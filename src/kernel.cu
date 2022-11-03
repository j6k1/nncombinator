#include<cmath>

__device__ float _exp(float x) {
    return __expf(x);
}

__device__ double _exp(double x) {
    return exp(x);
}

__device__ float _fmax(float a, float b) {
    return std::fmax(a,b);
}

__device__ double _fmax(double a, double b) {
    return std::fmax(a,b);
}
#define BLOCK_SHARED 1024

template<typename T>

__device__ void sigmoid_forward(T *input_output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = input_output[i];

        x = 1.0 / (1.0 + _exp(-x));

        input_output[i] = x;
    }
}
template<typename T>

__device__ void relu_forward(T *input_output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = 0.0;

        if (input_output[i] > 0.0 || isnan(input_output[i])) {
            x = 1.0;
        }

        input_output[i] = input_output[i] * x;
    }
}
template<typename T>

__device__ void swish_forward(T *input_output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = input_output[i];

        x = x * (1.0 / (1.0 + _exp(-x)));
        input_output[i] = x;
    }
}
template<typename T>

__device__ void tanh_forward(T *input_output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = input_output[i];

        input_output[i] = (_exp(x) - _exp(-x)) / (_exp(x) + _exp(-x));
    }
}
template<typename T>

__device__ void softmax_forward(T *input_output, const size_t units_len, const size_t batch_len) {
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    T *alpha_sdata = &sdata[0];

    T *sum_sdata = &sdata[BLOCK_SHARED];

    size_t tid = threadIdx.x;
    size_t batch_index = blockIdx.x;

    if (tid < units_len && batch_index < batch_len) {
        size_t end_block = batch_index * units_len + units_len;
        size_t distance = blockDim.x;

        sum_sdata[tid] = 0;
        alpha_sdata[tid] = 0.0/0.0;

        for (size_t i = batch_index * units_len + tid; i < end_block; i += distance) {
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],input_output[i]);
        }

        __syncthreads();

        if (tid < 512) {
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],alpha_sdata[tid + 512]);
        }
        __syncthreads();

        if (tid < 256) {
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],alpha_sdata[tid + 256]);
        }
        __syncthreads();

        if (tid < 128) {
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],alpha_sdata[tid + 128]);
        }
        __syncthreads();

        if (tid < 64) {
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],alpha_sdata[tid + 64]);
        }
        __syncthreads();

        if (tid < 32) {
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],alpha_sdata[tid + 32]);
        }
        __syncthreads();

        if (tid > 0 && tid < 32) {
            alpha_sdata[0] = _fmax(alpha_sdata[tid],alpha_sdata[0]);
        }
        __syncthreads();

        T alpha = alpha_sdata[0];

        T scale = 1e7;

        for (size_t i = batch_index * units_len + tid; i < end_block; i += distance) {
            sum_sdata[tid] += _exp(input_output[i] - alpha) * scale;
        }
        __syncthreads();

        if (tid < 512) {
            sum_sdata[tid] += sum_sdata[tid + 512];
        }
        __syncthreads();

        if (tid < 256) {
            sum_sdata[tid] += sum_sdata[tid + 256];
        }
        __syncthreads();

        if (tid < 128) {
            sum_sdata[tid] += sum_sdata[tid + 128];
        }
        __syncthreads();

        if (tid < 64) {
            sum_sdata[tid] += sum_sdata[tid + 64];
        }
        __syncthreads();

        if (tid < 32) {
            sum_sdata[tid] += sum_sdata[tid + 32];
        }
        __syncthreads();

        if (tid < 32) {
            sum_sdata[tid] += sum_sdata[tid + 32];
        }
        __syncthreads();

        if (tid < 16) {
            sum_sdata[tid] += sum_sdata[tid + 16];
        }
        __syncthreads();

        if (tid < 8) {
            sum_sdata[tid] += sum_sdata[tid + 8];
        }
        __syncthreads();


        if (tid < 4) {
            sum_sdata[tid] += sum_sdata[tid + 4];
        }
        __syncthreads();


        if (tid < 2) {
            sum_sdata[tid] += sum_sdata[tid + 2];
        }
        __syncthreads();

        if (tid < 1) {
            sum_sdata[tid] += sum_sdata[tid + 1];
        }
        __syncthreads();
        
        T sum = sum_sdata[0] / scale;

        for (size_t i = batch_index * units_len + tid; i < end_block; i += distance) {
            T number = _exp(input_output[i] - alpha);
            T x = number / sum;

            input_output[i] = x;
        }
    }
}
template<typename T>

__device__ void sigmoid_backward(const T *o, const T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = u[i];
        x = (1.0 - x) * x;

        loss[i] = x * loss[i];
    }
}
template<typename T>

__device__ void relu_backward(const T *o, const T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = 0.0;

        if (u[i] > 0.0) {
            x = 1.0;
        }

        loss[i] = x * loss[i];
    }
}
template<typename T>

__device__ void swish_backward(const T *o, const T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = u[i];
        x = x + 1.0 / (1.0 + _exp(-o[i])) * (1.0 - x);

        loss[i] = x;
    }
}
template<typename T>

__device__ void tanh_backward(const T *o, const T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = u[i];
        x = 1.0 - x * x;

        loss[i] = x * loss[i];
    }
}
template<typename T>

__device__ void softmax_backward(const T *o, const T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = u[i];
        x = x * (1.0 - x);

        loss[i] = loss[i] * x;
    }
}
template<typename T>

__device__ void reduce_linear_batch(const T *input, T *output, const int nlen, const int batch_size) {
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    if (blockIdx.x < nlen) {
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x + tid * nlen;
        unsigned int n = nlen * batch_size;
        unsigned int distance = blockDim.x * nlen;

        sdata[tid] = (T)0;

        while (i < n) {
            sdata[tid] += input[i];
            i += distance;
        }
        __syncthreads();

        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();

        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();

        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();

        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();

        if (tid < 32) {
            sdata[tid] += sdata[tid + 32];
        }

        __syncthreads();

        if (tid > 0 && tid < 32) {
            sdata[0] += sdata[tid];
        }

        if (tid == 0) {
            output[blockIdx.x] = sdata[0];
        }
    }
}
template<typename T>

__device__ void loss_linear_batch_by_canonical_link(const T *expected, T *actual, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        actual[i] = actual[i] - expected[i];
    }
}
template<typename T>

__device__ void loss_linear_batch_mse_derive(const T *t, T *r, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockDim.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        r[i] = r[i] - t[i];
    }
}
template<typename T>

__device__ void loss_linear_batch_cross_entropy_derive(const T *t, T *r, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockDim.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        r[i] = -(r[i] / (t[i] + (T)1e-7)) + (1.0 - t[i]) / (1.0 - r[i]);
    }
}
template<typename T>

__device__ void loss_linear_batch_cross_entropy_multiclass_derive(const T *t, T *r, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockDim.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        r[i] = -t[i] / r[i];
    }
}
extern "C" {
	__global__ void sigmoid_forward_float(float *input_output, const size_t units_len, const size_t batch_len) {
        sigmoid_forward(input_output,units_len,batch_len);
	}

	__global__ void relu_forward_float(float *input_output, const size_t units_len, const size_t batch_len) {
        relu_forward(input_output,units_len,batch_len);
    }

	__global__ void swish_forward_float(float *input_output, const size_t units_len, const size_t batch_len) {
        swish_forward(input_output,units_len,batch_len);
    }

	__global__ void tanh_forward_float(float *input_output, const size_t units_len, const size_t batch_len) {
        tanh_forward(input_output,units_len,batch_len);
    }

	__global__ void softmax_forward_float(float *input_output, const size_t units_len, const size_t batch_len) {
        softmax_forward(input_output,units_len,batch_len);
    }

	__global__ void sigmoid_backward_float(const float * o, const float *u, float *loss, const size_t units_len, const size_t batch_len) {
        sigmoid_backward(o,u,loss,units_len,batch_len);
    }

	__global__ void relu_backward_float(const float *o, const float *u, float *loss, const size_t units_len, const size_t batch_len) {
        relu_backward(o,u,loss,units_len,batch_len);
    }

	__global__ void swish_backward_float(const float *o, const float *u, float *loss, const size_t units_len, const size_t batch_len) {
        swish_backward(o,u,loss,units_len,batch_len);
    }

	__global__ void tanh_backward_float(const float *o, const float *u, float *loss, const size_t units_len, const size_t batch_len) {
        tanh_backward(o,u,loss,units_len,batch_len);
    }

	__global__ void softmax_backward_float(const float *o, const float *u, float *loss, const size_t units_len, const size_t batch_len) {
        softmax_backward(o,u,loss,units_len,batch_len);
    }
	__global__ void sigmoid_forward_double(double *input_output, const size_t units_len, const size_t batch_len) {
        sigmoid_forward(input_output,units_len,batch_len);
	}

	__global__ void relu_forward_double(double *input_output, const size_t units_len, const size_t batch_len) {
        relu_forward(input_output,units_len,batch_len);
    }

	__global__ void swish_forward_double(double *input_output, const size_t units_len, const size_t batch_len) {
        swish_forward(input_output,units_len,batch_len);
    }

	__global__ void tanh_forward_double(double *input_output, const size_t units_len, const size_t batch_len) {
        tanh_forward(input_output,units_len,batch_len);
    }

	__global__ void softmax_forward_double(double *input_output, const size_t units_len, const size_t batch_len) {
        softmax_forward(input_output,units_len,batch_len);
    }

	__global__ void sigmoid_backward_double(const double *o, const double *u, double *loss, const size_t units_len, const size_t batch_len) {
        sigmoid_backward(o,u,loss,units_len,batch_len);
    }

	__global__ void relu_backward_double(const double *o, const double *u, double *loss, const size_t units_len, const size_t batch_len) {
        relu_backward(o,u,loss,units_len,batch_len);
    }

	__global__ void swish_backward_double(const double *o, const double *u, double *loss, const size_t units_len, const size_t batch_len) {
        swish_backward(o,u,loss,units_len,batch_len);
    }

	__global__ void tanh_backward_double(const double *o, const double *u, double *loss, const size_t units_len, const size_t batch_len) {
        tanh_backward(o,u,loss,units_len,batch_len);
    }

	__global__ void softmax_backward_double(const double * o, const double *u, double *loss, const size_t units_len, const size_t batch_len) {
        softmax_backward(o,u,loss,units_len,batch_len);
    }

    __global__ void reduce_linear_batch_float(const float *input, float *output, const int nlen, const int batch_size) {
        reduce_linear_batch(input,output,nlen,batch_size);
    }

    __global__ void reduce_linear_batch_double(const double *input, double *output, const int nlen, const int batch_size) {
        reduce_linear_batch(input,output,nlen,batch_size);
    }

    __global__ void loss_linear_batch_by_canonical_link_float(const float *expected, float *actual, const int nlen, const int batch_size) {
        loss_linear_batch_by_canonical_link(expected,actual,nlen,batch_size);
    }

    __global__ void loss_linear_batch_by_canonical_link_double(const double *expected, double *actual, const int nlen, const int batch_size) {
        loss_linear_batch_by_canonical_link(expected,actual,nlen,batch_size);
    }

    __global__ void loss_linear_batch_mse_derive_float(const float *t, float *r, const int nlen, const int batch_size) {
        loss_linear_batch_mse_derive(t,r,nlen,batch_size);
    }

    __global__ void loss_linear_batch_mse_derive_double(const double *t, double *r, const int nlen, const int batch_size) {
        loss_linear_batch_mse_derive(t,r,nlen,batch_size);
    }

    __global__ void loss_linear_batch_cross_entropy_derive_float(const float *t, float *r, const int nlen, const int batch_size) {
        loss_linear_batch_cross_entropy_derive(t,r,nlen,batch_size);
    }

    __global__ void loss_linear_batch_cross_entropy_derive_double(const double *t, double *r, const int nlen, const int batch_size) {
        loss_linear_batch_cross_entropy_derive(t,r,nlen,batch_size);
    }

    __global__ void loss_linear_batch_cross_entropy_multiclass_derive_float(const float *t, float *r, const int nlen, const int batch_size) {
        loss_linear_batch_cross_entropy_multiclass_derive(t,r,nlen,batch_size);
    }

    __global__ void loss_linear_batch_cross_entropy_multiclass_derive_double(const double *t, double *r, const int nlen, const int batch_size) {
        loss_linear_batch_cross_entropy_multiclass_derive(t,r,nlen,batch_size);
    }
}