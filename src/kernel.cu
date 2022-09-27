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

        if (input_output[i] < 0.0) {
            input_output[i] = 0.0;
        }
    }
}
template<typename T>

__device__ void swish_forwaxd(T *input_output, const size_t units_len, const size_t batch_len) {
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
    return;

    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    T *alpha_sdata = &sdata[0];

    T *sum_sdata = &sdata[BLOCK_SHARED];

    size_t tid = threadIdx.x;
    size_t batch_index = blockIdx.x;

    if (tid < units_len && batch_index < batch_len) {
        unsigned int i = batch_index * units_len + tid;
        unsigned int end_block = (batch_index + 1) * units_len;
        unsigned int distance = blockDim.x;

        sum_sdata[tid] = (T)0;
        alpha_sdata[tid] = (T)0/(T)0;

        while (i < end_block) {
            sum_sdata[tid] += input_output[i];
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],input_output[i]);
            i += distance;
        }
        __syncthreads();

        if (tid < 512) {
            sum_sdata[tid] += sum_sdata[tid + 512];
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],alpha_sdata[tid + 512]);
        }
        __syncthreads();

        if (tid < 256) {
            sum_sdata[tid] += sum_sdata[tid + 256];
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],alpha_sdata[tid + 256]);
        }
        __syncthreads();

        if (tid < 128) {
            sum_sdata[tid] += sum_sdata[tid + 128];
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],alpha_sdata[tid + 128]);
        }
        __syncthreads();

        if (tid < 64) {
            sum_sdata[tid] += sum_sdata[tid + 64];
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],alpha_sdata[tid + 64]);
        }
        __syncthreads();

        if (tid < 32) {
            sum_sdata[tid] += sum_sdata[tid + 32];
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],alpha_sdata[tid + 32]);
        }
        __syncthreads();

        if (tid > 0 && tid < 32) {
            sum_sdata[0] += sum_sdata[tid];
            alpha_sdata[tid] = _fmax(alpha_sdata[tid],alpha_sdata[0]);
        }
        __syncthreads();

        T sum = sum_sdata[0];
        T alpha = alpha_sdata[0];

        for (size_t i = batch_index == 0 ? tid : batch_index * units_len + tid; i < end_block; i+=distance) {
            T number = _exp(input_output[i] - alpha);
            T x = number / sum;

            input_output[i] = x;
        }
    }
}
template<typename T>

__device__ void softmax_preprocessing(const T *input, const size_t units_len, const size_t batch_len, T *alpha, T *sum) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        T a = 0.0;
        T s = 0.0;

        for (size_t i=0; i < units_len; i++) {
            size_t idx = batch_index == 0 ? index : batch_index * units_len + index;

            if (input[idx] > a) {
                a = input[idx];
            }

            s += input[idx];
        }

        alpha[batch_index] = a;
        sum[batch_index] = s;
    }
}
template<typename T>

__device__ void sigmoid_backward(T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T e = 1.0 / (1.0 + _exp(-u[i]));
        T x = e * (1.0 - e);

        loss[i] = x * loss[i];
    }
}
template<typename T>

__device__ void relu_backward(T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        if (u[i] <= 0.0) {
            loss[i] = 0.0;
        }
    }
}
template<typename T>

__device__ void swish_backward(T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = u[i];

        x = x * (1.0 / (1.0 + _exp(-x))) +
                (1.0 / (1.0 + _exp(-x))) * (1.0 - (x * (1.0 / (1.0 + _exp(-x)))));

        loss[i] = x;
    }
}
template<typename T>

__device__ void tanh_backward(T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = u[i];
        T e = (_exp(x) - _exp(-x)) / (_exp(x) + _exp(-x));

        x = 1.0 - e * e;

        loss[i] = x * loss[i];
    }
}
template<typename T>

__device__ void softmax_backward(T *u, T *loss, const size_t units_len, const size_t batch_len) {
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
extern "C" {
	__global__ void sigmoid_forward_float(float *input_output, const size_t units_len, const size_t batch_len) {
        sigmoid_forward(input_output,units_len,batch_len);
	}

	__global__ void relu_forward_float(float *input_output, const size_t units_len, const size_t batch_len) {
        relu_forward(input_output,units_len,batch_len);
    }

	__global__ void swish_forwaxd_float(float *input_output, const size_t units_len, const size_t batch_len) {
        swish_forwaxd(input_output,units_len,batch_len);
    }

	__global__ void tanh_forward_float(float *input_output, const size_t units_len, const size_t batch_len) {
        tanh_forward(input_output,units_len,batch_len);
    }

	__global__ void softmax_forward_float(float *input_output, const size_t units_len, const size_t batch_len) {
        softmax_forward(input_output,units_len,batch_len);
    }

    __global__ void softmax_preprocessing_float(const float *input, const size_t units_len, const size_t batch_len, float *alpha, float *sum) {
        softmax_preprocessing(input,units_len,batch_len,alpha,sum);
    }

	__global__ void sigmoid_backward_float(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        sigmoid_backward(u,loss,units_len,batch_len);
    }

	__global__ void relu_backward_float(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        relu_backward(u,loss,units_len,batch_len);
    }

	__global__ void swish_backward_float(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        swish_backward(u,loss,units_len,batch_len);
    }

	__global__ void tanh_backward_float(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        tanh_backward(u,loss,units_len,batch_len);
    }

	__global__ void softmax_backward_float(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        softmax_backward(u,loss,units_len,batch_len);
    }
	__global__ void sigmoid_forward_double(float *input_output, const size_t units_len, const size_t batch_len) {
        sigmoid_forward(input_output,units_len,batch_len);
	}

	__global__ void relu_forward_double(float *input_output, const size_t units_len, const size_t batch_len) {
        relu_forward(input_output,units_len,batch_len);
    }

	__global__ void swish_forwaxd_double(float *input_output, const size_t units_len, const size_t batch_len) {
        swish_forwaxd(input_output,units_len,batch_len);
    }

	__global__ void tanh_forward_double(float *input_output, const size_t units_len, const size_t batch_len) {
        tanh_forward(input_output,units_len,batch_len);
    }

	__global__ void softmax_forward_double(float *input_output, const size_t units_len, const size_t batch_len) {
        softmax_forward(input_output,units_len,batch_len);
    }

    __global__ void softmax_preprocessing_double(const float *input, const size_t units_len, const size_t batch_len, float *alpha, float *sum) {
        softmax_preprocessing(input,units_len,batch_len,alpha,sum);
    }

	__global__ void sigmoid_backward_double(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        sigmoid_backward(u,loss,units_len,batch_len);
    }

	__global__ void relu_backward_double(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        relu_backward(u,loss,units_len,batch_len);
    }

	__global__ void swish_backward_double(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        swish_backward(u,loss,units_len,batch_len);
    }

	__global__ void tanh_backward_double(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        tanh_backward(u,loss,units_len,batch_len);
    }

	__global__ void softmax_backward_double(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        softmax_backward(u,loss,units_len,batch_len);
    }

    __global__ void reduce_linear_batch_float(const float *input, float *output, const int nlen, const int batch_size) {
        reduce_linear_batch(input,output,nlen,batch_size);
    }

    __global__ void reduce_linear_batch_double(const double *input, double *output, const int nlen, const int batch_size) {
        reduce_linear_batch(input,output,nlen,batch_size);
    }
}