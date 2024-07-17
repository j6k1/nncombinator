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

__device__ float _sqrt(float x) {
    return sqrtf(x);
}

__device__ double _sqrt(double x) {
    return sqrt(x);
}

__device__ float _add(float x, float y) {
    return __fadd_rn(x,y);
}

__device__ double _add(double x, double y) {
    return __dadd_rn(x,y);
}

__device__ float _fma(float x, float y, float z) {
    return fmaf(x,y,z);
}

__device__ double _fma(double x, double y, double z) {
    return fma(x,y,z);
}

#define BLOCK_SHARED 1024
#define BLOCK_SHARED_SMALL 32
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

        if (!(input_output[i] > 0.0)) {
            input_output[i] = 0.0;
        }
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

        for (size_t i = batch_index * units_len + tid; i < end_block; i += distance) {
            sum_sdata[tid] += _exp(input_output[i] - alpha);
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
        
        T sum = sum_sdata[0];

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

        T x = o[i];
        x = x * (1.0 - x);

        loss[i] = x * loss[i];
    }
}
template<typename T>

__device__ void relu_backward(const T *o, const T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        if (!(u[i] > 0.0)) {
            loss[i] = 0.0;
        }
    }
}
template<typename T>

__device__ void swish_backward(const T *o, const T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = o[i];
        x = x + 1.0 / (1.0 + _exp(-u[i])) * (1.0 - x);

        loss[i] = x * loss[i];
    }
}
template<typename T>

__device__ void tanh_backward(const T *o, const T *u, T *loss, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = o[i];
        x = 1.0 - x * x;

        loss[i] = x * loss[i];
    }
}
template<typename T>

__device__ void softmax_backward(const T *o, const T *u, T *loss, const size_t units_len, const size_t batch_len) {
    extern __shared__ char smem[];
    T *sum_sdata = reinterpret_cast<T*>(smem);

    size_t tid = threadIdx.x;
    size_t batch_index = blockIdx.x;

    if (tid < units_len && batch_index < batch_len) {
        size_t end_block = batch_index * units_len + units_len;
        size_t distance = blockDim.x;

        T scale = 1e7;

        sum_sdata[tid] = 0;

        for (size_t i = batch_index * units_len + tid; i < end_block; i += distance) {
            sum_sdata[tid] += (loss[i] * -o[i]) * scale;
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
            T yk = o[i];
            T l = loss[i];
            loss[i] = sum * yk + l * (yk * yk + (yk * (1.0 - yk)));
        }
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

        T acc = 0.0;

        while (i < n) {
            acc += input[i];
            i += distance;
        }
        __syncthreads();


        acc += __shfl_down_sync(0xffffffff,acc,16);
        acc += __shfl_down_sync(0xffffffff,acc,8);
        acc += __shfl_down_sync(0xffffffff,acc,4);
        acc += __shfl_down_sync(0xffffffff,acc,2);
        acc += __shfl_down_sync(0xffffffff,acc,1);

        if (tid % 32 == 0) {
            sdata[tid / 32] = acc;
        }
        __syncthreads();

        if (tid < 32) {
            acc = sdata[tid];

            acc += __shfl_down_sync(0xffffffff,acc,16);
            acc += __shfl_down_sync(0xffffffff,acc,8);
            acc += __shfl_down_sync(0xffffffff,acc,4);
            acc += __shfl_down_sync(0xffffffff,acc,2);
            acc += __shfl_down_sync(0xffffffff,acc,1);
        }

        if (tid == 0) {
            output[blockIdx.x] = acc;
        }
    }
}
template<typename T>

__device__ void forward_linear_batch(const T *input, const T *units, const T *bias, T *output,
                                     const size_t input_len, const size_t output_len, const size_t batch_size) {
    extern __shared__ char smem[];

    T *sdata_sum = reinterpret_cast<T*>(&smem[0]);
    T *sdata_c = reinterpret_cast<T*>(&smem[BLOCK_SHARED * sizeof(T)]);

    if (blockIdx.x < output_len * batch_size) {
        size_t batch_index = blockIdx.x / output_len;

        size_t tid = threadIdx.x;
        size_t out_index = blockIdx.x - output_len * batch_index;
        size_t i = batch_index * input_len;
        size_t j = tid;
        size_t tid_warp = tid % 32;
        size_t distance = blockDim.x;

        sdata_sum[tid] = 0.0;
        sdata_c[tid] = 0.0;

        T acc = 0.0;
        T c = 0.0;

        /**
         * Kahan summation algorithm
         */
        while (j < input_len) {
            const T y = input[i + j] * units[j * output_len + out_index] - c;
            const T t = acc + y;
            c = (t - acc) - y;
            acc = t;
            j += distance;
        }

        {
            T dc = 0.0;
            T dacc = 0.0;

            dc = __shfl_down_sync(0xffffffff,c,16);
            dacc = __shfl_down_sync(0xffffffff,acc,16);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,8);
            dacc = __shfl_down_sync(0xffffffff,acc,8);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,4);
            dacc = __shfl_down_sync(0xffffffff,acc,4);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,2);
            dacc = __shfl_down_sync(0xffffffff,acc,2);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,1);
            dacc = __shfl_down_sync(0xffffffff,acc,1);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }
        }

        if (tid_warp == 0) {
            sdata_c[tid / 32] = c;
            sdata_sum[tid / 32] = acc;
        }
        __syncthreads();

        if (tid < 32) {
            c = sdata_c[tid];
            acc = sdata_sum[tid];

            const size_t org_index = tid * 32;

            T dc = 0.0;
            T dacc = 0.0;

            dc = __shfl_down_sync(0xffffffff,c,16);
            dacc = __shfl_down_sync(0xffffffff,acc,16);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,8);
            dacc = __shfl_down_sync(0xffffffff,acc,8);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,4);
            dacc = __shfl_down_sync(0xffffffff,acc,4);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,2);
            dacc = __shfl_down_sync(0xffffffff,acc,2);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,1);
            dacc = __shfl_down_sync(0xffffffff,acc,1);

            {
                const T y = dacc - c - dc;
                const T t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }
        }

        if (tid == 0) {
            const T y = bias[out_index] - c;
            const T t = acc + y;
            output[blockIdx.x] = t;
        }
    }
}

template<typename T>

__device__ void backward_linear_batch(const T *loss, const T *units, T *output,
                                     const size_t input_len, const size_t output_len, const size_t batch_size) {
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    if (blockIdx.x < input_len * batch_size) {
        size_t batch_index = blockIdx.x / input_len;

        size_t tid = threadIdx.x;
        size_t tid_warp = tid % 32;
        size_t input_index = blockIdx.x - input_len * batch_index;
        size_t i = batch_index * output_len;
        size_t j = tid;
        size_t distance = blockDim.x;

        sdata[tid] = (T)0;

        T acc = 0.0;

        while (j < output_len) {
            acc += loss[i + j] * units[input_index * output_len + j];
            j += distance;
        }

        acc += __shfl_down_sync(0xffffffff,acc,16);
        acc += __shfl_down_sync(0xffffffff,acc,8);
        acc += __shfl_down_sync(0xffffffff,acc,4);
        acc += __shfl_down_sync(0xffffffff,acc,2);
        acc += __shfl_down_sync(0xffffffff,acc,1);

        if (tid_warp == 0) {
            sdata[tid / 32] = acc;
        }
        __syncthreads();

        if (tid < 32) {
            acc = sdata[tid];

            acc += __shfl_down_sync(0xffffffff,acc,16);
            acc += __shfl_down_sync(0xffffffff,acc,8);
            acc += __shfl_down_sync(0xffffffff,acc,4);
            acc += __shfl_down_sync(0xffffffff,acc,2);
            acc += __shfl_down_sync(0xffffffff,acc,1);
        }

        if (tid == 0) {
            output[blockIdx.x] = acc;
        }
    }
}

template<typename T>

__device__ void linear_gradient_batch(const T *loss, const T *input, T *output,
                                      const size_t input_len, const size_t output_len,
                                      const size_t units_size, const size_t batch_size) {
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    if (blockIdx.x < units_size) {
        size_t tid = threadIdx.x;
        size_t tid_warp = tid % 32;
        size_t i = blockIdx.x / output_len;
        size_t j = blockIdx.x - output_len * i;
        size_t k = tid;

        i = i + k * input_len;
        j = j + k * output_len;

        size_t distance = blockDim.x;

        sdata[tid] = (T)0;

        T acc = 0.0;

        while (k < batch_size) {
            acc += loss[j] * input[i];
            k += distance;
            i += distance * input_len;
            j += distance * output_len;
        }

        acc += __shfl_down_sync(0xffffffff,acc,16);
        acc += __shfl_down_sync(0xffffffff,acc,8);
        acc += __shfl_down_sync(0xffffffff,acc,4);
        acc += __shfl_down_sync(0xffffffff,acc,2);
        acc += __shfl_down_sync(0xffffffff,acc,1);

        if (tid_warp == 0) {
            sdata[tid / 32] = acc;
        }
        __syncthreads();

        if (tid < 32) {
            acc = sdata[tid];

            acc += __shfl_down_sync(0xffffffff,acc,16);
            acc += __shfl_down_sync(0xffffffff,acc,8);
            acc += __shfl_down_sync(0xffffffff,acc,4);
            acc += __shfl_down_sync(0xffffffff,acc,2);
            acc += __shfl_down_sync(0xffffffff,acc,1);
        }

        if (tid == 0) {
            output[blockIdx.x] = acc;
        }
    }
}

template<typename T>

__device__ void loss_linear_batch_by_canonical_link(const T *expected, T *actual, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        actual[i] = (actual[i] - expected[i]) / batch_size;
    }
}
template<typename T>

__device__ void loss_linear_batch_mse_derive(const T *t, T *r, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        r[i] = (r[i] - t[i]) / batch_size;
    }
}
template<typename T>

__device__ void loss_linear_batch_cross_entropy_derive(const T *t, T *r, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        r[i] = -((r[i] / (t[i] + (T)1e-7)) + (1.0 - t[i]) / (1.0 - r[i])) / batch_size;
    }
}
template<typename T>

__device__ void loss_linear_batch_cross_entropy_multiclass_derive(const T *t, T *r, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        r[i] = -(t[i] / r[i]) / batch_size;
    }
}

template<typename T>

__device__ void update_with_sgd(T *weight, const T *grad, const size_t size, const T a, const T lambda) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size) {
        T w = weight[index];

        w = w - a * (grad[index] + lambda * w);

        weight[index] = w;
    }
}

template<typename T>

__device__ void update_with_momentum_sgd(T *weight, const T *grad, const size_t size, const T a, const T mu, const T lambda, T *vt) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size) {
        T w = weight[index];
        T _vt = vt[index];
        T e = grad[index];

        _vt = mu * _vt - (a * (e + lambda * w));

        w = w + _vt;

        weight[index] = w;
        vt[index] = _vt;
    }
}

template<typename T>

__device__ void update_with_adagrad(T *weight, const T *grad, const size_t size, const T a, const T eps, T *gt) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size) {
        T w = weight[index];
        T _gt = gt[index];
        const T e = grad[index];

        _gt = _gt + e * e;

        w = w - a * (e / (_sqrt(_gt) + eps));

        weight[index] = w;
        gt[index] = _gt;
    }
}

template<typename T>

__device__ void update_with_rmsprop(T *weight, const T *grad, const size_t size, const T a, const T mu, const T eps, T *gt) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size) {
        T w = weight[index];
        T _gt = gt[index];
        const T e = grad[index];

        _gt = mu * _gt + (1 - mu) * e * e;
        w = w - a * e / (_sqrt(_gt) + eps);

        weight[index] = w;
        gt[index] = _gt;
    }
}

template<typename T>

__device__ void update_with_adam(T *weight, const T *grad, const size_t size, const T a, const T eps, T *mt, T *vt, const T b1, const T b2, T b1t, T b2t) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size) {
        T w = weight[index];
        T _mt = mt[index];
        T _vt = vt[index];
        const T e = grad[index];

        _mt = b1 * _mt + (1 - b1) * e;
        _vt = b2 * _vt + (1 - b2) * e * e;

        w = w - a * (_mt / (1 - b1t)) / _sqrt((_vt / (1 - b2t)) + eps);

        weight[index] = w;
        mt[index] = _mt;
        vt[index] = _vt;
    }
}

template<typename T>

__device__ void forward_diff_linear(const size_t *indexes, const T *input, const T *units, T *output, const size_t output_size, const size_t diff_len) {
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    if (blockIdx.x < output_size) {
        unsigned int out_index = blockIdx.x;
        unsigned int tid = threadIdx.x;
        unsigned int i = tid;
        unsigned int n = diff_len;
        unsigned int distance = blockDim.x;

        sdata[tid] = (T)0;

        while (i < n) {
            sdata[tid] += units[indexes[i] * output_size + out_index] * input[i];
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

        __syncthreads();

        if (tid == 0) {
            output[out_index] += sdata[0];
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

    __global__ void forward_linear_batch_float(const float *input, const float *units, const float *bias, float *output,
                                         const size_t input_len, const size_t output_len, const size_t batch_size) {
        forward_linear_batch(input,units,bias,output,input_len,output_len,batch_size);
    }

    __global__ void forward_linear_batch_double(const double *input, const double *units, const double *bias, double *output,
                                         const size_t input_len, const size_t output_len, const size_t batch_size) {
        forward_linear_batch(input,units,bias,output,input_len,output_len,batch_size);
    }

    __global__ void backward_linear_batch_float(const float *loss, const float *units, float *output,
                                         const size_t input_len, const size_t output_len, const size_t batch_size) {
        backward_linear_batch(loss,units,output,input_len,output_len,batch_size);
    }

    __global__ void backward_linear_batch_double(const double *loss, const double *units, double *output,
                                         const size_t input_len, const size_t output_len, const size_t batch_size) {
        backward_linear_batch(loss,units,output,input_len,output_len,batch_size);
    }

    __global__ void linear_gradient_batch_float(const float *loss, const float *input, float *output,
                                                const size_t input_len, const size_t output_len,
                                                const size_t units_size, const size_t batch_size) {
        linear_gradient_batch(loss,input,output,input_len,output_len,units_size,batch_size);
    }

    __global__ void linear_gradient_batch_double(const double *loss, const double *input, double *output,
                                                 const size_t input_len, const size_t output_len,
                                                 const size_t units_size, const size_t batch_size) {
        linear_gradient_batch(loss,input,output,input_len,output_len,units_size,batch_size);
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
    __global__ void update_with_sgd_float(float *weight, const float *grad, const size_t size, const float a, const float lambda) {
        update_with_sgd(weight,grad,size,a,lambda);
    }
    __global__ void update_with_sgd_double(double *weight, const double *grad, const size_t size, const double a, const double lambda) {
        update_with_sgd(weight,grad,size,a,lambda);
    }

    __global__ void update_with_momentum_sgd_float(float *weight, const float *grad, const size_t size, const float a, const float mu, const float lambda, float *vt) {
        update_with_momentum_sgd(weight,grad,size,a,mu,lambda,vt);
    }

    __global__ void update_with_momentum_sgd_double(double *weight, const double *grad, const size_t size, const double a, const double mu, const double lambda, double *vt) {
        update_with_momentum_sgd(weight,grad,size,a,mu,lambda,vt);
    }

    __global__ void update_with_adagrad_float(float *weight, const float *grad, const size_t size, const float a, const float eps, float *gt) {
        update_with_adagrad(weight,grad,size,a,eps,gt);
    }

    __global__ void update_with_adagrad_double(double *weight, const double *grad, const size_t size, const double a, const double eps, double *gt) {
        update_with_adagrad(weight,grad,size,a,eps,gt);
    }

    __global__ void update_with_rmsprop_float(float *weight, const float *grad, const size_t size, const float a, const float mu, const float eps, float *gt) {
        update_with_rmsprop(weight,grad,size,a,mu,eps,gt);
    }

    __global__ void update_with_rmsprop_double(double *weight, const double *grad, const size_t size, const double a, const double mu, double eps, double *gt) {
        update_with_rmsprop(weight,grad,size,a,mu,eps,gt);
    }

    __global__ void update_with_adam_float(float *weight, const float *grad, const size_t size, const float a, const float eps, float *mt, float *vt, const float b1, const float b2, float b1t, float b2t) {
        update_with_adam(weight,grad,size,a,eps,mt,vt,b1,b2,b1t,b2t);
    }

    __global__ void update_with_adam_double(double *weight, const double *grad, const size_t size, const double a, const double eps, double *mt, double *vt, const double b1, const double b2, double b1t, double b2t) {
        update_with_adam(weight,grad,size,a,eps,mt,vt,b1,b2,b1t,b2t);
    }

    __global__ void forward_diff_linear_float(const size_t *indexes, const float *input, const float *units, float *output, const size_t output_size, const size_t diff_len) {
        forward_diff_linear(indexes,input,units,output,output_size,diff_len);
    }

    __global__ void forward_diff_linear_double(const size_t *indexes, const double *input, const double *units, double *output, const size_t output_size, const size_t diff_len) {
        forward_diff_linear(indexes,input,units,output,output_size,diff_len);
    }
}