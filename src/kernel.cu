#include<cmath>
#include<mma.h>
#include<cuda.h>
#include<cuda_runtime.h>

using namespace nvcuda;

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

__device__ half _to_half(float x) {
    return __float2half(x);
}

__device__ half _to_half(double x) {
    return __double2half(x);
}

__device__ size_t calc_index(size_t x, size_t y, size_t leading_dimension) {
    return y * leading_dimension + x;
}

__device__ size_t calc_transposed_index(size_t x, size_t y, size_t leading_dimension) {
    return x * leading_dimension + y;
}

#define BLOCK_SHARED 1024
#define BLOCK_SHARED_SMALL 32
#define TILE_SIZE 16
#define TILE_SIZE_2D 256
template<typename T>

__device__ void sigmoid_forward(const T *input, T *output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = input[i];

        x = 1.0 / (1.0 + _exp(-x));

        output[i] = x;
    }
}
template<typename T>

__device__ void relu_forward(const T *input, T *output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        output[i] = _fmax(input[i],(T)0.0);
    }
}
template<typename T>

__device__ void swish_forward(const T *input, T *output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = input[i];

        x = x * (1.0 / (1.0 + _exp(-x)));
        output[i] = x;
    }
}
template<typename T>

__device__ void tanh_forward(const T *input, T *output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = input[i];

        output[i] = (_exp(x) - _exp(-x)) / (_exp(x) + _exp(-x));
    }
}
template<typename T>

__device__ void softmax_forward(const T *input, T *output, const size_t units_len, const size_t batch_len) {
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    T *alpha_sdata = &sdata[0];

    T *sum_sdata = &sdata[BLOCK_SHARED_SMALL];

    size_t tid = threadIdx.x;
    size_t tid_warp = tid % 32;

    size_t batch_index = blockIdx.x;

    if (tid < units_len && batch_index < batch_len) {
        size_t end_block = batch_index * units_len + units_len;
        size_t distance = blockDim.x;

        if (tid < 32) {
            sum_sdata[tid] = 0;
            alpha_sdata[tid] = 0.0/0.0;
        }
        __syncthreads();

        T alpha = 0.0;

        for (size_t i = batch_index * units_len + tid; i < end_block; i += distance) {
            alpha = _fmax(alpha,input[i]);
        }

        alpha = _fmax(alpha,__shfl_down_sync(0xffffffff,alpha,16));
        alpha = _fmax(alpha,__shfl_down_sync(0xffffffff,alpha,8));
        alpha = _fmax(alpha,__shfl_down_sync(0xffffffff,alpha,4));
        alpha = _fmax(alpha,__shfl_down_sync(0xffffffff,alpha,2));
        alpha = _fmax(alpha,__shfl_down_sync(0xffffffff,alpha,1));

        if (tid_warp == 0) {
            alpha_sdata[tid / 32] = alpha;
        }
        __syncthreads();

        if (tid < 32) {
            alpha = alpha_sdata[tid];

            alpha = _fmax(alpha,__shfl_down_sync(0xffffffff,alpha,16));
            alpha = _fmax(alpha,__shfl_down_sync(0xffffffff,alpha,8));
            alpha = _fmax(alpha,__shfl_down_sync(0xffffffff,alpha,4));
            alpha = _fmax(alpha,__shfl_down_sync(0xffffffff,alpha,2));
            alpha = _fmax(alpha,__shfl_down_sync(0xffffffff,alpha,1));
        }

        if (tid == 0) {
            alpha_sdata[0] = alpha;
        }
        __syncthreads();

        T sum = 0.0;

        for (size_t i = batch_index * units_len + tid; i < end_block; i += distance) {
            sum += _exp(input[i] - alpha);
        }

        sum += __shfl_down_sync(0xffffffff,sum,16);
        sum += __shfl_down_sync(0xffffffff,sum,8);
        sum += __shfl_down_sync(0xffffffff,sum,4);
        sum += __shfl_down_sync(0xffffffff,sum,2);
        sum += __shfl_down_sync(0xffffffff,sum,1);

        if (tid_warp == 0) {
            sum_sdata[tid / 32] = sum;
        }
        __syncthreads();

        if (tid < 32) {
            sum = sum_sdata[tid];

            sum += __shfl_down_sync(0xffffffff,sum,16);
            sum += __shfl_down_sync(0xffffffff,sum,8);
            sum += __shfl_down_sync(0xffffffff,sum,4);
            sum += __shfl_down_sync(0xffffffff,sum,2);
            sum += __shfl_down_sync(0xffffffff,sum,1);
        }

        if (tid == 0) {
            sum_sdata[0] = sum;
        }
        __syncthreads();

        sum = sum_sdata[0];

        for (size_t i = batch_index * units_len + tid; i < end_block; i += distance) {
            T number = _exp(input[i] - alpha);
            T x = number / sum;

            output[i] = x;
        }
    }
}
template<typename T>

__device__ void sigmoid_backward(const T *o, const T *u, const T *loss, T *output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = o[i];
        x = x * (1.0 - x);

        output[i] = x * loss[i];
    }
}
template<typename T>

__device__ void relu_backward(const T *o, const T *u, const T *loss, T *output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        if (!(u[i] > 0.0)) {
            output[i] = 0.0;
        } else {
            output[i] = loss[i];
        }
    }
}
template<typename T>

__device__ void swish_backward(const T *o, const T *u, const T *loss, T *output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = o[i];
        x = x + 1.0 / (1.0 + _exp(-u[i])) * (1.0 - x);

        output[i] = x * loss[i];
    }
}
template<typename T>

__device__ void tanh_backward(const T *o, const T *u, const T *loss, T *output, const size_t units_len, const size_t batch_len) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T x = o[i];
        x = 1.0 - x * x;

        output[i] = x * loss[i];
    }
}
template<typename T>

__device__ void softmax_backward(const T *o, const T *u, const T *loss, T *output, const size_t units_len, const size_t batch_len) {
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
            output[i] = sum * yk + l * (yk * yk + (yk * (1.0 - yk)));
        }
    }
}
template<typename T>

__device__ void reduce_linear_batch(const T *input, T *output, const int nlen, const int batch_size) {
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    if (blockIdx.x < nlen && blockDim.x * blockIdx.z + threadIdx.x < batch_size) {
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x + tid * nlen;
        unsigned int distance = blockDim.x * nlen;

        if (tid < 32) {
            sdata[tid] = (T)0;
        }
        __syncthreads();

        T acc = 0.0;

        acc += input[i + blockIdx.z * distance];

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
            atomicAdd(&output[blockIdx.x],acc);
        }
    }
}
template<typename T>

__device__ void forward_linear_batch(const T *input, const T *units, const T *bias, T *output,
                                     const size_t input_len, const size_t output_len, const size_t batch_size) {
    extern __shared__ char smem[];

    if (blockIdx.x < output_len * batch_size && blockIdx.z * blockDim.x + threadIdx.x < input_len) {
        T *sdata_sum = reinterpret_cast<T*>(&smem[0]);
        T *sdata_c = reinterpret_cast<T*>(&smem[BLOCK_SHARED_SMALL * sizeof(T)]);

        size_t batch_index = blockIdx.x / output_len;

        size_t tid = threadIdx.x;
        size_t out_index = blockIdx.x - output_len * batch_index;
        size_t i = batch_index * input_len;
        size_t j = tid;
        size_t tid_warp = tid % 32;
        size_t distance = blockDim.x;

        if (tid < 32) {
            sdata_sum[tid] = 0.0;
            sdata_c[tid] = 0.0;
        }
        __syncthreads();

        T acc = 0.0;
        T c = 0.0;

        j += blockIdx.z * distance;

        acc = input[i + j] * units[j * output_len + out_index] - c;

        T dc = 0.0;
        T dacc = 0.0;
        T y;
        T t;

        /**
         * Kahan summation algorithm
         */
        {
            dc = __shfl_down_sync(0xffffffff,c,16);
            dacc = __shfl_down_sync(0xffffffff,acc,16);

            {
                y = dacc - c - dc;
                t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,8);
            dacc = __shfl_down_sync(0xffffffff,acc,8);

            {
                y = dacc - c - dc;
                t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,4);
            dacc = __shfl_down_sync(0xffffffff,acc,4);

            {
                y = dacc - c - dc;
                t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,2);
            dacc = __shfl_down_sync(0xffffffff,acc,2);

            {
                y = dacc - c - dc;
                t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,1);
            dacc = __shfl_down_sync(0xffffffff,acc,1);

            {
                y = dacc - c - dc;
                t = acc + y;
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

            dc = 0.0;
            dacc = 0.0;

            dc = __shfl_down_sync(0xffffffff,c,16);
            dacc = __shfl_down_sync(0xffffffff,acc,16);

            {
                y = dacc - c - dc;
                t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,8);
            dacc = __shfl_down_sync(0xffffffff,acc,8);

            {
                y = dacc - c - dc;
                t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,4);
            dacc = __shfl_down_sync(0xffffffff,acc,4);

            {
                y = dacc - c - dc;
                t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,2);
            dacc = __shfl_down_sync(0xffffffff,acc,2);

            {
                y = dacc - c - dc;
                t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }

            dc = __shfl_down_sync(0xffffffff,c,1);
            dacc = __shfl_down_sync(0xffffffff,acc,1);

            {
                y = dacc - c - dc;
                t = acc + y;
                c = (t - acc) - y;
                acc = t;
            }
        }

        if (tid == 0) {
            y = bias[out_index] - c;
            t = acc + y;
            atomicAdd(&output[blockIdx.x], t);
        }
    }
}

template<typename T>

__device__ void backward_linear_batch(const T *loss, const T *units, T *output,
                                     const size_t input_len, const size_t output_len, const size_t batch_size) {
    extern __shared__ char smem[];

    float *sdata_c = reinterpret_cast<float*>(&smem[0]);
    half *sdata_a = reinterpret_cast<half*>(&smem[TILE_SIZE_2D * sizeof(float)]);
    half *sdata_b = reinterpret_cast<half*>(&smem[TILE_SIZE_2D * sizeof(float) + TILE_SIZE_2D * sizeof(half)]);

    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> c_frag;

    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    size_t bx = blockIdx.x * TILE_SIZE;
    size_t by = blockIdx.y * TILE_SIZE;

    if (ty < 2) {
        wmma::fill_fragment(c_frag, 0.0f);
    }

    __syncthreads();

    for (int k = 0; k < output_len; k += TILE_SIZE) {
        if (k+tx < output_len && by + ty < batch_size) {
            sdata_a[ty * TILE_SIZE + tx] = _to_half(loss[calc_index(k+tx,by+ty,output_len)]);
        } else {
            sdata_a[ty * TILE_SIZE + tx] = __float2half(0.0f);
        }

        if (k+ty < output_len && bx + tx < input_len) {
            sdata_b[ty * TILE_SIZE + tx] = _to_half(units[calc_transposed_index(bx+tx,k+ty,output_len)]);
        } else {
            sdata_b[ty * TILE_SIZE + tx] = __float2half(0.0f);
        }

        __syncthreads();

        if (ty < 2) {
            wmma::load_matrix_sync(a_frag, sdata_a, TILE_SIZE);
            wmma::load_matrix_sync(b_frag, sdata_b, TILE_SIZE);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    if (ty < 2) {
        wmma::store_matrix_sync(sdata_c, c_frag, TILE_SIZE, wmma::mem_row_major);
    }

    __syncthreads();

    if (tx + bx < input_len && ty + by < batch_size) {
        output[calc_index(tx+bx,ty+by,input_len)] = (T)sdata_c[ty * TILE_SIZE + tx];
    }
}

template<typename T>

__device__ void linear_gradient_batch(const T *loss, const T *input, T *output,
                                      const size_t input_len, const size_t output_len,
                                      const size_t units_size, const size_t batch_size) {
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    if (blockIdx.x < units_size && blockDim.x * blockIdx.z + threadIdx.x < batch_size) {
        size_t tid = threadIdx.x;
        size_t tid_warp = tid % 32;
        size_t i = blockIdx.x / output_len;
        size_t j = blockIdx.x - output_len * i;
        size_t k = tid;

        i = i + k * input_len;
        j = j + k * output_len;

        size_t distance = blockDim.x;

        if (tid < 32) {
            sdata[tid] = (T)0;
        }
        __syncthreads();

        T acc = 0.0;

        acc += loss[j + distance * output_len * blockIdx.z] * input[i + distance * input_len * blockIdx.z];

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
            atomicAdd(&output[blockIdx.x],acc);
        }
    }
}

template<typename T>

__device__ void loss_linear_batch_by_canonical_link(const T *expected, const T *actual,T *output, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        output[i] = (actual[i] - expected[i]) / batch_size;
    }
}
template<typename T>

__device__ void loss_linear_batch_mse_derive(const T *t, const T *r, T* output, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        output[i] = (r[i] - t[i]) / batch_size;
    }
}
template<typename T>

__device__ void loss_linear_batch_cross_entropy_derive(const T *t, const T *r, T* output, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        output[i] = -((r[i] / (t[i] + (T)1e-7)) + (1.0 - t[i]) / (1.0 - r[i])) / batch_size;
    }
}
template<typename T>

__device__ void loss_linear_batch_cross_entropy_multiclass_derive(const T *t, const T *r, T* output, const int nlen, const int batch_size) {
    const size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (batch_index < batch_size && index < nlen) {
        const size_t i = batch_index * nlen + index;
        output[i] = -(t[i] / r[i]) / batch_size;
    }
}

template<typename T>

__device__ void update_with_sgd(T *weight, const T *grad, const size_t size, const T a, const T weight_decay) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size) {
        T w = weight[index];

        w = w - a * (grad[index] + weight_decay * w);

        weight[index] = w;
    }
}

template<typename T>

__device__ void update_with_momentum_sgd(T *weight, const T *grad, const size_t size, const T a, const T mu, const T weight_decay, T *vt) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size) {
        T w = weight[index];
        T _vt = vt[index];
        T e = grad[index];

        _vt = mu * _vt - (a * (e + weight_decay * w));

        w = w + _vt;

        weight[index] = w;
        vt[index] = _vt;
    }
}

template<typename T>

__device__ void update_with_adagrad(T *weight, const T *grad, const size_t size,
                                    const T a, const T weight_decay, const T eps, T *gt) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size) {
        T w = weight[index];
        T _gt = gt[index];
        T e = grad[index];

        e += weight_decay * w;

        _gt = _gt + e * e;

        w = w - a * (e / (_sqrt(_gt) + eps));

        weight[index] = w;
        gt[index] = _gt;
    }
}

template<typename T>

__device__ void update_with_rmsprop(T *weight, const T *grad, const size_t size,
                                    const T lr, const T weight_decay, const T alpha, const T mu,
                                    const T eps, T *gt, T *bt) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size) {
        T w = weight[index];
        T _gt = gt[index];
        T _bt = bt[index];

        T e = grad[index];

        e += weight_decay * w;

        _gt = alpha * _gt + (1 - alpha) * e * e;

        _bt += mu * _bt + e / (_sqrt(_gt) + eps);
        w = w - lr * _bt;

        weight[index] = w;
        gt[index] = _gt;
        bt[index] = _bt;
    }
}

template<typename T>

__device__ void update_with_adam(T *weight, const T *grad, const size_t size,
                                 const T a, const T weight_decay, const T eps,
                                 T *mt, T *vt, const T b1, const T b2, T b1t, T b2t) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < size) {
        T w = weight[index];
        T _mt = mt[index];
        T _vt = vt[index];
        T e = grad[index];

        e += weight_decay * w;

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
	__global__ void sigmoid_forward_float(const float *input, float *output, const size_t units_len, const size_t batch_len) {
        sigmoid_forward(input,output,units_len,batch_len);
	}

	__global__ void relu_forward_float(const float *input, float *output, const size_t units_len, const size_t batch_len) {
        relu_forward(input,output,units_len,batch_len);
    }

	__global__ void swish_forward_float(const float *input, float *output, const size_t units_len, const size_t batch_len) {
        swish_forward(input,output,units_len,batch_len);
    }

	__global__ void tanh_forward_float(const float *input, float *output, const size_t units_len, const size_t batch_len) {
        tanh_forward(input,output,units_len,batch_len);
    }

	__global__ void softmax_forward_float(const float *input, float *output, const size_t units_len, const size_t batch_len) {
        softmax_forward(input,output,units_len,batch_len);
    }

	__global__ void sigmoid_backward_float(const float * o, const float *u, const float *loss, float *output, const size_t units_len, const size_t batch_len) {
        sigmoid_backward(o,u,loss,output,units_len,batch_len);
    }

	__global__ void relu_backward_float(const float *o, const float *u, const float *loss, float *output, const size_t units_len, const size_t batch_len) {
        relu_backward(o,u,loss,output,units_len,batch_len);
    }

	__global__ void swish_backward_float(const float *o, const float *u, const float *loss, float *output, const size_t units_len, const size_t batch_len) {
        swish_backward(o,u,loss,output,units_len,batch_len);
    }

	__global__ void tanh_backward_float(const float *o, const float *u, const float *loss, float *output, const size_t units_len, const size_t batch_len) {
        tanh_backward(o,u,loss,output,units_len,batch_len);
    }

	__global__ void softmax_backward_float(const float *o, const float *u, const float *loss, float *output, const size_t units_len, const size_t batch_len) {
        softmax_backward(o,u,loss,output,units_len,batch_len);
    }
	__global__ void sigmoid_forward_double(const double *input, double *output, const size_t units_len, const size_t batch_len) {
        sigmoid_forward(input,output,units_len,batch_len);
	}

	__global__ void relu_forward_double(const double *input, double *output, const size_t units_len, const size_t batch_len) {
        relu_forward(input,output,units_len,batch_len);
    }

	__global__ void swish_forward_double(const double *input, double *output, const size_t units_len, const size_t batch_len) {
        swish_forward(input,output,units_len,batch_len);
    }

	__global__ void tanh_forward_double(const double *input, double *output, const size_t units_len, const size_t batch_len) {
        tanh_forward(input,output,units_len,batch_len);
    }

	__global__ void softmax_forward_double(const double *input, double *output, const size_t units_len, const size_t batch_len) {
        softmax_forward(input,output,units_len,batch_len);
    }

	__global__ void sigmoid_backward_double(const double *o, const double *u, const double *loss, double *output, const size_t units_len, const size_t batch_len) {
        sigmoid_backward(o,u,loss,output,units_len,batch_len);
    }

	__global__ void relu_backward_double(const double *o, const double *u, const double *loss, double *output, const size_t units_len, const size_t batch_len) {
        relu_backward(o,u,loss,output,units_len,batch_len);
    }

	__global__ void swish_backward_double(const double *o, const double *u, const double *loss, double *output, const size_t units_len, const size_t batch_len) {
        swish_backward(o,u,loss,output,units_len,batch_len);
    }

	__global__ void tanh_backward_double(const double *o, const double *u, const double *loss, double *output, const size_t units_len, const size_t batch_len) {
        tanh_backward(o,u,loss,output,units_len,batch_len);
    }

	__global__ void softmax_backward_double(const double * o, const double *u, const double *loss, double *output, const size_t units_len, const size_t batch_len) {
        softmax_backward(o,u,loss,output,units_len,batch_len);
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

    __global__ void loss_linear_batch_by_canonical_link_float(const float *expected, const float *actual, float *output, const int nlen, const int batch_size) {
        loss_linear_batch_by_canonical_link(expected,actual,output,nlen,batch_size);
    }

    __global__ void loss_linear_batch_by_canonical_link_double(const double *expected, const double *actual, double *output, const int nlen, const int batch_size) {
        loss_linear_batch_by_canonical_link(expected,actual,output,nlen,batch_size);
    }

    __global__ void loss_linear_batch_mse_derive_float(const float *t, const float *r, float* output, const int nlen, const int batch_size) {
        loss_linear_batch_mse_derive(t,r,output,nlen,batch_size);
    }

    __global__ void loss_linear_batch_mse_derive_double(const double *t, const double *r, double *output, const int nlen, const int batch_size) {
        loss_linear_batch_mse_derive(t,r,output,nlen,batch_size);
    }

    __global__ void loss_linear_batch_cross_entropy_derive_float(const float *t, const float *r, float* output, const int nlen, const int batch_size) {
        loss_linear_batch_cross_entropy_derive(t,r,output,nlen,batch_size);
    }

    __global__ void loss_linear_batch_cross_entropy_derive_double(const double *t, const double *r, double *output, const int nlen, const int batch_size) {
        loss_linear_batch_cross_entropy_derive(t,r,output,nlen,batch_size);
    }

    __global__ void loss_linear_batch_cross_entropy_multiclass_derive_float(const float *t, const float *r, float* output, const int nlen, const int batch_size) {
        loss_linear_batch_cross_entropy_multiclass_derive(t,r,output,nlen,batch_size);
    }

    __global__ void loss_linear_batch_cross_entropy_multiclass_derive_double(const double *t, const double *r, double *output, const int nlen, const int batch_size) {
        loss_linear_batch_cross_entropy_multiclass_derive(t,r,output,nlen,batch_size);
    }
    __global__ void update_with_sgd_float(float *weight, const float *grad, const size_t size, const float a, const float weight_decay) {
        update_with_sgd(weight,grad,size,a,weight_decay);
    }
    __global__ void update_with_sgd_double(double *weight, const double *grad, const size_t size, const double a, const double weight_decay) {
        update_with_sgd(weight,grad,size,a,weight_decay);
    }

    __global__ void update_with_momentum_sgd_float(float *weight, const float *grad, const size_t size, const float a, const float mu, const float weight_decay, float *vt) {
        update_with_momentum_sgd(weight,grad,size,a,mu,weight_decay,vt);
    }

    __global__ void update_with_momentum_sgd_double(double *weight, const double *grad, const size_t size, const double a, const double mu, const double weight_decay, double *vt) {
        update_with_momentum_sgd(weight,grad,size,a,mu,weight_decay,vt);
    }

    __global__ void update_with_adagrad_float(float *weight, const float *grad, const size_t size,
                                              const float a, const float weight_decay, const float eps, float *gt) {
        update_with_adagrad(weight,grad,size,a,weight_decay,eps,gt);
    }

    __global__ void update_with_adagrad_double(double *weight, const double *grad, const size_t size,
                                               const double a, const double weight_decay, const double eps, double *gt) {
        update_with_adagrad(weight,grad,size,a,weight_decay,eps,gt);
    }

    __global__ void update_with_rmsprop_float(float *weight, const float *grad, const size_t size,
                                              const float lr, const float weight_decay,
                                              const float alpha, const float mu,
                                              const float eps,
                                              float *gt, float *bt) {
        update_with_rmsprop(weight,grad,size,lr,alpha,mu,weight_decay,eps,gt,bt);
    }

    __global__ void update_with_rmsprop_double(double *weight, const double *grad, const size_t size,
                                               const double lr, const double weight_decay,
                                               const double alpha, const double mu,
                                               const double eps,
                                               double *gt, double *bt) {
        update_with_rmsprop(weight,grad,size,lr,alpha,mu,weight_decay,eps,gt,bt);
    }

    __global__ void update_with_adam_float(float *weight, const float *grad, const size_t size,
                                           const float a, const float weight_decay,
                                           const float eps, float *mt, float *vt, const float b1, const float b2, float b1t, float b2t) {
        update_with_adam(weight,grad,size,a,weight_decay,eps,mt,vt,b1,b2,b1t,b2t);
    }

    __global__ void update_with_adam_double(double *weight, const double *grad, const size_t size,
                                            const double a, const double weight_decay,
                                            const double eps, double *mt, double *vt, const double b1, const double b2, double b1t, double b2t) {
        update_with_adam(weight,grad,size,a,weight_decay,eps,mt,vt,b1,b2,b1t,b2t);
    }

    __global__ void forward_diff_linear_float(const size_t *indexes, const float *input, const float *units, float *output, const size_t output_size, const size_t diff_len) {
        forward_diff_linear(indexes,input,units,output,output_size,diff_len);
    }

    __global__ void forward_diff_linear_double(const size_t *indexes, const double *input, const double *units, double *output, const size_t output_size, const size_t diff_len) {
        forward_diff_linear(indexes,input,units,output,output_size,diff_len);
    }
}