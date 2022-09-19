__device__ float _exp(float x) {
    return __expf(x);
}

__device__ double _exp(double x) {
    return exp(x);
}

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

__device__ void softmax_forward(T *input_output, const size_t units_len, const size_t batch_len, const T *alpha, const T *sum) {
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

    if (index < units_len && batch_index < batch_len) {
        size_t i = batch_index == 0 ? index : batch_index * units_len + index;

        T number = _exp(input_output[i] - alpha[batch_index]);
        T x = number / sum[batch_index];

        input_output[i] = x;
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

	__global__ void softmax_forward_float(float *input_output, const size_t units_len, const size_t batch_len, const float *alpha, const float *sum) {
        softmax_forward(input_output,units_len,batch_len,alpha,sum);
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

	__global__ void softmax_forward_double(float *input_output, const size_t units_len, const size_t batch_len, const float *alpha, const float *sum) {
        softmax_forward(input_output,units_len,batch_len,alpha,sum);
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
}