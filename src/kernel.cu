extern "C" {
	__global__ void sigmoid_forward_float(float *input_output, const size_t units_len, const size_t batch_len) {
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

        if (index < units_len && batch_index < batch_len) {
            size_t i = batch_index == 0 ? index : batch_index * units_len + index;

            float x = input_output[i];

            x = 1.0 / (1.0 + __expf(-x));

            input_output[i] = x;
        }
	}

	__global__ void relu_forward_float(float *input_output, const size_t units_len, const size_t batch_len, const float *alpha, const float *sum) {
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

        if (index < units_len && batch_index < batch_len) {
            size_t i = batch_index == 0 ? index : batch_index * units_len + index;

            if (input_output[i] < 0.0) {
                input_output[i] = 0.0;
            }
        }
    }

	__global__ void swish_forwaxd_float(float *input_output, const size_t units_len, const size_t batch_len, const float *alpha, const float *sum) {
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

        if (index < units_len && batch_index < batch_len) {
            size_t i = batch_index == 0 ? index : batch_index * units_len + index;

            float x = input_output[i];

            x = x * (1.0 / (1.0 + __expf(-x)));
            input_output[i] = x;
        }
    }

	__global__ void tanh_forward_float(float *input_output, const size_t units_len, const size_t batch_len, const float *alpha, const float *sum) {
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

        if (index < units_len && batch_index < batch_len) {
            size_t i = batch_index == 0 ? index : batch_index * units_len + index;

            float x = input_output[i];

            input_output[i] = (__expf(x) - __expf(-x)) / (__expf(x) + __expf(-x));
        }
    }

	__global__ void softmax_forward_float(float *input_output, const size_t units_len, const size_t batch_len, const float *alpha, const float *sum) {
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

        if (index < units_len && batch_index < batch_len) {
            size_t i = batch_index == 0 ? index : batch_index * units_len + index;

            float number = __expf(input_output[i] - alpha[batch_index]);
            float x = number / sum[batch_index];

            input_output[i] = x;
        }
    }

    __global__ void softmax_preprocessing_float(const float *input, const size_t units_len, const size_t batch_len, float *alpha, float *sum) {
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

        if (index < units_len && batch_index < batch_len) {
            float a = 0.0;
            float s = 0.0;

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

	__global__ void sigmoid_backward_float(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

        if (index < units_len && batch_index < batch_len) {
            size_t i = batch_index == 0 ? index : batch_index * units_len + index;

            float e = 1.0 / (1.0 + __expf(-u[i]));
            float x = e * (1.0 - e);

            loss[i] = x * loss[i];
        }
    }

	__global__ void relu_backward_float(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

        if (index < units_len && batch_index < batch_len) {
            size_t i = batch_index == 0 ? index : batch_index * units_len + index;

            if (u[i] <= 0.0) {
                loss[i] = 0.0;
            }
        }
    }

	__global__ void swish_backward_float(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

        if (index < units_len && batch_index < batch_len) {
            size_t i = batch_index == 0 ? index : batch_index * units_len + index;

            float x = u[i];

            x = x * (1.0 / (1.0 + __expf(-x))) +
                    (1.0 / (1.0 + __expf(-x))) * (1.0 - (x * (1.0 / (1.0 + __expf(-x)))));

            loss[i] = x;
        }
    }

	__global__ void tanh_backward_float(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

        if (index < units_len && batch_index < batch_len) {
            size_t i = batch_index == 0 ? index : batch_index * units_len + index;

            float x = u[i];
            float e = (__expf(x) - __expf(-x)) / (__expf(x) + __expf(-x));

            x = 1.0 - e * e;

            loss[i] = x * loss[i];
        }
    }

	__global__ void softmax_backward_float(float *u, float *loss, const size_t units_len, const size_t batch_len) {
        size_t index = blockDim.x * blockIdx.x + threadIdx.x;
        size_t batch_index = blockDim.y * blockIdx.y + threadIdx.y;

        if (index < units_len && batch_index < batch_len) {
            size_t i = batch_index == 0 ? index : batch_index * units_len + index;

            float x = u[i];
            x = x * (1.0 - x);

            loss[i] = loss[i] * x;
        }
    }
}