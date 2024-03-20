//! This module is related to the cuda implementation of the activation function

use std::marker::PhantomData;
use libc::{c_void, size_t};
use crate::cuda::{AsKernelPtr, CudaPtr, DataTypeInfo, Kernel, KernelArgs};

extern "C" {
    fn sigmoid_forward_float(input_output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn relu_forward_float(input_output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn swish_forward_float(input_output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn tanh_forward_float(input_output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn softmax_forward_float(input_output: *mut f32, len: size_t, batch_len: size_t) -> c_void;
    fn sigmoid_backward_float(o: *const f32, u: *const f32, loss: *mut f32, units_len: size_t, batch_len: size_t) -> c_void;
    fn relu_backward_float(o: *const f32, u: *const f32, loss: *mut f32, units_len: size_t, batch_len: size_t) -> c_void;
    fn swish_backward_float(o: *const f32, u: *const f32, loss: *mut f32, units_len: size_t, batch_len: size_t) -> c_void;
    fn tanh_backward_float(o: *const f32, u: *const f32, loss: *mut f32, units_len: size_t, batch_len: size_t) -> c_void;
    fn softmax_backward_float(o: *const f32, u: *const f32, loss: *mut f32, units_len: size_t, batch_len: size_t) -> c_void;
    fn sigmoid_forward_double(input_output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn relu_forward_double(input_output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn swish_forward_double(input_output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn tanh_forward_double(input_output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn softmax_forward_double(input_output: *mut f64, len: size_t, batch_len: size_t) -> c_void;
    fn sigmoid_backward_double(o: *const f64, u: *const f64, loss: *mut f64, units_len: size_t, batch_len: size_t) -> c_void;
    fn relu_backward_double(o: *const f64, u: *const f64, loss: *mut f64, units_len: size_t, batch_len: size_t) -> c_void;
    fn swish_backward_double(o: *const f64, u: *const f64, loss: *mut f64, units_len: size_t, batch_len: size_t) -> c_void;
    fn tanh_backward_double(o: *const f64, u: *const f64, loss: *mut f64, units_len: size_t, batch_len: size_t) -> c_void;
    fn softmax_backward_double(o: *const f64, u: *const f64, loss: *mut f64, units_len: size_t, batch_len: size_t) -> c_void;
}
/// Defines the list of passed to the cuda kernel function for the arguments of the activation function.
pub struct ActivationForwardArgs<T> where T: DataTypeInfo {
    /// Input buffer (shared with output buffer)
    pub input_output: CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list at the time of activation function forward.
impl<T> ActivationForwardArgs<T> where T: DataTypeInfo {
    /// Create a ActivationForwardArgs instance
    /// # Arguments
    /// * `input_output` - Input buffer (shared with output buffer)
    /// * `units_len` - count of inputs and outputs of linear layer weights
    /// * `batch_lne` - batches count
    pub fn new(input_output:CudaPtr<T>,units_len:usize,batch_len:usize) -> ActivationForwardArgs<T> {
        ActivationForwardArgs {
            input_output: input_output,
            units_len: units_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for ActivationForwardArgs<T> where T: DataTypeInfo {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.input_output,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
/// Create an instance of an object representing the argument list during error back propagation of the activation function.
pub struct ActivationBackwardArgs<T> where T: DataTypeInfo {
    o: CudaPtr<T>,
    u: CudaPtr<T>,
    /// loss value
    pub loss: CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the list of arguments during error back propagation of the activation function.
impl<T> ActivationBackwardArgs<T> where T: DataTypeInfo {
    /// Create a ActivationBackwardArgs instance
    /// # Arguments
    /// * `u` - Input values from upper layers
    /// * `loss` - loss value
    /// * `units_len` - count of inputs and outputs of linear layer weights
    /// * `batch_len` - batch count
    pub fn new(o:CudaPtr<T>,u:CudaPtr<T>,loss: CudaPtr<T>,units_len:usize,batch_len:usize) -> ActivationBackwardArgs<T> {
        ActivationBackwardArgs {
            o: o,
            u: u,
            loss: loss,
            units_len: units_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for ActivationBackwardArgs<T> where T: DataTypeInfo {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.o,
            &mut self.u,
            &mut self.loss,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
/// Sigmoid activation function implementation
pub struct SigmoidForward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SigmoidForward<T> where T: DataTypeInfo {
    /// Create a SigmoidForward instance
    pub fn new() -> SigmoidForward<T> {
        SigmoidForward {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for SigmoidForward<f32> {
    const FUNC_PTR: *const c_void = sigmoid_forward_float as *const c_void;
    type Args = ActivationForwardArgs<f32>;
}
impl Kernel for SigmoidForward<f64> {
    const FUNC_PTR: *const c_void = sigmoid_forward_double as *const c_void;
    type Args = ActivationForwardArgs<f64>;
}
/// Implementation of derivatives of the sigmoid activation function
pub struct SigmoidBackward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SigmoidBackward<T> where T: DataTypeInfo {
    /// Create a SigmoidBackward instance
    pub fn new() -> SigmoidBackward<T> {
        SigmoidBackward {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for SigmoidBackward<f32> {
    const FUNC_PTR: *const c_void = sigmoid_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<f32>;
}
impl Kernel for SigmoidBackward<f64> {
    const FUNC_PTR: *const c_void = sigmoid_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<f64>;
}
/// ReLu activation function implementation activation function implementation
pub struct ReLuForward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> ReLuForward<T> where T: DataTypeInfo {
    /// Create a ReLuForward instance
    pub fn new() -> ReLuForward<T> {
        ReLuForward {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for ReLuForward<f32> {
    const FUNC_PTR: *const c_void = relu_forward_float as *const c_void;
    type Args = ActivationForwardArgs<f32>;
}
impl Kernel for ReLuForward<f64> {
    const FUNC_PTR: *const c_void = relu_forward_double as *const c_void;
    type Args = ActivationForwardArgs<f64>;
}
/// Implementation of derivatives of the ReLu activation function
pub struct ReLuBackward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> ReLuBackward<T> where T: DataTypeInfo {
    /// Create a ReLuBackward instance
    pub fn new() -> ReLuBackward<T> {
        ReLuBackward {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for ReLuBackward<f32> {
    const FUNC_PTR: *const c_void = relu_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<f32>;
}
impl Kernel for ReLuBackward<f64> {
    const FUNC_PTR: *const c_void = relu_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<f64>;
}
/// Swish activation function implementation
pub struct SwishForward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SwishForward<T> where T: DataTypeInfo {
    /// Create a SwishForward instance
    pub fn new() -> SwishForward<T> {
        SwishForward {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for SwishForward<f32> {
    const FUNC_PTR: *const c_void = swish_forward_float as *const c_void;
    type Args = ActivationForwardArgs<f32>;
}
impl Kernel for SwishForward<f64> {
    const FUNC_PTR: *const c_void = swish_forward_double as *const c_void;
    type Args = ActivationForwardArgs<f64>;
}
/// Implementation of derivatives of the Swish activation function
pub struct SwishBackward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SwishBackward<T> where T: DataTypeInfo {
    /// Create a SwishBackward instance
    pub fn new() -> SwishBackward<T> {
        SwishBackward {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for SwishBackward<f32> {
    const FUNC_PTR: *const c_void = swish_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<f32>;
}
impl Kernel for SwishBackward<f64> {
    const FUNC_PTR: *const c_void = swish_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<f64>;
}
/// Tanh activation function implementation
pub struct TanhForward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> TanhForward<T> where T: DataTypeInfo {
    /// Create a TanhForward instance
    pub fn new() -> TanhForward<T> {
        TanhForward {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for TanhForward<f32> {
    const FUNC_PTR: *const c_void = tanh_forward_float as *const c_void;
    type Args = ActivationForwardArgs<f32>;
}
impl Kernel for TanhForward<f64> {
    const FUNC_PTR: *const c_void = tanh_forward_double as *const c_void;
    type Args = ActivationForwardArgs<f64>;
}
/// Implementation of derivatives of the Tanh activation function
pub struct TanhBackward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> TanhBackward<T> where T: DataTypeInfo {
    /// Create a TanhBackward instance
    pub fn new() -> TanhBackward<T> {
        TanhBackward {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for TanhBackward<f32> {
    const FUNC_PTR: *const c_void = tanh_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<f32>;
}
impl Kernel for TanhBackward<f64> {
    const FUNC_PTR: *const c_void = tanh_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<f64>;
}
/// SoftMax activation function implementation
pub struct SoftMaxForward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SoftMaxForward<T> where T: DataTypeInfo {
    /// Create a SoftMaxForward instance
    pub fn new() -> SoftMaxForward<T> {
        SoftMaxForward {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for SoftMaxForward<f32> {
    const FUNC_PTR: *const c_void = softmax_forward_float as *const c_void;
    type Args = ActivationForwardArgs<f32>;
}
impl Kernel for SoftMaxForward<f64> {
    const FUNC_PTR: *const c_void = softmax_forward_double as *const c_void;
    type Args = ActivationForwardArgs<f64>;
}
/// Implementation of derivatives of the softmax activation function
pub struct SoftMaxBackward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SoftMaxBackward<T> where T: DataTypeInfo {
    /// Create a SoftMaxForward instance
    pub fn new() -> SoftMaxBackward<T> {
        SoftMaxBackward {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for SoftMaxBackward<f32> {
    const FUNC_PTR: *const c_void = softmax_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<f32>;
}
impl Kernel for SoftMaxBackward<f64> {
    const FUNC_PTR: *const c_void = softmax_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<f64>;
}
