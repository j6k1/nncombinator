use std::marker::PhantomData;
use libc::{c_void, size_t};
use crate::cuda::{AsMutKernelPtr, CudaPtr, DataTypeInfo, Kernel, KernelArgs};

extern "C" {
    fn sigmoid_forward_float(input_output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn relu_forward_float(input_output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn swish_forward_float(input_output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn tanh_forward_float(input_output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn softmax_forward_float(input_output: *mut f32, len: size_t, batch_len: size_t, alpha: *const f32, sum: *const f32) -> c_void;
    fn sigmoid_backward_float(u: *const f32, loss: *mut f32, units_len: size_t, batch_len: size_t) -> c_void;
    fn relu_backward_float(u: *const f32, loss: *mut f32, units_len: size_t, batch_len: size_t) -> c_void;
    fn swish_backward_float(u: *const f32, loss: *mut f32, units_len: size_t, batch_len: size_t) -> c_void;
    fn tanh_backward_float(u: *const f32, loss: *mut f32, units_len: size_t, batch_len: size_t) -> c_void;
    fn softmax_backward_float(u: *const f32, loss: *mut f32, units_len: size_t, batch_len: size_t) -> c_void;
    fn softmax_preprocessing_float(input: *const f32, len: size_t, batch_len: size_t, alpha: *mut f32, sum: *mut f32) -> c_void;
    fn sigmoid_forward_double(input_output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn relu_forward_double(input_output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn swish_forward_double(input_output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn tanh_forward_double(input_output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn softmax_forward_double(input_output: *mut f64, len: size_t, batch_len: size_t, alpha: *const f64, sum: *const f64) -> c_void;
    fn sigmoid_backward_double(u: *const f64, loss: *mut f64, units_len: size_t, batch_len: size_t) -> c_void;
    fn relu_backward_double(u: *const f64, loss: *mut f64, units_len: size_t, batch_len: size_t) -> c_void;
    fn swish_backward_double(u: *const f64, loss: *mut f64, units_len: size_t, batch_len: size_t) -> c_void;
    fn tanh_backward_double(u: *const f64, loss: *mut f64, units_len: size_t, batch_len: size_t) -> c_void;
    fn softmax_backward_double(u: *const f64, loss: *mut f64, units_len: size_t, batch_len: size_t) -> c_void;
    fn softmax_preprocessing_double(input: *const f64, len: size_t, batch_len: size_t, alpha: *mut f64, sum: *mut f64) -> c_void;
}
pub struct ActivationForwardArgs<T> where T: DataTypeInfo {
    pub input_output: CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
}
impl<T> ActivationForwardArgs<T> where T: DataTypeInfo {
    pub fn new(input_output:CudaPtr<T>,units_len:usize,batch_len:usize) -> ActivationForwardArgs<T> {
        ActivationForwardArgs {
            input_output: input_output,
            units_len: units_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for ActivationForwardArgs<T> where T: DataTypeInfo {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.input_output,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct ActivationBackwardArgs<T> where T: DataTypeInfo {
    u: CudaPtr<T>,
    pub loss: CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
}
impl<T> ActivationBackwardArgs<T> where T: DataTypeInfo {
    pub fn new(u:CudaPtr<T>,loss: CudaPtr<T>,units_len:usize,batch_len:usize) -> ActivationBackwardArgs<T> {
        ActivationBackwardArgs {
            u: u,
            loss: loss,
            units_len: units_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for ActivationBackwardArgs<T> where T: DataTypeInfo {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.u,
            &mut self.loss,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct SigmoidForward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SigmoidForward<T> where T: DataTypeInfo {
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
pub struct SigmoidBackward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SigmoidBackward<T> where T: DataTypeInfo {
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
pub struct ReLuForward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> ReLuForward<T> where T: DataTypeInfo {
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
pub struct ReLuBackward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> ReLuBackward<T> where T: DataTypeInfo {
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
pub struct SwishForward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SwishForward<T> where T: DataTypeInfo {
    pub fn new() -> SwishForward<T> {
        SwishForward {
            t: PhantomData::<T>
        }
    }
}
impl<T> Kernel for SwishForward<T> where T: DataTypeInfo {
    const FUNC_PTR: *const c_void = swish_forward_float as *const c_void;
    type Args = ActivationForwardArgs<T>;
}
pub struct SwishBackward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SwishBackward<T> where T: DataTypeInfo {
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
pub struct TanhForward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> TanhForward<T> where T: DataTypeInfo {
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
pub struct TanhBackward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> TanhBackward<T> where T: DataTypeInfo {
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
pub struct SoftMaxForward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SoftMaxForward<T> where T: DataTypeInfo {
    pub fn new() -> SoftMaxForward<T> {
        SoftMaxForward {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for SoftMaxForward<f32> {
    const FUNC_PTR: *const c_void = softmax_forward_float as *const c_void;
    type Args = ActivationSoftMaxForwardArgs<f32>;
}
impl Kernel for SoftMaxForward<f64> {
    const FUNC_PTR: *const c_void = softmax_forward_double as *const c_void;
    type Args = ActivationSoftMaxForwardArgs<f64>;
}
pub struct ActivationSoftMaxForwardArgs<T> where T: DataTypeInfo {
    pub input_output: CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
    alpha: CudaPtr<T>,
    sum: CudaPtr<T>
}
impl<T> ActivationSoftMaxForwardArgs<T> where T: DataTypeInfo {
    pub fn new(input_output:CudaPtr<T>,units_len:usize,batch_len:usize,alpha:CudaPtr<T>,sum:CudaPtr<T>)
               -> ActivationSoftMaxForwardArgs<T> {

        ActivationSoftMaxForwardArgs {
            input_output: input_output,
            units_len: units_len,
            batch_len: batch_len,
            alpha,
            sum
        }
    }
}
impl<T> KernelArgs for ActivationSoftMaxForwardArgs<T> where T: DataTypeInfo {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.input_output,
            &mut self.units_len,
            &mut self.batch_len,
            &mut self.alpha,
            &mut self.sum
        ]
    }
}
pub struct SoftMaxPreprocessing<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SoftMaxPreprocessing<T> where T: DataTypeInfo {
    pub fn new() -> SoftMaxPreprocessing<T> {
        SoftMaxPreprocessing {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for SoftMaxPreprocessing<f32> {
    const FUNC_PTR: *const c_void = softmax_preprocessing_float as *const c_void;
    type Args = ActivationSoftMaxPreprocessingArgs<f32>;
}
impl Kernel for SoftMaxPreprocessing<f64> {
    const FUNC_PTR: *const c_void = softmax_preprocessing_float as *const c_void;
    type Args = ActivationSoftMaxPreprocessingArgs<f64>;
}
pub struct ActivationSoftMaxPreprocessingArgs<T> where T: DataTypeInfo {
    input: CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
    pub alpha: CudaPtr<T>,
    pub sum: CudaPtr<T>
}
impl<T> ActivationSoftMaxPreprocessingArgs<T> where T: DataTypeInfo {
    pub fn new(input:CudaPtr<T>,units_len:usize,batch_len:usize,alpha:CudaPtr<T>,sum:CudaPtr<T>)
               -> ActivationSoftMaxPreprocessingArgs<T> {

        ActivationSoftMaxPreprocessingArgs {
            input: input,
            units_len: units_len,
            batch_len: batch_len,
            alpha,
            sum
        }
    }
}
impl<T> KernelArgs for ActivationSoftMaxPreprocessingArgs<T> where T: DataTypeInfo {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.input,
            &mut self.units_len,
            &mut self.batch_len,
            &mut self.alpha,
            &mut self.sum
        ]
    }
}
pub struct SoftMaxBackward<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> SoftMaxBackward<T> where T: DataTypeInfo {
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
