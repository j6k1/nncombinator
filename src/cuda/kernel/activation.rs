use std::marker::PhantomData;
use libc::{c_void, size_t};
use crate::cuda::{AsMutKernelPtr, CudaPtr, Kernel, KernelArgs};

pub trait DataType {}

impl DataType for f32 {}

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
}
pub struct ActivationForwardArgs<'a,T> where T: DataType + 'a {
    input_output: &'a mut CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
}
impl<'a,T> ActivationForwardArgs<'a,T> where T: DataType + 'a {
    pub fn new(input_output:&'a mut CudaPtr<T>,units_len:usize,batch_len:usize) -> ActivationForwardArgs<'a,T> {
        ActivationForwardArgs {
            input_output: input_output,
            units_len: units_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T> KernelArgs for ActivationForwardArgs<'a,T> where T: DataType + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            self.input_output,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct ActivationBackwardArgs<'a,T> where T: DataType + 'a {
    u: &'a mut CudaPtr<T>,
    loss: &'a mut CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
}
impl<'a,T> ActivationBackwardArgs<'a,T> where T: DataType + 'a {
    pub fn new(u:&'a mut CudaPtr<T>,loss: &'a mut CudaPtr<T>,units_len:usize,batch_len:usize) -> ActivationBackwardArgs<'a,T> {
        ActivationBackwardArgs {
            u: u,
            loss: loss,
            units_len: units_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T> KernelArgs for ActivationBackwardArgs<'a,T> where T: DataType + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            self.u,
            self.loss,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct SigmoidForward<'a,T> where T: DataType + 'a {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>
}
impl<'a,T> SigmoidForward<'a,T> where T: DataType + 'a {
    pub fn new() -> SigmoidForward<'a,T> {
        SigmoidForward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>
        }
    }
}
impl<'a,T> Kernel for SigmoidForward<'a,T> where T: DataType + 'a {
    const FUNC_PTR: *const c_void = sigmoid_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,T>;
}
pub struct SigmoidBackward<'a,T> where T: DataType + 'a {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>
}
impl<'a,T> SigmoidBackward<'a,T> where T: DataType + 'a {
    pub fn new() -> SigmoidBackward<'a,T> {
        SigmoidBackward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>
        }
    }
}
impl<'a,T> Kernel for SigmoidBackward<'a,T> where T: DataType + 'a {
    const FUNC_PTR: *const c_void = sigmoid_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,T>;
}
pub struct ReLuForward<'a,T> where T: DataType + 'a {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>
}
impl<'a,T> ReLuForward<'a,T> where T: DataType + 'a {
    pub fn new() -> ReLuForward<'a,T> {
        ReLuForward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>
        }
    }
}
impl<'a,T> Kernel for ReLuForward<'a,T> where T: DataType + 'a {
    const FUNC_PTR: *const c_void = relu_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,T>;
}
pub struct ReLuBackward<'a,T> where T: DataType + 'a {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>
}
impl<'a,T> ReLuBackward<'a,T> where T: DataType + 'a {
    pub fn new() -> ReLuBackward<'a,T> {
        ReLuBackward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>
        }
    }
}
impl<'a,T> Kernel for ReLuBackward<'a,T> where T: DataType + 'a {
    const FUNC_PTR: *const c_void = relu_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,T>;
}
pub struct SwishForward<'a,T> where T: DataType + 'a {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>
}
impl<'a,T> SwishForward<'a,T> where T: DataType + 'a {
    pub fn new() -> SwishForward<'a,T> {
        SwishForward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>
        }
    }
}
impl<'a,T> Kernel for SwishForward<'a,T> where T: DataType + 'a {
    const FUNC_PTR: *const c_void = swish_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,T>;
}
pub struct SwishBackward<'a,T> where T: DataType + 'a {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>
}
impl<'a,T> SwishBackward<'a,T> where T: DataType + 'a {
    pub fn new() -> SwishBackward<'a,T> {
        SwishBackward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>
        }
    }
}
impl<'a,T> Kernel for SwishBackward<'a,T> where T: DataType + 'a {
    const FUNC_PTR: *const c_void = swish_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,T>;
}
pub struct TanhForward<'a,T> where T: DataType + 'a {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>
}
impl<'a,T> TanhForward<'a,T> where T: DataType + 'a {
    pub fn new() -> TanhForward<'a,T> {
        TanhForward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>
        }
    }
}
impl<'a,T> Kernel for TanhForward<'a,T> where T: DataType + 'a {
    const FUNC_PTR: *const c_void = tanh_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,T>;
}
pub struct TanhBackward<'a,T> where T: DataType + 'a {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>
}
impl<'a,T> TanhBackward<'a,T> where T: DataType + 'a {
    pub fn new() -> TanhBackward<'a,T> {
        TanhBackward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>
        }
    }
}
impl<'a,T> Kernel for TanhBackward<'a,T> where T: DataType + 'a {
    const FUNC_PTR: *const c_void = tanh_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,T>;
}
pub struct SoftMaxForward<'a,T> where T: DataType + 'a {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>
}
impl<'a,T> SoftMaxForward<'a,T> where T: DataType + 'a {
    pub fn new() -> SoftMaxForward<'a,T> {
        SoftMaxForward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>
        }
    }
}
impl<'a,T> Kernel for SoftMaxForward<'a,T> where T: DataType + 'a {
    const FUNC_PTR: *const c_void = softmax_forward_float as *const c_void;
    type Args = ActivationSoftMaxForwardArgs<'a,T>;
}
pub struct ActivationSoftMaxForwardArgs<'a,T> where T: DataType + 'a {
    input_output: &'a mut CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
    alpha: &'a mut CudaPtr<T>,
    sum: &'a mut CudaPtr<T>
}
impl<'a,T> ActivationSoftMaxForwardArgs<'a,T> where T: DataType + 'a {
    pub fn new(input_output:&'a mut CudaPtr<T>,units_len:usize,batch_len:usize,alpha:&'a mut CudaPtr<T>,sum:&'a mut CudaPtr<T>)
               -> ActivationSoftMaxForwardArgs<'a,T> {

        ActivationSoftMaxForwardArgs {
            input_output: input_output,
            units_len: units_len,
            batch_len: batch_len,
            alpha,
            sum
        }
    }
}
impl<'a,T> KernelArgs for ActivationSoftMaxForwardArgs<'a,T> where T: DataType + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            self.input_output,
            &mut self.units_len,
            &mut self.batch_len,
            self.alpha,
            self.sum
        ]
    }
}
pub struct SoftMaxPreprocessing<'a,T> where T: DataType + 'a {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>
}
impl<'a,T> SoftMaxPreprocessing<'a,T> where T: DataType + 'a {
    pub fn new() -> SoftMaxPreprocessing<'a,T> {
        SoftMaxPreprocessing {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>
        }
    }
}
impl<'a,T> Kernel for SoftMaxPreprocessing<'a,T> where T: DataType + 'a {
    const FUNC_PTR: *const c_void = softmax_preprocessing_float as *const c_void;
    type Args = ActivationSoftMaxPreprocessingArgs<'a,T>;
}
pub struct ActivationSoftMaxPreprocessingArgs<'a,T> where T: DataType + 'a {
    input: &'a mut CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
    alpha: &'a mut CudaPtr<T>,
    sum: &'a mut CudaPtr<T>
}
impl<'a,T> ActivationSoftMaxPreprocessingArgs<'a,T> where T: DataType + 'a {
    pub fn new(input:&'a mut CudaPtr<T>,units_len:usize,batch_len:usize,alpha:&'a mut CudaPtr<T>,sum:&'a mut CudaPtr<T>)
               -> ActivationSoftMaxPreprocessingArgs<'a,T> {

        ActivationSoftMaxPreprocessingArgs {
            input: input,
            units_len: units_len,
            batch_len: batch_len,
            alpha,
            sum
        }
    }
}
impl<'a,T> KernelArgs for ActivationSoftMaxPreprocessingArgs<'a,T> where T: DataType + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            self.input,
            &mut self.units_len,
            &mut self.batch_len,
            self.alpha,
            self.sum
        ]
    }
}
pub struct SoftMaxBackward<'a,T> where T: DataType + 'a {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>
}
impl<'a,T> SoftMaxBackward<'a,T> where T: DataType + 'a {
    pub fn new() -> SoftMaxBackward<'a,T> {
        SoftMaxBackward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>
        }
    }
}
impl<'a,T> Kernel for SoftMaxBackward<'a,T> where T: DataType + 'a {
    const FUNC_PTR: *const c_void = softmax_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,T>;
}
