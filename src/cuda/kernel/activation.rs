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
    fn softmax_preprocessing_float(input: *const f32, len: size_t, batch_len: size_t, alpha: *mut f32, sum: *mut f32) -> c_void;
}
pub struct ActivationForwardArgs<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    input_output: &'a mut T,
    units_len: usize,
    batch_len: usize,
    v:PhantomData<V>
}
impl<'a,T,V> ActivationForwardArgs<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new(input_output:&'a mut T,units_len:usize,batch_len:usize) -> ActivationForwardArgs<'a,T,V> {
        ActivationForwardArgs {
            input_output: input_output,
            units_len: units_len,
            batch_len: batch_len,
            v:PhantomData::<V>
        }
    }
}
impl<'a,T,V> KernelArgs for ActivationForwardArgs<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            self.input_output,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct ActivationBackwardArgs<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    u: &'a mut T,
    loss: &'a mut T,
    units_len: usize,
    batch_len: usize,
    v:PhantomData<V>
}
impl<'a,T,V> ActivationBackwardArgs<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new(u:&'a mut T,loss: &'a mut T,units_len:usize,batch_len:usize) -> ActivationBackwardArgs<'a,T,V> {
        ActivationBackwardArgs {
            u: u,
            loss: loss,
            units_len: units_len,
            batch_len: batch_len,
            v:PhantomData::<V>
        }
    }
}
impl<'a,T,V> KernelArgs for ActivationBackwardArgs<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            self.u,
            self.loss,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct SigmoidForward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>,
    v:PhantomData<V>
}
impl<'a,T,V> SigmoidForward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new() -> SigmoidForward<'a,T,V> {
        SigmoidForward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>,
            v: PhantomData::<V>
        }
    }
}
impl<'a,T> Kernel for SigmoidForward<'a,T,f32> where T: AsMutKernelPtr + 'a {
    const FUNC_PTR: *const c_void = sigmoid_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,T,f32>;
}
pub struct SigmoidBackward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>,
    v:PhantomData<V>
}
impl<'a,T,V> SigmoidBackward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new() -> SigmoidBackward<'a,T,V> {
        SigmoidBackward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>,
            v: PhantomData::<V>
        }
    }
}
impl<'a,T> Kernel for SigmoidBackward<'a,T,f32> where T: AsMutKernelPtr + 'a {
    const FUNC_PTR: *const c_void = sigmoid_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,T,f32>;
}
pub struct ReLuForward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>,
    v:PhantomData<V>
}
impl<'a,T,V> ReLuForward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new() -> ReLuForward<'a,T,V> {
        ReLuForward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>,
            v: PhantomData::<V>
        }
    }
}
impl<'a,T> Kernel for ReLuForward<'a,T,f32> where T: AsMutKernelPtr + 'a {
    const FUNC_PTR: *const c_void = relu_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,T,f32>;
}
pub struct ReLuBackward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>,
    v:PhantomData<V>
}
impl<'a,T,V> ReLuBackward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new() -> ReLuBackward<'a,T,V> {
        ReLuBackward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>,
            v: PhantomData::<V>
        }
    }
}
impl<'a,T> Kernel for ReLuBackward<'a,T,f32> where T: AsMutKernelPtr + 'a {
    const FUNC_PTR: *const c_void = relu_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,T,f32>;
}
pub struct SwishForward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>,
    v:PhantomData<V>
}
impl<'a,T,V> SwishForward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new() -> SwishForward<'a,T,V> {
        SwishForward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>,
            v: PhantomData::<V>
        }
    }
}
impl<'a,T> Kernel for SwishForward<'a,T,f32> where T: AsMutKernelPtr + 'a {
    const FUNC_PTR: *const c_void = swish_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,T,f32>;
}
pub struct SwishBackward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>,
    v:PhantomData<V>
}
impl<'a,T,V> SwishBackward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new() -> SwishBackward<'a,T,V> {
        SwishBackward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>,
            v: PhantomData::<V>
        }
    }
}
impl<'a,T> Kernel for SwishBackward<'a,T,f32> where T: AsMutKernelPtr + 'a {
    const FUNC_PTR: *const c_void = swish_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,T,f32>;
}
pub struct TanhForward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>,
    v:PhantomData<V>
}
impl<'a,T,V> TanhForward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new() -> TanhForward<'a,T,V> {
        TanhForward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>,
            v: PhantomData::<V>
        }
    }
}
impl<'a,T> Kernel for TanhForward<'a,T,f32> where T: AsMutKernelPtr + 'a {
    const FUNC_PTR: *const c_void = tanh_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,T,f32>;
}
pub struct TanhBackward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>,
    v:PhantomData<V>
}
impl<'a,T,V> TanhBackward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new() -> TanhBackward<'a,T,V> {
        TanhBackward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>,
            v: PhantomData::<V>
        }
    }
}
impl<'a,T> Kernel for TanhBackward<'a,T,f32> where T: AsMutKernelPtr + 'a {
    const FUNC_PTR: *const c_void = tanh_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,T,f32>;
}
pub struct SoftMaxForward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>,
    v:PhantomData<V>
}
impl<'a,T,V> SoftMaxForward<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new() -> SoftMaxForward<'a,T,V> {
        SoftMaxForward {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>,
            v: PhantomData::<V>
        }
    }
}
impl<'a,T> Kernel for SoftMaxForward<'a,T,f32> where T: AsMutKernelPtr + 'a {
    const FUNC_PTR: *const c_void = softmax_forward_float as *const c_void;
    type Args = ActivationSoftMaxForwardArgs<'a,T>;
}
pub struct ActivationSoftMaxForwardArgs<'a,T> where T: AsMutKernelPtr + 'a {
    input_output: &'a mut T,
    units_len: usize,
    batch_len: usize,
    alpha: &'a mut T,
    sum: &'a mut T
}
impl<'a,T> ActivationSoftMaxForwardArgs<'a,T> where T: AsMutKernelPtr + 'a {
    pub fn new(input_output:&'a mut T,units_len:usize,batch_len:usize,alpha:&'a mut T,sum:&'a mut T)
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
impl<'a,T> KernelArgs for ActivationSoftMaxForwardArgs<'a,T> where T: AsMutKernelPtr + 'a {
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
pub struct SoftMaxPreprocessing<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    l:PhantomData<&'a ()>,
    t:PhantomData<T>,
    v:PhantomData<V>
}
impl<'a,T,V> SoftMaxPreprocessing<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new() -> SoftMaxPreprocessing<'a,T,V> {
        SoftMaxPreprocessing {
            l: PhantomData::<&'a ()>,
            t: PhantomData::<T>,
            v: PhantomData::<V>
        }
    }
}
impl<'a,T> Kernel for SoftMaxPreprocessing<'a,T,f32> where T: AsMutKernelPtr + 'a {
    const FUNC_PTR: *const c_void = softmax_preprocessing_float as *const c_void;
    type Args = ActivationSoftMaxPreprocessingArgs<'a,T,f32>;
}
pub struct ActivationSoftMaxPreprocessingArgs<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    input: &'a mut T,
    units_len: usize,
    batch_len: usize,
    alpha: &'a mut CudaPtr<V>,
    sum: &'a mut CudaPtr<V>
}
impl<'a,T,V> ActivationSoftMaxPreprocessingArgs<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
    pub fn new(input:&'a mut T,units_len:usize,batch_len:usize,alpha:&'a mut CudaPtr<V>,sum:&'a mut CudaPtr<V>)
               -> ActivationSoftMaxPreprocessingArgs<'a,T,V> {

        ActivationSoftMaxPreprocessingArgs {
            input: input,
            units_len: units_len,
            batch_len: batch_len,
            alpha,
            sum
        }
    }
}
impl<'a,T,V> KernelArgs for ActivationSoftMaxPreprocessingArgs<'a,T,V> where T: AsMutKernelPtr + 'a, V: DataType {
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
