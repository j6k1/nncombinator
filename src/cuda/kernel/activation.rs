//! This module is related to the cuda implementation of the activation function

use std::marker::PhantomData;
use libc::{c_void, size_t};
use crate::cuda::{AsKernelPtr, CudaConstPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, DataTypeInfo, Kernel, KernelArgs};
use crate::ope::UnitValue;

extern "C" {
    fn sigmoid_forward_float(input: *const f32, output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn relu_forward_float(input: *const f32, output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn swish_forward_float(input: *const f32, output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn tanh_forward_float(input: *const f32, output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn softmax_forward_float(input: *const f32, output: *mut f32, len: size_t, batch_size: size_t) -> c_void;
    fn sigmoid_backward_float(o: *const f32, u: *const f32, loss: *const f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn relu_backward_float(o: *const f32, u: *const f32, loss: *const f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn swish_backward_float(o: *const f32, u: *const f32, loss: *const f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn tanh_backward_float(o: *const f32, u: *const f32, loss: *const f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn softmax_backward_float(o: *const f32, u: *const f32, loss: *const f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn sigmoid_forward_double(input: *const f64, output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn relu_forward_double(input: *const f64, output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn swish_forward_double(input: *const f64, output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn tanh_forward_double(input: *const f64, output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn softmax_forward_double(input: *const f64, output: *mut f64, len: size_t, batch_size: size_t) -> c_void;
    fn sigmoid_backward_double(o: *const f64, u: *const f64, loss: *const f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
    fn relu_backward_double(o: *const f64, u: *const f64, loss: *const f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
    fn swish_backward_double(o: *const f64, u: *const f64, loss: *const f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
    fn tanh_backward_double(o: *const f64, u: *const f64, loss: *const f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
    fn softmax_backward_double(o: *const f64, u: *const f64, loss: *const f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
}
/// Defines the list of passed to the cuda kernel function for the arguments of the activation function.
pub struct ActivationForwardArgs<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    input: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// Output buffer
    pub output: CudaTensor1dPtr<T,N>,
    units_len: usize,
    batch_size: usize,
}
/// Create an instance of an object representing the argument list at the time of activation function forward.
impl<'a,T,const N:usize> ActivationForwardArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a ActivationForwardArgs instance
    /// # Arguments
    /// * `input` - Input buffer
    /// * `output` - Output buffer
    pub fn new(input:&'a CudaTensor1dPtrView<'a,T,N>,output:CudaTensor1dPtr<T,N>) -> ActivationForwardArgs<'a,T,N> {
        ActivationForwardArgs {
            input: CudaConstPtr::new(input),
            output: output,
            units_len: N,
            batch_size: 1
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for ActivationForwardArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.input,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_size
        ]
    }
}
/// Create an instance of an object representing the argument list during error back propagation of the activation function.
pub struct ActivationBackwardArgs<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    o: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    u: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    loss: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// Output of error back propagation
    pub output: CudaTensor1dPtr<T,N>,
    units_len: usize,
    batch_size: usize,
}
/// Create an instance of an object representing the list of arguments during error back propagation of the activation function.
impl<'a,T,const N:usize> ActivationBackwardArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a ActivationBackwardArgs instance
    /// # Arguments
    /// * `o` - Output values
    /// * `u` - Input values from upper layers
    /// * `loss` - loss value
    /// * `output` - Output of error back propagation
    pub fn new(o: &'a CudaTensor1dPtrView<'a,T,N>,
               u: &'a CudaTensor1dPtrView<'a,T,N>,
               loss: &'a CudaTensor1dPtrView<'a,T,N>,
               output: CudaTensor1dPtr<T,N>) -> ActivationBackwardArgs<'a,T,N> {
        ActivationBackwardArgs {
            o: CudaConstPtr::new(o),
            u: CudaConstPtr::new(u),
            loss: CudaConstPtr::new(loss),
            output: output,
            units_len: N,
            batch_size: 1
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for ActivationBackwardArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.o,
            &mut self.u,
            &mut self.loss,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_size
        ]
    }
}
/// Defines the list of arguments passed to the cuda kernel function as arguments
/// to the activation function during batch execution.
pub struct ActivationBatchForwardArgs<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    input: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    /// Output buffer
    pub output: CudaVec<T,CudaTensor1dPtr<T,N>>,
    units_len: usize,
    batch_size: usize,
}
/// Create an instance of an object representing the argument list
/// of the forward propagation of the activation function during batch execution.
impl<'a,T,const N:usize> ActivationBatchForwardArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a ActivationBatchForwardArgs instance
    /// # Arguments
    /// * `input` - Input buffer
    /// * `output` - Output buffer
    /// * `batch_size` - batches count
    pub fn new(input:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,output:CudaVec<T,CudaTensor1dPtr<T,N>>, batch_size: usize)
        -> ActivationBatchForwardArgs<'a,T,N> {
        ActivationBatchForwardArgs {
            input: CudaConstPtr::new(input),
            output: output,
            units_len: N,
            batch_size: batch_size
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for ActivationBatchForwardArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.input,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_size
        ]
    }
}
/// Create an instance of an object representing the list of arguments during error back propagation
/// of the activation function during batch execution.
pub struct ActivationBatchBackwardArgs<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    o: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    u: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    loss: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    /// Output of error back propagation
    pub output: CudaVec<T,CudaTensor1dPtr<T,N>>,
    units_len: usize,
    batch_size: usize,
}
/// Instantiate an object representing the list of arguments during error back propagation
/// of the activation function during batch execution.
impl<'a,T,const N:usize> ActivationBatchBackwardArgs<'a, T, N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a ActivationBatchBackwardArgs instance
    /// # Arguments
    /// * `o` - Output values
    /// * `u` - Input values from upper layers
    /// * `loss` - loss value
    /// * `output` - Output of error back propagation
    /// * `batch_size` - batch count
    pub fn new(o: &'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,
               u: &'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,
               loss: &'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,
               output: CudaVec<T,CudaTensor1dPtr<T,N>>,batch_size: usize) -> ActivationBatchBackwardArgs<'a, T, N> {
        ActivationBatchBackwardArgs {
            o: CudaConstPtr::new(o),
            u: CudaConstPtr::new(u),
            loss: CudaConstPtr::new(loss),
            output: output,
            units_len: N,
            batch_size: batch_size
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for ActivationBatchBackwardArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.o,
            &mut self.u,
            &mut self.loss,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_size
        ]
    }
}
/// Sigmoid activation function implementation
pub struct SigmoidForward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SigmoidForward<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SigmoidForward instance
    pub fn new() -> SigmoidForward<'a,T,N> {
        SigmoidForward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SigmoidForward<'a,f32,N> {
    const FUNC_PTR: *const c_void = sigmoid_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SigmoidForward<'a,f64,N> {
    const FUNC_PTR: *const c_void = sigmoid_forward_double as *const c_void;
    type Args = ActivationForwardArgs<'a,f64,N>;
}
/// Implementation of derivatives of the sigmoid activation function
pub struct SigmoidBackward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SigmoidBackward<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SigmoidBackward instance
    pub fn new() -> SigmoidBackward<'a,T,N> {
        SigmoidBackward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SigmoidBackward<'a,f32,N> {
    const FUNC_PTR: *const c_void = sigmoid_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SigmoidBackward<'a,f64,N> {
    const FUNC_PTR: *const c_void = sigmoid_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<'a,f64,N>;
}
/// Implementation of sigmoid activation functions for batch execution
pub struct SigmoidBatchForward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SigmoidBatchForward<'a, T, N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SigmoidForwardForBatch instance
    pub fn new() -> SigmoidBatchForward<'a, T, N> {
        SigmoidBatchForward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SigmoidBatchForward<'a, f32, N> {
    const FUNC_PTR: *const c_void = sigmoid_forward_float as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SigmoidBatchForward<'a, f64, N> {
    const FUNC_PTR: *const c_void = sigmoid_forward_double as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f64,N>;
}
/// Implement derivatives of the sigmoid activation function for batch execution
pub struct SigmoidBatchBackward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SigmoidBatchBackward<'a, T, N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SigmoidBackwardForBatch instance
    pub fn new() -> SigmoidBatchBackward<'a, T, N> {
        SigmoidBatchBackward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SigmoidBatchBackward<'a, f32, N> {
    const FUNC_PTR: *const c_void = sigmoid_backward_float as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SigmoidBatchBackward<'a, f64, N> {
    const FUNC_PTR: *const c_void = sigmoid_backward_double as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f64,N>;
}
/// ReLu activation function implementation activation function implementation
pub struct ReLuForward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> ReLuForward<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a ReLuForward instance
    pub fn new() -> ReLuForward<'a,T,N> {
        ReLuForward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for ReLuForward<'a,f32,N> {
    const FUNC_PTR: *const c_void = relu_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for ReLuForward<'a,f64,N> {
    const FUNC_PTR: *const c_void = relu_forward_double as *const c_void;
    type Args = ActivationForwardArgs<'a,f64,N>;
}
/// Implementation of derivatives of the ReLu activation function
pub struct ReLuBackward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> ReLuBackward<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a ReLuBackward instance
    pub fn new() -> ReLuBackward<'a,T,N> {
        ReLuBackward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for ReLuBackward<'a,f32,N> {
    const FUNC_PTR: *const c_void = relu_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for ReLuBackward<'a,f64,N> {
    const FUNC_PTR: *const c_void = relu_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<'a,f64,N>;
}
/// Implementation of ReLu activation functions for batch execution
pub struct ReLuBatchForward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> ReLuBatchForward<'a, T, N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a ReLuForwardBatch instance
    pub fn new() -> ReLuBatchForward<'a, T, N> {
        ReLuBatchForward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for ReLuBatchForward<'a, f32, N> {
    const FUNC_PTR: *const c_void = relu_forward_float as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for ReLuBatchForward<'a, f64, N> {
    const FUNC_PTR: *const c_void = relu_forward_double as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f64,N>;
}
/// Implement derivatives of the ReLu activation function for batch execution
pub struct ReLuBatchBackward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> ReLuBatchBackward<'a, T, N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a ReLuBackwardForBatch instance
    pub fn new() -> ReLuBatchBackward<'a, T, N> {
        ReLuBatchBackward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for ReLuBatchBackward<'a, f32, N> {
    const FUNC_PTR: *const c_void = relu_backward_float as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for ReLuBatchBackward<'a, f64, N> {
    const FUNC_PTR: *const c_void = relu_backward_double as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f64,N>;
}
/// Swish activation function implementation
pub struct SwishForward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SwishForward<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SwishForward instance
    pub fn new() -> SwishForward<'a,T,N> {
        SwishForward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SwishForward<'a,f32,N> {
    const FUNC_PTR: *const c_void = swish_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SwishForward<'a,f64,N> {
    const FUNC_PTR: *const c_void = swish_forward_double as *const c_void;
    type Args = ActivationForwardArgs<'a,f64,N>;
}
/// Implementation of derivatives of the Swish activation function
pub struct SwishBackward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SwishBackward<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SwishBackward instance
    pub fn new() -> SwishBackward<'a,T,N> {
        SwishBackward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SwishBackward<'a,f32,N> {
    const FUNC_PTR: *const c_void = swish_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SwishBackward<'a,f64,N> {
    const FUNC_PTR: *const c_void = swish_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<'a,f64,N>;
}
/// Implementation of Swish activation functions for batch execution
pub struct SwishBatchForward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SwishBatchForward<'a, T, N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SwishForwardForBatch instance
    pub fn new() -> SwishBatchForward<'a, T, N> {
        SwishBatchForward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SwishBatchForward<'a, f32, N> {
    const FUNC_PTR: *const c_void = swish_forward_float as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SwishBatchForward<'a, f64, N> {
    const FUNC_PTR: *const c_void = swish_forward_double as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f64,N>;
}
/// Implement derivatives of the Swish activation function for batch execution
pub struct SwishBatchBackward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SwishBatchBackward<'a, T, N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SwishBackwardForBatch instance
    pub fn new() -> SwishBatchBackward<'a, T, N> {
        SwishBatchBackward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SwishBatchBackward<'a, f32, N> {
    const FUNC_PTR: *const c_void = swish_backward_float as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SwishBatchBackward<'a, f64, N> {
    const FUNC_PTR: *const c_void = swish_backward_double as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f64,N>;
}
/// Tanh activation function implementation
pub struct TanhForward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> TanhForward<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a TanhForward instance
    pub fn new() -> TanhForward<'a,T,N> {
        TanhForward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for TanhForward<'a,f32,N> {
    const FUNC_PTR: *const c_void = tanh_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for TanhForward<'a,f64,N> {
    const FUNC_PTR: *const c_void = tanh_forward_double as *const c_void;
    type Args = ActivationForwardArgs<'a,f64,N>;
}
/// Implementation of derivatives of the Tanh activation function
pub struct TanhBackward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> TanhBackward<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a TanhBackward instance
    pub fn new() -> TanhBackward<'a,T,N> {
        TanhBackward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for TanhBackward<'a,f32,N> {
    const FUNC_PTR: *const c_void = tanh_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for TanhBackward<'a,f64,N> {
    const FUNC_PTR: *const c_void = tanh_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<'a,f64,N>;
}
/// Implementation of Tanh activation functions for batch execution
pub struct TanhBatchForward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> TanhBatchForward<'a, T, N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a TanhForwardForBatch instance
    pub fn new() -> TanhBatchForward<'a, T, N> {
        TanhBatchForward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for TanhBatchForward<'a, f32, N> {
    const FUNC_PTR: *const c_void = tanh_forward_float as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for TanhBatchForward<'a, f64, N> {
    const FUNC_PTR: *const c_void = tanh_forward_double as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f64,N>;
}
/// Implement derivatives of the Tanh activation function for batch execution
pub struct TanhBatchBackward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> TanhBatchBackward<'a, T, N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a TanhBackwardForBatch instance
    pub fn new() -> TanhBatchBackward<'a, T, N> {
        TanhBatchBackward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for TanhBatchBackward<'a, f32, N> {
    const FUNC_PTR: *const c_void = tanh_backward_float as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for TanhBatchBackward<'a, f64, N> {
    const FUNC_PTR: *const c_void = tanh_backward_double as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f64,N>;
}
/// SoftMax activation function implementation
pub struct SoftMaxForward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SoftMaxForward<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SoftMaxForward instance
    pub fn new() -> SoftMaxForward<'a,T,N> {
        SoftMaxForward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SoftMaxForward<'a,f32,N> {
    const FUNC_PTR: *const c_void = softmax_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SoftMaxForward<'a,f64,N> {
    const FUNC_PTR: *const c_void = softmax_forward_double as *const c_void;
    type Args = ActivationForwardArgs<'a,f64,N>;
}
/// Implementation of derivatives of the softmax activation function
pub struct SoftMaxBackward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SoftMaxBackward<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SoftMaxForward instance
    pub fn new() -> SoftMaxBackward<'a,T,N> {
        SoftMaxBackward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SoftMaxBackward<'a,f32,N> {
    const FUNC_PTR: *const c_void = softmax_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SoftMaxBackward<'a,f64,N> {
    const FUNC_PTR: *const c_void = softmax_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<'a,f64,N>;
}
/// Implementation of Softmax activation functions for batch execution
pub struct SoftMaxBatchForward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SoftMaxBatchForward<'a, T, N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SoftMaxForwardForBatch instance
    pub fn new() -> SoftMaxBatchForward<'a, T, N> {
        SoftMaxBatchForward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SoftMaxBatchForward<'a, f32, N> {
    const FUNC_PTR: *const c_void = softmax_forward_float as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SoftMaxBatchForward<'a, f64, N> {
    const FUNC_PTR: *const c_void = softmax_forward_double as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f64,N>;
}
/// Implement derivatives of the Softmax activation function for batch execution
pub struct SoftMaxBatchBackward<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> SoftMaxBatchBackward<'a, T, N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a SoftMaxForwardForBatch instance
    pub fn new() -> SoftMaxBatchBackward<'a, T, N> {
        SoftMaxBatchBackward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for SoftMaxBatchBackward<'a, f32, N> {
    const FUNC_PTR: *const c_void = softmax_backward_float as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for SoftMaxBatchBackward<'a, f64, N> {
    const FUNC_PTR: *const c_void = softmax_backward_double as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f64,N>;
}
