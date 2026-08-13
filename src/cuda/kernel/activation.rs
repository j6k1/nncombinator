//! This module is related to the cuda implementation of the activation function

use std::ffi::c_uint;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem;
use cuda_runtime_sys::dim3;
use libc::{c_void, size_t};
use crate::cuda::{AsCudaMutPtr, AsKernelPtr, AsMutKernelPtr, CudaConstPtr, CudaMutPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, DataTypeInfo, Kernel, KernelArgs, KernelLaunchConfig};
use crate::cuda::allocator::CudaAllocator;

extern "C" {
    fn sigmoid_forward_float(input: *const f32, output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn relu_forward_float(input: *const f32, output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn clipped_relu_forward_float(input: *const f32, ceiling: f32, output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn leaky_relu_forward_float(input: *const f32, output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn swish_forward_float(input: *const f32, output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn tanh_forward_float(input: *const f32, output: *mut f32, len: size_t, units_len: size_t) -> c_void;
    fn softmax_forward_float(input: *const f32, output: *mut f32, len: size_t, batch_size: size_t) -> c_void;
    fn sigmoid_backward_float(o: *const f32, u: *const f32, loss: *const f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn relu_backward_float(o: *const f32, u: *const f32, loss: *const f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn clipped_relu_backward_float(o: *const f32, u: *const f32, loss: *const f32, ceiling: f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn leaky_relu_backward_float(o: *const f32, u: *const f32, loss: *const f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn swish_backward_float(o: *const f32, u: *const f32, loss: *const f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn tanh_backward_float(o: *const f32, u: *const f32, loss: *const f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn softmax_backward_float(o: *const f32, u: *const f32, loss: *const f32, output: *mut f32, units_len: size_t, batch_size: size_t) -> c_void;
    fn sigmoid_forward_double(input: *const f64, output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn relu_forward_double(input: *const f64, output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn clipped_relu_forward_double(input: *const f64, ceiling: f64, output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn leaky_relu_forward_double(input: *const f64, output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn swish_forward_double(input: *const f64, output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn tanh_forward_double(input: *const f64, output: *mut f64, len: size_t, units_len: size_t) -> c_void;
    fn softmax_forward_double(input: *const f64, output: *mut f64, len: size_t, batch_size: size_t) -> c_void;
    fn sigmoid_backward_double(o: *const f64, u: *const f64, loss: *const f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
    fn relu_backward_double(o: *const f64, u: *const f64, loss: *const f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
    fn clipped_relu_backward_double(o: *const f64, u: *const f64, loss: *const f64, ceiling: f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
    fn leaky_relu_backward_double(o: *const f64, u: *const f64, loss: *const f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
    fn swish_backward_double(o: *const f64, u: *const f64, loss: *const f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
    fn tanh_backward_double(o: *const f64, u: *const f64, loss: *const f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
    fn softmax_backward_double(o: *const f64, u: *const f64, loss: *const f64, output: *mut f64, units_len: size_t, batch_size: size_t) -> c_void;
}
/// Defines the list of passed to the cuda kernel function for the arguments of the activation function.
pub struct ActivationForwardArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          A: CudaAllocator +'a {
    input: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// Output buffer
    pub output: CudaTensor1dPtr<T,A,N>,
    units_len: usize,
    batch_size: usize,
}
/// Create an instance of an object representing the argument list at the time of activation function forward.
impl<'a,T,A,const N:usize> ActivationForwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator +'a {
    /// Create a ActivationForwardArgs instance
    /// # Arguments
    /// * `input` - Input buffer
    /// * `output` - Output buffer
    pub fn new(input:&'a CudaTensor1dPtrView<'a,T,N>,output:CudaTensor1dPtr<T,A,N>) -> ActivationForwardArgs<'a,T,A,N> {
        ActivationForwardArgs {
            input: CudaConstPtr::new(input),
            output: output,
            units_len: N,
            batch_size: 1
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for ActivationForwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a {
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
pub struct ActivationBackwardArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          A: CudaAllocator +'a {
    o: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    u: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    loss: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// Output of error back propagation
    pub output: CudaTensor1dPtr<T,A,N>,
    units_len: usize,
    batch_size: usize,
}
/// Create an instance of an object representing the list of arguments during error
/// back propagation of the activation function.
impl<'a,T,A,const N:usize> ActivationBackwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator +'a {
    /// Create a ActivationBackwardArgs instance
    /// # Arguments
    /// * `o` - Output values
    /// * `u` - Input values from upper layers
    /// * `loss` - loss value
    /// * `output` - Output of error back propagation
    pub fn new(o: &'a CudaTensor1dPtrView<'a,T,N>,
               u: &'a CudaTensor1dPtrView<'a,T,N>,
               loss: &'a CudaTensor1dPtrView<'a,T,N>,
               output: CudaTensor1dPtr<T,A,N>) -> ActivationBackwardArgs<'a,T,A,N> {
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
impl<'a,T,A,const N:usize> KernelArgs for ActivationBackwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a {
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
pub struct ActivationBatchForwardArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          A: CudaAllocator +'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    input: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    /// Output buffer
    pub output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
    units_len: usize,
    batch_size: usize,
}
/// Create an instance of an object representing the argument list
/// of the forward propagation of the activation function during batch execution.
impl<'a,T,A,const N:usize> ActivationBatchForwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// Create a ActivationBatchForwardArgs instance
    /// # Arguments
    /// * `input` - Input buffer
    /// * `output` - Output buffer
    /// * `batch_size` - batches count
    pub fn new(input:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,output:CudaVec<T,CudaTensor1dPtr<T,A,N>,A>, batch_size: usize)
        -> ActivationBatchForwardArgs<'a,T,A,N> {
        ActivationBatchForwardArgs {
            input: CudaConstPtr::new(input),
            output: output,
            units_len: N,
            batch_size: batch_size
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for ActivationBatchForwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
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
pub struct ActivationBatchBackwardArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          A: CudaAllocator + 'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    o: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    u: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    loss: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    /// Output of error back propagation
    pub output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
    units_len: usize,
    batch_size: usize,
}
/// Instantiate an object representing the list of arguments during error back propagation
/// of the activation function during batch execution.
impl<'a,T,A,const N:usize> ActivationBatchBackwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// Create a ActivationBatchBackwardArgs instance
    /// # Arguments
    /// * `o` - Output values
    /// * `u` - Input values from upper layers
    /// * `loss` - loss value
    /// * `output` - Output of error back propagation
    /// * `batch_size` - batch count
    pub fn new(o: &'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               u: &'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               loss: &'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,batch_size: usize) -> ActivationBatchBackwardArgs<'a,T,A,N> {
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
impl<'a,T,A,const N:usize> KernelArgs for ActivationBatchBackwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
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
/// Define the list of arguments passed to the CUDA kernel function as arguments
/// for the forward propagation of ClippedReLU.
pub struct ClippedReLuForwardArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static + AsKernelPtr,
          A: CudaAllocator +'a,
          CudaMutPtr<'a,T,A>: AsMutKernelPtr {
    input: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    ceiling: T,
    /// Output buffer
    pub output: CudaTensor1dPtr<T,A,N>,
    units_len: usize,
    batch_size: usize,
}
/// Create an instance of an object representing the argument list at the time of activation function forward for ClippedReLU.
impl<'a,T,A,const N:usize> ClippedReLuForwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a + AsKernelPtr,
          A: CudaAllocator +'a {
    /// Create a ClippedReLuForwardArgs instance
    /// # Arguments
    /// * `input` - Input buffer
    /// * `ceiling` - Ceiling value for clipping
    /// * `output` - Output buffer
    pub fn new(input:&'a CudaTensor1dPtrView<'a,T,N>, ceiling: T, output:CudaTensor1dPtr<T,A,N>) -> ClippedReLuForwardArgs<'a,T,A,N> {
        ClippedReLuForwardArgs {
            input: CudaConstPtr::new(input),
            ceiling: ceiling,
            output: output,
            units_len: N,
            batch_size: 1
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for ClippedReLuForwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a + AsKernelPtr,
          A: CudaAllocator + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.input,
            &mut self.ceiling,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_size
        ]
    }
}
/// Creates an instance of an object representing the list of arguments for backpropagation in ClippedReLU.
pub struct ClippedReLuBackwardArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static + AsKernelPtr,
          A: CudaAllocator +'a {
    o: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    u: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    loss: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    ceiling: T,
    /// Output of error back propagation
    pub output: CudaTensor1dPtr<T,A,N>,
    units_len: usize,
    batch_size: usize,
}
impl<'a,T,A,const N:usize> ClippedReLuBackwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a + AsKernelPtr,
          A: CudaAllocator +'a {
    /// Create a ClippedReLuBackwardArgs instance
    /// # Arguments
    /// * `o` - Output values
    /// * `u` - Input values from upper layers
    /// * `loss` - loss value
    /// * `ceiling` - Ceiling value for clipping
    /// * `output` - Output of error back propagation
    pub fn new(o: &'a CudaTensor1dPtrView<'a,T,N>,
               u: &'a CudaTensor1dPtrView<'a,T,N>,
               loss: &'a CudaTensor1dPtrView<'a,T,N>,
               ceiling: T,
               output: CudaTensor1dPtr<T,A,N>) -> ClippedReLuBackwardArgs<'a,T,A,N> {
        ClippedReLuBackwardArgs {
            o: CudaConstPtr::new(o),
            u: CudaConstPtr::new(u),
            loss: CudaConstPtr::new(loss),
            ceiling: ceiling,
            output: output,
            units_len: N,
            batch_size: 1
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for ClippedReLuBackwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a + AsKernelPtr,
          A: CudaAllocator + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.o,
            &mut self.u,
            &mut self.loss,
            &mut self.ceiling,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_size
        ]
    }
}
/// Defines the list of arguments passed to the CUDA kernel function,
/// which are passed as arguments for the forward propagation in ClippedReLu during batch execution.
pub struct ClippedReLuBatchForwardArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static + AsKernelPtr,
          A: CudaAllocator +'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    input: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    ceiling: T,
    /// Output buffer
    pub output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
    units_len: usize,
    batch_size: usize,
}
impl<'a,T,A,const N:usize> ClippedReLuBatchForwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a + AsKernelPtr,
          A: CudaAllocator + 'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// Create a ClippedReLuBatchForwardArgs instance
    /// # Arguments
    /// * `input` - Input buffer
    /// * `output` - Output buffer
    /// * `batch_size` - batches count
    /// * `ceiling` - Ceiling value for clipping
    pub fn new(input:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>, ceiling: T, output:CudaVec<T,CudaTensor1dPtr<T,A,N>,A>, batch_size: usize)
        -> ClippedReLuBatchForwardArgs<'a,T,A,N> {
        ClippedReLuBatchForwardArgs {
            input: CudaConstPtr::new(input),
            ceiling: ceiling,
            output: output,
            units_len: N,
            batch_size: batch_size
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for ClippedReLuBatchForwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a + AsKernelPtr,
          A: CudaAllocator + 'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.input,
            &mut self.ceiling,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_size
        ]
    }
}
/// Defines the list of arguments passed to the CUDA kernel function.
/// These are passed as arguments for the backward propagation in ClippedReLu during batch execution.
pub struct ClippedReLuBatchBackwardArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static + AsKernelPtr,
          A: CudaAllocator + 'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    o: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    u: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    loss: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    ceiling: T,
    /// Output of error back propagation
    pub output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
    units_len: usize,
    batch_size: usize,
}
impl<'a,T,A,const N:usize> ClippedReLuBatchBackwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a + AsKernelPtr,
          A: CudaAllocator + 'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// Create a ClippedReLuBatchBackwardArgs instance
    /// # Arguments
    /// * `o` - Output values
    /// * `u` - Input values from upper layers
    /// * `loss` - loss value
    /// * `ceiling` - Ceiling value for clipping
    /// * `output` - Output of error back propagation
    /// * `batch_size` - batch count
    pub fn new(o: &'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               u: &'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               loss: &'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               ceiling: T,
               output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
               batch_size: usize) -> ClippedReLuBatchBackwardArgs<'a,T,A,N> {
        ClippedReLuBatchBackwardArgs {
            o: CudaConstPtr::new(o),
            u: CudaConstPtr::new(u),
            loss: CudaConstPtr::new(loss),
            output: output,
            ceiling: ceiling,
            units_len: N,
            batch_size: batch_size
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for ClippedReLuBatchBackwardArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a + AsKernelPtr,
          A: CudaAllocator + 'a,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.o,
            &mut self.u,
            &mut self.loss,
            &mut self.ceiling,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_size
        ]
    }
}
/// Sigmoid activation function implementation
pub struct SigmoidForward<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SigmoidForward<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a {
    /// Create a SigmoidForward instance
    pub fn new() -> SigmoidForward<'a,T,A,N> {
        SigmoidForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SigmoidForward<'a,f32,A,N> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = sigmoid_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for SigmoidForward<'a,f64,A,N> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = sigmoid_forward_double as *const c_void;
    type Args = ActivationForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of derivatives of the sigmoid activation function
pub struct SigmoidBackward<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SigmoidBackward<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a {
    /// Create a SigmoidBackward instance
    pub fn new() -> SigmoidBackward<'a,T,A,N> {
        SigmoidBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SigmoidBackward<'a,f32,A,N> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = sigmoid_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for SigmoidBackward<'a,f64,A,N> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = sigmoid_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of sigmoid activation functions for batch execution
pub struct SigmoidBatchForward<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SigmoidBatchForward<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a {
    /// Create a SigmoidForwardForBatch instance
    pub fn new() -> SigmoidBatchForward<'a,T,A,N> {
        SigmoidBatchForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SigmoidBatchForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = sigmoid_forward_float as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for SigmoidBatchForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = sigmoid_forward_double as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implement derivatives of the sigmoid activation function for batch execution
pub struct SigmoidBatchBackward<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'a,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SigmoidBatchBackward<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'a {
    /// Create a SigmoidBackwardForBatch instance
    pub fn new() -> SigmoidBatchBackward<'a,T,A,N> {
        SigmoidBatchBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SigmoidBatchBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = sigmoid_backward_float as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for SigmoidBatchBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = sigmoid_backward_double as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// ReLu activation function implementation activation function implementation
pub struct ReLuForward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> ReLuForward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a ReLuForward instance
    pub fn new() -> ReLuForward<'a,T,A,N> {
        ReLuForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for ReLuForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = relu_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for ReLuForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = relu_forward_double as *const c_void;
    type Args = ActivationForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of derivatives of the ReLu activation function
pub struct ReLuBackward<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> ReLuBackward<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    /// Create a ReLuBackward instance
    pub fn new() -> ReLuBackward<'a,T,A,N> {
        ReLuBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for ReLuBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = relu_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for ReLuBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = relu_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of ReLu activation functions for batch execution
pub struct ReLuBatchForward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> ReLuBatchForward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a ReLuForwardBatch instance
    pub fn new() -> ReLuBatchForward<'a,T,A,N> {
        ReLuBatchForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for ReLuBatchForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = relu_forward_float as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for ReLuBatchForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = relu_forward_double as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implement derivatives of the ReLu activation function for batch execution
pub struct ReLuBatchBackward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> ReLuBatchBackward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a ReLuBackwardForBatch instance
    pub fn new() -> ReLuBatchBackward<'a,T,A,N> {
        ReLuBatchBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for ReLuBatchBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = relu_backward_float as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for ReLuBatchBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = relu_backward_double as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of ClippedReLu activation functions for batch execution
pub struct ClippedReLuForward<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo + AsKernelPtr,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> ClippedReLuForward<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo + AsKernelPtr,
          A: CudaAllocator + 'a {
    /// Create a ClippedReLuForward instance
    pub fn new() -> ClippedReLuForward<'a,T,A,N> {
        ClippedReLuForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for ClippedReLuForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = clipped_relu_forward_float as *const c_void;
    type Args = ClippedReLuForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for ClippedReLuForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = clipped_relu_forward_double as *const c_void;
    type Args = ClippedReLuForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of derivatives of the ClippedReLU activation function
pub struct ClippedReLuBackward<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo + AsKernelPtr,
          A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> ClippedReLuBackward<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo + AsKernelPtr,
          A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    /// Create a ClippedReLuBackward instance
    pub fn new() -> ClippedReLuBackward<'a,T,A,N> {
        ClippedReLuBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for ClippedReLuBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = clipped_relu_backward_float as *const c_void;
    type Args = ClippedReLuBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for ClippedReLuBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = clipped_relu_backward_double as *const c_void;
    type Args = ClippedReLuBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of ClippedReLU activation functions for batch execution
pub struct ClippedReLuBatchForward<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo + AsKernelPtr, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> ClippedReLuBatchForward<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo + AsKernelPtr, A: CudaAllocator + 'a {
    /// Create a ClippedReLuForwardBatch instance
    pub fn new() -> ClippedReLuBatchForward<'a,T,A,N> {
        ClippedReLuBatchForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for ClippedReLuBatchForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = clipped_relu_forward_float as *const c_void;
    type Args = ClippedReLuBatchForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for ClippedReLuBatchForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = clipped_relu_forward_double as *const c_void;
    type Args = ClippedReLuBatchForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implement derivatives of the ClippedReLU activation function for batch execution
pub struct ClippedReLuBatchBackward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> ClippedReLuBatchBackward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a ClippedReLuBackwardForBatch instance
    pub fn new() -> ClippedReLuBatchBackward<'a,T,A,N> {
        ClippedReLuBatchBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for ClippedReLuBatchBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = clipped_relu_backward_float as *const c_void;
    type Args = ClippedReLuBatchBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for ClippedReLuBatchBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = clipped_relu_backward_double as *const c_void;
    type Args = ClippedReLuBatchBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Swish activation function implementation
pub struct SwishForward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SwishForward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a SwishForward instance
    pub fn new() -> SwishForward<'a,T,A,N> {
        SwishForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SwishForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = swish_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for SwishForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = swish_forward_double as *const c_void;
    type Args = ActivationForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of derivatives of the Swish activation function
pub struct SwishBackward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SwishBackward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a SwishBackward instance
    pub fn new() -> SwishBackward<'a,T,A,N> {
        SwishBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SwishBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = swish_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for SwishBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = swish_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of Swish activation functions for batch execution
pub struct SwishBatchForward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SwishBatchForward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a SwishForwardForBatch instance
    pub fn new() -> SwishBatchForward<'a,T,A,N> {
        SwishBatchForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SwishBatchForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = swish_forward_float as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for SwishBatchForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = swish_forward_double as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implement derivatives of the Swish activation function for batch execution
pub struct SwishBatchBackward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SwishBatchBackward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a SwishBackwardForBatch instance
    pub fn new() -> SwishBatchBackward<'a,T,A,N> {
        SwishBatchBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SwishBatchBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = swish_backward_float as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for SwishBatchBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = swish_backward_double as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Tanh activation function implementation
pub struct TanhForward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> TanhForward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a TanhForward instance
    pub fn new() -> TanhForward<'a,T,A,N> {
        TanhForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for TanhForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = tanh_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for TanhForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = tanh_forward_double as *const c_void;
    type Args = ActivationForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of derivatives of the Tanh activation function
pub struct TanhBackward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> TanhBackward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a TanhBackward instance
    pub fn new() -> TanhBackward<'a,T,A,N> {
        TanhBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for TanhBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = tanh_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for TanhBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = tanh_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of Tanh activation functions for batch execution
pub struct TanhBatchForward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> TanhBatchForward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a TanhForwardForBatch instance
    pub fn new() -> TanhBatchForward<'a,T,A,N> {
        TanhBatchForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for TanhBatchForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = tanh_forward_float as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for TanhBatchForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = tanh_forward_double as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implement derivatives of the Tanh activation function for batch execution
pub struct TanhBatchBackward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> TanhBatchBackward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a TanhBackwardForBatch instance
    pub fn new() -> TanhBatchBackward<'a,T,A,N> {
        TanhBatchBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for TanhBatchBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = tanh_backward_float as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for TanhBatchBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = tanh_backward_double as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// SoftMax activation function implementation
pub struct SoftMaxForward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SoftMaxForward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a SoftMaxForward instance
    pub fn new() -> SoftMaxForward<'a,T,A,N> {
        SoftMaxForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SoftMaxForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = softmax_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: 1, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 32 * mem::size_of::<f32>() * 2
        }
    }
}
impl<'a,A,const N:usize> Kernel for SoftMaxForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = softmax_forward_double as *const c_void;
    type Args = ActivationForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: 1, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 32 * mem::size_of::<f64>() * 2
        }
    }
}
/// Implementation of derivatives of the softmax activation function
pub struct SoftMaxBackward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SoftMaxBackward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a SoftMaxForward instance
    pub fn new() -> SoftMaxBackward<'a,T,A,N> {
        SoftMaxBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SoftMaxBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = softmax_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: 1, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 1024 * mem::size_of::<f32>()
        }
    }
}
impl<'a,A,const N:usize> Kernel for SoftMaxBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = softmax_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: 1, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 1024 * mem::size_of::<f64>()
        }
    }
}
/// Implementation of Softmax activation functions for batch execution
pub struct SoftMaxBatchForward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SoftMaxBatchForward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a SoftMaxForwardForBatch instance
    pub fn new() -> SoftMaxBatchForward<'a,T,A,N> {
        SoftMaxBatchForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SoftMaxBatchForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = softmax_forward_float as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: args.batch_size as c_uint, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 32 * mem::size_of::<f32>() * 2
        }
    }
}
impl<'a,A,const N:usize> Kernel for SoftMaxBatchForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = softmax_forward_double as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: args.batch_size as c_uint, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 32 * mem::size_of::<f64>() * 2
        }
    }
}
/// Implement derivatives of the Softmax activation function for batch execution
pub struct SoftMaxBatchBackward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> SoftMaxBatchBackward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a SoftMaxForwardForBatch instance
    pub fn new() -> SoftMaxBatchBackward<'a,T,A,N> {
        SoftMaxBatchBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for SoftMaxBatchBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = softmax_backward_float as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: args.batch_size as c_uint, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 1024 * mem::size_of::<f32>()
        }
    }
}
impl<'a,A,const N:usize> Kernel for SoftMaxBatchBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = softmax_backward_double as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: args.batch_size as c_uint, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 1024 * mem::size_of::<f64>()
        }
    }
}
/// LeakyReLu activation function implementation activation function implementation
pub struct LeakyReLuForward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LeakyReLuForward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a LeakyReLuForward instance
    pub fn new() -> LeakyReLuForward<'a,T,A,N> {
        LeakyReLuForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LeakyReLuForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = leaky_relu_forward_float as *const c_void;
    type Args = ActivationForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LeakyReLuForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = leaky_relu_forward_double as *const c_void;
    type Args = ActivationForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of derivatives of the LeakyReLu activation function
pub struct LeakyReLuBackward<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LeakyReLuBackward<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    /// Create a LeakyReLuBackward instance
    pub fn new() -> LeakyReLuBackward<'a,T,A,N> {
        LeakyReLuBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LeakyReLuBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = leaky_relu_backward_float as *const c_void;
    type Args = ActivationBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LeakyReLuBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaMutPtr<'a,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = leaky_relu_backward_double as *const c_void;
    type Args = ActivationBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implementation of LeakyReLu activation functions for batch execution
pub struct LeakyReLuBatchForward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LeakyReLuBatchForward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a LeakyReLuForwardBatch instance
    pub fn new() -> LeakyReLuBatchForward<'a,T,A,N> {
        LeakyReLuBatchForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LeakyReLuBatchForward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = leaky_relu_forward_float as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LeakyReLuBatchForward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = leaky_relu_forward_double as *const c_void;
    type Args = ActivationBatchForwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Implement derivatives of the LeakyReLu activation function for batch execution
pub struct LeakyReLuBatchBackward<'a,T,A,const N:usize> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LeakyReLuBatchBackward<'a,T,A,N> where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo, A: CudaAllocator + 'a {
    /// Create a LeakyReLuBackwardForBatch instance
    pub fn new() -> LeakyReLuBatchBackward<'a,T,A,N> {
        LeakyReLuBatchBackward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LeakyReLuBatchBackward<'a,f32,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = leaky_relu_backward_float as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LeakyReLuBatchBackward<'a,f64,A,N>
    where A: CudaAllocator + 'a,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = leaky_relu_backward_double as *const c_void;
    type Args = ActivationBatchBackwardArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}

