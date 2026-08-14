//! Implementation of various loss functions
use core::fmt::Debug;
use std::ffi::c_uint;
use std::marker::PhantomData;
use cuda_runtime_sys::dim3;
use libc::{c_int, c_void};
use crate::cuda::{AsConstKernelPtr, AsCudaMutPtr, AsKernelPtr, AsMutKernelPtr, CudaConstPtr, CudaMutPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, DataTypeInfo, Kernel, KernelArgs, KernelLaunchConfig};
use crate::cuda::allocator::CudaAllocator;

extern "C" {
    fn loss_linear_batch_mse_derive_float(r: *const f32, t: *const f32, output: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_mse_derive_double(r: *const f64, t: *const f64, output: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_derive_float(r: *const f32, t: *const f32, output: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_derive_double(r: *const f64, t: *const f64, output: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_multiclass_derive_float(r: *const f32, t: *const f32, output: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_multiclass_derive_double(r: *const f64, t: *const f64, output: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
}
/// Define a list to be passed to the cuda kernel function during mini-batch execution as the argument of mse.
pub struct LinearBatchMseArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// expected value
    expected: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the list of arguments to
/// compute the loss function mse during mini-batch execution.
impl<'a,T,A,const N:usize> LinearBatchMseArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// Create a LinearBatchMseArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(t:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,r:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,out_len:usize,batch_len:usize) -> LinearBatchMseArgs<'a,T,A,N> {
        LinearBatchMseArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for LinearBatchMseArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.output,
            &mut self.out_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearBatchMse<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    t:PhantomData<T>,
    a:PhantomData<A>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LinearBatchMse<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// Create a LinearBatchMse instance
    pub fn new() -> LinearBatchMse<'a,T,A,N> {
        LinearBatchMse {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearBatchMse<'a,f32,A,N>
    where A: CudaAllocator + 'static,
          CudaTensor1dPtr<f32,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = loss_linear_batch_mse_derive_float as *const c_void;
    type Args = LinearBatchMseArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_len + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearBatchMse<'a,f64,A,N>
    where A: CudaAllocator + 'static,
          CudaTensor1dPtr<f64,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = loss_linear_batch_mse_derive_double as *const c_void;
    type Args = LinearBatchMseArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_len + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Defines the list passed to the cuda kernel function as the argument of mse.
pub struct LinearMseArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    /// expected value
    expected: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    pub output: CudaTensor1dPtr<T,A,N>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list for computing the loss function mse.
impl<'a,T,A,const N:usize> LinearMseArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    /// Create a LinearMseArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    pub fn new(t:&'a CudaTensor1dPtrView<'a,T,N>,
               r:&'a CudaTensor1dPtrView<'a,T,N>,
               output: CudaTensor1dPtr<T,A,N>,
               out_len:usize) -> LinearMseArgs<'a,T,A,N> {
        LinearMseArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: 1
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for LinearMseArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.output,
            &mut self.out_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearMse<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    t:PhantomData<T>,
    a:PhantomData<A>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LinearMse<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    /// Create a LinearMse instance
    pub fn new() -> LinearMse<'a,T,A,N> {
        LinearMse {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearMse<'a,f32,A,N> where A: CudaAllocator + 'static {
    const FUNC_PTR: *const c_void = loss_linear_batch_mse_derive_float as *const c_void;
    type Args = LinearMseArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearMse<'a,f64,A,N> where A: CudaAllocator + 'static {
    const FUNC_PTR: *const c_void = loss_linear_batch_mse_derive_double as *const c_void;
    type Args = LinearMseArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Defines the list that is passed to the cuda kernel function as cross-entropy arguments during mini-batch execution.
pub struct LinearBatchCrossEntropyArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// expected value
    expected: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing a list of arguments to calculate
/// the result of passing a mini-batch to the loss function cross entropy.
impl<'a,T,A,const N:usize> LinearBatchCrossEntropyArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// Create a LinearBatchCrossEntropyArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(t:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               r:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
               out_len:usize,batch_len:usize) -> LinearBatchCrossEntropyArgs<'a,T,A,N> {
        LinearBatchCrossEntropyArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for LinearBatchCrossEntropyArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.output,
            &mut self.out_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearBatchCrossEntropy<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    t:PhantomData<T>,
    a:PhantomData<A>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LinearBatchCrossEntropy<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// Create a LinearBatchCrossEntropy instance
    pub fn new() -> LinearBatchCrossEntropy<'a,T,A,N> {
        LinearBatchCrossEntropy {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearBatchCrossEntropy<'a,f32,A,N>
    where A: CudaAllocator + 'static,
          CudaTensor1dPtr<f32,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_derive_float as *const c_void;
    type Args = LinearBatchCrossEntropyArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_len + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearBatchCrossEntropy<'a,f64,A,N>
    where A: CudaAllocator + 'static,
          CudaTensor1dPtr<f64,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_derive_double as *const c_void;
    type Args = LinearBatchCrossEntropyArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_len + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Defines the list passed to the cuda kernel function as the argument of cross entropy.
pub struct LinearCrossEntropyArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    /// expected value
    expected: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    pub output: CudaTensor1dPtr<T,A,N>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list for computing the loss function cross entropy.
impl<'a,T,A,const N:usize> LinearCrossEntropyArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    /// Create a LinearCrossEntropyArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    pub fn new(t:&'a CudaTensor1dPtrView<'a,T,N>,
               r:&'a CudaTensor1dPtrView<'a,T,N>,
               output: CudaTensor1dPtr<T,A,N>,
               out_len:usize) -> LinearCrossEntropyArgs<'a,T,A,N> {
        LinearCrossEntropyArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: 1
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for LinearCrossEntropyArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.output,
            &mut self.out_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearCrossEntropy<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    t:PhantomData<T>,
    a:PhantomData<A>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LinearCrossEntropy<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    /// Create a LinearCrossEntropy instance
    pub fn new() -> LinearCrossEntropy<'a,T,A,N> {
        LinearCrossEntropy {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearCrossEntropy<'a,f32,A,N> where A: CudaAllocator + 'static {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_derive_float as *const c_void;
    type Args = LinearCrossEntropyArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearCrossEntropy<'a,f64,A,N> where A: CudaAllocator + 'static {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_derive_double as *const c_void;
    type Args = LinearCrossEntropyArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Defines the list that is passed to the cuda kernel function as arguments
/// to the croos entropy multiclass during mini-batch execution.
pub struct LinearBatchCrossEntropyMulticlassArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// expected value
    expected: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing a list of arguments to compute the result of passing a mini-batch
/// to the loss function cross entropy multiclass.
impl<'a,T,A,const N:usize> LinearBatchCrossEntropyMulticlassArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// Create a LinearBatchCrossEntropyMulticlassArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(t:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               r:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
               out_len:usize,batch_len:usize) -> LinearBatchCrossEntropyMulticlassArgs<'a,T,A,N> {
        LinearBatchCrossEntropyMulticlassArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for LinearBatchCrossEntropyMulticlassArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.output,
            &mut self.out_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearBatchCrossEntropyMulticlass<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    t:PhantomData<T>,
    a:PhantomData<A>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LinearBatchCrossEntropyMulticlass<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<T,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<T,CudaTensor1dPtr<T,A,N>,A>: AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutKernelPtr {
    /// Create a LinearBatchCrossEntropyMulticlass instance
    pub fn new() -> LinearBatchCrossEntropyMulticlass<'a,T,A,N> {
        LinearBatchCrossEntropyMulticlass {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearBatchCrossEntropyMulticlass<'a,f32,A,N>
    where A: CudaAllocator + 'static,
          CudaTensor1dPtr<f32,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'b> CudaMutPtr<'b,f32,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_multiclass_derive_float as *const c_void;
    type Args = LinearBatchCrossEntropyMulticlassArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_len + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearBatchCrossEntropyMulticlass<'a,f64,A,N>
    where A: CudaAllocator + 'static,
          CudaTensor1dPtr<f64,A,N>: AsConstKernelPtr + AsKernelPtr + Debug + 'a,
          for<'b> CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'b> CudaMutPtr<'b,f64,A>: AsMutKernelPtr {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_multiclass_derive_double as *const c_void;
    type Args = LinearBatchCrossEntropyMulticlassArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_len + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Defines the list passed to the cuda kernel function as the argument of croos entropy multiclass
pub struct LinearCrossEntropyMulticlassArgs<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    /// expected value
    expected: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    pub output: CudaTensor1dPtr<T,A,N>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list for computing the loss function cross entropy multiclass.
impl<'a,T,A,const N:usize> LinearCrossEntropyMulticlassArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    /// Create a LinearCrossEntropyMulticlassArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    pub fn new(t:&'a CudaTensor1dPtrView<T,N>,
               r:&'a CudaTensor1dPtrView<'a,T,N>,
               output: CudaTensor1dPtr<T,A,N>,
               out_len:usize) -> LinearCrossEntropyMulticlassArgs<'a,T,A,N> {
        LinearCrossEntropyMulticlassArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: 1
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for LinearCrossEntropyMulticlassArgs<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.output,
            &mut self.out_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearCrossEntropyMulticlass<'a,T,A,const N:usize>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    t:PhantomData<T>,
    a:PhantomData<A>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LinearCrossEntropyMulticlass<'a,T,A,N>
    where T: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          A: CudaAllocator + 'static {
    /// Create a LinearCrossEntropyMulticlass instance
    pub fn new() -> LinearCrossEntropyMulticlass<'a,T,A,N> {
        LinearCrossEntropyMulticlass {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearCrossEntropyMulticlass<'a,f32,A,N> where A: CudaAllocator + 'static {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_multiclass_derive_float as *const c_void;
    type Args = LinearCrossEntropyMulticlassArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LinearCrossEntropyMulticlass<'a,f64,A,N> where A: CudaAllocator + 'static {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_multiclass_derive_double as *const c_void;
    type Args = LinearCrossEntropyMulticlassArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
