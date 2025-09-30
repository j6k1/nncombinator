//! Implementation of a device that performs various calculations for neural networks

use std::ffi::c_uint;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem;
use cuda_runtime_sys::dim3;
use libc::{c_int, c_void, size_t};
use crate::cuda::{AsKernelPtr, CudaConstPtr, CudaPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaTensor2dPtr, CudaVec, CudaVecView, DataTypeInfo, Kernel, KernelArgs, KernelLaunchConfig};
use crate::cuda::allocator::CudaAllocator;
use crate::ope::UnitValue;

extern "C" {
    fn reduce_linear_batch_float(input: *const f32, output: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn reduce_linear_batch_double(input: *const f64, output: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn forward_linear_batch_float(input: *const f32, units: *const f32, bias: *const f32, output: *mut f32, input_len: size_t, output_len: size_t, batch_size: size_t) -> c_void;
    fn forward_linear_batch_double(input: *const f64, units: *const f64, bias: *const f64, output: *mut f64, input_len: size_t, output_len: size_t, batch_size: size_t) -> c_void;
    fn backward_linear_batch_float(loss: *const f32, units: *const f32, output: *mut f32, input_len: size_t, output_len: size_t, batch_size: size_t) -> c_void;
    fn backward_linear_batch_double(loss: *const f64, units: *const f64, output: *mut f64, input_len: size_t, output_len: size_t, batch_size: size_t) -> c_void;
    fn linear_gradient_batch_float(loss: *const f32, input: *const f32, output: *mut f32, input_len: size_t, output_len: size_t, units_size: size_t, batch_size: size_t) -> c_void;
    fn linear_gradient_batch_double(loss: *const f64, input: *const f64, output: *mut f64, input_len: size_t, output_len: size_t, units_size: size_t, batch_size: size_t) -> c_void;
    fn loss_linear_batch_by_canonical_link_float(expected: *const f32, actual: *const f32, output: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_by_canonical_link_double(expected: *const f64, actual: *const f64, output: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn forward_diff_linear_float(indexes: *const size_t, input: *const f32, units: *const f32, bias: *const f32, output: *mut f32, output_size: size_t, diff_len: size_t) -> c_void;
    fn forward_diff_linear_double(indexes: *const size_t, input: *const f64, units: *const f64, bias: *const f32, output: *mut f64, output_size: size_t, diff_len: size_t) -> c_void;
    fn addbias_batch_float(bias: *const f32, input_output: *mut f32, units_len: c_int, batch_size: c_int) -> c_void;
    fn addbias_batch_double(bias: *const f64, input_output: *mut f64, units_len: c_int, batch_size: c_int) -> c_void;
}
/// Defines the list that is passed to the cuda kernel function as arguments for the convolution calculation.
pub struct ReduceLinearBatchArgs<'a,T,A,const N:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator {
    input: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    /// output
    pub output: CudaTensor1dPtr<T,A,N>,
    units_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list during convolution computation.
impl<'a,T,A,const N:usize> ReduceLinearBatchArgs<'a,T,A,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator {
    /// Create a ReduceLinearBatchArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `output` - output (All elements must be initialized to zero.)
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch_count
    pub fn new(input:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               output:CudaTensor1dPtr<T,A,N>,out_len:usize,batch_len:usize) -> ReduceLinearBatchArgs<'a,T,A,N> {
        ReduceLinearBatchArgs {
            input: CudaConstPtr::new(input),
            output: output,
            units_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for ReduceLinearBatchArgs<'a,T,A,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.input,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
/// Implementation of convolutional computation
pub struct ReduceLinearBatch<'a,T,A,const N:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>,
    n:PhantomData<[();N]>
}
impl<'a,T,A,const N:usize> ReduceLinearBatch<'a,T,A,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator {
    /// Create a ReduceLinearBatch instance
    pub fn new() -> ReduceLinearBatch<'a,T,A,N> {
        ReduceLinearBatch {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l:PhantomData::<&'a ()>,
            n:PhantomData::<[();N]>
        }
    }
}
impl<'a,A,const N:usize> Kernel for ReduceLinearBatch<'a,f32,A,N> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = reduce_linear_batch_float as *const c_void;
    type Args = ReduceLinearBatchArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: N as c_uint, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 32 * mem::size_of::<f32>()
        }
    }
}
impl<'a,A,const N:usize> Kernel for ReduceLinearBatch<'a,f64,A,N> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = reduce_linear_batch_double as *const c_void;
    type Args = ReduceLinearBatchArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: N as c_uint, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 32 * mem::size_of::<f64>()
        }
    }
}
/// Defines the list that is passed to the cuda kernel function as arguments
/// for the calculation that applies the canonical link during the mini-batch execution.
pub struct LossLinearBatchByCanonicalLinkArgs<'a,T,A,const N:usize>
    where T: DataTypeInfo + UnitValue<T>,
          A: CudaAllocator {
    expected: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    /// Actual Value
    actual: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
    units_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list
/// for the regular link application calculation at the time of mini-batch execution.
impl<'a,T,A,const N:usize> LossLinearBatchByCanonicalLinkArgs<'a,T,A,N>
    where T: DataTypeInfo + UnitValue<T>,
          A: CudaAllocator {
    /// Create a LossLinearBatchByCanonicalLinkArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(expected:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               actual:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,N>>,
               output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
               units_len:usize,batch_len:usize) -> LossLinearBatchByCanonicalLinkArgs<'a,T,A,N> {
        LossLinearBatchByCanonicalLinkArgs {
            expected: CudaConstPtr::new(expected),
            actual: CudaConstPtr::new(actual),
            output: output,
            units_len: units_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for LossLinearBatchByCanonicalLinkArgs<'a,T,A,N>
    where T: DataTypeInfo + UnitValue<T>, A: CudaAllocator {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct LossLinearBatchByCanonicalLink<'a,T,A,const N:usize>
    where T: DataTypeInfo + UnitValue<T>, A: CudaAllocator {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LossLinearBatchByCanonicalLink<'a,T,A,N>
    where T: DataTypeInfo + UnitValue<T>, A: CudaAllocator {
    /// Create a LossLinearBatchByCanonicalLink instance
    pub fn new() -> LossLinearBatchByCanonicalLink<'a,T,A,N> {
        LossLinearBatchByCanonicalLink {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LossLinearBatchByCanonicalLink<'a,f32,A,N> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_float as *const c_void;
    type Args = LossLinearBatchByCanonicalLinkArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_len + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LossLinearBatchByCanonicalLink<'a,f64,A,N> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_double as *const c_void;
    type Args = LossLinearBatchByCanonicalLinkArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_len + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Defines the list that is passed to the cuda kernel function as the argument for the calculation of applying canonical link.
pub struct LossLinearByCanonicalLinkArgs<'a,T,A,const N:usize>
    where T: DataTypeInfo + UnitValue<T>, A: CudaAllocator {
    expected: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// Actual Value
    actual: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    pub output: CudaTensor1dPtr<T,A,N>,
    units_len: usize,
    batch_len: usize,
}
/// Create an instance of an object that represents the argument list for the canonical link application calculation.
impl<'a,T,A,const N:usize> LossLinearByCanonicalLinkArgs<'a,T,A,N>
    where T: DataTypeInfo + UnitValue<T>, A: CudaAllocator {
    /// Create a LossLinearByCanonicalLinkArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    pub fn new(expected:&'a CudaTensor1dPtrView<'a,T,N>,
               actual:&'a CudaTensor1dPtrView<'a,T,N>,
               output: CudaTensor1dPtr<T,A,N>,
               units_len:usize) -> LossLinearByCanonicalLinkArgs<'a,T,A,N> {
        LossLinearByCanonicalLinkArgs {
            expected: CudaConstPtr::new(expected),
            actual: CudaConstPtr::new(actual),
            output: output,
            units_len: units_len,
            batch_len: 1
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for LossLinearByCanonicalLinkArgs<'a,T,A,N>
    where T: DataTypeInfo + UnitValue<T>, A: CudaAllocator {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct LossLinearByCanonicalLink<'a,T,A,const N:usize>
    where T: DataTypeInfo + UnitValue<T>, A: CudaAllocator {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> LossLinearByCanonicalLink<'a,T,A,N>
    where T: DataTypeInfo + UnitValue<T>, A: CudaAllocator {
    /// Create a LossLinearByCanonicalLink instance
    pub fn new() -> LossLinearByCanonicalLink<'a,T,A,N> {
        LossLinearByCanonicalLink {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for LossLinearByCanonicalLink<'a,f32,A,N> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_float as *const c_void;
    type Args = LossLinearByCanonicalLinkArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
impl<'a,A,const N:usize> Kernel for LossLinearByCanonicalLink<'a,f64,A,N> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_double as *const c_void;
    type Args = LossLinearByCanonicalLinkArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0
        }
    }
}
/// Defines the list that is passed to the cuda kernel function as arguments for forward propagation difference calculations.
pub struct DiffLinearForwardArgs<'a,T,A,const NI:usize,const NO:usize>
    where T: Debug + Default,
          A: CudaAllocator {
    indexes: CudaPtr<usize,A>,
    input: CudaPtr<T,A>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,A,NI,NO>>,
    pub output: CudaTensor1dPtr<T,A,NO>,
    output_size: usize,
    diff_len: usize
}
/// Create an instance of an object representing the argument list for the forward propagation difference calculation.
impl<'a,T,A,const NI:usize,const NO:usize> DiffLinearForwardArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default,
          A: CudaAllocator {
    /// Create a DiffLinearForwardArgs instance
    /// # Arguments
    ///
    /// * `indexes` - List of index corresponding to inputs for difference calculation
    /// * `input` - Value of input used for difference calculation
    /// * `units` - List of weights to be multiplied by the input
    /// * `output` - output
    /// * `ountput_size` - Number of Outputs
    /// * `diff_len` - Number of differential inputs
    pub fn new(indexes:CudaPtr<usize,A>,input:CudaPtr<T,A>,
               units: &'a CudaTensor2dPtr<T,A,NI,NO>,
               output: CudaTensor1dPtr<T,A,NO>,
               output_size:usize,diff_len:usize)
        -> DiffLinearForwardArgs<'a,T,A,NI,NO> {
        DiffLinearForwardArgs {
            indexes,
            input,
            units: CudaConstPtr::new(units),
            output,
            output_size,
            diff_len
        }
    }
}
impl<'a,T,A,const NI:usize,const NO:usize> KernelArgs for DiffLinearForwardArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default,
          A: CudaAllocator {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.indexes,
            &mut self.input,
            &mut self.units,
            &mut self.output,
            &mut self.output_size,
            &mut self.diff_len
        ]
    }
}
pub struct DiffLinearForward<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default,
          A: CudaAllocator {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const NI:usize,const NO:usize> DiffLinearForward<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default,
          A: CudaAllocator {
    /// Create a DiffLinearForward instance
    pub fn new() -> DiffLinearForward<'a,T,A,NI,NO> {
        DiffLinearForward {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for DiffLinearForward<'a,f32,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = forward_diff_linear_float as *const c_void;
    type Args = DiffLinearForwardArgs<'a,f32,A,NI,NO>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NO + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 1024 * mem::size_of::<f32>()
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for DiffLinearForward<'a,f64,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = forward_diff_linear_double as *const c_void;
    type Args = DiffLinearForwardArgs<'a,f64,A,NI,NO>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NO + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 1024 * mem::size_of::<f64>()
        }
    }
}
/// Defines the list that is passed to the cuda kernel function
/// as arguments for the mini-batch computation
/// of forward propagation of linear layers.
pub struct ForwardLinearBatchArgs<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    input: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,NI>>>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,A,NI,NO>>,
    bias: CudaConstPtr<'a,CudaTensor1dPtr<T,A,NO>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,A,NO>,A>,
    input_len: usize,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing the argument list during
/// the forward propagation calculation of a mini-batch of linear layers.
impl<'a,T,A,const NI:usize,const NO:usize> ForwardLinearBatchArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a ForwardLinearBatchArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `units` - weight
    /// * `bias` - bias
    /// * `output` - output (All elements must be initialized to zero.)
    /// * `batch_len` - batch_count
    pub fn new(input: &'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,NI>>,
               units: &'a CudaTensor2dPtr<T,A,NI,NO>,
               bias: &'a CudaTensor1dPtr<T,A,NO>,
               output:CudaVec<T,CudaTensor1dPtr<T,A,NO>,A>, batch_size: usize) -> ForwardLinearBatchArgs<'a,T,A,NI,NO> {
        ForwardLinearBatchArgs {
            input: CudaConstPtr::new(input),
            units: CudaConstPtr::new(units),
            bias: CudaConstPtr::new(bias),
            output: output,
            input_len: NI,
            output_len: NO,
            batch_size: batch_size
        }
    }
}
impl<'a,T,A,const NI:usize,const NO:usize> KernelArgs for ForwardLinearBatchArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.input,
            &mut self.units,
            &mut self.bias,
            &mut self.output,
            &mut self.input_len,
            &mut self.output_len,
            &mut self.batch_size
        ]
    }
}
/// Implementation of forward propagation calculations for mini-batches of linear layers
pub struct ForwardLinearBatch<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const NI:usize,const NO:usize> ForwardLinearBatch<'a,T,A,NI,NO,>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a ForwardLinearBatch instance
    pub fn new() -> ForwardLinearBatch<'a,T,A,NI,NO> {
        ForwardLinearBatch {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            a: PhantomData::<A>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for ForwardLinearBatch<'a,f32,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = forward_linear_batch_float as *const c_void;
    type Args = ForwardLinearBatchArgs<'a,f32,A,NI,NO>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NI + 15) as c_uint / 16, y: (args.batch_size + 15) as c_uint / 16, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for ForwardLinearBatch<'a,f64,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = forward_linear_batch_double as *const c_void;
    type Args = ForwardLinearBatchArgs<'a,f64,A,NI,NO>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NI + 15) as c_uint / 16, y: (args.batch_size + 15) as c_uint / 16, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
/// Defines the list that is passed to the cuda kernel function as arguments for the computation
/// of the forward propagation of the linear layer.
pub struct ForwardLinearArgs<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    input: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,NI>>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,A,NI,NO>>,
    bias: CudaConstPtr<'a,CudaTensor1dPtr<T,A,NO>>,
    pub output: CudaTensor1dPtr<T,A,NO>,
    input_len: usize,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing the argument list during
/// the forward propagation calculation of the linear layer.
impl<'a,T,A,const NI:usize,const NO:usize> ForwardLinearArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a ForwardLinearArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `units` - weight
    /// * `bias` - bias.
    /// * `output` - output (All elements must be initialized to zero.)
    pub fn new(input: &'a CudaTensor1dPtrView<'a,T,NI>,
               units: &'a CudaTensor2dPtr<T,A,NI,NO>,
               bias: &'a CudaTensor1dPtr<T,A,NO>,
               output:CudaTensor1dPtr<T,A,NO>) -> ForwardLinearArgs<'a,T,A,NI,NO> {
        ForwardLinearArgs {
            input: CudaConstPtr::new(input),
            units: CudaConstPtr::new(units),
            bias: CudaConstPtr::new(bias),
            output: output,
            input_len: NI,
            output_len: NO,
            batch_size: 1
        }
    }
}
impl<'a,T,A,const NI:usize,const NO:usize> KernelArgs for ForwardLinearArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.input,
            &mut self.units,
            &mut self.bias,
            &mut self.output,
            &mut self.input_len,
            &mut self.output_len,
            &mut self.batch_size
        ]
    }
}
/// Implementation of forward propagation calculations for linear layers
pub struct ForwardLinear<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const NI:usize,const NO:usize> ForwardLinear<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a ForwardLinear instance
    pub fn new() -> ForwardLinear<'a,T,A,NI,NO> {
        ForwardLinear {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            a: PhantomData::<A>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for ForwardLinear<'a,f32,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = forward_linear_batch_float as *const c_void;
    type Args = ForwardLinearArgs<'a,f32,A,NI,NO>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NI + 15) as c_uint / 16, y: 1, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for ForwardLinear<'a,f64,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = forward_linear_batch_double as *const c_void;
    type Args = ForwardLinearArgs<'a,f64,A,NI,NO>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NI + 15) as c_uint / 16, y: 1, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
/// Defines the list passed to the cuda kernel function as arguments
/// for the computation of the error back propagation of a mini-batch of linear layers.
pub struct BackwardLinearBatchArgs<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    loss: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,NO>>>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,A,NI,NO>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,A,NI>,A>,
    input_len: usize,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing a list of arguments
/// during the computation of the error back propagation of a mini-batch of linear layers.
impl<'a,T,A,const NI:usize,const NO:usize> BackwardLinearBatchArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a BackwardLinearBatchArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `units` - weight
    /// * `output` - output (All elements must be initialized to zero.)
    /// * `batch_len` - batch_count
    pub fn new(loss:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,NO>>,
               units: &'a CudaTensor2dPtr<T,A,NI,NO>,
               output:CudaVec<T,CudaTensor1dPtr<T,A,NI>,A>, batch_size: usize) -> BackwardLinearBatchArgs<'a,T,A,NI,NO> {
        BackwardLinearBatchArgs {
            loss: CudaConstPtr::new(loss),
            units: CudaConstPtr::new(units),
            output: output,
            input_len: NI,
            output_len: NO,
            batch_size: batch_size
        }
    }
}
impl<'a,T,A,const NI:usize,const NO:usize> KernelArgs for BackwardLinearBatchArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.loss,
            &mut self.units,
            &mut self.output,
            &mut self.input_len,
            &mut self.output_len,
            &mut self.batch_size
        ]
    }
}
/// Implementation of mini-batch error back propagation computation for linear layers
pub struct BackwardLinearBatch<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const NI:usize,const NO:usize> BackwardLinearBatch<'a,T,A,NI,NO,>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a BackwardLinearBatch instance
    pub fn new() -> BackwardLinearBatch<'a,T,A,NI,NO> {
        BackwardLinearBatch {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            a: PhantomData::<A>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for BackwardLinearBatch<'a,f32,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = backward_linear_batch_float as *const c_void;
    type Args = BackwardLinearBatchArgs<'a,f32,A,NI,NO>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NO + 15) as c_uint / 16, y: (args.batch_size + 15) as c_uint / 16, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for BackwardLinearBatch<'a,f64,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = backward_linear_batch_double as *const c_void;
    type Args = BackwardLinearBatchArgs<'a,f64,A,NI,NO>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NO + 15) as c_uint / 16, y: (args.batch_size + 15) as c_uint / 16, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
/// Defines the list that is passed to the cuda kernel function as arguments for
/// the computation of the error back propagation of the linear layer.
pub struct BackwardLinearArgs<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    loss: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,NO>>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,A,NI,NO>>,
    pub output: CudaTensor1dPtr<T,A,NI>,
    input_len: usize,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing the list of arguments during
/// the computation of the error back propagation of the linear layer.
impl<'a,T,A,const NI:usize,const NO:usize> BackwardLinearArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a BackwardLinearArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `units` - weight
    /// * `output` - output (All elements must be initialized to zero.)
    pub fn new(loss:&'a CudaTensor1dPtrView<'a,T,NO>,
               units: &'a CudaTensor2dPtr<T,A,NI,NO>,
               output:CudaTensor1dPtr<T,A,NI>) -> BackwardLinearArgs<'a,T,A,NI,NO> {
        BackwardLinearArgs {
            loss: CudaConstPtr::new(loss),
            units: CudaConstPtr::new(units),
            output: output,
            input_len: NI,
            output_len: NO,
            batch_size: 1
        }
    }
}
impl<'a,T,A,const NI:usize,const NO:usize> KernelArgs for BackwardLinearArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.loss,
            &mut self.units,
            &mut self.output,
            &mut self.input_len,
            &mut self.output_len,
            &mut self.batch_size
        ]
    }
}
/// Implementation of error back propagation calculations for linear layers
pub struct BackwardLinear<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const NI:usize,const NO:usize> BackwardLinear<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a BackwardLinear instance
    pub fn new() -> BackwardLinear<'a,T,A,NI,NO> {
        BackwardLinear {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            a: PhantomData::<A>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for BackwardLinear<'a,f32,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = backward_linear_batch_float as *const c_void;
    type Args = BackwardLinearArgs<'a,f32,A,NI,NO>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NO + 15) as c_uint / 16, y: 1, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for BackwardLinear<'a,f64,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = backward_linear_batch_double as *const c_void;
    type Args = BackwardLinearArgs<'a,f64,A,NI,NO>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NO + 15) as c_uint / 16, y: 1, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
/// Defines the list that is passed to the cuda kernel function as arguments
/// for the calculation of the amount of update of the linear layer weights during the mini-batch.
pub struct LinearGradientBatchArgs<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    loss: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,NO>>>,
    input: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,NI>>>,
    pub output: CudaTensor2dPtr<T,A,NI,NO>,
    input_len: usize,
    output_len: usize,
    units_size: usize,
    batch_size: usize
}
/// Create an instance of an object representing the list of arguments
///s for calculating the amount of updates to the linear layer weights during a mini-batch run.
impl<'a,T,A,const NI:usize,const NO:usize> LinearGradientBatchArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a LinearGradientBatchArgs instance
    /// # Arguments
    /// * `loss` - loss
    /// * `input` - input
    /// * `output` - output (All elements must be initialized to zero.)
    /// * `batch_len` - batch_count
    pub fn new(loss:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,NO>>, input:&'a CudaVecView<'a,T,CudaTensor1dPtrView<'a,T,NI>>,
               output:CudaTensor2dPtr<T,A,NI,NO>, batch_size: usize) -> LinearGradientBatchArgs<'a,T,A,NI,NO> {
        LinearGradientBatchArgs {
            loss: CudaConstPtr::new(loss),
            input: CudaConstPtr::new(input),
            output: output,
            input_len: NI,
            output_len: NO,
            units_size: NI * NO,
            batch_size: batch_size
        }
    }
}
impl<'a,T,A,const NI:usize,const NO:usize> KernelArgs for LinearGradientBatchArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.loss,
            &mut self.input,
            &mut self.output,
            &mut self.input_len,
            &mut self.output_len,
            &mut self.units_size,
            &mut self.batch_size
        ]
    }
}
/// Implementation of gradient calculation during mini-batch execution of linear layers
pub struct LinearGradientBatch<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    a:PhantomData<A>,
    lt:PhantomData<&'a ()>
}
impl<'a,T,A,const NI:usize,const NO:usize> LinearGradientBatch<'a,T,A,NI,NO,>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a LinearGradientBatch instance
    pub fn new() -> LinearGradientBatch<'a,T,A,NI,NO> {
        LinearGradientBatch {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            a: PhantomData::<A>,
            lt:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for LinearGradientBatch<'a,f32,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = linear_gradient_batch_float as *const c_void;
    type Args = LinearGradientBatchArgs<'a,f32,A,NI,NO>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NO + 15) as c_uint / 16, y: (NI + 15) as c_uint / 16, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for LinearGradientBatch<'a,f64,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = linear_gradient_batch_double as *const c_void;
    type Args = LinearGradientBatchArgs<'a,f64,A,NI,NO>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NO + 15) as c_uint / 16, y: (NI + 15) as c_uint / 16, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
/// Defines the list that is passed to the cuda kernel function as arguments
/// for the computation of the amount of update of the linear layer weights.
pub struct LinearGradientArgs<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    loss: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,NO>>,
    input: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,NI>>,
    pub output: CudaTensor2dPtr<T,A,NI,NO>,
    input_len: usize,
    output_len: usize,
    units_size: usize,
    batch_size: usize
}
/// Create an instance of an object representing the argument list
/// for the calculation of the update amount of the linear layer weights.
impl<'a,T,A,const NI:usize,const NO:usize> LinearGradientArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a LinearGradientBatchArgs instance
    /// # Arguments
    /// * `loss` - loss
    /// * `input` - input
    /// * `output` - output (All elements must be initialized to zero.)
    pub fn new(loss:&'a CudaTensor1dPtrView<'a,T,NO>, input:&'a CudaTensor1dPtrView<'a,T,NI>,
               output:CudaTensor2dPtr<T,A,NI,NO>) -> LinearGradientArgs<'a,T,A,NI,NO> {
        LinearGradientArgs {
            loss: CudaConstPtr::new(loss),
            input: CudaConstPtr::new(input),
            output: output,
            input_len: NI,
            output_len: NO,
            units_size: NI * NO,
            batch_size: 1
        }
    }
}
impl<'a,T,A,const NI:usize,const NO:usize> KernelArgs for LinearGradientArgs<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.loss,
            &mut self.input,
            &mut self.output,
            &mut self.input_len,
            &mut self.output_len,
            &mut self.units_size,
            &mut self.batch_size
        ]
    }
}
/// Implementation of gradient calculation for linear layers
pub struct LinearGradient<'a,T,A,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    a: PhantomData<A>,
    lt:PhantomData<&'a ()>
}
impl<'a,T,A,const NI:usize,const NO:usize> LinearGradient<'a,T,A,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a LinearGradientBatch instance
    pub fn new() -> LinearGradient<'a,T,A,NI,NO> {
        LinearGradient {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            a: PhantomData::<A>,
            lt:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for LinearGradient<'a,f32,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = linear_gradient_batch_float as *const c_void;
    type Args = LinearGradientArgs<'a,f32,A,NI,NO>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NO + 15) as c_uint / 16, y: (NI + 15) as c_uint / 16, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
impl<'a,A,const NI:usize,const NO:usize> Kernel for LinearGradient<'a,f64,A,NI,NO> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = linear_gradient_batch_double as *const c_void;
    type Args = LinearGradientArgs<'a,f64,A,NI,NO>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (NO + 15) as c_uint / 16, y: (NI + 15) as c_uint / 16, z: 1 },
            block_dim: dim3 { x: 16, y: 16, z: 1 },
            shared_memory_size: 2 * 256 * mem::size_of::<f32>() / 2 + 256 * mem::size_of::<f32>(),
        }
    }
}
/// Defines the list of arguments passed to the cuda function
/// that performs the addition of the bias to the mini-batch.
pub struct AddBiasBatchArgs<'a,T,A,const N:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    bias: CudaConstPtr<'a,CudaTensor1dPtr<T,A,N>>,
    pub input_output: CudaVec<T,CudaTensor1dPtr<T,A,N>,A>,
    units_len: usize,
    batch_size: usize
}
/// Create an instance of the type that represents the list of arguments passed to the bias addition process
impl<'a,T,A,const N:usize> AddBiasBatchArgs<'a,T,A,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a AddBiasBatchArgs instance
    /// # Arguments
    /// * `bias` - bias
    /// * `input_output` - input and output (Updated in-place)
    /// * `units_len` - units_len
    /// * `batch_len` - batch_count
    pub fn new(bias: &'a CudaTensor1dPtr<T,A,N>,
               input_output:CudaVec<T,CudaTensor1dPtr<T,A,N>,A>, batch_size: usize) -> AddBiasBatchArgs<'a,T,A,N> {
        AddBiasBatchArgs {
            bias: CudaConstPtr::new(bias),
            input_output: input_output,
            units_len: N,
            batch_size: batch_size
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for AddBiasBatchArgs<'a,T,A,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.bias,
            &mut self.input_output,
            &mut self.units_len,
            &mut self.batch_size
        ]
    }
}
/// Implementation of process to add bias to mini-batch
pub struct AddBiasBatch<'a,T,A,const N:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> AddBiasBatch<'a,T,A,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a AddBiasBatch instance
    pub fn new() -> AddBiasBatch<'a,T,A,N> {
        AddBiasBatch {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            n:PhantomData::<[();N]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for AddBiasBatch<'a,f32,A,N> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = addbias_batch_float as *const c_void;
    type Args = AddBiasBatchArgs<'a,f32,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0,
        }
    }
}
impl<'a,A,const N:usize> Kernel for AddBiasBatch<'a,f64,A,N> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = addbias_batch_double as *const c_void;
    type Args = AddBiasBatchArgs<'a,f64,A,N>;

    fn launch_config(&self, args: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 31) as c_uint / 32, y: (args.batch_size + 31) as c_uint / 32, z: 1 },
            block_dim: dim3 { x: 32, y: 32, z: 1 },
            shared_memory_size: 0,
        }
    }
}
/// Defines the type of the argument list passed to the kernel as arguments for bias addition.
pub struct AddBiasArgs<'a,T,A,const N:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    bias: CudaConstPtr<'a,CudaTensor1dPtr<T,A,N>>,
    pub input_output: CudaTensor1dPtr<T,A,N>,
    units_len: usize,
    batch_size: usize
}
/// Create an instance of the type of the argument list passed to the bias addition process
impl<'a,T,A,const N:usize> AddBiasArgs<'a,T,A,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a AddBiasArgs instance
    /// # Arguments
    /// * `bias` - bias.
    /// * `input_output` - input and output (Updated in-place)
    pub fn new(bias: &'a CudaTensor1dPtr<T,A,N>,
               input_output:CudaTensor1dPtr<T,A,N>) -> AddBiasArgs<'a,T,A,N> {
        AddBiasArgs {
            bias: CudaConstPtr::new(bias),
            input_output: input_output,
            units_len: N,
            batch_size: 1
        }
    }
}
impl<'a,T,A,const N:usize> KernelArgs for AddBiasArgs<'a,T,A,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.bias,
            &mut self.input_output,
            &mut self.units_len,
            &mut self.batch_size
        ]
    }
}
/// Implementation of the process of adding bias
pub struct AddBias<'a,T,A,const N:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    t:PhantomData<T>,
    a:PhantomData<A>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A,const N:usize> AddBias<'a,T,A,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T>,
          A: CudaAllocator + 'a {
    /// Create a AddBias instance
    pub fn new() -> AddBias<'a,T,A,N> {
        AddBias {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            n:PhantomData::<[();N]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,A,const N:usize> Kernel for AddBias<'a,f32,A,N> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = addbias_batch_float as *const c_void;
    type Args = AddBiasArgs<'a,f32,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0,
        }
    }
}
impl<'a,A,const N:usize> Kernel for AddBias<'a,f64,A,N> where A: CudaAllocator + 'a {
    const FUNC_PTR: *const c_void = addbias_batch_double as *const c_void;
    type Args = AddBiasArgs<'a,f64,A,N>;

    fn launch_config(&self, _: &Self::Args) -> KernelLaunchConfig {
        KernelLaunchConfig {
            grid_dim: dim3 { x: (N + 1023) as c_uint / 1024, y: 1, z: 1 },
            block_dim: dim3 { x: 1024, y: 1, z: 1 },
            shared_memory_size: 0,
        }
    }
}
