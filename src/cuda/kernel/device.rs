//! Implementation of a device that performs various calculations for neural networks

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::DerefMut;
use libc::{c_int, c_void, size_t};
use crate::cuda::{AsKernelPtr, CudaConstPtr, CudaMemoryPoolPtr, CudaPtr, CudaTensor1dPtr, CudaTensor2dPtr, DataTypeInfo, Kernel, KernelArgs};

extern "C" {
    fn reduce_linear_batch_float(input: *const f32, output: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn reduce_linear_batch_double(input: *const f64, output: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_by_canonical_link_float(expected: *const f32, actual: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_by_canonical_link_double(expected: *const f64, actual: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn forward_diff_linear_float(indexes: *const size_t, input: *const f32, units: *const f32, output: *mut f32, output_size: size_t, diff_len: size_t) -> c_void;
    fn forward_diff_linear_double(indexes: *const size_t, input: *const f64, units: *const f64, output: *mut f64, output_size: size_t, diff_len: size_t) -> c_void;
}
/// Defines the list that is passed to the cuda kernel function as arguments for the convolution calculation.
pub struct ReduceLinearBatchArgs<T,const N:usize> where T: DataTypeInfo + Debug + Default {
    input: CudaPtr<T>,
    /// output
    pub output: CudaTensor1dPtr<T,N>,
    units_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list during convolution computation.
impl<T,const N:usize> ReduceLinearBatchArgs<T,N> where T: DataTypeInfo + Debug + Default {
    /// Create a ReduceLinearBatchArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `output` - output
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch_count
    pub fn new(input:CudaPtr<T>,output:CudaTensor1dPtr<T,N>,out_len:usize,batch_len:usize) -> ReduceLinearBatchArgs<T,N> {
        ReduceLinearBatchArgs {
            input: input,
            output: output,
            units_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T,const N:usize> KernelArgs for ReduceLinearBatchArgs<T,N> where T: DataTypeInfo + Debug + Default {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.input,
            self.output.deref_mut(),
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
/// Implementation of convolutional computation
pub struct ReduceLinearBatch<T,const N:usize> where T: DataTypeInfo + Debug + Default {
    t:PhantomData<T>,
    n:PhantomData<[();N]>
}
impl<T,const N:usize> ReduceLinearBatch<T,N> where T: DataTypeInfo + Debug + Default {
    /// Create a ReduceLinearBatch instance
    pub fn new() -> ReduceLinearBatch<T,N> {
        ReduceLinearBatch {
            t: PhantomData::<T>,
            n:PhantomData::<[();N]>
        }
    }
}
impl<const N:usize> Kernel for ReduceLinearBatch<f32,N> {
    const FUNC_PTR: *const c_void = reduce_linear_batch_float as *const c_void;
    type Args = ReduceLinearBatchArgs<f32,N>;
}
impl<const N:usize> Kernel for ReduceLinearBatch<f64,N> {
    const FUNC_PTR: *const c_void = reduce_linear_batch_double as *const c_void;
    type Args = ReduceLinearBatchArgs<f64,N>;
}
/// Defines the list that is passed to the cuda kernel function as the argument for the calculation of applying canonical link.
pub struct LossLinearBatchByCanonicalLinkArgs<T> where T: DataTypeInfo {
    expected: CudaPtr<T>,
    /// Actual Value
    pub actual: CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
}
/// Create an instance of an object that represents the argument list for the canonical link application calculation.
impl<T> LossLinearBatchByCanonicalLinkArgs<T> where T: DataTypeInfo {
    /// Create a LossLinearBatchByCanonicalLinkArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(expected:CudaPtr<T>,actual:CudaPtr<T>,units_len:usize,batch_len:usize) -> LossLinearBatchByCanonicalLinkArgs<T> {
        LossLinearBatchByCanonicalLinkArgs {
            expected: expected,
            actual: actual,
            units_len: units_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for LossLinearBatchByCanonicalLinkArgs<T> where T: DataTypeInfo {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct LossLinearBatchByCanonicalLink<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> LossLinearBatchByCanonicalLink<T> where T: DataTypeInfo {
    /// Create a LossLinearBatchByCanonicalLink instance
    pub fn new() -> LossLinearBatchByCanonicalLink<T> {
        LossLinearBatchByCanonicalLink {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for LossLinearBatchByCanonicalLink<f32> {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_float as *const c_void;
    type Args = LossLinearBatchByCanonicalLinkArgs<f32>;
}
impl Kernel for LossLinearBatchByCanonicalLink<f64> {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_double as *const c_void;
    type Args = LossLinearBatchByCanonicalLinkArgs<f64>;
}
/// Defines the list that is passed to the cuda kernel function as arguments for forward propagation difference calculations.
pub struct DiffLinearForwardArgs<'a,T,const NI:usize,const NO:usize> where T: Debug + Default {
    indexes: CudaMemoryPoolPtr<usize>,
    input: CudaMemoryPoolPtr<T>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,NI,NO>>,
    pub output: CudaTensor1dPtr<T,NO>,
    output_size: usize,
    diff_len: usize
}
/// Create an instance of an object representing the argument list for the forward propagation difference calculation.
impl<'a,T,const NI:usize,const NO:usize> DiffLinearForwardArgs<'a,T,NI,NO> where T: DataTypeInfo + Debug + Default {
    /// Create a DiffLinearForwardArgs instance
    /// # Arguments
    ///
    /// * `indexes` - List of index corresponding to inputs for difference calculation
    /// * `input` - Value of input used for difference calculation
    /// * `units` - List of weights to be multiplied by the input
    /// * `output` - output
    /// * `ountput_size` - Number of Outputs
    /// * `diff_len` - Number of differential inputs
    pub fn new(indexes:CudaMemoryPoolPtr<usize>,input:CudaMemoryPoolPtr<T>,
               units: &'a CudaTensor2dPtr<T,NI,NO>,
               output: CudaTensor1dPtr<T,NO>,
               output_size:usize,diff_len:usize)
        -> DiffLinearForwardArgs<'a,T,NI,NO> {
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
impl<'a,T,const NI:usize,const NO:usize> KernelArgs for DiffLinearForwardArgs<'a,T,NI,NO> where T: DataTypeInfo + Debug + Default {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            &mut self.indexes,
            &mut self.input,
            &mut self.units,
            self.output.deref_mut(),
            &mut self.output_size,
            &mut self.diff_len
        ]
    }
}
pub struct DiffLinearForward<'a,T,const NI:usize,const NO:usize> where T: DataTypeInfo + Debug + Default {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const NI:usize,const NO:usize> DiffLinearForward<'a,T,NI,NO> where T: DataTypeInfo + Debug + Default {
    /// Create a DiffLinearForward instance
    pub fn new() -> DiffLinearForward<'a,T,NI,NO> {
        DiffLinearForward {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const NI:usize,const NO:usize> Kernel for DiffLinearForward<'a,f32,NI,NO> {
    const FUNC_PTR: *const c_void = forward_diff_linear_float as *const c_void;
    type Args = DiffLinearForwardArgs<'a,f32,NI,NO>;
}
impl<'a,const NI:usize,const NO:usize> Kernel for DiffLinearForward<'a,f64,NI,NO> {
    const FUNC_PTR: *const c_void = forward_diff_linear_double as *const c_void;
    type Args = DiffLinearForwardArgs<'a,f64,NI,NO>;
}
