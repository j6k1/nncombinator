//! Implementation of a device that performs various calculations for neural networks

use std::marker::PhantomData;
use libc::{c_int, c_void};
use crate::cuda::{AsMutKernelPtr, CudaPtr, DataTypeInfo, Kernel, KernelArgs};

extern "C" {
    fn reduce_linear_batch_float(input: *const f32, output: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn reduce_linear_batch_double(input: *const f64, output: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_by_canonical_link_float(expected: *const f32, actual: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_by_canonical_link_double(expected: *const f64, actual: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
}
/// Defines the list that is passed to the cuda kernel function as arguments for the convolution calculation.
pub struct ReduceLinearBatchArgs<T> where T: DataTypeInfo {
    input: CudaPtr<T>,
    /// output
    pub output: CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list during convolution computation.
impl<T> ReduceLinearBatchArgs<T> where T: DataTypeInfo {
    /// Create a ReduceLinearBatchArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `output` - output
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch_count
    pub fn new(input:CudaPtr<T>,output:CudaPtr<T>,out_len:usize,batch_len:usize) -> ReduceLinearBatchArgs<T> {
        ReduceLinearBatchArgs {
            input: input,
            output: output,
            units_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for ReduceLinearBatchArgs<T> where T: DataTypeInfo {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.input,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
/// Implementation of convolutional computation
pub struct ReduceLinearBatch<T> where T: DataTypeInfo {
    t:PhantomData<T>
}
impl<T> ReduceLinearBatch<T> where T: DataTypeInfo {
    /// Create a ReduceLinearBatch instance
    pub fn new() -> ReduceLinearBatch<T> {
        ReduceLinearBatch {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for ReduceLinearBatch<f32> {
    const FUNC_PTR: *const c_void = reduce_linear_batch_float as *const c_void;
    type Args = ReduceLinearBatchArgs<f32>;
}
impl Kernel for ReduceLinearBatch<f64> {
    const FUNC_PTR: *const c_void = reduce_linear_batch_double as *const c_void;
    type Args = ReduceLinearBatchArgs<f64>;
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
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
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
