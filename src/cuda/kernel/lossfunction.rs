//! Implementation of various loss functions
use std::marker::PhantomData;
use libc::{c_int, c_void};
use crate::cuda::{AsMutKernelPtr, CudaPtr, Kernel, KernelArgs};

extern "C" {
    fn loss_linear_batch_mse_derive_float(r: *const f32, t: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_mse_derive_double(r: *const f64, t: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_derive_float(r: *const f32, t: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_derive_double(r: *const f64, t: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_multiclass_derive_float(r: *const f32, t: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_multiclass_derive_double(r: *const f64, t: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
}
/// Defines the list passed to the cuda kernel function as the argument of mse.
pub struct LinearBatchMseArgs<T> where T: AsMutKernelPtr {
    /// expected value
    expected: CudaPtr<T>,
    /// actual value
    pub actual: CudaPtr<T>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list for computing the loss function mse.
impl<T> LinearBatchMseArgs<T> where T: AsMutKernelPtr {
    /// Create a LinearBatchMseArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(t:CudaPtr<T>,r:CudaPtr<T>,out_len:usize,batch_len:usize) -> LinearBatchMseArgs<T> {
        LinearBatchMseArgs {
            expected: t,
            actual: r,
            out_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for LinearBatchMseArgs<T> where T: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.out_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearBatchMse<T> where T: AsMutKernelPtr {
    t:PhantomData<T>
}
impl<T> LinearBatchMse<T> where T: AsMutKernelPtr {
    /// Create a LinearBatchMse instance
    pub fn new() -> LinearBatchMse<T> {
        LinearBatchMse {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for LinearBatchMse<f32> {
    const FUNC_PTR: *const c_void = loss_linear_batch_mse_derive_float as *const c_void;
    type Args = LinearBatchMseArgs<f32>;
}
impl Kernel for LinearBatchMse<f64> {
    const FUNC_PTR: *const c_void = loss_linear_batch_mse_derive_double as *const c_void;
    type Args = LinearBatchMseArgs<f64>;
}
/// Defines the list passed to the cuda kernel function as the argument of cross entropy.
pub struct LinearBatchCrossEntropyArgs<T> where T: AsMutKernelPtr {
    /// expected value
    expected: CudaPtr<T>,
    /// actual value
    pub actual: CudaPtr<T>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list for computing the loss function cross entropy.
impl<T> LinearBatchCrossEntropyArgs<T> where T: AsMutKernelPtr {
    /// Create a LinearBatchCrossEntropyArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(t:CudaPtr<T>,r:CudaPtr<T>,out_len:usize,batch_len:usize) -> LinearBatchCrossEntropyArgs<T> {
        LinearBatchCrossEntropyArgs {
            expected: t,
            actual: r,
            out_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for LinearBatchCrossEntropyArgs<T> where T: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.out_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearBatchCrossEntropy<T> where T: AsMutKernelPtr {
    t:PhantomData<T>
}
impl<T> LinearBatchCrossEntropy<T> where T: AsMutKernelPtr {
    /// Create a LinearBatchCrossEntropy instance
    pub fn new() -> LinearBatchCrossEntropy<T> {
        LinearBatchCrossEntropy {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for LinearBatchCrossEntropy<f32> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_derive_float as *const c_void;
    type Args = LinearBatchCrossEntropyArgs<f32>;
}
impl Kernel for LinearBatchCrossEntropy<f64> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_derive_double as *const c_void;
    type Args = LinearBatchCrossEntropyArgs<f64>;
}
/// Defines the list passed to the cuda kernel function as the argument of croos entropy multiclass
pub struct LinearBatchCrossEntropyMulticlassArgs<T> where T: AsMutKernelPtr {
    /// expected value
    expected: CudaPtr<T>,
    /// actual value
    pub actual: CudaPtr<T>,
    out_len: usize,
    batch_len: usize,
}
impl<T> LinearBatchCrossEntropyMulticlassArgs<T> where T: AsMutKernelPtr {
    /// Create a LinearBatchCrossEntropyMulticlassArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(t:CudaPtr<T>,r:CudaPtr<T>,out_len:usize,batch_len:usize) -> LinearBatchCrossEntropyMulticlassArgs<T> {
        LinearBatchCrossEntropyMulticlassArgs {
            expected: t,
            actual: r,
            out_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for LinearBatchCrossEntropyMulticlassArgs<T> where T: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.out_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearBatchCrossEntropyMulticlass<T> where T: AsMutKernelPtr {
    t:PhantomData<T>
}
impl<T> LinearBatchCrossEntropyMulticlass<T> where T: AsMutKernelPtr {
    /// Create a LinearBatchCrossEntropyMulticlass instance
    pub fn new() -> LinearBatchCrossEntropyMulticlass<T> {
        LinearBatchCrossEntropyMulticlass {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for LinearBatchCrossEntropyMulticlass<f32> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_multiclass_derive_float as *const c_void;
    type Args = LinearBatchCrossEntropyMulticlassArgs<f32>;
}
impl Kernel for LinearBatchCrossEntropyMulticlass<f64> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_multiclass_derive_double as *const c_void;
    type Args = LinearBatchCrossEntropyMulticlassArgs<f64>;
}
