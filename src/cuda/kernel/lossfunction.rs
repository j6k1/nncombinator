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

pub struct LinearBatchMseArgs<T> where T: AsMutKernelPtr {
    expected: CudaPtr<T>,
    pub actual: CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
}
impl<T> LinearBatchMseArgs<T> where T: AsMutKernelPtr {
    pub fn new(r:CudaPtr<T>,t:CudaPtr<T>,out_len:usize,batch_len:usize) -> LinearBatchMseArgs<T> {
        LinearBatchMseArgs {
            expected: r,
            actual: t,
            units_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for LinearBatchMseArgs<T> where T: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearBatchMse<T> where T: AsMutKernelPtr {
    t:PhantomData<T>
}
impl<T> LinearBatchMse<T> where T: AsMutKernelPtr {
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

pub struct LinearBatchCrossEntropyArgs<T> where T: AsMutKernelPtr {
    expected: CudaPtr<T>,
    pub actual: CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
}
impl<T> LinearBatchCrossEntropyArgs<T> where T: AsMutKernelPtr {
    pub fn new(r:CudaPtr<T>,t:CudaPtr<T>,out_len:usize,batch_len:usize) -> LinearBatchCrossEntropyArgs<T> {
        LinearBatchCrossEntropyArgs {
            expected: r,
            actual: t,
            units_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for LinearBatchCrossEntropyArgs<T> where T: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearBatchCrossEntropy<T> where T: AsMutKernelPtr {
    t:PhantomData<T>
}
impl<T> LinearBatchCrossEntropy<T> where T: AsMutKernelPtr {
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

pub struct LinearBatchCrossEntropyMulticlassArgs<T> where T: AsMutKernelPtr {
    expected: CudaPtr<T>,
    pub actual: CudaPtr<T>,
    units_len: usize,
    batch_len: usize,
}
impl<T> LinearBatchCrossEntropyMulticlassArgs<T> where T: AsMutKernelPtr {
    pub fn new(r:CudaPtr<T>,t:CudaPtr<T>,out_len:usize,batch_len:usize) -> LinearBatchCrossEntropyMulticlassArgs<T> {
        LinearBatchCrossEntropyMulticlassArgs {
            expected: r,
            actual: t,
            units_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for LinearBatchCrossEntropyMulticlassArgs<T> where T: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct LinearBatchCrossEntropyMulticlass<T> where T: AsMutKernelPtr {
    t:PhantomData<T>
}
impl<T> LinearBatchCrossEntropyMulticlass<T> where T: AsMutKernelPtr {
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
