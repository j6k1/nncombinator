use std::marker::PhantomData;
use libc::{c_int, c_void};
use crate::cuda::{AsMutKernelPtr, CudaHostPtr, CudaPtr, Kernel, KernelArgs};

extern "C" {
    fn reduce_linear_batch_float(input: *const f32, output: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn reduce_linear_batch_double(input: *const f64, output: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_by_canonical_link_float(expected: *const f32, actual: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_by_canonical_link_double(expected: *const f64, actual: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
}

pub struct ReduceLinearBatchArgs<T> where T: AsMutKernelPtr {
    input: T,
    pub output: T,
    units_len: usize,
    batch_len: usize,
}
impl<T> ReduceLinearBatchArgs<T> where T: AsMutKernelPtr {
    pub fn new(input:T,output:T,out_len:usize,batch_len:usize) -> ReduceLinearBatchArgs<T> {
        ReduceLinearBatchArgs {
            input: input,
            output: output,
            units_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for ReduceLinearBatchArgs<T> where T: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.input,
            &mut self.output,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct ReduceLinearBatch<T> where T: AsMutKernelPtr {
    t:PhantomData<T>
}
impl<T> ReduceLinearBatch<T> where T: AsMutKernelPtr {
    pub fn new() -> ReduceLinearBatch<T> {
        ReduceLinearBatch {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for ReduceLinearBatch<CudaPtr<f32>> {
    const FUNC_PTR: *const c_void = reduce_linear_batch_float as *const c_void;
    type Args = ReduceLinearBatchArgs<CudaPtr<f32>>;
}
impl Kernel for ReduceLinearBatch<CudaPtr<f64>> {
    const FUNC_PTR: *const c_void = reduce_linear_batch_double as *const c_void;
    type Args = ReduceLinearBatchArgs<CudaPtr<f64>>;
}
pub struct LossLinearBatchByCanonicalLinkArgs<T> where T: AsMutKernelPtr {
    expected: T,
    pub actual: T,
    units_len: usize,
    batch_len: usize,
}
impl<T> LossLinearBatchByCanonicalLinkArgs<T> where T: AsMutKernelPtr {
    pub fn new(expected:T,actual:T,out_len:usize,batch_len:usize) -> LossLinearBatchByCanonicalLinkArgs<T> {
        LossLinearBatchByCanonicalLinkArgs {
            expected: expected,
            actual: actual,
            units_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<T> KernelArgs for LossLinearBatchByCanonicalLinkArgs<T> where T: AsMutKernelPtr {
    fn as_vec(&mut self) -> Vec<&mut dyn AsMutKernelPtr> {
        vec![
            &mut self.expected,
            &mut self.actual,
            &mut self.units_len,
            &mut self.batch_len
        ]
    }
}
pub struct LossLinearBatchByCanonicalLink<T> where T: AsMutKernelPtr {
    t:PhantomData<T>
}
impl<T> LossLinearBatchByCanonicalLink<T> where T: AsMutKernelPtr {
    pub fn new() -> LossLinearBatchByCanonicalLink<T> {
        LossLinearBatchByCanonicalLink {
            t: PhantomData::<T>
        }
    }
}
impl Kernel for LossLinearBatchByCanonicalLink<CudaPtr<f32>> {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_float as *const c_void;
    type Args = LossLinearBatchByCanonicalLinkArgs<CudaPtr<f32>>;
}
impl Kernel for LossLinearBatchByCanonicalLink<CudaPtr<f64>> {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_double as *const c_void;
    type Args = LossLinearBatchByCanonicalLinkArgs<CudaPtr<f64>>;
}
