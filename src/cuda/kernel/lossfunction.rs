//! Implementation of various loss functions
use std::marker::PhantomData;
use libc::{c_int, c_void};
use crate::cuda::{AsKernelPtr, CudaConstPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, DataTypeInfo, Kernel, KernelArgs};
use crate::ope::UnitValue;

extern "C" {
    fn loss_linear_batch_mse_derive_float(r: *const f32, t: *const f32, output: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_mse_derive_double(r: *const f64, t: *const f64, output: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_derive_float(r: *const f32, t: *const f32, output: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_derive_double(r: *const f64, t: *const f64, output: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_multiclass_derive_float(r: *const f32, t: *const f32, output: *mut f32, nlen: c_int, batch_size: c_int) -> c_void;
    fn loss_linear_batch_cross_entropy_multiclass_derive_double(r: *const f64, t: *const f64, output: *mut f64, nlen: c_int, batch_size: c_int) -> c_void;
}
/// Define a list to be passed to the cuda kernel function during mini-batch execution as the argument of mse.
pub struct LinearBatchMseArgs<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    /// expected value
    expected: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,N>>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the list of arguments to
/// compute the loss function mse during mini-batch execution.
impl<'a,T,const N:usize> LinearBatchMseArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearBatchMseArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(t:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,r:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,
               output: CudaVec<T,CudaTensor1dPtr<T,N>>,out_len:usize,batch_len:usize) -> LinearBatchMseArgs<'a,T,N> {
        LinearBatchMseArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for LinearBatchMseArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
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
pub struct LinearBatchMse<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> LinearBatchMse<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearBatchMse instance
    pub fn new() -> LinearBatchMse<'a,T,N> {
        LinearBatchMse {
            t: PhantomData::<T>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for LinearBatchMse<'a,f32,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_mse_derive_float as *const c_void;
    type Args = LinearBatchMseArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for LinearBatchMse<'a,f64,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_mse_derive_double as *const c_void;
    type Args = LinearBatchMseArgs<'a,f64,N>;
}
/// Defines the list passed to the cuda kernel function as the argument of mse.
pub struct LinearMseArgs<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    /// expected value
    expected: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    pub output: CudaTensor1dPtr<T,N>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list for computing the loss function mse.
impl<'a,T,const N:usize> LinearMseArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearMseArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    pub fn new(t:&'a CudaTensor1dPtrView<'a,T,N>,
               r:&'a CudaTensor1dPtrView<'a,T,N>,
               output: CudaTensor1dPtr<T,N>,
               out_len:usize) -> LinearMseArgs<'a,T,N> {
        LinearMseArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: 1
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for LinearMseArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
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
pub struct LinearMse<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> LinearMse<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearMse instance
    pub fn new() -> LinearMse<'a,T,N> {
        LinearMse {
            t: PhantomData::<T>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for LinearMse<'a,f32,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_mse_derive_float as *const c_void;
    type Args = LinearMseArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for LinearMse<'a,f64,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_mse_derive_double as *const c_void;
    type Args = LinearMseArgs<'a,f64,N>;
}
/// Defines the list that is passed to the cuda kernel function as cross-entropy arguments during mini-batch execution.
pub struct LinearBatchCrossEntropyArgs<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    /// expected value
    expected: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,N>>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing a list of arguments to calculate
/// the result of passing a mini-batch to the loss function cross entropy.
impl<'a,T,const N:usize> LinearBatchCrossEntropyArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearBatchCrossEntropyArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(t:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,
               r:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,
               output: CudaVec<T,CudaTensor1dPtr<T,N>>,
               out_len:usize,batch_len:usize) -> LinearBatchCrossEntropyArgs<'a,T,N> {
        LinearBatchCrossEntropyArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for LinearBatchCrossEntropyArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
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
pub struct LinearBatchCrossEntropy<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> LinearBatchCrossEntropy<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearBatchCrossEntropy instance
    pub fn new() -> LinearBatchCrossEntropy<'a,T,N> {
        LinearBatchCrossEntropy {
            t: PhantomData::<T>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for LinearBatchCrossEntropy<'a,f32,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_derive_float as *const c_void;
    type Args = LinearBatchCrossEntropyArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for LinearBatchCrossEntropy<'a,f64,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_derive_double as *const c_void;
    type Args = LinearBatchCrossEntropyArgs<'a,f64,N>;
}
/// Defines the list passed to the cuda kernel function as the argument of cross entropy.
pub struct LinearCrossEntropyArgs<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    /// expected value
    expected: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    pub output: CudaTensor1dPtr<T,N>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list for computing the loss function cross entropy.
impl<'a,T,const N:usize> LinearCrossEntropyArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearCrossEntropyArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    pub fn new(t:&'a CudaTensor1dPtrView<'a,T,N>,
               r:&'a CudaTensor1dPtrView<'a,T,N>,
               output: CudaTensor1dPtr<T,N>,
               out_len:usize) -> LinearCrossEntropyArgs<'a, T, N> {
        LinearCrossEntropyArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: 1
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for LinearCrossEntropyArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
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
pub struct LinearCrossEntropy<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> LinearCrossEntropy<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearCrossEntropy instance
    pub fn new() -> LinearCrossEntropy<'a,T,N> {
        LinearCrossEntropy {
            t: PhantomData::<T>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for LinearCrossEntropy<'a,f32,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_derive_float as *const c_void;
    type Args = LinearCrossEntropyArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for LinearCrossEntropy<'a,f64,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_derive_double as *const c_void;
    type Args = LinearCrossEntropyArgs<'a,f64,N>;
}
/// Defines the list that is passed to the cuda kernel function as arguments
/// to the croos entropy multiclass during mini-batch execution.
pub struct LinearBatchCrossEntropyMulticlassArgs<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    /// expected value
    expected: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,N>>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing a list of arguments to compute the result of passing a mini-batch
/// to the loss function cross entropy multiclass.
impl<'a,T,const N:usize> LinearBatchCrossEntropyMulticlassArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearBatchCrossEntropyMulticlassArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(t:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,
               r:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,
               output: CudaVec<T,CudaTensor1dPtr<T,N>>,
               out_len:usize,batch_len:usize) -> LinearBatchCrossEntropyMulticlassArgs<'a,T,N> {
        LinearBatchCrossEntropyMulticlassArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for LinearBatchCrossEntropyMulticlassArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
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
pub struct LinearBatchCrossEntropyMulticlass<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> LinearBatchCrossEntropyMulticlass<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearBatchCrossEntropyMulticlass instance
    pub fn new() -> LinearBatchCrossEntropyMulticlass<'a,T,N> {
        LinearBatchCrossEntropyMulticlass {
            t: PhantomData::<T>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for LinearBatchCrossEntropyMulticlass<'a,f32,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_multiclass_derive_float as *const c_void;
    type Args = LinearBatchCrossEntropyMulticlassArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for LinearBatchCrossEntropyMulticlass<'a,f64,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_multiclass_derive_double as *const c_void;
    type Args = LinearBatchCrossEntropyMulticlassArgs<'a,f64,N>;
}
/// Defines the list passed to the cuda kernel function as the argument of croos entropy multiclass
pub struct LinearCrossEntropyMulticlassArgs<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    /// expected value
    expected: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// actual value
    actual: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    pub output: CudaTensor1dPtr<T,N>,
    out_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list for computing the loss function cross entropy multiclass.
impl<'a,T,const N:usize> LinearCrossEntropyMulticlassArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearCrossEntropyMulticlassArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    pub fn new(t:&'a CudaTensor1dPtrView<T,N>,
               r:&'a CudaTensor1dPtrView<'a,T,N>,
               output: CudaTensor1dPtr<T,N>,
               out_len:usize) -> LinearCrossEntropyMulticlassArgs<'a,T,N> {
        LinearCrossEntropyMulticlassArgs {
            expected: CudaConstPtr::new(t),
            actual: CudaConstPtr::new(r),
            output: output,
            out_len: out_len,
            batch_len: 1
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for LinearCrossEntropyMulticlassArgs<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
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
pub struct LinearCrossEntropyMulticlass<'a,T,const N:usize> where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    n:PhantomData<[();N]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> LinearCrossEntropyMulticlass<'a,T,N> where T: DataTypeInfo + UnitValue<T> {
    /// Create a LinearCrossEntropyMulticlass instance
    pub fn new() -> LinearCrossEntropyMulticlass<'a,T,N> {
        LinearCrossEntropyMulticlass {
            t: PhantomData::<T>,
            n: PhantomData::<[();N]>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for LinearCrossEntropyMulticlass<'a,f32,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_multiclass_derive_float as *const c_void;
    type Args = LinearCrossEntropyMulticlassArgs<'a, f32, N>;
}
impl<'a,const N:usize> Kernel for LinearCrossEntropyMulticlass<'a,f64,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_cross_entropy_multiclass_derive_double as *const c_void;
    type Args = LinearCrossEntropyMulticlassArgs<'a,f64,N>;
}
