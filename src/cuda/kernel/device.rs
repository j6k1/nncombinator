//! Implementation of a device that performs various calculations for neural networks

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::DerefMut;
use libc::{c_int, c_void, size_t};
use crate::cuda::{AsKernelPtr, CudaConstPtr, CudaMemoryPoolPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaTensor2dPtr, CudaVec, CudaVecView, DataTypeInfo, Kernel, KernelArgs};
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
}
/// Defines the list that is passed to the cuda kernel function as arguments for the convolution calculation.
pub struct ReduceLinearBatchArgs<'a,T,const N:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    input: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    /// output
    pub output: CudaTensor1dPtr<T,N>,
    units_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list during convolution computation.
impl<'a,T,const N:usize> ReduceLinearBatchArgs<'a,T,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a ReduceLinearBatchArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `output` - output (All elements must be initialized to zero.)
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch_count
    pub fn new(input:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,
               output:CudaTensor1dPtr<T,N>,out_len:usize,batch_len:usize) -> ReduceLinearBatchArgs<'a,T,N> {
        ReduceLinearBatchArgs {
            input: CudaConstPtr::new(input),
            output: output,
            units_len: out_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for ReduceLinearBatchArgs<'a,T,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
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
pub struct ReduceLinearBatch<'a,T,const N:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>,
    n:PhantomData<[();N]>
}
impl<'a,T,const N:usize> ReduceLinearBatch<'a,T,N>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a ReduceLinearBatch instance
    pub fn new() -> ReduceLinearBatch<'a,T,N> {
        ReduceLinearBatch {
            t: PhantomData::<T>,
            l:PhantomData::<&'a ()>,
            n:PhantomData::<[();N]>
        }
    }
}
impl<'a,const N:usize> Kernel for ReduceLinearBatch<'a,f32,N> {
    const FUNC_PTR: *const c_void = reduce_linear_batch_float as *const c_void;
    type Args = ReduceLinearBatchArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for ReduceLinearBatch<'a,f64,N> {
    const FUNC_PTR: *const c_void = reduce_linear_batch_double as *const c_void;
    type Args = ReduceLinearBatchArgs<'a,f64,N>;
}
/// Defines the list that is passed to the cuda kernel function as arguments
/// for the calculation that applies the canonical link during the mini-batch execution.
pub struct LossLinearBatchByCanonicalLinkArgs<'a,T,const N:usize>
    where T: DataTypeInfo + UnitValue<T> {
    expected: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    /// Actual Value
    actual: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,N>>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,N>>,
    units_len: usize,
    batch_len: usize,
}
/// Create an instance of an object representing the argument list
/// for the regular link application calculation at the time of mini-batch execution.
impl<'a,T,const N:usize> LossLinearBatchByCanonicalLinkArgs<'a,T,N>
    where T: DataTypeInfo + UnitValue<T> {
    /// Create a LossLinearBatchByCanonicalLinkArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    /// * `batch_len` - batch count
    pub fn new(expected:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,
               actual:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,N>>,
               output: CudaVec<T,CudaTensor1dPtr<T,N>>,
               units_len:usize,batch_len:usize) -> LossLinearBatchByCanonicalLinkArgs<'a,T,N> {
        LossLinearBatchByCanonicalLinkArgs {
            expected: CudaConstPtr::new(expected),
            actual: CudaConstPtr::new(actual),
            output: output,
            units_len: units_len,
            batch_len: batch_len
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for LossLinearBatchByCanonicalLinkArgs<'a,T,N>
    where T: DataTypeInfo + UnitValue<T> {
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
pub struct LossLinearBatchByCanonicalLink<'a,T,const N:usize>
    where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> LossLinearBatchByCanonicalLink<'a,T,N>
    where T: DataTypeInfo + UnitValue<T> {
    /// Create a LossLinearBatchByCanonicalLink instance
    pub fn new() -> LossLinearBatchByCanonicalLink<'a,T,N> {
        LossLinearBatchByCanonicalLink {
            t: PhantomData::<T>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for LossLinearBatchByCanonicalLink<'a,f32,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_float as *const c_void;
    type Args = LossLinearBatchByCanonicalLinkArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for LossLinearBatchByCanonicalLink<'a,f64,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_double as *const c_void;
    type Args = LossLinearBatchByCanonicalLinkArgs<'a,f64,N>;
}
/// Defines the list that is passed to the cuda kernel function as the argument for the calculation of applying canonical link.
pub struct LossLinearByCanonicalLinkArgs<'a,T,const N:usize>
    where T: DataTypeInfo + UnitValue<T> {
    expected: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    /// Actual Value
    actual: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,N>>,
    pub output: CudaTensor1dPtr<T,N>,
    units_len: usize,
    batch_len: usize,
}
/// Create an instance of an object that represents the argument list for the canonical link application calculation.
impl<'a,T,const N:usize> LossLinearByCanonicalLinkArgs<'a,T,N>
    where T: DataTypeInfo + UnitValue<T> {
    /// Create a LossLinearByCanonicalLinkArgs instance
    /// # Arguments
    /// * `expected` - Expected Value
    /// * `actual` - Actual Value
    /// * `out_len` - Number of scalar values in output
    pub fn new(expected:&'a CudaTensor1dPtrView<'a,T,N>,
               actual:&'a CudaTensor1dPtrView<'a,T,N>,
               output: CudaTensor1dPtr<T,N>,
               units_len:usize) -> LossLinearByCanonicalLinkArgs<'a,T,N> {
        LossLinearByCanonicalLinkArgs {
            expected: CudaConstPtr::new(expected),
            actual: CudaConstPtr::new(actual),
            output: output,
            units_len: units_len,
            batch_len: 1
        }
    }
}
impl<'a,T,const N:usize> KernelArgs for LossLinearByCanonicalLinkArgs<'a,T,N>
    where T: DataTypeInfo + UnitValue<T> {
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
pub struct LossLinearByCanonicalLink<'a,T,const N:usize>
    where T: DataTypeInfo + UnitValue<T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const N:usize> LossLinearByCanonicalLink<'a,T,N>
    where T: DataTypeInfo + UnitValue<T> {
    /// Create a LossLinearByCanonicalLink instance
    pub fn new() -> LossLinearByCanonicalLink<'a,T,N> {
        LossLinearByCanonicalLink {
            t: PhantomData::<T>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,const N:usize> Kernel for LossLinearByCanonicalLink<'a,f32,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_float as *const c_void;
    type Args = LossLinearByCanonicalLinkArgs<'a,f32,N>;
}
impl<'a,const N:usize> Kernel for LossLinearByCanonicalLink<'a,f64,N> {
    const FUNC_PTR: *const c_void = loss_linear_batch_by_canonical_link_double as *const c_void;
    type Args = LossLinearByCanonicalLinkArgs<'a,f64,N>;
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
/// Defines the list that is passed to the cuda kernel function
/// as arguments for the mini-batch computation
/// of forward propagation of linear layers.
pub struct ForwardLinearBatchArgs<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    input: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,NI>>>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,NI,NO>>,
    bias: CudaConstPtr<'a,CudaTensor1dPtr<T,NO>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,NO>>,
    input_len: usize,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing the argument list during
/// the forward propagation calculation of a mini-batch of linear layers.
impl<'a,T,const NI:usize,const NO:usize> ForwardLinearBatchArgs<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a ForwardLinearBatchArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `units` - weight
    /// * `bias` - bias
    /// * `output` - output (All elements must be initialized to zero.)
    /// * `batch_len` - batch_count
    pub fn new(input: &'a CudaVecView<'a,T,CudaTensor1dPtr<T,NI>>,
               units: &'a CudaTensor2dPtr<T,NI,NO>,
               bias: &'a CudaTensor1dPtr<T,NO>,
               output:CudaVec<T,CudaTensor1dPtr<T,NO>>, batch_size: usize) -> ForwardLinearBatchArgs<'a,T,NI,NO> {
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
impl<'a,T,const NI:usize,const NO:usize> KernelArgs for ForwardLinearBatchArgs<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
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
pub struct ForwardLinearBatch<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const NI:usize,const NO:usize> ForwardLinearBatch<'a,T,NI,NO,>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a ForwardLinearBatch instance
    pub fn new() -> ForwardLinearBatch<'a,T,NI,NO> {
        ForwardLinearBatch {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,const NI:usize,const NO:usize> Kernel for ForwardLinearBatch<'a,f32,NI,NO> {
    const FUNC_PTR: *const c_void = forward_linear_batch_float as *const c_void;
    type Args = ForwardLinearBatchArgs<'a,f32,NI,NO>;
}
impl<'a,const NI:usize,const NO:usize> Kernel for ForwardLinearBatch<'a,f64,NI,NO> {
    const FUNC_PTR: *const c_void = forward_linear_batch_double as *const c_void;
    type Args = ForwardLinearBatchArgs<'a,f64,NI,NO>;
}
/// Defines the list that is passed to the cuda kernel function as arguments for the computation
/// of the forward propagation of the linear layer.
pub struct ForwardLinearArgs<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    input: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,NI>>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,NI,NO>>,
    bias: CudaConstPtr<'a,CudaTensor1dPtr<T,NO>>,
    pub output: CudaTensor1dPtr<T,NO>,
    input_len: usize,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing the argument list during
/// the forward propagation calculation of the linear layer.
impl<'a,T,const NI:usize,const NO:usize> ForwardLinearArgs<'a, T, NI, NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a ForwardLinearArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `units` - weight
    /// * `bias` - bias.
    /// * `output` - output (All elements must be initialized to zero.)
    pub fn new(input: &'a CudaTensor1dPtrView<'a,T,NI>,
               units: &'a CudaTensor2dPtr<T,NI,NO>,
               bias: &'a CudaTensor1dPtr<T,NO>,
               output:CudaTensor1dPtr<T,NO>) -> ForwardLinearArgs<'a, T, NI, NO> {
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
impl<'a,T,const NI:usize,const NO:usize> KernelArgs for ForwardLinearArgs<'a, T, NI, NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
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
pub struct ForwardLinear<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const NI:usize,const NO:usize> ForwardLinear<'a, T, NI, NO, >
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a ForwardLinear instance
    pub fn new() -> ForwardLinear<'a, T, NI, NO> {
        ForwardLinear {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,const NI:usize,const NO:usize> Kernel for ForwardLinear<'a, f32, NI, NO> {
    const FUNC_PTR: *const c_void = forward_linear_batch_float as *const c_void;
    type Args = ForwardLinearArgs<'a,f32,NI,NO>;
}
impl<'a,const NI:usize,const NO:usize> Kernel for ForwardLinear<'a, f64, NI, NO> {
    const FUNC_PTR: *const c_void = forward_linear_batch_double as *const c_void;
    type Args = ForwardLinearArgs<'a,f64,NI,NO>;
}
/// Defines the list passed to the cuda kernel function as arguments
/// for the computation of the error back propagation of a mini-batch of linear layers.
pub struct BackwardLinearBatchArgs<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    loss: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,NO>>>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,NI,NO>>,
    pub output: CudaVec<T,CudaTensor1dPtr<T,NI>>,
    input_len: usize,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing a list of arguments
/// during the computation of the error back propagation of a mini-batch of linear layers.
impl<'a,T,const NI:usize,const NO:usize> BackwardLinearBatchArgs<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a BackwardLinearBatchArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `units` - weight
    /// * `output` - output (All elements must be initialized to zero.)
    /// * `batch_len` - batch_count
    pub fn new(loss:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,NO>>,
               units: &'a CudaTensor2dPtr<T,NI,NO>,
               output:CudaVec<T,CudaTensor1dPtr<T,NI>>, batch_size: usize) -> BackwardLinearBatchArgs<'a,T,NI,NO> {
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
impl<'a,T,const NI:usize,const NO:usize> KernelArgs for BackwardLinearBatchArgs<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
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
pub struct BackwardLinearBatch<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const NI:usize,const NO:usize> BackwardLinearBatch<'a,T,NI,NO,>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a BackwardLinearBatch instance
    pub fn new() -> BackwardLinearBatch<'a,T,NI,NO> {
        BackwardLinearBatch {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,const NI:usize,const NO:usize> Kernel for BackwardLinearBatch<'a,f32,NI,NO> {
    const FUNC_PTR: *const c_void = backward_linear_batch_float as *const c_void;
    type Args = BackwardLinearBatchArgs<'a,f32,NI,NO>;
}
impl<'a,const NI:usize,const NO:usize> Kernel for BackwardLinearBatch<'a,f64,NI,NO> {
    const FUNC_PTR: *const c_void = backward_linear_batch_double as *const c_void;
    type Args = BackwardLinearBatchArgs<'a,f64,NI,NO>;
}
/// Defines the list that is passed to the cuda kernel function as arguments for
/// the computation of the error back propagation of the linear layer.
pub struct BackwardLinearArgs<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    loss: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,NO>>,
    units: CudaConstPtr<'a,CudaTensor2dPtr<T,NI,NO>>,
    pub output: CudaTensor1dPtr<T,NI>,
    input_len: usize,
    output_len: usize,
    batch_size: usize
}
/// Create an instance of an object representing the list of arguments during
/// the computation of the error back propagation of the linear layer.
impl<'a,T,const NI:usize,const NO:usize> BackwardLinearArgs<'a, T, NI, NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a BackwardLinearArgs instance
    /// # Arguments
    /// * `input` - input
    /// * `units` - weight
    /// * `output` - output (All elements must be initialized to zero.)
    pub fn new(loss:&'a CudaTensor1dPtrView<'a,T,NO>,
               units: &'a CudaTensor2dPtr<T,NI,NO>,
               output:CudaTensor1dPtr<T,NI>) -> BackwardLinearArgs<'a, T, NI, NO> {
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
impl<'a,T,const NI:usize,const NO:usize> KernelArgs for BackwardLinearArgs<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
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
pub struct BackwardLinear<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    l:PhantomData<&'a ()>
}
impl<'a,T,const NI:usize,const NO:usize> BackwardLinear<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a BackwardLinear instance
    pub fn new() -> BackwardLinear<'a,T,NI,NO> {
        BackwardLinear {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            l:PhantomData::<&'a ()>
        }
    }
}
impl<'a,const NI:usize,const NO:usize> Kernel for BackwardLinear<'a,f32,NI,NO> {
    const FUNC_PTR: *const c_void = backward_linear_batch_float as *const c_void;
    type Args = BackwardLinearArgs<'a,f32,NI,NO>;
}
impl<'a,const NI:usize,const NO:usize> Kernel for BackwardLinear<'a,f64,NI,NO> {
    const FUNC_PTR: *const c_void = backward_linear_batch_double as *const c_void;
    type Args = BackwardLinearArgs<'a,f64,NI,NO>;
}
/// Defines the list that is passed to the cuda kernel function as arguments
/// for the calculation of the amount of update of the linear layer weights during the mini-batch.
pub struct LinearGradientBatchArgs<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    loss: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,NO>>>,
    input: CudaConstPtr<'a,CudaVecView<'a,T,CudaTensor1dPtr<T,NI>>>,
    pub output: CudaTensor2dPtr<T,NI,NO>,
    input_len: usize,
    output_len: usize,
    units_size: usize,
    batch_size: usize
}
/// Create an instance of an object representing the list of arguments
///s for calculating the amount of updates to the linear layer weights during a mini-batch run.
impl<'a,T,const NI:usize,const NO:usize> LinearGradientBatchArgs<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a LinearGradientBatchArgs instance
    /// # Arguments
    /// * `loss` - loss
    /// * `input` - input
    /// * `output` - output (All elements must be initialized to zero.)
    /// * `batch_len` - batch_count
    pub fn new(loss:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,NO>>, input:&'a CudaVecView<'a,T,CudaTensor1dPtr<T,NI>>,
               output:CudaTensor2dPtr<T,NI,NO>, batch_size: usize) -> LinearGradientBatchArgs<'a,T,NI,NO> {
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
impl<'a,T,const NI:usize,const NO:usize> KernelArgs for LinearGradientBatchArgs<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
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
pub struct LinearGradientBatch<'a,T,const NI:usize,const NO:usize> where T: DataTypeInfo + Debug + Default {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    lt:PhantomData<&'a ()>
}
impl<'a,T,const NI:usize,const NO:usize> LinearGradientBatch<'a,T,NI,NO,>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a LinearGradientBatch instance
    pub fn new() -> LinearGradientBatch<'a,T,NI,NO> {
        LinearGradientBatch {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            lt:PhantomData::<&'a ()>
        }
    }
}
impl<'a,const NI:usize,const NO:usize> Kernel for LinearGradientBatch<'a,f32,NI,NO> {
    const FUNC_PTR: *const c_void = linear_gradient_batch_float as *const c_void;
    type Args = LinearGradientBatchArgs<'a,f32,NI,NO>;
}
impl<'a,const NI:usize,const NO:usize> Kernel for LinearGradientBatch<'a,f64,NI,NO> {
    const FUNC_PTR: *const c_void = linear_gradient_batch_double as *const c_void;
    type Args = LinearGradientBatchArgs<'a,f64,NI,NO>;
}
/// Defines the list that is passed to the cuda kernel function as arguments
/// for the computation of the amount of update of the linear layer weights.
pub struct LinearGradientArgs<'a,T,const NI:usize,const NO:usize>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    loss: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,NO>>,
    input: CudaConstPtr<'a,CudaTensor1dPtrView<'a,T,NI>>,
    pub output: CudaTensor2dPtr<T,NI,NO>,
    input_len: usize,
    output_len: usize,
    units_size: usize,
    batch_size: usize
}
/// Create an instance of an object representing the argument list
/// for the calculation of the update amount of the linear layer weights.
impl<'a,T,const NI:usize,const NO:usize> LinearGradientArgs<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a LinearGradientBatchArgs instance
    /// # Arguments
    /// * `loss` - loss
    /// * `input` - input
    /// * `output` - output (All elements must be initialized to zero.)
    pub fn new(loss:&'a CudaTensor1dPtrView<'a,T,NO>, input:&'a CudaTensor1dPtrView<'a,T,NI>,
               output:CudaTensor2dPtr<T,NI,NO>) -> LinearGradientArgs<'a,T,NI,NO> {
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
impl<'a,T,const NI:usize,const NO:usize> KernelArgs for LinearGradientArgs<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
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
pub struct LinearGradient<'a,T,const NI:usize,const NO:usize> where T: DataTypeInfo + Debug + Default {
    t:PhantomData<T>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>,
    lt:PhantomData<&'a ()>
}
impl<'a,T,const NI:usize,const NO:usize> LinearGradient<'a,T,NI,NO>
    where T: DataTypeInfo + Debug + Default + UnitValue<T> {
    /// Create a LinearGradientBatch instance
    pub fn new() -> LinearGradient<'a,T,NI,NO> {
        LinearGradient {
            t: PhantomData::<T>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>,
            lt:PhantomData::<&'a ()>
        }
    }
}
impl<'a,const NI:usize,const NO:usize> Kernel for LinearGradient<'a,f32,NI,NO> {
    const FUNC_PTR: *const c_void = linear_gradient_batch_float as *const c_void;
    type Args = LinearGradientArgs<'a,f32,NI,NO>;
}
impl<'a,const NI:usize,const NO:usize> Kernel for LinearGradient<'a,f64,NI,NO> {
    const FUNC_PTR: *const c_void = linear_gradient_batch_double as *const c_void;
    type Args = LinearGradientArgs<'a,f64,NI,NO>;
}
