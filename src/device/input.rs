//! Implementation of the calculation process for input layers

use std::fmt::Debug;
use crate::device::{Device, DeviceCpu};
use crate::error::{TypeConvertError};
use crate::layer::BatchDataType;
#[cfg(feature = "cuda")]
use crate::cuda::allocator::CudaAllocator;
#[cfg(feature = "cuda")]
use crate::cuda::{ToCuda};
#[cfg(feature = "cuda")]
use crate::device::{DeviceGpu};

/// Trait that defines the function of processing data input in the input layer
/// into a form that can be passed to subsequent intermediate layers.
pub trait DeviceInput<U,I>: Device<U>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: BatchDataType + Debug + 'static,
          <I as BatchDataType>::Type: Debug + 'static {
    type Output: Debug + 'static;
    type BatchOutput: Debug + 'static;

    /// Type conversion during forward propagation
    /// # Arguments
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TypeConvertError`]
    ///
    fn forward_input(&self,input: I) -> Result<Self::Output,TypeConvertError>;

    /// Type conversion during forward propagation in batch
    /// # Arguments
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TypeConvertError`]
    ///
    fn batch_forward_input(&self,input: <I as BatchDataType>::Type) -> Result<Self::BatchOutput,TypeConvertError>;
}

impl<U,I> DeviceInput<U,I> for DeviceCpu
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: BatchDataType + Debug + 'static,
          <I as BatchDataType>::Type: Debug + 'static {
    type Output = I;
    type BatchOutput = <I as BatchDataType>::Type;

    fn forward_input(&self,input: I) -> Result<Self::Output,TypeConvertError> {
        Ok(input)
    }

    fn batch_forward_input(&self,input: <I as BatchDataType>::Type) -> Result<Self::BatchOutput,TypeConvertError> {
        Ok(input)
    }
}
#[cfg(feature = "cuda")]

impl<U,I,A> DeviceInput<U,I> for DeviceGpu<A>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: BatchDataType + ToCuda<U,A> + Debug + 'static,
          <I as BatchDataType>::Type: ToCuda<U,A> + Debug + 'static,
          <I as ToCuda<U,A>>::Output: Debug + 'static,
          <<I as BatchDataType>::Type as ToCuda<U,A>>::Output: Debug + 'static,
          A: CudaAllocator,
          DeviceGpu<A>: Device<U> {
    type Output = <I as ToCuda<U,A>>::Output;
    type BatchOutput = <<I as BatchDataType>::Type as ToCuda<U,A>>::Output;

    fn forward_input(&self,input: I) -> Result<Self::Output,TypeConvertError> {
        input.to_cuda(self)
    }

    fn batch_forward_input(&self, input: <I as BatchDataType>::Type) -> Result<Self::BatchOutput, TypeConvertError> {
        input.to_cuda(self)
    }
}