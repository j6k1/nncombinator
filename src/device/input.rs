//! Implementation of the calculation process for input layers

use std::fmt::Debug;
use crate::cuda::ToCuda;
use crate::device::{Device, DeviceCpu, DeviceGpu};
use crate::error::{TypeConvertError};
use crate::layer::BatchDataType;
use crate::ope::UnitValue;

/// Trait that defines the function of processing data input in the input layer
/// into a form that can be passed to subsequent intermediate layers.
pub trait DeviceInput<U,I>: Device<U>
    where U: UnitValue<U>,
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

impl<U,I> DeviceInput<U,I> for DeviceCpu<U>
    where U: UnitValue<U>,
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

impl<U,I> DeviceInput<U,I> for DeviceGpu<U> 
    where U: UnitValue<U>,
          I: BatchDataType + ToCuda<U> + Debug + 'static,
          <I as BatchDataType>::Type: ToCuda<U> + Debug + 'static,
          <I as ToCuda<U>>::Output: Debug + 'static,
          <<I as BatchDataType>::Type as ToCuda<U>>::Output: Debug + 'static,
          DeviceGpu<U>: Device<U> {
    type Output = <I as ToCuda<U>>::Output;
    type BatchOutput = <<I as BatchDataType>::Type as ToCuda<U>>::Output;

    fn forward_input(&self,input: I) -> Result<Self::Output,TypeConvertError> {
        input.to_cuda(self)
    }

    fn batch_forward_input(&self, input: <I as BatchDataType>::Type) -> Result<Self::BatchOutput, TypeConvertError> {
        input.to_cuda(self)
    }
}