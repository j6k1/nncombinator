//! Implementation of the calculation process for bridge layers

use std::fmt::Debug;
use crate::cuda::allocator::CudaAllocator;
use crate::cuda::{AsCudaView, CudaView};
use crate::device::{DeviceCpu, DeviceGpu};
use crate::error::{CudaError, EvaluateError, TrainingError, TypeConvertError};
use crate::layer::{BatchDataType};

/// Trait that defines the implementation of various calculation processes in the bridge layer
pub trait DeviceBridge<U,PI,CI>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          PI: BatchDataType + Debug,
          CI: BatchDataType + Debug {
    /// Perform generalization of bias data
    /// # Arguments
    /// `input` - input data.
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn bridge_forward<'a>(&self, input:&'a PI) -> Result<CI, EvaluateError>;
    /// Error back propagation calculation
    /// # Arguments
    /// * `input` - input data.
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn bridge_backward<'a>(&self, input: &'a CI) -> Result<PI, TrainingError>;
    /// Forward propagation calculation in batch
    /// # Arguments
    /// * `input` - input data.
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_bridge_forward<'a>(&self,input: &'a <PI as BatchDataType>::Type) ->
        Result<<CI as BatchDataType>::Type,TrainingError>;
    /// Error back propagation in batch
    /// # Arguments
    /// * `loss` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_bridge_backward<'a>(&self, input:&'a <CI as BatchDataType>::Type) ->
        Result<<PI as BatchDataType>::Type, TrainingError>;
}
impl<U,PI,CI> DeviceBridge<U,PI,CI> for DeviceCpu<U>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          for<'a> PI: From<&'a CI> + BatchDataType + Debug,
          for<'a> CI: From<&'a PI> + BatchDataType + Debug,
          for<'a> <PI as BatchDataType>::Type: From<&'a <CI as BatchDataType>::Type>,
          for<'a> <CI as BatchDataType>::Type: From<&'a <PI as BatchDataType>::Type> {
    fn bridge_forward<'a>(&self, input: &'a PI) -> Result<CI,EvaluateError> {
        Ok(input.into())
    }

    fn bridge_backward<'a>(&self, input: &'a CI) -> Result<PI,TrainingError> {
        Ok(input.into())
    }

    fn batch_bridge_forward<'a>(&self, input: &'a <PI as BatchDataType>::Type)
        -> Result<<CI as BatchDataType>::Type, TrainingError> {
        Ok(input.into())
    }

    fn batch_bridge_backward<'a>(&self, input: &'a <CI as BatchDataType>::Type)
        -> Result<<PI as BatchDataType>::Type, TrainingError> {
        Ok(input.into())
    }
}
impl<U,A,PI,CI> DeviceBridge<U,PI,CI> for DeviceGpu<U,A>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          A: CudaAllocator,
          for<'a> PI: AsCudaView<'a>,
          for<'a> CI: AsCudaView<'a>,
          for<'a> PI: TryFrom<<CI as CudaView<'a>>::Type,Error=CudaError> + BatchDataType + Debug,
          for<'a> CI: TryFrom<<PI as CudaView<'a>>::Type,Error=CudaError> + BatchDataType + Debug,
          for<'a> &'a <PI as BatchDataType>::Type: AsCudaView<'a>,
          for<'a> &'a <CI as BatchDataType>::Type: AsCudaView<'a>,
          for<'a> <PI as BatchDataType>::Type: TryFrom<<&'a <CI as BatchDataType>::Type as CudaView<'a>>::Type,Error=TypeConvertError>,
          for<'a> <CI as BatchDataType>::Type: TryFrom<<&'a <PI as BatchDataType>::Type as CudaView<'a>>::Type,Error=TypeConvertError> {
    fn bridge_forward<'a>(&self, input: &'a PI) -> Result<CI, EvaluateError> {
        Ok(input.as_cuda_view().try_into()?)
    }

    fn bridge_backward<'a>(&self, input: &'a CI) -> Result<PI, TrainingError> {
        Ok(input.as_cuda_view().try_into()?)
    }

    fn batch_bridge_forward<'a>(&self, input: &'a <PI as BatchDataType>::Type)
        -> Result<<CI as BatchDataType>::Type, TrainingError> {
        Ok(input.as_cuda_view().try_into()?)
    }

    fn batch_bridge_backward<'a>(&self, input: &'a <CI as BatchDataType>::Type)
        -> Result<<PI as BatchDataType>::Type, TrainingError> {
        Ok(input.as_cuda_view().try_into()?)
    }
}