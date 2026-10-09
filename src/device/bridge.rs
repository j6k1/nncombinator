//! Implementation of the calculation process for bridge layers

use std::fmt::Debug;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelIterator;
use simdoperator::{AccVector, Vector};
use simdoperator::backend::common::Backend;
use simdoperator::error::InstantiationError;
use crate::arr::{Arr, ArrView, AsView, SerializedVecView, TryAsSimd};
use crate::cast::Assume;
use crate::cuda::allocator::CudaAllocator;
use crate::cuda::{AsCudaView, CudaView};
use crate::device::{DeviceCpu, DeviceGpu, DeviceSimd};
use crate::error::{EvaluateError, TrainingError, TypeConvertError};
use crate::layer::{BatchDataType, BatchSize};
use crate::mem::AsRawSlice;

/// Trait that defines the implementation of various calculation processes in the bridge layer
pub trait DeviceBridge<U,SO,PI,CI>
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
        Result<<CI as BatchDataType>::Type,EvaluateError>;
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
/// Trait that defines the implementation of various calculation processes in the bridge layer
pub trait DeviceBridgeExactSize<U,SO,PI,CI,const N:usize>
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
    fn batch_bridge_forward<'a>(&self,input: &'a <PI as BatchDataType>::Type) -> Result<<CI as BatchDataType>::Type,EvaluateError>;
    /// Error back propagation in batch
    /// # Arguments
    /// * `loss` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_bridge_backward<'a>(&self, input:&'a <CI as BatchDataType>::Type) -> Result<<PI as BatchDataType>::Type, TrainingError>;
}
impl<U,SO,PI,CI> DeviceBridge<U,SO,PI,CI> for DeviceCpu
    where U: Default + Clone + Copy + Debug + Send + Sync + Assume<SO> + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + Assume<U> + 'static,
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
        -> Result<<CI as BatchDataType>::Type, EvaluateError> {
        Ok(input.into())
    }

    fn batch_bridge_backward<'a>(&self, input: &'a <CI as BatchDataType>::Type)
        -> Result<<PI as BatchDataType>::Type, TrainingError> {
        Ok(input.into())
    }
}
impl<U,SO,A,PI,CI> DeviceBridge<U,SO,PI,CI> for DeviceGpu<A>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          A: CudaAllocator,
          for<'a> &'a PI: AsCudaView<'a>,
          for<'a> &'a CI: AsCudaView<'a>,
          for<'a> PI: TryFrom<<&'a CI as CudaView<'a>>::Type> + BatchDataType + Debug,
          for<'a> CI: TryFrom<<&'a PI as CudaView<'a>>::Type> + BatchDataType + Debug,
          for<'a> &'a <PI as BatchDataType>::Type: AsCudaView<'a>,
          for<'a> &'a <CI as BatchDataType>::Type: AsCudaView<'a>,
          for<'a> <PI as BatchDataType>::Type: TryFrom<<&'a <CI as BatchDataType>::Type as CudaView<'a>>::Type>,
          for<'a> <CI as BatchDataType>::Type: TryFrom<<&'a <PI as BatchDataType>::Type as CudaView<'a>>::Type>,
          for<'a> EvaluateError: From<<<PI as BatchDataType>::Type as TryFrom<<&'a <CI as BatchDataType>::Type as CudaView<'a>>::Type>>::Error>,
          for<'a> EvaluateError: From<<<CI as BatchDataType>::Type as TryFrom<<&'a <PI as BatchDataType>::Type as CudaView<'a>>::Type>>::Error>,
          for<'a> TrainingError: From<<<PI as BatchDataType>::Type as TryFrom<<&'a <CI as BatchDataType>::Type as CudaView<'a>>::Type>>::Error>,
          for<'a> TrainingError: From<<<CI as BatchDataType>::Type as TryFrom<<&'a <PI as BatchDataType>::Type as CudaView<'a>>::Type>>::Error>,
          for<'a> EvaluateError: From<<PI as TryFrom<<&'a CI as CudaView<'a>>::Type>>::Error>,
          for<'a> EvaluateError: From<<CI as TryFrom<<&'a PI as CudaView<'a>>::Type>>::Error>,
          for<'a> TrainingError: From<<PI as TryFrom<<&'a CI as CudaView<'a>>::Type>>::Error>,
          for<'a> TrainingError: From<<CI as TryFrom<<&'a PI as CudaView<'a>>::Type>>::Error> {
    fn bridge_forward<'a>(&self, input: &'a PI) -> Result<CI, EvaluateError> {
        Ok(input.as_cuda_view().try_into()?)
    }

    fn bridge_backward<'a>(&self, input: &'a CI) -> Result<PI, TrainingError> {
        Ok(input.as_cuda_view().try_into()?)
    }

    fn batch_bridge_forward<'a>(&self, input: &'a <PI as BatchDataType>::Type)
        -> Result<<CI as BatchDataType>::Type, EvaluateError> {
        Ok(input.as_cuda_view().try_into()?)
    }

    fn batch_bridge_backward<'a>(&self, input: &'a <CI as BatchDataType>::Type)
        -> Result<<PI as BatchDataType>::Type, TrainingError> {
        Ok(input.as_cuda_view().try_into()?)
    }
}
impl<U,SO,PI,CI,BE: Backend + Clone> DeviceBridge<U,SO,PI,CI> for DeviceSimd<BE>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          PI: 'static,
          CI: 'static,
          <PI as BatchDataType>::Type: BatchSize + Debug + From<Vec<PI>> + Clone,
          <CI as BatchDataType>::Type: BatchSize + Debug + From<Vec<CI>> + Clone,
          EvaluateError: From<InstantiationError>,
          TrainingError: From<InstantiationError>,
          for<'a> PI: AsView<'a> + BatchDataType + Debug + Send + Sync + 'static,
          for<'a> CI: AsView<'a> + BatchDataType + Debug + Send + Sync + 'static,
          for<'a> &'a PI: TryAsSimd<'a,BE,Error=InstantiationError>,
          for<'a> &'a CI: TryAsSimd<'a,BE,Error=InstantiationError>,
          for<'a> <PI as AsView<'a>>::ViewType: TryAsSimd<'a,BE,Error=InstantiationError> + 'a,
          for<'a> <CI as AsView<'a>>::ViewType: TryAsSimd<'a,BE,Error=InstantiationError> + 'a,
          for<'a,'b> <<PI as AsView<'a>>::ViewType as TryAsSimd<'a,BE>>::Output<'b>: Send + Sync + 'static,
          for<'a,'b> <<CI as AsView<'a>>::ViewType as TryAsSimd<'a,BE>>::Output<'b>: Send + Sync + 'static,
          for<'a,'b> PI: From<<&'a CI as TryAsSimd<'a,BE>>::Output<'a>> + From<<<CI as AsView<'a>>::ViewType as TryAsSimd<'a,BE>>::Output<'b>>,
          for<'a,'b> CI: From<<&'a PI as TryAsSimd<'a,BE>>::Output<'a>> + From<<<PI as AsView<'a>>::ViewType as TryAsSimd<'a,BE>>::Output<'b>>,
          for<'a> <PI as AsView<'a>>::ViewType: TryAsSimd<'a,BE,Error=InstantiationError> + 'a,
          for<'a> <CI as AsView<'a>>::ViewType: TryAsSimd<'a,BE,Error=InstantiationError> + 'a,
          for<'a> SerializedVecView<'a,U,PI>: TryFrom<&'a <PI as BatchDataType>::Type,Error=TypeConvertError>,
          for<'a> SerializedVecView<'a,U,CI>: TryFrom<&'a <CI as BatchDataType>::Type,Error=TypeConvertError>,
          for<'a> SerializedVecView<'a,U,PI>: IntoParallelIterator<Item=<PI as AsView<'a>>::ViewType>,
          for<'a> SerializedVecView<'a,U,CI>: IntoParallelIterator<Item=<CI as AsView<'a>>::ViewType>,
          for<'a> SerializedVecView<'a,U,PI>: IntoParallelRefIterator<'a,Item=<PI as AsView<'a>>::ViewType>,
          for<'a> SerializedVecView<'a,U,CI>: IntoParallelRefIterator<'a,Item=<CI as AsView<'a>>::ViewType>,
          for<'a> <SerializedVecView<'a,U,PI> as IntoParallelIterator>::Item: 'a,
          for<'a> <SerializedVecView<'a,U,CI> as IntoParallelIterator>::Item: 'a {
    fn bridge_forward<'a>(&self, input: &'a PI) -> Result<CI,EvaluateError> {
        Ok(input.try_as_simd()?.into())
    }

    fn bridge_backward<'a>(&self, input: &'a CI) -> Result<PI,TrainingError> {
        Ok(input.try_as_simd()?.into())
    }

    fn batch_bridge_forward<'a>(&self, input: &'a <PI as BatchDataType>::Type)
                                -> Result<<CI as BatchDataType>::Type, EvaluateError> {
        Ok(SerializedVecView::<'a,U,PI>::try_from(input)?.par_iter().map(|i| {
            i.try_as_simd().map(|i| CI::from(i)).map_err(|e| EvaluateError::from(e))
        }).collect::<Result<Vec<CI>,EvaluateError>>()?.into())
    }

    fn batch_bridge_backward<'a>(&self, input: &'a <CI as BatchDataType>::Type)
                                 -> Result<<PI as BatchDataType>::Type, TrainingError> {
        Ok(SerializedVecView::<'a,U,CI>::try_from(input)?.par_iter().map(|i| {
            i.try_as_simd().map(|i| PI::from(i)).map_err(|e| EvaluateError::from(e))
        }).collect::<Result<Vec<PI>,EvaluateError>>()?.into())
    }
}
