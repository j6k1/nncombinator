//! Computational processes used in the implementation of scale layers

use std::fmt::Debug;
use std::ops::{Mul,Div};
use crate::arr::{Arr, ArrView, AsView, IntoConverter, MakeView, SerializedVec, SerializedVecView, SliceSize};
use crate::device::{DeviceCpu};
use crate::error::{EvaluateError, TrainingError, TypeConvertError};
use crate::layer::{BatchDataType, BatchSize};

/// Trait that defines the implementation of inverse scaling processes in the scale layer.
pub trait DeviceScale<U,IO,const N: usize>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          IO: BatchDataType + Debug,
          <IO as BatchDataType>::Type: BatchSize + Debug {
    type Scale;
    /// Forward propagation calculation.
    ///
    /// # Arguments
    /// * `scale` - input scale
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn inverse_scaling<'a>(&self, scale: &'a Self::Scale, input: &'a IO) -> Result<IO, EvaluateError>;

    /// Error back propagation calculation.
    ///
    /// # Arguments
    /// * `scale` - input scale
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn scaling<'a>(&self, scale: &'a Self::Scale, input: &'a IO) -> Result<IO, EvaluateError>;

    /// Forward propagation calculation in batch.
    ///
    /// # Arguments
    /// * `scale` - input scale
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_inverse_scaling<'a>(&self, scale: &'a Self::Scale, input: &'a <IO as BatchDataType>::Type)
                                 -> Result<<IO as BatchDataType>::Type, TrainingError>;

    /// Error back propagation calculation in batch.
    ///
    /// # Arguments
    /// * `scale` - input scale
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_scaling<'a>(&self, scale: &'a Self::Scale, input: &'a <IO as BatchDataType>::Type)
                         -> Result<<IO as BatchDataType>::Type, TrainingError>;
}
impl<U,IO,const N:usize> DeviceScale<U,IO,N> for DeviceCpu
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          IO: BatchDataType + Debug + Clone,
          <IO as BatchDataType>::Type: BatchSize + Debug,
          IO: From<Arr<U,N>>,
          Arr<U,N>: From<IO>,
          SerializedVec<U,Arr<U,N>>: IntoConverter,
          <IO as BatchDataType>::Type: TryFrom<<SerializedVec<U,Arr<U,N>> as IntoConverter>::Converter,Error=TypeConvertError>,
          for<'a> Arr<U,N>: SliceSize + AsView<'a,ViewType=ArrView<'a,U,N>> + MakeView<'a,U> + Clone + Send + Sync,
          for<'a> Arr<U,N>: Mul<<Arr<U,N> as AsView<'a>>::ViewType,Output=Arr<U,N>> + Send + Sync,
          for<'a> Arr<U,N>: Div<<Arr<U,N> as AsView<'a>>::ViewType,Output=Arr<U,N>> + Send + Sync,
          for<'a> ArrView<'a,U,N>: From<&'a IO> + Mul<Output=Arr<U,N>> + Div<Output=Arr<U,N>>,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a <IO as BatchDataType>::Type,Error=TypeConvertError> {
    type Scale = Arr<U,N>;
    fn inverse_scaling<'a>(&self, scale: &'a Arr<U,N>, input: &'a IO) -> Result<IO, EvaluateError> {
        let view  = ArrView::<'a,U,N>::from(input);

        Ok((view / scale.as_view()).into())
    }

    fn scaling<'a>(&self, scale: &'a Arr<U,N>, input: &'a IO) -> Result<IO, EvaluateError> {
        let view = ArrView::<'a,U,N>::from(input);

        Ok((view * scale.as_view()).into())
    }

    fn batch_inverse_scaling<'a>(&self, scale: &'a Arr<U,N>, input: &'a <IO as BatchDataType>::Type) -> Result<<IO as BatchDataType>::Type, TrainingError> {
        let view  = SerializedVecView::<'a,U,Arr<U,N>>::try_from(input)?;

        Ok(SerializedVec::from(view.iter().map(|i| {
            i / scale.as_view()
        }).collect::<Vec<Arr<U,N>>>()).into_converter().try_into()?)
    }

    fn batch_scaling<'a>(&self, scale: &'a Arr<U,N>, input: &'a <IO as BatchDataType>::Type) -> Result<<IO as BatchDataType>::Type, TrainingError> {
        let view  = SerializedVecView::<'a,U,Arr<U,N>>::try_from(input)?;


        Ok(SerializedVec::from(view.iter().map(|i| {
            i * scale.as_view()
        }).collect::<Vec<Arr<U,N>>>()).into_converter().try_into()?)
    }
}
