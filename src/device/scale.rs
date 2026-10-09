//! Computational processes used in the implementation of scale layers

use std::fmt::Debug;
use std::ops::{Mul, Div};
use rayon::prelude::{IntoParallelRefIterator};
use rayon::prelude::ParallelIterator;
use simdoperator::{AccVector, Vector};
use simdoperator::backend::common::Backend;
use crate::arr::{Arr, SerializedVecView};
use crate::device::{DeviceCpu, DeviceSimd};
use crate::error::{EvaluateError, TypeConvertError};
use crate::layer::{BatchDataType, BatchSize};
use crate::mem::AsRawSlice;
use crate::ope::One;

/// Trait that defines the implementation of inverse scaling processes in the scale layer.
pub trait DeviceScale<U,IO,const N: usize>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          IO: BatchDataType + Debug,
          <IO as BatchDataType>::Type: BatchSize + Debug {
    /// inverse scaling calculation.
    ///
    /// # Arguments
    /// * `scale` - input scale
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn inverse_scaling<'a>(&self, scale: U, input: &'a IO) -> Result<IO, EvaluateError>;
    /// scaling calculation.
    ///
    /// # Arguments
    /// * `scale` - input scale
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn scaling<'a>(&self, scale: U, input: &'a IO) -> Result<IO, EvaluateError>;
    /// Duplicate the input exactly as it is and return it
    ///
    /// # Arguments
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn identity<'a>(&self, input: &'a IO) -> Result<IO, EvaluateError>;
    /// batch inverse scaling calculation in batch.
    ///
    /// # Arguments
    /// * `scale` - input scale
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_inverse_scaling<'a>(&self, scale: U, input: &'a <IO as BatchDataType>::Type)
                                 -> Result<<IO as BatchDataType>::Type, EvaluateError>;
    /// batch scaling calculation in batch.
    ///
    /// # Arguments
    /// * `scale` - input scale
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_scaling<'a>(&self, scale: U, input: &'a <IO as BatchDataType>::Type)
                         -> Result<<IO as BatchDataType>::Type, EvaluateError>;
    /// Duplicate the input exactly as it is and return it
    ///
    /// # Arguments
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_identity<'a>(&self, input: &'a <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, EvaluateError>;
}
impl<U,IO,const N:usize> DeviceScale<U,IO,N> for DeviceCpu
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          IO: BatchDataType + Debug + Clone,
          for<'a> &'a IO: Mul<U,Output=IO> + Div<U,Output=IO>,
          <IO as BatchDataType>::Type: BatchSize + Debug + Clone,
          for<'a> &'a <IO as BatchDataType>::Type: Mul<U,Output=<IO as BatchDataType>::Type> +
                                                   Div<U,Output=<IO as BatchDataType>::Type> {
    fn inverse_scaling<'a>(&self, scale: U, input: &'a IO) -> Result<IO, EvaluateError> {
        Ok(input / scale)
    }

    fn scaling<'a>(&self, scale: U, input: &'a IO) -> Result<IO, EvaluateError> {
        Ok(input * scale)
    }

    fn identity<'a>(&self, input: &'a IO) -> Result<IO, EvaluateError> {
        Ok(input.clone())
    }

    fn batch_inverse_scaling<'a>(&self, scale: U, input: &'a <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, EvaluateError> {
        Ok(input / scale)
    }

    fn batch_scaling<'a>(&self, scale: U, input: &'a <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, EvaluateError> {
        Ok(input * scale)
    }

    fn batch_identity<'a>(&self, input: &'a <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, EvaluateError> {
        Ok(input.clone())
    }
}
impl<U,IO,BE:Backend,const N:usize> DeviceScale<U,IO,N> for DeviceSimd<BE>
    where U: Default + Clone + Copy + One + Div<U,Output=U> + Debug + Send + Sync + 'static,
          IO: BatchDataType + Debug + Clone + AsRawSlice<U> + From<Box<[U]>>,
          <IO as BatchDataType>::Type: BatchSize + Debug + From<Vec<Arr<U,N>>> + Clone,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a <IO as BatchDataType>::Type,Error=TypeConvertError>,
          for<'b> Vector<'b,U,N>: Mul<U,Output=AccVector<U,N>> {
    fn inverse_scaling<'a>(&self, scale: U, input: &'a IO) -> Result<IO, EvaluateError> {
        let i = Vector::try_from(input.as_raw_slice())?;
        let s = i * (U::one() / scale);

        Ok(IO::try_from(Box::from(s))?)
    }

    fn scaling<'a>(&self, scale: U, input: &'a IO) -> Result<IO, EvaluateError> {
        let i = Vector::try_from(input.as_raw_slice())?;
        let s = i * scale;

        Ok(IO::try_from(Box::from(s))?)
    }

    fn identity<'a>(&self, input: &'a IO) -> Result<IO, EvaluateError> {
        Ok(input.clone())
    }

    fn batch_inverse_scaling<'a>(&self, scale: U, input: &'a <IO as BatchDataType>::Type)
                                 -> Result<<IO as BatchDataType>::Type, EvaluateError> {
        Ok(SerializedVecView::<'a,U,Arr<U,N>>::try_from(input)?.par_iter().map(|input| {
            let i = Vector::try_from(input.as_raw_slice()).map_err(|e| {
                EvaluateError::from(e)
            })?;

            let s = i * (U::one() / scale);

            Box::<[U]>::from(s).try_into().map_err(|e| {
                EvaluateError::from(e)
            })
        }).collect::<Result<Vec<Arr<U,N>>, _>>()?.into())
    }

    fn batch_scaling<'a>(&self, scale: U, input: &'a <IO as BatchDataType>::Type)
                         -> Result<<IO as BatchDataType>::Type, EvaluateError> {
        Ok(SerializedVecView::<'a,U,Arr<U,N>>::try_from(input)?.par_iter().map(|input| {
            let i = Vector::try_from(input.as_raw_slice()).map_err(|e| {
                EvaluateError::from(e)
            })?;

            let s = i * scale;

            Box::<[U]>::from(s).try_into().map_err(|e| {
                EvaluateError::from(e)
            })
        }).collect::<Result<Vec<Arr<U,N>>, _>>()?.into())
    }

    fn batch_identity<'a>(&self, input: &'a <IO as BatchDataType>::Type)
                          -> Result<<IO as BatchDataType>::Type, EvaluateError> {
        Ok(input.clone())
    }
}
