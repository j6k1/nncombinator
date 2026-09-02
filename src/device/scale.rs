//! Computational processes used in the implementation of scale layers

use std::fmt::Debug;
use std::ops::{Mul, Div};
use crate::device::{DeviceCpu};
use crate::error::{EvaluateError};
use crate::layer::{BatchDataType, BatchSize};

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
