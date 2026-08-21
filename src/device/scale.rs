//! Computational processes used in the implementation of scale layers

use std::fmt::Debug;
use crate::error::{EvaluateError, TrainingError};
use crate::layer::{BatchDataType, BatchSize};

/// Trait that defines the implementation of inverse scaling processes in the scale layer.
pub trait DeviceScale<U,IO,const N: usize>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          IO: BatchDataType + Debug,
          <IO as BatchDataType>::Type: BatchSize + Debug {
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
    fn forward_inverse_scale<'a>(&self, scale: &'a IO, input: &'a IO) -> Result<IO, EvaluateError>;

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
    fn backward_inverse_scale<'a>(&self, scale: &'a IO, input: IO) -> Result<IO, TrainingError>;

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
    fn batch_forward_inverse_scale<'a>(&self, scale: &'a IO, input: &'a <IO as BatchDataType>::Type)
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
    fn batch_backward_inverse_scale<'a>(&self, scale: &'a IO, input: <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, TrainingError>;
}
