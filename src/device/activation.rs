use crate::activation::{Activation, BatchActivation};
use crate::arr::{Arr, ArrView, SerializedVec, SerializedVecView};
use crate::device::{Device, DeviceCpu, DeviceGpu};
use crate::error::{EvaluateError, TrainingError};
use crate::ope::UnitValue;

/// Trait that defines the implementation of various calculation processes in the activation layer
pub trait DeviceActivation<U,C,T,R,A>: Device<U>
    where U: UnitValue<U> {
    /// Apply the activation function
    /// # Arguments
    /// * `f` - Activation function object
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn apply(&self, f:&A, input:&T) -> Result<R, EvaluateError>;
    /// Apply derivatives of the activation function
    /// # Arguments
    /// * `f` - Activation function object
    /// * `o` - Input from upper layers
    /// * `loss` - Losses calculated at lower tiers
    /// * `u` - Value before passing through the activation function of the input from the upper layer
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn derive(&self, f:&A, o:&T, loss:&T, u:&T) -> Result<R, TrainingError>;
    /// Apply the activation function
    /// # Arguments
    /// * `f` - Activation function object
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_apply(&self, f:&A, input:&C) -> Result<SerializedVec<U,R>, TrainingError>;
    /// Apply derivatives of the activation function
    /// # Arguments
    /// * `f` - Activation function object
    /// * `o` - Input from upper layers
    /// * `loss` - Losses calculated at lower tiers
    /// * `u` - Value before passing through the activation function of the input from the upper layer
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_derive(&self, f:&A, o:&C, loss:&C, u:&C) -> Result<SerializedVec<U,R>, TrainingError>;
}
impl<'a,U,A,const N:usize> DeviceActivation<U,SerializedVecView<'a,U,Arr<U,N>>,ArrView<'a,U,N>,Arr<U,N>,A> for DeviceCpu<U>
    where U: UnitValue<U>,
          A: Activation<U,ArrView<'a,U,N>,Arr<U,N>,Self>,
          A: BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,Self> {
    #[inline]
    fn apply(&self, f: &A, input: &ArrView<'a, U, N>) -> Result<Arr<U, N>, EvaluateError> {
        f.apply(self, input)
    }

    #[inline]
    fn derive(&self, f: &A, o: &ArrView<'a, U, N>, loss: &ArrView<'a, U, N>, u: &ArrView<'a, U, N>) -> Result<Arr<U, N>, TrainingError> {
        f.derive(self, o, loss, u)
    }

    #[inline]
    fn batch_apply(&self, f: &A, input: &SerializedVecView<'a, U, Arr<U, N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        f.batch_apply(self, input)
    }

    #[inline]
    fn batch_derive(&self, f: &A, o: &SerializedVecView<'a, U, Arr<U, N>>, loss: &SerializedVecView<'a, U, Arr<U, N>>, u: &SerializedVecView<'a, U, Arr<U, N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        f.batch_derive(self, o, loss, u)
    }
}
impl<'a,U,A,const N:usize> DeviceActivation<U,SerializedVecView<'a,U,Arr<U,N>>,ArrView<'a,U,N>,Arr<U,N>,A> for DeviceGpu<U>
    where U: UnitValue<U>,
          A: Activation<U,ArrView<'a,U,N>,Arr<U,N>,Self>,
          A: BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,Self>,
          DeviceGpu<U>: Device<U> {
    #[inline]
    fn apply(&self, f: &A, input: &ArrView<'a, U, N>) -> Result<Arr<U, N>, EvaluateError> {
        f.apply(self,input)
    }

    #[inline]
    fn derive(&self, f: &A, o: &ArrView<'a, U, N>, loss: &ArrView<'a, U, N>, u: &ArrView<'a, U, N>) -> Result<Arr<U, N>, TrainingError> {
        f.derive(self,o,loss,u)
    }

    #[inline]
    fn batch_apply(&self, f: &A, input: &SerializedVecView<'a, U, Arr<U, N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        f.batch_apply(self,input)
    }

    #[inline]
    fn batch_derive(&self, f: &A, o: &SerializedVecView<'a, U, Arr<U, N>>, loss: &SerializedVecView<'a, U, Arr<U, N>>, u: &SerializedVecView<'a, U, Arr<U, N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        f.batch_derive(self,o,loss,u)
    }
}