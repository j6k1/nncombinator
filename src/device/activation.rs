use crate::activation::{Activation, BatchActivation};
use crate::arr::{Arr, ArrView, IntoConverter, SerializedVec, SerializedVecView};
use crate::cuda::{CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView};
use crate::device::{Device, DeviceCpu, DeviceGpu};
use crate::error::{EvaluateError, TrainingError, TypeConvertError};
use crate::layer::BatchDataType;
use crate::lossfunction::LossFunction;
use crate::ope::UnitValue;

/// Trait that defines the implementation of various calculation processes in the activation layer
pub trait DeviceActivation<U,I,A,const N:usize>: Device<U>
    where U: UnitValue<U>,
          I: BatchDataType {
    /// Apply the activation function
    /// # Arguments
    /// * `f` - Activation function object
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn apply(&self, f:&A, input:&I) -> Result<I, EvaluateError>;
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
    fn derive(&self, f:&A, o:&I, loss:&I, u:&I) -> Result<I, TrainingError>;
    /// Apply the activation function
    /// # Arguments
    /// * `f` - Activation function object
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_apply(&self, f:&A, input:&<I as BatchDataType>::Type) -> Result<<I as BatchDataType>::Type, TrainingError>;
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
    fn batch_derive(&self, f:&A, o:&<I as BatchDataType>::Type, loss:&<I as BatchDataType>::Type, u:&<I as BatchDataType>::Type)
        -> Result<<I as BatchDataType>::Type, TrainingError>;
    /// Returns whether or not the canonical linkage function can be used.
    /// # Arguments
    /// * `f` - Activation function object
    /// * `l` - loss function
    fn is_canonical_link<L: LossFunction<U>>(&self,f:&A,l:&L) -> bool;
}
impl<'a,U,I,A,const N:usize> DeviceActivation<U,I,A,N> for DeviceCpu<U>
    where U: UnitValue<U>,
          I: BatchDataType,
          I: From<Arr<U,N>>,
          SerializedVec<U,Arr<U,N>>: IntoConverter,
          <I as BatchDataType>::Type: TryFrom<<SerializedVec<U,Arr<U,N>> as IntoConverter>::Converter,Error=TypeConvertError>,
          for<'b> A: Activation<U,ArrView<'b,U,N>,Arr<U,N>,Self>,
          for<'b> A: BatchActivation<U,SerializedVecView<'b,U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,Self>,
          for<'b> ArrView<'b,U,N>: From<&'b I>,
          for<'b> SerializedVecView<'b,U,Arr<U,N>>: TryFrom<&'b <I as BatchDataType>::Type,Error=TypeConvertError> {
    #[inline]
    fn apply(&self, f: &A, input: &I) -> Result<I, EvaluateError> {
        Ok(f.apply(self, &input.into())?.into())
    }

    #[inline]
    fn derive(&self, f: &A, o: &I, loss: &I, u: &I) -> Result<I, TrainingError> {
        Ok(f.derive(self, &o.into(), &loss.into(), &u.into())?.into())
    }

    #[inline]
    fn batch_apply(&self, f: &A, input: &<I as BatchDataType>::Type) -> Result<<I as BatchDataType>::Type, TrainingError> {
        Ok(f.batch_apply(self, &input.try_into()?)?.into_converter().try_into()?)
    }

    #[inline]
    fn batch_derive(&self, f: &A, o: &<I as BatchDataType>::Type, loss: &<I as BatchDataType>::Type, u: &<I as BatchDataType>::Type)
        -> Result<<I as BatchDataType>::Type, TrainingError> {
        Ok(f.batch_derive(self, &o.try_into()?, &loss.try_into()?, &u.try_into()?).unwrap().into_converter().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, f: &A, l: &L) -> bool {
        f.is_canonical_link(l)
    }
}
impl<'a,U,I,A,const N:usize> DeviceActivation<U,I,A,N> for DeviceGpu<U>
    where U: UnitValue<U>,
          I: BatchDataType,
          I: From<CudaTensor1dPtr<U,N>>,
          DeviceGpu<U>: Device<U>,
          CudaTensor1dPtr<U,N>: From<I>,
          CudaVec<U,CudaTensor1dPtr<U,N>>: IntoConverter,
          <I as BatchDataType>::Type: TryFrom<<CudaVec<U,CudaTensor1dPtr<U,N>> as IntoConverter>::Converter,Error=TypeConvertError>,
          for<'b> A: Activation<U,CudaTensor1dPtrView<'b,U,N>,CudaTensor1dPtr<U,N>,Self>,
          for<'b> A: BatchActivation<U,CudaVecView<'b,U,CudaTensor1dPtr<U,N>>,CudaVec<U,CudaTensor1dPtr<U,N>>,Self>,
          for<'b> CudaTensor1dPtrView<'b,U,N>: From<&'b I>,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtr<U,N>>: TryFrom<&'b <I as BatchDataType>::Type,Error=TypeConvertError> {
    #[inline]
    fn apply(&self, f: &A, input: &I) -> Result<I, EvaluateError> {
        Ok(f.apply(self, &input.into())?.into())
    }

    #[inline]
    fn derive(&self, f: &A, o: &I, loss: &I, u: &I) -> Result<I, TrainingError> {
        Ok(f.derive(self, &o.into(), &loss.into(), &u.into())?.into())
    }

    #[inline]
    fn batch_apply(&self, f: &A, input: &<I as BatchDataType>::Type) -> Result<<I as BatchDataType>::Type, TrainingError> {
        Ok(f.batch_apply(self, &input.try_into()?)?.into_converter().try_into()?)
    }

    #[inline]
    fn batch_derive(&self, f: &A, o: &<I as BatchDataType>::Type, loss: &<I as BatchDataType>::Type, u: &<I as BatchDataType>::Type)
                    -> Result<<I as BatchDataType>::Type, TrainingError> {
        Ok(f.batch_derive(self, &o.try_into()?, &loss.try_into()?, &u.try_into()?).unwrap().into_converter().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, f: &A, l: &L) -> bool {
        f.is_canonical_link(l)
    }
}
