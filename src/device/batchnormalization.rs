//! Implementation of the calculation process for batch normalization
use std::fmt::Debug;
use std::ops::DerefMut;
use rcudnn::{API};
use rcudnn_sys::cudnnBatchNormMode_t::{CUDNN_BATCHNORM_PER_ACTIVATION, CUDNN_BATCHNORM_SPATIAL};
use rcudnn_sys::{cudnnBatchNormalizationBackward, cudnnBatchNormalizationForwardInference, cudnnBatchNormalizationForwardTraining, cudnnDeriveBNTensorDescriptor, cudnnStatus_t};

use crate::arr::{Arr, ArrView, IntoConverter, SerializedVec, SerializedVecView};
use crate::ope::Sum;
use crate::collection::Broadcast;
use crate::computational_graph::{BroadcastNode, GraphNode, SqrtNode, SquareNode, SumNode};
use crate::cuda::{AsMutVoidPtr, AsVoidPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, DataTypeInfo, Memory, MemoryMoveTo};
use crate::cuda::cudnn::tensor::CudnnTensor4dDescriptor;
use crate::device::{DeviceCpu, DeviceGpu, DeviceMemoryPool};
use crate::error::{EvaluateError, TrainingError, TypeConvertError};
use crate::layer::{BatchDataType, BatchSize};
use crate::ope::UnitValue;

/// Features defining the implementation of the various computational processes in the batch normalization layer
pub trait DeviceBatchNorm<U,C,I,const N:usize>
    where U: UnitValue<U>,
          I: BatchDataType + Debug + 'static,
          <I as BatchDataType>::Type: Debug + 'static {
    /// Forward propagation calculation
    /// # Arguments
    /// * `input` - input
    /// * `scale` - γ
    /// * `bias` - β
    /// * `estimated_mean` - μΒ
    /// * `estimated_variance` - σΒ
    ///
    /// output = γ * ((input - μΒ) / sqrt(σ^2Β + 1e-6)) + β
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn forward_batch_norm<'a>(&self, input: &'a I, scale: &C, bias: &C,
                          estimated_mean: &C, estimated_variance: &C) -> Result<I,EvaluateError>;
    /// Forward propagation calculation (implemented in training mode)
    /// # Arguments
    /// * `input` - input
    /// * `scale` - γ
    /// * `bias` - β
    /// * `estimated_mean` - μΒ
    /// * `estimated_variance` - σΒ
    ///
    /// output = (γ * ((input - μΒ) / sqrt(σ^2Β + 1e-6)) + β,μΒ,1 / (σΒ + 1e-6))
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn forward_batch_norm_train<'a>(&self, input: &'a I, scale: &C, bias: &C,
                                estimated_mean: &C, estimated_variance: &C) -> Result<(I,C,C),EvaluateError>;
    /// Forward propagation calculation in batch
    /// # Arguments
    /// * `input` - input
    /// * `scale` - γ
    /// * `bias` - β
    /// * `estimated_mean` - μΒ
    /// * `estimated_variance` - σΒ
    ///
    /// output = γ * ((input - μΒ) / sqrt(σ^2Β + 1e-6)) + β
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn batch_forward_batch_norm<'a>(&self, input: &'a <I as BatchDataType>::Type, scale: &C , bias: &C,
                                estimated_mean: &C, estimated_variance: &C) -> Result<<I as BatchDataType>::Type,EvaluateError>;
    /// Forward propagation calculation in batch (implemented in training mode)
    /// # Arguments
    /// * `input` - input
    /// * `scale` - γ
    /// * `bias` - β
    /// * `running_mean` - μΒ
    /// * `running_variance` - σΒ
    ///
    /// running_mean = running_mean * momentum + (1 - momentum) * μΒ
    /// running_variance = running_variance * momentum + (1 - momentum) * μΒ
    ///
    /// output = (γ * ((input - μΒ) / sqrt(σ^2Β + 1e-6)) + β,,μΒ,1 / (σΒ + 1e-6),running_mean,running_variance)
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn batch_forward_batch_norm_train<'a>(&self, input: &'a <I as BatchDataType>::Type, scale: &C, bias: &C,
                                      running_mean: &C, running_variance: &C, momentum: U)
                                      -> Result<(<I as BatchDataType>::Type,C,C,C,C),TrainingError>;
    /// Error back propagation calculation
    /// # Arguments
    /// * `loss` - loss input
    /// * `input` - input
    /// * `scale` - γ
    /// * `saved_mean` - μΒ calculated during forward propagation
    /// * `saved_inv_variance` - Inverse of σΒ calculated during forward propagati (1 / (σΒ + 1e-6))
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_batch_norm<'a>(&self, loss: &'a I, input: &'a I, scale: &C,
                           saved_mean: &C, saved_inv_variance: &C) -> Result<(I,C,C), TrainingError>;
    /// Error back propagation calculation in batch
    /// # Arguments
    /// * `loss` - loss input
    /// * `input` - input
    /// * `scale` - γ
    /// * `saved_mean` - μΒ calculated during forward propagation
    /// * `saved_inv_variance` - Inverse of σΒ calculated during forward propagati (1 / (σΒ + 1e-6))
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_batch_norm<'a>(&self, loss:&'a <I as BatchDataType>::Type, input: &'a <I as BatchDataType>::Type,
                                     scale: &C, saved_mean: &C, saved_inv_variance: &C)
        -> Result<(<I as BatchDataType>::Type,C,C), TrainingError>;
}
impl<U,I,const N:usize> DeviceBatchNorm<U,Arr<U,N>,I,N> for DeviceCpu<U>
    where U: UnitValue<U>,
          I: BatchDataType + Debug + From<Arr<U,N>> + 'static,
          <I as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: TryFrom<<SerializedVec<U,Arr<U,N>> as IntoConverter>::Converter,Error=TypeConvertError>,
          SerializedVec<U,Arr<U,N>>: IntoConverter,
          for<'a> ArrView<'a,U,N>: From<&'a I>,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a <I as BatchDataType>::Type,Error=TypeConvertError> {
    #[inline]
    fn forward_batch_norm<'a>(&self, input: &'a I, scale: &Arr<U,N>, bias: &Arr<U,N>,
                          estimated_mean: &Arr<U,N>, estimated_variance: &Arr<U,N>) -> Result<I,EvaluateError> {
        let input = ArrView::<'a,U,N>::from(input);

        let eps = U::from_f64(1e-6).ok_or(EvaluateError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        Ok(Arr::try_from(input.iter()
            .zip(scale.iter())
            .zip(bias.iter())
            .zip(estimated_mean.iter())
            .zip(estimated_variance.iter())
            .map(|((((&i,&scale),&bias),&mean),&variance)| {
                scale * ((i - mean) / SqrtNode::new().forward(variance + eps)) + bias
            }).collect::<Vec<U>>())?.into())
    }

    #[inline]
    fn forward_batch_norm_train<'a>(&self, input: &'a I,
                                scale: &Arr<U,N>,
                                bias: &Arr<U,N>,
                                estimated_mean: &Arr<U,N>,
                                estimated_variance: &Arr<U,N>) -> Result<(I,Arr<U,N>,Arr<U,N>),EvaluateError> {
        let input = ArrView::<'a,U,N>::from(input);

        let eps = U::from_f64(1e-6).ok_or(EvaluateError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        Ok((Arr::try_from(input.iter()
                .zip(scale.iter())
                .zip(bias.iter())
                .zip(estimated_mean.iter())
                .zip(estimated_variance.iter())
                .map(|((((&i,&scale),&bias),&mean),&variance)| {
                    scale * ((i - mean) / SqrtNode::new().forward(variance + eps)) + bias
                }).collect::<Vec<U>>())?.into(),
            estimated_mean.clone(),
            estimated_variance.iter().map(|&v| U::one() / SqrtNode::new().forward(v + eps)).collect::<Vec<U>>().try_into()?
        ))
    }

    #[inline]
    fn batch_forward_batch_norm<'a>(&self, input: &'a <I as BatchDataType>::Type, scale: &Arr<U,N>, bias: &Arr<U,N>,
                                    estimated_mean: &Arr<U,N>, estimated_variance: &Arr<U,N>)
        -> Result<<I as BatchDataType>::Type, EvaluateError> {
        let input = SerializedVecView::<'a,U,Arr<U,N>>::try_from(input)?;

        let eps = U::from_f64(1e-6).ok_or(EvaluateError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        Ok(SerializedVec::from(input.iter().map(|input| {
            input.iter()
                .zip(scale.iter())
                .zip(bias.iter())
                .zip(estimated_mean.iter())
                .zip(estimated_variance.iter())
                .map(|((((&i,&scale),&bias),&mean),&variance)| {
                    scale * (i - mean) / SqrtNode::new().forward(variance + eps) + bias
                }).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?).into_converter().try_into()?)
    }

    #[inline]
    fn batch_forward_batch_norm_train<'a>(&self, input: &'a <I as BatchDataType>::Type,
                                      scale: &Arr<U,N>, bias: &Arr<U,N>,
                                      running_mean: &Arr<U,N>, running_variance: &Arr<U,N>,
                                      momentum: U)
                                      -> Result<(<I as BatchDataType>::Type,Arr<U,N>,Arr<U,N>,Arr<U,N>,Arr<U,N>), TrainingError> {
        let input = SerializedVecView::<'a,U,Arr<U,N>>::try_from(input)?;

        let eps = U::from_f64(1e-6).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let n = input.len();
        let un = U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let un_inv = U::from_f64(1.).ok_or(TrainingError::TypeCastError(
            String::from(
                "Error in type conversion from usize."
            )
        ))? / un;

        let mean:Arr<U,N> = SumNode::<U,SerializedVecView<'_,U,Arr<U,N>>>::new().forward(input) * un_inv;

        let variance:SerializedVec<U,Arr<U,N>> = (input - Broadcast::<Arr<U,N>>(mean.clone()))
            .iter()
            .map(|i| {
                i.iter().map(|&i| {
                    SquareNode::new().forward(i)
                }).collect::<Vec<U>>().try_into()
            }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into();
        let variance = variance.sum() * un_inv;

        let inv_variance:Arr<U,N> = variance.iter().map(|&v| U::one() / SqrtNode::new().forward(v + eps)).collect::<Vec<U>>().try_into()?;

        let o:SerializedVec<U,Arr<U,N>> = Broadcast(inv_variance.clone()) * (input - Broadcast(mean.clone()));

        let running_mean = running_mean * momentum + &mean * (U::one() - momentum);
        let running_variance = running_variance * momentum + variance * (U::one() - momentum);

        let o = (BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().forward((scale,n)) * o) + Broadcast(bias.clone());

        Ok((o.into_converter().try_into()?,mean,inv_variance,running_mean,running_variance))
    }

    #[inline]
    fn backward_batch_norm<'a>(&self, loss: &'a I, input: &'a I,
                           scale: &Arr<U,N>, saved_mean: &Arr<U,N>, saved_inv_variance: &Arr<U,N>)
                           -> Result<(I, Arr<U,N>, Arr<U,N>), TrainingError> {
        let loss = ArrView::<'a,U,N>::from(loss);
        let input = ArrView::<'a,U,N>::from(input);

        let b = loss.clone();

        let x = input - saved_mean;

        let s = (&x * saved_inv_variance) * loss;

        let dx1 = scale * loss;
        let dx2 = &dx1 * saved_inv_variance;
        let dx3 = &x * dx1;
        let dx4 =  -(saved_inv_variance * saved_inv_variance) * dx3;
        let dx5 = dx4 * (saved_inv_variance * U::from_f64(0.5).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from f64.")
        ))?);
        let dx6 = &x * dx5 * U::from_usize(2).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;
        let dx7 = dx2 + dx6;
        let dx8 = &dx7;
        let dx9 = -&dx7;
        let dx = dx8 + dx9;

        Ok((dx.into(),s,b.into()))
    }

    #[inline]
    fn batch_backward_batch_norm<'a>(&self, loss: &'a <I as BatchDataType>::Type,
                                 input: &'a <I as BatchDataType>::Type,
                                 scale: &Arr<U,N>,
                                 saved_mean: &Arr<U,N>, saved_inv_variance: &Arr<U,N>)
                                 -> Result<(<I as BatchDataType>::Type, Arr<U,N>, Arr<U,N>), TrainingError> {
        let loss = SerializedVecView::<'a,U,Arr<U,N>>::try_from(loss)?;
        let input = SerializedVecView::<'a,U,Arr<U,N>>::try_from(input)?;

        let n = input.len();

        let un = U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let un_inv = U::from_usize(1).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))? / un;

        let b = BroadcastNode::<U,SerializedVecView<'_,U,Arr<U,N>>>::new().backward(loss);

        let x = BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().forward((saved_mean,n));
        let x2 = input - &x;
        let iv = BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().forward((saved_inv_variance,n));

        let s = BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().backward(&(&x2 * &iv * loss));

        let dx1 = Broadcast(scale.clone()) * loss;
        let dx2 = &dx1 * iv;
        let dx3 = BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().backward(&(&x2 * dx1));
        let dx4 = -(saved_inv_variance * saved_inv_variance) * dx3;
        let dx5 = dx4 * (saved_inv_variance * U::from_f64(0.5).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from f64.")
        ))?);
        let dx6 = SumNode::<U,SerializedVec<U,Arr<U,N>>>::new().backward((&(dx5 * un_inv),n));
        let dx7 = x2 * dx6 * U::from_usize(2).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;
        let dx8 = dx2 + dx7;
        let dx9 = &dx8;
        let dx10 = -&dx8;
        let dx11 = BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().backward(&dx10);
        let dx12 = SumNode::<U,SerializedVec<U,Arr<U,N>>>::new().backward((&dx11,n)) * un_inv;

        let dx = dx9 + dx12;

        Ok((dx.into_converter().try_into()?,s,b))
    }
}
impl<U,I,const N:usize> DeviceBatchNorm<U,CudaTensor1dPtr<U,N>,I,N> for DeviceGpu<U>
    where U: UnitValue<U> + DataTypeInfo + AsVoidPtr,
          I: BatchDataType + Debug + From<CudaTensor1dPtr<U,N>> + 'static,
          <I as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: TryFrom<<CudaVec<U,CudaTensor1dPtr<U,N>> as IntoConverter>::Converter,Error=TypeConvertError>,
          CudaVec<U,CudaTensor1dPtr<U,N>>: IntoConverter,
          for<'a> CudaTensor1dPtrView<'a,U,N>: From<&'a I>,
          for<'a> CudaVecView<'a,U,CudaTensor1dPtr<U,N>>: TryFrom<&'a <I as BatchDataType>::Type,Error=TypeConvertError>,
          f64: From<U> {
    fn forward_batch_norm<'a>(&self, input: &'a I, scale: &CudaTensor1dPtr<U,N>, bias: &CudaTensor1dPtr<U,N>,
                          estimated_mean: &CudaTensor1dPtr<U,N>, estimated_variance: &CudaTensor1dPtr<U,N>)
        -> Result<I,EvaluateError> {
        let input = CudaTensor1dPtrView::<'a,U,N>::from(input);

        let len = N as i32;

        let mut output_ptr = CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;

        let bn_scale_bias_mean_var_desc = API::create_tensor_descriptor()?;
        let xd = CudnnTensor4dDescriptor::<U>::new(1,len as usize,1,1)?;

        unsafe {
            match cudnnDeriveBNTensorDescriptor(bn_scale_bias_mean_var_desc,*xd.id_c(),CUDNN_BATCHNORM_SPATIAL) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => (),
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the vs is invalid.")));
                },
                status => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }

        let alpha = U::one();
        let beta = U::default();

        let eps = 1e-6;

        unsafe {
            match cudnnBatchNormalizationForwardInference(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_SPATIAL,
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                *xd.id_c(),
                input.as_void_ptr(),
                *xd.id_c(),
                output_ptr.as_mut_void_ptr(),
                bn_scale_bias_mean_var_desc,
                scale.as_void_ptr(),
                bias.as_void_ptr(),
                estimated_mean.as_void_ptr(),
                estimated_variance.as_void_ptr(),
                eps as f64) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {
                    return Ok(output_ptr.into());
                },
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => {
                    return Err(EvaluateError::CudnnError(rcudnn::Error::NotSupported("The function does not support the provided configuration.")));
                },
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the CdnBatchNormalizationForwardInference is invalid.")));
                },
                status => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }
    }

    fn forward_batch_norm_train<'a>(&self, input: &'a I,
                                scale: &CudaTensor1dPtr<U,N>,
                                bias: &CudaTensor1dPtr<U,N>,
                                estimated_mean: &CudaTensor1dPtr<U,N>,
                                estimated_variance: &CudaTensor1dPtr<U,N>) -> Result<(I,CudaTensor1dPtr<U,N>,CudaTensor1dPtr<U,N>),EvaluateError> {
        let input = CudaTensor1dPtrView::<'a,U,N>::from(input);

        let len = N as i32;

        let mut output_ptr = CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;

        let bn_scale_bias_mean_var_desc = API::create_tensor_descriptor()?;
        let xd = CudnnTensor4dDescriptor::<U>::new(1,len as usize,1,1)?;

        unsafe {
            match cudnnDeriveBNTensorDescriptor(bn_scale_bias_mean_var_desc,*xd.id_c(),CUDNN_BATCHNORM_SPATIAL) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => (),
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the vs is invalid.")));
                },
                status => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }

        let alpha = U::one();
        let beta = U::default();

        let eps = U::from_f64(1e-6).ok_or(
            EvaluateError::TypeCastError(String::from("An error occurred in floating point type conversion.")))?;

        let mut mean = CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;
        let mut inv_variance = CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;

        estimated_mean.memcpy_to(mean.deref_mut(),N)?;
        inv_variance.memcpy(estimated_variance.read_to_vec()?.into_boxed_slice()
                                                                .iter()
                                                                .map(|&v| U::one() / SqrtNode::new().forward(v + eps))
                                                                .collect::<Vec<U>>().as_ptr(),N)?;

        let eps = 1e-6;

        unsafe {
            match cudnnBatchNormalizationForwardInference(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_SPATIAL,
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                *xd.id_c(),
                input.as_void_ptr(),
                *xd.id_c(),
                output_ptr.as_mut_void_ptr(),
                bn_scale_bias_mean_var_desc,
                scale.as_void_ptr(),
                bias.as_void_ptr(),
                estimated_mean.as_void_ptr(),
                estimated_variance.as_void_ptr(),
                eps as f64) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {
                    return Ok((output_ptr.into(),mean,inv_variance));
                },
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => {
                    return Err(EvaluateError::CudnnError(rcudnn::Error::NotSupported("The function does not support the provided configuration.")));
                },
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the CdnBatchNormalizationForwardInference is invalid.")));
                },
                status => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }
    }

    fn batch_forward_batch_norm<'a>(&self, input: &'a <I as BatchDataType>::Type,
                                    scale: &CudaTensor1dPtr<U,N>,
                                    bias: &CudaTensor1dPtr<U,N>,
                                    estimated_mean: &CudaTensor1dPtr<U,N>, estimated_variance: &CudaTensor1dPtr<U,N>)
        -> Result<<I as BatchDataType>::Type, EvaluateError> {
        let input = CudaVecView::<'a,U,CudaTensor1dPtr<U,N>>::try_from(input)?;

        let len = input.size();

        let mut output_ptr = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,&self.memory_pool)?;

        let len = len as i32;

        let bn_scale_bias_mean_var_desc = API::create_tensor_descriptor()?;
        let xd = CudnnTensor4dDescriptor::<U>::new(len as usize,N,1,1)?;

        unsafe {
            match cudnnDeriveBNTensorDescriptor(bn_scale_bias_mean_var_desc,*xd.id_c(),CUDNN_BATCHNORM_SPATIAL) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => (),
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the vs is invalid.")));
                },
                status => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }

        let alpha = U::one();
        let beta = U::default();

        let eps = 1e-6;

        unsafe {
            match cudnnBatchNormalizationForwardInference(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_SPATIAL,
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                *xd.id_c(),
                input.as_void_ptr(),
                *xd.id_c(),
                output_ptr.as_mut_void_ptr(),
                bn_scale_bias_mean_var_desc,
                scale.as_void_ptr(),
                bias.as_void_ptr(),
                estimated_mean.as_void_ptr(),
                estimated_variance.as_void_ptr(),
                eps as f64) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {
                    return Ok(output_ptr.into_converter().try_into()?);
                },
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => {
                    return Err(EvaluateError::CudnnError(rcudnn::Error::NotSupported("The function does not support the provided configuration.")));
                },
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the CdnBatchNormalizationForwardInference is invalid.")));
                },
                status => {
                    return Err(EvaluateError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }
    }

    fn batch_forward_batch_norm_train<'a>(&self, input: &'a <I as BatchDataType>::Type,
                                      scale: &CudaTensor1dPtr<U,N>, bias: &CudaTensor1dPtr<U,N>,
                                      running_mean: &CudaTensor1dPtr<U,N>, running_variance: &CudaTensor1dPtr<U,N>,
                                      momentum: U)
        -> Result<(<I as BatchDataType>::Type,
                   CudaTensor1dPtr<U,N>,
                   CudaTensor1dPtr<U,N>,
                   CudaTensor1dPtr<U,N>,
                   CudaTensor1dPtr<U,N>), TrainingError> {
        let input = CudaVecView::<'a,U,CudaTensor1dPtr<U,N>>::try_from(input)?;

        let len = input.size();

        let mut output_ptr = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,self.get_memory_pool())?;

        let len = len as i32;

        let bn_scale_bias_mean_var_desc = API::create_tensor_descriptor()?;
        let xd = CudnnTensor4dDescriptor::<U>::new(len as usize,N,1,1)?;

        unsafe {
            match cudnnDeriveBNTensorDescriptor(bn_scale_bias_mean_var_desc,*xd.id_c(),CUDNN_BATCHNORM_SPATIAL) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => (),
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the vs is invalid.")));
                },
                status => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }

        let alpha = U::one();
        let beta = U::default();

        let eps = 1e-6;

        let mut new_running_mean = CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;
        let mut new_running_variance = CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;

        running_mean.memcpy_to(new_running_mean.deref_mut(), N)?;
        running_variance.memcpy_to(new_running_variance.deref_mut(), N)?;

        let mut mean = CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;
        let mut inv_variance = CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;

        unsafe {
            match cudnnBatchNormalizationForwardTraining(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_PER_ACTIVATION,
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                *xd.id_c(),
                input.as_void_ptr(),
                *xd.id_c(),
                output_ptr.as_mut_void_ptr(),
                bn_scale_bias_mean_var_desc,
                scale.as_void_ptr(),
                bias.as_void_ptr(),
                1. - f64::from(momentum),
                new_running_mean.as_mut_void_ptr(),
                new_running_variance.as_mut_void_ptr(),
                eps as f64,
                mean.as_mut_void_ptr(),
                inv_variance.as_mut_void_ptr()) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {
                    return Ok((output_ptr.into_converter().try_into()?,
                               mean,
                               inv_variance,
                               new_running_mean,
                               new_running_variance));
                },
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => {
                    return Err(TrainingError::CudnnError(rcudnn::Error::NotSupported("The function does not support the provided configuration.")));
                },
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the CdnBatchNormalizationForwardInference is invalid.")));
                },
                status => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }
    }

    fn backward_batch_norm<'a>(&self, loss: &'a I, input: &'a I,
                               scale: &CudaTensor1dPtr<U,N>,
                               saved_mean: &CudaTensor1dPtr<U,N>,
                               saved_inv_variance: &CudaTensor1dPtr<U,N>)
        -> Result<(I, CudaTensor1dPtr<U,N>, CudaTensor1dPtr<U,N>), TrainingError> {
        let loss = CudaTensor1dPtrView::<'a,U,N>::from(loss);
        let input = CudaTensor1dPtrView::<'a,U,N>::from(input);

        let len = N as i32;

        let mut output_ptr = CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;

        let bn_scale_bias_diff_desc = API::create_tensor_descriptor()?;
        let xd = CudnnTensor4dDescriptor::<U>::new(1,len as usize,1,1)?;

        unsafe {
            match cudnnDeriveBNTensorDescriptor(bn_scale_bias_diff_desc,*xd.id_c(),CUDNN_BATCHNORM_SPATIAL) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => (),
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the vs is invalid.")));
                },
                status => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }

        let eps = 1e-6;

        let alpha = U::one();
        let beta = U::default();

        let mut result_scale= CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;
        let mut result_bias = CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;

        unsafe {
            match cudnnBatchNormalizationBackward(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_PER_ACTIVATION,
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                *xd.id_c(),
                input.as_void_ptr(),
                *xd.id_c(),
                loss.as_void_ptr(),
                *xd.id_c(),
                output_ptr.as_mut_void_ptr(),
                bn_scale_bias_diff_desc,
                scale.as_void_ptr(),
                result_scale.as_mut_void_ptr(),
                result_bias.as_mut_void_ptr(),
                eps as f64,
                saved_mean.as_void_ptr(),
                saved_inv_variance.as_void_ptr()) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {
                    return Ok((output_ptr.into(),result_scale,result_bias));
                },
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => {
                    return Err(TrainingError::CudnnError(rcudnn::Error::NotSupported("The function does not support the provided configuration.")));
                },
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the CdnBatchNormalizationForwardInference is invalid.")));
                },
                status => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }
    }

    fn batch_backward_batch_norm<'a>(&self, loss: &'a <I as BatchDataType>::Type,
                                 input: &'a <I as BatchDataType>::Type,
                                 scale: &CudaTensor1dPtr<U,N>,
                                 saved_mean: &CudaTensor1dPtr<U,N>, saved_inv_variance: &CudaTensor1dPtr<U,N>)
        -> Result<(<I as BatchDataType>::Type, CudaTensor1dPtr<U,N>, CudaTensor1dPtr<U,N>), TrainingError> {

        let loss = CudaVecView::<'a,U,CudaTensor1dPtr<U,N>>::try_from(loss)?;
        let input = CudaVecView::<'a,U,CudaTensor1dPtr<U,N>>::try_from(input)?;

        let len = input.size();

        let mut output_ptr = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,self.get_memory_pool())?;

        let be_scale_bias_diff_desc = API::create_tensor_descriptor()?;
        let xd = CudnnTensor4dDescriptor::<U>::new(len as usize,N,1,1)?;

        unsafe {
            match cudnnDeriveBNTensorDescriptor(be_scale_bias_diff_desc, *xd.id_c(), CUDNN_BATCHNORM_SPATIAL) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => (),
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the vs is invalid.")));
                },
                status => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }

        let eps = 1e-6;

        let alpha = U::one();
        let beta = U::default();

        let mut result_scale= CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;
        let mut result_bias = CudaTensor1dPtr::<U,N>::new(self.get_memory_pool())?;

        unsafe {
            match cudnnBatchNormalizationBackward(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_PER_ACTIVATION,
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                *xd.id_c(),
                input.as_void_ptr(),
                *xd.id_c(),
                loss.as_void_ptr(),
                *xd.id_c(),
                output_ptr.as_mut_void_ptr(),
                be_scale_bias_diff_desc,
                scale.as_void_ptr(),
                result_scale.as_mut_void_ptr(),
                result_bias.as_mut_void_ptr(),
                eps as f64,
                saved_mean.as_void_ptr(),
                saved_inv_variance.as_void_ptr()) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {
                    return Ok((output_ptr.into_converter().try_into()?,result_scale,result_bias));
                },
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => {
                    return Err(TrainingError::CudnnError(rcudnn::Error::NotSupported("The function does not support the provided configuration.")));
                },
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::BadParam("The parameter passed to the CdnBatchNormalizationForwardInference is invalid.")));
                },
                status => {
                    return Err(TrainingError::CudnnError(
                        rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }
    }
}
