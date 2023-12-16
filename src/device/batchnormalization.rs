//! Implementation of the calculation process for batch normalization
use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};
use rcudnn::{API};
use rcudnn_sys::cudnnBatchNormMode_t::{CUDNN_BATCHNORM_PER_ACTIVATION, CUDNN_BATCHNORM_SPATIAL};
use rcudnn_sys::{cudnnBatchNormalizationBackward, cudnnBatchNormalizationForwardInference, cudnnBatchNormalizationForwardTraining, cudnnDeriveBNTensorDescriptor, cudnnStatus_t};

use crate::arr::{Arr, ArrView, SerializedVec, SerializedVecView};
use crate::ope::Sum;
use crate::collection::Broadcast;
use crate::computational_graph::{BroadcastNode, GraphNode, SqrtNode, SquareNode, SumNode};
use crate::cuda::{AsMutVoidPtr, AsPtr, AsVoidPtr, CudaMemoryPoolPtr, CudaPtr, DataTypeInfo, Memory};
use crate::cuda::cudnn::tensor::CudnnTensor4dDescriptor;
use crate::cuda::mem::CachedTensor;
use crate::device::{DeviceCpu, DeviceGpu};
use crate::error::{EvaluateError, TrainingError};
use crate::mem::AsRawSlice;
use crate::ope::UnitValue;

/// Features defining the implementation of the various computational processes in the batch normalization layer
pub trait DeviceBatchNorm<U,C,T,const N:usize>
    where U: UnitValue<U> {
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
    fn forward_batch_norm<'a>(&self, input: &ArrView<'a,U,N>, scale: &C, bias: &C,
                          estimated_mean: &C, estimated_variance: &C) -> Result<Arr<U,N>,EvaluateError>;
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
    fn forward_batch_norm_train<'a>(&self, input: ArrView<'a,U,N>, scale: &C, bias: &C,
                                estimated_mean: &C, estimated_variance: &C) -> Result<(Arr<U,N>,T,T),EvaluateError>;
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
    fn batch_forward_batch_norm<'a>(&self, input: SerializedVecView<'a,U,Arr<U,N>>, scale: &C , bias: &C,
                                estimated_mean: &C, estimated_variance: &C) -> Result<SerializedVec<U,Arr<U,N>>,EvaluateError>;
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
    fn batch_forward_batch_norm_train<'a>(&self, input: SerializedVecView<'a,U,Arr<U,N>>, scale: &C, bias: &C,
                                      running_mean: &C, running_variance: &C, momentum: U)
                                      -> Result<(SerializedVec<U,Arr<U,N>>,T,T,Arr<U,N>,Arr<U,N>),TrainingError>;
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
    fn backward_batch_norm<'a>(&self, loss:ArrView<'a,U,N>, input: ArrView<'a,U,N>, scale: &C,
                           saved_mean: &T, saved_inv_variance: &T) -> Result<(Arr<U,N>,Arr<U,N>,Arr<U,N>), TrainingError>;
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
    fn batch_backward_batch_norm<'a>(&self, loss:SerializedVecView<'a,U,Arr<U,N>>, input: SerializedVecView<'a,U,Arr<U,N>>,
                                 scale: &C, saved_mean: &T, saved_inv_variance: &T) -> Result<(SerializedVec<U,Arr<U,N>>,Arr<U,N>,Arr<U,N>), TrainingError>;
}
impl<U,const N:usize> DeviceBatchNorm<U,Arr<U,N>,Arr<U,N>,N> for DeviceCpu<U>
    where U: UnitValue<U> {
    fn forward_batch_norm<'a>(&self, input: &ArrView<'a,U,N>, scale: &Arr<U,N>, bias: &Arr<U,N>,
                          estimated_mean: &Arr<U,N>, estimated_variance: &Arr<U,N>) -> Result<Arr<U,N>,EvaluateError> {
        let eps = U::from_f64(1e-6).ok_or(EvaluateError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        Ok(input.par_iter()
            .zip(scale.par_iter())
            .zip(bias.par_iter())
            .zip(estimated_mean.par_iter())
            .zip(estimated_variance.par_iter())
            .map(|((((&i,&scale),&bias),&mean),&variance)| {
                scale * ((i - mean) / SqrtNode::new().forward(variance + eps)) + bias
            }).collect::<Vec<U>>().try_into()?)
    }

    fn forward_batch_norm_train<'a>(&self, input: ArrView<'a,U,N>,
                                scale: &Arr<U,N>,
                                bias: &Arr<U,N>,
                                estimated_mean: &Arr<U,N>,
                                estimated_variance: &Arr<U,N>) -> Result<(Arr<U,N>,Arr<U,N>,Arr<U,N>),EvaluateError> {
        let eps = U::from_f64(1e-6).ok_or(EvaluateError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        Ok((input.par_iter()
                .zip(scale.par_iter())
                .zip(bias.par_iter())
                .zip(estimated_mean.par_iter())
                .zip(estimated_variance.par_iter())
                .map(|((((&i,&scale),&bias),&mean),&variance)| {
                    scale * ((i - mean) / SqrtNode::new().forward(variance + eps)) + bias
                }).collect::<Vec<U>>().try_into()?,
            estimated_mean.clone(),
            estimated_variance.par_iter().map(|&v| U::one() / SqrtNode::new().forward(v + eps)).collect::<Vec<U>>().try_into()?
        ))
    }

    fn batch_forward_batch_norm<'a>(&self, input: SerializedVecView<'a,U,Arr<U,N>>, scale: &Arr<U,N>, bias: &Arr<U,N>,
                                estimated_mean: &Arr<U,N>, estimated_variance: &Arr<U,N>) -> Result<SerializedVec<U,Arr<U,N>>, EvaluateError> {

        let eps = U::from_f64(1e-6).ok_or(EvaluateError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        Ok(input.par_iter().map(|input| {
            input.par_iter()
                .zip(scale.par_iter())
                .zip(bias.par_iter())
                .zip(estimated_mean.par_iter())
                .zip(estimated_variance.par_iter())
                .map(|((((&i,&scale),&bias),&mean),&variance)| {
                    scale * (i - mean) / SqrtNode::new().forward(variance + eps) + bias
                }).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }

    fn batch_forward_batch_norm_train<'a>(&self, input: SerializedVecView<'a,U,Arr<U,N>>,
                                      scale: &Arr<U,N>, bias: &Arr<U,N>,
                                      running_mean: &Arr<U,N>, running_variance: &Arr<U,N>,
                                      momentum: U)
                                      -> Result<(SerializedVec<U,Arr<U,N>>,Arr<U,N>,Arr<U,N>,Arr<U,N>,Arr<U,N>), TrainingError> {

        let eps = U::from_f64(1e-6).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let n = input.len();
        let un = U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let mean:Arr<U,N> = SumNode::<U,SerializedVecView<'_,U,Arr<U,N>>>::new().forward(input) / un;

        let variance:SerializedVec<U,Arr<U,N>> = (input - Broadcast::<Arr<U,N>>(mean.clone()))
            .par_iter()
            .map(|i| {
                i.par_iter().map(|&i| {
                    SquareNode::new().forward(i)
                }).collect::<Vec<U>>().try_into()
            }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into();
        let variance = variance.sum() / un;

        let inv_variance:Arr<U,N> = variance.par_iter().map(|&v| U::one() / SqrtNode::new().forward(v + eps)).collect::<Vec<U>>().try_into()?;

        let o:SerializedVec<U,Arr<U,N>> = Broadcast(inv_variance.clone()) * (input - Broadcast(mean.clone()));

        let running_mean = running_mean * momentum + &mean * (U::one() - momentum);
        let running_variance = running_variance * momentum + variance * (U::one() - momentum);

        let o = (BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().forward((scale,n)) * o) + Broadcast(bias.clone());

        Ok((o,mean,inv_variance,running_mean,running_variance))
    }

    fn backward_batch_norm<'a>(&self, loss: ArrView<'a,U,N>, input: ArrView<'a,U,N>,
                           scale: &Arr<U,N>, saved_mean: &Arr<U,N>, saved_inv_variance: &Arr<U,N>)
                           -> Result<(Arr<U,N>, Arr<U,N>, Arr<U,N>), TrainingError> {
        let b = loss.clone();

        let x = input - saved_mean;

        let s = (&x * saved_inv_variance) * loss;

        let dx1 = scale * loss;
        let dx2 = &dx1 * saved_inv_variance;
        let dx3 = &x * dx1;
        let dx4 =  -(saved_inv_variance * saved_inv_variance) * dx3;
        let dx5 = dx4 * (saved_inv_variance / U::from_f64(2.).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from f64.")
        ))?);
        let dx6 = &x * dx5 * U::from_usize(2).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;
        let dx7 = dx2 + dx6;
        let dx8 = &dx7;
        let dx9 = -&dx7;
        let dx = dx8 + dx9;

        Ok((dx,s,b.into()))
    }

    fn batch_backward_batch_norm<'a>(&self, loss: SerializedVecView<'a,U,Arr<U,N>>,
                                 input: SerializedVecView<'a,U,Arr<U,N>>,
                                 scale: &Arr<U,N>,
                                 saved_mean: &Arr<U,N>, saved_inv_variance: &Arr<U,N>)
                                 -> Result<(SerializedVec<U,Arr<U,N>>, Arr<U,N>, Arr<U,N>), TrainingError> {
        let n = input.len();

        let un = U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let b = BroadcastNode::<U,SerializedVecView<'_,U,Arr<U,N>>>::new().backward(loss);

        let x = BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().forward((saved_mean,n));
        let x2 = input - &x;
        let iv = BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().forward((saved_inv_variance,n));

        let s = BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().backward(&(&x2 * &iv * loss));

        let dx1 = Broadcast(scale.clone()) * loss;
        let dx2 = &dx1 * iv;
        let dx3 = BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().backward(&(&x2 * dx1));
        let dx4 = -(saved_inv_variance * saved_inv_variance) * dx3;
        let dx5 = dx4 * (saved_inv_variance / U::from_f64(2.).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from f64.")
        ))?);
        let dx6 = SumNode::<U,SerializedVec<U,Arr<U,N>>>::new().backward((&(dx5 / un),n));
        let dx7 = x2 * dx6 * U::from_usize(2).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;
        let dx8 = dx2 + dx7;
        let dx9 = &dx8;
        let dx10 = -&dx8;
        let dx11 = BroadcastNode::<U,&SerializedVec<U,Arr<U,N>>>::new().backward(&dx10);
        let dx12 = SumNode::<U,SerializedVec<U,Arr<U,N>>>::new().backward((&dx11,n)) / un;

        let dx = dx9 + dx12;

        Ok((dx,s,b))
    }
}
impl<U,const N:usize> DeviceBatchNorm<U,CachedTensor<U,Arr<U,N>>,CudaPtr<U>,N> for DeviceGpu<U>
    where U: UnitValue<U> + DataTypeInfo + AsVoidPtr,
          f64: From<U> {
    fn forward_batch_norm<'a>(&self, input: &ArrView<'a,U,N>, scale: &CachedTensor<U,Arr<U,N>>, bias: &CachedTensor<U,Arr<U,N>>,
                          estimated_mean: &CachedTensor<U,Arr<U,N>>, estimated_variance: &CachedTensor<U,Arr<U,N>>)
        -> Result<Arr<U,N>,EvaluateError> {
        let len = input.len() as i32;

        let mut input_ptr = CudaMemoryPoolPtr::<U>::new(len as usize,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::<U>::new(len as usize,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize)?;

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
                input_ptr.as_void_ptr(),
                *xd.id_c(),
                output_ptr.as_mut_void_ptr(),
                bn_scale_bias_mean_var_desc,
                scale.as_void_ptr(),
                bias.as_void_ptr(),
                estimated_mean.as_void_ptr(),
                estimated_variance.as_void_ptr(),
                eps as f64) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {
                    return Ok(output_ptr.read_to_vec()?.try_into()?);
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

    fn forward_batch_norm_train<'a>(&self, input: ArrView<'a,U,N>,
                                scale: &CachedTensor<U,Arr<U,N>>,
                                bias: &CachedTensor<U,Arr<U,N>>,
                                estimated_mean: &CachedTensor<U,Arr<U,N>>,
                                estimated_variance: &CachedTensor<U,Arr<U,N>>) -> Result<(Arr<U,N>,CudaPtr<U>,CudaPtr<U>),EvaluateError> {
        let len = input.len() as i32;

        let mut input_ptr = CudaMemoryPoolPtr::<U>::new(len as usize,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::<U>::new(len as usize,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize)?;

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

        let mut mean = CudaPtr::<U>::new(N)?;
        let mut inv_variance = CudaPtr::<U>::new(N)?;

        mean.memcpy(estimated_mean.as_ptr(),N)?;
        inv_variance.memcpy(estimated_variance.iter().map(|&v| U::one() / SqrtNode::new().forward(v + eps)).collect::<Vec<U>>().as_ptr(),N)?;

        let eps = 1e-6;

        unsafe {
            match cudnnBatchNormalizationForwardInference(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_SPATIAL,
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                *xd.id_c(),
                input_ptr.as_void_ptr(),
                *xd.id_c(),
                output_ptr.as_mut_void_ptr(),
                bn_scale_bias_mean_var_desc,
                scale.as_void_ptr(),
                bias.as_void_ptr(),
                estimated_mean.as_void_ptr(),
                estimated_variance.as_void_ptr(),
                eps as f64) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {
                    return Ok((output_ptr.read_to_vec()?.try_into()?,mean,inv_variance));
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

    fn batch_forward_batch_norm<'a>(&self, input: SerializedVecView<'a,U,Arr<U,N>>, scale: &CachedTensor<U,Arr<U,N>>, bias: &CachedTensor<U,Arr<U,N>>,
                                estimated_mean: &CachedTensor<U,Arr<U,N>>, estimated_variance: &CachedTensor<U,Arr<U,N>>)
        -> Result<SerializedVec<U,Arr<U,N>>, EvaluateError> {
        let len = input.len() as i32;

        let mut input_ptr = CudaMemoryPoolPtr::<U>::new(len as usize * N,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::<U>::new(len as usize * N,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize * N)?;

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
                input_ptr.as_void_ptr(),
                *xd.id_c(),
                output_ptr.as_mut_void_ptr(),
                bn_scale_bias_mean_var_desc,
                scale.as_void_ptr(),
                bias.as_void_ptr(),
                estimated_mean.as_void_ptr(),
                estimated_variance.as_void_ptr(),
                eps as f64) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {
                    return Ok(output_ptr.read_to_vec()?.try_into()?);
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

    fn batch_forward_batch_norm_train<'a>(&self, input: SerializedVecView<'a,U,Arr<U,N>>,
                                      scale: &CachedTensor<U,Arr<U,N>>, bias: &CachedTensor<U,Arr<U,N>>,
                                      running_mean: &CachedTensor<U,Arr<U,N>>, running_variance: &CachedTensor<U,Arr<U,N>>,
                                      momentum: U)
                                      -> Result<(SerializedVec<U,Arr<U,N>>,CudaPtr<U>,CudaPtr<U>,Arr<U,N>,Arr<U,N>), TrainingError> {
        let len = input.len() as i32;

        let mut input_ptr = CudaMemoryPoolPtr::<U>::new(len as usize * N,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::<U>::new(len as usize * N,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize * N)?;

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

        let mut running_mean_ptr = CudaPtr::<U>::new(N)?;
        let mut running_variance_ptr = CudaPtr::<U>::new(N)?;

        running_mean_ptr.memcpy(running_mean.as_raw_slice().as_ptr(),N)?;
        running_variance_ptr.memcpy(running_variance.as_raw_slice().as_ptr(),N)?;

        let mut mean = CudaPtr::<U>::new(N)?;
        let mut inv_variance = CudaPtr::<U>::new(N)?;

        unsafe {
            match cudnnBatchNormalizationForwardTraining(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_PER_ACTIVATION,
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                *xd.id_c(),
                input_ptr.as_void_ptr(),
                *xd.id_c(),
                output_ptr.as_mut_void_ptr(),
                bn_scale_bias_mean_var_desc,
                scale.as_void_ptr(),
                bias.as_void_ptr(),
                1. - f64::from(momentum),
                running_mean_ptr.as_mut_void_ptr(),
                running_variance_ptr.as_mut_void_ptr(),
                eps as f64,
                mean.as_mut_void_ptr(),
                inv_variance.as_mut_void_ptr()) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {
                    return Ok((output_ptr.read_to_vec()?.try_into()?,
                               mean,
                               inv_variance,
                               running_mean_ptr.read_to_vec()?.try_into()?,
                               running_variance_ptr.read_to_vec()?.try_into()?));
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

    fn backward_batch_norm<'a>(&self, loss: ArrView<'a,U,N>, input: ArrView<'a,U,N>,
                           scale: &CachedTensor<U,Arr<U,N>>, saved_mean: &CudaPtr<U>, saved_inv_variance: &CudaPtr<U>)
                           -> Result<(Arr<U,N>, Arr<U,N>, Arr<U,N>), TrainingError> {
        let len = input.len() as i32;

        let mut loss_ptr = CudaMemoryPoolPtr::<U>::new(len as usize,&self.memory_pool)?;
        let mut input_ptr = CudaMemoryPoolPtr::<U>::new(len as usize,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::<U>::new(len as usize,&self.memory_pool)?;

        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),len as usize)?;
        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize)?;

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

        let mut result_scale= CudaPtr::<U>::new(N)?;
        let mut result_bias = CudaPtr::<U>::new(N)?;

        unsafe {
            match cudnnBatchNormalizationBackward(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_PER_ACTIVATION,
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                *xd.id_c(),
                input_ptr.as_void_ptr(),
                *xd.id_c(),
                loss_ptr.as_void_ptr(),
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
                    return Ok((output_ptr.read_to_vec()?.try_into()?,result_scale.read_to_vec()?.try_into()?,result_bias.read_to_vec()?.try_into()?));
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

    fn batch_backward_batch_norm<'a>(&self, loss: SerializedVecView<'a,U,Arr<U,N>>,
                                 input: SerializedVecView<'a,U,Arr<U,N>>,
                                 scale: &CachedTensor<U,Arr<U,N>>,
                                 saved_mean: &CudaPtr<U>, saved_inv_variance: &CudaPtr<U>)
                                 -> Result<(SerializedVec<U,Arr<U,N>>, Arr<U,N>, Arr<U,N>), TrainingError> {
        let len = input.len() as i32;

        let mut loss_ptr = CudaMemoryPoolPtr::<U>::new(len as usize * N,&self.memory_pool)?;
        let mut input_ptr = CudaMemoryPoolPtr::<U>::new(len as usize * N,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::<U>::new(len as usize * N,&self.memory_pool)?;

        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),len as usize * N)?;
        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize * N)?;

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

        let mut result_scale= CudaPtr::<U>::new(N)?;
        let mut result_bias = CudaPtr::<U>::new(N)?;

        unsafe {
            match cudnnBatchNormalizationBackward(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_PER_ACTIVATION,
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                alpha.as_void_ptr(),
                beta.as_void_ptr(),
                *xd.id_c(),
                input_ptr.as_void_ptr(),
                *xd.id_c(),
                loss_ptr.as_void_ptr(),
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
                    return Ok((output_ptr.read_to_vec()?.try_into()?,result_scale.read_to_vec()?.try_into()?,result_bias.read_to_vec()?.try_into()?));
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
