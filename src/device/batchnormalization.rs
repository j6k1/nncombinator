use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};
use rcudnn::{API, TensorDescriptor};
use rcudnn::utils::DataType;
use rcudnn_sys::cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL;
use rcudnn_sys::{cudnnBatchNormalizationBackward, cudnnBatchNormalizationForwardInference, cudnnBatchNormalizationForwardTraining, cudnnDeriveBNTensorDescriptor, cudnnStatus_t};

use crate::arr::{Arr, VecArr};
use crate::ope::Sum;
use crate::collection::Broadcast;
use crate::computational_graph::{BroadcastNode, GraphNode, SqrtNode, SquareNode, SumNode};
use crate::cuda::{AsMutVoidPtr, AsPtr, AsVoidPtr, CudaMemoryPoolPtr, CudaPtr, Memory};
use crate::cuda::mem::CachedTensor;
use crate::device::{DeviceCpu, DeviceGpu};
use crate::error::{EvaluateError, TrainingError};
use crate::mem::AsRawSlice;
use crate::ope::UnitValue;

/// Features defining the implementation of the various computational processes in the batch normalization layer
pub trait DeviceBatchNorm<U,C,T,const N:usize>
    where U: UnitValue<U> {
    fn forward_batch_norm(&self, input: &Arr<U,N>, scale: &C, bias: &C,
                          estimated_mean: &C, estimated_variance: &C) -> Result<Arr<U,N>,EvaluateError>;
    fn forward_batch_norm_train(&self, input: &Arr<U,N>, scale: &C, bias: &C,
                                estimated_mean: &C, estimated_variance: &C) -> Result<(Arr<U,N>,T,T),EvaluateError>;
    fn batch_forward_batch_norm(&self, input: &VecArr<U,Arr<U,N>>, scale: &C , bias: &C,
                                estimated_mean: &C, estimated_variance: &C) -> Result<VecArr<U,Arr<U,N>>,EvaluateError>;
    fn batch_forward_batch_norm_train(&self, input: &VecArr<U,Arr<U,N>>, scale: &C, bias: &C,
                                      running_mean: &C, running_variance: &C, momentum: U)
                                      -> Result<(VecArr<U,Arr<U,N>>,T,T,Arr<U,N>,Arr<U,N>),TrainingError>;
    fn backward_batch_norm(&self, loss:&Arr<U,N>, input: &Arr<U,N>, scale: &C,
                           saved_mean: &T, saved_inv_variance: &T) -> Result<(Arr<U, N>,Arr<U,N>,Arr<U,N>), TrainingError>;
    fn batch_backward_batch_norm(&self, loss:&VecArr<U,Arr<U,N>>, input: &VecArr<U,Arr<U,N>>,
                                 scale: &C, saved_mean: &T, saved_inv_variance: &T) -> Result<(VecArr<U,Arr<U, N>>,Arr<U,N>,Arr<U,N>), TrainingError>;
}
impl<U,const N:usize> DeviceBatchNorm<U,Arr<U,N>,Arr<U,N>,N> for DeviceCpu<U>
    where U: UnitValue<U> {
    fn forward_batch_norm(&self, input: &Arr<U, N>, scale: &Arr<U, N>, bias: &Arr<U, N>,
                          estimated_mean: &Arr<U, N>, estimated_variance: &Arr<U, N>) -> Result<Arr<U, N>,EvaluateError> {
        let eps = U::from_f64(1e-6).ok_or(EvaluateError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        Ok(input.par_iter()
            .zip(scale.par_iter())
            .zip(bias.par_iter())
            .zip(estimated_mean.par_iter())
            .zip(estimated_variance.par_iter())
            .map(|((((&i,&scale),&bias),&mean),&variance)| {
                scale * ((i - mean) / (variance + eps)) + bias
            }).collect::<Vec<U>>().try_into()?)
    }

    fn forward_batch_norm_train(&self, input: &Arr<U, N>,
                                scale: &Arr<U, N>,
                                bias: &Arr<U, N>,
                                estimated_mean: &Arr<U, N>,
                                estimated_variance: &Arr<U, N>) -> Result<(Arr<U,N>,Arr<U,N>,Arr<U,N>),EvaluateError> {
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

    fn batch_forward_batch_norm(&self, input: &VecArr<U, Arr<U, N>>, scale: &Arr<U, N>, bias: &Arr<U, N>,
                                estimated_mean: &Arr<U, N>, estimated_variance: &Arr<U, N>) -> Result<VecArr<U, Arr<U, N>>, EvaluateError> {

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

    fn batch_forward_batch_norm_train(&self, input: &VecArr<U, Arr<U, N>>,
                                      scale: &Arr<U, N>, bias: &Arr<U, N>,
                                      running_mean: &Arr<U, N>, running_variance: &Arr<U, N>,
                                      momentum: U)
                                      -> Result<(VecArr<U,Arr<U,N>>,Arr<U,N>,Arr<U,N>,Arr<U,N>,Arr<U,N>), TrainingError> {

        let eps = U::from_f64(1e-6).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let n = input.len();
        let un = U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let mean:Arr<U,N> = SumNode::new().forward(input) / un;

        let variance:VecArr<U,Arr<U,N>> = (input - Broadcast::<Arr<U,N>>(mean.clone()))
            .par_iter()
            .map(|i| {
                i.par_iter().map(|&i| {
                    SquareNode::new().forward(i)
                }).collect::<Vec<U>>().try_into()
            }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into();
        let variance = variance.sum() / un;

        let inv_variance:Arr<U,N> = variance.par_iter().map(|&v| U::one() / SqrtNode::new().forward(v + eps)).collect::<Vec<U>>().try_into()?;

        let o:VecArr<U,Arr<U,N>> = Broadcast(inv_variance.clone()) * (input - Broadcast(mean.clone()));

        let running_mean = running_mean * momentum + &mean * (U::one() - momentum);
        let running_variance = running_variance * momentum + variance * (U::one() - momentum);

        let o = (BroadcastNode::new().forward((scale,n)) * o) + Broadcast(bias.clone());

        Ok((o,mean,inv_variance,running_mean,running_variance))
    }

    fn backward_batch_norm(&self, loss: &Arr<U, N>, input: &Arr<U, N>,
                           scale: &Arr<U, N>, saved_mean: &Arr<U, N>, saved_inv_variance: &Arr<U, N>)
                           -> Result<(Arr<U, N>, Arr<U, N>, Arr<U, N>), TrainingError> {
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

        Ok((dx,s,b))
    }

    fn batch_backward_batch_norm(&self, loss: &VecArr<U, Arr<U, N>>,
                                 input: &VecArr<U,Arr<U,N>>,
                                 scale: &Arr<U, N>,
                                 saved_mean: &Arr<U, N>, saved_inv_variance: &Arr<U, N>)
                                 -> Result<(VecArr<U, Arr<U, N>>, Arr<U, N>, Arr<U, N>), TrainingError> {
        let n = input.len();

        let un = U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let b = BroadcastNode::new().backward(loss);

        let x = BroadcastNode::new().forward((saved_mean,n));
        let x2 = input - &x;
        let iv = BroadcastNode::new().forward((saved_inv_variance,n));

        let s = BroadcastNode::new().backward(&(&x2 * &iv * loss));

        let dx1 = Broadcast(scale.clone()) * loss;
        let dx2 = &dx1 * iv;
        let dx3 = BroadcastNode::new().backward(&(&x2 * dx1));
        let dx4 = -(saved_inv_variance * saved_inv_variance) * dx3;
        let dx5 = dx4 * (saved_inv_variance / U::from_f64(2.).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from f64.")
        ))?);
        let dx6 = SumNode::new().backward((&(dx5 / un),n));
        let dx7 = x2 * dx6 * U::from_usize(2).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;
        let dx8 = dx2 + dx7;
        let dx9 = &dx8;
        let dx10 = -&dx8;
        let dx11 = BroadcastNode::new().backward(&dx10);
        let dx12 = SumNode::new().backward((&dx11,n)) / un;

        let dx = dx9 + dx12;

        Ok((dx,s,b))
    }
}
impl<const N:usize> DeviceBatchNorm<f32,CachedTensor<f32,Arr<f32,N>>,CudaPtr<f32>,N> for DeviceGpu<f32> {
    fn forward_batch_norm(&self, input: &Arr<f32,N>, scale: &CachedTensor<f32,Arr<f32,N>>, bias: &CachedTensor<f32,Arr<f32,N>>,
                          estimated_mean: &CachedTensor<f32,Arr<f32,N>>, estimated_variance: &CachedTensor<f32,Arr<f32,N>>)
        -> Result<Arr<f32,N>,EvaluateError> {
        let len = input.len() as i32;

        let mut input_ptr = CudaMemoryPoolPtr::new(len as usize,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(len as usize,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize)?;

        let bn_scale_bias_mean_var_desc = API::create_tensor_descriptor()?;
        let xd = TensorDescriptor::new(&[1,1,1,len],&[len,len,len,1],DataType::Float)?;

        unsafe {
            cudnnDeriveBNTensorDescriptor(bn_scale_bias_mean_var_desc,*xd.id_c(),CUDNN_BATCHNORM_SPATIAL);
        }

        let alpha = 1.;
        let beta = 0.;
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

    fn forward_batch_norm_train(&self, input: &Arr<f32,N>,
                                scale: &CachedTensor<f32,Arr<f32,N>>,
                                bias: &CachedTensor<f32,Arr<f32,N>>,
                                estimated_mean: &CachedTensor<f32,Arr<f32,N>>,
                                estimated_variance: &CachedTensor<f32,Arr<f32,N>>) -> Result<(Arr<f32,N>,CudaPtr<f32>,CudaPtr<f32>),EvaluateError> {
        let len = input.len() as i32;

        let mut input_ptr = CudaMemoryPoolPtr::new(len as usize,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(len as usize,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize)?;

        let bn_scale_bias_mean_var_desc = API::create_tensor_descriptor()?;
        let xd = TensorDescriptor::new(&[1,1,1,len],&[len,len,len,1],DataType::Float)?;

        unsafe {
            cudnnDeriveBNTensorDescriptor(bn_scale_bias_mean_var_desc,*xd.id_c(),CUDNN_BATCHNORM_SPATIAL);
        }

        let alpha = 1.;
        let beta = 0.;
        let eps = 1e-6;

        let mut mean = CudaPtr::new(len as usize)?;
        let mut inv_variance = CudaPtr::new(len as usize)?;

        mean.memcpy(estimated_mean.as_ptr(),len as usize)?;
        inv_variance.memcpy(estimated_variance.iter().map(|v| 1. / SqrtNode::new().forward(v + eps)).collect::<Vec<f32>>().as_ptr(),len as usize)?;

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

    fn batch_forward_batch_norm(&self, input: &VecArr<f32,Arr<f32,N>>, scale: &CachedTensor<f32,Arr<f32,N>>, bias: &CachedTensor<f32,Arr<f32,N>>,
                                estimated_mean: &CachedTensor<f32,Arr<f32,N>>, estimated_variance: &CachedTensor<f32,Arr<f32,N>>)
        -> Result<VecArr<f32,Arr<f32,N>>, EvaluateError> {
        let len = input.len() as i32;

        let mut input_ptr = CudaMemoryPoolPtr::<f32>::new(len as usize * N,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::<f32>::new(len as usize * N,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize * N)?;

        let bn_scale_bias_mean_var_desc = API::create_tensor_descriptor()?;
        let xd = TensorDescriptor::new(&[1,1,len,N as i32],
                                           &[len * N as i32,len * N as i32,N as i32,1],
                                           DataType::Float)?;

        unsafe {
            cudnnDeriveBNTensorDescriptor(bn_scale_bias_mean_var_desc,*xd.id_c(),CUDNN_BATCHNORM_SPATIAL);
        }

        let alpha = 1.;
        let beta = 0.;
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

    fn batch_forward_batch_norm_train(&self, input: &VecArr<f32,Arr<f32,N>>,
                                      scale: &CachedTensor<f32,Arr<f32,N>>, bias: &CachedTensor<f32,Arr<f32,N>>,
                                      _: &CachedTensor<f32,Arr<f32,N>>, _: &CachedTensor<f32,Arr<f32,N>>,
                                      momentum: f32)
                                      -> Result<(VecArr<f32,Arr<f32,N>>,CudaPtr<f32>,CudaPtr<f32>,Arr<f32,N>,Arr<f32,N>), TrainingError> {
        let len = input.len() as i32;

        let mut input_ptr = CudaMemoryPoolPtr::<f32>::new(len as usize * N,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::<f32>::new(len as usize * N,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize * N)?;

        let bn_scale_bias_mean_var_desc = API::create_tensor_descriptor()?;
        let xd = TensorDescriptor::new(&[1,1,len,N as i32],
                                           &[len * N as i32,len * N as i32,N as i32,1],
                                           DataType::Float)?;

        unsafe {
            cudnnDeriveBNTensorDescriptor(bn_scale_bias_mean_var_desc,*xd.id_c(),CUDNN_BATCHNORM_SPATIAL);
        }

        let alpha = 1.;
        let beta = 0.;
        let eps = 1e-6;

        let mut running_mean = CudaPtr::new(len as usize)?;
        let mut running_variance = CudaPtr::new(len as usize)?;

        let mut mean = CudaPtr::new(len as usize)?;
        let mut inv_variance = CudaPtr::new(len as usize)?;

        unsafe {
            match cudnnBatchNormalizationForwardTraining(
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
                momentum as f64,
                running_mean.as_mut_void_ptr(),
                running_variance.as_mut_void_ptr(),
                eps as f64,
                mean.as_mut_void_ptr(),
                inv_variance.as_mut_void_ptr()) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => {
                    return Ok((output_ptr.read_to_vec()?.try_into()?,
                               mean,
                               inv_variance,
                               running_mean.read_to_vec()?.try_into()?,
                               running_variance.read_to_vec()?.try_into()?));
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

    fn backward_batch_norm(&self, loss: &Arr<f32,N>, input: &Arr<f32,N>,
                           scale: &CachedTensor<f32,Arr<f32,N>>, saved_mean: &CudaPtr<f32>, saved_inv_variance: &CudaPtr<f32>)
                           -> Result<(Arr<f32,N>, Arr<f32,N>, Arr<f32,N>), TrainingError> {
        let len = input.len() as i32;

        let mut loss_ptr = CudaMemoryPoolPtr::<f32>::new(len as usize,&self.memory_pool)?;
        let mut input_ptr = CudaMemoryPoolPtr::<f32>::new(len as usize,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::<f32>::new(len as usize,&self.memory_pool)?;

        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),len as usize)?;
        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize)?;

        let be_scale_bias_diff_desc = API::create_tensor_descriptor()?;
        let xd = TensorDescriptor::new(&[1,1,1,len],&[len,len,len,1],DataType::Float)?;

        unsafe {
            cudnnDeriveBNTensorDescriptor(be_scale_bias_diff_desc, *xd.id_c(), CUDNN_BATCHNORM_SPATIAL);
        }

        let eps = 1e-6;

        let mut result_scale= CudaPtr::new(len as usize)?;
        let mut result_bias = CudaPtr::new(len as usize)?;

        unsafe {
            match cudnnBatchNormalizationBackward(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_SPATIAL,
                (1.).as_void_ptr(),
                (0.).as_void_ptr(),
                (1.).as_void_ptr(),
                (1.).as_void_ptr(),
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

    fn batch_backward_batch_norm(&self, loss: &VecArr<f32,Arr<f32,N>>,
                                 input: &VecArr<f32,Arr<f32,N>>,
                                 scale: &CachedTensor<f32,Arr<f32,N>>,
                                 saved_mean: &CudaPtr<f32>, saved_inv_variance: &CudaPtr<f32>)
                                 -> Result<(VecArr<f32,Arr<f32,N>>, Arr<f32,N>, Arr<f32,N>), TrainingError> {
        let len = input.len() as i32;

        let mut loss_ptr = CudaMemoryPoolPtr::<f32>::new(len as usize * N,&self.memory_pool)?;
        let mut input_ptr = CudaMemoryPoolPtr::<f32>::new(len as usize * N,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::<f32>::new(len as usize * N,&self.memory_pool)?;

        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),len as usize * N)?;
        input_ptr.memcpy(input.as_raw_slice().as_ptr(),len as usize * N)?;

        let be_scale_bias_diff_desc = API::create_tensor_descriptor()?;
        let xd = TensorDescriptor::new(&[1,1,len,N as i32],
                                           &[len * N as i32, len * N as i32, N as i32,1],
                                           DataType::Float)?;

        unsafe {
            cudnnDeriveBNTensorDescriptor(be_scale_bias_diff_desc, *xd.id_c(), CUDNN_BATCHNORM_SPATIAL);
        }

        let eps = 1e-6;

        let mut result_scale= CudaPtr::new(len as usize)?;
        let mut result_bias = CudaPtr::new(len as usize)?;

        unsafe {
            match cudnnBatchNormalizationBackward(
                *self.cudnn.id_c(),
                CUDNN_BATCHNORM_SPATIAL,
                (1.).as_void_ptr(),
                (0.).as_void_ptr(),
                (1.).as_void_ptr(),
                (1.).as_void_ptr(),
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
