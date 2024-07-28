//! Implementation of the calculation process for bias layers

use std::fmt::Debug;
use std::iter;
use std::ops::DerefMut;
use libc::c_int;
use rcublas_sys::{cublasDaxpy_v2, cublasSaxpy_v2, cublasStatus_t};
use crate::arr::{Arr, ArrView, IntoConverter, SerializedVec, SerializedVecView};
use crate::collection::Broadcast;
use crate::cuda::{AsMutPtr, AsPtr, CudaPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, Memory, MemoryMoveTo};
use crate::device::{DeviceCpu, DeviceGpu, DeviceMemoryPool, DeviceReduce};
use crate::error::{EvaluateError, TrainingError, TypeConvertError};
use crate::layer::{BatchDataType, BatchSize};
use crate::ope::UnitValue;

/// Trait that defines the implementation of various calculation processes in the bias layer
pub trait DeviceBias<U,T,IO,const N: usize>
    where U: UnitValue<U>,
          IO: BatchDataType + Debug,
          <IO as BatchDataType>::Type: BatchSize + Debug {
    /// Forward propagation calculation
    /// # Arguments
    /// * `bias` - bias weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn forward_bias<'a>(&self, bias:&T, input:&'a IO) -> Result<IO, EvaluateError>;
    /// Error back propagation calculation
    /// # Arguments
    /// * `loss` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_bias<'a>(&self, input: IO) -> Result<IO, TrainingError>;
    /// Calculate the gradient of the weights
    /// # Arguments
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_bias_weight_gradient<'a>(&self, loss: &'a IO) -> Result<T, TrainingError>;
    /// Forward propagation calculation in batch
    /// # Arguments
    /// * `bias` - bias weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_forward_bias<'a>(&self,bias:&T,input: &'a <IO as BatchDataType>::Type) -> Result<<IO as BatchDataType>::Type,TrainingError>;
    /// Error back propagation in batch
    /// # Arguments
    /// * `loss` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_bias<'a>(&self, input: <IO as BatchDataType>::Type) -> Result<<IO as BatchDataType>::Type, TrainingError>;
    /// Calculate the gradient of the weights in batch
    /// # Arguments
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_bias_weight_gradient<'a>(&self, loss: &'a <IO as BatchDataType>::Type) -> Result<T, TrainingError>;
}
impl<U,IO,const N:usize> DeviceBias<U,Arr<U,N>,IO,N> for DeviceCpu<U>
    where U: UnitValue<U>,
          IO: BatchDataType + Debug + Clone,
          <IO as BatchDataType>::Type: BatchSize + Debug,
          IO: From<Arr<U,N>>,
          Arr<U,N>: From<IO>,
          SerializedVec<U,Arr<U,N>>: IntoConverter,
          <IO as BatchDataType>::Type: TryFrom<<SerializedVec<U,Arr<U,N>> as IntoConverter>::Converter,Error=TypeConvertError>,
          for<'a> ArrView<'a,U,N>: From<&'a IO>,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a <IO as BatchDataType>::Type,Error=TypeConvertError>,
          Self: DeviceReduce<<IO as BatchDataType>::Type,Arr<U,N>,U,N> {
    fn forward_bias<'a>(&self, bias: &Arr<U,N>, input: &'a IO) -> Result<IO, EvaluateError> {
        Ok((ArrView::<'a,U,N>::from(input) + bias).into())
    }

    fn backward_bias<'a>(&self, input: IO) -> Result<IO, TrainingError> {
        Ok(input)
    }

    fn backward_bias_weight_gradient<'a>(&self, loss: &'a IO) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().into())
    }

    fn batch_forward_bias<'a>(&self, bias: &Arr<U,N>, input: &'a <IO as BatchDataType>::Type) -> Result<<IO as BatchDataType>::Type, TrainingError> {
        Ok((SerializedVecView::<'a,U,Arr<U,N>>::try_from(input)? + Broadcast(bias.clone())).into_converter().try_into()?)
    }

    fn batch_backward_bias<'a>(&self, input: <IO as BatchDataType>::Type) -> Result<<IO as BatchDataType>::Type, TrainingError> {
        Ok(input)
    }

    fn batch_backward_bias_weight_gradient<'a>(&self, loss: &'a <IO as BatchDataType>::Type) -> Result<Arr<U,N>, TrainingError> {
        self.reduce(loss)
    }
}
impl<IO,const N:usize> DeviceBias<f32,CudaTensor1dPtr<f32,N>,IO,N> for DeviceGpu<f32>
    where IO: BatchDataType + Debug,
          <IO as BatchDataType>::Type: BatchSize + Debug,
          IO: From<CudaTensor1dPtr<f32,N>>,
          CudaVec<f32,CudaTensor1dPtr<f32,N>>: IntoConverter,
          <IO as BatchDataType>::Type: TryFrom<<CudaVec<f32,CudaTensor1dPtr<f32,N>> as IntoConverter>::Converter,Error=TrainingError>,
          for<'a> CudaTensor1dPtrView<'a,f32,N>: From<&'a IO>,
          for<'a> CudaVecView<'a,f32,CudaTensor1dPtr<f32,N>>: TryFrom<&'a <IO as BatchDataType>::Type,Error=TrainingError>,
          Self: DeviceReduce<<IO as BatchDataType>::Type,CudaTensor1dPtr<f32,N>,f32,N> {
    fn forward_bias<'a>(&self, bias: &CudaTensor1dPtr<f32,N>, input: &'a IO) -> Result<IO, EvaluateError> {
        let input_ptr = CudaTensor1dPtrView::<'a,f32,N>::from(input);
        let mut output_ptr = CudaTensor1dPtr::<f32,N>::new(self.get_memory_pool())?;

        bias.memcpy_to(output_ptr.deref_mut(),N)?;

        let alpha = CudaPtr::try_from(1.0f32)?;

        match unsafe {
            cublasSaxpy_v2 (
                *self.cublas.id_c(),
                N as c_int,
                alpha.as_ptr(),
                input_ptr.as_ptr(),
                1,
                output_ptr.as_mut_ptr(),
                1
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output_ptr.into()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => {
                return Err(EvaluateError::CublasError(rcublas::Error::NotInitialized));
            },
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => {
                return Err(EvaluateError::CublasError(rcublas::Error::InvalidValue(
                    "Parameters m or n are less than 0, or incx or incy was specified as 0."
                )));
            },
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => {
                return Err(EvaluateError::CublasError(rcublas::Error::ExecutionFailed));
            },
            status => {
                return Err(EvaluateError::CublasError(rcublas::Error::Unknown(
                    "Unable to get cuBLAS cublasSgemv_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    fn backward_bias<'a>(&self, input: IO) -> Result<IO, TrainingError> {
        Ok(input)
    }

    fn backward_bias_weight_gradient<'a>(&self, loss: &'a IO) -> Result<CudaTensor1dPtr<f32,N>, TrainingError> {
        let loss = CudaTensor1dPtrView::<f32,N>::from(loss);

        let mut p = CudaTensor1dPtr::<f32,N>::new(self.get_memory_pool())?;

        p.memcpy(loss.as_ptr(),N)?;

        Ok(p)
    }
    fn batch_forward_bias<'a>(&self, bias: &CudaTensor1dPtr<f32,N>, input: &'a <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, TrainingError> {
        let len = input.size();

        let bias = iter::repeat(bias.read_to_vec()?.into_boxed_slice().iter().cloned().collect::<Vec<f32>>())
            .take(input.size()).collect::<Vec<Vec<f32>>>()
            .into_iter().flatten().collect::<Vec<f32>>();

        let input_ptr = CudaVecView::<'a,f32,CudaTensor1dPtr<f32,N>>::try_from(input)?;
        let mut output_ptr = CudaVec::<f32,CudaTensor1dPtr::<f32,N>>::new(len,self.get_memory_pool())?;

        output_ptr.memcpy(bias.as_ptr(),N * len)?;

        let alpha = CudaPtr::try_from(1.0f32)?;

        match unsafe {
            cublasSaxpy_v2 (
                *self.cublas.id_c(),
                (N * len) as c_int,
                alpha.as_ptr(),
                input_ptr.as_ptr(),
                1,
                output_ptr.as_mut_ptr(),
                1
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output_ptr.into_converter().try_into()?),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => {
                return Err(TrainingError::CublasError(rcublas::Error::NotInitialized));
            },
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => {
                return Err(TrainingError::CublasError(rcublas::Error::InvalidValue(
                    "Parameters m or n are less than 0, or incx or incy was specified as 0."
                )));
            },
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => {
                return Err(TrainingError::CublasError(rcublas::Error::ExecutionFailed));
            },
            status => {
                return Err(TrainingError::CublasError(rcublas::Error::Unknown(
                    "Unable to get cuBLAS cublasSgemv_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    fn batch_backward_bias<'a>(&self, input: <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, TrainingError> {
        Ok(input)
    }

    fn batch_backward_bias_weight_gradient<'a>(&self, loss: &'a <IO as BatchDataType>::Type)
        -> Result<CudaTensor1dPtr<f32,N>, TrainingError> {
        self.reduce(loss)
    }
}
impl<IO,const N:usize> DeviceBias<f64,CudaTensor1dPtr<f64,N>,IO,N> for DeviceGpu<f64>
    where IO: BatchDataType + Debug,
          <IO as BatchDataType>::Type: BatchSize + Debug,
          IO: From<CudaTensor1dPtr<f64,N>>,
          CudaVec<f64,CudaTensor1dPtr<f64,N>>: IntoConverter,
          <IO as BatchDataType>::Type: TryFrom<<CudaVec<f64,CudaTensor1dPtr<f64,N>> as IntoConverter>::Converter,Error=TrainingError>,
          for<'a> CudaTensor1dPtrView<'a,f64,N>: From<&'a IO>,
          for<'a> CudaVecView<'a,f64,CudaTensor1dPtr<f64,N>>: TryFrom<&'a <IO as BatchDataType>::Type,Error=TrainingError>,
          Self: DeviceReduce<<IO as BatchDataType>::Type,CudaTensor1dPtr<f64,N>,f64,N> {
    fn forward_bias<'a>(&self, bias: &CudaTensor1dPtr<f64,N>, input: &'a IO) -> Result<IO, EvaluateError> {
        let input_ptr = CudaTensor1dPtrView::<'a,f64,N>::from(input);
        let mut output_ptr = CudaTensor1dPtr::<f64,N>::new(self.get_memory_pool())?;

        bias.memcpy_to(output_ptr.deref_mut(),N)?;

        let alpha = CudaPtr::try_from(1.0f64)?;

        match unsafe {
            cublasDaxpy_v2 (
                *self.cublas.id_c(),
                N as c_int,
                alpha.as_ptr(),
                input_ptr.as_ptr(),
                1,
                output_ptr.as_mut_ptr(),
                1
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output_ptr.into()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => {
                return Err(EvaluateError::CublasError(rcublas::Error::NotInitialized));
            },
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => {
                return Err(EvaluateError::CublasError(rcublas::Error::InvalidValue(
                    "Parameters m or n are less than 0, or incx or incy was specified as 0."
                )));
            },
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => {
                return Err(EvaluateError::CublasError(rcublas::Error::ExecutionFailed));
            },
            status => {
                return Err(EvaluateError::CublasError(rcublas::Error::Unknown(
                    "Unable to get cuBLAS cublasSgemv_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    fn backward_bias<'a>(&self, input: IO) -> Result<IO, TrainingError> {
        Ok(input)
    }

    fn backward_bias_weight_gradient<'a>(&self, loss: &'a IO) -> Result<CudaTensor1dPtr<f64,N>, TrainingError> {
        let loss = CudaTensor1dPtrView::<f64,N>::from(loss);

        let mut p = CudaTensor1dPtr::<f64,N>::new(self.get_memory_pool())?;

        p.memcpy(loss.as_ptr(),N)?;

        Ok(p)
    }

    fn batch_forward_bias<'a>(&self, bias: &CudaTensor1dPtr<f64,N>, input: &'a <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, TrainingError> {
        let len = input.size();

        let bias = iter::repeat(bias.read_to_vec()?.into_boxed_slice().iter().cloned().collect::<Vec<f64>>())
            .take(input.size()).collect::<Vec<Vec<f64>>>()
            .into_iter().flatten().collect::<Vec<f64>>();

        let input_ptr = CudaVecView::<'a,f64,CudaTensor1dPtr<f64,N>>::try_from(input)?;
        let mut output_ptr = CudaVec::<f64,CudaTensor1dPtr<f64,N>>::new(len,&self.memory_pool)?;

        output_ptr.memcpy(bias.as_ptr(),N * len)?;

        let alpha = CudaPtr::try_from(1.0f64)?;

        match unsafe {
            cublasDaxpy_v2 (
                *self.cublas.id_c(),
                (N * len) as c_int,
                alpha.as_ptr(),
                input_ptr.as_ptr(),
                1,
                output_ptr.as_mut_ptr(),
                1
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output_ptr.into_converter().try_into()?),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => {
                return Err(TrainingError::CublasError(rcublas::Error::NotInitialized));
            },
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => {
                return Err(TrainingError::CublasError(rcublas::Error::InvalidValue(
                    "Parameters m or n are less than 0, or incx or incy was specified as 0."
                )));
            },
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => {
                return Err(TrainingError::CublasError(rcublas::Error::ExecutionFailed));
            },
            status => {
                return Err(TrainingError::CublasError(rcublas::Error::Unknown(
                    "Unable to get cuBLAS cublasSgemv_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    fn batch_backward_bias<'a>(&self, input: <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, TrainingError> {
        Ok(input)
    }

    fn batch_backward_bias_weight_gradient<'a>(&self, loss: &'a <IO as BatchDataType>::Type)
        -> Result<CudaTensor1dPtr<f64,N>, TrainingError> {
        self.reduce(loss)
    }
}

