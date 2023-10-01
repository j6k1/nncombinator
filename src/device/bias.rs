//! Implementation of the calculation process for bias layers

use libc::c_int;
use rcublas_sys::{cublasDaxpy_v2, cublasSaxpy_v2, cublasStatus_t};
use crate::arr::{Arr, ArrView, SerializedVec, SerializedVecView};
use crate::collection::Broadcast;
use crate::cuda::{AsMutPtr, AsPtr, CudaMemoryPoolPtr, CudaPtr, Memory};
use crate::cuda::mem::CachedTensor;
use crate::device::{Device, DeviceCpu, DeviceGpu};
use crate::error::{EvaluateError, TrainingError};
use crate::mem::AsRawSlice;
use crate::ope::UnitValue;

/// Trait that defines the implementation of various calculation processes in the bias layer
pub trait DeviceBias<U,T,const N: usize> where U: UnitValue<U> {
    /// Forward propagation calculation
    /// # Arguments
    /// * `bias` - bias weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn forward_bias<'a>(&self, bias:&T, input:ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError>;
    /// Error back propagation calculation
    /// # Arguments
    /// * `loss` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_bias<'a>(&self, input:Arr<U,N>) -> Result<Arr<U,N>, TrainingError>;
    /// Calculate the gradient of the weights
    /// # Arguments
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_bias_weight_gradient<'a>(&self, loss: ArrView<'a,U,N>)
                                    -> Result<ArrView<'a,U,N>, TrainingError>;
    /// Forward propagation calculation in batch
    /// # Arguments
    /// * `bias` - bias weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_forward_bias<'a>(&self,bias:&T,input:SerializedVecView<'a,U,Arr<U,N>>)
                                -> Result<SerializedVec<U,Arr<U,N>>,TrainingError>;
    /// Error back propagation in batch
    /// # Arguments
    /// * `loss` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_bias<'a>(&self, input: SerializedVec<U,Arr<U,N>>)
                                 -> Result<SerializedVec<U,Arr<U,N>>, TrainingError>;
    /// Calculate the gradient of the weights in batch
    /// # Arguments
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_bias_weight_gradient<'a>(&self, loss: SerializedVecView<'a,U,Arr<U,N>>)
                                          -> Result<Arr<U,N>, TrainingError>;
}
impl<U,const N:usize> DeviceBias<U,Arr<U,N>,N> for DeviceCpu<U> where U: UnitValue<U> {
    fn forward_bias<'a>(&self, bias: &Arr<U,N>, input: ArrView<'a, U,N>) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input + bias)
    }

    fn backward_bias<'a>(&self, input: Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        Ok(input)
    }

    fn backward_bias_weight_gradient<'a>(&self, loss: ArrView<'a, U,N>) -> Result<ArrView<'a,U,N>, TrainingError> {
        Ok(loss)
    }

    fn batch_forward_bias<'a>(&self, bias: &Arr<U,N>, input: SerializedVecView<'a, U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U,N>>, TrainingError> {
        Ok(input + Broadcast(bias.clone()))
    }

    fn batch_backward_bias<'a>(&self, input: SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U,N>>, TrainingError> {
        Ok(input)
    }

    fn batch_backward_bias_weight_gradient<'a>(&self, loss: SerializedVecView<'a, U, Arr<U,N>>) -> Result<Arr<U,N>, TrainingError> {
        self.batch_linear_reduce(loss)
    }
}
impl<const N:usize> DeviceBias<f32,CachedTensor<f32,Arr<f32,N>>,N> for DeviceGpu<f32> {
    fn forward_bias<'a>(&self, bias: &CachedTensor<f32,Arr<f32,N>>, input: ArrView<'a,f32,N>) -> Result<Arr<f32,N>, EvaluateError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(N,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(N,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),N)?;
        output_ptr.memcpy(bias.as_raw_slice().as_ptr(),N)?;

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
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output_ptr.read_to_vec()?.try_into()?),
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

    fn backward_bias<'a>(&self, input: Arr<f32,N>) -> Result<Arr<f32,N>, TrainingError> {
        Ok(input)
    }

    fn backward_bias_weight_gradient<'a>(&self, loss: ArrView<'a,f32,N>) -> Result<ArrView<'a,f32,N>, TrainingError> {
        Ok(loss)
    }

    fn batch_forward_bias<'a>(&self, bias: &CachedTensor<f32,Arr<f32,N>>, input: SerializedVecView<'a,f32,Arr<f32,N>>) -> Result<SerializedVec<f32,Arr<f32,N>>, TrainingError> {
        let len = input.len();

        let mut input_ptr = CudaMemoryPoolPtr::new(N,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(N,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),N * len)?;
        output_ptr.memcpy(bias.as_raw_slice().as_ptr(),N * len)?;

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
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output_ptr.read_to_vec()?.try_into()?),
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

    fn batch_backward_bias<'a>(&self, input: SerializedVec<f32,Arr<f32,N>>) -> Result<SerializedVec<f32,Arr<f32,N>>, TrainingError> {
        Ok(input)
    }

    fn batch_backward_bias_weight_gradient<'a>(&self, loss: SerializedVecView<'a,f32,Arr<f32,N>>) -> Result<Arr<f32,N>, TrainingError> {
        self.batch_linear_reduce(loss)
    }
}
impl<const N:usize> DeviceBias<f64,CachedTensor<f64,Arr<f64,N>>,N> for DeviceGpu<f64> {
    fn forward_bias<'a>(&self, bias: &CachedTensor<f64,Arr<f64,N>>, input: ArrView<'a,f64,N>) -> Result<Arr<f64,N>, EvaluateError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(N,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(N,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),N)?;
        output_ptr.memcpy(bias.as_raw_slice().as_ptr(),N)?;

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
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output_ptr.read_to_vec()?.try_into()?),
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

    fn backward_bias<'a>(&self, input: Arr<f64,N>) -> Result<Arr<f64,N>, TrainingError> {
        Ok(input)
    }

    fn backward_bias_weight_gradient<'a>(&self, loss: ArrView<'a,f64,N>) -> Result<ArrView<'a,f64,N>, TrainingError> {
        Ok(loss)
    }

    fn batch_forward_bias<'a>(&self, bias: &CachedTensor<f64,Arr<f64,N>>, input: SerializedVecView<'a,f64, Arr<f64,N>>) -> Result<SerializedVec<f64, Arr<f64,N>>, TrainingError> {
        let len = input.len();

        let mut input_ptr = CudaMemoryPoolPtr::new(N,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(N,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),N * len)?;
        output_ptr.memcpy(bias.as_raw_slice().as_ptr(),N * len)?;

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
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output_ptr.read_to_vec()?.try_into()?),
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

    fn batch_backward_bias<'a>(&self, input: SerializedVec<f64,Arr<f64,N>>) -> Result<SerializedVec<f64,Arr<f64,N>>, TrainingError> {
        Ok(input)
    }

    fn batch_backward_bias_weight_gradient<'a>(&self, loss: SerializedVecView<'a,f64,Arr<f64,N>>) -> Result<Arr<f64,N>, TrainingError> {
        self.batch_linear_reduce(loss)
    }
}

