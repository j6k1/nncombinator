//! Implementation of the calculation process for full connected layers

use std::mem;
use cuda_runtime_sys::dim3;
use libc::c_uint;
use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};
use rcublas_sys::{cublasDgemm_v2, cublasDgemv_v2, cublasOperation_t, cublasSgemm_v2, cublasSgemv_v2, cublasStatus_t};
use crate::arr::{Arr, Arr2, ArrView, DiffArr, SerializedVec, SerializedVecView};
use crate::cuda::{AsMutPtr, AsPtr, CudaMemoryPoolPtr, CudaPtr, CudaTensor1dPtr, CudaTensor2dPtr, DataTypeInfo, ffi, Kernel, Memory, MemoryMoveTo};
use crate::cuda::kernel::device::{DiffLinearForward, DiffLinearForwardArgs};
use crate::device::{DeviceCpu, DeviceGpu, DeviceMemoryPool, DeviceReduce};
use crate::error::{EvaluateError, TrainingError};
use crate::mem::AsRawSlice;
use crate::ope::UnitValue;
use crate::ope::Product;

/// Trait that defines the implementation of various calculation processes in the linear layer
pub trait DeviceLinear<U,T,B,const NI: usize,const NO: usize> where U: UnitValue<U> {
    /// Forward propagation calculation
    /// # Arguments
    /// * `bias` - bias weights
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn forward_linear<'a>(&self, bias:&B, units:&T, input:ArrView<'a,U,NI>) -> Result<Arr<U, NO>, EvaluateError>;
    /// Error back propagation calculation
    /// # Arguments
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_linear<'a>(&self, units:&T, input:&'a Arr<U,NO>) -> Result<Arr<U, NI>, TrainingError>;
    /// Calculate the gradient of the weights
    /// # Arguments
    /// * `o` - Input values from upper layers
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_weight_gradient<'a>(&self, o: ArrView<'a,U,NI>, loss: &'a Arr<U,NO>)
                                -> Result<T, TrainingError>;
    /// Forward propagation calculation in batch
    /// # Arguments
    /// * `bias` - bias weights
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_forward_linear<'a>(&self,bias:&B,units:&T,input:SerializedVecView<'a,U,Arr<U,NI>>)
                            -> Result<SerializedVec<U,Arr<U,NO>>,TrainingError>;
    /// Error back propagation in batch
    /// # Arguments
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_linear<'a>(&self, units: &T, input: &'a SerializedVec<U,Arr<U,NO>>)
                             -> Result<SerializedVec<U,Arr<U, NI>>, TrainingError>;
    /// Calculate the gradient of the weights in batch
    /// # Arguments
    /// * `o` - Input values from upper layers
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_weight_gradient<'a>(&self, o: SerializedVecView<'a,U,Arr<U,NI>>, loss: &'a SerializedVec<U,Arr<U,NO>>)
                                      -> Result<T, TrainingError>;
    /// convolutional calculation
    /// # Arguments
    /// * `loss` - loss
    fn batch_linear_reduce<'a>(&self, loss: SerializedVecView<'a,U,Arr<U,NO>>) -> Result<B,TrainingError>;
}
impl<U,const NI: usize,const NO: usize> DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> for DeviceCpu<U> where U: UnitValue<U> {
    #[inline]
    fn forward_linear<'a>(&self, bias: &Arr<U, NO>, units: &Arr2<U, NI, NO>, input: ArrView<'a,U, NI>) -> Result<Arr<U, NO>, EvaluateError> {
        Ok(input.product(units) + bias)
    }

    #[inline]
    fn backward_linear<'a>(&self, units: &Arr2<U, NI, NO>, input: &'a Arr<U,NO>) -> Result<Arr<U, NI>, TrainingError> {
        Ok(units.par_iter().map(|u| {
            u.par_iter().zip(input.par_iter())
                .map(|(&w,&l)| w * l).reduce(|| U::default(), |acc,g|{
                acc + g
            })
        }).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))?)
    }

    #[inline]
    fn backward_weight_gradient<'a>(&self, o: ArrView<'a,U,NI>, loss: &'a Arr<U,NO>) -> Result<Arr2<U, NI, NO>, TrainingError> {
        Ok(o.par_iter().cloned().map(|o| {
            loss.par_iter().cloned().map(|l| o * l).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,NO>>,_>>()?.try_into().map_err(|e| TrainingError::from(e))?)
    }

    #[inline]
    fn batch_backward_linear<'a>(&self, units: &Arr2<U, NI, NO>, input: &'a SerializedVec<U,Arr<U, NO>>)
                             -> Result<SerializedVec<U,Arr<U, NI>>, TrainingError> {
        Ok(input.par_iter().map(|l| {
            units.par_iter().map(|u| {
                u.par_iter().zip(l.par_iter()).map(|(&w, &l)| w * l)
                    .reduce(|| U::default(), |acc, l| {
                        acc + l
                    })
            }).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,NI>>,_>>()?.into())
    }

    #[inline]
    fn batch_forward_linear<'a>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:SerializedVecView<'a,U,Arr<U,NI>>)
                            -> Result<SerializedVec<U,Arr<U,NO>>,TrainingError> {
        input.par_iter().map(|input| {
            input.par_iter().zip(units.par_iter()).map(|(&i, unit)| {
                unit.par_iter().map(|&w| {
                    i * w
                }).collect::<Vec<U>>()
            }).collect::<Vec<Vec<U>>>()
        }).map(|o| o.par_iter().cloned().reduce(|| vec![U::default();NO], |acc, o| {
            acc.par_iter()
                .zip(o.par_iter())
                .map(|(&acc, &o)| acc + o).collect::<Vec<U>>()
        })).map(|o| {
            o.par_iter().zip(bias.par_iter()).map(|(&o, &b)| {
                o + b
            }).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U, NO>>, _>>().map(|r| r.into()).map_err(|e| TrainingError::from(e))
    }

    #[inline]
    fn batch_backward_weight_gradient<'a>(&self, o: SerializedVecView<'a,U,Arr<U,NI>>, loss: &'a SerializedVec<U, Arr<U,NO>>)
                                      -> Result<Arr2<U,NI,NO>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter()).map(|(o,l)| o.par_iter().map(|&o| {
            l.par_iter().map(|&l| o * l).collect::<Vec<U>>()
        }).collect::<Vec<Vec<U>>>()).reduce(|| vec![vec![U::default();NO];NI], |acc,g| {
            acc.par_iter().zip(g.par_iter()).map(|(acc,g)| {
                acc.par_iter().zip(g.par_iter()).map(|(&acc,&g)| acc + g).collect::<Vec<U>>()
            }).collect::<Vec<Vec<U>>>()
        }).par_iter().cloned().map(|v| {
            v.try_into()
        }).collect::<Result<Vec<Arr<U,NO>>,_>>()?.try_into().map_err(|e| TrainingError::from(e))?)
    }

    #[inline]
    fn batch_linear_reduce<'a>(&self, loss: SerializedVecView<'a,U,Arr<U,NO>>) -> Result<Arr<U,NO>,TrainingError> {
        self.reduce(loss)
    }
}
impl<const NI: usize, const NO: usize> DeviceLinear<f32,CudaTensor2dPtr<f32,NI,NO>,CudaTensor1dPtr<f32,NO>,NI,NO> for DeviceGpu<f32> {
    #[inline]
    fn forward_linear<'a>(&self, bias: &CudaTensor1dPtr<f32,NO>, units: &CudaTensor2dPtr<f32,NI,NO>, input: ArrView<'a,f32,NI>)
                      -> Result<Arr<f32, NO>, EvaluateError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI)?;
        bias.memcpy_to(&mut output_ptr,NO)?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(1.0f32)?;

        match unsafe {
            cublasSgemv_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           NI as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           1,
                           beta.as_ptr(),
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

    #[inline]
    fn backward_linear<'a>(&self, units: &CudaTensor2dPtr<f32,NI,NO>, input: &'a Arr<f32,NO>) -> Result<Arr<f32, NI>, TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NO)?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(0.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_T,
                           cublasOperation_t::CUBLAS_OP_N,
                           NI as ::libc::c_int,
                           1 as libc::c_int,
                           NO as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NO as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NI as ::libc::c_int
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
                    "Unable to get cuBLAS cublasSgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn backward_weight_gradient<'a>(&self, o: ArrView<'a,f32,NI>, loss: &'a Arr<f32,NO>) -> Result<CudaTensor2dPtr<f32, NI,NO>, TrainingError> {
        let mut o_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let mut output_ptr = CudaTensor2dPtr::<f32,NI,NO>::new(&self.memory_pool)?;

        o_ptr.memcpy(o.as_raw_slice().as_ptr(),NI)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),NO)?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(0.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           NI as libc::c_int,
                           1 as ::libc::c_int,
                           alpha.as_ptr(),
                           loss_ptr.as_ptr(),
                           NO as libc::c_int,
                           o_ptr.as_ptr(),
                           1 as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                ffi::device_synchronize()?;

                Ok(output_ptr)
            },
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
                    "Unable to get cuBLAS cublasSgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn batch_forward_linear<'a>(&self,bias:&CudaTensor1dPtr<f32,NO>,units:&CudaTensor2dPtr<f32,NI,NO>,input:SerializedVecView<'a,f32,Arr<f32,NI>>)
                            -> Result<SerializedVec<f32,Arr<f32,NO>>,TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NI * input.len() ,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NO * input.len(),&self.memory_pool)?;

        let bias = rayon::iter::repeat(bias.read_to_vec()?.into_boxed_slice().iter().cloned().collect::<Vec<f32>>())
                                                .take(input.len()).collect::<Vec<Vec<f32>>>()
                                                .into_iter().flatten().collect::<Vec<f32>>();

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI * input.len())?;
        output_ptr.memcpy(bias.as_slice().as_ptr(),NO * input.len())?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(1.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           input.len() as libc::c_int,
                           NI as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NI as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                Ok(output_ptr.read_to_vec()?.try_into()?)
            },
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
                    "Unable to get cuBLAS cublasSgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn batch_backward_linear<'a>(&self, units: &CudaTensor2dPtr<f32, NI, NO>, input: &'a SerializedVec<f32,Arr<f32, NO>>) -> Result<SerializedVec<f32, Arr<f32, NI>>, TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NO * input.len(),&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI * input.len(),&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NO * input.len())?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(0.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_T,
                           cublasOperation_t::CUBLAS_OP_N,
                           NI as libc::c_int,
                           input.len() as ::libc::c_int,
                           NO as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NO as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NI as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                Ok(output_ptr.read_to_vec()?.try_into()?)
            },
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
                    "Unable to get cuBLAS cublasSgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn batch_backward_weight_gradient<'a>(&self, o: SerializedVecView<'a,f32,Arr<f32, NI>>, loss: &'a SerializedVec<f32, Arr<f32, NO>>)
        -> Result<CudaTensor2dPtr<f32, NI, NO>, TrainingError> {
        let n = o.len();

        let mut o_ptr = CudaMemoryPoolPtr::new(NI * n,&self.memory_pool)?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO * n,&self.memory_pool)?;
        let mut output_ptr = CudaTensor2dPtr::<f32,NI,NO>::new(&self.memory_pool)?;

        o_ptr.memcpy(o.as_raw_slice().as_ptr(),NI * n)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),NO * n)?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(0.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_T,
                           NO as ::libc::c_int,
                           NI as libc::c_int,
                           n as ::libc::c_int,
                           alpha.as_ptr(),
                           loss_ptr.as_ptr(),
                           NO as libc::c_int,
                           o_ptr.as_ptr(),
                           NI as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                ffi::device_synchronize()?;

                Ok(output_ptr)
            },
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
                    "Unable to get cuBLAS cublasSgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn batch_linear_reduce<'a>(&self, loss: SerializedVecView<'a,f32,Arr<f32,NO>>) -> Result<CudaTensor1dPtr<f32,NO>,TrainingError> {
        self.reduce(loss)
    }
}
impl<const NI: usize, const NO: usize> DeviceLinear<f64,CudaTensor2dPtr<f64,NI,NO>,CudaTensor1dPtr<f64,NO>,NI,NO> for DeviceGpu<f64> {
    #[inline]
    fn forward_linear<'a>(&self, bias: &CudaTensor1dPtr<f64,NO>, units: &CudaTensor2dPtr<f64,NI,NO>, input: ArrView<'a,f64,NI>)
                      -> Result<Arr<f64, NO>, EvaluateError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI)?;
        bias.memcpy_to(&mut output_ptr,NO)?;

        let alpha = CudaPtr::try_from(1.0f64)?;
        let beta = CudaPtr::try_from(1.0f64)?;

        match unsafe {
            cublasDgemv_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           NI as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           1,
                           beta.as_ptr(),
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
                    "Unable to get cuBLAS cublasDgemv_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn backward_linear<'a>(&self, units: &CudaTensor2dPtr<f64,NI,NO>, input: &'a Arr<f64,NO>)
                       -> Result<Arr<f64, NI>, TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NO)?;

        let alpha = CudaPtr::try_from(1.0f64)?;
        let beta = CudaPtr::try_from(0.0f64)?;

        match unsafe {
            cublasDgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_T,
                           cublasOperation_t::CUBLAS_OP_N,
                           NI as ::libc::c_int,
                           1 as libc::c_int,
                           NO as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NO as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NI as ::libc::c_int
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
                    "Unable to get cuBLAS cublasDgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn backward_weight_gradient<'a>(&self, o: ArrView<'a,f64,NI>, loss: &'a Arr<f64,NO>) -> Result<CudaTensor2dPtr<f64,NI,NO>, TrainingError> {
        let mut o_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let mut output_ptr = CudaTensor2dPtr::<f64,NI,NO>::new(&self.memory_pool)?;

        o_ptr.memcpy(o.as_raw_slice().as_ptr(),NI)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),NO)?;

        let alpha = CudaPtr::try_from(1.0f64)?;
        let beta = CudaPtr::try_from(0.0f64)?;

        match unsafe {
            cublasDgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           NI as libc::c_int,
                           1 as ::libc::c_int,
                           alpha.as_ptr(),
                           loss_ptr.as_ptr(),
                           NO as libc::c_int,
                           o_ptr.as_ptr(),
                           1 as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                ffi::device_synchronize()?;

                Ok(output_ptr)
            },
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
                    "Unable to get cuBLAS cublasDgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn batch_forward_linear<'a>(&self,bias:&CudaTensor1dPtr<f64,NO>,units:&CudaTensor2dPtr<f64,NI,NO>,input:SerializedVecView<'a,f64,Arr<f64,NI>>)
                            -> Result<SerializedVec<f64,Arr<f64,NO>>,TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NI * input.len() ,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NO * input.len(),&self.memory_pool)?;

        let bias = rayon::iter::repeat(bias.read_to_vec()?.into_boxed_slice().iter().cloned().collect::<Vec<f64>>())
                                        .take(input.len()).collect::<Vec<Vec<f64>>>()
                                        .into_iter().flatten().collect::<Vec<f64>>();

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI * input.len())?;
        output_ptr.memcpy(bias.as_slice().as_ptr(),NO * input.len())?;

        let alpha = CudaPtr::try_from(1.0f64)?;
        let beta = CudaPtr::try_from(1.0f64)?;

        match unsafe {
            cublasDgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           input.len() as libc::c_int,
                           NI as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NI as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                Ok(output_ptr.read_to_vec()?.try_into()?)
            },
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
                    "Unable to get cuBLAS cublasSgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn batch_backward_linear<'a>(&self, units: &CudaTensor2dPtr<f64, NI, NO>, input: &'a SerializedVec<f64,Arr<f64, NO>>)
        -> Result<SerializedVec<f64, Arr<f64, NI>>, TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NO * input.len(),&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI * input.len(),&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NO * input.len())?;

        let alpha = CudaPtr::try_from(1.0f64)?;
        let beta = CudaPtr::try_from(0.0f64)?;

        match unsafe {
            cublasDgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_T,
                           cublasOperation_t::CUBLAS_OP_N,
                           NI as libc::c_int,
                           input.len() as ::libc::c_int,
                           NO as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NO as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NI as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                Ok(output_ptr.read_to_vec()?.try_into()?)
            },
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
                    "Unable to get cuBLAS cublasDgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn batch_backward_weight_gradient<'a>(&self, o: SerializedVecView<'a,f64,Arr<f64, NI>>, loss: &'a SerializedVec<f64,Arr<f64,NO>>)
        -> Result<CudaTensor2dPtr<f64, NI, NO>, TrainingError> {
        let n = o.len();

        let mut o_ptr = CudaMemoryPoolPtr::new(NI * n,&self.memory_pool)?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO * n,&self.memory_pool)?;
        let mut output_ptr = CudaTensor2dPtr::<f64,NI,NO>::new(&self.memory_pool)?;

        o_ptr.memcpy(o.as_raw_slice().as_ptr(),NI * n)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),NO * n)?;

        let alpha = CudaPtr::try_from(1.0f64)?;
        let beta = CudaPtr::try_from(0.0f64)?;

        match unsafe {
            cublasDgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_T,
                           NO as ::libc::c_int,
                           NI as libc::c_int,
                           n as ::libc::c_int,
                           alpha.as_ptr(),
                           loss_ptr.as_ptr(),
                           NO as libc::c_int,
                           o_ptr.as_ptr(),
                           NI as libc::c_int,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                ffi::device_synchronize()?;

                Ok(output_ptr)
            },
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
                    "Unable to get cuBLAS cublasDgemm_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    #[inline]
    fn batch_linear_reduce<'a>(&self, loss: SerializedVecView<'a,f64,Arr<f64,NO>>) -> Result<CudaTensor1dPtr<f64,NO>,TrainingError> {
        self.reduce(loss)
    }
}
/// Trait that defines the implementation of various computational processes in the differentially applicable linear layer
pub trait DeviceDiffLinear<'a,U,T,const NI: usize,const NO: usize>
    where U: UnitValue<U> {
    fn forward_diff_linear(&self,input:&'a DiffArr<U,NI>,units: &'a T,output: ArrView<'a,U,NO>) -> Result<Arr<U, NO>,EvaluateError>;
}
impl<'a,U,const NI:usize,const NO:usize> DeviceDiffLinear<'a,U,Arr2<U,NI,NO>,NI,NO> for DeviceCpu<U>
    where U: UnitValue<U> {
    fn forward_diff_linear(&self, input: &'a DiffArr<U, NI>, units: &'a Arr2<U, NI, NO>, output: ArrView<'a,U,NO>) -> Result<Arr<U, NO>,EvaluateError> {
        let mut output:Arr<U,NO> = output.into();

        for &(i,d) in input.iter() {
            for (o,j) in output.iter_mut().zip(0..NO) {
                *o += units[(i,j)] * d;
            }
        }
        Ok(output)
    }
}
impl<'a,U,const NI:usize,const NO:usize> DeviceDiffLinear<'a,U,CudaTensor2dPtr<U,NI,NO>,NI,NO> for DeviceGpu<U>
    where U: UnitValue<U> + DataTypeInfo,
          for<'b> DiffLinearForward<'b,U,NI,NO>: Kernel<Args=DiffLinearForwardArgs<'b,U,NI,NO>> {
    fn forward_diff_linear(&self, input: &'a DiffArr<U, NI>, units: &'a CudaTensor2dPtr<U, NI, NO>, output: ArrView<'a,U,NO>)
        -> Result<Arr<U, NO>,EvaluateError> {
        let len = input.len();

        let (indexes,input) = input.iter().fold((Vec::new(),Vec::new()), | mut acc, &(i,d)| {
            acc.0.push(i);
            acc.1.push(d);

            acc
        });

        let mut indexes_ptr = CudaMemoryPoolPtr::new(len,self.get_memory_pool())?;
        let mut input_ptr = CudaMemoryPoolPtr::new(len,self.get_memory_pool())?;

        indexes_ptr.memcpy(indexes.as_ptr(),len)?;
        input_ptr.memcpy(input.as_ptr(),len)?;

        let mut output_ptr = CudaTensor1dPtr::<U,NO>::new(self.get_memory_pool())?;

        output_ptr.memcpy(output.as_ptr(),NO)?;

        let mut args = DiffLinearForwardArgs::new(indexes_ptr,input_ptr,units,output_ptr,NO,len);

        let mut kernel = DiffLinearForward::new();

        kernel.launch(dim3 { x: NO as c_uint, y: 1, z: 1},
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<U>())?;

        Ok(args.output.read_to_vec()?.try_into()?)
    }
}
