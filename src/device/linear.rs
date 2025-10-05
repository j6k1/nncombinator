//! Implementation of the calculation process for full connected layers

use std::fmt::Debug;
use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};
use rcublas_sys::{cublasDgemm_v2, cublasOperation_t, cublasSgemm_v2, cublasStatus_t};
use crate::arr::{Arr, Arr2, ArrView, DiffArr, IntoConverter, SerializedVec, SerializedVecView};
use crate::cuda::{AsConstKernelPtr, AsCudaMutPtr, AsCudaPtr, AsCudaReadOnlyPtr, AsCudaView, AsKernelPtr, AsMutPtr, AsPtr, CudaMutPtr, CudaPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaTensor2dPtr, CudaVec, CudaVecView, CudaView, MemorySize, MemoryType};
use crate::cuda::{DataTypeInfo, Kernel, MemoryMoveTo, WriteMemory};
use crate::cuda::allocator::CudaAllocator;
use crate::cuda::kernel::device::{AddBias, AddBiasArgs, AddBiasBatch, AddBiasBatchArgs, DiffLinearForward, DiffLinearForwardArgs, ForwardLinear, ForwardLinearArgs, LinearGradient, LinearGradientArgs, ReduceLinearBatch, ReduceLinearBatchArgs};
use crate::device::{DeviceCpu, DeviceGpu, DeviceAllocator, DeviceReduce};
use crate::error::{EvaluateError, TrainingError, TypeConvertError};
use crate::layer::{BatchDataType, BatchSize, DiffInput};
use crate::ope::UnitValue;
use crate::ope::Product;

/// Trait that defines the implementation of various calculation processes in the linear layer
pub trait DeviceLinear<U,T,B,I,const NI: usize,const NO: usize>
    where U: UnitValue<U>,
          I: BatchDataType {
    type Output: BatchDataType + Debug + 'static;
    type BatchOutput: Debug + 'static;
    type LossOutput: BatchDataType + Debug + 'static;
    type BatchLossOutput: Debug + 'static;
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
    fn forward_linear<'a>(&self, bias:&B, units:&T, input:&'a I) -> Result<Self::Output, EvaluateError>;
    /// Error back propagation calculation
    /// # Arguments
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_linear<'a>(&self, units:&T, input:&'a Self::Output) -> Result<Self::LossOutput, TrainingError>;
    /// Calculate the gradient of the weights
    /// # Arguments
    /// * `o` - Input values from upper layers
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_weight_gradient<'a>(&self, o: &'a I, loss: &'a Self::Output) -> Result<T, TrainingError>;
    /// Calculate the gradient of the bias weights
    /// # Arguments
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_bias_weight_gradient<'a>(&self, loss: Self::Output) -> Result<B, TrainingError>;
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
    fn batch_forward_linear<'a>(&self,bias:&B,units:&T,input: &'a <I as BatchDataType>::Type)
        -> Result<Self::BatchOutput,TrainingError>;
    /// Error back propagation in batch
    /// # Arguments
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_linear<'a>(&self, units: &T, input: &'a Self::BatchOutput)
        -> Result<Self::BatchLossOutput, TrainingError>;
    /// Calculate the gradient of the weights in batch
    /// # Arguments
    /// * `o` - Input values from upper layers
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_weight_gradient<'a>(&self, o: &'a <I as BatchDataType>::Type, loss: &'a Self::BatchOutput)
        -> Result<T, TrainingError>;
    /// convolutional calculation
    /// # Arguments
    /// * `loss` - loss
    fn batch_linear_reduce<'a>(&self, loss: &'a Self::BatchOutput) -> Result<B,TrainingError>;
}
impl<U,I,const NI: usize,const NO: usize> DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,I,NI,NO> for DeviceCpu<U>
    where U: UnitValue<U>,
          I: BatchDataType + From<Arr<U,NI>> + Debug + 'static,
          <I as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: TryFrom<<SerializedVec<U,Arr<U,NI>> as IntoConverter>::Converter,Error=TypeConvertError>,
          SerializedVec<U,Arr<U,NI>>: IntoConverter,
          for<'a> ArrView<'a,U,NI>: From<&'a I>,
          for<'a> SerializedVecView<'a,U,Arr<U,NI>>: TryFrom<&'a <I as BatchDataType>::Type,Error=TypeConvertError>,
          Self: DeviceReduce<SerializedVec<U,Arr<U,NO>>,Arr<U,NO>,U,NO> {
    type Output = Arr<U,NO>;
    type BatchOutput = <Arr<U,NO> as BatchDataType>::Type;
    type LossOutput = I;
    type BatchLossOutput = <I as BatchDataType>::Type;
    #[inline]
    fn forward_linear<'a>(&self, bias: &Arr<U, NO>, units: &Arr2<U, NI, NO>, input: &'a I) -> Result<Arr<U, NO>, EvaluateError> {
        Ok(ArrView::<'a,U,NI>::from(input).product(units) + bias)
    }

    #[inline]
    fn backward_linear<'a>(&self, units: &Arr2<U,NI,NO>, input: &'a Arr<U,NO>) -> Result<I, TrainingError> {
        Ok(Arr::<U,NI>::try_from(units.iter().map(|u| {
            u.iter().zip(input.iter())
                .map(|(&w,&l)| w * l).fold(U::default(), |acc,g|{
                acc + g
            })
        }).collect::<Vec<U>>()).map_err(|e| TrainingError::from(e))?.into())
    }

    #[inline]
    fn backward_weight_gradient<'a>(&self, o: &'a I, loss: &'a Arr<U,NO>) -> Result<Arr2<U,NI,NO>, TrainingError> {
        Ok(ArrView::<'a,U,NI>::from(o).iter().cloned().map(|o| {
            loss.iter().cloned().map(|l| o * l).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,NO>>,_>>()?.try_into().map_err(|e| TrainingError::from(e))?)
    }

    fn backward_bias_weight_gradient<'a>(&self, loss: Self::Output) -> Result<Arr<U,NO>, TrainingError> {
        Ok(loss.into())
    }
    #[inline]
    fn batch_backward_linear<'a>(&self, units: &Arr2<U,NI,NO>, input: &'a SerializedVec<U,Arr<U,NO>>)
                             -> Result<<I as BatchDataType>::Type, TrainingError> {
        Ok(SerializedVec::<U,Arr<U,NI>>::from(input.par_iter().map(|l| {
            units.iter().map(|u| {
                u.iter().zip(l.iter())
                    .map(|(&w,&l)| w * l).fold(U::default(), |acc,g|{
                    acc + g
                })
            }).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,NI>>,_>>()?).into_converter().try_into()?)
    }

    #[inline]
    fn batch_forward_linear<'a>(&self,bias: &Arr<U,NO>, units: &Arr2<U,NI,NO>, input: &'a <I as BatchDataType>::Type)
                            -> Result<SerializedVec<U,Arr<U,NO>>,TrainingError> {
        Ok(SerializedVecView::<'a,U,Arr<U,NI>>::try_from(input)?.par_iter().map(|input| {
            input.product(units) + bias
        }).collect::<Vec<Arr<U,NO>>>().into())
    }

    #[inline]
    fn batch_backward_weight_gradient<'a>(&self, o: &'a <I as BatchDataType>::Type, loss: &'a SerializedVec<U,Arr<U,NO>>)
                                      -> Result<Arr2<U,NI,NO>, TrainingError> {
        Ok(SerializedVecView::<'a,U,Arr<U,NI>>::try_from(o)?.par_iter().zip(loss.par_iter()).map(|(o,l)| {
            o.iter().cloned().map(|o| {
                l.iter().cloned().map(|l| o * l).collect::<Vec<U>>().try_into()
            }).collect::<Result<Vec<Arr<U,NO>>,_>>()?.try_into()
        }).reduce(|| Ok(Arr2::new()), | acc, g | {
            acc.and_then(| mut acc | g.and_then(|g| {
                for (mut acc,g) in acc.iter_mut().zip(g.iter()) {
                    for (acc,&g) in acc.iter_mut().zip(g.iter()) {
                        *acc += g;
                    }
                }

                Ok(acc)
            }))
        })?)
    }

    #[inline]
    fn batch_linear_reduce<'a>(&self, loss: &'a SerializedVec<U,Arr<U,NO>>) -> Result<Arr<U,NO>,TrainingError> {
        self.reduce(loss)
    }
}
impl<I,A,const NI: usize, const NO: usize> DeviceLinear<f32,CudaTensor2dPtr<f32,A,NI,NO>,CudaTensor1dPtr<f32,A,NO>,I,NI,NO> for DeviceGpu<f32,A>
    where I: BatchDataType + MemorySize + AsConstKernelPtr + AsKernelPtr + From<CudaTensor1dPtr<f32,A,NI>> + Debug + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize + IntoConverter + 'static,
          <I as BatchDataType>::Type: TryFrom<<CudaVec<f32,CudaTensor1dPtr<f32,A,NI>,A> as IntoConverter>::Converter,Error=TypeConvertError>,
          A: CudaAllocator + MemoryType + 'static,
          CudaPtr<f32,A>: AsPtr<f32> + WriteMemory<f32>,
          CudaVec<f32,CudaTensor1dPtr<f32,A,NI>,A>: IntoConverter,
          CudaTensor1dPtr<f32,A,NO>: AsConstKernelPtr + AsKernelPtr + MemorySize + MemoryMoveTo<f32,CudaTensor1dPtr<f32,A,NO>>,
          Self: DeviceReduce<CudaVec<f32,CudaTensor1dPtr<f32,A,NO>,A>,CudaTensor1dPtr<f32,A,NO>,f32,NO>,
          for<'a> I: CudaView<'a>,
          for<'a> <I as BatchDataType>::Type: CudaView<'a>,
          for<'a> CudaTensor1dPtr<f32,A,NI>: AsCudaPtr<'a> + AsCudaMutPtr<Pointee=f32,Allocator=A> + AsCudaReadOnlyPtr<Pointee=f32>,
          for<'a> CudaMutPtr<'a,f32,A>: AsMutPtr<f32>,
          for<'a> CudaTensor1dPtr<f32,A,NO>: CudaView<'a>,
          for<'a> &'a I: AsCudaView<'a>,
          for<'a> &'a <I as BatchDataType>::Type: AsCudaView<'a>,
          for<'a> &'a CudaTensor1dPtr<f32,A,NO>: AsCudaView<'a>,
          for<'a> CudaTensor1dPtrView<'a,f32,NI>: MemorySize + AsConstKernelPtr + From<<I as CudaView<'a>>::Type> + From<<&'a I as CudaView<'a>>::Type>,
          for<'a> CudaTensor1dPtrView<'a,f32,NO>: MemorySize + AsConstKernelPtr +
                                                  From<<CudaTensor1dPtr<f32,A,NO> as CudaView<'a>>::Type> +
                                                  From<<&'a CudaTensor1dPtr<f32,A,NO> as CudaView<'a>>::Type>,
          for<'a> <&'a <I as BatchDataType>::Type as CudaView<'a>>::Type: IntoConverter,
          for<'a> CudaVecView<'a,f32,CudaTensor1dPtrView<'a,f32,NI>>: TryFrom<<<&'a <I as BatchDataType>::Type as CudaView<'a>>::Type as IntoConverter>::Converter,Error=TypeConvertError>,
          for<'a> CudaVecView<'a,f32,CudaTensor1dPtrView<'a,f32,NO>>: TryFrom<&'a CudaVec<f32,CudaTensor1dPtr<f32,A,NO>,A>,Error=TypeConvertError>,
          for<'a> AddBias<'a,f32,A,NO>: Kernel<Args=AddBiasArgs<'a,f32,A,NO>>,
          for<'a> AddBiasBatch<'a,f32,A,NO>: Kernel<Args=AddBiasBatchArgs<'a,f32,A,NO>>,
          for<'a> ReduceLinearBatch::<'a,f32,A,NO>: Kernel<Args=ReduceLinearBatchArgs<'a,f32,A,NO>> {
    type Output = CudaTensor1dPtr<f32,A,NO>;
    type BatchOutput = CudaVec<f32,CudaTensor1dPtr<f32,A,NO>,A>;
    type LossOutput = I;
    type BatchLossOutput = <I as BatchDataType>::Type;
    #[inline]
    fn forward_linear<'a>(&self, bias: &CudaTensor1dPtr<f32,A,NO>, units: &CudaTensor2dPtr<f32,A,NI,NO>, input: &'a I)
                          -> Result<CudaTensor1dPtr<f32,A,NO>, EvaluateError> {
        let input = CudaTensor1dPtrView::<f32,NI>::from(input.as_cuda_view());
        let mut output = CudaTensor1dPtr::<f32,A,NO>::new(self.get_allocator())?;
        bias.memcpy_to(&mut output,NO)?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(1.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           1,
                           NI as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input.as_ptr(),
                           NI as libc::c_int,
                           beta.as_ptr(),
                           output.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output),
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
    fn backward_linear<'a>(&self, units: &CudaTensor2dPtr<f32,A,NI,NO>, input: &'a Self::Output)
                           -> Result<I, TrainingError> {
        let input = input.as_cuda_view();
        let input = CudaTensor1dPtrView::<f32,NO>::from(input);
        let mut output = CudaTensor1dPtr::<f32,A,NI>::new(self.get_allocator())?;

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
                           input.as_ptr(),
                           NO as libc::c_int,
                           beta.as_ptr(),
                           output.as_mut_ptr(),
                           NI as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output.into()),
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
    fn backward_weight_gradient<'a>(&self, o: &'a I, loss: &'a Self::Output) -> Result<CudaTensor2dPtr<f32,A,NI,NO>, TrainingError> {
        let o_ptr = CudaTensor1dPtrView::<f32,NI>::from(o.as_cuda_view());
        let loss = loss.as_cuda_view();
        let loss_ptr = CudaTensor1dPtrView::<f32,NO>::from(loss);
        let mut output_ptr = CudaTensor2dPtr::<f32,A,NI,NO>::new(self.get_allocator())?;

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

    fn backward_bias_weight_gradient<'a>(&self, loss: Self::Output) -> Result<CudaTensor1dPtr<f32,A,NO>, TrainingError> {
        Ok(loss.into())
    }
    #[inline]
    fn batch_forward_linear<'a>(&self,bias:&CudaTensor1dPtr<f32,A,NO>,units:&CudaTensor2dPtr<f32,A,NI,NO>,
                                input: &'a <I as BatchDataType>::Type)
                                -> Result<Self::BatchOutput,TrainingError> {
        let size = input.size();
        let converter = input.as_cuda_view().into_converter();
        let input_ptr = CudaVecView::<f32,CudaTensor1dPtrView<f32,NI>>::try_from(converter)?;
        let mut output = CudaVec::<f32,CudaTensor1dPtr<f32,A,NO>,A>::new(size,self.get_allocator())?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(0.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           size as libc::c_int,
                           NI as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NI as libc::c_int,
                           beta.as_ptr(),
                           output.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => (),
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

        let mut args = AddBiasBatchArgs::new(
            bias,
            output,
            size
        );

        let mut kernel = AddBiasBatch::<'_,f32,A,NO>::new();

        kernel.launch(&mut args)?;

        Ok(args.input_output)

    }

    #[inline]
    fn batch_backward_linear<'a>(&self, units: &CudaTensor2dPtr<f32,A, NI, NO>, input: &'a Self::BatchOutput)
                                 -> Result<<I as BatchDataType>::Type, TrainingError> {
        let n = input.size();

        let input = CudaVecView::<f32,CudaTensor1dPtrView<f32,NO>>::try_from(input)?;
        let mut output = CudaVec::<f32,CudaTensor1dPtr<f32,A,NI>,A>::new(n,self.get_allocator())?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(0.0f32)?;

        match unsafe {
            cublasSgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_T,
                           cublasOperation_t::CUBLAS_OP_N,
                           NI as libc::c_int,
                           input.size() as ::libc::c_int,
                           NO as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input.as_ptr(),
                           NO as libc::c_int,
                           beta.as_ptr(),
                           output.as_mut_ptr(),
                           NI as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                Ok(output.into_converter().try_into()?)
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
    fn batch_backward_weight_gradient<'a>(&self, o: &'a <I as BatchDataType>::Type,
                                          loss: &'a Self::BatchOutput)
                                          -> Result<CudaTensor2dPtr<f32,A, NI, NO>, TrainingError> {
        let n = loss.size();

        let converter = o.as_cuda_view().into_converter();
        let o_ptr = CudaVecView::<f32,CudaTensor1dPtrView<f32,NI>>::try_from(converter)?;
        let loss_ptr = CudaVecView::<f32,CudaTensor1dPtrView<f32,NO>>::try_from(loss)?;
        let mut output_ptr = CudaTensor2dPtr::<f32,A,NI,NO>::new(self.get_allocator())?;

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
    fn batch_linear_reduce<'a>(&self, loss: &'a Self::BatchOutput) -> Result<CudaTensor1dPtr<f32,A,NO>,TrainingError> {
        self.reduce(loss)
    }
}

impl<I,A,const NI: usize, const NO: usize> DeviceLinear<f64,CudaTensor2dPtr<f64,A,NI,NO>,CudaTensor1dPtr<f64,A,NO>,I,NI,NO> for DeviceGpu<f64,A>
    where I: BatchDataType + From<CudaTensor1dPtr<f64,A,NI>> + Debug + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'static,
          <I as BatchDataType>::Type: TryFrom<<CudaVec<f64,CudaTensor1dPtr<f64,A,NI>,A> as IntoConverter>::Converter,Error=TypeConvertError>,
          A: CudaAllocator + 'static,
          CudaTensor1dPtr<f64,A,NO>: MemoryMoveTo<f64,CudaTensor1dPtr<f64,A,NO>>,
          CudaVec<f64,CudaTensor1dPtr<f64,A,NI>,A>: IntoConverter,
          Self: DeviceReduce<CudaVec<f64,CudaTensor1dPtr<f64,A,NO>,A>,CudaTensor1dPtr<f64,A,NO>,f64,NO>,
          for<'a> CudaTensor1dPtrView<'a,f64,NI>: From<&'a I>,
          for<'a> CudaVecView<'a,f64,CudaTensor1dPtrView<'a,f64,NO>>: TryFrom<&'a CudaVec<f64,CudaTensor1dPtr<f64,A,NO>,A>,Error=TypeConvertError>,
          for<'a> CudaVecView<'a,f64,CudaTensor1dPtrView<'a,f64,NI>>: TryFrom<&'a <I as BatchDataType>::Type,Error=TypeConvertError>,
          for<'a> AddBias<'a,f64,A,NO>: Kernel<Args=AddBiasArgs<'a,f64,A,NO>>,
          for<'a> AddBiasBatch<'a,f64,A,NO>: Kernel<Args=AddBiasBatchArgs<'a,f64,A,NO>>,
          for<'b> ReduceLinearBatch::<'b,f64,A,NO>: Kernel<Args=ReduceLinearBatchArgs<'b,f64,A,NO>> {
    type Output = CudaTensor1dPtr<f64,A,NO>;
    type BatchOutput = CudaVec<f64,CudaTensor1dPtr<f64,A,NO>,A>;
    type LossOutput = I;
    type BatchLossOutput = <I as BatchDataType>::Type;
    #[inline]
    fn forward_linear<'a>(&self, bias: &CudaTensor1dPtr<f64,A,NO>, units: &CudaTensor2dPtr<f64,A,NI,NO>, input: &'a I)
                          -> Result<CudaTensor1dPtr<f64,A,NO>, EvaluateError> {
        let input = CudaTensor1dPtrView::<f64,NI>::from(input);
        let mut output = CudaTensor1dPtr::<f64,A,NO>::new(self.get_allocator())?;
        bias.memcpy_to(&mut output,NO)?;

        let alpha = CudaPtr::try_from(1.0f64)?;
        let beta = CudaPtr::try_from(1.0f64)?;

        match unsafe {
            cublasDgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           1,
                           NI as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input.as_ptr(),
                           NI as libc::c_int,
                           beta.as_ptr(),
                           output.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output),
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
    fn backward_linear<'a>(&self, units: &CudaTensor2dPtr<f64,A,NI,NO>, input: &'a Self::Output)
                           -> Result<I, TrainingError> {
        let input = CudaTensor1dPtrView::<f64,NO>::from(input);
        let mut output = CudaTensor1dPtr::<f64,A,NI>::new(self.get_allocator())?;

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
                           input.as_ptr(),
                           NO as libc::c_int,
                           beta.as_ptr(),
                           output.as_mut_ptr(),
                           NI as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(output.into()),
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
    fn backward_weight_gradient<'a>(&self, o: &'a I, loss: &'a Self::Output) -> Result<CudaTensor2dPtr<f64,A,NI,NO>, TrainingError> {
        let o_ptr = CudaTensor1dPtrView::<f64,NI>::from(o);
        let loss_ptr = CudaTensor1dPtrView::<f64,NO>::from(loss);
        let mut output_ptr = CudaTensor2dPtr::<f64,A,NI,NO>::new(self.get_allocator())?;

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

    fn backward_bias_weight_gradient<'a>(&self, loss: Self::Output) -> Result<CudaTensor1dPtr<f64,A,NO>, TrainingError> {
        Ok(loss.into())
    }
    #[inline]
    fn batch_forward_linear<'a>(&self,bias:&CudaTensor1dPtr<f64,A,NO>,units:&CudaTensor2dPtr<f64,A,NI,NO>,
                                input: &'a <I as BatchDataType>::Type)
                                -> Result<Self::BatchOutput,TrainingError> {
        let input_ptr = CudaVecView::<f64,CudaTensor1dPtrView<f64,NI>>::try_from(input)?;
        let mut output = CudaVec::<f64,CudaTensor1dPtr<f64,A,NO>,A>::new(input.size(),self.get_allocator())?;

        let alpha = CudaPtr::try_from(1.0f64)?;
        let beta = CudaPtr::try_from(0.0f64)?;

        match unsafe {
            cublasDgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_N,
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           input.size() as libc::c_int,
                           NI as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input_ptr.as_ptr(),
                           NI as libc::c_int,
                           beta.as_ptr(),
                           output.as_mut_ptr(),
                           NO as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => (),
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

        let mut args = AddBiasBatchArgs::new(
            bias,
            output,
            input.size()
        );

        let mut kernel = AddBiasBatch::<'_,f64,A,NO>::new();

        kernel.launch(&mut args)?;

        Ok(args.input_output)

    }

    #[inline]
    fn batch_backward_linear<'a>(&self, units: &CudaTensor2dPtr<f64, A, NI, NO>, input: &'a Self::BatchOutput)
                                 -> Result<<I as BatchDataType>::Type, TrainingError> {
        let n = input.size();

        let input = CudaVecView::<f64,CudaTensor1dPtrView<f64,NO>>::try_from(input)?;
        let mut output = CudaVec::<f64,CudaTensor1dPtr<f64,A,NI>,A>::new(n,self.get_allocator())?;

        let alpha = CudaPtr::try_from(1.0f64)?;
        let beta = CudaPtr::try_from(0.0f64)?;

        match unsafe {
            cublasDgemm_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_T,
                           cublasOperation_t::CUBLAS_OP_N,
                           NI as libc::c_int,
                           input.size() as ::libc::c_int,
                           NO as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NO as libc::c_int,
                           input.as_ptr(),
                           NO as libc::c_int,
                           beta.as_ptr(),
                           output.as_mut_ptr(),
                           NI as ::libc::c_int
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => {
                Ok(output.into_converter().try_into()?)
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
    fn batch_backward_weight_gradient<'a>(&self, o: &'a <I as BatchDataType>::Type,
                                          loss: &'a Self::BatchOutput)
                                          -> Result<CudaTensor2dPtr<f64, A, NI, NO>, TrainingError> {
        let n = loss.size();

        let o_ptr = CudaVecView::<f64,CudaTensor1dPtrView<f64,NI>>::try_from(o)?;
        let loss_ptr = CudaVecView::<f64,CudaTensor1dPtrView<f64,NO>>::try_from(loss)?;
        let mut output_ptr = CudaTensor2dPtr::<f64,A,NI,NO>::new(self.get_allocator())?;

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
    fn batch_linear_reduce<'a>(&self, loss: &'a Self::BatchOutput) -> Result<CudaTensor1dPtr<f64,A,NO>,TrainingError> {
        self.reduce(loss)
    }
}

/// Trait that defines the implementation of various computational processes in the differentially applicable linear layer
pub trait DeviceDiffLinear<'a,U,I,T,const NI: usize,const NO: usize>
    where U: UnitValue<U> {
    type Output: Debug + 'static;
    fn forward_diff_linear(&self, units: &T, input: I) -> Result<Self::Output, EvaluateError>;
    fn clone_diff_linear_forward_output(&self, output: &Self::Output) -> Result<Self::Output, EvaluateError>;
}
impl<'a,U,const NI:usize,const NO:usize> DeviceDiffLinear<'a,U,DiffInput<'a,DiffArr<U,NI>,Arr<U,NO>>,Arr2<U,NI,NO>,NI,NO> for DeviceCpu<U>
    where U: UnitValue<U> {
    type Output = Arr<U,NO>;
    #[inline]
    fn forward_diff_linear(&self, units: &Arr2<U, NI, NO>, input: DiffInput<DiffArr<U,NI>,Arr<U,NO>>) -> Result<Arr<U, NO>,EvaluateError> {
        let mut output = input.output.clone();

        for &(i,d) in input.diff.iter() {
            for (o,j) in output.iter_mut().zip(0..NO) {
                *o += units[(i,j)] * d;
            }
        }
        Ok(output)
    }

    fn clone_diff_linear_forward_output(&self, output: &Self::Output) -> Result<Self::Output, EvaluateError> {
        Ok(output.clone())
    }
}
impl<'a,U,A,const NI:usize,const NO:usize> DeviceDiffLinear<'a,U,DiffInput<'a,DiffArr<U,NI>,CudaTensor1dPtr<U,A,NO>>,CudaTensor2dPtr<U,A,NI,NO>,NI,NO> for DeviceGpu<U,A>
    where U: UnitValue<U> + DataTypeInfo,
          A: CudaAllocator + 'static,
          CudaPtr<U,A>: WriteMemory<U>,
          CudaPtr<usize,A>: WriteMemory<usize>,
          CudaTensor1dPtr<U,A,NI>: WriteMemory<U>,
          CudaTensor1dPtr<U,A,NO>: WriteMemory<U> + MemoryMoveTo<U,CudaTensor1dPtr<U,A,NO>>,
          for<'b> CudaTensor1dPtrView<'b,U,NI>: From<&'b CudaTensor1dPtr<U,A,NI>>,
          for<'b> ForwardLinear::<'b,U,A,NI,NO>: Kernel<Args=ForwardLinearArgs<'b,U,A,NI,NO>>,
          for<'b> LinearGradient::<'b,U,A,NI,NO>: Kernel<Args=LinearGradientArgs<'b,U,A,NI,NO>>,
          for<'b> ReduceLinearBatch::<'b,U,A,NO>: Kernel<Args=ReduceLinearBatchArgs<'b,U,A,NO>>,
          for<'b> DiffLinearForward<'b,U,A,NI,NO>: Kernel<Args=DiffLinearForwardArgs<'b,U,A,NI,NO>> {
    type Output = CudaTensor1dPtr<U,A,NO>;

    #[inline]
    fn forward_diff_linear(&self, units: &CudaTensor2dPtr<U,A,NI,NO>, input: DiffInput<'a,DiffArr<U,NI>,CudaTensor1dPtr<U,A,NO>>)
        -> Result<CudaTensor1dPtr<U,A,NO>,EvaluateError> {
        let len = input.diff.len();
        let output = input.output;

        let (indexes, input) = input.diff.iter().fold((Vec::new(), Vec::new()), |mut acc, &(i, d)| {
            acc.0.push(i);
            acc.1.push(d);

            acc
        });

        let mut indexes_ptr = CudaPtr::new(len, self.get_allocator())?;
        let mut input_ptr = CudaPtr::new(len, self.get_allocator())?;

        indexes_ptr.memcpy(indexes.as_ptr(), len)?;
        input_ptr.memcpy(input.as_ptr(), len)?;

        let mut output_ptr = CudaTensor1dPtr::<U,A,NO>::new(self.get_allocator())?;

        output.memcpy_to(&mut output_ptr,NO)?;

        let mut args = DiffLinearForwardArgs::new(indexes_ptr, input_ptr, units, output_ptr, NO, len);

        let mut kernel = DiffLinearForward::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn clone_diff_linear_forward_output(&self, output: &Self::Output) -> Result<Self::Output, EvaluateError> {
        let mut o = CudaTensor1dPtr::<U,A,NO>::new(self.get_allocator())?;

        output.memcpy_to(&mut o, NO)?;

        Ok(o)
    }
}
