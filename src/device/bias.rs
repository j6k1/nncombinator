//! Implementation of the calculation process for bias layers

use std::fmt::Debug;
use std::ops::Add;
#[cfg(feature = "cuda")]
use libc::c_int;
#[cfg(feature = "cuda")]
use rcublas_sys::{cublasDaxpy_v2, cublasSaxpy_v2, cublasStatus_t};
use crate::arr::{Arr, ArrView, IntoConverter, SerializedVec, SerializedVecView};
use crate::collection::Broadcast;
use crate::device::{DeviceCpu, DeviceReduce};
use crate::error::{EvaluateError, GeneralizationError, SpecializationError, TrainingError, TypeConvertError};
use crate::layer::{BatchDataType, BatchSize};
use crate::mem::AsRawSlice;
#[cfg(feature = "cuda")]
use crate::cuda::{AsMutPtr, AsPtr, CudaPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, ReadMemory, WriteMemory, MemoryMoveTo, AsCudaMutPtr, CudaMutPtr, AsCudaPtr, Kernel};
#[cfg(feature = "cuda")]
use crate::cuda::allocator::CudaAllocator;
#[cfg(feature = "cuda")]
use crate::cuda::kernel::device::{AddBiasBatch, AddBiasBatchArgs};
#[cfg(feature = "cuda")]
use crate::device::{DeviceGpu, DeviceAllocator};

/// Trait that defines the implementation of various calculation processes in the bias layer
pub trait DeviceBias<U,T,IO,const N: usize>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          IO: BatchDataType + Debug,
          <IO as BatchDataType>::Type: BatchSize + Debug {
    /// Perform generalization of bias data
    /// # Arguments
    /// * `bias` - Set of biases applied to the output of the bias layer
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`GeneralizationError`]
    fn generalization_bias(&self,bias:&T) -> Result<Arr<U,N>, GeneralizationError>;
    /// Perform specialization of bias data
    /// # Arguments
    /// * `bias` - Set of biases applied to the output of the bias layer
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`SpecializationError`]
    fn specialization_bias(&self,bias:Arr<U,N>) -> Result<T, SpecializationError>;
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
impl<U,IO,const N:usize> DeviceBias<U,Arr<U,N>,IO,N> for DeviceCpu
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          IO: BatchDataType + Debug + Clone,
          <IO as BatchDataType>::Type: BatchSize + Debug,
          IO: From<Arr<U,N>>,
          Arr<U,N>: From<IO>,
          SerializedVec<U,Arr<U,N>>: IntoConverter,
          <IO as BatchDataType>::Type: TryFrom<<SerializedVec<U,Arr<U,N>> as IntoConverter>::Converter,Error=TypeConvertError>,
          for<'a> ArrView<'a,U,N>: From<&'a IO> + Add<&'a Arr<U,N>,Output=Arr<U,N>>,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a <IO as BatchDataType>::Type,Error=TypeConvertError>,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: Add<Broadcast<Arr<U,N>>,Output=SerializedVec<U,Arr<U,N>>>,
          Self: DeviceReduce<<IO as BatchDataType>::Type,Arr<U,N>,U,N> {
    #[inline]
    fn generalization_bias(&self, bias:&Arr<U,N>) -> Result<Arr<U,N>, GeneralizationError> {
        Ok(bias.clone())
    }
    #[inline]
    fn specialization_bias(&self, bias: Arr<U,N>) -> Result<Arr<U,N>, SpecializationError> {
        Ok(bias)
    }
    #[inline]
    fn forward_bias<'a>(&self, bias: &Arr<U,N>, input: &'a IO) -> Result<IO, EvaluateError> {
        Ok((ArrView::<'a,U,N>::from(input) + bias).into())
    }

    #[inline]
    fn backward_bias<'a>(&self, input: IO) -> Result<IO, TrainingError> {
        Ok(input)
    }

    #[inline]
    fn backward_bias_weight_gradient<'a>(&self, loss: &'a IO) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().into())
    }

    #[inline]
    fn batch_forward_bias<'a>(&self, bias: &Arr<U,N>, input: &'a <IO as BatchDataType>::Type) -> Result<<IO as BatchDataType>::Type, TrainingError> {
        Ok((SerializedVecView::<'a,U,Arr<U,N>>::try_from(input)? + Broadcast(bias.clone())).into_converter().try_into()?)
    }

    #[inline]
    fn batch_backward_bias<'a>(&self, input: <IO as BatchDataType>::Type) -> Result<<IO as BatchDataType>::Type, TrainingError> {
        Ok(input)
    }

    #[inline]
    fn batch_backward_bias_weight_gradient<'a>(&self, loss: &'a <IO as BatchDataType>::Type) -> Result<Arr<U,N>, TrainingError> {
        self.reduce(loss)
    }
}
#[cfg(feature = "cuda")]
impl<IO,A,const N:usize> DeviceBias<f32,CudaTensor1dPtr<f32,A,N>,IO,N> for DeviceGpu<A>
    where IO: BatchDataType + Debug,
          <IO as BatchDataType>::Type: BatchSize + Debug,
          IO: From<CudaTensor1dPtr<f32,A,N>> + AsCudaMutPtr<Pointee=f32,Allocator=A>,
          A: CudaAllocator,
          CudaTensor1dPtr<f32,A,N>: ReadMemory<f32> +
                                    AsCudaMutPtr<Pointee=f32,Allocator=A> +
                                    AsMutPtr<f32> +
                                    MemoryMoveTo<f32,CudaTensor1dPtr<f32,A,N>>,
          CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A>: IntoConverter,
          <IO as BatchDataType>::Type: TryFrom<<CudaVec<f32,CudaTensor1dPtr<f32,A,N>,A> as IntoConverter>::Converter,Error=TypeConvertError>,
          for<'a> IO: AsCudaPtr<'a>,
          for<'a> <IO as AsCudaPtr<'a>>::Pointer: AsPtr<f32> + MemoryMoveTo<f32,CudaMutPtr<'a,f32,A>>,
          for<'a> <IO as BatchDataType>::Type: AsCudaPtr<'a>,
          for<'a> <<IO as BatchDataType>::Type as AsCudaPtr<'a>>::Pointer: AsPtr<f32> + MemoryMoveTo<f32,CudaMutPtr<'a,f32,A>>,
          for<'a> CudaMutPtr<'a,f32,A>: WriteMemory<f32> + AsMutPtr<f32>,
          for<'a> CudaTensor1dPtrView<'a,f32,N>: From<&'a IO>,
          for<'a> CudaVecView<'a,f32,CudaTensor1dPtrView<'a,f32,N>>: TryFrom<&'a <IO as BatchDataType>::Type,Error=TypeConvertError>,
          for<'a> AddBiasBatch<'a,f32,A,N>: Kernel<Args=AddBiasBatchArgs<'a,f32,A,N>>,
          Self: DeviceReduce<<IO as BatchDataType>::Type,CudaTensor1dPtr<f32,A,N>,f32,N> {
    #[inline]
    fn generalization_bias(&self, bias: &CudaTensor1dPtr<f32,A,N>) -> Result<Arr<f32,N>, GeneralizationError> {
        Ok(bias.read_to_vec()?.try_into()?)
    }
    #[inline]
    fn specialization_bias(&self, bias: Arr<f32,N>) -> Result<CudaTensor1dPtr<f32,A,N>, SpecializationError> {
        let mut b = CudaTensor1dPtr::new(self.get_allocator())?;

        b.memcpy(bias.as_raw_slice().as_ptr(),N)?;

        Ok(b)
    }
    #[inline]
    fn forward_bias<'a>(&self, bias: &CudaTensor1dPtr<f32,A,N>, input: &'a IO) -> Result<IO, EvaluateError> {
        let input_ptr = CudaTensor1dPtrView::<'a,f32,N>::from(input);
        let mut output_ptr = CudaTensor1dPtr::<f32,A,N>::new(self.get_allocator())?;

        bias.memcpy_to(&mut output_ptr,N)?;

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

    #[inline]
    fn backward_bias<'a>(&self, input: IO) -> Result<IO, TrainingError> {
        Ok(input)
    }

    #[inline]
    fn backward_bias_weight_gradient<'a>(&self, loss: &'a IO) -> Result<CudaTensor1dPtr<f32,A,N>, TrainingError> {
        let mut p = CudaTensor1dPtr::<f32,A,N>::new(self.get_allocator())?;

        loss.as_cuda_ptr().memcpy_to(&mut p.as_cuda_mut_ptr(),N)?;

        Ok(p)
    }
    #[inline]
    fn batch_forward_bias<'a>(&self, bias: &CudaTensor1dPtr<f32,A,N>, input: &'a <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, TrainingError> {
        let len = input.size();

        let mut input_output = CudaVec::<f32,CudaTensor1dPtr::<f32,A,N>,A>::new(len,self.get_allocator())?;
        input.as_cuda_ptr().memcpy_to(&mut input_output.as_cuda_mut_ptr(),N * len)?;

        let mut args = AddBiasBatchArgs::new(
            bias,
            input_output,
            len
        );

        let mut kernel = AddBiasBatch::<'_,f32,A,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.input_output.into_converter().try_into()?)
    }

    #[inline]
    fn batch_backward_bias<'a>(&self, input: <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, TrainingError> {
        Ok(input)
    }

    #[inline]
    fn batch_backward_bias_weight_gradient<'a>(&self, loss: &'a <IO as BatchDataType>::Type)
        -> Result<CudaTensor1dPtr<f32,A,N>, TrainingError> {
        self.reduce(loss)
    }
}
#[cfg(feature = "cuda")]
impl<IO,A,const N:usize> DeviceBias<f64,CudaTensor1dPtr<f64,A,N>,IO,N> for DeviceGpu<A>
    where IO: BatchDataType + Debug,
          <IO as BatchDataType>::Type: BatchSize + Debug,
          IO: From<CudaTensor1dPtr<f64,A,N>> + AsCudaMutPtr<Pointee=f64,Allocator=A>,
          A: CudaAllocator,
          CudaTensor1dPtr<f64,A,N>: ReadMemory<f64> +
          AsCudaMutPtr<Pointee=f64,Allocator=A> +
          AsMutPtr<f64> +
          MemoryMoveTo<f64,CudaTensor1dPtr<f64,A,N>>,
          CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A>: IntoConverter,
          <IO as BatchDataType>::Type: TryFrom<<CudaVec<f64,CudaTensor1dPtr<f64,A,N>,A> as IntoConverter>::Converter,Error=TypeConvertError>,
          for<'a> IO: AsCudaPtr<'a>,
          for<'a> <IO as AsCudaPtr<'a>>::Pointer: AsPtr<f64> + MemoryMoveTo<f64,CudaMutPtr<'a,f64,A>>,
          for<'a> <IO as BatchDataType>::Type: AsCudaPtr<'a>,
          for<'a> <<IO as BatchDataType>::Type as AsCudaPtr<'a>>::Pointer: AsPtr<f64> + MemoryMoveTo<f64,CudaMutPtr<'a,f64,A>>,
          for<'a> CudaMutPtr<'a,f64,A>: WriteMemory<f64> + AsMutPtr<f64>,
          for<'a> CudaTensor1dPtrView<'a,f64,N>: From<&'a IO>,
          for<'a> AddBiasBatch<'a,f64,A,N>: Kernel<Args=AddBiasBatchArgs<'a,f64,A,N>>,
          for<'a> CudaVecView<'a,f64,CudaTensor1dPtrView<'a,f64,N>>: TryFrom<&'a <IO as BatchDataType>::Type,Error=TypeConvertError>,
          Self: DeviceReduce<<IO as BatchDataType>::Type,CudaTensor1dPtr<f64,A,N>,f64,N> {
    #[inline]
    fn generalization_bias(&self, bias: &CudaTensor1dPtr<f64,A,N>) -> Result<Arr<f64,N>, GeneralizationError> {
        Ok(bias.read_to_vec()?.try_into()?)
    }
    #[inline]
    fn specialization_bias(&self, bias: Arr<f64,N>) -> Result<CudaTensor1dPtr<f64,A,N>, SpecializationError> {
        let mut b = CudaTensor1dPtr::new(self.get_allocator())?;

        b.memcpy(bias.as_raw_slice().as_ptr(),N)?;

        Ok(b)
    }
    #[inline]
    fn forward_bias<'a>(&self, bias: &CudaTensor1dPtr<f64,A,N>, input: &'a IO) -> Result<IO, EvaluateError> {
        let input_ptr = CudaTensor1dPtrView::<'a,f64,N>::from(input);
        let mut output_ptr = CudaTensor1dPtr::<f64,A,N>::new(self.get_allocator())?;

        bias.memcpy_to(&mut output_ptr,N)?;

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

    #[inline]
    fn backward_bias<'a>(&self, input: IO) -> Result<IO, TrainingError> {
        Ok(input)
    }

    #[inline]
    fn backward_bias_weight_gradient<'a>(&self, loss: &'a IO) -> Result<CudaTensor1dPtr<f64,A,N>, TrainingError> {
        let loss = CudaTensor1dPtrView::<f64,N>::from(loss);

        let mut p = CudaTensor1dPtr::<f64,A,N>::new(self.get_allocator())?;

        p.memcpy(loss.as_ptr(),N)?;

        Ok(p)
    }

    #[inline]
    fn batch_forward_bias<'a>(&self, bias: &CudaTensor1dPtr<f64,A,N>, input: &'a <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, TrainingError> {
        let len = input.size();

        let mut input_output = CudaVec::<f64,CudaTensor1dPtr<f64,A,N>,A>::new(len,&self.allocator)?;
        input.as_cuda_ptr().memcpy_to(&mut input_output.as_cuda_mut_ptr(),N * len)?;

        let mut args = AddBiasBatchArgs::new(
            bias,
            input_output,
            len
        );

        let mut kernel = AddBiasBatch::<'_,f64,A,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.input_output.into_converter().try_into()?)
    }

    #[inline]
    fn batch_backward_bias<'a>(&self, input: <IO as BatchDataType>::Type)
        -> Result<<IO as BatchDataType>::Type, TrainingError> {
        Ok(input)
    }

    #[inline]
    fn batch_backward_bias_weight_gradient<'a>(&self, loss: &'a <IO as BatchDataType>::Type)
        -> Result<CudaTensor1dPtr<f64,A,N>, TrainingError> {
        self.reduce(loss)
    }
}

