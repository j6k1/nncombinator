//! Computational processes used in the implementation of neural networks
pub mod linear;
pub mod batchnormalization;
pub mod bias;
pub mod activation;
pub mod output;
pub mod input;

use std::marker::PhantomData;
use std::fmt::Debug;
use std::ops::{Add, Div};
use num_traits::FromPrimitive;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[cfg(feature = "cuda")]
use std::rc::Rc;
#[cfg(feature = "cuda")]
use rcublas::Context;
#[cfg(feature = "cuda")]
use rcublas_sys::{cublasDscal_v2, cublasHandle_t, cublasSscal_v2, cublasStatus_t};
#[cfg(feature = "cuda")]
use rcublas::api::PointerMode;
#[cfg(feature = "cuda")]
use rcudnn::{Cudnn};
#[cfg(feature = "cuda")]
use rcudnn_sys::cudnnHandle_t;
use crate::arr::{Arr, SerializedVecView};
use crate::error::{TrainingError, TypeConvertError};
use crate::error::EvaluateError::TypeCastError;
use crate::mem::AsRawSlice;
use crate::error::{DeviceError};
#[cfg(feature = "cuda")]
use crate::layer::BatchSize;
#[cfg(feature = "cuda")]
use crate::cuda::{AsCudaMutPtr, AsMutPtr, AsPtr, CudaPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVecView, DataTypeInfo, Kernel, MemorySize};
#[cfg(feature = "cuda")]
use crate::cuda::allocator::CudaAllocator;
#[cfg(feature = "cuda")]
use crate::cuda::kernel::device::{ReduceLinearBatch, ReduceLinearBatchArgs};

/// Trait that defines devices responsible for various computational processes of neural networks
pub trait Device<U>: Clone
    where U: Default + Clone + Debug + Send + Sync + 'static {
}
/// Characteristics defining devices responsible for various convolutional computations of neural networks
pub trait DeviceReduce<T,R,U,const N:usize>
    where U: Default + Clone + Debug + Send + Sync + 'static {
    /// Convolutional computation of input
    /// # Arguments
    /// * `input` - convolutional input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn reduce<'a>(&self, input: &'a T) -> Result<R, TrainingError>;
}
/// Characteristics defining the device responsible for batch averaging calculations in neural networks
pub trait DeviceBatchAveraging<T,U>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static {
    /// Perform batch averaging
    /// # Arguments
    /// * `input` - input tensor
    /// * `batch_size` - batch size
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_averaging<'a>(&self, input: T,batch_size:usize) -> Result<T,TrainingError>;
}
/// Implementation of Device to be computed by CPU
pub struct DeviceCpu<U>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static {
    u:PhantomData<U>,
}
impl<U> DeviceCpu<U>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static {
    /// note: For the sake of implementation uniformity,
    /// DeviceCpu::new is defined as if it may return a DeviceError of type Result,
    /// but this error is never actually returned.
    pub fn new() -> Result<DeviceCpu<U>,DeviceError> {
        Ok(DeviceCpu {
            u: PhantomData::<U>
        })
    }
}
impl<U> Device<U> for DeviceCpu<U>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static {
}
impl<T,U,const N:usize> DeviceReduce<T,Arr<U,N>,U,N> for DeviceCpu<U>
    where U: Default + Clone + Copy + Debug + Send + Sync + Add<Output=U> + 'static,
          T: BatchSize,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a T,Error=TypeConvertError> {
    #[inline]
    fn reduce<'a>(&self, input: &'a T) -> Result<Arr<U,N>,  TrainingError> {
        Ok(SerializedVecView::<'a,U,Arr<U,N>>::try_from(input)?.par_iter()
            .map(|i| i.into())
            .map(|i| Ok(i)).reduce(|| Ok(Arr::new()), |acc,i| {
            acc.and_then(|acc| i.and_then(|i| {
                acc.par_iter().cloned()
                    .zip(i.par_iter().cloned())
                    .map(|(acc, i)| acc + i).collect::<Vec<U>>().try_into()
            }))
        })?)
    }
}
impl<T,U> DeviceBatchAveraging<T,U> for DeviceCpu<U>
    where U: Default + Clone + Copy + Debug + Send + Sync + Div<Output=U> + 'static + FromPrimitive,
          T: AsRawSlice<U> + TryFrom<Vec<U>,Error=TypeConvertError> {
    #[inline]
    fn batch_averaging<'a>(&self, input: T,batch_size:usize) -> Result<T,TrainingError> {
        let batch_size = U::from_usize(batch_size).ok_or(TypeCastError(
            format!("Failed to convert batch size type.")
        ))?;

        Ok(input.as_raw_slice().par_iter().cloned().map(|i| i / batch_size).collect::<Vec<U>>().try_into()?)
    }
}
impl<U> Clone for DeviceCpu<U>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static {
    fn clone(&self) -> Self {
        DeviceCpu {
            u:PhantomData::<U>
        }
    }
}
/// cublas context
#[cfg(feature = "cuda")]
pub struct CublasContext {
    raw:Rc<Context>
}
#[cfg(feature = "cuda")]
impl CublasContext {
    /// Create an instance of CublasContext
    /// # Arguments
    /// * `pointer_mode` - Host or Device
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcublas::error::Error`]
    pub fn new(pointer_mode:PointerMode) -> Result<CublasContext, rcublas::error::Error> {
        let mut context = Context::new()?;

        context.set_pointer_mode(pointer_mode)?;

        Ok(CublasContext {
            raw: Rc::new(context)
        })
    }

    /// Returns a reference to the raw handle (pointer) of the cublas context
    pub fn id_c(&self) -> &cublasHandle_t {
        self.raw.id_c()
    }

    /// Returns the PointerMode that has been set.
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcublas::error::Error`]
    pub fn pointer_mode(&self) -> Result<PointerMode, rcublas::error::Error> {
        self.raw.pointer_mode()
    }
}
#[cfg(feature = "cuda")]
impl Clone for CublasContext {
    fn clone(&self) -> Self {
        CublasContext {
            raw: Rc::clone(&self.raw)
        }
    }
}
/// cudnn context
#[cfg(feature = "cuda")]
pub struct CudnnContext {
    raw:Rc<Cudnn>
}
#[cfg(feature = "cuda")]
impl CudnnContext {
    /// Create an instance of CudnnContext
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    pub fn new() -> Result<CudnnContext, rcudnn::Error> {
        let cudnn = Cudnn::new()?;

        Ok(CudnnContext {
            raw: Rc::new(cudnn)
        })
    }

    /// Returns a reference to the raw handle (pointer) of the cudnn context
    pub fn id_c(&self) -> &cudnnHandle_t {
        self.raw.id_c()
    }
}
#[cfg(feature = "cuda")]
impl Clone for CudnnContext {
    fn clone(&self) -> Self {
        CudnnContext {
            raw: Rc::clone(&self.raw)
        }
    }
}
/// Implementation of Device to be computed by GPU
#[cfg(feature = "cuda")]
pub struct DeviceGpu<U,A> where A: CudaAllocator {
    u:PhantomData<U>,
    cublas:CublasContext,
    cudnn:CudnnContext,
    /// Memory pool for cuda memory allocation
    allocator:A
}
#[cfg(feature = "cuda")]
impl<U,A> DeviceGpu<U,A>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static, A: CudaAllocator {
    /// Create an instance of DeviceGpu
    /// # Arguments
    /// * `memory_pool` - Memory pool for cuda memory allocation
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`DeviceError`]
    pub fn new(allocator:&A) -> Result<DeviceGpu<U,A>,DeviceError> {
        let context = CublasContext::new(PointerMode::Device)?;
        let cudnn = CudnnContext::new()?;

        Ok(DeviceGpu {
            u:PhantomData::<U>,
            cublas:context,
            cudnn:cudnn,
            allocator:allocator.clone()
        })
    }

    /// Returns the CublasContext owned by itself
    pub fn cublas(&self) -> &CublasContext {
        &self.cublas
    }

    /// Returns the CudnnContext owned by itself
    pub fn cudnn(&self) -> &CudnnContext {
        &self.cudnn
    }
}
/// A trait defining the implementation of a memory allocator for CUDA
#[cfg(feature = "cuda")]
pub trait DeviceAllocator<A: CudaAllocator> {
    /// Returns the memory pool object owned by itself
    fn get_allocator(&self) -> &A;
}
#[cfg(feature = "cuda")]
impl<U,A> DeviceAllocator<A> for DeviceGpu<U,A> where A: CudaAllocator {
    fn get_allocator(&self) -> &A {
        &self.allocator
    }
}
#[cfg(feature = "cuda")]
impl<A: CudaAllocator> Device<f32> for DeviceGpu<f32,A> {
}
#[cfg(feature = "cuda")]
impl<U,T,A,const N:usize> DeviceReduce<T,CudaTensor1dPtr<U,A,N>,U,N> for DeviceGpu<U,A>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          T: BatchSize,
          A: CudaAllocator,
          for<'a> CudaVecView<'a,U,CudaTensor1dPtrView<'a,U,N>>: TryFrom<&'a T,Error=TypeConvertError>,
          for<'a> ReduceLinearBatch::<'a,U,A,N>: Kernel<Args=ReduceLinearBatchArgs<'a,U,A,N>> {
    #[inline]
    fn reduce<'a>(&self, input: &'a T) -> Result<CudaTensor1dPtr<U,A,N>, TrainingError> {
        let input_ptr = input.try_into()?;
        let output_ptr = CudaTensor1dPtr::<U,A,N>::new(&self.allocator)?;

        let mut args = ReduceLinearBatchArgs::new(&input_ptr,output_ptr,N,input.size());

        let mut kernel = ReduceLinearBatch::<U,A,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }
}
#[cfg(feature = "cuda")]
impl<T,A> DeviceBatchAveraging<T,f32> for DeviceGpu<f32,A>
    where A: CudaAllocator,
          T: MemorySize + AsCudaMutPtr<Pointee=f32,Allocator=A> {
    fn batch_averaging<'a>(&self, input: T, batch_size: usize) -> Result<T, TrainingError> {
        let batch_size = f32::from_usize(batch_size).ok_or(TypeCastError(
            format!("Failed to convert batch size type.")
        ))?;

        let mut input = input;

        let alpha = CudaPtr::try_from(1. / batch_size)?;

        let tensor_size = T::size();

        match unsafe {
            cublasSscal_v2(*self.cublas.id_c(),
                           tensor_size as ::libc::c_int,
                           alpha.as_ptr(),
                           input.as_mut_ptr(),
                           1
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(input),
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
                    "Unable to get cuBLAS cublasSscal_v2",
                    status as i32 as u64
                )));
            }
        }

    }
}
#[cfg(feature = "cuda")]
impl<T,A> DeviceBatchAveraging<T,f64> for DeviceGpu<f64,A>
    where A: CudaAllocator,
          T: MemorySize + AsCudaMutPtr<Pointee=f64,Allocator=A> {
    fn batch_averaging<'a>(&self, input: T, batch_size: usize) -> Result<T, TrainingError> {
        let batch_size = f64::from_usize(batch_size).ok_or(TypeCastError(
            format!("Failed to convert batch size type.")
        ))?;

        let mut input = input;

        let alpha = CudaPtr::try_from(1. / batch_size)?;

        let tensor_size = T::size();

        match unsafe {
            cublasDscal_v2(*self.cublas.id_c(),
                           tensor_size as ::libc::c_int,
                           alpha.as_ptr(),
                           input.as_mut_ptr(),
                           1
            )
        } {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(input),
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
                    "Unable to get cuBLAS cublasDscal_v2",
                    status as i32 as u64
                )));
            }
        }
    }
}
#[cfg(feature = "cuda")]
impl<A: CudaAllocator> Device<f64> for DeviceGpu<f64,A> {
}
#[cfg(feature = "cuda")]
impl<U,A> Clone for DeviceGpu<U,A>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static + Debug,
          A: CudaAllocator {
    fn clone(&self) -> Self {
        DeviceGpu {
            u:PhantomData::<U>,
            cublas:self.cublas.clone(),
            cudnn:self.cudnn.clone(),
            allocator:self.allocator.clone()
        }
    }
}
