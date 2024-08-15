//! Computational processes used in the implementation of neural networks
pub mod linear;
pub mod batchnormalization;
pub mod bias;
pub mod activation;
pub mod output;
pub mod input;

use std::marker::PhantomData;
use std::{mem};
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use cuda_runtime_sys::dim3;
use libc::{c_uint};
use rcublas::Context;
use rcublas_sys::{cublasHandle_t};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rcublas::api::PointerMode;
use rcudnn::{Cudnn};
use rcudnn_sys::cudnnHandle_t;
use crate::arr::{Arr, SerializedVecView};
use crate::cuda::{CudaTensor1dPtr, CudaVecView, DataTypeInfo, Kernel};
use crate::cuda::kernel::device::{ReduceLinearBatch, ReduceLinearBatchArgs};
use crate::cuda::mem::{MemoryPool};
use crate::error::{DeviceError, TrainingError, TypeConvertError};
use crate::layer::BatchSize;
use crate::UnitValue;

/// Trait that defines devices responsible for various computational processes of neural networks
pub trait Device<U>: Clone where U: UnitValue<U> {
}
/// Characteristics defining devices responsible for various convolutional computations of neural networks
pub trait DeviceReduce<T,R,U,const N:usize> where U: UnitValue<U> {
    fn reduce<'a>(&self, input: &'a T) -> Result<R, TrainingError>;
}
/// Implementation of Device to be computed by CPU
pub struct DeviceCpu<U> where U: UnitValue<U> {
    u:PhantomData<U>,
}
impl<U> DeviceCpu<U> where U: UnitValue<U> {
    /// note: For the sake of implementation uniformity,
    /// DeviceCpu::new is defined as if it may return a DeviceError of type Result,
    /// but this error is never actually returned.
    pub fn new() -> Result<DeviceCpu<U>,DeviceError> {
        Ok(DeviceCpu {
            u: PhantomData::<U>
        })
    }
}
impl<U> Device<U> for DeviceCpu<U> where U: UnitValue<U> {
}
impl<U,T,const N:usize> DeviceReduce<T,Arr<U,N>,U,N> for DeviceCpu<U>
    where U: UnitValue<U> + Debug,
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
impl<U> Clone for DeviceCpu<U> where U: UnitValue<U> {
    fn clone(&self) -> Self {
        DeviceCpu {
            u:PhantomData::<U>
        }
    }
}
/// cublas context
pub struct CublasContext {
    raw:Rc<Context>
}
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
impl Clone for CublasContext {
    fn clone(&self) -> Self {
        CublasContext {
            raw: Rc::clone(&self.raw)
        }
    }
}
/// cudnn context
pub struct CudnnContext {
    raw:Rc<Cudnn>
}
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
impl Clone for CudnnContext {
    fn clone(&self) -> Self {
        CudnnContext {
            raw: Rc::clone(&self.raw)
        }
    }
}
/// Implementation of Device to be computed by GPU
pub struct DeviceGpu<U> {
    u:PhantomData<U>,
    cublas:CublasContext,
    cudnn:CudnnContext,
    /// Memory pool for cuda memory allocation
    pub memory_pool:Arc<Mutex<MemoryPool>>
}
impl<U> DeviceGpu<U> where U: UnitValue<U> {
    /// Create an instance of DeviceGpu
    /// # Arguments
    /// * `memory_pool` - Memory pool for cuda memory allocation
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`DeviceError`]
    pub fn new(memory_pool:&Arc<Mutex<MemoryPool>>) -> Result<DeviceGpu<U>,DeviceError> {
        let context = CublasContext::new(PointerMode::Device)?;
        let cudnn = CudnnContext::new()?;

        Ok(DeviceGpu {
            u:PhantomData::<U>,
            cublas:context,
            cudnn:cudnn,
            memory_pool:Arc::clone(memory_pool)
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
pub trait DeviceMemoryPool {
    /// Returns the memory pool object owned by itself
    fn get_memory_pool(&self) -> &Arc<Mutex<MemoryPool>>;
}
impl<U> DeviceMemoryPool for DeviceGpu<U> {
    fn get_memory_pool(&self) -> &Arc<Mutex<MemoryPool>> {
        &self.memory_pool
    }
}
impl Device<f32> for DeviceGpu<f32> {
}
impl<U,T,const N:usize> DeviceReduce<T,CudaTensor1dPtr<U,N>,U,N> for DeviceGpu<U>
    where U: UnitValue<U> + DataTypeInfo,
          T: BatchSize,
          for<'a> CudaVecView<'a,U,CudaTensor1dPtr<U,N>>: TryFrom<&'a T,Error=TypeConvertError>,
          for<'a> ReduceLinearBatch::<'a,U,N>: Kernel<Args=ReduceLinearBatchArgs<'a,U,N>> {
    #[inline]
    fn reduce<'a>(&self, input: &'a T) -> Result<CudaTensor1dPtr<U, N>, TrainingError> {
        let input_ptr = input.try_into()?;
        let output_ptr = CudaTensor1dPtr::<U,N>::with_initializer(&self.memory_pool,Default::default)?;

        let mut args = ReduceLinearBatchArgs::new(&input_ptr,output_ptr,N,input.size());

        let mut kernel = ReduceLinearBatch::<U,N>::new();

        kernel.launch(dim3 { x: N as c_uint, y: 1, z: (input.size() as c_uint + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,32 * mem::size_of::<U>())?;

        Ok(args.output)
    }
}
impl Device<f64> for DeviceGpu<f64> {
}
impl<U> Clone for DeviceGpu<U> where U: UnitValue<U> + Debug {
    fn clone(&self) -> Self {
        DeviceGpu {
            u:PhantomData::<U>,
            cublas:self.cublas.clone(),
            cudnn:self.cudnn.clone(),
            memory_pool:Arc::clone(&self.memory_pool)
        }
    }
}
