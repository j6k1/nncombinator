//! Computational processes used in the implementation of neural networks
pub mod linear;
pub mod batchnormalization;
pub mod bias;

use std::marker::PhantomData;
use std::{mem};
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use cuda_runtime_sys::dim3;
use libc::{c_uint};
use num_traits::FromPrimitive;
use rcublas::Context;
use rcublas_sys::{cublasHandle_t};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rcublas::api::PointerMode;
use rcudnn::Cudnn;
use rcudnn_sys::cudnnHandle_t;
use crate::arr::{Arr, SerializedVec, SerializedVecView};
use crate::cuda::{CudaPtr, CudaTensor1dPtr, DataTypeInfo, Kernel, Memory};
use crate::cuda::kernel::device::{LossLinearBatchByCanonicalLink, LossLinearBatchByCanonicalLinkArgs, ReduceLinearBatch, ReduceLinearBatchArgs};
use crate::cuda::mem::{MemoryPool};
use crate::error::{DeviceError, TrainingError};
use crate::lossfunction::LossFunction;
use crate::mem::{AsRawSlice};
use crate::UnitValue;

/// Trait that defines devices responsible for various computational processes of neural networks
pub trait Device<U>: Clone where U: UnitValue<U> {
    /// Calculation of Losses
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    /// * `lossf` - loss function
    fn loss_linear<L,const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>, lossf: &L) -> Arr<U, N>
        where L: LossFunction<U>;
    /// Calculation of Losses by canonical link
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    fn loss_linear_by_canonical_link<const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>) -> Arr<U, N>;
    /// Calculation of total Losses
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    /// * `lossf` - loss function
    fn loss_linear_total<L: LossFunction<U>,const N:usize>(&self,exptected:&Arr<U,N>,actual:&Arr<U,N>,lossf:&L) -> U;
    /// Calculation of loss during batch execution by canonical link
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn loss_linear_batch_by_canonical_link<const N: usize>(&self, expected: &SerializedVec<U,Arr<U, N>>, actual: &SerializedVec<U,Arr<U, N>>)
                                                               -> Result<SerializedVec<U,Arr<U, N>>, TrainingError> where f64: From<U>;
    /// Calculation of total Losses (all batch)
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    /// * `lossf` - loss function
    fn batch_loss_linear_total<L: LossFunction<U>,const N:usize>(&self,exptected:&SerializedVec<U,Arr<U,N>>,actual:&SerializedVec<U,Arr<U,N>>,lossf:&L)
        -> Result<U,TrainingError> where f64: From<U> + FromPrimitive, f64: FromPrimitive {
        let n = f64::from_usize(exptected.len()).ok_or(TrainingError::TypeCastError(
            String::from("An error occurred when casting the batch size data type to f64.")
        ))?;

        let loss = actual.par_iter().zip(exptected.par_iter()).map(|(a,e)| {
            a.par_iter().cloned()
                .zip(e.par_iter().cloned())
                .reduce(|| (U::default(),U::default()), |(sum,d),(a,e)| {
                    (sum + lossf.apply(a,e),d)
                })
        }).map(|(sum,_)| sum).reduce(|| U::default(), |sum,l| sum + l);

        U::from_f64(f64::from(loss) / n).ok_or(TrainingError::TypeCastError(
            String::from("An error occurred in the type conversion of the total loss.")
        ))
    }
}
/// Characteristics defining devices responsible for various convolutional computations of neural networks
pub trait DeviceReduce<T,R,U,const N:usize> where U: UnitValue<U> {
    fn reduce(&self, input: T) -> Result<R, TrainingError>;
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
    fn loss_linear<L,const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>, lossf: &L) -> Arr<U, N>
        where L: LossFunction<U> {

        let mut loss = Arr::new();

        for (loss,(a, e))in loss.iter_mut().zip(actual.iter().zip(expected.iter())) {
            *loss = lossf.derive(*a, *e);
        }

        loss
    }

    fn loss_linear_by_canonical_link<const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>) -> Arr<U, N> {
        let mut loss = Arr::new();

        for (l, (a, e)) in loss.iter_mut().zip(actual.iter().zip(expected.iter())) {
            *l = *a - *e;
        }

        loss
    }

    fn loss_linear_total<L: LossFunction<U>, const N: usize>(&self, exptected: &Arr<U, N>, actual: &Arr<U, N>, lossf: &L) -> U {
        actual.iter().zip(exptected.iter()).fold(U::default(),| mut acc,(&a,&e) | {
            acc += lossf.apply(a,e);
            acc
        })
    }

    fn loss_linear_batch_by_canonical_link<const N: usize>(&self, expected: &SerializedVec<U,Arr<U, N>>, actual: &SerializedVec<U,Arr<U, N>>)
                                                               -> Result<SerializedVec<U,Arr<U, N>>, TrainingError> where f64: From<U> {
        let n = U::from_usize(actual.len()).ok_or(TrainingError::TypeCastError(
            String::from("An error occurred when casting the batch size data type to U.")
        ))?;

        Ok(actual.par_iter().zip(expected.par_iter()).map(|(a,e)| {
            a.par_iter().zip(e.par_iter())
                .map(|(&a,&e)| (a - e) / n).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> DeviceReduce<SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,U,N> for DeviceCpu<U>
    where U: UnitValue<U> + Debug {
    #[inline]
    fn reduce(&self, loss: SerializedVecView<'a,U,Arr<U, N>>) -> Result<Arr<U,N>,  TrainingError> {
        Ok(loss.par_iter()
            .map(|l| l.into())
            .map(|l| Ok(l)).reduce(|| Ok(Arr::new()), |acc,l| {
            acc.and_then(|acc| l.and_then(|l| {
                acc.par_iter().cloned()
                    .zip(l.par_iter().cloned())
                    .map(|(acc, l)| acc + l).collect::<Vec<U>>().try_into()
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
    fn loss_linear<L,const N: usize>(&self, expected: &Arr<f32, N>, actual: &Arr<f32, N>, lossf: &L) -> Arr<f32, N>
        where L: LossFunction<f32> {

        let mut loss = Arr::new();

        for (loss,(a, e))in loss.iter_mut().zip(actual.iter().zip(expected.iter())) {
            *loss = lossf.derive(*a, *e);
        }

        loss
    }

    fn loss_linear_by_canonical_link<const N: usize>(&self, expected: &Arr<f32, N>, actual: &Arr<f32, N>) -> Arr<f32, N> {
        let mut loss = Arr::new();

        for (l, (a, e)) in loss.iter_mut().zip(actual.iter().zip(expected.iter())) {
            *l = *a - *e;
        }

        loss
    }

    fn loss_linear_total<L: LossFunction<f32>, const N: usize>(&self, exptected: &Arr<f32, N>, actual: &Arr<f32, N>, lossf: &L) -> f32 {
        actual.iter().zip(exptected.iter()).fold(0.,| mut acc,(&a,&e) | {
            acc += lossf.apply(a,e);
            acc
        })
    }

    fn loss_linear_batch_by_canonical_link<const N: usize>(&self, expected: &SerializedVec<f32, Arr<f32, N>>, actual: &SerializedVec<f32, Arr<f32, N>>)
        -> Result<SerializedVec<f32, Arr<f32, N>>, TrainingError> {
        let mut expected_ptr = CudaPtr::new(expected.len() * N).unwrap();
        expected_ptr.memcpy(expected.as_raw_slice().as_ptr(), expected.len() * N).unwrap();

        let mut actual_ptr = CudaPtr::new(actual.len() * N).unwrap();
        actual_ptr.memcpy(actual.as_raw_slice().as_ptr(), actual.len() * N).unwrap();

        let mut args = LossLinearBatchByCanonicalLinkArgs::new(expected_ptr, actual_ptr, N, expected.len());

        let mut kernel = LossLinearBatchByCanonicalLink::<f32>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32, y: (expected.len() as c_uint + 32 - 1) / 32, z: 1},
                      dim3 { x: 32, y: 32, z: 1 },&mut args,0).unwrap();

        Ok(args.actual.read_to_vec()?.try_into()?)
    }
}
impl<'a,U,const N:usize> DeviceReduce<SerializedVecView<'a,U,Arr<U,N>>,CudaTensor1dPtr<U,N>,U,N> for DeviceGpu<U>
    where U: UnitValue<U> + DataTypeInfo,
          ReduceLinearBatch::<U,N>: Kernel<Args=ReduceLinearBatchArgs<U,N>> {
    #[inline]
    fn reduce(&self, input: SerializedVecView<'a, U, Arr<U, N>>) -> Result<CudaTensor1dPtr<U, N>, TrainingError> {
        let mut loss_ptr = CudaPtr::new(input.len() * N)?;
        loss_ptr.memcpy(input.as_raw_slice().as_ptr(),input.len() * N)?;
        let output_ptr = CudaTensor1dPtr::<U,N>::with_initializer(&self.memory_pool,Default::default)?;

        let mut args = ReduceLinearBatchArgs::new(loss_ptr,output_ptr,N,input.len());

        let mut kernel = ReduceLinearBatch::<U,N>::new();

        kernel.launch(dim3 { x: N as c_uint, y: 1, z: (input.len() as c_uint + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,32 * mem::size_of::<U>())?;

        Ok(args.output)
    }
}
impl Device<f64> for DeviceGpu<f64> {
    fn loss_linear<L,const N: usize>(&self, expected: &Arr<f64, N>, actual: &Arr<f64, N>, lossf: &L) -> Arr<f64, N>
        where L: LossFunction<f64> {

        let mut loss = Arr::new();

        for (loss,(a, e))in loss.iter_mut().zip(actual.iter().zip(expected.iter())) {
            *loss = lossf.derive(*a, *e);
        }

        loss
    }

    fn loss_linear_by_canonical_link<const N: usize>(&self, expected: &Arr<f64, N>, actual: &Arr<f64, N>) -> Arr<f64, N> {
        let mut loss = Arr::new();

        for (l, (a, e)) in loss.iter_mut().zip(actual.iter().zip(expected.iter())) {
            *l = *a - *e;
        }

        loss
    }

    fn loss_linear_total<L: LossFunction<f64>, const N: usize>(&self, exptected: &Arr<f64, N>, actual: &Arr<f64, N>, lossf: &L) -> f64 {
        actual.iter().zip(exptected.iter()).fold(0.,| mut acc,(&a,&e) | {
            acc += lossf.apply(a,e);
            acc
        })
    }

    fn loss_linear_batch_by_canonical_link<const N: usize>(&self, expected: &SerializedVec<f64, Arr<f64, N>>, actual: &SerializedVec<f64, Arr<f64, N>>)
        -> Result<SerializedVec<f64, Arr<f64, N>>, TrainingError> {
        let mut expected_ptr = CudaPtr::new(expected.len() * N).unwrap();
        expected_ptr.memcpy(expected.as_raw_slice().as_ptr(), expected.len() * N).unwrap();

        let mut actual_ptr = CudaPtr::new(actual.len() * N).unwrap();
        actual_ptr.memcpy(actual.as_raw_slice().as_ptr(), actual.len() * N).unwrap();

        let mut args = LossLinearBatchByCanonicalLinkArgs::new(expected_ptr, actual_ptr, N, expected.len());

        let mut kernel = LossLinearBatchByCanonicalLink::<f64>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32, y: (expected.len() as c_uint + 32 - 1) / 32, z: 1},
                      dim3 { x: 32, y: 32, z: 1 },&mut args,0).unwrap();

        Ok(args.actual.read_to_vec()?.try_into()?)
    }
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
