use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use rcublas::Context;
use rcublas_sys::{cublasDgemv_v2, cublasOperation_t, cublasStatus_t, cublasSgemv_v2, cublasHandle_t};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rcublas::api::PointerMode;
use rcudnn::{Cudnn, TensorDescriptor};
use rcudnn::utils::DataType;
use crate::activation::Activation;
use crate::arr::{Arr, Arr2};
use crate::cuda::{AsMutPtr, AsPtr, CudaMemoryPoolPtr, CudaPtr, Memory};
use crate::cuda::mem::{CachedTensor, MemoryPool};
use crate::error::{DeviceError, EvaluateError, TrainingError};
use crate::lossfunction::LossFunction;
use crate::mem::{AsRawSlice};
use crate::UnitValue;

pub trait Device<U>: Clone where U: UnitValue<U> {
    fn loss_linear<L,const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>, lossf: &L) -> Arr<U, N>
        where L: LossFunction<U>;
    fn loss_linear_by_canonical_link<const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>) -> Arr<U, N>;
    fn loss_linear_total<L: LossFunction<U>,const N:usize>(&self,exptected:&Arr<U,N>,actual:&Arr<U,N>,lossf:&L) -> U;
}
pub trait DeviceLinear<U> where U: UnitValue<U> {
    fn forward_linear<const NI:usize,const NO:usize>(&self, bias:&Arr<U,NO>, units:&CachedTensor<U,Arr2<U,NI,NO>>, input:&Arr<U,NI>) -> Result<Arr<U, NO>, EvaluateError>;
    fn backward_linear<const NI:usize,const NO:usize>(&self, units:&CachedTensor<U,Arr2<U,NI,NO>>, input:&Arr<U,NO>) -> Result<Arr<U, NI>, TrainingError>;
}
pub trait DeviceActivation<U>: Device<U> where U: UnitValue<U> {
    fn batch_loss_linear_by_activaton<A,const N:usize>(&self, o:&Vec<Arr<U,N>>, loss:&Vec<Arr<U,N>>, u:&Vec<Arr<U,N>>, activation:&A)
                                                           -> Result<Vec<Arr<U, N>>, TrainingError>
        where A: Activation<U,Arr<U,N>,Self>;
}
pub trait DataTypeInfo {
    fn cudnn_data_type() -> DataType;
    fn size() -> usize;
}
impl DataTypeInfo for f32 {
    fn cudnn_data_type() -> DataType {
        DataType::Float
    }
    fn size() -> usize {
        4_usize
    }
}
impl DataTypeInfo for f64 {
    fn cudnn_data_type() -> DataType {
        DataType::Double
    }
    fn size() -> usize {
        8_usize
    }
}

pub struct DeviceCpu<U> where U: UnitValue<U> {
    u:PhantomData<U>,
}
impl<U> DeviceCpu<U> where U: UnitValue<U> {
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
}
impl<U> Clone for DeviceCpu<U> where U: UnitValue<U> {
    fn clone(&self) -> Self {
        DeviceCpu {
            u:PhantomData::<U>
        }
    }
}
impl<U> DeviceCpu<U> where U: UnitValue<U> {
    pub fn forward_linear<const NI: usize, const NO: usize>(&self, bias: &Arr<U, NO>, units: &Arr2<U, NI, NO>, input: &Arr<U, NI>) -> Result<Arr<U, NO>, EvaluateError> {
        let mut output:Arr<U,NO> = Arr::new();

        for (o,w) in output.iter_mut().zip(bias.iter()) {
            *o += *w;
        }

        for (i,u) in input.iter().zip(units.iter()) {
            for (o,w) in output.iter_mut().zip(u.iter()) {
                *o += *i * *w;
            }
        }

        Ok(output)
    }

    pub fn backward_linear<const NI: usize, const NO: usize>(&self, units: &Arr2<U, NI, NO>, input: &Arr<U, NO>) -> Result<Arr<U, NI>, TrainingError> {
        let mut r = Arr::new();

        for (r,u) in r.iter_mut().zip(units.iter()) {
            for (w,l) in u.iter().zip(input.iter()) {
                *r += *w * *l;
            }
        }

        Ok(r)
    }

    pub fn batch_loss_linear_by_activaton<A,const N:usize>(&self, o:&Vec<Arr<U,N>>, loss:&Vec<Arr<U,N>>, u:&Vec<Arr<U,N>>, activation:&A)
                                                       -> Result<Vec<Arr<U, N>>, TrainingError>
        where A: Activation<U,Arr<U,N>,Self> + Send + Sync + 'static {

        o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            activation.derive(&self,o,l,u)
        }).collect::<Result<Vec<Arr<U,N>>,_>>()
    }

    pub fn loss_linear_batch<L,const N: usize>(&self, expected: &Vec<Arr<U, N>>, actual: &Vec<Arr<U, N>>, lossf: &L)
        -> Result<Vec<Arr<U, N>>, TrainingError>
        where L: LossFunction<U> {

        actual.par_iter().zip(expected.par_iter()).map(|(a,e)| {
            a.par_iter()
             .zip(e.par_iter())
             .map(|(&a,&e)| lossf.derive(a,e))
             .collect::<Vec<U>>()
             .try_into().map_err(|e| TrainingError::from(e))
        }).collect::<Result<Vec<Arr<U,N>>,_>>()
    }

    pub fn loss_linear_batch_by_canonical_link<const N: usize>(&self, expected: &Vec<Arr<U, N>>, actual: &Vec<Arr<U, N>>)
        -> Result<Vec<Arr<U, N>>, TrainingError> {
        actual.par_iter().zip(expected.par_iter()).map(|(a,e)| {
            a.par_iter().zip(e.par_iter())
             .map(|(&a,&e)| a - e).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))
        }).collect::<Result<Vec<Arr<U,N>>,_>>()
    }

    pub fn backward_linear_batch<const NI:usize, const NO: usize>(&self, units: &Arr2<U, NI, NO>, input: &Vec<Arr<U, NO>>)
                                                                  -> Result<Vec<Arr<U, NI>>, TrainingError> {
        input.par_iter().map(|input| {
            units.par_iter().map(|u| {
                u.par_iter().cloned().zip(input.par_iter().cloned())
                    .reduce(|| (U::default(),U::default()), | (sum,d), (w,l) | (sum + w * l,d))
            }).map(|(r,_)| r).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))
        }).collect::<Result<Vec<Arr<U,NI>>,_>>()
    }

    pub fn batch_loss_linear_total<L: LossFunction<U>,const N:usize>(&self,exptected:&Vec<Arr<U,N>>,actual:&Vec<Arr<U,N>>,lossf:&L) -> U {
        actual.par_iter().zip(exptected.par_iter()).map(|(a,e)| {
            a.par_iter().cloned()
             .zip(e.par_iter().cloned())
             .reduce(|| (U::default(),U::default()), |(sum,d),(a,e)| {
                 (sum + lossf.apply(a,e),d)
             })
        }).map(|(sum,_)| sum).reduce(|| U::default(), |sum,l| sum + l)
    }

    pub fn batch_forward_linear<const NI:usize,const NO:usize>(&self,input:&Vec<Arr<U,NI>>,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>) -> Result<Vec<Arr<U,NO>>,TrainingError> {
        input.par_iter().map(|input| {
            input.par_iter().zip(units.par_iter()).map(|(&i, unit)| {
                unit.par_iter().map(|&w| {
                    i * w
                }).collect::<Vec<U>>()
            }).collect::<Vec<Vec<U>>>()
        }).flat_map(|o| o.par_iter().cloned().map(|o| o.try_into()).reduce(|| Ok(Arr::<U,NO>::new()), |acc, o| {
            acc.and_then(|acc| o.and_then(|o| {
                acc.par_iter()
                    .zip(o.par_iter())
                    .map(|(&acc, &o)| acc + o).collect::<Vec<U>>().try_into()
            }))
        })).map(|o| {
            o.par_iter().zip(bias.par_iter()).map(|(&o, &b)| {
                o + b
            }).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U, NO>>, _>>().map_err(|e| TrainingError::from(e))
    }
}
pub struct CublasContext {
    raw:Rc<Context>
}
impl CublasContext {
    pub fn new(pointer_mode:PointerMode) -> Result<CublasContext, rcublas::error::Error> {
        let mut context = Context::new()?;

        context.set_pointer_mode(pointer_mode)?;

        Ok(CublasContext {
            raw: Rc::new(context)
        })
    }

    pub fn id_c(&self) -> &cublasHandle_t {
        self.raw.id_c()
    }

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
pub struct DeviceGpu<U> {
    u:PhantomData<U>,
    cublas:CublasContext,
    pub memory_pool:Arc<Mutex<MemoryPool>>
}
impl<U> DeviceGpu<U> where U: UnitValue<U> {
    pub fn new(memory_pool:MemoryPool) -> Result<DeviceGpu<U>,DeviceError> {
        let mut context = CublasContext::new(PointerMode::Device)?;

        Ok(DeviceGpu {
            u:PhantomData::<U>,
            cublas:context,
            memory_pool:Arc::new(Mutex::new(memory_pool))
        })
    }

    pub fn cublas(&self) -> &CublasContext {
        &self.cublas
    }
}
pub trait DeviceMemoryPool {
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
}
impl DeviceLinear<f32> for DeviceGpu<f32> {
    fn forward_linear<const NI:usize,const NO:usize>(&self, bias: &Arr<f32,NO>, units: &CachedTensor<f32,Arr2<f32,NI,NO>>, input: &Arr<f32,NI>)
                                                     -> Result<Arr<f32, NO>, EvaluateError> {

        let mut input_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI)?;
        output_ptr.memcpy(bias.as_raw_slice().as_ptr(),NO)?;

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

    fn backward_linear<const NI:usize, const NO: usize>(&self, units: &CachedTensor<f32,Arr2<f32,NI,NO>>, input: &Arr<f32,NO>)
                                                        -> Result<Arr<f32, NI>, TrainingError> {

        let mut input_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI)?;

        let alpha = CudaPtr::try_from(1.0f32)?;
        let beta = CudaPtr::try_from(0.0f32)?;

        match unsafe {
            cublasSgemv_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_T,
                           NI as ::libc::c_int,
                           NO as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NI as libc::c_int,
                           input_ptr.as_ptr(),
                           1,
                           beta.as_ptr(),
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
}
impl DeviceActivation<f32> for DeviceGpu<f32> {
    fn batch_loss_linear_by_activaton<A,const N:usize>(&self, o:&Vec<Arr<f32,N>>, loss:&Vec<Arr<f32,N>>, u:&Vec<Arr<f32,N>>, activation:&A)
                                                       -> Result<Vec<Arr<f32, N>>, TrainingError>
        where A: Activation<f32,Arr<f32,N>,Self>
    {
        todo!()
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
}
impl DeviceLinear<f64> for DeviceGpu<f64> {
    fn forward_linear<const NI:usize,const NO:usize>(&self, bias: &Arr<f64,NO>, units: &CachedTensor<f64,Arr2<f64,NI,NO>>, input: &Arr<f64,NI>)
                                                     -> Result<Arr<f64, NO>, EvaluateError> {

        let mut input_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI)?;
        output_ptr.memcpy(bias.as_raw_slice().as_ptr(),NO)?;

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
                    "Unable to get cuBLAS cublasSgemv_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    fn backward_linear<const NI:usize, const NO: usize>(&self, units: &CachedTensor<f64,Arr2<f64,NI,NO>>, input: &Arr<f64,NO>)
                                                        -> Result<Arr<f64, NI>, TrainingError> {

        let mut input_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI)?;

        let alpha = CudaPtr::try_from(1.0f64)?;
        let beta = CudaPtr::try_from(0.0f64)?;

        match unsafe {
            cublasDgemv_v2(*self.cublas.id_c(),
                           cublasOperation_t::CUBLAS_OP_T,
                           NI as ::libc::c_int,
                           NO as ::libc::c_int,
                           alpha.as_ptr(),
                           units.as_ptr(),
                           NI as libc::c_int,
                           input_ptr.as_ptr(),
                           1,
                           beta.as_ptr(),
                           output_ptr.as_mut_ptr(),
                           1,
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
}
impl DeviceActivation<f64> for DeviceGpu<f64> {
    fn batch_loss_linear_by_activaton<A,const N:usize>(&self, o:&Vec<Arr<f64,N>>, loss:&Vec<Arr<f64,N>>, u:&Vec<Arr<f64,N>>, activation:&A)
                                                       -> Result<Vec<Arr<f64, N>>, TrainingError>
        where A: Activation<f64,Arr<f64,N>,Self>
    {
        todo!()
    }
}
impl<U> Clone for DeviceGpu<U> where U: UnitValue<U> {
    fn clone(&self) -> Self {
        DeviceGpu {
            u:PhantomData::<U>,
            cublas:self.cublas.clone(),
            memory_pool:Arc::clone(&self.memory_pool)
        }
    }
}
impl<U> DeviceGpu<U> where U: DataTypeInfo + Default + Debug {
    pub fn linear_activation_forward<F,const N:usize>(&self, input:&Arr<U,N>,callback:F) -> Result<Arr<U,N>,EvaluateError>
        where U: UnitValue<U>,
              F: FnOnce(&Cudnn,
                        &TensorDescriptor,
                        CudaPtr<U>) -> Result<Arr<U,N>,EvaluateError> {
        let cudnn = Cudnn::new()?;
        let desc = TensorDescriptor::new(&[1,1,N as i32],&[N as i32,N as i32,1], U::cudnn_data_type())?;
        let mut input_output = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(),N)?;

        callback(&cudnn,
                 &desc,
                 input_output)
    }

    pub fn linear_activation_backward<F,const N:usize>(&self, o:&Arr<U,N>,loss:&Arr<U,N>,u:&Arr<U,N>,callback:F) -> Result<Arr<U,N>,TrainingError>
        where U: UnitValue<U>,
              F: FnOnce(&Cudnn,
                  &TensorDescriptor,
                  CudaPtr<U>,
                  &TensorDescriptor,
                  CudaPtr<U>,
                  &TensorDescriptor,
                  CudaPtr<U>,
                  ) -> Result<Arr<U,N>,TrainingError> {
        let cudnn = Cudnn::new()?;
        let o_desc = TensorDescriptor::new(&[1,1,N as i32], &[N as i32,N as i32,1], U::cudnn_data_type())?;
        let mut o_ptr = CudaPtr::new(N)?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(), N)?;

        let loss_desc = TensorDescriptor::new(&[1,1,N as i32], &[N as i32,N as i32,1], U::cudnn_data_type())?;
        let mut loss_ptr = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let u_desc = TensorDescriptor::new(&[1,1,N as i32], &[N as i32,N as i32,1], U::cudnn_data_type())?;
        let mut u_ptr = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        callback(&cudnn,
                 &o_desc,
                 o_ptr,
                 &loss_desc,
                 loss_ptr,
                 &u_desc,
                 u_ptr)
    }
}
