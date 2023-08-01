//! Computational processes used in the implementation of neural networks

use std::marker::PhantomData;
use std::{mem};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use cuda_runtime_sys::dim3;
use libc::{c_uint};
use num_traits::FromPrimitive;
use rcublas::Context;
use rcublas_sys::{cublasDgemm_v2, cublasOperation_t, cublasStatus_t, cublasSgemm_v2, cublasHandle_t, cublasSgemv_v2, cublasDgemv_v2};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rcublas::api::PointerMode;
use crate::arr::{Arr, Arr2, VecArr};
use crate::collection::Broadcast;
use crate::computational_graph::{BroadcastNode, GraphNode, SquareNode, SumNode};
use crate::cuda::{AsMutPtr, AsPtr, CudaMemoryPoolPtr, CudaPtr, Kernel, Memory};
use crate::cuda::kernel::device::{LossLinearBatchByCanonicalLink, LossLinearBatchByCanonicalLinkArgs, ReduceLinearBatch, ReduceLinearBatchArgs};
use crate::cuda::mem::{CachedTensor, MemoryPool};
use crate::error::{DeviceError, EvaluateError, SizeMismatchError, TrainingError};
use crate::lossfunction::LossFunction;
use crate::mem::{AsRawSlice};
use crate::UnitValue;
use crate::ope::{Arithmetic, Sum};

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
    fn loss_linear_batch_by_canonical_link<const N: usize>(&self, expected: &VecArr<U,Arr<U, N>>, actual: &VecArr<U,Arr<U, N>>)
                                                               -> Result<VecArr<U,Arr<U, N>>, TrainingError> where f64: From<U>;

    /// convolutional calculation
    /// # Arguments
    /// * `loss` - loss
    fn batch_linear_reduce<const N: usize>(&self, loss:&VecArr<U,Arr<U,N>>) -> Result<Arr<U,N>,  TrainingError>;
    /// Calculation of total Losses (all batch)
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    /// * `lossf` - loss function
    fn batch_loss_linear_total<L: LossFunction<U>,const N:usize>(&self,exptected:&VecArr<U,Arr<U,N>>,actual:&VecArr<U,Arr<U,N>>,lossf:&L)
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
/// Trait that defines the implementation of various calculation processes in the linear layer
pub trait DeviceLinear<U,T,const NI: usize,const NO: usize> where U: UnitValue<U> {
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
    fn forward_linear(&self, bias:&Arr<U,NO>, units:&T, input:&Arr<U,NI>) -> Result<Arr<U, NO>, EvaluateError>;
    /// Error back propagation calculation
    /// # Arguments
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_linear(&self, units:&T, input:&Arr<U,NO>) -> Result<Arr<U, NI>, TrainingError>;
    /// Calculate the gradient of the weights
    /// # Arguments
    /// * `o` - Input values from upper layers
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_weight_gradient(&self, o: &Arr<U, NI>, loss: &Arr<U,NO>)
                                -> Result<Arr2<U, NI,NO>, TrainingError>;
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
    fn batch_forward_linear(&self,bias:&Arr<U,NO>,units:&T,input:&VecArr<U,Arr<U,NI>>)
                            -> Result<VecArr<U,Arr<U,NO>>,TrainingError>;
    /// Error back propagation in batch
    /// # Arguments
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_linear(&self, units: &T, input: &VecArr<U,Arr<U, NO>>)
                             -> Result<VecArr<U,Arr<U, NI>>, TrainingError>;
    /// Calculate the gradient of the weights in batch
    /// # Arguments
    /// * `o` - Input values from upper layers
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_weight_gradient(&self, o: &VecArr<U,Arr<U, NI>>, loss: &VecArr<U,Arr<U,NO>>)
                                      -> Result<Arr2<U, NI, NO>, TrainingError>;
}
/// Features defining the implementation of the various computational processes in the batch normalization layer
pub trait DeviceBatchNorm<U,T,C,const N:usize>
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

    fn loss_linear_batch_by_canonical_link<const N: usize>(&self, expected: &VecArr<U,Arr<U, N>>, actual: &VecArr<U,Arr<U, N>>)
                                                               -> Result<VecArr<U,Arr<U, N>>, TrainingError> where f64: From<U> {
        Ok(actual.par_iter().zip(expected.par_iter()).map(|(a,e)| {
            a.par_iter().zip(e.par_iter())
                .map(|(&a,&e)| a - e).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }

    fn batch_linear_reduce<const N: usize>(&self, loss: &VecArr<U, Arr<U, N>>) -> Result<Arr<U,N>,  TrainingError> {
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
impl<U,const NI: usize,const NO: usize> DeviceLinear<U,Arr2<U,NI,NO>,NI,NO> for DeviceCpu<U> where U: UnitValue<U> {
    fn forward_linear(&self, bias: &Arr<U, NO>, units: &Arr2<U, NI, NO>, input: &Arr<U, NI>) -> Result<Arr<U, NO>, EvaluateError> {
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

    fn backward_linear(&self, units: &Arr2<U, NI, NO>, input: &Arr<U, NO>) -> Result<Arr<U, NI>, TrainingError> {
        Ok(units.par_iter().map(|u| {
            u.par_iter().zip(input.par_iter())
                .map(|(&w,&l)| w * l).reduce(|| U::default(), |acc,g|{
                acc + g
            })
        }).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))?)
    }

    fn backward_weight_gradient(&self, o: &Arr<U, NI>, loss: &Arr<U, NO>) -> Result<Arr2<U, NI, NO>, TrainingError> {
        Ok(o.par_iter().cloned().map(|o| {
            loss.par_iter().cloned().map(|l| o * l).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,NO>>,_>>()?.try_into().map_err(|e| TrainingError::from(e))?)
    }

    fn batch_backward_linear(&self, units: &Arr2<U, NI, NO>, input: &VecArr<U,Arr<U, NO>>)
                             -> Result<VecArr<U,Arr<U, NI>>, TrainingError> {
        Ok(input.par_iter().map(|l| {
            units.par_iter().map(|u| {
                u.par_iter().zip(l.par_iter()).map(|(&w, &l)| w * l)
                    .reduce(|| U::default(), |acc, l| {
                        acc + l
                    })
            }).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,NI>>,_>>()?.into())
    }

    fn batch_forward_linear(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:&VecArr<U,Arr<U,NI>>)
                            -> Result<VecArr<U,Arr<U,NO>>,TrainingError> {
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

    fn batch_backward_weight_gradient(&self, o: &VecArr<U, Arr<U, NI>>, loss: &VecArr<U, Arr<U, NO>>)
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
/// Implementation of Device to be computed by GPU
pub struct DeviceGpu<U> {
    u:PhantomData<U>,
    cublas:CublasContext,
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

        Ok(DeviceGpu {
            u:PhantomData::<U>,
            cublas:context,
            memory_pool:Arc::clone(memory_pool)
        })
    }

    /// Returns the CublasContext owned by itself
    pub fn cublas(&self) -> &CublasContext {
        &self.cublas
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

    fn loss_linear_batch_by_canonical_link<const N: usize>(&self, expected: &VecArr<f32, Arr<f32, N>>, actual: &VecArr<f32, Arr<f32, N>>)
        -> Result<VecArr<f32, Arr<f32, N>>, TrainingError> {
        let mut expected_ptr = CudaPtr::new(expected.len() * N).unwrap();
        expected_ptr.memcpy(expected.as_raw_slice().as_ptr(), expected.len() * N).unwrap();

        let mut actual_ptr = CudaPtr::new(actual.len() * N).unwrap();
        actual_ptr.memcpy(actual.as_raw_slice().as_ptr(), actual.len() * N).unwrap();

        let mut args = LossLinearBatchByCanonicalLinkArgs::new(expected_ptr, actual_ptr, N, expected.len());

        let mut kernel = LossLinearBatchByCanonicalLink::<f32>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32, y: (expected.len() as c_uint + 32 - 1) / 32, z: 1},
                      dim3 { x: 32, y: 32, z: 1 },&mut args,0).unwrap();

        Ok(args.actual.read_to_vec()?.into())
    }

    fn batch_linear_reduce<const N: usize>(&self, loss: &VecArr<f32, Arr<f32, N>>) -> Result<Arr<f32, N>, TrainingError> {
        let mut loss_ptr = CudaPtr::new(loss.len() * N).unwrap();
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),loss.len() * N).unwrap();
        let output_ptr = CudaPtr::new(N).unwrap();

        let mut args = ReduceLinearBatchArgs::new(loss_ptr,output_ptr,N,loss.len());

        let mut kernel = ReduceLinearBatch::<f32>::new();

        kernel.launch(dim3 { x: N as c_uint, y: 1, z: 1},
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<f32>()).unwrap();

        Ok(args.output.read_to_vec()?.try_into()?)
    }
}
impl<const NI: usize, const NO: usize> DeviceLinear<f32,CachedTensor<f32,Arr2<f32,NI,NO>>,NI,NO> for DeviceGpu<f32> {
    fn forward_linear(&self, bias: &Arr<f32,NO>, units: &CachedTensor<f32,Arr2<f32,NI,NO>>, input: &Arr<f32,NI>)
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

    fn backward_linear(&self, units: &CachedTensor<f32,Arr2<f32,NI,NO>>, input: &Arr<f32,NO>)
                                                        -> Result<Arr<f32, NI>, TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let mut units_ptr = CudaMemoryPoolPtr::new(NI * NO,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NO)?;
        units_ptr.memcpy(units.as_raw_slice().as_ptr(),NI * NO)?;

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

    fn backward_weight_gradient(&self, o: &Arr<f32, NI>, loss: &Arr<f32, NO>) -> Result<Arr2<f32, NI,NO>, TrainingError> {
        let mut o_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI * NO,&self.memory_pool)?;

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

    fn batch_forward_linear(&self,bias:&Arr<f32,NO>,units:&CachedTensor<f32,Arr2<f32,NI,NO>>,input:&VecArr<f32,Arr<f32,NI>>)
                            -> Result<VecArr<f32,Arr<f32,NO>>,TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NI * input.len() ,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NO * input.len(),&self.memory_pool)?;

        let bias_inv:VecArr<f32,Arr<f32,NO>> = bias.par_iter().map(|&b| {
            rayon::iter::repeat(b).take(input.len()).collect::<Vec<f32>>()
        }).collect::<Vec<Vec<f32>>>().into_iter().flatten().collect::<Vec<f32>>().into();

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI * input.len())?;
        output_ptr.memcpy(bias_inv.as_raw_slice().as_ptr(),NO * input.len())?;

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
                Ok(output_ptr.read_to_vec()?.into())
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

    fn batch_backward_linear(&self, units: &CachedTensor<f32, Arr2<f32, NI, NO>>, input: &VecArr<f32, Arr<f32, NO>>) -> Result<VecArr<f32, Arr<f32, NI>>, TrainingError> {
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
                Ok(output_ptr.read_to_vec()?.into())
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

    fn batch_backward_weight_gradient(&self, o: &VecArr<f32, Arr<f32, NI>>, loss: &VecArr<f32, Arr<f32, NO>>)
                                      -> Result<Arr2<f32, NI, NO>, TrainingError> {
        let n = o.len();

        let mut o_ptr = CudaMemoryPoolPtr::new(NI * n,&self.memory_pool)?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO * n,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI * NO,&self.memory_pool)?;

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

    fn loss_linear_batch_by_canonical_link<const N: usize>(&self, expected: &VecArr<f64, Arr<f64, N>>, actual: &VecArr<f64, Arr<f64, N>>)
        -> Result<VecArr<f64, Arr<f64, N>>, TrainingError> {
        let mut expected_ptr = CudaPtr::new(expected.len() * N).unwrap();
        expected_ptr.memcpy(expected.as_raw_slice().as_ptr(), expected.len() * N).unwrap();

        let mut actual_ptr = CudaPtr::new(actual.len() * N).unwrap();
        actual_ptr.memcpy(actual.as_raw_slice().as_ptr(), actual.len() * N).unwrap();

        let mut args = LossLinearBatchByCanonicalLinkArgs::new(expected_ptr, actual_ptr, N, expected.len());

        let mut kernel = LossLinearBatchByCanonicalLink::<f64>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32, y: (expected.len() as c_uint + 32 - 1) / 32, z: 1},
                      dim3 { x: 32, y: 32, z: 1 },&mut args,0).unwrap();

        Ok(args.actual.read_to_vec()?.into())
    }

    fn batch_linear_reduce<const N: usize>(&self, loss: &VecArr<f64, Arr<f64, N>>) -> Result<Arr<f64, N>, TrainingError> {
        let mut loss_ptr = CudaPtr::new(loss.len() * N).unwrap();
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),loss.len() * N).unwrap();
        let output_ptr = CudaPtr::new(N).unwrap();

        let mut args = ReduceLinearBatchArgs::new(loss_ptr,output_ptr,N,loss.len());

        let mut kernel = ReduceLinearBatch::<f64>::new();

        kernel.launch(dim3 { x: N as c_uint, y: 1, z: 1},
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<f64>()).unwrap();

        Ok(args.output.read_to_vec()?.try_into()?)
    }
}
impl<const NI: usize, const NO: usize> DeviceLinear<f64,CachedTensor<f64,Arr2<f64,NI,NO>>,NI,NO> for DeviceGpu<f64> {
    fn forward_linear(&self, bias: &Arr<f64,NO>, units: &CachedTensor<f64,Arr2<f64,NI,NO>>, input: &Arr<f64,NI>)
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
                    "Unable to get cuBLAS cublasDgemv_v2",
                    status as i32 as u64
                )));
            }
        }
    }

    fn backward_linear(&self, units: &CachedTensor<f64,Arr2<f64,NI,NO>>, input: &Arr<f64,NO>)
                                                        -> Result<Arr<f64, NI>, TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let mut units_ptr = CudaMemoryPoolPtr::new(NI * NO,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NO)?;
        units_ptr.memcpy(units.as_raw_slice().as_ptr(),NI * NO)?;

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

    fn backward_weight_gradient(&self, o: &Arr<f64, NI>, loss: &Arr<f64, NO>) -> Result<Arr2<f64,NI,NO>, TrainingError> {
        let mut o_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI * NO,&self.memory_pool)?;

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

    fn batch_forward_linear(&self,bias:&Arr<f64,NO>,units:&CachedTensor<f64,Arr2<f64,NI,NO>>,input:&VecArr<f64,Arr<f64,NI>>)
                            -> Result<VecArr<f64,Arr<f64,NO>>,TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NI * input.len() ,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NO * input.len(),&self.memory_pool)?;

        let bias_inv:VecArr<f64,Arr<f64,NO>> = bias.par_iter().map(|&b| {
            rayon::iter::repeat(b).take(input.len()).collect::<Vec<f64>>()
        }).collect::<Vec<Vec<f64>>>().into_iter().flatten().collect::<Vec<f64>>().into();

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI * input.len())?;
        output_ptr.memcpy(bias_inv.as_raw_slice().as_ptr(),NO * input.len())?;

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
                Ok(output_ptr.read_to_vec()?.into())
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

    fn batch_backward_linear(&self, units: &CachedTensor<f64, Arr2<f64, NI, NO>>, input: &VecArr<f64, Arr<f64, NO>>) -> Result<VecArr<f64, Arr<f64, NI>>, TrainingError> {
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
                Ok(output_ptr.read_to_vec()?.into())
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

    fn batch_backward_weight_gradient(&self, o: &VecArr<f64, Arr<f64, NI>>, loss: &VecArr<f64, Arr<f64, NO>>)
                                      -> Result<Arr2<f64, NI, NO>, TrainingError> {
        let n = o.len();

        let mut o_ptr = CudaMemoryPoolPtr::new(NI * n,&self.memory_pool)?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO * n,&self.memory_pool)?;
        let mut output_ptr = CudaMemoryPoolPtr::new(NI * NO,&self.memory_pool)?;

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
}
impl<U,const N:usize> DeviceBatchNorm<U,Arr<U,N>,Arr<U,N>,N> for DeviceCpu<U>
    where U: UnitValue<U>,
          for<'a> Arr<U,N>: Arithmetic<&'a Arr<U,N>,Arr<U,N>> + TryFrom<Vec<U>,Error = SizeMismatchError> +
                            Arithmetic<U,Arr<U,N>>,
          for<'a> &'a Arr<U,N>: Arithmetic<&'a Arr<U,N>,Arr<U,N>> + TryFrom<Vec<U>,Error = SizeMismatchError> + Arithmetic<U,Arr<U,N>>,
          for<'data> VecArr<U,Arr<U,N>>: Arithmetic<&'data VecArr<U,Arr<U,N>>, VecArr<U,Arr<U,N>>> +
                                         Arithmetic<U,VecArr<U,Arr<U,N>>> +
                                         Arithmetic<Broadcast<Arr<U,N>>,VecArr<U,Arr<U,N>>>,
          for<'data> &'data VecArr<U,Arr<U,N>>: Arithmetic<&'data VecArr<U,Arr<U,N>>,VecArr<U,Arr<U,N>>> +
                                                Arithmetic<U,VecArr<U,Arr<U,N>>> +
                                                Arithmetic<Broadcast<Arr<U,N>>,VecArr<U,Arr<U,N>>> {
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
                 scale * (i - mean) / (variance + eps) + bias
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
                scale * (i - mean) / (variance + eps) + bias
            }).collect::<Vec<U>>().try_into()?,
            estimated_mean.clone(),
            estimated_variance.par_iter().map(|&v| U::one() / (v + eps)).collect::<Vec<U>>().try_into()?
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
                    scale * (i - mean) / (variance + eps) + bias
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

        let mean:Arr<U,N> = SumNode::new().forward(input).par_iter().map(|&i| {
            U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
                "Error in type conversion from usize."
            ))).map(|n| i / n)
        }).collect::<Result<Vec<U>,_>>()?.try_into()?;

        let variance:VecArr<U,Arr<U,N>> = (input - Broadcast::<Arr<U,N>>(mean.clone()))
                               .par_iter()
                               .map(|i| {
                                    i.par_iter().map(|&i| {
                                        SquareNode::new().forward(i)
                                    }).collect::<Vec<U>>().try_into()
                                }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into();
        let variance = variance.sum() / U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let inv_variance = variance.par_iter().map(|&v| U::one() / (v + eps)).collect::<Vec<U>>().try_into()?;

        let o:VecArr<U,Arr<U,N>> = Broadcast(<Vec<U> as TryInto<Arr<U,N>>>::try_into(variance.par_iter().map(|&v| {
            U::one() / (v + eps)
        }).collect::<Vec<U>>())?) * &(input - Broadcast(mean.clone()));

        let running_mean = running_mean.par_iter().zip(mean.par_iter()).map(|(&rm,&m)| {
            rm * momentum + m * (U::one() - momentum)
        }).collect::<Vec<U>>().try_into()?;

        let running_variance = running_variance.par_iter().zip(variance.par_iter()).map(|(&rv,&v)| {
            rv * momentum + v * (U::one() - momentum)
        }).collect::<Vec<U>>().try_into()?;

        let o = &(&o * &BroadcastNode::new().forward((scale,n))) + Broadcast(bias.clone());

        Ok((o,mean,inv_variance,running_mean,running_variance))
    }

    fn backward_batch_norm(&self, loss: &Arr<U, N>, input: &Arr<U, N>,
                           scale: &Arr<U, N>, saved_mean: &Arr<U, N>, saved_inv_variance: &Arr<U, N>)
                           -> Result<(Arr<U, N>, Arr<U, N>, Arr<U, N>), TrainingError> {
        let b = loss.clone();

        let x = input - saved_mean;

        let s = &(&x * saved_inv_variance) * loss;

        let dx1 = scale * loss;
        let dx2 = &dx1 * saved_inv_variance;
        let dx3 = &x * &dx1;
        let dx4 = &dx3 - &(saved_inv_variance * saved_inv_variance);
        let dx5 = &dx4 / &(saved_inv_variance * U::from_f64(2.).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from f64.")
        ))?);
        let dx6 = &x * &dx5 * U::from_usize(2).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;
        let dx7 = &dx2 + &dx6;
        let dx8 = dx7.clone();
        let dx9 = -&dx7;
        let dx = &dx8 + &dx9;

        Ok((dx,s,b))
    }

    fn batch_backward_batch_norm(&self, loss: &VecArr<U, Arr<U, N>>,
                                 input: &VecArr<U,Arr<U,N>>,
                                 scale: &Arr<U, N>,
                                 saved_mean: &Arr<U, N>, saved_inv_variance: &Arr<U, N>)
        -> Result<(VecArr<U, Arr<U, N>>, Arr<U, N>, Arr<U, N>), TrainingError> {
        let n = input.len();

        let b = BroadcastNode::new().backward(loss);

        let x = BroadcastNode::new().forward((saved_mean,n));
        let x2 = input - &x;
        let iv = BroadcastNode::new().forward((saved_inv_variance,n));

        let s = BroadcastNode::new().backward(&(&x2 * &iv * loss));

        let dx1 = Broadcast(scale.clone()) * loss;
        let dx2 = &dx1 * &iv;
        let dx3 = BroadcastNode::new().backward(&(&x2 * &dx1));
        let dx4 = &(-(saved_inv_variance * saved_inv_variance)) * &dx3;
        let dx5 = &dx4 / &(saved_inv_variance * U::from_f64(2.).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from f64.")
        ))?);
        let dx6 = SumNode::new().backward((&(dx5 / U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?),n));
        let dx7 = &x2 * &dx6 * U::from_usize(2).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;
        let dx8 = &dx2 + &dx7;
        let dx9 = dx8.clone();
        let dx10 = -dx8;
        let dx11 = BroadcastNode::new().backward(&dx10);
        let dx12 = SumNode::new().backward((&dx11,n)) / U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;
        let dx = &dx9 + &dx12;

        Ok((dx,s,b))
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
