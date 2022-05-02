use std::marker::PhantomData;
use cublas::Context;
use cublas_sys::{cublasOperation_t, cublasSgemm_v2};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rcudnn::{Cudnn, TensorDescriptor};
use rcudnn::utils::DataType;
use crate::activation::Activation;
use crate::arr::{Arr, Arr2};
use crate::error::TrainingError;
use crate::lossfunction::LossFunction;
use crate::mem::{AsRawMutSlice, AsRawSlice};
use crate::activation::ActivationImplType;
use crate::UnitValue;

pub trait Device<U>: Clone + Send + Sync + 'static where U: UnitValue<U> {
    fn forward_linear<const NI:usize,const NO:usize>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:&Arr<U,NI>) -> Arr<U,NO>;
    fn backward_linear<const NI:usize,const NO:usize>(&self, units:&Arr2<U,NI,NO>, input:&Arr<U,NO>) -> Arr<U,NI>;
    fn loss_linear<L,const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>, lossf: &L) -> Arr<U, N>
        where L: LossFunction<U>;
    fn loss_linear_by_canonical_link<const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>) -> Arr<U, N>;
    fn loss_linear_total<L: LossFunction<U>,const N:usize>(&self,exptected:&Arr<U,N>,actual:&Arr<U,N>,lossf:&L) -> U;
    fn batch_loss_linear_by_activaton<A,I,const N:usize>(&self, o:&Vec<Arr<U,N>>, loss:&Vec<Arr<U,N>>, u:&Vec<Arr<U,N>>, activation:&A)
        -> Result<Vec<Arr<U, N>>, TrainingError>
        where A: Activation<U,Arr<U,N>,Self,I>, I: ActivationImplType;
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
    max_threads:usize,
}
impl<U> DeviceCpu<U> where U: UnitValue<U> {
    pub fn with_max_threads(c:usize) -> DeviceCpu<U> {
        DeviceCpu {
            u:PhantomData::<U>,
            max_threads:c
        }
    }

    pub fn new() -> DeviceCpu<U> {
        DeviceCpu::with_max_threads(16)
    }

    pub fn get_max_threads(&self) -> usize {
        self.max_threads
    }
}
impl<U> Device<U> for DeviceCpu<U> where U: UnitValue<U> {
    fn forward_linear<const NI:usize,const NO:usize>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:&Arr<U,NI>) -> Arr<U,NO> {
        let mut output:Arr<U,NO> = Arr::new();

        for (o,w) in output.iter_mut().zip(bias.iter()) {
            *o += *w;
        }

        for (i,u) in input.iter().zip(units.iter()) {
            for (o,w) in output.iter_mut().zip(u.iter()) {
                *o += *i * *w;
            }
        }

        output
    }

    fn backward_linear<const NI:usize, const NO: usize>(&self, units: &Arr2<U,NI,NO>, input: &Arr<U,NO>) -> Arr<U, NI> {
        let mut r = Arr::new();

        for (r,u) in r.iter_mut().zip(units.iter()) {
            for (w,l) in u.iter().zip(input.iter()) {
                *r += *w * *l;
            }
        }

        r
    }

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

    fn batch_loss_linear_by_activaton<A,I,const N:usize>(&self, o:&Vec<Arr<U,N>>, loss:&Vec<Arr<U,N>>, u:&Vec<Arr<U,N>>, activation:&A)
                                                       -> Result<Vec<Arr<U, N>>, TrainingError>
        where A: Activation<U,Arr<U,N>,Self,I>, I: ActivationImplType {

        o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            Ok(activation.derive(&self,o,l,u))
        }).collect::<Result<Vec<Arr<U,N>>,_>>()
    }
}
impl<U> Clone for DeviceCpu<U> where U: UnitValue<U> {
    fn clone(&self) -> Self {
        DeviceCpu {
            u:PhantomData::<U>,
            max_threads:self.max_threads
        }
    }
}
impl<U> DeviceCpu<U> where U: UnitValue<U> {
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
        }).map(|o| o.par_iter().cloned().map(|o| o.try_into()).reduce(|| Ok(Arr::new()), |acc, o| {
            acc.and_then(|acc| o.and_then(|o| {
                acc.par_iter()
                    .zip(o.par_iter())
                    .map(|(&acc, &o)| acc + o)
                    .zip(bias.par_iter()).map(|(acc,&b)| {
                        acc + b
                    }).collect::<Vec<U>>().try_into()
            }))
        })).collect::<Result<Vec<Arr<U, NO>>, _>>().map_err(|e| TrainingError::from(e))
    }
}
pub struct DeviceGpu<U> {
    u:PhantomData<U>
}
impl<U> DeviceGpu<U> where U: UnitValue<U> {
    pub fn new() -> DeviceGpu<U> {
        DeviceGpu {
            u:PhantomData::<U>,
        }
    }
}
impl Device<f32> for DeviceGpu<f32> {
    fn forward_linear<const NI:usize,const NO:usize>(&self,bias:&Arr<f32,NO>,units:&Arr2<f32,NI,NO>,input:&Arr<f32,NI>) -> Arr<f32,NO> {
        let context = Context::new().unwrap();

        let mut output = bias.clone();

        unsafe {
            cublasSgemm_v2(*context.id_c(),
                          cublasOperation_t::CUBLAS_OP_N,
                          cublasOperation_t::CUBLAS_OP_N,
                          NO as ::libc::c_int,
                          1,
                          NI as ::libc::c_int,
                          &1.0f32 as *const f32,
                          units.as_raw_slice().as_ptr(),
                          NO as libc::c_int,
                          input.as_raw_slice().as_ptr(),
                          NO as libc::c_int,
                          &1.0f32 as *const f32,
                          output.as_raw_mut_slice().as_mut_ptr(),
                          NO as ::libc::c_int
            );
        }

        output
    }

    fn backward_linear<const NI:usize, const NO: usize>(&self, units: &Arr2<f32,NI,NO>, input: &Arr<f32,NO>) -> Arr<f32, NI> {
        let mut output:Arr<f32,NI> = Arr::new();

        let context = Context::new().unwrap();

        unsafe {
            cublasSgemm_v2(*context.id_c(),
                           cublasOperation_t::CUBLAS_OP_T,
                           cublasOperation_t::CUBLAS_OP_N,
                           NO as ::libc::c_int,
                           1,
                           NO as ::libc::c_int,
                           &1.0f32 as *const f32,
                           units.as_raw_slice().as_ptr(),
                           NI as libc::c_int,
                           input.as_raw_slice().as_ptr(),
                           NI as libc::c_int,
                           &0.0f32 as *const f32,
                           output.as_raw_mut_slice().as_mut_ptr(),
                           NI as ::libc::c_int
            );
        }

        output
    }

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

    fn batch_loss_linear_by_activaton<A,I,const N:usize>(&self, o:&Vec<Arr<f32,N>>, loss:&Vec<Arr<f32,N>>, u:&Vec<Arr<f32,N>>, activation:&A)
        -> Result<Vec<Arr<f32, N>>, TrainingError>
        where A: Activation<f32,Arr<f32,N>,Self,I>, I: ActivationImplType
    {
        o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            Ok(activation.derive(&self,o,l,u))
        }).collect::<Result<Vec<Arr<f32,N>>,_>>()
    }
}
impl<U> Clone for DeviceGpu<U> where U: UnitValue<U> {
    fn clone(&self) -> Self {
        DeviceGpu {
            u:PhantomData::<U>,
        }
    }
}

impl<U> DeviceGpu<U> where U: DataTypeInfo {
    pub fn linear_activation_forward<F,const N:usize>(&self, callback:F) -> Arr<U,N>
        where U: UnitValue<U>,
              F: FnOnce(&Cudnn,
                        &TensorDescriptor,
                        &TensorDescriptor,
                        *mut libc::c_void) {
        let cudnn = Cudnn::new().unwrap();
        let src_desc = TensorDescriptor::new(&[1,1,N as i32],&[N as i32,N as i32,1], U::cudnn_data_type()).unwrap();
        let dest_desc = TensorDescriptor::new(&[1,1,N as i32],&[N as i32,N as i32,1], U::cudnn_data_type()).unwrap();

        let mut dest_data = Arr::<U,N>::new();

        callback(&cudnn,
                 &src_desc,
                 &dest_desc,
                 dest_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
        dest_data
    }

    pub fn linear_activation_backward<F,const N:usize>(&self, callback:F) -> Arr<U,N>
        where U: UnitValue<U>,
              F: FnOnce(&Cudnn,
                  &TensorDescriptor,
                  &TensorDescriptor,
                  &TensorDescriptor,
                  &TensorDescriptor,
                  *mut libc::c_void
                  ) {
        let cudnn = Cudnn::new().unwrap();
        let src_desc = TensorDescriptor::new(&[1,1,N as i32],&[N as i32,N as i32,1], U::cudnn_data_type()).unwrap();
        let src_diff_desc = TensorDescriptor::new(&[1,1,N as i32],&[N as i32,N as i32,1], U::cudnn_data_type()).unwrap();
        let dest_desc = TensorDescriptor::new(&[1,1,N as i32],&[N as i32,N as i32,1],U::cudnn_data_type()).unwrap();
        let dest_diff_desc = TensorDescriptor::new(&[1,1,N as i32],&[N as i32,N as i32,1], U::cudnn_data_type()).unwrap();

        let mut dest_diff_data = Arr::<U,N>::new();

        callback(&cudnn,
                 &src_desc,
                 &src_diff_desc,
                 &dest_desc,
                 &dest_diff_desc,
                 dest_diff_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
        dest_diff_data
    }
}
