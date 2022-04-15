use std::marker::PhantomData;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelIterator};
use crate::activation::Activation;
use crate::arr::{Arr, Arr2};
use crate::error::TrainingError;
use crate::lossfunction::LossFunction;
use crate::mem::AsRawSlice;
use crate::UnitValue;

pub trait Device<U>: Clone + Send + Sync + 'static where U: UnitValue<U> {
    fn forward_linear<const NI:usize,const NO:usize>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:&Arr<U,NI>) -> Arr<U,NO>;
    fn backward_linear<const NI:usize,const NO:usize>(&self, units:&Arr2<U,NI,NO>, input:&Arr<U,NO>) -> Arr<U,NI>;
    fn loss_linear<L,const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>, lossf: &L) -> Arr<U, N>
        where L: LossFunction<U>;
    fn loss_linear_by_canonical_link<const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>) -> Arr<U, N>;
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
            a.as_raw_slice()
             .par_iter()
             .zip(e.as_raw_slice().par_iter())
             .map(|(&a,&e)| lossf.derive(a,e))
             .collect::<Vec<U>>()
             .try_into().map_err(|e| TrainingError::from(e))
        }).collect::<Result<Vec<Arr<U,N>>,_>>()
    }

    pub fn loss_linear_batch_by_canonical_link<const N: usize>(&self, expected: &Vec<Arr<U, N>>, actual: &Vec<Arr<U, N>>)
        -> Result<Vec<Arr<U, N>>, TrainingError> {
        actual.par_iter().zip(expected.par_iter()).map(|(a,e)| {
            a.as_raw_slice()
             .par_iter().zip(e.as_raw_slice().par_iter())
             .map(|(&a,&e)| a - e).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))
        }).collect::<Result<Vec<Arr<U,N>>,_>>()
    }

    pub fn backward_linear_batch<const NI:usize, const NO: usize>(&self, units: &Arr2<U, NI, NO>, input: &Vec<Arr<U, NO>>)
                                                                  -> Result<Vec<Arr<U, NI>>, TrainingError> {
        input.par_iter().map(|input| {
            units.iter().par_bridge().map(|u| {
                u.as_raw_slice().par_iter().cloned().zip(input.as_raw_slice().par_iter().cloned())
                    .reduce(|| (U::default(),U::default()), | (sum,d), (w,l) | (sum + w * l,d))
            }).map(|(r,_)| r).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))
        }).collect::<Result<Vec<Arr<U,NI>>,_>>()
    }

    pub fn batch_loss_linear_by_activaton<A: Activation<U,Arr<U,N>,Self>,const N:usize>(&self, loss:Vec<Arr<U,N>>, u:&Vec<Arr<U,N>>, activation:&A) -> Result<Vec<Arr<U, N>>, TrainingError>
    {
        loss.par_iter().zip(u.par_iter()).map(|(l,u)| {
            l.as_raw_slice().par_iter().zip(activation.derive(self,u).as_raw_slice().par_iter()).map(|(&l,&u)| {
                l * u
            }).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))
        }).collect::<Result<Vec<Arr<U,N>>,_>>()
    }

    pub fn batch_loss_linear_total<L: LossFunction<U>,const N:usize>(&self,exptected:&Vec<Arr<U,N>>,actual:&Vec<Arr<U,N>>,lossf:&L) -> U {
        actual.par_iter().zip(exptected.par_iter()).map(|(a,e)| {
            a.as_raw_slice()
             .par_iter().cloned()
             .zip(e.as_raw_slice().par_iter().cloned())
             .reduce(|| (U::default(),U::default()), |(sum,d),(a,e)| {
                 (sum + lossf.apply(a,e),d)
             })
        }).map(|(sum,_)| sum).reduce(|| U::default(), |sum,l| sum + l)
    }

    pub fn batch_forward_linear<const NI:usize,const NO:usize>(&self,input:&Vec<Arr<U,NI>>,units:&Arr2<U,NI,NO>) -> Result<Vec<Arr<U,NO>>,TrainingError> {
        input.par_iter().map(|input| {
            input.iter().zip(units.iter()).map(|(&i, unit)| {
                unit.iter().par_bridge().map(|&w| {
                    i * w
                }).collect::<Vec<U>>()
            }).collect::<Vec<Vec<U>>>()
        }).map(|o| o.par_iter().cloned().map(|o| o.try_into()).reduce(|| Ok(Arr::new()), |acc, o| {
            acc.and_then(|acc| o.and_then(|o| {
                acc.as_raw_slice()
                    .par_iter()
                    .zip(o.as_raw_slice().par_iter())
                    .map(|(&acc, &o)| acc + o).collect::<Vec<U>>().try_into()
            }))
        })).collect::<Result<Vec<Arr<U, NO>>, _>>().map_err(|e| TrainingError::from(e))
    }
}