use std::marker::PhantomData;
use crate::arr::{Arr, Arr2};
use crate::lossfunction::LossFunction;
use crate::UnitValue;

pub trait Device<U>: Clone where U: UnitValue<U> {
    fn forward_linear<const NI:usize,const NO:usize>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:&Arr<U,NI>) -> Arr<U,NO>;
    fn backward_liner<const NI:usize,const NO:usize>(&self,units:&Arr2<U,NI,NO>,input:&Arr<U,NO>) -> Arr<U,NI>;
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

    fn backward_liner<const NI:usize, const NO: usize>(&self, units: &Arr2<U, NI, NO>, input: &Arr<U, NO>) -> Arr<U, NI> {
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