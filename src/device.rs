use std::marker::PhantomData;
use crate::activation::Activation;
use crate::arr::{Arr, Arr2};
use crate::layer::{ForwardAll, LinearLayer};
use crate::lossfunction::LossFunction;
use crate::UnitValue;

pub trait Device<U>: Clone where U: UnitValue<U> {
    fn forward_linear<const NI:usize,const NO:usize>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:&Arr<U,NI>) -> Arr<U,NO>;
    fn loss_linear<L,const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>, f: &Arr<U,N>, lossf: &L) -> Arr<U, N>
        where L: LossFunction<U>;
}
pub struct DeviceCpu<U> where U: UnitValue<U> {
    u:PhantomData<U>
}
impl<U> DeviceCpu<U> where U: UnitValue<U> {
    pub fn new() -> DeviceCpu<U> {
        DeviceCpu {
            u:PhantomData::<U>
        }
    }
}
impl<U> Device<U> for DeviceCpu<U> where U: UnitValue<U> {
    fn forward_linear<const NI:usize,const NO:usize>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:&Arr<U,NI>) -> Arr<U,NO> {
        let mut output:Arr<U,NO> = Arr::new();

        let b = U::bias();

        for (o,w) in output.iter_mut().zip(bias.iter()) {
            *o += b * *w;
        }

        for (i,u) in input.iter().zip(units.iter()) {
            for (o,w) in output.iter_mut().skip(1).zip(u.iter()) {
                *o += *i * *w;
            }
        }

        output
    }

    fn loss_linear<L,const N: usize>(&self, expected: &Arr<U, N>, actual: &Arr<U, N>, f: &Arr<U,N>, lossf: &L) -> Arr<U, N>
        where L: LossFunction<U> {

        let mut loss = Arr::new();

        for (((a, e), loss),f) in actual.iter()
                                                            .zip(expected.iter())
                                                            .zip(loss.iter_mut()).zip(f.iter()) {
            *loss = lossf.derive(*a, *e) * *f;
        }

        loss
    }
}
impl<U> Clone for DeviceCpu<U> where U: UnitValue<U> {
    fn clone(&self) -> Self {
        DeviceCpu {
            u:PhantomData::<U>
        }
    }
}