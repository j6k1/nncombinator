use std::marker::PhantomData;
use crate::UnitValue;
use crate::arr::*;
use crate::device::*;

pub trait Activation<U,T,D> where U: UnitValue<U>, D: Device<U> {
    fn apply(&self,device:&D,input:&T) -> T;
    fn derive(&self,device:&D,input:&T) -> T;
}
pub struct Identity<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> Identity<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&D) -> Identity<U,D> {
        Identity {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>> where U: UnitValue<U> {
    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        (*input).clone()
    }

    fn derive(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for it in r.iter_mut() {
            *it = U::one()
        }

        r
    }
}
pub struct Sigmoid<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> Sigmoid<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&DeviceCpu<U>) -> Sigmoid<U,D> {
        Sigmoid {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>> where U: UnitValue<U> {
    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = U::one() / (U::one() + (-*i).exp());
        }

        r
    }

    fn derive(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            let e = U::one() / (U::one() + (-*i).exp());
            *r = e * (U::one() - e);
        }

        r
    }
}
