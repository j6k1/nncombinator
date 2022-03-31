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
pub struct ReLu<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> ReLu<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&DeviceCpu<U>) -> ReLu<U,D> {
        ReLu {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>> where U: UnitValue<U> {
    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = if *i > U::default() {
                *i
            } else {
                U::default()
            };
        }

        r
    }

    fn derive(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            if *i > U::default() {
                U::one()
            } else {
                U::default()
            };
        }

        r
    }
}
pub struct Swish<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> Swish<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&DeviceCpu<U>) -> Swish<U,D> {
        Swish {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>> where U: UnitValue<U> {
    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = *r * (U::one() / (U::one() + (-*i).exp()))
        }

        r
    }

    fn derive(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = *r * (U::one() / (U::one() + (-*i).exp())) +
                (U::one() / (U::one() + (-*i).exp())) * (U::one() - (*i * (U::one() / (U::one() + (-*i).exp()))))
        }

        r
    }
}
pub struct Tanh<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> Tanh<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&DeviceCpu<U>) -> Tanh<U,D> {
        Tanh {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>> where U: UnitValue<U> {
    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = i.tanh();
        }

        r
    }

    fn derive(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            let e = i.tanh();
            *r = U::one() - e * e;
        }

        r
    }
}
pub struct SoftMax<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> SoftMax<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&DeviceCpu<U>) -> SoftMax<U,D> {
        SoftMax {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>> where U: UnitValue<U> {
    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        let alpha = input.iter().fold(U::initial_max_value(), |m, &v| v.max(&m));
        let sum = input.iter().fold(U::default(),|acc, &x| acc + (x - alpha).exp());

        for (r,i) in r.iter_mut().zip(input.iter()) {
            let number = (*i - alpha).exp();
            *r = *i / number;
        }

        r
    }

    fn derive(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = *i * (U::one() - *i);
        }

        r
    }
}
