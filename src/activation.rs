use std::collections::HashSet;
use std::marker::PhantomData;
use crate::UnitValue;
use crate::arr::*;
use crate::device::*;
use crate::lossfunction::LossFunction;

pub trait Activation<U,T,D> where U: UnitValue<U>, D: Device<U> {
    fn apply(&self,device:&D,input:&T) -> T;
    fn derive(&self,device:&D,input:&T) -> T;
    fn is_canonical_link<L: LossFunction<U>>(&self,l:&L) -> bool;
}
pub struct Identity<'a,U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>,
    c:HashSet<&'a str>
}
impl<'a,U,D> Identity<'a,U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&D) -> Identity<U,D> {
        let mut c = HashSet::new();
        c.insert("mse");

        Identity {
            u: PhantomData::<U>,
            d:PhantomData::<D>,
            c:c
        }
    }
}
impl<'a,U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Identity<'a,U,DeviceCpu<U>> where U: UnitValue<U> {
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

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
pub struct Sigmoid<'a,U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>,
    c:HashSet<&'a str>
}
impl<'a,U,D> Sigmoid<'a,U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&DeviceCpu<U>) -> Sigmoid<U,D> {
        let mut c = HashSet::new();
        c.insert("crossentropy");

        Sigmoid {
            u: PhantomData::<U>,
            d:PhantomData::<D>,
            c:c
        }
    }
}
impl<'a,U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Sigmoid<'a,U,DeviceCpu<U>> where U: UnitValue<U> {
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

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
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

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        false
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

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        false
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

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        false
    }
}
pub struct SoftMax<'a,U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>,
    c:HashSet<&'a str>
}
impl<'a,U,D> SoftMax<'a,U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&DeviceCpu<U>) -> SoftMax<U,D> {
        let mut c = HashSet::new();
        c.insert("crossentropymulticlass");

        SoftMax {
            u: PhantomData::<U>,
            d:PhantomData::<D>,
            c:c
        }
    }
}
impl<'a,U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for SoftMax<'a,U,DeviceCpu<U>> where U: UnitValue<U> {
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

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
