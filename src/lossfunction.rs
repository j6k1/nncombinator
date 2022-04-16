use std::marker::PhantomData;
use crate::UnitValue;

pub trait LossFunction<U>: Send + Sync + 'static where U: Clone + Copy + UnitValue<U> {
    fn derive(&self,r:U,t:U) -> U;
    fn apply(&self,r:U,t:U) -> U;
    fn name(&self) -> &'static str;
}
pub struct Mse<U> where U: Clone + Copy + UnitValue<U> {
    u:PhantomData<U>
}
impl<U> Mse<U> where U: UnitValue<U> {
    pub fn new() -> Mse<U> {
        Mse {
            u:PhantomData::<U>
        }
    }
}
impl<U> LossFunction<U> for Mse<U> where U: Clone + Copy + UnitValue<U> {
    fn derive(&self, r: U, t: U) -> U {
        r - t
    }

    fn apply(&self, r: U, t: U) -> U {
        (r - t) * (r - t) / U::from_f64(2.).unwrap()
    }

    fn name(&self) -> &'static str {
        "mse"
    }
}
pub struct CrossEntropy<U>  where U: Clone + Copy + UnitValue<U> {
    u:PhantomData<U>
}
impl<U> CrossEntropy<U> where U: Clone + Copy + UnitValue<U> {
    pub fn new() -> CrossEntropy<U> {
        CrossEntropy {
            u:PhantomData::<U>
        }
    }
}
impl<U> LossFunction<U> for CrossEntropy<U> where U: Clone + Copy + UnitValue<U> {
    fn derive(&self, r: U, t: U) -> U {
        -(r / (t + U::from_f64(1e-7).unwrap())) + (U::one() - t) / (U::one() - r)
    }

    fn apply(&self, r: U, t: U) -> U {
        let rl = r.ln();

        if rl == U::neg_infinity() || rl == U::infinity() {
            (U::one() - t) * (U::one() - r).ln()
        } else {
            -t * rl + (U::one() - t) * (U::one() - r).ln()
        }
    }

    fn name(&self) -> &'static str {
        "crossentropy"
    }
}
pub struct CrossEntropyMulticlass<U> where U: Clone + Copy + UnitValue<U> {
    u:PhantomData<U>
}
impl<U> CrossEntropyMulticlass<U> where U: Clone + Copy + UnitValue<U>{
    pub fn new() -> CrossEntropyMulticlass<U> {
        CrossEntropyMulticlass {
            u:PhantomData::<U>
        }
    }
}
impl<U> LossFunction<U> for CrossEntropyMulticlass<U> where U: Clone + Copy + UnitValue<U> {
    fn derive(&self, r: U, t: U) -> U {
        -t / r
    }

    fn apply(&self, r: U, t: U) -> U {
        let rl = r.ln();

        if rl == U::neg_infinity() || rl == U::infinity() {
            U::default()
        } else {
            -t * rl
        }
    }

    fn name(&self) -> &'static str {
        "crossentropymulticlass"
    }
}