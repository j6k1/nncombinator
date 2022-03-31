use crate::UnitValue;

pub trait Activation<U,T> where U: UnitValue<U> {
    fn apply(&self,input:&T) -> T;
    fn derive(&self,input:&T) -> T;
}
