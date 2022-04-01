use crate::UnitValue;

pub trait LossFunction<U> where U: Clone + Copy + UnitValue<U> {
    fn derive(&self,r:U,t:U) -> U;
    fn name(&self) -> &'static str;
}
