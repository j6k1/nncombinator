use crate::UnitValue;

pub trait LossFunction<U> where U: Clone + Copy + UnitValue<U> {
    fn derive(r:U,t:U) -> U;
}
