use crate::UnitValue;

pub trait Optimizer<U> where U: Clone + Copy + UnitValue<U> {
    fn update(&mut self,e:U,w:&mut U) -> U;
}
