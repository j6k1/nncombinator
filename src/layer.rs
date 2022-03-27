use std::marker::PhantomData;
use crate::arr::Arr;
use crate::{Cons, Nil, Stack, Train};
use crate::Forward;
use crate::Backward;
use crate::Optimizer;
use crate::ForwardAll;

struct InputLayer<O> {
    o:PhantomData<O>
}
impl<O> InputLayer<O> {
    pub fn new() -> InputLayer<O> {
        InputLayer {
            o:PhantomData::<O>
        }
    }
}
impl<O> ForwardAll for InputLayer<O> {
    type Input = O;
    type Output = Self::Input;
    fn forward_all(&self,input:Self::Input) -> Self::Output {
        input
    }
}
impl<U,O,OP> Train<U,OP> for InputLayer<O> where OP: Optimizer<U> {
    type OutStack = Cons<Nil,Self::Output>;

    fn train(&mut self, input:Self::Input, optimizer:&mut OP) -> Self::OutStack {
        Cons(Nil,input)
    }
}
struct LinearLayer<U,P,const NI:usize,const NO:usize>
    where P: Forward<Arr<U,NI>>, U: Default + Clone + Copy {
    parent:P,
    units:Vec<Vec<U>>
}
impl<U,P,const NI:usize,const NO:usize> LinearLayer<U,P,NI,NO>
    where P: Forward<Arr<U,NI>>, U: Default + Clone + Copy {
    fn new<UI: FnMut() -> U, BI: FnMut() -> U>(parent:P,mut ui:UI,mut bi:BI) -> LinearLayer<U,P,NI,NO> {
        let mut units:Vec<Vec<U>> = (0..(NI)).map(|_| (0..NO).map(|_| ui()).collect()).collect();
        units.push((0..NO).map(|_| bi()).collect());

        LinearLayer {
            parent:parent,
            units: units
        }
    }
}
impl<U,P,const NI:usize,const NO:usize> Forward<Arr<U,NO>> for LinearLayer<U,P,NI,NO>
    where P: Forward<Arr<U,NI>> + ForwardAll<Output=Arr<U,NI>>, U: Default + Clone + Copy {

    type Input = Arr<U,NI>;

    fn forward(&self,input:&Self::Input) -> Arr<U,NO> {
        Arr::new()
    }
}
impl<U,P,const NI:usize,const NO:usize> ForwardAll for LinearLayer<U,P,NI,NO>
    where P: Forward<Arr<U,NI>> + ForwardAll<Output=Arr<U,NI>>,
          U: Default + Clone + Copy {
    type Input = <P as ForwardAll>::Input;
    type Output = Arr<U,NO>;
    fn forward_all(&self, input: Self::Input) -> Arr<U,NO> {
        self.forward(&self.parent.forward_all(input))
    }
}
impl<U,P,OP,const NI:usize,const NO:usize> Train<U,OP> for LinearLayer<U,P,NI,NO>
    where P: Train<U,OP> + Forward<Arr<U,NI>> + ForwardAll<Output=Arr<U,NI>>,
          U: Default + Clone + Copy,
          OP: Optimizer<U> {
    type OutStack = Cons<<P as Train<U,OP>>::OutStack,Self::Output>;

    fn train(&mut self, input: Self::Input, optimizer: &mut OP) -> Self::OutStack {
        let r = self.parent.train(input, optimizer);
        let u = r.map(|r| self.forward(r));

        Cons(r,u)
    }
}
