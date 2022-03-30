use std::marker::PhantomData;
use crate::arr::{Arr, Arr2};
use crate::{Cons, Nil, Stack};
use crate::ope::UnitValue;
use crate::Optimizer;

pub trait Forward<O> {
    type Input;
    fn forward(&self,input:&Self::Input) -> O;
}
pub trait ForwardAll {
    type Input;
    type Output;
    fn forward_all(&self,input:Self::Input) -> Self::Output;
}
pub trait Backward<I,U,S>: PreTrain<U> where U: UnitValue<U>, S: Stack {
    fn backward<OP: Optimizer<U>>(&mut self,input:I,stack:S,optimizer:&mut OP);
}
pub trait PreTrain<U>: ForwardAll where U: UnitValue<U> {
    type OutStack: Stack<Head=Self::Output>;
    fn pre_train<OP: Optimizer<U>>(&mut self, input:Self::Input, optimizer:&mut OP) -> Self::OutStack;
}
pub trait Train<U>: PreTrain<U> where U: UnitValue<U> {
    fn train<OP: Optimizer<U>>(&mut self, input:Self::Input, optimizer:&mut OP);
}
pub trait AddLayer: ForwardAll where Self: Sized {
    fn add_layer<C,F>(self,f:F) -> C where C: ForwardAll, F: FnOnce(Self) -> C;
}
pub trait AddLayerTrain<U>: PreTrain<U> where Self: Sized, U: UnitValue<U> {
    fn add_layer_train<C,F>(self,f:F) -> C where C: Train<U>, F: FnOnce(Self) -> C;
}
impl<T> AddLayer for T where T: ForwardAll + Sized {
    fn add_layer<C, F>(self, f: F) -> C where C: ForwardAll, F: FnOnce(Self) -> C {
        f(self)
    }
}
impl<T,U> AddLayerTrain<U> for T where T: PreTrain<U> + Sized, U: UnitValue<U> {
    fn add_layer_train<C, F>(self, f: F) -> C where C: Train<U>, F: FnOnce(Self) -> C {
        f(self)
    }
}
pub struct InputLayer<U,O> where U: UnitValue<U> {
    u:PhantomData<U>,
    o:PhantomData<O>
}
impl<U,O> InputLayer<U,O> where U: UnitValue<U> {
    pub fn new() -> InputLayer<U,O> {
        InputLayer {
            u:PhantomData::<U>,
            o:PhantomData::<O>
        }
    }
}
impl<U,O> ForwardAll for InputLayer<U,O> where U: UnitValue<U> {
    type Input = O;
    type Output = Self::Input;
    fn forward_all(&self,input:Self::Input) -> Self::Output {
        input
    }
}
impl<U,O> PreTrain<U> for InputLayer<U,O> where U: UnitValue<U> {
    type OutStack = Cons<Nil,Self::Output>;

    fn pre_train<OP: Optimizer<U>>(&mut self, input:Self::Input, optimizer:&mut OP) -> Self::OutStack {
        Cons(Nil,input)
    }
}
pub struct LinearLayer<U,P,const NI:usize,const NO:usize>
    where P: ForwardAll, U: Default + Clone + Copy + UnitValue<U> {
    parent:P,
    units:Arr2<U,NI,NO>,
    bias:Arr<U,NO>
}
impl<U,P,const NI:usize,const NO:usize> LinearLayer<U,P,NI,NO>
    where P: ForwardAll, U: Default + Clone + Copy + UnitValue<U> {
    pub fn new<UI: FnMut() -> U, BI: FnMut() -> U>(parent:P,mut ui:UI,mut bi:BI) -> LinearLayer<U,P,NI,NO> {
        let mut units:Arr2<U,NI,NO> = Arr2::new();
        let mut bias:Arr<U,NO> = Arr::new();

        for mut it in units.iter_mut() {
            for it in it.iter_mut() {
                *it = ui();
            }
        }

        for it in bias.iter_mut() {
            *it = bi();
        }

        LinearLayer {
            parent:parent,
            units: units,
            bias:bias
        }
    }
}
impl<U,P,const NI:usize,const NO:usize> Forward<Arr<U,NO>> for LinearLayer<U,P,NI,NO>
    where P: ForwardAll<Output=Arr<U,NI>>, U: Default + Clone + Copy + UnitValue<U> {

    type Input = Arr<U,NI>;

    fn forward(&self,input:&Self::Input) -> Arr<U,NO> {
        let mut output:Arr<U,NO> = Arr::new();

        let bias = U::bias();

        for (mut o,w) in output.iter_mut().zip(self.bias.iter()) {
            *o += bias * *w;
        }

        for (i,u) in input.iter().zip(self.units.iter()) {
            for (o,w) in output.iter_mut().skip(1).zip(u.iter()) {
                *o += *i * *w;
            }
        }

        output
    }
}
impl<U,P,const NI:usize,const NO:usize> ForwardAll for LinearLayer<U,P,NI,NO>
    where P: ForwardAll<Output=Arr<U,NI>>,
          U: Default + Clone + Copy + UnitValue<U> {
    type Input = <P as ForwardAll>::Input;
    type Output = Arr<U,NO>;
    fn forward_all(&self, input: Self::Input) -> Arr<U,NO> {
        self.forward(&self.parent.forward_all(input))
    }
}
impl<U,P,const NI:usize,const NO:usize> PreTrain<U> for LinearLayer<U,P,NI,NO>
    where P: PreTrain<U> + ForwardAll<Output=Arr<U,NI>>,
          U: Default + Clone + Copy + UnitValue<U> {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::Output>;

    fn pre_train<OP: Optimizer<U>>(&mut self, input: Self::Input, optimizer: &mut OP) -> Self::OutStack {
        let r = self.parent.pre_train(input, optimizer);
        let u = r.map(|r| self.forward(r));

        Cons(r,u)
    }
}
impl<U,P,const NI:usize,const NO:usize> Train<U> for LinearLayer<U,P,NI,NO>
    where P: PreTrain<U> + ForwardAll<Output=Arr<U,NI>>,
          U: Default + Clone + Copy + UnitValue<U> {
    fn train<OP: Optimizer<U>>(&mut self, input: Self::Input, optimizer: &mut OP) {
        let r = self.pre_train(input, optimizer);
    }
}
