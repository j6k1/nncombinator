use std::marker::PhantomData;
use crate::arr::*;
use crate::device::*;
use crate::{Cons, Nil, Stack};
use crate::activation::Activation;
use crate::ope::UnitValue;
use crate::lossfunction::*;
use crate::optimizer::*;

pub trait Forward<I,O> {
    fn forward(&self,input:&I) -> O;
}
pub trait ForwardAll {
    type Input;
    type Output;
    fn forward_all(&self,input:Self::Input) -> Self::Output;
}
pub trait BackwardAll<U>: PreTrain<U> where U: UnitValue<U> {
    fn backward_all<OP: Optimizer<U>>(&mut self,input:Self::Output, stack:Self::OutStack, optimizer:&mut OP);
    fn is_canonical_link<L: LossFunction<U>>(&self,_:&L) -> bool {
        false
    }
}
pub trait Loss<U>: BackwardAll<U> where U: UnitValue<U> {
    fn loss<L: LossFunction<U>>(&mut self,loss:&Self::Output,lossf:&L,stack:Self::OutStack) -> (Self::OutStack,Self::Output);
}
pub trait Backward<U,I,O> where U: UnitValue<U> {
    fn backward(&mut self, input:I) -> O;
}
pub trait PreTrain<U>: ForwardAll where U: UnitValue<U> {
    type OutStack: Stack<Head=Self::Output> + Sized;
    fn pre_train(&mut self, input:Self::Input) -> Self::OutStack;
}
pub trait Train<U>: PreTrain<U> where U: UnitValue<U> {
    fn train<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, expected:Self::Output, input:Self::Input, optimizer:&mut OP, lossf:&L);
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

    fn pre_train(&mut self, input:Self::Input) -> Self::OutStack {
        Cons(Nil,input)
    }
}
impl<U,const N:usize> BackwardAll<U> for InputLayer<U,Arr<U,N>> where U: UnitValue<U> {
    fn backward_all<OP: Optimizer<U>>(&mut self, _: Self::Output, _:Self::OutStack, _: &mut OP) {
        
    }
}
pub struct ActivationLayer<U,P,A,T,D> where P: ForwardAll,
                                            U: UnitValue<U>,
                                            D: Device<U>,
                                            A: Activation<U,T,D> {
    parent:P,
    f:A,
    device:D,
    u:PhantomData<U>,
    t:PhantomData<T>,
}
impl<U,P,A,T,D> ActivationLayer<U,P,A,T,D> where P: ForwardAll<Output=T>,
                                                 U: UnitValue<U>,
                                                 D: Device<U>,
                                                 A: Activation<U,T,D> {
    pub fn new(parent:P,f:A,device:&D) -> ActivationLayer<U,P,A,T,D> {
        ActivationLayer {
            parent:parent,
            f:f,
            device:device.clone(),
            u:PhantomData::<U>,
            t:PhantomData::<T>,
        }
    }
}
impl<U,P,A,T,D> ForwardAll for ActivationLayer<U,P,A,T,D>
    where P: ForwardAll<Output=T>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,T,D> {
    type Input = <P as ForwardAll>::Input;
    type Output = <P as ForwardAll>::Output;
    fn forward_all(&self, input: Self::Input) -> Self::Output {
        self.forward(&self.parent.forward_all(input))
    }
}
impl<U,P,A,T,D> Forward<<P as ForwardAll>::Output,T> for ActivationLayer<U,P,A,T,D>
    where P: ForwardAll<Output=T>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,T,D> {
    fn forward(&self, input: &<P as ForwardAll>::Output) -> T {
        self.f.apply(&self.device,input)
    }
}
impl<U,P,A,T,D> PreTrain<U> for ActivationLayer<U,P,A,T,D>
    where P: PreTrain<U> + ForwardAll<Output=T>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,T,D> {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack, Self::Output>;

    fn pre_train(&mut self, input: Self::Input) -> Self::OutStack {
        let r = self.parent.pre_train(input);
        let u = r.map(|r| self.forward(r));

        Cons(r,u)
    }
}
impl<U,P,A,T,D> BackwardAll<U> for ActivationLayer<U,P,A,T,D>
    where P: PreTrain<U> + ForwardAll<Output=T> + BackwardAll<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,T,D> {
    fn backward_all<OP: Optimizer<U>>(&mut self, input: Self::Output, stack:Self::OutStack, optimizer: &mut OP) {
        let (s,_) = stack.pop();

        self.parent.backward_all(input, s, optimizer);
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.f.is_canonical_link(l)
    }
}
impl<U,P,A,D,const N:usize> Loss<U> for ActivationLayer<U,P,A,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Output=Arr<U,N>> + BackwardAll<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,Arr<U,N>,D>,
          Self: ForwardAll<Output=Arr<U,N>> {
    fn loss<L: LossFunction<U>>(&mut self, loss: &Self::Output, lossf:&L, stack: Self::OutStack) -> (Self::OutStack, Self::Output) {
        let (s,actual) = stack.pop();
        let mut r = Arr::new();

        s.map(|u| {
            for (r, (l,u)) in r.iter_mut()
                                            .zip(loss.iter().zip(self.f.derive(&self.device,u).iter())) {
                *r = *l * *u;
            }
        });

        (Cons(s,actual),r)
    }
}
pub struct LinearOutputLayer<U,P,D,O>
    where P: ForwardAll, U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    u:PhantomData<U>,
    o:PhantomData<O>,
    parent:P,
    _device:D,
}
impl<U,P,D,O> LinearOutputLayer<U,P,D,O>
    where P: ForwardAll, U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    pub fn new(parent:P,device:&D) -> LinearOutputLayer<U,P,D,O> {
        LinearOutputLayer {
            u:PhantomData::<U>,
            o:PhantomData::<O>,
            parent:parent,
            _device:device.clone(),
        }
    }
}
impl<U,P,D,O> ForwardAll for LinearOutputLayer<U,P,D,O>
    where P: ForwardAll<Output=O>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    type Input = P::Input;
    type Output = P::Output;
    fn forward_all(&self, input: Self::Input) -> Self::Output {
        self.parent.forward_all(input)
    }
}
impl<U,P,D,O> PreTrain<U> for LinearOutputLayer<U,P,D,O>
    where P: PreTrain<U> + ForwardAll<Output=O>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    type OutStack = P::OutStack;

    fn pre_train(&mut self, input: Self::Input) -> Self::OutStack {
        self.parent.pre_train(input)
    }
}
impl<U,P,D,const N:usize> BackwardAll<U> for LinearOutputLayer<U,P,D,Arr<U,N>>
    where P: BackwardAll<U> + ForwardAll<Output=Arr<U,N>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    fn backward_all<OP: Optimizer<U>>(&mut self, input: Self::Output, stack:Self::OutStack, optimizer: &mut OP) {
        self.parent.backward_all(input, stack, optimizer);
    }
}
impl<U,P,D,const N:usize> Train<U> for LinearOutputLayer<U,P,D,Arr<U,N>>
    where P: BackwardAll<U> + ForwardAll<Output=Arr<U,N>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          Self: ForwardAll<Output=Arr<U,N>> {
    fn train<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, expected: Self::Output, input: Self::Input, optimizer: &mut OP, lossf: &L) {
        let stack = self.pre_train(input);

        let (stack,loss) = if self.parent.is_canonical_link(lossf) {

            let loss = stack.map(|actual| {
                let mut loss = Arr::new();

                for (l, (a, e)) in loss.iter_mut().zip(actual.iter().zip(expected.iter())) {
                    *l = *a - *e;
                }

                loss
            });

            (stack,loss)
        } else {
            let loss = stack.map(|actual| {
                let mut loss = Arr::new();

                for (l, (a, e)) in loss.iter_mut().zip(actual.iter().zip(expected.iter())) {
                    *l = lossf.derive(*a,*e);
                }

                loss
            });

            self.parent.loss(&loss,lossf,stack)
        };

        self.backward_all(loss,stack,optimizer);
    }
}
pub struct LinearLayer<U,P,D,const NI:usize,const NO:usize>
    where P: ForwardAll, U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    parent:P,
    device:D,
    units:Arr2<U,NI,NO>,
    bias:Arr<U,NO>
}
impl<U,P,D,const NI:usize,const NO:usize> LinearLayer<U,P,D,NI,NO>
    where P: ForwardAll, U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    pub fn new<UI: FnMut() -> U, BI: FnMut() -> U>(parent:P,device:&D,mut ui:UI,mut bi:BI) -> LinearLayer<U,P,D,NI,NO> {
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
            device:device.clone(),
            units: units,
            bias:bias
        }
    }
}
impl<U,P,D,const NI:usize,const NO:usize> Forward<Arr<U,NI>,Arr<U,NO>> for LinearLayer<U,P,D,NI,NO>
    where P: ForwardAll<Output=Arr<U,NI>>, U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {

    fn forward(&self,input:&Arr<U,NI>) -> Arr<U,NO> {
        self.device.forward_linear(&self.bias,&self.units,input)
    }
}
impl<U,P,D,const NI:usize,const NO:usize> ForwardAll for LinearLayer<U,P,D,NI,NO>
    where P: ForwardAll<Output=Arr<U,NI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    type Input = <P as ForwardAll>::Input;
    type Output = Arr<U,NO>;
    fn forward_all(&self, input: Self::Input) -> Arr<U,NO> {
        self.forward(&self.parent.forward_all(input))
    }
}
impl<U,P,D,const NI:usize,const NO:usize> PreTrain<U> for LinearLayer<U,P,D,NI,NO>
    where P: PreTrain<U> + ForwardAll<Output=Arr<U,NI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::Output>;

    fn pre_train(&mut self, input: Self::Input) -> Self::OutStack {
        let r = self.parent.pre_train(input);
        let u = r.map(|r| self.forward(r));

        Cons(r,u)
    }
}
impl<U,P,D,const NI:usize,const NO:usize> Backward<U,Arr<U,NO>,Arr<U,NI>> for LinearLayer<U,P,D,NI,NO>
    where U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          P: ForwardAll + BackwardAll<U>{
    fn backward(&mut self, input: Arr<U,NO>) -> Arr<U,NI> {
        self.device.backward_liner(&self.bias,&self.units,&input)
    }
}
impl<U,P,D,const NI:usize,const NO:usize> BackwardAll<U> for LinearLayer<U,P,D,NI,NO>
    where P: BackwardAll<U> + ForwardAll<Output=Arr<U,NI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    fn backward_all<OP: Optimizer<U>>(&mut self, input: Self::Output, stack:Self::OutStack, optimizer: &mut OP) {
        let (s,o) = stack.pop();

        let loss = self.backward(input);

        self.parent.backward_all(loss, s, optimizer);
    }
}
impl<U,P,D,const NI:usize,const NO:usize> Loss<U> for LinearLayer<U,P,D,NI,NO>
    where P: BackwardAll<U> + ForwardAll<Output=Arr<U,NI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    fn loss<L: LossFunction<U>>(&mut self, loss: &Self::Output, lossf: &L, stack: Self::OutStack) -> (Self::OutStack, Self::Output) {
        (stack,loss.clone())
    }
}
