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
pub trait BackwardAll<U>: Train<U> where U: UnitValue<U> {
    type LossInput;
    fn backward_all<OP: Optimizer<U>>(&mut self,input:Self::LossInput, optimizer:&mut OP);
    fn derive(&mut self,input:&Self::LossInput) -> Self::LossInput;
}
pub trait Backward<U,I,O> where U: UnitValue<U> {
    fn backward(&mut self, loss:I) -> O;
}
pub trait PreTrain<U>: ForwardAll where U: UnitValue<U> {
    type OutStack: Stack<Head=Self::Output>;
    fn pre_train(&mut self, input:Self::Input) -> Self::OutStack;
}
pub trait Train<U>: PreTrain<U> where U: UnitValue<U> {
    fn train<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input:Self::Input, stack:Self::OutStack, optimizer:&mut OP, lossf:&L);
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
        todo!()
    }
}
impl<U,P,A,T,D> Train<U> for ActivationLayer<U,P,A,T,D>
    where P: PreTrain<U> + ForwardAll<Output=T>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,T,D> {
    fn train<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::Input, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) {
        let r = self.pre_train(input);

        let (s,input) = r.pop();

        //input.map(|input| self.backward_all(input,optimizer));
    }
}
impl<U,P,A,T,D> BackwardAll<U> for ActivationLayer<U,P,A,T,D>
    where P: PreTrain<U> + ForwardAll<Output=T> + BackwardAll<U,LossInput=T>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,T,D> {
    type LossInput = T;

    fn backward_all<OP: Optimizer<U>>(&mut self, input: Self::LossInput, optimizer: &mut OP) {
        let loss = self.backward(input);
        self.parent.backward_all(loss,optimizer);
    }
    fn derive(&mut self,input:&Self::LossInput) -> Self::LossInput {
        self.f.derive(&self.device,input)
    }
}
impl<U,P,A,T,D> Backward<U,T,T> for ActivationLayer<U,P,A,T,D>
    where P: PreTrain<U> + ForwardAll<Output=T>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,T,D> {
    fn backward(&mut self, input: T) -> T {
        self.f.derive(&self.device,&input)
    }
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
impl<U,O> Train<U> for InputLayer<U,O> where U: UnitValue<U> {
    fn train<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::Input, stack: Self::OutStack, optimizer: &mut OP, lossf: &L) {
        
    }
}
impl<U,O> BackwardAll<U> for InputLayer<U,O> where U: UnitValue<U> {
    type LossInput = ();

    fn backward_all<OP: Optimizer<U>>(&mut self, input: Self::LossInput, optimizer: &mut OP) {
        
    }
    fn derive(&mut self,input:&Self::LossInput) -> Self::LossInput {
        ()
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
impl<U,P,D,const NI:usize,const NO:usize> Train<U> for LinearLayer<U,P,D,NI,NO>
    where P: PreTrain<U> + ForwardAll<Output=Arr<U,NI>> + BackwardAll<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    fn train<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::Input, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) {
        let r = self.pre_train(input);

        let (s,input) = r.pop();

        input.map(|input| self.backward_all(input,optimizer));
    }
}
impl<U,P,D,const NI:usize,const NO:usize> Backward<U,Arr<U,NO>,Arr<U,NI>> for LinearLayer<U,P,D,NI,NO>
    where U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          P: ForwardAll {
    fn backward(&mut self, loss: Arr<U,NO>) -> Arr<U,NI> {
        todo!()
    }
}
impl<U,P,D,const NI:usize,const NO:usize> BackwardAll<U> for LinearLayer<U,P,D,NI,NO>
    where P: BackwardAll<U> + ForwardAll<Output=Arr<U, NI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> {
    type LossInput = Arr<U,NO>;

    fn backward_all<OP: Optimizer<U>>(&mut self, input: Self::LossInput, optimizer: &mut OP) {
        self.backward(input);
    }

    fn derive(&mut self, input: &Self::LossInput) -> Self::LossInput {
        let mut r = Arr::new();

        for it in r.iter_mut() {
            *it = U::one();
        }

        r
    }
}
