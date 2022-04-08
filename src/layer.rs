use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::*;
use crate::device::*;
use crate::persistence::*;
use crate::{Cons, Nil, Stack};
use crate::activation::Activation;
use crate::error::ConfigReadError;
use crate::ope::UnitValue;
use crate::lossfunction::*;
use crate::optimizer::*;

pub enum DiffInput<T,U,const NI:usize,const NO:usize> where U: UnitValue<U> {
    Diff(T,Arr<U,NO>),
    NotDiff(Arr<U,NI>)
}
pub trait Forward<I,O> {
    fn forward(&self,input:&I) -> O;
}
pub trait ForwardAll {
    type Input: Debug;
    type Output: Debug;
    fn forward_all(&self,input:Self::Input) -> Self::Output;
}
pub trait BackwardAll<U>: PreTrain<U> where U: UnitValue<U> {
    type LossInput: Debug;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self,input:Self::LossInput, stack:Self::OutStack, optimizer:&mut OP, lossf:&L);
    fn is_canonical_link<L: LossFunction<U>>(&self,_:&L) -> bool {
        false
    }
}
pub trait Loss<U>: BackwardAll<U> where U: UnitValue<U> {
    fn loss<L: LossFunction<U>>(&mut self,loss:Self::LossInput,_:&L,stack:Self::OutStack) -> (Self::OutStack,Self::LossInput) {
        (stack,loss)
    }
}
pub trait Backward<U,I,O> where U: UnitValue<U> {
    fn backward(&mut self, input:I) -> O;
}
pub trait PreTrain<U>: ForwardAll where U: UnitValue<U> {
    type OutStack: Stack<Head=Self::Output> + Sized + Debug;
    fn pre_train(&mut self, input:Self::Input) -> Self::OutStack;
}
pub trait ForwardDiff<U>: PreTrain<U> where U: UnitValue<U> {
    fn forward_diff(&mut self, input:Self::Input) -> Self::OutStack;
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
impl<T,U> ForwardDiff<U> for T where T: PreTrain<U> + Sized, U: UnitValue<U> {
    fn forward_diff(&mut self, input: Self::Input) -> Self::OutStack {
        self.pre_train(input)
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
impl<U,O> Persistence<U,TextFilePersistence<U>> for InputLayer<U,O>
    where U: UnitValue<U> + FromStr + Sized {
    fn load(&mut self, _: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut TextFilePersistence<U>) {

    }
}
impl<U,O> ForwardAll for InputLayer<U,O> where U: UnitValue<U>, O: Debug {
    type Input = O;
    type Output = O;
    fn forward_all(&self,input:Self::Input) -> Self::Output {
        input
    }
}
impl<U,O> PreTrain<U> for InputLayer<U,O> where U: UnitValue<U>, O: Debug {
    type OutStack = Cons<Nil,Self::Output>;

    fn pre_train(&mut self, input:Self::Input) -> Self::OutStack {
        Cons(Nil,input)
    }
}
impl<U,O> BackwardAll<U> for InputLayer<U,O> where U: UnitValue<U>, O: Debug {
    type LossInput = O;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, _: Self::LossInput, _:Self::OutStack, _: &mut OP, _:&L) {
        
    }
}
impl<U,O> Loss<U> for InputLayer<U,O> where U: UnitValue<U>, O: Debug {}
pub struct ActivationLayer<U,P,A,I,PI,D> where P: ForwardAll,
                                            U: UnitValue<U>,
                                            D: Device<U>,
                                            A: Activation<U,PI,D>,
                                            PI: Debug,
                                            I: Debug {
    parent:P,
    f:A,
    device:D,
    u:PhantomData<U>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
}
impl<U,P,A,I,PI,D> ActivationLayer<U,P,A,I,PI,D> where P: ForwardAll<Output=PI>,
                                                 U: UnitValue<U>,
                                                 D: Device<U>,
                                                 A: Activation<U,PI,D>,
                                                 PI: Debug,
                                                 I: Debug {
    pub fn new(parent:P,f:A,device:&D) -> ActivationLayer<U,P,A,I,PI,D> {
        ActivationLayer {
            parent:parent,
            f:f,
            device:device.clone(),
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            pi:PhantomData::<PI>,
        }
    }
}
impl<U,P,A,I,PI,D> Persistence<U,TextFilePersistence<U>> for ActivationLayer<U,P,A,I,PI,D>
    where P: ForwardAll<Output=PI> + Persistence<U,TextFilePersistence<U>>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr + Sized,
          D: Device<U>,
          A: Activation<U,PI,D>,
          PI: Debug,
          I: Debug {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) {
        self.parent.save(persistence)
    }
}
impl<U,P,A,I,PI,D> ForwardAll for ActivationLayer<U,P,A,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,PI,D>,
          PI: Debug,
          I: Debug {
    type Input = I;
    type Output = PI;
    fn forward_all(&self, input: Self::Input) -> Self::Output {
        self.forward(&self.parent.forward_all(input))
    }
}
impl<U,P,A,I,PI,D> Forward<PI,PI> for ActivationLayer<U,P,A,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,PI,D>,
          PI: Debug,
          I: Debug {
    fn forward(&self, input: &PI) -> PI {
        self.f.apply(&self.device,input)
    }
}
impl<U,P,A,D,I,const N:usize> PreTrain<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,Arr<U,N>,D>,
          I: Debug {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack, Self::Output>;

    fn pre_train(&mut self, input: Self::Input) -> Self::OutStack {
        let r = self.parent.pre_train(input);

        let u = r.map(|r| self.forward(r));

        Cons(r,u)
    }
}
impl<U,P,A,D,I,const N:usize> BackwardAll<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,Arr<U,N>,D>,
          I: Debug {
    type LossInput = Arr<U,N>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) {
        let (s,_) = stack.pop();

        self.parent.backward_all(input, s, optimizer,lossf);
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.f.is_canonical_link(l)
    }
}
impl<U,P,A,D,I,const N:usize> Loss<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,Arr<U,N>,D>,
          I: Debug {
    fn loss<L: LossFunction<U>>(&mut self, loss: Arr<U,N>, _:&L, stack: Self::OutStack) -> (Self::OutStack, Self::LossInput) {
        let (s,actual) = stack.pop();
        let mut r = Arr::new();

        s.map(|u| {
            for (r, (l,u)) in r.iter_mut()
                                                .zip(loss.iter().zip(self.f.derive(&self.device,u).iter())) {
                *r += *l * *u;
            }
        });

        (Cons(s,actual),r)
    }
}
pub struct LinearOutputLayer<U,P,D,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          IO: Debug,
          I: Debug {
    u:PhantomData<U>,
    i:PhantomData<I>,
    io:PhantomData<IO>,
    parent:P,
    device:D,
}
impl<U,P,D,I,IO> LinearOutputLayer<U,P,D,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          IO: Debug,
          I: Debug {
    pub fn new(parent:P,device:&D) -> LinearOutputLayer<U,P,D,I,IO> {
        LinearOutputLayer {
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            io:PhantomData::<IO>,
            parent:parent,
            device:device.clone(),
        }
    }
}
impl<U,P,D,I,IO> Persistence<U,TextFilePersistence<U>> for LinearOutputLayer<U,P,D,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U> + Persistence<U,TextFilePersistence<U>>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr + Sized,
          D: Device<U>,
          IO: Debug,
          I: Debug {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;
        persistence.verify_eof()
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) {
        self.parent.save(persistence)
    }
}
impl<U,P,D,I,IO> ForwardAll for LinearOutputLayer<U,P,D,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          IO: Debug,
          I: Debug {
    type Input = I;
    type Output = IO;
    fn forward_all(&self, input: Self::Input) -> Self::Output {
        self.parent.forward_all(input)
    }
}
impl<U,P,D,I,IO> PreTrain<U> for LinearOutputLayer<U,P,D,I,IO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=IO> + BackwardAll<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          IO: Debug,
          I: Debug {
    type OutStack = P::OutStack;

    fn pre_train(&mut self, input: Self::Input) -> Self::OutStack {
        self.parent.pre_train(input)
    }
}
impl<U,P,D,I,const NO:usize> BackwardAll<U> for LinearOutputLayer<U,P,D,I,Arr<U,NO>>
    where P: BackwardAll<U,LossInput=Arr<U,NO>> + ForwardAll<Input=I,Output=Arr<U,NO>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    type LossInput = Arr<U,NO>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) {
        self.parent.backward_all(input, stack, optimizer, lossf);
    }
}
impl<U,P,D,I,const NO:usize> Train<U> for LinearOutputLayer<U,P,D,I,Arr<U,NO>>
    where P: BackwardAll<U,LossInput=Arr<U,NO>> + ForwardAll<Input=I,Output=Arr<U,NO>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    fn train<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, expected: Self::Output, input: Self::Input, optimizer: &mut OP, lossf: &L) {
        let stack = self.pre_train(input);

        let (stack,loss) = if self.parent.is_canonical_link(lossf) {
            let loss = stack.map(|actual| {
                self.device.loss_linear_by_canonical_link(&expected, &actual)
            });

            (stack,loss)
        } else {
            let loss = stack.map(|actual| {
                self.device.loss_linear(&expected,&actual,lossf)
            });

            self.parent.loss(loss,lossf,stack)
        };

        self.backward_all(loss,stack,optimizer,lossf);
    }
}
pub struct LinearLayer<U,P,D,I,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>>, U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    parent:P,
    device:D,
    units:Arr2<U,NI,NO>,
    bias:Arr<U,NO>
}
impl<U,P,D,I,const NI:usize,const NO:usize> LinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>>, U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    pub fn new<UI: FnMut() -> U, BI: FnMut() -> U>(parent:P,device:&D,mut ui:UI,mut bi:BI) -> LinearLayer<U,P,D,I,NI,NO> {
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
impl<U,P,D,I,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>> for LinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + Persistence<U,TextFilePersistence<U>>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          D: Device<U>, ConfigReadError: From<<U as FromStr>::Err>,
          I: Debug {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        for mut u in self.units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) {
        self.parent.save(persistence);

        persistence.write(UnitOrMarker::LayerStart);

        for b in self.bias.iter() {
            persistence.write(UnitOrMarker::Unit(*b));
        }

        for u in self.units.iter() {
            persistence.write(UnitOrMarker::UnitsStart);
            for w in u.iter() {
                persistence.write(UnitOrMarker::Unit(*w));
            }
        }
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> Forward<Arr<U,NI>,Arr<U,NO>> for LinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {

    fn forward(&self,input:&Arr<U,NI>) -> Arr<U,NO> {
        self.device.forward_linear(&self.bias,&self.units,input)
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> ForwardAll for LinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    type Input = I;
    type Output = Arr<U,NO>;
    fn forward_all(&self, input: Self::Input) -> Arr<U,NO> {
        self.forward(&self.parent.forward_all(input))
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> PreTrain<U> for LinearLayer<U,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,NI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::Output>;

    fn pre_train(&mut self, input: Self::Input) -> Self::OutStack {
        let r = self.parent.pre_train(input);

        let u = r.map(|r| self.forward(r));

        Cons(r,u)
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> Backward<U,&Arr<U,NO>,Arr<U,NI>> for LinearLayer<U,P,D,I,NI,NO>
    where U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U>,
          I: Debug {
    fn backward(&mut self, input: &Arr<U,NO>) -> Arr<U,NI> {
        self.device.backward_liner(&self.units,input)
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> BackwardAll<U> for LinearLayer<U,P,D,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> + ForwardAll<Input=I,Output=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    type LossInput = Arr<U,NO>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) {
        let loss = input;

        let (s,_) = stack.pop();

        {
            for l in loss.iter() {
                for w in self.bias.iter_mut() {
                    optimizer.update(*l, w);
                }
            }

            s.map(|o| {
                for (mut u,(l,o)) in self.units.iter_mut().zip(loss.iter().zip(o.iter())) {
                    for w in u.iter_mut() {
                        optimizer.update(*l * *o, w);
                    }
                }
            });
        }

        let loss= self.backward(&loss);

        let (s,loss) = self.parent.loss(loss,lossf,s);

        self.parent.backward_all(loss, s, optimizer, lossf);
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> Loss<U> for LinearLayer<U,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
}
pub struct DiffLinearLayer<U,P,D,I,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    parent:P,
    device:D,
    units:Arr2<U,NI,NO>,
    bias:Arr<U,NO>
}
impl<U,P,D,I,const NI:usize,const NO:usize> DiffLinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    pub fn new<UI: FnMut() -> U, BI: FnMut() -> U>(parent:P,device:&D,mut ui:UI,mut bi:BI) -> DiffLinearLayer<U,P,D,I,NI,NO> {
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

        DiffLinearLayer {
            parent:parent,
            device:device.clone(),
            units: units,
            bias:bias
        }
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>> for DiffLinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + Persistence<U,TextFilePersistence<U>>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          D: Device<U>, ConfigReadError: From<<U as FromStr>::Err>,
          I: Debug {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        for mut u in self.units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) {
        self.parent.save(persistence);

        persistence.write(UnitOrMarker::LayerStart);

        for b in self.bias.iter() {
            persistence.write(UnitOrMarker::Unit(*b));
        }

        for u in self.units.iter() {
            persistence.write(UnitOrMarker::UnitsStart);
            for w in u.iter() {
                persistence.write(UnitOrMarker::Unit(*w));
            }
        }
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> ForwardAll for DiffLinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    type Input = I;
    type Output = Arr<U,NO>;

    fn forward_all(&self, input: Self::Input) -> Arr<U,NO> {
        let input = self.parent.forward_all(input);

        match input {
            DiffInput::Diff(d, mut output) => {
                for &(i,d) in d.iter() {
                    for (o,j) in output.iter_mut().zip(0..NO) {
                        *o += self.units[(i,j)] * d;
                    }
                }
                output
            },
            DiffInput::NotDiff(input) => {
                self.device.forward_linear(&self.bias,&self.units,&input)
            }
        }
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> PreTrain<U> for DiffLinearLayer<U,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Arr<U,NO>>;

    fn pre_train(&mut self, input: Self::Input) -> Self::OutStack {
        let s = self.parent.pre_train(input);

        let u = s.map(|input| {
            match input {
                DiffInput::Diff(d, output) => {
                    let mut output = output.clone();

                    for &(i, d) in d.iter() {
                        for (o, j) in output.iter_mut().zip(0..NO) {
                            *o += self.units[(i, j)] * d;
                        }
                    }
                    output
                },
                DiffInput::NotDiff(input) => {
                    self.device.forward_linear(&self.bias, &self.units, &input)
                }
            }
        });

        Cons(s,u)
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> Backward<U,&Arr<U,NO>,Arr<U,NI>> for DiffLinearLayer<U,P,D,I,NI,NO>
    where U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + BackwardAll<U>,
          I: Debug {
    fn backward(&mut self, input: &Arr<U,NO>) -> Arr<U,NI> {
        self.device.backward_liner(&self.units,input)
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> BackwardAll<U> for DiffLinearLayer<U,P,D,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
    type LossInput = Arr<U,NO>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) {
        let loss = input;

        let (s,_) = stack.pop();

        {
            for l in loss.iter() {
                for w in self.bias.iter_mut() {
                    optimizer.update(*l, w);
                }
            }

            s.map(|o| {
                match o {
                    DiffInput::Diff(_, o) => {
                        for (mut u,(l,o)) in self.units.iter_mut().zip(loss.iter().zip(o.iter())) {
                            for w in u.iter_mut() {
                                optimizer.update(*l * *o, w);
                            }
                        }
                    },
                    DiffInput::NotDiff(input) => {
                       let o = self.device.forward_linear(&self.bias, &self.units, input);

                        for (mut u,(l,o)) in self.units.iter_mut().zip(loss.iter().zip(o.iter())) {
                            for w in u.iter_mut() {
                                optimizer.update(*l * *o, w);
                            }
                        }
                    }
                };
            });
        }

        let loss= self.backward(&loss);

        let (s,loss) = self.parent.loss(loss,lossf,s);

        self.parent.backward_all(loss, s, optimizer, lossf);
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> Loss<U> for DiffLinearLayer<U,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug {
}
