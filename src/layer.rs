use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use crate::arr::*;
use crate::device::*;
use crate::persistence::*;
use crate::{Cons, Nil, Stack};
use crate::activation::{Activation, ActivationImplType};
use crate::error::{ConfigReadError, EvaluateError, PersistenceError, TrainingError};
use crate::ope::UnitValue;
use crate::lossfunction::*;
use crate::optimizer::*;

#[derive(Debug)]
pub enum DiffInput<T,U,const NI:usize,const NO:usize>
    where U: UnitValue<U> + Clone + Copy + Debug, T: Debug {
    Diff(T,Arr<U,NO>),
    NotDiff(Arr<U,NI>)
}
pub trait Forward<I,O> {
    fn forward(&self,input:&I) -> O;
}
pub trait ForwardAll {
    type Input: Debug;
    type Output: Debug;
    fn forward_all(&self, input:Self::Input) -> Result<Self::Output, EvaluateError>;
}
pub trait BackwardAll<U>: PreTrain<U> where U: UnitValue<U> {
    type LossInput: Debug;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input:Self::LossInput, stack:Self::OutStack, optimizer:&mut OP, lossf:&L) -> Result<(), TrainingError>;
    fn is_canonical_link<L: LossFunction<U>>(&self,_:&L) -> bool {
        false
    }
}
pub trait Loss<U>: BackwardAll<U> where U: UnitValue<U> {
    fn loss<L: LossFunction<U>>(&mut self, loss:Self::LossInput, _:&L, stack:Self::OutStack) -> Result<(Self::OutStack, Self::LossInput), TrainingError> {
        Ok((stack,loss))
    }
}
pub trait Backward<U,I,O> where U: UnitValue<U> {
    fn backward(&mut self, input:I) -> O;
}
pub trait PreTrain<U>: ForwardAll where U: UnitValue<U> {
    type OutStack: Stack<Head=Self::Output> + Sized + Debug + Send + Sync + 'static;
    fn pre_train(&self, input:Self::Input) -> Result<Self::OutStack, EvaluateError>;
}
pub trait ForwardDiff<U>: PreTrain<U> where U: UnitValue<U> {
    fn forward_diff(&self, input:Self::Input) -> Result<Self::OutStack, EvaluateError>;
}
pub trait Train<U>: PreTrain<U> where U: UnitValue<U> {
    fn train<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, expected:Self::Output, input:Self::Input, optimizer:&mut OP, lossf:&L) -> Result<U, TrainingError>;
}
pub trait AskDiffInput<U>: PreTrain<U> where U: UnitValue<U> {
    type DiffInput: Debug;
    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput;
}
pub trait BatchForwardBase: ForwardAll {
    type BatchInput: Debug;
    type BatchOutput: Debug;
}
pub trait BatchForward: BatchForwardBase {
    fn batch_forward(&self,input:Self::BatchInput) -> Result<Self::BatchOutput, TrainingError>;
}
pub trait BatchBackward<U>: BatchPreTrainBase<U> where U: UnitValue<U> {
    type BatchLossInput: Debug;
    fn batch_backward<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input:Self::BatchLossInput, stack:Self::BatchOutStack, optimizer:&mut OP, lossf:&L) -> Result<(), TrainingError>;
}
pub trait BatchLoss<U>: BatchBackward<U> + Loss<U> where U: UnitValue<U> {
    fn batch_loss<L: LossFunction<U>>(&self, loss:Self::BatchLossInput, _:&L, stack:Self::BatchOutStack) -> Result<(Self::BatchOutStack, Self::BatchLossInput), TrainingError> {
        Ok((stack,loss))
    }
}
pub trait BatchPreTrainBase<U>: BatchForwardBase + PreTrain<U> where U: UnitValue<U> {
    type BatchOutStack: Stack<Head=Self::BatchOutput> + Sized + Debug;
}
pub trait BatchPreTrain<U>: BatchPreTrainBase<U> + BatchForwardBase + BatchForward + where U: UnitValue<U> {
    fn batch_pre_train(&self, input:Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError>;
}
pub trait BatchTrain<U>: BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + PreTrain<U> where U: UnitValue<U> {
    fn batch_train<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, expected:Self::BatchOutput, input:Self::BatchInput, optimizer:&mut OP, lossf:&L) -> Result<U, TrainingError>;
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
    fn forward_diff(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        self.pre_train(input)
    }
}
pub struct InputLayer<U,O,LI> where U: UnitValue<U> {
    u:PhantomData<U>,
    o:PhantomData<O>,
    l:PhantomData<LI>
}
impl<U,O,LI> InputLayer<U,O,LI> where U: UnitValue<U> {
    pub fn new() -> InputLayer<U,O,LI> {
        InputLayer {
            u:PhantomData::<U>,
            o:PhantomData::<O>,
            l:PhantomData::<LI>
        }
    }
}
impl<U,O,LI> Persistence<U,TextFilePersistence<U>,Specialized> for InputLayer<U,O,LI>
    where U: UnitValue<U> + FromStr + Sized {
    fn load(&mut self, _: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<T,U,O,LI> Persistence<U,T,Linear> for InputLayer<U,O,LI>
    where T: LinearPersistence<U>, U: UnitValue<U> {
    fn load(&mut self, _: &mut T) -> Result<(),ConfigReadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut T) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<U,O,LI> ForwardAll for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type Input = O;
    type Output = O;
    fn forward_all(&self, input:Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(input)
    }
}
impl<U,O,LI> PreTrain<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type OutStack = Cons<Nil,Self::Output>;

    fn pre_train(&self, input:Self::Input) -> Result<Self::OutStack, EvaluateError> {
        Ok(Cons(Nil,input))
    }
}
impl<U,O,LI> BackwardAll<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type LossInput = LI;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, _: Self::LossInput, _:Self::OutStack, _: &mut OP, _:&L) -> Result<(), TrainingError> {
        Ok(())
    }
}
impl<U,O,LI> Loss<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {}
impl<U,O,LI> BatchForwardBase for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type BatchInput = Vec<O>;
    type BatchOutput = Vec<O>;
}
impl<U,O,LI> BatchForward for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput,TrainingError> {
        Ok(input)
    }
}
impl<U,O,LI> BatchPreTrainBase<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type BatchOutStack = Cons<Nil,Vec<O>>;
}
impl<U,O,LI> BatchPreTrain<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        Ok(Cons(Nil,input))
    }
}
impl<U,O,LI> BatchBackward<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type BatchLossInput = Vec<LI>;
    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, _: Self::BatchLossInput, _: Self::BatchOutStack, _: &mut OP, _: &L) -> Result<(), TrainingError> {
        Ok(())
    }
}
impl<U,O,LI> BatchLoss<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {}
pub struct ActivationLayer<U,P,A,I,PI,D,AI> where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI>,
                                               U: UnitValue<U>,
                                               D: Device<U>,
                                               AI: ActivationImplType,
                                               A: Activation<U,PI,D,AI>,
                                               PI: Debug,
                                               I: Debug + Send + Sync {
    parent:P,
    f:A,
    device:D,
    u:PhantomData<U>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
    ai:PhantomData<AI>,
}
impl<U,P,A,I,PI,D,AI> ActivationLayer<U,P,A,I,PI,D,AI>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,PI,D,AI>,
          PI: Debug,
          I: Debug + Send + Sync {
    pub fn new(parent:P,f:A,device:&D) -> ActivationLayer<U,P,A,I,PI,D,AI> {
        ActivationLayer {
            parent:parent,
            f:f,
            device:device.clone(),
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            pi:PhantomData::<PI>,
            ai:PhantomData::<AI>
        }
    }
}
impl<U,P,A,I,PI,D,AI> Persistence<U,TextFilePersistence<U>,Specialized> for ActivationLayer<U,P,A,I,PI,D,AI>
    where P: ForwardAll<Input=I,Output=PI> + Persistence<U,TextFilePersistence<U>,Specialized> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U> + std::str::FromStr,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,PI,D,AI>,
          PI: Debug,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<T,U,P,A,I,PI,D,AI> Persistence<U,T,Linear> for ActivationLayer<U,P,A,I,PI,D,AI>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,PI,D,AI>,
          PI: Debug,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,A,I,PI,D,AI> ForwardAll for ActivationLayer<U,P,A,I,PI,D,AI>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,PI,D,AI>,
          PI: Debug,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = PI;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(self.forward(&self.parent.forward_all(input)?))
    }
}
impl<U,P,A,I,PI,D,AI> Forward<PI,PI> for ActivationLayer<U,P,A,I,PI,D,AI>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,PI,D,AI>,
          PI: Debug,
          I: Debug + Send + Sync {
    fn forward(&self, input: &PI) -> PI {
        self.f.apply(&self.device,input)
    }
}
impl<U,P,A,D,I,AI,const N:usize> PreTrain<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D,AI>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> +
             BackwardAll<U,LossInput=Arr<U,N>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,Arr<U,N>,D,AI>,
          I: Debug + Send + Sync {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack, Self::Output>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r));

        Ok(Cons(r,u))
    }
}
impl<U,P,A,D,I,AI,const N:usize> BackwardAll<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D,AI>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,Arr<U,N>,D,AI>,
          I: Debug + Send + Sync {
    type LossInput = Arr<U,N>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) -> Result<(), TrainingError> {
        let (s,_) = stack.pop();

        self.parent.backward_all(input, s, optimizer,lossf)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.f.is_canonical_link(l)
    }
}
impl<U,P,A,D,I,AI,const N:usize> AskDiffInput<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D,AI>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> +
             BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> + AskDiffInput<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,Arr<U,N>,D,AI>,
          I: Debug + Send + Sync {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map_remaining(|s| self.parent.ask_diff_input(s))
    }
}
impl<U,P,A,D,I,AI,const N:usize> Loss<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D,AI>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> +
             BackwardAll<U,LossInput=Arr<U,N>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,Arr<U,N>,D,AI>,
          I: Debug + Send + Sync {
    fn loss<L: LossFunction<U>>(&mut self, loss: Self::LossInput, _:&L, stack: Self::OutStack) -> Result<(Self::OutStack, Self::LossInput), TrainingError> {
        let (s,o) = stack.pop();

        let r = s.map(|u| self.f.derive(&self.device,&o,&loss,u));

        Ok((Cons(s,o),r))
    }
}
impl<U,P,A,D,I,AI,const N:usize> BatchForwardBase for ActivationLayer<U,P,A,I,Arr<U,N>,D,AI>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,N>>> + BatchPreTrainBase<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=Vec<Arr<U,N>>> + Send + Sync + 'static,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,Arr<U,N>,D,AI>,
          I: Debug + Send + Sync {
    type BatchInput = Vec<I>;
    type BatchOutput = Vec<Arr<U,N>>;
}
impl<U,P,A,D,I,AI,const N:usize> BatchForward for ActivationLayer<U,P,A,I,Arr<U,N>,D,AI>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,N>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=Vec<Arr<U,N>>> +
             Send + Sync + 'static,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,Arr<U,N>,D,AI>,
          I: Debug + Send + Sync {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(input.par_iter().map(|i| {
            self.f.apply(&self.device,&i)
        }).collect::<Vec<Arr<U,N>>>())
    }
}
impl<U,P,A,D,I,AI,const N:usize> BatchPreTrainBase<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D,AI>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,N>>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=Vec<Arr<U,N>>> +
             Send + Sync + 'static,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,Arr<U,N>,D,AI>,
          I: Debug + Send + Sync {
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack, Self::BatchOutput>;
}
impl<U,P,A,D,I,AI,const N:usize> BatchPreTrain<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D,AI>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,N>>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=Vec<Arr<U,N>>> +
             Send + Sync + 'static,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,Arr<U,N>,D,AI>,
          I: Debug + Send + Sync {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| input.par_iter().map(|i| {
            self.f.apply(&self.device,&i)
        }).collect::<Vec<Arr<U,N>>>());

        Ok(Cons(r,u))
    }
}
impl<U,P,A,D,I,AI,const N:usize> BatchBackward<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D,AI>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,N>>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=Vec<Arr<U,N>>> +
             Send + Sync + 'static,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,Arr<U,N>,D,AI>,
          I: Debug + Send + Sync {
    type BatchLossInput = Vec<Arr<U,N>>;
    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, optimizer: &mut OP, lossf: &L) -> Result<(), TrainingError> {
        let (s,_) = stack.pop();

        self.parent.batch_backward(input, s, optimizer, lossf)
    }
}
impl<U,P,A,D,I,AI,const N:usize> BatchLoss<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D,AI>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> +
             BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,N>>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=Vec<Arr<U,N>>> +
             Send + Sync + 'static,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          AI: ActivationImplType,
          A: Activation<U,Arr<U,N>,D,AI>,
          I: Debug + Send + Sync {
    fn batch_loss<L: LossFunction<U>>(&self, loss: Self::BatchLossInput, _: &L, stack: Self::BatchOutStack) -> Result<(Self::BatchOutStack, Self::BatchLossInput), TrainingError> {
        let (s,o) = stack.pop();

        let r = s.map(|u| {
            self.device.batch_loss_linear_by_activaton(&o,&loss, u, &self.f)
        })?;

        Ok((Cons(s,o),r))
    }
}
pub struct LinearOutputLayer<U,P,D,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U,LossInput=IO> + PreTrain<U> + Loss<U> +,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          IO: Debug,
          I: Debug + Send + Sync {
    u:PhantomData<U>,
    i:PhantomData<I>,
    io:PhantomData<IO>,
    parent:P,
    device:D,
}
impl<U,P,D,I,IO> LinearOutputLayer<U,P,D,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U,LossInput=IO> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          IO: Debug,
          I: Debug + Send + Sync {
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
impl<U,P,D,I,IO> Persistence<U,TextFilePersistence<U>,Specialized> for LinearOutputLayer<U,P,D,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U,LossInput=IO> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr + Sized,
          D: Device<U>,
          IO: Debug,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;
        persistence.verify_eof()
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<T,U,P,D,I,IO> Persistence<U,T,Linear> for LinearOutputLayer<U,P,D,I,IO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=IO> + BackwardAll<U,LossInput=IO> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          IO: Debug,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;
        persistence.verify_eof()
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,D,I,IO> ForwardAll for LinearOutputLayer<U,P,D,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U,LossInput=IO> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          IO: Debug,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = IO;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.parent.forward_all(input)
    }
}
impl<U,P,D,I,IO> PreTrain<U> for LinearOutputLayer<U,P,D,I,IO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=IO> + BackwardAll<U,LossInput=IO> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          IO: Debug,
          I: Debug + Send + Sync {
    type OutStack = P::OutStack;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        self.parent.pre_train(input)
    }
}
impl<U,P,D,I,const NO:usize> BackwardAll<U> for LinearOutputLayer<U,P,D,I,Arr<U,NO>>
    where P: BackwardAll<U,LossInput=Arr<U,NO>> +
             ForwardAll<Input=I,Output=Arr<U,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type LossInput = Arr<U,NO>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) -> Result<(), TrainingError> {
        self.parent.backward_all(input, stack, optimizer, lossf)
    }
}
impl<U,P,D,I,const NO:usize> AskDiffInput<U> for LinearOutputLayer<U,P,D,I,Arr<U,NO>>
    where P: BackwardAll<U,LossInput=Arr<U,NO>> +
             ForwardAll<Input=I,Output=Arr<U,NO>> + PreTrain<U> + Loss<U> + AskDiffInput<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        self.parent.ask_diff_input(stack)
    }
}
impl<U,P,D,I,const NO:usize> Train<U> for LinearOutputLayer<U,P,D,I,Arr<U,NO>>
    where P: BackwardAll<U,LossInput=Arr<U,NO>> +
             ForwardAll<Input=I,Output=Arr<U,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    fn train<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, expected: Self::Output, input: Self::Input, optimizer: &mut OP, lossf: &L) -> Result<U, TrainingError> {
        let stack = self.pre_train(input)?;

        let total_loss = stack.map(|l| self.device.loss_linear_total(&expected,l,lossf));

        let (stack,loss) = if self.parent.is_canonical_link(lossf) {
            let loss = stack.map(|actual| {
                self.device.loss_linear_by_canonical_link(&expected, &actual)
            });

            (stack,loss)
        } else {
            let loss = stack.map(|actual| {
                self.device.loss_linear(&expected,&actual,lossf)
            });

            self.parent.loss(loss,lossf,stack)?
        };

        self.backward_all(loss,stack,optimizer,lossf)?;

        Ok(total_loss)
    }
}
impl<U,P,I,IO> BatchForwardBase for LinearOutputLayer<U,P,DeviceCpu<U>,I,IO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=IO> + BackwardAll<U,LossInput=IO> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<IO>> + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          IO: Debug,
          I: Debug + Send + Sync {
    type BatchInput = Vec<I>;
    type BatchOutput = Vec<IO>;
}
impl<U,P,I,IO> BatchForward for LinearOutputLayer<U,P,DeviceCpu<U>,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U,LossInput=IO> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<IO>> + BatchForward + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          IO: Debug,
          I: Debug + Send + Sync {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        self.parent.batch_forward(input)
    }
}
impl<U,P,I,IO> BatchPreTrainBase<U> for LinearOutputLayer<U,P,DeviceCpu<U>,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U,LossInput=IO> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<IO>> + BatchForward +
             BatchPreTrainBase<U> + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          IO: Debug,
          I: Debug + Send + Sync {
    type BatchOutStack = P::BatchOutStack;
}
impl<U,P,I,IO> BatchPreTrain<U> for LinearOutputLayer<U,P,DeviceCpu<U>,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U,LossInput=IO> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<IO>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          IO: Debug,
          I: Debug + Send + Sync {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        self.parent.batch_pre_train(input)
    }    
}
impl<U,P,I,IO> BatchBackward<U> for LinearOutputLayer<U,P,DeviceCpu<U>,I,IO>
    where P: ForwardAll<Input=I,Output=IO> + BackwardAll<U,LossInput=IO> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<IO>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=Vec<IO>> + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          IO: Debug,
          I: Debug + Send + Sync {
    type BatchLossInput = Vec<IO>;

    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, optimizer: &mut OP, lossf: &L) -> Result<(), TrainingError> {
        self.parent.batch_backward(input,stack,optimizer,lossf)
    }
}
impl<U,P,I,const N:usize> BatchTrain<U> for LinearOutputLayer<U,P,DeviceCpu<U>,I,Arr<U,N>>
    where P: ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,N>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=Vec<Arr<U,N>>> + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
    fn batch_train<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, expected:Self::BatchOutput, input:Self::BatchInput, optimizer:&mut OP, lossf:&L) -> Result<U, TrainingError> {
        let stack = self.batch_pre_train(input)?;

        let total_loss = stack.map(|l| self.device.batch_loss_linear_total(&expected,l,lossf));

        let (stack,loss) = if self.parent.is_canonical_link(lossf) {
            let loss = stack.map(|actual| {
                self.device.loss_linear_batch_by_canonical_link(&expected, &actual)
            })?;

            (stack,loss)
        } else {
            let loss = stack.map(|actual| {
                self.device.loss_linear_batch(&expected,&actual,lossf)
            })?;

            self.parent.batch_loss(loss,lossf,stack)?
        };

        self.parent.batch_backward(loss,stack,optimizer,lossf)?;

        Ok(total_loss)
    }
}
pub struct LinearLayer<U,P,D,I,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    parent:P,
    device:D,
    units:Arr2<U,NI,NO>,
    bias:Arr<U,NO>
}
impl<U,P,D,I,const NI:usize,const NO:usize> LinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
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
impl<U,P,D,I,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>,Specialized> for LinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          D: Device<U>, ConfigReadError: From<<U as FromStr>::Err>,
          I: Debug + Send + Sync {
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

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

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

        Ok(())
    }
}
impl<T,U,P,D,I,const NI:usize,const NO:usize> Persistence<U,T,Linear> for LinearLayer<U,P,D,I,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
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

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for b in self.bias.iter() {
            persistence.write(*b)?;
        }

        for u in self.units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> Forward<Arr<U,NI>,Arr<U,NO>> for LinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {

    fn forward(&self,input:&Arr<U,NI>) -> Arr<U,NO> {
        self.device.forward_linear(&self.bias,&self.units,input)
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> ForwardAll for LinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = Arr<U,NO>;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(self.forward(&self.parent.forward_all(input)?))
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> PreTrain<U> for LinearLayer<U,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::Output>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r));

        Ok(Cons(r,u))
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> Backward<U,&Arr<U,NO>,Arr<U,NI>> for LinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    fn backward(&mut self, input: &Arr<U,NO>) -> Arr<U,NI> {
        self.device.backward_linear(&self.units, input)
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> BackwardAll<U> for LinearLayer<U,P,D,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> + ForwardAll<Input=I,Output=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type LossInput = Arr<U,NO>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) -> Result<(), TrainingError> {
        let loss = input;

        let (s,_) = stack.pop();

        {
            for l in loss.iter() {
                for w in self.bias.iter_mut() {
                    optimizer.update(*l, w);
                }
            }

            s.map(|o| {
                for (mut u,o) in self.units.iter_mut().zip(o.iter()) {
                    for (w,l) in u.iter_mut().zip(loss.iter()) {
                        optimizer.update(*l * *o, w);
                    }
                }
            });
        }

        let loss= self.backward(&loss);

        let (s,loss) = self.parent.loss(loss,lossf,s)?;

        self.parent.backward_all(loss, s, optimizer, lossf)
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> AskDiffInput<U> for LinearLayer<U,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,NI>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U> + AskDiffInput<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map_remaining(|s| self.parent.ask_diff_input(s))
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> Loss<U> for LinearLayer<U,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
}
impl<U,P,D,I,const NI:usize,const NO:usize> BatchForwardBase for LinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,NI>>> + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type BatchInput = Vec<I>;
    type BatchOutput = Vec<Arr<U,NO>>;
}
impl<U,P,I,const NI:usize,const NO:usize> BatchForward for LinearLayer<U,P,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,NI>>> + BatchForward + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        self.device.batch_forward_linear(&input,&self.bias,&self.units,)
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> BatchPreTrainBase<U> for LinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,Self::BatchOutput>;
}
impl<U,P,I,const NI:usize,const NO:usize> BatchPreTrain<U> for LinearLayer<U,P,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| self.device.batch_forward_linear(input,&self.bias,&self.units))?;

        Ok(Cons(r,u))
    }
}
impl<U,P,I,const NI:usize,const NO:usize> BatchBackward<U> for LinearLayer<U,P,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=Vec<Arr<U,NI>>> + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
    type BatchLossInput = Vec<Arr<U,NO>>;

    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, optimizer: &mut OP, lossf: &L) -> Result<(), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;
        {
            let batch_size = loss.len() as f64;
            let batch_size = U::from_f64(batch_size).ok_or(TrainingError::ToLargeInput(batch_size))?;

            let loss = loss.par_iter()
                           .cloned()
                           .map(|l| Ok(l)).reduce(|| Ok(Arr::<U,NO>::new()), |acc,o| {
                acc.and_then(|acc| o.and_then(|o| {
                    acc.par_iter().cloned()
                        .zip(o.par_iter().cloned())
                        .map(|(acc, o)| acc + o).collect::<Vec<U>>().try_into()
                }))
            })?;

            {
                for (w,&l) in self.bias.iter_mut().zip(loss.iter()) {
                    optimizer.update(l / batch_size, w);
                }

                s.map(|o| {
                    o.par_iter().cloned().map(|o| Ok(o)).reduce(|| Ok(Arr::new()), |acc,o| {
                        acc.and_then(|acc| o.and_then(|o| {
                            acc.par_iter().zip(o.par_iter()).map(|(&acc, &o)| {
                                acc + o
                            }).collect::<Vec<U>>().try_into()
                        }))
                    }).map(|o| {
                        for (mut u, o) in self.units.iter_mut().zip(o.iter()) {
                            for (w, &l) in u.iter_mut().zip(loss.iter()) {
                                optimizer.update(l * *o / batch_size, w);
                            }
                        }
                    })
                })?;
            }
        }

        let loss = self.device.backward_linear_batch(&self.units,&loss)?;

        let (s,loss) = self.parent.batch_loss(loss,lossf,s)?;

        self.parent.batch_backward(loss, s, optimizer, lossf)
    }
}
impl<U,P,I,const NI:usize,const NO:usize> BatchLoss<U> for LinearLayer<U,P,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=Vec<I>,BatchOutput=Vec<Arr<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=Vec<Arr<U,NI>>> + Send + Sync + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
}
pub struct DiffLinearLayer<U,P,D,I,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    parent:P,
    device:D,
    units:Arr2<U,NI,NO>,
    bias:Arr<U,NO>
}
impl<U,P,D,I,const NI:usize,const NO:usize> DiffLinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
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
impl<U,P,D,I,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>,Specialized> for DiffLinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          D: Device<U>, ConfigReadError: From<<U as FromStr>::Err>,
          I: Debug + Send + Sync {
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

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

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

        Ok(())
    }
}
impl<T,U,P,D,I,const NI:usize,const NO:usize> Persistence<U,T,Linear> for DiffLinearLayer<U,P,D,I,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
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

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for b in self.bias.iter() {
            persistence.write(*b)?;
        }

        for u in self.units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> ForwardAll for DiffLinearLayer<U,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = Arr<U,NO>;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_all(input)?;

        Ok(match input {
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
        })
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> PreTrain<U> for DiffLinearLayer<U,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Arr<U,NO>>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let s = self.parent.pre_train(input)?;

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

        Ok(Cons(s,u))
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> Backward<U,&Arr<U,NO>,Arr<U,NI>> for DiffLinearLayer<U,P,D,I,NI,NO>
    where U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          I: Debug + Send + Sync {
    fn backward(&mut self, input: &Arr<U,NO>) -> Arr<U,NI> {
        self.device.backward_linear(&self.units, input)
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> BackwardAll<U> for DiffLinearLayer<U,P,D,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type LossInput = Arr<U,NO>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) -> Result<(), TrainingError> {
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
                        for (mut u,o) in self.units.iter_mut().zip(o.iter()) {
                            for (w,l) in u.iter_mut().zip(loss.iter()) {
                                optimizer.update(*l * *o, w);
                            }
                        }
                    },
                    DiffInput::NotDiff(input) => {
                        let o = self.device.forward_linear(&self.bias, &self.units, input);

                        for (mut u,o) in self.units.iter_mut().zip(o.iter()) {
                            for (w,l) in u.iter_mut().zip(loss.iter()) {
                                optimizer.update(*l * *o, w);
                            }
                        }
                    }
                };
            });
        }

        let loss= self.backward(&loss);

        let (s,loss) = self.parent.loss(loss,lossf,s)?;

        self.parent.backward_all(loss, s, optimizer, lossf)
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> AskDiffInput<U> for DiffLinearLayer<U,P,D,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    type DiffInput = Arr<U,NO>;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map(|o| o.clone())
    }
}
impl<U,P,D,I,const NI:usize,const NO:usize> Loss<U> for DiffLinearLayer<U,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
}
