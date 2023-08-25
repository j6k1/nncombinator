//! Implementation of Activation layers

use std::fmt::Debug;
use std::marker::PhantomData;
use crate::activation::{Activation, BatchActivation};
use crate::arr::{Arr, VecArr};
use crate::{Cons, Stack};
use crate::device::Device;
use crate::error::{ConfigReadError, EvaluateError, PersistenceError, TrainingError};
use crate::layer::{AskDiffInput, BackwardAll, BatchBackward, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, Forward, ForwardAll, Loss, PreTrain};
use crate::lossfunction::LossFunction;
use crate::ope::UnitValue;
use crate::optimizer::Optimizer;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence};

/// Activation layer Implementation
pub struct ActivationLayer<U,P,A,I,PI,D> where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
                                               U: UnitValue<U>,
                                               D: Device<U>,
                                               A: Activation<U,PI,PI,D>,
                                               PI: Debug + Send + Sync + 'static,
                                               I: Debug + Send + Sync {
    parent:P,
    f:A,
    device:D,
    u:PhantomData<U>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
}
impl<U,P,A,I,PI,D> ActivationLayer<U,P,A,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U>,
          D: Device<U>,
          A: Activation<U,PI,PI,D>,
          PI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    /// Create and return an instance of ActivationLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
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
impl<U,P,A,I,PI,D> Persistence<U,TextFilePersistence<U>,Specialized> for ActivationLayer<U,P,A,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + Persistence<U,TextFilePersistence<U>,Specialized> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U> + std::str::FromStr,
          D: Device<U>,
          A: Activation<U,PI,PI,D>,
          PI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<T,U,P,A,I,PI,D> Persistence<U,T,Linear> for ActivationLayer<U,P,A,I,PI,D>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U>,
          D: Device<U>,
          A: Activation<U,PI,PI,D>,
          PI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,A,I,PI,D> ForwardAll for ActivationLayer<U,P,A,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,PI,PI,D>,
          PI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = PI;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,P,A,I,PI,D> Forward<PI,Result<PI,EvaluateError>> for ActivationLayer<U,P,A,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,PI,PI,D>,
          PI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    fn forward(&self, input: &PI) -> Result<PI,EvaluateError> {
        self.f.apply(&self.device, input)
    }
}
impl<U,P,A,I,D,const N:usize> PreTrain<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> +
             BackwardAll<U,LossInput=Arr<U,N>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,Arr<U,N>,Arr<U,N>,D>,
          I: Debug + Send + Sync {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack, Self::Output>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r))?;

        Ok(Cons(r,u))
    }
}
impl<U,P,A,I,D,const N:usize> BackwardAll<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,Arr<U,N>,Arr<U,N>,D>,
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
impl<U,P,A,I,D,const N:usize> AskDiffInput<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> +
             BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> + AskDiffInput<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,Arr<U,N>,Arr<U,N>,D>,
          I: Debug + Send + Sync {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map_remaining(|s| self.parent.ask_diff_input(s))
    }
}
impl<U,P,A,I,D,const N:usize> Loss<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> +
             BackwardAll<U,LossInput=Arr<U,N>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,Arr<U,N>,Arr<U,N>,D>,
          I: Debug + Send + Sync {
    fn loss<L: LossFunction<U>>(&mut self, loss: Self::LossInput, _:&L, stack: Self::OutStack) -> Result<(Self::OutStack, Self::LossInput), TrainingError> {
        let (s,o) = stack.pop();

        let r = s.map(|u| self.f.derive(&self.device, &o, &loss, u))?;

        Ok((Cons(s,o),r))
    }
}
impl<U,P,A,I,D,const N:usize> BatchForwardBase for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Arr<U,N>>> + BatchPreTrainBase<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=VecArr<U,Arr<U,N>>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,Arr<U,N>,Arr<U,N>,D>,
          I: Debug + Send + Sync {
    type BatchInput = VecArr<U,I>;
    type BatchOutput = VecArr<U,Arr<U,N>>;
}
impl<U,P,A,I,D,const N:usize> BatchForward for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Arr<U,N>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=VecArr<U,Arr<U,N>>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: BatchActivation<U,Arr<U,N>,Arr<U,N>,D>,
          I: Debug + Send + Sync {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        self.f.batch_apply(&self.device,&input)
    }
}
impl<U,P,A,I,D,const N:usize> BatchPreTrainBase<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Arr<U,N>>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=VecArr<U,Arr<U,N>>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,Arr<U,N>,Arr<U,N>,D>,
          I: Debug + Send + Sync {
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack, Self::BatchOutput>;
}
impl<U,P,A,I,D,const N:usize> BatchPreTrain<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Arr<U,N>>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=VecArr<U,Arr<U,N>>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: BatchActivation<U,Arr<U,N>,Arr<U,N>,D>,
          I: Debug + Send + Sync {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| {
            self.f.batch_apply(&self.device,&input)
        })?;

        Ok(Cons(r,u))
    }
}
impl<U,P,A,I,D,const N:usize> BatchBackward<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> + BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Arr<U,N>>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=VecArr<U,Arr<U,N>>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: Activation<U,Arr<U,N>,Arr<U,N>,D>,
          I: Debug + Send + Sync {
    type BatchLossInput = VecArr<U,Arr<U,N>>;
    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, optimizer: &mut OP, lossf: &L) -> Result<(), TrainingError> {
        let (s,_) = stack.pop();

        self.parent.batch_backward(input, s, optimizer, lossf)
    }
}
impl<U,P,A,I,D,const N:usize> BatchLoss<U> for ActivationLayer<U,P,A,I,Arr<U,N>,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,N>> +
             BackwardAll<U,LossInput=Arr<U,N>> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Arr<U,N>>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=VecArr<U,Arr<U,N>>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          A: BatchActivation<U,Arr<U,N>,Arr<U,N>,D> + Activation<U,Arr<U,N>,Arr<U,N>,D>,
          I: Debug + Send + Sync {
    fn batch_loss<L: LossFunction<U>>(&self, loss: Self::BatchLossInput, _: &L, stack: Self::BatchOutStack) -> Result<(Self::BatchOutStack, Self::BatchLossInput), TrainingError> {
        let (s,o) = stack.pop();

        let r = s.map(|u| {
            self.f.batch_derive(&self.device,&o,&loss, u)
        })?;

        Ok((Cons(s,o),r))
    }
}
