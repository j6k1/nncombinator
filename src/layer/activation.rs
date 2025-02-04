//! Implementation of Activation layers

use std::fmt::Debug;
use std::marker::PhantomData;
use crate::{Cons, Stack};
use crate::device::activation::DeviceActivation;
use crate::device::Device;
use crate::error::{ConfigReadError, EvaluateError, PersistenceError, TrainingError, TypeConvertError};
use crate::layer::{AskDiffInput, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, Forward, ForwardAll, Loss, PreTrain, UpdateWeight};
use crate::lossfunction::LossFunction;
use crate::ope::UnitValue;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence};

/// Activation layer Implementation
pub struct ActivationLayer<U,P,A,I,PI,D,const N:usize> where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
                                               U: UnitValue<U>,
                                               D: Device<U> + DeviceActivation<U,PI,A,N>,
                                               PI: Debug + BatchDataType + 'static,
                                               I: Debug + Send + Sync {
    parent:P,
    f:A,
    device:D,
    u:PhantomData<U>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
}
impl<U,P,A,I,PI,D,const N:usize> ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          I: Debug + Send + Sync {
    /// Create and return an instance of ActivationLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `f` - Activation Function
    /// * `device` - Device object used for neural network computation
    pub fn new(parent:P,f:A,device:&D) -> ActivationLayer<U,P,A,I,PI,D,N> {
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
impl<U,P,A,I,PI,D,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + Persistence<U,TextFilePersistence<U>,Specialized> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U> + std::str::FromStr,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<T,U,P,A,I,PI,D,const N:usize> Persistence<U,T,Linear> for ActivationLayer<U,P,A,I,PI,D,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,A,I,PI,D,const N:usize> ForwardAll for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = PI;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,P,A,I,PI,D,const N:usize> Forward<PI,Result<PI,EvaluateError>> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          I: Debug + Send + Sync {
    fn forward(&self, input: &PI) -> Result<PI,EvaluateError> {
        self.device.apply(&self.f, &input)
    }
}
impl<U,P,A,I,PI,D,const N:usize> PreTrain<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
    type PreOutput = PI;
    type OutStack = Cons<<P as PreTrain<U>>::OutStack, Self::PreOutput>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r))?;

        Ok(Cons(r,u))
    }
}
impl<U,P,A,I,PI,D,const N:usize> BackwardAll<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
    type LossInput = PI;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        self.parent.backward_all(input.into(), s, lossf)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.device.is_canonical_link(&self.f,l)
    }
}
impl<U,P,A,I,PI,D,const N:usize> UpdateWeight<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
    type GradientStack = <P as UpdateWeight<U>>::GradientStack;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        Ok(self.parent.update_weight(stack)?)
    }
}
impl<U,P,A,I,PI,D,const N:usize> AskDiffInput<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + AskDiffInput<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Result<Self::DiffInput,TypeConvertError> {
        stack.map_remaining(|s| self.parent.ask_diff_input(s))
    }
}
impl<U,P,A,I,PI,D,const N:usize> Loss<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
    fn loss<L: LossFunction<U>>(&mut self, loss: Self::LossInput, _:&L, stack: Self::OutStack) -> Result<(Self::OutStack, Self::LossInput), TrainingError> {
        let (s,o) = stack.pop();

        let r = s.map(|u| self.device.derive(&self.f, &o, &loss, u))?;

        Ok((Cons(s,o),r.into()))
    }
}
impl<U,P,A,I,PI,D,const N:usize> BatchForwardBase for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,P,A,I,PI,D,const N:usize> BatchForward for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_apply(&self.f,&input)?)
    }
}
impl<U,P,A,I,PI,D,const N:usize> BatchPreTrainBase<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack, Self::BatchPreOutput>;
}
impl<U,P,A,I,PI,D,const N:usize> BatchPreTrain<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| {
            self.device.batch_apply(&self.f,input)
        })?;

        Ok(Cons(r,u))
    }
}
impl<U,P,A,I,PI,D,const N:usize> BatchBackward<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        self.parent.batch_backward(input, s, lossf)
    }
}
impl<U,P,A,I,PI,D,const N:usize> BatchLoss<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    fn batch_loss<L: LossFunction<U>>(&self, loss: Self::BatchLossInput, _: &L, stack: Self::BatchOutStack) -> Result<(Self::BatchOutStack, Self::BatchLossInput), TrainingError> {
        let (s,o) = stack.pop();

        let r = s.map(|u| {
            self.device.batch_derive(&self.f, &o, &loss, u)
        })?;

        Ok((Cons(s,o),r))
    }
}
