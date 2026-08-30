//! Implementation of Activation layers

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use std::str::FromStr;
use crate::{Cons, Stack};
use crate::activation::ActivationBuilder;
use crate::device::activation::DeviceActivation;
use crate::device::Device;
use crate::error::{ModelLoadError, EvaluateError, PersistenceError, TrainingError};
use crate::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, ContinueForward, Forward, ForwardAll, ForwardDiff, Loss, PartialForward, PreTrain, UpdateWeight, OnStep, PersistProgress, InputTensorScalar, OutputTensorScalar, InputScale, OutputScale, MaxInputValue, PreTrainBase, BatchOutputScale, BatchSize, BackwardBase, BatchBackwardBase};
use crate::lossfunction::LossFunction;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextRecord};

/// Activation layer Implementation
pub struct ActivationLayer<U,P,A,DA,I,PI,LI,D,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    parent:P,
    f:A,
    df:DA,
    device:D,
    u:PhantomData<U>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
    li:PhantomData<LI>,
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + Device<<LI as OutputTensorScalar>::Scalar>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync,
          <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Debug + Send + Sync + 'static {
    /// Create and return an instance of ActivationLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `f` - Activation Function
    /// * `device` - Device object used for neural network computation
    pub fn new<AB>(parent:P,activation_builder:AB,device:&D) -> ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
        where AB: ActivationBuilder<U,<LI as OutputTensorScalar>::Scalar,D,ForwardActivation=A,BackwardActivation=DA> {
        let (f,df) = activation_builder.build();

        ActivationLayer {
            parent:parent,
            f:f,
            df:df,
            device:device.clone(),
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            pi:PhantomData::<PI>,
            li:PhantomData::<LI>,
        }
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> InputTensorScalar for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
          PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    type Scalar = U;
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> OutputTensorScalar for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    type Scalar = U;
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> Persistence<TextFilePersistence,Specialized> for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + Persistence<TextFilePersistence,Specialized> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> + PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + std::str::FromStr,
          D: Device<U>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write_layer_start();
        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,P,A,DA,I,PI,LI,D,const N:usize> Persistence<T,Linear> for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + Persistence<T,Linear> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> + PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> ForwardAll for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PreTrainBase<PreOutput=PI> + PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = PI;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> Forward<PI,Result<PI,EvaluateError>> for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PreTrainBase<PreOutput=PI> + PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    fn forward(&self, input: &PI) -> Result<PI,EvaluateError> {
        self.device.apply(&self.f, &input)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> PreTrainBase for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PreTrainBase<PreOutput=PI> + PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    type PreOutput = PI;
    type OutStack = Cons<<P as PreTrainBase>::OutStack, Self::PreOutput>;
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> PreTrain for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PreTrainBase<PreOutput=PI> + PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync {

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r))?;

        Ok(Cons(r,u))
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> BackwardBase for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             OutputTensorScalar<Scalar=U> +,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<<LI as OutputTensorScalar>::Scalar,LI,DA,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync,
          <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Debug + Send + Sync + 'static {
    type LossInputScalar = <LI as OutputTensorScalar>::Scalar;
    type LossInput = LI;
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> BackwardAll<<LI as OutputTensorScalar>::Scalar> for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             OutputTensorScalar<Scalar=U> +,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<<LI as OutputTensorScalar>::Scalar,LI,DA,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync,
          <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Debug + Send + Sync + 'static {
    type LossOutput = <P as BackwardAll<<LI as OutputTensorScalar>::Scalar>>::LossOutput;

    fn backward_all(&mut self, input: Self::LossInput, stack:Self::OutStack)
        -> Result<(<Self as BackwardAll<<LI as OutputTensorScalar>::Scalar>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        self.parent.backward_all(input.into(), s)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> UpdateWeight for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> + UpdateWeight +
             OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + BatchDataType+ OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    type GradientStack = <P as UpdateWeight>::GradientStack;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        Ok(self.parent.update_weight(stack,batch_size)?)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> PartialForward for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PartialForward + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          LI: Debug + BatchDataType+ OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    type PartialInput = <P as PartialForward>::PartialInput;
    type PartialOutput = <P as PartialForward>::PartialOutput;
    type DiffInput = <P as PartialForward>::DiffInput;
    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward(input)?)
    }

    fn partial_forward_by_diff(&self, input:Self::DiffInput, partial_input:&Self::PartialInput) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward_by_diff(input, partial_input)?)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> ForwardDiff for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PartialForward + ForwardDiff + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    fn forward_diff(&self, input: Self::DiffInput, partial_input:&Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_diff(input, partial_input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> ContinueForward for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PartialForward + ContinueForward + OutputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U> + DeviceActivation<U,PI,A,N>,
      PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
      LI: Debug + BatchDataType + OutputTensorScalar + 'static,
      I: Debug + Send + Sync {
    fn continue_forward<'a>(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.continue_forward(input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> Loss<<LI as OutputTensorScalar>::Scalar> for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             OutputTensorScalar<Scalar=U> + OutputScale<Scale=LI,ScaledOutput=LI,ScalingInput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<<LI as OutputTensorScalar>::Scalar,LI,DA,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + BatchDataType + Send + Sync,
          <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Debug + Send + Sync + 'static,
          <P as OutputScale>::Scale: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <LI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Debug + Send + Sync + 'static {
    fn loss(&mut self, loss: Self::LossInput, stack: Self::OutStack) -> Result<(Self::OutStack, Self::LossInput), TrainingError> {
        let (s,o) = stack.pop();

        let r = {
            let o = self.parent.scaling_mapper(&o)?;

            s.map(|u| {
                self.parent.scaling_mapper(u).map(|u| {
                    self.device.derive(&self.df,o.deref(),&loss,u.deref())
                })
            })??
        };

        Ok((Cons(s,o),r))
    }

    fn is_canonical_link<L: LossFunction<<LI as OutputTensorScalar>::Scalar>>(&self, lossf: &L) -> bool {
        self.device.is_canonical_link(&self.df,lossf)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> BatchForwardBase for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize + 'static {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> BatchForward for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain +
             ForwardAll<Input=I,Output=PI> + BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchForward + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize + 'static {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_apply(&self.f,&input)?)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> BatchPreTrainBase for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug,
          <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Debug + Send + Sync + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize + 'static {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = Cons<<P as BatchPreTrainBase>::BatchOutStack, Self::BatchPreOutput>;
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> BatchPreTrain for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain  + ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug,
          <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Debug + Send + Sync + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize + 'static {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| {
            self.device.batch_apply(&self.f,input)
        })?;

        Ok(Cons(r,u))
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> BatchBackwardBase for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain  + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI> + BackwardAll<<LI as OutputTensorScalar>::Scalar> +
             OutputScale<Scale=LI,ScaledOutput=LI,ScalingInput=PI> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type> +
             BatchBackward<<LI as OutputTensorScalar>::Scalar> +
             OutputTensorScalar<Scalar=U> + BatchOutputScale,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<<LI as OutputTensorScalar>::Scalar,LI,DA,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <LI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Send + Sync + Debug + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize + 'static {
    type BatchLossInput = <LI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackwardBase>::BatchLossOutput;
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> BatchBackward<<LI as OutputTensorScalar>::Scalar> for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain  + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI> + BackwardAll<<LI as OutputTensorScalar>::Scalar> +
             OutputScale<Scale=LI,ScaledOutput=LI,ScalingInput=PI> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type> +
             BatchBackward<<LI as OutputTensorScalar>::Scalar> +
             OutputTensorScalar<Scalar=U> + BatchOutputScale,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<<LI as OutputTensorScalar>::Scalar,LI,DA,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Send + Sync + Debug + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <LI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize + 'static {
    fn batch_backward(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack)
        -> Result<(<Self as BatchBackwardBase>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        self.parent.batch_backward(input, s)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> BatchLoss<<LI as OutputTensorScalar>::Scalar> for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain  + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI> + BackwardAll<<LI as OutputTensorScalar>::Scalar> +
             OutputScale<Scale=LI,ScaledOutput=LI,ScalingInput=PI> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type> +
             BatchBackward<<LI as OutputTensorScalar>::Scalar> +
             OutputTensorScalar<Scalar=U> + BatchOutputScale,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<<LI as OutputTensorScalar>::Scalar,LI,DA,N>,
          PI: Debug + BatchDataType + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <LI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Send + Sync + Debug + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize + 'static,
          Self: ForwardAll<Output=PI> {
    fn batch_loss(&self, loss: Self::BatchLossInput, stack: Self::BatchOutStack) -> Result<(Self::BatchOutStack, Self::BatchLossInput), TrainingError> {
        let (s,o) = stack.pop();

        let r = {
            let o = self.parent.batch_scaling_mapper(&o)?;

            s.map(|u| {
                self.parent.batch_scaling_mapper(u).map(|u| {
                    self.device.batch_derive(&self.df, o.deref(), &loss, u.deref())
                })
            })??
       };

        Ok((Cons(s,o),r))
    }
}

impl<U,P,A,DA,I,PI,LI,D,const N:usize> OnStep for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PreTrain + OnStep + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        Ok(self.parent.on_step(step)?)
    }

    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        Ok(self.parent.on_frequently_step(step,frequently_step)?)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> PersistProgress<TextFilePersistence,Specialized> for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> +
             PersistProgress<TextFilePersistence,Specialized> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> + PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + std::str::FromStr,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)
    }

    fn save_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;
        persistence.write_layer_start();
        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,P,A,DA,I,PI,LI,D,const N:usize> PersistProgress<T,Linear> for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> +
             PersistProgress<T,Linear> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> + PreTrain + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> InputScale for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PreTrain + OutputTensorScalar<Scalar=U> + InputScale,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U> + DeviceActivation<U,PI,A,N>,
      PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
      LI: Debug + BatchDataType + OutputTensorScalar + 'static,
      I: Debug + Send + Sync,
      Self: ForwardAll<Output=PI> {
    fn scale_mean(&self) -> f32 {
        self.parent.scale_mean()
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> OutputScale for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             OutputTensorScalar<Scalar=U> + OutputScale<Scale=LI,ScaledOutput=LI,ScalingInput=PI> + 'static,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U> + DeviceActivation<U,PI,A,N>,
      PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
      LI: Debug + BatchDataType + OutputTensorScalar + 'static,
      I: Debug + BatchDataType + Send + Sync + 'static,
      <I as BatchDataType>::Type: BatchSize + Debug + 'static,
      <PI as BatchDataType>::Type: BatchSize + Debug + 'static,
      <LI as BatchDataType>::Type: BatchSize + Debug + 'static,
      <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Send + Sync + Debug + 'static {
    type ScalingDevice = <P as OutputScale>::ScalingDevice;
    type Scale = LI;
    type ScaledOutput = LI;
    type ScalingInput = PI;
    type Mapper<'a> = <P as OutputScale>::Mapper<'a> where Self: 'a, <P as OutputScale>::Mapper<'a>: Deref<Target=LI>;
    fn scaling_mapper<'a>(&'a self, input: &'a PI) -> Result<Self::Mapper<'a>,EvaluateError>
        where Self: 'a, <P as OutputScale>::Mapper<'a>: Deref<Target=LI> {
        self.parent.scaling_mapper(input)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> BatchOutputScale for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             OutputScale<Scale=LI,ScaledOutput=LI,ScalingInput=PI> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchOutputScale + OutputTensorScalar<Scalar=U> + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          LI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + BatchDataType + Send + Sync + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'static,
          <PI as BatchDataType>::Type: BatchSize + Debug + 'static,
          <LI as BatchDataType>::Type: BatchSize + Debug + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize + 'static,
          <LI as OutputTensorScalar>::Scalar: Default + Clone + Copy + Send + Sync + Debug + 'static {
    type BatchMapper<'a> = <P as BatchOutputScale>::BatchMapper<'a> where Self: 'a, <P as BatchOutputScale>::BatchMapper<'a>: Deref<Target=<LI as BatchDataType>::Type>;
    fn batch_scaling_mapper<'a>(&'a self, input: &'a <PI as BatchDataType>::Type) -> Result<Self::BatchMapper<'a>,EvaluateError>
        where Self: 'a, <P as BatchOutputScale>::BatchMapper<'a>: Deref<Target=<LI as BatchDataType>::Type> {
        self.parent.batch_scaling_mapper(input)
    }
}
impl<U,P,A,DA,I,PI,LI,D,const N:usize> MaxInputValue for ActivationLayer<U,P,A,DA,I,PI,LI,D,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<<LI as OutputTensorScalar>::Scalar,LossInput=LI> +
             PreTrain + OutputTensorScalar<Scalar=U> + InputScale + MaxInputValue<Scalar=usize>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U> + DeviceActivation<U,PI,A,N>,
      PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
      LI: Debug + BatchDataType + OutputTensorScalar + 'static,
      I: Debug + Send + Sync {
    type Scalar = usize;
    fn max_input_value(&self) -> Self::Scalar {
        self.parent.max_input_value()
    }
}
