//! Implementation of inverse scaling layer

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Deref;
use std::str::FromStr;
use crate::{Cons, Stack};
use crate::device::clone::DeviceClone;
use crate::device::Device;
use crate::device::scale::DeviceScale;
use crate::error::{ModelLoadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError, InvalidStateError};
use crate::layer::{Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchPreTrain, BatchPreTrainBase, BatchSize, ContinueForward, Forward, ForwardAll, ForwardDiff, PartialForward, PreTrain, UpdateWeight, OnStep, PersistProgress, InputTensorScalar, OutputTensorScalar, InputScale, OutputScale, PreTrainBase, BatchOutputScale, BackwardBase, BatchBackwardBase};
use crate::mapper::{BatchIdentityMapper, IdentityMapper};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextRecord};

/// Trait for InverseScalingLayer instance creation.
pub trait ScalingLayerInstantiation<U,P,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    /// Create and return an instance.
    ///
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    fn instantiation(parent:P,device:&D) -> Result<ScalingLayer<U,P,D,I,PI,N>,LayerInstantiationError>;
}

/// Inverse scaling layer implementation.
pub struct ScalingLayer<U,P,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    parent:P,
    device:D,
    u:PhantomData<U>,
    i:PhantomData<I>,
    pi:PhantomData<PI>
}
impl<U,P,D,I,PI,const N:usize> InputTensorScalar for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    type Scalar = U;
}
impl<U,P,D,I,PI,const N:usize> OutputTensorScalar for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    type Scalar = U;
}
impl<U,P,D,I,PI,const N:usize> Persistence<TextFilePersistence,Specialized> for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> +
             Persistence<TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err>,
          D: Device<U> + DeviceScale<U,PI,N,Scale=PI>,
          <PI as BatchDataType>::Type: Debug + BatchSize {
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
impl<T,U,P,D,I,PI,const N:usize> Persistence<T,Linear> for ScalingLayer<U,P,D,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> + Persistence<T,Linear>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          D: Device<U> + DeviceScale<U,PI,N,Scale=PI>,
          <PI as BatchDataType>::Type: Debug + BatchSize {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,D,I,PI,const N:usize> Forward<PI,Result<PI,EvaluateError>> for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          <PI as BatchDataType>::Type: Debug + BatchSize,
          for<'a> <P as OutputScale>::Mapper<'a>: Deref<Target=PI>,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> {
    fn forward(&self,input:&PI) -> Result<PI,EvaluateError> {
        Ok(self.device.cloned(self.parent.scaling_mapper(input)?.deref())?)
    }
}
impl<U,P,D,I,PI,const N:usize> ForwardAll for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          for<'a> <P as OutputScale>::Mapper<'a>: Deref<Target=PI>,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> {
    type Input = I;
    type Output = PI;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,P,D,I,PI,const N:usize> PreTrainBase for ScalingLayer<U,P,D,I,PI,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          for<'a>D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> {
    type PreOutput = PI;
    type OutStack = Cons<<P as PreTrainBase>::OutStack, Self::PreOutput>;
}
impl<U,P,D,I,PI,const N:usize> PreTrain for ScalingLayer<U,P,D,I,PI,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          for<'a> <P as OutputScale>::Mapper<'a>: Deref<Target=PI>,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> {
    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;
        let u = r.map(|r| self.forward(r))?;

        Ok(Cons(r,u))
    }
}
impl<U,P,D,I,PI,const N:usize> Backward<U,PI,Result<PI,TrainingError>> for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          for<'a> <P as OutputScale>::Mapper<'a>: Deref<Target=PI>,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> {
    fn backward(&mut self, input: PI) -> Result<PI,TrainingError> {
        Ok(self.device.cloned(self.parent.scaling_mapper(&input)?.deref())?)
    }
}
impl<U,P,D,I,PI,const N:usize> BackwardBase for ScalingLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI,LossInputScalar=U> + ForwardAll<Input=I,Output=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> {
    type LossInputScalar = U;
    type LossInput = PI;
}
impl<U,P,D,I,PI,const N:usize> BackwardAll<U> for ScalingLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI,LossInputScalar=U> + ForwardAll<Input=I,Output=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> {
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all(&mut self, input: Self::LossInput, stack:Self::OutStack)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();
        let next_loss = self.backward(input)?;

        self.parent.backward_all(next_loss.into(), s)
    }
}
impl<U,P,D,I,PI,const N:usize> UpdateWeight for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> + UpdateWeight,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> {
    type GradientStack = <P as UpdateWeight>::GradientStack;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        self.parent.update_weight(stack,batch_size)
    }
}
impl<U,P,D,I,PI,const N:usize> PartialForward for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          Self: ForwardAll<Input=I,Output=PI>,
          Self: PreTrain,
          for<'a> <P as OutputScale>::Mapper<'a>: Deref<Target=PI>,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> {
    type PartialInput = <P as PartialForward>::PartialInput;
    type PartialOutput = <P as PartialForward>::PartialOutput;
    type DiffInput = <P as PartialForward>::DiffInput;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        self.parent.partial_forward(input)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput, partial_input:&Self::PartialInput)
        -> Result<Self::PartialOutput, EvaluateError> {
        self.parent.partial_forward_by_diff(input, partial_input)
    }
}
impl<U,P,D,I,PI,const N:usize> ForwardDiff for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward + ForwardDiff +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      I: Debug + Send + Sync,
      PI: Debug + BatchDataType + 'static,
      <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
      Self: ForwardAll<Input=I,Output=PI>,
      Self: PreTrain,
      for<'a> <P as OutputScale>::Mapper<'a>: Deref<Target=PI>,
      for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> {
    fn forward_diff(&self, input: Self::DiffInput, partial_input:&Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_diff(input, partial_input)?;

        self.forward(&input)
    }
}
impl<U,P,D,I,PI,const N:usize> ContinueForward for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward + ContinueForward +
          BackwardAll<U,LossInput=PI,LossInputScalar=U> + PreTrainBase<PreOutput=PI> + PreTrain +
          InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      I: Debug + Send + Sync,
      PI: Debug + BatchDataType + 'static,
      <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
      Self: ForwardAll<Input=I,Output=PI>,
      Self: PreTrain,
      for<'a> <P as OutputScale>::Mapper<'a>: Deref<Target=PI>,
      for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> {
    fn continue_forward(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.continue_forward(input)?;

        self.forward(&input)
    }
}
impl<U,P,D,I,PI,const N:usize> BatchForwardBase for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> +
             BatchOutputScale +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          Self: ForwardAll,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> + DeviceClone<'a,<PI as BatchDataType>::Type> {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,P,D,I,PI,const N:usize> BatchForward for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> +
             BatchOutputScale +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> +
                     DeviceClone<'a,<PI as BatchDataType>::Type> {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;
        let input = self.parent.batch_scaling_mapper(&input)?;

        Ok(self.device.cloned(input.deref())?)
    }
}
impl<U,P,D,I,PI,const N:usize> BatchPreTrainBase for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> +
             BatchOutputScale +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          Self: PreTrain<PreOutput=PI>,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> + DeviceClone<'a,<PI as BatchDataType>::Type> {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = Cons<<P as BatchPreTrainBase>::BatchOutStack,Self::BatchPreOutput>;
}
impl<U,P,D,I,PI,const N:usize> BatchPreTrain for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> +
             BatchOutputScale +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> + DeviceClone<'a,<PI as BatchDataType>::Type> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let s = self.parent.batch_pre_train(input)?;

        let u = s.map(|input| {
            self.parent.batch_scaling_mapper(input).and_then(|input| {
                self.device.cloned(input.deref())
            })
        })?;

        Ok(Cons(s,u))
    }
}
impl<U,P,D,I,PI,const N:usize> BatchBackwardBase for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> +
             BatchOutputScale +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> + DeviceClone<'a,<PI as BatchDataType>::Type> {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackwardBase>::BatchLossOutput;
}
impl<U,P,D,I,PI,const N:usize> BatchBackward<U> for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> +
             BatchOutputScale +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          for<'a> D: Device<U> + DeviceScale<U,PI,N,Scale=PI> + DeviceClone<'a,PI> + DeviceClone<'a,<PI as BatchDataType>::Type> {
    fn batch_backward(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack)
        -> Result<(<Self as BatchBackwardBase>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let next_loss = self.device.cloned(self.parent.batch_scaling_mapper(&input)?.deref())?;

        self.parent.batch_backward(next_loss, s)
    }
}
impl<U,P,D,I,PI,const N:usize> OnStep for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> + OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        self.parent.on_step(step)
    }
    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        self.parent.on_frequently_step(step,frequently_step)
    }
}
impl<U,P,D,I,PI,const N:usize> PersistProgress<TextFilePersistence,Specialized> for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> +
             PersistProgress<TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          D: Device<U> + DeviceScale<U,PI,N,Scale=PI>,
          <PI as BatchDataType>::Type: Debug + BatchSize,
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
impl<T,U,P,D,I,PI,const N:usize> PersistProgress<T,Linear> for ScalingLayer<U,P,D,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> +
             PersistProgress<T,Linear>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          D: Device<U> + DeviceScale<U,PI,N,Scale=PI>,
          <PI as BatchDataType>::Type: Debug + BatchSize {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)
    }
}
impl<U,P,D,I,PI,const N:usize> InputScale for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain + InputScale + OutputScale<Scale=PI,ScaledOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
      D: Device<U> + DeviceScale<U,PI,N,Scale=PI>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      I: Debug + Send + Sync,
      PI: Debug + BatchDataType + 'static,
      <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
      Self: ForwardAll<Output=PI>,
      for<'a> D: DeviceClone<'a,PI> {
    fn scale_mean(&self) -> f32 {
        self.parent.scale_mean()
    }
}
impl<U,P,D,I,PI,const N:usize> OutputScale for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
      D: Device<U> + DeviceScale<U,PI,N,Scale=PI>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      I: Debug + Send + Sync,
      PI: Debug + BatchDataType + 'static,
      <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
      for<'a> D: DeviceClone<'a,PI> {
    type ScalingDevice = D;
    type Scale = PI;
    type ScaledOutput = PI;
    type Mapper<'a> = IdentityMapper<'a,PI,Self::ScalingDevice> where Self: 'a;

    fn scaling_mapper<'a>(&'a self, input: &'a <Self as ForwardAll>::Output) -> Result<Self::Mapper<'a>,EvaluateError> where Self: 'a {
        Ok(IdentityMapper::new(input))
    }
}
impl<U,P,D,I,PI,const N:usize> BatchOutputScale for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI> +
             BatchOutputScale +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type>,
      D: Device<U> + DeviceScale<U,PI,N,Scale=PI>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      I: Debug + Send + Sync + BatchDataType,
      PI: Debug + BatchDataType + 'static,
      <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
      <I as BatchDataType>::Type: Debug,
      Self: ForwardAll<Output=PI>,
      Self: BatchForwardBase<BatchOutput=<PI as BatchDataType>::Type>,
      for<'a> D: DeviceClone<'a,PI> {
    type BatchMapper<'a> = BatchIdentityMapper<'a,PI,Self::ScalingDevice> where Self: 'a;

    fn batch_scaling_mapper<'a>(&'a self, input: &'a<PI as BatchDataType>::Type) -> Result<Self::BatchMapper<'a>,EvaluateError> where Self: 'a {
        Ok(BatchIdentityMapper::new(input))
    }
}
impl<U,P,D,I,PI,const N:usize> ScalingLayerInstantiation<U,P,D,I,PI,N> for ScalingLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + OutputScale<Scale=PI,ScaledOutput=PI>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          D: Device<U> + DeviceScale<U,PI,N,Scale=PI>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    fn instantiation(parent: P, device: &D)
        -> Result<ScalingLayer<U,P,D,I,PI,N>, LayerInstantiationError> {
        Ok(ScalingLayer {
            parent: parent,
            device: device.clone(),
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            pi:PhantomData::<PI>
        })
    }
}
/// Builder for InverseScalingLayer instance creation.
pub struct InverseScalingLayerBuilder<const N:usize> {
}
impl<const N:usize> InverseScalingLayerBuilder<N> {
    pub fn new() -> InverseScalingLayerBuilder<N> {
        InverseScalingLayerBuilder {}
    }

    /// Create an instance of InverseScalingLayer.
    ///
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build<U,P,D,I,PI>(&self,parent:P,device:&D)
        -> Result<ScalingLayer<U,P,D,I,PI,N>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> +
                 BackwardAll<U,LossInput=PI,LossInputScalar=U> + PreTrain +
                 InputTensorScalar + OutputTensorScalar<Scalar=U> + OutputScale<Scale=PI,ScaledOutput=PI>,
              U: Default + Clone + Copy + Debug + Send + Sync + 'static,
              D: Device<U>,
              I: Debug + Send + Sync + BatchDataType,
              PI: Debug + BatchDataType,
              <I as BatchDataType>::Type: Debug + Send + Sync + 'static,
              <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
              ScalingLayer<U,P,D,I,PI,N>: ScalingLayerInstantiation<U,P,D,I,PI,N> {
        ScalingLayer::instantiation(parent, device)
    }
}
