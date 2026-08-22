//! Implementation of a Layer That Supports Quantization

use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::{MakeView, MakeViewMut, SliceSize};
use crate::device::Device;
use crate::error::{ModelLoadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use crate::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchPreTrain, BatchPreTrainBase, ContinueForward, ForwardAll, ForwardDiff, PartialForward, PreTrain, UpdateWeight, OnStep, PersistProgress, InputTensorScalar, OutputTensorScalar, InputScale, OutputScale};
use crate::mem::AsRawSlice;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextRecord};
use crate::{Cons, Stack};
use crate::device::bridge::DeviceBridge;

/// Dequantize layer Implementation
pub struct DequantizeLayer<U,SO,P,I,PI,CI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
      U: Debug + Default + Clone + Copy + Send + Sync + 'static,
      SO : Debug + Default + Clone + Copy + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + 'static,
      CI: Debug + 'static,
      I: Debug + Send + Sync {
    parent:P,
    device:D,
    u:PhantomData<U>,
    so:PhantomData<SO>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
    ci:PhantomData<CI>
}
impl<U,SO,P,I,PI,CI,D> InputTensorScalar for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + 'static,
      CI: Debug + 'static,
      I: Debug + Send + Sync {
    type Scalar = U;
}
impl<U,SO,P,I,PI,CI,D> OutputTensorScalar for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + 'static,
      CI: Debug + 'static,
      I: Debug + Send + Sync {
    type Scalar = SO;
}
impl<U,SO,P,I,PI,CI,D> Persistence<TextFilePersistence,Specialized> for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: ForwardAll<Input=I,Output=PI> + Persistence<TextFilePersistence,Specialized> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> + PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
      SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + 'static,
      CI: Debug + 'static,
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
impl<T,U,SO,P,I,PI,CI,D> Persistence<T,Linear> for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + Persistence<T,Linear> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> + PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static,
          CI: Debug + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,SO,P,I,PI,CI,D> ForwardAll for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + 'static + BatchDataType + InputTensorScalar,
          CI: Debug + 'static + BatchDataType + OutputTensorScalar,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = CI;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(self.device.bridge_forward(&self.parent.forward_all(input)?)?)
    }
}
impl<U,SO,P,I,PI,CI,D> PreTrain for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + 'static + BatchDataType + InputTensorScalar,
          CI: Debug + 'static + BatchDataType + OutputTensorScalar,
          I: Debug + Send + Sync {
    type PreOutput = CI;
    type OutStack = Cons<<P as PreTrain>::OutStack,Self::PreOutput>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let s = self.parent.pre_train(input)?;

        let r = s.map(|o| self.device.bridge_forward(o))?;

        Ok(s.push(r))
    }
}
impl<U,SO,P,I,PI,CI,D> BackwardAll<SO> for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             InputTensorScalar + OutputTensorScalar,
          U: Debug + Debug + Default + Clone + Copy + Send + Sync + 'static,
          SO : Debug + Debug + Default + Clone + Copy + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + 'static + BatchDataType + InputTensorScalar,
          CI: Debug + 'static + BatchDataType + OutputTensorScalar,
          I: Debug + Send + Sync {
    type LossInputScalar = SO;
    type LossInput = PI;
    type LossOutput = <P as BackwardAll<SO>>::LossOutput;

    fn backward_all(&mut self, input: Self::LossInput, stack:Self::OutStack)
                    -> Result<(<Self as BackwardAll<SO>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        Ok(self.parent.backward_all(input, s)?)
    }
}
impl<U,SO,P,I,PI,CI,D> UpdateWeight for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> + UpdateWeight +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + BatchDataType + 'static,
          CI: Debug + BatchDataType + 'static,
          I: Debug + Send + Sync {
    type GradientStack = <P as UpdateWeight>::GradientStack;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        Ok(self.parent.update_weight(stack,batch_size)?)
    }
}
impl<U,SO,P,I,PI,CI,D> PartialForward for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             PartialForward + InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + BatchDataType + InputTensorScalar + 'static,
          CI: Debug + BatchDataType + 'static + OutputTensorScalar,
          I: Debug + Send + Sync {
    type PartialInput = <P as PartialForward>::PartialInput;
    type PartialOutput = <P as PartialForward>::PartialOutput;
    type DiffInput = <P as PartialForward>::DiffInput;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward(input)?)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput, partial_input:&Self::PartialInput)
                               -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward_by_diff(input,partial_input)?)
    }
}
impl<U,SO,P,I,PI,CI,D> ForwardDiff for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             PartialForward + ForwardDiff +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + BatchDataType + InputTensorScalar + 'static,
          CI: Debug + BatchDataType + 'static + OutputTensorScalar,
          I: Debug + Send + Sync {
    fn forward_diff(&self, input: Self::DiffInput, partial_input:&Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        Ok(self.device.bridge_forward(&self.parent.forward_diff(input,partial_input)?)?)
    }
}
impl<U,SO,P,I,PI,CI,D> ContinueForward for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             PartialForward + ContinueForward +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + BatchDataType + 'static + InputTensorScalar,
          CI: Debug + BatchDataType + 'static + OutputTensorScalar,
          I: Debug + Send + Sync {
    fn continue_forward(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        Ok(self.device.bridge_forward(&self.parent.continue_forward(input)?)?)
    }
}

impl<U,SO,P,I,PI,CI,D> BatchForwardBase for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchBackward<SO,BatchLossInput=<CI as BatchDataType>::Type> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + BatchDataType + InputTensorScalar + 'static,
          CI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <CI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug,
          for<'a> CI: Debug + SliceSize + AsRawSlice<SO> + MakeView<'a,SO> + MakeViewMut<'a,SO> + 'static {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <CI as BatchDataType>::Type;
}
impl<U,SO,P,I,PI,CI,D> BatchForward for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchForward + BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackward<SO,BatchLossInput=<CI as BatchDataType>::Type> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + BatchDataType + InputTensorScalar + 'static,
          CI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <CI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug,
          for<'a> CI: Debug + SliceSize + AsRawSlice<SO> + MakeView<'a,SO> + MakeViewMut<'a,SO> + 'static {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        Ok(self.device.batch_bridge_forward(&self.parent.batch_forward(input)?)?)
    }
}
impl<U,SO,P,I,PI,CI,D> BatchPreTrainBase for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain +
             BatchBackward<SO,BatchLossInput=<CI as BatchDataType>::Type> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + BatchDataType + InputTensorScalar + 'static,
          CI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + 'static,
          <CI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          for<'a> <CI as BatchDataType>::Type: Debug,
          for<'a> CI: Debug + SliceSize + AsRawSlice<SO> + MakeView<'a,SO> + MakeViewMut<'a,SO> + 'static,
          for<'a> CI: Debug + 'static + BatchDataType {
    type BatchPreOutput = <CI as BatchDataType>::Type;
    type BatchOutStack = Cons<<P as BatchPreTrainBase>::BatchOutStack,Self::BatchPreOutput>;
}
impl<U,SO,P,I,PI,CI,D> BatchPreTrain for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain +
             BatchBackward<SO,BatchLossInput=<CI as BatchDataType>::Type> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + BatchDataType + InputTensorScalar + 'static,
          CI: Debug + BatchDataType + OutputTensorScalar + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + 'static,
          <CI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          for<'a> CI: Debug + SliceSize + AsRawSlice<SO> + MakeView<'a,SO> + MakeViewMut<'a,SO> + 'static {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let s = self.parent.batch_pre_train(input)?;

        let r = s.map(|o| self.device.batch_bridge_forward(o))?;

        Ok(s.push(r))
    }
}
impl<U,SO,P,I,PI,CI,D> BatchBackward<SO> for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain +
             BatchBackward<SO,BatchLossInput=<CI as BatchDataType>::Type> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + BatchDataType + InputTensorScalar + 'static,
          CI: Debug + BatchDataType + OutputTensorScalar + 'static,
          <PI as BatchDataType>::Type: Debug,
          <CI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug,
          for<'a> <CI as BatchDataType>::Type: Debug,
          for<'a> CI: Debug + SliceSize + AsRawSlice<SO> + MakeView<'a,SO> + MakeViewMut<'a,SO> + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug {
    type BatchLossInput = <CI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackward<SO>>::BatchLossOutput;
    fn batch_backward(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack)
                      -> Result<(<Self as BatchBackward<SO>>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        Ok(self.parent.batch_backward(input, s)?)
    }
}
impl<U,SO,P,I,PI,CI,D> OnStep for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          CI: Debug {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        Ok(self.parent.on_step(step)?)
    }
    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        Ok(self.parent.on_frequently_step(step,frequently_step)?)
    }
}
impl<U,SO,P,I,PI,CI,D> PersistProgress<TextFilePersistence,Specialized> for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: ForwardAll<Input=I,Output=PI> +
             PersistProgress<TextFilePersistence,Specialized> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static,
          CI: Debug + 'static,
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
impl<T,U,SO,P,I,PI,CI,D> PersistProgress<T,Linear> for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> +
          PersistProgress<T,Linear> +
          BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
          PreTrain<PreOutput=PI> +
          InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static,
          CI: Debug + 'static,
          I: Debug + Send + Sync {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)
    }
}
impl<U,SO,P,I,PI,CI,D> InputScale for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             PreTrain<PreOutput=PI> + InputScale +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + 'static + BatchDataType + InputTensorScalar,
          CI: Debug + 'static + BatchDataType + OutputTensorScalar,
          I: Debug + Send + Sync {
    fn scale_mean(&self) -> f32 {
        self.parent.scale_mean()
    }
}
impl<U,SO,P,I,PI,CI,D> OutputScale for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             PreTrain<PreOutput=PI> + OutputScale +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBridge<U,SO,PI,CI>,
          PI: Debug + 'static + BatchDataType + InputTensorScalar,
          CI: Debug + 'static + BatchDataType + OutputTensorScalar,
          I: Debug + Send + Sync {
    type Scale = <P as OutputScale>::Scale;
    fn scale(&self) -> &Self::Scale {
        self.parent.scale()
    }
}
/// Trait for DequantizeLayer instance creation
pub trait DequantizeLayerInstantiation<U,SO,P,I,PI,CI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<SO,LossInput=PI,LossInputScalar=SO> + PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static,
          CI: Debug + 'static,
          I: Debug + Send + Sync + 'static + BatchDataType,
          <I as BatchDataType>::Type: Debug + Send + Sync + 'static {
    /// Creates and returns a layer that dequantizes and outputs only the output of the quantized layer.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    fn instantiation(parent:P,device:&D) -> Result<DequantizeLayer<U, SO, P, I, PI, CI, D>,LayerInstantiationError>;
}
impl<U,SO,P,I,PI,CI,D> DequantizeLayerInstantiation<U, SO, P, I, PI, CI, D> for DequantizeLayer<U, SO, P, I, PI, CI, D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<SO,LossInput=PI,LossInputScalar=SO> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static,
          CI: Debug + 'static,
          I: Debug + Send + Sync + 'static + BatchDataType,
          <I as BatchDataType>::Type: Debug + Send + Sync + 'static {
    /// Create and return an instance of DequantizeLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    fn instantiation(parent:P,device:&D) -> Result<DequantizeLayer<U, SO, P, I, PI, CI, D>,LayerInstantiationError> {
        Ok(DequantizeLayer {
            parent:parent,
            device:device.clone(),
            u:PhantomData::<U>,
            so:PhantomData::<SO>,
            i:PhantomData::<I>,
            pi:PhantomData::<PI>,
            ci:PhantomData::<CI>
        })
    }
}
/// Builder for DequantizeLayer instance creation
pub struct DequantizeLayerBuilder<SO,CI>
    where CI: Debug + OutputTensorScalar + 'static {
    so:PhantomData<SO>,
    ci:PhantomData<CI>
}
impl<SO,CI> DequantizeLayerBuilder<SO, CI>
    where CI: Debug + OutputTensorScalar<Scalar=SO> + 'static {
    pub fn new() -> DequantizeLayerBuilder<SO, CI> {
        DequantizeLayerBuilder {
            so:PhantomData::<SO>,
            ci:PhantomData::<CI>
        }
    }

    /// Create an instance of DequantizeLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build<U,P,I,PI,D>(&self,parent:P,device:&D) -> Result<DequantizeLayer<U, SO, P, I, PI, CI, D>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> +
                 BackwardAll<SO,LossInput=PI,LossInputScalar=SO> + PreTrain<PreOutput=PI> + InputTensorScalar + OutputTensorScalar,
              U: Default + Clone + Copy + Debug + Send + Sync + 'static,
              SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
              D: Device<U>,
              PI: Debug + InputTensorScalar + 'static,
              CI: Debug + OutputTensorScalar<Scalar=SO> + 'static,
              I: Debug + Send + Sync + 'static + BatchDataType,
              <I as BatchDataType>::Type: Debug + Send + Sync + 'static,
              DequantizeLayer<U, SO, P, I, PI, CI, D>: DequantizeLayerInstantiation<U, SO, P, I, PI, CI, D>
    {
        DequantizeLayer::<U, SO, P, I, PI, CI, D>::instantiation(parent, device)
    }
}


