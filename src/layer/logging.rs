//! Implementation of a layer for log collection

use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::device::Device;
use crate::error::{ModelLoadError, EvaluateError, PersistenceError, TrainingError};
use crate::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchPreTrain, BatchPreTrainBase, ContinueForward, ForwardAll, ForwardDiff, PartialForward, PreTrain, UpdateWeight, OnStep, PersistProgress, InputTensorScalar, OutputTensorScalar, InputScale, Bridge, BatchBridge, InputMax, PreTrainBase, BatchSize, BackwardBase, BatchBackwardBase, Loss, BatchLoss, BridgeBase, BridgeRepr, BatchBridgeRepr};
use crate::lossfunction::LossFunction;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextRecord};
use crate::Stack;

/// Logging layer Implementation
pub struct LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    parent:P,
    device:PhantomData<D>,
    u:PhantomData<U>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
    li:PhantomData<LI>,
    // Added logger fields
    forward_loggers: Vec<Box<dyn Fn(&PI) -> Result<(),EvaluateError> + 'static>>,
    backward_loggers: Vec<Box<dyn Fn(&LI) -> Result<(), TrainingError> + 'static>>,
    gradient_loggers: Vec<Box<dyn Fn(&<<P as UpdateWeight>::GradientStack as crate::Stack>::Head) -> Result<(),TrainingError> + 'static>>,
    batch_forward_loggers: Vec<Box<dyn Fn(&<PI as BatchDataType>::Type) -> Result<(),TrainingError> + 'static>>,
    batch_backward_loggers: Vec<Box<dyn Fn(&<LI as BatchDataType>::Type) -> Result<(),TrainingError> + 'static>>,
}
impl<U,P,I,PI,LI,D> InputTensorScalar for LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type Scalar = U;
}
impl<U,P,I,PI,LI,D> OutputTensorScalar for LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type Scalar = U;
}
impl<U,P,I,PI,LI,D> LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    /// Create and return an instance of LoggingLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    pub fn new(parent:P,_:&D) -> LoggingLayer<U,P,I,PI,LI,D> {
        LoggingLayer {
            parent:parent,
            device:PhantomData::<D>,
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            pi:PhantomData::<PI>,
            li:PhantomData::<LI>,
            forward_loggers: Vec::new(),
            backward_loggers: Vec::new(),
            gradient_loggers: Vec::new(),
            batch_forward_loggers: Vec::new(),
            batch_backward_loggers: Vec::new(),
        }
    }
    pub fn add_forward_logger<F>(&mut self, logger: F) where F: Fn(&PI) -> Result<(),EvaluateError> + 'static {
        self.forward_loggers.push(Box::new(logger));
    }

    pub fn add_backward_logger<F>(&mut self, logger: F) where F: Fn(&LI) -> Result<(),TrainingError> + 'static {
        self.backward_loggers.push(Box::new(logger));
    }

    pub fn add_gradient_logger<F>(&mut self, logger: F)
        where F: Fn(&<<P as UpdateWeight>::GradientStack as crate::Stack>::Head) -> Result<(),TrainingError> + 'static {
        self.gradient_loggers.push(Box::new(logger));
    }

    pub fn add_batch_forward_logger<F>(&mut self, logger: F)
        where F: Fn(&<PI as BatchDataType>::Type) -> Result<(),TrainingError> + 'static {
        self.batch_forward_loggers.push(Box::new(logger));
    }

    pub fn add_batch_backward_logger<F>(&mut self, logger: F)
        where F: Fn(&<LI as BatchDataType>::Type) -> Result<(),TrainingError> + 'static {
        self.batch_backward_loggers.push(Box::new(logger));
    }
}
impl<U,P,I,PI,LI,D> Persistence<TextFilePersistence,Specialized> for LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> + Persistence<TextFilePersistence,Specialized> +
             BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err>,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
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
impl<T,U,P,I,PI,LI,D> Persistence<T,Linear> for LoggingLayer<U,P,I,PI,LI,D>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + Persistence<T,Linear> +
             BackwardBase<LossInput=LI>  + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,I,PI,LI,D> ForwardAll for LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type Input = I;
    type Output = PI;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        let r = self.parent.forward_all(input)?;

        for logger in self.forward_loggers.iter() {
            logger(&r)?;
        }

        Ok(r)
    }
}
impl<U,P,I,PI,LI,D> PreTrainBase for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type PreOutput = PI;
    type OutStack = <P as PreTrainBase>::OutStack;
}
impl<U,P,I,PI,LI,D> PreTrain for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let s = self.parent.pre_train(input)?;

        for logger in self.forward_loggers.iter() {
            s.map(|r| {
                logger(r)
            })?;
        }
        Ok(s)
    }
}
impl<U,P,I,PI,LI,D> BackwardBase for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type LossInputScalar = <LI as InputTensorScalar>::Scalar;
    type LossInput = <P as BackwardBase>::LossInput;
}
impl<U,P,I,PI,LI,D> BackwardAll<<LI as InputTensorScalar>::Scalar> for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  + BackwardAll<<LI as InputTensorScalar>::Scalar> +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type LossOutput = <P as BackwardAll<<LI as InputTensorScalar>::Scalar>>::LossOutput;

    fn backward_all(&mut self, input: Self::LossInput, stack:Self::OutStack)
        -> Result<(<Self as BackwardAll<<LI as InputTensorScalar>::Scalar>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        for logger in self.backward_loggers.iter() {
            logger(&input)?;
        }

        Ok(self.parent.backward_all(input, stack)?.into())
    }
}
impl<U,P,I,PI,LI,D> Loss<<LI as InputTensorScalar>::Scalar> for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  + BackwardAll<<LI as InputTensorScalar>::Scalar> +
             Loss<<LI as InputTensorScalar>::Scalar> +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    fn loss(&mut self, loss: Self::LossInput, stack: Self::OutStack) -> Result<(Self::OutStack, Self::LossInput), TrainingError> {
        self.parent.loss(loss,stack)
    }

    fn is_canonical_link<L: LossFunction<<LI as InputTensorScalar>::Scalar>>(&self, lossf: &L) -> bool {
        self.parent.is_canonical_link(lossf)
    }
}
impl<U,P,I,PI,LI,D> UpdateWeight for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type GradientStack = <P as UpdateWeight>::GradientStack;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        for logger in self.gradient_loggers.iter() {
            stack.map(|r| {
                logger(r)
            })?;
        }

        Ok(self.parent.update_weight(stack,batch_size)?)
    }
}
impl<U,P,I,PI,LI,D> PartialForward for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             PartialForward,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type PartialInput = <P as PartialForward>::PartialInput;
    type PartialOutput = <P as PartialForward>::PartialOutput;
    type DiffInput = <P as PartialForward>::DiffInput;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward(input)?)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput)
        -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward_by_diff(input,partial_input)?)
    }
}
impl<U,P,I,PI,LI,D> ForwardDiff for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             PartialForward + ForwardDiff +
             BackwardBase<LossInput=LI>  +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + BatchDataType,
      LI: Debug + 'static + BatchDataType + InputTensorScalar,
      I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    fn forward_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let r = self.parent.forward_diff(input,partial_input)?;

        for logger in self.forward_loggers.iter() {
            logger(&r)?;
        }

        Ok(r)
    }
}
impl<U,P,I,PI,LI,D> ContinueForward for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
          PartialForward + ContinueForward +
          BackwardBase<LossInput=LI>  +
          InputTensorScalar + OutputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + BatchDataType,
      LI: Debug + 'static + BatchDataType + InputTensorScalar,
      I: Debug + Send + Sync,
      <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    fn continue_forward(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let r = self.parent.continue_forward(input)?;

        for logger in self.forward_loggers.iter() {
            logger(&r)?;
        }

        Ok(r)
    }
}
impl<U,P,I,PI,LI,D> BatchForwardBase for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             Bridge<RealScale=LI, RealOutput=LI, SourceInput=PI> +
             BackwardBase<LossInput=LI>  +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: BatchSize + Debug,
          <LI as BatchDataType>::Type: BatchSize + Debug,
          <I as BatchDataType>::Type: BatchSize + Debug {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,P,I,PI,LI,D> BatchForward for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             Bridge<RealScale=LI, RealOutput=LI, SourceInput=PI> +
             BackwardBase<LossInput=LI>  +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: BatchSize + Debug,
          <LI as BatchDataType>::Type: BatchSize + Debug,
          <I as BatchDataType>::Type: BatchSize + Debug,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let r = self.parent.batch_forward(input)?;

        for logger in self.batch_forward_loggers.iter() {
            logger(&r)?;
        }

        Ok(r)
    }
}
impl<U,P,I,PI,LI,D> BatchPreTrainBase for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type> +
             BatchBackward<<LI as InputTensorScalar>::Scalar>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = <P as BatchPreTrainBase>::BatchOutStack;
}
impl<U,P,I,PI,LI,D> BatchPreTrain for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             Bridge<RealScale=LI, RealOutput=LI, SourceInput=PI> +
             BackwardBase<LossInput=LI>  +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type> +
             BatchBackward<<LI as InputTensorScalar>::Scalar>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: BatchSize + Debug,
          <LI as BatchDataType>::Type: BatchSize + Debug,
          <I as BatchDataType>::Type: BatchSize + Debug,
          <P as BackwardBase>::LossInput: BatchDataType + OutputTensorScalar + Debug + 'static,
          <LI as InputTensorScalar>::Scalar: Debug + 'static {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let s = self.parent.batch_pre_train(input)?;

        for logger in self.batch_forward_loggers.iter() {
            s.map(|r| {
                logger(r)
            })?;
        }

        Ok(s)
    }
}
impl<U,P,I,PI,LI,D> BatchBackwardBase for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type> +
             BatchBackward<<LI as InputTensorScalar>::Scalar>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug,
          <P as BackwardBase>::LossInput: BatchDataType + OutputTensorScalar + Debug + 'static,
          <LI  as BatchDataType>::Type: Debug + 'static,
          <LI as InputTensorScalar>::Scalar: Debug + 'static {
    type BatchLossInput = <LI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackwardBase>::BatchLossOutput;
}
impl<U,P,I,PI,LI,D> BatchBackward<<LI as InputTensorScalar>::Scalar> for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type> +
             BatchBackward<<LI as InputTensorScalar>::Scalar>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug,
          <P as BackwardBase>::LossInput: BatchDataType + OutputTensorScalar + Debug + 'static,
          <LI  as BatchDataType>::Type: Debug + 'static {
    fn batch_backward(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack)
        -> Result<(<Self as BatchBackwardBase>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        for logger in self.batch_backward_loggers.iter() {
            logger(&input)?;
        }

        let r = self.parent.batch_backward(input, stack)?;

        Ok(r)
    }
}
impl<U,P,I,PI,LI,D> BatchLoss<<LI as InputTensorScalar>::Scalar>
    for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type> +
             BatchBackward<<LI as InputTensorScalar>::Scalar> +
             BatchLoss<<LI as InputTensorScalar>::Scalar>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug,
          <P as BackwardBase>::LossInput: BatchDataType + OutputTensorScalar + Debug + 'static,
          <LI  as BatchDataType>::Type: Debug + 'static {
    fn batch_loss(&self, loss: Self::BatchLossInput, stack: Self::BatchOutStack) -> Result<(Self::BatchOutStack, Self::BatchLossInput), TrainingError> {
        self.parent.batch_loss(loss,stack)
    }
}
impl<U,P,I,PI,LI,D> OnStep for LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U> + OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        Ok(self.parent.on_step(step)?)
    }
    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        Ok(self.parent.on_frequently_step(step,frequently_step)?)
    }
}
impl<U,P,I,PI,LI,D> PersistProgress<TextFilePersistence,Specialized> for LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             PersistProgress<TextFilePersistence,Specialized> +
             BackwardBase<LossInput=LI>  + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err>,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
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
impl<T,U,P,I,PI,LI,D> PersistProgress<T,Linear> for LoggingLayer<U,P,I,PI,LI,D>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> +
             PersistProgress<T,Linear> +
             BackwardBase<LossInput=LI>  + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)
    }
}
impl<U,P,I,PI,LI,D> InputScale for LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain + InputScale +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + 'static + BatchDataType,
      LI: Debug + 'static + BatchDataType + InputTensorScalar,
      I: Debug + Send + Sync,
      <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    fn input_scale(&self) -> f32 {
        self.parent.input_scale()
    }
}
impl<U,P,I,PI,LI,D> BridgeBase for LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain +
             BridgeBase<RealScale=LI, RealOutput=LI, SourceInput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <LI as BatchDataType>::Type: Debug + BatchSize,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type UseDevice = <P as BridgeBase>::UseDevice;
    type RealScale = <P as BridgeBase>::RealScale;
    type RealOutput = <P as BridgeBase>::RealOutput;
    type SourceInput = <P as BridgeBase>::SourceInput;
}
impl<U,P,I,PI,LI,D> Bridge for LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain +
             BridgeBase<RealScale=LI, RealOutput=LI, SourceInput=PI> + Bridge +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
      I: Debug + Send + Sync,
      <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
      <LI as BatchDataType>::Type: BatchSize + Debug,
      <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type RealMapper<'a> = <P as Bridge>::RealMapper<'a> where Self: 'a;
    fn as_real<'a>(&'a self, input:&'a PI) -> Result<Self::RealMapper<'a>,EvaluateError> where Self: 'a {
        self.parent.as_real(input)
    }
}
impl<U,P,I,PI,LI,D> BridgeRepr for LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             BridgeBase<RealScale=LI, RealOutput=LI, SourceInput=PI> +
             BridgeRepr<RepresentationOutput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <LI as BatchDataType>::Type: Debug + BatchSize,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type RepresentationOutput = PI;
    type ReprMapper<'a> = <P as BridgeRepr>::ReprMapper<'a> where Self: 'a;
    fn as_repr<'a>(&'a self, input: &'a Self::RealOutput) -> Result<Self::ReprMapper<'a>, EvaluateError>
        where Self: 'a {
        self.parent.as_repr(input)
    }
}
impl<U,P,I,PI,LI,D> BatchBridge for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             BridgeBase<RealScale=LI, RealOutput=LI, SourceInput=PI> + Bridge +
             BatchBridge +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: BatchSize + Debug,
          <LI as BatchDataType>::Type: BatchSize + Debug,
          <I as BatchDataType>::Type: BatchSize + Debug {
    type BatchRealMapper<'a> = <P as BatchBridge>::BatchRealMapper<'a> where Self: 'a;
    fn batch_as_real<'a>(&'a self, input:&'a <Self as BatchForwardBase>::BatchOutput) -> Result<Self::BatchRealMapper<'a>,EvaluateError> where Self: 'a {
        self.parent.batch_as_real(input)
    }
}
impl<U,P,I,PI,LI,D> BatchBridgeRepr for LoggingLayer<U,P,I,PI,LI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             BridgeBase<RealScale=LI, RealOutput=LI, SourceInput=PI> + BridgeRepr<RepresentationOutput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U> + BatchBridgeRepr +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase +
             BatchBackwardBase<BatchLossInput=<LI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          LI: Debug + 'static + BatchDataType + InputTensorScalar,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: BatchSize + Debug,
          <LI as BatchDataType>::Type: Debug + BatchSize,
          <I as BatchDataType>::Type: BatchSize + Debug,
          <P as BackwardBase>::LossInput: BatchDataType + Debug + 'static {
    type BatchReprMapper<'a> = <P as BatchBridgeRepr>::BatchReprMapper<'a> where Self: 'a;
    fn batch_as_repr<'a>(&'a self, input: &'a <LI as BatchDataType>::Type) -> Result<Self::BatchReprMapper<'a>,EvaluateError> where Self: 'a {
        self.parent.batch_as_repr(input)
    }
}
impl<U,P,I,PI,LI,D> InputMax for LoggingLayer<U,P,I,PI,LI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardBase<LossInput=LI>  +
             PreTrainBase<PreOutput=PI> + PreTrain + InputScale + InputMax +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + 'static + BatchDataType,
      LI: Debug + 'static + BatchDataType + InputTensorScalar,
      I: Debug + Send + Sync,
      <P as BackwardBase>::LossInput: BatchDataType + OutputTensorScalar + Debug + 'static {
    type Scalar = <P as InputMax>::Scalar;
    fn input_max(&self) -> Self::Scalar {
        self.parent.input_max()
    }
}
