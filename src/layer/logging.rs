//! Implementation of a layer for log collection

use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::device::Device;
use crate::error::{ModelLoadError, EvaluateError, PersistenceError, TrainingError};
use crate::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, ContinueForward, ForwardAll, ForwardDiff, Loss, PartialForward, PreTrain, UpdateWeight, OnStep, PersistProgress, InputTensorScalar, OutputTensorScalar};
use crate::lossfunction::LossFunction;
use crate::ope::UnitValue;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextRecord};
use crate::Stack;

/// Logging layer Implementation
pub struct LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<PreOutput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: UnitValue<U>,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          I: Debug + Send + Sync {
    parent:P,
    device:PhantomData<D>,
    u:PhantomData<U>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
    // Added logger fields
    forward_loggers: Vec<Box<dyn Fn(&PI) -> Result<(),EvaluateError> + 'static>>,
    backward_loggers: Vec<Box<dyn Fn(&PI) -> Result<(), TrainingError> + 'static>>,
    gradient_loggers: Vec<Box<dyn Fn(&<<P as UpdateWeight>::GradientStack as crate::Stack>::Head) -> Result<(),TrainingError> + 'static>>,
    batch_forward_loggers: Vec<Box<dyn Fn(&<PI as BatchDataType>::Type) -> Result<(),TrainingError> + 'static>>,
    batch_backward_loggers: Vec<Box<dyn Fn(&<PI as BatchDataType>::Type) -> Result<(),TrainingError> + 'static>>,
}
impl<U,P,I,PI,D> InputTensorScalar<U> for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<PreOutput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: UnitValue<U>,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          I: Debug + Send + Sync {}
impl<U,P,I,PI,D> OutputTensorScalar<U> for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<PreOutput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: UnitValue<U>,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          I: Debug + Send + Sync {}
impl<U,P,I,PI,D> LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<PreOutput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: UnitValue<U>,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          I: Debug + Send + Sync {
    /// Create and return an instance of LoggingLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    pub fn new(parent:P,_:&D) -> LoggingLayer<U,P,I,PI,D> {
        LoggingLayer {
            parent:parent,
            device:PhantomData::<D>,
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            pi:PhantomData::<PI>,
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

    pub fn add_backward_logger<F>(&mut self, logger: F) where F: Fn(&PI) -> Result<(),TrainingError> + 'static {
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
        where F: Fn(&<PI as BatchDataType>::Type) -> Result<(),TrainingError> + 'static {
        self.batch_backward_loggers.push(Box::new(logger));
    }
}
impl<U,P,I,PI,D> Persistence<U,TextFilePersistence,Specialized> for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + Persistence<U,TextFilePersistence,Specialized> +
             BackwardAll<U,LossInput=PI> +
             PreTrain<PreOutput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: UnitValue<U> + std::str::FromStr,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
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
impl<T,U,P,I,PI,D> Persistence<U,T,Linear> for LoggingLayer<U,P,I,PI,D>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=PI> + PreTrain<PreOutput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: UnitValue<U>,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,I,PI,D> ForwardAll for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<PreOutput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          I: Debug + Send + Sync {
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
impl<U,P,I,PI,D> PreTrain for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrain<PreOutput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
    type PreOutput = PI;
    type OutStack = <P as PreTrain>::OutStack;

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
impl<U,P,I,PI,D> BackwardAll<U> for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync, {
    type LossInput = PI;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        for logger in self.backward_loggers.iter() {
            logger(&input)?;
        }

        Ok(self.parent.backward_all(input, stack, lossf)?.into())
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, lossf: &L) -> bool {
        self.parent.is_canonical_link(lossf)
    }
}
impl<U,P,I,PI,D> UpdateWeight for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             Loss<U> + UpdateWeight + OutputTensorScalar<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync, {
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
impl<U,P,I,PI,D> PartialForward for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + OutputTensorScalar<U> +
             PartialForward,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
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
impl<U,P,I,PI,D> ForwardDiff for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             PartialForward + ForwardDiff +
             BackwardAll<U,LossInput=PI> + Loss<U> + OutputTensorScalar<U>,
      U: Default + Clone + Copy + UnitValue<U>,
      D: Device<U>,
      PI: Debug + BatchDataType,
      I: Debug + Send + Sync {
    fn forward_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let r = self.parent.forward_diff(input,partial_input)?;

        for logger in self.forward_loggers.iter() {
            logger(&r)?;
        }

        Ok(r)
    }
}
impl<U,P,I,PI,D> ContinueForward for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
          PartialForward + ContinueForward +
          BackwardAll<U,LossInput=PI> + Loss<U> + OutputTensorScalar<U>,
      U: Default + Clone + Copy + UnitValue<U>,
      D: Device<U>,
      PI: Debug + BatchDataType,
      I: Debug + Send + Sync {
    fn continue_forward(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let r = self.parent.continue_forward(input)?;

        for logger in self.forward_loggers.iter() {
            logger(&r)?;
        }

        Ok(r)
    }
}
impl<U,P,I,PI,D> Loss<U> for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
    fn loss<L: LossFunction<U>>(&mut self, loss: Self::LossInput, lossf: &L, stack: Self::OutStack) -> Result<(Self::OutStack, Self::LossInput), TrainingError> {
        Ok(self.parent.loss(loss,lossf,stack)?)
    }
}
impl<U,P,I,PI,D> BatchForwardBase for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + OutputTensorScalar<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,P,I,PI,D> BatchForward for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + OutputTensorScalar<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> + BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let r = self.parent.batch_forward(input)?;

        for logger in self.batch_forward_loggers.iter() {
            logger(&r)?;
        }

        Ok(r)
    }
}
impl<U,P,I,PI,D> BatchPreTrainBase for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + OutputTensorScalar<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> + BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = <P as BatchPreTrainBase>::BatchOutStack;
}
impl<U,P,I,PI,D> BatchPreTrain for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + OutputTensorScalar<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> + BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
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
impl<U,P,I,PI,D> BatchBackward<U> for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + OutputTensorScalar<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> + BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;
    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        for logger in self.batch_backward_loggers.iter() {
            logger(&input)?;
        }

        let r = self.parent.batch_backward(input, stack, lossf)?;

        Ok(r)
    }
}
impl<U,P,I,PI,D> BatchLoss<U> for LoggingLayer<U,P,I,PI,D>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + OutputTensorScalar<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug {
    fn batch_loss<L: LossFunction<U>>(&self, loss: Self::BatchLossInput, lossf: &L, stack: Self::BatchOutStack) -> Result<(Self::BatchOutStack, Self::BatchLossInput), TrainingError> {
        Ok(self.parent.batch_loss(loss,lossf,stack)?)
    }
}
impl<U,P,I,PI,D> OnStep for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<PreOutput=PI> + Loss<U> + OutputTensorScalar<U> + OnStep,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        Ok(self.parent.on_step(step)?)
    }
    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        Ok(self.parent.on_frequently_step(step,frequently_step)?)
    }
}
impl<U,P,I,PI,D> PersistProgress<TextFilePersistence,Specialized> for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             PersistProgress<TextFilePersistence,Specialized> +
             BackwardAll<U,LossInput=PI> + PreTrain<PreOutput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: UnitValue<U> + std::str::FromStr,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
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
impl<T,U,P,I,PI,D> PersistProgress<T,Linear> for LoggingLayer<U,P,I,PI,D>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> +
             PersistProgress<T,Linear> +
             BackwardAll<U,LossInput=PI> + PreTrain<PreOutput=PI> + Loss<U> + OutputTensorScalar<U>,
          U: UnitValue<U>,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          I: Debug + Send + Sync {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)
    }
}
