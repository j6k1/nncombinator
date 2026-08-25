//! Implementation of a layer for log collection

use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::device::Device;
use crate::error::{ModelLoadError, EvaluateError, PersistenceError, TrainingError};
use crate::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchPreTrain, BatchPreTrainBase, ContinueForward, ForwardAll, ForwardDiff, PartialForward, PreTrain, UpdateWeight, OnStep, PersistProgress, InputTensorScalar, OutputTensorScalar, InputScale, OutputScale, MaxInputValue, PreTrainBase};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextRecord};
use crate::Stack;

/// Logging layer Implementation
pub struct LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
impl<U,P,I,PI,D> InputTensorScalar for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          I: Debug + Send + Sync {
    type Scalar = U;
}
impl<U,P,I,PI,D> OutputTensorScalar for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + 'static + BatchDataType,
          I: Debug + Send + Sync {
    type Scalar = U;
}
impl<U,P,I,PI,D> LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
impl<U,P,I,PI,D> Persistence<TextFilePersistence,Specialized> for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> + Persistence<TextFilePersistence,Specialized> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
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
impl<T,U,P,I,PI,D> Persistence<T,Linear> for LoggingLayer<U,P,I,PI,D>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + Persistence<T,Linear> +
             BackwardAll<U,LossInput=PI> + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
impl<U,P,I,PI,D> PreTrainBase for LoggingLayer<U,P,I,PI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
    type PreOutput = PI;
    type OutStack = <P as PreTrainBase>::OutStack;
}
impl<U,P,I,PI,D> PreTrain for LoggingLayer<U,P,I,PI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
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
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
    type LossInputScalar = U;
    type LossInput = PI;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all(&mut self, input: Self::LossInput, stack:Self::OutStack)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        for logger in self.backward_loggers.iter() {
            logger(&input)?;
        }

        Ok(self.parent.backward_all(input, stack)?.into())
    }
}
impl<U,P,I,PI,D> UpdateWeight for LoggingLayer<U,P,I,PI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             PartialForward,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             PartialForward + ForwardDiff +
             BackwardAll<U,LossInput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
          PartialForward + ContinueForward +
          BackwardAll<U,LossInput=PI> +
          InputTensorScalar + OutputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
impl<U,P,I,PI,D> BatchForwardBase for LoggingLayer<U,P,I,PI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,P,I,PI,D> BatchForward for LoggingLayer<U,P,I,PI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = <P as BatchPreTrainBase>::BatchOutStack;
}
impl<U,P,I,PI,D> BatchPreTrain for LoggingLayer<U,P,I,PI,D>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             InputTensorScalar + OutputTensorScalar<Scalar=U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase + BatchPreTrain<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;
    fn batch_backward(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        for logger in self.batch_backward_loggers.iter() {
            logger(&input)?;
        }

        let r = self.parent.batch_backward(input, stack)?;

        Ok(r)
    }
}
impl<U,P,I,PI,D> OnStep for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U> + OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
             BackwardAll<U,LossInput=PI> + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
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
             BackwardAll<U,LossInput=PI> + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
impl<U,P,I,PI,D> InputScale for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain + InputScale +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + 'static + BatchDataType,
      I: Debug + Send + Sync {
    fn scale_mean(&self) -> f32 {
        self.parent.scale_mean()
    }
}
impl<U,P,I,PI,D> OutputScale for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain + OutputScale +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + 'static + BatchDataType,
      I: Debug + Send + Sync {
    type ScalingDevice = <P as OutputScale>::ScalingDevice;
    type Scale = <P as OutputScale>::Scale;
    type ScaledOutput = <P as OutputScale>::ScaledOutput;
    type Mapper<'a> = <P as OutputScale>::Mapper<'a> where Self: 'a;
    fn scaling_mapper<'a>(&self, input:&'a PI) -> Result<Self::Mapper<'a>,TrainingError> where Self: 'a {
        self.parent.scaling_mapper(input)
    }

}
impl<U,P,I,PI,D> MaxInputValue for LoggingLayer<U,P,I,PI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain + InputScale + MaxInputValue +
             InputTensorScalar + OutputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U>,
      PI: Debug + 'static + BatchDataType,
      I: Debug + Send + Sync {
    type Scalar = <P as MaxInputValue>::Scalar;
    fn max_input_value(&self) -> Self::Scalar {
        self.parent.max_input_value()
    }
}
