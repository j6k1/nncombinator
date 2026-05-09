//! Implementation of bias layer

use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::{Arr, IntoConverter};
use crate::{Cons, Stack};
use crate::device::{Device, DeviceBatchAveraging};
use crate::device::bias::DeviceBias;
use crate::error::{ModelLoadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use crate::layer::{Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, BatchSize, ContinueForward, Forward, ForwardAll, ForwardDiff, Loss, PartialForward, PreTrain, UpdateWeight, OnStep, PersistProgress};
use crate::lossfunction::LossFunction;
use crate::ope::{UnitValue};
use crate::optimizer::{Optimizer, OptimizerBuilder};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextPersistence, TextRecord};

/// Trait for BiasLayer instance creation
pub trait BiasLayerInstantiation<U,C,P,OP,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,D> {
    /// Create and return an instance with the specified scale, bias, and momentum.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `b`- optimizer builder
    ///
    fn instantiation<UI: FnMut() -> U,B: OptimizerBuilder<U,D,Output=OP>>(parent:P,device:&D,ui:UI,b:&B) -> Result<BiasLayer<U,C,P,OP,D,I,PI,N>,LayerInstantiationError>;
}
/// Bias Layer Implementation
pub struct BiasLayer<U,C,P,OP,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,D> {
    parent:P,
    device:D,
    bias:C,
    u:PhantomData<U>,
    optimizer:OP
}
impl<U,C,P,OP,D,I,PI,const N:usize> Persistence<U,TextFilePersistence,Specialized> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          OP: Optimizer<U,D>,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + BatchSize {
    fn load(&mut self, persistence: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)?;

        let mut bias = Arr::<U,N>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        self.bias = self.device.specialization_bias(bias)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write_layer_start();

        let bias = self.device.generalization_bias(&self.bias)?;

        persistence.write_units_start();

        for b in bias.iter() {
            persistence.write(*b);
        }

        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,C,P,OP,D,I,PI,const N:usize> Persistence<U,T,Linear> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + BatchSize {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)?;

        let mut bias = Arr::<U,N>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        self.bias = self.device.specialization_bias(bias)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        let bias = self.device.generalization_bias(&self.bias)?;

        for b in bias.iter() {
            persistence.write(*b)?;
        }

        Ok(())
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> Forward<PI,Result<PI,EvaluateError>> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + BatchSize {

    fn forward(&self,input:&PI) -> Result<PI,EvaluateError> {
        self.device.forward_bias(&self.bias,input)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> ForwardAll for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    type Input = I;
    type Output = PI;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> PreTrain<U> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    type PreOutput = PI;
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::PreOutput>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r))?;

        Ok(Cons(r,u))
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> Backward<U,PI,Result<PI,TrainingError>> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    fn backward(&mut self, input: PI) -> Result<PI,TrainingError> {
        self.device.backward_bias(input)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BackwardAll<U> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> + ForwardAll<Input=I,Output=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBias<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C> {
    type LossInput = PI;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let g = self.device.backward_bias_weight_gradient(&loss)?;

        let next_loss= self.backward(loss)?;

        let (s,next_loss) = self.parent.loss(next_loss.into(),lossf,s)?;

        let (l,s) = self.parent.backward_all(next_loss, s, lossf)?;

        Ok((l,Cons(s,g)))
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> UpdateWeight<U> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchAveraging<C,U>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          OP: Optimizer<U,D>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,C>;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        let (s,bias) = stack.pop();

        let bias = self.device.batch_averaging(bias,batch_size)?;

        self.optimizer.update((&bias).into(),(&mut self.bias).into())?;

        Ok(self.parent.update_weight(s,batch_size)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> PartialForward for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward<DiffOutput=PI> +
             BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          Self: ForwardAll<Input=I,Output=PI>,
          Self: PreTrain<U> {
    type PartialOutput = <P as PartialForward>::PartialOutput;
    type PartialOutputByDiff = <P as PartialForward>::PartialOutputByDiff;
    type DiffInput = <P as PartialForward>::DiffInput;
    type DiffOutput = PI;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward(input)?)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput) -> Result<Self::PartialOutputByDiff, EvaluateError> {
        Ok(self.parent.partial_forward_by_diff(input)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> ForwardDiff for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward<DiffOutput=PI> + ForwardDiff +
             BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
      D: Device<U> + DeviceBias<U,C,PI,N>,
      U: Default + Clone + Copy + Send + UnitValue<U>,
      I: Debug + Send + Sync,
      PI: Debug + BatchDataType + 'static,
      OP: Optimizer<U,D>,
      <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
      Self: ForwardAll<Input=I,Output=PI>,
      Self: PreTrain<U> {
    fn forward_diff(&self, input: Self::DiffInput) -> Result<Self::DiffOutput, EvaluateError> {
        let input = self.parent.forward_diff(input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> ContinueForward for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward<DiffOutput=PI> + ContinueForward<ConinueOutput=PI> +
          BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
      D: Device<U> + DeviceBias<U,C,PI,N>,
      U: Default + Clone + Copy + Send + UnitValue<U>,
      I: Debug + Send + Sync,
      PI: Debug + BatchDataType + 'static,
      OP: Optimizer<U,D>,
      <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
      Self: ForwardAll<Input=I,Output=PI>,
      Self: PreTrain<U> {
    type ConinueOutput = Self::Output;
    fn continue_forward(&self, input: &Self::PartialOutput) -> Result<Self::ConinueOutput, EvaluateError> {
        let input = self.parent.continue_forward(input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> Loss<U> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          D: Device<U> + DeviceBias<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C> {
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchForwardBase for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          OP: Optimizer<U,D>,
          Self: ForwardAll {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchForward for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          OP: Optimizer<U,D> {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_forward_bias(&self.bias,&input)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchPreTrainBase<U> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          C: Debug,
          OP: Optimizer<U,D>,
          Self: PreTrain<U,PreOutput=PI> {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,Self::BatchPreOutput>;
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchPreTrain<U> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <I as BatchDataType>::Type: Debug,
          C: Debug,
          OP: Optimizer<U,D> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| self.device.batch_forward_bias(&self.bias,input))?;

        Ok(Cons(r,u))
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchBackward<U> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + IntoConverter + 'static,
          <I as BatchDataType>::Type: Debug,
          C: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBias<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C> {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let g = self.device.batch_backward_bias_weight_gradient(&loss)?;

        let next_loss = self.device.batch_backward_bias(loss)?;

        let (
            s,next_loss
        ) = self.parent.batch_loss(next_loss,lossf,s)?;

        let (l,s) = self.parent.batch_backward(next_loss, s, lossf)?;

        Ok((l,Cons(s,g)))
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchLoss<U> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + IntoConverter + 'static,
          <I as BatchDataType>::Type: Debug,
          C: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBias<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C> {
}
// OnStep implementation
impl<U,C,P,OP,D,I,PI,const N:usize> OnStep for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> + OnStep,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,D> {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        self.optimizer.on_step(step)?;
        Ok(self.parent.on_step(step)?)
    }
    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        self.optimizer.on_frequently_step(step,frequently_step)?;
        Ok(self.parent.on_frequently_step(step,frequently_step)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> PersistProgress<TextFilePersistence,Specialized> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> +
             PersistProgress<TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          OP: Optimizer<U,D> + Persistence<U,TextFilePersistence,Specialized>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + BatchSize,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)?;

        self.optimizer.load(persistence)?;

        Ok(())
    }

    fn save_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;

        persistence.write_layer_start();

        persistence.write_units_start();

        self.optimizer.save(persistence)?;

        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,C,P,OP,D,I,PI,const N:usize> PersistProgress<T,Linear> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> +
             PersistProgress<T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          OP: Optimizer<U,D> + Persistence<U,T,Linear>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + BatchSize {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)?;

        self.optimizer.load(persistence)?;

        Ok(())
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;

        self.optimizer.save(persistence)?;

        Ok(())
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BiasLayerInstantiation<U,C,P,OP,D,I,PI,N> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBias<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static {
    fn instantiation<UI: FnMut() -> U,B: OptimizerBuilder<U,D,Output=OP>>(parent: P, device: &D, ui: UI, b: &B)
        -> Result<BiasLayer<U,C,P,OP,D,I,PI,N>, LayerInstantiationError> {
        let mut ui = ui;

        let mut bias = Arr::new();

        for it in bias.iter_mut() {
            *it = ui();
        }

        let bias = device.specialization_bias(bias)?;

        Ok(BiasLayer {
            parent: parent,
            device: device.clone(),
            bias: bias,
            u:PhantomData::<U>,
            optimizer:b.build(N)?
        })
    }
}
/// Trait for BiasLayer instance creation
pub struct BiasLayerBuilder<const N:usize> {

}
impl<const N:usize> BiasLayerBuilder<N> {
    pub fn new() -> BiasLayerBuilder<N> {
        BiasLayerBuilder {}
    }

    /// Create an instance of BiasLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `b` - optimizer builder
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build<U,C,P,D,I,PI,UI,OP,B>(&self,parent:P,device:&D,ui:UI,b:&B)
        -> Result<BiasLayer<U,C,P,OP,D,I,PI,N>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> +
                 BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + Send + UnitValue<U>,
              D: Device<U>,
              I: Debug + Send + Sync + BatchDataType,
              PI: Debug + BatchDataType,
              <I as BatchDataType>::Type: Debug + Send + Sync + 'static,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              UI: FnMut() -> U,
              BiasLayer<U,C,P,OP,D,I,PI,N>: BiasLayerInstantiation<U,C,P,OP,D,I,PI,N> {
        BiasLayer::instantiation(parent,device,ui,b)
    }
}
