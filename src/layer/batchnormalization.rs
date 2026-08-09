//! Batch normalization layer implementation
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::{Arr};
use crate::{Cons, Stack};
use crate::device::{Device, DeviceBatchAveraging};
use crate::device::batchnormalization::DeviceBatchNorm;
use crate::error::{ModelLoadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use crate::layer::{Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, ContinueForward, Forward, ForwardAll, ForwardDiff, Loss, PartialForward, PreTrain, UpdateWeight, OnStep, PersistProgress};
use crate::lossfunction::LossFunction;
use crate::ope::{UnitValue};
use crate::optimizer::{Optimizer, OptimizerBuilder};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextPersistence, TextRecord};

/// Structure that holds information related to mean and variance calculated during forward propagation during learning.
#[derive(Debug)]
pub struct MeanAndVariance<T> {
    /// Population mean per batch
    pub running_mean:T,
    /// Population variance per batch
    pub running_variance:T,
    /// Mean computed by the process of forward propagation during learning
    pub saved_mean:T,
    /// Variance computed by the process of forward propagation during learning
    pub saved_inv_variance:T
}
/// Trait for BatchNormalizationLayer instance creation
pub trait BatchNormalizationLayerInstantiation<U,C,P,OP,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D> {
    /// Create and return an instance with the specified scale, bias, and momentum.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `scale` - γ
    /// * `bias` - β
    /// * `momentum`- Learning rate when updating running_mean and running_variance
    /// * `b`- optimizer builder
    ///
    /// y = γx + β
    fn with_params<B: OptimizerBuilder<U,D,Output=OP>>(parent:P,device:&D,scale:Arr<U,N>,bias:Arr<U,N>,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>,LayerInstantiationError>;

    /// Create and return an instance with the momentum.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `momentum`- Learning rate when updating running_mean and running_variance
    /// * `b`- optimizer builder
    ///
    /// γ = 1, β = 0
    /// y = γx + β
    fn with_momentum<B: OptimizerBuilder<U,D,Output=OP>>(parent:P,device:&D,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>,LayerInstantiationError>;

    /// Create and return an instance.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `b`- optimizer builder
    ///
    /// γ = 1, β = 0
    /// y = γx + β
    /// momentum = 0.9
    fn new<B: OptimizerBuilder<U,D,Output=OP>>(parent:P,device:&D,b:&B) -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>,LayerInstantiationError>;
}
///  BatchNormalization Layer Implementation
pub struct BatchNormalizationLayer<U,C,P,OP,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D> {
    parent:P,
    device:D,
    scale: C,
    bias: C,
    momentum: U,
    running_mean: C,
    running_variance: C,
    pi:PhantomData<PI>,
    scale_optimizer:OP,
    bias_optimizer:OP
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchNormalizationLayerInstantiation<U,C,P,OP,D,I,PI,N>
    for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static {
    fn with_params<B: OptimizerBuilder<U,D,Output=OP>>(parent:P,device:&D,scale:Arr<U,N>,bias:Arr<U,N>,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>,LayerInstantiationError> {

        let running_mean = Arr::new();
        let mut running_variance = Arr::new();

        for v in running_variance.iter_mut() {
            *v = U::one();
        }

        let bias = device.specialization_vars(bias)?;
        let scale = device.specialization_vars(scale)?;
        let running_mean = device.specialization_vars(running_mean)?;
        let running_variance = device.specialization_vars(running_variance)?;

        Ok(BatchNormalizationLayer {
            parent:parent,
            device:device.clone(),
            scale:scale,
            bias:bias,
            momentum:momentum,
            running_mean:running_mean,
            running_variance:running_variance,
            pi:PhantomData::<PI>,
            scale_optimizer:b.build(N)?,
            bias_optimizer:b.build(N)?
        })
    }

    fn with_momentum<B: OptimizerBuilder<U,D,Output=OP>>(parent:P,device:&D,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>,LayerInstantiationError> {
        let mut scale = Arr::new();

        for i in scale.iter_mut() {
            *i = U::one();
        }

        Self::with_params(parent,device,scale,Arr::new(),momentum,b)
    }

    fn new<B: OptimizerBuilder<U,D,Output=OP>>(parent:P,device:&D,b:&B)
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>,LayerInstantiationError> {
        Self::with_momentum(parent,device,U::from_f64(0.9).expect("An error occurred in floating point type conversion."),b)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> Persistence<U,TextFilePersistence,Specialized>
    for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err>,
          <PI as BatchDataType>::Type: Debug + 'static {
    fn load(&mut self, persistence: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)?;

        let mut scale = Arr::<U,N>::new();

        for i in scale.iter_mut() {
            *i = persistence.read()?;
        }

        let mut bias = Arr::<U,N>::new();

        for i in bias.iter_mut() {
            *i = persistence.read()?;
        }

        let mut running_mean = Arr::<U,N>::new();

        for i in running_mean.iter_mut() {
            *i = persistence.read()?;
        }

        let mut running_variance = Arr::<U,N>::new();

        for i in running_variance.iter_mut() {
            *i = persistence.read()?;
        }

        self.scale = self.device.specialization_vars(scale)?;
        self.bias = self.device.specialization_vars(bias)?;
        self.running_mean = self.device.specialization_vars(running_mean)?;
        self.running_variance = self.device.specialization_vars(running_variance)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write_layer_start();

        persistence.write_units_start();

        let scale = self.device.generalization_vars(&self.scale)?;
        let bias = self.device.generalization_vars(&self.bias)?;
        let running_mean = self.device.generalization_vars(&self.running_mean)?;
        let running_variance = self.device.generalization_vars(&self.running_variance)?;

        for i in scale.iter() {
            persistence.write(*i);
        }

        persistence.write_units_start();

        for i in bias.iter() {
            persistence.write(*i);
        }

        persistence.write_units_start();

        for i in running_mean.iter() {
            persistence.write(*i);
        }

        persistence.write_units_start();

        for i in running_variance.iter() {
            persistence.write(*i);
        }

        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,C,P,OP,D,I,PI,const N:usize> Persistence<U,T,Linear> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)?;

        let mut scale = Arr::<U,N>::new();

        for i in scale.iter_mut() {
            *i = persistence.read()?;
        }

        let mut bias = Arr::<U,N>::new();

        for i in bias.iter_mut() {
            *i = persistence.read()?;
        }

        let mut running_mean = Arr::<U,N>::new();

        for i in running_mean.iter_mut() {
            *i = persistence.read()?;
        }

        let mut running_variance = Arr::<U,N>::new();

        for i in running_variance.iter_mut() {
            *i = persistence.read()?;
        }

        self.scale = self.device.specialization_vars(scale)?;
        self.bias = self.device.specialization_vars(bias)?;
        self.running_mean = self.device.specialization_vars(running_mean)?;
        self.running_variance = self.device.specialization_vars(running_variance)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        let scale = self.device.generalization_vars(&self.scale)?;
        let bias = self.device.generalization_vars(&self.bias)?;
        let running_mean = self.device.generalization_vars(&self.running_mean)?;
        let running_variance = self.device.generalization_vars(&self.running_variance)?;

        for i in scale.iter() {
            persistence.write(*i)?;
        }

        for i in bias.iter() {
            persistence.write(*i)?;
        }

        for i in running_mean.iter() {
            persistence.write(*i)?;
        }

        for i in running_variance.iter() {
            persistence.write(*i)?;
        }

        Ok(())
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> Forward<PI,Result<PI,EvaluateError>> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static {
    fn forward(&self,input:&PI) -> Result<PI,EvaluateError> {
        self.device.forward_batch_norm(input,&self.scale,&self.bias,&self.running_mean,&self.running_variance)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> ForwardAll for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static {
    type Input = I;
    type Output = PI;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(self.forward(&self.parent.forward_all(input)?)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> PreTrain<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static {
    type PreOutput = PI;
    type OutStack = Cons<Cons<<P as PreTrain<U>>::OutStack,(C,C)>,Self::PreOutput>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let s = self.parent.pre_train(input)?;

        let (u,m,iv) = s.map(|i| {
            self.device.forward_batch_norm_train(i,
                                                 &self.scale,
                                                 &self.bias,
                                                 &self.running_mean,
                                                 &self.running_variance)
        })?;

        Ok(s.push((m,iv)).push(u))
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> Backward<U,(&PI,&PI,&C,&C),Result<(PI,C,C),TrainingError>>
    for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>

    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static {
    fn backward(&mut self, (loss,input,saved_mean,saved_inv_variance): (&PI,&PI,&C,&C))
        -> Result<(PI,C,C),TrainingError> {
        self.device.backward_batch_norm(loss,
                                        input,
                                        &self.scale,
                                        saved_mean,
                                        saved_inv_variance)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BackwardAll<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C> {
    type LossInput = PI;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {

        let (s,_) = stack.pop();
        let (s,(m,iv)) = s.pop();

        let loss = input;

        let (loss,scale,bias) = s.map(|input| {
            self.backward((&loss,input,&m,&iv))
        })?;

        let (s,loss) = self.parent.loss(loss,lossf,s)?;

        let (l,s) = self.parent.backward_all(loss, s, lossf)?;

        Ok((l,Cons(s,(scale,bias,None))))
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> UpdateWeight<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchAveraging<C,U>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(C,C,Option<(C,C)>)>;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        let (s,(scale,bias,saved)) = stack.pop();

        let bias = self.device.batch_averaging(bias,batch_size)?;
        let scale = self.device.batch_averaging(scale,batch_size)?;

        self.bias_optimizer.update((&bias).into(),(&mut self.bias).into())?;
        self.scale_optimizer.update((&scale).into(),(&mut self.scale).into())?;

        if let Some((running_mean,running_variance)) = saved {
            self.running_mean = running_mean;
            self.running_variance = running_variance;
        }

        Ok(self.parent.update_weight(s,batch_size)?)
    }
}
impl<U,P,OP,D,C,I,PI,const N:usize> PartialForward for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PartialForward<DiffOutput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static,
          Self: ForwardAll<Input=I,Output=PI> + PreTrain<U> {
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
impl<U,P,OP,D,C,I,PI,const N:usize> ForwardDiff for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
          PartialForward<DiffOutput=PI> + ForwardDiff + PreTrain<U,PreOutput=PI> + Loss<U>,
      U: Default + Clone + Copy + Send + UnitValue<U>,
      D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
      I: Debug + Send + Sync,
      PI: BatchDataType + Debug + 'static,
      OP: Optimizer<U,D>,
      <PI as BatchDataType>::Type: Debug + 'static,
      Self: ForwardAll<Input=I,Output=PI> + PreTrain<U> {
    fn forward_diff(&self, input: Self::DiffInput) -> Result<Self::DiffOutput, EvaluateError> {
        let input = self.parent.forward_diff(input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,P,OP,D,C,I,PI,const N:usize> ContinueForward for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
          PartialForward<DiffOutput=PI> + ContinueForward<ConinueOutput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
      U: Default + Clone + Copy + Send + UnitValue<U>,
      D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
      I: Debug + Send + Sync,
      PI: BatchDataType + Debug + 'static,
      OP: Optimizer<U,D>,
      <PI as BatchDataType>::Type: Debug + 'static,
      Self: ForwardAll<Input=I,Output=PI> + PreTrain<U> {
    type ConinueOutput = Self::Output;
    fn continue_forward(&self, input: &Self::PartialOutput) -> Result<Self::ConinueOutput, EvaluateError> {
        let input = self.parent.continue_forward(input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> Loss<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C> {
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchForwardBase for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: BatchDataType + Debug + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug + 'static,
          Self: ForwardAll {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchForward for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: BatchDataType + Debug + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug + 'static {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_forward_batch_norm(&input,&self.scale,&self.bias,&self.running_mean,&self.running_variance)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchPreTrainBase<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          C: Debug,
          OP: Optimizer<U,D>,
          PI: BatchDataType + Debug + 'static,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          Self: PreTrain<U,PreOutput=PI> {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = Cons<Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,MeanAndVariance<C>>,Self::BatchPreOutput>;
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchPreTrain<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U>,
          U: Default + Clone + Copy + Debug + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          C: Debug,
          OP: Optimizer<U,D>,
          PI: BatchDataType + Debug + 'static,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          Self: PreTrain<U,PreOutput=PI> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let s = self.parent.batch_pre_train(input)?;

        let (u,mean,inv_variance,running_mean,running_variance) = s.map(|input| {
            self.device.batch_forward_batch_norm_train(input,&self.scale,&self.bias,
                                                       &self.running_mean,&self.running_variance,self.momentum)
        })?;

        Ok(s.push(MeanAndVariance {
            running_mean: running_mean,
            running_variance: running_variance,
            saved_mean: mean,
            saved_inv_variance: inv_variance
        }).push(u))
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchBackward<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,D>,
          PI: BatchDataType + Debug + 'static,
          C: Debug,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C> {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let loss = input;

        let (s, _) = stack.pop();

        let (s,MeanAndVariance {
            running_mean,
            running_variance,
            saved_mean,
            saved_inv_variance
        }) = s.pop();

        let (loss,scale,bias) = s.map(|input| {
            self.device.batch_backward_batch_norm(&loss,input,&self.scale,&saved_mean,&saved_inv_variance)
        })?;

        let (s,
             loss
        ) = self.parent.batch_loss(loss,lossf,s)?;

        let (l,s) = self.parent.batch_backward(loss, s, lossf)?;

        Ok((l,Cons(s,(scale,bias,Some((running_mean,running_variance))))))
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchLoss<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,D>,
          PI: BatchDataType + Debug + 'static,
          C: Debug,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C> {
}
// OnStep implementation
impl<U,C,P,OP,D,I,PI,const N:usize> OnStep for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> + OnStep,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,D> {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        self.scale_optimizer.on_step(step)?;
        self.bias_optimizer.on_step(step)?;
        Ok(self.parent.on_step(step)?)
    }

    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        self.scale_optimizer.on_frequently_step(step,frequently_step)?;
        self.bias_optimizer.on_frequently_step(step,frequently_step)?;
        Ok(self.parent.on_frequently_step(step,frequently_step)?)
    }
}
/// Builder for BatchNormalizationLayer instance creation
pub struct BatchNormalizationLayerBuilder<const N:usize> {
    n:PhantomData<[();N]>
}
impl<const N:usize> BatchNormalizationLayerBuilder<N> {
    /// Create an instance of BatchNormalizationLayerBuilder
    pub fn new() -> BatchNormalizationLayerBuilder<N> {
        BatchNormalizationLayerBuilder {
            n:PhantomData::<[();N]>
        }
    }
}
impl<const N:usize> BatchNormalizationLayerBuilder<N> {
    /// Create an instance of BatchNormalizationLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `scale` - γ
    /// * `bias` - β
    /// * `momentum`- Learning rate when updating running_mean and running_variance
    /// * `b`- optimizer builder
    ///
    /// y = γx + β
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build_with_params<U,C,P,OP,D,I,PI,B>(&self,parent: P,device:&D,scale:Arr<U,N>,bias:Arr<U,N>,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + Send + UnitValue<U>,
              D: Device<U>,
              I: Debug + Send + Sync,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              BatchNormalizationLayer<U,C,P,OP,D,I,PI,N> : BatchNormalizationLayerInstantiation<U,C,P,OP,D,I,PI,N> {
        Ok(BatchNormalizationLayer::with_params(parent,device,scale,bias,momentum,b)?)
    }

    /// Create an instance of BatchNormalizationLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `momentum`- Learning rate when updating running_mean and running_variance
    /// * `b`- optimizer builder
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build_with_momentum<U,C,P,OP,D,I,PI,B: OptimizerBuilder<U,D>>(&self,parent:P,device:&D,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + Send + UnitValue<U>,
              D: Device<U>,
              I: Debug + Send + Sync,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              BatchNormalizationLayer<U,C,P,OP,D,I,PI,N> : BatchNormalizationLayerInstantiation<U,C,P,OP,D,I,PI,N>{
        Ok(BatchNormalizationLayer::with_momentum(parent,device,momentum,b)?)
    }

    /// Create an instance of BatchNormalizationLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `b`- optimizer builder
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build<U,C,P,OP,D,I,PI,B>(&self,parent: P,device:&D,b:&B)
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + Send + UnitValue<U>,
              D: Device<U>,
              I: Debug + Send + Sync,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              BatchNormalizationLayer<U,C,P,OP,D,I,PI,N> : BatchNormalizationLayerInstantiation<U,C,P,OP,D,I,PI,N> {
        Ok(BatchNormalizationLayer::new(parent,device,b)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> PersistProgress<TextFilePersistence,Specialized>
    for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> +
             PersistProgress<TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,D> + Persistence<U,TextFilePersistence,Specialized>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)?;

        self.scale_optimizer.load(persistence)?;
        self.bias_optimizer.load(persistence)?;

        Ok(())
    }

    fn save_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;
        persistence.write_layer_start();

        persistence.write_units_start();
        self.scale_optimizer.save(persistence)?;
        persistence.write_units_start();
        self.bias_optimizer.save(persistence)?;

        persistence.write_layer_end();
        Ok(())
    }
}
impl<T,U,C,P,OP,D,I,PI,const N:usize> PersistProgress<T,Linear> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> +
             PersistProgress<T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,D> + Persistence<U,T,Linear>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)?;

        self.scale_optimizer.load(persistence)?;
        self.bias_optimizer.load(persistence)?;

        Ok(())
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;

        self.scale_optimizer.save(persistence)?;
        self.bias_optimizer.save(persistence)?;

        Ok(())
    }
}
