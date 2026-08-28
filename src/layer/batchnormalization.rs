//! Batch normalization layer implementation
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use num_traits::FromPrimitive;
use crate::arr::{Arr};
use crate::{Cons, Stack};
use crate::device::{Device, DeviceBatchAveraging};
use crate::device::batchnormalization::DeviceBatchNorm;
use crate::error::{ModelLoadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use crate::layer::{Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchPreTrain, BatchPreTrainBase, ContinueForward, Forward, ForwardAll, ForwardDiff, PartialForward, PreTrain, UpdateWeight, OnStep, PersistProgress, InputTensorScalar, OutputTensorScalar, TensorSize, OutputTensorSize, InputTensorSize, InputScale, PreTrainBase, OutputScale, BatchSize, BatchOutputScale, BackwardBase, BatchBackwardBase};
use crate::mapper::{BatchDataMapper, BatchIdentityMapper, IdentityMapper};
use crate::ope::One;
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
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: InputTensorSize<N> + OutputTensorSize<N>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          [();N]: TensorSize {
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
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: InputTensorSize<N> + OutputTensorSize<N>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          [();N]: TensorSize {
    parent:P,
    device:D,
    scale: C,
    bias: C,
    momentum: U,
    running_mean: C,
    running_variance: C,
    pi:PhantomData<PI>,
    n:PhantomData<[();N]>,
    scale_optimizer:OP,
    bias_optimizer:OP
}
impl<U,C,P,OP,D,I,PI,const N:usize> InputTensorScalar for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: InputTensorSize<N> + OutputTensorSize<N>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          [();N]: TensorSize {
    type Scalar = U;
}
impl<U,C,P,OP,D,I,PI,const N:usize> OutputTensorScalar for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          PI: InputTensorSize<N> + OutputTensorSize<N>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          [();N]: TensorSize {
    type Scalar = U;
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchNormalizationLayerInstantiation<U,C,P,OP,D,I,PI,N>
    for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + One + FromPrimitive +
             Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize {
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
            n:PhantomData::<[();N]>,
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
impl<U,C,P,OP,D,I,PI,const N:usize> Persistence<TextFilePersistence,Specialized>
    for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain + Persistence<TextFilePersistence,Specialized> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> +
              Debug + InputTensorScalar + OutputTensorScalar + 'static,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err>,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize {
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
impl<T,U,C,P,OP,D,I,PI,const N:usize> Persistence<T,Linear> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain + Persistence<T,Linear> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize {
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
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize {
    fn forward(&self,input:&PI) -> Result<PI,EvaluateError> {
        self.device.forward_batch_norm(input,&self.scale,&self.bias,&self.running_mean,&self.running_variance)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> ForwardAll for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize {
    type Input = I;
    type Output = PI;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(self.forward(&self.parent.forward_all(input)?)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> PreTrainBase for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
          InputTensorScalar + OutputTensorScalar + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize
{
    type PreOutput = PI;
    type OutStack = Cons<Cons<<P as PreTrainBase>::OutStack, (C, C)>, Self::PreOutput>;
}
impl<U,C,P,OP,D,I,PI,const N:usize> PreTrain for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize {
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

    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> +
              Debug + InputTensorScalar + OutputTensorScalar + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize {
    fn backward(&mut self, (loss,input,saved_mean,saved_inv_variance): (&PI,&PI,&C,&C))
        -> Result<(PI,C,C),TrainingError> {
        self.device.backward_batch_norm(loss,
                                        input,
                                        &self.scale,
                                        saved_mean,
                                        saved_inv_variance)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BackwardBase for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + InputTensorSize<N> + OutputTensorSize<N> + Debug +
          InputTensorScalar + OutputTensorScalar + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C>,
          [();N]: TensorSize {
    type LossInputScalar = U;
    type LossInput = PI;
}
impl<U,C,P,OP,D,I,PI,const N:usize> BackwardAll<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C>,
          [();N]: TensorSize {
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all(&mut self, input: Self::LossInput, stack:Self::OutStack)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {

        let (s,_) = stack.pop();
        let (s,(m,iv)) = s.pop();

        let loss = input;

        let (loss,scale,bias) = s.map(|input| {
            self.backward((&loss,input,&m,&iv))
        })?;

        let (l,s) = self.parent.backward_all(loss, s)?;

        Ok((l,Cons(s,(scale,bias,None))))
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> UpdateWeight for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain + UpdateWeight +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchAveraging<C,U>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C>,
          [();N]: TensorSize {
    type GradientStack = Cons<<P as UpdateWeight>::GradientStack,(C,C,Option<(C,C)>)>;

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
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PartialForward + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize,
          Self: ForwardAll<Input=I,Output=PI> + PreTrain {
    type PartialInput = <P as PartialForward>::PartialInput;
    type PartialOutput = <P as PartialForward>::PartialOutput;
    type DiffInput = <P as PartialForward>::DiffInput;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward(input)?)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput, partial_input:&Self::PartialInput)
        -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward_by_diff(input, partial_input)?)
    }
}
impl<U,P,OP,D,C,I,PI,const N:usize> ForwardDiff for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
          PartialForward + ForwardDiff + PreTrainBase<PreOutput=PI> + PreTrain +
          InputTensorScalar + OutputTensorScalar,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
      I: Debug + Send + Sync,
      PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
          InputTensorScalar + OutputTensorScalar + 'static,
      OP: Optimizer<U,D>,
      <PI as BatchDataType>::Type: Debug + 'static,
      [();N]: TensorSize,
      Self: ForwardAll<Input=I,Output=PI> + PreTrain {
    fn forward_diff(&self, input: Self::DiffInput, partial_input:&Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_diff(input, partial_input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,P,OP,D,C,I,PI,const N:usize> ContinueForward for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
          PartialForward + ContinueForward + PreTrainBase<PreOutput=PI> + PreTrain +
          InputTensorScalar + OutputTensorScalar,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
      I: Debug + Send + Sync,
      PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
          InputTensorScalar + OutputTensorScalar + 'static,
      OP: Optimizer<U,D>,
      <PI as BatchDataType>::Type: Debug + 'static,
      [();N]: TensorSize,
      Self: ForwardAll<Input=I,Output=PI> + PreTrain {
    fn continue_forward(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.continue_forward(input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchForwardBase for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize,
          Self: ForwardAll {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchForward for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_forward_batch_norm(&input,&self.scale,&self.bias,&self.running_mean,&self.running_variance)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchPreTrainBase for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          C: Debug,
          OP: Optimizer<U,D>,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          [();N]: TensorSize,
          Self: PreTrain<PreOutput=PI> {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = Cons<Cons<<P as BatchPreTrainBase>::BatchOutStack,MeanAndVariance<C>>,Self::BatchPreOutput>;
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchPreTrain for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          C: Debug,
          OP: Optimizer<U,D>,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          [();N]: TensorSize,
          Self: PreTrain<PreOutput=PI> {
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
impl<U,C,P,OP,D,I,PI,const N:usize> BatchBackwardBase for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,D>,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
          InputTensorScalar + OutputTensorScalar + 'static,
          C: Debug,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C>,
          [();N]: TensorSize {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackwardBase>::BatchLossOutput;
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchBackward<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,D>,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          C: Debug,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N> + DeviceBatchAveraging<C,U>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C>,
          [();N]: TensorSize {
    fn batch_backward(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack)
        -> Result<(<Self as BatchBackwardBase>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
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

        let (l,s) = self.parent.batch_backward(loss, s)?;

        Ok((l,Cons(s,(scale,bias,Some((running_mean,running_variance))))))
    }
}
// OnStep implementation
impl<U,C,P,OP,D,I,PI,const N:usize> OnStep for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: InputTensorSize<N> + OutputTensorSize<N>,
          OP: Optimizer<U,D>,
          [();N]: TensorSize {
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
pub struct BatchNormalizationLayerBuilder<const N:usize>
    where [();N]: TensorSize {
    n:PhantomData<[();N]>
}
impl<const N:usize> BatchNormalizationLayerBuilder<N>
    where [();N]: TensorSize {
    /// Create an instance of BatchNormalizationLayerBuilder
    pub fn new() -> BatchNormalizationLayerBuilder<N> {
        BatchNormalizationLayerBuilder {
            n:PhantomData::<[();N]>
        }
    }
}
impl<const N:usize> BatchNormalizationLayerBuilder<N>
    where [();N]: TensorSize {
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
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
                 PreTrain +
                 InputTensorScalar + OutputTensorScalar,
              U: Default + Clone + Copy + Debug + Send + Sync + 'static,
              D: Device<U>,
              PI: InputTensorSize<N> + OutputTensorSize<N> + InputTensorScalar + OutputTensorScalar,
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
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
                 PreTrain +
                 InputTensorScalar + OutputTensorScalar,
              U: Default + Clone + Copy + Debug + Send + Sync + 'static,
              D: Device<U>,
              PI: InputTensorSize<N> + OutputTensorSize<N> + InputTensorScalar + OutputTensorScalar,
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
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
                 PreTrain +
                 InputTensorScalar + OutputTensorScalar,
              U: Default + Clone + Copy + Debug + Send + Sync + 'static,
              D: Device<U>,
              PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
                  InputTensorScalar + OutputTensorScalar,
              I: Debug + Send + Sync,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              BatchNormalizationLayer<U,C,P,OP,D,I,PI,N> : BatchNormalizationLayerInstantiation<U,C,P,OP,D,I,PI,N> {
        Ok(BatchNormalizationLayer::new(parent,device,b)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> PersistProgress<TextFilePersistence,Specialized>
    for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar +
             PersistProgress<TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
              InputTensorScalar + OutputTensorScalar + 'static,
          OP: Optimizer<U,D> + Persistence<TextFilePersistence,Specialized>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err>,
          [();N]: TensorSize {
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
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar +
             PersistProgress<T,Linear>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> +
              Debug + InputTensorScalar + OutputTensorScalar + 'static,
          OP: Optimizer<U,D> + Persistence<T,Linear>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static,
          [();N]: TensorSize {
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
impl<U,C,P,OP,D,I,PI,const N:usize> InputScale for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain + InputScale +
             InputTensorScalar + OutputTensorScalar,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
      I: Debug + Send + Sync,
      PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
      InputTensorScalar + OutputTensorScalar + 'static,
      OP: Optimizer<U,D>,
      <PI as BatchDataType>::Type: Debug + 'static,
      [();N]: TensorSize {
    fn scale_mean(&self) -> f32 {
        self.parent.scale_mean()
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> OutputScale for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
      I: Debug + Send + Sync,
      PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
          InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> + 'static,
      OP: Optimizer<U,D>,
      <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
      [();N]: TensorSize {
    type ScalingDevice = D;
    type Scale = PI;
    type ScaledOutput = PI;

    type Mapper<'a> = IdentityMapper<'a,PI,Self::ScalingDevice> where Self: 'a;

    fn scaling_mapper<'a>(&'a self, input: &'a <Self as ForwardAll>::Output) -> Result<Self::Mapper<'a>, EvaluateError> where Self: 'a {
        Ok(IdentityMapper::new(input))
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> BatchOutputScale for BatchNormalizationLayer<U,C,P,OP,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardBase<LossInput=PI,LossInputScalar=U> +
             BackwardAll<U> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
      I: Debug + BatchDataType + Send + Sync,
      PI: BatchDataType + InputTensorSize<N> + OutputTensorSize<N> + Debug +
          InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> + 'static,
      C: Debug,
      OP: Optimizer<U,D>,
      <I as BatchDataType>::Type: Debug + 'static,
      <PI as BatchDataType>::Type: Debug + BatchSize +
                                   InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> + 'static,
      [();N]: TensorSize {
    type BatchMapper<'a> = BatchIdentityMapper<'a,PI,Self::ScalingDevice> where Self: 'a;

    fn batch_scaling_mapper<'a>(&'a self, input: &'a <PI as BatchDataType>::Type) -> Result<Self::BatchMapper<'a>, EvaluateError> where Self: 'a {
        Ok(BatchIdentityMapper::new(input))
    }
}