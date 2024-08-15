//! Batch normalization layer implementation
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::{Arr};
use crate::{Cons, Stack};
use crate::cuda::{CudaPtr, CudaTensor1dPtr, Memory};
use crate::device::{Device, DeviceCpu, DeviceGpu, DeviceMemoryPool};
use crate::device::batchnormalization::DeviceBatchNorm;
use crate::error::{ConfigReadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use crate::layer::{AskDiffInput, Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, Forward, ForwardAll, Loss, PreTrain, UpdateWeight};
use crate::lossfunction::LossFunction;
use crate::mem::AsRawSlice;
use crate::ope::{UnitValue};
use crate::optimizer::{Optimizer, OptimizerBuilder};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, UnitOrMarker};

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
pub trait BatchNormalizationLayerInstantiation<U,C,P,OP,D,I,PI,S,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
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
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>,LayerInstantiationError>;

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
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>,LayerInstantiationError>;

    /// Create and return an instance.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `b`- optimizer builder
    ///
    /// γ = 1, β = 0
    /// y = γx + β
    /// momentum = 0.9
    fn new<B: OptimizerBuilder<U,D,Output=OP>>(parent:P,device:&D,b:&B) -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>,LayerInstantiationError>;
}
///  BatchNormalization Layer Implementation
pub struct BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
          OP: Optimizer<U,D> {
    parent:P,
    device:D,
    scale: C,
    bias: C,
    momentum: U,
    running_mean: C,
    running_variance: C,
    pi:PhantomData<PI>,
    s:PhantomData<S>,
    scale_optimizer:OP,
    bias_optimizer:OP
}
impl<U,P,OP,I,PI,const N:usize> BatchNormalizationLayerInstantiation<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    for BatchNormalizationLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>> {
    fn with_params<B: OptimizerBuilder<U,DeviceCpu<U>,Output=OP>>(parent:P,device:&DeviceCpu<U>,scale:Arr<U,N>,bias:Arr<U,N>,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>,LayerInstantiationError> {

        let running_mean = Arr::new();
        let mut running_variance = Arr::new();

        for v in running_variance.iter_mut() {
            *v = U::one();
        }

        Ok(BatchNormalizationLayer {
            parent:parent,
            device:device.clone(),
            scale:scale,
            bias:bias,
            momentum:momentum,
            running_mean:running_mean,
            running_variance:running_variance,
            pi:PhantomData::<PI>,
            s:PhantomData::<Arr<U,N>>,
            scale_optimizer:b.build(N)?,
            bias_optimizer:b.build(N)?
        })
    }

    fn with_momentum<B: OptimizerBuilder<U,DeviceCpu<U>,Output=OP>>(parent:P,device:&DeviceCpu<U>,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>,LayerInstantiationError> {
        let mut scale = Arr::new();

        for i in scale.iter_mut() {
            *i = U::one();
        }

        Self::with_params(parent,device,scale,Arr::new(),momentum,b)
    }

    fn new<B: OptimizerBuilder<U,DeviceCpu<U>,Output=OP>>(parent:P,device:&DeviceCpu<U>,b:&B)
        -> Result<BatchNormalizationLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>,LayerInstantiationError> {
        Self::with_momentum(parent,device,U::from_f64(0.9).expect("An error occurred in floating point type conversion."),b)
    }
}
impl<U,P,OP,I,PI,const N:usize> BatchNormalizationLayerInstantiation<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>
    for BatchNormalizationLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> {
    fn with_params<B: OptimizerBuilder<U,DeviceGpu<U>,Output=OP>>(parent:P,device:&DeviceGpu<U>,scale:Arr<U,N>,bias:Arr<U,N>,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>,LayerInstantiationError> {
        let mut scale_ptr = CudaTensor1dPtr::new(device.get_memory_pool())?;

        scale_ptr.memcpy(scale.as_raw_slice().as_ptr(),N)?;

        let mut bias_ptr = CudaTensor1dPtr::new(device.get_memory_pool())?;

        bias_ptr.memcpy(bias.as_raw_slice().as_ptr(),N)?;

        Ok(BatchNormalizationLayer {
            parent:parent,
            device:device.clone(),
            scale:scale_ptr,
            bias:bias_ptr,
            momentum:momentum,
            running_mean:CudaTensor1dPtr::with_initializer(device.get_memory_pool(),Default::default)?,
            running_variance:CudaTensor1dPtr::with_initializer(device.get_memory_pool(),|| U::one())?,
            pi:PhantomData::<PI>,
            s:PhantomData::<CudaPtr<U>>,
            scale_optimizer:b.build(N)?,
            bias_optimizer:b.build(N)?
        })
    }

    fn with_momentum<B: OptimizerBuilder<U,DeviceGpu<U>,Output=OP>>(parent:P,device:&DeviceGpu<U>,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>,LayerInstantiationError> {
        let mut scale = Arr::new();

        for i in scale.iter_mut() {
            *i = U::one();
        }

        Self::with_params(parent,device,scale,Arr::new(),momentum,b)
    }

    fn new<B: OptimizerBuilder<U,DeviceGpu<U>,Output=OP>>(parent:P,device:&DeviceGpu<U>,b:&B)
        -> Result<BatchNormalizationLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>,LayerInstantiationError> {
        Self::with_momentum(parent,device,U::from_f64(0.9).expect("An error occurred in floating point type conversion."),b)
    }
}
impl<U,P,OP,I,PI,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized>
    for BatchNormalizationLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>>,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for i in self.scale.iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.bias.iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.running_mean.iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.running_variance.iter_mut() {
            *i = persistence.read()?;
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        persistence.write(UnitOrMarker::UnitsStart);

        for i in self.scale.iter() {
            persistence.write(UnitOrMarker::Unit(*i));
        }

        persistence.write(UnitOrMarker::UnitsStart);

        for i in self.bias.iter() {
            persistence.write(UnitOrMarker::Unit(*i));
        }

        persistence.write(UnitOrMarker::UnitsStart);

        for i in self.running_mean.iter() {
            persistence.write(UnitOrMarker::Unit(*i));
        }

        persistence.write(UnitOrMarker::UnitsStart);

        for i in self.running_variance.iter() {
            persistence.write(UnitOrMarker::Unit(*i));
        }

        Ok(())
    }
}
impl<T,U,P,OP,I,PI,const N:usize> Persistence<U,T,Linear>
    for BatchNormalizationLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>> {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for i in self.scale.iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.bias.iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.running_mean.iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.running_variance.iter_mut() {
            *i = persistence.read()?;
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for i in self.scale.iter() {
            persistence.write(*i)?;
        }

        for i in self.bias.iter() {
            persistence.write(*i)?;
        }

        for i in self.running_mean.iter() {
            persistence.write(*i)?;
        }

        for i in self.running_variance.iter() {
            persistence.write(*i)?;
        }

        Ok(())
    }
}
impl<U,P,OP,I,PI,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized>
    for BatchNormalizationLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
                 PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
              U: Default + Clone + Copy + UnitValue<U> + FromStr,
              I: Debug + Send + Sync,
              OP: Optimizer<U,DeviceGpu<U>>,
              ConfigReadError: From<<U as FromStr>::Err>,
              DeviceGpu<U>: Device<U> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
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

        self.scale.memcpy(scale.as_raw_slice().as_ptr(),N)?;
        self.bias.memcpy(bias.as_raw_slice().as_ptr(),N)?;
        self.running_mean.memcpy(running_mean.as_raw_slice().as_ptr(),N)?;
        self.running_variance.memcpy(running_variance.as_raw_slice().as_ptr(),N)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        persistence.write(UnitOrMarker::UnitsStart);

        let scale = self.scale.read_to_vec()?;

        for i in scale.iter() {
            persistence.write(UnitOrMarker::Unit(*i));
        }

        persistence.write(UnitOrMarker::UnitsStart);

        let bias = self.bias.read_to_vec()?;

        for i in bias.iter() {
            persistence.write(UnitOrMarker::Unit(*i));
        }

        persistence.write(UnitOrMarker::UnitsStart);

        let running_mean = self.running_mean.read_to_vec()?;

        for i in running_mean.iter() {
            persistence.write(UnitOrMarker::Unit(*i));
        }

        persistence.write(UnitOrMarker::UnitsStart);

        let running_variance = self.running_variance.read_to_vec()?;

        for i in running_variance.iter() {
            persistence.write(UnitOrMarker::Unit(*i));
        }

        Ok(())
    }
}
impl<T,U,P,OP,I,PI,const N:usize> Persistence<U,T,Linear>
    for BatchNormalizationLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>
        where T: LinearPersistence<U>,
              P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
                 PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
              U: Default + Clone + Copy + UnitValue<U>,
              I: Debug + Send + Sync,
              OP: Optimizer<U,DeviceGpu<U>>,
              DeviceGpu<U>: Device<U> {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
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

        self.scale.memcpy(scale.as_raw_slice().as_ptr(),N)?;
        self.bias.memcpy(bias.as_raw_slice().as_ptr(),N)?;
        self.running_mean.memcpy(running_mean.as_raw_slice().as_ptr(),N)?;
        self.running_variance.memcpy(running_variance.as_raw_slice().as_ptr(),N)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        let scale = Arr::<U,N>::try_from(self.scale.read_to_vec()?)?;

        for i in scale.iter() {
            persistence.write(*i)?;
        }

        let bias = Arr::<U,N>::try_from(self.bias.read_to_vec()?)?;

        for i in bias.iter() {
            persistence.write(*i)?;
        }

        let running_mean = Arr::<U,N>::try_from(self.running_mean.read_to_vec()?)?;

        for i in running_mean.iter() {
            persistence.write(*i)?;
        }

        let running_variance = Arr::<U,N>::try_from(self.running_variance.read_to_vec()?)?;

        for i in running_variance.iter() {
            persistence.write(*i)?;
        }

        Ok(())
    }
}
impl<U,C,P,OP,D,I,PI,S,const N:usize> Forward<PI,Result<PI,EvaluateError>> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          S: Debug + Sized + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static {
    fn forward(&self,input:&PI) -> Result<PI,EvaluateError> {
        self.device.forward_batch_norm(input,&self.scale,&self.bias,&self.running_mean,&self.running_variance)
    }
}
impl<U,C,P,OP,D,I,PI,S,const N:usize> ForwardAll for BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          S: Debug + Sized + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + 'static {
    type Input = I;
    type Output = PI;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(self.forward(&self.parent.forward_all(input)?)?)
    }
}
impl<U,C,P,OP,D,I,PI,S,const N:usize> PreTrain<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          S: Debug + Sized + 'static,
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
impl<U,C,P,OP,D,I,PI,S,const N:usize> Backward<U,(&PI,&PI,&C,&C),Result<(PI,C,C),TrainingError>>
    for BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>

    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          S: Debug + Sized + 'static,
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
impl<U,P,OP,I,PI,const N:usize> BackwardAll<U> for BatchNormalizationLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,DeviceCpu<U>>,
          DeviceCpu<U>: Device<U> + DeviceBatchNorm<U,Arr<U,N>,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,N>> {
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
impl<U,P,OP,I,PI,const N:usize> BackwardAll<U> for BatchNormalizationLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> + DeviceBatchNorm<U,CudaTensor1dPtr<U,N>,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,N>> {
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
impl<U,P,OP,I,PI,const N:usize> UpdateWeight<U> for BatchNormalizationLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,DeviceCpu<U>>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,N>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(Arr<U,N>,Arr<U,N>,Option<(Arr<U,N>,Arr<U,N>)>)>;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,(scale,bias,saved)) = stack.pop();

        self.bias_optimizer.update((&bias).into(),(&mut self.bias).into())?;
        self.scale_optimizer.update((&scale).into(),(&mut self.scale).into())?;

        if let Some((running_mean,running_variance)) = saved {
            for (it, &m) in self.running_mean.iter_mut().zip(running_mean.iter()) {
                *it = m;
            }

            for (it, &v) in self.running_variance.iter_mut().zip(running_variance.iter()) {
                *it = v;
            }
        }

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,P,OP,I,PI,const N:usize> UpdateWeight<U> for BatchNormalizationLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,N>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(CudaTensor1dPtr<U,N>,CudaTensor1dPtr<U,N>,Option<(CudaTensor1dPtr<U,N>,CudaTensor1dPtr<U,N>)>)>;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,(scale,bias,saved)) = stack.pop();

        self.bias_optimizer.update((&bias).into(),(&mut self.bias).into())?;
        self.scale_optimizer.update((&scale).into(),(&mut self.scale).into())?;

        if let Some((running_mean,running_variance)) = saved {
            self.running_mean = running_mean;
            self.running_variance = running_variance;
        }

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,P,OP,C,I,PI,S,const N:usize> AskDiffInput<U> for BatchNormalizationLayer<U,C,P,OP,DeviceCpu<U>,I,PI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,OutStack = <<<Self as PreTrain<U>>::OutStack as Stack>::Remaining as Stack>::Remaining> + Loss<U> + AskDiffInput<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          S: Debug + Sized + 'static,
          C: Debug,
          OP: Optimizer<U,DeviceCpu<U>>,
          <PI as BatchDataType>::Type: Debug + 'static,
          Self: PreTrain<U,PreOutput=PI> {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map_remaining(|s| s.map_remaining(|s| self.parent.ask_diff_input(s)))
    }
}
impl<U,P,OP,I,PI,const N:usize> Loss<U> for BatchNormalizationLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: BatchDataType + Debug + 'static,
          OP: Optimizer<U,DeviceCpu<U>>,
          DeviceCpu<U>: Device<U> + DeviceBatchNorm<U,Arr<U,N>,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,N>> {
}
impl<U,P,OP,I,PI,const N:usize> Loss<U> for BatchNormalizationLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          PI: Debug + BatchDataType + 'static,
          DeviceGpu<U>: Device<U> + DeviceBatchNorm<U,CudaTensor1dPtr<U,N>,PI,N>,
          <PI as BatchDataType>::Type: Debug + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,N>> {
}
impl<U,C,P,OP,D,I,PI,S,const N:usize> BatchForwardBase for BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: BatchDataType + Debug + 'static,
          S: Debug + Sized + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug + 'static,
          Self: ForwardAll {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,C,P,OP,D,I,PI,S,const N:usize> BatchForward for BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          PI: BatchDataType + Debug + 'static,
          S: Debug + Sized + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug + 'static {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_forward_batch_norm(&input,&self.scale,&self.bias,&self.running_mean,&self.running_variance)?)
    }
}
impl<U,C,P,OP,D,I,PI,S,const N:usize> BatchPreTrainBase<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          S: Debug + Sized + 'static,
          C: Debug,
          OP: Optimizer<U,D>,
          PI: BatchDataType + Debug + 'static,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          Self: PreTrain<U,PreOutput=PI> {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = Cons<Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,MeanAndVariance<C>>,Self::BatchPreOutput>;
}
impl<U,C,P,OP,D,I,PI,S,const N:usize> BatchPreTrain<U> for BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U>,
          U: Default + Clone + Copy + Debug + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,PI,N>,
          I: Debug + Send + Sync + BatchDataType,
          S: Debug + Sized + 'static,
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
impl<U,P,OP,I,PI,const N:usize> BatchBackward<U> for BatchNormalizationLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,DeviceCpu<U>>,
          PI: BatchDataType + Debug + 'static,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          DeviceCpu<U>: Device<U> + DeviceBatchNorm<U,Arr<U,N>,PI,N>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,N>> {
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
impl<U,P,OP,I,PI,const N:usize> BatchBackward<U> for BatchNormalizationLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,DeviceGpu<U>>,
          PI: BatchDataType + Debug + 'static,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          DeviceGpu<U>: Device<U> + DeviceBatchNorm<U,CudaTensor1dPtr<U,N>,PI,N>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,N>> {
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
impl<U,P,OP,I,PI,const N:usize> BatchLoss<U> for BatchNormalizationLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,DeviceCpu<U>>,
          PI: BatchDataType + Debug + 'static,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          DeviceCpu<U>: Device<U> + DeviceBatchNorm<U,Arr<U,N>,PI,N>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,N>> {
}
impl<U,P,OP,I,PI,const N:usize> BatchLoss<U> for BatchNormalizationLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,CudaPtr<U>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,DeviceGpu<U>>,
          PI: BatchDataType + Debug + 'static,
          <PI as BatchDataType>::Type: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          DeviceGpu<U>: Device<U> + DeviceBatchNorm<U,CudaTensor1dPtr<U,N>,PI,N>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,N>> {
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
    pub fn build_with_params<U,C,P,OP,D,I,PI,S,B>(&self,parent: P,device:&D,scale:Arr<U,N>,bias:Arr<U,N>,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + Send + UnitValue<U>,
              D: Device<U>,
              I: Debug + Send + Sync,
              S: Debug + Sized + 'static,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N> : BatchNormalizationLayerInstantiation<U,C,P,OP,D,I,PI,S,N> {
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
    pub fn build_with_momentum<U,C,P,OP,D,I,PI,S,B: OptimizerBuilder<U,D>>(&self,parent:P,device:&D,momentum:U,b:&B)
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + Send + UnitValue<U>,
              D: Device<U>,
              I: Debug + Send + Sync,
              S: Debug + Sized + 'static,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N> : BatchNormalizationLayerInstantiation<U,C,P,OP,D,I,PI,S,N>{
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
    pub fn build<U,C,P,OP,D,I,PI,S,B>(&self,parent: P,device:&D,b:&B)
        -> Result<BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + Send + UnitValue<U>,
              D: Device<U>,
              I: Debug + Send + Sync,
              S: Debug + Sized + 'static,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              BatchNormalizationLayer<U,C,P,OP,D,I,PI,S,N> : BatchNormalizationLayerInstantiation<U,C,P,OP,D,I,PI,S,N> {
        Ok(BatchNormalizationLayer::new(parent,device,b)?)
    }
}
