use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::{Arr, VecArr};
use crate::{Cons, Stack};
use crate::cuda::{CudaPtr};
use crate::cuda::mem::CachedTensor;
use crate::device::{Device, DeviceCpu, DeviceGpu, DeviceMemoryPool};
use crate::device::batchnormalization::DeviceBatchNorm;
use crate::error::{ConfigReadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use crate::layer::{AskDiffInput, Backward, BackwardAll, BatchBackward, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, Forward, ForwardAll, Loss, PreTrain};
use crate::lossfunction::LossFunction;
use crate::ope::{UnitValue};
use crate::optimizer::Optimizer;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, UnitOrMarker};

/// Structure that holds information related to mean and variance calculated during forward propagation during learning.
#[derive(Debug)]
pub struct MeanAndVariance<U,T,const N:usize> where U: UnitValue<U> {
    pub running_mean:Arr<U,N>,
    pub running_variance:Arr<U,N>,
    pub saved_mean:T,
    pub saved_inv_variance:T
}
/// Trait for BatchNormalizationLayer instance creation
pub trait BatchNormalizationLayerInstantiation<U,C,P,D,I,PI,BI,S,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static {
    /// Create and return an instance with the specified scale, bias, and momentum.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `scale` - γ
    /// * `bias` - β
    /// * `momentum`- Learning rate when updating running_mean and running_variance
    ///
    /// y = γx + β
    fn with_params(parent:P,device:&D,scale:Arr<U,N>,bias:Arr<U,N>,momentum:U)
        -> Result<BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>,LayerInstantiationError>;

    /// Create and return an instance with the momentum.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `momentum`- Learning rate when updating running_mean and running_variance
    ///
    /// γ = 1, β = 0
    /// y = γx + β
    fn with_momentum(parent:P,device:&D,momentum:U)
        -> Result<BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>,LayerInstantiationError>;

    /// Create and return an instance.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    /// γ = 1, β = 0
    /// y = γx + β
    /// momentum = 0.9
    fn new(parent:P,device:&D) -> Result<BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>,LayerInstantiationError>;
}
///  BatchNormalization Layer Implementation
pub struct BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static {
    parent:P,
    device:D,
    scale: C,
    bias: C,
    momentum: U,
    running_mean: C,
    running_variance: C,
    pi:PhantomData<PI>,
    bi:PhantomData<BI>,
    s:PhantomData<S>
}
impl<U,P,I,PI,BI,const N:usize> BatchNormalizationLayerInstantiation<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,BI,Arr<U,N>,N>
    for BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,BI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI> {
    fn with_params(parent:P,device:&DeviceCpu<U>,scale:Arr<U,N>,bias:Arr<U,N>,momentum:U)
                       -> Result<BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,BI,Arr<U,N>,N>,LayerInstantiationError> {

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
            bi:PhantomData::<BI>,
            s:PhantomData::<Arr<U,N>>
        })
    }

    fn with_momentum(parent:P,device:&DeviceCpu<U>,momentum:U)
                         -> Result<BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,BI,Arr<U,N>,N>,LayerInstantiationError> {
        let mut scale = Arr::new();

        for i in scale.iter_mut() {
            *i = U::one();
        }

        Self::with_params(parent,device,scale,Arr::new(),momentum)
    }

    fn new(parent:P,device:&DeviceCpu<U>) -> Result<BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,BI,Arr<U,N>,N>,LayerInstantiationError> {
        Self::with_momentum(parent,device,U::from_f64(0.9).expect("An error occurred in floating point type conversion."))
    }
}
impl<U,P,I,PI,BI,const N:usize> BatchNormalizationLayerInstantiation<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,BI,CudaPtr<U>,N>
    for BatchNormalizationLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,BI,CudaPtr<U>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> {
    fn with_params(parent:P,device:&DeviceGpu<U>,scale:Arr<U,N>,bias:Arr<U,N>,momentum:U)
        -> Result<BatchNormalizationLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,BI,CudaPtr<U>,N>,LayerInstantiationError> {
        let running_mean = Arr::new();
        let mut running_variance = Arr::new();

        for v in running_variance.iter_mut() {
            *v = U::one();
        }

        Ok(BatchNormalizationLayer {
            parent:parent,
            device:device.clone(),
            scale:CachedTensor::new(scale,device.get_memory_pool())?,
            bias:CachedTensor::new(bias,device.get_memory_pool())?,
            momentum:momentum,
            running_mean:CachedTensor::new(running_mean,device.get_memory_pool())?,
            running_variance:CachedTensor::new(running_variance,device.get_memory_pool())?,
            pi:PhantomData::<PI>,
            bi:PhantomData::<BI>,
            s:PhantomData::<CudaPtr<U>>
        })
    }

    fn with_momentum(parent:P,device:&DeviceGpu<U>,momentum:U)
                         -> Result<BatchNormalizationLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,BI,CudaPtr<U>,N>,LayerInstantiationError> {
        let mut scale = Arr::new();

        for i in scale.iter_mut() {
            *i = U::one();
        }

        Self::with_params(parent,device,scale,Arr::new(),momentum)
    }

    fn new(parent:P,device:&DeviceGpu<U>)
        -> Result<BatchNormalizationLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,BI,CudaPtr<U>,N>,LayerInstantiationError> {
        Self::with_momentum(parent,device,U::from_f64(0.9).expect("An error occurred in floating point type conversion."))
    }
}
impl<U,P,I,PI,BI,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized>
    for BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,BI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static,
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
impl<T,U,P,I,PI,BI,const N:usize> Persistence<U,T,Linear>
    for BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,BI,Arr<U,N>,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static {
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

        for i in self.bias.iter() {
            persistence.write(*i)?;
        }

        for i in self.scale.iter() {
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
impl<U,P,I,PI,BI,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized>
    for BatchNormalizationLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,BI,CudaPtr<U>,N>
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
                 PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
              U: Default + Clone + Copy + UnitValue<U> + FromStr,
              I: Debug + Send + Sync,
              Arr<U,N>: From<PI>,
              PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
              VecArr<U,Arr<U,N>>: From<BI>,
              BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static,
              ConfigReadError: From<<U as FromStr>::Err>,
              DeviceGpu<U>: Device<U> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for i in self.scale.scoped_mut().iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.bias.scoped_mut().iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.running_mean.scoped_mut().iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.running_variance.scoped_mut().iter_mut() {
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
impl<T,U,P,I,PI,BI,const N:usize> Persistence<U,T,Linear>
    for BatchNormalizationLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,BI,CudaPtr<U>,N>
        where T: LinearPersistence<U>,
              P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
                 PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
              U: Default + Clone + Copy + UnitValue<U>,
              I: Debug + Send + Sync,
              Arr<U,N>: From<PI>,
              PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
              VecArr<U,Arr<U,N>>: From<BI>,
              BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static,
              DeviceGpu<U>: Device<U> {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for i in self.scale.scoped_mut().iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.bias.scoped_mut().iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.running_mean.scoped_mut().iter_mut() {
            *i = persistence.read()?;
        }

        for i in self.running_variance.scoped_mut().iter_mut() {
            *i = persistence.read()?;
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for i in self.bias.iter() {
            persistence.write(*i)?;
        }

        for i in self.scale.iter() {
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
impl<U,C,P,D,I,PI,BI,S,const N:usize> Forward<Arr<U,N>,Result<Arr<U,N>,EvaluateError>> for BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,S,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static {
    fn forward(&self,input:&Arr<U,N>) -> Result<Arr<U,N>,EvaluateError> {
        self.device.forward_batch_norm(input,&self.scale,&self.bias,&self.running_mean,&self.running_variance)
    }
}
impl<U,C,P,D,I,PI,BI,S,const N:usize> ForwardAll for BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,S,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static {
    type Input = I;
    type Output = PI;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(self.forward(&self.parent.forward_all(input)?.into()).map(|r| r.into())?)
    }
}
impl<U,C,P,D,I,PI,BI,S,const N:usize> PreTrain<U> for BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,S,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static {
    type OutStack = Cons<Cons<<P as PreTrain<U>>::OutStack,(S,S)>,Self::Output>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let s = self.parent.pre_train(input)?;

        let (s,r) = s.take_map(|i| {
            let i = i.into();
            let r = self.device.forward_batch_norm_train(&i,
                                                                             &self.scale,
                                                                             &self.bias,
                                                                             &self.running_mean,
                                                                             &self.running_variance);
            (i.into(),r)
        });

        let (u,m,iv) = r?;

        Ok(s.push((m,iv)).push(u.into()))
    }
}
impl<U,C,P,D,I,PI,BI,S,const N:usize> Backward<U,(&Arr<U,N>,&Arr<U,N>,&S,&S),Result<(Arr<U,N>,Arr<U,N>,Arr<U,N>),TrainingError>>
    for BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>

    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,S,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static {
    fn backward(&mut self, (loss,input,saved_mean,saved_inv_variance): (&Arr<U,N>,&Arr<U,N>,&S,&S)) -> Result<(Arr<U,N>,Arr<U,N>,Arr<U,N>),TrainingError> {
        self.device.backward_batch_norm(loss,
                                        input,
                                        &self.scale,
                                        saved_mean,
                                        saved_inv_variance)
    }
}
impl<U,P,I,PI,BI,const N:usize> BackwardAll<U> for BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,BI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static {
    type LossInput = PI;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L)
        -> Result<(), TrainingError> {

        let (s,_) = stack.pop();
        let (s,(m,iv)) = s.pop();

        let loss = input;

        let (s,r) = s.take_map(|input| {
            let x = input.into();
            let r = self.backward((&loss.into(),&x,&m,&iv));
            (x.into(),r)
        });

        let (loss,scale,bias) = r?;

        for (w,&g) in self.scale.iter_mut().zip(scale.iter()) {
            optimizer.update(g,w)
        }

        for (w,&g) in self.bias.iter_mut().zip(bias.iter()) {
            optimizer.update(g,w)
        }

        let (s,loss) = self.parent.loss(loss.into(),lossf,s)?;

        self.parent.backward_all(loss.into(), s, optimizer, lossf)
    }
}
impl<U,P,I,PI,BI,const N:usize> BackwardAll<U> for BatchNormalizationLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,BI,CudaPtr<U>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static,
          DeviceGpu<U>: Device<U> + DeviceBatchNorm<U,CachedTensor<U,Arr<U,N>>,CudaPtr<U>,N> {
    type LossInput = PI;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L)
                                                         -> Result<(), TrainingError> {

        let (s,_) = stack.pop();
        let (s,(m,iv)) = s.pop();

        let loss = input;

        let (s,r) = s.take_map(|input| {
            let x = input.into();
            let r = self.backward((&loss.into(),&x,&m,&iv));
            (x.into(),r)
        });

        let (loss,scale,bias) = r?;

        for (w,&g) in self.scale.scoped_mut().iter_mut().zip(scale.iter()) {
            optimizer.update(g,w)
        }

        for (w,&g) in self.bias.scoped_mut().iter_mut().zip(bias.iter()) {
            optimizer.update(g,w)
        }

        let (s,loss) = self.parent.loss(loss.into(),lossf,s)?;

        self.parent.backward_all(loss.into(), s, optimizer, lossf)
    }
}
impl<U,P,C,I,PI,BI,S,const N:usize> AskDiffInput<U> for BatchNormalizationLayer<U,C,P,DeviceCpu<U>,I,PI,BI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,OutStack = <<<Self as PreTrain<U>>::OutStack as Stack>::Remaining as Stack>::Remaining> + Loss<U> + AskDiffInput<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static,
          Self: PreTrain<U> {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map_remaining(|s| s.map_remaining(|s| self.parent.ask_diff_input(s)))
    }
}
impl<U,P,I,PI,BI,const N:usize> Loss<U> for BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,BI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static {
}
impl<U,P,I,PI,BI,const N:usize> Loss<U> for BatchNormalizationLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,BI,CudaPtr<U>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static,
          DeviceGpu<U>: Device<U> + DeviceBatchNorm<U,CachedTensor<U,Arr<U,N>>,CudaPtr<U>,N> {
}
impl<U,C,P,D,I,PI,BI,S,const N:usize> BatchForwardBase for BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=BI> + BatchForward,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,S,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static,
          Self: ForwardAll {
    type BatchInput = VecArr<U,I>;
    type BatchOutput = BI;
}
impl<U,C,P,D,I,PI,BI,S,const N:usize> BatchForward for BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=BI> + BatchForward,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,S,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_forward_batch_norm(&input.into(),&self.scale,&self.bias,
                                                &self.running_mean,&self.running_variance)?.into())
    }
}
impl<U,C,P,D,I,PI,BI,S,const N:usize> BatchPreTrainBase<U> for BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=BI> + BatchForward +
             BatchPreTrainBase<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,S,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static,
          Self: PreTrain<U> {
    type BatchOutStack = Cons<Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,MeanAndVariance<U,S,N>>,Self::BatchOutput>;
}
impl<U,C,P,D,I,PI,BI,S,const N:usize> BatchPreTrain<U> for BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=BI> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,C,S,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static,
          Self: PreTrain<U> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let s = self.parent.batch_pre_train(input)?;

        let (s,r) = s.take_map(|input| {
            let input = input.into();
            let r = self.device.batch_forward_batch_norm_train(&input,&self.scale,&self.bias,
                                                                    &self.running_mean,&self.running_variance,self.momentum);
            (input.into(),r)
        });

        let (u,mean,inv_variance,running_mean,running_variance) = r?;

        Ok(s.push(MeanAndVariance {
            running_mean: running_mean,
            running_variance: running_variance,
            saved_mean: mean,
            saved_inv_variance: inv_variance
        }).push(u.into()))
    }
}
impl<U,P,I,PI,BI,const N:usize> BatchBackward<U> for BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,BI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=BI> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=BI>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static {
    type BatchLossInput = BI;

    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, optimizer: &mut OP, lossf: &L) -> Result<(), TrainingError> {
        let loss = input.into();

        let (s, _) = stack.pop();

        let (s,MeanAndVariance {
            running_mean,
            running_variance,
            saved_mean,
            saved_inv_variance
        }) = s.pop();

        let (s,r) = s.take_map(|input| {
            let x = input.into();
            let r = self.device.batch_backward_batch_norm(&loss,&x,&self.scale,&saved_mean,&saved_inv_variance);
            (x.into(),r)
        });

        let (loss,scale,bias) = r?;

        let (s,loss) = self.parent.batch_loss(loss.into(),lossf,s)?;

        for (w,&g) in self.scale.iter_mut().zip(scale.iter()) {
            optimizer.update(g,w)
        }

        for (w,&g) in self.bias.iter_mut().zip(bias.iter()) {
            optimizer.update(g,w)
        }

        self.running_mean = running_mean;
        self.running_variance = running_variance;

        self.parent.batch_backward(loss, s, optimizer, lossf)
    }
}
impl<U,P,I,PI,BI,const N:usize> BatchBackward<U> for BatchNormalizationLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,BI,CudaPtr<U>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=BI> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=BI>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static,
          DeviceGpu<U>: Device<U> + DeviceBatchNorm<U,CachedTensor<U,Arr<U,N>>,CudaPtr<U>,N> {
    type BatchLossInput = BI;

    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, optimizer: &mut OP, lossf: &L) -> Result<(), TrainingError> {
        let loss = input.into();

        let (s, _) = stack.pop();

        let (s,MeanAndVariance {
            running_mean,
            running_variance,
            saved_mean,
            saved_inv_variance
        }) = s.pop();

        let (s,r) = s.take_map(|input| {
            let x = input.into();
            let r = self.device.batch_backward_batch_norm(&loss,&x,&self.scale,&saved_mean,&saved_inv_variance);
            (x.into(),r)
        });

        let (loss,scale,bias) = r?;

        let (s,loss) = self.parent.batch_loss(loss.into(),lossf,s)?;

        for (w,&g) in self.scale.scoped_mut().iter_mut().zip(scale.iter()) {
            optimizer.update(g,w)
        }

        for (w,&g) in self.bias.scoped_mut().iter_mut().zip(bias.iter()) {
            optimizer.update(g,w)
        }

        for (it,&m) in self.running_mean.scoped_mut().iter_mut().zip(running_mean.iter()) {
            *it = m;
        }

        for (it,&v) in self.running_variance.scoped_mut().iter_mut().zip(running_variance.iter()) {
            *it = v;
        }

        self.parent.batch_backward(loss, s, optimizer, lossf)
    }
}
impl<U,P,I,PI,BI,const N:usize> BatchLoss<U> for BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,BI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=BI> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=BI>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static {
}
impl<U,P,I,PI,BI,const N:usize> BatchLoss<U> for BatchNormalizationLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,BI,CudaPtr<U>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=BI> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=BI>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
          VecArr<U,Arr<U,N>>: From<BI>,
          BI: From<VecArr<U,Arr<U,N>>> + Debug + Send + Sync + 'static,
          DeviceGpu<U>: Device<U> + DeviceBatchNorm<U,CachedTensor<U,Arr<U,N>>,CudaPtr<U>,N> {
}
/// Builder for BatchNormalizationLayer instance creation
pub struct BatchNormalizationLayerBuilder<U,C,P,D,I,PI,BI,S,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static {
    u:PhantomData<U>,
    c:PhantomData<C>,
    p:PhantomData<P>,
    d:PhantomData<D>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
    bi:PhantomData<BI>,
    s:PhantomData<S>,
    n:PhantomData<[();N]>
}
impl<U,C,P,D,I,PI,BI,S,const N:usize> BatchNormalizationLayerBuilder<U,C,P,D,I,PI,BI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static {
    /// Create an instance of BatchNormalizationLayerBuilder
    pub fn new() -> BatchNormalizationLayerBuilder<U,C,P,D,I,PI,BI,S,N> {
        BatchNormalizationLayerBuilder {
            u:PhantomData::<U>,
            c:PhantomData::<C>,
            p:PhantomData::<P>,
            d:PhantomData::<D>,
            i:PhantomData::<I>,
            pi:PhantomData::<PI>,
            bi:PhantomData::<BI>,
            s:PhantomData::<S>,
            n:PhantomData::<[();N]>
        }
    }
}
impl<U,C,P,D,I,PI,BI,S> BatchNormalizationLayerBuilder<U,C,P,D,I,PI,BI,S,0>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static {
    /// Create an instance of BatchNormalizationLayerBuilder with size
    ///
    /// # Types
    /// * `N` - input size
    pub fn with_size<const N:usize>() -> BatchNormalizationLayerBuilder<U,C,P,D,I,PI,BI,S,N> {
        BatchNormalizationLayerBuilder {
            u:PhantomData::<U>,
            c:PhantomData::<C>,
            p:PhantomData::<P>,
            d:PhantomData::<D>,
            i:PhantomData::<I>,
            pi:PhantomData::<PI>,
            bi:PhantomData::<BI>,
            s:PhantomData::<S>,
            n:PhantomData::<[();N]>
        }
    }
}
impl<U,C,P,D,I,PI,BI,S,const N:usize> BatchNormalizationLayerBuilder<U,C,P,D,I,PI,BI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          S: Debug + Sized + 'static,
          BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N> : BatchNormalizationLayerInstantiation<U,C,P,D,I,PI,BI,S,N> {
    /// Create an instance of BatchNormalizationLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `scale` - γ
    /// * `bias` - β
    /// * `momentum`- Learning rate when updating running_mean and running_variance
    ///
    /// y = γx + β
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build_with_params(&self,parent: P,device:&D,scale:Arr<U,N>,bias:Arr<U,N>,momentum:U)
        -> Result<BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>,LayerInstantiationError> {
        Ok(BatchNormalizationLayer::with_params(parent,device,scale,bias,momentum)?)
    }

    /// Create an instance of BatchNormalizationLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `momentum`- Learning rate when updating running_mean and running_variance
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build_with_momentum(&self,parent:P,device:&D,momentum:U)
                             -> Result<BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>,LayerInstantiationError> {
        Ok(BatchNormalizationLayer::with_momentum(parent,device,momentum)?)
    }

    /// Create an instance of BatchNormalizationLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build(&self,parent: P,device:&D)
        -> Result<BatchNormalizationLayer<U,C,P,D,I,PI,BI,S,N>,LayerInstantiationError> {
        Ok(BatchNormalizationLayer::new(parent,device)?)
    }
}
