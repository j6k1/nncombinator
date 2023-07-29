use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::{Arr, VecArr};
use crate::{Cons, Stack};
use crate::collection::Broadcast;
use crate::device::{Device, DeviceBatchNorm, DeviceCpu};
use crate::error::{ConfigReadError, EvaluateError, PersistenceError, SizeMismatchError, TrainingError};
use crate::layer::{Backward, BackwardAll, Forward, ForwardAll, Loss, PreTrain};
use crate::lossfunction::LossFunction;
use crate::ope::{Arithmetic, UnitValue};
use crate::optimizer::Optimizer;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, UnitOrMarker};

pub struct MeanAndVariance<U,T,const N:usize> where U: UnitValue<U> {
    pub running_mean:Arr<U,N>,
    pub running_variance:Arr<U,N>,
    pub saved_mean:T,
    pub saved_inv_variance:T
}
///  BatchNormalization Layer Implementation
pub struct BatchNormalizationLayer<U,C,P,D,I,PI,S,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          S: Debug + Sized + Send + Sync + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static {
    parent:P,
    device:D,
    scale: C,
    bias: C,
    momentum: U,
    running_mean: C,
    running_variance: C,
    pi:PhantomData<PI>,
    s:PhantomData<S>
}
impl<U,P,I,PI,const N:usize> BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static {
    /// Create and return an instance with the specified scale, bias, and momentum.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `scale` - γ
    /// * `bias` - β
    /// * `momentum`- Learning rate when updating running_mean and running_variance
    ///
    /// y = γx + β
    pub fn new(parent:P,device:&DeviceCpu<U>,scale:Arr<U,N>,bias:Arr<U,N>,momentum:U)
        -> BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,Arr<U,N>,N> {

        let running_mean = Arr::new();
        let mut running_variance = Arr::new();

        for v in running_variance.iter_mut() {
            *v = U::one();
        }

        BatchNormalizationLayer {
            parent:parent,
            device:device.clone(),
            scale:scale,
            bias:bias,
            momentum:momentum,
            running_mean:running_mean,
            running_variance:running_variance,
            pi:PhantomData::<PI>,
            s:PhantomData::<Arr<U,N>>
        }
    }
}
impl<U,P,I,PI,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized> for BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static,
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
impl<T,U,P,I,PI,const N:usize> Persistence<U,T,Linear> for BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
          PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static {
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
impl<U,C,P,D,I,PI,S,const N:usize> Forward<Arr<U,N>,Result<Arr<U,N>,EvaluateError>> for BatchNormalizationLayer<U,C,P,D,I,PI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,S,C,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + Send + Sync + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static {
    fn forward(&self,input:&Arr<U,N>) -> Result<Arr<U,N>,EvaluateError> {
        self.device.forward_batch_norm(input,&self.scale,&self.bias,&self.running_mean,&self.running_variance)
    }
}
impl<U,C,P,D,I,PI,S,const N:usize> ForwardAll for BatchNormalizationLayer<U,C,P,D,I,PI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,S,C,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + Send + Sync + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static {
    type Input = I;
    type Output = PI;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?.into()).map(|r| r.into())
    }
}
impl<U,C,P,D,I,PI,S,const N:usize> PreTrain<U> for BatchNormalizationLayer<U,C,P,D,I,PI,S,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,S,C,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + Send + Sync + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static {
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
impl<U,C,P,D,I,PI,S,const N:usize> Backward<U,(&Arr<U,N>,&Arr<U,N>,&S,&S),Result<(Arr<U,N>,Arr<U,N>,Arr<U,N>),TrainingError>>
    for BatchNormalizationLayer<U,C,P,D,I,PI,S,N>

    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBatchNorm<U,S,C,N>,
          I: Debug + Send + Sync,
          S: Debug + Sized + Send + Sync + 'static,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static {
    fn backward(&mut self, (loss,input,saved_mean,saved_inv_variance): (&Arr<U,N>,&Arr<U,N>,&S,&S)) -> Result<(Arr<U,N>,Arr<U,N>,Arr<U,N>),TrainingError> {
        self.device.backward_batch_norm(loss,
                                        input,
                                        &self.scale,
                                        saved_mean,
                                        saved_inv_variance)
    }
}
impl<U,P,I,PI,const N:usize> BackwardAll<U> for BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          for<'a> Arr<U,N>: Arithmetic<&'a Arr<U,N>,Arr<U,N>> + TryFrom<Vec<U>,Error = SizeMismatchError> +
                            Arithmetic<U,Arr<U,N>>,
          for<'a> &'a Arr<U,N>: Arithmetic<&'a Arr<U,N>,Arr<U,N>> + TryFrom<Vec<U>,Error = SizeMismatchError> + Arithmetic<U,Arr<U,N>>,
          for<'data> VecArr<U,Arr<U,N>>: Arithmetic<&'data VecArr<U,Arr<U,N>>, VecArr<U,Arr<U,N>>> +
                                         Arithmetic<U,VecArr<U,Arr<U,N>>> +
                                         Arithmetic<Broadcast<Arr<U,N>>,VecArr<U,Arr<U,N>>>,
          for<'data> &'data VecArr<U,Arr<U,N>>: Arithmetic<&'data VecArr<U,Arr<U,N>>,VecArr<U,Arr<U,N>>> +
                                                Arithmetic<U,VecArr<U,Arr<U,N>>> +
                                                Arithmetic<Broadcast<Arr<U,N>>,VecArr<U,Arr<U,N>>>,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static {
    type LossInput = PI;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L)
        -> Result<(), TrainingError> {

        let (s,x) = stack.pop();
        let (s,(m,iv)) = s.pop();

        let loss = input;
        let (loss,scale,bias) = self.backward((&loss.into(),&x.into(),&m,&iv))?;

        self.scale = scale;
        self.bias = bias;

        let (s,loss) = self.parent.loss(loss.into(),lossf,s)?;

        self.parent.backward_all(loss.into(), s, optimizer, lossf)
    }
}
impl<U,P,I,PI,const N:usize> Loss<U> for BatchNormalizationLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,Arr<U,N>,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          for<'a> Arr<U,N>: Arithmetic<&'a Arr<U,N>,Arr<U,N>> + TryFrom<Vec<U>,Error = SizeMismatchError> +
                            Arithmetic<U,Arr<U,N>>,
          for<'a> &'a Arr<U,N>: Arithmetic<&'a Arr<U,N>,Arr<U,N>> + TryFrom<Vec<U>,Error = SizeMismatchError> + Arithmetic<U,Arr<U,N>>,
          for<'data> VecArr<U,Arr<U,N>>: Arithmetic<&'data VecArr<U,Arr<U,N>>, VecArr<U,Arr<U,N>>> +
                                         Arithmetic<U,VecArr<U,Arr<U,N>>> +
                                         Arithmetic<Broadcast<Arr<U,N>>,VecArr<U,Arr<U,N>>>,
          for<'data> &'data VecArr<U,Arr<U,N>>: Arithmetic<&'data VecArr<U,Arr<U,N>>,VecArr<U,Arr<U,N>>> +
                                                Arithmetic<U,VecArr<U,Arr<U,N>>> +
                                                Arithmetic<Broadcast<Arr<U,N>>,VecArr<U,Arr<U,N>>>,
          Arr<U,N>: From<PI>,
          PI: From<Arr<U,N>> + Debug + Send + Sync + 'static {
}