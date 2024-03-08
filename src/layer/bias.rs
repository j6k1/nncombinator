//! Implementation of bias layer

use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::{Arr, ArrView, MakeView, MakeViewMut, SerializedVec, SerializedVecConverter, SerializedVecView, SliceSize};
use crate::{Cons, Stack};
use crate::cuda::mem::CachedTensor;
use crate::device::{Device, DeviceCpu, DeviceGpu, DeviceMemoryPool};
use crate::device::bias::DeviceBias;
use crate::error::{ConfigReadError, EvaluateError, LayerInstantiationError, PersistenceError, SizeMismatchError, TrainingError};
use crate::layer::{AskDiffInput, Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, Forward, ForwardAll, Loss, PreTrain, UpdateWeight};
use crate::lossfunction::LossFunction;
use crate::ope::{UnitValue};
use crate::optimizer::Optimizer;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, UnitOrMarker};

/// Trait for BiasLayer instance creation
pub trait BiasLayerInstantiation<U,C,P,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync {
    /// Create and return an instance with the specified scale, bias, and momentum.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    ///
    fn instantiation<UI: FnMut() -> U>(parent:P,device:&D,ui:UI) -> Result<BiasLayer<U,C,P,D,I,PI,N>,LayerInstantiationError>;
}
/// Bias Layer Implementation
pub struct BiasLayer<U,C,P,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync {
    parent:P,
    device:D,
    bias:C,
    u:PhantomData<U>
}
impl<U,P,I,PI,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized> for BiasLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        for b in self.bias.iter() {
            persistence.write(UnitOrMarker::Unit(*b));
        }

        Ok(())
    }
}
impl<U,P,I,PI,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized> for BiasLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          DeviceGpu<U>: Device<U>,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.scoped_mut().iter_mut() {
            *b = persistence.read()?;
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        for b in self.bias.iter() {
            persistence.write(UnitOrMarker::Unit(*b));
        }

        Ok(())
    }
}
impl<T,U,P,I,PI,const N:usize> Persistence<U,T,Linear> for BiasLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for b in self.bias.iter() {
            persistence.write(*b)?;
        }

        Ok(())
    }
}
impl<T,U,P,I,PI,const N:usize> Persistence<U,T,Linear> for BiasLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.scoped_mut().iter_mut() {
            *b = persistence.read()?;
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for b in self.bias.scoped_mut().iter() {
            persistence.write(*b)?;
        }

        Ok(())
    }
}
impl<U,C,P,D,I,PI,const N:usize> Forward<PI,Result<PI,EvaluateError>> for BiasLayer<U,C,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBias<U,C,N>,
          I: Debug + Send + Sync,
          for<'a> PI: Debug + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>>,
          for<'a> ArrView<'a,U,N>: From<&'a PI> {

    fn forward(&self,input:&PI) -> Result<PI,EvaluateError> {
        self.device.forward_bias(&self.bias,input.into()).map(|o| o.into())
    }
}
impl<U,C,P,D,I,PI,const N:usize> ForwardAll for BiasLayer<U,C,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          D: Device<U> + DeviceBias<U,C,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          for<'a> PI: Debug + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>> + 'static,
          for<'a> ArrView<'a,U,N>: From<&'a PI> {
    type Input = I;
    type Output = PI;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,C,P,D,I,PI,const N:usize> PreTrain<U> for BiasLayer<U,C,P,D,I,PI,N>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          D: Device<U> + DeviceBias<U,C,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          for<'a> PI: Debug + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>> + 'static,
          for<'a> ArrView<'a,U,N>: From<&'a PI> {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::Output>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r.into()))?;

        Ok(Cons(r,u))
    }
}
impl<U,C,P,D,I,PI,const N:usize> Backward<U,PI,Result<PI,TrainingError>> for BiasLayer<U,C,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceBias<U,C,N>,
          I: Debug + Send + Sync,
          for<'a> PI: Debug + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>>,
          Arr<U,N>: From<PI> {
    fn backward(&mut self, input: PI) -> Result<PI,TrainingError> {
        self.device.backward_bias(input.into()).map(|l| l.into())
    }
}
impl<U,P,I,PI,const N:usize> BackwardAll<U> for BiasLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> + ForwardAll<Input=I,Output=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          for<'a> PI: Debug + Clone + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>> + 'static,
          for<'a> ArrView<'a,U,N>: From<&'a PI> {
    type LossInput = PI;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let g = loss.clone().into();

        let next_loss= self.backward(loss)?;

        let (s,next_loss) = self.parent.loss(next_loss.into(),lossf,s)?;

        let (l,s) = self.parent.backward_all(next_loss, s, lossf)?;

        Ok((l,Cons(s,g)))
    }
}
impl<U,P,I,PI,const N:usize> BackwardAll<U> for BiasLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> + ForwardAll<Input=I,Output=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          for<'a> PI: Debug + Clone + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>> + 'static,
          DeviceGpu<U>: Device<U> + DeviceBias<U,CachedTensor<U,Arr<U,N>>,N>,
          for<'a> ArrView<'a,U,N>: From<&'a PI> {
    type LossInput = PI;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let g = loss.clone().into();

        let next_loss= self.backward(loss)?;

        let (s,next_loss) = self.parent.loss(next_loss.into(),lossf,s)?;

        let (l,s) = self.parent.backward_all(next_loss, s, lossf)?;

        Ok((l,Cons(s,g)))
    }
}
impl<U,P,I,PI,const N:usize> UpdateWeight<U> for BiasLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,Arr<U,N>>;

    fn update_weight<OP: Optimizer<U>>(&mut self, stack: Self::GradientStack, optimizer: &mut OP) -> Result<(), TrainingError> {
        let (s,bias) = stack.pop();

        for (w,&g) in self.bias.iter_mut().zip(bias.iter()) {
            optimizer.update(g, w);
        }

        Ok(self.parent.update_weight(s,optimizer)?)
    }
}
impl<U,P,I,PI,const N:usize> UpdateWeight<U> for BiasLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,Arr<U,N>>;

    fn update_weight<OP: Optimizer<U>>(&mut self, stack: Self::GradientStack, optimizer: &mut OP) -> Result<(), TrainingError> {
        let (s,bias) = stack.pop();

        for (w,&g) in self.bias.scoped_mut().iter_mut().zip(bias.iter()) {
            optimizer.update(g, w);
        }

        Ok(self.parent.update_weight(s,optimizer)?)
    }
}
impl<U,C,P,D,I,PI,const N:usize> AskDiffInput<U> for BiasLayer<U,C,P,D,I,PI,N>
    where P: PreTrain<U,OutStack=<<Self as PreTrain<U>>::OutStack as Stack>::Remaining> +
             ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> +
             AskDiffInput<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          Self: PreTrain<U> {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map_remaining(|s| self.parent.ask_diff_input(s))
    }
}
impl<U,P,I,PI,const N:usize> Loss<U> for BiasLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,N>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          Arr<U,N>: From<PI>,
          for<'a> PI: Debug + Clone + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>> + 'static,
          for<'a> ArrView<'a,U,N>: From<&'a PI> {
}
impl<U,P,I,PI,const N:usize> Loss<U> for BiasLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,N>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Clone + Sync + From<PI>,
          Arr<U,N>: From<PI>,
          for<'a> ArrView<'a,U,N>: From<&'a PI>,
          DeviceGpu<U>: Device<U>,
          Self: BackwardAll<U> {
}
impl<U,C,P,D,I,PI,const N:usize> BatchForwardBase for BiasLayer<U,C,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + Send + Sync,
          <I as BatchDataType>::Type: Debug,
          Self: ForwardAll {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = SerializedVec<U,PI>;
}
impl<U,C,P,D,I,PI,const N:usize> BatchForward for BiasLayer<U,C,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward,
          D: Device<U> + DeviceBias<U,C,N>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,Arr<U,N>>,Error=SizeMismatchError>,
          for<'a> PI: Debug + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>> + 'static,
          for<'a> ArrView<'a,U,N>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError> {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_forward_bias(&self.bias,(&input).try_into()?)?.into_converter().try_into()?)
    }
}
impl<U,C,P,D,I,PI,const N:usize> BatchPreTrainBase<U> for BiasLayer<U,C,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + Send + Sync,
          <I as BatchDataType>::Type: Debug,
          Self: PreTrain<U> {
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,Self::BatchOutput>;
}
impl<U,C,P,D,I,PI,const N:usize> BatchPreTrain<U> for BiasLayer<U,C,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceBias<U,C,N>,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,Arr<U,N>>,Error=SizeMismatchError>,
          for<'a> PI: Debug + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>> + 'static,
          for<'a> ArrView<'a,U,N>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| self.device.batch_forward_bias(&self.bias,input.try_into()?))?;

        Ok(Cons(r,u.into_converter().try_into()?))
    }
}
impl<U,P,I,PI,const N:usize> BatchBackward<U> for BiasLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          for<'a> PI: Debug + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>> + 'static,
          for<'a> ArrView<'a,U,N>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,Arr<U,N>>: TryFrom<SerializedVecConverter<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,Arr<U,N>>,Error=SizeMismatchError> {
    type BatchLossInput = SerializedVec<U,PI>;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let g = self.device.batch_backward_bias_weight_gradient((&loss).try_into()?)?;

        let next_loss = self.device.batch_backward_bias(loss.into_converter().try_into()?)?;

        let (
            s,next_loss
        ) = self.parent.batch_loss(next_loss.into_converter().try_into()?,lossf,s)?;

        let (l,s) = self.parent.batch_backward(next_loss, s, lossf)?;

        Ok((l,Cons(s,g)))
    }
}
impl<U,P,I,PI,const N:usize> BatchBackward<U> for BiasLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          for<'a> PI: Debug + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>> + 'static,
          for<'a> ArrView<'a,U,N>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,Arr<U,N>>: TryFrom<SerializedVecConverter<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,Arr<U,N>>,Error=SizeMismatchError>,
          DeviceGpu<U>: Device<U> + DeviceBias<U,CachedTensor<U,Arr<U,N>>,N> {
    type BatchLossInput = SerializedVec<U,Arr<U,N>>;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let g = self.device.batch_backward_bias_weight_gradient((&loss).try_into()?)?;

        let next_loss = self.device.batch_backward_bias(loss.into_converter().try_into()?)?;

        let (
            s,
            next_loss
        ) = self.parent.batch_loss(next_loss.into_converter().try_into()?,lossf,s)?;

        let (l,s) = self.parent.batch_backward(next_loss, s, lossf)?;

        Ok((l,Cons(s,g)))
    }
}
impl<U,P,I,PI,const N:usize> BatchLoss<U> for BiasLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          Arr<U,N>: From<PI>,
          for<'a> PI: Debug + Clone + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>> + 'static,
          for<'a> ArrView<'a,U,N>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,Arr<U,N>>: TryFrom<SerializedVecConverter<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,Arr<U,N>>,Error=SizeMismatchError>,{
}
impl<U,P,I,PI,const N:usize> BatchLoss<U> for BiasLayer<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          Arr<U,N>: From<PI>,
          for<'a> PI: Debug + Clone + Send + Sync + SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> + From<Arr<U,N>>,
          for<'a> ArrView<'a,U,N>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,N>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,Arr<U,N>>: TryFrom<SerializedVecConverter<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,Arr<U,N>>,Error=SizeMismatchError>,
          DeviceGpu<U>: Device<U>,
          Self: Loss<U> + BatchBackward<U> {
}
impl<U,P,I,PI,const N:usize> BiasLayerInstantiation<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,N> for BiasLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync {
    fn instantiation<UI: FnMut() -> U>(parent: P, device: &DeviceCpu<U>, ui: UI) -> Result<BiasLayer<U, Arr<U,N>, P, DeviceCpu<U>, I, PI, N>, LayerInstantiationError> {
        let mut ui = ui;

        let mut bias = Arr::new();

        for it in bias.iter_mut() {
            *it = ui();
        }

        Ok(BiasLayer {
            parent: parent,
            device: device.clone(),
            bias: bias,
            u:PhantomData::<U>
        })
    }
}
impl<U,P,I,PI,const N:usize> BiasLayerInstantiation<U,CachedTensor<U,Arr<U,N>>,P,DeviceGpu<U>,I,PI,N> for BiasLayer<U,Arr<U,N>,P,DeviceCpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> {
    fn instantiation<UI: FnMut() -> U>(parent: P, device: &DeviceGpu<U>, ui: UI) -> Result<BiasLayer<U, CachedTensor<U,Arr<U,N>>, P, DeviceGpu<U>, I, PI, N>, LayerInstantiationError> {
        let mut ui = ui;

        let mut bias = Arr::new();

        for it in bias.iter_mut() {
            *it = ui();
        }

        Ok(BiasLayer {
            parent: parent,
            device: device.clone(),
            bias: CachedTensor::new(bias,device.get_memory_pool())?,
            u:PhantomData::<U>
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
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build<U,C,P,D,I,PI,UI: FnMut() -> U>(&self,parent:P,device:&D,ui:UI)
                                             -> Result<BiasLayer<U,C,P,D,I,PI,N>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> +
                 BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + Send + UnitValue<U>,
              D: Device<U>,
              I: Debug + Send + Sync + BatchDataType,
              PI: Debug + Send + Sync,
              <I as BatchDataType>::Type: Debug + Send + Sync + 'static,
              BiasLayer<U,C,P,D,I,PI,N>: BiasLayerInstantiation<U,C,P,D,I,PI,N> {
        BiasLayer::instantiation(parent,device,ui)
    }
}
