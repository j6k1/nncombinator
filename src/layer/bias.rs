//! Implementation of bias layer

use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::{Arr, IntoConverter};
use crate::{Cons, Stack};
use crate::cuda::{CudaTensor1dPtr, Memory};
use crate::device::{Device, DeviceCpu, DeviceGpu, DeviceMemoryPool};
use crate::device::bias::DeviceBias;
use crate::error::{ConfigReadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use crate::layer::{AskDiffInput, Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, BatchSize, Forward, ForwardAll, Loss, PreTrain, UpdateWeight};
use crate::lossfunction::LossFunction;
use crate::mem::AsRawSlice;
use crate::ope::{UnitValue};
use crate::optimizer::{Optimizer, OptimizerBuilder};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, UnitOrMarker};

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
impl<U,P,OP,I,PI,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized> for BiasLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,DeviceCpu<U>>,
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
impl<U,P,OP,I,PI,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized> for BiasLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U>,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        let mut bias = Arr::<U,N>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        self.bias.memcpy(bias.as_raw_slice().as_ptr(),N)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        let bias = self.bias.read_to_vec()?;

        for b in bias.iter() {
            persistence.write(UnitOrMarker::Unit(*b));
        }

        Ok(())
    }
}
impl<T,U,P,OP,I,PI,const N:usize> Persistence<U,T,Linear> for BiasLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,DeviceCpu<U>> {
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
impl<T,U,P,OP,I,PI,const N:usize> Persistence<U,T,Linear> for BiasLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        let mut bias = Arr::<U,N>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        self.bias.memcpy(bias.as_raw_slice().as_ptr(),N)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        let bias = self.bias.read_to_vec()?;

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
impl<U,P,OP,I,PI,const N:usize> BackwardAll<U> for BiasLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> + ForwardAll<Input=I,Output=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          DeviceCpu<U>: Device<U> + DeviceBias<U,Arr<U,N>,PI,N>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          OP: Optimizer<U,DeviceCpu<U>>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,N>> {
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
impl<U,P,OP,I,PI,const N:usize> BackwardAll<U> for BiasLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> + ForwardAll<Input=I,Output=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          OP: Optimizer<U,DeviceGpu<U>>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          DeviceGpu<U>: Device<U> + DeviceBias<U,CudaTensor1dPtr<U,N>,PI,N>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,N>> {
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
impl<U,P,OP,I,PI,const N:usize> UpdateWeight<U> for BiasLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          OP: Optimizer<U,DeviceCpu<U>>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          OP: Optimizer<U,DeviceCpu<U>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,N>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,Arr<U,N>>;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,bias) = stack.pop();

        self.optimizer.update((&bias).into(),(&mut self.bias).into())?;

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,P,OP,I,PI,const N:usize> UpdateWeight<U> for BiasLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          OP: Optimizer<U,DeviceGpu<U>>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          DeviceGpu<U>: Device<U>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,N>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,CudaTensor1dPtr<U,N>>;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,bias) = stack.pop();

        self.optimizer.update((&bias).into(),(&mut self.bias).into())?;

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,C,P,OP,D,I,PI,const N:usize> AskDiffInput<U> for BiasLayer<U,C,P,OP,D,I,PI,N>
    where P: PreTrain<U,OutStack=<<Self as PreTrain<U>>::OutStack as Stack>::Remaining> +
             ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> +
             AskDiffInput<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          Self: PreTrain<U,PreOutput=PI> {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map_remaining(|s| self.parent.ask_diff_input(s))
    }
}
impl<U,P,OP,I,PI,const N:usize> Loss<U> for BiasLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          OP: Optimizer<U,DeviceCpu<U>>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          DeviceCpu<U>: Device<U> + DeviceBias<U,Arr<U,N>,PI,N>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,N>> {
}
impl<U,P,OP,I,PI,const N:usize> Loss<U> for BiasLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + 'static,
          OP: Optimizer<U,DeviceGpu<U>>,
          <PI as BatchDataType>::Type: Debug + BatchSize + 'static,
          DeviceGpu<U>: Device<U> + DeviceBias<U,CudaTensor1dPtr<U,N>,PI,N>,
          Self: BackwardAll<U>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,N>> {
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
          OP: Optimizer<U,D> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| self.device.batch_forward_bias(&self.bias,input))?;

        Ok(Cons(r,u))
    }
}
impl<U,P,OP,I,PI,const N:usize> BatchBackward<U> for BiasLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + IntoConverter + 'static,
          <I as BatchDataType>::Type: Debug,
          OP: Optimizer<U,DeviceCpu<U>>,
          DeviceCpu<U>: Device<U> + DeviceBias<U,Arr<U,N>,PI,N>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,N>> {
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
impl<U,P,OP,I,PI,const N:usize> BatchBackward<U> for BiasLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + IntoConverter + 'static,
          <I as BatchDataType>::Type: Debug,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> + DeviceBias<U,CudaTensor1dPtr<U,N>,PI,N>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,N>> {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let g = self.device.batch_backward_bias_weight_gradient(&loss)?;

        let next_loss = self.device.batch_backward_bias(loss)?;

        let (
            s,
            next_loss
        ) = self.parent.batch_loss(next_loss,lossf,s)?;

        let (l,s) = self.parent.batch_backward(next_loss, s, lossf)?;

        Ok((l,Cons(s,g)))
    }
}
impl<U,P,OP,I,PI,const N:usize> BatchLoss<U> for BiasLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + IntoConverter + 'static,
          <I as BatchDataType>::Type: Debug,
          OP: Optimizer<U,DeviceCpu<U>>,
          DeviceCpu<U>: Device<U> + DeviceBias<U,Arr<U,N>,PI,N>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,N>> {
}
impl<U,P,OP,I,PI,const N:usize> BatchLoss<U> for BiasLayer<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + BatchDataType + 'static,
          <PI as BatchDataType>::Type: Debug + BatchSize + IntoConverter + 'static,
          <I as BatchDataType>::Type: Debug,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> + DeviceBias<U,CudaTensor1dPtr<U,N>,PI,N>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,N>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,N>>,
          Self: Loss<U> + BatchBackward<U> {
}
impl<U,P,OP,I,PI,const N:usize> BiasLayerInstantiation<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,N> for BiasLayer<U,Arr<U,N>,P,OP,DeviceCpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          OP: Optimizer<U,DeviceCpu<U>> {
    fn instantiation<UI: FnMut() -> U,B: OptimizerBuilder<U,DeviceCpu<U>,Output=OP>>(parent: P, device: &DeviceCpu<U>, ui: UI, b: &B)
        -> Result<BiasLayer<U, Arr<U,N>, P, OP, DeviceCpu<U>, I, PI, N>, LayerInstantiationError> {
        let mut ui = ui;

        let mut bias = Arr::new();

        for it in bias.iter_mut() {
            *it = ui();
        }

        Ok(BiasLayer {
            parent: parent,
            device: device.clone(),
            bias: bias,
            u:PhantomData::<U>,
            optimizer:b.build(N)?
        })
    }
}
impl<U,P,OP,I,PI,const N:usize> BiasLayerInstantiation<U,CudaTensor1dPtr<U,N>,P,OP,DeviceGpu<U>,I,PI,N> for BiasLayer<U,Arr<U,N>,P,OP,DeviceGpu<U>,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> {
    fn instantiation<UI: FnMut() -> U,B: OptimizerBuilder<U,DeviceGpu<U>,Output=OP>>(parent: P, device: &DeviceGpu<U>, ui: UI, b: &B)
        -> Result<BiasLayer<U, CudaTensor1dPtr<U,N>, P, OP, DeviceGpu<U>, I, PI, N>, LayerInstantiationError> {
        Ok(BiasLayer {
            parent: parent,
            device: device.clone(),
            bias: CudaTensor1dPtr::with_initializer(device.get_memory_pool(),ui)?,
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
