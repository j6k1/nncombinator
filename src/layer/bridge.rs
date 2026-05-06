//! Implementation of a layer for inverse transformation of the error type during back propagation

use std::fmt::Debug;
use std::marker::PhantomData;
use crate::arr::{IntoConverter, MakeView, MakeViewMut, SerializedVec, SerializedVecConverter, SliceSize};
use crate::device::Device;
use crate::error::{ModelLoadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError, TypeConvertError};
use crate::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, ContinueForward, ForwardAll, ForwardDiff, Loss, PartialForward, PreTrain, UpdateWeight, OnStep};
use crate::lossfunction::LossFunction;
use crate::mem::AsRawSlice;
use crate::ope::UnitValue;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence};

/// Bridge layer Implementation
pub struct BridgeLayer<U,P,I,PI,CI,D> where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
                                                             U: UnitValue<U>,
                                                             D: Device<U>,
                                                             PI: Debug + 'static,
                                                             CI: Debug + 'static,
                                                             I: Debug + Send + Sync {
    parent:P,
    device:PhantomData<D>,
    u:PhantomData<U>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
    ci:PhantomData<CI>
}
impl<U,P,I,PI,CI,D> Persistence<U,TextFilePersistence,Specialized> for BridgeLayer<U,P,I,PI,CI,D>
    where P: ForwardAll<Input=I,Output=PI> + Persistence<U,TextFilePersistence,Specialized> +
             BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: UnitValue<U> + std::str::FromStr,
          D: Device<U>,
          PI: Debug + 'static,
          CI: Debug + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<T,U,P,I,PI,CI,D> Persistence<U,T,Linear> for BridgeLayer<U,P,I,PI,CI,D>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: UnitValue<U>,
          D: Device<U>,
          PI: Debug + 'static,
          CI: Debug + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,I,PI,CI,D> ForwardAll for BridgeLayer<U,P,I,PI,CI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + 'static,
          CI: Debug + 'static,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = PI;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.parent.forward_all(input)
    }
}
impl<U,P,I,PI,CI,D> PreTrain<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + From<CI>,
          CI: Debug + 'static,
          I: Debug + Send + Sync {
    type PreOutput = PI;
    type OutStack = <P as PreTrain<U>>::OutStack;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        Ok(self.parent.pre_train(input)?)
    }
}
impl<U,P,I,PI,CI,D> BackwardAll<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + From<CI>,
          CI: Debug + 'static,
          I: Debug + Send + Sync, {
    type LossInput = CI;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        Ok(self.parent.backward_all(input.into(), stack, lossf)?.into())
    }
}
impl<U,P,I,PI,CI,D> UpdateWeight<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + 
             Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + From<CI>,
          CI: Debug + 'static,
          I: Debug + Send + Sync, {
    type GradientStack = <P as UpdateWeight<U>>::GradientStack;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        Ok(self.parent.update_weight(stack,batch_size)?)
    }
}
impl<U,P,I,PI,CI,D> PartialForward for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + PartialForward<DiffOutput=PI>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + From<CI>,
          CI: Debug + 'static,
          I: Debug + Send + Sync {
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
impl<U,P,I,PI,CI,D> ForwardDiff for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             PartialForward<DiffOutput=PI> + ForwardDiff +
             BackwardAll<U,LossInput=PI> + Loss<U>,
      U: Default + Clone + Copy + UnitValue<U>,
      D: Device<U>,
      PI: Debug + From<CI>,
      CI: Debug + 'static,
      I: Debug + Send + Sync {
    fn forward_diff(&self, input: Self::DiffInput) -> Result<Self::DiffOutput, EvaluateError> {
        Ok(self.parent.forward_diff(input)?)
    }
}
impl<U,P,I,PI,CI,D> ContinueForward for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
          PartialForward<DiffOutput=PI> + ContinueForward<ConinueOutput=PI> +
          BackwardAll<U,LossInput=PI> + Loss<U>,
      U: Default + Clone + Copy + UnitValue<U>,
      D: Device<U>,
      PI: Debug + From<CI>,
      CI: Debug + 'static,
      I: Debug + Send + Sync {
    type ConinueOutput = Self::Output;
    fn continue_forward(&self, input: &Self::PartialOutput) -> Result<Self::ConinueOutput, EvaluateError> {
        Ok(self.parent.continue_forward(input)?)
    }
}
impl<U,P,I,PI,CI,D> Loss<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + From<CI>,
          CI: Debug + 'static,
          I: Debug + Send + Sync {}
impl<U,P,I,PI,CI,D> BatchForwardBase for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + From<CI> + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug,
          for<'a> CI: Debug + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,P,I,PI,CI,D> BatchForward for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + From<CI> + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug,
          for<'a> CI: Debug + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        self.parent.batch_forward(input)
    }
}
impl<U,P,I,PI,CI,D> BatchPreTrainBase<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<U> + BatchPreTrain<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + From<CI> + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug,
          for<'a> CI: Debug + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = <P as BatchPreTrainBase<U>>::BatchOutStack;
}
impl<U,P,I,PI,CI,D> BatchPreTrain<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<U> + BatchPreTrain<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + From<CI> + BatchDataType,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug,
          for<'a> CI: Debug + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        self.parent.batch_pre_train(input)
    }
}
impl<U,P,I,PI,CI,D> BatchBackward<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<U> + BatchPreTrain<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + From<CI> + BatchDataType,
          for<'a> CI: Debug + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          SerializedVec<U,CI>: IntoConverter,
          <PI as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type : TryFrom<<SerializedVec<U,CI> as IntoConverter>::Converter,Error=TypeConvertError> {
    type BatchLossInput = SerializedVec<U,CI>;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;
    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        self.parent.batch_backward(input.into_converter().try_into()?, stack, lossf)
    }
}
impl<U,P,I,PI,CI,D> BatchLoss<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<U> + BatchPreTrain<U,BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + From<CI> + BatchDataType,
          for<'a> CI: Debug + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type : TryFrom<SerializedVecConverter<U,CI>,Error=TypeConvertError> {

}
impl<U,P,I,PI,CI,D> OnStep for BridgeLayer<U,P,I,PI,CI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> + OnStep,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          CI: Debug {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        Ok(self.parent.on_step(step)?)
    }
}
/// Trait for BridgeLayer instance creation
pub trait BridgeLayerInstantiation<U,P,I,PI,CI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          PI: Debug + 'static,
          CI: Debug + 'static,
          I: Debug + Send + Sync + 'static + BatchDataType,
          <I as BatchDataType>::Type: Debug + Send + Sync + 'static {
    /// Create and return an instance with the specified scale, bias, and momentum.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    fn instantiation(parent:P,device:&D) -> Result<BridgeLayer<U,P,I,PI,CI,D>,LayerInstantiationError>;
}
impl<U,P,I,PI,CI,D> BridgeLayerInstantiation<U,P,I,PI,CI,D> for BridgeLayer<U,P,I,PI,CI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: UnitValue<U>,
          D: Device<U>,
          PI: Debug + 'static,
          CI: Debug + 'static,
          I: Debug + Send + Sync + 'static + BatchDataType,
          <I as BatchDataType>::Type: Debug + Send + Sync + 'static {
    /// Create and return an instance of BridgeLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    fn instantiation(parent:P,_:&D) -> Result<BridgeLayer<U,P,I,PI,CI,D>,LayerInstantiationError> {
        Ok(BridgeLayer {
            parent:parent,
            device:PhantomData::<D>,
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            pi:PhantomData::<PI>,
            ci:PhantomData::<CI>
        })
    }
}
/// Builder for BridgeLayer instance creation
pub struct BridgeLayerBuilder<CI> where CI: Debug + 'static {
    ci:PhantomData<CI>
}
impl<CI> BridgeLayerBuilder<CI> where CI: Debug + 'static {
    pub fn new() -> BridgeLayerBuilder<CI> {
        BridgeLayerBuilder {
            ci:PhantomData::<CI>
        }
    }

    /// Create an instance of BridgeLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build<U,P,I,PI,D>(&self,parent:P,device:&D) -> Result<BridgeLayer<U,P,I,PI,CI,D>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> +
                 BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
              U: Default + Clone + Copy + Send + UnitValue<U>,
              D: Device<U>,
              PI: Debug + 'static,
              CI: Debug + 'static,
              I: Debug + Send + Sync + 'static + BatchDataType,
              <I as BatchDataType>::Type: Debug + Send + Sync + 'static,
              BridgeLayer<U,P,I,PI,CI,D>: BridgeLayerInstantiation<U,P,I,PI,CI,D> {
        BridgeLayer::<U,P,I,PI,CI,D>::instantiation(parent,device)
    }
}
