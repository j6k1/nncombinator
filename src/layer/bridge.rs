//! Implementation of a layer for inverse transformation of the error type during back propagation

use std::fmt::Debug;
use std::marker::PhantomData;
use crate::arr::{MakeView, MakeViewMut, SerializedVec, SerializedVecConverter, SliceSize};
use crate::device::Device;
use crate::error::{ConfigReadError, EvaluateError, LayerInstantiationError, PersistenceError, SizeMismatchError, TrainingError};
use crate::layer::{AskDiffInput, BackwardAll, BatchBackward, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, ForwardAll, Loss, PreTrain};
use crate::lossfunction::LossFunction;
use crate::mem::AsRawSlice;
use crate::ope::UnitValue;
use crate::optimizer::Optimizer;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence};

/// Bridge layer Implementation
pub struct BridgeLayer<U,P,I,PI,CI,D> where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
                                                             U: UnitValue<U>,
                                                             D: Device<U>,
                                                             PI: Debug + Send + Sync + 'static,
                                                             CI: Debug + Send + Sync + 'static,
                                                             I: Debug + Send + Sync {
    parent:P,
    device:PhantomData<D>,
    u:PhantomData<U>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
    ci:PhantomData<CI>
}
impl<U,P,I,PI,CI,D> Persistence<U,TextFilePersistence<U>,Specialized> for BridgeLayer<U,P,I,PI,CI,D>
    where P: ForwardAll<Input=I,Output=PI> + Persistence<U,TextFilePersistence<U>,Specialized> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U> + std::str::FromStr,
          D: Device<U>,
          PI: Debug + Send + Sync + 'static,
          CI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<T,U,P,I,PI,CI,D> Persistence<U,T,Linear> for BridgeLayer<U,P,I,PI,CI,D>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + 'static,
          CI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,I,PI,CI,D> ForwardAll for BridgeLayer<U,P,I,PI,CI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + 'static,
          CI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = PI;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.parent.forward_all(input)
    }
}
impl<U,P,I,PI,CI,D> PreTrain<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + From<CI>,
          CI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    type OutStack = <P as PreTrain<U>>::OutStack;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        Ok(self.parent.pre_train(input)?)
    }
}
impl<U,P,I,PI,CI,D> BackwardAll<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + From<CI>,
          CI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync, {
    type LossInput = CI;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) -> Result<(), TrainingError> {
        Ok(self.parent.backward_all(input.into(), stack, optimizer,lossf)?.into())
    }
}
impl<U,P,I,PI,CI,D> AskDiffInput<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> + AskDiffInput<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + From<CI>,
          CI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        self.parent.ask_diff_input(stack)
    }
}
impl<U,P,I,PI,CI,D> Loss<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + From<CI>,
          CI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {}
impl<U,P,I,PI,CI,D> BatchForwardBase for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,PI>> + BatchPreTrainBase<U> + BatchBackward<U> +
             BatchLoss<U,BatchLossInput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + From<CI>,
          for<'a> CI: Debug + Send + Sync + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static,
          I: Debug + Send + Sync {
    type BatchInput = SerializedVec<U,I>;
    type BatchOutput = SerializedVec<U,PI>;
}
impl<U,P,I,PI,CI,D> BatchForward for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + From<CI>,
          for<'a> CI: Debug + Send + Sync + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static,
          I: Debug + Send + Sync,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,CI>,Error=SizeMismatchError> {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        self.parent.batch_forward(input)
    }
}
impl<U,P,I,PI,CI,D> BatchPreTrainBase<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,PI>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + From<CI>,
          for<'a> CI: Debug + Send + Sync + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static,
          I: Debug + Send + Sync {
    type BatchOutStack = <P as BatchPreTrainBase<U>>::BatchOutStack;
}
impl<U,P,I,PI,CI,D> BatchPreTrain<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,PI>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + From<CI>,
          for<'a> CI: Debug + Send + Sync + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static,
          I: Debug + Send + Sync,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,CI>,Error=SizeMismatchError> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        self.parent.batch_pre_train(input)
    }
}
impl<U,P,I,PI,CI,D> BatchBackward<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,PI>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + From<CI>,
          for<'a> CI: Debug + Send + Sync + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static,
          I: Debug + Send + Sync,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,CI>,Error=SizeMismatchError> {
    type BatchLossInput = SerializedVec<U,CI>;
    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, optimizer: &mut OP, lossf: &L) -> Result<(), TrainingError> {
        self.parent.batch_backward(input.into_converter().try_into()?, stack, optimizer, lossf)
    }
}
impl<U,P,I,PI,CI,D> BatchLoss<U> for BridgeLayer<U,P,I,PI,CI,D>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,PI>> +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + From<CI>,
          for<'a> CI: Debug + Send + Sync + SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> + 'static,
          I: Debug + Send + Sync,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,CI>,Error=SizeMismatchError> {

}
/// Trait for BridgeLayer instance creation
pub trait BridgeLayerInstantiation<U,P,I,PI,CI,D>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + 'static,
          CI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static {
    /// Create and return an instance with the specified scale, bias, and momentum.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    fn instantiation(parent:P,device:&D) -> Result<BridgeLayer<U,P,I,PI,CI,D>,LayerInstantiationError>;
}
impl<U,P,I,PI,CI,D> BridgeLayerInstantiation<U,P,I,PI,CI,D> for BridgeLayer<U,P,I,PI,CI,D>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + 'static,
          CI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + 'static {
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
pub struct BridgeLayerBuilder<CI> where CI: Debug + Send + Sync + 'static {
    ci:PhantomData<CI>
}
impl<CI> BridgeLayerBuilder<CI> where CI: Debug + Send + Sync + 'static {
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
                 BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + Send + UnitValue<U>,
              D: Device<U>,
              PI: Debug + Send + Sync + 'static,
              CI: Debug + Send + Sync + 'static,
              I: Debug + Send + Sync + 'static,
              SerializedVec<U,I>: Debug + Send + Sync + 'static,
              BridgeLayer<U,P,I,PI,CI,D>: BridgeLayerInstantiation<U,P,I,PI,CI,D> {
        BridgeLayer::<U,P,I,PI,CI,D>::instantiation(parent,device)
    }
}
