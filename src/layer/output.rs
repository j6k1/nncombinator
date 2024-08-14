//! Implementation of output layers
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::{Stack};
use crate::arr::{Arr, IntoConverter, SerializedVec};
use crate::device::Device;
use crate::device::output::DeviceLinearOutput;
use crate::error::{ConfigReadError, EvaluateError, PersistenceError, SizeMismatchError, TrainingError, TypeConvertError};
use crate::layer::{AskDiffInput, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, BatchSize, BatchTrain, ForwardAll, Loss, PreTrain, Train, UpdateWeight};
use crate::lossfunction::{BatchLossFunctionLinear, LossFunction, LossFunctionLinear};
use crate::ope::UnitValue;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence};

/// Layer implementation of the output layer (linear layer)
pub struct LinearOutputLayer<U,P,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    u:PhantomData<U>,
    i:PhantomData<I>,
    io:PhantomData<PI>,
    n:PhantomData<[();N]>,
    parent:P,
    device:D,
}
impl<U,P,D,I,PI,const N:usize> LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    /// Create and return an instance of LinearOutputLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    pub fn new(parent:P,device:&D) -> LinearOutputLayer<U,P,D,I,PI,N> {
        LinearOutputLayer {
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            io:PhantomData::<PI>,
            n:PhantomData::<[();N]>,
            parent:parent,
            device:device.clone(),
        }
    }
}
impl<U,P,D,I,PI,const N:usize> Persistence<U,TextFilePersistence<U>,Specialized> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr + Sized,
          D: Device<U>,
          PI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;
        persistence.verify_eof()
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<T,U,P,D,I,PI,const N:usize> Persistence<U,T,Linear> for LinearOutputLayer<U,P,D,I,PI,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          PI: Debug + Send + Sync + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;
        persistence.verify_eof()
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,D,I,PI,const N:usize> ForwardAll for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + 'static,
          I: Debug + Send + Sync,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N>,
          Arr<U,N>: From<PI> {
    type Input = I;
    type Output = Arr<U,N>;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(self.parent.forward_all(input)?.into())
    }
}
impl<U,P,D,I,PI,const N:usize> PreTrain<U> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + 'static,
          I: Debug + Send + Sync,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N>,
          Arr<U,N>: From<PI> {
    type PreOutput = PI;
    type OutStack = P::OutStack;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        self.parent.pre_train(input)
    }
}
impl<U,P,D,I,PI,const N:usize> BackwardAll<U> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> +
             ForwardAll<Input=I,Output=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + 'static,
          I: Debug + Send + Sync,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N>,
          Arr<U,N>: From<PI> {
    type LossInput = PI;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        self.parent.backward_all(input, stack, lossf)
    }
}
impl<U,P,D,I,PI,const N:usize> UpdateWeight<U> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> +
             ForwardAll<Input=I,Output=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + 'static,
          I: Debug + Send + Sync,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N> {
    type GradientStack = <P as UpdateWeight<U>>::GradientStack;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        Ok(self.parent.update_weight(stack)?)
    }
}
impl<U,P,D,I,PI,const N:usize> AskDiffInput<U> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> +
             ForwardAll<Input=I,Output=PI> + PreTrain<U,PreOutput=PI> + Loss<U> + AskDiffInput<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + 'static,
          I: Debug + Send + Sync,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N>,
          Arr<U,N>: From<PI> {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        self.parent.ask_diff_input(stack)
    }
}
impl<U,P,D,I,PI,L,const N:usize> Train<U,L> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> +
             ForwardAll<Input=I,Output=PI> + PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + 'static,
          I: Debug + Send + Sync,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI>,
          Arr<U,N>: From<PI>,
          for<'a> L: LossFunction<U> + LossFunctionLinear<'a,U,PI,D,N,Output=PI> {
    fn train(&mut self, expected: Self::Output, input: Self::Input, lossf: &L) -> Result<U, TrainingError> {
        let stack = self.pre_train(input)?;

        let total_loss = stack.map(|l| self.device.loss_linear_total(&expected,l,lossf))?;

        let (stack,loss) = if self.parent.is_canonical_link(lossf) {
            let loss = stack.map(|actual| {
                self.device.loss_linear_by_canonical_link(&expected, &actual)
            })?;

            (stack,loss)
        } else {
            let loss = stack.map(|actual| {
                self.device.loss_linear(&expected,&actual,lossf)
            })?;

            self.parent.loss(loss,lossf,stack)?
        };

        let (_,s) = self.backward_all(loss,stack,lossf)?;

        self.parent.update_weight(s)?;

        Ok(total_loss)
    }
}
impl<U,P,D,I,PI,const N:usize> BatchForwardBase for LinearOutputLayer<U,P,D,I,PI,N>
    where P: PreTrain<U,PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + BatchDataType + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + IntoConverter,
          <I as BatchDataType>::Type: Debug,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N>,
          Arr<U,N>: From<PI>,
          SerializedVec<U,Arr<U,N>>: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError> {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = SerializedVec<U,Arr<U,N>>;
}
impl<U,P,D,I,PI,const N:usize> BatchForward for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + BatchDataType + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + IntoConverter,
          <I as BatchDataType>::Type: Debug,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N>,
          Arr<U,N>: From<PI>,
          SerializedVec<U,Arr<U,N>>: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError> {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        Ok(self.parent.batch_forward(input)?.into_converter().try_into()?)
    }
}
impl<U,P,D,I,PI,const N:usize> BatchPreTrainBase<U> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + BatchDataType + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + IntoConverter,
          <I as BatchDataType>::Type: Debug,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N>,
          Arr<U,N>: From<PI>,
          SerializedVec<U,Arr<U,N>>: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError> {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = P::BatchOutStack;
}
impl<U,P,D,I,PI,const N:usize> BatchPreTrain<U> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + BatchDataType + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + IntoConverter,
          <I as BatchDataType>::Type: Debug,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N>,
          Arr<U,N>: From<PI>,
          SerializedVec<U,Arr<U,N>>: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        self.parent.batch_pre_train(input)
    }
}
impl<U,P,D,I,PI,const N:usize> BatchBackward<U> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + UpdateWeight<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + BatchDataType + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + IntoConverter,
          <I as BatchDataType>::Type: Debug + BatchSize,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N>,
          Arr<U,N>: From<PI>,
          SerializedVec<U,Arr<U,N>>: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError> {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        self.parent.batch_backward(input,stack,lossf)
    }
}
impl<U,P,D,I,PI,L,const N:usize> BatchTrain<U,D,L> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain<U> +
             BatchBackward<U> + UpdateWeight<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          PI: Debug + BatchDataType + Send + Sync + BatchDataType + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + IntoConverter,
          <I as BatchDataType>::Type: Debug + BatchSize,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,BatchIO=<PI as BatchDataType>::Type>,
          f64: From<U>,
          Arr<U,N>: From<PI>,
          SerializedVec<U,Arr<U,N>>: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError>,
          Self: UpdateWeight<U,GradientStack = <P as UpdateWeight<U>>::GradientStack>,
          for<'a> L: LossFunction<U> + BatchLossFunctionLinear<'a,U,<PI as BatchDataType>::Type,D,N,Output=<PI as BatchDataType>::Type> {
    fn batch_train(&mut self, expected:Self::BatchOutput, input:Self::BatchInput, lossf:&L) -> Result<U, TrainingError> {
        if expected.len() != input.size() {
            return Err(TrainingError::from(SizeMismatchError(expected.len(),input.size())));
        }

        let stack = self.batch_pre_train(input)?;

        let total_loss = stack.map(|l| self.device.batch_loss_linear_total(&expected,l,lossf))?;

        let (stack,loss) = if self.parent.is_canonical_link(lossf) {
            let loss = stack.map(|actual| {
                self.device.loss_linear_batch_by_canonical_link(&expected, &actual)
            })?;

            (stack,loss)
        } else {
            let loss = stack.map(|actual| {
                self.device.batch_loss_linear(&expected,actual,lossf)
            })?;

            self.parent.batch_loss(loss,lossf,stack)?
        };

        let (_,s) = self.parent.batch_backward(loss,stack,lossf)?;

        self.parent.update_weight(s)?;

        Ok(total_loss)
    }
}
