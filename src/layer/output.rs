//! Implementation of output layers
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, Sub};
use std::str::FromStr;
use num_traits::FromPrimitive;
use crate::{Stack};
use crate::arr::{Arr, SerializedVec};
use crate::bridge::ToHost;
use crate::cuda::DataTypeInfo;
use crate::device::{Device};
use crate::device::output::DeviceLinearOutput;
use crate::error::{ModelLoadError, EvaluateError, PersistenceError, SizeMismatchError, TrainingError};
use crate::layer::{BackwardAll, BackwardBase, BatchBackward, BatchBackwardBase, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, BatchSize, BatchTrain, ContinueForward, ForwardAll, ForwardDiff, InputTensorScalar, Loss, OnStep, OutputTensorScalar, PartialForward, PersistProgress, PreTrain, PreTrainBase, Step, Train, UpdateWeight};
use crate::lossfunction::{BatchLossFunctionLinear, LossFunction, LossFunctionLinear};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextPersistence, TextRecord, VerifyEof};

/// Layer implementation of the output layer (linear layer)
pub struct LinearOutputLayer<U,P,D,I,PI,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          D: Device<U>,
          PI: Debug + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync {
    u:PhantomData<U>,
    i:PhantomData<I>,
    io:PhantomData<PI>,
    n:PhantomData<[();N]>,
    parent:P,
    device:D,
    step_count:usize,
    frequently_steps:usize
}
impl<U,P,D,I,PI,const N:usize> LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          D: Device<U>,
          PI: Debug + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync {
    /// Create and return an instance of LinearOutputLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    pub fn new(parent:P,device:&D) -> Result<LinearOutputLayer<U,P,D,I,PI,N>,TrainingError> {
        let mut l = LinearOutputLayer {
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            io:PhantomData::<PI>,
            n:PhantomData::<[();N]>,
            parent:parent,
            device:device.clone(),
            step_count:0,
            frequently_steps:0
        };

        l.parent.on_step(0)?;
        
        Ok(l)
    }
}
impl<U,P,D,I,PI,const N:usize> InputTensorScalar for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo + FromStr + Sized,
          D: Device<U>,
          PI: Debug + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync,
          TextFilePersistence: VerifyEof {
    type Scalar = U;
}
impl<U,P,D,I,PI,const N:usize> OutputTensorScalar for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign +
             FromPrimitive + 'static + DataTypeInfo + FromStr + Sized,
          D: Device<U>,
          PI: Debug + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync,
          TextFilePersistence: VerifyEof {
    type Scalar = U;
}
impl<U,P,D,I,PI,const N:usize> Persistence<TextFilePersistence,Specialized> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             Persistence<TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo + FromStr + Sized,
          D: Device<U>,
          PI: Debug + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync,
          TextFilePersistence: VerifyEof {
    fn load(&mut self, persistence: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)?;
        persistence.verify_eof()
    }

    fn save(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<T,U,P,D,I,PI,const N:usize> Persistence<T,Linear> for LinearOutputLayer<U,P,D,I,PI,N>
    where T: LinearPersistence<U> + VerifyEof,
          P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain + Persistence<T,Linear> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          D: Device<U>,
          PI: Debug + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)?;
        persistence.verify_eof()
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,D,I,PI,const N:usize> ForwardAll for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync,
          <PI as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    type Input = I;
    type Output = <PI as ToHost<U>>::Output;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(self.parent.forward_all(input)?.to_host()?)
    }
}
impl<U,P,D,I,PI,const N:usize> PreTrainBase for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync +
          Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync,
          <PI as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    type PreOutput = PI;
    type OutStack = P::OutStack;
}
impl<U,P,D,I,PI,const N:usize> PreTrain for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync,
          <PI as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        self.parent.pre_train(input)
    }
}
impl<U,P,D,I,PI,const N:usize> BackwardBase for LinearOutputLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> +
             ForwardAll<Input=I,Output=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain + Loss<U> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync +
          Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync,
          <PI as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    type LossInputScalar = U;
    type LossInput = PI;
}
impl<U,P,D,I,PI,const N:usize> BackwardAll<U> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> +
             ForwardAll<Input=I,Output=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain + Loss<U> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync,
          <PI as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all(&mut self, input: Self::LossInput, stack:Self::OutStack)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        self.parent.backward_all(input, stack)
    }
}
impl<U,P,D,I,PI,const N:usize> UpdateWeight for LinearOutputLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> +
             ForwardAll<Input=I,Output=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + UpdateWeight,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync,
          <PI as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    type GradientStack = <P as UpdateWeight>::GradientStack;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        Ok(self.parent.update_weight(stack,batch_size)?)
    }
}
impl<U,P,D,I,PI,const N:usize> PartialForward for LinearOutputLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> +
             ForwardAll<Input=I,Output=PI> + PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             PartialForward,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync,
          <PI as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    type PartialInput = <P as PartialForward>::PartialInput;
    type PartialOutput = <P as PartialForward>::PartialOutput;
    type DiffInput = <P as PartialForward>::DiffInput;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput,EvaluateError> {
        Ok(self.parent.partial_forward(input)?)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput)
        -> Result<Self::PartialOutput,EvaluateError> {
        Ok(self.parent.partial_forward_by_diff(input,partial_input)?)
    }
}
impl<U,P,D,I,PI,const N:usize> ForwardDiff for LinearOutputLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> +
          ForwardAll<Input=I,Output=PI> + ForwardDiff +
          PreTrainBase<PreOutput=PI> + PreTrain + InputTensorScalar + OutputTensorScalar +
          PartialForward,
      U: Default + Clone + Copy + Debug + Send + Sync +
         Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
      PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
      I: Debug + Send + Sync,
      <PI as ToHost<U>>::Output: Debug + 'static,
      for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    fn forward_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        Ok(self.parent.forward_diff(input,partial_input)?.to_host()?)
    }
}
impl<U,P,D,I,PI,const N:usize> ContinueForward for LinearOutputLayer<U,P,D,I,PI,N>
    where P: BackwardAll<U,LossInput=PI> +
          ForwardAll<Input=I,Output=PI> + ForwardDiff +
          PreTrainBase<PreOutput=PI> + PreTrain + InputTensorScalar + OutputTensorScalar +
          PartialForward + ContinueForward,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync,
          <PI as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    fn continue_forward(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        Ok(self.parent.continue_forward(input)?.to_host()?)
    }
}
impl<U,P,D,I,PI,L,const N:usize> Train<U,L> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain + Loss<U> + InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync,
          <PI as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI>,
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

            self.parent.loss(loss,stack)?
        };

        let (_,s) = self.backward_all(loss,stack)?;

        self.parent.update_weight(s,1)?;

        Ok(total_loss)
    }
}
impl<U,P,D,I,PI,const N:usize> BatchForwardBase for LinearOutputLayer<U,P,D,I,PI,N>
    where P: PreTrainBase<PreOutput=PI> + PreTrain + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + ToHost<U,Output=SerializedVec<U,Arr<U,N>>>,
          <PI as ToHost<U>>::Output: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          <<PI as BatchDataType>::Type as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <<PI as BatchDataType>::Type as ToHost<U>>::Output;
}
impl<U,P,D,I,PI,const N:usize> BatchForward for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + ToHost<U,Output=SerializedVec<U,Arr<U,N>>>,
          <PI as ToHost<U>>::Output: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          <<PI as BatchDataType>::Type as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        Ok(self.parent.batch_forward(input)?.to_host()?)
    }
}
impl<U,P,D,I,PI,const N:usize> BatchPreTrainBase for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + ToHost<U,Output=SerializedVec<U,Arr<U,N>>>,
          <PI as ToHost<U>>::Output: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          <<PI as BatchDataType>::Type as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = P::BatchOutStack;
}
impl<U,P,D,I,PI,const N:usize> BatchPreTrain for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + ToHost<U,Output=SerializedVec<U,Arr<U,N>>>,
          <PI as ToHost<U>>::Output: Debug + 'static,
          <I as BatchDataType>::Type: Debug,
          <<PI as BatchDataType>::Type as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        self.parent.batch_pre_train(input)
    }
}
impl<U,P,D,I,PI,const N:usize> BatchBackwardBase for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain + Loss<U> +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackward<U> + UpdateWeight + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync +
          Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + ToHost<U,Output=SerializedVec<U,Arr<U,N>>>,
          <PI as ToHost<U>>::Output: Debug + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          <<PI as BatchDataType>::Type as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackwardBase>::BatchLossOutput;
}
impl<U,P,D,I,PI,const N:usize> BatchBackward<U> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain + Loss<U> +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackward<U> + UpdateWeight + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + ToHost<U,Output=SerializedVec<U,Arr<U,N>>>,
          <PI as ToHost<U>>::Output: Debug + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          <<PI as BatchDataType>::Type as ToHost<U>>::Output: Debug + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI> {
    fn batch_backward(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack)
        -> Result<(<Self as BatchBackwardBase>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        self.parent.batch_backward(input,stack)
    }
}
impl<U,P,D,I,PI,L,const N:usize> BatchTrain<U,D,L> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardBase<LossInput=PI> + BackwardAll<U> +
             PreTrainBase<PreOutput=PI> + PreTrain + Loss<U> +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackward<U> + UpdateWeight + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          PI: Debug + InputTensorScalar<Scalar=U> + BatchDataType + ToHost<U,Output=Arr<U,N>> + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug + ToHost<U,Output=SerializedVec<U,Arr<U,N>>> +
                                       InputTensorScalar<Scalar=U> +
                                       OutputTensorScalar<Scalar=U>,
          <PI as ToHost<U>>::Output: Debug + 'static,
          <I as BatchDataType>::Type: Debug + BatchSize,
          <<PI as BatchDataType>::Type as ToHost<U>>::Output: Debug +
                                                              InputTensorScalar<Scalar=U> +
                                                              OutputTensorScalar<Scalar=U> + 'static,
          for<'a> D: Device<U> + DeviceLinearOutput<'a,U,N,IO=PI,BatchIO=<PI as BatchDataType>::Type>,
          f64: From<U>,
          Self: UpdateWeight<GradientStack = <P as UpdateWeight>::GradientStack>,
          for<'a> L: LossFunction<U> + BatchLossFunctionLinear<'a,U,<PI as BatchDataType>::Type,D,N,Output=<PI as BatchDataType>::Type> {
    fn batch_train(&mut self, expected:Self::BatchOutput, input:Self::BatchInput, lossf:&L) -> Result<U, TrainingError> {
        if expected.len() != input.size() {
            return Err(TrainingError::from(SizeMismatchError(expected.len(),input.size())));
        }

        let batch_size = input.size();

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

            self.parent.batch_loss(loss,stack)?
        };

        let (_,s) = self.parent.batch_backward(loss,stack)?;

        self.parent.update_weight(s,batch_size)?;

        Ok(total_loss)
    }
}
impl<U,P,D,I,PI,const N:usize> Step for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          D: Device<U>,
          PI: Debug + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync {
    fn step(&mut self) -> Result<(),TrainingError> {
        self.step_count += 1;

        Ok(self.parent.on_step(self.step_count)?)
    }

    fn frequently_step(&mut self) -> Result<(),TrainingError> {
        self.frequently_steps += 1;
        Ok(self.parent.on_frequently_step(self.step_count,self.frequently_steps)?)
    }
}
impl<U,P,D,I,PI,const N:usize> PersistProgress<TextFilePersistence,Specialized> for LinearOutputLayer<U,P,D,I,PI,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             PersistProgress<TextFilePersistence,Specialized> + OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign +
             FromPrimitive + 'static + DataTypeInfo + FromStr + Sized,
          D: Device<U>,
          PI: Debug + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync,
          TextRecord: From<U> + From<u64>,
          ModelLoadError: From<<U as FromStr>::Err> + From<<u64 as FromStr>::Err> {
    fn load_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(),TrainingError> {
        self.parent.load_progress(persistence)?;
        let step_count:u64 = persistence.read()?;
        self.step_count = step_count as usize;
        let frequently_steps:u64 = persistence.read()?;
        self.frequently_steps = frequently_steps as usize;

        for _ in 0..self.step_count {
            self.step()?;
        }

        Ok(persistence.verify_eof()?)
    }

    fn save_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(),PersistenceError> {
        self.parent.save_progress(persistence)?;

        persistence.write_layer_start();
        persistence.write_units_start();

        persistence.write(self.step_count as u64);
        persistence.write(self.frequently_steps as u64);

        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,P,D,I,PI,const N:usize> PersistProgress<T,Linear> for LinearOutputLayer<U,P,D,I,PI,N>
    where T: LinearPersistence<U> + LinearPersistence<u64> + VerifyEof,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrainBase<PreOutput=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             PersistProgress<T,Linear> + OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync +
             Add<Output=U> + Sub<Output=U> + Div<Output=U> + AddAssign + FromPrimitive + 'static + DataTypeInfo,
          D: Device<U>,
          PI: Debug + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)?;
        let step_count:u64 = persistence.read()?;

        self.step_count = step_count as usize;

        let frequently_steps:u64 = persistence.read()?;
        self.frequently_steps = frequently_steps as usize;

        for _ in 0..self.step_count {
            self.step()?;
        }

        Ok(persistence.verify_eof()?)
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;
        persistence.write(self.step_count as u64)?;
        persistence.write(self.frequently_steps as u64)?;

        Ok(())
    }
}
