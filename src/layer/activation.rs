//! Implementation of Activation layers

use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::{Cons, Stack};
use crate::cuda::DataTypeInfo;
use crate::device::activation::DeviceActivation;
use crate::device::Device;
use crate::error::{ModelLoadError, EvaluateError, PersistenceError, TrainingError};
use crate::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, ContinueForward, Forward, ForwardAll, ForwardDiff, Loss, PartialForward, PreTrain, UpdateWeight, OnStep, PersistProgress, InputTensorScalar, OutputTensorScalar};
use crate::lossfunction::LossFunction;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextRecord};

/// Activation layer Implementation
pub struct ActivationLayer<U,P,A,I,PI,D,const N:usize>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             PreTrain + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          I: Debug + Send + Sync {
    parent:P,
    f:A,
    device:D,
    u:PhantomData<U>,
    i:PhantomData<I>,
    pi:PhantomData<PI>,
}
impl<U,P,A,I,PI,D,const N:usize> ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             PreTrain + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync {
    /// Create and return an instance of ActivationLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `f` - Activation Function
    /// * `device` - Device object used for neural network computation
    pub fn new(parent:P,f:A,device:&D) -> ActivationLayer<U,P,A,I,PI,D,N> {
        ActivationLayer {
            parent:parent,
            f:f,
            device:device.clone(),
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            pi:PhantomData::<PI>,
        }
    }
}
impl<U,P,A,I,PI,D,const N:usize> InputTensorScalar for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
          PreTrain + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync {
    type Scalar = U;
}
impl<U,P,A,I,PI,D,const N:usize> OutputTensorScalar for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             PreTrain + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync {
    type Scalar = U;
}
impl<U,P,A,I,PI,D,const N:usize> Persistence<U,TextFilePersistence,Specialized> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + Persistence<U,TextFilePersistence,Specialized> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> + PreTrain + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static + std::str::FromStr,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write_layer_start();
        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,P,A,I,PI,D,const N:usize> Persistence<U,T,Linear> for ActivationLayer<U,P,A,I,PI,D,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + Persistence<U,T,Linear> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> + PreTrain + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)
    }
}
impl<U,P,A,I,PI,D,const N:usize> ForwardAll for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             PreTrain<PreOutput=PI> + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = PI;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,P,A,I,PI,D,const N:usize> Forward<PI,Result<PI,EvaluateError>> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             PreTrain<PreOutput=PI> + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U> + 'static,
          I: Debug + Send + Sync {
    fn forward(&self, input: &PI) -> Result<PI,EvaluateError> {
        self.device.apply(&self.f, &input)
    }
}
impl<U,P,A,I,PI,D,const N:usize> PreTrain for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             PreTrain<PreOutput=PI> + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          I: Debug + Send + Sync {
    type PreOutput = PI;
    type OutStack = Cons<<P as PreTrain>::OutStack, Self::PreOutput>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r))?;

        Ok(Cons(r,u))
    }
}
impl<U,P,A,I,PI,D,const N:usize> BackwardAll<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          I: Debug + Send + Sync {
    type LossInputScalar = <P as OutputTensorScalar>::Scalar;
    type LossInput = PI;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U> + LossFunction<Self::LossInputScalar>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s,o) = stack.pop();
        let input = if self.is_canonical_link(lossf) {
            input
        } else {
            s.map(|u| self.device.derive(&self.f, &o, &input, u))?.into()
        };

        self.parent.backward_all(input.into(), s, lossf)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.device.is_canonical_link(&self.f,l)
    }
}
impl<U,P,A,I,PI,D,const N:usize> UpdateWeight for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> + UpdateWeight + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType,
          I: Debug + Send + Sync {
    type GradientStack = <P as UpdateWeight>::GradientStack;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        Ok(self.parent.update_weight(stack,batch_size)?)
    }
}
impl<U,P,A,I,PI,D,const N:usize> PartialForward for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> + PartialForward + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          I: Debug + Send + Sync {
    type PartialInput = <P as PartialForward>::PartialInput;
    type PartialOutput = <P as PartialForward>::PartialOutput;
    type DiffInput = <P as PartialForward>::DiffInput;
    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward(input)?)
    }

    fn partial_forward_by_diff(&self, input:Self::DiffInput, partial_input:&Self::PartialInput) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward_by_diff(input, partial_input)?)
    }
}
impl<U,P,A,I,PI,D,const N:usize> ForwardDiff for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             PartialForward + ForwardDiff + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          I: Debug + Send + Sync {
    fn forward_diff(&self, input: Self::DiffInput, partial_input:&Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_diff(input, partial_input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,P,A,I,PI,D,const N:usize> ContinueForward for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             PartialForward + ContinueForward + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
      U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
      D: Device<U> + DeviceActivation<U,PI,A,N>,
      PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
      I: Debug + Send + Sync {
    fn continue_forward<'a>(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.continue_forward(input)?;

        Ok(self.forward(&input)?)
    }
}
impl<U,P,A,I,PI,D,const N:usize> Loss<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          I: Debug + Send + Sync {
    fn loss<L: LossFunction<U> + LossFunction<Self::LossInputScalar>>(&mut self, loss: Self::LossInput, _:&L, stack: Self::OutStack) -> Result<(Self::OutStack, Self::LossInput), TrainingError> {
        let (s,o) = stack.pop();

        let r = s.map(|u| self.device.derive(&self.f, &o, &loss, u))?;

        Ok((Cons(s,o),r.into()))
    }
}
impl<U,P,A,I,PI,D,const N:usize> BatchForwardBase for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type> + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <PI as BatchDataType>::Type;
}
impl<U,P,A,I,PI,D,const N:usize> BatchForward for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type> + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_apply(&self.f,&input)?)
    }
}
impl<U,P,A,I,PI,D,const N:usize> BatchPreTrainBase for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain + BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type> + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    type BatchPreOutput = <PI as BatchDataType>::Type;
    type BatchOutStack = Cons<<P as BatchPreTrainBase>::BatchOutStack, Self::BatchPreOutput>;
}
impl<U,P,A,I,PI,D,const N:usize> BatchPreTrain for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain + BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type> + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| {
            self.device.batch_apply(&self.f,input)
        })?;

        Ok(Cons(r,u))
    }
}
impl<U,P,A,I,PI,D,const N:usize> BatchBackward<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain + BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type> + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    type BatchLossInput = <PI as BatchDataType>::Type;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U> + LossFunction<Self::LossInputScalar>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s,o) = stack.pop();
        let input = if self.is_canonical_link(lossf) {
            input
        } else {
            s.map(|u| {
                self.device.batch_derive(&self.f, &o, &input, u)
            })?
        };

        self.parent.batch_backward(input, s, lossf)
    }
}
impl<U,P,A,I,PI,D,const N:usize> BatchLoss<U> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: PreTrain<PreOutput=PI> + ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> + BatchPreTrain +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type> + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          I: Debug + Send + Sync + BatchDataType,
          <PI as BatchDataType>::Type: Debug,
          <I as BatchDataType>::Type: Debug {
    fn batch_loss<L: LossFunction<U>>(&self, loss: Self::BatchLossInput, _: &L, stack: Self::BatchOutStack) -> Result<(Self::BatchOutStack, Self::BatchLossInput), TrainingError> {
        let (s,o) = stack.pop();

        let r = s.map(|u| {
            self.device.batch_derive(&self.f, &o, &loss, u)
        })?;

        Ok((Cons(s,o),r))
    }
}

impl<U,P,A,I,PI,D,const N:usize> OnStep for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> +
             PreTrain + OnStep + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          I: Debug + Send + Sync {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        Ok(self.parent.on_step(step)?)
    }

    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        Ok(self.parent.on_frequently_step(step,frequently_step)?)
    }
}
impl<U,P,A,I,PI,D,const N:usize> PersistProgress<TextFilePersistence,Specialized> for ActivationLayer<U,P,A,I,PI,D,N>
    where P: ForwardAll<Input=I,Output=PI> +
             PersistProgress<TextFilePersistence,Specialized> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> + PreTrain + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static + std::str::FromStr,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          I: Debug + Send + Sync,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)
    }

    fn save_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;
        persistence.write_layer_start();
        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,P,A,I,PI,D,const N:usize> PersistProgress<T,Linear> for ActivationLayer<U,P,A,I,PI,D,N>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> +
             PersistProgress<T,Linear> +
             BackwardAll<U,LossInput=PI,LossInputScalar=<P as OutputTensorScalar>::Scalar> + PreTrain + OutputTensorScalar<Scalar=U> + InputTensorScalar<Scalar=U>,
          U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          D: Device<U> + DeviceActivation<U,PI,A,N>,
          PI: Debug + BatchDataType + 'static,
          I: Debug + Send + Sync {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)
    }
}

