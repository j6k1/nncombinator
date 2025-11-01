//! Implementation of Input layers
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::{Cons, Nil};
use crate::device::Device;
use crate::device::input::DeviceInput;
use crate::error::{ConfigReadError, EvaluateError, PersistenceError, TrainingError};
use crate::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, ForwardAll, ForwardDiff, Loss, OnStep, PartialForward, PreTrain, UpdateWeight};
use crate::lossfunction::LossFunction;
use crate::ope::UnitValue;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence};

pub struct InputLayer<U,O,LI,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    o:PhantomData<O>,
    l:PhantomData<LI>,
    device:D
}
impl<U,O,LI,D> InputLayer<U,O,LI,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of InputLayer
    pub fn new(device:&D) -> InputLayer<U,O,LI,D> {
        InputLayer {
            u:PhantomData::<U>,
            o:PhantomData::<O>,
            l:PhantomData::<LI>,
            device:device.clone()
        }
    }
}
impl<U,O,LI,D> Persistence<U,TextFilePersistence<U>,Specialized> for InputLayer<U,O,LI,D>
    where U: UnitValue<U> + FromStr + Sized, D: Device<U> {
    fn load(&mut self, _: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<T,U,O,LI,D> Persistence<U,T,Linear> for InputLayer<U,O,LI,D>
    where T: LinearPersistence<U>, U: UnitValue<U>, D: Device<U> {
    fn load(&mut self, _: &mut T) -> Result<(),ConfigReadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut T) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<U,O,LI,D> ForwardAll for InputLayer<U,O,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type Input = O;
    type Output = <D as DeviceInput<U,O>>::Output;
    fn forward_all(&self, input:Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(self.device.forward_input(input)?)
    }
}
impl<U,O,LI,D> PreTrain<U> for InputLayer<U,O,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type PreOutput = <D as DeviceInput<U,O>>::Output;
    type OutStack = Cons<Nil,Self::PreOutput>;

    fn pre_train(&self, input:Self::Input) -> Result<Self::OutStack, EvaluateError> {
        Ok(Cons(Nil,self.device.forward_input(input)?))
    }
}
impl<U,O,LI,D> BackwardAll<U> for InputLayer<U,O,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type LossInput = LI;
    type LossOutput = LI;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, _:Self::OutStack, _:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        Ok((input,Nil))
    }
}
impl<U,O,LI,D> UpdateWeight<U> for InputLayer<U,O,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type GradientStack = Nil;

    fn update_weight(&mut self, _: Self::GradientStack) -> Result<(), TrainingError> {
        Ok(())
    }
}
impl<U,O,LI,D> Loss<U> for InputLayer<U,O,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {}
impl<U,O,LI,D> BatchForwardBase for InputLayer<U,O,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type BatchInput = <O as BatchDataType>::Type;
    type BatchOutput = <D as DeviceInput<U,O>>::BatchOutput;
}
impl<U,O,LI,D> BatchForward for InputLayer<U,O,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput,TrainingError> {
        Ok(self.device.batch_forward_input(input)?)
    }
}
impl<U,O,LI,D> BatchPreTrainBase<U> for InputLayer<U,O,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type BatchPreOutput = <D as DeviceInput<U,O>>::BatchOutput;
    type BatchOutStack = Cons<Nil,Self::BatchPreOutput>;
}
impl<U,O,LI,D> BatchPreTrain<U> for InputLayer<U,O,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        Ok(Cons(Nil,self.device.batch_forward_input(input)?))
    }
}
impl<U,O,LI,D> BatchBackward<U> for InputLayer<U,O,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType,
          D: Device<U> + DeviceInput<U,O>,
          <LI as BatchDataType>::Type: Debug,
          <O as BatchDataType>::Type: Debug + 'static {
    type BatchLossInput = <LI as BatchDataType>::Type;
    type BatchLossOutput = <LI as BatchDataType>::Type;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, _: Self::BatchOutStack, _: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        Ok((input,Nil))
    }
}
impl<U,O,LI,D> BatchLoss<U> for InputLayer<U,O,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType,
          D: Device<U> + DeviceInput<U,O>,
          <LI as BatchDataType>::Type: Debug,
          <O as BatchDataType>::Type: Debug + 'static {
}
impl<U,O,LI,D> OnStep for InputLayer<U,O,LI,D> where U: UnitValue<U>, D: Device<U> {
    fn on_step(&mut self, _: usize) -> Result<(), TrainingError> {
        Ok(())
    }
}
pub struct DiffInputLayer<U,O,DI,PO,LI,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    o:PhantomData<O>,
    di:PhantomData<DI>,
    po:PhantomData<PO>,
    l:PhantomData<LI>,
    device:D
}
impl<U,O,DI,PO,LI,D> DiffInputLayer<U,O,DI,PO,LI,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of InputLayer
    pub fn new(device:&D) -> DiffInputLayer<U,O,DI,PO,LI,D> {
        DiffInputLayer {
            u:PhantomData::<U>,
            o:PhantomData::<O>,
            di:PhantomData::<DI>,
            po:PhantomData::<PO>,
            l:PhantomData::<LI>,
            device:device.clone()
        }
    }
}
impl<U,O,DI,PO,LI,D> Persistence<U,TextFilePersistence<U>,Specialized> for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: UnitValue<U> + FromStr + Sized, D: Device<U> {
    fn load(&mut self, _: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<T,U,O,DI,PO,LI,D> Persistence<U,T,Linear> for DiffInputLayer<U,O,DI,PO,LI,D>
    where T: LinearPersistence<U>, U: UnitValue<U>, D: Device<U> {
    fn load(&mut self, _: &mut T) -> Result<(),ConfigReadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut T) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<U,O,DI,PO,LI,D> ForwardAll for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
        DI: Debug,
        PO: Debug,
        LI: Debug,
        D: Device<U> + DeviceInput<U,O>,
        <O as BatchDataType>::Type: Debug + 'static {
    type Input = O;
    type Output = <D as DeviceInput<U,O>>::Output;
    fn forward_all(&self, input:Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(self.device.forward_input(input)?)
    }
}
impl<U,O,DI,PO,LI,D> PartialForward for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type PartialOutput = <D as DeviceInput<U,O>>::Output;
    type PartialOutputByDiff = DI;
    type DiffOutput = DI;
    type DiffInput = DI;
    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.device.forward_input(input)?)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput) -> Result<Self::PartialOutputByDiff, EvaluateError> {
        Ok(input)
    }
}
impl<U,O,DI,PO,LI,D> ForwardDiff for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    fn forward_diff(&self, input: Self::DiffInput) -> Result<Self::DiffOutput, EvaluateError> {
        Ok(input)
    }
}
impl<U,O,DI,PO,LI,D> PreTrain<U> for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type PreOutput = <D as DeviceInput<U,O>>::Output;
    type OutStack = Cons<Nil,Self::PreOutput>;

    fn pre_train(&self, input:Self::Input) -> Result<Self::OutStack, EvaluateError> {
        Ok(Cons(Nil,self.device.forward_input(input)?))
    }
}
impl<U,O,DI,PO,LI,D> BackwardAll<U> for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type LossInput = LI;
    type LossOutput = LI;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, _:Self::OutStack, _:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        Ok((input,Nil))
    }
}
impl<U,O,DI,PO,LI,D> UpdateWeight<U> for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type GradientStack = Nil;

    fn update_weight(&mut self, _: Self::GradientStack) -> Result<(), TrainingError> {
        Ok(())
    }
}
impl<U,O,DI,PO,LI,D> Loss<U> for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: UnitValue<U>,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {}
impl<U,O,DI,PO,LI,D> OnStep for DiffInputLayer<U,O,DI,PO,LI,D> where U: UnitValue<U>, D: Device<U> {
    fn on_step(&mut self, _: usize) -> Result<(), TrainingError> {
        Ok(())
    }
}