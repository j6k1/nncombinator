//! Implementation of Input layers
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::{Cons, Never, Nil};
use crate::device::Device;
use crate::device::input::DeviceInput;
use crate::error::{ModelLoadError, EvaluateError, PersistenceError, TrainingError};
use crate::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchPreTrain, BatchPreTrainBase, ForwardAll, InputTensorScalar, OnStep, InputScale, OutputTensorScalar, PartialForward, PersistProgress, PreTrain, UpdateWeight};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextRecord};

pub struct InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> {
    u:PhantomData<U>,
    o:PhantomData<O>,
    l:PhantomData<LI>,
    device:D
}
impl<U,O,LI,D> InputTensorScalar for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> {
    type Scalar = U;
}
impl<U,O,LI,D> OutputTensorScalar for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> {
    type Scalar = U;
}
impl<U,O,LI,D> InputLayer<U,O,LI,D> where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
                                          D: Device<U> {
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
impl<U,O,LI,D> Persistence<TextFilePersistence,Specialized> for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr + Sized,
          D: Device<U> {
    fn load(&mut self, _: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<T,U,O,LI,D> Persistence<T,Linear> for InputLayer<U,O,LI,D>
    where T: LinearPersistence<U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> {
    fn load(&mut self, _: &mut T) -> Result<(), ModelLoadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut T) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<U,O,LI,D> ForwardAll for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
impl<U,O,LI,D> PreTrain for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type LossInputScalar = U;
    type LossInput = LI;
    type LossOutput = LI;

    fn backward_all(&mut self, input: Self::LossInput, _:Self::OutStack)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        Ok((input,Nil))
    }
}
impl<U,O,LI,D> UpdateWeight for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type GradientStack = Nil;

    fn update_weight(&mut self, _: Self::GradientStack, _: usize) -> Result<(), TrainingError> {
        Ok(())
    }
}

impl<U,O,LI,D> BatchForwardBase for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type BatchInput = <O as BatchDataType>::Type;
    type BatchOutput = <D as DeviceInput<U,O>>::BatchOutput;
}
impl<U,O,LI,D> BatchForward for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput,TrainingError> {
        Ok(self.device.batch_forward_input(input)?)
    }
}
impl<U,O,LI,D> BatchPreTrainBase for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type BatchPreOutput = <D as DeviceInput<U,O>>::BatchOutput;
    type BatchOutStack = Cons<Nil,Self::BatchPreOutput>;
}
impl<U,O,LI,D> BatchPreTrain for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        Ok(Cons(Nil,self.device.batch_forward_input(input)?))
    }
}
impl<U,O,LI,D> BatchBackward<U> for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType,
          D: Device<U> + DeviceInput<U,O>,
          <LI as BatchDataType>::Type: Debug,
          <O as BatchDataType>::Type: Debug + 'static {
    type BatchLossInput = <LI as BatchDataType>::Type;
    type BatchLossOutput = <LI as BatchDataType>::Type;

    fn batch_backward(&mut self, input: Self::BatchLossInput, _: Self::BatchOutStack)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        Ok((input,Nil))
    }
}
impl<U,O,LI,D> OnStep for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static, D: Device<U> {
    fn on_step(&mut self, _: usize) -> Result<(), TrainingError> {
        Ok(())
    }
    fn on_frequently_step(&mut self, _: usize, _: usize) -> Result<(), TrainingError> {
        Ok(())
    }
}
impl<U,O,LI,D> PersistProgress<TextFilePersistence,Specialized> for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr + Sized,
          D: Device<U>,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load_progress(&mut self, _: &mut TextFilePersistence) -> Result<(), TrainingError> {
        Ok(())
    }

    fn save_progress(&mut self, _: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<T,U,O,LI,D> PersistProgress<T,Linear> for InputLayer<U,O,LI,D>
    where T: LinearPersistence<U>, U: Default + Clone + Copy + Debug + Send + Sync + 'static, D: Device<U> {
    fn load_progress(&mut self, _: &mut T) -> Result<(), TrainingError> {
        Ok(())
    }

    fn save_progress(&mut self, _: &mut T) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<U,O,LI,D> InputScale for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          D: Device<U> {
    fn scale_mean(&self) -> f32 {
        1.
    }
}
pub struct DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> {
    u:PhantomData<U>,
    o:PhantomData<O>,
    di:PhantomData<DI>,
    po:PhantomData<PO>,
    l:PhantomData<LI>,
    device:D
}
impl<U,O,DI,PO,LI,D> DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> {
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
impl<U,O,DI,PO,LI,D> InputTensorScalar for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> {
    type Scalar = U;
}
impl<U,O,DI,PO,LI,D> OutputTensorScalar for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> {
    type Scalar = U;
}
impl<U,O,DI,PO,LI,D> Persistence<TextFilePersistence,Specialized> for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr + Sized,
          D: Device<U> {
    fn load(&mut self, _: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<T,U,O,DI,PO,LI,D> Persistence<T,Linear> for DiffInputLayer<U,O,DI,PO,LI,D>
    where T: LinearPersistence<U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static, D: Device<U> {
    fn load(&mut self, _: &mut T) -> Result<(), ModelLoadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut T) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<U,O,DI,PO,LI,D> ForwardAll for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type PartialInput = Never;
    type PartialOutput = Never;
    type DiffInput = DI;
    /// When implementing diff application,
    /// do so in a lower-level layer and avoid calling this method of the `DiffInputLayer`.
    fn partial_forward(&self, _: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        // Since the argument is an enum type without a variant,
        // it cannot be instantiated, so this code will never be executed.
        unreachable!()
    }

    /// When implementing diff application,
    /// do so in a lower-level layer and avoid calling this method of the `DiffInputLayer`.
    fn partial_forward_by_diff(&self, _: Self::DiffInput, _: &Self::PartialInput)
        -> Result<Self::PartialOutput, EvaluateError> {
        // Since the argument is an enum type without a variant,
        // it cannot be instantiated, so this code will never be executed.
        unreachable!()
    }
}
impl<U,O,DI,PO,LI,D> PreTrain for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
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
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type LossInputScalar = U;
    type LossInput = LI;
    type LossOutput = LI;

    fn backward_all(&mut self, input: Self::LossInput, _:Self::OutStack)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        Ok((input,Nil))
    }
}
impl<U,O,DI,PO,LI,D> UpdateWeight for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type GradientStack = Nil;

    fn update_weight(&mut self, _: Self::GradientStack, _: usize) -> Result<(), TrainingError> {
        Ok(())
    }
}
impl<U,O,DI,PO,LI,D> OnStep for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static, D: Device<U> {
    fn on_step(&mut self, _: usize) -> Result<(), TrainingError> {
        Ok(())
    }
    fn on_frequently_step(&mut self, _: usize, _: usize) -> Result<(), TrainingError> {
        Ok(())
    }
}
impl<U,O,DI,PO,LI,D> PersistProgress<TextFilePersistence,Specialized> for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr + Sized,
          D: Device<U>,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load_progress(&mut self, _: &mut TextFilePersistence) -> Result<(), TrainingError> {
        Ok(())
    }

    fn save_progress(&mut self, _: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<T,U,O,DI,PO,LI,D> PersistProgress<T,Linear> for DiffInputLayer<U,O,DI,PO,LI,D>
    where T: LinearPersistence<U>, U: Default + Clone + Copy + Debug + Send + Sync + 'static, D: Device<U> {
    fn load_progress(&mut self, _: &mut T) -> Result<(), TrainingError> {
        Ok(())
    }

    fn save_progress(&mut self, _: &mut T) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<U,O,DI,PO,LI,D> InputScale for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> {
}
