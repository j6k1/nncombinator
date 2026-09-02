//! Implementation of Input layers
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::{Cons, Never, Nil};
use crate::arr::{Ones};
use crate::device::bridge::DeviceBridge;
use crate::device::Device;
use crate::device::input::DeviceInput;
use crate::device::scale::DeviceScale;
use crate::error::{ModelLoadError, EvaluateError, PersistenceError, TrainingError};
use crate::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchPreTrain, BatchPreTrainBase, BatchSize, ForwardAll, InputTensorScalar, OnStep, InputScale, OutputTensorScalar, PartialForward, PersistProgress, PreTrain, UpdateWeight, InputMax, PreTrainBase, Bridge, BatchBridge, BackwardBase, BatchBackwardBase, BridgeBase, BridgeRepr, BatchBridgeRepr};
use crate::mapper::{BatchIdentityMapper, BatchInternalReprMapper, IdentityMapper, InternalReprMapper};
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
impl<U,O,LI,D> PreTrainBase for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type PreOutput = <D as DeviceInput<U, O>>::Output;
    type OutStack = Cons<Nil, Self::PreOutput>;
}
impl<U,O,LI,D> PreTrain for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {

    fn pre_train(&self, input:Self::Input) -> Result<Self::OutStack, EvaluateError> {
        Ok(Cons(Nil,self.device.forward_input(input)?))
    }
}
impl<U,O,LI,D> BackwardBase for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type LossInputScalar = U;
    type LossInput = LI;
}
impl<U,O,LI,D> BackwardAll<U> for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
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
impl<U,O,LI,D> BatchBackwardBase for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType,
          D: Device<U> + DeviceInput<U,O>,
          <LI as BatchDataType>::Type: Debug,
          <O as BatchDataType>::Type: Debug + 'static {
    type BatchLossInput = <LI as BatchDataType>::Type;
    type BatchLossOutput = <LI as BatchDataType>::Type;
}
impl<U,O,LI,D> BatchBackward<U> for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType,
          D: Device<U> + DeviceInput<U,O>,
          <LI as BatchDataType>::Type: Debug,
          <O as BatchDataType>::Type: Debug + 'static {
    fn batch_backward(&mut self, input: Self::BatchLossInput, _: Self::BatchOutStack)
        -> Result<(<Self as BatchBackwardBase>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
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
    fn input_scale(&self) -> f32 {
        1.
    }
}
impl<U,O,LI,D> BridgeBase for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    type UseDevice = D;
    type RealScale = <D as DeviceInput<U,O>>::Output;
    type RealOutput = <D as DeviceInput<U,O>>::Output;
    type SourceInput = <D as DeviceInput<U,O>>::Output;
}
impl<U,O,LI,D> Bridge for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    type RealMapper<'a> = IdentityMapper<'a,<D as DeviceInput<U,O>>::Output,Self::UseDevice> where Self: 'a;

    fn as_real<'a>(&'a self, input: &'a <D as DeviceInput<U,O>>::Output) -> Result<Self::RealMapper<'a>,EvaluateError> where Self: 'a {
        Ok(IdentityMapper::new(input))
    }
}
impl<U,O,LI,D> BridgeRepr for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    type RepresentationOutput = <D as DeviceInput<U,O>>::Output;
    type ReprMapper<'a> = IdentityMapper<'a,<D as DeviceInput<U,O>>::Output,Self::UseDevice> where Self: 'a;

    fn as_repr<'a>(&'a self, input: &'a <D as DeviceInput<U,O>>::Output) -> Result<Self::ReprMapper<'a>,EvaluateError> where Self: 'a {
        Ok(IdentityMapper::new(input))
    }
}
impl<U,O,LI,D> BatchBridge for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O,BatchOutput=<<D as DeviceInput<U,O>>::Output as BatchDataType>::Type>,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static,
          <O as BatchDataType>::Type: Debug + 'static,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static {
    type BatchRealMapper<'a> = BatchIdentityMapper<'a,<D as DeviceInput<U,O>>::Output,Self::UseDevice> where Self: 'a;

    fn batch_as_real<'a>(&'a self, input: &'a <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type)
                         -> Result<Self::BatchRealMapper<'a>,EvaluateError> where Self: 'a {
        Ok(BatchIdentityMapper::new(input))
    }
}
impl<U,O,LI,D> BatchBridgeRepr for InputLayer<U,O,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O,BatchOutput=<<D as DeviceInput<U,O>>::Output as BatchDataType>::Type>,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static,
          <O as BatchDataType>::Type: Debug + 'static,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static {
    type BatchReprMapper<'a> = BatchIdentityMapper<'a,<D as DeviceInput<U,O>>::Output,Self::UseDevice> where Self: 'a;

    fn batch_as_repr<'a>(&'a self, input: &'a <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type)
                         -> Result<Self::BatchReprMapper<'a>,EvaluateError> where Self: 'a {
        Ok(BatchIdentityMapper::new(input))
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
impl<U,O,DI,PO,LI,D> PreTrainBase for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type PreOutput = <D as DeviceInput<U, O>>::Output;
    type OutStack = Cons<Nil, Self::PreOutput>;
}
impl<U,O,DI,PO,LI,D> PreTrain for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {

    fn pre_train(&self, input:Self::Input) -> Result<Self::OutStack, EvaluateError> {
        Ok(Cons(Nil,self.device.forward_input(input)?))
    }
}
impl<U,O,DI,PO,LI,D> BackwardBase for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type LossInputScalar = U;
    type LossInput = LI;
}
impl<U,O,DI,PO,LI,D> BackwardAll<U> for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
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
impl<U,O,DI,PO,LI,D> BridgeBase for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    type UseDevice = D;
    type RealScale = <D as DeviceInput<U,O>>::Output;
    type RealOutput = <D as DeviceInput<U,O>>::Output;
    type SourceInput = <D as DeviceInput<U,O>>::Output;
}
impl<U,O,DI,PO,LI,D> Bridge for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    type RealMapper<'a> = IdentityMapper<'a,<D as DeviceInput<U,O>>::Output,Self::UseDevice> where Self: 'a;

    fn as_real<'a>(&'a self, input: &'a <Self as ForwardAll>::Output) -> Result<Self::RealMapper<'a>,EvaluateError> where Self: 'a {
        Ok(IdentityMapper::new(input))
    }
}
impl<U,O,DI,PO,LI,D> BridgeRepr for DiffInputLayer<U,O,DI,PO,LI,D>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          DI: Debug,
          PO: Debug,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    type RepresentationOutput = <D as DeviceInput<U,O>>::Output;
    type ReprMapper<'a> = IdentityMapper<'a,<D as DeviceInput<U,O>>::Output,Self::UseDevice> where Self: 'a;

    fn as_repr<'a>(&'a self, input: &'a <Self as ForwardAll>::Output) -> Result<Self::ReprMapper<'a>,EvaluateError> where Self: 'a {
        Ok(IdentityMapper::new(input))
    }
}
pub struct QuantizedInputLayer<U,O,LI,D,const M:usize,const N:usize>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    u:PhantomData<U>,
    o:PhantomData<O>,
    l:PhantomData<LI>,
    device:D
}
impl<U,O,LI,D,const M: usize,const N:usize> InputTensorScalar for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    type Scalar = U;
}
impl<U,O,LI,D,const M: usize,const N:usize> OutputTensorScalar for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    type Scalar = U;
}
impl<U,O,LI,D,const M: usize,const N:usize> QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    /// Create an instance of QuantizedInputLayer
    pub fn new(device:&D) -> Result<QuantizedInputLayer<U,O,LI,D,M,N>,EvaluateError> {
        Ok(QuantizedInputLayer {
            u:PhantomData::<U>,
            o:PhantomData::<O>,
            l:PhantomData::<LI>,
            device:device.clone()
        })
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> Persistence<TextFilePersistence,Specialized> for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr + Sized,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    fn load(&mut self, _: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<T,U,O,LI,D,const M: usize,const N: usize> Persistence<T,Linear> for QuantizedInputLayer<U,O,LI,D,M,N>
    where T: LinearPersistence<U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    fn load(&mut self, _: &mut T) -> Result<(), ModelLoadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut T) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> ForwardAll for QuantizedInputLayer<U,O,LI,D,M,N>
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
impl<U,O,LI,D,const M: usize,const N:usize> PreTrainBase for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type PreOutput = <D as DeviceInput<U, O>>::Output;
    type OutStack = Cons<Nil, Self::PreOutput>;
}
impl<U,O,LI,D,const M: usize,const N:usize> PreTrain for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    fn pre_train(&self, input:Self::Input) -> Result<Self::OutStack, EvaluateError> {
        Ok(Cons(Nil,self.device.forward_input(input)?))
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> BackwardBase for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O> + DeviceInput<f32,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type LossInputScalar = f32;
    type LossInput = LI;
}
impl<U,O,LI,D,const M: usize,const N:usize> BackwardAll<f32> for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O> + DeviceInput<f32,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type LossOutput = LI;

    fn backward_all(&mut self, input: Self::LossInput, _:Self::OutStack)
                    -> Result<(<Self as BackwardAll<f32>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        Ok((input,Nil))
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> UpdateWeight for QuantizedInputLayer<U,O,LI,D,M,N>
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

impl<U,O,LI,D,const M: usize,const N:usize> BatchForwardBase for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type BatchInput = <O as BatchDataType>::Type;
    type BatchOutput = <D as DeviceInput<U,O>>::BatchOutput;
}
impl<U,O,LI,D,const M: usize,const N:usize> BatchForward for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput,TrainingError> {
        Ok(self.device.batch_forward_input(input)?)
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> BatchPreTrainBase for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    type BatchPreOutput = <D as DeviceInput<U,O>>::BatchOutput;
    type BatchOutStack = Cons<Nil,Self::BatchPreOutput>;
}
impl<U,O,LI,D,const M: usize,const N:usize> BatchPreTrain for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug,
          D: Device<U> + DeviceInput<U,O>,
          <O as BatchDataType>::Type: Debug + 'static {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        Ok(Cons(Nil,self.device.batch_forward_input(input)?))
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> BatchBackwardBase for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType,
          D: Device<U> + DeviceInput<U,O> + DeviceInput<f32,O>,
          <LI as BatchDataType>::Type: Debug,
          <O as BatchDataType>::Type: Debug + 'static {
    type BatchLossInput = <LI as BatchDataType>::Type;
    type BatchLossOutput = <LI as BatchDataType>::Type;
}
impl<U,O,LI,D,const M: usize,const N:usize> BatchBackward<f32> for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType,
          D: Device<U> + DeviceInput<U,O> + DeviceInput<f32,O>,
          <LI as BatchDataType>::Type: Debug,
          <O as BatchDataType>::Type: Debug + 'static {
    fn batch_backward(&mut self, input: Self::BatchLossInput, _: Self::BatchOutStack)
                      -> Result<(<Self as BatchBackwardBase>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        Ok((input,Nil))
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> OnStep for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    fn on_step(&mut self, _: usize) -> Result<(), TrainingError> {
        Ok(())
    }
    fn on_frequently_step(&mut self, _: usize, _: usize) -> Result<(), TrainingError> {
        Ok(())
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> PersistProgress<TextFilePersistence,Specialized> for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr + Sized,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load_progress(&mut self, _: &mut TextFilePersistence) -> Result<(), TrainingError> {
        Ok(())
    }

    fn save_progress(&mut self, _: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<T,U,O,LI,D,const M: usize,const N: usize> PersistProgress<T,Linear> for QuantizedInputLayer<U,O,LI,D,M,N>
    where T: LinearPersistence<U>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    fn load_progress(&mut self, _: &mut T) -> Result<(), TrainingError> {
        Ok(())
    }

    fn save_progress(&mut self, _: &mut T) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> InputScale for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    fn input_scale(&self) -> f32 {
        M as f32
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> BridgeBase for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType + 'static,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static,
          <LI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static {
    type UseDevice = D;
    type RealScale = LI;
    type RealOutput = LI;
    type SourceInput = <D as DeviceInput<U, O>>::Output;
}
impl<U,O,LI,D,const M: usize,const N:usize> Bridge for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType + 'static,
          D: Device<U> + DeviceInput<U,O> + DeviceBridge<U,f32,<D as DeviceInput<U,O>>::Output,LI>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static,
          <LI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static {
    type RealMapper<'a> = InternalReprMapper<'a,U,f32,<D as DeviceInput<U,O>>::Output,LI,D,N>
        where Self: 'a;
    fn as_real<'a>(&'a self, input: &'a <D as DeviceInput<U,O>>::Output) -> Result<Self::RealMapper<'a>, EvaluateError>
        where Self: 'a {
        InternalReprMapper::new(&self.device,input)
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> BridgeRepr for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType + 'static,
          D: Device<U> + DeviceInput<U,O> + DeviceBridge<f32,U,LI,<D as DeviceInput<U,O>>::Output>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static,
          <LI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static {
    type RepresentationOutput = <D as DeviceInput<U, O>>::Output;
    type ReprMapper<'a> = InternalReprMapper<'a,f32,U,LI,<D as DeviceInput<U,O>>::Output,D,N>
        where Self: 'a;
    fn as_repr<'a>(&'a self, input: &'a LI) -> Result<Self::ReprMapper<'a>, EvaluateError>
        where Self: 'a {
        InternalReprMapper::new(&self.device,input)
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> BatchBridge for QuantizedInputLayer<U,O,LI,D,M,N>
        where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType + 'static,
          D: Device<U> + DeviceInput<U,O> + DeviceBridge<U,f32,<D as DeviceInput<U,O>>::Output,LI> +
              DeviceScale<f32,LI,N>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static,
          <LI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static {
    type BatchRealMapper<'a> = BatchInternalReprMapper<'a,U,f32,<D as DeviceInput<U,O>>::Output,LI,D,N>
        where Self: 'a;
    fn batch_as_real<'a>(&'a self, input: &'a <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type)
                         -> Result<Self::BatchRealMapper<'a>,EvaluateError>
        where Self: 'a {
        Ok(BatchInternalReprMapper::new(&self.device,input)?)
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> BatchBridgeRepr for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          LI: Debug + BatchDataType + 'static,
          D: Device<U> + DeviceInput<U,O> + DeviceBridge<f32,U,LI,<D as DeviceInput<U,O>>::Output>,
          O: Debug + BatchDataType + Ones + 'static,
          <O as BatchDataType>::Type: Debug + 'static,
          <LI as BatchDataType>::Type: Debug + BatchSize + 'static,
          <D as DeviceInput<U,O>>::Output: Debug + BatchDataType + 'static,
          <<D as DeviceInput<U,O>>::Output as BatchDataType>::Type: Debug + BatchSize + 'static {
    type BatchReprMapper<'a> = BatchInternalReprMapper<'a,f32,U,LI,<D as DeviceInput<U,O>>::Output,D,N>
        where Self: 'a;
    fn batch_as_repr<'a>(&'a self, input: &'a <LI as BatchDataType>::Type)
        -> Result<Self::BatchReprMapper<'a>,EvaluateError> where Self: 'a {
        Ok(BatchInternalReprMapper::new(&self.device,input)?)
    }
}
impl<U,O,LI,D,const M: usize,const N:usize> InputMax for QuantizedInputLayer<U,O,LI,D,M,N>
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          O: Debug + BatchDataType + Send + Sync + 'static,
          D: Device<U> + DeviceInput<U,O>,
          O: Debug + BatchDataType + 'static,
          <O as BatchDataType>::Type: Debug + 'static {
    type Scalar = f32;
    fn input_max(&self) -> f32 {
        M as f32
    }
}
