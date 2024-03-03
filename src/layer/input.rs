//! Implementation of Input layers
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::SerializedVec;
use crate::{Cons, Nil};
use crate::error::{ConfigReadError, EvaluateError, PersistenceError, TrainingError};
use crate::layer::{BackwardAll, BatchBackward, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, ForwardAll, Loss, PreTrain, UpdateWeight};
use crate::lossfunction::LossFunction;
use crate::ope::UnitValue;
use crate::optimizer::Optimizer;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence};

pub struct InputLayer<U,O,LI> where U: UnitValue<U> {
    u:PhantomData<U>,
    o:PhantomData<O>,
    l:PhantomData<LI>
}
impl<U,O,LI> InputLayer<U,O,LI> where U: UnitValue<U> {
    /// Create an instance of InputLayer
    pub fn new() -> InputLayer<U,O,LI> {
        InputLayer {
            u:PhantomData::<U>,
            o:PhantomData::<O>,
            l:PhantomData::<LI>
        }
    }
}
impl<U,O,LI> Persistence<U,TextFilePersistence<U>,Specialized> for InputLayer<U,O,LI>
    where U: UnitValue<U> + FromStr + Sized {
    fn load(&mut self, _: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<T,U,O,LI> Persistence<U,T,Linear> for InputLayer<U,O,LI>
    where T: LinearPersistence<U>, U: UnitValue<U> {
    fn load(&mut self, _: &mut T) -> Result<(),ConfigReadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut T) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<U,O,LI> ForwardAll for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type Input = O;
    type Output = O;
    fn forward_all(&self, input:Self::Input) -> Result<Self::Output, EvaluateError> {
        Ok(input)
    }
}
impl<U,O,LI> PreTrain<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type OutStack = Cons<Nil,Self::Output>;

    fn pre_train(&self, input:Self::Input) -> Result<Self::OutStack, EvaluateError> {
        Ok(Cons(Nil,input))
    }
}
impl<U,O,LI> BackwardAll<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type LossInput = LI;
    type LossOutput = LI;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, _:Self::OutStack, _:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        Ok((input,Nil))
    }
}
impl<U,O,LI> UpdateWeight<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type GradientStack = Nil;

    fn update_weight<OP: Optimizer<U>>(&mut self, _: Self::GradientStack, _: &mut OP) -> Result<(), TrainingError> {
        Ok(())
    }
}
impl<U,O,LI> Loss<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {}
impl<U,O,LI> BatchForwardBase for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type BatchInput = SerializedVec<U,O>;
    type BatchOutput = SerializedVec<U,O>;
}
impl<U,O,LI> BatchForward for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput,TrainingError> {
        Ok(input)
    }
}
impl<U,O,LI> BatchPreTrainBase<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type BatchOutStack = Cons<Nil,SerializedVec<U,O>>;
}
impl<U,O,LI> BatchPreTrain<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        Ok(Cons(Nil,input))
    }
}
impl<U,O,LI> BatchBackward<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {
    type BatchLossInput = SerializedVec<U,LI>;
    type BatchLossOutput = SerializedVec<U,LI>;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, _: Self::BatchOutStack, _: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        Ok((input,Nil))
    }
}
impl<U,O,LI> BatchLoss<U> for InputLayer<U,O,LI> where U: UnitValue<U>, O: Debug + Send + Sync + 'static, LI: Debug {}
