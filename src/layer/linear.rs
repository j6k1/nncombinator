//! Implementation of all full connected layers
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Index;
use std::str::FromStr;
use crate::arr::{Arr, Arr2, DiffArr, SerializedVec};
use crate::{Cons, Stack};
use crate::cuda::mem::CachedTensor;
use crate::device::{Device, DeviceCpu, DeviceGpu, DeviceMemoryPool};
use crate::device::linear::DeviceLinear;
use crate::error::{ConfigReadError, CudaError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError, UnsupportedOperationError};
use crate::layer::{AskDiffInput, Backward, BackwardAll, BatchBackward, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, DiffInput, Forward, ForwardAll, Loss, PreTrain};
use crate::lossfunction::LossFunction;
use crate::ope::UnitValue;
use crate::optimizer::Optimizer;
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, UnitOrMarker};

/// Linear Layer Implementation
pub struct LinearLayer<U,C,P,D,I,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    parent:P,
    device:D,
    units:C,
    bias:Arr<U,NO>
}
impl<U,P,I,const NI:usize,const NO:usize> LinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
    /// Create and return an instance of LinearLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    pub fn new<UI: FnMut() -> U, BI: FnMut() -> U>(parent:P,device:&DeviceCpu<U>,mut ui:UI,mut bi:BI)
                                                   -> LinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO> {

        let mut units:Arr2<U,NI,NO> = Arr2::new();
        let mut bias:Arr<U,NO> = Arr::new();

        for mut it in units.iter_mut() {
            for it in it.iter_mut() {
                *it = ui();
            }
        }

        for it in bias.iter_mut() {
            *it = bi();
        }

        LinearLayer {
            parent:parent,
            device:device.clone(),
            units: units,
            bias:bias
        }
    }
}
impl<U,P,I,const NI:usize,const NO:usize> LinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> {
    /// Attempt to create and return an instance of LinearLayer.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new<UI: FnMut() -> U, BI: FnMut() -> U>(parent:P,device:&DeviceGpu<U>,mut ui:UI,mut bi:BI)
                                                   -> Result<LinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>,CudaError> {

        let mut units:Arr2<U,NI,NO> = Arr2::new();
        let mut bias:Arr<U,NO> = Arr::new();

        for mut it in units.iter_mut() {
            for it in it.iter_mut() {
                *it = ui();
            }
        }

        for it in bias.iter_mut() {
            *it = bi();
        }

        Ok(LinearLayer {
            parent:parent,
            device:device.clone(),
            units: CachedTensor::new(units,device.get_memory_pool())?,
            bias:bias
        })
    }
}
impl<U,P,I,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>,Specialized> for LinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        for mut u in self.units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        for b in self.bias.iter() {
            persistence.write(UnitOrMarker::Unit(*b));
        }

        for u in self.units.iter() {
            persistence.write(UnitOrMarker::UnitsStart);
            for w in u.iter() {
                persistence.write(UnitOrMarker::Unit(*w));
            }
        }

        Ok(())
    }
}
impl<U,P,I,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>,Specialized> for LinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U>,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        for mut u in self.units.scoped_mut().iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        for b in self.bias.iter() {
            persistence.write(UnitOrMarker::Unit(*b));
        }

        for u in self.units.iter() {
            persistence.write(UnitOrMarker::UnitsStart);
            for w in u.iter() {
                persistence.write(UnitOrMarker::Unit(*w));
            }
        }

        Ok(())
    }
}
impl<T,U,P,I,const NI:usize,const NO:usize> Persistence<U,T,Linear> for LinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        for mut u in self.units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for b in self.bias.iter() {
            persistence.write(*b)?;
        }

        for u in self.units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<T,U,P,I,const NI:usize,const NO:usize> Persistence<U,T,Linear> for LinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        for mut u in self.units.scoped_mut().iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for b in self.bias.iter() {
            persistence.write(*b)?;
        }

        for u in self.units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> Forward<Arr<U,NI>,Result<Arr<U,NO>,EvaluateError>> for LinearLayer<U,C,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceLinear<U,C,NI,NO>,
          I: Debug + Send + Sync {

    fn forward(&self,input:&Arr<U,NI>) -> Result<Arr<U,NO>,EvaluateError> {
        self.device.forward_linear(&self.bias,&self.units,&input)
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> ForwardAll for LinearLayer<U,C,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          D: Device<U> + DeviceLinear<U,C,NI,NO>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = Arr<U,NO>;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> PreTrain<U> for LinearLayer<U,C,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          D: Device<U> + DeviceLinear<U,C,NI,NO>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::Output>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r))?;

        Ok(Cons(r,u))
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> Backward<U,&Arr<U,NO>,Result<Arr<U,NI>,TrainingError>> for LinearLayer<U,C,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceLinear<U,C,NI,NO>,
          I: Debug + Send + Sync {
    fn backward(&mut self, input: &Arr<U,NO>) -> Result<Arr<U,NI>,TrainingError> {
        self.device.backward_linear(&self.units,input)
    }
}
impl<U,P,I,const NI:usize,const NO:usize> BackwardAll<U> for LinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> + ForwardAll<Input=I,Output=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
    type LossInput = Arr<U,NO>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) -> Result<(), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        {
            for (w,&g) in self.bias.iter_mut().zip(loss.iter()) {
                optimizer.update(g, w);
            }

            s.map(|o| {
                self.device.backward_weight_gradient(o,&loss).map(|g| {
                    for (mut u,g) in self.units.iter_mut().zip(g.iter()) {
                        for (w,&g) in u.iter_mut().zip(g.iter()) {
                            optimizer.update(g, w);
                        }
                    }
                })
            })?;
        }

        let loss= self.backward(&loss)?;

        let (s,loss) = self.parent.loss(loss,lossf,s)?;

        self.parent.backward_all(loss, s, optimizer, lossf)
    }
}
impl<U,P,I,const NI:usize,const NO:usize> BackwardAll<U> for LinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> + ForwardAll<Input=I,Output=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> + DeviceLinear<U,CachedTensor<U,Arr2<U,NI,NO>>,NI,NO> {
    type LossInput = Arr<U,NO>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) -> Result<(), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        {
            for (w,&g) in self.bias.iter_mut().zip(loss.iter()) {
                optimizer.update(g, w);
            }

            s.map(|o| {
                self.device.backward_weight_gradient(o,&loss).map(|g| {
                    for (mut u,g) in self.units.scoped_mut().iter_mut().zip(g.iter()) {
                        for (w,&g) in u.iter_mut().zip(g.iter()) {
                            optimizer.update(g, w);
                        }
                    }
                })
            })?;
        }

        let loss= self.backward(&loss)?;

        let (s,loss) = self.parent.loss(loss,lossf,s)?;

        self.parent.backward_all(loss, s, optimizer, lossf)
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> AskDiffInput<U> for LinearLayer<U,C,P,D,I,NI,NO>
    where P: PreTrain<U,OutStack=<<Self as PreTrain<U>>::OutStack as Stack>::Remaining> +
             ForwardAll<Input=I,Output=Arr<U,NI>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U> +
             AskDiffInput<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          Self: PreTrain<U> {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map_remaining(|s| self.parent.ask_diff_input(s))
    }
}
impl<U,P,I,const NI:usize,const NO:usize> Loss<U> for LinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
}
impl<U,P,I,const NI:usize,const NO:usize> Loss<U> for LinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U>,
          Self: BackwardAll<U> {
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> BatchForwardBase for LinearLayer<U,C,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Arr<U,NI>>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          Self: ForwardAll {
    type BatchInput = SerializedVec<U,I>;
    type BatchOutput = SerializedVec<U,Arr<U, NO>>;
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> BatchForward for LinearLayer<U,C,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Arr<U,NI>>> + BatchForward,
          D: Device<U> + DeviceLinear<U,C,NI,NO>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        self.device.batch_forward_linear(&self.bias,&self.units,&input)
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> BatchPreTrainBase<U> for LinearLayer<U,C,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Arr<U,NI>>> + BatchForward +
             BatchPreTrainBase<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          Self: PreTrain<U> {
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,Self::BatchOutput>;
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> BatchPreTrain<U> for LinearLayer<U,C,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Arr<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceLinear<U,C,NI,NO>,
          I: Debug + Send + Sync {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| self.device.batch_forward_linear(&self.bias,&self.units,input))?;

        Ok(Cons(r,u))
    }
}
impl<U,P,I,const NI:usize,const NO:usize> BatchBackward<U> for LinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Arr<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,Arr<U,NI>>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
    type BatchLossInput = SerializedVec<U,Arr<U,NO>>;

    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, optimizer: &mut OP, lossf: &L) -> Result<(), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        {
            {
                for (w,&g) in self.bias.iter_mut().zip(self.device.batch_linear_reduce(&loss)?.iter()) {
                    optimizer.update(g, w);
                }

                s.map(|o| {
                    self.device.batch_backward_weight_gradient(&o, &loss).map(|g| {
                        for (mut u, g) in self.units.iter_mut().zip(g.iter()) {
                            for (w, &g) in u.iter_mut().zip(g.iter()) {
                                optimizer.update(g, w);
                            }
                        }
                    })
                })?;
            }
        }

        let loss = self.device.batch_backward_linear(&self.units, &loss)?;

        let (s,loss) = self.parent.batch_loss(loss,lossf,s)?;

        self.parent.batch_backward(loss, s, optimizer, lossf)
    }
}
impl<U,P,I,const NI:usize,const NO:usize> BatchBackward<U> for LinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Arr<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,Arr<U,NI>>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> + DeviceLinear<U,CachedTensor<U,Arr2<U,NI,NO>>,NI,NO> {
    type BatchLossInput = SerializedVec<U,Arr<U,NO>>;

    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, optimizer: &mut OP, lossf: &L) -> Result<(), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        {
            {
                for (w,&g) in self.bias.iter_mut().zip(self.device.batch_linear_reduce(&loss)?.iter()) {
                    optimizer.update(g, w);
                }

                s.map(|o| {
                    self.device.batch_backward_weight_gradient(&o, &loss).map(|g| {
                        for (mut u, g) in self.units.scoped_mut().iter_mut().zip(g.iter()) {
                            for (w, &g) in u.iter_mut().zip(g.iter()) {
                                optimizer.update(g, w);
                            }
                        }
                    })
                })?;
            }
        }

        let loss = self.device.batch_backward_linear(&self.units, &loss)?;

        let (s,loss) = self.parent.batch_loss(loss,lossf,s)?;

        self.parent.batch_backward(loss, s, optimizer, lossf)
    }
}
impl<U,P,I,const NI:usize,const NO:usize> BatchLoss<U> for LinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Arr<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,Arr<U,NI>>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync {
}
impl<U,P,I,const NI:usize,const NO:usize> BatchLoss<U> for LinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Arr<U,NI>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,Arr<U,NI>>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U>,
          Self: Loss<U> + BatchBackward<U> {
}
/// Trait for LinearLayer instance creation
pub trait LinearLayerInstantiation<U,C,P,D,I,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          D: Device<U> {
    /// Create an instance of LinearLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    fn instantiation(parent:P,device:&D,ui: impl FnMut() -> U, bi: impl FnMut() -> U)
                     -> Result<LinearLayer<U,C,P,D,I,NI,NO>,LayerInstantiationError>;
}
impl<U,P,I,const NI:usize,const NO:usize> LinearLayerInstantiation<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    for LinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync {
    fn instantiation(parent: P, device:&DeviceCpu<U>,ui: impl FnMut() -> U, bi: impl FnMut() -> U)
                     -> Result<LinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>,LayerInstantiationError> {
        Ok(LinearLayer::<_,_,_,DeviceCpu<U>,_,NI,NO>::new(parent,device,ui,bi))
    }
}
impl<U,P,I,const NI:usize,const NO:usize> LinearLayerInstantiation<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    for LinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> {
    fn instantiation(parent: P, device:&DeviceGpu<U>, ui: impl FnMut() -> U, bi: impl FnMut() -> U)
                     -> Result<LinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>,LayerInstantiationError> {
        Ok(LinearLayer::<_,_,_,DeviceGpu<U>,_,NI,NO>::new(parent,device,ui,bi)?)
    }
}
/// Builder for LinearLayer instance creation
pub struct LinearLayerBuilder<U,C,P,D,I,const NI:usize,const NO:usize>
    where U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          D: Device<U> {
    u:PhantomData<U>,
    c:PhantomData<C>,
    p:PhantomData<P>,
    d:PhantomData<D>,
    i:PhantomData<I>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>
}
impl<U,C,P,D,I> LinearLayerBuilder<U,C,P,D,I,0,0>
    where U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          D: Device<U> {
    /// Create an instance of LinearLayerBuilder
    ///
    /// # Types
    /// * `N1` - input size
    /// * `N2` - output size
    pub fn new<const N1:usize,const N2:usize>() -> LinearLayerBuilder<U,C,P,D,I,N1,N2> {
        LinearLayerBuilder {
            u:PhantomData::<U>,
            c:PhantomData::<C>,
            p:PhantomData::<P>,
            d:PhantomData::<D>,
            i:PhantomData::<I>,
            ni:PhantomData::<[();N1]>,
            no:PhantomData::<[();N2]>
        }
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> LinearLayerBuilder<U,C,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=Arr<U,NI>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          D: Device<U> {
    /// Create an instance of LinearLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build(&self,parent: P, device:&D, ui: impl FnMut() -> U, bi: impl FnMut() -> U)
                 -> Result<LinearLayer<U,C,P,D,I,NI,NO>,LayerInstantiationError>
        where LinearLayer<U,C,P,D,I,NI,NO>: LinearLayerInstantiation<U,C,P,D,I,NI,NO> {

        LinearLayer::instantiation(parent,device,ui,bi)
    }
}
/// Implementation of differentially applicable linear layers
pub struct DiffLinearLayer<U,C,P,D,I,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync {
    parent:P,
    device:D,
    units:C,
    bias:Arr<U,NO>
}
impl<U,P,I,const NI:usize,const NO:usize> DiffLinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync {
    /// Create and return an instance of DiffLinearLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    pub fn new<UI: FnMut() -> U, BI: FnMut() -> U>(parent:P,device:&DeviceCpu<U>,mut ui:UI,mut bi:BI)
                                                   -> DiffLinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO> {

        let mut units:Arr2<U,NI,NO> = Arr2::new();
        let mut bias:Arr<U,NO> = Arr::new();

        for mut it in units.iter_mut() {
            for it in it.iter_mut() {
                *it = ui();
            }
        }

        for it in bias.iter_mut() {
            *it = bi();
        }

        DiffLinearLayer {
            parent:parent,
            device:device.clone(),
            units: units,
            bias:bias
        }
    }
}
impl<U,P,I,const NI:usize,const NO:usize> DiffLinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Debug + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> {
    /// Attempt to create and return an instance of DiffLinearLayer.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new<UI: FnMut() -> U, BI: FnMut() -> U>(parent:P,device:&DeviceGpu<U>,mut ui:UI,mut bi:BI)
                                                   -> Result<DiffLinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>,CudaError> {

        let mut units:Arr2<U,NI,NO> = Arr2::new();
        let mut bias:Arr<U,NO> = Arr::new();

        for mut it in units.iter_mut() {
            for it in it.iter_mut() {
                *it = ui();
            }
        }

        for it in bias.iter_mut() {
            *it = bi();
        }

        Ok(DiffLinearLayer {
            parent:parent,
            device:device.clone(),
            units: CachedTensor::new(units,device.get_memory_pool())?,
            bias:bias
        })
    }
}
impl<U,P,I,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>,Specialized> for DiffLinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        for mut u in self.units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        for b in self.bias.iter() {
            persistence.write(UnitOrMarker::Unit(*b));
        }

        for u in self.units.iter() {
            persistence.write(UnitOrMarker::UnitsStart);
            for w in u.iter() {
                persistence.write(UnitOrMarker::Unit(*w));
            }
        }

        Ok(())
    }
}
impl<U,P,I,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>,Specialized> for DiffLinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U>,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        for mut u in self.units.scoped_mut().iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        for b in self.bias.iter() {
            persistence.write(UnitOrMarker::Unit(*b));
        }

        for u in self.units.iter() {
            persistence.write(UnitOrMarker::UnitsStart);
            for w in u.iter() {
                persistence.write(UnitOrMarker::Unit(*w));
            }
        }

        Ok(())
    }
}
impl<T,U,P,I,const NI:usize,const NO:usize> Persistence<U,T,Linear> for DiffLinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        for mut u in self.units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for b in self.bias.iter() {
            persistence.write(*b)?;
        }

        for u in self.units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<T,U,P,I,const NI:usize,const NO:usize> Persistence<U,T,Linear> for DiffLinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for b in self.bias.iter_mut() {
            *b = persistence.read()?;
        }

        for mut u in self.units.scoped_mut().iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for b in self.bias.iter() {
            persistence.write(*b)?;
        }

        for u in self.units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> ForwardAll for DiffLinearLayer<U,C,P,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          C: Index<(usize,usize),Output=U>,
          D: Device<U> + DeviceLinear<U,C,NI,NO>,
          I: Debug + Send + Sync {
    type Input = I;
    type Output = Arr<U,NO>;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_all(input)?;

        match input {
            DiffInput::Diff(d, mut output) => {
                for &(i,d) in d.iter() {
                    for (o,j) in output.iter_mut().zip(0..NO) {
                        *o += self.units[(i,j)] * d;
                    }
                }
                Ok(output)
            },
            DiffInput::NotDiff(input) => {
                self.device.forward_linear(&self.bias,&self.units,&input)
            }
        }
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> PreTrain<U> for DiffLinearLayer<U,C,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          C: Index<(usize,usize),Output=U>,
          D: Device<U> + DeviceLinear<U,C,NI,NO>,
          I: Debug + Send + Sync {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::Output>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let s = self.parent.pre_train(input)?;

        let u = s.map(|input| {
            match input {
                DiffInput::Diff(d, output) => {
                    let mut output = output.clone();

                    for &(i, d) in d.iter() {
                        for (o, j) in output.iter_mut().zip(0..NO) {
                            *o += self.units[(i, j)] * d;
                        }
                    }
                    Ok(output)
                },
                DiffInput::NotDiff(input) => {
                    self.device.forward_linear(&self.bias,&self.units,&input)
                }
            }
        })?;

        Ok(Cons(s,u))
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> Backward<U,&Arr<U,NO>,Result<Arr<U,NI>,TrainingError>> for DiffLinearLayer<U,C,P,D,I,NI,NO>
    where U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceLinear<U,C,NI,NO>,
          C: Index<(usize,usize),Output=U>,
          P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          I: Debug + Send + Sync {
    fn backward(&mut self, input: &Arr<U,NO>) -> Result<Arr<U,NI>,TrainingError> {
        self.device.backward_linear(&self.units,input)
    }
}
impl<U,P,I,const NI:usize,const NO:usize> BackwardAll<U> for DiffLinearLayer<U,Arr2<U,NI,NO>,P,DeviceCpu<U>,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          Self: ForwardAll + PreTrain<U,OutStack=Cons<<P as PreTrain<U>>::OutStack,Arr<U,NO>>> {
    type LossInput = Arr<U,NO>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) -> Result<(), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        {
            for (w,&g) in self.bias.iter_mut().zip(loss.iter()) {
                optimizer.update(g, w);
            }

            s.map::<_,Result<(),TrainingError>>(|o| {
                match o {
                    DiffInput::Diff(_, _) => {
                        return Err(TrainingError::UnsupportedOperationError(UnsupportedOperationError(
                            String::from("Training from difference information is not supported.")
                        )));
                    },
                    DiffInput::NotDiff(o) => {
                        let g = self.device.backward_weight_gradient(&o,&loss);

                        g.map(|g| {
                            for (mut u,g) in self.units.iter_mut().zip(g.iter()) {
                                for (w,&g) in u.iter_mut().zip(g.iter()) {
                                    optimizer.update(g, w);
                                }
                            }
                        })?;
                    }
                }

                Ok(())
            })?;
        }

        let loss= self.backward(&loss)?;

        let (s,loss) = self.parent.loss(loss,lossf,s)?;

        self.parent.backward_all(loss, s, optimizer, lossf)
    }
}
impl<U,P,I,const NI:usize,const NO:usize> BackwardAll<U> for DiffLinearLayer<U,CachedTensor<U,Arr2<U,NI,NO>>,P,DeviceGpu<U>,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          DeviceGpu<U>: Device<U> + DeviceLinear<U,CachedTensor<U,Arr2<U,NI,NO>>,NI,NO>,
          Self: ForwardAll + PreTrain<U,OutStack=Cons<<P as PreTrain<U>>::OutStack,Arr<U,NO>>> {
    type LossInput = Arr<U,NO>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L) -> Result<(), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        {
            for (w,&g) in self.bias.iter_mut().zip(loss.iter()) {
                optimizer.update(g, w);
            }

            s.map::<_,Result<(),TrainingError>>(|o| {
                match o {
                    DiffInput::Diff(_, _) => {
                        return Err(TrainingError::UnsupportedOperationError(UnsupportedOperationError(
                            String::from("Training from difference information is not supported.")
                        )));
                    },
                    DiffInput::NotDiff(o) => {
                        let g = self.device.backward_weight_gradient(&o,&loss);

                        g.map(|g| {
                            for (mut u,g) in self.units.scoped_mut().iter_mut().zip(g.iter()) {
                                for (w,&g) in u.iter_mut().zip(g.iter()) {
                                    optimizer.update(g, w);
                                }
                            }
                        })?;
                    }
                }

                Ok(())
            })?;
        }

        let loss= self.backward(&loss)?;

        let (s,loss) = self.parent.loss(loss,lossf,s)?;

        self.parent.backward_all(loss, s, optimizer, lossf)
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> AskDiffInput<U> for DiffLinearLayer<U,C,P,D,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          C: Index<(usize,usize),Output=U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          Self: PreTrain<U,OutStack=Cons<<P as PreTrain<U>>::OutStack,Arr<U,NO>>> {
    type DiffInput = Arr<U,NO>;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map(|o| o.clone())
    }
}
impl<U,C,P,D,I,const NI:usize,const NO:usize> Loss<U> for DiffLinearLayer<U,C,P,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          C: Index<(usize,usize),Output=U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          Self: BackwardAll<U> {
}
