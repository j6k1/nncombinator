//! Implementation of all full connected layers
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::{Arr, Arr2, ArrView, DiffArr, SerializedVec, SerializedVecConverter, SerializedVecView};
use crate::{Cons, Stack};
use crate::cuda::{CudaTensor1dPtr, CudaTensor2dPtr, Memory};
use crate::device::{Device, DeviceCpu, DeviceGpu, DeviceMemoryPool};
use crate::device::linear::{DeviceDiffLinear, DeviceLinear};
use crate::error::{ConfigReadError, EvaluateError, LayerInstantiationError, PersistenceError, SizeMismatchError, TrainingError, UnsupportedOperationError};
use crate::layer::{AskDiffInput, Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, DiffInput, Forward, ForwardAll, Loss, PreTrain, UpdateWeight};
use crate::lossfunction::LossFunction;
use crate::mem::AsRawSlice;
use crate::ope::UnitValue;
use crate::optimizer::{Optimizer, OptimizerBuilder};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, UnitOrMarker};

/// Linear Layer Implementation
pub struct LinearLayer<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,D> {
    u:PhantomData<U>,
    parent:P,
    device:D,
    units:C,
    bias:BC,
    unit_optimizer:OP,
    bias_optimizer:OP,
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>> {
    /// Create and return an instance of LinearLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    /// * `b` - optimizer builder
    pub fn new<UI,BI,B>(parent:P,device:&DeviceCpu<U>,mut ui:UI,mut bi:BI, b:&B)
        -> Result<LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>,LayerInstantiationError>
        where UI: FnMut() -> U, BI: FnMut() -> U, B: OptimizerBuilder<U,DeviceCpu<U>,Output=OP> {

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
            u:PhantomData::<U>,
            parent:parent,
            device:device.clone(),
            units: units,
            bias:bias,
            unit_optimizer:b.build(NI*NO)?,
            bias_optimizer:b.build(NO)?
        })
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> {
    /// Attempt to create and return an instance of LinearLayer.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    /// * `b` - optimizer builder
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new<UI,BI,B>(parent:P,device:&DeviceGpu<U>,ui:UI,bi:BI, b:&B)
        -> Result<LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>,LayerInstantiationError>
        where UI: FnMut() -> U, BI: FnMut() -> U, B: OptimizerBuilder<U,DeviceGpu<U>,Output=OP> {

        let units = CudaTensor2dPtr::with_initializer(device.get_memory_pool(),ui)?;
        let bias = CudaTensor1dPtr::with_initializer(device.get_memory_pool(),bi)?;

        Ok(LinearLayer {
            u:PhantomData::<U>,
            parent:parent,
            device:device.clone(),
            units: units,
            bias: bias,
            unit_optimizer:b.build(NI*NO)?,
            bias_optimizer:b.build(NO)?
        })
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>,Specialized> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>>,
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
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>,Specialized>
    for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U>,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        let mut bias = Arr::<U,NO>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        let mut units = Arr2::<U,NI,NO>::new();

        for mut u in units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        self.bias.memcpy(bias.as_raw_slice().as_ptr(),NO)?;
        self.units.memcpy(units.as_raw_slice().as_ptr(),NI*NO)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        let bias = Arr::<U,NO>::try_from(self.bias.read_to_vec()?)?;
        let units = Arr2::<U,NI,NO>::try_from(self.units.read_to_vec()?)?;

        for b in bias.iter() {
            persistence.write(UnitOrMarker::Unit(*b));
        }

        for u in units.iter() {
            persistence.write(UnitOrMarker::UnitsStart);
            for w in u.iter() {
                persistence.write(UnitOrMarker::Unit(*w));
            }
        }

        Ok(())
    }
}
impl<T,U,P,I,PI,OP,const NI:usize,const NO:usize> Persistence<U,T,Linear> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>> {
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
impl<T,U,P,I,PI,OP,const NI:usize,const NO:usize> Persistence<U,T,Linear> 
    for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        let mut bias = Arr::<U,NO>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        let mut units = Arr2::<U,NI,NO>::new();

        for mut u in units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        self.bias.memcpy(bias.as_raw_slice().as_ptr(),NO)?;
        self.units.memcpy(units.as_raw_slice().as_ptr(),NI*NO)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        let bias = Arr::<U,NO>::try_from(self.bias.read_to_vec()?)?;

        for b in bias.iter() {
            persistence.write(*b)?;
        }

        let units = Arr2::<U,NI,NO>::try_from(self.units.read_to_vec()?)?;

        for u in units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> Forward<PI,Result<Arr<U,NO>,EvaluateError>> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceLinear<U,C,BC,NI,NO>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          for<'a> ArrView<'a,U,NI>: From<&'a PI> {

    fn forward(&self,input:&PI) -> Result<Arr<U,NO>,EvaluateError> {
        self.device.forward_linear(&self.bias,&self.units,input.into())
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> ForwardAll for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          D: Device<U> + DeviceLinear<U,C,BC,NI,NO>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          for<'a> ArrView<'a,U,NI>: From<&'a PI> {
    type Input = I;
    type Output = Arr<U,NO>;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> PreTrain<U> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          D: Device<U> + DeviceLinear<U,C,BC,NI,NO>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          for<'a> ArrView<'a,U,NI>: From<&'a PI> {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::Output>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r.into()))?;

        Ok(Cons(r,u))
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> Backward<U,&Arr<U,NO>,Result<Arr<U,NI>,TrainingError>> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceLinear<U,C,BC,NI,NO>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,D> {
    fn backward(&mut self, input: &Arr<U,NO>) -> Result<Arr<U,NI>,TrainingError> {
        self.device.backward_linear(&self.units,input.into())
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BackwardAll<U> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> + ForwardAll<Input=I,Output=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,DeviceCpu<U>>,
          for<'a> ArrView<'a,U,NI>: From<&'a PI>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>> {
    type LossInput = Arr<U,NO>;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let next_loss = self.backward(&loss)?;

        let g = s.map(|o| {
            self.device.backward_weight_gradient(o.into(),(&loss).into())
        })?;

        let bg = loss;

        let (s,loss) = self.parent.loss(next_loss.into(),lossf,s)?;

        let (l,s) = self.parent.backward_all(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BackwardAll<U> for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> + ForwardAll<Input=I,Output=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> + DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO>,
          for<'a> ArrView<'a,U,NI>: From<&'a PI>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor2dPtr<U,NI,NO>> {
    type LossInput = Arr<U,NO>;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let next_loss = self.backward(&loss)?;

        let g = s.map(|o| {
            self.device.backward_weight_gradient(o.into(),(&loss).into())
        })?;

        let mut bg = CudaTensor1dPtr::<U,NO>::new(self.device.get_memory_pool())?;

        bg.memcpy(loss.as_raw_slice().as_ptr(),NO)?;

        let (s,loss) = self.parent.loss(next_loss.into(),lossf,s)?;

        let (l,s) = self.parent.backward_all(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> AskDiffInput<U> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: PreTrain<U,OutStack=<<Self as PreTrain<U>>::OutStack as Stack>::Remaining> +
             ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U> +
             AskDiffInput<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          Self: PreTrain<U> {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map_remaining(|s| self.parent.ask_diff_input(s))
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> UpdateWeight<U> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,DeviceCpu<U>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(Arr2<U,NI,NO>,Arr<U,NO>)>;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        self.bias_optimizer.update((&bg).into(), (&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> UpdateWeight<U> for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor2dPtr<U,NI,NO>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>)>;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        self.bias_optimizer.update((&bg).into(), (&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> Loss<U> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,DeviceCpu<U>>,
          for<'a> ArrView<'a,U,NI>: From<&'a PI>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>> {
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> Loss<U> for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,DeviceGpu<U>>,
          for<'a> ArrView<'a,U,NI>: From<&'a PI>,
          DeviceGpu<U>: Device<U>,
          Self: BackwardAll<U>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor2dPtr<U,NI,NO>> {
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchForwardBase for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          Self: ForwardAll {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = SerializedVec<U,Arr<U, NO>>;
}
impl<U,C,BC,P,OP,D,I,PI,const NI:usize,const NO:usize> BatchForward for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward,
          D: Device<U> + DeviceLinear<U,C,BC,NI,NO>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          for<'a> ArrView<'a,U,NI>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,NI>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError> {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        self.device.batch_forward_linear(&self.bias,&self.units,(&input).try_into()?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchPreTrainBase<U> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          Self: PreTrain<U> {
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,Self::BatchOutput>;
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> Forward<SerializedVecView<'_,U,Arr<U,NI>>,Result<SerializedVec<U,Arr<U,NO>>,TrainingError>>
    for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceLinear<U,C,BC,NI,NO>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          for<'a> ArrView<'a,U,NI>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,NI>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError> {
    fn forward(&self, input: &SerializedVecView<'_,U,Arr<U,NI>>) -> Result<SerializedVec<U,Arr<U,NO>>,TrainingError> {
        Ok(self.device.batch_forward_linear(&self.bias,&self.units,*input)?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchPreTrain<U> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceLinear<U,C,BC,NI,NO>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          for<'a> ArrView<'a,U,NI>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,NI>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| self.forward(&SerializedVecView::try_from(input)?))?;

        Ok(Cons(r,u))
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BatchBackward<U> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,Arr<U,NI>>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,DeviceCpu<U>>,
          <I as BatchDataType>::Type: Debug,
          for<'a> ArrView<'a,U,NI>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,NI>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,Arr<U,NI>>,Error=SizeMismatchError>,
          DeviceCpu<U>: DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>> {
    type BatchLossInput = SerializedVec<U,Arr<U,NO>>;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let next_loss = self.device.batch_backward_linear(&self.units, &loss)?;

        let g = s.map(|o| {
            self.device.batch_backward_weight_gradient(o.try_into()?, &loss)
        })?;

        let bg = self.device.batch_linear_reduce((&loss).try_into()?)?;

        let (
            s,loss
        ) = self.parent.batch_loss(next_loss.into_converter().try_into()?,lossf,s)?;

        let (l,s) = self.parent.batch_backward(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BatchBackward<U> for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,Arr<U,NI>>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,DeviceGpu<U>>,
          <I as BatchDataType>::Type: Debug,
          for<'a> ArrView<'a,U,NI>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,NI>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,Arr<U,NI>>,Error=SizeMismatchError>,
          DeviceGpu<U>: Device<U> + DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor2dPtr<U,NI,NO>> {
    type BatchLossInput = SerializedVec<U,Arr<U,NO>>;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let next_loss = self.device.batch_backward_linear(&self.units, &loss)?;

        let g = s.map(|o| {
            self.device.batch_backward_weight_gradient(o.try_into()?, &loss)
        })?;

        let bg = self.device.batch_linear_reduce((&loss).try_into()?)?;

        let (
            s,
            loss
        ) = self.parent.batch_loss(next_loss.into_converter().try_into()?,lossf,s)?;

        let (l,s) = self.parent.batch_backward(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BatchLoss<U> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,Arr<U,NI>>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,DeviceCpu<U>>,
          <I as BatchDataType>::Type: Debug,
          for<'a> ArrView<'a,U,NI>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,NI>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,Arr<U,NI>>,Error=SizeMismatchError>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>> {
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BatchLoss<U> for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=SerializedVec<U,PI>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,Arr<U,NI>>>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + Send + Sync + From<Arr<U,NI>>,
          OP: Optimizer<U,DeviceGpu<U>>,
          <I as BatchDataType>::Type: Debug,
          for<'a> ArrView<'a,U,NI>: From<&'a PI>,
          for<'a> SerializedVecView<'a,U,Arr<U,NI>>: TryFrom<&'a SerializedVec<U,PI>,Error=SizeMismatchError>,
          SerializedVec<U,PI>: TryFrom<SerializedVecConverter<U,Arr<U,NI>>,Error=SizeMismatchError>,
          DeviceGpu<U>: Device<U>,
          Self: Loss<U> + BatchBackward<U> {
}
/// Trait for LinearLayer instance creation
pub trait LinearLayerInstantiation<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          D: Device<U> {
    /// Create an instance of LinearLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    /// * `b` - optimizer builder
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    fn instantiation<B: OptimizerBuilder<U,D,Output=OP>>(parent:P,device:&D,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
        -> Result<LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>,LayerInstantiationError>;
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> LinearLayerInstantiation<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>> {
    fn instantiation<B: OptimizerBuilder<U,DeviceCpu<U>,Output=OP>>(parent: P, device:&DeviceCpu<U>,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
        -> Result<LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>,LayerInstantiationError> {
        LinearLayer::<_,_,_,_,DeviceCpu<U>,_,_,_,NI,NO>::new(parent,device,ui,bi,b)
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> LinearLayerInstantiation<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> {
    fn instantiation<B: OptimizerBuilder<U,DeviceGpu<U>,Output=OP>>(parent: P, device:&DeviceGpu<U>, ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
        -> Result<LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>,LayerInstantiationError> {
        Ok(LinearLayer::<_,_,_,_,DeviceGpu<U>,_,_,_,NI,NO>::new(parent,device,ui,bi,b)?)
    }
}
/// Builder for LinearLayer instance creation
pub struct LinearLayerBuilder<const NI:usize,const NO:usize> {
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>
}
impl<const NI:usize,const NO:usize> LinearLayerBuilder<NI,NO> {
    /// Create an instance of LinearLayerBuilder
    pub fn new() -> LinearLayerBuilder<NI,NO> {
        LinearLayerBuilder {
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>
        }
    }
}
impl<const NI:usize,const NO:usize> LinearLayerBuilder<NI,NO> {
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
    pub fn build<U,C,BC,P,D,I,PI,OP,B>(&self,parent: P, device:&D, ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
        -> Result<LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=Arr<U,NI>> +
                 PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + UnitValue<U>,
              I: Debug + Send + Sync,
              PI: Debug + Send + Sync,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              D: Device<U>,
              LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>: LinearLayerInstantiation<U,C,BC,P,D,I,PI,OP,NI,NO> {

        LinearLayer::instantiation(parent,device,ui,bi,b)
    }
}
/// Implementation of differentially applicable linear layers
pub struct DiffLinearLayer<U,C,BC,P,OP,D,I,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D> {
    u:PhantomData<U>,
    parent:P,
    device:D,
    units:C,
    bias:BC,
    unit_optimizer: OP,
    bias_optimizer: OP
}
impl<U,P,OP,I,const NI:usize,const NO:usize> DiffLinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,OP,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>> {
    /// Create and return an instance of DiffLinearLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    /// * `b` - optimizer builder
    pub fn new<UI,BI,B>(parent:P,device:&DeviceCpu<U>,mut ui:UI,mut bi:BI, b: &B)
        -> Result<DiffLinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,OP,DeviceCpu<U>,I,NI,NO>,LayerInstantiationError>
        where UI: FnMut() -> U, BI: FnMut() -> U, B: OptimizerBuilder<U,DeviceCpu<U>,Output=OP> {

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
            u:PhantomData::<U>,
            parent:parent,
            device:device.clone(),
            units: units,
            bias:bias,
            unit_optimizer:b.build(NI*NO)?,
            bias_optimizer:b.build(NO)?
        })
    }
}
impl<U,P,OP,I,const NI:usize,const NO:usize> DiffLinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,OP,DeviceGpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Debug + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> {
    /// Attempt to create and return an instance of DiffLinearLayer.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    /// * `b` - optimizer builder
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new<UI,BI,B>(parent:P,device:&DeviceGpu<U>,ui:UI,bi:BI, b: &B)
        -> Result<DiffLinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,OP,DeviceGpu<U>,I,NI,NO>,LayerInstantiationError>
        where UI: FnMut() -> U, BI: FnMut() -> U, B: OptimizerBuilder<U,DeviceGpu<U>,Output=OP> {

        Ok(DiffLinearLayer {
            u:PhantomData::<U>,
            parent:parent,
            device:device.clone(),
            units:CudaTensor2dPtr::with_initializer(device.get_memory_pool(),ui)?,
            bias:CudaTensor1dPtr::with_initializer(device.get_memory_pool(),bi)?,
            unit_optimizer:b.build(NI*NO)?,
            bias_optimizer:b.build(NO)?
        })
    }
}
impl<U,P,OP,I,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>,Specialized>
    for DiffLinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,OP,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>>,
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
impl<U,P,OP,I,const NI:usize,const NO:usize> Persistence<U,TextFilePersistence<U>,Specialized>
    for DiffLinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,OP,DeviceGpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U>,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        let mut bias = Arr::<U,NO>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        let mut units = Arr2::<U,NI,NO>::new();

        for mut u in units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        self.bias.memcpy(bias.as_raw_slice().as_ptr(),NO)?;
        self.units.memcpy(units.as_raw_slice().as_ptr(),NI*NO)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        let bias = Arr::<U,NO>::try_from(self.bias.read_to_vec()?)?;

        for b in bias.iter() {
            persistence.write(UnitOrMarker::Unit(*b));
        }

        let units = Arr2::<U,NI,NO>::try_from(self.units.read_to_vec()?)?;

        for u in units.iter() {
            persistence.write(UnitOrMarker::UnitsStart);
            for w in u.iter() {
                persistence.write(UnitOrMarker::Unit(*w));
            }
        }

        Ok(())
    }
}
impl<T,U,P,OP,I,const NI:usize,const NO:usize> Persistence<U,T,Linear>
    for DiffLinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,OP,DeviceCpu<U>,I,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>> {
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
impl<T,U,P,OP,I,const NI:usize,const NO:usize> Persistence<U,T,Linear>
    for DiffLinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,OP,DeviceGpu<U>,I,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U> +
             Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        let mut bias = Arr::<U,NO>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        let mut units = Arr2::<U,NI,NO>::new();

        for mut u in units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        self.bias.memcpy(bias.as_raw_slice().as_ptr(),NO)?;
        self.units.memcpy(units.as_raw_slice().as_ptr(),NI*NO)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        let bias = Arr::<U,NO>::try_from(self.bias.read_to_vec()?)?;

        for b in bias.iter() {
            persistence.write(*b)?;
        }

        let units = Arr2::<U,NI,NO>::try_from(self.units.read_to_vec()?)?;

        for u in units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<U,C,BC,P,OP,D,I,const NI:usize,const NO:usize> ForwardAll for DiffLinearLayer<U,C,BC,P,OP,D,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          for<'a> D: Device<U> + DeviceLinear<U,C,BC,NI,NO> + DeviceDiffLinear<'a,U,C,NI,NO>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D> {
    type Input = I;
    type Output = Arr<U,NO>;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_all(input)?;

        match input {
            DiffInput::Diff(d, output) => {
                self.device.forward_diff_linear(&d,&self.units,(&output).into())
            },
            DiffInput::NotDiff(input) => {
                self.device.forward_linear(&self.bias,&self.units,(&input).into())
            }
        }
    }
}
impl<U,C,BC,P,OP,D,I,const NI:usize,const NO:usize> PreTrain<U> for DiffLinearLayer<U,C,BC,P,OP,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          for<'a> D: Device<U> + DeviceLinear<U,C,BC,NI,NO> + DeviceDiffLinear<'a,U,C,NI,NO>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D> {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::Output>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let s = self.parent.pre_train(input)?;

        let u = s.map(|input| {
            match input {
                DiffInput::Diff(d, output) => {
                    self.device.forward_diff_linear(d,&self.units,output.into())
                },
                DiffInput::NotDiff(input) => {
                    self.device.forward_linear(&self.bias,&self.units,input.into())
                }
            }
        })?;

        Ok(Cons(s,u))
    }
}
impl<U,C,BC,P,OP,D,I,const NI:usize,const NO:usize> Backward<U,&Arr<U,NO>,Result<Arr<U,NI>,TrainingError>> for DiffLinearLayer<U,C,BC,P,OP,D,I,NI,NO>
    where U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceLinear<U,C,BC,NI,NO>,
          P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + PreTrain<U> + Loss<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D> {
    fn backward(&mut self, input: &Arr<U,NO>) -> Result<Arr<U,NI>,TrainingError> {
        self.device.backward_linear(&self.units,input.into())
    }
}
impl<U,P,OP,I,const NI:usize,const NO:usize> BackwardAll<U> for DiffLinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,OP,DeviceCpu<U>,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>>,
          Self: ForwardAll + PreTrain<U,OutStack=Cons<<P as PreTrain<U>>::OutStack,Arr<U,NO>>> {
    type LossInput = Arr<U,NO>;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let next_loss= self.backward(&loss)?;

        let g = s.map(|o| {
            match o {
                DiffInput::Diff(_, _) => {
                    Err(TrainingError::UnsupportedOperationError(UnsupportedOperationError(
                        String::from("Training from difference information is not supported.")
                    )))
                },
                DiffInput::NotDiff(o) => {
                    self.device.backward_weight_gradient(o.into(),(&loss).into())
                }
            }
        })?;

        let bg = loss;

        let (s,loss) = self.parent.loss(next_loss,lossf,s)?;

        let (l,s) = self.parent.backward_all(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,OP,I,const NI:usize,const NO:usize> BackwardAll<U> for DiffLinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,OP,DeviceGpu<U>,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> + DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor2dPtr<U,NI,NO>>,
          Self: ForwardAll + PreTrain<U,OutStack=Cons<<P as PreTrain<U>>::OutStack,Arr<U,NO>>> {
    type LossInput = Arr<U,NO>;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let next_loss = self.backward(&loss)?;

        let g = s.map(|o| {
            match o {
                DiffInput::Diff(_, _) => {
                    Err(TrainingError::UnsupportedOperationError(UnsupportedOperationError(
                        String::from("Training from difference information is not supported.")
                    )))
                },
                DiffInput::NotDiff(o) => {
                    self.device.backward_weight_gradient(o.into(),(&loss).into())
                }
            }
        })?;

        let mut bg = CudaTensor1dPtr::<U,NO>::new(self.device.get_memory_pool())?;

        bg.memcpy(loss.as_raw_slice().as_ptr(),NO)?;

        let (s,loss) = self.parent.loss(next_loss,lossf,s)?;

        let (l,s) = self.parent.backward_all(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,OP,I,const NI:usize,const NO:usize> UpdateWeight<U> for DiffLinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,OP,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             PreTrain<U> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a mut Arr<U,NO>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(Arr2<U,NI,NO>,Arr<U,NO>)>;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        self.bias_optimizer.update((&bg).into(),(&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,P,OP,I,const NI:usize,const NO:usize> UpdateWeight<U> for DiffLinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,OP,DeviceGpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             PreTrain<U> + Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> &'a mut <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a mut CudaTensor2dPtr<U,NI,NO>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>)>;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        self.bias_optimizer.update((&bg).into(),(&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,C,BC,P,OP,D,I,const NI:usize,const NO:usize> AskDiffInput<U> for DiffLinearLayer<U,C,BC,P,OP,D,I,NI,NO>
    where P: BackwardAll<U,LossInput=Arr<U,NI>> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          Self: PreTrain<U,OutStack=Cons<<P as PreTrain<U>>::OutStack,Arr<U,NO>>> {
    type DiffInput = Arr<U,NO>;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map(|o| o.clone())
    }
}
impl<U,C,BC,P,OP,D,I,const NI:usize,const NO:usize> Loss<U> for DiffLinearLayer<U,C,BC,P,OP,D,I,NI,NO>
    where P: PreTrain<U> + ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=Arr<U,NI>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          Self: BackwardAll<U> {
}
/// Trait for DiffLinearLayer instance creation
pub trait DiffLinearLayerInstantiation<U,C,BC,P,OP,D,I,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          D: Device<U>,
          OP: Optimizer<U,D> {
    /// Create an instance of DiffLinearLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    /// * `b` - optimizer builder
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    fn instantiation<UI: FnMut() -> U, BI: FnMut() -> U, B: OptimizerBuilder<U,D,Output=OP>>(parent:P,device:&D,ui: UI, bi: BI, b: &B)
        -> Result<DiffLinearLayer<U,C,BC,P,OP,D,I,NI,NO>,LayerInstantiationError>;
}
impl<U,P,OP,I,const NI:usize,const NO:usize> DiffLinearLayerInstantiation<U,Arr2<U,NI,NO>,Arr<U,NO>,P,OP,DeviceCpu<U>,I,NI,NO>
    for DiffLinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,OP,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>> {
    fn instantiation<UI: FnMut() -> U, BI: FnMut() -> U, B: OptimizerBuilder<U,DeviceCpu<U>,Output=OP>>(parent: P, device:&DeviceCpu<U>,ui: UI, bi: BI, b: &B)
        -> Result<DiffLinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,OP,DeviceCpu<U>,I,NI,NO>,LayerInstantiationError> {
        Ok(DiffLinearLayer::<_,_,_,_,_,DeviceCpu<U>,_,NI,NO>::new(parent,device,ui,bi,b)?)
    }
}
impl<U,P,OP,I,const NI:usize,const NO:usize> DiffLinearLayerInstantiation<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,OP,DeviceGpu<U>,I,NI,NO>
    for DiffLinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,OP,DeviceGpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + BackwardAll<U,LossInput=Arr<U,NI>> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> {
    fn instantiation<UI: FnMut() -> U, BI: FnMut() -> U, B: OptimizerBuilder<U,DeviceGpu<U>,Output=OP>>(parent: P, device:&DeviceGpu<U>, ui: UI, bi: BI, b: &B)
        -> Result<DiffLinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,OP,DeviceGpu<U>,I,NI,NO>,LayerInstantiationError> {
        Ok(DiffLinearLayer::<_,_,_,_,_,DeviceGpu<U>,_,NI,NO>::new(parent,device,ui,bi,b)?)
    }
}
/// Builder for DiffLinearLayer instance creation
pub struct DiffLinearLayerBuilder<const NI:usize,const NO:usize> {
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>
}
impl<const NI:usize,const NO:usize> DiffLinearLayerBuilder<NI,NO> {
    /// Create an instance of DiffLinearLayerBuilder
    pub fn new() -> DiffLinearLayerBuilder<NI, NO> {
        DiffLinearLayerBuilder {
            ni: PhantomData::<[(); NI]>,
            no: PhantomData::<[(); NO]>
        }
    }
}
impl<const NI:usize,const NO:usize> DiffLinearLayerBuilder<NI,NO> {
    /// Create an instance of LinearLayers
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    /// * `b` - optimizer builder
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`LayerInstantiationError`]
    pub fn build<U,C,BC,P,OP,B,D,I>(&self,parent: P, device:&D, ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
                 -> Result<DiffLinearLayer<U,C,BC,P,OP,D,I,NI,NO>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + BackwardAll<U,LossInput=Arr<U,NI>> +
                 PreTrain<U> + Loss<U>,
              U: Default + Clone + Copy + UnitValue<U>,
              I: Debug + Send + Sync,
              D: Device<U>,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              DiffLinearLayer<U,C,BC,P,OP,D,I,NI,NO>: DiffLinearLayerInstantiation<U,C,BC,P,OP,D,I,NI,NO> {

        DiffLinearLayer::instantiation(parent,device,ui,bi,b)
    }
}
