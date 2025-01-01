//! Implementation of all full connected layers
use std::fmt::Debug;
use std::marker::PhantomData;
use std::str::FromStr;
use crate::arr::{Arr, Arr2, DiffArr, IntoConverter};
use crate::{Cons, Stack};
use crate::cuda::{CudaTensor1dPtr, CudaTensor2dPtr, DataTypeInfo, ReadMemory, WriteMemory};
use crate::device::{Device, DeviceCpu, DeviceGpu, DeviceMemoryPool};
use crate::device::linear::{DeviceDiffLinear, DeviceLinear};
use crate::error::{ConfigReadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError, TypeConvertError};
use crate::layer::{AskDiffInput, Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, BatchSize, DiffInput, Forward, ForwardAll, Loss, PreTrain, UpdateWeight};
use crate::lossfunction::LossFunction;
use crate::mem::AsRawSlice;
use crate::ope::UnitValue;
use crate::optimizer::{Optimizer, OptimizerBuilder};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, UnitOrMarker};

/// Linear Layer Implementation
pub struct LinearLayer<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
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
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug,
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
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug,
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
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: Debug,
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
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + UnitValue<U> + FromStr,
          I: Debug + Send + Sync,
          PI: Debug,
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
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug,
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
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug,
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
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> Forward<PI,Result<<Self as ForwardAll>::Output,EvaluateError>>
    for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO> {

    fn forward(&self,input:&PI) -> Result<<Self as ForwardAll>::Output,EvaluateError> {
        self.device.forward_linear(&self.bias,&self.units,input.into())
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> ForwardAll for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO> {
    type Input = I;
    type Output = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> PreTrain<U> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: PreTrain<U,PreOutput=PI> +
             ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          PI: Debug + BatchDataType + From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO> {
    type PreOutput = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output;
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::PreOutput>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r))?;

        Ok(Cons(r,u))
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize>
    Backward<U,&<D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output,Result<PI,TrainingError>> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          PI: Debug + BatchDataType + From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO> {
    fn backward(&mut self, input: &<D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output) -> Result<PI,TrainingError> {
        Ok(self.device.backward_linear(&self.units,input)?.into())
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BackwardAll<U> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: BackwardAll<U,LossInput=PI> + ForwardAll<Input=I,Output=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>>,
          PI: Debug + BatchDataType +
              From<<DeviceCpu<U> as DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>>::LossOutput>,
          DeviceCpu<U>: Device<U> + DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>> {
    type LossInput = <DeviceCpu<U> as DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>>::Output;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let next_loss = self.backward(&loss)?;

        let g = s.map(|o| {
            self.device.backward_weight_gradient(o.into(),&loss)
        })?;

        let bg = self.device.backward_bias_weight_gradient(loss)?;

        let (s,loss) = self.parent.loss(next_loss.into(),lossf,s)?;

        let (l,s) = self.parent.backward_all(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BackwardAll<U> for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: BackwardAll<U,LossInput=PI> + ForwardAll<Input=I,Output=PI> +
             PreTrain<U,PreOutput=PI> +
             Loss<U>,
          U: Default + Debug + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          PI: Debug + BatchDataType +
              From<<DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::LossOutput>,
          DeviceGpu<U>: Device<U> + DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor2dPtr<U,NI,NO>> {
    type LossInput = <DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::Output;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let next_loss = self.backward(&loss)?;

        let g = s.map(|o| {
            self.device.backward_weight_gradient(o.into(),&loss)
        })?;

        let bg = self.device.backward_bias_weight_gradient(loss)?;

        let (s,loss) = self.parent.loss(next_loss.into(),lossf,s)?;

        let (l,s) = self.parent.backward_all(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> AskDiffInput<U> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: PreTrain<U,OutStack=<<Self as PreTrain<U>>::OutStack as Stack>::Remaining> +
             ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI> + Loss<U> +
             AskDiffInput<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType,
          OP: Optimizer<U,D>,
          Self: PreTrain<U> {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Result<Self::DiffInput,TypeConvertError> {
        stack.map_remaining(|s| self.parent.ask_diff_input(s))
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> UpdateWeight<U> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> +
             Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>>,
          PI: Debug + BatchDataType + From<<DeviceCpu<U> as DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>>::LossOutput>,
          DeviceCpu<U>: Device<U> + DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(Arr2<U,NI,NO>,Arr<U,NO>)>;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        self.bias_optimizer.update((&bg).into(), (&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> UpdateWeight<U> for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> +
             Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          PI: Debug + BatchDataType + From<<DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::LossOutput>,
          DeviceGpu<U>: Device<U> + DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor2dPtr<U,NI,NO>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>)>;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        self.bias_optimizer.update((&bg).into(), (&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> Loss<U> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: PreTrain<U,PreOutput=PI> +
             ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>>,
          PI: Debug + BatchDataType + From<<DeviceCpu<U> as DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>>::LossOutput>,
          DeviceCpu<U>: Device<U> + DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>> {
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> Loss<U> for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: PreTrain<U,PreOutput=PI> +
             ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          PI: Debug + BatchDataType +
              From<<DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::LossOutput>,
          DeviceGpu<U>: Device<U> + DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>,
          Self: BackwardAll<U>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor2dPtr<U,NI,NO>> {
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchForwardBase for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> +
             Loss<U> + BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          PI: Debug + BatchDataType + BatchDataType,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput: Debug {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput;
}
impl<U,C,BC,P,OP,D,I,PI,const NI:usize,const NO:usize> BatchForward for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          PI: Debug + BatchDataType + From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput> + BatchDataType,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput: Debug {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_forward_linear(&self.bias,&self.units,&input)?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchPreTrainBase<U> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> + PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: BatchSize,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          PI: Debug + BatchDataType + From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput> + BatchDataType,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput: Debug,
          Self: PreTrain<U,PreOutput=<D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output> {
    type BatchPreOutput = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput;
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,Self::BatchPreOutput>;
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchPreTrain<U> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: BatchSize,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          PI: Debug + BatchDataType + From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput> + BatchDataType,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput: Debug {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| {
            self.device.batch_forward_linear(&self.bias,&self.units,input)
        })?;

        Ok(Cons(r,u))
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BatchBackward<U> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> +
             Loss<U> + BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          PI: BatchDataType,
          OP: Optimizer<U,DeviceCpu<U>>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: BatchSize,
          <PI as BatchDataType>::Type: IntoConverter,
          <PI as BatchDataType>::Type: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError> + Debug,
          PI: Debug + From<<DeviceCpu<U> as DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>>::LossOutput> + BatchDataType,
          DeviceCpu<U>: Device<U> + DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO,BatchLossOutput=<PI as BatchDataType>::Type>,
          <DeviceCpu<U> as DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>>::BatchOutput: Debug,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>> {
    type BatchLossInput = <DeviceCpu<U> as DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>>::BatchOutput;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let next_loss = self.device.batch_backward_linear(&self.units, &loss)?;

        let g = s.map(|o| {
            self.device.batch_backward_weight_gradient(o, &loss)
        })?;

        let bg = self.device.batch_linear_reduce(&loss)?;

        let (
            s,loss
        ) = self.parent.batch_loss(next_loss.into_converter().try_into()?,lossf,s)?;

        let (l,s) = self.parent.batch_backward(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BatchBackward<U> for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,DeviceGpu<U>>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: BatchSize,
          <PI as BatchDataType>::Type: IntoConverter,
          <PI as BatchDataType>::Type: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError> + Debug,
          PI: Debug + From<<DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::LossOutput> + BatchDataType,
          DeviceGpu<U>: Device<U> + DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO,BatchLossOutput=<PI as BatchDataType>::Type>,
          <DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::BatchOutput: Debug,
          <<DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::Output as BatchDataType>::Type: Debug,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor2dPtr<U,NI,NO>> {
    type BatchLossInput = <DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::BatchOutput;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward<L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, lossf: &L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let next_loss = self.device.batch_backward_linear(&self.units, &loss)?;

        let g = s.map(|o| {
            self.device.batch_backward_weight_gradient(o, &loss)
        })?;

        let bg = self.device.batch_linear_reduce(&loss)?;

        let (
            s,
            loss
        ) = self.parent.batch_loss(next_loss.into_converter().try_into()?,lossf,s)?;

        let (l,s) = self.parent.batch_backward(loss, s, lossf)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BatchLoss<U> for LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,DeviceCpu<U>>,
          PI: Debug + From<<DeviceCpu<U> as DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>>::LossOutput> + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: BatchSize,
          <PI as BatchDataType>::Type: IntoConverter,
          <PI as BatchDataType>::Type: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError> + Debug,
          DeviceCpu<U>: Device<U> + DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO,BatchLossOutput=<PI as BatchDataType>::Type>,
          <DeviceCpu<U> as DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>>::BatchOutput: Debug,
          <<DeviceCpu<U> as DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,PI,NI,NO>>::Output as BatchDataType>::Type: Debug,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>> {
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> BatchLoss<U> for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> +
             Loss<U> + BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<U,BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain<U> +
             BatchBackward<U> + BatchLoss<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Send + UnitValue<U> + DataTypeInfo,
          I: Debug + Send + Sync + BatchDataType,
          PI: Debug + From<<DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::LossOutput> + BatchDataType,
          OP: Optimizer<U,DeviceGpu<U>>,
          <PI as BatchDataType>::Type: IntoConverter,
          <PI as BatchDataType>::Type: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError> + Debug,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: BatchSize,
          <PI as BatchDataType>::Type: TryFrom<<<DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::BatchLossOutput as IntoConverter>::Converter> + Debug,
          DeviceGpu<U>: Device<U> + DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO,BatchLossOutput=<PI as BatchDataType>::Type>,
          <DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::BatchOutput: Debug,
          <<DeviceGpu<U> as DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,PI,NI,NO>>::Output as BatchDataType>::Type: Debug,
          Self: Loss<U> + BatchBackward<U> {
}
/// Trait for LinearLayer instance creation
pub trait LinearLayerInstantiation<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug,
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
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,DeviceCpu<U>> {
    fn instantiation<B: OptimizerBuilder<U,DeviceCpu<U>,Output=OP>>(parent: P, device:&DeviceCpu<U>,ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
        -> Result<LinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,DeviceCpu<U>,I,PI,OP,NI,NO>,LayerInstantiationError> {
        LinearLayer::<_,_,_,_,DeviceCpu<U>,_,_,_,NI,NO>::new(parent,device,ui,bi,b)
    }
}
impl<U,P,I,PI,OP,const NI:usize,const NO:usize> LinearLayerInstantiation<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    for LinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,DeviceGpu<U>,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
             PreTrain<U,PreOutput=PI> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          PI: Debug,
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
        where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI> +
                 PreTrain<U,PreOutput=PI> + Loss<U>,
              U: Default + Clone + Copy + UnitValue<U>,
              I: Debug + Send + Sync,
              PI: Debug,
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
             BackwardAll<U,LossInput=()> + PreTrain<U> + Loss<U>,
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
             BackwardAll<U,LossInput=()> + PreTrain<U> + Loss<U>,
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
             BackwardAll<U,LossInput=()> + PreTrain<U> + Loss<U> +
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
             BackwardAll<U,LossInput=()> + PreTrain<U> + Loss<U> +
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
             BackwardAll<U,LossInput=()> + PreTrain<U> + Loss<U> +
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
             BackwardAll<U,LossInput=()> + PreTrain<U> + Loss<U> +
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
             BackwardAll<U,LossInput=()> +
             PreTrain<U,PreOutput=DiffInput<DiffArr<U,NI>,U,NI,NO>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceDiffLinear<U,C,BC,NI,NO>,
          <D as DeviceDiffLinear<U,C,BC,NI,NO>>::Output: Debug + 'static {
    type Input = I;
    type Output = <D as DeviceDiffLinear<U,C,BC,NI,NO>>::Output;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_all(input)?;

        self.device.forward_diff_linear(&self.units,&self.bias,&input)
    }
}
impl<U,C,BC,P,OP,D,I,const NI:usize,const NO:usize> PreTrain<U> for DiffLinearLayer<U,C,BC,P,OP,D,I,NI,NO>
    where P: PreTrain<U,PreOutput=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=()> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceDiffLinear<U,C,BC,NI,NO>,
          <D as DeviceDiffLinear<U,C,BC,NI,NO>>::Output: Debug + 'static {
    type PreOutput = <D as DeviceDiffLinear<U,C,BC,NI,NO>>::Output;
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::PreOutput>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let s = self.parent.pre_train(input)?;

        let u = s.map(|input| {
            self.device.forward_diff_linear(&self.units,&self.bias,input)
        })?;

        Ok(Cons(s,u))
    }
}
impl<U,P,OP,I,const NI:usize,const NO:usize> BackwardAll<U> for DiffLinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,OP,DeviceCpu<U>,I,NI,NO>
    where P: BackwardAll<U,LossInput=()> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             PreTrain<U,PreOutput=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>>,
          DeviceCpu<U>: Device<U> + DeviceDiffLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO>,
          <DeviceCpu<U> as DeviceDiffLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO>>::Output: Debug + 'static,
          Arr<U,NO>: From<<DeviceCpu<U> as DeviceDiffLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO>>::Output>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>>,
          Self: ForwardAll + PreTrain<U,OutStack=Cons<<P as PreTrain<U>>::OutStack,Arr<U,NO>>> {
    type LossInput = <DeviceCpu<U> as DeviceDiffLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO>>::Output;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let g = s.map(|o| {
            self.device.backward_diff_weight_gradient(o,&loss)
        })?;

        let bg = loss;

        let (l,s) = self.parent.backward_all((), s, lossf)?;

        Ok((l,Cons(s,(g,bg.into()))))
    }
}
impl<U,P,OP,I,const NI:usize,const NO:usize> BackwardAll<U> for DiffLinearLayer<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,P,OP,DeviceGpu<U>,I,NI,NO>
    where P: BackwardAll<U,LossInput=()> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             PreTrain<U,PreOutput=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U> + DeviceDiffLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO>,
          <DeviceGpu<U> as DeviceDiffLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO>>::Output: Debug + 'static,
          CudaTensor1dPtr<U,NO>: From<<DeviceGpu<U> as DeviceDiffLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO>>::Output>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor2dPtr<U,NI,NO>>,
          Self: ForwardAll + PreTrain<U,OutStack=Cons<<P as PreTrain<U>>::OutStack,CudaTensor1dPtr<U,NO>>> {
    type LossInput = <DeviceGpu<U> as DeviceDiffLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO>>::Output;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all<L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let g = s.map(|o| {
            self.device.backward_diff_weight_gradient(o,&loss)
        })?;

        let bg = loss;

        let (l,s) = self.parent.backward_all((), s, lossf)?;

        Ok((l,Cons(s,(g,bg.into()))))
    }
}
impl<U,P,OP,I,const NI:usize,const NO:usize> UpdateWeight<U> for DiffLinearLayer<U,Arr2<U,NI,NO>,Arr<U,NO>,P,OP,DeviceCpu<U>,I,NI,NO>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             PreTrain<U> +
             Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr2<U,NI,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceCpu<U>>>::InternalType: From<&'a Arr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr2<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceCpu<U>>>::InternalUpdateType<'a>: From<&'a mut Arr<U,NO>> {
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
             PreTrain<U> +
             Loss<U> + UpdateWeight<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          DeviceGpu<U>: Device<U>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor1dPtr<U,NO>>,
          for<'a> &'a <OP as Optimizer<U,DeviceGpu<U>>>::InternalType: From<&'a CudaTensor2dPtr<U,NI,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor1dPtr<U,NO>>,
          for<'a> <OP as Optimizer<U,DeviceGpu<U>>>::InternalUpdateType<'a>: From<&'a mut CudaTensor2dPtr<U,NI,NO>> {
    type GradientStack = Cons<<P as UpdateWeight<U>>::GradientStack,(CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>)>;

    fn update_weight(&mut self, stack: Self::GradientStack) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        self.bias_optimizer.update((&bg).into(),(&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s)?)
    }
}
impl<U,C,BC,P,OP,I,const NI:usize,const NO:usize> AskDiffInput<U> for DiffLinearLayer<U,C,BC,P,OP,DeviceCpu<U>,I,NI,NO>
    where P: BackwardAll<U,LossInput=()> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             PreTrain<U,PreOutput=DiffInput<DiffArr<U,NI>,U,NI,NO>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          DeviceCpu<U>: Device<U> + DeviceDiffLinear<U,C,BC,NI,NO>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceCpu<U>>,
          Self: PreTrain<U,OutStack=Cons<<P as PreTrain<U>>::OutStack,Arr<U,NO>>> {
    type DiffInput = Arr<U,NO>;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Result<Self::DiffInput,TypeConvertError> {
        Ok(stack.map(|o| o.clone()))
    }
}
impl<U,C,BC,P,OP,I,const NI:usize,const NO:usize> AskDiffInput<U> for DiffLinearLayer<U,C,BC,P,OP,DeviceGpu<U>,I,NI,NO>
    where P: BackwardAll<U,LossInput=()> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             PreTrain<U,PreOutput=DiffInput<DiffArr<U,NI>,U,NI,NO>> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          DeviceGpu<U>: Device<U> + DeviceDiffLinear<U,C,BC,NI,NO>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,DeviceGpu<U>>,
          Arr<U,NO>: TryFrom<Vec<U>,Error=TypeConvertError>,
          Self: PreTrain<U,OutStack=Cons<<P as PreTrain<U>>::OutStack,CudaTensor1dPtr<U,NO>>> {
    type DiffInput = Arr<U,NO>;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Result<Self::DiffInput,TypeConvertError> {
        Ok(stack.map(|o| {
            o.read_to_vec().map(|r| r.try_into())
        })??)
    }
}
impl<U,C,BC,P,OP,D,I,const NI:usize,const NO:usize> Loss<U> for DiffLinearLayer<U,C,BC,P,OP,D,I,NI,NO>
    where P: PreTrain<U,PreOutput=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> +
             BackwardAll<U,LossInput=()> + Loss<U>,
          U: Default + Clone + Copy + UnitValue<U>,
          D: Device<U> + DeviceDiffLinear<U,C,BC,NI,NO>,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          BC: From<<D as DeviceDiffLinear<U,C,BC,NI,NO>>::Output>,
          Self: BackwardAll<U> {
}
/// Trait for DiffLinearLayer instance creation
pub trait DiffLinearLayerInstantiation<U,C,BC,P,OP,D,I,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + BackwardAll<U,LossInput=()> +
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
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + BackwardAll<U,LossInput=()> +
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
    where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + BackwardAll<U,LossInput=()> +
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
        where P: ForwardAll<Input=I,Output=DiffInput<DiffArr<U,NI>,U,NI,NO>> + BackwardAll<U,LossInput=()> +
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
