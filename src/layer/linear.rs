//! Implementation of all full connected layers
use std::fmt::Debug;
use std::marker::{PhantomData};
use std::ops::{Mul};
use std::str::FromStr;
use crate::arr::{Arr, Arr2, IntoConverter};
use crate::{Cons, Stack};
use crate::cast::Assume;
use crate::device::{Device, DeviceBatchAveraging};
use crate::device::linear::{DeviceDiffLinear, DeviceLinear, DeviceQuantizedLinear};
use crate::error::{ModelLoadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError, TypeConvertError};
use crate::layer::{Backward, BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchForwardBase, BatchPreTrain, BatchPreTrainBase, BatchSize, ContinueForward, Forward, ForwardAll, ForwardDiff, PartialForward, PreTrain, UpdateWeight, OnStep, PersistProgress, InputTensorScalar, OutputTensorScalar, TensorSize, OutputTensorSize, InputTensorSize, InputScale, OutputScale, MaxInputValue};
use crate::ope::{MaxValue};
use crate::optimizer::{Optimizer, OptimizerBuilder};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, TextPersistence, TextRecord};
use crate::quantization::Quantizable;

/// Linear Layer Implementation
pub struct LinearLayer<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain + 
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,D>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    u:PhantomData<U>,
    parent:P,
    device:D,
    units:C,
    bias:BC,
    unit_optimizer:OP,
    bias_optimizer:OP,
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> InputTensorScalar
    for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,D>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type Scalar = U;
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> OutputTensorScalar
    for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,D>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type Scalar = U;
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    /// Create and return an instance of LinearLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    /// * `b` - optimizer builder
    pub fn new<UI,BI,B>(parent:P,device:&D,mut ui:UI,mut bi:BI, b:&B)
        -> Result<LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>,LayerInstantiationError>
        where UI: FnMut() -> U, BI: FnMut() -> U, B: OptimizerBuilder<U,D,Output=OP> {

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

        let units = device.specialization_units(units)?;
        let bias = device.specialization_bias(bias)?;

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
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> Persistence<TextFilePersistence,Specialized> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + Persistence<TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn load(&mut self, persistence: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
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

        self.units = self.device.specialization_units(units)?;
        self.bias = self.device.specialization_bias(bias)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write_layer_start();

        let bias = self.device.generalization_bias(&self.bias)?;
        let units = self.device.generalization_units(&self.units)?;

        persistence.write_units_start();

        for b in bias.iter() {
            persistence.write(*b);
        }

        for u in units.iter() {
            persistence.write_units_start();
            for w in u.iter() {
                persistence.write(*w);
            }
        }

        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> Persistence<T,Linear> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + Persistence<T,Linear>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
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

        self.units = self.device.specialization_units(units)?;
        self.bias = self.device.specialization_bias(bias)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        let bias = self.device.generalization_bias(&self.bias)?;
        let units = self.device.generalization_units(&self.units)?;

        for b in bias.iter() {
            persistence.write(*b)?;
        }

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
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + InputTensorSize<NI> +
              InputTensorScalar + OutputTensorScalar,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {

    fn forward(&self,input:&PI) -> Result<<Self as ForwardAll>::Output,EvaluateError> {
        self.device.forward_linear(&self.bias,&self.units,input.into())
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> ForwardAll for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar + OutputTensorScalar,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type Input = I;
    type Output = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> PreTrain for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: PreTrain<PreOutput=PI> +
             ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          PI: Debug + InputTensorSize<NI> +
              BatchDataType + InputTensorScalar + OutputTensorScalar +
              From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type PreOutput = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output;
    type OutStack = Cons<<P as PreTrain>::OutStack,Self::PreOutput>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r))?;

        Ok(Cons(r,u))
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize>
    Backward<U,&<D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output,Result<PI,TrainingError>> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          OP: Optimizer<U,D>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar + OutputTensorScalar +
              From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn backward(&mut self, input: &<D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output) -> Result<PI,TrainingError> {
        Ok(self.device.backward_linear(&self.units,input)?.into())
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BackwardAll<U> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: BackwardAll<U,LossInput=PI,LossInputScalar=U> + ForwardAll<Input=I,Output=PI> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          C: Debug,
          BC: Debug,
          OP: Optimizer<U,D>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar + OutputTensorScalar +
              From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO> + DeviceBatchAveraging<C,U> + DeviceBatchAveraging<BC,U>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a BC>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut BC> ,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type LossInputScalar = U;
    type LossInput = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all(&mut self, input: Self::LossInput, stack:Self::OutStack)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let next_loss = self.backward(&loss)?;

        let g = s.map(|o| {
            self.device.backward_weight_gradient(o.into(),&loss)
        })?;

        let bg = self.device.backward_bias_weight_gradient(loss)?;

        let (l,s) = self.parent.backward_all(next_loss.into(), s)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> PartialForward for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar + OutputTensorScalar,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          Self: ForwardAll<Input=I>,
          Self: PreTrain {
    type PartialInput = <P as PartialForward>::PartialInput;
    type PartialOutput = <P as PartialForward>::PartialOutput;
    type DiffInput = <P as PartialForward>::DiffInput;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward(input)?)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput)
        -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward_by_diff(input,partial_input)?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> ForwardDiff for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward + ForwardDiff +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar + OutputTensorScalar,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          Self: ForwardAll<Input=I,Output=<D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output> + PreTrain {
    fn forward_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_diff(input,partial_input)?;

        Ok(self.device.forward_linear(&self.bias,&self.units,&input.into())?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> ContinueForward for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward + ContinueForward +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar + OutputTensorScalar,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          Self: ForwardAll<Input=I,Output=<D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output> + PreTrain {
    fn continue_forward(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.continue_forward(input)?;

        Ok(self.device.forward_linear(&self.bias,&self.units,&input.into())?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> UpdateWeight for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> + UpdateWeight +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          C: Debug,
          BC: Debug,
          OP: Optimizer<U,D>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO> + DeviceBatchAveraging<C,U> + DeviceBatchAveraging<BC,U>,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a BC>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut BC> {
    type GradientStack = Cons<<P as UpdateWeight>::GradientStack,(C,BC)>;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        let g = self.device.batch_averaging(g,batch_size)?;
        let bg = self.device.batch_averaging(bg,batch_size)?;

        self.bias_optimizer.update((&bg).into(), (&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s,batch_size)?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchForwardBase for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          PI: Debug + InputTensorSize<NI> + BatchDataType + BatchDataType +
              InputTensorScalar + OutputTensorScalar,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput: Debug,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput;
}
impl<U,C,BC,P,OP,D,I,PI,const NI:usize,const NO:usize> BatchForward for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar + OutputTensorScalar +
              From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput> + BatchDataType,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput: Debug,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_forward_linear(&self.bias,&self.units,&input)?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchPreTrainBase for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,D>,
          <PI as BatchDataType>::Type: BatchSize,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar + OutputTensorScalar +
              From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput> + BatchDataType,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput: Debug,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          Self: PreTrain<PreOutput=<D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output> {
    type BatchPreOutput = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput;
    type BatchOutStack = Cons<<P as BatchPreTrainBase>::BatchOutStack,Self::BatchPreOutput>;
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchPreTrain for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: BatchSize,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar + OutputTensorScalar +
              From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput> + BatchDataType,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput: Debug,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| {
            self.device.batch_forward_linear(&self.bias,&self.units,input)
        })?;

        Ok(Cons(r,u))
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchBackward<U> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> + InputTensorScalar + OutputTensorScalar + BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          C: Debug,
          BC: Debug,
          PI: BatchDataType + InputTensorSize<NI> + InputTensorScalar + OutputTensorScalar,
          OP: Optimizer<U,D>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: BatchSize,
          <PI as BatchDataType>::Type: IntoConverter,
          <PI as BatchDataType>::Type: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError> + Debug,
          PI: Debug + From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::LossOutput> + BatchDataType,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO,BatchLossOutput=<PI as BatchDataType>::Type> +
             DeviceBatchAveraging<C,U> + DeviceBatchAveraging<BC,U>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput: Debug,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a C>,
          for<'a> &'a <OP as Optimizer<U,D>>::InternalType: From<&'a BC>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut C>,
          for<'a> <OP as Optimizer<U,D>>::InternalUpdateType<'a>: From<&'a mut BC> {
    type BatchLossInput = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::BatchOutput;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let next_loss = self.device.batch_backward_linear(&self.units, &loss)?;

        let g = s.map(|o| {
            self.device.batch_backward_weight_gradient(o, &loss)
        })?;

        let bg = self.device.batch_linear_reduce(&loss)?;

        let (l,s) = self.parent.batch_backward(next_loss.into_converter().try_into()?, s)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
// OnStep implementation for LinearLayer
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> OnStep for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          PI: Debug,
          OP: Optimizer<U,D>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        self.unit_optimizer.on_step(step)?;
        self.bias_optimizer.on_step(step)?;
        Ok(self.parent.on_step(step)?)
    }
    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        self.unit_optimizer.on_frequently_step(step,frequently_step)?;
        self.bias_optimizer.on_frequently_step(step,frequently_step)?;
        Ok(self.parent.on_frequently_step(step,frequently_step)?)
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> PersistProgress<TextFilePersistence,Specialized> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar +
             PersistProgress<TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          OP: Optimizer<U,D> + Persistence<TextFilePersistence,Specialized>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn load_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)?;

        self.unit_optimizer.load(persistence)?;
        self.bias_optimizer.load(persistence)?;

        Ok(())
    }

    fn save_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;

        persistence.write_layer_start();

        persistence.write_units_start();

        self.unit_optimizer.save(persistence)?;
        persistence.write_units_start();

        self.bias_optimizer.save(persistence)?;

        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> PersistProgress<T,Linear> for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar +
             PersistProgress<T,Linear>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          OP: Optimizer<U,D> + Persistence<T,Linear>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)?;

        self.unit_optimizer.load(persistence)?;
        self.bias_optimizer.load(persistence)?;

        Ok(())
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;

        self.unit_optimizer.save(persistence)?;
        self.bias_optimizer.save(persistence)?;

        Ok(())
    }
}
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> InputScale for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain + InputScale +
             InputTensorScalar + OutputTensorScalar,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      I: Debug + Send + Sync,
      PI: Debug + InputTensorSize<NI> + BatchDataType,
      OP: Optimizer<U,D>,
      D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
      [();NI]: TensorSize,
      [();NO]: TensorSize {
    fn scale_mean(&self) -> f32 {
        self.parent.scale_mean()
    }
}
pub trait LinearLayerInstantiation<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + InputTensorScalar + OutputTensorScalar,
          OP: Optimizer<U,D>,
          D: Device<U>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
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
impl<U,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> LinearLayerInstantiation<U,C,BC,P,D,I,PI,OP,NI,NO>
    for LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType + InputTensorScalar + OutputTensorScalar,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn instantiation<B: OptimizerBuilder<U,D,Output=OP>>(parent: P, device:&D, ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
        -> Result<LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>,LayerInstantiationError> {
        Ok(LinearLayer::<_,_,_,_,_,_,_,_,NI,NO>::new(parent,device,ui,bi,b)?)
    }
}
/// Builder for LinearLayer instance creation
pub struct LinearLayerBuilder<const NI:usize,const NO:usize>
    where [();NI]: TensorSize,
          [();NO]: TensorSize {
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>
}
impl<const NI:usize,const NO:usize> LinearLayerBuilder<NI,NO>
    where[();NI]: TensorSize,
         [();NO]: TensorSize {
    /// Create an instance of LinearLayerBuilder
    pub fn new() -> LinearLayerBuilder<NI,NO> {
        LinearLayerBuilder {
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>
        }
    }
}
impl<const NI:usize,const NO:usize> LinearLayerBuilder<NI,NO>
    where [();NI]: TensorSize,
          [();NO]: TensorSize {
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
        where P: ForwardAll<Input=I,Output=PI> +
                 BackwardAll<U,LossInput=PI,LossInputScalar=U> +
                 PreTrain<PreOutput=PI> +
                 InputTensorScalar + OutputTensorScalar,
              U: Default + Clone + Copy + Debug + Send + Sync + 'static,
              I: Debug + Send + Sync,
              PI: Debug + InputTensorSize<NI> + InputTensorScalar + OutputTensorScalar,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              D: Device<U>,
              LinearLayer<U,C,BC,P,D,I,PI,OP,NI,NO>: LinearLayerInstantiation<U,C,BC,P,D,I,PI,OP,NI,NO> {

        LinearLayer::instantiation(parent,device,ui,bi,b)
    }
}
/// Implementation of differentially applicable linear layers


/// Quantized Linear Layer Implementation
pub struct QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain + 
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    u:PhantomData<U>,
    w:PhantomData<W>,
    parent:P,
    device:D,
    units:C,
    bias:BC,
    qunits:<C as Quantizable<W>>::Quantized,
    qbias:<BC as Quantizable<W>>::Quantized,
    scale:<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Scale,
    scale_mean:f32,
    next_input_max:usize,
    shift:usize,
    unit_optimizer:OP,
    bias_optimizer:OP,
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> InputTensorScalar
    for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type Scalar = U;
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> OutputTensorScalar
    for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type Scalar = U;
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain + InputScale +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + Mul<Output=U> + MaxValue +
             Assume<f32> + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + MaxValue + Assume<U> + 'static,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          i32: From<U>,
          f32: Assume<W>,
          <C as Quantizable<W>>::Quantized: InputTensorScalar<Scalar=W>,
          <BC as Quantizable<W>>::Quantized: InputTensorScalar<Scalar=W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize
{
    /// Create and return an instance of QuantizedLinearLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    /// * `b` - optimizer builder
    pub fn new<UI, BI, B>(parent: P, device: &D, mut ui: UI, mut bi: BI, b: &B)
                          -> Result<QuantizedLinearLayer<U, W, C, BC, P, D, I, PI, OP, NI, NO>, LayerInstantiationError>
    where
        UI: FnMut() -> f32,
        BI: FnMut() -> f32,
        B: OptimizerBuilder<f32, D, Output=OP>
    {
        let mut units: Arr2<f32, NI, NO> = Arr2::new();
        let mut bias: Arr<f32, NO> = Arr::new();

        for mut it in units.iter_mut() {
            for it in it.iter_mut() {
                *it = ui();
            }
        }

        for it in bias.iter_mut() {
            *it = bi();
        }

        let shift = Self::calculate_shift(parent.max_input_value());

        let units = device.specialization_units(units)?;
        let bias = device.specialization_bias(bias)?;

        let (scale,scale_mean,qunits,qbias,next_input_max) = {
            device.quantization(&units, &bias, shift, parent.max_input_value())?
        };

        Ok(QuantizedLinearLayer {
            u: PhantomData::<U>,
            w: PhantomData::<W>,
            parent: parent,
            device: device.clone(),
            units: units,
            bias: bias,
            qunits: qunits,
            qbias: qbias,
            scale: scale,
            scale_mean: scale_mean,
            next_input_max: next_input_max,
            shift: shift as usize,
            unit_optimizer: b.build(NI * NO)?,
            bias_optimizer: b.build(NO)?
        })
    }

    fn calculate_shift(max_input_value: usize) -> u32 {
        let s = (NI as i32 * max_input_value as i32 * i32::from(W::max_value().assume())) as u32;

        let shift = 31 - s.leading_zeros();

        shift.saturating_sub(2)
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> Persistence<TextFilePersistence,Specialized> for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain + InputScale + Persistence<TextFilePersistence,Specialized> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + MaxValue + Mul<Output=U> + Assume<f32> + 'static + FromStr,
          W: Default + Clone + Copy + Debug + Send + Sync + MaxValue + Assume<U> + 'static + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> + BatchDataType,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          TextRecord: From<f32>,
          ModelLoadError: From<<f32 as FromStr>::Err>,
          i32: From<U> + From<W>,
          f32: Assume<W>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn load(&mut self, persistence: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)?;

        let mut bias = Arr::<f32,NO>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        let mut units = Arr2::<f32,NI,NO>::new();

        for mut u in units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }
        let shift = Self::calculate_shift(self.parent.max_input_value());

        let units = self.device.specialization_units(units)?;
        let bias = self.device.specialization_bias(bias)?;

        let (scale,scale_mean,qunits,qbias,next_input_max) = {
            self.device.quantization(&units, &bias, shift, self.parent.max_input_value())?
        };

        self.units = units;
        self.bias = bias;
        self.qunits = qunits;
        self.qbias = qbias;
        self.scale = scale;
        self.scale_mean = scale_mean;
        self.shift = shift as usize;
        self.next_input_max = next_input_max;

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write_layer_start();

        let bias = self.device.generalization_bias(&self.bias)?;
        let units = self.device.generalization_units(&self.units)?;

        persistence.write_units_start();

        for b in bias.iter() {
            persistence.write(*b);
        }

        for u in units.iter() {
            persistence.write_units_start();
            for w in u.iter() {
                persistence.write(*w);
            }
        }

        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> Persistence<T,Linear> for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where T: LinearPersistence<f32>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain + InputScale +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize> + Persistence<T,Linear>,
          U: Default + Clone + Copy + Debug + Send + Sync + Mul<Output=U> + MaxValue + Assume<f32> + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + MaxValue + Assume<U> + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> + BatchDataType,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          i32: From<U> + From<W>,
          f32: Assume<W>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        self.parent.load(persistence)?;

        let mut bias = Arr::<f32,NO>::new();

        for b in bias.iter_mut() {
            *b = persistence.read()?;
        }

        let mut units = Arr2::<f32,NI,NO>::new();

        for mut u in units.iter_mut() {
            for w in u.iter_mut() {
                *w = persistence.read()?;
            }
        }

        let shift = Self::calculate_shift(self.parent.max_input_value());

        let units = self.device.specialization_units(units)?;
        let bias = self.device.specialization_bias(bias)?;

        let (scale,scale_mean,qunits,qbias,next_input_max) = {
            self.device.quantization(&units, &bias, shift, self.parent.max_input_value())?
        };

        self.units = units;
        self.bias = bias;
        self.qunits = qunits;
        self.qbias = qbias;
        self.scale = scale;
        self.scale_mean = scale_mean;
        self.shift = shift as usize;
        self.next_input_max = next_input_max;

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        let bias = self.device.generalization_bias(&self.bias)?;
        let units = self.device.generalization_units(&self.units)?;

        for b in bias.iter() {
            persistence.write(*b)?;
        }

        for u in units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> Forward<PI,Result<<Self as ForwardAll>::Output,EvaluateError>>
    for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + InputTensorSize<NI> +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {

    fn forward(&self,input:&PI) -> Result<<Self as ForwardAll>::Output,EvaluateError> {
        self.device.forward_linear(self.shift,&self.qbias,&self.qunits,input.into())
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> ForwardAll for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type Input = I;
    type Output = <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> PreTrain for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: PreTrain<PreOutput=PI> +
             ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          OP: Optimizer<f32,D>,
          PI: Debug + InputTensorSize<NI> +
              BatchDataType + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> +
              From<<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::LossOutput>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type PreOutput = <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output;
    type OutStack = Cons<<P as PreTrain>::OutStack,Self::PreOutput>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let r = self.parent.pre_train(input)?;

        let u = r.map(|r| self.forward(r))?;

        Ok(Cons(r,u))
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize>
    Backward<U,&<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output,Result<PI,TrainingError>> for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          OP: Optimizer<f32,D>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> +
              From<<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::LossOutput>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn backward(&mut self, input: &<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output) -> Result<PI,TrainingError> {
        Ok(self.device.backward_linear(&self.units,input,self.scale_mean,&self.scale)?.into())
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BackwardAll<U> for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: BackwardAll<U,LossInput=PI,LossInputScalar=U> + ForwardAll<Input=I,Output=PI> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + InputScale + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + MaxValue + Assume<f32> + Mul<Output=U> + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + MaxValue + Assume<U> + 'static,
          I: Debug + Send + Sync,
          C: Debug,
          BC: Debug,
          OP: Optimizer<f32,D>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> +
              From<<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::LossOutput>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO> +
             DeviceBatchAveraging<C,f32> + DeviceBatchAveraging<BC,f32>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          f32: Assume<W>,
          i32: From<U>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          for<'a> &'a <OP as Optimizer<f32,D>>::InternalType: From<&'a C>,
          for<'a> &'a <OP as Optimizer<f32,D>>::InternalType: From<&'a BC>,
          for<'a> <OP as Optimizer<f32,D>>::InternalUpdateType<'a>: From<&'a mut C>,
          for<'a> <OP as Optimizer<f32,D>>::InternalUpdateType<'a>: From<&'a mut BC> ,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type LossInputScalar = U;
    type LossInput = <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all(&mut self, input: Self::LossInput, stack:Self::OutStack)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let next_loss = self.backward(&loss)?;

        let g = s.map(|o| {
            self.device.backward_weight_gradient(o.into(),&loss,self.parent.scale_mean(),&self.scale)
        })?;

        let bg = self.device.backward_bias_weight_gradient(loss,&self.scale)?;

        let (l,s) = self.parent.backward_all(next_loss.into(), s)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> PartialForward for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          Self: ForwardAll<Input=I>,
          Self: PreTrain {
    type PartialInput = <P as PartialForward>::PartialInput;
    type PartialOutput = <P as PartialForward>::PartialOutput;
    type DiffInput = <P as PartialForward>::DiffInput;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward(input)?)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput)
        -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.parent.partial_forward_by_diff(input,partial_input)?)
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> ForwardDiff for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward + ForwardDiff +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          Self: ForwardAll<Input=I,Output=<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output> + PreTrain {
    fn forward_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_diff(input,partial_input)?;

        Ok(self.device.forward_linear(self.shift,&self.qbias,&self.qunits,&input.into())?)
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> ContinueForward for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + PartialForward + ContinueForward +
             BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          Self: ForwardAll<Input=I,Output=<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output> + PreTrain {
    fn continue_forward(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.continue_forward(input)?;

        Ok(self.device.forward_linear(self.shift,&self.qbias,&self.qunits,&input.into())?)
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> UpdateWeight for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> + UpdateWeight +
             InputTensorScalar + OutputTensorScalar + InputScale + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + Mul<Output=U> + MaxValue + Assume<f32> + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + MaxValue + Assume<U> + 'static,
          I: Debug + Send + Sync,
          C: Debug,
          BC: Debug,
          OP: Optimizer<f32,D>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> +
              From<<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::LossOutput>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO> + DeviceBatchAveraging<C,f32> + DeviceBatchAveraging<BC,f32>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          i32: From<U>,
          f32: Assume<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          <C as Quantizable<W>>::Quantized: InputTensorScalar<Scalar=W>,
          <BC as Quantizable<W>>::Quantized: InputTensorScalar<Scalar=W>,
          for<'a> &'a <OP as Optimizer<f32,D>>::InternalType: From<&'a C>,
          for<'a> &'a <OP as Optimizer<f32,D>>::InternalType: From<&'a BC>,
          for<'a> <OP as Optimizer<f32,D>>::InternalUpdateType<'a>: From<&'a mut C>,
          for<'a> <OP as Optimizer<f32,D>>::InternalUpdateType<'a>: From<&'a mut BC> {
    type GradientStack = Cons<<P as UpdateWeight>::GradientStack,(C,BC)>;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        let g = self.device.batch_averaging(g,batch_size)?;
        let bg = self.device.batch_averaging(bg,batch_size)?;

        self.bias_optimizer.update((&bg).into(), (&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        let r = self.parent.update_weight(s,batch_size)?;

        let shift = Self::calculate_shift(self.parent.max_input_value());

        let (scale,scale_mean,qunits,qbias,next_input_max) = {
            self.device.quantization(&self.units, &self.bias, shift as u32, self.parent.max_input_value())?
        };

        self.scale = scale;
        self.scale_mean = scale_mean;
        self.qunits = qunits;
        self.qbias = qbias;
        self.next_input_max = next_input_max;

        Ok(r)
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchForwardBase for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          <I as BatchDataType>::Type: Debug,
          PI: Debug + InputTensorSize<NI> + BatchDataType + BatchDataType +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::BatchOutput: Debug,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type BatchInput = <I as BatchDataType>::Type;
    type BatchOutput = <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::BatchOutput;
}
impl<U,W,C,BC,P,OP,D,I,PI,const NI:usize,const NO:usize> BatchForward for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<f32,D>,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> +
              From<<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::LossOutput> + BatchDataType,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::BatchOutput: Debug,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_forward_linear(self.shift,&self.qbias,&self.qunits,&input)?)
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchPreTrainBase for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> + BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<f32,D>,
          <PI as BatchDataType>::Type: BatchSize,
          <I as BatchDataType>::Type: Debug,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> +
              From<<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::LossOutput> + BatchDataType,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::BatchOutput: Debug,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          Self: PreTrain<PreOutput=<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output> {
    type BatchPreOutput = <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::BatchOutput;
    type BatchOutStack = Cons<<P as BatchPreTrainBase>::BatchOutStack,Self::BatchPreOutput>;
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchPreTrain for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync + BatchDataType,
          OP: Optimizer<f32,D>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: BatchSize,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          PI: Debug + InputTensorSize<NI> + BatchDataType +
              InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> +
              From<<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::LossOutput> + BatchDataType,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::BatchOutput: Debug,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| {
            self.device.batch_forward_linear(self.shift,&self.qbias,&self.qunits,input)
        })?;

        Ok(Cons(r,u))
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> BatchBackward<U> for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> + InputTensorScalar + OutputTensorScalar + InputScale + MaxInputValue<Scalar=usize> +
             BatchForwardBase<BatchInput=<I as BatchDataType>::Type,BatchOutput=<PI as BatchDataType>::Type> +
             BatchForward +
             BatchPreTrainBase<BatchPreOutput=<PI as BatchDataType>::Type> +
             BatchPreTrain +
             BatchBackward<U,BatchLossInput=<PI as BatchDataType>::Type>,
          U: Default + Clone + Copy + Debug + Mul<Output=U> + MaxValue + Assume<f32> + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + MaxValue + Assume<U> + 'static,
          I: Debug + Send + Sync + BatchDataType,
          C: Debug,
          BC: Debug,
          PI: BatchDataType + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          <I as BatchDataType>::Type: Debug,
          <PI as BatchDataType>::Type: BatchSize,
          <PI as BatchDataType>::Type: IntoConverter,
          <PI as BatchDataType>::Type: TryFrom<<<PI as BatchDataType>::Type as IntoConverter>::Converter,Error=TypeConvertError> + Debug,
          PI: Debug + From<<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::LossOutput> + BatchDataType,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO> +
             DeviceBatchAveraging<C,f32> + DeviceBatchAveraging<BC,f32>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::BatchOutput: Debug,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::BatchLossOutput: IntoConverter,
          <PI as BatchDataType>::Type: TryFrom<<<D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::BatchLossOutput as IntoConverter>::Converter,Error=TypeConvertError>,
          [();NI]: TensorSize,
          [();NO]: TensorSize,
          f32: Assume<W>,
          i32: From<U>,
          for<'a> &'a <OP as Optimizer<f32,D>>::InternalType: From<&'a C>,
          for<'a> &'a <OP as Optimizer<f32,D>>::InternalType: From<&'a BC>,
          for<'a> <OP as Optimizer<f32,D>>::InternalUpdateType<'a>: From<&'a mut C>,
          for<'a> <OP as Optimizer<f32,D>>::InternalUpdateType<'a>: From<&'a mut BC> {
    type BatchLossInput = <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::BatchOutput;
    type BatchLossOutput = <P as BatchBackward<U>>::BatchLossOutput;

    fn batch_backward(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        let next_loss = self.device.batch_backward_linear(&self.units, &loss, self.parent.scale_mean(),&self.scale)?;

        let g = s.map(|o| {
            self.device.batch_backward_weight_gradient(o, &loss, self.scale_mean, &self.scale)
        })?;

        let bg = self.device.batch_backward_bias_gradient(&loss,&self.scale)?;

        let (l,s) = self.parent.batch_backward(next_loss.into_converter().try_into()?, s)?;

        Ok((l,Cons(s,(g,bg))))
    }
}
// OnStep implementation for QuantizedLinearLayer
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> OnStep for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize> + OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> + BatchDataType,
          OP: Optimizer<f32,D>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        self.unit_optimizer.on_step(step)?;
        self.bias_optimizer.on_step(step)?;
        Ok(self.parent.on_step(step)?)
    }
    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        self.unit_optimizer.on_frequently_step(step,frequently_step)?;
        self.bias_optimizer.on_frequently_step(step,frequently_step)?;
        Ok(self.parent.on_frequently_step(step,frequently_step)?)
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> PersistProgress<TextFilePersistence,Specialized> for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize> +
             PersistProgress<TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> + BatchDataType,
          OP: Optimizer<f32,D> + Persistence<TextFilePersistence,Specialized>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn load_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)?;

        self.unit_optimizer.load(persistence)?;
        self.bias_optimizer.load(persistence)?;

        Ok(())
    }

    fn save_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;

        persistence.write_layer_start();

        persistence.write_units_start();

        self.unit_optimizer.save(persistence)?;
        persistence.write_units_start();

        self.bias_optimizer.save(persistence)?;

        persistence.write_layer_end();

        Ok(())
    }
}
impl<T,U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> PersistProgress<T,Linear> for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize> +
             PersistProgress<T,Linear>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> + BatchDataType,
          OP: Optimizer<f32,D> + Persistence<T,Linear>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)?;

        self.unit_optimizer.load(persistence)?;
        self.bias_optimizer.load(persistence)?;

        Ok(())
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;

        self.unit_optimizer.save(persistence)?;
        self.bias_optimizer.save(persistence)?;

        Ok(())
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> InputScale for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + InputTensorSize<NI> +
          InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn scale_mean(&self) -> f32 {
        self.scale_mean
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> OutputScale for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      W: Default + Clone + Copy + Debug + Send + Sync + 'static,
      I: Debug + Send + Sync,
      PI: Debug + BatchDataType + InputTensorSize<NI> +
      InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
      OP: Optimizer<f32,D>,
      D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
      <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
      C: Quantizable<W>,
      BC: Quantizable<W>,
      [();NI]: TensorSize,
      [();NO]: TensorSize {
    type Scale = <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Scale;
    fn scale(&self) -> &Self::Scale {
        &self.scale
    }
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> MaxInputValue for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + BatchDataType + InputTensorSize<NI> +
          InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          <D as DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    type Scalar = usize;

    fn max_input_value(&self) -> Self::Scalar {
        self.next_input_max
    }
}
/// Trait for QuantizedLinearLayer instance creation
pub trait QuantizedLinearLayerInstantiation<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> + BatchDataType,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    /// Create an instance of QuantizedLinearLayers
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
    fn instantiation<B: OptimizerBuilder<f32,D,Output=OP>>(parent:P,device:&D,ui: impl FnMut() -> f32, bi: impl FnMut() -> f32, b: &B)
        -> Result<QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>,LayerInstantiationError>;
}
impl<U,W,C,BC,P,D,I,PI,OP,const NI:usize,const NO:usize> QuantizedLinearLayerInstantiation<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    for QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=PI,LossInputScalar=U> +
             PreTrain<PreOutput=PI> + InputScale +
             InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
          U: Default + Clone + Copy + Debug + Send + Sync + Mul<Output=U> + MaxValue + Assume<f32> + 'static,
          W: Default + Clone + Copy + Debug + Send + Sync + MaxValue + Assume<U> + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U>,
          OP: Optimizer<f32,D>,
          D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
          C: Quantizable<W>,
          BC: Quantizable<W>,
          <C as Quantizable<W>>::Quantized: InputTensorScalar<Scalar=W>,
          <BC as Quantizable<W>>::Quantized: InputTensorScalar<Scalar=W>,
          i32: From<U> + From<W>,
          f32: Assume<W>,
          [();NI]: TensorSize,
          [();NO]: TensorSize {
    fn instantiation<B: OptimizerBuilder<f32,D,Output=OP>>(parent: P, device:&D, ui: impl FnMut() -> f32, bi: impl FnMut() -> f32, b: &B)
        -> Result<QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>,LayerInstantiationError> {
        Ok(QuantizedLinearLayer::<_,_,_,_,_,_,_,_,_,NI,NO>::new(parent,device,ui,bi,b)?)
    }
}
/// Builder for QuantizedLinearLayer instance creation
pub struct QuantizedLinearLayerBuilder<W,const NI:usize,const NO:usize>
    where [();NI]: TensorSize,
          [();NO]: TensorSize {
    w:PhantomData<W>,
    ni:PhantomData<[();NI]>,
    no:PhantomData<[();NO]>
}
impl<W,const NI:usize,const NO:usize> QuantizedLinearLayerBuilder<W,NI,NO>
    where[();NI]: TensorSize,
         [();NO]: TensorSize {
    /// Create an instance of QuantizedLinearLayerBuilder
    pub fn new() -> QuantizedLinearLayerBuilder<W,NI,NO> {
        QuantizedLinearLayerBuilder {
            w:PhantomData::<W>,
            ni:PhantomData::<[();NI]>,
            no:PhantomData::<[();NO]>
        }
    }
}
impl<W,const NI:usize,const NO:usize> QuantizedLinearLayerBuilder<W,NI,NO>
    where [();NI]: TensorSize,
          [();NO]: TensorSize {
    /// Create an instance of QuantizedLinearLayers
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
    pub fn build<U,C,BC,P,D,I,PI,OP,B>(&self,parent: P, device:&D, ui: impl FnMut() -> f32, bi: impl FnMut() -> f32, b: &B)
        -> Result<QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> +
                 BackwardAll<U,LossInput=PI,LossInputScalar=U> +
                 PreTrain<PreOutput=PI> +
                 InputTensorScalar + OutputTensorScalar + MaxInputValue<Scalar=usize>,
              U: Default + Clone + Copy + Debug + Send + Sync + 'static,
              W: Default + Clone + Copy + Debug + Send + Sync + 'static,
              I: Debug + Send + Sync,
              PI: Debug + InputTensorSize<NI> + InputTensorScalar<Scalar=U> + OutputTensorScalar<Scalar=U> + BatchDataType,
              OP: Optimizer<f32,D>,
              D: Device<U> + Device<f32> + DeviceQuantizedLinear<U,W,C,BC,PI,NI,NO>,
              C: Quantizable<W>,
              BC: Quantizable<W>,
              B: OptimizerBuilder<f32,D,Output=OP>,
              i32: From<W>,
              [();NI]: TensorSize,
              [();NO]: TensorSize,
              QuantizedLinearLayer<U,W,C,BC,P,D,I,PI,OP,NI,NO>: QuantizedLinearLayerInstantiation<U,W,C,BC,P,D,I,PI,OP,NI,NO> {

        QuantizedLinearLayer::instantiation(parent,device,ui,bi,b)
    }
}
/// Implementation of differentially applicable linear layers

pub struct DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=PI> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug,
          OP: Optimizer<U,D> {
    u:PhantomData<U>,
    pi:PhantomData<PI>,
    di:PhantomData<DI>,
    l:PhantomData<&'a ()>,
    parent:P,
    device:D,
    units:C,
    bias:BC,
    unit_optimizer: OP,
    bias_optimizer: OP
}
impl<'a,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize> InputTensorScalar
    for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug,
          OP: Optimizer<U,D> {
    type Scalar = U;
}
impl<'a,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize> OutputTensorScalar
    for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug,
          OP: Optimizer<U,D> {
    type Scalar = U;
}
impl<'a,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize> DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=()> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          DI: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO> {
    /// Create and return an instance of DiffLinearLayer
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    /// * `bi` - Callback to generate weight of bias
    /// * `b` - optimizer builder
    pub fn new<UI,BI,B>(parent:P,device:&D,mut ui:UI,mut bi:BI, b: &B)
        -> Result<DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>,LayerInstantiationError>
        where UI: FnMut() -> U, BI: FnMut() -> U, B: OptimizerBuilder<U,D,Output=OP> {

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

        let units = device.specialization_units(units)?;
        let bias = device.specialization_bias(bias)?;

        Ok(DiffLinearLayer {
            u:PhantomData::<U>,
            pi:PhantomData::<PI>,
            di:PhantomData::<DI>,
            l:PhantomData::<&'a ()>,
            parent:parent,
            device:device.clone(),
            units: units,
            bias:bias,
            unit_optimizer:b.build(NI*NO)?,
            bias_optimizer:b.build(NO)?
        })
    }
}
impl<'a,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize> Persistence<TextFilePersistence,Specialized>
    for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=()> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             Persistence<TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          DI: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence) -> Result<(), ModelLoadError> {
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

        self.bias = self.device.specialization_bias(bias)?;
        self.units = self.device.specialization_units(units)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write_layer_start();

        let bias = self.device.generalization_bias(&self.bias)?;
        let units = self.device.generalization_units(&self.units)?;

        persistence.write_units_start();

        for b in bias.iter() {
            persistence.write(*b);
        }

        for u in units.iter() {
            persistence.write_units_start();
            for w in u.iter() {
                persistence.write(*w);
            }
        }

        persistence.write_layer_end();

        Ok(())
    }
}
impl<'a,T,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize> Persistence<T,Linear>
    for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=()> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             Persistence<T,Linear>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          DI: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO> {
    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
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

        self.bias = self.device.specialization_bias(bias)?;
        self.units = self.device.specialization_units(units)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        let bias = self.device.generalization_bias(&self.bias)?;
        let units = self.device.generalization_units(&self.units)?;

        for b in bias.iter() {
            persistence.write(*b)?;
        }

        for u in units.iter() {
            for w in u.iter() {
                persistence.write(*w)?;
            }
        }

        Ok(())
    }
}
impl<'a,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize> ForwardAll for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=()> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceDiffLinear<'a,U,DI,C,NI,NO> + DeviceLinear<U,C,BC,PI,NI,NO> {
    type Input = I;
    type Output = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output;

    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        let input = self.parent.forward_all(input)?;

        Ok(self.device.forward_linear(&self.bias,&self.units,&input)?)
    }
}
impl<'a,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize> PreTrain for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: PreTrain<PreOutput=PI> +
             ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=()> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceDiffLinear<'a,U,DI,C,NI,NO> + DeviceLinear<U,C,BC,PI,NI,NO>,
          <D as DeviceDiffLinear<'a,U,DI,C,NI,NO>>::Output: OutputTensorSize<NO> + Debug + 'static {
    type PreOutput = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output;
    type OutStack = Cons<<P as PreTrain>::OutStack,Self::PreOutput>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let s = self.parent.pre_train(input)?;

        let u = s.map(|input| {
            self.device.forward_linear(&self.bias,&self.units,input)
        })?;

        Ok(Cons(s,u))
    }
}
impl<'a,U,C,BC,P,D,OP,I,DI,PI,const NI:usize,const NO:usize> BackwardAll<U> for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: BackwardAll<U,LossInput=(),LossInputScalar=U> +
             ForwardAll<Input=I,Output=PI> +
             PreTrain<PreOutput=PI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          C: Debug,
          BC: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceDiffLinear<'a,U,DI,C,NI,NO> + DeviceLinear<U,C,BC,PI,NI,NO> +
             DeviceBatchAveraging<C,U> + DeviceBatchAveraging<BC,U>,
          <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output: OutputTensorSize<NO> + Debug + 'static,
          BC: From<<D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output>,
          for<'b> &'b <OP as Optimizer<U,D>>::InternalType: From<&'b C>,
          for<'b> &'b <OP as Optimizer<U,D>>::InternalType: From<&'b BC>,
          for<'b> <OP as Optimizer<U,D>>::InternalUpdateType<'b>: From<&'b mut C>,
          for<'b> <OP as Optimizer<U,D>>::InternalUpdateType<'b>: From<&'b mut BC>,
          Self: ForwardAll + PreTrain<OutStack=Cons<<P as PreTrain>::OutStack,BC>> {
    type LossInputScalar = U;
    type LossInput = <D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output;
    type LossOutput = <P as BackwardAll<U>>::LossOutput;

    fn backward_all(&mut self, input: Self::LossInput, stack:Self::OutStack)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError> {
        let (s,_) = stack.pop();

        let loss = input;

        let g = s.map(|o| {
            self.device.backward_weight_gradient(o,&loss)
        })?;

        let bg = loss;

        let (l,s) = self.parent.backward_all((), s)?;

        Ok((l,Cons(s,(g,bg.into()))))
    }
}
impl<'a,U,C,BC,P,D,OP,I,DI,PI,const NI:usize,const NO:usize> UpdateWeight for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             PreTrain + UpdateWeight +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          C: Debug,
          BC: Debug,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceBatchAveraging<C,U> + DeviceBatchAveraging<BC,U>,
          for<'b> &'b <OP as Optimizer<U,D>>::InternalType: From<&'b C>,
          for<'b> &'b <OP as Optimizer<U,D>>::InternalType: From<&'b BC>,
          for<'b> <OP as Optimizer<U,D>>::InternalUpdateType<'b>: From<&'b mut C>,
          for<'b> <OP as Optimizer<U,D>>::InternalUpdateType<'b>: From<&'b mut BC> {
    type GradientStack = Cons<<P as UpdateWeight>::GradientStack,(C,BC)>;

    fn update_weight(&mut self, stack: Self::GradientStack, batch_size: usize) -> Result<(), TrainingError> {
        let (s,(g,bg)) = stack.pop();

        let g = self.device.batch_averaging(g,batch_size)?;
        let bg = self.device.batch_averaging(bg,batch_size)?;

        self.bias_optimizer.update((&bg).into(),(&mut self.bias).into())?;
        self.unit_optimizer.update((&g).into(),(&mut self.units).into())?;

        Ok(self.parent.update_weight(s,batch_size)?)
    }
}
impl<'a,U,C,BC,P,D,OP,I,DI,PI,const NI:usize,const NO:usize> PartialForward for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: BackwardAll<U,LossInput=()> +
             ForwardAll<Input=I,Output=PI> +
             PreTrain<PreOutput=PI> +
             PartialForward<DiffInput=DI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO,Output=<D as DeviceDiffLinear<'a,U,DI,C,NI,NO>>::Output> +
             DeviceDiffLinear<'a,U,DI,C,NI,NO>,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          OP: Optimizer<U,D>,
          Self: ForwardAll<Input=I> +
                PreTrain {
    type PartialInput = <D as DeviceDiffLinear<'a,U,DI,C,NI,NO>>::Output;
    type PartialOutput = <D as DeviceDiffLinear<'a,U,DI,C,NI,NO>>::Output;
    type DiffInput = DI;

    fn partial_forward(&self, input: Self::Input) -> Result<Self::PartialOutput, EvaluateError> {
        let input = self.parent.forward_all(input)?;
        Ok(self.device.forward_linear(&self.bias,&self.units,&input)?)
    }

    fn partial_forward_by_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput)
        -> Result<Self::PartialOutput, EvaluateError> {
        Ok(self.device.forward_diff_linear(&self.units,input,partial_input)?)
    }
}
impl<'a,U,C,BC,P,D,OP,I,DI,PI,const NI:usize,const NO:usize> ForwardDiff for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: BackwardAll<U,LossInput=()> +
             ForwardAll<Input=I,Output=PI> +
             PreTrain<PreOutput=PI> +
             PartialForward<DiffInput=DI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceDiffLinear<'a,U,DI,C,NI,NO> +
          DeviceLinear<U,C,BC,PI,NI,NO,Output=<D as DeviceDiffLinear<'a,U,DI,C,NI,NO>>::Output>,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          OP: Optimizer<U,D>,
          Self: ForwardAll<Input=I,Output=<D as DeviceLinear<U,C,BC,PI,NI,NO>>::Output> +
                PreTrain {
    fn forward_diff(&self, input: Self::DiffInput, partial_input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        Ok(self.device.forward_diff_linear(&self.units,input,partial_input)?)
    }
}
impl<'a,U,C,BC,P,D,OP,I,DI,PI,const NI:usize,const NO:usize> ContinueForward for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: BackwardAll<U,LossInput=()> +
             ForwardAll<Input=I,Output=PI> +
             PreTrain<PreOutput=PI> +
             PartialForward<DiffInput=DI> +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U> + DeviceDiffLinear<'a,U,DI,C,NI,NO> +
             DeviceLinear<U,C,BC,PI,NI,NO,Output=<D as DeviceDiffLinear<'a,U,DI,C,NI,NO>>::Output>,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          OP: Optimizer<U,D>,
          Self: ForwardAll<Input=I,Output=Self::PartialInput> +
                PreTrain {
    fn continue_forward(&self, input: &Self::PartialInput) -> Result<Self::Output, EvaluateError> {
        Ok(self.device.clone_diff_linear_forward_output(input)?)
    }
}
// OnStep implementation for DiffLinearLayer
impl<'a,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize> OnStep for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + PreTrain +
             InputTensorScalar + OutputTensorScalar + OnStep,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: Device<U>,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug,
          OP: Optimizer<U,D> {
    fn on_step(&mut self, step: usize) -> Result<(), TrainingError> {
        self.unit_optimizer.on_step(step)?;
        self.bias_optimizer.on_step(step)?;
        Ok(self.parent.on_step(step)?)
    }
    
    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError> {
        self.unit_optimizer.on_frequently_step(step,frequently_step)?;
        self.bias_optimizer.on_frequently_step(step,frequently_step)?;
        Ok(self.parent.on_frequently_step(step,frequently_step)?)
    }
}
impl<'a,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize> PersistProgress<TextFilePersistence,Specialized>
    for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=()> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             PersistProgress<TextFilePersistence,Specialized>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static + FromStr,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          DI: Debug,
          OP: Optimizer<U,D> + Persistence<TextFilePersistence,Specialized>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO>,
          TextRecord: From<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)?;

        self.unit_optimizer.load(persistence)?;
        self.bias_optimizer.load(persistence)?;

        Ok(())
    }

    fn save_progress(&mut self, persistence: &mut TextFilePersistence) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;

        persistence.write_layer_start();
        persistence.write_units_start();

        self.unit_optimizer.save(persistence)?;

        persistence.write_units_start();

        self.bias_optimizer.save(persistence)?;

        persistence.write_layer_end();

        Ok(())
    }
}
impl<'a,T,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize> PersistProgress<T,Linear>
    for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=()> + PreTrain +
             InputTensorScalar + OutputTensorScalar +
             PersistProgress<T,Linear>,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          DI: Debug,
          OP: Optimizer<U,D> + Persistence<T,Linear>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO> {
    fn load_progress(&mut self, persistence: &mut T) -> Result<(), TrainingError> {
        self.parent.load_progress(persistence)?;

        self.unit_optimizer.load(persistence)?;
        self.bias_optimizer.load(persistence)?;

        Ok(())
    }

    fn save_progress(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save_progress(persistence)?;

        self.unit_optimizer.save(persistence)?;
        self.bias_optimizer.save(persistence)?;

        Ok(())
    }
}
impl<'a,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize> InputScale
    for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> +
             BackwardAll<U,LossInput=()> +
             PreTrain<PreOutput=PI> + InputScale +
             InputTensorScalar + OutputTensorScalar,
      U: Default + Clone + Copy + Debug + Send + Sync + 'static,
      I: Debug + Send + Sync,
      DI: Debug,
      PI: Debug + InputTensorSize<NI> + BatchDataType,
      OP: Optimizer<U,D>,
      D: Device<U> + DeviceDiffLinear<'a,U,DI,C,NI,NO> + DeviceLinear<U,C,BC,PI,NI,NO> {
    fn scale_mean(&self) -> f32 {
        self.parent.scale_mean()
    }
}
/// Trait for DiffLinearLayer instance creation
pub trait DiffLinearLayerInstantiation<'a,U,C,BC,P,OP,D,I,DI,PI,const NI:usize,const NO:usize>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=()> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
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
        -> Result<DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>,LayerInstantiationError>;
}
impl<'a,U,C,BC,P,D,OP,I,DI,PI,const NI:usize,const NO:usize> DiffLinearLayerInstantiation<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    for DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>
    where P: ForwardAll<Input=I,Output=PI> + BackwardAll<U,LossInput=()> +
             PreTrain +
             InputTensorScalar + OutputTensorScalar,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: Debug + Send + Sync,
          DI: Debug,
          PI: Debug + InputTensorSize<NI> + BatchDataType,
          OP: Optimizer<U,D>,
          D: Device<U> + DeviceLinear<U,C,BC,PI,NI,NO> {
    fn instantiation<UI: FnMut() -> U, BI: FnMut() -> U, B: OptimizerBuilder<U,D,Output=OP>>(parent: P, device:&D,ui: UI, bi: BI, b: &B)
        -> Result<DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>,LayerInstantiationError> {
        Ok(DiffLinearLayer::<'a,_,_,_,_,_,_,_,_,PI,NI,NO>::new(parent,device,ui,bi,b)?)
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
    pub fn build<'a,U,C,BC,P,OP,B,D,I,DI,PI>(&self,parent: P, device:&D, ui: impl FnMut() -> U, bi: impl FnMut() -> U, b: &B)
                 -> Result<DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>,LayerInstantiationError>
        where P: ForwardAll<Input=I,Output=PI> +
                 BackwardAll<U,LossInput=()> +
                 PreTrain +
                 InputTensorScalar + OutputTensorScalar,
              U: Default + Clone + Copy + Debug + Send + Sync + 'static,
              I: Debug + Send + Sync,
              DI: Debug,
              PI: Debug + InputTensorSize<NI> + BatchDataType,
              D: Device<U>,
              OP: Optimizer<U,D>,
              B: OptimizerBuilder<U,D,Output=OP>,
              DiffLinearLayer<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO>: DiffLinearLayerInstantiation<'a,U,C,BC,P,OP,D,I,DI,PI,NI,NO> {

        DiffLinearLayer::instantiation(parent,device,ui,bi,b)
    }
}
