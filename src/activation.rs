//! Activation Function Implementation

use std::collections::HashSet;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_uint;
use cuda_runtime_sys::dim3;
use rayon::prelude::{FromParallelIterator, IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use crate::UnitValue;
use crate::arr::*;
use crate::cuda::{CudaPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, DataTypeInfo, Kernel};
use crate::cuda::kernel::activation::{ActivationBackwardArgs, ActivationBatchBackwardArgs, ActivationBatchForwardArgs, ActivationForwardArgs, ReLuBackward, ReLuBatchBackward, ReLuForward, ReLuBatchForward, SigmoidBackward, SigmoidBatchBackward, SigmoidForward, SigmoidBatchForward, SoftMaxBackward, SoftMaxBatchBackward, SoftMaxForward, SoftMaxBatchForward, SwishBackward, SwishBatchBackward, SwishForward, TanhBackward, TanhBatchBackward, TanhForward, TanhBatchForward, SwishBatchForward};
use crate::device::*;
use crate::error::{CudaError, EvaluateError, TrainingError};
use crate::layer::BatchSize;
use crate::lossfunction::LossFunction;

/// Trait defining activation functions
pub trait Activation<U,T,R,D> where U: UnitValue<U>, D: Device<U> {
    /// Apply the activation function
    /// # Arguments
    /// * `device` - Device objects available for processing
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn apply(&self, device:&D, input:&T) -> Result<R, EvaluateError>;
    /// Apply derivatives of the activation function
    /// # Arguments
    /// * `device` - Device objects available for processing
    /// * `o` - Input from upper layers
    /// * `loss` - Losses calculated at lower tiers
    /// * `u` - Value before passing through the activation function of the input from the upper layer
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn derive(&self, device:&D, o:&T, loss:&T, u:&T) -> Result<R, TrainingError>;
    /// Returns whether or not the canonical linkage function can be used.
    /// # Arguments
    /// * `l` - loss function
    fn is_canonical_link<L: LossFunction<U>>(&self,l:&L) -> bool;
}

/// Trait that defines the activation function during batch processing
pub trait BatchActivation<U,T,R,D> where U: UnitValue<U>, D: Device<U> {
    /// Apply the activation function
    /// # Arguments
    /// * `device` - Device objects available for processing
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_apply(&self, device:&D, input:&T) -> Result<R, TrainingError>;
    /// Apply derivatives of the activation function
    /// # Arguments
    /// * `device` - Device objects available for processing
    /// * `o` - Input from upper layers
    /// * `loss` - Losses calculated at lower tiers
    /// * `u` - Value before passing through the activation function of the input from the upper layer
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_derive(&self, device:&D, o:&T, loss:&T, u:&T) -> Result<R, TrainingError>;
}
/// Identity Implementation
pub struct Identity<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>,
    c:HashSet<&'static str>
}
impl<U,D> Identity<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of Identity
    pub fn new(_:&D) -> Identity<U,D> {
        let mut c = HashSet::new();
        c.insert("mse");

        Identity {
            u: PhantomData::<U>,
            d:PhantomData::<D>,
            c:c
        }
    }
}
impl<U,I,const N:usize> Activation<U,I,Arr<U,N>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
    where U: UnitValue<U>, I: Iterator<Item=U> + Clone {

    fn apply(&self, _: &DeviceCpu<U>, input: &I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &I, loss: &I, _: &I) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().collect::<Vec<U>>().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        Ok((*input).clone())
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, _: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        Ok((*loss).clone())
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        Ok((*input).clone().into())
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, _: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        Ok((*loss).clone().into())
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,const N:usize> Activation<U,CudaTensor1dPtrView<'a,U,N>,CudaTensor1dPtr<U,N>,DeviceGpu<U>> for Identity<U,DeviceGpu<U>>
    where U: UnitValue<U>, DeviceGpu<U>: Device<U> {

    fn apply(&self, _: &DeviceGpu<U>, input: &CudaTensor1dPtrView<'a,U,N>) -> Result<CudaTensor1dPtr<U,N>, EvaluateError> {
        Ok(input.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>,
              _: &CudaTensor1dPtrView<'a,U,N>, loss: &CudaTensor1dPtrView<'a,U,N>, _: &CudaTensor1dPtrView<U,N>)
        -> Result<CudaTensor1dPtr<U,N>, TrainingError> {
        Ok(loss.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> BatchActivation<U,SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, _: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok((*input).clone())
    }

    fn batch_derive(&self, _: &DeviceCpu<U>, _: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, _: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok((*loss).clone())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,SerializedVecView<'a,U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, _: &DeviceCpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.into())
    }

    fn batch_derive(&self, _: &DeviceCpu<U>,
                    _: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>,
                    _: &SerializedVecView<'a,U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(loss.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,CudaVec<U,CudaTensor1dPtr<U,N>>,DeviceGpu<U>>
    for Identity<U,DeviceGpu<U>>
    where U: UnitValue<U>, DeviceGpu<U>: Device<U> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        Ok(input.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U>,
                    _: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    loss: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    _: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>) -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        Ok(loss.try_into()?)
    }
}
/// Sigmoid Implementation
pub struct Sigmoid<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>,
    c:HashSet<&'static str>
}
impl<U,D> Sigmoid<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of Sigmoid
    pub fn new(_:&D) -> Sigmoid<U,D> {
        let mut c = HashSet::new();
        c.insert("crossentropy");

        Sigmoid {
            u: PhantomData::<U>,
            d:PhantomData::<D>,
            c:c
        }
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,I,const N:usize> Activation<U,I,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: Iterator<Item=U> + Clone {
    fn apply(&self, _: &DeviceCpu<U>, input: &I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().map(|i| U::one() / (U::one() + (-i).exp())).collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, o: &I, loss: &I, _: &I) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().zip(o.clone()).map(|(l,o)| o * (U::one() - o) * l).collect::<Vec<U>>().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,const N:usize> Activation<U,CudaTensor1dPtrView<'a,U,N>,CudaTensor1dPtr<U,N>,DeviceGpu<U>> for Sigmoid<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          for<'b> SigmoidForward<'b,U,N>: Kernel<Args=ActivationForwardArgs<'b,U,N>>,
          for<'b> SigmoidBackward<'b,U,N>: Kernel<Args=ActivationBackwardArgs<'b,U,N>> {

    fn apply(&self, device: &DeviceGpu<U>, input: &CudaTensor1dPtrView<'a,U,N>) -> Result<CudaTensor1dPtr<U,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = ActivationForwardArgs::new(input,output);

        let mut kernel = SigmoidForward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U>,
              o: &CudaTensor1dPtrView<'a,U,N>, loss: &CudaTensor1dPtrView<'a,U,N>, u: &CudaTensor1dPtrView<'a,U,N>)
        -> Result<CudaTensor1dPtr<U,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = ActivationBackwardArgs::new(o, u, loss, output);

        let mut kernel = SigmoidBackward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> BatchActivation<U,SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,SerializedVecView<'a,U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,CudaVec<U,CudaTensor1dPtr<U,N>>,DeviceGpu<U>>
    for Sigmoid<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          for<'b> SigmoidBatchForward<'b,U,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,N>>,
          for<'b> SigmoidBatchBackward<'b,U,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U>, input: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>) -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        let len = input.size();
        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,device.get_memory_pool())?;

        let mut args = ActivationBatchForwardArgs::new(input,output,len);

        let mut kernel = SigmoidBatchForward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                             y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U>,
                    o: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    loss: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    u: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>) -> Result<CudaVec<U, CudaTensor1dPtr<U, N>>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,device.get_memory_pool())?;

        let mut args = ActivationBatchBackwardArgs::new(o, u, loss, output, len);

        let mut kernel = SigmoidBatchBackward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }
}
/// ReLu Implementation
pub struct ReLu<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> ReLu<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of ReLu
    pub fn new(_:&D) -> ReLu<U,D> {
        ReLu {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,I,const N:usize> Activation<U,I,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: Iterator<Item=U> + Clone {
    fn apply(&self, _: &DeviceCpu<U>, input: &I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().map(|i| {
            i.max(&U::default())
        }).collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &I, loss: &I, u: &I) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().zip(u.clone()).map(|(l,u)| {
            if u > U::default() {
                l
            } else {
                U::default()
            }
        }).collect::<Vec<U>>().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,CudaTensor1dPtrView<'a,U,N>,CudaTensor1dPtr<U,N>,DeviceGpu<U>> for ReLu<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          for<'b> ReLuForward<'b,U,N>: Kernel<Args=ActivationForwardArgs<'b,U,N>>,
          for<'b> ReLuBackward<'b,U,N>: Kernel<Args=ActivationBackwardArgs<'b,U,N>> {
    fn apply(&self, device: &DeviceGpu<U>, input: &CudaTensor1dPtrView<'a,U,N>) -> Result<CudaTensor1dPtr<U,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = ActivationForwardArgs::new(input, output);

        let mut kernel = ReLuForward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U>,
              o: &CudaTensor1dPtrView<'a,U,N>, loss: &CudaTensor1dPtrView<'a,U,N>, u: &CudaTensor1dPtrView<'a,U,N>)
        -> Result<CudaTensor1dPtr<U,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = ActivationBackwardArgs::new(o, u, loss, output);

        let mut kernel = ReLuBackward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> BatchActivation<U,SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,SerializedVecView<'a,U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,CudaVec<U,CudaTensor1dPtr<U,N>>,DeviceGpu<U>>
    for ReLu<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          for<'b> ReLuBatchForward<'b,U,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,N>>,
          for<'b> ReLuBatchBackward<'b,U,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U>, input: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,device.get_memory_pool())?;

        let mut args = ActivationBatchForwardArgs::new(input, output, len);

        let mut kernel = ReLuBatchForward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
            y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U>,
                    o: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    loss: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    u: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>) -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,device.get_memory_pool())?;

        let mut args = ActivationBatchBackwardArgs::new(o, u, loss, output, len);

        let mut kernel = ReLuBatchBackward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }
}
/// Swish Implementation
pub struct Swish<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> Swish<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of Swish
    pub fn new(_:&D) -> Swish<U,D> {
        Swish {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,I,const N:usize> Activation<U,I,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: Iterator<Item=U> + Clone {
    fn apply(&self, _: &DeviceCpu<U>, input: &I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().map(|i| i * (U::one() / (U::one() + (-i).exp()))).collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, o: &I, loss: &I, u: &I) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().zip(o.clone()).zip(u.clone()).map(|((l,o),u)| {
            (o + U::one() / (U::one() + (-u).exp()) * (U::one() - o)) * l
        }).collect::<Vec<U>>().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,CudaTensor1dPtrView<'a,U,N>,CudaTensor1dPtr<U,N>,DeviceGpu<U>> for Swish<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
             for<'b> SwishForward<'b,U,N>: Kernel<Args=ActivationForwardArgs<'b,U,N>>,
             for<'b> SwishBackward<'b,U,N>: Kernel<Args=ActivationBackwardArgs<'b,U,N>> {

    fn apply(&self, device: &DeviceGpu<U>, input: &CudaTensor1dPtrView<'a,U,N>) -> Result<CudaTensor1dPtr<U,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = ActivationForwardArgs::new(input,output);

        let mut kernel = SwishForward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U>,
              o: &CudaTensor1dPtrView<'a,U,N>, loss: &CudaTensor1dPtrView<'a,U,N>, u: &CudaTensor1dPtrView<'a,U,N>)
        -> Result<CudaTensor1dPtr<U,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = ActivationBackwardArgs::new(o, u, loss, output);

        let mut kernel = SwishBackward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> BatchActivation<U,SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,SerializedVecView<'a,U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,CudaVec<U,CudaTensor1dPtr<U,N>>,DeviceGpu<U>>
    for Swish<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          for<'b> SwishBatchForward<'b,U,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,N>>,
          for<'b> SwishBatchBackward<'b,U,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U>, input: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,device.get_memory_pool())?;

        let mut args = ActivationBatchForwardArgs::new(input,output,len);

        let mut kernel = SwishBatchForward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U>,
                    o: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    loss: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    u: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>) -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,device.get_memory_pool())?;

        let mut args = ActivationBatchBackwardArgs::new(o, u, loss, output, len);

        let mut kernel = SwishBatchBackward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }
}
/// Tanh Implementation
pub struct Tanh<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> Tanh<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of Tanh
    pub fn new(_:&D) -> Tanh<U,D> {
        Tanh {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,I,const N:usize> Activation<U,I,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: Iterator<Item=U> + Clone {
    fn apply(&self, _: &DeviceCpu<U>, input: &I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().map(|i| i.tanh()).collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, o: &I, loss: &I, _: &I) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().zip(o.clone()).map(|(l,o)| {
            (U::one() - o * o) * l
        }).collect::<Vec<U>>().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,CudaTensor1dPtrView<'a,U,N>,CudaTensor1dPtr<U,N>,DeviceGpu<U>> for Tanh<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          for<'b> TanhForward<'b,U,N>: Kernel<Args=ActivationForwardArgs<'b,U,N>>,
          for<'b> TanhBackward<'b,U,N>: Kernel<Args=ActivationBackwardArgs<'b,U,N>> {

    fn apply(&self, device: &DeviceGpu<U>, input: &CudaTensor1dPtrView<'a,U,N>) -> Result<CudaTensor1dPtr<U,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = ActivationForwardArgs::new(input,output);

        let mut kernel = TanhForward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U>,
              o: &CudaTensor1dPtrView<'a,U,N>, loss: &CudaTensor1dPtrView<'a,U,N>, u: &CudaTensor1dPtrView<'a,U,N>)
        -> Result<CudaTensor1dPtr<U,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = ActivationBackwardArgs::new(o, u, loss, output);

        let mut kernel = TanhBackward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> BatchActivation<U,SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,SerializedVecView<'a,U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,CudaVec<U,CudaTensor1dPtr<U,N>>,DeviceGpu<U>>
    for Tanh<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          for<'b> TanhBatchForward<'b,U,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,N>>,
          for<'b> TanhBatchBackward<'b,U,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U>, input: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,device.get_memory_pool())?;

        let mut args = ActivationBatchForwardArgs::new(input,output,len);

        let mut kernel = TanhBatchForward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U>,
                    o: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    loss: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    u: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>) -> Result<CudaVec<U, CudaTensor1dPtr<U, N>>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,device.get_memory_pool())?;

        let mut args = ActivationBatchBackwardArgs::new(o, u, loss, output, len);

        let mut kernel = TanhBatchBackward::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }
}
/// SoftMax Implementation
pub struct SoftMax<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>,
    c:HashSet<&'static str>
}
impl<U,D> SoftMax<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of SoftMax
    pub fn new(_:&D) -> SoftMax<U,D> {
        let mut c = HashSet::new();
        c.insert("crossentropymulticlass");

        SoftMax {
            u: PhantomData::<U>,
            d:PhantomData::<D>,
            c:c
        }
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,I,const N:usize> Activation<U,I,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: Iterator<Item=U> + Clone {
    fn apply(&self, _: &DeviceCpu<U>, input: &I) -> Result<Arr<U,N>, EvaluateError> {
        let alpha = input.clone().fold(U::initial_max_value(), |m,v| {
            v.max(&m)
        });
        let sum = input.clone().map(|x| (x - alpha).exp()).fold(U::default(),
            |acc,x| {
            acc + x
        });
        Ok(input.clone().map(|i| {
            let number = (i - alpha).exp();
            number / sum
        }).collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, o: &I, loss: &I, _: &I) -> Result<Arr<U,N>, TrainingError> {
        let scale = U::from_f64(1e7).expect("Error in type conversion from f64.");

        let sum = loss.clone().zip(o.clone()).map(|(l,o)| {
            (l * -o) * scale
        }).fold(U::default(), |acc,x| {
            acc + x
        }) / scale;

        Ok(loss.clone().zip(o.clone()).map(|(l,o)| {
           sum * o + l * (o * o + (o * (U::one() - o)))
        }).collect::<Vec<U>>().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,const N:usize> Activation<U,CudaTensor1dPtrView<'a,U,N>,CudaTensor1dPtr<U,N>,DeviceGpu<U>> for SoftMax<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          CudaPtr<U>: TryFrom<U,Error=CudaError>,
          for<'b> SoftMaxForward<'b,U,N>: Kernel<Args=ActivationForwardArgs<'b,U,N>>,
          for<'b> SoftMaxBackward<'b,U,N>: Kernel<Args=ActivationBackwardArgs<'b,U,N>> {

    fn apply(&self, device: &DeviceGpu<U>, input: &CudaTensor1dPtrView<'a,U,N>) -> Result<CudaTensor1dPtr<U,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = ActivationForwardArgs::new(input,output);

        let mut kernel = SoftMaxForward::<'_,U,N>::new();

        kernel.launch(dim3 { x: 1, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 32 * mem::size_of::<U>() * 2)?;

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U>,
              o: &CudaTensor1dPtrView<'a,U,N>, loss: &CudaTensor1dPtrView<'a,U,N>, u: &CudaTensor1dPtrView<'a,U,N>)
        -> Result<CudaTensor1dPtr<U,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = ActivationBackwardArgs::new(o, u, loss, output);

        let mut kernel = SoftMaxBackward::<'_,U,N>::new();

        kernel.launch(dim3 { x: 1, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>())?;

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> BatchActivation<U,SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,SerializedVecView<'a,U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVecView<'a,U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,CudaVec<U,CudaTensor1dPtr<U,N>>,DeviceGpu<U>>
    for SoftMax<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          for<'b> SoftMaxBatchForward<'b,U,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,N>>,
          for<'b> SoftMaxBatchBackward<'b,U,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U>, input: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,device.get_memory_pool())?;

        let mut args = ActivationBatchForwardArgs::new(input, output, len);

        let mut kernel = SoftMaxBatchForward::<'_,U,N>::new();

        kernel.launch(dim3 { x: len as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 32 * mem::size_of::<U>() * 2)?;

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U>,
                    o: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    loss: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>,
                    u: &CudaVecView<'a,U,CudaTensor1dPtr<U,N>>) -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(len,device.get_memory_pool())?;

        let mut args = ActivationBatchBackwardArgs::new(o, u, loss, output, len);

        let mut kernel = SoftMaxBatchBackward::<'_,U,N>::new();

        kernel.launch(dim3 { x: len as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>())?;

        Ok(args.output)
    }
}
