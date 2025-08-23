//! Activation Function Implementation

use std::collections::HashSet;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_uint;
use cuda_runtime_sys::dim3;
use rayon::prelude::{FromParallelIterator, IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use crate::UnitValue;
use crate::arr::*;
use crate::cuda::{CudaPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, DataTypeInfo, Kernel, WriteMemory};
use crate::cuda::allocator::CudaAllocator;
use crate::cuda::kernel::activation::{ActivationBackwardArgs, ActivationBatchBackwardArgs, ActivationBatchForwardArgs, ActivationForwardArgs, ReLuBackward, ReLuBatchBackward, ReLuForward, ReLuBatchForward, SigmoidBackward, SigmoidBatchBackward, SigmoidForward, SigmoidBatchForward, SoftMaxBackward, SoftMaxBatchBackward, SoftMaxForward, SoftMaxBatchForward, SwishBackward, SwishBatchBackward, SwishForward, TanhBackward, TanhBatchBackward, TanhForward, TanhBatchForward, SwishBatchForward};
use crate::device::*;
use crate::error::{EvaluateError, TrainingError};
use crate::layer::{BatchDataType, BatchSize};
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
impl<'a,U,I,AC,const N:usize> Activation<U,I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for Identity<U,DeviceGpu<U,AC>>
    where U: UnitValue<U>,
          I: 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=EvaluateError>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator {

    fn apply(&self, _: &DeviceGpu<U,AC>, input: &'a I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        Ok(input.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U,AC>,
              _: &I, loss: &I, _: &I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
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
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a <I as BatchDataType>::Type,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for Identity<U,DeviceGpu<U,AC>>
    where U: UnitValue<U>,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: Clone + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=TrainingError>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: TryFrom<&'a <I as BatchDataType>::Type,Error=TrainingError>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator {

    fn batch_apply(&self, _: &DeviceGpu<U,AC>, input: &'a <I as BatchDataType>::Type)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,TrainingError> {
        Ok(input.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U,AC>,
                    _: &'a <I as BatchDataType>::Type,
                    loss: &'a <I as BatchDataType>::Type,
                    _: &'a <I as BatchDataType>::Type) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
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
impl<'a,U,I,AC,const N:usize> Activation<U,I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for Sigmoid<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtrView<'a,U,N>: TryFrom<&'a I,Error=EvaluateError>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=EvaluateError>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: TryFrom<&'a <I as BatchDataType>::Type,Error=EvaluateError>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> SigmoidForward<'b,U,AC,N>: Kernel<Args=ActivationForwardArgs<'b,U,AC,N>>,
          for<'b> SigmoidBackward<'b,U,AC,N>: Kernel<Args=ActivationBackwardArgs<'b,U,AC,N>> {

    fn apply(&self, device: &DeviceGpu<U,AC>, input: &I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let mut args = ActivationForwardArgs::new(&input.try_into()?,output);

        let mut kernel = SigmoidForward::<'_,U,AC,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>,
              o: &I, loss: &I, u: &I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let mut args = ActivationBackwardArgs::new(&o.try_into()?, &u.try_into()?, &loss.try_into()?, output);

        let mut kernel = SigmoidBackward::<'_,U,AC,N>::new();

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
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a <I as BatchDataType>::Type,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for Sigmoid<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: BatchSize + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtrView<'a,U,N>: TryFrom<&'a I,Error=TrainingError>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=TrainingError>,
          CudaVecView<'a,U,CudaTensor1dPtrView<'a,U,N>>: TryFrom<&'a <I as BatchDataType>::Type,Error=TrainingError>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> SigmoidBatchForward<'b,U,AC,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,AC,N>>,
          for<'b> SigmoidBatchBackward<'b,U,AC,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a <I as BatchDataType>::Type) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();
        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let mut args = ActivationBatchForwardArgs::new(&input.try_into()?,output,len);

        let mut kernel = SigmoidBatchForward::<'_,U,AC,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                             y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>,
                    o: &'a <I as BatchDataType>::Type,
                    loss: &'a <I as BatchDataType>::Type,
                    u: &'a <I as BatchDataType>::Type) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let mut args = ActivationBatchBackwardArgs::new(&o.try_into()?, &u.try_into()?, &loss.try_into()?, output, len);

        let mut kernel = SigmoidBatchBackward::<'_,U,AC,N>::new();

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
impl<'a,U,I,AC,const N:usize> Activation<U,I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for ReLu<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: BatchSize + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtrView<'a,U,N>: TryFrom<&'a I,Error=EvaluateError>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=EvaluateError>,
          <I as BatchDataType>::Type: IntoConverter,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: TryFrom<&'a <I as BatchDataType>::Type,Error=EvaluateError>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> ReLuForward<'b,U,AC,N>: Kernel<Args=ActivationForwardArgs<'b,U,AC,N>>,
          for<'b> ReLuBackward<'b,U,AC,N>: Kernel<Args=ActivationBackwardArgs<'b,U,AC,N>> {
    fn apply(&self, device: &DeviceGpu<U,AC>, input: &I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let mut args = ActivationForwardArgs::new(&input.try_into()?, output);

        let mut kernel = ReLuForward::<'_,U,AC,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>,
              o: &I, loss: &I, u: &I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let mut args = ActivationBackwardArgs::new(&o.try_into()?, &u.try_into()?, &loss.try_into()?, output);

        let mut kernel = ReLuBackward::<'_,U,AC,N>::new();

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
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a <I as BatchDataType>::Type,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for ReLu<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: BatchSize + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtrView<'a,U,N>: TryFrom<&'a I,Error=TrainingError>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=TrainingError>,
          CudaVecView<'a,U,CudaTensor1dPtrView<'a,U,N>>: TryFrom<&'a <I as BatchDataType>::Type,Error=TrainingError>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> ReLuBatchForward<'b,U,AC,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,AC,N>>,
          for<'b> ReLuBatchBackward<'b,U,AC,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a <I as BatchDataType>::Type)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let mut args = ActivationBatchForwardArgs::new(&input.try_into()?, output, len);

        let mut kernel = ReLuBatchForward::<'_,U,AC,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
            y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>,
                    o: &'a <I as BatchDataType>::Type,
                    loss: &'a <I as BatchDataType>::Type,
                    u: &'a <I as BatchDataType>::Type) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let mut args = ActivationBatchBackwardArgs::new(&o.try_into()?, &u.try_into()?, &loss.try_into()?, output, len);

        let mut kernel = ReLuBatchBackward::<'_,U,AC,N>::new();

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
impl<'a,U,I,AC,const N:usize> Activation<U,I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for Swish<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: BatchSize + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtrView<'a,U,N>: TryFrom<&'a I,Error=EvaluateError>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=EvaluateError>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: TryFrom<&'a <I as BatchDataType>::Type,Error=EvaluateError>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> SwishForward<'b,U,AC,N>: Kernel<Args=ActivationForwardArgs<'b,U,AC,N>>,
          for<'b> SwishBackward<'b,U,AC,N>: Kernel<Args=ActivationBackwardArgs<'b,U,AC,N>> {
    fn apply(&self, device: &DeviceGpu<U,AC>, input: &I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let mut args = ActivationForwardArgs::new(&input.try_into()?,output);

        let mut kernel = SwishForward::<'_,U,AC,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>,
              o: &I, loss: &I, u: &I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let mut args = ActivationBackwardArgs::new(&o.try_into()?, &u.try_into()?, &loss.try_into()?, output);

        let mut kernel = SwishBackward::<'_,U,AC,N>::new();

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
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a <I as BatchDataType>::Type,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for Swish<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: BatchSize + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtrView<'a,U,N>: TryFrom<&'a I,Error=TrainingError>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=TrainingError>,
          CudaVecView<'a,U,CudaTensor1dPtrView<'a,U,N>>: TryFrom<&'a <I as BatchDataType>::Type,Error=TrainingError>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> SwishBatchForward<'b,U,AC,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,AC,N>>,
          for<'b> SwishBatchBackward<'b,U,AC,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a <I as BatchDataType>::Type)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let mut args = ActivationBatchForwardArgs::new(&input.try_into()?,output,len);

        let mut kernel = SwishBatchForward::<'_,U,AC,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>,
                    o: &'a <I as BatchDataType>::Type,
                    loss: &'a <I as BatchDataType>::Type,
                    u: &'a <I as BatchDataType>::Type) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let mut args = ActivationBatchBackwardArgs::new(&o.try_into()?, &u.try_into()?, &loss.try_into()?, output, len);

        let mut kernel = SwishBatchBackward::<'_,U,AC,N>::new();

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
impl<'a,U,I,AC,const N:usize> Activation<U,I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for Tanh<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: BatchSize + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtrView<'a,U,N>: TryFrom<&'a I,Error=EvaluateError>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=EvaluateError>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> TanhForward<'b,U,AC,N>: Kernel<Args=ActivationForwardArgs<'b,U,AC,N>>,
          for<'b> TanhBackward<'b,U,AC,N>: Kernel<Args=ActivationBackwardArgs<'b,U,AC,N>> {

    fn apply(&self, device: &DeviceGpu<U,AC>, input: &I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let mut args = ActivationForwardArgs::new(&input.try_into()?,output);

        let mut kernel = TanhForward::<'_,U,AC,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>,
              o: &I, loss: &I, u: &I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let mut args = ActivationBackwardArgs::new(&o.try_into()?, &u.try_into()?, &loss.try_into()?, output);

        let mut kernel = TanhBackward::<'_,U,AC,N>::new();

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
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a <I as BatchDataType>::Type,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for Tanh<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: BatchSize + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtrView<'a,U,N>: TryFrom<&'a I,Error=TrainingError>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=TrainingError>,
          CudaVecView<'a,U,CudaTensor1dPtrView<'a,U,N>>: TryFrom<&'a <I as BatchDataType>::Type,Error=TrainingError>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> TanhBatchForward<'b,U,AC,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,AC,N>>,
          for<'b> TanhBatchBackward<'b,U,AC,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a <I as BatchDataType>::Type)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let mut args = ActivationBatchForwardArgs::new(&input.try_into()?,output,len);

        let mut kernel = TanhBatchForward::<'_,U,AC,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (len as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>,
                    o: &'a <I as BatchDataType>::Type,
                    loss: &'a <I as BatchDataType>::Type,
                    u: &'a <I as BatchDataType>::Type) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let mut args = ActivationBatchBackwardArgs::new(&o.try_into()?, &u.try_into()?, &loss.try_into()?, output, len);

        let mut kernel = TanhBatchBackward::<'_,U,AC,N>::new();

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
impl<'a,U,I,AC,const N:usize> Activation<U,I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for SoftMax<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: BatchSize + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtrView<'a,U,N>: TryFrom<&'a I,Error=EvaluateError>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=EvaluateError>,
          <I as BatchDataType>::Type: IntoConverter,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> SoftMaxForward<'b,U,AC,N>: Kernel<Args=ActivationForwardArgs<'b,U,AC,N>>,
          for<'b> SoftMaxBackward<'b,U,AC,N>: Kernel<Args=ActivationBackwardArgs<'b,U,AC,N>> {

    fn apply(&self, device: &DeviceGpu<U,AC>, input: &I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let mut args = ActivationForwardArgs::new(&input.try_into()?,output);

        let mut kernel = SoftMaxForward::<'_,U,AC,N>::new();

        kernel.launch(dim3 { x: 1, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 32 * mem::size_of::<U>() * 2)?;

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>,
              o: &I, loss: &I, u: &I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let mut args = ActivationBackwardArgs::new(&o.try_into()?, &u.try_into()?, &loss.try_into()?, output);

        let mut kernel = SoftMaxBackward::<'_,U,AC,N>::new();

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
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a <I as BatchDataType>::Type,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for SoftMax<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: BatchSize + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaTensor1dPtrView<'a,U,N>: TryFrom<&'a I,Error=TrainingError>,
          CudaTensor1dPtr<U,AC,N>: TryFrom<&'a I,Error=TrainingError>,
          CudaVecView<'a,U,CudaTensor1dPtrView<'a,U,N>>: TryFrom<&'a <I as BatchDataType>::Type,Error=TrainingError>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> SoftMaxBatchForward<'b,U,AC,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,AC,N>>,
          for<'b> SoftMaxBatchBackward<'b,U,AC,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a <I as BatchDataType>::Type)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let mut args = ActivationBatchForwardArgs::new(&input.try_into()?, output, len);

        let mut kernel = SoftMaxBatchForward::<'_,U,AC,N>::new();

        kernel.launch(dim3 { x: len as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 32 * mem::size_of::<U>() * 2)?;

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>,
                    o: &'a <I as BatchDataType>::Type,
                    loss: &'a <I as BatchDataType>::Type,
                    u: &'a <I as BatchDataType>::Type) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let mut args = ActivationBatchBackwardArgs::new(&o.try_into()?, &u.try_into()?, &loss.try_into()?, output, len);

        let mut kernel = SoftMaxBatchBackward::<'_,U,AC,N>::new();

        kernel.launch(dim3 { x: len as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>())?;

        Ok(args.output)
    }
}
