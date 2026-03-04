//! Activation Function Implementation

use std::collections::HashSet;
use std::marker::PhantomData;
use rayon::prelude::{FromParallelIterator, IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use crate::UnitValue;
use crate::arr::*;
use crate::device::*;
use crate::error::{CudaError, EvaluateError, TrainingError, TypeConvertError};
use crate::layer::{BatchDataType, BatchSize};
use crate::lossfunction::LossFunction;
use crate::cuda::{AsConstKernelPtr, AsCudaMutPtr, AsCudaView, AsKernelPtr, AsMutKernelPtr, CudaMutPtr, CudaPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, CudaView, DataTypeInfo, Kernel, MemorySize, TryClone, WriteMemory};
use crate::cuda::allocator::CudaAllocator;
use crate::cuda::kernel::activation::{ActivationBackwardArgs, ActivationBatchBackwardArgs, ActivationBatchForwardArgs, ActivationForwardArgs, ReLuBackward, ReLuBatchBackward, ReLuForward, ReLuBatchForward, SigmoidBackward, SigmoidBatchBackward, SigmoidForward, SigmoidBatchForward, SoftMaxBackward, SoftMaxBatchBackward, SoftMaxForward, SoftMaxBatchForward, SwishBackward, SwishBatchBackward, SwishForward, TanhBackward, TanhBatchBackward, TanhForward, TanhBatchForward, SwishBatchForward, LeakyReLuBatchBackward, LeakyReLuBatchForward, LeakyReLuBackward, LeakyReLuForward, ClippedReLuForward, ClippedReLuForwardArgs, ClippedReLuBackward, ClippedReLuBackwardArgs, ClippedReLuBatchForward, ClippedReLuBatchForwardArgs, ClippedReLuBatchBackward, ClippedReLuBatchBackwardArgs};

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
    fn apply<'a>(&self, device:&D, input:T) -> Result<R, EvaluateError>;
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
    fn derive<'a>(&self, device:&D, o:T, loss:T, u:T) -> Result<R, TrainingError>;
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
    fn batch_apply<'a>(&self, device:&D, input:T) -> Result<R, TrainingError>;
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
    fn batch_derive<'a>(&self, device:&D, o:T, loss:T, u:T) -> Result<R, TrainingError>;
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
impl<'a,U,I,const N:usize> Activation<U,&'a I,Arr<U,N>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
    where U: UnitValue<U>, I: Iterator<Item=U> + Clone {

    fn apply(&self, _: &DeviceCpu<U>, input: &'a I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &'a I, loss: &'a I, _: &'a I) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().collect::<Vec<U>>().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> Activation<U,&Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
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

    fn apply(&self, _: &DeviceCpu<U>, input: ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().into())
    }

    fn derive(&self, _: &DeviceCpu<U>, _: ArrView<'a,U,N>, loss: ArrView<'a,U,N>, _: ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().into())
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,I,AC,const N:usize> Activation<U,&'a I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for Identity<U,DeviceGpu<U,AC>>
    where U: UnitValue<U>,
          CudaPtr<U,AC>: WriteMemory<U>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          I: TryClone<Error=CudaError>,
          CudaTensor1dPtr<U,AC,N>: From<I>,
          EvaluateError: From<CudaError> {

    fn apply(&self, _: &DeviceGpu<U,AC>, input: &'a I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        Ok(input.try_clone()?.into())
    }

    fn derive(&self, _: &DeviceGpu<U,AC>,
              _: &'a I, loss: &'a I, _: &'a I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        Ok(loss.try_clone()?.into())
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> BatchActivation<U,&SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
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

    fn batch_apply(&self, _: &DeviceCpu<U>, input: SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok((&input).into())
    }

    fn batch_derive(&self, _: &DeviceCpu<U>,
                    _: SerializedVecView<'a,U,Arr<U,N>>,
                    loss: SerializedVecView<'a,U,Arr<U,N>>,
                    _: SerializedVecView<'a,U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok((&loss).into())
    }
}
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a I,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for Identity<U,DeviceGpu<U,AC>>
    where U: UnitValue<U>,
          CudaPtr<U,AC>: WriteMemory<U>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          I: IntoConverter + TryClone<Error=CudaError>,
          I: TryFrom<<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC> as IntoConverter>::Converter,Error=TypeConvertError>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: IntoConverter + TryClone<Error=CudaError>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: TryFrom<<I as IntoConverter>::Converter,Error=TypeConvertError>,
          TrainingError: From<TypeConvertError> {

    fn batch_apply(&self, _: &DeviceGpu<U,AC>, input: &'a I)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,TrainingError> {
        Ok((*input).try_clone()?.into_converter().try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U,AC>,
                    _: &'a I,
                    loss: &'a I,
                    _: &'a I) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        Ok((*loss).try_clone()?.into_converter().try_into()?)
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
impl<U,const N:usize> Activation<U,&Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
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

    fn apply(&self, device: &DeviceCpu<U>, input: ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: ArrView<'a,U,N>, loss: ArrView<'a,U,N>, u: ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,I,const N:usize> Activation<U,&'a I,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: Iterator<Item=U> + Clone {
    fn apply(&self, _: &DeviceCpu<U>, input: &'a I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().map(|i| U::one() / (U::one() + (-i).exp())).collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, o: &'a I, loss: &'a I, _: &'a I) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().zip(o.clone()).map(|(l,o)| o * (U::one() - o) * l).collect::<Vec<U>>().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,I,AC,const N:usize> Activation<U,&'a I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for Sigmoid<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> I: CudaView<'b>,
          for<'b> &'b I: AsCudaView<'b>,
          for<'b> CudaTensor1dPtrView<'b,U,N>: From<<&'b I as CudaView<'b>>::Type>,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> SigmoidForward<'b,U,AC,N>: Kernel<Args=ActivationForwardArgs<'b,U,AC,N>>,
          for<'b> SigmoidBackward<'b,U,AC,N>: Kernel<Args=ActivationBackwardArgs<'b,U,AC,N>> {

    fn apply(&self, device: &DeviceGpu<U,AC>, input: &'a I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let input = input.as_cuda_view().into();

        let mut args = ActivationForwardArgs::new(&input,output);

        let mut kernel = SigmoidForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let o = o.as_cuda_view().into();
        let u = u.as_cuda_view().into();
        let loss = loss.as_cuda_view().into();

        let mut args = ActivationBackwardArgs::new(&o, &u, &loss, output);

        let mut kernel = SigmoidBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> BatchActivation<U,&SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
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

    fn batch_apply(&self, device: &DeviceCpu<U>, input: SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: SerializedVecView<'a,U,Arr<U,N>>,
                    loss: SerializedVecView<'a,U,Arr<U,N>>, u: SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a I,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for Sigmoid<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchSize,
          AC: CudaAllocator + 'a,
          DeviceGpu<U,AC>: Device<U>,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: AsCudaMutPtr<Pointee=U,Allocator=AC>,
          for<'b> CudaMutPtr<'b,U,AC>: AsMutKernelPtr,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b I,Error=TypeConvertError>,
          for<'b> SigmoidBatchForward<'b,U,AC,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,AC,N>>,
          for<'b> SigmoidBatchBackward<'b,U,AC,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a I) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();
        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let input = input.try_into()?;

        let mut args = ActivationBatchForwardArgs::new(&input,output,len);

        let mut kernel = SigmoidBatchForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let o = o.try_into()?;
        let u = u.try_into()?;
        let loss = loss.try_into()?;

        let mut args = ActivationBatchBackwardArgs::new(&o,&u,&loss,output,len);

        let mut kernel = SigmoidBatchBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

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
impl<U,const N:usize> Activation<U,&Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
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

    fn apply(&self, device: &DeviceCpu<U>, input: ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: ArrView<'a,U,N>, loss: ArrView<'a,U,N>, u: ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,I,const N:usize> Activation<U,&'a I,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: Iterator<Item=U> + Clone {
    fn apply(&self, _: &DeviceCpu<U>, input: &'a I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().map(|i| {
            i.max(&U::default())
        }).collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, o: &'a I, loss: &'a I, _: &'a I) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().zip(o.clone()).map(|(l,o)| {
            if o > U::default() {
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
impl<'a,U,I,AC,const N:usize> Activation<U,&'a I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for ReLu<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          CudaPtr<U,AC>: WriteMemory<U>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator + 'a,
          for<'b> I: CudaView<'b>,
          for<'b> &'b I: AsCudaView<'b>,
          for<'b> CudaTensor1dPtrView<'b,U,N>: From<<&'b I as CudaView<'b>>::Type>,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> ReLuForward<'b,U,AC,N>: Kernel<Args=ActivationForwardArgs<'b,U,AC,N>>,
          for<'b> ReLuBackward<'b,U,AC,N>: Kernel<Args=ActivationBackwardArgs<'b,U,AC,N>> {
    fn apply(&self, device: &DeviceGpu<U,AC>, input: &'a I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let input = input.as_cuda_view().into();

        let mut args = ActivationForwardArgs::new(&input, output);

        let mut kernel = ReLuForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let o = o.as_cuda_view().into();
        let u = u.as_cuda_view().into();
        let loss = loss.as_cuda_view().into();

        let mut args = ActivationBackwardArgs::new(&o, &u, &loss, output);

        let mut kernel = ReLuBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> BatchActivation<U,&SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
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

    fn batch_apply(&self, device: &DeviceCpu<U>, input: SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: SerializedVecView<'a,U,Arr<U,N>>,
                    loss: SerializedVecView<'a,U,Arr<U,N>>, u: SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a I,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for ReLu<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchSize,
          AC: CudaAllocator + 'a,
          DeviceGpu<U,AC>: Device<U>,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: AsCudaMutPtr<Pointee=U,Allocator=AC>,
          for<'b> CudaMutPtr<'b,U,AC>: AsMutKernelPtr,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b I,Error=TypeConvertError>,
          for<'b> ReLuBatchForward<'b,U,AC,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,AC,N>>,
          for<'b> ReLuBatchBackward<'b,U,AC,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a I)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let input = input.try_into()?;

        let mut args = ActivationBatchForwardArgs::new(&input, output, len);

        let mut kernel = ReLuBatchForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let o = o.try_into()?;
        let u = u.try_into()?;
        let loss = loss.try_into()?;

        let mut args = ActivationBatchBackwardArgs::new(&o,&u,&loss,output,len);

        let mut kernel = ReLuBatchBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }
}
/// ClippedReLu Implementation
pub struct ClippedReLu<U,D> where U: UnitValue<U>, D: Device<U> {
    ceiling: U,
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> ClippedReLu<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of ClippedReLu
    pub fn new(_:&D, ceiling: U) -> ClippedReLu<U,D> {
        ClippedReLu {
            ceiling,
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U,const N:usize> Activation<U,&Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for ClippedReLu<U,DeviceCpu<U>>
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
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for ClippedReLu<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: ArrView<'a,U,N>, loss: ArrView<'a,U,N>, u: ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,I,const N:usize> Activation<U,&'a I,Arr<U,N>,DeviceCpu<U>> for ClippedReLu<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: Iterator<Item=U> + Clone {
    fn apply(&self, _: &DeviceCpu<U>, input: &'a I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().map(|i| {
            i.max(&U::default()).min(&self.ceiling)
        }).collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, o: &'a I, loss: &'a I, _: &'a I) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().zip(o.clone()).map(|(l,o)| {
            if o > U::default() && o <= self.ceiling {
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
impl<'a,U,I,AC,const N:usize> Activation<U,&'a I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for ClippedReLu<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo + AsKernelPtr,
          CudaPtr<U,AC>: WriteMemory<U>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator + 'a,
          for<'b> I: CudaView<'b>,
          for<'b> &'b I: AsCudaView<'b>,
          for<'b> CudaTensor1dPtrView<'b,U,N>: From<<&'b I as CudaView<'b>>::Type>,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> ClippedReLuForward<'b,U,AC,N>: Kernel<Args=ClippedReLuForwardArgs<'b,U,AC,N>>,
          for<'b> ClippedReLuBackward<'b,U,AC,N>: Kernel<Args=ClippedReLuBackwardArgs<'b,U,AC,N>> {
    fn apply(&self, device: &DeviceGpu<U,AC>, input: &'a I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let input = input.as_cuda_view().into();

        let mut args = ClippedReLuForwardArgs::new(&input, self.ceiling, output);

        let mut kernel = ClippedReLuForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let o = o.as_cuda_view().into();
        let u = u.as_cuda_view().into();
        let loss = loss.as_cuda_view().into();

        let mut args = ClippedReLuBackwardArgs::new(&o, &u, &loss, self.ceiling, output);

        let mut kernel = ClippedReLuBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> BatchActivation<U,&SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for ClippedReLu<U,DeviceCpu<U>>
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
impl<'a,U,const N:usize> BatchActivation<U,SerializedVecView<'a,U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for ClippedReLu<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: SerializedVecView<'a,U,Arr<U,N>>, loss: SerializedVecView<'a,U,Arr<U,N>>, u: SerializedVecView<'a,U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a I,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for ClippedReLu<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo + AsKernelPtr,
          I: BatchSize,
          AC: CudaAllocator + 'a,
          DeviceGpu<U,AC>: Device<U>,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: AsCudaMutPtr<Pointee=U,Allocator=AC>,
          for<'b> CudaMutPtr<'b,U,AC>: AsMutKernelPtr,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b I,Error=TypeConvertError>,
          for<'b> ClippedReLuBatchForward<'b,U,AC,N>: Kernel<Args=ClippedReLuBatchForwardArgs<'b,U,AC,N>>,
          for<'b> ClippedReLuBatchBackward<'b,U,AC,N>: Kernel<Args=ClippedReLuBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a I)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let input = input.try_into()?;

        let mut args = ClippedReLuBatchForwardArgs::new(&input, self.ceiling, output, len);

        let mut kernel = ClippedReLuBatchForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let o = o.try_into()?;
        let u = u.try_into()?;
        let loss = loss.try_into()?;

        let mut args = ClippedReLuBatchBackwardArgs::new(&o,&u,&loss,self.ceiling,output,len);

        let mut kernel = ClippedReLuBatchBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }
}

/// LeakyReLu Implementation
pub struct LeakyReLu<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> LeakyReLu<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of LeakyReLu
    pub fn new(_:&D) -> LeakyReLu<U,D> {
        LeakyReLu {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U,const N:usize> Activation<U,&Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for LeakyReLu<U,DeviceCpu<U>>
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
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for LeakyReLu<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: ArrView<'a,U,N>, loss: ArrView<'a,U,N>, u: ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,I,const N:usize> Activation<U,&'a I,Arr<U,N>,DeviceCpu<U>> for LeakyReLu<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: Iterator<Item=U> + Clone {
    fn apply(&self, _: &DeviceCpu<U>, input: &'a I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().map(|i| {
            i.max(&U::default()) + U::from_f64(0.01).unwrap() * i.min(&U::default())
        }).collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, o: &'a I, loss: &'a I, _: &'a I) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().zip(o.clone()).map(|(l,o)| {
            if o >= U::default() {
                l
            } else {
                l * U::from_f64(0.01).unwrap()
            }
        }).collect::<Vec<U>>().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,I,AC,const N:usize> Activation<U,&'a I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for LeakyReLu<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          CudaPtr<U,AC>: WriteMemory<U>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator + 'a,
          for<'b> I: CudaView<'b>,
          for<'b> &'b I: AsCudaView<'b>,
          for<'b> CudaTensor1dPtrView<'b,U,N>: From<<&'b I as CudaView<'b>>::Type>,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> LeakyReLuForward<'b,U,AC,N>: Kernel<Args=ActivationForwardArgs<'b,U,AC,N>>,
          for<'b> LeakyReLuBackward<'b,U,AC,N>: Kernel<Args=ActivationBackwardArgs<'b,U,AC,N>> {
    fn apply(&self, device: &DeviceGpu<U,AC>, input: &'a I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let input = input.as_cuda_view().into();

        let mut args = ActivationForwardArgs::new(&input, output);

        let mut kernel = LeakyReLuForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I)
              -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let o = o.as_cuda_view().into();
        let u = u.as_cuda_view().into();
        let loss = loss.as_cuda_view().into();

        let mut args = ActivationBackwardArgs::new(&o, &u, &loss, output);

        let mut kernel = LeakyReLuBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> BatchActivation<U,&SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for LeakyReLu<U,DeviceCpu<U>>
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
impl<'a,U,const N:usize> BatchActivation<U,SerializedVecView<'a,U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for LeakyReLu<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: SerializedVecView<'a,U,Arr<U,N>>,
                    loss: SerializedVecView<'a,U,Arr<U,N>>, u: SerializedVecView<'a,U,Arr<U,N>>)
                    -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a I,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for LeakyReLu<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchSize,
          AC: CudaAllocator + 'a,
          DeviceGpu<U,AC>: Device<U>,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: AsCudaMutPtr<Pointee=U,Allocator=AC>,
          for<'b> CudaMutPtr<'b,U,AC>: AsMutKernelPtr,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b I,Error=TypeConvertError>,
          for<'b> LeakyReLuBatchForward<'b,U,AC,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,AC,N>>,
          for<'b> LeakyReLuBatchBackward<'b,U,AC,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a I)
                   -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let input = input.try_into()?;

        let mut args = ActivationBatchForwardArgs::new(&input, output, len);

        let mut kernel = LeakyReLuBatchForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let o = o.try_into()?;
        let u = u.try_into()?;
        let loss = loss.try_into()?;

        let mut args = ActivationBatchBackwardArgs::new(&o,&u,&loss,output,len);

        let mut kernel = LeakyReLuBatchBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

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
impl<U,const N:usize> Activation<U,&Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
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

    fn apply(&self, device: &DeviceCpu<U>, input: ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: ArrView<'a,U,N>, loss: ArrView<'a,U,N>, u: ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,I,const N:usize> Activation<U,&'a I,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: Iterator<Item=U> + Clone {
    fn apply(&self, _: &DeviceCpu<U>, input: &'a I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().map(|i| i * (U::one() / (U::one() + (-i).exp()))).collect::<Vec<U>>().try_into()?)
    }

    fn derive(&self, _: &DeviceCpu<U>, o: &'a I, loss: &'a I, u: &'a I) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone().zip(o.clone()).zip(u.clone()).map(|((l,o),u)| {
            (o + U::one() / (U::one() + (-u).exp()) * (U::one() - o)) * l
        }).collect::<Vec<U>>().try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,I,AC,const N:usize> Activation<U,&'a I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for Swish<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          CudaPtr<U,AC>: WriteMemory<U>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator + 'a,
          for<'b> I: CudaView<'b>,
          for<'b> &'b I: AsCudaView<'b>,
          for<'b> CudaTensor1dPtrView<'b,U,N>: From<<&'b I as CudaView<'b>>::Type>,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> SwishForward<'b,U,AC,N>: Kernel<Args=ActivationForwardArgs<'b,U,AC,N>>,
          for<'b> SwishBackward<'b,U,AC,N>: Kernel<Args=ActivationBackwardArgs<'b,U,AC,N>> {
    fn apply(&self, device: &DeviceGpu<U,AC>, input: &'a I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let input = input.as_cuda_view().into();

        let mut args = ActivationForwardArgs::new(&input,output);

        let mut kernel = SwishForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let o = o.as_cuda_view().into();
        let u = u.as_cuda_view().into();
        let loss = loss.as_cuda_view().into();

        let mut args = ActivationBackwardArgs::new(&o, &u, &loss, output);

        let mut kernel = SwishBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> BatchActivation<U,&SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
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

    fn batch_apply(&self, device: &DeviceCpu<U>, input: SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: SerializedVecView<'a,U,Arr<U,N>>,
                    loss: SerializedVecView<'a,U,Arr<U,N>>, u: SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a I,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for Swish<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchSize,
          AC: CudaAllocator + 'a,
          DeviceGpu<U,AC>: Device<U>,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: AsCudaMutPtr<Pointee=U,Allocator=AC>,
          for<'b> CudaMutPtr<'b,U,AC>: AsMutKernelPtr,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b I,Error=TypeConvertError>,
          for<'b> SwishBatchForward<'b,U,AC,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,AC,N>>,
          for<'b> SwishBatchBackward<'b,U,AC,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a I)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();

        let input = input.try_into()?;

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let mut args = ActivationBatchForwardArgs::new(&input,output,len);

        let mut kernel = SwishBatchForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>,
                    o: &'a I,
                    loss: &'a I,
                    u: &'a I) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let o = o.try_into()?;
        let u = u.try_into()?;
        let loss = loss.try_into()?;

        let mut args = ActivationBatchBackwardArgs::new(&o,&u,&loss,output,len);

        let mut kernel = SwishBatchBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

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
impl<U,const N:usize> Activation<U,&Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
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

    fn apply(&self, device: &DeviceCpu<U>, input: ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: ArrView<'a,U,N>, loss: ArrView<'a,U,N>, u: ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,I,const N:usize> Activation<U,&'a I,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
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
impl<'a,U,I,AC,const N:usize> Activation<U,&'a I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for Tanh<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: BatchSize + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> I: CudaView<'b>,
          for<'b> &'b I: AsCudaView<'b>,
          for<'b> CudaTensor1dPtrView<'b,U,N>: From<<&'b I as CudaView<'b>>::Type>,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> TanhForward<'b,U,AC,N>: Kernel<Args=ActivationForwardArgs<'b,U,AC,N>>,
          for<'b> TanhBackward<'b,U,AC,N>: Kernel<Args=ActivationBackwardArgs<'b,U,AC,N>> {

    fn apply(&self, device: &DeviceGpu<U,AC>, input: &'a I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let input = input.as_cuda_view().into();

        let mut args = ActivationForwardArgs::new(&input,output);

        let mut kernel = TanhForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let o = o.as_cuda_view().into();
        let u = u.as_cuda_view().into();
        let loss = loss.as_cuda_view().into();

        let mut args = ActivationBackwardArgs::new(&o, &u, &loss, output);

        let mut kernel = TanhBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> BatchActivation<U,&SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
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

    fn batch_apply(&self, device: &DeviceCpu<U>, input: SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: SerializedVecView<'a,U,Arr<U,N>>,
                    loss: SerializedVecView<'a,U,Arr<U,N>>, u: SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a I,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for Tanh<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchSize,
          AC: CudaAllocator + 'a,
          DeviceGpu<U,AC>: Device<U>,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: AsCudaMutPtr<Pointee=U,Allocator=AC>,
          for<'b> CudaMutPtr<'b,U,AC>: AsMutKernelPtr,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b I,Error=TypeConvertError>,
          for<'b> TanhBatchForward<'b,U,AC,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,AC,N>>,
          for<'b> TanhBatchBackward<'b,U,AC,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a I)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let input = input.try_into()?;

        let mut args = ActivationBatchForwardArgs::new(&input,output,len);

        let mut kernel = TanhBatchForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let o = o.try_into()?;
        let u = u.try_into()?;
        let loss = loss.try_into()?;

        let mut args = ActivationBatchBackwardArgs::new(&o,&u,&loss,output,len);

        let mut kernel = TanhBatchBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

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
impl<U,const N:usize> Activation<U,&Arr<U,N>,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
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

    fn apply(&self, device: &DeviceCpu<U>, input: ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: ArrView<'a,U,N>, loss: ArrView<'a,U,N>, u: ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.iter().cloned(),&loss.iter().cloned(),&u.iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,I,const N:usize> Activation<U,&'a I,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: Iterator<Item=U> + Clone {
    fn apply(&self, _: &DeviceCpu<U>, input: &'a I) -> Result<Arr<U,N>, EvaluateError> {
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

    fn derive(&self, _: &DeviceCpu<U>, o: &'a I, loss: &'a I, _: &'a I) -> Result<Arr<U,N>, TrainingError> {
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
impl<'a,U,I,AC,const N:usize> Activation<U,&'a I,CudaTensor1dPtr<U,AC,N>,DeviceGpu<U,AC>> for SoftMax<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchDataType + 'a,
          <I as BatchDataType>::Type: BatchSize + 'a,
          CudaPtr<U,AC>: WriteMemory<U>,
          <I as BatchDataType>::Type: IntoConverter,
          DeviceGpu<U,AC>: Device<U>,
          AC: CudaAllocator,
          for<'b> I: CudaView<'b>,
          for<'b> &'b I: AsCudaView<'b>,
          for<'b> CudaTensor1dPtrView<'b,U,N>: From<<&'b I as CudaView<'b>>::Type>,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> SoftMaxForward<'b,U,AC,N>: Kernel<Args=ActivationForwardArgs<'b,U,AC,N>>,
          for<'b> SoftMaxBackward<'b,U,AC,N>: Kernel<Args=ActivationBackwardArgs<'b,U,AC,N>> {

    fn apply(&self, device: &DeviceGpu<U,AC>, input: &'a I) -> Result<CudaTensor1dPtr<U,AC,N>, EvaluateError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let input = input.as_cuda_view().into();

        let mut args = ActivationForwardArgs::new(&input,output);

        let mut kernel = SoftMaxForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I)
        -> Result<CudaTensor1dPtr<U,AC,N>, TrainingError> {
        let output = CudaTensor1dPtr::<U,AC,N>::new(device.get_allocator())?;

        let o = o.as_cuda_view().into();
        let u = u.as_cuda_view().into();
        let loss = loss.as_cuda_view().into();

        let mut args = ActivationBackwardArgs::new(&o, &u, &loss, output);

        let mut kernel = SoftMaxBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> BatchActivation<U,&SerializedVec<U,Arr<U,N>>,SerializedVec<U,Arr<U,N>>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
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

    fn batch_apply(&self, device: &DeviceCpu<U>, input: SerializedVecView<'a,U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: SerializedVecView<'a,U,Arr<U,N>>,
                    loss: SerializedVecView<'a,U,Arr<U,N>>, u: SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.iter().cloned(), &l.iter().cloned(), &u.iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,I,AC,const N:usize> BatchActivation<U,&'a I,CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>,DeviceGpu<U,AC>>
    for SoftMax<U,DeviceGpu<U,AC>>
    where U: UnitValue<U> + DataTypeInfo,
          I: BatchSize,
          AC: CudaAllocator + 'a,
          DeviceGpu<U,AC>: Device<U>,
          CudaPtr<U,AC>: WriteMemory<U>,
          CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>: AsCudaMutPtr<Pointee=U,Allocator=AC>,
          for<'b> CudaMutPtr<'b,U,AC>: AsMutKernelPtr,
          for<'b> CudaTensor1dPtr<U,AC,N>: AsConstKernelPtr + AsKernelPtr + MemorySize,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b I,Error=TypeConvertError>,
          for<'b> SoftMaxBatchForward<'b,U,AC,N>: Kernel<Args=ActivationBatchForwardArgs<'b,U,AC,N>>,
          for<'b> SoftMaxBatchBackward<'b,U,AC,N>: Kernel<Args=ActivationBatchBackwardArgs<'b,U,AC,N>> {

    fn batch_apply(&self, device: &DeviceGpu<U,AC>, input: &'a I)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = input.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let input = input.try_into()?;

        let mut args = ActivationBatchForwardArgs::new(&input, output, len);

        let mut kernel = SoftMaxBatchForward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn batch_derive(&self, device: &DeviceGpu<U,AC>, o: &'a I, loss: &'a I, u: &'a I) -> Result<CudaVec<U,CudaTensor1dPtr<U,AC,N>,AC>, TrainingError> {
        let len = loss.size();

        let output = CudaVec::<U,CudaTensor1dPtr<U,AC,N>,AC>::new(len,device.get_allocator())?;

        let o = o.try_into()?;
        let u = u.try_into()?;
        let loss = loss.try_into()?;

        let mut args = ActivationBatchBackwardArgs::new(&o,&u,&loss,output,len);

        let mut kernel = SoftMaxBatchBackward::<'_,U,AC,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }
}
