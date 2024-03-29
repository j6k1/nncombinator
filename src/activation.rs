//! Activation Function Implementation

use std::collections::HashSet;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_uint;
use cuda_runtime_sys::dim3;
use rayon::prelude::{FromParallelIterator, IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use crate::UnitValue;
use crate::arr::*;
use crate::cuda::{CudaPtr, DataTypeInfo, Kernel, Memory};
use crate::cuda::kernel::activation::{
    ActivationBackwardArgs,
    ActivationForwardArgs,
    ReLuBackward,
    ReLuForward,
    SigmoidBackward,
    SigmoidForward,
    SoftMaxBackward,
    SoftMaxForward,
    SwishBackward,
    SwishForward,
    TanhBackward,
    TanhForward
};
use crate::device::*;
use crate::error::{CudaError, EvaluateError, TrainingError};
use crate::lossfunction::LossFunction;
use crate::mem::AsRawSlice;

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
pub trait BatchActivation<U,T,C,R,D>: Activation<U,T,R,D> where U: UnitValue<U>, D: Device<U> {
    /// Apply the activation function
    /// # Arguments
    /// * `device` - Device objects available for processing
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_apply(&self, device:&D, input:&C) -> Result<SerializedVec<U,R>, TrainingError>;
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
    fn batch_derive(&self, device:&D, o:&C, loss:&C, u:&C) -> Result<SerializedVec<U,R>, TrainingError>;
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
    where U: UnitValue<U>, I: IndexedParallelIterator<Item=U> + IntoParallelIterator + Clone {

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
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceGpu<U>> for Identity<U,DeviceGpu<U>>
    where U: UnitValue<U>, DeviceGpu<U>: Device<U> {

    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        Ok((*input).clone())
    }

    fn derive(&self, _: &DeviceGpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, _: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        Ok((*loss).clone())
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, _: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok((*input).clone())
    }

    fn batch_derive(&self, _: &DeviceCpu<U>, _: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, _: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok((*loss).clone())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, _: &DeviceCpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.into())
    }

    fn batch_derive(&self, _: &DeviceCpu<U>, _: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, _: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(loss.into())
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for Identity<U,DeviceGpu<U>>
    where U: UnitValue<U>, DeviceGpu<U>: Device<U> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok((*input).clone())
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, _: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, _: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok((*loss).clone())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for Identity<U,DeviceGpu<U>>
    where U: UnitValue<U>, DeviceGpu<U>: Device<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.into())
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, _: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, _: &SerializedVecView<'a,U,Arr<U,N>>)
                    -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(loss.into())
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
        self.apply(device,&input.par_iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.par_iter().cloned(),&loss.par_iter().cloned(),&u.par_iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.par_iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.par_iter().cloned(),&loss.par_iter().cloned(),&u.par_iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,I,const N:usize> Activation<U,I,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: IndexedParallelIterator<Item=U> + Clone,
          <I as IntoParallelIterator>::Iter: IndexedParallelIterator<Item=U>{

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
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceGpu<U>> for Sigmoid<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
             SigmoidForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
             SigmoidBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = SigmoidForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N)?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,1);

        let mut kernel = SigmoidBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceGpu<U>> for Sigmoid<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          SigmoidForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          SigmoidBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = SigmoidForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N)?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,1);

        let mut kernel = SigmoidBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.par_iter().cloned(), &l.par_iter().cloned(), &u.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.par_iter().cloned(), &l.par_iter().cloned(), &u.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for Sigmoid<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          SigmoidForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          SigmoidBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N * input.len())?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N * input.len())?;

        let mut args = ActivationForwardArgs::new(input_output, N,input.len());

        let mut kernel = SigmoidForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                             y: (input.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N * o.len())?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N * o.len())?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N * u.len())?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N * u.len())?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N * loss.len())?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N * loss.len())?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,loss.len());

        let mut kernel = SigmoidBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (loss.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for Sigmoid<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          SigmoidForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          SigmoidBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N * input.len())?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N * input.len())?;

        let mut args = ActivationForwardArgs::new(input_output, N,input.len());

        let mut kernel = SigmoidForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
            y: (input.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N * o.len())?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N * o.len())?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N * u.len())?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N * u.len())?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N * loss.len())?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N * loss.len())?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,loss.len());

        let mut kernel = SigmoidBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
            y: (loss.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
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
        self.apply(device,&input.par_iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.par_iter().cloned(),&loss.par_iter().cloned(),&u.par_iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.par_iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.par_iter().cloned(),&loss.par_iter().cloned(),&u.par_iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,I,const N:usize> Activation<U,I,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: IndexedParallelIterator<Item=U> + Clone,
          <I as IntoParallelIterator>::Iter: IndexedParallelIterator<Item=U> {
    fn apply(&self, _: &DeviceCpu<U>, input: &I) -> Result<Arr<U,N>, EvaluateError> {
        Ok(input.clone().map(|i| {
            if i > U::default() || i.is_nan() {
                i
            } else {
                U::default()
            }
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
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceGpu<U>> for ReLu<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          ReLuForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          ReLuBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {
    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {

        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = ReLuForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N)?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,1);

        let mut kernel = ReLuBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceGpu<U>> for ReLu<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          ReLuForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          ReLuBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {
    fn apply(&self, _: &DeviceGpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {

        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = ReLuForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N)?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,1);

        let mut kernel = ReLuBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.par_iter().cloned(), &l.par_iter().cloned(), &u.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.par_iter().cloned(), &l.par_iter().cloned(), &u.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for ReLu<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          ReLuForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          ReLuBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N * input.len())?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N * input.len())?;

        let mut args = ActivationForwardArgs::new(input_output, N,input.len());

        let mut kernel = ReLuForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
            y: (input.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N * o.len())?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N * o.len())?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N * u.len())?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N * u.len())?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N * loss.len())?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N * loss.len())?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,loss.len());

        let mut kernel = ReLuBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (loss.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for ReLu<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          ReLuForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          ReLuBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVecView<'a,U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N * input.len())?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N * input.len())?;

        let mut args = ActivationForwardArgs::new(input_output, N,input.len());

        let mut kernel = ReLuForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
            y: (input.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N * o.len())?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N * o.len())?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N * u.len())?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N * u.len())?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N * loss.len())?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N * loss.len())?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,loss.len());

        let mut kernel = ReLuBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
            y: (loss.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
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
        self.apply(device,&input.par_iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.par_iter().cloned(),&loss.par_iter().cloned(),&u.par_iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.par_iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.par_iter().cloned(),&loss.par_iter().cloned(),&u.par_iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,I,const N:usize> Activation<U,I,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: IndexedParallelIterator<Item=U> + Clone,
          <I as IntoParallelIterator>::Iter: IndexedParallelIterator<Item=U> {

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
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceGpu<U>> for Swish<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
             SwishForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
             SwishBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = SwishForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N)?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,1);

        let mut kernel = SwishBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceGpu<U>> for Swish<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          SwishForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          SwishBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = SwishForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N)?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,1);

        let mut kernel = SwishBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.par_iter().cloned(), &l.par_iter().cloned(), &u.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.par_iter().cloned(), &l.par_iter().cloned(), &u.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for Swish<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          SwishForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          SwishBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N * input.len())?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N * input.len())?;

        let mut args = ActivationForwardArgs::new(input_output, N,input.len());

        let mut kernel = SwishForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (input.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N * o.len())?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N * o.len())?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N * u.len())?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N * u.len())?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N * loss.len())?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N * loss.len())?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,loss.len());

        let mut kernel = SwishBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (loss.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for Swish<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          SwishForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          SwishBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVecView<'a,U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N * input.len())?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N * input.len())?;

        let mut args = ActivationForwardArgs::new(input_output, N,input.len());

        let mut kernel = SwishForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
            y: (input.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N * o.len())?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N * o.len())?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N * u.len())?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N * u.len())?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N * loss.len())?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N * loss.len())?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,loss.len());

        let mut kernel = SwishBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
            y: (loss.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
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
        self.apply(device,&input.par_iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.par_iter().cloned(),&loss.par_iter().cloned(),&u.par_iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.par_iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.par_iter().cloned(),&loss.par_iter().cloned(),&u.par_iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,I,const N:usize> Activation<U,I,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: IndexedParallelIterator<Item=U> + Clone,
          <I as IntoParallelIterator>::Iter: IndexedParallelIterator<Item=U> {

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
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceGpu<U>> for Tanh<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
             TanhForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
             TanhBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = TanhForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N)?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,1);

        let mut kernel = TanhBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceGpu<U>> for Tanh<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          TanhForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          TanhBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = TanhForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N)?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,1);

        let mut kernel = TanhBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.par_iter().cloned(), &l.par_iter().cloned(), &u.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVecView<'a,U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.par_iter().cloned(), &l.par_iter().cloned(), &u.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for Tanh<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          TanhForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          TanhBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N * input.len())?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N * input.len())?;

        let mut args = ActivationForwardArgs::new(input_output, N,input.len());

        let mut kernel = TanhForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (input.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N * o.len())?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N * o.len())?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N * u.len())?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N * u.len())?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N * loss.len())?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N * loss.len())?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,loss.len());

        let mut kernel = TanhBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (loss.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for Tanh<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
          TanhForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          TanhBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVecView<'a,U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N * input.len())?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N * input.len())?;

        let mut args = ActivationForwardArgs::new(input_output, N,input.len());

        let mut kernel = TanhForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
            y: (input.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N * o.len())?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N * o.len())?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N * u.len())?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N * u.len())?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N * loss.len())?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N * loss.len())?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,loss.len());

        let mut kernel = TanhBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
            y: (loss.len() as c_uint + 32 - 1) / 32, z: 1 },
                      dim3 { x: 32, y: 32, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
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
        self.apply(device,&input.par_iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.par_iter().cloned(),&loss.par_iter().cloned(),&u.par_iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, device: &DeviceCpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply(device,&input.par_iter().cloned())
    }

    fn derive(&self, device: &DeviceCpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive(device,&o.par_iter().cloned(),&loss.par_iter().cloned(),&u.par_iter().cloned(),)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,I,const N:usize> Activation<U,I,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          I: IndexedParallelIterator<Item=U> + Clone,
          <I as IntoParallelIterator>::Iter: IndexedParallelIterator<Item=U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &I) -> Result<Arr<U,N>, EvaluateError> {
        let alpha = input.clone().reduce(|| U::initial_max_value(), |m,v| {
            v.max(&m)
        });
        let sum = input.clone().map(|x| (x - alpha).exp()).reduce(|| U::default(),
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
        }).reduce(|| U::default(), |acc,x| {
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
impl<U,const N:usize> Activation<U,Arr<U,N>,Arr<U,N>,DeviceGpu<U>> for SoftMax<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          CudaPtr<U>: TryFrom<U,Error=CudaError>,
          SoftMaxForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          SoftMaxBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N, 1);

        let mut kernel = SoftMaxForward::<U>::new();

        kernel.launch(dim3 { x: 1, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>() * 2)?;

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N)?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,1);

        let mut kernel = SoftMaxBackward::<U>::new();

        kernel.launch(dim3 { x: 1, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>())?;

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<'a,U,const N:usize> Activation<U,ArrView<'a,U,N>,Arr<U,N>,DeviceGpu<U>> for SoftMax<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          CudaPtr<U>: TryFrom<U,Error=CudaError>,
          SoftMaxForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          SoftMaxBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &ArrView<'a,U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N, 1);

        let mut kernel = SoftMaxForward::<U>::new();

        kernel.launch(dim3 { x: 1, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>() * 2)?;

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, o: &ArrView<'a,U,N>, loss: &ArrView<'a,U,N>, u: &ArrView<'a,U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N)?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,1);

        let mut kernel = SoftMaxBackward::<U>::new();

        kernel.launch(dim3 { x: 1, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>())?;

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVec<U, Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.par_iter().cloned(), &l.par_iter().cloned(), &u.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          Vec<Arr<U,N>>: FromParallelIterator<Arr<U,N>> {

    fn batch_apply(&self, device: &DeviceCpu<U>, input: &SerializedVecView<'a,U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(input.par_iter().map(|i| {
            self.apply(device, &i.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,EvaluateError>>().map_err(|e| TrainingError::from(e))?.into())
    }

    fn batch_derive(&self, device: &DeviceCpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter().zip(u.par_iter())).map(|(o,(l,u))| {
            self.derive(device, &o.par_iter().cloned(), &l.par_iter().cloned(), &u.par_iter().cloned())
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
impl<U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVec<U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for SoftMax<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          CudaPtr<U>: TryFrom<U,Error=CudaError>,
          SoftMaxForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          SoftMaxBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N * input.len())?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N * input.len())?;

        let mut args = ActivationForwardArgs::new(input_output, N, input.len());

        let mut kernel = SoftMaxForward::<U>::new();

        kernel.launch(dim3 { x: input.len() as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>() * 2)?;

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, o: &SerializedVec<U,Arr<U,N>>, loss: &SerializedVec<U,Arr<U,N>>, u: &SerializedVec<U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N * o.len())?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N * o.len())?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N * u.len())?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N * u.len())?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N * loss.len())?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N * loss.len())?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,loss.len());

        let mut kernel = SoftMaxBackward::<U>::new();

        kernel.launch(dim3 { x: loss.len() as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>())?;

        Ok(args.loss.read_to_vec()?.try_into()?)
    }
}
impl<'a,U,const N:usize> BatchActivation<U,Arr<U,N>,SerializedVecView<'a,U,Arr<U,N>>,Arr<U,N>,DeviceGpu<U>> for SoftMax<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          CudaPtr<U>: TryFrom<U,Error=CudaError>,
          SoftMaxForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          SoftMaxBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn batch_apply(&self, _: &DeviceGpu<U>, input: &SerializedVecView<'a,U,Arr<U,N>>) -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N * input.len())?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N * input.len())?;

        let mut args = ActivationForwardArgs::new(input_output, N, input.len());

        let mut kernel = SoftMaxForward::<U>::new();

        kernel.launch(dim3 { x: input.len() as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>() * 2)?;

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn batch_derive(&self, _: &DeviceGpu<U>, o: &SerializedVecView<'a,U,Arr<U,N>>,
                    loss: &SerializedVecView<'a,U,Arr<U,N>>, u: &SerializedVecView<'a,U,Arr<U,N>>)
        -> Result<SerializedVec<U, Arr<U, N>>, TrainingError> {
        let mut o_ptr: CudaPtr<U> = CudaPtr::new(N * o.len())?;
        o_ptr.memcpy(o.as_raw_slice().as_ptr(),N * o.len())?;

        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N * u.len())?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N * u.len())?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N * loss.len())?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N * loss.len())?;

        let mut args = ActivationBackwardArgs::new(o_ptr, u_ptr, loss_ptr, N,loss.len());

        let mut kernel = SoftMaxBackward::<U>::new();

        kernel.launch(dim3 { x: loss.len() as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>())?;

        Ok(args.loss.read_to_vec()?.try_into()?)
    }
}
