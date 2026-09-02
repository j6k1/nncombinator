//! Implementation of the Data Projection Function

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Deref};
use crate::device::bridge::DeviceBridge;
use crate::device::scale::DeviceScale;
use crate::error::{EvaluateError};
use crate::layer::{BatchDataType, BatchSize};

pub trait DataMapper<'a,I,O,D>: Sized + Deref<Target=O> + 'a
    where I: Debug + Sized + 'a,
          O: Debug + Sized + 'a,
          D: 'static {
}
pub trait BatchDataMapper<'a,I,O,D>: Sized
    where I: Debug + Sized + 'a,
      O: Debug + BatchDataType + Sized + 'a,
      D: 'static,
      Self: Deref<Target=<O as BatchDataType>::Type> + 'a {
}
pub struct IdentityMapper<'a,I,D>
    where I: Debug + Sized + 'a,
          Self: Sized + 'a {
    source: &'a I,
    device: PhantomData<D>,
}
impl<'a,I,D> IdentityMapper<'a,I,D>
    where I: Debug + Sized + 'a,
          Self: Sized + 'a {
    pub fn new(source: &'a I) -> Self {
        IdentityMapper {
            source:source,
            device:PhantomData::<D>,
        }
    }
}
impl<'a,I,D> Deref for IdentityMapper<'a,I,D>
    where I: Debug + Sized + 'a,
          Self: Sized + 'a {
    type Target = I;
    fn deref(&self) -> &Self::Target {
        self.source
    }
}
impl<'a,I,D> DataMapper<'a,I,I,D> for IdentityMapper<'a,I,D>
    where I: Debug + Sized + 'a,
          D: 'static,
          Self: Sized + 'a {}
pub struct ScalingMapper<'a,U,I,O,D,const N:usize>
    where I: Debug + Sized + BatchDataType + 'static,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<U,O,N> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'static,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    u: PhantomData<U>,
    i: PhantomData<I>,
    dst: O,
    device: PhantomData<D>,
    l: PhantomData<&'a ()>
}
impl<'a,U,I,O,D,const N:usize> ScalingMapper<'a,U,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<U,O,N> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    pub fn new(device:&'a D ,_: &I, bridged: &O, scale: U) -> Result<ScalingMapper<'a,U,I,O,D,N>,EvaluateError> {
        Ok(ScalingMapper {
            u:PhantomData::<U>,
            i:PhantomData::<I>,
            dst:device.scaling(scale,bridged)?,
            device:PhantomData::<D>,
            l:PhantomData::<&'a ()>
        })
    }
}
impl<'a,U,I,O,D,const N:usize> Deref for ScalingMapper<'a,U,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<U,O,N> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    type Target = O;
    fn deref(&self) -> &Self::Target {
        &self.dst
    }
}
impl<'a,U,I,O,D,const N:usize> DataMapper<'a,I,O,D> for ScalingMapper<'a,U,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<U,O,N> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {}
pub struct BatchScalingMapper<'a,U,I,O,D,const N:usize>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<U,O,N> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    i: PhantomData<I>,
    u: PhantomData<U>,
    dst: <O as BatchDataType>::Type,
    device: PhantomData<D>,
    l: PhantomData<&'a ()>
}
impl<'a,U,I,O,D,const N:usize> BatchScalingMapper<'a,U,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<U,O,N> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    pub fn new(device:&'a D,_: &<I as BatchDataType>::Type, bridged: &<O as BatchDataType>::Type, scale: U)
        -> Result<BatchScalingMapper<'a,U,I,O,D,N>,EvaluateError> {
        Ok(BatchScalingMapper {
            i: PhantomData::<I>,
            u: PhantomData::<U>,
            dst:device.batch_scaling(scale,bridged)?,
            device:PhantomData::<D>,
            l:PhantomData::<&'a ()>,
        })
    }
}
impl<'a,U,I,O,D,const N:usize> Deref for BatchScalingMapper<'a,U,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<U,O,N> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    type Target = <O as BatchDataType>::Type;
    fn deref(&self) -> &Self::Target {
        &self.dst
    }
}
impl<'a,U,I,O,D,const N:usize> BatchDataMapper<'a,I,O,D> for BatchScalingMapper<'a,U,I,O,D,N>
    where I: Debug + Sized + BatchDataType,
          O: Debug + Sized + BatchDataType,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<U,O,N> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug,
          <O as BatchDataType>::Type: BatchSize + Debug,
          Self: Sized {}
pub struct InternalReprMapper<'a,U,SO,I,O,D,const N:usize>
    where I: Debug + Sized + BatchDataType + 'static,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceBridge<U,SO,I,O> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'static,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    u: PhantomData<U>,
    so: PhantomData<SO>,
    i: PhantomData<I>,
    dst: O,
    device: PhantomData<D>,
    l: PhantomData<&'a ()>
}
impl<'a,U,SO,I,O,D,const N:usize> InternalReprMapper<'a,U,SO,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceBridge<U,SO,I,O> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    pub fn new(device:&'a D ,input: &I) -> Result<InternalReprMapper<'a,U,SO,I,O,D,N>,EvaluateError> {
        Ok(InternalReprMapper {
            u:PhantomData::<U>,
            so:PhantomData::<SO>,
            i:PhantomData::<I>,
            dst:device.bridge_forward(&input)?,
            device:PhantomData::<D>,
            l:PhantomData::<&'a ()>
        })
    }
}
impl<'a,U,SO,I,O,D,const N:usize> Deref for InternalReprMapper<'a,U,SO,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceBridge<U,SO,I,O> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    type Target = O;
    fn deref(&self) -> &Self::Target {
        &self.dst
    }
}
impl<'a,U,SO,I,O,D,const N:usize> DataMapper<'a,I,O,D> for InternalReprMapper<'a,U,SO,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceBridge<U,SO,I,O> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {}
pub struct BatchInternalReprMapper<'a,U,SO,I,O,D,const N:usize>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceBridge<U,SO,I,O> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    i: PhantomData<I>,
    u: PhantomData<U>,
    so: PhantomData<SO>,
    dst: <O as BatchDataType>::Type,
    device: PhantomData<D>,
    l: PhantomData<&'a ()>
}
impl<'a,U,SO,I,O,D,const N:usize> BatchInternalReprMapper<'a,U,SO,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceBridge<U,SO,I,O> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    pub fn new(device:&'a D,input: &<I as BatchDataType>::Type)
               -> Result<BatchInternalReprMapper<'a,U,SO,I,O,D,N>,EvaluateError> {
        Ok(BatchInternalReprMapper {
            i: PhantomData::<I>,
            u: PhantomData::<U>,
            so: PhantomData::<SO>,
            dst:device.batch_bridge_forward(input)?,
            device:PhantomData::<D>,
            l:PhantomData::<&'a ()>,
        })
    }
}
impl<'a,U,SO,I,O,D,const N:usize> Deref for BatchInternalReprMapper<'a,U,SO,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'static,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceBridge<U,SO,I,O> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized {
    type Target = <O as BatchDataType>::Type;
    fn deref(&self) -> &Self::Target {
        &self.dst
    }
}
impl<'a,U,SO,I,O,D,const N:usize> BatchDataMapper<'a,I,O,D> for BatchInternalReprMapper<'a,U,SO,I,O,D,N>
    where I: Debug + Sized + BatchDataType,
          O: Debug + Sized + BatchDataType,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceBridge<U,SO,I,O> + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug,
          <O as BatchDataType>::Type: BatchSize + Debug,
          Self: Sized {}
pub struct BatchIdentityMapper<'a,I,D>
    where I: Debug + BatchDataType + Sized + 'a,
          Self: Sized + 'a {
    source: &'a <I as BatchDataType>::Type,
    device: PhantomData<D>,
}
impl<'a,I,D> BatchIdentityMapper<'a,I,D>
    where I: Debug + BatchDataType + Sized + 'a,
          Self: Sized + 'a {
    pub fn new(source: &'a <I as BatchDataType>::Type) -> Self {
        BatchIdentityMapper {
            source:source,
            device:PhantomData::<D>,
        }
    }
}
impl<'a,I,D> Deref for BatchIdentityMapper<'a,I,D>
    where I: Debug + BatchDataType + Sized + 'a,
          Self: Sized + 'a {
    type Target = <I as BatchDataType>::Type;
    fn deref(&self) -> &Self::Target {
        self.source
    }
}
impl<'a,I,D> BatchDataMapper<'a,I,I,D> for BatchIdentityMapper<'a,I,D>
    where I: Debug + BatchDataType + Sized + 'a,
          D: 'static,
          Self: Sized + 'a {}
