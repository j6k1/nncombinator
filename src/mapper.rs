//! Implementation of the Data Projection Function

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Deref};
use crate::device::bridge::DeviceBridge;
use crate::device::scale::DeviceScale;
use crate::error::{EvaluateError, TrainingError};
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
pub struct ScalingMapper<'a,U,SO,SC,I,O,D,const N:usize>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'a,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + Debug + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized + 'a {
    u: PhantomData<U>,
    so: PhantomData<SO>,
    _scale: &'a SC,
    i: PhantomData<I>,
    dst: O,
    device: PhantomData<D>,
}
impl<'a,U,SO,SC,I,O,D,const N:usize> ScalingMapper<'a,U,SO,SC,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'a,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + Debug + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized + 'a {
    pub fn new(device:&'a D ,_: &'a I, bridged: O, scale: &'a SC) -> Result<ScalingMapper<'a,U,SO,SC,I,O,D,N>,EvaluateError> {
        Ok(ScalingMapper {
            u:PhantomData::<U>,
            so:PhantomData::<SO>,
            i:PhantomData::<I>,
            _scale:scale,
            dst:device.scaling(scale,&bridged)?,
            device:PhantomData::<D>
        })
    }
}
impl<'a,U,SO,SC,I,O,D,const N:usize> Deref for ScalingMapper<'a,U,SO,SC,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'a,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + Debug + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized + 'a {
    type Target = O;
    fn deref(&self) -> &Self::Target {
        &self.dst
    }
}
impl<'a,U,SO,SC,I,O,D,const N:usize> DataMapper<'a,I,O,D> for ScalingMapper<'a,U,SO,SC,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'a,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + Debug + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static,
          Self: Sized + 'a {}
pub struct BatchScalingMapper<'a,U,SO,SC,I,O,D,const N:usize>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'a,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + Debug + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'a,
          Self: Sized + 'a {
    i: PhantomData<I>,
    u: PhantomData<U>,
    so: PhantomData<SO>,
    _scale: &'a SC,
    dst: <O as BatchDataType>::Type,
    device: PhantomData<D>,
}
impl<'a,U,SO,SC,I,O,D,const N:usize> BatchScalingMapper<'a,U,SO,SC,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'a,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + DeviceBridge<U,SO,I,O> + Debug + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'a,
          Self: Sized + 'a {
    pub fn new(device:&'a D,_: &'a <I as BatchDataType>::Type, bridged: <O as BatchDataType>::Type, scale: &'a SC)
        -> Result<BatchScalingMapper<'a,U,SO,SC,I,O,D,N>,EvaluateError> {
        Ok(BatchScalingMapper {
            i: PhantomData::<I>,
            u: PhantomData::<U>,
            so: PhantomData::<SO>,
            _scale:scale,
            dst:device.batch_scaling(scale,&bridged)?,
            device:PhantomData::<D>
        })
    }
}
impl<'a,U,SO,SC,I,O,D,const N:usize> Deref for BatchScalingMapper<'a,U,SO,SC,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'a,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + DeviceBridge<U,SO,I,O> + Debug + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'a,
          Self: Sized + 'a {
    type Target = <O as BatchDataType>::Type;
    fn deref(&self) -> &Self::Target {
        &self.dst
    }
}
impl<'a,U,SO,SC,I,O,D,const N:usize> BatchDataMapper<'a,I,O,D> for BatchScalingMapper<'a,U,SO,SC,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'a,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + DeviceBridge<U,SO,I,O> + Debug + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'a,
          Self: Sized + 'a {}
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
