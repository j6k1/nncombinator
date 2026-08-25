//! Implementation of the Data Projection Function

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Deref};
use crate::device::bridge::DeviceBridge;
use crate::device::scale::DeviceScale;
use crate::error::TrainingError;
use crate::layer::{BatchDataType, BatchSize};

pub trait DataMapper<'a,I,O,D>: Deref<Target=O> + Sized + 'a
    where I: Debug + Sized + 'a,
          O: Debug + Sized + 'a,
          D: 'static {
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
          O: Debug + Sized + BatchDataType,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + DeviceBridge<U,SO,I,O> + Debug + 'static,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static {
    i: PhantomData<I>,
    u: PhantomData<U>,
    so: PhantomData<SO>,
    scale: &'a SC,
    dst: O,
    device: PhantomData<D>,
}
impl<'a,U,SO,SC,I,O,D,const N:usize> ScalingMapper<'a,U,SO,SC,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + DeviceBridge<U,SO,I,O> + Debug + 'static,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static {
    pub fn new(device:&'a D,source: &'a I,scale: &'a SC) -> Result<ScalingMapper<'a,U,SO,SC,I,O,D,N>,TrainingError> {
        Ok(ScalingMapper {
            i:PhantomData::<I>,
            u:PhantomData::<U>,
            so:PhantomData::<SO>,
            scale:scale,
            dst:device.scaling(scale,&device.bridge_forward(source)?)?,
            device:PhantomData::<D>
        })
    }
}
impl<'a,U,SO,SC,I,O,D,const N:usize> Deref for ScalingMapper<'a,U,SO,SC,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + DeviceBridge<U,SO,I,O> + Debug + 'static,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static {
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
          D: DeviceScale<SO,O,N,Scale=SC> + DeviceBridge<U,SO,I,O> + Debug + 'static,
          <O as BatchDataType>::Type: BatchSize + Debug + 'static {}
pub struct BatchScalingMapper<'a,U,SO,SC,I,O,D,const N:usize>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'a,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + DeviceBridge<U,SO,I,O> + Debug + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'a {
    u: PhantomData<U>,
    so: PhantomData<SO>,
    scale: &'a SC,
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
          <O as BatchDataType>::Type: BatchSize + Debug + 'a {
    pub fn new(device:&'a D,source: &'a <I as BatchDataType>::Type,scale: &'a SC) -> Result<BatchScalingMapper<'a,U,SO,SC,I,O,D,N>,TrainingError> {
        Ok(BatchScalingMapper {
            u: PhantomData::<U>,
            so: PhantomData::<SO>,
            scale:scale,
            dst:device.batch_scaling(scale,&device.batch_bridge_forward(source)?)?,
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
          <O as BatchDataType>::Type: BatchSize + Debug + 'a {
    type Target = <O as BatchDataType>::Type;
    fn deref(&self) -> &Self::Target {
        &self.dst
    }
}
impl<'a,U,SO,SC,I,O,D,const N:usize> DataMapper<'a,I,<O as BatchDataType>::Type,D> for BatchScalingMapper<'a,U,SO,SC,I,O,D,N>
    where I: Debug + Sized + BatchDataType + 'a,
          O: Debug + Sized + BatchDataType + 'a,
          U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          SO: Default + Clone + Copy + Debug + Send + Sync + 'static,
          D: DeviceScale<SO,O,N,Scale=SC> + DeviceBridge<U,SO,I,O> + Debug + 'static,
          <I as BatchDataType>::Type: BatchSize + Debug + 'a,
          <O as BatchDataType>::Type: BatchSize + Debug + 'a {}
