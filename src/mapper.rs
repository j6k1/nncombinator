//! Implementation of the Data Projection Function

use std::convert::Infallible;
use std::error::Error;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Deref, Mul};
use crate::device::{DeviceCpu};
use crate::arr::{Arr, ArrView, SerializedVec};
use crate::cast::Assume;
use crate::error::TrainingError;

pub trait DataMapper<'b,I,O,D>: Deref<Target=O> + Sized + 'b
    where I: Debug + Sized + 'b,
          O: Debug + Sized + 'b,
          D: Sized + 'b {
}
pub struct IdentityMapper<'b,I,D>
    where I: Debug + Sized + 'b,
          D: Sized + 'b,
          Self: Sized + 'b {
    source: &'b I,
    device: PhantomData<D>,
}
impl<'b,I,D> IdentityMapper<'b,I,D>
    where I: Debug + Sized + 'b,
          D: Sized + 'b,
          Self: Sized + 'b {
    pub fn new(source: &'b I) -> Self {
        IdentityMapper {
            source:source,
            device:PhantomData::<D>,
        }
    }
}
impl<'b,I,D> Deref for IdentityMapper<'b,I,D>
    where I: Debug + Sized + 'b,
          D: Sized + 'b,
          Self: Sized + 'b {
    type Target = I;
    fn deref(&self) -> &Self::Target {
        self.source
    }
}
impl<'b,I,D> DataMapper<'b,I,I,D> for IdentityMapper<'b,I,D>
    where I: Debug + Sized + 'b,
          D: Sized + 'b,
          Self: Sized + 'b {}
pub struct ScalingMapper<'a,'b,I,S,O,D>
    where I: Debug + Sized + 'b,
          O: Debug + Sized + 'b,
          D: Sized + 'b,
          S: Debug + Sized + 'a {
    source: &'b I,
    dst: O,
    scale: &'a S,
    device: PhantomData<D>,
}
impl<'a,'b,I,S,O,D> ScalingMapper<'a,'b,I,S,O,D>
    where I: Scaling<S,O,D> + Debug + Sized + 'b,
          O: Debug + Sized + 'b,
          S: Debug + Sized + 'a,
          D: Sized + 'b,
          TrainingError: From<<I as Scaling<S,O,D>>::Error> {
    pub fn new(device:&'b D,source: &'b I,scale: &'a S) -> Result<ScalingMapper<'a,'b,I,S,O,D>,<I as Scaling<S,O,D>>::Error> {
        Ok(ScalingMapper {
            source:source,
            dst:source.scaling(device,scale)?,
            device:PhantomData::<D>,
            scale:scale
        })
    }
}
impl<'a,'b,I,S,O,D> Deref for ScalingMapper<'a,'b,I,S,O,D>
    where I: Debug + Sized + 'b,
          O: Debug + Sized + 'b,
          S: Debug + Sized + 'a,
          D: Sized + 'b,
          Self: Sized + 'a + 'b {
    type Target = O;
    fn deref(&self) -> &Self::Target {
        &self.dst
    }
}
impl<'a,'b,I,S,O,D> DataMapper<'b,I,O,D> for ScalingMapper<'a,'b,I,S,O,D>
    where I: Scaling<S,O,D> + Debug + Sized + 'b,
          O: Debug + Sized + 'b,
          S: Debug + Sized + 'a,
          D: Sized + 'b,
          Self: Sized + 'a + 'b,
          TrainingError: From<<I as Scaling<S,O,D>>::Error> {}
pub trait MapperBuilder<'a,I,O,D>: Sized
    where for<'b> I: Debug + Sized + 'b,
          for<'b> O: Debug + Sized + 'b,
          for<'b> D: Sized + 'b {
    type Mapper<'b: 'a>: DataMapper<'b,I,O,D> where Self: 'b;
    fn build<'b>(&self,device:&'b D,source: &'b I) -> Result<Self::Mapper<'b>,TrainingError>;
}
#[derive(Debug)]
pub struct IdentityMapperBuilder<I> {
    i: PhantomData<I>
}
impl<I> IdentityMapperBuilder<I> {
    pub fn new() -> IdentityMapperBuilder<I> {
        IdentityMapperBuilder {
            i: PhantomData::<I>
        }
    }
}
impl<'a,I,D> MapperBuilder<'a,I,I,D> for IdentityMapperBuilder<I>
    where for<'b> I: 'b + Debug + Sized,
          for<'b> D: 'b + Sized,
          TrainingError: From<Infallible> {
    type Mapper<'b: 'a> = IdentityMapper<'b,I,D> where Self: 'b;
    fn build<'b>(&self, _: &'b D, source: &'b I) -> Result<IdentityMapper<'b,I,D>,TrainingError> {
        Ok(IdentityMapper::new(source))
    }
}
pub trait Scaling<S,O,D>
    where TrainingError: From<Self::Error> {
    type Error: Error;
    fn scaling(&self,device:&D,scale: &S) -> Result<O,Self::Error>;
}
impl<SI,SO,const N:usize> Scaling<Arr<SO,N>,Arr<SO,N>,DeviceCpu> for Arr<SI,N>
    where SI: Default + Clone + Copy + Send + Sync + Assume<SO> + 'static,
          SO: Default + Clone + Copy + Send + Sync + Mul<Output=SO> + 'static,
          for<'b> Arr<SO,N>: From<&'b Arr<SI,N>> {
    type Error = Infallible;
    fn scaling(&self,_:&DeviceCpu,scale: &Arr<SO,N>) -> Result<Arr<SO,N>,Infallible> {
        Ok(Arr::<SO,N>::from(self) * scale)
    }
}
pub struct ScalingMapperBuilder<'a,I,S,O>
    where for<'b> I: Debug + Sized + 'b,
          for<'b> O: Debug + Sized + 'b,
          for<'b> S: Debug + Sized + 'a,
          Self: Sized + 'a {
    scale: &'a S,
    dst: PhantomData<O>,
    source: PhantomData<I>
}
impl<'a,I,S,O> ScalingMapperBuilder<'a,I,S,O>
    where for<'b> I: Debug + Sized + 'b,
          for<'b> O: Debug + Sized + 'b,
          S: Debug + Sized + 'a {
    pub fn new(scale:&'a S) -> ScalingMapperBuilder<'a,I,S,O> {
        ScalingMapperBuilder {
            scale,
            source: PhantomData::<I>,
            dst: PhantomData::<O>,
        }
    }
}
impl<'a,I,S,O,D> MapperBuilder<'a,I,O,D> for ScalingMapperBuilder<'a,I,S,O>
    where for<'b> I: Debug + Sized + Scaling<S,O,D> + 'b,
          for<'b> S: Debug + Sized + 'a,
          for<'b> O: Debug + Sized + 'b,
          for<'b> D: Sized + 'b,
          S: Debug + Sized + 'a,
          Self: Sized + 'a,
          TrainingError: From<<I as Scaling<S,O,D>>::Error> {
    type Mapper<'b: 'a> = ScalingMapper<'a,'b,I,S,O,D> where Self: 'b;
    fn build<'b>(&self,device:&'b D,source: &'b I) -> Result<ScalingMapper<'a,'b,I,S,O,D>,TrainingError> {
        Ok(ScalingMapper::new(device,source,self.scale)?)
    }
}
impl<SI,SO,const N:usize> Scaling<Arr<SO,N>,SerializedVec<SO,Arr<SO,N>>,DeviceCpu> for SerializedVec<SI,Arr<SI,N>>
    where SI: Default + Clone + Copy + Send + Sync + Assume<SO> + 'static,
          SO: Default + Clone + Copy + Send + Sync + Mul<Output=SO> + 'static,
          SerializedVec<SO,Arr<SI,N>>: From<Vec<Arr<SO,N>>>,
          for<'b> Arr<SO,N>: From<ArrView<'b,SI,N>> {
    type Error = Infallible;
    fn scaling(&self,_:&DeviceCpu,scale: &Arr<SO,N>) -> Result<SerializedVec<SO,Arr<SO,N>>,Infallible> {
        Ok(self.iter().map(|i| {
            Arr::<SO,N>::from(i) * scale
        }).collect::<Vec<Arr<SO,N>>>().into())
    }
}
