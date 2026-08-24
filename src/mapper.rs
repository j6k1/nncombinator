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

pub trait DataMapper<'a,'b,I,O,D>: Deref<Target=O> + Sized + 'a + 'b
    where I: Debug + Sized + 'b,
          O: Debug + Sized + 'b,
          D: Sized + 'b {
}
pub struct IdentityMapper<'a,'b,I>
    where I: Debug + Sized + 'b,
          Self: Sized + 'a + 'b + Deref<Target=I> {
    source: &'b I,
    l: PhantomData<&'a ()>{
}
impl<'a,'b,I> IdentityMapper<'a,'b,I>
    where I: Debug + Sized + 'b,
          Self: Sized + 'a + 'b + Deref<Target=I> {
    pub fn new(source: &'b I) -> Self {
        IdentityMapper {
            source:source,
            l: PhantomData::<&'a ()>,
        }
    }
}
impl<'a,'b,I> Deref for IdentityMapper<'a,'b,I>
    where I: Debug + Sized + 'b,
          Self: Sized + 'a + 'b {
    type Target = I;
    fn deref(&self) -> &Self::Target {
        self.source
    }
}
impl<'a,'b,I,D> DataMapper<'a,'b,I,I,D> for IdentityMapper<'a,'b,I>
    where I: Debug + Sized + 'b,
          D: Sized + 'b,
          Self: Sized + 'a + 'b {}
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
          D: Sized + 'b {
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
    where I: Scaling<S,O,D> + Debug + Sized + 'b,
          O: Debug + Sized + 'b,
          S: Debug + Sized + 'a,
          D: Sized + 'b,
          Self: Sized + 'a + 'b {
    type Target = O;
    fn deref(&self) -> &Self::Target {
        &self.dst
    }
}
impl<'a,'b,I,S,O,D> DataMapper<'a,'b,I,O,D> for ScalingMapper<'a,'b,I,S,O,D>
    where I: Scaling<S,O,D> + Debug + Sized + 'b,
          O: Debug + Sized + 'b,
          S: Debug + Sized + 'a,
          D: Sized + 'b,
          Self: Sized + 'a + 'b + Deref<Target=O>{}
pub trait MapperBuilder<'a,I,O>: Sized + 'a
    where TrainingError: From<Self::Error> {
    type Error: Error;
    type Mapper<'b,D>: DataMapper<'a,'b,I,O,D> where I: Debug + Sized + 'b, O: Debug + Sized + 'b, D: Sized + 'b;
    fn build<'b,D>(&self,device:&'b D,source: &'b I) -> Result<Self::Mapper<'b,D>,Self::Error>;
}
#[derive(Debug)]
pub struct IdentityMapperBuilder<'a,I> {
    i: PhantomData<I>,
    lt: PhantomData<&'a ()>,
}
impl<'a,I> IdentityMapperBuilder<'a,I> {
    pub fn new() -> IdentityMapperBuilder<'a,I> {
        IdentityMapperBuilder {
            i: PhantomData::<I>,
            lt: PhantomData::<&'a ()>,
        }
    }
}
impl<'a,I,O> MapperBuilder<'a,I,O> for IdentityMapperBuilder<'a,I>
    where for<'b> IdentityMapper<'a,'b,I>: 'a + 'b + Sized,
          for<'b> I: 'b + Debug + Sized {
    type Error = Infallible;
    type Mapper<'b,D> = IdentityMapper<'a,'b,I>;
    fn build<'b,D>(&self, _: &'b D, source: &'b I) -> Result<IdentityMapper<'a,'b,I>, Self::Error> {
        Ok(IdentityMapper::new(source))
    }
}
pub trait Scaling<S,O,D> {
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
    where I: Debug + Sized,
          for<'b> O: Debug + Sized + 'b,
          for<'b> S: Debug + Sized + 'a,
          Self: Sized + 'a {
    scale: &'a S,
    dst: PhantomData<O>,
    source: PhantomData<I>
}
impl<'a,I,S,O> ScalingMapperBuilder<'a,I,S,O>
    where I: Debug + Sized,
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
impl<'a,I,S,O> MapperBuilder<'a,I,O> for ScalingMapperBuilder<'a,I,S,O>
    where for<'b> I: Debug + Sized + 'b,
          for<'b> O: Debug + Sized + 'b,
          S: Debug + Sized + 'a,
          for<'b> Self: 'a + 'b + Sized,
          Self: Sized + 'a + Deref<Target=O> {
    type Error = TrainingError;
    type Mapper<'b,D> = ScalingMapper<'a,'b,I,S,O,D>;
    fn build<'b,D>(&self,device:&'b D,source: &'b I) -> Result<ScalingMapper<'a,'b,I,S,O,D>,Self::Error> {
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
