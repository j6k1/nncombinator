//! Implementation of the Data Projection Function

use std::convert::Infallible;
use std::error::Error;
use std::marker::PhantomData;
use std::ops::{Deref, Mul};
use crate::device::{DeviceCpu};
use crate::arr::{Arr, ArrView, SerializedVec};
use crate::cast::Assume;
use crate::error::TypeConvertError;

pub trait DataMapper<'a,I,O,D>: Deref<Target=O> {
}
pub struct IdentityMapper<'a,I> {
    source: &'a I
}
impl<'a,I> IdentityMapper<'a,I> {
    pub fn new(source: &'a I) -> Self {
        IdentityMapper {
            source:source
        }
    }
}
impl<'a,I> Deref for IdentityMapper<'a,I> {
    type Target = I;
    fn deref(&self) -> &Self::Target {
        self.source
    }
}
impl<'a,I> DataMapper<'a,I,I,DeviceCpu> for IdentityMapper<'a,I> {}
pub struct ScalingMapper<'a,I,S,O,D> {
    source: &'a I,
    dst: O,
    scale: PhantomData<S>,
    device: PhantomData<D>,
}
impl<'a,I,S,O,D> ScalingMapper<'a,I,S,O,D>
    where I: Scaling<'a,S,O,D> {
    pub fn new(device:&'a D,source: &'a I,scale: &'a S) -> Result<Self,<I as Scaling<'a,S,O,D>>::Error> {
        Ok(ScalingMapper {
            source:source,
            dst:source.scaling(device,scale)?,
            device:PhantomData::<D>,
            scale:PhantomData::<S>,
        })
    }
}
impl<'a,I,S,O,D> Deref for ScalingMapper<'a,I,S,O,D> {
    type Target = O;
    fn deref(&self) -> &Self::Target {
        &self.dst
    }
}
impl<'a,I,S,O,D> DataMapper<'a,I,O,D> for ScalingMapper<'a,I,S,O,D> {}
pub trait MapperBuilder<'a,I,O,D>
    where I: 'a + Sized {
    type Error: Error;
    type Mapper: DataMapper<'a,I,O,D>;
    fn build(&self,device:&'a D,source: &'a I) -> Result<Self::Mapper,Self::Error>;
}
pub struct IdentityMapperBuilder<'a,I> {
    i: PhantomData<I>,
    l: PhantomData<&'a ()>,
}
impl<'a,I> IdentityMapperBuilder<'a,I> {
    pub fn new() -> IdentityMapperBuilder<'a,I> {
        IdentityMapperBuilder {
            i: PhantomData::<I>,
            l: PhantomData::<&'a ()>,
        }
    }
}
impl<'a,I> MapperBuilder<'a,I,I,DeviceCpu> for IdentityMapperBuilder<'a,I>
    where I: 'a + Sized {
    type Error = Infallible;
    type Mapper = IdentityMapper<'a,I>;
    fn build(&self, device: &'a DeviceCpu, source: &'a I) -> Result<IdentityMapper<'a,I>, Self::Error> {
        Ok(IdentityMapper::new(source))
    }
}
pub trait Scaling<'a,S,O,D> {
    type Error: Error;
    fn scaling(&self,device:&'a D,scale: &'a S) -> Result<O,Self::Error>;
}
impl<'a,SI,SO,const N:usize> Scaling<'a,Arr<SO,N>,Arr<SO,N>,DeviceCpu> for Arr<SI,N>
    where SI: Default + Clone + Copy + Send + Sync + Assume<SO> + 'static,
          SO: Default + Clone + Copy + Send + Sync + Mul<Output=SO> + 'static {
    type Error = Infallible;
    fn scaling(&self,_:&DeviceCpu,scale: &'a Arr<SO,N>) -> Result<Arr<SO,N>,Infallible> {
        let mut dst = Arr::<SO,N>::new();

        for (d,(&i,&s)) in dst.iter_mut().zip(self.iter().zip(scale.iter())) {
            *d = i.assume() * s;
        }

        Ok(dst)
    }
}
pub struct ScalingMapperBuilder<'a,I,S,O,D> {
    scale: &'a S,
    dst: PhantomData<O>,
    source: PhantomData<I>,
    device: PhantomData<D>,
}
impl<'a,I,S,O,D> ScalingMapperBuilder<'a,I,S,O,D> {
    pub fn new(scale:&'a S) -> ScalingMapperBuilder<'a,I,S,O,D> {
        ScalingMapperBuilder {
            scale,
            source: PhantomData::<I>,
            device: PhantomData::<D>,
            dst: PhantomData::<O>,
        }
    }
}
impl<'a,I,S,O,D> MapperBuilder<'a,I,O,D> for ScalingMapperBuilder<'a,I,S,O,D>
    where I: Scaling<'a,S,O,D> + 'a + Sized {
    type Error = <I as Scaling<'a,S,O,D>>::Error;
    type Mapper = ScalingMapper<'a,I,S,O,D>;
    fn build(&self,device:&'a D,source: &'a I) -> Result<ScalingMapper<'a,I,S,O,D>,Self::Error> {
        Ok(ScalingMapper::new(device,source,self.scale)?)
    }
}
impl<'a,SI,SO,const N:usize> Scaling<'a,Arr<SO,N>,SerializedVec<SO,Arr<SO,N>>,DeviceCpu> for SerializedVec<SI,Arr<SI,N>>
    where SI: Default + Clone + Copy + Send + Sync + Assume<SO> + 'static,
          SO: Default + Clone + Copy + Send + Sync + Mul<Output=SO> + 'static,
          SerializedVec<SO,Arr<SI,N>>: From<Vec<Arr<SO,N>>>,
          for<'b> Arr<SO,N>: From<ArrView<'b,SI,N>> {
    type Error = TypeConvertError;
    fn scaling(&self,_:&DeviceCpu,scale: &'a Arr<SO,N>) -> Result<SerializedVec<SO,Arr<SO,N>>,TypeConvertError> {
        Ok(self.iter().map(|i| {
            Arr::<SO,N>::from(i) * scale
        }).collect::<Vec<Arr<SO,N>>>().into())
    }
}
