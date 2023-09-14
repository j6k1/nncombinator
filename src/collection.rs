//! Definition and implementation of various collection typess
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use rayon::prelude::{ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use crate::arr::{AsView, MakeView, SerializedVec, SerializedVecView, SliceSize};
use crate::derive_arithmetic;

/// Trait to transform the shape
pub trait ReShape<A,R> {
    fn reshape(self,args:A) -> R;
}
/// A type that expresses a broadcast of the inner type when computing with the inner type and its type collection
#[derive(Debug,Clone)]
pub struct Broadcast<T>(pub T) where T: Clone;

derive_arithmetic! (Broadcast<T> > &'a SerializedVec<U,T> = SerializedVec<U,T>);
derive_arithmetic! (Broadcast<T> > SerializedVec<U,T> = SerializedVec<U,T>);

derive_arithmetic! (&'a SerializedVec<U,T> > Broadcast<T> = SerializedVec<U,T>);
derive_arithmetic! (SerializedVec<U,T> > Broadcast<T> = SerializedVec<U,T>);

derive_arithmetic! (Broadcast<T> > &'a SerializedVecView<'a,U,T> = SerializedVec<U,T>);
derive_arithmetic! (Broadcast<T> > SerializedVecView<'a,U,T> = SerializedVec<U,T>);

derive_arithmetic! (&'a SerializedVecView<'a,U,T> > Broadcast<T> = SerializedVec<U,T>);
derive_arithmetic! (SerializedVecView<'a,U,T> > Broadcast<T> = SerializedVec<U,T>);
