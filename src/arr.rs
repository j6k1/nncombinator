//! Array-related data structures such as fixed-length arrays

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Div, Index, IndexMut, Mul, Neg, Sub};
use std;
use rayon::iter::{plumbing};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use crate::error::{IndexOutBoundError, SizeMismatchError};
use crate::mem::{AsRawMutSlice, AsRawSlice};
use crate::ope::Sum;

/// Trait that returns the number of elements in the slice held by itself
pub trait SliceSize {
    const SIZE: usize;

    fn slice_size() -> usize {
        Self::SIZE
    }
}
/// Fixed-length one-dimensional array implementation
#[derive(Debug,Eq,PartialEq)]
pub struct Arr<T,const N:usize> where T: Default + Clone + Send {
    arr:Box<[T]>
}
impl<T,const N:usize> Arr<T,N> where T: Default + Clone + Send {
    /// Create an instance of Arr
    pub fn new() -> Arr<T,N> {
        let mut arr = Vec::with_capacity(N);
        arr.resize_with(N,Default::default);

        Arr {
            arr:arr.into_boxed_slice()
        }
    }
}
impl<T,const N:usize> Deref for Arr<T,N> where T: Default + Clone + Send {
    type Target = Box<[T]>;
    fn deref(&self) -> &Self::Target {
        &self.arr
    }
}
impl<T,const N:usize> DerefMut for Arr<T,N> where T: Default + Clone + Send  {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.arr
    }
}
impl<T,const N:usize> Clone for Arr<T,N> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        Arr {
            arr:self.arr.clone()
        }
    }
}
impl<'a,T,const N:usize> MakeView<'a,T> for Arr<T,N> where T: Default + Clone + Send + Sync + 'a {
    type ViewType = ArrView<'a,T,N>;

    fn make_view(arr: &'a [T]) -> Self::ViewType {
        ArrView {
            arr: arr
        }
    }
}
impl<'a,T,const N:usize> MakeViewMut<'a,T> for Arr<T,N> where T: Default + Clone + Send + Sync + 'a {
    type ViewType = ArrView<'a,T,N>;

    fn make_view_mut(arr: &'a mut [T]) -> Self::ViewType {
        ArrView {
            arr: arr
        }
    }
}
impl<'data,U,const N:usize> From<ArrView<'data,U,N>> for Arr<U,N> where U: Default + Clone + Copy + Send {
    fn from(view: ArrView<'data,U, N>) -> Self {
        let mut v = Vec::new();

        v.extend_from_slice(view.arr);

        Arr {
            arr:v.into_boxed_slice()
        }
    }
}
impl<T,const N:usize> TryFrom<Vec<T>> for Arr<T,N> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
        let s = v.into_boxed_slice();

        if s.len() != N {
            Err(SizeMismatchError(s.len(),N))
        } else {
            Ok(Arr {
                arr: s
            })
        }
    }
}
impl<T,const N:usize> TryFrom<Box<[T]>> for Arr<T,N> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(arr: Box<[T]>) -> Result<Self, Self::Error> {
        if arr.len() != N {
            Err(SizeMismatchError(arr.len(),N))
        } else {
            Ok(Arr { arr: arr })
        }
    }
}
impl<T,const N:usize> From<Arr<T,N>> for Box<[T]> where T: Default + Clone + Send {
    fn from(value: Arr<T,N>) -> Self {
        value.arr
    }
}
impl<T,const N:usize> SliceSize for Arr<T,N> where T: Default + Clone + Send {
    const SIZE: usize = N;
}
impl<'a,T,const N:usize> Add<T> for &'a Arr<T,N> where T: Add<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn add(self, rhs: T) -> Self::Output {
        let mut r = Arr::new();

        for (it,&l) in r.iter_mut().zip(self.iter()) {
            *it = l + rhs;
        }
        r
    }
}
impl<T,const N:usize> Add<T> for Arr<T,N> where T: Add<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn add(self, rhs: T) -> Self::Output {
        &self + rhs
    }
}
impl<'a,T,const N:usize> Sub<T> for &'a Arr<T,N> where T: Sub<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn sub(self, rhs: T) -> Self::Output {
        let mut r = Arr::new();

        for (it,&l) in r.iter_mut().zip(self.iter()) {
            *it = l - rhs;
        }
        r
    }
}
impl<T,const N:usize> Sub<T> for Arr<T,N> where T: Sub<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn sub(self, rhs: T) -> Self::Output {
        &self - rhs
    }
}
impl<'a,T,const N:usize> Mul<T> for &'a Arr<T,N> where T: Mul<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut r = Arr::new();

        for (it,&l) in r.iter_mut().zip(self.iter()) {
            *it = l * rhs;
        }
        r
    }
}
impl<T,const N:usize> Mul<T> for Arr<T,N> where T: Mul<T> + Mul<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn mul(self, rhs: T) -> Self::Output {
        &self * rhs
    }
}
impl<'a,T,const N:usize> Div<T> for &'a Arr<T,N> where T: Div<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn div(self, rhs: T) -> Self::Output {
        let mut r = Arr::new();

        for (it,&l) in r.iter_mut().zip(self.iter()) {
            *it = l / rhs;
        }
        r
    }
}
impl<T,const N:usize> Div<T> for Arr<T,N> where T: Div<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn div(self, rhs: T) -> Self::Output {
        &self / rhs
    }
}
impl<T,const N:usize> Add<&Arr<T,N>> for &Arr<T,N>
    where T: Add<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn add(self, rhs: &Arr<T,N>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(&l,&r)| l + r)
            .collect::<Vec<T>>().try_into().expect("An error occurred in the add of Arrand Arr.")
    }
}
impl<T,const N:usize> Add<&Arr<T,N>> for Arr<T,N>
    where T: Add<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn add(self, rhs: &Arr<T, N>) -> Self::Output {
        &self + rhs
    }
}
impl<T,const N:usize> Add<Arr<T,N>> for &Arr<T,N>
    where T: Add<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn add(self, rhs: Arr<T, N>) -> Self::Output {
        self + &rhs
    }
}
impl<T,const N:usize> Add<Arr<T,N>> for Arr<T,N>
    where T: Add<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn add(self, rhs: Arr<T, N>) -> Self::Output {
        &self + &rhs
    }
}
impl<T,const N:usize> Sub<&Arr<T,N>> for &Arr<T,N>
    where T: Sub<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn sub(self, rhs: &Arr<T,N>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(&l,&r)| l - r)
            .collect::<Vec<T>>().try_into().expect("An error occurred in the sub of Arr and Arr.")
    }
}
impl<T,const N:usize> Sub<&Arr<T,N>> for Arr<T,N>
    where T: Sub<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn sub(self, rhs: &Arr<T, N>) -> Self::Output {
        &self - rhs
    }
}
impl<T,const N:usize> Sub<Arr<T,N>> for &Arr<T,N>
    where T: Sub<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn sub(self, rhs: Arr<T, N>) -> Self::Output {
        self - &rhs
    }
}
impl<T,const N:usize> Sub<Arr<T,N>> for Arr<T,N>
    where T: Sub<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn sub(self, rhs: Arr<T, N>) -> Self::Output {
        &self - &rhs
    }
}
impl<T,const N:usize> Mul<&Arr<T,N>> for &Arr<T,N>
    where T: Mul<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn mul(self, rhs: &Arr<T,N>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(&l,&r)| l * r)
            .collect::<Vec<T>>().try_into().expect("An error occurred when multiplying Arr by Arr.")
    }
}
impl<T,const N:usize> Mul<&Arr<T,N>> for Arr<T,N>
    where T: Mul<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn mul(self, rhs: &Arr<T, N>) -> Self::Output {
        &self * rhs
    }
}
impl<T,const N:usize> Mul<Arr<T,N>> for &Arr<T,N>
    where T: Mul<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn mul(self, rhs: Arr<T, N>) -> Self::Output {
        self * &rhs
    }
}
impl<T,const N:usize> Mul<Arr<T,N>> for Arr<T,N>
    where T: Mul<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn mul(self, rhs: Arr<T, N>) -> Self::Output {
        &self * &rhs
    }
}
impl<T,const N:usize> Div<&Arr<T,N>> for &Arr<T,N>
    where T: Div<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn div(self, rhs: &Arr<T,N>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(&l,&r)| l / r)
            .collect::<Vec<T>>().try_into().expect("An error occurred in the division of Arr and Arr.")
    }
}
impl<T,const N:usize> Div<&Arr<T,N>> for Arr<T,N>
    where T: Div<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn div(self, rhs: &Arr<T, N>) -> Self::Output {
        &self / rhs
    }
}
impl<T,const N:usize> Div<Arr<T,N>> for &Arr<T,N>
    where T: Div<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn div(self, rhs: Arr<T, N>) -> Self::Output {
        self / &rhs
    }
}
impl<T,const N:usize> Div<Arr<T,N>> for Arr<T,N>
    where T: Div<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn div(self, rhs: Arr<T, N>) -> Self::Output {
        &self / &rhs
    }
}
impl<'a,T,const N:usize> Neg for &'a Arr<T,N>
    where T: Neg<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn neg(self) -> Self::Output {
        self.par_iter().map(|&v| -v)
            .collect::<Vec<T>>().try_into().expect("An error occurred during the sign reversal operation for each element of Arr.")
    }
}
impl<T,const N:usize> Neg for Arr<T,N>
    where T: Neg<Output=T> + Clone + Copy + Default + Send + Sync + 'static,
          for<'a> &'a Arr<T,N>: Neg<Output = Arr<T,N>> {
    type Output = Arr<T,N>;

    fn neg(self) -> Self::Output {
        (&self).neg()
    }
}
impl<'a,T,const N:usize> AsRawSlice<T> for Arr<T,N> where T: Default + Clone + Send {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N:usize> AsRawMutSlice<'a,T> for Arr<T,N> where T: Default + Clone + Send {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T] {
        &mut self.arr
    }
}
/// Fixed-length 2D array implementation
#[derive(Debug,Eq,PartialEq)]
pub struct Arr2<T,const N1:usize, const N2:usize> where T: Default {
    arr:Box<[T]>
}
impl<T,const N1:usize, const N2:usize> Arr2<T,N1,N2> where T: Default {
    /// Create an instance of Arr2
    pub fn new() -> Arr2<T,N1,N2> {
        let mut arr = Vec::with_capacity(N1 * N2);
        arr.resize_with(N1*N2,Default::default);

        Arr2 {
            arr:arr.into_boxed_slice()
        }
    }

    /// Obtaining a immutable iterator
    pub fn iter<'a>(&'a self) -> Arr2Iter<'a,T,N2> {
        Arr2Iter(&*self.arr)
    }

    /// Obtaining a mutable iterator
    pub fn iter_mut<'a>(&'a mut self) -> Arr2IterMut<'a,T,N2> {
        Arr2IterMut(&mut *self.arr)
    }
}
impl<T,const N1:usize, const N2:usize> Clone for Arr2<T,N1,N2> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        Arr2 {
            arr:self.arr.clone()
        }
    }
}
impl<T,const N1:usize, const N2:usize> Index<(usize,usize)> for Arr2<T,N1,N2> where T: Default {
    type Output = T;

    fn index(&self, (y,x): (usize, usize)) -> &Self::Output {
        if y >= N1 {
            panic!("index out of bounds: the len is {} but the index is {}",N1,y);
        } else if x >= N2 {
            panic!("index out of bounds: the len is {} but the index is {}",N2,x);
        }
        &self.arr[y * N2 + x]
    }
}
impl<T,const N1:usize, const N2:usize> IndexMut<(usize,usize)> for Arr2<T,N1,N2> where T: Default {
    fn index_mut(&mut self, (y,x): (usize, usize)) -> &mut Self::Output {
        if y >= N1 {
            panic!("index out of bounds: the len is {} but the index is {}",N1,y);
        } else if x >= N2 {
            panic!("index out of bounds: the len is {} but the index is {}",N2,x);
        }
        &mut self.arr[y * N2 + x]
    }
}
impl<'a,T,const N1:usize, const N2: usize> AsRawSlice<T> for Arr2<T,N1,N2> where T: Default + Clone + Send {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N1:usize, const N2:usize> AsRawMutSlice<'a,T> for Arr2<T,N1,N2> where T: Default + Clone + Send {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T] {
        &mut self.arr
    }
}
impl<T,const N1:usize, const N2: usize> TryFrom<Vec<T>> for Arr2<T,N1,N2> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
        if v.len() != N1 * N2 {
            Err(SizeMismatchError(v.len(),N1 * N2))
        } else {
            let arr = v.into_boxed_slice();

            Ok(Arr2 {
                arr:arr
            })
        }
    }
}
impl<T,const N1:usize, const N2: usize> TryFrom<Vec<Arr<T,N2>>> for Arr2<T,N1,N2> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(v: Vec<Arr<T,N2>>) -> Result<Self, Self::Error> {
        if v.len() != N1 {
            Err(SizeMismatchError(v.len(),N1))
        } else {
            let mut buffer = Vec::with_capacity(N1 * N2);

            for v in v.into_iter() {
                buffer.extend_from_slice(&v);
            }
            Ok(Arr2 {
                arr: buffer.into_boxed_slice()
            })
        }
    }
}
impl<T,const N1:usize,const N2:usize> SliceSize for Arr2<T,N1,N2> where T: Default + Clone + Send {
    const SIZE: usize = N1 * N2;
}
/// Fixed-length 3D array implementation
#[derive(Debug,Eq,PartialEq)]
pub struct Arr3<T,const N1:usize, const N2:usize, const N3:usize> where T: Default {
    arr:Box<[T]>
}
impl<T,const N1:usize,const N2:usize,const N3:usize> Clone for Arr3<T,N1,N2,N3> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        Arr3 {
            arr:self.arr.clone()
        }
    }
}
impl<T,const N1:usize,const N2:usize,const N3:usize> Arr3<T,N1,N2,N3> where T: Default {
    /// Create an instance of Arr3
    pub fn new() -> Arr3<T,N1,N2,N3> {
        let mut arr = Vec::with_capacity(N1 * N2 * N3);
        arr.resize_with(N1*N2*N3,Default::default);

        Arr3 {
            arr:arr.into_boxed_slice()
        }
    }

    /// Obtaining a immutable iterator
    pub fn iter<'a>(&'a self) -> Arr3Iter<'a,T,N2,N3> {
        Arr3Iter(&*self.arr)
    }

    /// Obtaining a mutable iterator
    pub fn iter_mut<'a>(&'a mut self) -> Arr3IterMut<'a,T,N2,N3> {
        Arr3IterMut(&mut *self.arr)
    }
}
impl<T,const N1:usize, const N2:usize, const N3:usize> Index<(usize,usize,usize)> for Arr3<T,N1,N2,N3> where T: Default {
    type Output = T;

    fn index(&self, (z,y,x): (usize, usize, usize)) -> &Self::Output {
        if z >= N1 {
            panic!("index out of bounds: the len is {} but the index is {}",N1,z);
        } else if y >= N2 {
            panic!("index out of bounds: the len is {} but the index is {}",N2,y);
        } else if x >= N3 {
            panic!("index out of bounds: the len is {} but the index is {}",N3,x);
        }
        &self.arr[z * N2 * N3 + y * N3 + x]
    }
}
impl<T,const N1:usize, const N2:usize, const N3:usize> IndexMut<(usize,usize,usize)> for Arr3<T,N1,N2,N3> where T: Default {
    fn index_mut(&mut self, (z,y,x): (usize, usize, usize)) -> &mut Self::Output {
        if z >= N1 {
            panic!("index out of bounds: the len is {} but the index is {}",N1,z);
        } else if y >= N2 {
            panic!("index out of bounds: the len is {} but the index is {}",N2,y);
        } else if x >= N3 {
            panic!("index out of bounds: the len is {} but the index is {}",N3,x);
        }
        &mut self.arr[z * N2 * N3 + y * N3 + x]
    }
}
impl<T,const N1:usize, const N2: usize, const N3:usize> TryFrom<Vec<Arr2<T,N2,N3>>> for Arr3<T,N1,N2,N3> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(v: Vec<Arr2<T,N2,N3>>) -> Result<Self, Self::Error> {
        if v.len() != N1 {
            Err(SizeMismatchError(v.len(),N1))
        } else {
            let mut buffer = Vec::with_capacity(N1 * N2 * N3);

            for v in v.into_iter() {
                buffer.extend_from_slice(&v.arr);
            }
            Ok(Arr3 {
                arr: buffer.into_boxed_slice()
            })
        }
    }
}
impl<T,const N1:usize,const N2:usize,const N3:usize> SliceSize for Arr3<T,N1,N2,N3>
    where T: Default + Clone + Send {
    const SIZE: usize = N1 * N2 * N3;
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> AsRawSlice<T> for Arr3<T,N1,N2,N3> where T: Default + Clone + Send {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> AsRawMutSlice<'a,T> for Arr3<T,N1,N2,N3> where T: Default + Clone + Send {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T] {
        &mut self.arr
    }
}
/// Fixed-length 4D array implementation
#[derive(Debug,Eq,PartialEq)]
pub struct Arr4<T,const N1:usize, const N2:usize, const N3:usize, const N4:usize> where T: Default {
    arr:Box<[T]>
}
impl<T,const N1:usize,const N2:usize,const N3:usize, const N4:usize> Clone for Arr4<T,N1,N2,N3,N4> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        Arr4 {
            arr:self.arr.clone()
        }
    }
}
impl<T,const N1:usize,const N2:usize,const N3:usize, const N4:usize> Arr4<T,N1,N2,N3,N4> where T: Default {
    /// Create an instance of Arr4
    pub fn new() -> Arr4<T,N1,N2,N3,N4> {
        let mut arr = Vec::with_capacity(N1 * N2 * N3 * N4);
        arr.resize_with(N1*N2*N3*N4,Default::default);

        Arr4 {
            arr:arr.into_boxed_slice()
        }
    }

    /// Obtaining a immutable iterator
    pub fn iter<'a>(&'a self) -> Arr4Iter<'a,T,N2,N3,N4> {
        Arr4Iter(&*self.arr)
    }

    /// Obtaining a mutable iterator
    pub fn iter_mut<'a>(&'a mut self) -> Arr4IterMut<'a,T,N2,N3,N4> {
        Arr4IterMut(&mut *self.arr)
    }
}
impl<T,const N1:usize, const N2:usize, const N3:usize, const N4:usize> Index<(usize,usize,usize,usize)> for Arr4<T,N1,N2,N3,N4>
    where T: Default {
    type Output = T;

    fn index(&self, (i,z,y,x): (usize, usize, usize, usize)) -> &Self::Output {
        if i >= N1 {
            panic!("index out of bounds: the len is {} but the index is {}",N1,i);
        } else if z >= N2 {
            panic!("index out of bounds: the len is {} but the index is {}",N2,z);
        } else if y >= N3 {
            panic!("index out of bounds: the len is {} but the index is {}",N3,y);
        } else if x >= N4 {
            panic!("index out of bounds: the len is {} but the index is {}",N4,x);
        }
        &self.arr[i * N2 * N3 * N4 + z * N3 * N4 + y * N4 + x]
    }
}
impl<T,const N1:usize, const N2:usize, const N3:usize, const N4:usize> IndexMut<(usize,usize,usize,usize)> for Arr4<T,N1,N2,N3,N4>
    where T: Default {
    fn index_mut(&mut self, (i,z,y,x): (usize, usize, usize, usize)) -> &mut Self::Output {
        if i >= N1 {
            panic!("index out of bounds: the len is {} but the index is {}",N1,i);
        } else if z >= N2 {
            panic!("index out of bounds: the len is {} but the index is {}",N2,z);
        } else if y >= N3 {
            panic!("index out of bounds: the len is {} but the index is {}",N3,y);
        } else if x >= N4 {
            panic!("index out of bounds: the len is {} but the index is {}",N4,x);
        }
        &mut self.arr[i * N2 * N3 * N4 + z * N3 * N4 + y * N4 + x]
    }
}
impl<T,const N1:usize, const N2: usize, const N3:usize, const N4:usize> TryFrom<Vec<Arr3<T,N2,N3,N4>>>
    for Arr4<T,N1,N2,N3,N4> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(v: Vec<Arr3<T,N2,N3,N4>>) -> Result<Self, Self::Error> {
        if v.len() != N1 {
            Err(SizeMismatchError(v.len(),N1))
        } else {
            let mut buffer = Vec::with_capacity(N1 * N2 * N3 * N4);

            for v in v.into_iter() {
                buffer.extend_from_slice(&v.arr);
            }
            Ok(Arr4 {
                arr: buffer.into_boxed_slice()
            })
        }
    }
}
impl<T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> SliceSize for Arr4<T,N1,N2,N3,N4>
    where T: Default + Clone + Send {
    const SIZE: usize = N1 * N2 * N3 * N4;
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> AsRawSlice<T> for Arr4<T,N1,N2,N3,N4> where T: Default + Clone + Send {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> AsRawMutSlice<'a,T> for Arr4<T,N1,N2,N3,N4> where T: Default + Clone + Send {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T] {
        &mut self.arr
    }
}
/// Implementation of an immutable view of a fixed-length 1D array
#[derive(Debug,Eq,PartialEq)]
pub struct ArrView<'a,T,const N:usize> {
    pub(crate) arr:&'a [T]
}
impl<'a,T,const N:usize> Clone for ArrView<'a,T,N> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        ArrView {
            arr:self.arr
        }
    }
}
impl<'a,T,const N:usize> Copy for ArrView<'a,T,N> where Self: Clone {}
impl<'a,T,const N:usize> Deref for ArrView<'a,T,N> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.arr
    }
}
impl<'a,T,const N:usize> From<&'a Arr<T,N>> for ArrView<'a,T,N> where T: Default + Clone + Send {
    fn from(value: &'a Arr<T, N>) -> Self {
        ArrView {
            arr: &value.arr
        }
    }
}
impl<'a,T,const N:usize> TryFrom<&'a [T]> for ArrView<'a,T,N> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(arr: &'a [T]) -> Result<Self, Self::Error> {
        if arr.len() != N {
            Err(SizeMismatchError(arr.len(),N))
        } else {
            Ok(ArrView { arr: arr })
        }
    }
}
impl<'a,T,const N:usize> AsRawSlice<T> for ArrView<'a,T,N> where T: Default + Clone {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N:usize> SliceSize for ArrView<'a,T,N> where T: Default + Clone + Send {
    const SIZE: usize = N;
}
impl<'a,T,const N:usize> Add<T> for &'a ArrView<'a,T,N>
    where T: Add<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn add(self, rhs: T) -> Self::Output {
        let mut r = Arr::new();

        for (it,&l) in r.iter_mut().zip(self.iter()) {
            *it = l + rhs;
        }
        r
    }
}
impl<'a,T,const N:usize> Add<T> for ArrView<'a,T,N>
    where T: Add<Output=T> + Clone + Copy + Default + Send,
          for<'b> &'b ArrView<'b,T,N>: Add<T,Output=Arr<T,N>> {
    type Output = Arr<T,N>;

    fn add(self, rhs: T) -> Self::Output {
        &self + rhs
    }
}
impl<'a,T,const N:usize> Sub<T> for &'a ArrView<'a,T,N>
    where T: Sub<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn sub(self, rhs: T) -> Self::Output {
        let mut r = Arr::new();

        for (it,&l) in r.iter_mut().zip(self.iter()) {
            *it = l - rhs;
        }
        r
    }
}
impl<'a,T,const N:usize> Sub<T> for ArrView<'a,T,N>
    where T: Sub<Output=T> + Clone + Copy + Default + Send,
          for<'b> &'b ArrView<'b,T,N>: Sub<T,Output=Arr<T,N>> {
    type Output = Arr<T,N>;

    fn sub(self, rhs: T) -> Self::Output {
        &self - rhs
    }
}
impl<'a,T,const N:usize> Mul<T> for &'a ArrView<'a,T,N>
    where T: Mul<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut r = Arr::new();

        for (it,&l) in r.iter_mut().zip(self.iter()) {
            *it = l * rhs;
        }
        r
    }
}
impl<'a,T,const N:usize> Mul<T> for ArrView<'a,T,N>
    where T: Mul<Output=T> + Clone + Copy + Default + Send,
          for<'b> &'b ArrView<'b,T,N>: Mul<T,Output=Arr<T,N>> {
    type Output = Arr<T,N>;

    fn mul(self, rhs: T) -> Self::Output {
        &self * rhs
    }
}
impl<'a,T,const N:usize> Div<T> for &'a ArrView<'a,T,N>
    where T: Div<Output=T> + Clone + Copy + Default + Send {
    type Output = Arr<T,N>;

    fn div(self, rhs: T) -> Self::Output {
        let mut r = Arr::new();

        for (it,&l) in r.iter_mut().zip(self.iter()) {
            *it = l / rhs;
        }
        r
    }
}
impl<'a,T,const N:usize> Div<T> for ArrView<'a,T,N>
    where T: Div<Output=T> + Clone + Copy + Default + Send,
          for<'b> &'b ArrView<'b,T,N>: Div<T,Output=Arr<T,N>> {
    type Output = Arr<T,N>;

    fn div(self, rhs: T) -> Self::Output {
        &self / rhs
    }
}
impl<'a,'b: 'a,T,const N:usize> Add<ArrView<'b,T,N>> for ArrView<'a,T,N>
    where T: Add<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn add(self, rhs: ArrView<'b,T,N>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(&l,&r)| l + r)
            .collect::<Vec<T>>().try_into().expect("An error occurred in the add of ArrView and ArrView.")
    }
}
impl<'a,'b: 'a,T,const N:usize> Sub<ArrView<'b,T,N>> for ArrView<'a,T,N>
    where T: Sub<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn sub(self, rhs: ArrView<'b,T,N>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(&l,&r)| l - r)
            .collect::<Vec<T>>().try_into().expect("An error occurred in the add of ArrView and ArrView.")
    }
}
impl<'a,'b: 'a,T,const N:usize> Mul<ArrView<'b,T,N>> for ArrView<'a,T,N>
    where T: Mul<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn mul(self, rhs: ArrView<'b,T,N>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(&l,&r)| l * r)
            .collect::<Vec<T>>().try_into().expect("An error occurred when multiplying ArrView by ArrView.")
    }
}
impl<'a,'b: 'a,T,const N:usize> Div<ArrView<'b,T,N>> for ArrView<'a,T,N>
    where T: Div<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn div(self, rhs: ArrView<'b,T,N>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(&l,&r)| l / r)
            .collect::<Vec<T>>().try_into().expect("An error occurred in the division of ArrView and ArrView.")
    }
}
impl<'a,T,const N:usize> Neg for ArrView<'a,T,N>
    where T: Neg<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
    type Output = Arr<T,N>;

    fn neg(self) -> Self::Output {
        self.par_iter().map(|&v| -v)
            .collect::<Vec<T>>().try_into().expect("An error occurred during the sign reversal operation for each element of ArrView.")
    }
}
/// Implementation of an mutable view of a fixed-length 1D array
#[derive(Debug,Eq,PartialEq)]
pub struct ArrViewMut<'a,T,const N:usize> {
    pub(crate) arr:&'a mut [T]
}
impl<'a,T,const N:usize> Deref for ArrViewMut<'a,T,N> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.arr
    }
}
impl<'a,T,const N:usize> DerefMut for ArrViewMut<'a,T,N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.arr
    }
}
impl<'a,T,const N:usize> TryFrom<&'a mut [T]> for ArrViewMut<'a,T,N> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(arr: &'a mut [T]) -> Result<Self, Self::Error> {
        if arr.len() != N {
            Err(SizeMismatchError(arr.len(),N))
        } else {
            Ok(ArrViewMut { arr: arr })
        }
    }
}
impl<'a,T,const N:usize> SliceSize for ArrViewMut<'a,T,N> where T: Default + Clone + Send {
    const SIZE: usize = N;
}
impl<'a,T,const N:usize> AsRawSlice<T> for ArrViewMut<'a,T,N> where T: Default + Clone {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N:usize> AsRawMutSlice<'a,T> for ArrViewMut<'a,T,N> where T: Default + Clone + Send {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T] {
        &mut self.arr
    }
}
/// Implementation of a immutable view of a fixed-length 2D array
#[derive(Debug,Eq,PartialEq)]
pub struct Arr2View<'a,T,const N1:usize,const N2:usize> {
    arr:&'a [T]
}
impl<'a,T,const N1:usize,const N2:usize> Arr2View<'a,T,N1,N2> {
    pub fn iter(&'a self) -> Arr2Iter<'a,T,N2> {
        Arr2Iter(&self.arr)
    }
}
impl<'a,T,const N1:usize,const N2:usize> AsRawSlice<T> for Arr2View<'a,T,N1,N2> {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N1:usize,const N2:usize> SliceSize for Arr2View<'a,T,N1,N2> where T: Default + Clone + Send {
    const SIZE: usize = N1 * N2;
}
/// Implementation of an immutable iterator for fixed-length 2D arrays
#[derive(Debug,Eq,PartialEq)]
pub struct Arr2Iter<'a,T,const N:usize>(&'a [T]);
impl<'a,T,const N:usize> Arr2Iter<'a,T,N> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        N
    }
}
impl<'a,T,const N:usize> Iterator for Arr2Iter<'a,T,N> {
    type Item = ArrView<'a,T,N>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(ArrView {
                arr:l
            })
        }
    }
}
impl<'a,T,const N:usize> AsRawSlice<T> for Arr2Iter<'a,T,N> {
    fn as_raw_slice(&self) -> &[T] {
        &self.0
    }
}
/// Implementation of a mutable view of a fixed-length 2D array
#[derive(Debug,Eq,PartialEq)]
pub struct Arr2ViewMut<'a,T,const N1:usize,const N2:usize> {
    arr: &'a mut [T]
}
impl<'a,T,const N1:usize,const N2:usize> Arr2ViewMut<'a,T,N1,N2> {
    pub fn iter_mut(&'a mut self) -> Arr2IterMut<'a,T,N2> {
        Arr2IterMut(&mut self.arr)
    }
}
impl<'a,T,const N1:usize,const N2:usize> SliceSize for Arr2ViewMut<'a,T,N1,N2> where T: Default + Clone + Send {
    const SIZE: usize = N1 * N2;
}
impl<'a,T,const N1:usize,const N2:usize> AsRawSlice<T> for Arr2ViewMut<'a,T,N1,N2> where T: Default + Clone {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N1:usize,const N2:usize> AsRawMutSlice<'a,T> for Arr2ViewMut<'a,T,N1,N2> where T: Default + Clone + Send {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T] {
        &mut self.arr
    }
}
/// Implementation of an mutable iterator for fixed-length 2D arrays
#[derive(Debug,Eq,PartialEq)]
pub struct Arr2IterMut<'a,T,const N:usize>(&'a mut [T]);

impl<'a,T,const N:usize> Arr2IterMut<'a,T,N> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        N
    }
}
impl<'a,T,const N:usize> Iterator for Arr2IterMut<'a,T,N> {
    type Item = ArrViewMut<'a,T,N>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.0 = r;

            Some(ArrViewMut {
                arr:l
            })
        }
    }
}
/// Implementation of a immutable view of a fixed-length 3D array
#[derive(Debug,Eq,PartialEq)]
pub struct Arr3View<'a,T,const N1:usize,const N2:usize,const N3:usize> {
    arr: &'a [T]
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Arr3View<'a,T,N1,N2,N3> {
    pub fn iter(&'a self) -> Arr3Iter<'a,T,N1,N2> {
        Arr3Iter(&self.arr)
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> AsRawSlice<T> for Arr3View<'a,T,N1,N2,N3> {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> SliceSize for Arr3View<'a,T,N1,N2,N3> where T: Default + Clone + Send {
    const SIZE: usize = N1 * N2 * N3;
}
/// Implementation of an immutable iterator for fixed-length 3D arrays
#[derive(Debug,Eq,PartialEq)]
pub struct Arr3Iter<'a,T,const N1:usize,const N2:usize>(&'a [T]);

impl<'a,T,const N1:usize,const N2:usize> Arr3Iter<'a,T,N1,N2> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        N1 * N2
    }
}
impl<'a,T,const N1:usize,const N2:usize> Iterator for Arr3Iter<'a,T,N1,N2> {
    type Item = Arr2View<'a,T,N1,N2>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(Arr2View {
                arr: l
            })
        }
    }
}
/// Implementation of a mutable view of a fixed-length 3D array
#[derive(Debug,Eq,PartialEq)]
pub struct Arr3ViewMut<'a,T,const N1:usize,const N2:usize,const N3:usize> {
    arr:&'a mut [T]
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Arr3ViewMut<'a,T,N1,N2,N3> {
    pub fn iter_mut(&'a mut self) -> Arr3IterMut<'a,T,N1,N2> {
        Arr3IterMut(&mut self.arr)
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> AsRawSlice<T> for Arr3ViewMut<'a,T,N1,N2,N3> {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> AsRawMutSlice<'a,T> for Arr3ViewMut<'a,T,N1,N2,N3> where T: Default + Clone + Send {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T] {
        &mut self.arr
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> SliceSize for Arr3ViewMut<'a,T,N1,N2,N3> where T: Default + Clone + Send {
    const SIZE: usize = N1 * N2 * N3;
}
/// Implementation of an mutable iterator for fixed-length 3D arrays
#[derive(Debug,Eq,PartialEq)]
pub struct Arr3IterMut<'a,T,const N1:usize,const N2:usize>(&'a mut [T]);

impl<'a,T,const N1:usize,const N2:usize> Arr3IterMut<'a,T,N1,N2> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        N1 * N2
    }
}
impl<'a,T,const N1:usize,const N2:usize> Iterator for Arr3IterMut<'a,T,N1,N2> {
    type Item = Arr2ViewMut<'a,T,N1,N2>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.0 = r;

            Some(Arr2ViewMut {
                arr: l
            })
        }
    }
}
/// Implementation of an immutable iterator for fixed-length 4D arrays
#[derive(Debug,Eq,PartialEq)]
pub struct Arr4Iter<'a,T,const N1:usize,const N2:usize,const N3:usize>(&'a [T]);

impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Arr4Iter<'a,T,N1,N2,N3> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        N1 * N2 * N3
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Iterator for Arr4Iter<'a,T,N1,N2,N3> {
    type Item = Arr3View<'a,T,N1,N2,N3>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(Arr3View {
                arr: l
            })
        }
    }
}
/// Implementation of an mutable iterator for fixed-length 4D arrays
#[derive(Debug,Eq,PartialEq)]
pub struct Arr4IterMut<'a,T,const N1:usize,const N2:usize, const N3:usize>(&'a mut [T]);

impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Arr4IterMut<'a,T,N1,N2,N3> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        N1 * N2 * N3
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Iterator for Arr4IterMut<'a,T,N1,N2,N3> {
    type Item = Arr3ViewMut<'a,T,N1,N2,N3>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.0 = r;

            Some(Arr3ViewMut {
                arr: l
            })
        }
    }
}
/// Array difference information
#[derive(Debug,Clone)]
pub struct DiffArr<T,const N:usize> where T: Debug {
    items:Vec<(usize,T)>
}

impl<T,const N:usize> DiffArr<T,N> where T: Debug {
    /// Create an instance of DiffArr
    pub fn new() -> DiffArr<T,N> {
        DiffArr {
            items:Vec::new()
        }
    }

    /// Adding Elements
    /// # Arguments
    /// * `i` - index
    /// * `v` - value
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`IndexOutBoundError`]
    pub fn push(&mut self,i:usize,v:T) -> Result<(),IndexOutBoundError> {
        if i >= N {
            return Err(IndexOutBoundError::new(N,i));
        }

        self.items.push((i,v));

        Ok(())
    }

    /// Obtaining a immutable iterator
    pub fn iter<'a>(&'a self) -> impl Iterator<Item=&(usize,T)> {
        self.items.iter()
    }
}
impl<T,const N:usize> Mul<T> for DiffArr<T,N>
    where T: Mul<T> + Mul<Output=T> + Clone + Copy + Debug {

    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut r = self;

        for it in r.items.iter_mut() {
            it.1 = it.1 * rhs
        }

        r
    }
}
impl<T,const N:usize> Div<T> for DiffArr<T,N>
    where T: Div<T> + Div<Output=T> + Clone + Copy + Debug {

    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut r = self;

        for it in r.items.iter_mut() {
            it.1 = it.1 / rhs
        }

        r
    }
}
/// Converter via for zero-cost conversion of SerializedVec<U,T> to SerializedVec<U,R>
pub struct SerializedVecConverter<U,T>
    where U: Default + Clone + Copy + Send,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> {
    arr:Box<[U]>,
    len:usize,
    u:PhantomData<U>,
    t:PhantomData<T>
}
/// Implementation of fixed-length arrays whose size is not specified by a type parameter
#[derive(Debug,Eq,PartialEq,Clone)]
pub struct SerializedVec<U,T> {
    arr:Box<[U]>,
    len:usize,
    u:PhantomData<U>,
    t:PhantomData<T>
}
impl<U,T> SerializedVec<U,T> where U: Default + Clone + Copy + Send {
    /// get the number of element
    pub fn len(&self) -> usize {
        self.len
    }
}
impl<U,T> SerializedVec<U,T>
    where U: Default + Clone + Copy + Send,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> {
    /// Create a SerializedVec instance of the specified size
    /// # Arguments
    /// * `size`- Size to be secured
    pub fn with_size(size:usize) -> SerializedVec<U,T> {
        let mut arr = Vec::with_capacity(T::slice_size() * size);

        arr.resize_with(T::slice_size() * size,Default::default);

        SerializedVec {
            arr:arr.into_boxed_slice(),
            len:size,
            u:PhantomData::<U>,
            t:PhantomData::<T>
        }
    }

    /// Obtaining a immutable iterator
    pub fn iter(&self) -> SerializedVecIter<U,T> {
        SerializedVecIter {
            arr:&*self.arr,
            u:PhantomData::<U>,
            t:PhantomData::<T>,
        }
    }

    /// Obtaining a mutable iterator
    pub fn iter_mut(&mut self) -> SerializedVecIterMut<U,T> {
        SerializedVecIterMut {
            arr:&mut self.arr,
            u:PhantomData::<U>,
            t:PhantomData::<T>,
        }
    }

    /// Conversion to converter to SerializedVec with different internal types
    pub fn into_converter(self) -> SerializedVecConverter<U,T> {
        SerializedVecConverter {
            arr:self.arr,
            len:self.len,
            u:PhantomData::<U>,
            t:PhantomData::<T>,
        }
    }
}
impl<U,T> From<Vec<T>> for SerializedVec<U,T>
    where U: Default + Clone + Copy + Send,
          for<'a> T: SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U> {
    fn from(items: Vec<T>) -> Self {
        let len = items.len();

        let mut buffer = Vec::with_capacity(len * T::slice_size());

        for item in items.into_iter() {
            buffer.extend_from_slice(item.as_raw_slice());
        }

        SerializedVec {
            arr:buffer.into_boxed_slice(),
            len:len,
            u:PhantomData::<U>,
            t:PhantomData::<T>
        }
    }
}
impl<'data,U,const N:usize> From<Vec<ArrView<'data,U,N>>> for SerializedVec<U,Arr<U,N>> where U: Default + Clone + Copy + Send {
    fn from(items: Vec<ArrView<'data,U, N>>) -> Self {
        let len = items.len();

        let mut buffer = Vec::with_capacity(len * N);

        for item in items.into_iter() {
            buffer.extend_from_slice(&item);
        }

        SerializedVec {
            arr:buffer.into_boxed_slice(),
            len:len,
            u:PhantomData::<U>,
            t:PhantomData::<Arr<U,N>>
        }
    }
}
impl<U,const N:usize> TryFrom<Vec<U>> for SerializedVec<U,Arr<U,N>> where U: Default + Clone + Copy + Send {
    type Error = SizeMismatchError;

    fn try_from(items: Vec<U>) -> Result<Self,SizeMismatchError> {
        if items.len() % N != 0 {
            Err(SizeMismatchError(items.len(),N))
        } else {
            let len = items.len() / N;

            Ok(SerializedVec {
                arr: items.into_boxed_slice(),
                len: len,
                u:PhantomData::<U>,
                t: PhantomData::<Arr<U, N>>
            })
        }
    }
}
impl<U,T,R> TryFrom<SerializedVecConverter<U,T>> for SerializedVec<U,R>
    where U: Default + Clone + Copy + Send,
          for<'a> T: SliceSize + AsRawSlice<U> + MakeView<'a,U> + MakeViewMut<'a,U>,
          for<'b> R: SliceSize + AsRawSlice<U> + MakeView<'b,U> + MakeViewMut<'b,U> {
    type Error = SizeMismatchError;

    fn try_from(s: SerializedVecConverter<U,T>) -> Result<Self,SizeMismatchError> {
        if T::slice_size() != R::slice_size() {
            Err(SizeMismatchError(T::slice_size(),R::slice_size()))
        } else {
            Ok(SerializedVec {
                arr: s.arr,
                len: s.len,
                u:PhantomData::<U>,
                t: PhantomData::<R>
            })
        }
    }
}
impl<U,const N:usize> TryFrom<Box<[U]>> for SerializedVec<U,Arr<U,N>> where U: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(arr: Box<[U]>) -> Result<Self, Self::Error> {
        if arr.len() % N != 0 {
            Err(SizeMismatchError(arr.len(),N))
        } else {
            let len = arr.len() / N;

            Ok(SerializedVec {
                arr: arr,
                len: len,
                u:PhantomData::<U>,
                t: PhantomData::<Arr<U, N>>
            })
        }
    }
}
impl<U,T> From<SerializedVec<U,T>> for Box<[U]> where U: Default + Clone + Send {
    fn from(value: SerializedVec<U,T>) -> Self {
        value.arr
    }
}
impl<'a,T,const N:usize> AsRawSlice<T> for SerializedVec<T,Arr<T,N>> where T: Default + Clone + Send {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N:usize> AsRawMutSlice<'a,T> for SerializedVec<T,Arr<T,N>> where T: Default + Clone + Send {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T] {
        &mut self.arr
    }
}
impl<'a,U,const N:usize> Add<U> for &'a SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn add(self, rhs: U) -> Self::Output {
        self.par_iter().map(|l| l + rhs).collect::<Vec<Arr<U,N>>>().into()
    }
}
impl<U,const N:usize> Add<U> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Add<U,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn add(self, rhs: U) -> Self::Output {
        &self + rhs
    }
}
impl<'a,U,const N:usize> Sub<U> for &'a SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn sub(self, rhs: U) -> Self::Output {
        self.par_iter().map(|l| l - rhs).collect::<Vec<Arr<U,N>>>().into()
    }
}
impl<U,const N:usize> Sub<U> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Sub<U,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn sub(self, rhs: U) -> Self::Output {
        &self - rhs
    }
}
impl<'a,U,const N:usize> Mul<U> for &'a SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn mul(self, rhs: U) -> Self::Output {
        self.par_iter().map(|l| l * rhs).collect::<Vec<Arr<U,N>>>().into()
    }
}
impl<U,const N:usize> Mul<U> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Mul<U,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn mul(self, rhs: U) -> Self::Output {
        &self * rhs
    }
}
impl<'a,U,const N:usize> Div<U> for &'a SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn div(self, rhs: U) -> Self::Output {
        self.par_iter().map(|l| l / rhs).collect::<Vec<Arr<U,N>>>().into()
    }
}
impl<U,const N:usize> Div<U> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Div<U,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn div(self, rhs: U) -> Self::Output {
        &self / rhs
    }
}
impl<'a,'b: 'a,U,const N:usize> Add<&'b SerializedVec<U,Arr<U,N>>> for &'a SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn add(self, rhs: &'b SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(l,r)| l + r).collect::<Vec<Arr<U,N>>>().into()
    }
}
impl<U,const N:usize> Add<&SerializedVec<U,Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Add<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn add(self, rhs: &SerializedVec<U,Arr<U,N>>) -> Self::Output {
        &self + rhs
    }
}
impl<U,const N:usize> Add<SerializedVec<U,Arr<U,N>>> for &SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Add<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn add(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self + &rhs
    }
}
impl<U,const N:usize> Add<SerializedVec<U,Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Add<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn add(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        &self + &rhs
    }
}
impl<'a,'b: 'a,U,const N:usize> Sub<&'b SerializedVec<U,Arr<U,N>>> for &'a SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn sub(self, rhs: &'b SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(l,r)| l - r).collect::<Vec<Arr<U,N>>>().into()
    }
}
impl<U,const N:usize> Sub<&SerializedVec<U,Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Sub<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn sub(self, rhs: &SerializedVec<U,Arr<U,N>>) -> Self::Output {
        &self - rhs
    }
}
impl<U,const N:usize> Sub<SerializedVec<U,Arr<U,N>>> for &SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Sub<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn sub(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self - &rhs
    }
}
impl<U,const N:usize> Sub<SerializedVec<U,Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Sub<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn sub(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        &self - &rhs
    }
}
impl<'a,'b: 'a,U,const N:usize> Mul<&'b SerializedVec<U,Arr<U,N>>> for &'a SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn mul(self, rhs: &'b SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(l,r)| l * r).collect::<Vec<Arr<U,N>>>().into()
    }
}
impl<U,const N:usize> Mul<&SerializedVec<U,Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Mul<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn mul(self, rhs: &SerializedVec<U,Arr<U,N>>) -> Self::Output {
        &self * rhs
    }
}
impl<U,const N:usize> Mul<SerializedVec<U,Arr<U,N>>> for &SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Mul<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn mul(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self * &rhs
    }
}
impl<U,const N:usize> Mul<SerializedVec<U,Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Mul<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn mul(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        &self * &rhs
    }
}
impl<'a,'b: 'a,U,const N:usize> Div<&'b SerializedVec<U,Arr<U,N>>> for &'a SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn div(self, rhs: &'b SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rhs.par_iter()).map(|(l,r)| l / r).collect::<Vec<Arr<U,N>>>().into()
    }
}
impl<U,const N:usize> Div<&SerializedVec<U,Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Div<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn div(self, rhs: &SerializedVec<U,Arr<U,N>>) -> Self::Output {
        &self / rhs
    }
}
impl<U,const N:usize> Div<SerializedVec<U,Arr<U,N>>> for &SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Div<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn div(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self / &rhs
    }
}
impl<U,const N:usize> Div<SerializedVec<U,Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Div<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn div(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        &self / &rhs
    }
}
impl<'a,U,const N:usize> Neg for &'a SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Neg<Output=U> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn neg(self) -> Self::Output {
        self.par_iter().map(|v| -v).collect::<Vec<Arr<U,N>>>().into()
    }
}
impl<U,const N:usize> Neg for SerializedVec<U,Arr<U,N>>
    where U: Send + Sync + Default + Clone + Copy + 'static + Neg<Output=U>,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Neg<Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn neg(self) -> Self::Output {
        (&self).neg()
    }
}
impl<U,const N:usize> Sum for SerializedVec<U,Arr<U,N>>
    where U: Add<Output=U> + Clone + Copy + Default + Send + Sync + 'static,
          for<'a> ArrView<'a,U,N>: Add<ArrView<'a,U,N>,Output = Arr<U,N>> {
    type Output = Arr<U,N>;

    fn sum(&self) -> Self::Output {
        self.par_iter().fold(|| Arr::new(), |acc:Arr<U,N>,r| {
            ArrView::from(&acc) + r
        }).reduce(|| Arr::new(), |acc,r| &acc + &r)
    }
}
/// Trait that returns a view holding an immutable reference to the slice to be owned
pub trait MakeView<'a,T> {
    /// Returned View type
    type ViewType: Send + Sync + 'a;

    /// Create a view
    /// # Arguments
    /// * 'arr' - Slice of view references
    ///
    fn make_view(arr:&'a [T]) -> Self::ViewType;
}
/// Trait that returns a view holding an mutable reference to the slice to be owned
pub trait MakeViewMut<'a,T> {
    /// Returned View type
    type ViewType: Send + Sync + 'a;

    /// Create a view
    /// # Arguments
    /// * 'arr' - Slice of view references
    ///
    fn make_view_mut(arr:&'a mut [T]) -> Self::ViewType;
}
/// SerializedVec's Immutable Iterator
#[derive(Debug,Eq,PartialEq)]
pub struct SerializedVecIter<'a,U,T> where T: SliceSize + MakeView<'a,U> {
    arr: &'a [U],
    t:PhantomData<T>,
    u:PhantomData<U>
}

impl<'a,U,T> SerializedVecIter<'a,U,T> where T: SliceSize + MakeView<'a,U> {
    /// Number of elements encompassed by the iterator element
    #[inline]
    fn element_size(&self) -> usize {
        T::slice_size()
    }
}
impl<'a,U,T> Iterator for SerializedVecIter<'a,U,T> where T: SliceSize + MakeView<'a,U> {
    type Item = <T as MakeView<'a,U>>::ViewType;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.arr = r;

            Some(T::make_view(l))
        }
    }
}

/// SerializedVec's mutable Iterator
#[derive(Debug,Eq,PartialEq)]
pub struct SerializedVecIterMut<'a,U,T> where T: SliceSize + MakeViewMut<'a,U> {
    arr: &'a mut [U],
    t:PhantomData<T>,
    u:PhantomData<U>
}

impl<'a,U,T> SerializedVecIterMut<'a,U,T> where T: SliceSize + MakeViewMut<'a,U> {
    /// Number of elements encompassed by the iterator element
    #[inline]
    fn element_size(&self) -> usize {
        T::slice_size()
    }
}
impl<'a,U,T> Iterator for SerializedVecIterMut<'a,U,T> where T: SliceSize + MakeViewMut<'a,U> {
    type Item = <T as MakeViewMut<'a,U>>::ViewType;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.arr = r;

            Some(T::make_view_mut(l))
        }
    }
}
impl<'data,T, const N:usize> IntoParallelRefIterator<'data> for ArrView<'data,T,N>
    where T: Send + Sync + Default + Clone + 'static {
    type Iter = rayon::slice::Iter<'data,T>;
    type Item = &'data T;

    fn par_iter(&'data self) -> Self::Iter {
        <&[T]>::into_par_iter(&self.arr)
    }
}
impl<'data,T, const N:usize> IntoParallelRefIterator<'data> for &'data ArrView<'data,T,N>
    where T: Send + Sync + Default + Clone + 'static {
    type Iter = rayon::slice::Iter<'data,T>;
    type Item = &'data T;

    fn par_iter(&'data self) -> Self::Iter {
        <&[T]>::into_par_iter(&*self.arr)
    }
}
impl<'data,T, const N:usize> IntoParallelRefIterator<'data> for Arr<T,N>
    where T: Send + Sync + Default + Clone + 'static {
    type Iter = rayon::slice::Iter<'data,T>;
    type Item = &'data T;

    fn par_iter(&'data self) -> Self::Iter {
        <&[T]>::into_par_iter(&self.arr)
    }
}
impl<'data,T, const N:usize> IntoParallelRefIterator<'data> for &'data Arr<T,N>
    where T: Send + Sync + Default + Clone + 'static {
    type Iter = rayon::slice::Iter<'data,T>;
    type Item = &'data T;

    fn par_iter(&'data self) -> Self::Iter {
        <&[T]>::into_par_iter(&self.arr)
    }
}
/// ParallelIterator implementation for Arr2
#[derive(Debug)]
pub struct Arr2ParIter<'data,T,const N1:usize,const N2:usize>(&'data [T]);

/// Implementation of plumbing::Producer for Arr2
#[derive(Debug)]
pub struct Arr2IterProducer<'data,T,const N1:usize,const N2:usize>(&'data [T]);

impl<'data,T,const N1:usize, const N2:usize> Arr2IterProducer<'data,T,N1,N2> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        N2
    }
}
impl<'data,T,const N1:usize,const N2:usize> Iterator for Arr2IterProducer<'data,T,N1,N2> {
    type Item = ArrView<'data,T,N2>;

    fn next(&mut self) -> Option<ArrView<'data,T,N2>> {
        let slice = std::mem::replace(&mut self.0, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(ArrView {
                arr:l
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (N1, Some(N1))
    }
}
impl<'data,T,const N1:usize,const N2:usize> std::iter::ExactSizeIterator for Arr2IterProducer<'data,T,N1,N2> {
    fn len(&self) -> usize {
        N1
    }
}
impl<'data,T,const N1:usize,const N2:usize> std::iter::DoubleEndedIterator for Arr2IterProducer<'data,T,N1,N2> {
    fn next_back(&mut self) -> Option<ArrView<'data,T,N2>> {
        let slice = std::mem::replace(&mut self.0, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.0.len() - self.element_size());

            self.0 = l;

            Some(ArrView {
                arr:r
            })
        }
    }
}
impl<'data, T: Send + Sync + 'static,const N1:usize,const N2:usize> plumbing::Producer for Arr2IterProducer<'data,T,N1,N2> {
    type Item = ArrView<'data,T,N2>;
    type IntoIter = Self;

    fn into_iter(self) -> Self { self }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let (l,r) = self.0.split_at(mid * N2);

        (Arr2IterProducer(l),Arr2IterProducer(r))
    }
}
impl<'data, T: Send + Sync + 'static,const N1: usize, const N2: usize> ParallelIterator for Arr2ParIter<'data,T,N1,N2> {
    type Item = ArrView<'data,T,N2>;

    fn opt_len(&self) -> Option<usize> { Some(IndexedParallelIterator::len(self)) }

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: plumbing::UnindexedConsumer<Self::Item>,
    {
        self.drive(consumer)
    }
}
impl<'data, T: Send + Sync + 'static, const N1: usize, const N2: usize> IndexedParallelIterator for Arr2ParIter<'data,T,N1,N2> {
    fn len(&self) -> usize { N1 }

    fn drive<C>(self, consumer: C) -> C::Result
        where
            C: plumbing::Consumer<Self::Item>,
    {
        plumbing::bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
        where
            CB: plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(Arr2IterProducer::<T,N1,N2>(&self.0))
    }
}
impl<'data,T, const N1:usize, const N2:usize> IntoParallelRefIterator<'data> for Arr2<T,N1,N2>
    where T: Send + Sync + 'static + Default {
    type Iter = Arr2ParIter<'data,T,N1,N2>;
    type Item = ArrView<'data,T,N2>;

    fn par_iter(&'data self) -> Self::Iter {
        Arr2ParIter(&self.arr)
    }
}
impl<'data,T, const N1:usize, const N2:usize> IntoParallelRefIterator<'data> for Arr2View<'data,T,N1,N2>
    where T: Send + Sync + 'static + Default {
    type Iter = Arr2ParIter<'data,T,N1,N2>;
    type Item = ArrView<'data,T,N2>;

    fn par_iter(&'data self) -> Self::Iter {
        Arr2ParIter(&self.arr)
    }
}
/// ParallelIterator implementation for Arr3
#[derive(Debug)]
pub struct Arr3ParIter<'data,T,const N1:usize,const N2:usize,const N3:usize>(&'data [T]);

/// Implementation of plumbing::Producer for Arr3
#[derive(Debug)]
pub struct Arr3IterProducer<'data,T,const N1:usize,const N2:usize,const N3:usize>(&'data [T]);

impl<'data,T,const N1:usize, const N2:usize, const N3:usize> Arr3IterProducer<'data,T,N1,N2,N3> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        N1 * N2
    }
}
impl<'data,T,const N1:usize,const N2:usize,const N3:usize> Iterator for Arr3IterProducer<'data,T,N1,N2,N3> {
    type Item = Arr2View<'data,T,N2,N3>;

    fn next(&mut self) -> Option<Arr2View<'data,T,N2,N3>> {
        let slice = std::mem::replace(&mut self.0, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(Arr2View {
                arr: l
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (N1, Some(N1))
    }
}
impl<'data,T,const N1:usize,const N2:usize,const N3:usize> std::iter::ExactSizeIterator for Arr3IterProducer<'data,T,N1,N2,N3> {
    fn len(&self) -> usize {
        N1
    }
}
impl<'data,T,const N1:usize,const N2:usize,const N3:usize> std::iter::DoubleEndedIterator for Arr3IterProducer<'data,T,N1,N2,N3> {
    fn next_back(&mut self) -> Option<Arr2View<'data,T,N2,N3>> {
        let slice = std::mem::replace(&mut self.0, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.0.len() - self.element_size());

            self.0 = l;

            Some(Arr2View {
                arr:r
            })
        }
    }
}
impl<'data, T: Send + Sync + 'static,const N1:usize,const N2:usize,const N3:usize> plumbing::Producer for Arr3IterProducer<'data,T,N1,N2,N3> {
    type Item = Arr2View<'data,T,N2,N3>;
    type IntoIter = Self;

    fn into_iter(self) -> Self { self }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let (l,r) = self.0.split_at(mid * N2 * N3);

        (Arr3IterProducer(l),Arr3IterProducer(r))
    }
}
impl<'data, T: Send + Sync + 'static,const N1: usize, const N2: usize, const N3:usize> ParallelIterator for Arr3ParIter<'data,T,N1,N2,N3> {
    type Item = Arr2View<'data,T,N2,N3>;

    fn opt_len(&self) -> Option<usize> { Some(IndexedParallelIterator::len(self)) }

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: plumbing::UnindexedConsumer<Self::Item>,
    {
        self.drive(consumer)
    }
}
impl<'data, T: Send + Sync + 'static, const N1: usize, const N2: usize, const N3:usize> IndexedParallelIterator for Arr3ParIter<'data,T,N1,N2,N3> {
    fn len(&self) -> usize { N1 }

    fn drive<C>(self, consumer: C) -> C::Result
        where
            C: plumbing::Consumer<Self::Item>,
    {
        plumbing::bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
        where
            CB: plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(Arr3IterProducer::<T,N1,N2,N3>(&self.0))
    }
}
impl<'data,T, const N1:usize, const N2:usize, const N3:usize> IntoParallelRefIterator<'data> for Arr3<T,N1,N2,N3>
    where T: Send + Sync + 'static + Default {
    type Iter = Arr3ParIter<'data,T,N1,N2,N3>;
    type Item = Arr2View<'data,T,N2,N3>;

    fn par_iter(&'data self) -> Self::Iter {
        Arr3ParIter(&self.arr)
    }
}
impl<'data,T, const N1:usize, const N2:usize, const N3:usize> IntoParallelRefIterator<'data> for Arr3View<'data,T,N1,N2,N3>
    where T: Send + Sync + 'static + Default {
    type Iter = Arr3ParIter<'data,T,N1,N2,N3>;
    type Item = Arr2View<'data,T,N2,N3>;

    fn par_iter(&'data self) -> Self::Iter {
        Arr3ParIter(&self.arr)
    }
}
/// ParallelIterator implementation for Arr4
#[derive(Debug)]
pub struct Arr4ParIter<'data,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize>(&'data [T]);

/// Implementation of plumbing::Producer for Arr4
#[derive(Debug)]
pub struct Arr4IterProducer<'data,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize>(&'data [T]);

impl<'data,T,const N1:usize, const N2:usize, const N3:usize, const N4:usize> Arr4IterProducer<'data,T,N1,N2,N3,N4> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        N1 * N2 * N3
    }
}
impl<'data,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> Iterator for Arr4IterProducer<'data,T,N1,N2,N3,N4> {
    type Item = Arr3View<'data,T,N2,N3,N4>;

    fn next(&mut self) -> Option<Arr3View<'data,T,N2,N3,N4>> {
        let slice = std::mem::replace(&mut self.0, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(Arr3View {
                arr: l
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (N1, Some(N1))
    }
}
impl<'data,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> std::iter::ExactSizeIterator for Arr4IterProducer<'data,T,N1,N2,N3,N4> {
    fn len(&self) -> usize {
        N1
    }
}
impl<'data,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> std::iter::DoubleEndedIterator for Arr4IterProducer<'data,T,N1,N2,N3,N4> {
    fn next_back(&mut self) -> Option<Arr3View<'data,T,N2,N3,N4>> {
        let slice = std::mem::replace(&mut self.0, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.0.len() - self.element_size());

            self.0 = l;

            Some(Arr3View {
                arr:r
            })
        }
    }
}
impl<'data, T: Send + Sync + 'static,const N1:usize,const N2:usize,const N3:usize,const N4:usize> plumbing::Producer
    for Arr4IterProducer<'data,T,N1,N2,N3,N4> {
    type Item = Arr3View<'data,T,N2,N3,N4>;
    type IntoIter = Self;

    fn into_iter(self) -> Self { self }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let (l,r) = self.0.split_at(mid * N2 * N3);

        (Arr4IterProducer(l),Arr4IterProducer(r))
    }
}
impl<'data, T: Send + Sync + 'static,const N1: usize, const N2: usize, const N3:usize,const N4:usize> ParallelIterator
    for Arr4ParIter<'data,T,N1,N2,N3,N4> {
    type Item = Arr3View<'data,T,N2,N3,N4>;

    fn opt_len(&self) -> Option<usize> { Some(IndexedParallelIterator::len(self)) }

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: plumbing::UnindexedConsumer<Self::Item>,
    {
        self.drive(consumer)
    }
}
impl<'data, T: Send + Sync + 'static, const N1: usize, const N2: usize, const N3:usize, const N4:usize> IndexedParallelIterator
    for Arr4ParIter<'data,T,N1,N2,N3,N4> {
    fn len(&self) -> usize { N1 }

    fn drive<C>(self, consumer: C) -> C::Result
        where
            C: plumbing::Consumer<Self::Item>,
    {
        plumbing::bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
        where
            CB: plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(Arr4IterProducer::<T,N1,N2,N3,N4>(&self.0))
    }
}
impl<'data,T, const N1:usize, const N2:usize, const N3:usize, const N4:usize> IntoParallelRefIterator<'data> for Arr4<T,N1,N2,N3,N4>
    where T: Send + Sync + 'static + Default {
    type Iter = Arr4ParIter<'data,T,N1,N2,N3,N4>;
    type Item = Arr3View<'data,T,N2,N3,N4>;

    fn par_iter(&'data self) -> Self::Iter {
        Arr4ParIter(&self.arr)
    }
}
/// Implementation of ParallelIterator for SerializedVec
#[derive(Debug)]
pub struct SerializedVecParIter<'data,C,T> {
    arr: &'data [T],
    t:PhantomData<C>,
    len: usize
}

/// Implementation of plumbing::Producer for SerializedVec
#[derive(Debug)]
pub struct SerializedVecIterProducer<'data,C,T> {
    arr: &'data [T],
    t:PhantomData<C>,
    len: usize
}

impl<'data,C,T> SerializedVecIterProducer<'data,C,T>
    where T: Default + Clone + Send,
          C: SliceSize + MakeView<'data,T> + Send {
    #[inline]
    /// Number of elements encompassed by the iterator element
    fn element_size(&self) -> usize {
        C::slice_size()
    }
}
impl<'data,C,T> Iterator for SerializedVecIterProducer<'data,C,T>
    where T: Default + Clone + Send,
          C: SliceSize + MakeView<'data,T> + Send {
    type Item = <C as MakeView<'data,T>>::ViewType;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.arr = r;

            Some(C::make_view(l))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        ({self.len}, Some(self.len))
    }
}
impl<'data,C,T> std::iter::ExactSizeIterator for SerializedVecIterProducer<'data,C,T>
    where T: Default + Clone + Send,
          C: SliceSize + MakeView<'data,T> + Send {
    fn len(&self) -> usize {
        self.len
    }
}
impl<'data,C,T> std::iter::DoubleEndedIterator for SerializedVecIterProducer<'data,C,T>
    where T: Default + Clone + Send,
          C: SliceSize + MakeView<'data,T> + Send {
    fn next_back(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.arr.len() - self.element_size());

            self.arr = l;

            Some(C::make_view(r))
        }
    }
}
impl<'data, C, T> plumbing::Producer for SerializedVecIterProducer<'data,C,T>
    where T: Default + Clone + Send + Sync + 'static,
          C: SliceSize + MakeView<'data,T> + Send {
    type Item = <C as MakeView<'data,T>>::ViewType;
    type IntoIter = Self;

    fn into_iter(self) -> Self { self }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let (l,r) = self.arr.split_at(mid * self.element_size());

        (SerializedVecIterProducer {
            arr: l,
            t:PhantomData::<C>,
            len:self.len
        },SerializedVecIterProducer {
            arr: r,
            t:PhantomData::<C>,
            len:self.len
        })
    }
}
impl<'data, C, T> ParallelIterator for SerializedVecParIter<'data,C,T>
    where T: Default + Clone + Send + Sync + 'static,
          C: SliceSize + MakeView<'data,T> + Send {
    type Item = <C as MakeView<'data,T>>::ViewType;

    fn opt_len(&self) -> Option<usize> { Some(IndexedParallelIterator::len(self)) }

    fn drive_unindexed<CS>(self, consumer: CS) -> CS::Result
        where
            CS: plumbing::UnindexedConsumer<Self::Item>,
    {
        self.drive(consumer)
    }
}
impl<'data, C, T> IndexedParallelIterator for SerializedVecParIter<'data,C,T>
    where T: Default + Clone + Send + Sync + 'static,
          C: SliceSize + MakeView<'data,T> + Send {
    fn len(&self) -> usize { self.len }

    fn drive<CS>(self, consumer: CS) -> CS::Result
        where
            CS: plumbing::Consumer<Self::Item>,
    {
        plumbing::bridge(self, consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
        where
            CB: plumbing::ProducerCallback<Self::Item>,
    {
        callback.callback(SerializedVecIterProducer::<'data,C,T> {
            arr:&self.arr,
            t:PhantomData::<C>,
            len: self.len
        })
    }
}
impl<'data,C,T> IntoParallelRefIterator<'data> for SerializedVec<T,C>
    where T: Default + Clone + Copy + Send + Sync + 'static,
          C: SliceSize + MakeView<'data,T> + Send {
    type Iter = SerializedVecParIter<'data,C,T>;
    type Item = <C as MakeView<'data,T>>::ViewType;

    fn par_iter(&'data self) -> Self::Iter {
        SerializedVecParIter {
            arr: &self.arr,
            t:PhantomData::<C>,
            len: self.len
        }
    }
}
impl<'data,C,T> IntoParallelRefIterator<'data> for &'data SerializedVec<T,C>
    where T: Default + Clone + Copy + Send + Sync + 'static,
          C: SliceSize + MakeView<'data,T> + Send {
    type Iter = SerializedVecParIter<'data,C,T>;
    type Item = <C as MakeView<'data,T>>::ViewType;

    fn par_iter(&'data self) -> Self::Iter {
        SerializedVecParIter {
            arr: &self.arr,
            t:PhantomData::<C>,
            len: self.len
        }
    }
}
