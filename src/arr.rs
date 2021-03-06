use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Div, Index, IndexMut, Mul};
use std;
use rayon::iter::plumbing;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use crate::error::{IndexOutBoundError, SizeMismatchError};
use crate::mem::{AsRawMutSlice, AsRawSlice};

#[derive(Debug,Eq,PartialEq)]
pub struct Arr<T,const N:usize> where T: Default + Clone + Send {
    arr:Box<[T]>
}
impl<T,const N:usize> Arr<T,N> where T: Default + Clone + Send {
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
        Arr{
            arr:self.arr.clone()
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
impl<T,const N:usize> Mul<T> for Arr<T,N> where T: Mul<T> + Mul<Output=T> + Clone + Copy + Default + Send {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut r = self;

        for it in r.iter_mut() {
            *it = *it * rhs;
        }
        r
    }
}
impl<T,const N:usize> Div<T> for Arr<T,N> where T: Div<T> + Div<Output=T> + Clone + Copy + Default + Send {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut r = self;

        for it in r.iter_mut() {
            *it = *it / rhs;
        }
        r
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
#[derive(Debug,Eq,PartialEq)]
pub struct Arr2<T,const N1:usize, const N2:usize> where T: Default {
    arr:Box<[T]>
}
impl<T,const N1:usize, const N2:usize> Arr2<T,N1,N2> where T: Default {
    pub fn new() -> Arr2<T,N1,N2> {
        let mut arr = Vec::with_capacity(N1 * N2);
        arr.resize_with(N1*N2,Default::default);

        Arr2 {
            arr:arr.into_boxed_slice()
        }
    }

    pub fn iter<'a>(&'a self) -> Arr2Iter<'a,T,N2> {
        Arr2Iter(&*self.arr)
    }

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
    pub fn new() -> Arr3<T,N1,N2,N3> {
        let mut arr = Vec::with_capacity(N1 * N2 * N3);
        arr.resize_with(N1*N2*N3,Default::default);

        Arr3 {
            arr:arr.into_boxed_slice()
        }
    }

    pub fn iter<'a>(&'a self) -> Arr3Iter<'a,T,N2,N3> {
        Arr3Iter(&*self.arr)
    }

    pub fn iter_mut<'a>(&'a mut self) -> Arr3IterMut<'a,T,N2,N3> {
        Arr3IterMut(&mut *self.arr)
    }
}
impl<T,const N1:usize, const N2:usize, const N3:usize> Index<(usize,usize,usize)> for Arr3<T,N1,N2,N3> where T: Default {
    type Output = T;

    fn index(&self, (y,x,z): (usize, usize, usize)) -> &Self::Output {
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
    pub fn new() -> Arr3<T,N1,N2,N3> {
        let mut arr = Vec::with_capacity(N1 * N2 * N3 * N4);
        arr.resize_with(N1*N2*N3*N4,Default::default);

        Arr3 {
            arr:arr.into_boxed_slice()
        }
    }

    pub fn iter<'a>(&'a self) -> Arr4Iter<'a,T,N2,N3,N4> {
        Arr4Iter(&*self.arr)
    }

    pub fn iter_mut<'a>(&'a mut self) -> Arr4IterMut<'a,T,N2,N3,N4> {
        Arr4IterMut(&mut *self.arr)
    }
}
impl<T,const N1:usize, const N2:usize, const N3:usize, const N4:usize> Index<(usize,usize,usize,usize)> for Arr4<T,N1,N2,N3,N4>
    where T: Default {
    type Output = T;

    fn index(&self, (i,y,x,z): (usize, usize, usize, usize)) -> &Self::Output {
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
    fn index_mut(&mut self, (i,y,x,z): (usize, usize, usize, usize)) -> &mut Self::Output {
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
#[derive(Debug,Eq,PartialEq)]
pub struct ArrView<'a,T,const N:usize> {
    arr:&'a [T]
}
impl<'a,T,const N:usize> Clone for ArrView<'a,T,N> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        ArrView {
            arr:self.arr.clone()
        }
    }
}
impl<'a,T,const N:usize> Deref for ArrView<'a,T,N> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.arr
    }
}
#[derive(Debug,Eq,PartialEq)]
pub struct ArrViewMut<'a,T,const N:usize> {
    arr:&'a mut [T]
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
impl<'a,T,const N:usize> AsRawSlice<T> for ArrView<'a,T,N> where T: Default + Clone {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
#[derive(Debug,Eq,PartialEq)]
pub struct Arr2Iter<'a,T,const N:usize>(&'a [T]);

impl<'a,T,const N:usize> Arr2Iter<'a,T,N> {
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
#[derive(Debug,Eq,PartialEq)]
pub struct Arr2IterMut<'a,T,const N:usize>(&'a mut [T]);

impl<'a,T,const N:usize> Arr2IterMut<'a,T,N> {
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
#[derive(Debug,Eq,PartialEq)]
pub struct Arr3Iter<'a,T,const N1:usize,const N2:usize>(&'a [T]);

impl<'a,T,const N1:usize,const N2:usize> Arr3Iter<'a,T,N1,N2> {
    const fn element_size(&self) -> usize {
        N1 * N2
    }
}
impl<'a,T,const N1:usize,const N2:usize> Iterator for Arr3Iter<'a,T,N1,N2> {
    type Item = Arr2Iter<'a,T,N2>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(Arr2Iter(l))
        }
    }
}
#[derive(Debug,Eq,PartialEq)]
pub struct Arr3IterMut<'a,T,const N1:usize,const N2:usize>(&'a mut [T]);

impl<'a,T,const N1:usize,const N2:usize> Arr3IterMut<'a,T,N1,N2> {
    const fn element_size(&self) -> usize {
        N1 * N2
    }
}
impl<'a,T,const N1:usize,const N2:usize> Iterator for Arr3IterMut<'a,T,N1,N2> {
    type Item = Arr2IterMut<'a,T,N2>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.0 = r;

            Some(Arr2IterMut(l))
        }
    }
}
#[derive(Debug,Eq,PartialEq)]
pub struct Arr4Iter<'a,T,const N1:usize,const N2:usize,const N3:usize>(&'a [T]);

impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Arr4Iter<'a,T,N1,N2,N3> {
    const fn element_size(&self) -> usize {
        N1 * N2 * N3
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Iterator for Arr4Iter<'a,T,N1,N2,N3> {
    type Item = Arr3Iter<'a,T,N2,N3>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(Arr3Iter(l))
        }
    }
}
#[derive(Debug,Eq,PartialEq)]
pub struct Arr4IterMut<'a,T,const N1:usize,const N2:usize, const N3:usize>(&'a mut [T]);

impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Arr4IterMut<'a,T,N1,N2,N3> {
    const fn element_size(&self) -> usize {
        N1 * N2 * N3
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Iterator for Arr4IterMut<'a,T,N1,N2,N3> {
    type Item = Arr3IterMut<'a,T,N2,N3>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.0 = r;

            Some(Arr3IterMut(l))
        }
    }
}
#[derive(Debug,Clone)]
pub struct DiffArr<T,const N:usize> where T: Debug {
    items:Vec<(usize,T)>
}

impl<T,const N:usize> DiffArr<T,N> where T: Debug {
    pub fn new() -> DiffArr<T,N> {
        DiffArr {
            items:Vec::new()
        }
    }

    pub fn push(&mut self,i:usize,v:T) -> Result<(),IndexOutBoundError> {
        if i >= N {
            return Err(IndexOutBoundError::new(N,i));
        }

        self.items.push((i,v));

        Ok(())
    }

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
#[derive(Debug,Eq,PartialEq)]
pub struct VecArr<U,T> {
    arr:Box<[U]>,
    len:usize,
    t:PhantomData<T>
}
impl<U,const N:usize> VecArr<U,Arr<U,N>> where U: Default + Clone + Copy + Send {
    pub fn with_size(size:usize) -> VecArr<U,Arr<U,N>> {
        let mut arr = Vec::with_capacity(N * size);

        arr.resize_with(N * size,Default::default);

        VecArr {
            arr:arr.into_boxed_slice(),
            len:size,
            t:PhantomData::<Arr<U,N>>
        }
    }

    pub fn iter(&self) -> VecArrIter<U,N> {
        VecArrIter(&*self.arr)
    }

    pub fn iter_mut(&mut self) -> VecArrIterMut<U,N> {
        VecArrIterMut(&mut *self.arr)
    }

    pub fn len(&self) -> usize {
        self.len
    }
}
impl<U,const N:usize> From<Vec<Arr<U,N>>> for VecArr<U,Arr<U,N>> where U: Default + Clone + Copy + Send {
    fn from(items: Vec<Arr<U, N>>) -> Self {
        let len = items.len();

        let mut buffer = Vec::with_capacity(len * N);

        for item in items.into_iter() {
            buffer.extend_from_slice(&item);
        }

        VecArr {
            arr:buffer.into_boxed_slice(),
            len:len,
            t:PhantomData::<Arr<U,N>>
        }
    }
}
impl<'a,T,const N:usize> AsRawSlice<T> for VecArr<T,Arr<T,N>> where T: Default + Clone + Send {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const N:usize> AsRawMutSlice<'a,T> for VecArr<T,Arr<T,N>> where T: Default + Clone + Send {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T] {
        &mut self.arr
    }
}
#[derive(Debug,Eq,PartialEq)]
pub struct VecArrIter<'a,T,const N:usize>(&'a [T]);

impl<'a,T,const N:usize> VecArrIter<'a,T,N> {
    const fn element_size(&self) -> usize {
        N
    }
}
impl<'a,T,const N:usize> Iterator for VecArrIter<'a,T,N> {
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
#[derive(Debug,Eq,PartialEq)]
pub struct VecArrIterMut<'a,T,const N:usize>(&'a mut [T]);

impl<'a,T,const N:usize> VecArrIterMut<'a,T,N> {
    const fn element_size(&self) -> usize {
        N
    }
}
impl<'a,T,const N:usize> Iterator for VecArrIterMut<'a,T,N> {
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
impl<'data,T, const N:usize> IntoParallelRefIterator<'data> for ArrView<'data,T,N>
    where T: Send + Sync + 'static + Default + Clone {
    type Iter = rayon::slice::Iter<'data,T>;
    type Item = &'data T;

    fn par_iter(&'data self) -> Self::Iter {
        <&[T]>::into_par_iter(&self.arr)
    }
}
impl<'data,T, const N:usize> IntoParallelRefIterator<'data> for Arr<T,N>
    where T: Send + Sync + 'static + Default + Clone {
    type Iter = rayon::slice::Iter<'data,T>;
    type Item = &'data T;

    fn par_iter(&'data self) -> Self::Iter {
        <&[T]>::into_par_iter(&self.arr)
    }
}
#[derive(Debug)]
pub struct Arr2ParIter<'data,T,const N1:usize,const N2:usize>(&'data [T]);

#[derive(Debug)]
pub struct Arr2IterProducer<'data,T,const N1:usize,const N:usize>(&'data [T]);

impl<'data,T,const N1:usize, const N2:usize> Arr2IterProducer<'data,T,N1,N2> {
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
        ({N1*N2-1}, Some(N1*N2-1))
    }
}
impl<'data,T,const N1:usize,const N2:usize> std::iter::ExactSizeIterator for Arr2IterProducer<'data,T,N1,N2> {
    fn len(&self) -> usize {
        N1 * N2
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
#[derive(Debug)]
pub struct VecArrParIter<'data,C,T> {
    arr: &'data [T],
    t:PhantomData<C>,
    len: usize
}

#[derive(Debug)]
pub struct VecArrIterProducer<'data,C,T> {
    arr: &'data [T],
    t:PhantomData<C>,
    len: usize
}

impl<'data,T,const N:usize> VecArrIterProducer<'data,Arr<T,N>,T> where T: Default + Clone + Send {
    fn element_size(&self) -> usize {
        N
    }
}
impl<'data,T,const N:usize> Iterator for VecArrIterProducer<'data,Arr<T,N>,T> where T: Default + Clone + Send {
    type Item = ArrView<'data,T,N>;

    fn next(&mut self) -> Option<ArrView<'data,T,N>> {
        let slice = std::mem::replace(&mut self.arr, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.arr = r;

            Some(ArrView {
                arr:l
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        ({self.len}, Some(self.len))
    }
}
impl<'data,T,const N:usize> std::iter::ExactSizeIterator for VecArrIterProducer<'data,Arr<T,N>,T> where T: Default + Clone + Send {
    fn len(&self) -> usize {
        self.len
    }
}
impl<'data,T,const N:usize> std::iter::DoubleEndedIterator for VecArrIterProducer<'data,Arr<T,N>,T> where T: Default + Clone + Send {
    fn next_back(&mut self) -> Option<ArrView<'data,T,N>> {
        let slice = std::mem::replace(&mut self.arr, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.arr.len() - self.element_size());

            self.arr = l;

            Some(ArrView {
                arr:r
            })
        }
    }
}
impl<'data, T: Send + Sync + 'static,const N:usize> plumbing::Producer for VecArrIterProducer<'data,Arr<T,N>,T> where T: Default + Clone + Send {
    type Item = ArrView<'data,T,N>;
    type IntoIter = Self;

    fn into_iter(self) -> Self { self }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let (l,r) = self.arr.split_at(mid * N);

        (VecArrIterProducer {
            arr: l,
            t:PhantomData::<Arr<T,N>>,
            len:self.len
        },VecArrIterProducer {
            arr: r,
            t:PhantomData::<Arr<T,N>>,
            len:self.len
        })
    }
}
impl<'data, T: Send + Sync + 'static,const N:usize> ParallelIterator for VecArrParIter<'data,Arr<T,N>,T> where T: Default + Clone + Send {
    type Item = ArrView<'data,T,N>;

    fn opt_len(&self) -> Option<usize> { Some(IndexedParallelIterator::len(self)) }

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: plumbing::UnindexedConsumer<Self::Item>,
    {
        self.drive(consumer)
    }
}
impl<'data, T: Send + Sync + 'static, const N:usize> IndexedParallelIterator for VecArrParIter<'data,Arr<T,N>,T> where T: Default + Clone + Send {
    fn len(&self) -> usize { self.len }

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
        callback.callback(VecArrIterProducer::<'data,Arr<T,N>,T> {
            arr:&self.arr,
            t:PhantomData::<Arr<T,N>>,
            len: self.len
        })
    }
}
impl<'data,T, const N:usize> IntoParallelRefIterator<'data> for VecArr<T,Arr<T,N>>
    where T: Default + Clone + Send + Sync + 'static {
    type Iter = VecArrParIter<'data,Arr<T,N>,T>;
    type Item = ArrView<'data,T,N>;

    fn par_iter(&'data self) -> Self::Iter {
        VecArrParIter {
            arr: &self.arr,
            t:PhantomData::<Arr<T,N>>,
            len: self.len
        }
    }
}
