use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};
use rayon::iter::plumbing;
use rayon::prelude::{ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use crate::arr::{Arr, ArrView, ArrViewMut, VecArr};
use crate::mem::AsRawSlice;

#[derive(Debug,Clone)]
pub struct Broadcast<T>(pub T) where T: Clone;
impl<'a,U,const N:usize> Add<&'a VecArr<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn add(self, rhs: &'a VecArr<U,Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l + r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the add of Broadcast and VecArr.").into()
    }
}
impl<U,const N:usize> Add<VecArr<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output = U> + 'static,
          for<'a> Broadcast<Arr<U,N>>: Add<&'a VecArr<U,Arr<U,N>>,Output = VecArr<U,Arr<U,N>>>, {
    type Output = VecArr<U,Arr<U,N>>;

    fn add(self, rhs: VecArr<U,Arr<U,N>>) -> Self::Output {
        self + &rhs
    }
}
impl<'a,U,const N:usize> Add<Broadcast<Arr<U,N>>> for &'a VecArr<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn add(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l + r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the add of Broadcast and VecArr.").into()
    }
}
impl<U,const N:usize> Add<Broadcast<Arr<U,N>>> for VecArr<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output = U> + 'static,
          for<'a> &'a VecArr<U,Arr<U,N>>: Add<Broadcast<Arr<U,N>>,Output = VecArr<U,Arr<U,N>>> {
    type Output = VecArr<U,Arr<U,N>>;

    fn add(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        &self + rhs
    }
}
impl<'a,U,const N:usize> Sub<&'a VecArr<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Sub<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn sub(self, rhs: &'a VecArr<U,Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l - r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the sub of Broadcast and VecArr.").into()
    }
}
impl<U,const N:usize> Sub<VecArr<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Sub<Output = U> + 'static,
          for<'a> Broadcast<Arr<U,N>>: Sub<&'a VecArr<U,Arr<U,N>>,Output = VecArr<U,Arr<U,N>>>, {
    type Output = VecArr<U,Arr<U,N>>;

    fn sub(self, rhs: VecArr<U,Arr<U,N>>) -> Self::Output {
        self - &rhs
    }
}
impl<'a,U,const N:usize> Sub<Broadcast<Arr<U,N>>> for &'a VecArr<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Sub<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn sub(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l - r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the sub of Broadcast and VecArr.").into()
    }
}
impl<U,const N:usize> Sub<Broadcast<Arr<U,N>>> for VecArr<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Sub<Output = U> + 'static,
          for<'a> &'a VecArr<U,Arr<U,N>>: Sub<Broadcast<Arr<U,N>>,Output = VecArr<U,Arr<U,N>>> {
    type Output = VecArr<U,Arr<U,N>>;

    fn sub(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        &self - rhs
    }
}
impl<'a,U,const N:usize> Mul<&'a VecArr<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Mul<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn mul(self, rhs: &'a VecArr<U,Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l * r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the mul of Broadcast and VecArr.").into()
    }
}
impl<U,const N:usize> Mul<VecArr<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Mul<Output = U> + 'static,
          for<'a> Broadcast<Arr<U,N>>: Mul<&'a VecArr<U,Arr<U,N>>,Output = VecArr<U,Arr<U,N>>>, {
    type Output = VecArr<U,Arr<U,N>>;

    fn mul(self, rhs: VecArr<U,Arr<U,N>>) -> Self::Output {
        self * &rhs
    }
}
impl<'a,U,const N:usize> Mul<Broadcast<Arr<U,N>>> for &'a VecArr<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Mul<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn mul(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l * r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the mul of Broadcast and VecArr.").into()
    }
}
impl<U,const N:usize> Mul<Broadcast<Arr<U,N>>> for VecArr<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Mul<Output = U> + 'static,
          for<'a> &'a VecArr<U,Arr<U,N>>: Mul<Broadcast<Arr<U,N>>,Output = VecArr<U,Arr<U,N>>> {
    type Output = VecArr<U,Arr<U,N>>;

    fn mul(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        &self * rhs
    }
}
impl<'a,U,const N:usize> Div<&'a VecArr<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Div<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn div(self, rhs: &'a VecArr<U,Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l / r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the div of Broadcast and VecArr.").into()
    }
}
impl<'a,U,const N:usize> Div<Broadcast<Arr<U,N>>> for &'a VecArr<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Div<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn div(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l / r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the div of Broadcast and VecArr.").into()
    }
}
impl<U,const N:usize> Div<VecArr<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Div<Output = U> + 'static,
          for<'a> Broadcast<Arr<U,N>>: Div<&'a VecArr<U,Arr<U,N>>,Output = VecArr<U,Arr<U,N>>>, {
    type Output = VecArr<U,Arr<U,N>>;

    fn div(self, rhs: VecArr<U,Arr<U,N>>) -> Self::Output {
        self / &rhs
    }
}
impl<U,const N:usize> Div<Broadcast<Arr<U,N>>> for VecArr<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Div<Output = U> + 'static,
          for<'a> &'a VecArr<U,Arr<U,N>>: Div<Broadcast<Arr<U,N>>,Output = VecArr<U,Arr<U,N>>> {
    type Output = VecArr<U,Arr<U,N>>;

    fn div(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        &self / rhs
    }
}

/// Image implementation
#[derive(Debug,Eq,PartialEq)]
pub struct Image<T,const C:usize, const H:usize, const W:usize> where T: Default {
    arr:Box<[T]>
}
impl<T,const C:usize,const H:usize,const W:usize> Clone for Image<T,C,H,W> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        Image {
            arr:self.arr.clone()
        }
    }
}
impl<T,const C:usize,const H:usize,const W:usize> Image<T,C,H,W> where T: Default {
    /// Create an instance of Image
    pub fn new() -> Image<T,C,H,W> {
        let mut arr = Vec::with_capacity(C * H * W);
        arr.resize_with(C*H*W,Default::default);

        Image {
            arr:arr.into_boxed_slice()
        }
    }

    /// Obtaining a immutable iterator
    pub fn iter<'a>(&'a self) -> ImageIter<'a,T,H,W> {
        ImageIter(&*self.arr)
    }

    /// Obtaining a mutable iterator
    pub fn iter_mut<'a>(&'a mut self) -> ImageIterMut<'a,T,H,W> {
        ImageIterMut(&mut *self.arr)
    }
}
impl<T,const C:usize, const H:usize, const W:usize> Index<(usize,usize,usize)> for Image<T,C,H,W> where T: Default {
    type Output = T;

    fn index(&self, (c,y,x): (usize, usize, usize)) -> &Self::Output {
        if c >= C {
            panic!("index out of bounds: the len is {} but the index is {}",C,c);
        } else if y >= H {
            panic!("index out of bounds: the len is {} but the index is {}",H,y);
        } else if x >= W {
            panic!("index out of bounds: the len is {} but the index is {}",W,x);
        }
        &self.arr[c * H * W + y * W + x]
    }
}
impl<T,const C:usize, const H:usize, const W:usize> IndexMut<(usize,usize,usize)> for Image<T,C,H,W> where T: Default {
    fn index_mut(&mut self, (c,y,x): (usize, usize, usize)) -> &mut Self::Output {
        if c >= C {
            panic!("index out of bounds: the len is {} but the index is {}",C,c);
        } else if y >= H {
            panic!("index out of bounds: the len is {} but the index is {}",H,y);
        } else if x >= W {
            panic!("index out of bounds: the len is {} but the index is {}",W,x);
        }
        &mut self.arr[c * H * W + y * W + x]
    }
}
impl<'a,T,const C:usize,const H:usize,const W:usize> AsRawSlice<T> for Image<T,C,H,W> where T: Default {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
/// Implementation of an immutable view of a Image
#[derive(Debug,Eq,PartialEq)]
pub struct ImageView<'a,T,const H:usize,const W:usize> {
    arr:&'a [T]
}
impl<'a,T,const H:usize,const W:usize> Clone for ImageView<'a,T,H,W> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        ImageView {
            arr:self.arr
        }
    }
}
impl<'a,T,const H:usize,const W:usize> ImageView<'a,T,H,W> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        W
    }
}
impl<'a,T,const H:usize,const W:usize> Iterator for ImageView<'a,T,H,W> {
    type Item = ArrView<'a,T,W>;

    fn next(&mut self) -> Option<Self::Item> {
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
}
impl<'a,T,const H:usize, const W:usize> Index<(usize,usize)> for ImageView<'a,T,H,W> where T: Default {
    type Output = T;

    fn index(&self, (y,x): (usize, usize)) -> &Self::Output {
        if y >= H {
            panic!("index out of bounds: the len is {} but the index is {}",H,y);
        } else if x >= W {
            panic!("index out of bounds: the len is {} but the index is {}",W,x);
        }
        &self.arr[y * W + x]
    }
}
impl<'a,T,const H:usize,const W:usize> AsRawSlice<T> for ImageView<'a,T,H,W> {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
/// Implementation of an mutable view of a Image
#[derive(Debug,Eq,PartialEq)]
pub struct ImageViewMut<'a,T,const H:usize,const W:usize> {
    arr:&'a mut [T]
}
impl<'a,T,const H:usize,const W:usize> ImageViewMut<'a,T,H,W> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        W
    }
}
impl<'a,T,const H:usize,const W:usize> Iterator for ImageViewMut<'a,T,H,W> {
    type Item = ArrViewMut<'a,T,W>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.arr = r;

            Some(ArrViewMut {
                arr:l
            })
        }
    }
}
impl<'a,T,const H:usize, const W:usize> Index<(usize,usize)> for ImageViewMut<'a,T,H,W> where T: Default {
    type Output = T;

    fn index(&self, (y,x): (usize, usize)) -> &Self::Output {
        if y >= H {
            panic!("index out of bounds: the len is {} but the index is {}",H,y);
        } else if x >= W {
            panic!("index out of bounds: the len is {} but the index is {}",W,x);
        }
        &self.arr[y * W + x]
    }
}
impl<'a,T,const H:usize, const W:usize> IndexMut<(usize,usize)> for ImageViewMut<'a,T,H,W> where T: Default {
    fn index_mut(&mut self, (y,x): (usize, usize)) -> &mut Self::Output {
        if y >= H {
            panic!("index out of bounds: the len is {} but the index is {}",H,y);
        } else if x >= W {
            panic!("index out of bounds: the len is {} but the index is {}",W,x);
        }
        &mut self.arr[y * W + x]
    }
}
/// Implementation of an immutable iterator for fixed-length 3D arrays
#[derive(Debug,Eq,PartialEq)]
pub struct ImageIter<'a,T,const H:usize,const W:usize>(&'a [T]);

impl<'a,T,const H:usize,const W:usize> ImageIter<'a,T,H,W> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        H * W
    }
}
impl<'a,T,const H:usize,const W:usize> Iterator for ImageIter<'a,T,H,W> {
    type Item = ImageView<'a,T,H,W>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(ImageView { arr: l })
        }
    }
}
/// Implementation of an mutable iterator for fixed-length 3D arrays
#[derive(Debug,Eq,PartialEq)]
pub struct ImageIterMut<'a,T,const H:usize,const W:usize>(&'a mut [T]);

impl<'a,T,const H:usize,const W:usize> ImageIterMut<'a,T,H,W> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        H * W
    }
}
impl<'a,T,const H:usize,const W:usize> Iterator for ImageIterMut<'a,T,H,W> {
    type Item = ImageViewMut<'a,T,H,W>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.0 = r;

            Some(ImageViewMut { arr: l })
        }
    }
}
/// ParallelIterator implementation for Image
#[derive(Debug)]
pub struct ImageParIter<'data,T,const C:usize,const H:usize,const W:usize>(&'data [T]);

/// Implementation of plumbing::Producer for Image
#[derive(Debug)]
pub struct ImageIterProducer<'data,T,const C:usize,const H:usize,const W:usize>(&'data [T]);

impl<'data,T,const C:usize, const H:usize, const W:usize> ImageIterProducer<'data,T,C,H,W> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        H * W
    }
}
impl<'data,T,const C:usize,const H:usize,const W:usize> Iterator for ImageIterProducer<'data,T,C,H,W> {
    type Item = ImageView<'data,T,H,W>;

    fn next(&mut self) -> Option<ImageView<'data,T,H,W>> {
        let slice = std::mem::replace(&mut self.0, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(ImageView {
                arr:l
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (C, Some(C))
    }
}
impl<'data,T,const C:usize,const H:usize,const W:usize> std::iter::ExactSizeIterator for ImageIterProducer<'data,T,C,H,W> {
    fn len(&self) -> usize {
        C
    }
}
impl<'data,T,const C:usize,const H:usize,const W:usize> std::iter::DoubleEndedIterator for ImageIterProducer<'data,T,C,H,W> {
    fn next_back(&mut self) -> Option<ImageView<'data,T,H,W>> {
        let slice = std::mem::replace(&mut self.0, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.0.len() - self.element_size());

            self.0 = l;

            Some(ImageView {
                arr:r
            })
        }
    }
}
impl<'data, T: Send + Sync + 'static,const C:usize,const H:usize,const W:usize> plumbing::Producer for ImageIterProducer<'data,T,C,H,W> {
    type Item = ImageView<'data,T,H,W>;
    type IntoIter = Self;

    fn into_iter(self) -> Self { self }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let (l,r) = self.0.split_at(mid * H * W);

        (ImageIterProducer(l),ImageIterProducer(r))
    }
}
impl<'data, T: Send + Sync + 'static,const C: usize, const H: usize, const W:usize> ParallelIterator for ImageParIter<'data,T,C,H,W> {
    type Item = ImageView<'data,T,H,W>;

    fn opt_len(&self) -> Option<usize> { Some(IndexedParallelIterator::len(self)) }

    fn drive_unindexed<CS>(self, consumer: CS) -> CS::Result
        where
            CS: plumbing::UnindexedConsumer<Self::Item>,
    {
        self.drive(consumer)
    }
}
impl<'data, T: Send + Sync + 'static, const C: usize, const H: usize, const W:usize> IndexedParallelIterator for ImageParIter<'data,T,C,H,W> {
    fn len(&self) -> usize { C }

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
        callback.callback(ImageIterProducer::<T,C,H,W>(&self.0))
    }
}
impl<'data,T, const C:usize, const H:usize, const W:usize> IntoParallelRefIterator<'data> for Image<T,C,H,W>
    where T: Send + Sync + 'static + Default {
    type Iter = ImageParIter<'data,T,C,H,W>;
    type Item = ImageView<'data,T,H,W>;

    fn par_iter(&'data self) -> Self::Iter {
        ImageParIter(&self.arr)
    }
}
/// ParallelIterator implementation for ImageView
#[derive(Debug)]
pub struct ImageViewParIter<'data,T,const H:usize,const W:usize>(&'data [T]);

/// Implementation of plumbing::Producer for ImageView
#[derive(Debug)]
pub struct ImageViewIterProducer<'data,T,const H:usize,const W:usize>(&'data [T]);

impl<'data,T,const H:usize, const W:usize> ImageViewIterProducer<'data,T,H,W> {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        W
    }
}
impl<'data,T,const H:usize,const W:usize> Iterator for ImageViewIterProducer<'data,T,H,W> {
    type Item = ArrView<'data,T,W>;

    fn next(&mut self) -> Option<ArrView<'data,T,W>> {
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
        (H, Some(H))
    }
}
impl<'data,T,const H:usize,const W:usize> std::iter::ExactSizeIterator for ImageViewIterProducer<'data,T,H,W> {
    fn len(&self) -> usize {
        H
    }
}
impl<'data,T,const H:usize,const W:usize> std::iter::DoubleEndedIterator for ImageViewIterProducer<'data,T,H,W> {
    fn next_back(&mut self) -> Option<ArrView<'data,T,W>> {
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
impl<'data, T: Send + Sync + 'static,const H:usize,const W:usize> plumbing::Producer for ImageViewIterProducer<'data,T,H,W> {
    type Item = ArrView<'data,T,W>;
    type IntoIter = Self;

    fn into_iter(self) -> Self { self }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let (l,r) = self.0.split_at(mid * W);

        (ImageViewIterProducer(l),ImageViewIterProducer(r))
    }
}
impl<'data, T: Send + Sync + 'static,const H: usize, const W:usize> ParallelIterator for ImageViewParIter<'data,T,H,W> {
    type Item = ArrView<'data,T,W>;

    fn opt_len(&self) -> Option<usize> { Some(IndexedParallelIterator::len(self)) }

    fn drive_unindexed<CS>(self, consumer: CS) -> CS::Result
        where
            CS: plumbing::UnindexedConsumer<Self::Item>,
    {
        self.drive(consumer)
    }
}
impl<'data, T: Send + Sync + 'static, const H: usize, const W:usize> IndexedParallelIterator for ImageViewParIter<'data,T,H,W> {
    fn len(&self) -> usize { H }

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
        callback.callback(ImageViewIterProducer::<T,H,W>(&self.0))
    }
}
impl<'data,T, const H:usize, const W:usize> IntoParallelRefIterator<'data> for ImageView<'data,T,H,W>
    where T: Send + Sync + 'static + Default {
    type Iter = ImageViewParIter<'data,T,H,W>;
    type Item = ArrView<'data,T,W>;

    fn par_iter(&'data self) -> Self::Iter {
        ImageViewParIter(&self.arr)
    }
}
