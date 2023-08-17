use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};
use rayon::iter::plumbing;
use rayon::prelude::{ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use crate::arr::{Arr, ArrView, ArrViewMut, VecArr};
use crate::error::SizeMismatchError;
use crate::mem::{AsRawMutSlice, AsRawSlice};

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

/// Images implementation
#[derive(Debug,Eq,PartialEq)]
pub struct Images<T,const C:usize, const H:usize, const W:usize> where T: Default + Clone + Send {
    arr:Box<[T]>
}
impl<T,const C:usize,const H:usize,const W:usize> Clone for Images<T,C,H,W> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        Images {
            arr:self.arr.clone()
        }
    }
}
impl<T,const C:usize,const H:usize,const W:usize> Images<T,C,H,W> where T: Default + Clone + Send {
    /// Create an instance of Images
    pub fn new() -> Images<T,C,H,W> {
        let mut arr = Vec::with_capacity(C * H * W);
        arr.resize_with(C*H*W,Default::default);

        Images {
            arr:arr.into_boxed_slice()
        }
    }

    /// Obtaining a immutable iterator
    pub fn iter<'a>(&'a self) -> ImageIter<'a,T,H,W> {
        ImageIter{ arr: &*self.arr }
    }

    /// Obtaining a mutable iterator
    pub fn iter_mut<'a>(&'a mut self) -> ImageIterMut<'a,T,H,W> {
        ImageIterMut{ arr: &mut *self.arr }
    }
}
impl<T,const C:usize, const H:usize, const W:usize> Index<(usize,usize,usize)> for Images<T,C,H,W> where T: Default + Clone + Send {
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
impl<T,const C:usize, const H:usize, const W:usize> IndexMut<(usize,usize,usize)> for Images<T,C,H,W> where T: Default + Clone + Send {
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
impl<T,const C:usize,const H:usize,const W:usize> TryFrom<Vec<Image<T,H,W>>> for Images<T,C,H,W> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(items: Vec<Image<T,H,W>>) -> Result<Self,SizeMismatchError> {
        if items.len() != C {
            Err(SizeMismatchError(items.len(),C))
        } else {
            let mut buffer = Vec::with_capacity(C * H * W);

            for v in items.into_iter() {
                buffer.extend_from_slice(&v.arr);
            }
            Ok(Images {
                arr: buffer.into_boxed_slice()
            })
        }
    }
}
/// Implementation of an immutable iterator for image
#[derive(Debug,Eq,PartialEq)]
pub struct ImageIter<'a,T,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:&'a [T],
}
impl<'a,T,const H:usize,const W:usize> ImageIter<'a,T,H,W> where T: Default + Clone + Send {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        H * W
    }
}
impl<'a,T,const H:usize,const W:usize> Iterator for ImageIter<'a,T,H,W> where T: Default + Clone + Send {
    type Item = ImageView<'a,T,H,W>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.arr = r;

            Some(ImageView{ arr: l })
        }
    }
}
/// Implementation of an mutable iterator for image
#[derive(Debug,Eq,PartialEq)]
pub struct ImageIterMut<'a,T,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:&'a mut [T],
}
impl<'a,T,const H:usize,const W:usize> ImageIterMut<'a,T,H,W> where T: Default + Clone + Send {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        H * W
    }
}
impl<'a,T,const H:usize,const W:usize> Iterator for ImageIterMut<'a,T,H,W> where T: Default + Clone + Send {
    type Item = ImageViewMut<'a,T,H,W>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.arr = r;

            Some(ImageViewMut{ arr: l })
        }
    }
}
impl<'a,T,const C:usize,const H:usize,const W:usize> AsRawSlice<T> for Images<T,C,H,W> where T: Default + Clone + Send {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
/// Implementation of an immutable view of a Images
#[derive(Debug,Eq,PartialEq)]
pub struct ImagesView<'a,T,const C:usize, const H:usize, const W:usize> where T: Default + Clone + Send {
    arr:&'a [T],
}
impl<'a,T,const C:usize,const H:usize,const W:usize> ImagesView<'a,T,C,H,W> where T: Default + Clone + Send {
    /// Obtaining a immutable iterator
    pub fn iter(&self) -> ImageIter<'a,T,H,W> {
        ImageIter{ arr: self.arr }
    }
}
impl<'a,T,const C:usize,const H:usize,const W:usize> Clone for ImagesView<'a,T,C,H,W> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        ImagesView{ arr: self.arr }
    }
}
impl<'a,T,const C:usize, const H:usize, const W:usize> Index<(usize,usize,usize)> for ImagesView<'a,T,C,H,W>
    where T: Default + Clone + Send {
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
impl<'a,T,const C:usize,const H:usize,const W:usize> AsRawSlice<T> for ImagesView<'a,T,C,H,W>
    where T: Default + Clone + Send {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
/// Implementation of an mutable view of a Images
#[derive(Debug,Eq,PartialEq)]
pub struct ImagesViewMut<'a,T,const C:usize, const H:usize, const W:usize> where T: Default + Clone + Send {
    arr:&'a mut [T],
}
impl<'a,T,const C:usize,const H:usize,const W:usize> ImagesViewMut<'a,T,C,H,W> where T: Default + Clone + Send {
    /// Obtaining a immutable iterator
    pub fn iter(&'a self) -> ImageIter<'a,T,H,W> {
        ImageIter{ arr: self.arr }
    }
    /// Obtaining a mutable iterator
    pub fn iter_mut(&'a mut self) -> ImageIterMut<'a,T,H,W> {
        ImageIterMut{ arr: &mut self.arr }
    }
}
impl<'a,T,const C:usize, const H:usize, const W:usize> Index<(usize,usize,usize)> for ImagesViewMut<'a,T,C,H,W>
    where T: Default + Clone + Send {
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
impl<'a,T,const C:usize, const H:usize, const W:usize> IndexMut<(usize,usize,usize)> for ImagesViewMut<'a,T,C,H,W>
    where T: Default + Clone + Send {
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
/// Image Implementation
#[derive(Debug,Eq,PartialEq)]
pub struct Image<T,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:Box<[T]>,
}
impl<T,const H:usize,const W:usize> Image<T,H,W> where T: Default + Clone + Send {
    /// Create an instance of Image
    pub fn new() -> Image<T,H,W> {
        let mut arr = Vec::with_capacity(H * W);
        arr.resize_with(H * W,Default::default);

        Image {
            arr:arr.into_boxed_slice()
        }
    }
    /// Obtaining a immutable iterator
    pub fn iter<'a>(&'a self) -> ImageView<'a,T,H,W> {
        ImageView{ arr: &*self.arr }
    }

    /// Obtaining a mutable iterator
    pub fn iter_mut<'a>(&'a mut self) -> ImageViewMut<'a,T,H,W> {
        ImageViewMut{ arr: &mut *self.arr }
    }
}
impl<T,const H:usize,const W:usize> Clone for Image<T,H,W> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        Image {
            arr:self.arr.clone(),
        }
    }
}
impl<T,const H:usize, const W:usize> Index<(usize,usize)> for Image<T,H,W> where T: Default + Clone + Send {
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
impl<T,const H:usize,const W:usize> TryFrom<Vec<Arr<T,W>>> for Image<T,H,W> where T: Default + Clone + Send {
    type Error = SizeMismatchError;

    fn try_from(items: Vec<Arr<T,W>>) -> Result<Self,SizeMismatchError> {
        if items.len() != H {
            Err(SizeMismatchError(items.len(),H))
        } else {
            let mut buffer = Vec::with_capacity(H * W);

            for v in items.into_iter() {
                buffer.extend_from_slice(&v);
            }
            Ok(Image {
                arr: buffer.into_boxed_slice(),
            })
        }
    }
}
impl<T,const H:usize,const W:usize> AsRawSlice<T> for Image<T,H,W> where T: Default + Clone + Send {
    fn as_raw_slice(&self) -> &[T] {
        &*self.arr
    }
}
impl<'a,T,const H:usize,const W:usize> AsRawMutSlice<'a,T> for Image<T,H,W> where T: Default + Clone + Send {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T] {
        &mut *self.arr
    }
}
/// Implementation of an immutable view of a Image
#[derive(Debug,Eq,PartialEq)]
pub struct ImageView<'a,T,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:&'a [T],
}
impl<'a,T,const H:usize,const W:usize> Clone for ImageView<'a,T,H,W> where T: Default + Clone + Send {
    fn clone(&self) -> Self {
        ImageView{ arr: self.arr }
    }
}
impl<'a,T,const H:usize,const W:usize> ImageView<'a,T,H,W> where T: Default + Clone + Send {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        W
    }
}
impl<'a,T,const H:usize,const W:usize> Iterator for ImageView<'a,T,H,W> where T: Default + Clone + Send {
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
impl<'a,T,const H:usize, const W:usize> Index<(usize,usize)> for ImageView<'a,T,H,W>
    where T: Default + Clone + Send {
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
impl<'a,T,const H:usize,const W:usize> AsRawSlice<T> for ImageView<'a,T,H,W> where T: Default + Clone + Send {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
/// Implementation of an mutable view of a Image
#[derive(Debug,Eq,PartialEq)]
pub struct ImageViewMut<'a,T,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:&'a mut [T],
}
impl<'a,T,const H:usize,const W:usize> ImageViewMut<'a,T,H,W> where T: Default + Clone + Send {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        W
    }
}
impl<'a,T,const H:usize,const W:usize> Iterator for ImageViewMut<'a,T,H,W> where T: Default + Clone + Send {
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
impl<'a,T,const H:usize, const W:usize> Index<(usize,usize)> for ImageViewMut<'a,T,H,W> where T: Default + Clone + Send {
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
impl<'a,T,const H:usize, const W:usize> IndexMut<(usize,usize)> for ImageViewMut<'a,T,H,W>
    where T: Default + Clone + Send {
    fn index_mut(&mut self, (y,x): (usize, usize)) -> &mut Self::Output {
        if y >= H {
            panic!("index out of bounds: the len is {} but the index is {}",H,y);
        } else if x >= W {
            panic!("index out of bounds: the len is {} but the index is {}",W,x);
        }
        &mut self.arr[y * W + x]
    }
}
/// ParallelIterator implementation for Image
#[derive(Debug)]
pub struct ImageParIter<'data,T,const C:usize,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:&'data [T],
}
/// Implementation of plumbing::Producer for Image
#[derive(Debug)]
pub struct ImageIterProducer<'data,T,const C:usize,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:&'data [T],
}
impl<'data,T,const C:usize, const H:usize, const W:usize> ImageIterProducer<'data,T,C,H,W> where T: Default + Clone + Send {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        H * W
    }
}
impl<'data,T,const C:usize,const H:usize,const W:usize> Iterator for ImageIterProducer<'data,T,C,H,W> where T: Default + Clone + Send {
    type Item = ImageView<'data,T,H,W>;

    fn next(&mut self) -> Option<ImageView<'data,T,H,W>> {
        let slice = std::mem::replace(&mut self.arr, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.arr = r;

            Some(ImageView{ arr: l })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (C, Some(C))
    }
}
impl<'data,T,const C:usize,const H:usize,const W:usize> std::iter::ExactSizeIterator for ImageIterProducer<'data,T,C,H,W>
    where T: Default + Clone + Send{
    fn len(&self) -> usize {
        C
    }
}
impl<'data,T,const C:usize,const H:usize,const W:usize> std::iter::DoubleEndedIterator for ImageIterProducer<'data,T,C,H,W>
    where T: Default + Clone + Send {
    fn next_back(&mut self) -> Option<ImageView<'data,T,H,W>> {
        let slice = std::mem::replace(&mut self.arr, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.arr.len() - self.element_size());

            self.arr = l;

            Some(ImageView{ arr: r })
        }
    }
}
impl<'data, T: Send + Sync + 'static,const C:usize,const H:usize,const W:usize> plumbing::Producer
    for ImageIterProducer<'data,T,C,H,W> where T: Default + Clone + Send {
    type Item = ImageView<'data,T,H,W>;
    type IntoIter = Self;

    fn into_iter(self) -> Self { self }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let (l,r) = self.arr.split_at(mid * H * W);

        (ImageIterProducer{ arr: l },ImageIterProducer{ arr: r })
    }
}
impl<'data, T: Send + Sync + 'static,const C: usize, const H: usize, const W:usize> ParallelIterator
    for ImageParIter<'data,T,C,H,W> where T: Default + Clone + Send {
    type Item = ImageView<'data,T,H,W>;

    fn opt_len(&self) -> Option<usize> { Some(IndexedParallelIterator::len(self)) }

    fn drive_unindexed<CS>(self, consumer: CS) -> CS::Result
        where
            CS: plumbing::UnindexedConsumer<Self::Item>,
    {
        self.drive(consumer)
    }
}
impl<'data, T: Send + Sync + 'static, const C: usize, const H: usize, const W:usize> IndexedParallelIterator
    for ImageParIter<'data,T,C,H,W> where T: Default + Clone + Send {
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
        callback.callback(ImageIterProducer::<T, C, H, W>{ arr: &self.arr })
    }
}
impl<'data,T, const C:usize, const H:usize, const W:usize> IntoParallelRefIterator<'data> for Images<T,C,H,W>
    where T: Default + Clone + Send + Sync + 'static {
    type Iter = ImageParIter<'data,T,C,H,W>;
    type Item = ImageView<'data,T,H,W>;

    fn par_iter(&'data self) -> Self::Iter {
        ImageParIter{ arr: &self.arr }
    }
}
/// Implement a fixed-length image array whose size is not specified by a type parameter.
#[derive(Debug,Eq,PartialEq,Clone)]
pub struct VecImages<T,const C:usize,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:Box<[T]>,
    len:usize,
}
impl<T,const C:usize,const H:usize,const W:usize> VecImages<T,C,H,W> where T: Default + Clone + Copy + Send {
    /// get the number of element
    pub fn len(&self) -> usize {
        self.len
    }
}
impl<T,const C:usize,const H:usize,const W:usize> VecImages<T,C,H,W> where T: Default + Clone + Copy + Send {
    /// Create a VecImages instance of the specified size
    /// # Arguments
    /// * `size`- Size to be secured
    pub fn with_size(size:usize) -> VecImages<T,C,H,W> {
        let mut arr = Vec::with_capacity(C * H * W * size);

        arr.resize_with(C * H * W * size,Default::default);

        VecImages {
            arr:arr.into_boxed_slice(),
            len:size,
        }
    }

    /// Obtaining a immutable iterator
    pub fn iter(&self) -> ImagesIter<T,C,H,W> {
        ImagesIter{ arr: &*self.arr }
    }

    /// Obtaining a mutable iterator
    pub fn iter_mut(&mut self) -> ImagesIterMut<T,C,H,W> {
        ImagesIterMut{ arr: &mut *self.arr }
    }
}
impl<T,const C:usize,const H:usize,const W:usize> From<Vec<Images<T,C,H,W>>> for VecImages<T,C,H,W>
    where T: Default + Clone + Copy + Send {

    fn from(items: Vec<Images<T,C,H,W>>) -> Self {
        let len = items.len();

        let mut buffer = Vec::with_capacity(len * C * H * W);

        for item in items.into_iter() {
            buffer.extend_from_slice(&item.arr);
        }

        VecImages {
            arr:buffer.into_boxed_slice(),
            len:len,
        }
    }
}
impl<'data,T,const C:usize,const H:usize,const W:usize> From<Vec<ImagesView<'data,T,C,H,W>>>
    for VecImages<T,C,H,W> where T: Default + Clone + Copy + Send {

    fn from(items: Vec<ImagesView<'data,T,C,H,W>>) -> Self {
        let len = items.len();

        let mut buffer = Vec::with_capacity(len * C * H * W);

        for item in items.into_iter() {
            buffer.extend_from_slice(&item.arr);
        }

        VecImages {
            arr:buffer.into_boxed_slice(),
            len:len,
        }
    }
}
/// VecImages's Immutable Iterator
#[derive(Debug,Eq,PartialEq)]
pub struct ImagesIter<'a,T,const C:usize,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:&'a [T],
}
impl<'a,T,const C:usize,const H:usize,const W:usize> ImagesIter<'a,T,C,H,W> where T: Default + Clone + Send {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        C * H * W
    }
}
impl<'a,T,const C:usize,const H:usize,const W:usize> Iterator for ImagesIter<'a,T,C,H,W> where T: Default + Clone + Send {
    type Item = ImagesView<'a,T,C,H,W>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.arr = r;

            Some(ImagesView{ arr: l })
        }
    }
}

/// VecImages's mutable Iterator
#[derive(Debug,Eq,PartialEq)]
pub struct ImagesIterMut<'a,T,const C:usize,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:&'a mut [T],
}
impl<'a,T,const C:usize,const H:usize,const W:usize> ImagesIterMut<'a,T,C,H,W> where T: Default + Clone + Send {
    /// Number of elements encompassed by the iterator element
    const fn element_size(&self) -> usize {
        C * H * W
    }
}
impl<'a,T,const C:usize,const H:usize,const W:usize> Iterator for ImagesIterMut<'a,T,C,H,W> where T: Default + Clone + Send {
    type Item = ImagesViewMut<'a,T,C,H,W>;

    fn next(&mut self) -> Option<Self::Item> {
        let slice = std::mem::replace(&mut self.arr, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.arr = r;

            Some(ImagesViewMut{ arr: l })
        }
    }
}
impl<'a,T,const C:usize,const H:usize,const W:usize> AsRawSlice<T> for VecImages<T,C,H,W> where T: Default + Clone + Send {
    fn as_raw_slice(&self) -> &[T] {
        &self.arr
    }
}
impl<'a,T,const C:usize,const H:usize,const W:usize> AsRawMutSlice<'a,T> for VecImages<T,C,H,W> where T: Default + Clone + Send {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T] {
        &mut self.arr
    }
}
/// ParallelIterator implementation for Images
#[derive(Debug)]
pub struct ImagesParIter<'data,T,const C:usize,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:&'data [T],
    len:usize,
}
pub struct ImagesIterProducer<'data,T,const C:usize,const H:usize,const W:usize> where T: Default + Clone + Send {
    arr:&'data [T],
    len:usize,
}
/// Implementation of plumbing::Producer for VecImages
impl<'data,T,const C:usize,const H:usize,const W:usize> ImagesIterProducer<'data,T,C,H,W>
    where T: Default + Clone + Send {
    /// Number of elements encompassed by the iterator element
    fn element_size(&self) -> usize {
        C * H * W
    }
}
impl<'data,T,const C:usize,const H:usize,const W:usize> Iterator for ImagesIterProducer<'data,T,C,H,W>
    where T: Default + Clone + Send {
    type Item = ImagesView<'data,T,C,H,W>;

    fn next(&mut self) -> Option<ImagesView<'data,T,C,H,W>> {
        let slice = std::mem::replace(&mut self.arr, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.arr = r;

            Some(ImagesView{ arr: l })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        ({self.len}, Some(self.len))
    }
}
impl<'data,T,const C:usize,const H:usize,const W:usize> std::iter::ExactSizeIterator
    for ImagesIterProducer<'data,T,C,H,W> where T: Default + Clone + Send {

    fn len(&self) -> usize {
        self.len
    }
}
impl<'data,T,const C:usize,const H:usize,const W:usize> std::iter::DoubleEndedIterator
    for ImagesIterProducer<'data,T,C,H,W> where T: Default + Clone + Send {

    fn next_back(&mut self) -> Option<ImagesView<'data,T,C,H,W>> {
        let slice = std::mem::replace(&mut self.arr, &mut []);

        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.arr.len() - self.element_size());

            self.arr = l;

            Some(ImagesView{ arr: r })
        }
    }
}
impl<'data, T: Send + Sync + 'static,const C:usize,const H:usize,const W:usize> plumbing::Producer
    for ImagesIterProducer<'data,T,C,H,W> where T: Default + Clone + Send {

    type Item = ImagesView<'data,T,C,H,W>;
    type IntoIter = Self;

    fn into_iter(self) -> Self { self }

    fn split_at(self, mid: usize) -> (Self, Self) {
        let (l,r) = self.arr.split_at(mid * C * H * W);

        (ImagesIterProducer {
            arr: l,
            len: self.len
        },ImagesIterProducer {
            arr: r,
            len: self.len
        })
    }
}
impl<'data, T: Send + Sync + 'static,const C:usize,const H:usize,const W:usize> ParallelIterator
    for ImagesParIter<'data,T,C,H,W> where T: Default + Clone + Send {

    type Item = ImagesView<'data,T,C,H,W>;

    fn opt_len(&self) -> Option<usize> { Some(IndexedParallelIterator::len(self)) }

    fn drive_unindexed<CS>(self, consumer: CS) -> CS::Result
        where
            CS: plumbing::UnindexedConsumer<Self::Item>,
    {
        self.drive(consumer)
    }
}
impl<'data, T: Send + Sync + 'static, const C:usize, const H:usize, const W:usize> IndexedParallelIterator
    for ImagesParIter<'data,T,C,H,W> where T: Default + Clone + Send {

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
        callback.callback(ImagesIterProducer {
            arr: &self.arr,
            len: self.len
        })
    }
}
impl<'data,T, const C:usize, const H:usize, const W:usize> IntoParallelRefIterator<'data> for VecImages<T,C,H,W>
    where T: Default + Clone + Copy + Send + Sync + 'static {
    type Iter = ImagesParIter<'data,T,C,H,W>;
    type Item = ImagesView<'data,T,C,H,W>;

    fn par_iter(&'data self) -> Self::Iter {
        ImagesParIter {
            arr: &self.arr,
            len: self.len
        }
    }
}
impl<'data,T, const C:usize, const H:usize, const W:usize> IntoParallelRefIterator<'data> for &'data VecImages<T,C,H,W>
    where T: Default + Clone + Copy + Send + Sync + 'static {
    type Iter = ImagesParIter<'data,T,C,H,W>;
    type Item = ImagesView<'data,T,C,H,W>;

    fn par_iter(&'data self) -> Self::Iter {
        ImagesParIter {
            arr: &self.arr,
            len: self.len
        }
    }
}
