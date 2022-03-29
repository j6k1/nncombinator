use std::cmp::Ordering;
use std::ops::{Deref, DerefMut};
use std::path::Iter;
use std::sync::Arc;

pub struct Arr<T,const N:usize> where T: Default {
    arr:Box<[T]>
}
impl<T,const N:usize> Arr<T,N> where T: Default {
    pub fn new() -> Arr<T,N> {
        let mut arr = Vec::with_capacity(N);
        arr.resize_with(N,Default::default);

        Arr {
            arr:arr.into_boxed_slice()
        }
    }
}
impl<T,const N:usize> Deref for Arr<T,N> where T: Default {
    type Target = Box<[T]>;
    fn deref(&self) -> &Self::Target {
        &self.arr
    }
}
impl<T,const N:usize> DerefMut for Arr<T,N> where T: Default  {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.arr
    }
}
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

    pub fn iter_mut<'a>(&'a mut self) -> Arr2Iter<'a,T,N2> {
        Arr2Iter(&mut *self.arr)
    }
}
pub struct ArrView<'a,T,const N:usize> {
    arr:&'a [T]
}
impl<'a,T,const N:usize> Deref for ArrView<'a,T,N> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.arr
    }
}
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
pub struct Arr2Iter<'a,T,const N:usize>(&'a [T]);

impl<'a,T,const N:usize> Arr2Iter<'a,T,N> {
    const fn element_size(&self) -> usize {
        N * std::mem::size_of::<T>()
    }
}
impl<'a,T,const N:usize> Iterator for Arr2Iter<'a,T,N> {
    type Item = ArrView<'a,T,N>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut slice = std::mem::replace(&mut self.0, &mut []);
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
pub struct Arr2IterMut<'a,T,const N:usize>(&'a mut [T]);

impl<'a,T,const N:usize> Arr2IterMut<'a,T,N> {
    const fn element_size(&self) -> usize {
        N * std::mem::size_of::<T>()
    }
}
impl<'a,T,const N:usize> Iterator for Arr2IterMut<'a,T,N> {
    type Item = ArrViewMut<'a,T,N>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut slice = std::mem::replace(&mut self.0, &mut []);
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