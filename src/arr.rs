use std::cmp::Ordering;
use std::ops::{Deref, DerefMut, Index, IndexMut};
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

    pub fn iter_mut<'a>(&'a mut self) -> Arr2IterMut<'a,T,N2> {
        Arr2IterMut(&mut *self.arr)
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
pub struct Arr3<T,const N1:usize, const N2:usize, const N3:usize> where T: Default {
    arr:Box<[T]>
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
pub struct Arr4<T,const N1:usize, const N2:usize, const N3:usize, const N4:usize> where T: Default {
    arr:Box<[T]>
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
        N
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
        N
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
pub struct Arr3Iter<'a,T,const N1:usize,const N2:usize>(&'a [T]);

impl<'a,T,const N1:usize,const N2:usize> Arr3Iter<'a,T,N1,N2> {
    const fn element_size(&self) -> usize {
        N1 * N2
    }
}
impl<'a,T,const N1:usize,const N2:usize> Iterator for Arr3Iter<'a,T,N1,N2> {
    type Item = Arr2Iter<'a,T,N2>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(Arr2Iter(l))
        }
    }
}
pub struct Arr3IterMut<'a,T,const N1:usize,const N2:usize>(&'a mut [T]);

impl<'a,T,const N1:usize,const N2:usize> Arr3IterMut<'a,T,N1,N2> {
    const fn element_size(&self) -> usize {
        N1 * N2
    }
}
impl<'a,T,const N1:usize,const N2:usize> Iterator for Arr3IterMut<'a,T,N1,N2> {
    type Item = Arr2IterMut<'a,T,N2>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.0 = r;

            Some(Arr2IterMut(l))
        }
    }
}
pub struct Arr4Iter<'a,T,const N1:usize,const N2:usize,const N3:usize>(&'a [T]);

impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Arr4Iter<'a,T,N1,N2,N3> {
    const fn element_size(&self) -> usize {
        N1 * N2 * N3
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Iterator for Arr4Iter<'a,T,N1,N2,N3> {
    type Item = Arr3Iter<'a,T,N2,N3>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at(self.element_size());

            self.0 = r;

            Some(Arr3Iter(l))
        }
    }
}
pub struct Arr4IterMut<'a,T,const N1:usize,const N2:usize, const N3:usize>(&'a mut [T]);

impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Arr4IterMut<'a,T,N1,N2,N3> {
    const fn element_size(&self) -> usize {
        N1 * N2 * N3
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> Iterator for Arr4IterMut<'a,T,N1,N2,N3> {
    type Item = Arr3IterMut<'a,T,N2,N3>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut slice = std::mem::replace(&mut self.0, &mut []);
        if slice.is_empty() {
            None
        } else {
            let (l,r) = slice.split_at_mut(self.element_size());

            self.0 = r;

            Some(Arr3IterMut(l))
        }
    }
}