use std::ops::{Deref, DerefMut};

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