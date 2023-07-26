use std::ops::{Add, Div, Mul, Sub};
use rayon::prelude::{ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use crate::arr::{Arr, VecArr};

#[derive(Clone)]
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
impl<'a,U,const N:usize> Add<Broadcast<Arr<U,N>>> for &'a VecArr<U,Arr<U,N>>
    where U: Default + Clone + Send + Sync,
          Broadcast<Arr<U,N>>: Add<&'a VecArr<U,Arr<U,N>>,Output = VecArr<U,Arr<U,N>>>,
          Arr<U,N>: Clone {
    type Output = VecArr<U,Arr<U,N>>;

    fn add(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        rhs + self
    }
}
impl<'a,U,const N:usize> Sub<&'a VecArr<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Sub<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn sub(self, rhs: &'a VecArr<U,Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l - r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the add of Broadcast and VecArr.").into()
    }
}
impl<'a,U,const N:usize> Sub<Broadcast<Arr<U,N>>> for &'a VecArr<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Sub<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn sub(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(rhs.0).take(self.len()).zip(self.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l - r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the add of Broadcast and VecArr.").into()
    }
}
impl<'a,U,const N:usize> Mul<&'a VecArr<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Mul<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn mul(self, rhs: &'a VecArr<U,Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l * r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the add of Broadcast and VecArr.").into()
    }
}
impl<'a,U,const N:usize> Mul<Broadcast<Arr<U,N>>> for &'a VecArr<U,Arr<U,N>>
    where U: Default + Clone + Send + Sync,
          Broadcast<Arr<U,N>>: Mul<&'a VecArr<U,Arr<U,N>>,Output = VecArr<U,Arr<U,N>>>,
          Arr<U,N>: Clone {
    type Output = VecArr<U,Arr<U,N>>;

    fn mul(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        rhs * self
    }
}
impl<'a,U,const N:usize> Div<&'a VecArr<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Div<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn div(self, rhs: &'a VecArr<U,Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l / r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the add of Broadcast and VecArr.").into()
    }
}
impl<'a,U,const N:usize> Div<Broadcast<Arr<U,N>>> for &'a VecArr<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Div<Output = U> + 'static {
    type Output = VecArr<U,Arr<U,N>>;

    fn div(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(rhs.0).take(self.len()).zip(self.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l / r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the add of Broadcast and VecArr.").into()
    }
}
