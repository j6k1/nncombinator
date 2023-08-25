//! Definition and implementation of various collection typess
use std::ops::{Add, Div, Mul, Sub};
use rayon::prelude::{ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use crate::arr::{Arr, SerializedVec};

/// A type that expresses a broadcast of the inner type when computing with the inner type and its type collection
#[derive(Debug,Clone)]
pub struct Broadcast<T>(pub T) where T: Clone;
impl<'a,U,const N:usize> Add<&'a SerializedVec<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output = U> + 'static {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn add(self, rhs: &'a SerializedVec<U,Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l + r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the add of Broadcast and SerializedVec.").into()
    }
}
impl<U,const N:usize> Add<SerializedVec<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output = U> + 'static,
          for<'a> Broadcast<Arr<U,N>>: Add<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>>, {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn add(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self + &rhs
    }
}
impl<'a,U,const N:usize> Add<Broadcast<Arr<U,N>>> for &'a SerializedVec<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output = U> + 'static {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn add(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l + r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the add of Broadcast and SerializedVec.").into()
    }
}
impl<U,const N:usize> Add<Broadcast<Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output = U> + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Add<Broadcast<Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn add(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        &self + rhs
    }
}
impl<'a,U,const N:usize> Sub<&'a SerializedVec<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Sub<Output = U> + 'static {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn sub(self, rhs: &'a SerializedVec<U,Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l - r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the sub of Broadcast and SerializedVec.").into()
    }
}
impl<U,const N:usize> Sub<SerializedVec<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Sub<Output = U> + 'static,
          for<'a> Broadcast<Arr<U,N>>: Sub<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>>, {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn sub(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self - &rhs
    }
}
impl<'a,U,const N:usize> Sub<Broadcast<Arr<U,N>>> for &'a SerializedVec<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Sub<Output = U> + 'static {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn sub(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l - r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the sub of Broadcast and SerializedVec.").into()
    }
}
impl<U,const N:usize> Sub<Broadcast<Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Sub<Output = U> + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Sub<Broadcast<Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn sub(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        &self - rhs
    }
}
impl<'a,U,const N:usize> Mul<&'a SerializedVec<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Mul<Output = U> + 'static {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn mul(self, rhs: &'a SerializedVec<U,Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l * r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the mul of Broadcast and SerializedVec.").into()
    }
}
impl<U,const N:usize> Mul<SerializedVec<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Mul<Output = U> + 'static,
          for<'a> Broadcast<Arr<U,N>>: Mul<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>>, {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn mul(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self * &rhs
    }
}
impl<'a,U,const N:usize> Mul<Broadcast<Arr<U,N>>> for &'a SerializedVec<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Mul<Output = U> + 'static {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn mul(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l * r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the mul of Broadcast and SerializedVec.").into()
    }
}
impl<U,const N:usize> Mul<Broadcast<Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Mul<Output = U> + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Mul<Broadcast<Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn mul(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        &self * rhs
    }
}
impl<'a,U,const N:usize> Div<&'a SerializedVec<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Div<Output = U> + 'static {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn div(self, rhs: &'a SerializedVec<U,Arr<U,N>>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l / r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the div of Broadcast and SerializedVec.").into()
    }
}
impl<'a,U,const N:usize> Div<Broadcast<Arr<U,N>>> for &'a SerializedVec<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Div<Output = U> + 'static {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn div(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l.par_iter().zip(r.par_iter()).map(|(&l,&r)| l / r).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>().expect("An error occurred in the div of Broadcast and SerializedVec.").into()
    }
}
impl<U,const N:usize> Div<SerializedVec<U,Arr<U,N>>> for Broadcast<Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Div<Output = U> + 'static,
          for<'a> Broadcast<Arr<U,N>>: Div<&'a SerializedVec<U,Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>>, {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn div(self, rhs: SerializedVec<U,Arr<U,N>>) -> Self::Output {
        self / &rhs
    }
}
impl<U,const N:usize> Div<Broadcast<Arr<U,N>>> for SerializedVec<U,Arr<U,N>>
    where U: Default + Clone + Copy + Send + Sync + Div<Output = U> + 'static,
          for<'a> &'a SerializedVec<U,Arr<U,N>>: Div<Broadcast<Arr<U,N>>,Output = SerializedVec<U,Arr<U,N>>> {
    type Output = SerializedVec<U,Arr<U,N>>;

    fn div(self, rhs: Broadcast<Arr<U,N>>) -> Self::Output {
        &self / rhs
    }
}
