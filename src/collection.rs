//! Definition and implementation of various collection typess
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use rayon::prelude::{ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use crate::arr::{AsView, MakeView, SerializedVec, SerializedVecView, SliceSize};

/// A type that expresses a broadcast of the inner type when computing with the inner type and its type collection
#[derive(Debug,Clone)]
pub struct Broadcast<T>(pub T) where T: Clone;
impl<'a,U,T> Add<&'a SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Add<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Add<Output=T> + Add<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: &'a SerializedVec<U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l + r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Add<SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Add<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Add<Output=T> + Add<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: SerializedVec<U,T>) -> Self::Output {
        self + &rhs
    }
}
impl<'a,U,T> Sub<&'a SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Sub<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Sub<Output=T> + Sub<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: &'a SerializedVec<U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l - r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Sub<SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Sub<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Sub<Output=T> + Sub<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: SerializedVec<U,T>) -> Self::Output {
        self - &rhs
    }
}
impl<'a,U,T> Mul<&'a SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Mul<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Mul<Output=T> + Mul<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: &'a SerializedVec<U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l * r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Mul<SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Mul<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Mul<Output=T> + Mul<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: SerializedVec<U,T>) -> Self::Output {
        self * &rhs
    }
}
impl<'a,U,T> Div<&'a SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Div<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Div<Output=T> + Div<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: &'a SerializedVec<U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l / r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Div<SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Div<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Div<Output=T> + Div<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: SerializedVec<U,T>) -> Self::Output {
        self / &rhs
    }
}
impl<'a,U,T> Add<Broadcast<T>> for &'a SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Add<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Add<Output=T> + Add<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l + r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Add<Broadcast<T>> for SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Add<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Add<Output=T> + Add<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: Broadcast<T>) -> Self::Output {
        &self + rhs
    }
}
impl<'a,U,T> Sub<Broadcast<T>> for &'a SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Sub<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Sub<Output=T> + Sub<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l - r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Sub<Broadcast<T>> for SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Sub<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Sub<Output=T> + Sub<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: Broadcast<T>) -> Self::Output {
        &self - rhs
    }
}
impl<'a,U,T> Mul<Broadcast<T>> for &'a SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Mul<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Mul<Output=T> + Mul<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l * r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Mul<Broadcast<T>> for SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Mul<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Mul<Output=T> + Mul<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: Broadcast<T>) -> Self::Output {
        &self * rhs
    }
}
impl<'a,U,T> Div<Broadcast<T>> for &'a SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Div<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Div<Output=T> + Div<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l / r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Div<Broadcast<T>> for SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Div<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Div<Output=T> + Div<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: Broadcast<T>) -> Self::Output {
        &self / rhs
    }
}
impl<'a,U,T> Add<&'a SerializedVecView<'a,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Add<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Add<Output=T> + Add<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: &'a SerializedVecView<'a,U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l + r
        }).collect::<Vec<T>>().into()
    }
}
impl<'a,U,T> Add<SerializedVecView<'a,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Add<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Add<Output=T> + Add<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: SerializedVecView<'a,U,T>) -> Self::Output {
        self + &rhs
    }
}
impl<'a,U,T> Sub<&'a SerializedVecView<'a,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Sub<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Sub<Output=T> + Sub<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: &'a SerializedVecView<'a,U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l - r
        }).collect::<Vec<T>>().into()
    }
}
impl<'a,U,T> Sub<SerializedVecView<'a,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Sub<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Sub<Output=T> + Sub<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: SerializedVecView<'a,U,T>) -> Self::Output {
        self - &rhs
    }
}
impl<'a,U,T> Mul<&'a SerializedVecView<'a,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Mul<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Mul<Output=T> + Mul<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: &'a SerializedVecView<'a,U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l * r
        }).collect::<Vec<T>>().into()
    }
}
impl<'a,U,T> Mul<SerializedVecView<'a,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Mul<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Mul<Output=T> + Mul<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: SerializedVecView<'a,U,T>) -> Self::Output {
        self * &rhs
    }
}
impl<'a,U,T> Div<&'a SerializedVecView<'a,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Div<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Div<Output=T> + Div<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>>{
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: &'a SerializedVecView<'a,U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l / r
        }).collect::<Vec<T>>().into()
    }
}
impl<'a,U,T> Div<SerializedVecView<'a,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Div<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Div<Output=T> + Div<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: SerializedVecView<'a,U,T>) -> Self::Output {
        self / &rhs
    }
}
impl<'a,U,T> Add<Broadcast<T>> for &'a SerializedVecView<'a,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Add<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Add<Output=T> + Add<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l + r
        }).collect::<Vec<T>>().into()
    }
}
impl<'b,U,T> Add<Broadcast<T>> for SerializedVecView<'b,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Add<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Add<Output=T> + Add<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: Broadcast<T>) -> Self::Output {
        &self + rhs
    }
}
impl<'a,U,T> Sub<Broadcast<T>> for &'a SerializedVecView<'a,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Sub<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Sub<Output=T> + Sub<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l - r
        }).collect::<Vec<T>>().into()
    }
}
impl<'b,U,T> Sub<Broadcast<T>> for SerializedVecView<'b,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Sub<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Sub<Output=T> + Sub<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: Broadcast<T>) -> Self::Output {
        &self - rhs
    }
}
impl<'a,U,T> Mul<Broadcast<T>> for &'a SerializedVecView<'a,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Mul<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Mul<Output=T> + Mul<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l * r
        }).collect::<Vec<T>>().into()
    }
}
impl<'a,U,T> Mul<Broadcast<T>> for SerializedVecView<'a,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Mul<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Mul<Output=T> + Mul<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: Broadcast<T>) -> Self::Output {
        &self * rhs
    }
}
impl<'a,U,T> Div<Broadcast<T>> for &'a SerializedVecView<'a,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Div<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Div<Output=T> + Div<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l / r
        }).collect::<Vec<T>>().into()
    }
}
impl<'a,U,T> Div<Broadcast<T>> for SerializedVecView<'a,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                     Div<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
          for<'data> <T as AsView<'data>>::ViewType: Send + Div<Output=T> + Div<T,Output=T>,
          SerializedVec<U,T>: From<Vec<T>> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: Broadcast<T>) -> Self::Output {
        &self / rhs
    }
}
