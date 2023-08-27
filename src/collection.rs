//! Definition and implementation of various collection typess
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
use rayon::prelude::{ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use crate::arr::{AsView, MakeView, MakeViewMut, SerializedVec, SerializedVecView, SliceSize};
use crate::mem::AsRawSlice;

/// A type that expresses a broadcast of the inner type when computing with the inner type and its type collection
#[derive(Debug,Clone)]
pub struct Broadcast<T>(pub T) where T: Clone;
impl<'b,U,T> Add<&'b SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
                  AsRawSlice<U> + Send + Clone + Add<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: &'b SerializedVec<U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l + r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Add<SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
                  AsRawSlice<U> + Send + Clone + Add<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: SerializedVec<U,T>) -> Self::Output {
        self + &rhs
    }
}
impl<'b,U,T> Sub<&'b SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
                  AsRawSlice<U> + Send + Clone + Sub<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: &'b SerializedVec<U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l - r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Sub<SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
                  AsRawSlice<U> + Send + Clone + Sub<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: SerializedVec<U,T>) -> Self::Output {
        self - &rhs
    }
}
impl<'b,U,T> Mul<&'b SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Mul<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: &'b SerializedVec<U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l * r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Mul<SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Mul<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: SerializedVec<U,T>) -> Self::Output {
        self * &rhs
    }
}
impl<'b,U,T> Div<&'b SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Div<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: &'b SerializedVec<U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l / r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Div<SerializedVec<U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Div<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: SerializedVec<U,T>) -> Self::Output {
        self / &rhs
    }
}
impl<'a,U,T> Add<Broadcast<T>> for &'a SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'b> T: SliceSize + MakeView<'b,U> + MakeViewMut<'b,U> + AsRawSlice<U> + Send + Clone,
          <T as AsView<'a>>::ViewType: Add<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l + r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Add<Broadcast<T>> for SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone,
          for<'a> <T as AsView<'a>>::ViewType: Add<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: Broadcast<T>) -> Self::Output {
        &self + rhs
    }
}
impl<'a,U,T> Sub<Broadcast<T>> for &'a SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'b> T: SliceSize + MakeView<'b,U> + MakeViewMut<'b,U> + AsRawSlice<U> + Send + Clone,
          <T as AsView<'a>>::ViewType: Sub<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l - r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Sub<Broadcast<T>> for SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone,
          for<'a> <T as AsView<'a>>::ViewType: Sub<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: Broadcast<T>) -> Self::Output {
        &self - rhs
    }
}
impl<'a,U,T> Mul<Broadcast<T>> for &'a SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'b> T: SliceSize + MakeView<'b,U> + MakeViewMut<'b,U> + AsRawSlice<U> + Send + Clone,
          <T as AsView<'a>>::ViewType: Mul<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l * r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Mul<Broadcast<T>> for SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone,
          for<'a> <T as AsView<'a>>::ViewType: Mul<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: Broadcast<T>) -> Self::Output {
        &self * rhs
    }
}
impl<'a,U,T> Div<Broadcast<T>> for &'a SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'b> T: SliceSize + MakeView<'b,U> + MakeViewMut<'b,U> + AsRawSlice<U> + Send + Clone,
          <T as AsView<'a>>::ViewType: Div<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l / r
        }).collect::<Vec<T>>().into()
    }
}
impl<U,T> Div<Broadcast<T>> for SerializedVec<U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone,
          for<'a> <T as AsView<'a>>::ViewType: Div<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: Broadcast<T>) -> Self::Output {
        &self / rhs
    }
}
impl<'b,U,T> Add<&'b SerializedVecView<'b,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Add<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: &'b SerializedVecView<'b,U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l + r
        }).collect::<Vec<T>>().into()
    }
}
impl<'b,U,T> Add<SerializedVecView<'b,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Add<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: SerializedVecView<'b,U,T>) -> Self::Output {
        self + &rhs
    }
}
impl<'b,U,T> Sub<&'b SerializedVecView<'b,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Sub<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: &'b SerializedVecView<'b,U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l - r
        }).collect::<Vec<T>>().into()
    }
}
impl<'b,U,T> Sub<SerializedVecView<'b,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Sub<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: SerializedVecView<'b,U,T>) -> Self::Output {
        self - &rhs
    }
}
impl<'b,U,T> Mul<&'b SerializedVecView<'b,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Mul<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: &'b SerializedVecView<'b,U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l * r
        }).collect::<Vec<T>>().into()
    }
}
impl<'b,U,T> Mul<SerializedVecView<'b,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Mul<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: SerializedVecView<'b,U,T>) -> Self::Output {
        self * &rhs
    }
}
impl<'b,U,T> Div<&'b SerializedVecView<'b,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Div<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: &'b SerializedVecView<'b,U,T>) -> Self::Output {
        rayon::iter::repeat(self.0).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
            l / r
        }).collect::<Vec<T>>().into()
    }
}
impl<'b,U,T> Div<SerializedVecView<'b,U,T>> for Broadcast<T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone + Div<<T as AsView<'a>>::ViewType,Output=T> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: SerializedVecView<'b,U,T>) -> Self::Output {
        self / &rhs
    }
}
impl<'a,U,T> Add<Broadcast<T>> for &'a SerializedVecView<'a,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'b> T: SliceSize + MakeView<'b,U> + MakeViewMut<'b,U> + AsRawSlice<U> + Send + Clone,
          <T as AsView<'a>>::ViewType: Add<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l + r
        }).collect::<Vec<T>>().into()
    }
}
impl<'b,U,T> Add<Broadcast<T>> for SerializedVecView<'b,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone,
          for<'a> <T as AsView<'a>>::ViewType: Add<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn add(self, rhs: Broadcast<T>) -> Self::Output {
        &self + rhs
    }
}
impl<'a,U,T> Sub<Broadcast<T>> for &'a SerializedVecView<'a,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'b> T: SliceSize + MakeView<'b,U> + MakeViewMut<'b,U> + AsRawSlice<U> + Send + Clone,
          <T as AsView<'a>>::ViewType: Sub<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l - r
        }).collect::<Vec<T>>().into()
    }
}
impl<'b,U,T> Sub<Broadcast<T>> for SerializedVecView<'b,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone,
          for<'a> <T as AsView<'a>>::ViewType: Sub<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn sub(self, rhs: Broadcast<T>) -> Self::Output {
        &self - rhs
    }
}
impl<'a,U,T> Mul<Broadcast<T>> for &'a SerializedVecView<'a,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'b> T: SliceSize + MakeView<'b,U> + MakeViewMut<'b,U> + AsRawSlice<U> + Send + Clone,
          <T as AsView<'a>>::ViewType: Mul<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l * r
        }).collect::<Vec<T>>().into()
    }
}
impl<'b,U,T> Mul<Broadcast<T>> for SerializedVecView<'b,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone,
          for<'a> <T as AsView<'a>>::ViewType: Mul<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn mul(self, rhs: Broadcast<T>) -> Self::Output {
        &self * rhs
    }
}
impl<'a,U,T> Div<Broadcast<T>> for &'a SerializedVecView<'a,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'b> T: SliceSize + MakeView<'b,U> + MakeViewMut<'b,U> + AsRawSlice<U> + Send + Clone,
          <T as AsView<'a>>::ViewType: Div<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: Broadcast<T>) -> Self::Output {
        self.par_iter().zip(rayon::iter::repeat(rhs.0).take(self.len())).map(|(l,r)| {
            l / r
        }).collect::<Vec<T>>().into()
    }
}
impl<'b,U,T> Div<Broadcast<T>> for SerializedVecView<'b,U,T>
    where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
          for<'a> T: SliceSize + MakeView<'a,U> + MakeViewMut<'a,U> +
          AsRawSlice<U> + Send + Clone,
          for<'a> <T as AsView<'a>>::ViewType: Div<T,Output=T> {
    type Output = SerializedVec<U,T>;

    fn div(self, rhs: Broadcast<T>) -> Self::Output {
        &self / rhs
    }
}
