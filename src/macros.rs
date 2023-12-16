/// Macros for automatic derivation of the implementation of the four arithmetic operations
///
/// # Arguments
/// * `$lt` - the left hand side type
/// * `$rt` - the right hand side type
/// * `$clt` - Converted type of the left-hand side type
/// * `$crt` - Converted type of the right-hand side type
/// * `$ot` - output type
#[macro_export]
macro_rules! derive_arithmetic {
    ( Broadcast<T> > $rt:ty = $ot:ty) => {
        impl<'a,U,T> Add<$rt> for Broadcast<T>
            where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                             Add<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Add<Output=T> + Add<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            fn add(self, rhs: $rt) -> Self::Output {
                rayon::iter::repeat(self.0.clone()).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
                    l + r
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Sub<$rt> for Broadcast<T>
            where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                             Sub<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Sub<Output=T> + Sub<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            fn sub(self, rhs: $rt) -> Self::Output {
                rayon::iter::repeat(self.0.clone()).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
                    l - r
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Mul<$rt> for Broadcast<T>
            where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                             Mul<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Mul<Output=T> + Mul<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            fn mul(self, rhs: $rt) -> Self::Output {
                rayon::iter::repeat(self.0.clone()).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
                    l * r
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Div<$rt> for Broadcast<T>
            where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                             Div<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Div<Output=T> + Div<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            fn div(self, rhs: $rt) -> Self::Output {
                rayon::iter::repeat(self.0.clone()).take(rhs.len()).zip(rhs.par_iter()).map(|(l,r)| {
                    l / r
                }).collect::<Vec<T>>().into()
            }
        }
    };
    ( $lt:ty > Broadcast<T> = $ot:ty) => {
        impl<'a,U,T> Add<Broadcast<T>> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                             Add<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Add<Output=T> + Add<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;
        
            fn add(self, rhs: Broadcast<T>) -> Self::Output {
                self.par_iter().zip(rayon::iter::repeat(rhs.0.clone()).take(self.len())).map(|(l,r)| {
                    l + r
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Sub<Broadcast<T>> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                             Sub<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Sub<Output=T> + Sub<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            fn sub(self, rhs: Broadcast<T>) -> Self::Output {
                self.par_iter().zip(rayon::iter::repeat(rhs.0.clone()).take(self.len())).map(|(l,r)| {
                    l - r
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Mul<Broadcast<T>> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                             Mul<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Mul<Output=T> + Mul<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            fn mul(self, rhs: Broadcast<T>) -> Self::Output {
                self.par_iter().zip(rayon::iter::repeat(rhs.0.clone()).take(self.len())).map(|(l,r)| {
                    l * r
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Div<Broadcast<T>> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone +
                             Div<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Div<Output=T> + Div<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            fn div(self, rhs: Broadcast<T>) -> Self::Output {
                self.par_iter().zip(rayon::iter::repeat(rhs.0.clone()).take(self.len())).map(|(l,r)| {
                    l / r
                }).collect::<Vec<T>>().into()
            }
        }
    };
    ( $lt:ty > $rt:ty = $ot:ty) => {
        impl<'a,U,T> Add<$rt> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
                  for<'b> T: SliceSize + MakeView<'b,U> + Send + Sync,
                  for<'b> <T as AsView<'b>>::ViewType: Send + Add<Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            fn add(self, rhs: $rt) -> Self::Output {
                self.par_iter().zip(rhs.par_iter()).map(|(l,r)| l + r).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Sub<$rt> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
                  for<'b> T: SliceSize + MakeView<'b,U> + Send + Sync,
                  for<'b> <T as AsView<'b>>::ViewType: Send + Sub<Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            fn sub(self, rhs: $rt) -> Self::Output {
                self.par_iter().zip(rhs.par_iter()).map(|(l,r)| l - r).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Mul<$rt> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
                  for<'b> T: SliceSize + MakeView<'b,U> + Send + Sync,
                  for<'b> <T as AsView<'b>>::ViewType: Send + Mul<Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            fn mul(self, rhs: $rt) -> Self::Output {
                self.par_iter().zip(rhs.par_iter()).map(|(l,r)| l * r).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Div<$rt> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
                  for<'b> T: SliceSize + MakeView<'b,U> + Send + Sync,
                  for<'b> <T as AsView<'b>>::ViewType: Send + Div<Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            fn div(self, rhs: $rt) -> Self::Output {
                self.par_iter().zip(rhs.par_iter()).map(|(l,r)| l / r).collect::<Vec<T>>().into()
            }
        }
    };
    ( $lt:ty > $rt:ty = r $clt:ty > r $crt:ty = $ot:ty) => {
        impl<'a,U,T> Add<$rt> for $lt
            where for<'b> &'b $clt: Add<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt>,
            T: Clone {
            type Output = $ot;

            fn add(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) + <&$crt>::from(&rhs)
            }
        }

        impl<'a,U,T> Sub<$rt> for $lt
            where for<'b> &'b $clt: Sub<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt>,
            T: Clone {
            type Output = $ot;

            fn sub(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) - <&$crt>::from(&rhs)
            }
        }

        impl<'a,U,T> Mul<$rt> for $lt
            where for<'b> &'b $clt: Mul<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt>,
            T: Clone {
            type Output = $ot;

            fn mul(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) * <&$crt>::from(&rhs)
            }
        }

        impl<'a,U,T> Div<$rt> for $lt
            where for<'b> &'b $clt: Div<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt>,
            T: Clone {
            type Output = $ot;

            fn div(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) / <&$crt>::from(&rhs)
            }
        }
    };
}
/// Macro for automatic derivation of the implementation of the four arithmetic operations of Arr,ArrView.ss
///
/// # Arguments
/// * `$lt` - the left hand side type
/// * `$rt` - the right hand side type
/// * `$clt` - Converted type of the left-hand side type
/// * `$crt` - Converted type of the right-hand side type
/// * `$ot` - output type
#[macro_export]
macro_rules! derive_arr_like_arithmetic {
    ($lt:ty > $rt:ty = $ot:ty) => {
        impl<'a,T,const N:usize> Add<$rt> for $lt
            where T: Add<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
            type Output = $ot;

            fn add(self, rhs: $rt) -> Self::Output {
                self.iter().zip(rhs.iter()).map(|(&l,&r)| l + r)
                    .collect::<Vec<T>>().try_into().expect("An error occurred in the add of Arr and Arr.")
            }
        }

        impl<'a,T,const N:usize> Sub<$rt> for $lt
            where T: Sub<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
            type Output = $ot;

            fn sub(self, rhs: $rt) -> Self::Output {
                self.iter().zip(rhs.iter()).map(|(&l,&r)| l - r)
                    .collect::<Vec<T>>().try_into().expect("An error occurred in the sub of Arr and Arr.")
            }
        }

        impl<'a,T,const N:usize> Mul<$rt> for $lt
            where T: Mul<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
            type Output = $ot;

            fn mul(self, rhs: $rt) -> Self::Output {
                self.iter().zip(rhs.iter()).map(|(&l,&r)| l * r)
                    .collect::<Vec<T>>().try_into().expect("An error occurred in the mul of Arr and Arr.")
            }
        }

        impl<'a,T,const N:usize> Div<$rt> for $lt
            where T: Div<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
            type Output = $ot;

            fn div(self, rhs: $rt) -> Self::Output {
                self.iter().zip(rhs.iter()).map(|(&l,&r)| l / r)
                    .collect::<Vec<T>>().try_into().expect("An error occurred in the sub of Arr and Arr.")
            }
        }
    };
    ( $lt:ty > $rt:ty = r $clt:ty > r $crt:ty = $ot:ty) => {
        impl<'a,T,const N:usize> Add<$rt> for $lt
            where for<'b> &'b $clt: Add<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt>,
            T: Default + Clone + Send {
            type Output = $ot;

            fn add(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) + <&$crt>::from(&rhs)
            }
        }

        impl<'a,T,const N:usize> Sub<$rt> for $lt
            where for<'b> &'b $clt: Sub<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt>,
            T: Default + Clone + Send {
            type Output = $ot;

            fn sub(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) - <&$crt>::from(&rhs)
            }
        }

        impl<'a,T,const N:usize> Mul<$rt> for $lt
            where for<'b> &'b $clt: Mul<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt>,
            T: Default + Clone + Send {
            type Output = $ot;

            fn mul(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) * <&$crt>::from(&rhs)
            }
        }

        impl<'a,T,const N:usize> Div<$rt> for $lt
            where for<'b> &'b $clt: Div<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt>,
            T: Default + Clone + Send {
            type Output = $ot;

            fn div(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) / <&$crt>::from(&rhs)
            }
        }
    }
}
