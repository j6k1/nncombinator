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
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone + Send + Sync,
                  for<'data> &'data T: Add<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            #[inline]
            fn add(self, rhs: $rt) -> Self::Output {
                rhs.iter().map(|r| {
                    &self.0 + r
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Sub<$rt> for Broadcast<T>
            where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone + Send + Sync,
                  for<'data> &'data T: Sub<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            #[inline]
            fn sub(self, rhs: $rt) -> Self::Output {
                rhs.iter().map(|r| {
                    &self.0 - r
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Mul<$rt> for Broadcast<T>
            where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone + Send + Sync,
                  for<'data> &'data T: Mul<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            #[inline]
            fn mul(self, rhs: $rt) -> Self::Output {
                rhs.iter().map(|r| {
                    &self.0 * r
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Div<$rt> for Broadcast<T>
            where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone + Send + Sync,
                  for<'data> &'data T: Div<<T as AsView<'data>>::ViewType,Output=T> + Send + Sync,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            #[inline]
            fn div(self, rhs: $rt) -> Self::Output {
                rhs.iter().map(|r| {
                    &self.0 / r
                }).collect::<Vec<T>>().into()
            }
        }
    };
    ( $lt:ty > Broadcast<T> = $ot:ty) => {
        impl<'a,U,T> Add<Broadcast<T>> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Add<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Add<&'data T,Output=T> + Add<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;
        
            #[inline]
            fn add(self, rhs: Broadcast<T>) -> Self::Output {
                self.iter().map(|l| {
                    l + &rhs.0
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Sub<Broadcast<T>> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Sub<&'data T,Output=T> + Add<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            #[inline]
            fn sub(self, rhs: Broadcast<T>) -> Self::Output {
                self.iter().map(|l| {
                    l - &rhs.0
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Mul<Broadcast<T>> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Mul<&'data T,Output=T> + Add<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            #[inline]
            fn mul(self, rhs: Broadcast<T>) -> Self::Output {
                self.iter().map(|l| {
                    l * &rhs.0
                }).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Div<Broadcast<T>> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
                  for<'data> T: SliceSize + MakeView<'data,U> + Clone + Send + Sync,
                  for<'data> <T as AsView<'data>>::ViewType: Send + Div<&'data T,Output=T> + Add<T,Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            #[inline]
            fn div(self, rhs: Broadcast<T>) -> Self::Output {
                self.iter().map(|l| {
                    l / &rhs.0
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

            #[inline]
            fn add(self, rhs: $rt) -> Self::Output {
                self.iter().zip(rhs.iter()).map(|(l,r)| l + r).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Sub<$rt> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Sub<Output=U>,
                  for<'b> T: SliceSize + MakeView<'b,U> + Send + Sync,
                  for<'b> <T as AsView<'b>>::ViewType: Send + Sub<Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            #[inline]
            fn sub(self, rhs: $rt) -> Self::Output {
                self.iter().zip(rhs.iter()).map(|(l,r)| l - r).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Mul<$rt> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Mul<Output=U>,
                  for<'b> T: SliceSize + MakeView<'b,U> + Send + Sync,
                  for<'b> <T as AsView<'b>>::ViewType: Send + Mul<Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            #[inline]
            fn mul(self, rhs: $rt) -> Self::Output {
                self.iter().zip(rhs.iter()).map(|(l,r)| l * r).collect::<Vec<T>>().into()
            }
        }

        impl<'a,U,T> Div<$rt> for $lt
            where U: Send + Sync + Default + Clone + Copy + 'static + Div<Output=U>,
                  for<'b> T: SliceSize + MakeView<'b,U> + Send + Sync,
                  for<'b> <T as AsView<'b>>::ViewType: Send + Div<Output=T>,
                  $ot: From<Vec<T>> {
            type Output = $ot;

            #[inline]
            fn div(self, rhs: $rt) -> Self::Output {
                self.iter().zip(rhs.iter()).map(|(l,r)| l / r).collect::<Vec<T>>().into()
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

            #[inline]
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

            #[inline]
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

            #[inline]
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

            #[inline]
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

            #[inline]
            fn add(self, rhs: $rt) -> Self::Output {
                self.iter().zip(rhs.iter()).map(|(&l,&r)| l + r)
                    .collect::<Vec<T>>().try_into().expect("An error occurred in the add of Arr and Arr.")
            }
        }

        impl<'a,T,const N:usize> Sub<$rt> for $lt
            where T: Sub<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
            type Output = $ot;

            #[inline]
            fn sub(self, rhs: $rt) -> Self::Output {
                self.iter().zip(rhs.iter()).map(|(&l,&r)| l - r)
                    .collect::<Vec<T>>().try_into().expect("An error occurred in the sub of Arr and Arr.")
            }
        }

        impl<'a,T,const N:usize> Mul<$rt> for $lt
            where T: Mul<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
            type Output = $ot;

            #[inline]
            fn mul(self, rhs: $rt) -> Self::Output {
                self.iter().zip(rhs.iter()).map(|(&l,&r)| l * r)
                    .collect::<Vec<T>>().try_into().expect("An error occurred in the mul of Arr and Arr.")
            }
        }

        impl<'a,T,const N:usize> Div<$rt> for $lt
            where T: Div<Output=T> + Clone + Copy + Default + Send + Sync + 'static {
            type Output = $ot;

            #[inline]
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

            #[inline]
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

            #[inline]
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

            #[inline]
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

            #[inline]
            fn div(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) / <&$crt>::from(&rhs)
            }
        }
    }
}
pub const FASTEXP_C1:f32 = 0.69314697759916432673321651236619800329208374023437;
pub const FASTEXP_C2:f32 = 0.24022242085378028852993281816452508792281150817871;
pub const FASTEXP_C3:f32 = 5.5507337432541360711102385039339424110949039459229e-2;
pub const FASTEXP_C4:f32 = 9.6715126395259202324306002651610469911247491836548e-3;
pub const FASTEXP_C5:f32 = 1.326472719636653634089906717008489067666232585907e-3;
pub const FASTEXP_C1_F64: f64 = 1.000000000000000000000000000000000000;
pub const FASTEXP_C2_F64: f64 = 0.499999999999999999999999999999999999;
pub const FASTEXP_C3_F64: f64 = 0.166666666666666666666666666666666666;
pub const FASTEXP_C4_F64: f64 = 0.041666666666666666666666666666666666;
pub const FASTEXP_C5_F64: f64 = 0.008333333333333333333333333333333333;
pub const FASTEXP_C6_F64: f64 = 0.001388888888888888888888888888888888;
pub const INV_LN2:f64 = 1. / std::f64::consts::LN_2;
#[macro_export]
macro_rules! fast_exp_f64 {
    ($x:expr) => {{
        let x = $x;
        let y = x * std::f64::consts::LOG2_E;

        let k = (x * INV_LN2).floor();

        let r = x - k * std::f64::consts::LN_2;
        let r2 = r * r;
        let r3 = r * r2;
        let r4 = r2 * r2;
        let r5 = r * r4;
        let r6 = r3 * r3;

        let w = 1. + FASTEXP_C1_F64 * r +
                     FASTEXP_C2_F64 * r2 +
                     FASTEXP_C3_F64 * r3 +
                     FASTEXP_C4_F64 * r4 +
                     FASTEXP_C5_F64 * r5 +
                     FASTEXP_C6_F64 * r6;

        let z = f64::from_bits(((k as i64 + 1023) as u64) << 52);

        z * w
    }};
}
#[macro_export]
macro_rules! fast_exp_f32 {
    ($x:expr) => {{
        let y = $x * std::f32::consts::LOG2_E;
        let n = (y + 12582912.0) as i32 - 12582912;
        let a = y - n as f32;
        let a2 = a * a;
        let a4 = a2 * a2;

        let p01 = FASTEXP_C1 + FASTEXP_C2 * a;
        let p23 = FASTEXP_C3 + FASTEXP_C4 * a;
        let p45 = FASTEXP_C5;

        let w = 1. + a * p01 + a2 * p23 + a4 * p45;

        let z = f32::from_bits(((n + 127) as u32) << 23);

        z * w
    }}
}
