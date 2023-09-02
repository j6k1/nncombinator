#[macro_export]
macro_rules! derive_arithmetic {
    ( $lt:ty > $rt:ty = r $clt:ty > r $crt:ty = $ot:ty) => {
        impl<'a,U,T> Add<$rt> for $lt
            where for<'b> &'b $clt: Add<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt> {
            type Output = $ot;

            fn add(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) + <&$crt>::from(&rhs)
            }
        }

        impl<'a,U,T> Sub<$rt> for $lt
            where for<'b> &'b $clt: Sub<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt> {
            type Output = $ot;

            fn sub(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) - <&$crt>::from(&rhs)
            }
        }

        impl<'a,U,T> Mul<$rt> for $lt
            where for<'b> &'b $clt: Mul<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt> {
            type Output = $ot;

            fn mul(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) * <&$crt>::from(&rhs)
            }
        }

        impl<'a,U,T> Div<$rt> for $lt
            where for<'b> &'b $clt: Div<&'b $crt,Output=$ot>,
            for<'b> &'b $clt: From<&'b $lt>,
            for<'b> &'b $crt: From<&'b $rt> {
            type Output = $ot;

            fn div(self, rhs: $rt) -> Self::Output {
                <&$clt>::from(&self) / <&$crt>::from(&rhs)
            }
        }
    };
    ($lt:ty > $rt:ty = $clt:ty > $crt:ty = $ot:ty) => {
        impl<'a,U,T> Add<$rt> for $lt
            where for<'b> U: Send + 'b,
                  for<'b> T: Send + 'b,
                  for<'b> $clt: Add<$crt,Output=$ot>,
                  for<'b> $clt: FromRef<'b,$lt>,
                  for<'b> $crt: FromRef<'b,$rt> {
            type Output = $ot;

            fn add(self, rhs: $rt) -> Self::Output {
                <$clt>::from_ref(&self) + <$crt>::from_ref(&rhs)
            }
        }

        impl<'a,U,T> Sub<$rt> for $lt
            where for<'b> U: Send + 'b,
                  for<'b> T: Send + 'b,
                  for<'b> $clt: Sub<$crt,Output=$ot>,
                  for<'b> $clt: FromRef<'b,$lt>,
                  for<'b> $crt: FromRef<'b,$rt> {
            type Output = $ot;

            fn sub(self, rhs: $rt) -> Self::Output {
                <$clt>::from_ref(&self) - <$crt>::from_ref(&rhs)
            }
        }

        impl<'a,U,T> Mul<$rt> for $lt
            where for<'b> U: Send + 'b,
                  for<'b> T: Send + 'b,
                  for<'b> $clt: Mul<$crt,Output=$ot>,
                  for<'b> $clt: FromRef<'b,$lt>,
                  for<'b> $crt: FromRef<'b,$rt> {
            type Output = $ot;

            fn mul(self, rhs: $rt) -> Self::Output {
                <$clt>::from_ref(&self) * <$crt>::from_ref(&rhs)
            }
        }

        impl<'a,U,T> Div<$rt> for $lt
            where for<'b> U: Send + 'b,
                  for<'b> T: Send + 'b,
                  for<'b> $clt: Div<$crt,Output=$ot>,
                  for<'b> $clt: FromRef<'b,$lt>,
                  for<'b> $crt: FromRef<'b,$rt> {
            type Output = $ot;

            fn div(self, rhs: $rt) -> Self::Output {
                <$clt>::from_ref(&self) / <$crt>::from_ref(&rhs)
            }
        }
    }
}
