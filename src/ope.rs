//! Definition and implementation of various operations on Units

use std::ops::{Add, Mul, Sub, Div, AddAssign, Neg};
use std::fmt::Debug;
use num_traits::FromPrimitive;

/// Trait to consolidate traits that define various operations in the unit
pub trait UnitValue<T>: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> +
    AddAssign + PartialOrd +
    Clone + Copy + Default + Debug + Send + Sync + 'static +
    Exp + Tanh + Cos + Ln + One + Max + Min + MaxValue + InitialMaxValue + Abs + Sqrt +
    Infinity + Neginfinity + IsNaN +
    Bias + FromPrimitive {
}
/// Trait to calculate bias
pub trait Bias where Self: Sized {
    fn bias() -> Self;
}
impl Bias for f64 {
    #[inline]
    fn bias() -> f64 {
        1f64
    }
}
impl Bias for f32 {
    #[inline]
    fn bias() -> f32 {
        1f32
    }
}
/// Trait to calculate max
pub trait Max {
    fn max(&self,other:&Self) -> Self;
}
impl Max for f64 {
    #[inline]
    fn max(&self,other:&f64) -> f64 {
        (*self).max(*other)
    }
}
impl Max for f32 {
    #[inline]
    fn max(&self,other:&f32) -> f32 {
        (*self).max(*other)
    }
}
/// Trait to calculate min
pub trait Min {
    fn min(&self,other:&Self) -> Self;
}
impl Min for f64 {
    #[inline]
    fn min(&self,other:&f64) -> f64 {
        (*self).min(*other)
    }
}
impl Min for f32 {
    #[inline]
    fn min(&self,other:&f32) -> f32 {
        (*self).min(*other)
    }
}
/// Trait to calculate the maximum value defined for the type
pub trait MaxValue {
    fn max_value() -> Self;
}
impl MaxValue for f64 {
    #[inline]
    fn max_value() -> f64 {
        f64::MAX
    }
}
/// Trait to calculate the minimum value defined for the type
impl MaxValue for f32 {
    #[inline]
    fn max_value() -> f32 {
        f32::MAX
    }
}
/// Trait to calculate the initial value defined for the type
pub trait InitialMaxValue {
    fn initial_max_value() -> Self;
}
impl InitialMaxValue for f64 {
    fn initial_max_value() -> f64 {
        0.0/0.0
    }
}
impl InitialMaxValue for f32 {
    fn initial_max_value() -> f32 {
        0.0/0.0
    }
}
/// Trait that returns 1 of that type
pub trait One {
    fn one() -> Self;
}
impl One for f64 {
    #[inline]
    fn one() -> f64 {
        1f64
    }
}
impl One for f32 {
    #[inline]
    fn one() -> f32 {
        1f32
    }
}
/// Trait that returns the result of applying exp
pub trait Exp {
    fn exp(&self) -> Self;
}
const FASTEXP_C1:f32 = 0.69314697759916432673321651236619800329208374023437;
const FASTEXP_C2:f32 = 0.24022242085378028852993281816452508792281150817871;
const FASTEXP_C3:f32 = 5.5507337432541360711102385039339424110949039459229e-2;
const FASTEXP_C4:f32 = 9.6715126395259202324306002651610469911247491836548e-3;
const FASTEXP_C5:f32 = 1.326472719636653634089906717008489067666232585907e-3;
const FASTEXP_C1_F64: f64 = 1.000000000000000000000000000000000000;
const FASTEXP_C2_F64: f64 = 0.499999999999999999999999999999999999;
const FASTEXP_C3_F64: f64 = 0.166666666666666666666666666666666666;
const FASTEXP_C4_F64: f64 = 0.041666666666666666666666666666666666;
const FASTEXP_C5_F64: f64 = 0.008333333333333333333333333333333333;
const FASTEXP_C6_F64: f64 = 0.001388888888888888888888888888888888;
const INV_LN2:f64 = 1. / std::f64::consts::LN_2;

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
impl Exp for f64 {
    #[inline(always)]
    fn exp(&self) -> f64 {
        (*self).exp()
    }
}
impl Exp for f32 {
    #[inline(always)]
    fn exp(&self) -> f32 {
        (*self).exp()
    }
}
/// Trait that returns the result of applying tanh
pub trait Tanh {
    fn tanh(&self) -> Self;
}
impl Tanh for f64 {
    #[inline]
    fn tanh(&self) -> f64 {
        (*self).tanh()
    }
}
impl Tanh for f32 {
    #[inline]
    fn tanh(&self) -> f32 {
        (*self).tanh()
    }
}
/// Trait that returns the result of applying cosine
pub trait Cos {
    fn cos(&self) -> Self;
}
impl Cos for f64 {
    #[inline]
    fn cos(&self) -> f64 {
        (*self).cos()
    }
}
impl Cos for f32 {
    #[inline]
    fn cos(&self) -> f32 {
        (*self).cos()
    }
}
/// Trait to return absolute value
pub trait Abs {
    fn abs(&self) -> Self;
}
impl Abs for f64 {
    #[inline]
    fn abs(&self) -> f64 {
        (*self).abs()
    }
}
impl Abs for f32 {
    #[inline]
    fn abs(&self) -> f32 {
        (*self).abs()
    }
}
/// Trait that returns the result of applying sqrt
pub trait Sqrt {
    fn sqrt(&self) -> Self;
}
impl Sqrt for f64 {
    #[inline]
    fn sqrt(&self) -> f64 {
        (*self).sqrt()
    }
}
impl Sqrt for f32 {
    #[inline]
    fn sqrt(&self) -> f32 {
        (*self).sqrt()
    }
}
/// Trait that returns the natural logarithm
pub trait Ln {
    fn ln(&self) -> Self;
}
impl Ln for f64 {
    #[inline]
    fn ln(&self) -> f64 {
        (*self).ln()
    }
}
impl Ln for f32 {
    #[inline]
    fn ln(&self) -> f32 {
        (*self).ln()
    }
}
/// Trait that returns negative infinity defined on the type
pub trait Neginfinity {
    fn neg_infinity() -> Self;
}
impl Neginfinity for f64 {
    #[inline]
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }
}

impl Neginfinity for f32 {
    #[inline]
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }
}
/// Trait that returns infinity defined on the type
pub trait Infinity {
    fn infinity() -> Self;
}
impl Infinity for f64 {
    #[inline]
    fn infinity() -> f64 {
        f64::INFINITY
    }
}
impl Infinity for f32 {
    #[inline]
    fn infinity() -> f32 {
        f32::INFINITY
    }
}
/// NaN or not? Trait that returns the
pub trait IsNaN {
    fn is_nan(&self) -> bool;
}
impl IsNaN for f64 {
    #[inline]
    fn is_nan(&self) -> bool {
        (*self).is_nan()
    }
}

impl IsNaN for f32 {
    #[inline]
    fn is_nan(&self) -> bool {
        (*self).is_nan()
    }
}
impl UnitValue<f64> for f64 {}
impl UnitValue<f32> for f32 {}
/// Type that represents the sum of elements
pub trait Sum {
    type Output;

    fn sum(&self) -> Self::Output;
}
/// Trait defining the operation of the product of matrices
pub trait Product<Rhs> {
    type Output;

    fn product(self,rhs:Rhs) -> Self::Output;
}
/// Type indicating support for four arithmetic operations
pub trait Arithmetic<Rhs = Self,O = Self>: Add<Rhs,Output = O> + Sub<Rhs,Output = O> +
                                           Mul<Rhs,Output = O> + Div<Rhs,Output = O> + Neg<Output = O> {}
impl<T,Rhs,O> Arithmetic<Rhs,O> for T where T: Add<Rhs,Output = O> + Sub<Rhs,Output = O> +
                                               Mul<Rhs,Output = O> + Div<Rhs,Output = O> + Neg<Output = O> {}
