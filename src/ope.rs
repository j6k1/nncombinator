use std::ops::{Add, Mul, Sub, Div, AddAssign, Neg};
use std::fmt::Debug;
use num_traits::FromPrimitive;

pub trait UnitValue<T>: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> +
    AddAssign + PartialOrd +
    Clone + Copy + Default + Debug + Send + Sync + 'static +
    Exp + Tanh + Ln + One + Max + Min + MaxValue + InitialMaxValue + Abs + Sqrt +
    Infinity + Neginfinity + IsNaN +
    Bias + FromPrimitive {
}
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
pub trait MaxValue {
    fn max_value() -> Self;
}
impl MaxValue for f64 {
    #[inline]
    fn max_value() -> f64 {
        f64::MAX
    }
}
impl MaxValue for f32 {
    #[inline]
    fn max_value() -> f32 {
        f32::MAX
    }
}
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
pub trait Exp {
    fn exp(&self) -> Self;
}
impl Exp for f64 {
    #[inline]
    fn exp(&self) -> f64 {
        (*self).exp()
    }
}
impl Exp for f32 {
    #[inline]
    fn exp(&self) -> f32 {
        (*self).exp()
    }
}
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