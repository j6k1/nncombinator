use std::ops::{Add, Mul, Sub, Div, AddAssign, Neg};
use std::fmt::Debug;

pub trait UnitValue<T>: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> +
    AddAssign + PartialOrd +
    Clone + Copy + Default + Debug + Send + Sync + 'static +
    Exp + Tanh + One + Max + Min + MaxValue + InitialMaxValue + Abs + Bias {
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
