//! Implementation of Type Conversion
pub trait Assume<T> {
    fn assume(self) -> T;
}
impl Assume<f32> for f64 {
    #[inline]
    fn assume(self) -> f32 {
        self as f32
    }
}
impl Assume<i8> for f64 {
    #[inline]
    fn assume(self) -> i8 {
        self as i8
    }
}
impl Assume<i16> for f64 {
    #[inline]
    fn assume(self) -> i16 {
        self as i16
    }
}
impl Assume<f64> for f32 {
    #[inline]
    fn assume(self) -> f64 {
        self as f64
    }
}
impl Assume<i16> for f32 {
    #[inline]
    fn assume(self) -> i16 {
        self as i16
    }
}
impl Assume<i8> for f32 {
    #[inline]
    fn assume(self) -> i8 {
        self as i8
    }
}
impl Assume<f64> for i8 {
    #[inline]
    fn assume(self) -> f64 {
        self as f64
    }
}
impl Assume<f32> for i8 {
    #[inline]
    fn assume(self) -> f32 {
        self as f32
    }
}
impl Assume<i16> for i8 {
    #[inline]
    fn assume(self) -> i16 {
        self as i16
    }
}
impl Assume<i32> for i8 {
    #[inline]
    fn assume(self) -> i32 {
        self as i32
    }
}
impl Assume<f32> for i16 {
    #[inline]
    fn assume(self) -> f32 {
        self as f32
    }
}
impl Assume<f64> for i16 {
    #[inline]
    fn assume(self) -> f64 {
        self as f64
    }
}
impl Assume<i32> for i16 {
    #[inline]
    fn assume(self) -> i32 {
        self as i32
    }
}
impl Assume<i8> for i16 {
    #[inline]
    fn assume(self) -> i8 {
        self as i8
    }
}
impl Assume<i16> for i32 {
    #[inline]
    fn assume(self) -> i16 {
        self as i16
    }
}