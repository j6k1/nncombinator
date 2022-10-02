//! Definition and implementation of optimizers to be used during training

use std::collections::HashMap;
use crate::UnitValue;

/// Optimizer Definition
pub trait Optimizer<U> where U: Clone + Copy + UnitValue<U> {
    /// Update Weights
    /// # Arguments
    /// * `e` - error
    /// * `w` - weight
    fn update(&mut self,e:U,w:&mut U);
}
/// SGD Implementation
pub struct SGD<U> where U: UnitValue<U> {
    /// Learning rate
    a:U,
    lambda:U
}
impl<U> SGD<U> where U: UnitValue<U> {
    /// Create an instance of SGD
    /// # Arguments
    /// * `a` - Learning rate
    pub fn new(a:U) -> SGD<U> {
        SGD {
            a:a,
            lambda:U::default()
        }
    }
    pub fn with_lambda(a:U,lambda:U) -> SGD<U> {
        SGD {
            a:a,
            lambda:lambda,
        }
    }
}
impl<U> Optimizer<U> for SGD<U> where U: UnitValue<U> {
    #[inline]
    fn update(&mut self, e: U, w: &mut U) {
        let a = self.a;
        let lambda = self.lambda;
        *w = *w - a * (e + lambda * *w);
    }
}
/// MomentumSGD Implementation
pub struct MomentumSGD<U> where U: UnitValue<U> {
    a:U,
    mu:U,
    lambda:U,
    vt:HashMap<*const U,U>
}
impl<U> MomentumSGD<U> where U: UnitValue<U> {
    /// Create an instance of MomentumSGD
    /// # Arguments
    /// * `a` - Learning rate
    pub fn new(a:U) -> MomentumSGD<U> {
        MomentumSGD {
            a:a,
            mu:U::from_f64(0.9).expect("Error in type conversion from f64."),
            lambda:U::default(),
            vt:HashMap::new()
        }
    }
    /// Create an instance of MomentumSGD with additional parameters other than the default values
    /// # Arguments
    /// * `a` - Learning rate
    /// * `mu` - mu
    /// * `lambda` - lambda
    ///
    /// note: See the mu and lambda sections of the MomentumSGD algorithm formula.
    pub fn with_params(a:U,mu:U,lambda:U) -> MomentumSGD<U> {
        MomentumSGD {
            a:a,
            mu:mu,
            lambda:lambda,
            vt:HashMap::new()
        }
    }
}
impl<U> Optimizer<U> for MomentumSGD<U> where U: UnitValue<U> {
    #[inline]
    fn update(&mut self, e: U, w: &mut U) {
        let vt = self.vt.entry(w as * const U).or_insert(U::default());

        let a = self.a;
        let mu = self.mu;

        let lambda = self.lambda;
        *vt = mu * *vt - (U::one() - mu) * a * (e + lambda * *w);
        *w = *w + *vt;
    }
}
/// Adagrad Implementation
pub struct Adagrad<U> where U: UnitValue<U> {
    a:U,
    gt:HashMap<*const U,U>,
    eps:U
}
impl<U> Adagrad<U> where U: UnitValue<U> {
    /// Create an instance of Adagrad
    pub fn new() -> Adagrad<U> {
        Adagrad {
            a:U::from_f64(0.01).expect("Error in type conversion from f64."),
            gt:HashMap::new(),
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U> for Adagrad<U> where U: UnitValue<U> {
    #[inline]
    fn update(&mut self, e: U, w: &mut U) {
        let gt = self.gt.entry(w as *const U).or_insert(U::default());

        let a = self.a;

        *gt += e * e;
        *w = *w - a * e / (gt.sqrt() + self.eps);
    }
}
/// Adagrad Implementation
pub struct RMSprop<U> where U: UnitValue<U> {
    a:U,
    mu:U,
    gt:HashMap<*const U,U>,
    eps:U
}
impl<U> RMSprop<U> where U: UnitValue<U> {
    /// Create an instance of RMSprop
    pub fn new() -> RMSprop<U> {
        RMSprop {
            a:U::from_f64(0.0001f64).expect("Error in type conversion from f64."),
            mu:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            gt:HashMap::new(),
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U> for RMSprop<U> where U: UnitValue<U> {
    #[inline]
    fn update(&mut self, e: U, w: &mut U) {
        let gt = self.gt.entry(w as *const U).or_insert(U::default());

        let a = self.a;
        let mu = self.mu;

        *gt = mu * *gt + (U::one() - mu) * e * e;
        *w = *w - a * e / (gt.sqrt() + self.eps);
    }
}
/// Adam Implementation
pub struct Adam<U> where U: UnitValue<U> {
    a:U,
    mt:HashMap<* const U,U>,
    vt:HashMap<* const U,U>,
    b1:U,
    b2:U,
    b1t:U,
    b2t:U,
    eps:U
}
impl<U> Adam<U> where U: UnitValue<U> {
    /// Create an instance of Adam
    pub fn new() -> Adam<U> {
        Adam {
            a:U::from_f64(0.001f64).expect("Error in type conversion from f64."),
            mt:HashMap::new(),
            vt:HashMap::new(),
            b1:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            b2:U::from_f64(0.999f64).expect("Error in type conversion from f64."),
            b1t:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            b2t:U::from_f64(0.999f64).expect("Error in type conversion from f64."),
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U> for Adam<U> where U: UnitValue<U> {
    #[inline]
    fn update(&mut self, e: U, w: &mut U) {
        let mt = self.mt.entry(w as *const U).or_insert(U::default());
        let vt = self.vt.entry(w as *const U).or_insert(U::default());

        let a = self.a;
        let b1 = self.b1;
        let b2 = self.b2;
        let b1t = self.b1t;
        let b2t = self.b2t;

        *mt = b1 * *mt + (U::one() - self.b1) * e;
        *vt = b2 * *vt + (U::one() - self.b2) * e * e;

        *w = *w - a * (*mt / (U::one() - b1t)) / ((*vt / (U::one() - b2t)) + self.eps).sqrt();

        self.b1t = b1t * b1;
        self.b2t = b2t * b2;
    }
}
