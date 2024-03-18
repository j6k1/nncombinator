//! Definition and implementation of optimizers to be used during training

use std::marker::PhantomData;
use crate::device::{Device, DeviceCpu, DeviceGpu};
use crate::UnitValue;

/// OptimizerBuilder Definition
pub trait OptimizerBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    type Output: Optimizer<U,D>;
    fn build(&self,size:usize) -> Self::Output;
}
/// Optimizer Definition
pub trait Optimizer<U,D> where U: Clone + Copy + UnitValue<U>, D: Device<U> {
    type Target: ?Sized;
    type Grad: ?Sized;
    /// Update Weights
    /// # Arguments
    /// * `e` - error
    /// * `w` - weight
    fn update(&mut self, e:&Self::Grad, w:&mut Self::Target);
}
/// SGD Implementation
pub struct SGD<U,D> where U: UnitValue<U>, D: Device<U> {
    d:PhantomData<D>,
    /// Learning rate
    a:U,
    /// Weight decay
    lambda:U
}
impl<U,D> SGD<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of SGD
    /// # Arguments
    /// * `a` - Learning rate
    pub fn new(a:U) -> SGD<U,D> {
        SGD {
            d:PhantomData::<D>,
            a:a,
            lambda:U::default()
        }
    }
    /// Create an instance of SGD
    /// # Arguments
    /// * `a` - Learning rate
    /// * `lambda` - Weight decay
    pub fn with_lambda(a:U,lambda:U) -> SGD<U,D> {
        SGD {
            d:PhantomData::<D>,
            a:a,
            lambda:lambda,
        }
    }
}
impl<U> Optimizer<U,DeviceCpu<U>> for SGD<U,DeviceCpu<U>> where U: UnitValue<U>, DeviceCpu<U>: Device<U> {
    type Target = [U];
    type Grad = [U];

    #[inline]
    fn update(&mut self, e: &[U], w: &mut [U]) {
        let a = self.a;
        let lambda = self.lambda;

        for (w,&e) in w.iter_mut().zip(e.iter()) {
            *w = *w - a * (e + lambda * *w);
        }
    }
}
impl<U> Optimizer<U,DeviceGpu<U>> for SGD<U,DeviceGpu<U>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    type Target = [U];
    type Grad = [U];

    #[inline]
    fn update(&mut self, e: &[U], w: &mut [U]) {
        let a = self.a;
        let lambda = self.lambda;

        for (w,&e) in w.iter_mut().zip(e.iter()) {
            *w = *w - a * (e + lambda * *w);
        }
    }
}
/// Implementation of a builder to generate SGD optimizers
pub struct SGDBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    d:PhantomData<D>,
    /// Learning rate
    a:U,
    /// Weight decay
    lambda:U
}
impl<U,D> SGDBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of SGD
    /// # Arguments
    /// * `a` - Learning rate
    pub fn new(a:U) -> SGDBuilder<U,D> {
        SGDBuilder {
            d:PhantomData::<D>,
            a:a,
            lambda:U::default()
        }
    }

    /// Create an instance of SGD
    /// # Arguments
    /// * `a` - Learning rate
    /// * `lambda` - Weight decay
    pub fn with_lambda(a:U,lambda:U) -> SGDBuilder<U,D> {
        SGDBuilder {
            d:PhantomData::<D>,
            a:a,
            lambda:lambda,
        }
    }
}
impl<U,D> OptimizerBuilder<U,D> for SGDBuilder<U,D> where U: UnitValue<U>, D: Device<U>, SGD<U,D>: Optimizer<U,D> {
    type Output = SGD<U,D>;

    fn build(&self,_:usize) -> Self::Output {
        SGD::with_lambda(self.a,self.lambda)
    }
}
/// MomentumSGD Implementation
pub struct MomentumSGD<U,D,T> where U: UnitValue<U>, D: Device<U> {
    d:PhantomData<D>,
    size:usize,
    a:U,
    mu:U,
    lambda:U,
    vt:T
}
impl<U> MomentumSGD<U,DeviceCpu<U>,Box<[U]>> where U: UnitValue<U> {
    /// Create an instance of MomentumSGD
    /// # Arguments
    /// * `a` - Learning rate
    pub fn new(_:&DeviceCpu<U>,size:usize,a:U) -> MomentumSGD<U,DeviceCpu<U>,Box<[U]>> {
        MomentumSGD {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            a:a,
            mu:U::from_f64(0.9).expect("Error in type conversion from f64."),
            lambda:U::default(),
            vt:vec![U::default();size].into_boxed_slice()
        }
    }
    /// Create an instance of MomentumSGD with additional parameters other than the default values
    /// # Arguments
    /// * `a` - Learning rate
    /// * `mu` - mu
    /// * `lambda` - Weight decay
    ///
    /// note: See the mu and lambda sections of the MomentumSGD algorithm formula.
    pub fn with_params(_:&DeviceCpu<U>,size:usize,a:U,mu:U,lambda:U) -> MomentumSGD<U,DeviceCpu<U>,Box<[U]>> {
        MomentumSGD {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            a:a,
            mu:mu,
            lambda:lambda,
            vt:vec![U::default();size].into_boxed_slice()
        }
    }
}
impl<U> Optimizer<U,DeviceCpu<U>> for MomentumSGD<U,DeviceCpu<U>,Box<[U]>> where U: UnitValue<U> {
    type Target = [U];
    type Grad = [U];

    #[inline]
    fn update(&mut self, e: & [U], w: &mut [U]) {
        let a = self.a;
        let mu = self.mu;

        let lambda = self.lambda;

        for ((w,&e),vt) in w.iter_mut().zip(e.iter()).zip(self.vt.iter_mut()) {
            *vt = mu * *vt - a * (e + lambda * *w);
            *w = *w + *vt;
        }
    }
}
impl<U> MomentumSGD<U,DeviceGpu<U>,Box<[U]>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    /// Create an instance of MomentumSGD
    /// # Arguments
    /// * `a` - Learning rate
    pub fn new(device:&DeviceGpu<U>,size:usize,a:U) -> MomentumSGD<U,DeviceGpu<U>,Box<[U]>> {
        MomentumSGD {
            d:PhantomData::<DeviceGpu<U>>,
            size:size,
            a:a,
            mu:U::from_f64(0.9).expect("Error in type conversion from f64."),
            lambda:U::default(),
            vt:vec![U::default();size].into_boxed_slice()
        }
    }
    /// Create an instance of MomentumSGD with additional parameters other than the default values
    /// # Arguments
    /// * `a` - Learning rate
    /// * `mu` - mu
    /// * `lambda` - Weight decay
    ///
    /// note: See the mu and lambda sections of the MomentumSGD algorithm formula.
    pub fn with_params(device:&DeviceGpu<U>,size:usize,a:U,mu:U,lambda:U) -> MomentumSGD<U,DeviceGpu<U>,Box<[U]>> {
        MomentumSGD {
            d:PhantomData::<DeviceGpu<U>>,
            size:size,
            a:a,
            mu:mu,
            lambda:lambda,
            vt:vec![U::default();size].into_boxed_slice()
        }
    }
}
impl<U> Optimizer<U,DeviceGpu<U>> for MomentumSGD<U,DeviceGpu<U>,Box<[U]>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    type Target = [U];
    type Grad = [U];

    #[inline]
    fn update(&mut self, e: & [U], w: &mut [U]) {
        let a = self.a;
        let mu = self.mu;

        let lambda = self.lambda;

        for ((w,&e),vt) in w.iter_mut().zip(e.iter()).zip(self.vt.iter_mut()) {
            *vt = mu * *vt - a * (e + lambda * *w);
            *w = *w + *vt;
        }
    }
}
/// Implementation of a builder to generate MomentumSGD optimizers
pub struct MomentumSGDBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    device:D,
    a:U,
    mu:U,
    lambda:U
}
impl<U,D> MomentumSGDBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of MomentumSGD
    /// # Arguments
    /// * `a` - Learning rate
    pub fn new(device:&D,a:U) -> MomentumSGDBuilder<U,D> {
        MomentumSGDBuilder {
            device:device.clone(),
            a:a,
            mu:U::from_f64(0.9).expect("Error in type conversion from f64."),
            lambda:U::default()
        }
    }
    /// Create an instance of MomentumSGD with additional parameters other than the default values
    /// # Arguments
    /// * `a` - Learning rate
    /// * `mu` - mu
    /// * `lambda` - Weight decay
    ///
    /// note: See the mu and lambda sections of the MomentumSGD algorithm formula.
    pub fn with_params(device:&D,a:U,mu:U,lambda:U) -> MomentumSGDBuilder<U,D> {
        MomentumSGDBuilder {
            device:device.clone(),
            a:a,
            mu:mu,
            lambda:lambda
        }
    }
}
impl<U> OptimizerBuilder<U,DeviceCpu<U>> for MomentumSGDBuilder<U,DeviceCpu<U>>
    where U: UnitValue<U>, MomentumSGD<U,DeviceCpu<U>,Box<[U]>>: Optimizer<U,DeviceCpu<U>> {
    type Output = MomentumSGD<U,DeviceCpu<U>,Box<[U]>>;

    fn build(&self,size:usize) -> Self::Output {
        MomentumSGD::<_,DeviceCpu<U>,_>::with_params(&self.device,size,self.a,self.mu,self.lambda)
    }
}
impl<U> OptimizerBuilder<U,DeviceGpu<U>> for MomentumSGDBuilder<U,DeviceGpu<U>>
    where U: UnitValue<U>, DeviceGpu<U>: Device<U>, MomentumSGD<U,DeviceGpu<U>,Box<[U]>>: Optimizer<U,DeviceGpu<U>> {
    type Output = MomentumSGD<U,DeviceGpu<U>,Box<[U]>>;

    fn build(&self,size:usize) -> Self::Output {
        MomentumSGD::<_,DeviceGpu<U>,_>::with_params(&self.device,size,self.a,self.mu,self.lambda)
    }
}
/// Adagrad Implementation
pub struct Adagrad<U,D,T> where U: UnitValue<U>, D: Device<U> {
    d:PhantomData<D>,
    size:usize,
    a:U,
    gt:T,
    eps:U
}
impl<U> Adagrad<U,DeviceCpu<U>,Box<[U]>> where U: UnitValue<U> {
    /// Create an instance of Adagrad
    pub fn new(_:&DeviceCpu<U>,size:usize) -> Adagrad<U,DeviceCpu<U>,Box<[U]>> {
        Adagrad {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            a:U::from_f64(0.01).expect("Error in type conversion from f64."),
            gt:vec![U::default();size].into_boxed_slice(),
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U,DeviceCpu<U>> for Adagrad<U,DeviceCpu<U>,Box<[U]>> where U: UnitValue<U> {
    type Target = [U];
    type Grad = [U];

    #[inline]
    fn update(&mut self, e: &[U], w: &mut [U]) {
        let a = self.a;

        for ((w,&e),gt) in w.iter_mut().zip(e.iter()).zip(self.gt.iter_mut()) {
            *gt += e * e;
            *w = *w - a * e / (gt.sqrt() + self.eps);
        }
    }
}
impl<U> Adagrad<U,DeviceGpu<U>,Box<[U]>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    /// Create an instance of Adagrad
    pub fn new(device:&DeviceGpu<U>,size:usize) -> Adagrad<U,DeviceGpu<U>,Box<[U]>> {
        Adagrad {
            d:PhantomData::<DeviceGpu<U>>,
            size:size,
            a:U::from_f64(0.01).expect("Error in type conversion from f64."),
            gt:vec![U::default();size].into_boxed_slice(),
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U,DeviceGpu<U>> for Adagrad<U,DeviceGpu<U>,Box<[U]>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    type Target = [U];
    type Grad = [U];

    #[inline]
    fn update(&mut self, e: &[U], w: &mut [U]) {
        let a = self.a;

        for ((w,&e),gt) in w.iter_mut().zip(e.iter()).zip(self.gt.iter_mut()) {
            *gt += e * e;
            *w = *w - a * e / (gt.sqrt() + self.eps);
        }
    }
}
/// Implementation of a builder to generate Adagrad optimizers
pub struct AdagradBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    device:D
}
impl<U,D> AdagradBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&D) -> AdagradBuilder<U,D> {
        AdagradBuilder {
            u:PhantomData::<U>,
            device:device.clone()
        }
    }
}
impl<U> OptimizerBuilder<U,DeviceCpu<U>> for AdagradBuilder<U,DeviceCpu<U>>
    where U: UnitValue<U>, Adagrad<U,DeviceCpu<U>,Box<[U]>>: Optimizer<U,DeviceCpu<U>> {
    type Output = Adagrad<U,DeviceCpu<U>,Box<[U]>>;

    fn build(&self,size:usize) -> Self::Output {
        Adagrad::<_,DeviceCpu<U>,_>::new(&self.device,size)
    }
}
impl<U> OptimizerBuilder<U,DeviceGpu<U>> for AdagradBuilder<U,DeviceGpu<U>>
    where U: UnitValue<U>, DeviceGpu<U>: Device<U>, Adagrad<U,DeviceGpu<U>,Box<[U]>>: Optimizer<U,DeviceGpu<U>> {
    type Output = Adagrad<U,DeviceGpu<U>,Box<[U]>>;

    fn build(&self,size:usize) -> Self::Output {
        Adagrad::<_,DeviceGpu<U>,_>::new(&self.device,size)
    }
}
/// RMSprop Implementation
pub struct RMSprop<U,D,T> where U: UnitValue<U>, D: Device<U> {
    d:PhantomData<D>,
    size:usize,
    a:U,
    mu:U,
    gt:T,
    eps:U
}
impl<U> RMSprop<U,DeviceCpu<U>,Box<[U]>> where U: UnitValue<U> {
    /// Create an instance of RMSprop
    pub fn new(_:&DeviceCpu<U>,size:usize) -> RMSprop<U,DeviceCpu<U>,Box<[U]>> {
        RMSprop {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            a:U::from_f64(0.0001f64).expect("Error in type conversion from f64."),
            mu:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            gt:vec![U::default();size].into_boxed_slice(),
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U,DeviceCpu<U>> for RMSprop<U,DeviceCpu<U>,Box<[U]>> where U: UnitValue<U> {
    type Target = [U];
    type Grad = [U];

    #[inline]
    fn update(&mut self, e: &[U], w: &mut [U]) {
        let a = self.a;
        let mu = self.mu;

        for ((w,&e),gt) in w.iter_mut().zip(e.iter()).zip(self.gt.iter_mut()) {
            *gt = mu * *gt + (U::one() - mu) * e * e;
            *w = *w - a * e / (gt.sqrt() + self.eps);
        }
    }
}
impl<U> RMSprop<U,DeviceGpu<U>,Box<[U]>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    /// Create an instance of RMSprop
    pub fn new(device:&DeviceGpu<U>,size:usize) -> RMSprop<U,DeviceGpu<U>,Box<[U]>> {
        RMSprop {
            d:PhantomData::<DeviceGpu<U>>,
            size:size,
            a:U::from_f64(0.0001f64).expect("Error in type conversion from f64."),
            mu:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            gt:vec![U::default();size].into_boxed_slice(),
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U,DeviceGpu<U>> for RMSprop<U,DeviceGpu<U>,Box<[U]>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    type Target = [U];
    type Grad = [U];

    #[inline]
    fn update(&mut self, e: &[U], w: &mut [U]) {
        let a = self.a;
        let mu = self.mu;

        for ((w,&e),gt) in w.iter_mut().zip(e.iter()).zip(self.gt.iter_mut()) {
            *gt = mu * *gt + (U::one() - mu) * e * e;
            *w = *w - a * e / (gt.sqrt() + self.eps);
        }
    }
}
/// Implementation of a builder to generate RMSprop optimizers
pub struct RMSpropBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    device:D
}
impl<U,D> RMSpropBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&D) -> RMSpropBuilder<U,D> {
        RMSpropBuilder {
            u:PhantomData::<U>,
            device:device.clone()
        }
    }
}
impl<U> OptimizerBuilder<U,DeviceCpu<U>> for RMSpropBuilder<U,DeviceCpu<U>>
    where U: UnitValue<U>, RMSprop<U,DeviceCpu<U>,Box<[U]>>: Optimizer<U,DeviceCpu<U>> {
    type Output = RMSprop<U,DeviceCpu<U>,Box<[U]>>;

    fn build(&self,size:usize) -> Self::Output {
        RMSprop::<_,DeviceCpu<U>,_>::new(&self.device,size)
    }
}
impl<U> OptimizerBuilder<U,DeviceGpu<U>> for RMSpropBuilder<U,DeviceGpu<U>>
    where U: UnitValue<U>, DeviceGpu<U>: Device<U>, RMSprop<U,DeviceGpu<U>,Box<[U]>>: Optimizer<U,DeviceGpu<U>> {
    type Output = RMSprop<U,DeviceGpu<U>,Box<[U]>>;

    fn build(&self,size:usize) -> Self::Output {
        RMSprop::<_,DeviceGpu<U>,_>::new(&self.device,size)
    }
}
/// Adam Implementation
pub struct Adam<U,D,T> where U: UnitValue<U>, D: Device<U> {
    d:PhantomData<D>,
    size:usize,
    a:U,
    mt:T,
    vt:T,
    b1:U,
    b2:U,
    b1t:U,
    b2t:U,
    eps:U
}
impl<U> Adam<U,DeviceCpu<U>,Box<[U]>> where U: UnitValue<U> {
    /// Create an instance of Adam
    pub fn new(_:&DeviceCpu<U>,size:usize) -> Adam<U,DeviceCpu<U>,Box<[U]>> {
        Adam {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            a:U::from_f64(0.001f64).expect("Error in type conversion from f64."),
            mt:vec![U::default();size].into_boxed_slice(),
            vt:vec![U::default();size].into_boxed_slice(),
            b1:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            b2:U::from_f64(0.999f64).expect("Error in type conversion from f64."),
            b1t:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            b2t:U::from_f64(0.999f64).expect("Error in type conversion from f64."),
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U,DeviceCpu<U>> for Adam<U,DeviceCpu<U>,Box<[U]>> where U: UnitValue<U> {
    type Target = [U];
    type Grad = [U];

    #[inline]
    fn update(&mut self, e: &[U], w: &mut [U]) {
        let a = self.a;
        let b1 = self.b1;
        let b2 = self.b2;
        let b1t = self.b1t;
        let b2t = self.b2t;

        for ((w,&e),(mt,vt)) in w.iter_mut().zip(e.iter()).zip(self.mt.iter_mut().zip(self.vt.iter_mut())) {
            *mt = b1 * *mt + (U::one() - self.b1) * e;
            *vt = b2 * *vt + (U::one() - self.b2) * e * e;

            *w = *w - a * (*mt / (U::one() - b1t)) / ((*vt / (U::one() - b2t)) + self.eps).sqrt();
        }

        self.b1t = b1t * b1;
        self.b2t = b2t * b2;
    }
}
impl<U> Adam<U,DeviceGpu<U>,Box<[U]>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    /// Create an instance of Adam
    pub fn new(device:&DeviceGpu<U>,size:usize) -> Adam<U,DeviceGpu<U>,Box<[U]>> {
        Adam {
            d:PhantomData::<DeviceGpu<U>>,
            size:size,
            a:U::from_f64(0.001f64).expect("Error in type conversion from f64."),
            mt:vec![U::default();size].into_boxed_slice(),
            vt:vec![U::default();size].into_boxed_slice(),
            b1:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            b2:U::from_f64(0.999f64).expect("Error in type conversion from f64."),
            b1t:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            b2t:U::from_f64(0.999f64).expect("Error in type conversion from f64."),
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U,DeviceGpu<U>> for Adam<U,DeviceGpu<U>,Box<[U]>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    type Target = [U];
    type Grad = [U];

    #[inline]
    fn update(&mut self, e: &[U], w: &mut [U]) {
        let a = self.a;
        let b1 = self.b1;
        let b2 = self.b2;
        let b1t = self.b1t;
        let b2t = self.b2t;

        for ((w,&e),(mt,vt)) in w.iter_mut().zip(e.iter()).zip(self.mt.iter_mut().zip(self.vt.iter_mut())) {
            *mt = b1 * *mt + (U::one() - self.b1) * e;
            *vt = b2 * *vt + (U::one() - self.b2) * e * e;

            *w = *w - a * (*mt / (U::one() - b1t)) / ((*vt / (U::one() - b2t)) + self.eps).sqrt();
        }

        self.b1t = b1t * b1;
        self.b2t = b2t * b2;
    }
}
/// Implementation of a builder to generate Adam optimizers
pub struct AdamBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    device:D
}
impl<U,D> AdamBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(device:&D) -> AdamBuilder<U,D> {
        AdamBuilder {
            u:PhantomData::<U>,
            device:device.clone()
        }
    }
}
impl<U> OptimizerBuilder<U,DeviceCpu<U>> for AdamBuilder<U,DeviceCpu<U>>
    where U: UnitValue<U>, Adam<U,DeviceCpu<U>,Box<[U]>>: Optimizer<U,DeviceCpu<U>> {
    type Output = Adam<U,DeviceCpu<U>,Box<[U]>>;

    fn build(&self,size:usize) -> Self::Output {
        Adam::<_,DeviceCpu<U>,_>::new(&self.device,size)
    }
}
impl<U> OptimizerBuilder<U,DeviceGpu<U>> for AdamBuilder<U,DeviceGpu<U>>
    where U: UnitValue<U>, DeviceGpu<U>: Device<U>, Adam<U,DeviceGpu<U>,Box<[U]>>: Optimizer<U,DeviceGpu<U>> {
    type Output = Adam<U,DeviceGpu<U>,Box<[U]>>;

    fn build(&self,size:usize) -> Self::Output {
        Adam::<_,DeviceGpu<U>,_>::new(&self.device,size)
    }
}
