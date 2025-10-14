//! Definition and implementation of optimizers to be used during training

use core::fmt::Debug;
use std::marker::PhantomData;
use crate::device::{Device, DeviceCpu, DeviceGpu, DeviceAllocator};
use crate::{UnitValue};
use crate::arr::ShieldSlice;
use crate::cuda::{CudaMutPtr, CudaPtr, kernel, Kernel, WriteMemory};
use crate::cuda::allocator::CudaAllocator;
use crate::cuda::kernel::optimizer::{AdagradArgs, AdamArgs, AdamWArgs, MomentumSGDArgs, RMSpropArgs, SGDArgs};
use crate::error::{OptimizerBuildError, TrainingError};

/// OptimizerBuilder Definition
pub trait OptimizerBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    type Output: Optimizer<U,D>;
    /// Create and return an optimizer
    /// # Arguments
    /// * `size` - Total number of weights to be optimized
    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError>;
}
/// Optimizer Definition
pub trait Optimizer<U,D> where U: Clone + Copy + UnitValue<U>, D: Device<U> {
    type InternalType: ?Sized;
    type InternalUpdateType<'a>: ?Sized;
    /// Update Weights
    /// # Arguments
    /// * `e` - error
    /// * `w` - weight
    fn update<'a>(&mut self, e:&'a Self::InternalType, w:Self::InternalUpdateType<'a>) -> Result<(),TrainingError>;
}
/// Optimizer State Definition
pub trait OptimizerState<U,D> where U: Clone + Copy + UnitValue<U>, D: Device<U> {
    /// State type
    type Type;
}
/// SGD Implementation
pub struct SGD<U,D> where U: UnitValue<U>, D: Device<U> {
    d:PhantomData<D>,
    size: usize,
    /// Learning rate
    lr:U,
    /// Weight decay
    weight_decay:U
}
impl<U,D> SGD<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of SGD
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn new(size: usize,lr:U) -> SGD<U,D> {
        SGD {
            d:PhantomData::<D>,
            size:size,
            lr:lr,
            weight_decay:U::default()
        }
    }
    /// Create an instance of SGD
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `weight_decay` - Weight decay
    pub fn with_params(size: usize,lr:U,weight_decay:U) -> SGD<U,D> {
        SGD {
            d:PhantomData::<D>,
            size:size,
            lr:lr,
            weight_decay:weight_decay,
        }
    }
}
impl<U> Optimizer<U,DeviceCpu<U>> for SGD<U,DeviceCpu<U>> where U: UnitValue<U>, DeviceCpu<U>: Device<U> {
    type InternalType = [U];
    type InternalUpdateType<'a> = ShieldSlice<'a,U>;

    #[inline]
    fn update<'a>(&mut self, e: &'a [U], w: Self::InternalUpdateType<'a>) -> Result<(),TrainingError> {
        let mut w = w;
        let a = self.lr;
        let weight_decay = self.weight_decay;

        for (w,&e) in w.iter_mut().zip(e.iter()) {
            *w = *w - a * (e + weight_decay * *w);
        }

        Ok(())
    }
}
impl<U,A> Optimizer<U,DeviceGpu<U,A>> for SGD<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator + 'static,
          DeviceGpu<U,A>: Device<U>,
          for<'a> kernel::optimizer::SGD<'a,U,A>: Kernel<Args=SGDArgs<'a,U,A>> {
    type InternalType = CudaPtr<U,A>;
    type InternalUpdateType<'a> = CudaMutPtr<'a,U,A>;

    #[inline]
    fn update<'a>(&mut self, e: &'a CudaPtr<U,A>, w: CudaMutPtr<'a,U,A>) -> Result<(),TrainingError> {
        let mut w = w;
        let mut args = SGDArgs::new(&mut w,e,self.size,self.lr,self.weight_decay);

        let mut kernel = kernel::optimizer::SGD::<'_,U,A>::new();

        kernel.launch(&mut args)?;

        Ok(())
    }
}
/// Implementation of a builder to generate SGD optimizers
pub struct SGDBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    d:PhantomData<D>,
    /// Learning rate
    lr:U,
    /// Weight decay
    weight_decay:U
}
impl<U,D> SGDBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of SGDBuilder
    /// # Arguments
    /// * `device` - device
    pub fn new(_:&D) -> SGDBuilder<U,D> {
        SGDBuilder {
            d:PhantomData::<D>,
            lr:U::from_f64(0.001).expect("Error in type conversion from f64."),
            weight_decay:U::default()
        }
    }

    /// Replaces the value of field lr in SGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> SGDBuilder<U,D> {
        SGDBuilder {
            d:PhantomData::<D>,
            lr:lr,
            weight_decay:self.weight_decay,
        }
    }

    /// Replaces the value of field weight_decay in SGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Weight Decay
    pub fn weight_decay(self,weight_decay:U) -> SGDBuilder<U,D> {
        SGDBuilder {
            d:PhantomData::<D>,
            lr:self.lr,
            weight_decay:weight_decay,
        }
    }
}
impl<U,D> OptimizerBuilder<U,D> for SGDBuilder<U,D> where U: UnitValue<U>, D: Device<U>, SGD<U,D>: Optimizer<U,D> {
    type Output = SGD<U,D>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(SGD::with_params(size,self.lr,self.weight_decay))
    }
}
/// MomentumSGD Implementation
pub struct MomentumSGD<U,D>
    where U: UnitValue<U>, D: Device<U>,
          Self: OptimizerState<U,D> {
    d:PhantomData<D>,
    size:usize,
    lr:U,
    mu:U,
    weight_decay:U,
    vt:<Self as OptimizerState<U,D>>::Type
}
impl<U> MomentumSGD<U,DeviceCpu<U>> where U: UnitValue<U> {
    /// Create an instance of MomentumSGD
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn new(_:&DeviceCpu<U>,size:usize,lr:U) -> MomentumSGD<U,DeviceCpu<U>> {
        MomentumSGD {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            lr:lr,
            mu:U::from_f64(0.9).expect("Error in type conversion from f64."),
            weight_decay:U::default(),
            vt:vec![U::default();size].into_boxed_slice()
        }
    }
    /// Create an instance of MomentumSGD with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `mu` - mu
    /// * `weight_decay` - Weight decay
    ///
    /// note: See the mu and weight_decay sections of the MomentumSGD algorithm formula.
    pub fn with_params(_:&DeviceCpu<U>,size:usize,lr:U,mu:U,weight_decay:U) -> MomentumSGD<U,DeviceCpu<U>> {
        MomentumSGD {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            lr:lr,
            mu:mu,
            weight_decay:weight_decay,
            vt:vec![U::default();size].into_boxed_slice()
        }
    }
}
impl<U> Optimizer<U,DeviceCpu<U>> for MomentumSGD<U,DeviceCpu<U>> where U: UnitValue<U> {
    type InternalType = [U];
    type InternalUpdateType<'a> = ShieldSlice<'a,U>;

    #[inline]
    fn update<'a>(&mut self, e: &[U], w: Self::InternalUpdateType<'a>) -> Result<(),TrainingError> {
        let mut w = w;
        let a = self.lr;
        let mu = self.mu;

        let weight_decay = self.weight_decay;

        for ((w,&e),vt) in w.iter_mut().zip(e.iter()).zip(self.vt.iter_mut()) {
            *vt = mu * *vt - a * (e + weight_decay * *w);
            *w = *w + *vt;
        }

        Ok(())
    }
}
impl<U,A> MomentumSGD<U,DeviceGpu<U,A>>
    where U: UnitValue<U> + Debug + Default,
          A: CudaAllocator,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: WriteMemory<U> {
    /// Create an instance of MomentumSGD
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn new(device:&DeviceGpu<U,A>,size:usize,lr:U)
        -> Result<MomentumSGD<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        Ok(MomentumSGD {
            d:PhantomData::<DeviceGpu<U,A>>,
            size:size,
            lr:lr,
            mu:U::from_f64(0.9).expect("Error in type conversion from f64."),
            weight_decay:U::default(),
            vt: CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?
        })
    }
    /// Create an instance of MomentumSGD with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `mu` - mu
    /// * `weight_decay` - Weight decay
    ///
    /// note: See the mu and weight_decay sections of the MomentumSGD algorithm formula.
    pub fn with_params(device:&DeviceGpu<U,A>,size:usize,lr:U,mu:U,weight_decay:U)
        -> Result<MomentumSGD<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        Ok(MomentumSGD {
            d:PhantomData::<DeviceGpu<U,A>>,
            size:size,
            lr:lr,
            mu:mu,
            weight_decay:weight_decay,
            vt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?
        })
    }
}
impl<U,A> Optimizer<U,DeviceGpu<U,A>> for MomentumSGD<U,DeviceGpu<U,A>>
    where U: UnitValue<U> + Debug + Default,
          A: CudaAllocator + 'static,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: WriteMemory<U>,
          for<'a> kernel::optimizer::MomentumSGD<'a,U,A>: Kernel<Args=MomentumSGDArgs<'a,U,A>> {
    type InternalType = CudaPtr<U,A>;
    type InternalUpdateType<'a> = CudaMutPtr<'a,U,A>;

    #[inline]
    fn update<'a>(&mut self, e: &CudaPtr<U,A>, w: CudaMutPtr<'a,U,A>) -> Result<(),TrainingError> {
        let mut w = w;
        let mut args = MomentumSGDArgs::new(&mut w,e,self.size,self.lr,self.mu,self.weight_decay,&mut self.vt);

        let mut kernel = kernel::optimizer::MomentumSGD::<'_,U,A>::new();

        kernel.launch(&mut args)?;

        Ok(())
    }
}
impl<U> OptimizerState<U,DeviceCpu<U>> for MomentumSGD<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          DeviceCpu<U>: Device<U> {
    type Type = Box<[U]>;
}
impl<U,A> OptimizerState<U,DeviceGpu<U,A>> for MomentumSGD<U,DeviceGpu<U,A>>
    where U: UnitValue<U> + Debug + Default,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    type Type = CudaPtr<U,A>;
}
/// Implementation of a builder to generate MomentumSGD optimizers
pub struct MomentumSGDBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    device:D,
    lr:U,
    mu:U,
    weight_decay:U
}
impl<U,D> MomentumSGDBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of MomentumSGDBuilder
    /// # Arguments
    /// * `device` - device
    pub fn new(device:&D) -> MomentumSGDBuilder<U,D> {
        MomentumSGDBuilder {
            device:device.clone(),
            lr:U::from_f64(0.001).expect("Error in type conversion from f64."),
            mu:U::from_f64(0.9).expect("Error in type conversion from f64."),
            weight_decay:U::default()
        }
    }

    /// Replaces the value of field lr in MomentumSGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> MomentumSGDBuilder<U,D> {
        MomentumSGDBuilder {
            device:self.device,
            lr:lr,
            mu:self.mu,
            weight_decay:self.weight_decay,
        }
    }

    /// Replaces the value of field weight_decay in MomentumSGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Learning rate
    pub fn weight_decay(self,weight_decay:U) -> MomentumSGDBuilder<U,D> {
        MomentumSGDBuilder {
            device:self.device,
            lr:self.lr,
            mu:self.mu,
            weight_decay:weight_decay
        }
    }

    /// Replaces the value of field mu in MomentumSGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `mu` - momentum
    pub fn mu(self,mu:U) -> MomentumSGDBuilder<U,D> {
        MomentumSGDBuilder {
            device:self.device,
            lr:self.lr,
            mu:mu,
            weight_decay:self.weight_decay,
        }
    }
}
impl<U> OptimizerBuilder<U,DeviceCpu<U>> for MomentumSGDBuilder<U,DeviceCpu<U>>
    where U: UnitValue<U>, MomentumSGD<U,DeviceCpu<U>>: Optimizer<U,DeviceCpu<U>> {
    type Output = MomentumSGD<U,DeviceCpu<U>>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(MomentumSGD::<_,DeviceCpu<U>>::with_params(&self.device,size,self.lr,self.mu,self.weight_decay))
    }
}
impl<U,A> OptimizerBuilder<U,DeviceGpu<U,A>> for MomentumSGDBuilder<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          MomentumSGD<U,DeviceGpu<U,A>>: Optimizer<U,DeviceGpu<U,A>> {
    type Output = MomentumSGD<U,DeviceGpu<U,A>>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        MomentumSGD::<_,DeviceGpu<U,A>>::with_params(&self.device,size,self.lr,self.mu,self.weight_decay)
    }
}
/// Adagrad Implementation
pub struct Adagrad<U,D>
    where U: UnitValue<U>, D: Device<U>,
          Self: OptimizerState<U,D> {
    d:PhantomData<D>,
    size:usize,
    lr:U,
    gt:<Self as OptimizerState<U,D>>::Type,
    weight_decay:U,
    eps:U
}
impl<U> Adagrad<U,DeviceCpu<U>> where U: UnitValue<U> {
    /// Create an instance of Adagrad
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    pub fn new(device:&DeviceCpu<U>,size:usize) -> Adagrad<U,DeviceCpu<U>> {
        Adagrad::<U,DeviceCpu<U>>::with_params(
            device,size,
            U::from_f64(0.01).expect("Error in type conversion from f64."),
            U::default()
        )
    }

    /// Create an instance of Adagrad with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_params(_:&DeviceCpu<U>,size:usize,lr:U,weight_decay:U) -> Adagrad<U,DeviceCpu<U>> {
        Adagrad {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            lr:lr,
            gt:vec![U::default();size].into_boxed_slice(),
            weight_decay: weight_decay,
            eps:U::from_f64(1e-10f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U,DeviceCpu<U>> for Adagrad<U,DeviceCpu<U>> where U: UnitValue<U> {
    type InternalType = [U];
    type InternalUpdateType<'a> = ShieldSlice<'a,U>;

    #[inline]
    fn update<'a>(&mut self, e: &[U], w: Self::InternalUpdateType<'a>) -> Result<(),TrainingError> {
        let mut w = w;
        let a = self.lr;
        let weight_decay = self.weight_decay;

        for ((w,&e),gt) in w.iter_mut().zip(e.iter()).zip(self.gt.iter_mut()) {
            let e = e + weight_decay * *w;

            *gt += e * e;
            *w = *w - a * (e / (gt.sqrt() + self.eps));
        }

        Ok(())
    }
}
impl<U,A> Adagrad<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of Adagrad
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    pub fn new(device:&DeviceGpu<U,A>,size:usize) -> Result<Adagrad<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        Adagrad::<U,DeviceGpu<U,A>>::with_params(
            device,size,
            U::from_f64(0.01).expect("Error in type conversion from f64."),
            U::default()
        )
    }

    /// Create an instance of Adagrad with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_params(device:&DeviceGpu<U,A>,size:usize,lr:U,weight_decay:U) -> Result<Adagrad<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        Ok(Adagrad {
            d:PhantomData::<DeviceGpu<U,A>>,
            size:size,
            lr:lr,
            gt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            weight_decay:weight_decay,
            eps:U::from_f64(1e-10f64).expect("Error in type conversion from f64.")
        })
    }
}
impl<U,A> Optimizer<U,DeviceGpu<U,A>> for Adagrad<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator + 'static,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          for<'a> kernel::optimizer::Adagrad<'a,U,A>: Kernel<Args=AdagradArgs<'a,U,A>> {
    type InternalType = CudaPtr<U,A>;
    type InternalUpdateType<'a> = CudaMutPtr<'a,U,A>;

    #[inline]
    fn update<'a>(&mut self, e: &'a CudaPtr<U,A>, w: CudaMutPtr<'a,U,A>) -> Result<(),TrainingError> {
        let mut w = w;
        let mut args = AdagradArgs::new(&mut w,e,self.size,self.lr,self.weight_decay,self.eps,&mut self.gt);

        let mut kernel = kernel::optimizer::Adagrad::<'_,U,A>::new();

        kernel.launch(&mut args)?;

        Ok(())
    }
}
impl<U> OptimizerState<U,DeviceCpu<U>> for Adagrad<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          DeviceCpu<U>: Device<U> {
    type Type = Box<[U]>;
}
impl<U,A> OptimizerState<U,DeviceGpu<U,A>> for Adagrad<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    type Type = CudaPtr<U,A>;
}
/// Implementation of a builder to generate Adagrad optimizers
pub struct AdagradBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    lr:U,
    weight_decay:U,
    device:D
}
impl<U,D> AdagradBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of AdagradBuilder
    /// # Arguments
    /// * `device` - device
    pub fn new(device:&D) -> AdagradBuilder<U,D> {
        AdagradBuilder {
            lr:U::from_f64(0.01).expect("Error in type conversion from f64."),
            weight_decay:U::default(),
            device:device.clone()
        }
    }

    /// Replaces the value of field lr in AdagradBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> AdagradBuilder<U,D> {
        AdagradBuilder {
            lr:lr,
            weight_decay:self.weight_decay,
            device:self.device
        }
    }

    /// Replaces the value of field weight_decay in AdagradBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Learning rate
    pub fn weight_decay(self,weight_decay:U) -> AdagradBuilder<U,D> {
        AdagradBuilder {
            lr:self.lr,
            weight_decay:weight_decay,
            device:self.device
        }
    }
}
impl<U> OptimizerBuilder<U,DeviceCpu<U>> for AdagradBuilder<U,DeviceCpu<U>>
    where U: UnitValue<U>, Adagrad<U,DeviceCpu<U>>: Optimizer<U,DeviceCpu<U>> {
    type Output = Adagrad<U,DeviceCpu<U>>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(Adagrad::<_,DeviceCpu<U>>::with_params(&self.device,size,self.lr,self.weight_decay))
    }
}
impl<U,A> OptimizerBuilder<U,DeviceGpu<U,A>> for AdagradBuilder<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          Adagrad<U,DeviceGpu<U,A>>: Optimizer<U,DeviceGpu<U,A>> {
    type Output = Adagrad<U,DeviceGpu<U,A>>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Adagrad::<_,DeviceGpu<U,A>>::with_params(&self.device,size,self.lr,self.weight_decay)
    }
}
/// RMSprop Implementation
pub struct RMSprop<U,D>
    where U: UnitValue<U>, D: Device<U>,
          Self: OptimizerState<U,D> {
    d:PhantomData<D>,
    size:usize,
    lr:U,
    weight_decay:U,
    alpha:U,
    mu:U,
    gt:<Self as OptimizerState<U,D>>::Type,
    bt:<Self as OptimizerState<U,D>>::Type,
    eps:U
}
impl<U> RMSprop<U,DeviceCpu<U>> where U: UnitValue<U> {
    /// Create an instance of RMSprop
    /// # Arguments
    /// * `size` - input size
    pub fn new(device:&DeviceCpu<U>,size:usize) -> RMSprop<U,DeviceCpu<U>> {
        RMSprop::<U,DeviceCpu<U>>::with_lr(device,size,U::from_f64(0.0001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of RMSprop with Learning rate
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceCpu<U>,size:usize,lr:U) -> RMSprop<U,DeviceCpu<U>> {
        RMSprop::<U,DeviceCpu<U>>::with_params(
            device,size,lr,
            U::default(),
            U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            U::default(),
        )
    }

    /// Create an instance of RMSprop with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `alpha` - alpha
    /// * `mu` - momentum
    pub fn with_params(_:&DeviceCpu<U>,size:usize,lr:U,weight_decay:U,alpha:U,mu:U) -> RMSprop<U,DeviceCpu<U>> {
        RMSprop {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            lr:lr,
            weight_decay:weight_decay,
            alpha:alpha,
            mu:mu,
            gt:vec![U::default();size].into_boxed_slice(),
            bt:vec![U::default();size].into_boxed_slice(),
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U,DeviceCpu<U>> for RMSprop<U,DeviceCpu<U>> where U: UnitValue<U> {
    type InternalType = [U];
    type InternalUpdateType<'a> = ShieldSlice<'a,U>;

    #[inline]
    fn update<'a>(&mut self, e: &'a [U], w: Self::InternalUpdateType<'a>) -> Result<(),TrainingError> {
        let mut w = w;
        let a = self.lr;
        let weight_decay = self.weight_decay;
        let alpha = self.alpha;
        let mu = self.mu;

        for ((w,&e),(gt,bt)) in w.iter_mut().zip(e.iter())
                                                            .zip(self.gt.iter_mut().zip(self.bt.iter_mut())) {
            let e = e + weight_decay * *w;

            *gt = alpha * *gt + (U::one() - alpha) * e * e;
            *bt = mu * *bt + e / (gt.sqrt() + self.eps);

            *w = *w - a * *bt;
        }

        Ok(())
    }
}
impl<U,A> RMSprop<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of RMSprop
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    pub fn new(device:&DeviceGpu<U,A>,size:usize)
        -> Result<RMSprop<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        RMSprop::<U,DeviceGpu<U,A>>::with_lr(device,size,U::from_f64(0.0001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of RMSprop with Learning rate
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceGpu<U,A>,size:usize,lr:U)
        -> Result<RMSprop<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        RMSprop::<U,DeviceGpu<U,A>>::with_params(
            device,size,
            lr,
            U::default(),
            U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            U::default()
        )
    }

    /// Create an instance of RMSprop with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `weight_decay` - Weight Decay
    /// * `alpha` - alpha
    /// * `mu` - mu
    pub fn with_params(device:&DeviceGpu<U,A>,size:usize,lr:U,weight_decay:U,alpha:U,mu:U)
        -> Result<RMSprop<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        Ok(RMSprop {
            d:PhantomData::<DeviceGpu<U,A>>,
            size:size,
            lr:lr,
            weight_decay:weight_decay,
            alpha:alpha,
            mu:mu,
            gt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            bt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        })
    }
}
impl<U,A> Optimizer<U,DeviceGpu<U,A>> for RMSprop<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator + 'static,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          for<'a> kernel::optimizer::RMSprop<'a,U,A>: Kernel<Args=RMSpropArgs<'a,U,A>> {
    type InternalType = CudaPtr<U,A>;
    type InternalUpdateType<'a> = CudaMutPtr<'a,U,A>;

    #[inline]
    fn update<'a>(&mut self, e: &'a CudaPtr<U,A>, w: CudaMutPtr<'a,U,A>) -> Result<(),TrainingError> {
        let mut w = w;
        let mut args = RMSpropArgs::new(&mut w,e,self.size,self.lr,self.weight_decay,self.alpha,self.mu,self.eps,&mut self.gt, &mut self.bt);

        let mut kernel = kernel::optimizer::RMSprop::<'_,U,A>::new();

        kernel.launch(&mut args)?;

        Ok(())
    }
}
impl<U> OptimizerState<U,DeviceCpu<U>> for RMSprop<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          DeviceCpu<U>: Device<U> {
    type Type = Box<[U]>;
}
impl<U,A> OptimizerState<U,DeviceGpu<U,A>> for RMSprop<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    type Type = CudaPtr<U,A>;
}
/// Implementation of a builder to generate RMSprop optimizers
pub struct RMSpropBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    lr:U,
    weight_decay:U,
    alpha:U,
    mu:U,
    device:D
}
impl<U,D> RMSpropBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of RMSpropBuilder with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    pub fn new(device:&D) -> RMSpropBuilder<U,D> {
        RMSpropBuilder {
            lr:U::from_f64(0.0001f64).expect("Error in type conversion from f64."),
            weight_decay:U::default(),
            alpha:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            mu:U::default(),
            device:device.clone()
        }
    }

    /// Replaces the value of field lr in RMSpropBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> RMSpropBuilder<U,D> {
        RMSpropBuilder {
            device:self.device,
            lr:lr,
            weight_decay:self.weight_decay,
            alpha:self.alpha,
            mu:self.mu
        }
    }

    /// Replaces the value of field weight_decay in MomentumSGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Learning rate
    pub fn weight_decay(self,weight_decay:U) -> RMSpropBuilder<U,D> {
        RMSpropBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:weight_decay,
            alpha:self.alpha,
            mu:self.mu,
        }
    }

    /// Replaces the value of field alpha in RMSpropBuilder with the passed value and returns it.
    /// # Arguments
    /// * `alpha` - alpha
    pub fn alpha(self,alpha:U) -> RMSpropBuilder<U,D> {
        RMSpropBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            alpha:alpha,
            mu:self.mu,
        }
    }

    /// Replaces the value of field mu in RMSpropBuilder with the passed value and returns it.
    /// # Arguments
    /// * `mu` - momentum
    pub fn mu(self,mu:U) -> RMSpropBuilder<U,D> {
        RMSpropBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            alpha:self.alpha,
            mu:mu,
        }
    }
}
impl<U> OptimizerBuilder<U,DeviceCpu<U>> for RMSpropBuilder<U,DeviceCpu<U>>
    where U: UnitValue<U>, RMSprop<U,DeviceCpu<U>>: Optimizer<U,DeviceCpu<U>> {
    type Output = RMSprop<U,DeviceCpu<U>>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(RMSprop::<_,DeviceCpu<U>>::with_params(&self.device,size,self.lr,self.weight_decay,self.alpha,self.mu))
    }
}
impl<U,A> OptimizerBuilder<U,DeviceGpu<U,A>> for RMSpropBuilder<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          RMSprop<U,DeviceGpu<U,A>>: Optimizer<U,DeviceGpu<U,A>> {
    type Output = RMSprop<U,DeviceGpu<U,A>>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        RMSprop::<_,DeviceGpu<U,A>>::with_params(&self.device,size,self.lr,self.weight_decay,self.alpha,self.mu)
    }
}
/// Adam Implementation
pub struct Adam<U,D>
    where U: UnitValue<U>, D: Device<U>,
          Self: OptimizerState<U,D> {
    d:PhantomData<D>,
    size:usize,
    lr:U,
    weight_decay:U,
    mt:<Self as OptimizerState<U,D>>::Type,
    vt:<Self as OptimizerState<U,D>>::Type,
    b1:U,
    b2:U,
    b1t:U,
    b2t:U,
    eps:U
}
impl<U> Adam<U,DeviceCpu<U>> where U: UnitValue<U> {
    /// Create an instance of Adam
    /// # Arguments
    /// * `size` - input size
    pub fn new(device:&DeviceCpu<U>,size:usize) -> Adam<U,DeviceCpu<U>> {
        Adam::<U,DeviceCpu<U>>::with_lr(device,size,U::from_f64(0.001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of Adam with Learning rate
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceCpu<U>,size:usize,lr:U) -> Adam<U,DeviceCpu<U>> {
        Adam::<U,DeviceCpu<U>>::with_params(device,size,
                          lr,
                          U::default(),
                          U::from_f64(0.9f64).expect("Error in type conversion from f64."),
                          U::from_f64(0.999f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of Adam with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `b1` - beta1
    /// * `b2` - beta2
    pub fn with_params(_:&DeviceCpu<U>,size:usize,lr:U,weight_decay:U,b1:U,b2:U) -> Adam<U,DeviceCpu<U>> {
        Adam {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            lr:lr,
            weight_decay:weight_decay,
            mt:vec![U::default();size].into_boxed_slice(),
            vt:vec![U::default();size].into_boxed_slice(),
            b1:b1,
            b2:b2,
            b1t:b1,
            b2t:b2,
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U,DeviceCpu<U>> for Adam<U,DeviceCpu<U>> where U: UnitValue<U> {
    type InternalType = [U];
    type InternalUpdateType<'a> = ShieldSlice<'a,U>;

    #[inline]
    fn update<'a>(&mut self, e: &'a [U], w: Self::InternalUpdateType<'a>) -> Result<(),TrainingError> {
        let mut w = w;
        let a = self.lr;
        let weight_decay = self.weight_decay;
        let b1 = self.b1;
        let b2 = self.b2;
        let b1t = self.b1t;
        let b2t = self.b2t;

        for ((w,&e),(mt,vt)) in w.iter_mut().zip(e.iter()).zip(self.mt.iter_mut().zip(self.vt.iter_mut())) {
            let e = e + weight_decay * *w;

            *mt = b1 * *mt + (U::one() - self.b1) * e;
            *vt = b2 * *vt + (U::one() - self.b2) * e * e;

            *w = *w - a * (*mt / (U::one() - b1t)) / ((*vt / (U::one() - b2t)) + self.eps).sqrt();
        }

        self.b1t = b1t * b1;
        self.b2t = b2t * b2;

        Ok(())
    }
}
impl<U,A> Adam<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of Adam
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    pub fn new(device:&DeviceGpu<U,A>,size:usize) -> Result<Adam<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        Adam::<U,DeviceGpu<U,A>>::with_lr(device,size,U::from_f64(0.001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of Adam with Learning rate
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceGpu<U,A>,size:usize,lr:U) ->Result<Adam<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        Adam::<U,DeviceGpu<U,A>>::with_params(device,size,lr,
                          U::default(),
                          U::from_f64(0.9f64).expect("Error in type conversion from f64."),
                          U::from_f64(0.999f64).expect("Error in type conversion from f64.")
        )
    }

    /// Create an instance of Adam with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `b1` - beta1
    /// * `b2` - beta2
    pub fn with_params(device:&DeviceGpu<U,A>,size:usize,lr:U,weight_decay:U,b1:U,b2:U) ->Result<Adam<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        Ok(Adam {
            d:PhantomData::<DeviceGpu<U,A>>,
            size:size,
            lr:lr,
            weight_decay:weight_decay,
            mt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            vt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            b1:b1,
            b2:b2,
            b1t:b1,
            b2t:b2,
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        })
    }
}
impl<U,A> Optimizer<U,DeviceGpu<U,A>> for Adam<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator + 'static,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          for<'a> kernel::optimizer::Adam<'a,U,A>: Kernel<Args=AdamArgs<'a,U,A>> {
    type InternalType = CudaPtr<U,A>;
    type InternalUpdateType<'a> = CudaMutPtr<'a,U,A>;

    #[inline]
    fn update<'a>(&mut self, e: &'a CudaPtr<U,A>, w: CudaMutPtr<'a,U,A>) -> Result<(),TrainingError> {
        let mut w = w;
        let mut args = AdamArgs::new(&mut w,e,self.size,self.lr,self.weight_decay,self.eps,
                                                 &mut self.mt,&mut self.vt,
                                                 self.b1,self.b2,self.b1t,self.b2t);

        let mut kernel = kernel::optimizer::Adam::<'_,U,A>::new();

        kernel.launch(&mut args)?;

        self.b1t = self.b1t * self.b1;
        self.b2t = self.b2t * self.b2;

        Ok(())
    }
}
impl<U> OptimizerState<U,DeviceCpu<U>> for Adam<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          DeviceCpu<U>: Device<U> {
    type Type = Box<[U]>;
}
impl<U,A> OptimizerState<U,DeviceGpu<U,A>> for Adam<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    type Type = CudaPtr<U,A>;
}
/// Implementation of a builder to generate Adam optimizers
pub struct AdamBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    lr:U,
    weight_decay:U,
    b1:U,
    b2:U,
    device:D
}
impl<U,D> AdamBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of AdamBuilder with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    pub fn new(device:&D) -> AdamBuilder<U,D> {
        AdamBuilder {
            lr:U::from_f64(0.001f64).expect("Error in type conversion from f64."),
            weight_decay:U::default(),
            b1:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            b2:U::from_f64(0.999f64).expect("Error in type conversion from f64."),
            device:device.clone()
        }
    }

    /// Replaces the value of field lr in AdamBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> AdamBuilder<U,D> {
        AdamBuilder {
            device:self.device,
            lr:lr,
            weight_decay:self.weight_decay,
            b1:self.b1,
            b2:self.b2
        }
    }

    /// Replaces the value of field weight_decay in AdamBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Learning rate
    pub fn weight_decay(self,weight_decay:U) -> AdamBuilder<U,D> {
        AdamBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:weight_decay,
            b1:self.b1,
            b2:self.b2
        }
    }

    /// Replaces the value of field b1 in AdamBuilder with the passed value and returns it.
    /// # Arguments
    /// * `b1` - b1
    pub fn b1(self,b1:U) -> AdamBuilder<U,D> {
        AdamBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            b1:b1,
            b2:self.b2
        }
    }

    /// Replaces the value of field b2 in AdamBuilder with the passed value and returns it.
    /// # Arguments
    /// * `b2` - b2
    pub fn b2(self,b2:U) -> AdamBuilder<U,D> {
        AdamBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            b1:self.b1,
            b2:b2
        }
    }
}
impl<U> OptimizerBuilder<U,DeviceCpu<U>> for AdamBuilder<U,DeviceCpu<U>>
    where U: UnitValue<U>, Adam<U,DeviceCpu<U>>: Optimizer<U,DeviceCpu<U>> {
    type Output = Adam<U,DeviceCpu<U>>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(Adam::<_,DeviceCpu<U>>::with_params(&self.device,size,self.lr,self.weight_decay,self.b1,self.b2))
    }
}
impl<U,A> OptimizerBuilder<U,DeviceGpu<U,A>> for AdamBuilder<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          Adam<U,DeviceGpu<U,A>>: Optimizer<U,DeviceGpu<U,A>> {
    type Output = Adam<U,DeviceGpu<U,A>>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Adam::<_,DeviceGpu<U,A>>::with_params(&self.device,size,self.lr,self.weight_decay,self.b1,self.b2)
    }
}
/// AdamW Implementation
pub struct AdamW<U,D>
    where U: UnitValue<U>, D: Device<U>,
          Self: OptimizerState<U,D> {
    d:PhantomData<D>,
    size:usize,
    lr:U,
    weight_decay:U,
    mt:<Self as OptimizerState<U,D>>::Type,
    vt:<Self as OptimizerState<U,D>>::Type,
    b1:U,
    b2:U,
    b1t:U,
    b2t:U,
    eps:U
}
impl<U> AdamW<U,DeviceCpu<U>> where U: UnitValue<U> {
    /// Create an instance of AdamW
    /// # Arguments
    /// * `size` - input size
    pub fn new(device:&DeviceCpu<U>,size:usize) -> AdamW<U,DeviceCpu<U>> {
        AdamW::<U,DeviceCpu<U>>::with_lr(device,size,U::from_f64(0.001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of AdamW with Learning rate
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceCpu<U>,size:usize,lr:U) -> AdamW<U,DeviceCpu<U>> {
        AdamW::<U,DeviceCpu<U>>::with_params(device,size,
                                             lr,
                                             U::default(),
                                             U::from_f64(0.9f64).expect("Error in type conversion from f64."),
                                             U::from_f64(0.999f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of AdamW with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `b1` - beta1
    /// * `b2` - beta2
    pub fn with_params(_:&DeviceCpu<U>,size:usize,lr:U,weight_decay:U,b1:U,b2:U) -> AdamW<U,DeviceCpu<U>> {
        AdamW {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            lr:lr,
            weight_decay:weight_decay,
            mt:vec![U::default();size].into_boxed_slice(),
            vt:vec![U::default();size].into_boxed_slice(),
            b1:b1,
            b2:b2,
            b1t:b1,
            b2t:b2,
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        }
    }
}
impl<U> Optimizer<U,DeviceCpu<U>> for AdamW<U,DeviceCpu<U>> where U: UnitValue<U> {
    type InternalType = [U];
    type InternalUpdateType<'a> = ShieldSlice<'a,U>;

    #[inline]
    fn update<'a>(&mut self, e: &'a [U], w: Self::InternalUpdateType<'a>) -> Result<(),TrainingError> {
        let mut w = w;
        let a = self.lr;
        let weight_decay = self.weight_decay;
        let b1 = self.b1;
        let b2 = self.b2;
        let b1t = self.b1t;
        let b2t = self.b2t;

        for ((w,&e),(mt,vt)) in w.iter_mut().zip(e.iter()).zip(self.mt.iter_mut().zip(self.vt.iter_mut())) {
            *w = *w - a * weight_decay * *w;

            *mt = b1 * *mt + (U::one() - self.b1) * e;
            *vt = b2 * *vt + (U::one() - self.b2) * e * e;

            *w = *w - a * (*mt / (U::one() - b1t)) / ((*vt / (U::one() - b2t)) + self.eps).sqrt();
        }

        self.b1t = b1t * b1;
        self.b2t = b2t * b2;

        Ok(())
    }
}
impl<U,A> AdamW<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of AdamW
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    pub fn new(device:&DeviceGpu<U,A>,size:usize) -> Result<AdamW<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        AdamW::<U,DeviceGpu<U,A>>::with_lr(device,size,U::from_f64(0.001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of AdamW with Learning rate
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceGpu<U,A>,size:usize,lr:U) ->Result<AdamW<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        AdamW::<U,DeviceGpu<U,A>>::with_params(device,size,lr,
                                               U::default(),
                                               U::from_f64(0.9f64).expect("Error in type conversion from f64."),
                                               U::from_f64(0.999f64).expect("Error in type conversion from f64.")
        )
    }

    /// Create an instance of AdamW with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `b1` - beta1
    /// * `b2` - beta2
    pub fn with_params(device:&DeviceGpu<U,A>,size:usize,lr:U,weight_decay:U,b1:U,b2:U) ->Result<AdamW<U,DeviceGpu<U,A>>,OptimizerBuildError> {
        Ok(AdamW {
            d:PhantomData::<DeviceGpu<U,A>>,
            size:size,
            lr:lr,
            weight_decay:weight_decay,
            mt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            vt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            b1:b1,
            b2:b2,
            b1t:b1,
            b2t:b2,
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64.")
        })
    }
}
impl<U,A> Optimizer<U,DeviceGpu<U,A>> for AdamW<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator + 'static,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          for<'a> kernel::optimizer::AdamW<'a,U,A>: Kernel<Args=AdamWArgs<'a,U,A>> {
    type InternalType = CudaPtr<U,A>;
    type InternalUpdateType<'a> = CudaMutPtr<'a,U,A>;

    #[inline]
    fn update<'a>(&mut self, e: &'a CudaPtr<U,A>, w: CudaMutPtr<'a,U,A>) -> Result<(),TrainingError> {
        let mut w = w;
        let mut args = AdamWArgs::new(&mut w,e,self.size,self.lr,self.weight_decay,self.eps,
                                     &mut self.mt,&mut self.vt,
                                     self.b1,self.b2,self.b1t,self.b2t);

        let mut kernel = kernel::optimizer::AdamW::<'_,U,A>::new();

        kernel.launch(&mut args)?;

        self.b1t = self.b1t * self.b1;
        self.b2t = self.b2t * self.b2;

        Ok(())
    }
}
impl<U> OptimizerState<U,DeviceCpu<U>> for AdamW<U,DeviceCpu<U>>
    where U: UnitValue<U>,
          DeviceCpu<U>: Device<U> {
    type Type = Box<[U]>;
}
impl<U,A> OptimizerState<U,DeviceGpu<U,A>> for AdamW<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    type Type = CudaPtr<U,A>;
}
/// Implementation of a builder to generate AdamW optimizers
pub struct AdamWBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    lr:U,
    weight_decay:U,
    b1:U,
    b2:U,
    device:D
}
impl<U,D> AdamWBuilder<U,D> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of AdamWBuilder with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    pub fn new(device:&D) -> AdamWBuilder<U,D> {
        AdamWBuilder {
            lr:U::from_f64(0.001f64).expect("Error in type conversion from f64."),
            weight_decay:U::default(),
            b1:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            b2:U::from_f64(0.999f64).expect("Error in type conversion from f64."),
            device:device.clone()
        }
    }

    /// Replaces the value of field lr in AdamWBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> AdamWBuilder<U,D> {
        AdamWBuilder {
            device:self.device,
            lr:lr,
            weight_decay:self.weight_decay,
            b1:self.b1,
            b2:self.b2
        }
    }

    /// Replaces the value of field weight_decay in AdamWBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Learning rate
    pub fn weight_decay(self,weight_decay:U) -> AdamWBuilder<U,D> {
        AdamWBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:weight_decay,
            b1:self.b1,
            b2:self.b2
        }
    }

    /// Replaces the value of field b1 in AdamWBuilder with the passed value and returns it.
    /// # Arguments
    /// * `b1` - b1
    pub fn b1(self,b1:U) -> AdamWBuilder<U,D> {
        AdamWBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            b1:b1,
            b2:self.b2
        }
    }

    /// Replaces the value of field b2 in AdamWBuilder with the passed value and returns it.
    /// # Arguments
    /// * `b2` - b2
    pub fn b2(self,b2:U) -> AdamWBuilder<U,D> {
        AdamWBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            b1:self.b1,
            b2:b2
        }
    }
}
impl<U> OptimizerBuilder<U,DeviceCpu<U>> for AdamWBuilder<U,DeviceCpu<U>>
    where U: UnitValue<U>, AdamW<U,DeviceCpu<U>>: Optimizer<U,DeviceCpu<U>> {
    type Output = AdamW<U,DeviceCpu<U>>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(AdamW::<_,DeviceCpu<U>>::with_params(&self.device,size,self.lr,self.weight_decay,self.b1,self.b2))
    }
}
impl<U,A> OptimizerBuilder<U,DeviceGpu<U,A>> for AdamWBuilder<U,DeviceGpu<U,A>>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          AdamW<U,DeviceGpu<U,A>>: Optimizer<U,DeviceGpu<U,A>> {
    type Output = AdamW<U,DeviceGpu<U,A>>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        AdamW::<_,DeviceGpu<U,A>>::with_params(&self.device,size,self.lr,self.weight_decay,self.b1,self.b2)
    }
}

