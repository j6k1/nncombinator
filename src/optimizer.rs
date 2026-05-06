//! Definition and implementation of optimizers to be used during training

use std::marker::PhantomData;
use crate::device::{Device, DeviceCpu};
use crate::{UnitValue};
use crate::arr::ShieldSlice;
use crate::error::{ModelLoadError, OptimizerBuildError, PersistenceError, TrainingError};
use crate::scheduler::{Scheduler, IdentityLR};
use std::str::FromStr;
use std::fmt::Debug;
#[cfg(feature = "cuda")]
use crate::cuda::{CudaMutPtr, CudaPtr, kernel, Kernel, WriteMemory};
#[cfg(feature = "cuda")]
use crate::cuda::allocator::CudaAllocator;
#[cfg(feature = "cuda")]
use crate::cuda::kernel::optimizer::{AdagradArgs, AdamArgs, AdamWArgs, MomentumSGDArgs, RMSpropArgs, SGDArgs};
#[cfg(feature = "cuda")]
use crate::cuda::ReadMemory;
#[cfg(feature = "cuda")]
use crate::device::{DeviceGpu, DeviceAllocator};
use crate::persistence::{Linear, LinearPersistence, Persistence, Specialized, TextFilePersistence, UnitOrMarker};

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
    /// Learning Progress Notification
    /// # Arguments
    /// * `step` - step count
    fn on_step(&mut self, step: usize) -> Result<(),TrainingError>;
}
/// Optimizer State Definition
pub trait OptimizerState<U,D> where U: Clone + Copy + UnitValue<U>, D: Device<U> {
    /// State type
    type Type;
}
/// SGD Implementation
pub struct SGD<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> {
    d:PhantomData<D>,
    #[allow(dead_code)]
    size: usize,
    /// Learning rate
    lr:U,
    /// Weight decay
    weight_decay:U,
    /// Learning rate scheduler
    scheduler: SD
}
impl<U,D> SGD<U,D,IdentityLR> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of SGD
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn new(size: usize,lr:U) -> SGD<U,D,IdentityLR> {
        SGD {
            d:PhantomData::<D>,
            size:size,
            lr:lr,
            weight_decay:U::default(),
            scheduler: IdentityLR
        }
    }
}

impl<U,D,S> SGD<U,D,S> where U: UnitValue<U>, D: Device<U>, S: Scheduler<U> {
    /// Create an instance of SGD
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `weight_decay` - Weight decay
    /// * `scheduler` - Learning rate scheduler
    pub fn with_params(size: usize,lr:U,weight_decay:U,scheduler: S) -> SGD<U,D,S> {
        SGD {
            d:PhantomData::<D>,
            size:size,
            lr:lr,
            weight_decay:weight_decay,
            scheduler
        }
    }
}
impl<U,SD> Optimizer<U,DeviceCpu<U>> for SGD<U,DeviceCpu<U>,SD> where U: UnitValue<U>, DeviceCpu<U>: Device<U>, SD: Scheduler<U> {
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

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Optimizer<U,DeviceGpu<U,A>> for SGD<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          A: CudaAllocator + 'static,
          DeviceGpu<U,A>: Device<U>,
          for<'a> kernel::optimizer::SGD<'a,U,A>: Kernel<Args=SGDArgs<'a,U,A>>, SD: Scheduler<U> {
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

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
impl<U,D,S> Persistence<U,TextFilePersistence<U>,Specialized> for SGD<U,D,S>
    where U: UnitValue<U> + FromStr,
          D: Device<U>,
          S: Scheduler<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn load(&mut self, _: &mut TextFilePersistence<U>) -> Result<(), ModelLoadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        Ok(())
    }
}
impl<T,U,D,S> Persistence<U,T,Linear> for SGD<U,D,S>
    where T: LinearPersistence<U>,
          U: UnitValue<U> + FromStr,
          D: Device<U>,
          S: Scheduler<U> {
    fn load(&mut self, _: &mut T) -> Result<(), ModelLoadError> {
        Ok(())
    }

    fn save(&mut self, _: &mut T) -> Result<(), PersistenceError> {
        Ok(())
    }
}
/// Implementation of a builder to generate SGD optimizers
pub struct SGDBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone {
    d:PhantomData<D>,
    /// Learning rate
    lr:U,
    /// Weight decay
    weight_decay:U,
    /// Learning rate scheduler
    scheduler: SD
}
impl<U,D> SGDBuilder<U,D,IdentityLR> where U: UnitValue<U>, D: Device<U> + Clone {
    /// Create an instance of SGDBuilder
    /// # Arguments
    /// * `device` - device
    pub fn new(_: &D) -> SGDBuilder<U, D, IdentityLR> {
        SGDBuilder {
            d: PhantomData::<D>,
            lr: U::from_f64(0.001).expect("Error in type conversion from f64."),
            weight_decay: U::default(),
            scheduler: IdentityLR
        }
    }
}
impl<U,D,SD> SGDBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone {
    /// Replaces the value of field lr in SGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> SGDBuilder<U,D,SD> {
        SGDBuilder {
            d:PhantomData::<D>,
            lr:lr,
            weight_decay:self.weight_decay,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field weight_decay in SGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Weight Decay
    pub fn weight_decay(self,weight_decay:U) -> SGDBuilder<U,D,SD> {
        SGDBuilder {
            d:PhantomData::<D>,
            lr:self.lr,
            weight_decay:weight_decay,
            scheduler:self.scheduler
        }
    }


    /// Replaces the value of field scheduler in SGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `scheduler` - learning rate scheduler
    pub fn scheduler<S>(self, scheduler: S) -> SGDBuilder<U,D,S>
        where S: Scheduler<U> + Clone {
        SGDBuilder {
            d:PhantomData::<D>,
            lr:self.lr,
            weight_decay:self.weight_decay,
            scheduler
        }
    }
}
impl<U,D,SD> OptimizerBuilder<U,D> for SGDBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone, SGD<U,D,SD>: Optimizer<U,D> {
    type Output = SGD<U,D,SD>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(SGD::with_params(size,self.lr,self.weight_decay,self.scheduler.clone()))
    }
}
/// MomentumSGD Implementation
pub struct MomentumSGD<U,D,SD>
    where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U>,
          Self: OptimizerState<U,D> {
    d:PhantomData<D>,
    #[allow(dead_code)]
    size:usize,
    lr:U,
    mu:U,
    weight_decay:U,
    vt:<Self as OptimizerState<U,D>>::Type,
    scheduler: SD
}
impl<U> MomentumSGD<U,DeviceCpu<U>,IdentityLR> where U: UnitValue<U>
{
    /// Create an instance of MomentumSGD
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn new(_: &DeviceCpu<U>, size: usize, lr: U) -> MomentumSGD<U, DeviceCpu<U>, IdentityLR> {
        MomentumSGD {
            d: PhantomData::<DeviceCpu<U>>,
            size: size,
            lr: lr,
            mu: U::from_f64(0.9).expect("Error in type conversion from f64."),
            weight_decay: U::default(),
            vt: vec![U::default(); size].into_boxed_slice(),
            scheduler: IdentityLR
        }
    }
}
impl<U,SD> MomentumSGD<U,DeviceCpu<U>,SD> where U: UnitValue<U>, SD: Scheduler<U> + Clone {
    /// Create an instance of MomentumSGD with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `mu` - mu
    /// * `weight_decay` - Weight decay
    ///
    /// note: See the mu and weight_decay sections of the MomentumSGD algorithm formula.
    pub fn with_params(_:&DeviceCpu<U>,size:usize,lr:U,mu:U,weight_decay:U, scheduler: SD) -> MomentumSGD<U,DeviceCpu<U>,SD> {
        MomentumSGD {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            lr:lr,
            mu:mu,
            weight_decay:weight_decay,
            vt:vec![U::default();size].into_boxed_slice(),
            scheduler
        }
    }
}
impl<U,SD> Optimizer<U,DeviceCpu<U>> for MomentumSGD<U,DeviceCpu<U>,SD> where U: UnitValue<U>, SD: Scheduler<U> {
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

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<U,A> MomentumSGD<U,DeviceGpu<U,A>,IdentityLR>
    where U: UnitValue<U> + Debug + Default,
          A: CudaAllocator,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: WriteMemory<U> {
    /// Create an instance of MomentumSGD
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn new(device: &DeviceGpu<U, A>, size: usize, lr: U)
               -> Result<MomentumSGD<U, DeviceGpu<U, A>, IdentityLR>, OptimizerBuildError> {
        Ok(MomentumSGD {
            d: PhantomData::<DeviceGpu<U, A>>,
            size: size,
            lr: lr,
            mu: U::from_f64(0.9).expect("Error in type conversion from f64."),
            weight_decay: U::default(),
            vt: CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            scheduler: IdentityLR
        })
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> MomentumSGD<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U> + Debug + Default,
          A: CudaAllocator,
          SD: Scheduler<U>,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: WriteMemory<U> {
    /// Create an instance of MomentumSGD with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `mu` - mu
    /// * `weight_decay` - Weight decay
    ///
    /// note: See the mu and weight_decay sections of the MomentumSGD algorithm formula.
    pub fn with_params(device:&DeviceGpu<U,A>,size:usize,lr:U,mu:U,weight_decay:U, scheduler: SD)
        -> Result<MomentumSGD<U,DeviceGpu<U,A>,SD>,OptimizerBuildError> {
        Ok(MomentumSGD {
            d:PhantomData::<DeviceGpu<U,A>>,
            size:size,
            lr:lr,
            mu:mu,
            weight_decay:weight_decay,
            vt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            scheduler
        })
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Optimizer<U,DeviceGpu<U,A>> for MomentumSGD<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U> + Debug + Default,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
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

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
impl<U,SD> OptimizerState<U,DeviceCpu<U>> for MomentumSGD<U,DeviceCpu<U>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          DeviceCpu<U>: Device<U> {
    type Type = Box<[U]>;
}
#[cfg(feature = "cuda")]
impl<U,A,SD> OptimizerState<U,DeviceGpu<U,A>> for MomentumSGD<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U> + Debug + Default,
          SD: Scheduler<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    type Type = CudaPtr<U,A>;
}
impl<U,SD> Persistence<U,TextFilePersistence<U>,Specialized> for MomentumSGD<U,DeviceCpu<U>,SD>
    where U: UnitValue<U> + FromStr,
          SD: Scheduler<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        for &vt in self.vt.iter() {
            persistence.write(UnitOrMarker::Unit(vt));
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), ModelLoadError> {
        for vt in self.vt.iter_mut() {
            *vt = persistence.read()?;
        }
        Ok(())
    }
}
impl<T,U,SD> Persistence<U,T,Linear> for MomentumSGD<U,DeviceCpu<U>,SD>
    where T: LinearPersistence<U>,
          U: UnitValue<U> + Debug + Default,
          SD: Scheduler<U> {
    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        for &vt in self.vt.iter() {
            persistence.write(vt)?;
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        for vt in self.vt.iter_mut() {
            *vt = persistence.read()?;
        }
        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Persistence<U,TextFilePersistence<U>,Specialized> for MomentumSGD<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U> + FromStr,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: ReadMemory<U> + WriteMemory<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        for &vt in self.vt.read_to_vec()?.iter() {
            persistence.write(UnitOrMarker::Unit(vt));
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), ModelLoadError> {
        let mut vt = vec![U::default();self.size];

        for vt in vt.iter_mut() {
            *vt = persistence.read()?;
        }

        self.vt.memcpy(vt.as_ptr(),self.size)?;

        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<T,U,A,SD> Persistence<U,T,Linear> for MomentumSGD<U,DeviceGpu<U,A>,SD>
    where T: LinearPersistence<U>,
          U: UnitValue<U> + Debug + Default,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: ReadMemory<U> + WriteMemory<U> {
    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        for &vt in self.vt.read_to_vec()?.iter() {
            persistence.write(vt)?;
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        let mut vt = vec![U::default();self.size];

        for vt in vt.iter_mut() {
            *vt = persistence.read()?;
        }

        self.vt.memcpy(vt.as_ptr(),self.size)?;

        Ok(())
    }
}
/// Implementation of a builder to generate MomentumSGD optimizers
pub struct MomentumSGDBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone {
    device:D,
    lr:U,
    mu:U,
    weight_decay:U,
    scheduler: SD
}
impl<U,D> MomentumSGDBuilder<U,D,IdentityLR> where U: UnitValue<U>, D: Device<U> + Clone
{
    /// Create an instance of MomentumSGDBuilder
    /// # Arguments
    /// * `device` - device
    pub fn new(device: &D) -> MomentumSGDBuilder<U, D, IdentityLR> {
        MomentumSGDBuilder {
            device: device.clone(),
            lr: U::from_f64(0.001).expect("Error in type conversion from f64."),
            mu: U::from_f64(0.9).expect("Error in type conversion from f64."),
            weight_decay: U::default(),
            scheduler: IdentityLR
        }
    }
}
impl<U,D,SD> MomentumSGDBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U> + Clone, SD: Scheduler<U> + Clone {
    /// Replaces the value of field lr in MomentumSGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> MomentumSGDBuilder<U,D,SD> {
        MomentumSGDBuilder {
            device:self.device,
            lr:lr,
            mu:self.mu,
            weight_decay:self.weight_decay,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field weight_decay in MomentumSGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Learning rate
    pub fn weight_decay(self,weight_decay:U) -> MomentumSGDBuilder<U,D,SD> {
        MomentumSGDBuilder {
            device:self.device,
            lr:self.lr,
            mu:self.mu,
            weight_decay:weight_decay,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field mu in MomentumSGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `mu` - momentum
    pub fn mu(self,mu:U) -> MomentumSGDBuilder<U,D,SD> {
        MomentumSGDBuilder {
            device:self.device,
            lr:self.lr,
            mu:mu,
            weight_decay:self.weight_decay,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field scheduler in MomentumSGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `scheduler` - learning rate scheduler
    pub fn scheduler<S>(self,scheduler:S) -> MomentumSGDBuilder<U,D,S> where S: Scheduler<U> + Clone {
        MomentumSGDBuilder {
            device:self.device,
            lr:self.lr,
            mu:self.mu,
            weight_decay:self.weight_decay,
            scheduler
        }
    }
}
impl<U,SD> OptimizerBuilder<U,DeviceCpu<U>> for MomentumSGDBuilder<U,DeviceCpu<U>,SD>
    where U: UnitValue<U>, SD: Scheduler<U> + Clone, MomentumSGD<U,DeviceCpu<U>,SD>: Optimizer<U,DeviceCpu<U>> {
    type Output = MomentumSGD<U,DeviceCpu<U>,SD>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(MomentumSGD::<U,DeviceCpu<U>,SD>::with_params(&self.device,size,self.lr,self.mu,self.weight_decay,self.scheduler.clone()))
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> OptimizerBuilder<U,DeviceGpu<U,A>> for MomentumSGDBuilder<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U> + Clone,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          MomentumSGD<U,DeviceGpu<U,A>,SD>: Optimizer<U,DeviceGpu<U,A>> {
    type Output = MomentumSGD<U,DeviceGpu<U,A>,SD>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        MomentumSGD::<U,DeviceGpu<U,A>,SD>::with_params(&self.device,size,self.lr,self.mu,self.weight_decay,self.scheduler.clone())
    }
}
/// Adagrad Implementation
pub struct Adagrad<U,D,SD>
    where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U>,
          Self: OptimizerState<U,D> {
    d:PhantomData<D>,
    #[allow(dead_code)]
    size:usize,
    lr:U,
    gt:<Self as OptimizerState<U,D>>::Type,
    weight_decay:U,
    eps:U,
    scheduler: SD
}
impl<U> Adagrad<U,DeviceCpu<U>,IdentityLR> where U: UnitValue<U> {
    /// Create an instance of Adagrad
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    pub fn new(device:&DeviceCpu<U>,size:usize) -> Adagrad<U,DeviceCpu<U>,IdentityLR> {
        Adagrad::<U,DeviceCpu<U>,IdentityLR>::with_params(
            device,size,
            U::from_f64(0.01).expect("Error in type conversion from f64."),
            U::default(),
            IdentityLR
        )
    }
}
impl<U,SD> Adagrad<U,DeviceCpu<U>,SD> where U: UnitValue<U>, SD: Scheduler<U> + Clone {
    /// Create an instance of Adagrad with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_params(_:&DeviceCpu<U>,size:usize,lr:U,weight_decay:U, scheduler: SD) -> Adagrad<U,DeviceCpu<U>,SD> {
        Adagrad {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            lr:lr,
            gt:vec![U::default();size].into_boxed_slice(),
            weight_decay: weight_decay,
            eps:U::from_f64(1e-10f64).expect("Error in type conversion from f64."),
            scheduler
        }
    }
}
impl<U,SD> Optimizer<U,DeviceCpu<U>> for Adagrad<U,DeviceCpu<U>,SD> where U: UnitValue<U>, SD: Scheduler<U> {
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

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<U,A> Adagrad<U,DeviceGpu<U,A>,IdentityLR>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of Adagrad
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    pub fn new(device:&DeviceGpu<U,A>,size:usize) -> Result<Adagrad<U,DeviceGpu<U,A>,IdentityLR>,OptimizerBuildError> {
        Adagrad::<U,DeviceGpu<U,A>,IdentityLR>::with_params(
            device,size,
            U::from_f64(0.01).expect("Error in type conversion from f64."),
            U::default(),
            IdentityLR
        )
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Adagrad<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          A: CudaAllocator,
          SD: Scheduler<U>,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of Adagrad with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_params(device:&DeviceGpu<U,A>,size:usize,lr:U,weight_decay:U, scheduler: SD) -> Result<Adagrad<U,DeviceGpu<U,A>,SD>,OptimizerBuildError> {
        Ok(Adagrad {
            d:PhantomData::<DeviceGpu<U,A>>,
            size:size,
            lr:lr,
            gt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            weight_decay:weight_decay,
            eps:U::from_f64(1e-10f64).expect("Error in type conversion from f64."),
            scheduler
        })
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Optimizer<U,DeviceGpu<U,A>> for Adagrad<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
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

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
impl<U,SD> OptimizerState<U,DeviceCpu<U>> for Adagrad<U,DeviceCpu<U>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          DeviceCpu<U>: Device<U> {
    type Type = Box<[U]>;
}
#[cfg(feature = "cuda")]
impl<U,A,SD> OptimizerState<U,DeviceGpu<U,A>> for Adagrad<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    type Type = CudaPtr<U,A>;
}
impl<U,SD> Persistence<U,TextFilePersistence<U>,Specialized> for Adagrad<U,DeviceCpu<U>,SD>
    where U: UnitValue<U> + FromStr,
          SD: Scheduler<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        for &gt in self.gt.iter() {
            persistence.write(UnitOrMarker::Unit(gt));
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), ModelLoadError> {
        for gt in self.gt.iter_mut() {
            *gt = persistence.read()?;
        }
        Ok(())
    }
}
impl<T,U,SD> Persistence<U,T,Linear> for Adagrad<U,DeviceCpu<U>,SD>
    where T: LinearPersistence<U>,
          U: UnitValue<U> + Debug + Default,
          SD: Scheduler<U> {
    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        for &gt in self.gt.iter() {
            persistence.write(gt)?;
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        for gt in self.gt.iter_mut() {
            *gt = persistence.read()?;
        }
        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Persistence<U,TextFilePersistence<U>,Specialized> for Adagrad<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U> + FromStr,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: ReadMemory<U> + WriteMemory<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        for &gt in self.gt.read_to_vec()?.iter() {
            persistence.write(UnitOrMarker::Unit(gt));
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), ModelLoadError> {
        let mut gt = vec![U::default();self.size];

        for gt in gt.iter_mut() {
            *gt = persistence.read()?;
        }

        self.gt.memcpy(gt.as_ptr(),self.size)?;

        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<T,U,A,SD> Persistence<U,T,Linear> for Adagrad<U,DeviceGpu<U,A>,SD>
    where T: LinearPersistence<U>,
          U: UnitValue<U> + Debug + Default,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: ReadMemory<U> + WriteMemory<U> {
    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        for &gt in self.gt.read_to_vec()?.iter() {
            persistence.write(gt)?;
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        let mut gt = vec![U::default();self.size];

        for gt in gt.iter_mut() {
            *gt = persistence.read()?;
        }

        self.gt.memcpy(gt.as_ptr(),self.size)?;

        Ok(())
    }
}
/// Implementation of a builder to generate Adagrad optimizers
pub struct AdagradBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone {
    lr:U,
    weight_decay:U,
    device:D,
    scheduler: SD
}
impl<U,D> AdagradBuilder<U,D,IdentityLR> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of AdagradBuilder
    /// # Arguments
    /// * `device` - device
    pub fn new(device:&D) -> AdagradBuilder<U,D,IdentityLR> {
        AdagradBuilder {
            lr:U::from_f64(0.01).expect("Error in type conversion from f64."),
            weight_decay:U::default(),
            device:device.clone(),
            scheduler: IdentityLR
        }
    }
}
impl<U,D,SD> AdagradBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone {
    /// Replaces the value of field lr in AdagradBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> AdagradBuilder<U,D,SD> {
        AdagradBuilder {
            lr:lr,
            weight_decay:self.weight_decay,
            device:self.device,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field weight_decay in AdagradBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Learning rate
    pub fn weight_decay(self,weight_decay:U) -> AdagradBuilder<U,D,SD> {
        AdagradBuilder {
            lr:self.lr,
            weight_decay:weight_decay,
            device:self.device,
            scheduler:self.scheduler
        }
    }


    /// Replaces the value of field scheduler in AdagradBuilder with the passed value and returns it.
    /// # Arguments
    /// * `scheduler` - learning rate scheduler
    pub fn scheduler<S>(self, scheduler: S) -> AdagradBuilder<U,D,S>
        where S: Scheduler<U> + Clone {
        AdagradBuilder {
            lr:self.lr,
            weight_decay:self.weight_decay,
            device:self.device,
            scheduler
        }
    }
}
impl<U,SD> OptimizerBuilder<U,DeviceCpu<U>> for AdagradBuilder<U,DeviceCpu<U>,SD>
    where U: UnitValue<U>, SD: Scheduler<U> + Clone, Adagrad<U,DeviceCpu<U>,SD>: Optimizer<U,DeviceCpu<U>> {
    type Output = Adagrad<U,DeviceCpu<U>,SD>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(Adagrad::<_,DeviceCpu<U>,SD>::with_params(&self.device,size,self.lr,self.weight_decay,self.scheduler.clone()))
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> OptimizerBuilder<U,DeviceGpu<U,A>> for AdagradBuilder<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U> + Clone,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          Adagrad<U,DeviceGpu<U,A>,SD>: Optimizer<U,DeviceGpu<U,A>> {
    type Output = Adagrad<U,DeviceGpu<U,A>,SD>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Adagrad::<_,DeviceGpu<U,A>,SD>::with_params(&self.device,size,self.lr,self.weight_decay,self.scheduler.clone())
    }
}
/// RMSprop Implementation
pub struct RMSprop<U,D,SD>
    where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U>,
          Self: OptimizerState<U,D> {
    d:PhantomData<D>,
    #[allow(dead_code)]
    size:usize,
    lr:U,
    weight_decay:U,
    alpha:U,
    mu:U,
    gt:<Self as OptimizerState<U,D>>::Type,
    bt:<Self as OptimizerState<U,D>>::Type,
    eps:U,
    scheduler: SD
}
impl<U> RMSprop<U,DeviceCpu<U>,IdentityLR> where U: UnitValue<U> {
    /// Create an instance of RMSprop
    /// # Arguments
    /// * `size` - input size
    pub fn new(device:&DeviceCpu<U>,size:usize) -> RMSprop<U,DeviceCpu<U>,IdentityLR> {
        Self::with_lr(device,size,U::from_f64(0.0001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of RMSprop with Learning rate
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceCpu<U>,size:usize,lr:U) -> RMSprop<U,DeviceCpu<U>,IdentityLR> {
        RMSprop::<U,DeviceCpu<U>,IdentityLR>::with_params(
            device,size,lr,
            U::default(),
            U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            U::default(),
            IdentityLR,
        )
    }
}
impl<U,SD> RMSprop<U,DeviceCpu<U>,SD> where U: UnitValue<U>, SD: Scheduler<U> + Clone {
    /// Create an instance of RMSprop with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `alpha` - alpha
    /// * `mu` - momentum
    pub fn with_params(_:&DeviceCpu<U>,size:usize,lr:U,weight_decay:U,alpha:U,mu:U, scheduler: SD) -> RMSprop<U,DeviceCpu<U>,SD> {
        RMSprop {
            d:PhantomData::<DeviceCpu<U>>,
            size:size,
            lr:lr,
            weight_decay:weight_decay,
            alpha:alpha,
            mu:mu,
            gt:vec![U::default();size].into_boxed_slice(),
            bt:vec![U::default();size].into_boxed_slice(),
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64."),
            scheduler
        }
    }
}
impl<U,SD> Optimizer<U,DeviceCpu<U>> for RMSprop<U,DeviceCpu<U>,SD> where U: UnitValue<U>, SD: Scheduler<U> {
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

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<U,A> RMSprop<U,DeviceGpu<U,A>,IdentityLR>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of RMSprop
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    pub fn new(device:&DeviceGpu<U,A>,size:usize)
        -> Result<RMSprop<U,DeviceGpu<U,A>,IdentityLR>,OptimizerBuildError> {
        Self::with_lr(device,size,U::from_f64(0.0001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of RMSprop with Learning rate
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceGpu<U,A>,size:usize,lr:U)
        -> Result<RMSprop<U,DeviceGpu<U,A>,IdentityLR>,OptimizerBuildError> {
        RMSprop::<U,DeviceGpu<U,A>,IdentityLR>::with_params(
            device,size,
            lr,
            U::default(),
            U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            U::default(),
            IdentityLR
        )
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> RMSprop<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of RMSprop with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `weight_decay` - Weight Decay
    /// * `alpha` - alpha
    /// * `mu` - mu
    pub fn with_params(device:&DeviceGpu<U,A>,size:usize,lr:U,weight_decay:U,alpha:U,mu:U, scheduler: SD)
        -> Result<RMSprop<U,DeviceGpu<U,A>,SD>,OptimizerBuildError> {
        Ok(RMSprop {
            d:PhantomData::<DeviceGpu<U,A>>,
            size:size,
            lr:lr,
            weight_decay:weight_decay,
            alpha:alpha,
            mu:mu,
            gt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            bt:CudaPtr::with_initializer(size, device.get_allocator(), Default::default)?,
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64."),
            scheduler
        })
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Optimizer<U,DeviceGpu<U,A>> for RMSprop<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
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

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
impl<U,SD> OptimizerState<U,DeviceCpu<U>> for RMSprop<U,DeviceCpu<U>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          DeviceCpu<U>: Device<U> {
    type Type = Box<[U]>;
}
#[cfg(feature = "cuda")]
impl<U,A,SD> OptimizerState<U,DeviceGpu<U,A>> for RMSprop<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    type Type = CudaPtr<U,A>;
}
impl<U,SD> Persistence<U,TextFilePersistence<U>,Specialized> for RMSprop<U,DeviceCpu<U>,SD>
    where U: UnitValue<U> + FromStr,
          SD: Scheduler<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        for &gt in self.gt.iter() {
            persistence.write(UnitOrMarker::Unit(gt));
        }

        for &bt in self.bt.iter() {
            persistence.write(UnitOrMarker::Unit(bt));
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), ModelLoadError> {
        for gt in self.gt.iter_mut() {
            *gt = persistence.read()?;
        }

        for bt in self.bt.iter_mut() {
            *bt = persistence.read()?;
        }
        Ok(())
    }
}
impl<T,U,SD> Persistence<U,T,Linear> for RMSprop<U,DeviceCpu<U>,SD>
    where T: LinearPersistence<U>,
          U: UnitValue<U> + Debug + Default,
          SD: Scheduler<U> {
    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        for &gt in self.gt.iter() {
            persistence.write(gt)?;
        }

        for &bt in self.bt.iter() {
            persistence.write(bt)?;
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        for gt in self.gt.iter_mut() {
            *gt = persistence.read()?;
        }

        for bt in self.bt.iter_mut() {
            *bt = persistence.read()?;
        }
        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Persistence<U,TextFilePersistence<U>,Specialized> for RMSprop<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U> + FromStr,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: ReadMemory<U> + WriteMemory<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        for &gt in self.gt.read_to_vec()?.iter() {
            persistence.write(UnitOrMarker::Unit(gt));
        }

        for &bt in self.bt.read_to_vec()?.iter() {
            persistence.write(UnitOrMarker::Unit(bt));
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), ModelLoadError> {
        let mut gt = vec![U::default();self.size];

        for gt in gt.iter_mut() {
            *gt = persistence.read()?;
        }

        self.gt.memcpy(gt.as_ptr(),self.size)?;

        let mut bt = vec![U::default();self.size];

        for bt in bt.iter_mut() {
            *bt = persistence.read()?;
        }

        self.bt.memcpy(bt.as_ptr(),self.size)?;

        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<T,U,A,SD> Persistence<U,T,Linear> for RMSprop<U,DeviceGpu<U,A>,SD>
    where T: LinearPersistence<U>,
          U: UnitValue<U> + Debug + Default,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: ReadMemory<U> + WriteMemory<U> {
    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        for &gt in self.gt.read_to_vec()?.iter() {
            persistence.write(gt)?;
        }

        for &bt in self.bt.read_to_vec()?.iter() {
            persistence.write(bt)?;
        }
        Ok(())
    }

    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        let mut gt = vec![U::default();self.size];

        for gt in gt.iter_mut() {
            *gt = persistence.read()?;
        }

        self.gt.memcpy(gt.as_ptr(),self.size)?;

        let mut bt = vec![U::default();self.size];

        for bt in bt.iter_mut() {
            *bt = persistence.read()?;
        }

        self.bt.memcpy(bt.as_ptr(),self.size)?;

        Ok(())
    }
}
/// Implementation of a builder to generate RMSprop optimizers
pub struct RMSpropBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone {
    lr:U,
    weight_decay:U,
    alpha:U,
    mu:U,
    device:D,
    scheduler: SD
}
impl<U,D> RMSpropBuilder<U,D,IdentityLR> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of RMSpropBuilder with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    pub fn new(device:&D) -> RMSpropBuilder<U,D,IdentityLR> {
        RMSpropBuilder {
            lr:U::from_f64(0.0001f64).expect("Error in type conversion from f64."),
            weight_decay:U::default(),
            alpha:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            mu:U::default(),
            device:device.clone(),
            scheduler: IdentityLR
        }
    }
}
impl<U,D,SD> RMSpropBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone {
    /// Replaces the value of field lr in RMSpropBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> RMSpropBuilder<U,D,SD> {
        RMSpropBuilder {
            device:self.device,
            lr:lr,
            weight_decay:self.weight_decay,
            alpha:self.alpha,
            mu:self.mu,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field weight_decay in MomentumSGDBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Learning rate
    pub fn weight_decay(self,weight_decay:U) -> RMSpropBuilder<U,D,SD> {
        RMSpropBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:weight_decay,
            alpha:self.alpha,
            mu:self.mu,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field alpha in RMSpropBuilder with the passed value and returns it.
    /// # Arguments
    /// * `alpha` - alpha
    pub fn alpha(self,alpha:U) -> RMSpropBuilder<U,D,SD> {
        RMSpropBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            alpha:alpha,
            mu:self.mu,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field mu in RMSpropBuilder with the passed value and returns it.
    /// # Arguments
    /// * `mu` - momentum
    pub fn mu(self,mu:U) -> RMSpropBuilder<U,D,SD> {
        RMSpropBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            alpha:self.alpha,
            mu:mu,
            scheduler:self.scheduler
        }
    }


    /// Replaces the value of field scheduler in RMSpropBuilder with the passed value and returns it.
    /// # Arguments
    /// * `scheduler` - learning rate scheduler
    pub fn scheduler<S>(self, scheduler: S) -> RMSpropBuilder<U,D,S>
        where S: Scheduler<U> + Clone {
        RMSpropBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            alpha:self.alpha,
            mu:self.mu,
            scheduler
        }
    }
}
impl<U,SD> OptimizerBuilder<U,DeviceCpu<U>> for RMSpropBuilder<U,DeviceCpu<U>,SD>
    where U: UnitValue<U>, SD: Scheduler<U> + Clone, RMSprop<U,DeviceCpu<U>,SD>: Optimizer<U,DeviceCpu<U>> {
    type Output = RMSprop<U,DeviceCpu<U>,SD>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(RMSprop::<_,DeviceCpu<U>,SD>::with_params(&self.device,size,self.lr,self.weight_decay,self.alpha,self.mu,self.scheduler.clone()))
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> OptimizerBuilder<U,DeviceGpu<U,A>> for RMSpropBuilder<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U> + Clone,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          RMSprop<U,DeviceGpu<U,A>,SD>: Optimizer<U,DeviceGpu<U,A>> {
    type Output = RMSprop<U,DeviceGpu<U,A>,SD>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        RMSprop::<_,DeviceGpu<U,A>,SD>::with_params(&self.device,size,self.lr,self.weight_decay,self.alpha,self.mu,self.scheduler.clone())
    }
}
/// Adam Implementation
pub struct Adam<U,D,SD>
    where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U>,
          Self: OptimizerState<U,D> {
    d:PhantomData<D>,
    #[allow(dead_code)]
    size:usize,
    lr:U,
    weight_decay:U,
    mt:<Self as OptimizerState<U,D>>::Type,
    vt:<Self as OptimizerState<U,D>>::Type,
    b1:U,
    b2:U,
    b1t:U,
    b2t:U,
    eps:U,
    scheduler: SD
}
impl<U> Adam<U,DeviceCpu<U>,IdentityLR> where U: UnitValue<U> {
    /// Create an instance of Adam
    /// # Arguments
    /// * `size` - input size
    pub fn new(device:&DeviceCpu<U>,size:usize) -> Adam<U,DeviceCpu<U>,IdentityLR> {
        Self::with_lr(device,size,U::from_f64(0.001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of Adam with Learning rate
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceCpu<U>,size:usize,lr:U) -> Adam<U,DeviceCpu<U>,IdentityLR> {
        Adam::<U,DeviceCpu<U>,IdentityLR>::with_params(device,size,
                          lr,
                          U::default(),
                          U::from_f64(0.9f64).expect("Error in type conversion from f64."),
                          U::from_f64(0.999f64).expect("Error in type conversion from f64."),
                          IdentityLR)
    }
}
impl<U,SD> Adam<U,DeviceCpu<U>,SD> where U: UnitValue<U>, SD: Scheduler<U> + Clone {
    /// Create an instance of Adam with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `b1` - beta1
    /// * `b2` - beta2
    pub fn with_params(_:&DeviceCpu<U>,size:usize,lr:U,weight_decay:U,b1:U,b2:U, scheduler: SD) -> Adam<U,DeviceCpu<U>,SD> {
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
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64."),
            scheduler
        }
    }
}
impl<U,SD> Optimizer<U,DeviceCpu<U>> for Adam<U,DeviceCpu<U>,SD> where U: UnitValue<U>, SD: Scheduler<U> {
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

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<U,A> Adam<U,DeviceGpu<U,A>,IdentityLR>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of Adam
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    pub fn new(device:&DeviceGpu<U,A>,size:usize) -> Result<Adam<U,DeviceGpu<U,A>,IdentityLR>,OptimizerBuildError> {
        Self::with_lr(device,size,U::from_f64(0.001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of Adam with Learning rate
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceGpu<U,A>,size:usize,lr:U) ->Result<Adam<U,DeviceGpu<U,A>,IdentityLR>,OptimizerBuildError> {
        Adam::<U,DeviceGpu<U,A>,IdentityLR>::with_params(device,size,lr,
                          U::default(),
                          U::from_f64(0.9f64).expect("Error in type conversion from f64."),
                          U::from_f64(0.999f64).expect("Error in type conversion from f64."),
                          IdentityLR
        )
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Adam<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of Adam with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `b1` - beta1
    /// * `b2` - beta2
    pub fn with_params(device:&DeviceGpu<U,A>,size:usize,lr:U,weight_decay:U,b1:U,b2:U, scheduler: SD) ->Result<Adam<U,DeviceGpu<U,A>,SD>,OptimizerBuildError> {
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
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64."),
            scheduler
        })
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Optimizer<U,DeviceGpu<U,A>> for Adam<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
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

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
impl<U,SD> OptimizerState<U,DeviceCpu<U>> for Adam<U,DeviceCpu<U>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          DeviceCpu<U>: Device<U> {
    type Type = Box<[U]>;
}
#[cfg(feature = "cuda")]
impl<U,A,SD> OptimizerState<U,DeviceGpu<U,A>> for Adam<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    type Type = CudaPtr<U,A>;
}
impl<U,SD> Persistence<U,TextFilePersistence<U>,Specialized> for Adam<U,DeviceCpu<U>,SD>
    where U: UnitValue<U> + FromStr,
          SD: Scheduler<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        for &mt in self.mt.iter() {
            persistence.write(UnitOrMarker::Unit(mt));
        }

        for &vt in self.vt.iter() {
            persistence.write(UnitOrMarker::Unit(vt));
        }

        persistence.write(UnitOrMarker::Unit(self.b1t));
        persistence.write(UnitOrMarker::Unit(self.b2t));

        Ok(())
    }

    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), ModelLoadError> {
        for mt in self.mt.iter_mut() {
            *mt = persistence.read()?;
        }

        for vt in self.vt.iter_mut() {
            *vt = persistence.read()?;
        }

        self.b1t = persistence.read()?;
        self.b2t = persistence.read()?;

        Ok(())
    }
}
impl<T,U,SD> Persistence<U,T,Linear> for Adam<U,DeviceCpu<U>,SD>
    where T: LinearPersistence<U>,
          U: UnitValue<U> + Debug + Default,
          SD: Scheduler<U> {
    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        for &mt in self.mt.iter() {
            persistence.write(mt)?;
        }

        for &vt in self.vt.iter() {
            persistence.write(vt)?;
        }

        persistence.write(self.b1t)?;
        persistence.write(self.b2t)?;

        Ok(())
    }

    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        for mt in self.mt.iter_mut() {
            *mt = persistence.read()?;
        }

        for vt in self.vt.iter_mut() {
            *vt = persistence.read()?;
        }

        self.b1t = persistence.read()?;
        self.b2t = persistence.read()?;

        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Persistence<U,TextFilePersistence<U>,Specialized> for Adam<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U> + FromStr,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: ReadMemory<U> + WriteMemory<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        for &mt in self.mt.read_to_vec()?.iter() {
            persistence.write(UnitOrMarker::Unit(mt));
        }

        for &vt in self.vt.read_to_vec()?.iter() {
            persistence.write(UnitOrMarker::Unit(vt));
        }

        persistence.write(UnitOrMarker::Unit(self.b1t));
        persistence.write(UnitOrMarker::Unit(self.b2t));

        Ok(())
    }

    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), ModelLoadError> {
        let mut mt = vec![U::default();self.size];

        for mt in mt.iter_mut() {
            *mt = persistence.read()?;
        }

        self.mt.memcpy(mt.as_ptr(),self.size)?;

        let mut vt = vec![U::default();self.size];

        for vt in vt.iter_mut() {
            *vt = persistence.read()?;
        }

        self.vt.memcpy(vt.as_ptr(),self.size)?;

        self.b1t = persistence.read()?;
        self.b2t = persistence.read()?;

        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<T,U,A,SD> Persistence<U,T,Linear> for Adam<U,DeviceGpu<U,A>,SD>
    where T: LinearPersistence<U>,
          U: UnitValue<U> + Debug + Default,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: ReadMemory<U> + WriteMemory<U> {
    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        for &mt in self.mt.read_to_vec()?.iter() {
            persistence.write(mt)?;
        }

        for &vt in self.vt.read_to_vec()?.iter() {
            persistence.write(vt)?;
        }

        persistence.write(self.b1t)?;
        persistence.write(self.b2t)?;

        Ok(())
    }

    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        let mut mt = vec![U::default();self.size];

        for mt in mt.iter_mut() {
            *mt = persistence.read()?;
        }

        self.mt.memcpy(mt.as_ptr(),self.size)?;

        let mut vt = vec![U::default();self.size];

        for vt in vt.iter_mut() {
            *vt = persistence.read()?;
        }

        self.vt.memcpy(vt.as_ptr(),self.size)?;

        self.b1t = persistence.read()?;
        self.b2t = persistence.read()?;

        Ok(())
    }
}
/// Implementation of a builder to generate Adam optimizers
pub struct AdamBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone {
    lr:U,
    weight_decay:U,
    b1:U,
    b2:U,
    device:D,
    scheduler: SD
}
impl<U,D> AdamBuilder<U,D,IdentityLR> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of AdamBuilder with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    pub fn new(device:&D) -> AdamBuilder<U,D,IdentityLR> {
        AdamBuilder {
            lr:U::from_f64(0.001f64).expect("Error in type conversion from f64."),
            weight_decay:U::default(),
            b1:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            b2:U::from_f64(0.999f64).expect("Error in type conversion from f64."),
            device:device.clone(),
            scheduler: IdentityLR
        }
    }
}
impl<U,D,SD> AdamBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone {
    /// Replaces the value of field lr in AdamBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> AdamBuilder<U,D,SD> {
        AdamBuilder {
            device:self.device,
            lr:lr,
            weight_decay:self.weight_decay,
            b1:self.b1,
            b2:self.b2,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field weight_decay in AdamBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Learning rate
    pub fn weight_decay(self,weight_decay:U) -> AdamBuilder<U,D,SD> {
        AdamBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:weight_decay,
            b1:self.b1,
            b2:self.b2,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field b1 in AdamBuilder with the passed value and returns it.
    /// # Arguments
    /// * `b1` - b1
    pub fn b1(self,b1:U) -> AdamBuilder<U,D,SD> {
        AdamBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            b1:b1,
            b2:self.b2,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field b2 in AdamBuilder with the passed value and returns it.
    /// # Arguments
    /// * `b2` - b2
    pub fn b2(self,b2:U) -> AdamBuilder<U,D,SD> {
        AdamBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            b1:self.b1,
            b2:b2,
            scheduler:self.scheduler
        }
    }


    /// Replaces the value of field scheduler in AdamBuilder with the passed value and returns it.
    /// # Arguments
    /// * `scheduler` - learning rate scheduler
    pub fn scheduler<S>(self, scheduler: S) -> AdamBuilder<U,D,S>
        where S: Scheduler<U> + Clone {
        AdamBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            b1:self.b1,
            b2:self.b2,
            scheduler
        }
    }
}
impl<U,SD> OptimizerBuilder<U,DeviceCpu<U>> for AdamBuilder<U,DeviceCpu<U>,SD>
    where U: UnitValue<U>, SD: Scheduler<U> + Clone, Adam<U,DeviceCpu<U>,SD>: Optimizer<U,DeviceCpu<U>> {
    type Output = Adam<U,DeviceCpu<U>,SD>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(Adam::<_,DeviceCpu<U>,SD>::with_params(&self.device,size,self.lr,self.weight_decay,self.b1,self.b2,self.scheduler.clone()))
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> OptimizerBuilder<U,DeviceGpu<U,A>> for AdamBuilder<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U> + Clone,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          Adam<U,DeviceGpu<U,A>,SD>: Optimizer<U,DeviceGpu<U,A>> {
    type Output = Adam<U,DeviceGpu<U,A>,SD>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Adam::<_,DeviceGpu<U,A>,SD>::with_params(&self.device,size,self.lr,self.weight_decay,self.b1,self.b2,self.scheduler.clone())
    }
}
/// AdamW Implementation
pub struct AdamW<U,D,SD>
    where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U>,
          Self: OptimizerState<U,D> {
    d:PhantomData<D>,
    #[allow(dead_code)]
    size:usize,
    lr:U,
    weight_decay:U,
    mt:<Self as OptimizerState<U,D>>::Type,
    vt:<Self as OptimizerState<U,D>>::Type,
    b1:U,
    b2:U,
    b1t:U,
    b2t:U,
    eps:U,
    scheduler: SD
}
impl<U> AdamW<U,DeviceCpu<U>,IdentityLR> where U: UnitValue<U> {
    /// Create an instance of AdamW
    /// # Arguments
    /// * `size` - input size
    pub fn new(device:&DeviceCpu<U>,size:usize) -> AdamW<U,DeviceCpu<U>,IdentityLR> {
        Self::with_lr(device,size,U::from_f64(0.001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of AdamW with Learning rate
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceCpu<U>,size:usize,lr:U) -> AdamW<U,DeviceCpu<U>,IdentityLR> {
        AdamW::<U,DeviceCpu<U>,IdentityLR>::with_params(device,size,
                                             lr,
                                             U::default(),
                                             U::from_f64(0.9f64).expect("Error in type conversion from f64."),
                                             U::from_f64(0.999f64).expect("Error in type conversion from f64."),
                                             IdentityLR)
    }
}
impl<U,SD> AdamW<U,DeviceCpu<U>,SD> where U: UnitValue<U>, SD: Scheduler<U> + Clone {
    /// Create an instance of AdamW with additional parameters other than the default values
    /// # Arguments
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `b1` - beta1
    /// * `b2` - beta2
    pub fn with_params(_:&DeviceCpu<U>,size:usize,lr:U,weight_decay:U,b1:U,b2:U, scheduler: SD) -> AdamW<U,DeviceCpu<U>,SD> {
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
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64."),
            scheduler
        }
    }
}
impl<U,SD> Optimizer<U,DeviceCpu<U>> for AdamW<U,DeviceCpu<U>,SD> where U: UnitValue<U>, SD: Scheduler<U> {
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
            *w = *w - weight_decay * *w;

            *mt = b1 * *mt + (U::one() - self.b1) * e;
            *vt = b2 * *vt + (U::one() - self.b2) * e * e;

            *w = *w - a * (*mt / (U::one() - b1t)) / ((*vt / (U::one() - b2t)) + self.eps).sqrt();
        }

        self.b1t = b1t * b1;
        self.b2t = b2t * b2;

        Ok(())
    }

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<U,A> AdamW<U,DeviceGpu<U,A>,IdentityLR>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of AdamW
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    pub fn new(device:&DeviceGpu<U,A>,size:usize) -> Result<AdamW<U,DeviceGpu<U,A>,IdentityLR>,OptimizerBuildError> {
        Self::with_lr(device,size,U::from_f64(0.001f64).expect("Error in type conversion from f64."))
    }

    /// Create an instance of AdamW with Learning rate
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    pub fn with_lr(device:&DeviceGpu<U,A>,size:usize,lr:U) ->Result<AdamW<U,DeviceGpu<U,A>,IdentityLR>,OptimizerBuildError> {
        AdamW::<U,DeviceGpu<U,A>,IdentityLR>::with_params(device,size,lr,
                                               U::default(),
                                               U::from_f64(0.9f64).expect("Error in type conversion from f64."),
                                               U::from_f64(0.999f64).expect("Error in type conversion from f64."),
                                               IdentityLR
        )
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> AdamW<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    /// Create an instance of AdamW with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    /// * `size` - input size
    /// * `lr` - Learning rate
    /// * `b1` - beta1
    /// * `b2` - beta2
    pub fn with_params(device:&DeviceGpu<U,A>,size:usize,lr:U,weight_decay:U,b1:U,b2:U, scheduler: SD) ->Result<AdamW<U,DeviceGpu<U,A>,SD>,OptimizerBuildError> {
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
            eps:U::from_f64(1e-8f64).expect("Error in type conversion from f64."),
            scheduler
        })
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Optimizer<U,DeviceGpu<U,A>> for AdamW<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
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

    fn on_step(&mut self, step: usize) -> Result<(),TrainingError> {
        self.lr = self.scheduler.schedule(self.lr, step)?;
        Ok(())
    }
}
impl<U,SD> OptimizerState<U,DeviceCpu<U>> for AdamW<U,DeviceCpu<U>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          DeviceCpu<U>: Device<U> {
    type Type = Box<[U]>;
}
#[cfg(feature = "cuda")]
impl<U,A,SD> OptimizerState<U,DeviceGpu<U,A>> for AdamW<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U> {
    type Type = CudaPtr<U,A>;
}
impl<U,SD> Persistence<U,TextFilePersistence<U>,Specialized> for AdamW<U,DeviceCpu<U>,SD>
    where U: UnitValue<U> + FromStr,
          SD: Scheduler<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        for &mt in self.mt.iter() {
            persistence.write(UnitOrMarker::Unit(mt));
        }

        for &vt in self.vt.iter() {
            persistence.write(UnitOrMarker::Unit(vt));
        }

        persistence.write(UnitOrMarker::Unit(self.b1t));
        persistence.write(UnitOrMarker::Unit(self.b2t));

        Ok(())
    }

    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), ModelLoadError> {
        for mt in self.mt.iter_mut() {
            *mt = persistence.read()?;
        }

        for vt in self.vt.iter_mut() {
            *vt = persistence.read()?;
        }

        self.b1t = persistence.read()?;
        self.b2t = persistence.read()?;

        Ok(())
    }
}
impl<T,U,SD> Persistence<U,T,Linear> for AdamW<U,DeviceCpu<U>,SD>
    where T: LinearPersistence<U>,
          U: UnitValue<U> + Debug + Default,
          SD: Scheduler<U> {
    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        for &mt in self.mt.iter() {
            persistence.write(mt)?;
        }

        for &vt in self.vt.iter() {
            persistence.write(vt)?;
        }

        persistence.write(self.b1t)?;
        persistence.write(self.b2t)?;

        Ok(())
    }

    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        for mt in self.mt.iter_mut() {
            *mt = persistence.read()?;
        }

        for vt in self.vt.iter_mut() {
            *vt = persistence.read()?;
        }

        self.b1t = persistence.read()?;
        self.b2t = persistence.read()?;

        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> Persistence<U,TextFilePersistence<U>,Specialized> for AdamW<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U> + FromStr,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: ReadMemory<U> + WriteMemory<U>,
          ModelLoadError: From<<U as FromStr>::Err> {
    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        for &mt in self.mt.read_to_vec()?.iter() {
            persistence.write(UnitOrMarker::Unit(mt));
        }

        for &vt in self.vt.read_to_vec()?.iter() {
            persistence.write(UnitOrMarker::Unit(vt));
        }

        persistence.write(UnitOrMarker::Unit(self.b1t));
        persistence.write(UnitOrMarker::Unit(self.b2t));

        Ok(())
    }

    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), ModelLoadError> {
        let mut mt = vec![U::default();self.size];

        for mt in mt.iter_mut() {
            *mt = persistence.read()?;
        }

        self.mt.memcpy(mt.as_ptr(),self.size)?;

        let mut vt = vec![U::default();self.size];

        for vt in vt.iter_mut() {
            *vt = persistence.read()?;
        }

        self.vt.memcpy(vt.as_ptr(),self.size)?;

        self.b1t = persistence.read()?;
        self.b2t = persistence.read()?;

        Ok(())
    }
}
#[cfg(feature = "cuda")]
impl<T,U,A,SD> Persistence<U,T,Linear> for AdamW<U,DeviceGpu<U,A>,SD>
    where T: LinearPersistence<U>,
          U: UnitValue<U> + Debug + Default,
          A: CudaAllocator + 'static,
          SD: Scheduler<U>,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: ReadMemory<U> + WriteMemory<U> {
    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        for &mt in self.mt.read_to_vec()?.iter() {
            persistence.write(mt)?;
        }

        for &vt in self.vt.read_to_vec()?.iter() {
            persistence.write(vt)?;
        }

        persistence.write(self.b1t)?;
        persistence.write(self.b2t)?;

        Ok(())
    }

    fn load(&mut self, persistence: &mut T) -> Result<(), ModelLoadError> {
        let mut mt = vec![U::default();self.size];

        for mt in mt.iter_mut() {
            *mt = persistence.read()?;
        }

        self.mt.memcpy(mt.as_ptr(),self.size)?;

        let mut vt = vec![U::default();self.size];

        for vt in vt.iter_mut() {
            *vt = persistence.read()?;
        }

        self.vt.memcpy(vt.as_ptr(),self.size)?;

        self.b1t = persistence.read()?;
        self.b2t = persistence.read()?;

        Ok(())
    }
}
/// Implementation of a builder to generate AdamW optimizers
pub struct AdamWBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone {
    lr:U,
    weight_decay:U,
    b1:U,
    b2:U,
    device:D,
    scheduler: SD
}
impl<U,D> AdamWBuilder<U,D,IdentityLR> where U: UnitValue<U>, D: Device<U> {
    /// Create an instance of AdamWBuilder with additional parameters other than the default values
    /// # Arguments
    /// * `device` - device
    pub fn new(device:&D) -> AdamWBuilder<U,D,IdentityLR> {
        AdamWBuilder {
            lr:U::from_f64(0.001f64).expect("Error in type conversion from f64."),
            weight_decay:U::default(),
            b1:U::from_f64(0.9f64).expect("Error in type conversion from f64."),
            b2:U::from_f64(0.999f64).expect("Error in type conversion from f64."),
            device:device.clone(),
            scheduler: IdentityLR
        }
    }
}
impl<U,D,SD> AdamWBuilder<U,D,SD> where U: UnitValue<U>, D: Device<U>, SD: Scheduler<U> + Clone {
    /// Replaces the value of field lr in AdamWBuilder with the passed value and returns it.
    /// # Arguments
    /// * `lr` - Learning rate
    pub fn lr(self,lr:U) -> AdamWBuilder<U,D,SD> {
        AdamWBuilder {
            device:self.device,
            lr:lr,
            weight_decay:self.weight_decay,
            b1:self.b1,
            b2:self.b2,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field weight_decay in AdamWBuilder with the passed value and returns it.
    /// # Arguments
    /// * `weight_decay` - Learning rate
    pub fn weight_decay(self,weight_decay:U) -> AdamWBuilder<U,D,SD> {
        AdamWBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:weight_decay,
            b1:self.b1,
            b2:self.b2,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field b1 in AdamWBuilder with the passed value and returns it.
    /// # Arguments
    /// * `b1` - b1
    pub fn b1(self,b1:U) -> AdamWBuilder<U,D,SD> {
        AdamWBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            b1:b1,
            b2:self.b2,
            scheduler:self.scheduler
        }
    }

    /// Replaces the value of field b2 in AdamWBuilder with the passed value and returns it.
    /// # Arguments
    /// * `b2` - b2
    pub fn b2(self,b2:U) -> AdamWBuilder<U,D,SD> {
        AdamWBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            b1:self.b1,
            b2:b2,
            scheduler:self.scheduler
        }
    }


    /// Replaces the value of field scheduler in AdamWBuilder with the passed value and returns it.
    /// # Arguments
    /// * `scheduler` - learning rate scheduler
    pub fn scheduler<S>(self, scheduler: S) -> AdamWBuilder<U,D,S>
        where S: Scheduler<U> + Clone {
        AdamWBuilder {
            device:self.device,
            lr:self.lr,
            weight_decay:self.weight_decay,
            b1:self.b1,
            b2:self.b2,
            scheduler
        }
    }
}
impl<U,SD> OptimizerBuilder<U,DeviceCpu<U>> for AdamWBuilder<U,DeviceCpu<U>,SD>
    where U: UnitValue<U>, SD: Scheduler<U> + Clone, AdamW<U,DeviceCpu<U>,SD>: Optimizer<U,DeviceCpu<U>> {
    type Output = AdamW<U,DeviceCpu<U>,SD>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        Ok(AdamW::<_,DeviceCpu<U>,SD>::with_params(&self.device,size,self.lr,self.weight_decay,self.b1,self.b2,self.scheduler.clone()))
    }
}
#[cfg(feature = "cuda")]
impl<U,A,SD> OptimizerBuilder<U,DeviceGpu<U,A>> for AdamWBuilder<U,DeviceGpu<U,A>,SD>
    where U: UnitValue<U>,
          SD: Scheduler<U> + Clone,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          DeviceGpu<U,A>: Device<U>,
          AdamW<U,DeviceGpu<U,A>,SD>: Optimizer<U,DeviceGpu<U,A>> {
    type Output = AdamW<U,DeviceGpu<U,A>,SD>;

    fn build(&self,size:usize) -> Result<Self::Output,OptimizerBuildError> {
        AdamW::<_,DeviceGpu<U,A>,SD>::with_params(&self.device,size,self.lr,self.weight_decay,self.b1,self.b2,self.scheduler.clone())
    }
}

