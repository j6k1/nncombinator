//! Implementation of a Learning Rate Scheduler

use std::f64::consts::PI;
use std::marker::PhantomData;
use std::sync::Arc;
use try_from_primitive::TryFromPrimitive;
use crate::error::TrainingError;
use crate::ope::UnitValue;

/// Trait that defines learning rate scheduler.
pub trait Scheduler<U> where U: UnitValue<U> {
    /// Retrieve the adjusted learning rate
    /// # Arguments
    /// * `lr` - learning rate
    /// * `step` - current training step
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn schedule(&mut self, lr: U, step: usize) -> Result<U,TrainingError>;

    /// Returns a combined scheduler that executes two schedulers sequentially.
    /// # Arguments
    /// * `milestone` - Threshold for the number of steps before delegating processing to the next scheduler
    /// * `next_scheduler` - Scheduler to be executed after the milestone
    ///
    fn seq<NS>(self, milestone: usize, next_scheduler: NS) -> SequentialLR<U,Self,NS>
        where NS: Scheduler<U> + Clone + Sized + 'static,
              Self:  Clone + Sized + 'static {

        SequentialLR::new(self, next_scheduler, milestone)
    }
}
/// Scheduler that does not update the learning rate
#[derive(Clone)]
pub struct IdentityLR;
impl<U> Scheduler<U> for IdentityLR where U: UnitValue<U> {
    fn schedule(&mut self, lr: U, _: usize) -> Result<U,TrainingError> {
        Ok(lr)
    }
}
/// A scheduler that adjusts the learning rate using a callback that calculates the ratio relative
/// to the initial learning rate
pub struct LambdaLR<U,F> where U: UnitValue<U>, F: Fn(usize) -> Result<U,TrainingError> {
    base_lr:U,
    callback: Arc<F>
}
impl<U,F> LambdaLR<U,F> where U: UnitValue<U>, F: Fn(usize) -> Result<U,TrainingError> {
    /// Create a new LambdaLR
    ///
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `callback` - Callback function that calculates the ratio relative to the initial learning rate
    /// # Returns
    /// * `LambdaLR` - A new LambdaLR
    pub fn new(base_lr:U, callback: F) -> Self {
        LambdaLR {
            base_lr,
            callback: Arc::new(callback)
        }
    }
}
impl<U,F> Scheduler<U> for LambdaLR<U,F> where U: UnitValue<U>, F: Fn(usize) -> Result<U,TrainingError> {
    fn schedule(&mut self, _: U, step: usize) -> Result<U,TrainingError> {
        Ok(self.base_lr * (self.callback)(step)?)
    }
}
impl<U,F> Clone for LambdaLR<U,F> where U: UnitValue<U>, F: Fn(usize) -> Result<U,TrainingError> + Clone {
    fn clone(&self) -> Self {
        LambdaLR {
            base_lr: self.base_lr,
            callback: Arc::clone(&self.callback)
        }
    }
}
/// A learning rate decay scheduler using gamma per step size
#[derive(Clone)]
pub struct StepLR<U> where U: UnitValue<U> {
    step_size: usize,
    gamma: U
}
impl<U> StepLR<U> where U: UnitValue<U> {
    /// Create a new instance of `StepLR` with the specified number of steps and decay rate.
    ///
    /// # Arguments
    /// * `step_size` - Number of steps to decay the learning rate. Decays every this many steps.
    ///
    /// * `gamma` - Decay rate when decaying the learning rate
    ///
    /// # Returns A new StepLR instance configured with the specified step size and decay rate.
    ///
    /// # Example
    /// ```
    /// use nncombinator::scheduler::{StepLR};
    /// let step_size = 75;
    /// let gamma = 0.3;
    /// let linear_warmup_lr = StepLR::new(step_size,gamma);
    /// ```
    pub fn new(step_size: usize, gamma: U) -> Self {
        StepLR { step_size, gamma }
    }
}
impl<U> Scheduler<U> for StepLR<U> where U: UnitValue<U> {
    fn schedule(&mut self, lr: U, step: usize) -> Result<U,TrainingError> {
        Ok(if step > 0 && step % self.step_size == 0 {
            lr * self.gamma
        } else {
            lr
        })
    }
}
/// A scheduler that gradually increases the learning rate from a low starting point
#[derive(Clone)]
pub struct LinearWarmupLR<U> where U: UnitValue<U> {
    warmup_steps: usize,
    base_lr: U,
    start_factor: U,
}
impl<U> LinearWarmupLR<U> where U: UnitValue<U> {
    /// Creates a new instance of `LinearWarmupLR` with the specified number of warmup steps
    /// and a base learning rate.
    ///
    /// # Arguments
    /// * `warmup_steps` - The number of steps over which the learning rate will linearly increase
    ///                   from zero to the base learning rate.
    /// * `base_lr` - The base learning rate value to be achieved after the warmup period.
    /// * `start_factor` - Initial learning rate coefficient
    ///
    /// # Returns
    /// A new `LinearWarmupLR` instance configured with the provided warmup steps and base learning rate.
    ///
    /// # Example
    /// ```
    /// use nncombinator::scheduler::LinearWarmupLR;
    /// let warmup_steps = 1000;
    /// let base_lr = 0.01;
    /// let start_factor = 0.1;
    /// let linear_warmup_lr = LinearWarmupLR::new(warmup_steps, base_lr, start_factor);
    /// ```
   pub fn new(warmup_steps: usize, base_lr: U, start_factor: U) -> Self {
        LinearWarmupLR {
            warmup_steps,
            base_lr,
            start_factor
        }
    }
}
impl<U> Scheduler<U> for LinearWarmupLR<U> where U: UnitValue<U> + TryFromPrimitive {
    fn schedule(&mut self, _: U, step: usize) -> Result<U,TrainingError> {
        Ok(if step < self.warmup_steps {
            self.base_lr * (self.start_factor + (U::one() - self.start_factor) *
                (U::try_from_f64(step as f64)? / U::try_from_f64(self.warmup_steps as f64)?)
            )
        } else {
            self.base_lr
        })
    }
}
/// Scheduler using cosine annealing schedule.
#[derive(Clone)]
pub struct CosineAnnealingLR<U> where U: UnitValue<U> {
    total_steps: usize,
    eta_min: U
}
impl<U> CosineAnnealingLR<U> where U: UnitValue<U> {
    /// Creates a new instance of `CosineAnnealingLR`.
    ///
    /// This method initializes the struct with the total number of steps and the
    /// minimum learning rate (eta_min) that the scheduler should decay towards.
    ///
    /// # Arguments
    /// * `total_steps`: - The total number of steps over which the learning rate
    ///   will be annealed using a cosine schedule.
    /// * `eta_min` - The minimum learning rate value to which the learning rate
    ///   will decay during the schedule.
    ///
    /// # Returns
    /// A new instance of `CosineAnnealingLR` initialized with the specified
    /// total steps and minimum learning rate.
    ///
    /// # Example
    /// ```
    /// use your_crate::CosineAnnealingLR;
    ///
    /// let scheduler = CosineAnnealingLR::new(100, 0.01);
    /// ```
    pub fn new(total_steps: usize, eta_min: U) -> Self {
        CosineAnnealingLR {
            total_steps,
            eta_min
        }
    }
}
impl<U> Scheduler<U> for CosineAnnealingLR<U> where U: UnitValue<U> {
    fn schedule(&mut self, lr: U, step: usize) -> Result<U, TrainingError> {
        Ok(self.eta_min + (lr - self.eta_min) * (
            (U::one() +
                (
                    (U::try_from_usize(step)? + U::one()) * U::try_from_f64(PI)? /
                     U::try_from_usize(self.total_steps)?
                ).cos()
            ) /
            (U::one() +
                (
                    U::try_from_usize(step)? * U::try_from_f64(PI)? /
                    U::try_from_usize(self.total_steps)?
                ).cos()
            )
        ))
    }
}
/// Scheduler that executes two schedulers sequentially.
pub struct SequentialLR<U,PS,S> where U: UnitValue<U>, PS: Scheduler<U> + Clone, S: Scheduler<U> + Clone {
    prev_scheduler: PS,
    next_scheduler: S,
    milestone: usize,
    u:PhantomData<U>
}

impl<U,PS,S> Clone for SequentialLR<U,PS,S>
    where U: UnitValue<U>, PS: Scheduler<U> + Clone, S: Scheduler<U> + Clone {
    fn clone(&self) -> Self {
        SequentialLR {
            prev_scheduler: self.prev_scheduler.clone(),
            next_scheduler: self.next_scheduler.clone(),
            milestone: self.milestone,
            u: PhantomData
        }
    }
}
impl<U,PS,S> SequentialLR<U,PS,S> where U: UnitValue<U>, PS: Scheduler<U> + Clone, S: Scheduler<U> + Clone {
    /// Creates a new instance of `SequentialLR`.
    ///
    /// This function initializes a sequential learning rate scheduler, which transitions
    /// between two schedulers (`prev_scheduler` and `next_scheduler`) based on a specified milestone.
    ///
    /// # Arguments
    /// * `prev_scheduler` - The learning rate scheduler to be used prior to the milestone.
    /// * `next_scheduler` - The learning rate scheduler to be used after the milestone.
    /// * `milestone` - An integer value that defines the transition point between `prev_scheduler` and `next_scheduler`.
    ///
    /// # Returns
    /// A new `SequentialLR` instance containing the provided schedulers and milestone.
    ///
    /// # Example
    /// ```
    /// use nncombinator::scheduler::LinearWarmupLR;
    /// use nncombinator::scheduler::IdentityLR;
    /// let warmup_steps = 1000;
    /// let base_lr = 0.01;
    /// let linear_warmup_lr = LinearWarmupLR::new(warmup_steps, base_lr);
    /// let sequential_lr = SequentialLR::new(linear_warmup_lr, IdentityLR, 1000);
    /// ```
    pub fn new(prev_scheduler: PS, next_scheduler: S, milestone: usize) -> Self {
        SequentialLR {
            prev_scheduler,
            next_scheduler,
            milestone,
            u: PhantomData
        }
    }
}
impl<U,PS,S> Scheduler<U> for SequentialLR<U,PS,S> where U: UnitValue<U>, PS: Scheduler<U> + Clone, S: Scheduler<U> + Clone {
    fn schedule(&mut self, lr: U, step: usize) -> Result<U,TrainingError> {
        if step < self.milestone {
            self.prev_scheduler.schedule(lr, step)
        } else {
            self.next_scheduler.schedule(lr, step - self.milestone)
        }
    }

    fn seq<NS>(self, milestone: usize, next_scheduler: NS) -> SequentialLR<U,Self,NS>
        where NS: Scheduler<U> + Clone + Sized + 'static,
              Self: Sized + 'static,
    {
        let joined_milestone = self.milestone + milestone;

        SequentialLR::new(self, next_scheduler, joined_milestone)
    }
}