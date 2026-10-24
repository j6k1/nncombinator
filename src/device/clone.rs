//! Implementation of the calculation process for clone layers

use simdoperator::backend::common::Backend;
#[cfg(feature = "cuda")]
use crate::cuda::allocator::CudaAllocator;
#[cfg(feature = "cuda")]
use crate::cuda::TryClone;
#[cfg(feature = "cuda")]
use crate::device::DeviceGpu;
use crate::device::{DeviceCpu, DeviceSimd};
use crate::error::{EvaluateError};

pub trait DeviceClone<'a,T> where T: 'static {
    /// clone calculation.
    ///
    /// # Arguments
    /// * `source` - source
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn cloned(&self, source: &'a T) -> Result<T, EvaluateError>;
}
impl<'a,T> DeviceClone<'a,T> for DeviceCpu
    where T: Clone + 'static {
    fn cloned(&self, source: &'a T) -> Result<T, EvaluateError> {
        Ok(source.clone())
    }
}
impl<'a,T,BE> DeviceClone<'a,T> for DeviceSimd<BE>
    where T: Clone + 'static,
          BE: Backend {
    fn cloned(&self, source: &'a T) -> Result<T, EvaluateError> {
        Ok(source.clone())
    }
}
#[cfg(feature = "cuda")]
impl<'a,T,A> DeviceClone<'a,T> for DeviceGpu<A>
    where T: TryClone + 'static,
          A: CudaAllocator + 'static,
          EvaluateError: From<<T as TryClone>::Error>{
    fn cloned(&self, source: &'a T) -> Result<T, EvaluateError> {
        Ok(source.try_clone()?)
    }
}