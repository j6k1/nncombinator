//! Implementing the loss function of a neural network

use std::marker::PhantomData;
use cuda_runtime_sys::dim3;
use libc::c_uint;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use crate::arr::{Arr, VecArr};
use crate::cuda::{AsMutKernelPtr, CudaPtr, Kernel, Memory};
use crate::cuda::kernel::lossfunction::{LinearBatchCrossEntropy, LinearBatchCrossEntropyArgs, LinearBatchCrossEntropyMulticlass, LinearBatchCrossEntropyMulticlassArgs, LinearBatchMse, LinearBatchMseArgs};
use crate::device::{Device, DeviceCpu, DeviceGpu};
use crate::error::{CudaError, TrainingError};
use crate::mem::AsRawSlice;
use crate::UnitValue;

/// Trait that defines the implementation of the loss function used in neural networks during training.
pub trait LossFunction<U>: Send + Sync + 'static where U: Clone + Copy + UnitValue<U> {
    /// Differentiation of loss functions
    /// # Arguments
    /// * `r` - actual value
    /// * `t` - expected value
    fn derive(&self,r:U,t:U) -> U;
    /// Applying the loss function
    /// # Arguments
    /// * `r` - actual value
    /// * `t` - expected value
    fn apply(&self,r:U,t:U) -> U;
    /// this loss function name
    fn name(&self) -> &'static str;
}
/// Trait defining the implementation of the loss function with batch processing
pub trait BatchLossFunction<U,D>: LossFunction<U> + Send + Sync + 'static
    where U: Clone + Copy + UnitValue<U>, D: Device<U> {
    /// Differentiation of loss functions
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    fn batch_linear_derive<const N: usize>(&self,_: &D,expected: &VecArr<U,Arr<U, N>>, actual: &VecArr<U,Arr<U, N>>)
                                           -> Result<VecArr<U,Arr<U, N>>, TrainingError> {
        Ok(actual.par_iter().zip(expected.par_iter()).map(|(a,e)| {
            a.par_iter()
                .zip(e.par_iter())
                .map(|(&a,&e)| self.derive(a,e))
                .collect::<Vec<U>>()
                .try_into().map_err(|e| TrainingError::from(e))
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
/// Mse implementation
pub struct Mse<U> where U: Clone + Copy + UnitValue<U> {
    u:PhantomData<U>
}
impl<U> Mse<U> where U: UnitValue<U> {
    /// Create a Mse instance
    pub fn new() -> Mse<U> {
        Mse {
            u:PhantomData::<U>
        }
    }
}
impl<U> LossFunction<U> for Mse<U> where U: Clone + Copy + UnitValue<U> {
    fn derive(&self, r: U, t: U) -> U {
        r - t
    }

    fn apply(&self, r: U, t: U) -> U {
        (r - t) * (r - t) / U::from_f64(2.).unwrap()
    }

    fn name(&self) -> &'static str {
        "mse"
    }
}
impl<U> BatchLossFunction<U,DeviceCpu<U>> for Mse<U> where U: Clone + Copy + UnitValue<U> {}
impl<U> BatchLossFunction<U,DeviceGpu<U>> for Mse<U> where U: Clone + Copy + UnitValue<U> + AsMutKernelPtr,
                                                              DeviceGpu<U>:  Device<U>,
                                                              CudaPtr<U>: TryFrom<U,Error=CudaError>,
                                                              LinearBatchMse<U>: Kernel<Args=LinearBatchMseArgs<U>> {
    fn batch_linear_derive<const N: usize>(&self, _: &DeviceGpu<U>, expected: &VecArr<U, Arr<U, N>>, actual: &VecArr<U, Arr<U, N>>) -> Result<VecArr<U, Arr<U, N>>, TrainingError> {
        let mut expected_ptr = CudaPtr::new(expected.len() * N).unwrap();
        expected_ptr.memcpy(expected.as_raw_slice().as_ptr(), expected.len() * N).unwrap();

        let mut actual_ptr = CudaPtr::new(actual.len() * N).unwrap();
        actual_ptr.memcpy(actual.as_raw_slice().as_ptr(), actual.len() * N).unwrap();

        let mut args = LinearBatchMseArgs::new(expected_ptr, actual_ptr, N, expected.len());

        let mut kernel = LinearBatchMse::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (expected.len() as c_uint + 32 - 1) / 32, z: 1},
                      dim3 { x: 32, y: 32, z: 1 },&mut args,0).unwrap();

        Ok(args.actual.read_to_vec()?.into())
    }
}
/// CrossEntropy implementation
pub struct CrossEntropy<U>  where U: Clone + Copy + UnitValue<U> {
    u:PhantomData<U>
}
impl<U> CrossEntropy<U> where U: Clone + Copy + UnitValue<U> {
    /// Create a CrossEntropy instance
    pub fn new() -> CrossEntropy<U> {
        CrossEntropy {
            u:PhantomData::<U>
        }
    }
}
impl<U> LossFunction<U> for CrossEntropy<U> where U: Clone + Copy + UnitValue<U> {
    fn derive(&self, r: U, t: U) -> U {
        -(r / (t + U::from_f64(1e-7).unwrap())) + (U::one() - t) / (U::one() - r)
    }

    fn apply(&self, r: U, t: U) -> U {
        -t * r.max(&U::from_f64(1e-7).unwrap()).ln() + (U::one() - t) * (U::one() - r).max(&U::from_f64(1e-7).unwrap()).ln()
    }

    fn name(&self) -> &'static str {
        "crossentropy"
    }
}
impl<U> BatchLossFunction<U,DeviceCpu<U>> for CrossEntropy<U> where U: Clone + Copy + UnitValue<U> {}
impl<U> BatchLossFunction<U,DeviceGpu<U>> for CrossEntropy<U> where U: Clone + Copy + UnitValue<U> + AsMutKernelPtr,
                                                                    DeviceGpu<U>:  Device<U>,
                                                                    CudaPtr<U>: TryFrom<U,Error=CudaError>,
                                                                    LinearBatchCrossEntropy<U>: Kernel<Args=LinearBatchCrossEntropyArgs<U>> {
    fn batch_linear_derive<const N: usize>(&self, _: &DeviceGpu<U>, expected: &VecArr<U, Arr<U, N>>, actual: &VecArr<U, Arr<U, N>>) -> Result<VecArr<U, Arr<U, N>>, TrainingError> {
        let mut expected_ptr = CudaPtr::new(expected.len() * N).unwrap();
        expected_ptr.memcpy(expected.as_raw_slice().as_ptr(), expected.len() * N).unwrap();

        let mut actual_ptr = CudaPtr::new(actual.len() * N).unwrap();
        actual_ptr.memcpy(actual.as_raw_slice().as_ptr(), actual.len() * N).unwrap();

        let mut args = LinearBatchCrossEntropyArgs::new(expected_ptr, actual_ptr, N, expected.len());

        let mut kernel = LinearBatchCrossEntropy::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (expected.len() as c_uint + 32 - 1) / 32, z: 1},
                      dim3 { x: 32, y: 32, z: 1 },&mut args,0).unwrap();

        Ok(args.actual.read_to_vec()?.into())
    }
}
/// CrossEntropyMulticlass implementation
pub struct CrossEntropyMulticlass<U> where U: Clone + Copy + UnitValue<U> {
    u:PhantomData<U>
}
impl<U> CrossEntropyMulticlass<U> where U: Clone + Copy + UnitValue<U> {
    /// Create a CrossEntropyMulticlass instance
    pub fn new() -> CrossEntropyMulticlass<U> {
        CrossEntropyMulticlass {
            u:PhantomData::<U>
        }
    }
}
impl<U> LossFunction<U> for CrossEntropyMulticlass<U> where U: Clone + Copy + UnitValue<U> {
    fn derive(&self, r: U, t: U) -> U {
        -t / r
    }

    fn apply(&self, r: U, t: U) -> U {
        -t * r.max(&U::from_f64(1e-7).unwrap()).ln()
    }

    fn name(&self) -> &'static str {
        "crossentropymulticlass"
    }
}
impl<U> BatchLossFunction<U,DeviceCpu<U>> for CrossEntropyMulticlass<U> where U: Clone + Copy + UnitValue<U> {}
impl<U> BatchLossFunction<U,DeviceGpu<U>> for CrossEntropyMulticlass<U> where U: Clone + Copy + UnitValue<U> + AsMutKernelPtr,
                                                                              DeviceGpu<U>:  Device<U>,
                                                                              CudaPtr<U>: TryFrom<U,Error=CudaError>,
                                                                              LinearBatchCrossEntropyMulticlass<U>: Kernel<Args=LinearBatchCrossEntropyMulticlassArgs<U>> {
    fn batch_linear_derive<const N: usize>(&self, _: &DeviceGpu<U>, expected: &VecArr<U, Arr<U, N>>, actual: &VecArr<U, Arr<U, N>>) -> Result<VecArr<U, Arr<U, N>>, TrainingError> {
        let mut expected_ptr = CudaPtr::new(expected.len() * N).unwrap();
        expected_ptr.memcpy(expected.as_raw_slice().as_ptr(), expected.len() * N).unwrap();

        let mut actual_ptr = CudaPtr::new(actual.len() * N).unwrap();
        actual_ptr.memcpy(actual.as_raw_slice().as_ptr(), actual.len() * N).unwrap();

        let mut args = LinearBatchCrossEntropyMulticlassArgs::new(expected_ptr, actual_ptr, N, expected.len());

        let mut kernel = LinearBatchCrossEntropyMulticlass::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (expected.len() as c_uint + 32 - 1) / 32, z: 1},
                      dim3 { x: 32, y: 32, z: 1 },&mut args,0).unwrap();

        Ok(args.actual.read_to_vec()?.into())
    }
}
