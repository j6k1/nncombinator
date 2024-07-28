//! Implementing the loss function of a neural network

use std::marker::PhantomData;
use cuda_runtime_sys::dim3;
use libc::c_uint;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use crate::arr::{Arr, ArrView, SerializedVec, SerializedVecView};
use crate::cuda::{CudaPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, DataTypeInfo, Kernel};
use crate::cuda::kernel::lossfunction::{LinearBatchCrossEntropy, LinearBatchCrossEntropyArgs, LinearBatchCrossEntropyMulticlass, LinearBatchCrossEntropyMulticlassArgs, LinearBatchMse, LinearBatchMseArgs, LinearCrossEntropy, LinearCrossEntropyArgs, LinearCrossEntropyMulticlass, LinearCrossEntropyMulticlassArgs, LinearMse, LinearMseArgs};
use crate::device::{Device, DeviceCpu, DeviceGpu, DeviceMemoryPool};
use crate::error::{CudaError, TrainingError};
use crate::layer::{BatchSize};
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
/// A property that defines the implementation of the loss function used in the linear layer when training a neural network.
pub trait LossFunctionLinear<'a,U,D,const N:usize>: LossFunction<U> + Send + Sync + 'static
    where U: Clone + Copy + UnitValue<U>, D: Device<U> {
    type Input;
    type Output;
    /// Differentiation of loss functions
    /// # Arguments
    /// * `actual` - actual value
    /// * `expected` - expected value
    fn linear_derive<'b>(&self,device:&D,actual:&'b Self::Input,expected:&'b Self::Input) -> Result<Self::Output,TrainingError>;
}
/// Trait defining the implementation of a linear layer loss function with batch processing
pub trait BatchLossFunctionLinear<'a,U,D,const N:usize>: LossFunction<U> + Send + Sync + 'static
    where U: Clone + Copy + UnitValue<U>, D: Device<U> {
    type Input: BatchSize;
    type Output: BatchSize;
    /// Differentiation of loss functions
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    fn batch_linear_derive<'b>(&self,_: &D,expected: &'b Self::Input, actual: &'b Self::Input) -> Result<Self::Output, TrainingError>;
}
impl<'a,T,U,const N:usize> LossFunctionLinear<'a,U,DeviceCpu<U>,N> for T
    where T: LossFunction<U>,
          U: UnitValue<U> {
    type Input = ArrView<'a,U,N>;
    type Output = Arr<U,N>;
    fn linear_derive<'b>(&self,_:&DeviceCpu<U>,actual: &'b ArrView<'b,U,N>, expected: &'b ArrView<'b,U,N>)
        -> Result<Arr<U,N>,TrainingError> {
        let mut loss = Arr::new();

        for (loss,(&a, &e))in loss.iter_mut().zip(actual.iter().zip(expected.iter())) {
            *loss = self.derive(a, e);
        }

        Ok(loss)
    }
}
impl<'a,T,U,const N:usize> BatchLossFunctionLinear<'a,U,DeviceCpu<U>,N> for T
    where T: LossFunction<U>,
          U: UnitValue<U> {
    type Input = SerializedVecView<'a,U,Arr<U,N>>;
    type Output = SerializedVec<U,Arr<U,N>>;
    fn batch_linear_derive<'b>(&self,_: &DeviceCpu<U>,expected: &'b SerializedVecView<'b,U,Arr<U, N>>,
                               actual: &'b SerializedVecView<'b,U,Arr<U, N>>)
        -> Result<SerializedVec<U,Arr<U, N>>, TrainingError> {
        let n = U::from_usize(actual.len()).ok_or(TrainingError::TypeCastError(
            String::from("An error occurred when casting the batch size data type to U.")
        ))?;

        Ok(actual.par_iter().zip(expected.par_iter()).map(|(a,e)| {
            a.par_iter()
                .zip(e.par_iter())
                .map(|(&a,&e)| self.derive(a,e) / n)
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
impl<'a,U,const N:usize> LossFunctionLinear<'a,U,DeviceGpu<U>,N> for Mse<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          for<'b> LinearMse<'b,U,N>: Kernel<Args=LinearMseArgs<'b,U,N>> {
    type Input = CudaTensor1dPtrView<'a,U,N>;
    type Output = CudaTensor1dPtr<U,N>;

    fn linear_derive<'b>(&self,device:&DeviceGpu<U>,actual: &'b CudaTensor1dPtrView<'b,U,N>, expected: &'b CudaTensor1dPtrView<'b,U,N>)
        -> Result<Self::Output,TrainingError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = LinearMseArgs::new(expected, actual, output, N);

        let mut kernel = LinearMse::<'a,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1024 - 1) / 1024, y: 1, z: 1},
                      dim3 { x: 1024, y: 32, z: 1 },&mut args,0)?;

        Ok(args.output)
    }
}
impl<'a,U,const N:usize> BatchLossFunctionLinear<'a,U,DeviceGpu<U>,N> for Mse<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
             DeviceGpu<U>:  Device<U>,
             for<'b> LinearBatchMse<'b,U,N>: Kernel<Args=LinearBatchMseArgs<'b,U,N>> {
    type Input = CudaVecView<'a,U,CudaTensor1dPtr<U,N>>;
    type Output = CudaVec<U,CudaTensor1dPtr<U,N>>;
    fn batch_linear_derive<'b>(&self, device: &DeviceGpu<U>, expected: &'b CudaVecView<'b,U,CudaTensor1dPtr<U,N>>,
                                      actual: &'b CudaVecView<'b,U,CudaTensor1dPtr<U,N>>)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(expected.size(),device.get_memory_pool())?;

        let mut args = LinearBatchMseArgs::new(expected, actual, output, N, expected.size());

        let mut kernel = LinearBatchMse::<'a,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (expected.size() as c_uint + 32 - 1) / 32, z: 1},
                      dim3 { x: 32, y: 32, z: 1 },&mut args,0)?;

        Ok(args.output)
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
impl<'a,U,const N:usize> LossFunctionLinear<'a,U,DeviceGpu<U>,N> for CrossEntropy<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          for<'b> LinearCrossEntropy<'b,U,N>: Kernel<Args=LinearCrossEntropyArgs<'b,U,N>> {
    type Input = CudaTensor1dPtrView<'a,U,N>;
    type Output = CudaTensor1dPtr<U,N>;

    fn linear_derive<'b>(&self,device:&DeviceGpu<U>,actual: &'b CudaTensor1dPtrView<'b,U,N>,
                         expected: &'b CudaTensor1dPtrView<'a,U,N>)
        -> Result<Self::Output,TrainingError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = LinearCrossEntropyArgs::new(expected, actual, output, N);

        let mut kernel = LinearCrossEntropy::<'a,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1024 - 1) / 1024, y: 1, z: 1},
                      dim3 { x: 1024, y: 32, z: 1 },&mut args,0)?;

        Ok(args.output)
    }
}
impl<'a,U,const N:usize> BatchLossFunctionLinear<'a,U,DeviceGpu<U>,N> for CrossEntropy<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>:  Device<U>,
          for<'b> LinearBatchCrossEntropy<'b,U,N>: Kernel<Args=LinearBatchCrossEntropyArgs<'b,U,N>> {
    type Input = CudaVecView<'a,U,CudaTensor1dPtr<U,N>>;
    type Output = CudaVec<U,CudaTensor1dPtr<U,N>>;
    fn batch_linear_derive<'b>(&self, device: &DeviceGpu<U>, expected: &'b CudaVecView<'b,U,CudaTensor1dPtr<U,N>>,
                                      actual: &'b CudaVecView<'b,U,CudaTensor1dPtr<U,N>>)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(expected.size(),device.get_memory_pool())?;

        let mut args = LinearBatchCrossEntropyArgs::new(expected, actual, output, N, expected.size());

        let mut kernel = LinearBatchCrossEntropy::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (expected.size() as c_uint + 32 - 1) / 32, z: 1},
                      dim3 { x: 32, y: 32, z: 1 },&mut args,0)?;

        Ok(args.output)
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
impl<'a,U,const N:usize> LossFunctionLinear<'a,U,DeviceGpu<U>,N> for CrossEntropyMulticlass<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          for<'b> LinearCrossEntropyMulticlass<'b,U,N>: Kernel<Args=LinearCrossEntropyMulticlassArgs<'b,U,N>> {
    type Input = CudaTensor1dPtrView<'a,U,N>;
    type Output = CudaTensor1dPtr<U,N>;

    fn linear_derive<'b>(&self,device:&DeviceGpu<U>,actual: &'b CudaTensor1dPtrView<'b,U,N>,
                         expected: &'b CudaTensor1dPtrView<'b,U,N>)
        -> Result<Self::Output,TrainingError> {
        let output = CudaTensor1dPtr::<U,N>::new(device.get_memory_pool())?;

        let mut args = LinearCrossEntropyMulticlassArgs::new(expected, actual, output, N);

        let mut kernel = LinearCrossEntropyMulticlass::<'a,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1024 - 1) / 1024, y: 1, z: 1},
                      dim3 { x: 1024, y: 32, z: 1 },&mut args,0)?;

        Ok(args.output)
    }
}
impl<'a,U,const N:usize> BatchLossFunctionLinear<'a,U,DeviceGpu<U>,N> for CrossEntropyMulticlass<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
             DeviceGpu<U>:  Device<U>,
             CudaPtr<U>: TryFrom<U,Error=CudaError>,
             for<'b> LinearBatchCrossEntropyMulticlass<'b,U,N>: Kernel<Args=LinearBatchCrossEntropyMulticlassArgs<'b,U,N>> {
    type Input = CudaVecView<'a,U,CudaTensor1dPtr<U,N>>;
    type Output = CudaVec<U,CudaTensor1dPtr<U,N>>;
    fn batch_linear_derive<'b>(&self, device: &DeviceGpu<U>, expected: &'b CudaVecView<'b,U,CudaTensor1dPtr<U,N>>,
                               actual: &'b CudaVecView<'b,U,CudaTensor1dPtr<U,N>>)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,N>>, TrainingError> {
        let output = CudaVec::<U,CudaTensor1dPtr<U,N>>::new(expected.size(),device.get_memory_pool())?;

        let mut args = LinearBatchCrossEntropyMulticlassArgs::new(expected, actual, output, N, expected.size());

        let mut kernel = LinearBatchCrossEntropyMulticlass::<'_,U,N>::new();

        kernel.launch(dim3 { x: (N as c_uint + 32 - 1) / 32,
                                     y: (expected.size() as c_uint + 32 - 1) / 32, z: 1},
                      dim3 { x: 32, y: 32, z: 1 },&mut args,0)?;

        Ok(args.output)
    }
}
