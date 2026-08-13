//! Implementing the loss function of a neural network

use std::marker::PhantomData;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use crate::arr::{Arr, ArrView, SerializedVec, SerializedVecView};
use crate::device::{Device, DeviceCpu};
use crate::error::{TrainingError, TypeConvertError};
use crate::layer::{BatchSize};
use crate::UnitValue;
#[cfg(feature = "cuda")]
use crate::cuda::{AsConstKernelPtr, AsCudaMutPtr, AsMutKernelPtr, CudaMutPtr, CudaPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, DataTypeInfo, Kernel, WriteMemory};
#[cfg(feature = "cuda")]
use crate::cuda::allocator::CudaAllocator;
#[cfg(feature = "cuda")]
use crate::cuda::kernel::lossfunction::{LinearBatchCrossEntropy, LinearBatchCrossEntropyArgs, LinearBatchCrossEntropyMulticlass, LinearBatchCrossEntropyMulticlassArgs, LinearBatchMse, LinearBatchMseArgs, LinearCrossEntropy, LinearCrossEntropyArgs, LinearCrossEntropyMulticlass, LinearCrossEntropyMulticlassArgs, LinearMse, LinearMseArgs};
#[cfg(feature = "cuda")]
use crate::device::{DeviceGpu, DeviceAllocator};

/// Trait that defines the implementation of the loss function used in neural networks during training.
pub trait LossFunction<U>: Send + Sync + 'static where U: Clone + Copy {
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
pub trait LossFunctionLinear<'a,U,I,D,const N:usize>: LossFunction<U> + Send + Sync + 'static
    where U: Clone + Copy + UnitValue<U>, D: Device<U> {
    type Output;
    /// Differentiation of loss functions
    /// # Arguments
    /// * `actual` - actual value
    /// * `expected` - expected value
    fn linear_derive(&self,device:&D,actual:&'a I,expected:&'a I) -> Result<Self::Output,TrainingError>;
}
/// Trait defining the implementation of a linear layer loss function with batch processing
pub trait BatchLossFunctionLinear<'a,U,I,D,const N:usize>: LossFunction<U> + Send + Sync + 'static
    where U: Clone + Copy + UnitValue<U>,
          D: Device<U> {
    type Output: BatchSize;
    /// Differentiation of loss functions
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    fn batch_linear_derive(&self,_: &D,expected: &'a I, actual: &'a I)
        -> Result<Self::Output, TrainingError>;
}
impl<'a,T,U,I,const N:usize> LossFunctionLinear<'a,U,I,DeviceCpu<U>,N> for T
    where T: LossFunction<U>,
          U: UnitValue<U>,
          for<'b> ArrView<'b,U,N>: From<&'b I> {
    type Output = Arr<U,N>;
    fn linear_derive(&self,_:&DeviceCpu<U>,actual: &'a I, expected: &'a I)
        -> Result<Arr<U,N>,TrainingError> {
        let actual = ArrView::<'a,U,N>::from(actual);
        let expected = ArrView::<'a,U,N>::from(expected);

        let mut loss = Arr::new();

        for (loss,(&a, &e))in loss.iter_mut().zip(actual.iter().zip(expected.iter())) {
            *loss = self.derive(a, e);
        }

        Ok(loss)
    }
}
impl<'a,T,U,I,const N:usize> BatchLossFunctionLinear<'a,U,I,DeviceCpu<U>,N> for T
    where T: LossFunction<U>,
          U: UnitValue<U>,
          I: BatchSize,
          for<'b> SerializedVecView<'b,U,Arr<U,N>>: TryFrom<&'b I,Error=TypeConvertError> {
    type Output = SerializedVec<U,Arr<U,N>>;
    fn batch_linear_derive(&self,_: &DeviceCpu<U>,expected: &'a I, actual: &'a I)
        -> Result<SerializedVec<U,Arr<U, N>>, TrainingError> {
        let actual = SerializedVecView::<'a,U,Arr<U,N>>::try_from(actual)?;
        let expected = SerializedVecView::<'a,U,Arr<U,N>>::try_from(expected)?;

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
#[cfg(feature = "cuda")]
impl<'a,U,I,A,const N:usize> LossFunctionLinear<'a,U,I,DeviceGpu<U,A>,N> for Mse<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
          I: 'a,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: WriteMemory<U>,
          for<'b> A: CudaAllocator + 'b,
          for<'b> CudaTensor1dPtrView<'a,U,N>: From<&'b I>,
          for<'b> LinearMse<'b,U,A,N>: Kernel<Args=LinearMseArgs<'b,U,A,N>> {
    type Output = CudaTensor1dPtr<U,A,N>;

    fn linear_derive(&self,device:&DeviceGpu<U,A>,actual: &'a I, expected: &'a I)
        -> Result<Self::Output,TrainingError> {
        let actual = CudaTensor1dPtrView::<'a,U,N>::from(actual);
        let expected = CudaTensor1dPtrView::<'a,U,N>::from(expected);

        let output = CudaTensor1dPtr::<U,A,N>::new(device.get_allocator())?;

        let mut args = LinearMseArgs::new(&expected, &actual, output, N);

        let mut kernel = LinearMse::<'a,U,A,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }
}
#[cfg(feature = "cuda")]
impl<'a,U,I,A,const N:usize> BatchLossFunctionLinear<'a,U,I,DeviceGpu<U,A>,N> for Mse<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
          I: 'a,
          A: CudaAllocator + 'a,
          DeviceGpu<U,A>:  Device<U>,
          CudaPtr<U,A>: WriteMemory<U>,
          for<'b> CudaTensor1dPtr<U,A,N>: AsConstKernelPtr + AsMutKernelPtr,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b I,Error=TypeConvertError>,
          for<'b> CudaVec<U,CudaTensor1dPtr<U,A,N>,A>: AsCudaMutPtr<Pointee=U,Allocator=A>,
          for<'b> CudaMutPtr<'b,U,A>: AsMutKernelPtr,
          for<'b> LinearBatchMse<'b,U,A,N>: Kernel<Args=LinearBatchMseArgs<'b,U,A,N>> {
    type Output = CudaVec<U,CudaTensor1dPtr<U,A,N>,A>;
    fn batch_linear_derive(&self, device: &DeviceGpu<U,A>, expected: &'a I, actual: &'a I)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,A,N>,A>, TrainingError> {
        let actual = CudaVecView::<'a,U,CudaTensor1dPtrView<'a,U,N>>::try_from(actual)?;
        let expected = CudaVecView::<'a,U,CudaTensor1dPtrView<'a,U,N>>::try_from(expected)?;

        let output = CudaVec::<U,CudaTensor1dPtr<U,A,N>,A>::new(expected.size(),device.get_allocator())?;

        let mut args = LinearBatchMseArgs::new(&expected, &actual, output, N, expected.size());

        let mut kernel = LinearBatchMse::<'a,U,A,N>::new();

        kernel.launch(&mut args)?;

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
        -(t / (r + U::from_f64(1e-7).unwrap())) + (U::one() - t) / (U::one() - r)
    }

    fn apply(&self, r: U, t: U) -> U {
        -t * r.max(&U::from_f64(1e-7).unwrap()).ln() - (U::one() - t) * (U::one() - r).max(&U::from_f64(1e-7).unwrap()).ln()
    }

    fn name(&self) -> &'static str {
        "crossentropy"
    }
}
#[cfg(feature = "cuda")]
impl<'a,U,I,A,const N:usize> LossFunctionLinear<'a,U,I,DeviceGpu<U,A>,N> for CrossEntropy<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
          I: 'a,
          A: CudaAllocator + 'a,
          DeviceGpu<U,A>: Device<U>,
          CudaPtr<U,A>: WriteMemory<U>,
          for<'b> CudaTensor1dPtrView<'b,U,N>: From<&'b I>,
          for<'b> CudaTensor1dPtr<U,A,N>: AsConstKernelPtr + AsMutKernelPtr,
          for<'b> CudaMutPtr<'b,U,A>: AsMutKernelPtr,
          for<'b> LinearCrossEntropy<'b,U,A,N>: Kernel<Args=LinearCrossEntropyArgs<'b,U,A,N>> {
    type Output = CudaTensor1dPtr<U,A,N>;

    fn linear_derive(&self,device:&DeviceGpu<U,A>,actual: &'a I, expected: &'a I) -> Result<Self::Output,TrainingError> {
        let actual = CudaTensor1dPtrView::<'a,U,N>::from(actual);
        let expected = CudaTensor1dPtrView::<'a,U,N>::from(expected);

        let output = CudaTensor1dPtr::<U,A,N>::new(device.get_allocator())?;

        let mut args = LinearCrossEntropyArgs::new(&expected, &actual, output, N);

        let mut kernel = LinearCrossEntropy::<'a,U,A,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }
}
#[cfg(feature = "cuda")]
impl<'a,U,I,A,const N:usize> BatchLossFunctionLinear<'a,U,I,DeviceGpu<U,A>,N> for CrossEntropy<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
          I: 'a,
          A: CudaAllocator + 'a,
          DeviceGpu<U,A>:  Device<U>,
          CudaPtr<U,A>: WriteMemory<U>,
          for<'b> CudaTensor1dPtr<U,A,N>: AsConstKernelPtr + AsMutKernelPtr,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b I,Error=TypeConvertError>,
          for<'b> CudaVec<U,CudaTensor1dPtr<U,A,N>,A>: AsCudaMutPtr<Pointee=U,Allocator=A>,
          for<'b> CudaMutPtr<'b,U,A>: AsMutKernelPtr,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b I,Error=TypeConvertError>,
          for<'b> LinearBatchCrossEntropy<'b,U,A,N>: Kernel<Args=LinearBatchCrossEntropyArgs<'b,U,A,N>> {
    type Output = CudaVec<U,CudaTensor1dPtr<U,A,N>,A>;
    fn batch_linear_derive(&self, device: &DeviceGpu<U,A>, expected: &'a I, actual: &'a I)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,A,N>,A>, TrainingError> {
        let actual = CudaVecView::<'a,U,CudaTensor1dPtrView<'a,U,N>>::try_from(actual)?;
        let expected = CudaVecView::<'a,U,CudaTensor1dPtrView<'a,U,N>>::try_from(expected)?;

        let output = CudaVec::<U,CudaTensor1dPtr<U,A,N>,A>::new(expected.size(),device.get_allocator())?;

        let mut args = LinearBatchCrossEntropyArgs::new(&expected, &actual, output, N, expected.size());

        let mut kernel = LinearBatchCrossEntropy::<'_,U,A,N>::new();

        kernel.launch(&mut args)?;

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
#[cfg(feature = "cuda")]
impl<'a,U,I,A,const N:usize> LossFunctionLinear<'a,U,I,DeviceGpu<U,A>,N> for CrossEntropyMulticlass<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
          I: 'a,
          DeviceGpu<U,A>: Device<U>,
          for<'b> A: CudaAllocator + 'b,
          CudaPtr<U,A>: WriteMemory<U>,
          for<'b> CudaTensor1dPtrView<'b,U,N>: From<&'b I>,
          for<'b> LinearCrossEntropyMulticlass<'b,U,A,N>: Kernel<Args=LinearCrossEntropyMulticlassArgs<'b,U,A,N>> {
    type Output = CudaTensor1dPtr<U,A,N>;

    fn linear_derive(&self,device:&DeviceGpu<U,A>,actual: &'a I,expected: &'a I) -> Result<Self::Output,TrainingError> {
        let actual = CudaTensor1dPtrView::<'a,U,N>::from(actual);
        let expected = CudaTensor1dPtrView::<'a,U,N>::from(expected);

        let output = CudaTensor1dPtr::<U,A,N>::new(device.get_allocator())?;

        let mut args = LinearCrossEntropyMulticlassArgs::new(&expected, &actual, output, N);

        let mut kernel = LinearCrossEntropyMulticlass::<'a,U,A,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }
}
#[cfg(feature = "cuda")]
impl<'a,U,I,A,const N:usize> BatchLossFunctionLinear<'a,U,I,DeviceGpu<U,A>,N> for CrossEntropyMulticlass<U>
    where U: Clone + Copy + UnitValue<U> + DataTypeInfo,
          I: 'a,
          A: CudaAllocator + 'a,
          DeviceGpu<U,A>:  Device<U>,
          CudaPtr<U,A>: WriteMemory<U>,
          for<'b> CudaTensor1dPtr<U,A,N>: AsConstKernelPtr + AsMutKernelPtr,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b I,Error=TypeConvertError>,
          for<'b> CudaVec<U,CudaTensor1dPtr<U,A,N>,A>: AsCudaMutPtr<Pointee=U,Allocator=A>,
          for<'b> CudaMutPtr<'b,U,A>: AsMutKernelPtr,
          for<'b> LinearBatchCrossEntropyMulticlass<'b,U,A,N>: Kernel<Args=LinearBatchCrossEntropyMulticlassArgs<'b,U,A,N>> {
    type Output = CudaVec<U,CudaTensor1dPtr<U,A,N>,A>;
    fn batch_linear_derive(&self, device: &DeviceGpu<U,A>, expected: &'a I, actual: &'a I)
        -> Result<CudaVec<U,CudaTensor1dPtr<U,A,N>,A>, TrainingError> {
        let actual = CudaVecView::<'a,U,CudaTensor1dPtrView<'a,U,N>>::try_from(actual)?;
        let expected = CudaVecView::<'a,U,CudaTensor1dPtrView<'a,U,N>>::try_from(expected)?;

        let output = CudaVec::<U,CudaTensor1dPtr<U,A,N>,A>::new(expected.size(),device.get_allocator())?;

        let mut args = LinearBatchCrossEntropyMulticlassArgs::new(&expected, &actual, output, N, expected.size());

        let mut kernel = LinearBatchCrossEntropyMulticlass::<'_,U,A,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }
}
