//! Implementation of the calculation process for output layers

use num_traits::FromPrimitive;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IndexedParallelIterator;
use crate::arr::{Arr, ArrView, SerializedVec, SerializedVecView};
use crate::device::{Device, DeviceCpu};
use crate::error::{TrainingError};
use crate::layer::{BatchDataType, BatchSize};
use crate::lossfunction::{BatchLossFunctionLinear, LossFunction, LossFunctionLinear};
use crate::ope::UnitValue;
#[cfg(feature = "cuda")]
use core::fmt::Debug;
#[cfg(feature = "cuda")]
use crate::error::{TypeConvertError};
#[cfg(feature = "cuda")]
use crate::cuda::{CudaTensor1dPtr, CudaTensor1dPtrView, CudaVec, CudaVecView, DataTypeInfo, Kernel, ToCuda, ReadMemory, WriteMemory, CudaPtr, AsMutPtr, AsCudaPtr};
#[cfg(feature = "cuda")]
use crate::cuda::allocator::CudaAllocator;
#[cfg(feature = "cuda")]
use crate::cuda::kernel::device::{LossLinearBatchByCanonicalLink, LossLinearBatchByCanonicalLinkArgs, LossLinearByCanonicalLink, LossLinearByCanonicalLinkArgs};
#[cfg(feature = "cuda")]
use crate::device::{DeviceGpu, DeviceAllocator};

/// Trait that defines the implementation of various calculation processes in the linear output layer
pub trait DeviceLinearOutput<'a,U,const N:usize>: Device<U>
    where U: UnitValue<U> {
    type IO: BatchDataType + 'static;
    type BatchIO: BatchSize + 'static;
    /// Calculation of Losses
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    /// * `lossf` - loss function
    fn loss_linear<L>(&self, expected: &'a Arr<U,N>, actual: &'a Self::IO, lossf: &L) -> Result<Self::IO,TrainingError>
        where for<'b> L: LossFunction<U> + LossFunctionLinear<'b,U,Self::IO,Self,N,Output=Self::IO>;
    /// Calculation of Losses by canonical link
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    fn loss_linear_by_canonical_link(&self, expected: &'a Arr<U,N>, actual: &'a Self::IO) -> Result<Self::IO,TrainingError>;
    /// Calculation of total Losses
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    /// * `lossf` - loss function
    fn loss_linear_total<L: LossFunction<U>>(&self, exptected:&'a Arr<U,N>, actual:&'a Self::IO, lossf:&L) -> Result<U,TrainingError>;
    /// Calculation of loss during batch execution by canonical link
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn loss_linear_batch_by_canonical_link(&self, expected: &'a SerializedVec<U,Arr<U,N>>,
                                               actual: &'a Self::BatchIO)
                                               -> Result<Self::BatchIO, TrainingError> where f64: From<U>;
    /// Calculation of Losses (all batch)
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    /// * `lossf` - loss function
    fn batch_loss_linear<L>(&self, expected: &'a SerializedVec<U,Arr<U,N>>,
                               actual: &'a Self::BatchIO, lossf: &L)
                               -> Result<Self::BatchIO,TrainingError>
        where for<'b> L: LossFunction<U> + BatchLossFunctionLinear<'b,U,Self::BatchIO,Self,N,Output=Self::BatchIO>;
    /// Calculation of total Losses (all batch)
    /// # Arguments
    /// * `expected` - expected value
    /// * `actual` - actual value
    /// * `lossf` - loss function
    fn batch_loss_linear_total<L: LossFunction<U>>(&self, exptected:&'a SerializedVec<U,Arr<U,N>>,
                                                      actual:&'a Self::BatchIO, lossf:&L)
        -> Result<U,TrainingError> where f64: From<U> + FromPrimitive, f64: FromPrimitive;
}
impl<'a,U,const N:usize> DeviceLinearOutput<'a,U,N> for DeviceCpu<U>
    where U: UnitValue<U> {
    type IO = Arr<U,N>;
    type BatchIO = SerializedVec<U,Arr<U,N>>;
    fn loss_linear<L>(&self, expected: &'a Arr<U,N>, actual: &'a Arr<U,N>, lossf: &L) -> Result<Arr<U,N>,TrainingError>
        where for<'b> L: LossFunction<U> + LossFunctionLinear<'b,U,Arr<U,N>,DeviceCpu<U>,N,Output=Arr<U,N>> {
        Ok(lossf.linear_derive(self,actual,expected)?)
    }

    fn loss_linear_by_canonical_link(&self, expected: &'a Arr<U,N>, actual: &'a Arr<U,N>) -> Result<Arr<U,N>,TrainingError> {
        let mut loss = Arr::new();

        for (l, (a, e)) in loss.iter_mut().zip(ArrView::<'a,U,N>::from(actual).iter().zip(expected.iter())) {
            *l = *a - *e;
        }

        Ok(loss)
    }

    fn loss_linear_total<L: LossFunction<U>>(&self, exptected: &'a Arr<U,N>, actual: &'a Arr<U,N>, lossf: &L)
        -> Result<U,TrainingError> {

        Ok(ArrView::<'a,U,N>::from(actual).iter().zip(exptected.iter()).fold(U::default(),| mut acc,(&a,&e) | {
            acc += lossf.apply(a,e);
            acc
        }))
    }

    fn loss_linear_batch_by_canonical_link(&self, expected: &'a SerializedVec<U,Arr<U,N>>,
                                               actual: &'a SerializedVec<U,Arr<U,N>>)
        -> Result<SerializedVec<U,Arr<U, N>>, TrainingError> where f64: From<U> {
        let n = U::from_usize(actual.size()).ok_or(TrainingError::TypeCastError(
            String::from("An error occurred when casting the batch size data type to U.")
        ))?;

        Ok(SerializedVecView::<'a,U,Arr<U,N>>::try_from(actual)?.par_iter().zip(expected.par_iter()).map(|(a,e)| {
            a.par_iter().zip(e.par_iter())
                .map(|(&a,&e)| (a - e) / n).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }

    fn batch_loss_linear<L>(&self, expected: &'a SerializedVec<U,Arr<U,N>>,
                               actual: &'a SerializedVec<U,Arr<U,N>>, lossf: &L)
        -> Result<SerializedVec<U,Arr<U,N>>, TrainingError>
        where for<'b> L: LossFunction<U> + BatchLossFunctionLinear<'b,U,Self::BatchIO,DeviceCpu<U>,N,Output=Self::BatchIO> {
        lossf.batch_linear_derive(self,expected,actual)
    }

    fn batch_loss_linear_total<L: LossFunction<U>>(&self, exptected: &'a SerializedVec<U,Arr<U,N>>,
                                                      actual: &'a SerializedVec<U,Arr<U,N>>, lossf: &L)
        -> Result<U, TrainingError> where f64: From<U> + FromPrimitive, f64: FromPrimitive {

        let n = f64::from_usize(exptected.len()).ok_or(TrainingError::TypeCastError(
            String::from("An error occurred when casting the batch size data type to f64.")
        ))?;

        let loss = SerializedVecView::<'a,U,Arr<U,N>>::try_from(actual)?.par_iter().zip(exptected.par_iter()).map(|(a,e)| {
            a.par_iter().cloned()
                .zip(e.par_iter().cloned())
                .reduce(|| (U::default(),U::default()), |(sum,d),(a,e)| {
                    (sum + lossf.apply(a,e),d)
                })
        }).map(|(sum,_)| sum).reduce(|| U::default(), |sum,l| sum + l);

        U::from_f64(f64::from(loss) / n).ok_or(TrainingError::TypeCastError(
            String::from("An error occurred in the type conversion of the total loss.")
        ))
    }
}
#[cfg(feature = "cuda")]
impl<'a,U,A,const N:usize> DeviceLinearOutput<'a,U,N> for DeviceGpu<U,A>
    where U: DataTypeInfo + UnitValue<U> + AsMutPtr<U>,
          A: CudaAllocator + 'static,
          DeviceGpu<U,A>: Device<U>,
          Arr<U,N>: ToCuda<U,A,Output=CudaTensor1dPtr<U,A,N>>,
          CudaPtr<U,A>: WriteMemory<U>,
          CudaTensor1dPtr<U,A,N>: Debug + ReadMemory<U> + WriteMemory<U>,
          CudaVec<U,CudaTensor1dPtr<U,A,N>,A>: ReadMemory<U> + 'a,
          SerializedVec<U,Arr<U,N>>: ToCuda<U,A,Output=CudaVec<U,CudaTensor1dPtr<U,A,N>,A>>,
          f64: From<U>,
          for<'b> &'b SerializedVec<U,Arr<U,N>>: ToCuda<U,A,Output=CudaVec<U,CudaTensor1dPtr<U,A,N>,A>>,
          for<'b> CudaVec<U,CudaTensor1dPtr<U,A,N>,A>: AsCudaPtr<'b>,
          for<'b> CudaVecView<'b,U,CudaTensor1dPtrView<'b,U,N>>: TryFrom<&'b CudaVec<U,CudaTensor1dPtr<U,A,N>,A>,Error=TypeConvertError>,
          for<'b> LossLinearBatchByCanonicalLink<'b,U,A,N>: Kernel<Args=LossLinearBatchByCanonicalLinkArgs<'b,U,A,N>>,
          for<'b> LossLinearByCanonicalLink<'b,U,A,N>: Kernel<Args=LossLinearByCanonicalLinkArgs<'b,U,A,N>> {
    type IO = CudaTensor1dPtr<U,A,N>;
    type BatchIO = CudaVec<U,CudaTensor1dPtr<U,A,N>,A>;
    fn loss_linear<L>(&self, expected: &'a Arr<U,N>, actual: &'a CudaTensor1dPtr<U,A,N>, lossf: &L)
                         -> Result<Self::IO, TrainingError>
        where for<'b> L: LossFunction<U> + LossFunctionLinear<'b,U,CudaTensor1dPtr<U,A,N>,DeviceGpu<U,A>,N,Output=CudaTensor1dPtr<U,A,N>> {
        Ok(lossf.linear_derive(self,&actual, &expected.to_cuda(self)?)?)
    }

    fn loss_linear_by_canonical_link(&self, expected: &'a Arr<U,N>, actual: &'a CudaTensor1dPtr<U,A,N>)
        -> Result<Self::IO, TrainingError> {
        let expected = expected.to_cuda(self)?;
        let expected = (&expected).into();

        let actual = actual.into();

        let output = CudaTensor1dPtr::<U,A,N>::new(self.get_allocator())?;

        let mut args = LossLinearByCanonicalLinkArgs::new(&expected, &actual, output.into(), N);

        let mut kernel = LossLinearByCanonicalLink::<'a,U,A,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn loss_linear_total<L: LossFunction<U>>(&self, exptected: &'a Arr<U,N>,
                                                actual: &'a CudaTensor1dPtr<U,A,N>, lossf: &L)
        -> Result<U, TrainingError> {
        Ok(actual.read_to_vec()?.iter().zip(exptected.iter()).fold(U::default(), |mut acc, (&a, &e)| {
            acc += lossf.apply(a, e);
            acc
        }))
    }

    fn loss_linear_batch_by_canonical_link(&self, expected: &'a SerializedVec<U,Arr<U,N>>,
                                           actual: &'a CudaVec<U,CudaTensor1dPtr<U,A,N>,A>)
        -> Result<Self::BatchIO, TrainingError> {
        let expected_ptr = expected.to_cuda(self)?;
        let expected_ptr = (&expected_ptr).try_into()?;

        let len = actual.size();

        let actual = actual.try_into()?;

        let output = CudaVec::<U,CudaTensor1dPtr<U,A,N>,A>::new(len, self.get_allocator())?;

        let mut args = LossLinearBatchByCanonicalLinkArgs::new(
            &expected_ptr,
            &actual,
            output,
            N,
            expected.len()
        );

        let mut kernel = LossLinearBatchByCanonicalLink::<'a,U,A,N>::new();

        kernel.launch(&mut args)?;

        Ok(args.output)
    }

    fn batch_loss_linear<L>(&self, expected: &'a SerializedVec<U, Arr<U,N>>,
                               actual: &'a CudaVec<U,CudaTensor1dPtr<U,A,N>,A>, lossf: &L)
                               -> Result<Self::BatchIO, TrainingError>
        where for<'b> L: LossFunction<U> + BatchLossFunctionLinear<'b,U,Self::BatchIO,DeviceGpu<U,A>,N,Output=Self::BatchIO> {
        let expected = expected.to_cuda(self)?;

        Ok(lossf.batch_linear_derive(self, &expected, actual)?)
    }
    fn batch_loss_linear_total<L: LossFunction<U>>(&self, exptected: &'a SerializedVec<U, Arr<U, N>>,
                                                      actual: &'a Self::BatchIO,
                                                      lossf: &L)
                                                      -> Result<U, TrainingError> {
        let actual = SerializedVec::<U,Arr<U,N>>::try_from(actual.read_to_vec()?.into_boxed_slice())?;

        let n = f64::from_usize(exptected.len()).ok_or(TrainingError::TypeCastError(
            String::from("An error occurred when casting the batch size data type to f64.")
        ))?;

        let loss = actual.par_iter().zip(exptected.par_iter()).map(|(a, e)| {
            a.iter().cloned()
                .zip(e.iter().cloned())
                .fold(U::default(), |sum, (a, e)| {
                    sum + lossf.apply(a, e)
                })
        }).fold(|| U::default(), | acc, s | {
            acc + s
        }).reduce(|| U::default(), | acc, s | acc + s);

        U::from_f64(f64::from(loss) / n).ok_or(TrainingError::TypeCastError(
            String::from("An error occurred in the type conversion of the total loss.")
        ))
    }
}