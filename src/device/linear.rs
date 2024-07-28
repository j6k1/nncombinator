//! Implementation of the calculation process for full connected layers

use std::fmt::Debug;
use std::mem;
use cuda_runtime_sys::dim3;
use libc::c_uint;
use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};
use crate::arr::{Arr, Arr2, ArrView, DiffArr, IntoConverter, SerializedVec, SerializedVecView};
use crate::cuda::{CudaMemoryPoolPtr, CudaTensor1dPtr, CudaTensor1dPtrView, CudaTensor2dPtr, CudaVec, CudaVecView, DataTypeInfo, Kernel, Memory};
use crate::cuda::kernel::device::{BackwardLinear, BackwardLinearArgs, BackwardLinearBatch, BackwardLinearBatchArgs, DiffLinearForward, DiffLinearForwardArgs, ForwardLinear, ForwardLinearArgs, ForwardLinearBatch, ForwardLinearBatchArgs, LinearGradient, LinearGradientArgs, LinearGradientBatch, LinearGradientBatchArgs, ReduceLinearBatch, ReduceLinearBatchArgs};
use crate::device::{DeviceCpu, DeviceGpu, DeviceMemoryPool, DeviceReduce};
use crate::error::{EvaluateError, TrainingError, TypeConvertError, UnsupportedOperationError};
use crate::layer::{BatchDataType, BatchSize, DiffInput};
use crate::ope::UnitValue;
use crate::ope::Product;

/// Trait that defines the implementation of various calculation processes in the linear layer
pub trait DeviceLinear<U,T,B,I,const NI: usize,const NO: usize>
    where U: UnitValue<U>,
          I: BatchDataType {
    type Output: BatchDataType + Debug + 'static;
    type BatchOutput: Debug + 'static;
    type LossOutput: BatchDataType + Debug + 'static;
    type BatchLossOutput: Debug + 'static;
    /// Forward propagation calculation
    /// # Arguments
    /// * `bias` - bias weights
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn forward_linear<'a>(&self, bias:&B, units:&T, input:&'a I) -> Result<Self::Output, EvaluateError>;
    /// Error back propagation calculation
    /// # Arguments
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_linear<'a>(&self, units:&T, input:&'a Self::Output) -> Result<Self::LossOutput, TrainingError>;
    /// Calculate the gradient of the weights
    /// # Arguments
    /// * `o` - Input values from upper layers
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_weight_gradient<'a>(&self, o: &'a I, loss: &'a Self::Output) -> Result<T, TrainingError>;
    /// Calculate the gradient of the bias weights
    /// # Arguments
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_bias_weight_gradient<'a>(&self, loss: Self::Output) -> Result<B, TrainingError>;
    /// Forward propagation calculation in batch
    /// # Arguments
    /// * `bias` - bias weights
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_forward_linear<'a>(&self,bias:&B,units:&T,input: &'a <I as BatchDataType>::Type)
        -> Result<Self::BatchOutput,TrainingError>;
    /// Error back propagation in batch
    /// # Arguments
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_linear<'a>(&self, units: &T, input: &'a Self::BatchOutput)
        -> Result<Self::BatchLossOutput, TrainingError>;
    /// Calculate the gradient of the weights in batch
    /// # Arguments
    /// * `o` - Input values from upper layers
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_weight_gradient<'a>(&self, o: &'a <I as BatchDataType>::Type, loss: &'a Self::BatchOutput)
        -> Result<T, TrainingError>;
    /// convolutional calculation
    /// # Arguments
    /// * `loss` - loss
    fn batch_linear_reduce<'a>(&self, loss: &'a Self::BatchOutput) -> Result<B,TrainingError>;
}
impl<U,I,const NI: usize,const NO: usize> DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,I,NI,NO> for DeviceCpu<U>
    where U: UnitValue<U>,
          I: BatchDataType,
          for<'a> ArrView<'a,U,NI>: From<&'a I>,
          for<'a> SerializedVecView<'a,U,Arr<U,NI>>: TryFrom<&'a <I as BatchDataType>::Type,Error=TypeConvertError>,
          Self: DeviceReduce<SerializedVec<U,Arr<U,NO>>,Arr<U,NO>,U,NO> {
    type Output = Arr<U,NO>;
    type BatchOutput = <Arr<U,NO> as BatchDataType>::Type;
    type LossOutput = Arr<U,NI>;
    type BatchLossOutput = SerializedVec<U,Arr<U,NI>>;
    #[inline]
    fn forward_linear<'a>(&self, bias: &Arr<U, NO>, units: &Arr2<U, NI, NO>, input: &'a I) -> Result<Arr<U, NO>, EvaluateError> {
        Ok(ArrView::<'a,U,NI>::from(input).product(units) + bias)
    }

    #[inline]
    fn backward_linear<'a>(&self, units: &Arr2<U,NI,NO>, input: &'a Arr<U,NO>) -> Result<Arr<U, NI>, TrainingError> {
        Ok(units.iter().map(|u| {
            u.iter().zip(input.iter())
                .map(|(&w,&l)| w * l).fold(U::default(), |acc,g|{
                acc + g
            })
        }).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))?)
    }

    #[inline]
    fn backward_weight_gradient<'a>(&self, o: &'a I, loss: &'a Arr<U,NO>) -> Result<Arr2<U,NI,NO>, TrainingError> {
        Ok(ArrView::<'a,U,NI>::from(o).iter().cloned().map(|o| {
            loss.iter().cloned().map(|l| o * l).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,NO>>,_>>()?.try_into().map_err(|e| TrainingError::from(e))?)
    }

    fn backward_bias_weight_gradient<'a>(&self, loss: Self::Output) -> Result<Arr<U,NO>, TrainingError> {
        Ok(loss.into())
    }
    #[inline]
    fn batch_backward_linear<'a>(&self, units: &Arr2<U,NI,NO>, input: &'a SerializedVec<U,Arr<U,NO>>)
                             -> Result<SerializedVec<U,Arr<U,NI>>, TrainingError> {
        Ok(input.par_iter().map(|l| {
            units.iter().map(|u| {
                u.iter().zip(l.iter())
                    .map(|(&w,&l)| w * l).fold(U::default(), |acc,g|{
                    acc + g
                })
            }).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,NI>>,_>>()?.into())
    }

    #[inline]
    fn batch_forward_linear<'a>(&self,bias: &Arr<U,NO>, units: &Arr2<U,NI,NO>, input: &'a <I as BatchDataType>::Type)
                            -> Result<SerializedVec<U,Arr<U,NO>>,TrainingError> {
        Ok(SerializedVecView::<'a,U,Arr<U,NI>>::try_from(input)?.par_iter().map(|input| {
            input.product(units) + bias
        }).collect::<Vec<Arr<U,NO>>>().into())
    }

    #[inline]
    fn batch_backward_weight_gradient<'a>(&self, o: &'a <I as BatchDataType>::Type, loss: &'a SerializedVec<U,Arr<U,NO>>)
                                      -> Result<Arr2<U,NI,NO>, TrainingError> {
        Ok(SerializedVecView::<'a,U,Arr<U,NI>>::try_from(o)?.par_iter().zip(loss.par_iter()).map(|(o,l)| {
            o.iter().cloned().map(|o| {
                l.iter().cloned().map(|l| o * l).collect::<Vec<U>>().try_into()
            }).collect::<Result<Vec<Arr<U,NO>>,_>>()?.try_into()
        }).reduce(|| Ok(Arr2::new()), | acc, g | {
            acc.and_then(| mut acc | g.and_then(|g| {
                for (mut acc,g) in acc.iter_mut().zip(g.iter()) {
                    for (acc,&g) in acc.iter_mut().zip(g.iter()) {
                        *acc += g;
                    }
                }

                Ok(acc)
            }))
        })?)
    }

    #[inline]
    fn batch_linear_reduce<'a>(&self, loss: &'a SerializedVec<U,Arr<U,NO>>) -> Result<Arr<U,NO>,TrainingError> {
        self.reduce(loss)
    }
}
impl<U,I,const NI: usize, const NO: usize> DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,I,NI,NO> for DeviceGpu<U>
    where U: DataTypeInfo + UnitValue<U>,
          I: BatchDataType,
          <I as BatchDataType>::Type: BatchSize,
          Self: DeviceReduce<CudaVec<U,CudaTensor1dPtr<U,NO>>,CudaTensor1dPtr<U,NO>,U,NO>,
          for<'a> CudaTensor1dPtrView<'a,U,NI>: From<&'a I>,
          for<'a> CudaVecView<'a,U,CudaTensor1dPtr<U,NI>>: TryFrom<&'a <I as BatchDataType>::Type,Error=TrainingError>,
          for<'b> ForwardLinear::<'b,U,NI,NO>: Kernel<Args=ForwardLinearArgs<'b,U,NI,NO>>,
          for<'b> BackwardLinear::<'b,U,NI,NO>: Kernel<Args=BackwardLinearArgs<'b,U,NI,NO>>,
          for<'b> LinearGradient::<'b,U,NI,NO>: Kernel<Args=LinearGradientArgs<'b,U,NI,NO>>,
          for<'b> ForwardLinearBatch::<'b,U,NI,NO>: Kernel<Args=ForwardLinearBatchArgs<'b,U,NI,NO>>,
          for<'b> BackwardLinearBatch::<'b,U,NI,NO>: Kernel<Args=BackwardLinearBatchArgs<'b,U,NI,NO>>,
          for<'b> LinearGradientBatch::<'b,U,NI,NO>: Kernel<Args=LinearGradientBatchArgs<'b,U,NI,NO>>,
          for<'b> ReduceLinearBatch::<'b,U,NO>: Kernel<Args=ReduceLinearBatchArgs<'b,U,NO>> {
    type Output = CudaTensor1dPtr<U,NO>;
    type BatchOutput = CudaVec<U,CudaTensor1dPtr<U,NO>>;
    type LossOutput = CudaTensor1dPtr<U,NI>;
    type BatchLossOutput = CudaVec<U,CudaTensor1dPtr<U,NI>>;
    #[inline]
    fn forward_linear<'a>(&self, bias: &CudaTensor1dPtr<U,NO>, units: &CudaTensor2dPtr<U,NI,NO>, input: &'a I)
                          -> Result<CudaTensor1dPtr<U,NO>, EvaluateError> {
        let input = input.into();
        let output = CudaTensor1dPtr::<U,NO>::with_initializer(self.get_memory_pool(),Default::default)?;

        let mut args = ForwardLinearArgs::new(
                                                   &input,
                                                   units,
                                                   bias,
                                                   output);

        let mut kernel = ForwardLinear::<U,NI,NO>::new();

        kernel.launch(dim3 { x: NO as c_uint, y: 1, z: (NI as c_uint + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,32 * 2 * mem::size_of::<U>())?;

        Ok(args.output)
    }

    #[inline]
    fn backward_linear<'a>(&self, units: &CudaTensor2dPtr<U,NI,NO>, input: &'a Self::Output)
        -> Result<Self::LossOutput, TrainingError> {
        let input_ptr = input.into();
        let output = CudaTensor1dPtr::<U,NI>::with_initializer(&self.memory_pool,Default::default)?;

        let mut args = BackwardLinearArgs::new(&input_ptr,
                                                    units,
                                                    output);

        let mut kernel = BackwardLinear::<U,NI,NO>::new();

        kernel.launch(dim3 { x: NI as c_uint, y: 1, z: (NO as c_uint + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,32 * mem::size_of::<U>())?;

        Ok(args.output.into())
    }

    #[inline]
    fn backward_weight_gradient<'a>(&self, o: &'a I, loss: &'a Self::Output) -> Result<CudaTensor2dPtr<U,NI,NO>, TrainingError> {
        let input_ptr = o.into();
        let loss_ptr = loss.into();
        let output = CudaTensor2dPtr::<U,NI,NO>::with_initializer(&self.memory_pool,Default::default)?;

        let mut args = LinearGradientArgs::new(
            &loss_ptr,
            &input_ptr,
            output
        );

        let mut kernel = LinearGradient::<U,NI,NO>::new();

        kernel.launch(dim3 { x: (NI * NO) as c_uint, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,32 * mem::size_of::<U>())?;

        Ok(args.output)
    }

    fn backward_bias_weight_gradient<'a>(&self, loss: Self::Output) -> Result<CudaTensor1dPtr<U,NO>, TrainingError> {
        Ok(loss.into())
    }
    #[inline]
    fn batch_forward_linear<'a>(&self,bias:&CudaTensor1dPtr<U,NO>,units:&CudaTensor2dPtr<U,NI,NO>,
                                input: &'a <I as BatchDataType>::Type)
                                -> Result<Self::BatchOutput,TrainingError> {
        let n = input.size();

        let input = input.try_into()?;
        let output = CudaVec::<U,CudaTensor1dPtr<U,NO>>::with_initializer(n,&self.memory_pool,Default::default)?;

        let mut args = ForwardLinearBatchArgs::new(&input,
                                                   units,
                                                   bias,
                                                   output,
                                                   n);

        let mut kernel = ForwardLinearBatch::<U,NI,NO>::new();

        kernel.launch(dim3 { x: (NO * n) as c_uint, y: 1, z: (NI as c_uint + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,32 * 2 * mem::size_of::<U>())?;

        Ok(args.output)
    }

    #[inline]
    fn batch_backward_linear<'a>(&self, units: &CudaTensor2dPtr<U, NI, NO>, input: &'a Self::BatchOutput)
        -> Result<Self::BatchLossOutput, TrainingError> {
        let n = input.size();

        let input_ptr = input.try_into()?;

        let output = CudaVec::<U,CudaTensor1dPtr<U,NI>>::with_initializer(n,&self.memory_pool,Default::default)?;

        let mut args = BackwardLinearBatchArgs::new(&input_ptr,
                                                    units,
                                                    output,
                                                    n);

        let mut kernel = BackwardLinearBatch::<U,NI,NO>::new();

        kernel.launch(dim3 { x: (NI * n) as c_uint, y: 1, z: (NO as c_uint + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,32 * mem::size_of::<U>())?;

        Ok(args.output.into_converter().try_into()?)
    }

    #[inline]
    fn batch_backward_weight_gradient<'a>(&self, o: &'a <I as BatchDataType>::Type,
                                          loss: &'a Self::BatchOutput)
        -> Result<CudaTensor2dPtr<U, NI, NO>, TrainingError> {
        let n = loss.size();

        let o = o.try_into()?;
        let loss_ptr = loss.try_into()?;
        let output = CudaTensor2dPtr::<U,NI,NO>::with_initializer(&self.memory_pool,Default::default)?;

        let mut args = LinearGradientBatchArgs::new(
            &loss_ptr,
            &o,
            output,
            n
        );

        let mut kernel = LinearGradientBatch::<U,NI,NO>::new();

        kernel.launch(dim3 { x: (NI * NO) as c_uint, y: 1, z: (n as c_uint + 1023) / 1024 },
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,32 * mem::size_of::<U>())?;

        Ok(args.output)
    }

    #[inline]
    fn batch_linear_reduce<'a>(&self, loss: &'a Self::BatchOutput) -> Result<CudaTensor1dPtr<U,NO>,TrainingError> {
        self.reduce(loss)
    }
}
/// Trait that defines the implementation of various computational processes in the differentially applicable linear layer
pub trait DeviceDiffLinear<U,T,B,const NI: usize,const NO: usize>
    where U: UnitValue<U> {
    type Output;
    fn forward_diff_linear<'a>(&self,units: &T,bias: &B, input: &'a DiffInput<DiffArr<U,NI>,U,NI,NO>) -> Result<Self::Output,EvaluateError>;
    fn backward_diff_weight_gradient<'a>(&self, o: &'a DiffInput<DiffArr<U,NI>,U,NI,NO>, loss: &'a Self::Output) -> Result<T, TrainingError>;
}
impl<U,const NI:usize,const NO:usize> DeviceDiffLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> for DeviceCpu<U>
    where U: UnitValue<U> {
    type Output = Arr<U,NO>;
    #[inline]
    fn forward_diff_linear<'a>(&self, units: &Arr2<U, NI, NO>, bias: &Arr<U,NO>, input: &'a DiffInput<DiffArr<U,NI>,U,NI,NO>) -> Result<Arr<U, NO>,EvaluateError> {
        match input {
            DiffInput::Diff(d,output) => {
                let mut output:Arr<U,NO> = output.clone();

                for &(i,d) in d.iter() {
                    for (o,j) in output.iter_mut().zip(0..NO) {
                        *o += units[(i,j)] * d;
                    }
                }
                Ok(output)
            },
            DiffInput::NotDiff(input) => {
                Ok(ArrView::<'a,U,NI>::from(input).product(units) + bias)
            }
        }
    }

    #[inline]
    fn backward_diff_weight_gradient<'a>(&self, o: &'a DiffInput<DiffArr<U,NI>,U,NI,NO>, loss: &'a Arr<U,NO>) -> Result<Arr2<U,NI,NO>, TrainingError> {
        match o {
            DiffInput::Diff(_,_) => {
                Err(TrainingError::UnsupportedOperationError(UnsupportedOperationError(
                    String::from("Training from difference information is not supported.")
                )))
            },
            DiffInput::NotDiff(o) => {
                Ok(ArrView::<'a,U,NI>::from(o).iter().cloned().map(|o| {
                    loss.iter().cloned().map(|l| o * l).collect::<Vec<U>>().try_into()
                }).collect::<Result<Vec<Arr<U,NO>>,_>>()?.try_into().map_err(|e| TrainingError::from(e))?)
            }
        }
    }
}
impl<U,const NI:usize,const NO:usize> DeviceDiffLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO> for DeviceGpu<U>
    where U: UnitValue<U> + DataTypeInfo,
          for<'b> ForwardLinear::<'b,U,NI,NO>: Kernel<Args=ForwardLinearArgs<'b,U,NI,NO>>,
          for<'b> LinearGradient::<'b,U,NI,NO>: Kernel<Args=LinearGradientArgs<'b,U,NI,NO>>,
          for<'b> ReduceLinearBatch::<'b,U,NO>: Kernel<Args=ReduceLinearBatchArgs<'b,U,NO>>,
          for<'b> DiffLinearForward<'b,U,NI,NO>: Kernel<Args=DiffLinearForwardArgs<'b,U,NI,NO>> {
    type Output = CudaTensor1dPtr<U,NO>;

    #[inline]
    fn forward_diff_linear<'a>(&self, units: &CudaTensor2dPtr<U,NI,NO>, bias: &CudaTensor1dPtr<U,NO>, input: &'a DiffInput<DiffArr<U,NI>,U,NI,NO>)
        -> Result<CudaTensor1dPtr<U,NO>,EvaluateError> {
        match input {
            DiffInput::Diff(d, output) => {
                let len = d.len();

                let (indexes, input) = d.iter().fold((Vec::new(), Vec::new()), |mut acc, &(i, d)| {
                    acc.0.push(i);
                    acc.1.push(d);

                    acc
                });

                let mut indexes_ptr = CudaMemoryPoolPtr::new(len, self.get_memory_pool())?;
                let mut input_ptr = CudaMemoryPoolPtr::new(len, self.get_memory_pool())?;

                indexes_ptr.memcpy(indexes.as_ptr(), len)?;
                input_ptr.memcpy(input.as_ptr(), len)?;

                let mut output_ptr = CudaTensor1dPtr::<U, NO>::new(self.get_memory_pool())?;

                output_ptr.memcpy(output.as_ptr(), NO)?;

                let mut args = DiffLinearForwardArgs::new(indexes_ptr, input_ptr, units, output_ptr, NO, len);

                let mut kernel = DiffLinearForward::new();

                kernel.launch(dim3 { x: NO as c_uint, y: 1, z: 1 },
                              dim3 { x: 1024, y: 1, z: 1 }, &mut args, 1024 * mem::size_of::<U>())?;

                Ok(args.output)
            },
            DiffInput::NotDiff(input) => {
                let output = CudaTensor1dPtr::<U, NO>::with_initializer(self.get_memory_pool(), Default::default)?;

                let mut input_ptr = CudaTensor1dPtr::<U,NI>::new(self.get_memory_pool())?;

                input_ptr.memcpy(input.as_ptr(),NI)?;

                let input_ptr = (&input_ptr).into();

                let mut args = ForwardLinearArgs::new(
                    &input_ptr,
                    units,
                    bias,
                    output);

                let mut kernel = ForwardLinear::<U, NI, NO>::new();

                kernel.launch(dim3 { x: NO as c_uint, y: 1, z: (NI as c_uint + 1023) / 1024 },
                              dim3 { x: 1024, y: 1, z: 1 }, &mut args, 32 * 2 * mem::size_of::<U>())?;

                Ok(args.output)
            }
        }
    }

    #[inline]
    fn backward_diff_weight_gradient<'a>(&self, o: &'a DiffInput<DiffArr<U,NI>,U,NI,NO>, loss: &'a Self::Output) -> Result<CudaTensor2dPtr<U,NI,NO>, TrainingError> {
        match o {
            DiffInput::Diff(_, _) => {
                Err(TrainingError::UnsupportedOperationError(UnsupportedOperationError(
                    String::from("Training from difference information is not supported.")
                )))
            },
            DiffInput::NotDiff(o) => {
                let mut input_ptr = CudaTensor1dPtr::<U,NI>::new(self.get_memory_pool())?;

                input_ptr.memcpy(o.as_ptr(),NI)?;

                let input_ptr = (&input_ptr).into();

                let loss_ptr = loss.into();
                let output = CudaTensor2dPtr::<U, NI, NO>::with_initializer(&self.memory_pool, Default::default)?;

                let mut args = LinearGradientArgs::new(
                    &loss_ptr,
                    &input_ptr,
                    output
                );

                let mut kernel = LinearGradient::<U, NI, NO>::new();

                kernel.launch(dim3 { x: (NI * NO) as c_uint, y: 1, z: 1 },
                              dim3 { x: 1024, y: 1, z: 1 }, &mut args, 32 * mem::size_of::<U>())?;

                Ok(args.output)
            }
        }
    }
}
