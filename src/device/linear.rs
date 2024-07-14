//! Implementation of the calculation process for full connected layers

use std::mem;
use cuda_runtime_sys::dim3;
use libc::c_uint;
use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};
use crate::arr::{Arr, Arr2, ArrView, DiffArr, SerializedVec, SerializedVecView};
use crate::cuda::{CudaConstPtr, CudaMemoryPoolPtr, CudaTensor1dPtr, CudaTensor2dPtr, DataTypeInfo, Kernel, Memory};
use crate::cuda::kernel::device::{BackwardLinearBatch, BackwardLinearBatchArgs, DiffLinearForward, DiffLinearForwardArgs, ForwardLinearBatch, ForwardLinearBatchArgs, LinearGradientBatch, LinearGradientBatchArgs, ReduceLinearBatch, ReduceLinearBatchArgs};
use crate::device::{DeviceCpu, DeviceGpu, DeviceMemoryPool, DeviceReduce};
use crate::error::{EvaluateError, TrainingError};
use crate::mem::AsRawSlice;
use crate::ope::UnitValue;
use crate::ope::Product;

/// Trait that defines the implementation of various calculation processes in the linear layer
pub trait DeviceLinear<U,T,B,const NI: usize,const NO: usize> where U: UnitValue<U> {
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
    fn forward_linear<'a>(&self, bias:&B, units:&T, input:ArrView<'a,U,NI>) -> Result<Arr<U, NO>, EvaluateError>;
    /// Error back propagation calculation
    /// # Arguments
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_linear<'a>(&self, units:&T, input:ArrView<'a,U,NO>) -> Result<Arr<U, NI>, TrainingError>;
    /// Calculate the gradient of the weights
    /// # Arguments
    /// * `o` - Input values from upper layers
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_weight_gradient<'a>(&self, o: ArrView<'a,U,NI>, loss: ArrView<'a,U,NO>)
                                -> Result<T, TrainingError>;
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
    fn batch_forward_linear<'a>(&self,bias:&B,units:&T,input:SerializedVecView<'a,U,Arr<U,NI>>)
                            -> Result<SerializedVec<U,Arr<U,NO>>,TrainingError>;
    /// Error back propagation in batch
    /// # Arguments
    /// * `units` - unit weights
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_linear<'a>(&self, units: &T, input: &'a SerializedVec<U,Arr<U,NO>>)
                             -> Result<SerializedVec<U,Arr<U, NI>>, TrainingError>;
    /// Calculate the gradient of the weights in batch
    /// # Arguments
    /// * `o` - Input values from upper layers
    /// * `loss` - loss
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_weight_gradient<'a>(&self, o: SerializedVecView<'a,U,Arr<U,NI>>, loss: &'a SerializedVec<U,Arr<U,NO>>)
                                      -> Result<T, TrainingError>;
    /// convolutional calculation
    /// # Arguments
    /// * `loss` - loss
    fn batch_linear_reduce<'a>(&self, loss: SerializedVecView<'a,U,Arr<U,NO>>) -> Result<B,TrainingError>;
}
impl<U,const NI: usize,const NO: usize> DeviceLinear<U,Arr2<U,NI,NO>,Arr<U,NO>,NI,NO> for DeviceCpu<U> where U: UnitValue<U> {
    #[inline]
    fn forward_linear<'a>(&self, bias: &Arr<U, NO>, units: &Arr2<U, NI, NO>, input: ArrView<'a,U, NI>) -> Result<Arr<U, NO>, EvaluateError> {
        Ok(input.product(units) + bias)
    }

    #[inline]
    fn backward_linear<'a>(&self, units: &Arr2<U, NI, NO>, input: ArrView<'a,U,NO>) -> Result<Arr<U, NI>, TrainingError> {
        Ok(units.iter().map(|u| {
            u.iter().zip(input.iter())
                .map(|(&w,&l)| w * l).fold(U::default(), |acc,g|{
                acc + g
            })
        }).collect::<Vec<U>>().try_into().map_err(|e| TrainingError::from(e))?)
    }

    #[inline]
    fn backward_weight_gradient<'a>(&self, o: ArrView<'a,U,NI>, loss: ArrView<'a,U,NO>) -> Result<Arr2<U, NI, NO>, TrainingError> {
        Ok(o.iter().cloned().map(|o| {
            loss.iter().cloned().map(|l| o * l).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,NO>>,_>>()?.try_into().map_err(|e| TrainingError::from(e))?)
    }

    #[inline]
    fn batch_backward_linear<'a>(&self, units: &Arr2<U, NI, NO>, input: &'a SerializedVec<U,Arr<U, NO>>)
                             -> Result<SerializedVec<U,Arr<U, NI>>, TrainingError> {
        Ok(input.par_iter().map(|l| {
            self.backward_linear(units,l)
        }).collect::<Result<Vec<Arr<U,NI>>,_>>()?.into())
    }

    #[inline]
    fn batch_forward_linear<'a>(&self,bias:&Arr<U,NO>,units:&Arr2<U,NI,NO>,input:SerializedVecView<'a,U,Arr<U,NI>>)
                            -> Result<SerializedVec<U,Arr<U,NO>>,TrainingError> {
        input.par_iter().map(|input| {
            self.forward_linear(bias, units, input)
        }).collect::<Result<Vec<Arr<U, NO>>, _>>().map(|r| r.into()).map_err(|e| TrainingError::from(e))
    }

    #[inline]
    fn batch_backward_weight_gradient<'a>(&self, o: SerializedVecView<'a,U,Arr<U,NI>>, loss: &'a SerializedVec<U, Arr<U,NO>>)
                                      -> Result<Arr2<U,NI,NO>, TrainingError> {
        Ok(o.par_iter().zip(loss.par_iter()).map(|(o,l)| {
            self.backward_weight_gradient(o, l)
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
    fn batch_linear_reduce<'a>(&self, loss: SerializedVecView<'a,U,Arr<U,NO>>) -> Result<Arr<U,NO>,TrainingError> {
        self.reduce(loss)
    }
}
impl<U,const NI: usize, const NO: usize> DeviceLinear<U,CudaTensor2dPtr<U,NI,NO>,CudaTensor1dPtr<U,NO>,NI,NO> for DeviceGpu<U>
    where U: DataTypeInfo + UnitValue<U>,
          for<'a> ForwardLinearBatch::<'a,U,NI,NO>: Kernel<Args=ForwardLinearBatchArgs<'a,U,NI,NO>>,
          for<'a> BackwardLinearBatch::<'a,U,NI,NO>: Kernel<Args=BackwardLinearBatchArgs<'a,U,NI,NO>>,
          LinearGradientBatch::<U,NI,NO>: Kernel<Args=LinearGradientBatchArgs<U,NI,NO>>,
          ReduceLinearBatch::<U,NO>: Kernel<Args=ReduceLinearBatchArgs<U,NO>> {
    #[inline]
    fn forward_linear<'a>(&self, bias: &CudaTensor1dPtr<U,NO>, units: &CudaTensor2dPtr<U,NI,NO>, input: ArrView<'a,U,NI>)
                          -> Result<Arr<U, NO>, EvaluateError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;
        let output = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI)?;

        let mut args = ForwardLinearBatchArgs::new(input_ptr,
                                                   CudaConstPtr::new(units),
                                                   CudaConstPtr::new(bias),
                                                   output,
                                                   1);

        let mut kernel = ForwardLinearBatch::<U,NI,NO>::new();

        kernel.launch(dim3 { x: NO as c_uint, y: 1, z: 1},
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * 2 * mem::size_of::<U>())?;

        kernel.device_synchronize()?;

        Ok(args.output.read_to_vec()?.try_into()?)
    }

    #[inline]
    fn backward_linear<'a>(&self, units: &CudaTensor2dPtr<U,NI,NO>, input: ArrView<'a,U,NO>) -> Result<Arr<U, NI>, TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let output = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NO)?;

        let mut args = BackwardLinearBatchArgs::new(input_ptr,
                                                    CudaConstPtr::new(units),
                                                    output,
                                                    1);

        let mut kernel = BackwardLinearBatch::<U,NI,NO>::new();

        kernel.launch(dim3 { x: NI as c_uint, y: 1, z: 1},
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<U>())?;

        Ok(args.output.read_to_vec()?.try_into()?)
    }

    #[inline]
    fn backward_weight_gradient<'a>(&self, o: ArrView<'a,U,NI>, loss: ArrView<'a,U,NO>) -> Result<CudaTensor2dPtr<U, NI,NO>, TrainingError> {
        let mut input_ptr = CudaMemoryPoolPtr::new(NI,&self.memory_pool)?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO,&self.memory_pool)?;
        let output = CudaTensor2dPtr::<U,NI,NO>::new(&self.memory_pool)?;

        input_ptr.memcpy(o.as_raw_slice().as_ptr(),NI)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),NO)?;

        let mut args = LinearGradientBatchArgs::new(
            loss_ptr,
            input_ptr,
            output,
            1
        );

        let mut kernel = LinearGradientBatch::<U,NI,NO>::new();

        kernel.launch(dim3 { x: (NI * NO) as c_uint, y: 1, z: 1},
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<U>())?;

        Ok(args.output)
    }

    #[inline]
    fn batch_forward_linear<'a>(&self,bias:&CudaTensor1dPtr<U,NO>,units:&CudaTensor2dPtr<U,NI,NO>,input:SerializedVecView<'a,U,Arr<U,NI>>)
                                -> Result<SerializedVec<U,Arr<U,NO>>,TrainingError> {
        let n = input.len();

        let mut input_ptr = CudaMemoryPoolPtr::new(NI*n,&self.memory_pool)?;
        let output = CudaMemoryPoolPtr::new(NO*n,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NI*n)?;

        let mut args = ForwardLinearBatchArgs::new(input_ptr,
                                                   CudaConstPtr::new(units),
                                                   CudaConstPtr::new(bias),
                                                   output,
                                                   n);

        let mut kernel = ForwardLinearBatch::<U,NI,NO>::new();

        kernel.launch(dim3 { x: (NO * n) as c_uint, y: 1, z: 1},
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * 2 * mem::size_of::<U>())?;

        kernel.device_synchronize()?;

        Ok(args.output.read_to_vec()?.try_into()?)
    }

    #[inline]
    fn batch_backward_linear<'a>(&self, units: &CudaTensor2dPtr<U, NI, NO>, input: &'a SerializedVec<U,Arr<U, NO>>) -> Result<SerializedVec<U, Arr<U, NI>>, TrainingError> {
        let n = input.len();

        let mut input_ptr = CudaMemoryPoolPtr::new(NO*n,&self.memory_pool)?;
        let output = CudaMemoryPoolPtr::new(NI*n,&self.memory_pool)?;

        input_ptr.memcpy(input.as_raw_slice().as_ptr(),NO*n)?;

        let mut args = BackwardLinearBatchArgs::new(input_ptr,
                                                    CudaConstPtr::new(units),
                                                    output,
                                                    n);

        let mut kernel = BackwardLinearBatch::<U,NI,NO>::new();

        kernel.launch(dim3 { x: (NI * n) as c_uint, y: 1, z: 1},
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<U>())?;

        Ok(args.output.read_to_vec()?.try_into()?)
    }

    #[inline]
    fn batch_backward_weight_gradient<'a>(&self, o: SerializedVecView<'a,U,Arr<U, NI>>, loss: &'a SerializedVec<U, Arr<U, NO>>)
                                          -> Result<CudaTensor2dPtr<U, NI, NO>, TrainingError> {
        let n = loss.len();

        let mut input_ptr = CudaMemoryPoolPtr::new(NI*n,&self.memory_pool)?;
        let mut loss_ptr = CudaMemoryPoolPtr::new(NO*n,&self.memory_pool)?;
        let output = CudaTensor2dPtr::<U,NI,NO>::new(&self.memory_pool)?;
        input_ptr.memcpy(o.as_raw_slice().as_ptr(),NI*n)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(),NO*n)?;

        let mut args = LinearGradientBatchArgs::new(
            loss_ptr,
            input_ptr,
            output,
            n
        );

        let mut kernel = LinearGradientBatch::<U,NI,NO>::new();

        kernel.launch(dim3 { x: (NI * NO) as c_uint, y: 1, z: 1},
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<U>())?;

        kernel.device_synchronize()?;

        Ok(args.output)
    }

    #[inline]
    fn batch_linear_reduce<'a>(&self, loss: SerializedVecView<'a,U,Arr<U,NO>>) -> Result<CudaTensor1dPtr<U,NO>,TrainingError> {
        self.reduce(loss)
    }
}
/// Trait that defines the implementation of various computational processes in the differentially applicable linear layer
pub trait DeviceDiffLinear<'a,U,T,const NI: usize,const NO: usize>
    where U: UnitValue<U> {
    fn forward_diff_linear(&self,input:&'a DiffArr<U,NI>,units: &'a T,output: ArrView<'a,U,NO>) -> Result<Arr<U, NO>,EvaluateError>;
}
impl<'a,U,const NI:usize,const NO:usize> DeviceDiffLinear<'a,U,Arr2<U,NI,NO>,NI,NO> for DeviceCpu<U>
    where U: UnitValue<U> {
    fn forward_diff_linear(&self, input: &'a DiffArr<U, NI>, units: &'a Arr2<U, NI, NO>, output: ArrView<'a,U,NO>) -> Result<Arr<U, NO>,EvaluateError> {
        let mut output:Arr<U,NO> = output.into();

        for &(i,d) in input.iter() {
            for (o,j) in output.iter_mut().zip(0..NO) {
                *o += units[(i,j)] * d;
            }
        }
        Ok(output)
    }
}
impl<'a,U,const NI:usize,const NO:usize> DeviceDiffLinear<'a,U,CudaTensor2dPtr<U,NI,NO>,NI,NO> for DeviceGpu<U>
    where U: UnitValue<U> + DataTypeInfo,
          for<'b> DiffLinearForward<'b,U,NI,NO>: Kernel<Args=DiffLinearForwardArgs<'b,U,NI,NO>> {
    fn forward_diff_linear(&self, input: &'a DiffArr<U, NI>, units: &'a CudaTensor2dPtr<U, NI, NO>, output: ArrView<'a,U,NO>)
        -> Result<Arr<U, NO>,EvaluateError> {
        let len = input.len();

        let (indexes,input) = input.iter().fold((Vec::new(),Vec::new()), | mut acc, &(i,d)| {
            acc.0.push(i);
            acc.1.push(d);

            acc
        });

        let mut indexes_ptr = CudaMemoryPoolPtr::new(len,self.get_memory_pool())?;
        let mut input_ptr = CudaMemoryPoolPtr::new(len,self.get_memory_pool())?;

        indexes_ptr.memcpy(indexes.as_ptr(),len)?;
        input_ptr.memcpy(input.as_ptr(),len)?;

        let mut output_ptr = CudaTensor1dPtr::<U,NO>::new(self.get_memory_pool())?;

        output_ptr.memcpy(output.as_ptr(),NO)?;

        let mut args = DiffLinearForwardArgs::new(indexes_ptr,input_ptr,units,output_ptr,NO,len);

        let mut kernel = DiffLinearForward::new();

        kernel.launch(dim3 { x: NO as c_uint, y: 1, z: 1},
                      dim3 { x: 1024, y: 1, z: 1 },&mut args,1024 * mem::size_of::<U>())?;

        Ok(args.output.read_to_vec()?.try_into()?)
    }
}
