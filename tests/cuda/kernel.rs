extern crate nncombinator;
extern crate rand;

use nncombinator::arr::{Arr, Arr2, IntoConverter, SerializedVec};
use nncombinator::cuda::{CudaTensor1dPtr, CudaTensor1dPtrView, CudaTensor2dPtr, CudaVec, CudaVecView, ReadMemory, WriteMemory, Kernel, AsCudaMutPtr, AsCudaView};
use nncombinator::cuda::allocator::{DeviceAlloc, MemoryPoolAllocator};
use nncombinator::cuda::kernel::device::{ForwardLinearBatch, ForwardLinearBatchArgs, BackwardLinearBatch, BackwardLinearBatchArgs, LinearGradientBatch, LinearGradientBatchArgs, ReduceLinearBatchArgs, ReduceLinearBatch};
use nncombinator::device::{DeviceCpu, DeviceReduce};
use rand::Rng;
use nncombinator::device::linear::DeviceLinear;
use crate::common::SHARED_MEMORY_POOL;
use crate::common::gen_inputs;
use crate::common::approx_eq_slice;
use crate::common::upload_inputs_to_device;
use std::convert::TryFrom;
use nncombinator::layer::BatchSize;

const NI: usize = 500;
const NO: usize = 600;
const BATCH: usize = 400;

type A = MemoryPoolAllocator<DeviceAlloc>;

#[test]
fn test_kernel_forward_linear_batch_matches_cpu() {
    let device = DeviceCpu::<f32>::new().unwrap();

    let (bias,units,inputs) = gen_inputs();

    // CPU result
    let cpu_out = device.batch_forward_linear(&bias,&units,&inputs).unwrap();

    // GPU kernel
    let alloc: &A = &SHARED_MEMORY_POOL;

    let (d_bias,d_units,d_inputs) = upload_inputs_to_device::<A>(alloc,&bias,&units,&inputs);

    // Prepare output buffer (zero-init required)
    let d_output = CudaVec::<f32,CudaTensor1dPtr<f32,A,NO>,A>::new(BATCH,alloc).unwrap();

    // Build args
    let d_inputs = &d_inputs;
    let conv = d_inputs.as_cuda_view().into_converter();
    let input_view = CudaVecView::<f32,CudaTensor1dPtrView<f32,NI>>::try_from(conv).unwrap();
    let mut args = ForwardLinearBatchArgs::<'_,f32,A,NI,NO>::new(&input_view,&d_units,&d_bias,d_output,BATCH);

    let mut kernel = ForwardLinearBatch::<'_,f32,A,NI,NO>::new();
    kernel.launch(&mut args).unwrap();

    let gpu_out = args.output.read_to_vec_with_size(BATCH * NO).unwrap();

    // Flatten CPU output
    let mut cpu_flat: Vec<f32> = Vec::with_capacity(BATCH * NO);

    for b in cpu_out.iter() {
        for &v in b.iter() {
            cpu_flat.push(v);
        }
    }

    approx_eq_slice(&gpu_out,&cpu_flat,2e-2);
}

#[test]
fn test_kernel_backward_linear_batch_matches_cpu() {
    let device = DeviceCpu::<f32>::new().unwrap();

    let mut rng = rand::thread_rng();

    // units and loss (input to backward is BatchOutput i.e.,batch of Arr<NO>)
    let mut units = Arr2::<f32,NI,NO>::new();
    for i in 0..NI { for j in 0..NO { units[(i,j)] = rng.gen::<f32>(); } }

    let mut loss_host: Vec<Arr<f32,NO>> = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let mut v = Arr::<f32,NO>::new();
        for x in v.iter_mut() { *x = rng.gen::<f32>(); }
        loss_host.push(v);
    }
    let loss = SerializedVec::<f32,Arr<f32,NO>>::from(loss_host);

    // CPU result
    let cpu_out = device.batch_backward_linear(&units,&loss).unwrap();

    // GPU kernel setup
    let alloc: &A = &SHARED_MEMORY_POOL;

    // Upload units
    let mut flat_units: Vec<f32> = Vec::with_capacity(NI * NO);
    for i in 0..NI { for j in 0..NO { flat_units.push(units[(i,j)]); } }
    let mut d_units = CudaTensor2dPtr::<f32,A,NI,NO>::new(alloc).unwrap();
    d_units.memcpy(flat_units.as_ptr(),flat_units.len()).unwrap();

    // Upload loss batch as device CudaVec<NO>
    let mut flat_loss: Vec<f32> = Vec::with_capacity(BATCH * NO);
    for b in loss.iter() {
        for &v in b.iter() { flat_loss.push(v); }
    }
    let mut d_loss = CudaVec::<f32,CudaTensor1dPtr<f32,A,NO>,A>::new(BATCH,alloc).unwrap();
    { let mut p = d_loss.as_cuda_mut_ptr(); p.memcpy(flat_loss.as_ptr(),flat_loss.len()).unwrap(); }

    // Output buffer (zero-init required)
    let d_output = CudaVec::<f32,CudaTensor1dPtr<f32,A,NI>,A>::new(BATCH,alloc).unwrap();

    // Build args
    let loss_view = CudaVecView::<f32,CudaTensor1dPtrView<f32,NO>>::try_from(&d_loss).unwrap();
    let mut args = BackwardLinearBatchArgs::<'_,f32,A,NI,NO>::new(&loss_view,&d_units,d_output,BATCH);

    let mut kernel = BackwardLinearBatch::<'_,f32,A,NI,NO>::new();
    kernel.launch(&mut args).unwrap();

    let gpu_out = args.output.read_to_vec_with_size(BATCH * NI).unwrap();

    // Flatten CPU output
    let mut cpu_flat: Vec<f32> = Vec::with_capacity(BATCH * NI);

    for b in cpu_out.iter() {
        for &v in b.iter() {
            cpu_flat.push(v);
        }
    }

    approx_eq_slice(&gpu_out,&cpu_flat,2e-2);
}

#[test]
fn test_kernel_linear_gradient_batch_matches_cpu() {
    let device = DeviceCpu::<f32>::new().unwrap();

    let mut rng = rand::thread_rng();

    // inputs and loss
    let mut inputs_host: Vec<Arr<f32,NI>> = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let mut v = Arr::<f32,NI>::new();
        for x in v.iter_mut() { *x = rng.gen::<f32>(); }
        inputs_host.push(v);
    }
    let inputs = SerializedVec::<f32,Arr<f32,NI>>::from(inputs_host);

    let mut loss_host: Vec<Arr<f32,NO>> = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let mut v = Arr::<f32,NO>::new();
        for x in v.iter_mut() { *x = rng.gen::<f32>(); }
        loss_host.push(v);
    }
    let loss = SerializedVec::<f32,Arr<f32,NO>>::from(loss_host);

    // CPU result (weight gradient)
    let cpu_grad = device.batch_backward_weight_gradient(&inputs,&loss).unwrap();

    // GPU kernel setup
    let alloc: &A = &SHARED_MEMORY_POOL;

    // Upload inputs batch
    let mut flat_inputs: Vec<f32> = Vec::with_capacity(BATCH * NI);
    for b in inputs.iter() {
        for &v in b.iter() { flat_inputs.push(v); }
    }
    let mut d_inputs = CudaVec::<f32,CudaTensor1dPtr<f32,A,NI>,A>::new(BATCH,alloc).unwrap();
    { let mut p = d_inputs.as_cuda_mut_ptr(); p.memcpy(flat_inputs.as_ptr(),flat_inputs.len()).unwrap(); }

    // Upload loss batch
    let mut flat_loss: Vec<f32> = Vec::with_capacity(BATCH * NO);
    for b in loss.iter() {
        for &v in b.iter() { flat_loss.push(v); }
    }
    let mut d_loss = CudaVec::<f32,CudaTensor1dPtr<f32,A,NO>,A>::new(BATCH,alloc).unwrap();
    { let mut p = d_loss.as_cuda_mut_ptr(); p.memcpy(flat_loss.as_ptr(),flat_loss.len()).unwrap(); }

    // Output buffer (zero-init required)
    let d_output = CudaTensor2dPtr::<f32,A,NI,NO>::with_initializer(alloc,|| 0.0f32).unwrap();

    // Build args
    let d_inputs = &d_inputs;
    let input_view_conv = d_inputs.as_cuda_view().into_converter();
    let input_view = CudaVecView::<f32,CudaTensor1dPtrView<f32,NI>>::try_from(input_view_conv).unwrap();
    let loss_view = CudaVecView::<f32,CudaTensor1dPtrView<f32,NO>>::try_from(&d_loss).unwrap();
    let mut args = LinearGradientBatchArgs::<'_,f32,A,NI,NO>::new(&loss_view,&input_view,d_output,BATCH);

    let mut kernel = LinearGradientBatch::<'_,f32,A,NI,NO>::new();
    kernel.launch(&mut args).unwrap();

    let gpu_out = args.output.read_to_vec_with_size(NI * NO).unwrap();

    let mut cpu_flat: Vec<f32> = Vec::with_capacity(BATCH * NO);
    for i in cpu_grad.iter() {
        for &v in i.iter() { cpu_flat.push(v); }
    }

    approx_eq_slice(&gpu_out,&cpu_flat,1e-2);
}
#[test]
fn test_kernel_reduce_linear_batch_matches_cpu() {
    let device = DeviceCpu::<f32>::new().unwrap();

    let mut rng = rand::thread_rng();

    let input = (0..1200).map(|_| {
        let mut v = Arr::<f32,1000>::new();

        for v in v.iter_mut() {
            *v = rng.gen::<f32>();
        }

        v
    }).collect::<Vec<_>>();

    let input = SerializedVec::<f32,Arr<f32,1000>>::from(input);

    let cpu_out:Arr<f32,1000> = device.reduce(&input).unwrap();

    let input = &input;

    let allocator:&A = &SHARED_MEMORY_POOL;

    let mut input_ptr = CudaVec::<f32,CudaTensor1dPtr<f32,A,1000>,A>::new(input.size(),allocator).unwrap();

    input_ptr.memcpy(input.as_ptr(),input.size()).unwrap();

    let input_ptr = &input_ptr;

    let input_ptr = input_ptr.as_cuda_view();

    let output_ptr = CudaTensor1dPtr::<f32,A,1000>::new(allocator).unwrap();

    let mut args = ReduceLinearBatchArgs::new(&input_ptr,output_ptr,1000,input.size());

    let mut kernel = ReduceLinearBatch::<f32,A,1000>::new();

    kernel.launch(&mut args).unwrap();

    let gpu_out = args.output.read_to_vec().unwrap();

    let mut cpu_flat: Vec<f32> = Vec::with_capacity(1200 * 1000);

    for &i in cpu_out.iter() {
        cpu_flat.push(i);
    }

    approx_eq_slice(&gpu_out,&cpu_flat,2e-4);
}