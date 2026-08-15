extern crate nncombinator;
extern crate rand;

use nncombinator::arr::{Arr, Arr2, SerializedVec};
use nncombinator::cuda::{CudaTensor1dPtr,CudaTensor2dPtr,CudaVec,ReadMemory,WriteMemory,AsCudaMutPtr};
use nncombinator::cuda::allocator::{ DeviceAlloc, MemoryPoolAllocator};
use nncombinator::device::{DeviceCpu, DeviceGpu};
use rand::Rng;
use nncombinator::device::linear::DeviceLinear;
use nncombinator::layer::{OutputTensorSize, TensorSize};
use crate::common::SHARED_MEMORY_POOL;
use crate::common::gen_inputs;
use crate::common::approx_eq_slice;
use crate::common::upload_inputs_to_device;

const NI: usize = 500;
const NO: usize = 600;
const BATCH: usize = 400;
type A = MemoryPoolAllocator<DeviceAlloc>;

#[test]
fn test_device_gpu_forward_linear_batch_matches_cpu()
    where DeviceGpu<A>: DeviceLinear<f64,CudaTensor2dPtr<f64,A,NI,NO>,CudaTensor1dPtr<f64,A,NO>,CudaTensor1dPtr<f64,A,NI>,NI,NO>,
          CudaTensor1dPtr<f64,A,NO>: OutputTensorSize<NO> {
    let device = DeviceCpu::new().unwrap();

    let (bias,units,inputs) = gen_inputs();

    // CPU result
    let cpu_out = device.batch_forward_linear(&bias,&units,&inputs).unwrap();

    // GPU kernel
    let alloc: &A = &SHARED_MEMORY_POOL;

    let device_gpu = DeviceGpu::<A>::new(alloc).unwrap();

    let (d_bias,d_units,d_inputs) = upload_inputs_to_device::<A>(alloc,&bias,&units,&inputs);

    // Build args
    let d_inputs:&CudaVec<f64,CudaTensor1dPtr<f64,A,NI>,A> = &d_inputs;
    let d_units:&CudaTensor2dPtr<f64,A,NI,NO> = &d_units;
    let d_bias:&CudaTensor1dPtr<f64,A,NO> = &d_bias;

    let d_output = device_gpu.batch_forward_linear(d_bias,d_units,d_inputs).unwrap();

    let gpu_out = d_output.read_to_vec().unwrap();
    // Flatten CPU output
    let mut cpu_flat: Vec<f64> = Vec::with_capacity(BATCH * NO);

    for b in cpu_out.iter() {
        for &v in b.iter() {
            cpu_flat.push(v);
        }
    }

    approx_eq_slice(&gpu_out,&cpu_flat,2e-2);
}

#[test]
fn test_device_gpu_backward_linear_batch_matches_cpu()
    where DeviceGpu<A>: DeviceLinear<f64,CudaTensor2dPtr<f64,A,NI,NO>,CudaTensor1dPtr<f64,A,NO>,CudaTensor1dPtr<f64,A,NI>,NI,NO> {
    let device = DeviceCpu::new().unwrap();

    let mut rng = rand::thread_rng();

    // units and loss (input to backward is BatchOutput i.e.,batch of Arr<NO>)
    let mut units = Arr2::<f64,NI,NO>::new();
    for i in 0..NI { for j in 0..NO { units[(i,j)] = rng.gen::<f64>(); } }

    let mut loss_host: Vec<Arr<f64,NO>> = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let mut v = Arr::<f64,NO>::new();
        for x in v.iter_mut() { *x = rng.gen::<f64>(); }
        loss_host.push(v);
    }
    let loss = SerializedVec::<f64,Arr<f64,NO>>::from(loss_host);

    // CPU result
    let cpu_out = device.batch_backward_linear(&units,&loss).unwrap();

    // GPU kernel setup
    let alloc: &A = &SHARED_MEMORY_POOL;

    let device_gpu = DeviceGpu::<A>::new(alloc).unwrap();

    // Upload units
    let mut flat_units: Vec<f64> = Vec::with_capacity(NI * NO);
    for i in 0..NI { for j in 0..NO { flat_units.push(units[(i,j)]); } }
    let mut d_units = CudaTensor2dPtr::<f64,A,NI,NO>::new(alloc).unwrap();
    d_units.memcpy(flat_units.as_ptr(),flat_units.len()).unwrap();

    // Upload loss batch as device CudaVec<NO>
    let mut flat_loss: Vec<f64> = Vec::with_capacity(BATCH * NO);
    for b in loss.iter() {
        for &v in b.iter() { flat_loss.push(v); }
    }
    let mut d_loss = CudaVec::<f64,CudaTensor1dPtr<f64,A,NO>,A>::new(BATCH,alloc).unwrap();
    { let mut p = d_loss.as_cuda_mut_ptr(); p.memcpy(flat_loss.as_ptr(),flat_loss.len()).unwrap(); }

    // Output buffer (zero-init required)
    let d_output = device_gpu.batch_backward_linear(&d_units,(&d_loss).into()).unwrap();

    let gpu_out = d_output.read_to_vec().unwrap();

    // Flatten CPU output
    let mut cpu_flat: Vec<f64> = Vec::with_capacity(BATCH * NI);

    for b in cpu_out.iter() {
        for &v in b.iter() {
            cpu_flat.push(v);
        }
    }

    approx_eq_slice(&gpu_out,&cpu_flat,2e-2);
}

#[test]
fn test_device_gpu_linear_gradient_batch_matches_cpu()
    where DeviceGpu<A>: DeviceLinear<f64,CudaTensor2dPtr<f64,A,NI,NO>,CudaTensor1dPtr<f64,A,NO>,CudaTensor1dPtr<f64,A,NI>,NI,NO> {
    let device = DeviceCpu::new().unwrap();

    let mut rng = rand::thread_rng();

    // inputs and loss
    let mut inputs_host: Vec<Arr<f64,NI>> = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let mut v = Arr::<f64,NI>::new();
        for x in v.iter_mut() { *x = rng.gen::<f64>(); }
        inputs_host.push(v);
    }
    let inputs = SerializedVec::<f64,Arr<f64,NI>>::from(inputs_host);

    let mut loss_host: Vec<Arr<f64,NO>> = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let mut v = Arr::<f64,NO>::new();
        for x in v.iter_mut() { *x = rng.gen::<f64>(); }
        loss_host.push(v);
    }
    let loss = SerializedVec::<f64,Arr<f64,NO>>::from(loss_host);

    // CPU result (weight gradient)
    let cpu_grad = device.batch_backward_weight_gradient(&inputs,&loss).unwrap();

    // GPU kernel setup
    let alloc: &A = &SHARED_MEMORY_POOL;

    let device_gpu = DeviceGpu::<A>::new(alloc).unwrap();

    // Upload inputs batch
    let mut flat_inputs: Vec<f64> = Vec::with_capacity(BATCH * NI);
    for b in inputs.iter() {
        for &v in b.iter() { flat_inputs.push(v); }
    }
    let mut d_inputs = CudaVec::<f64,CudaTensor1dPtr<f64,A,NI>,A>::new(BATCH,alloc).unwrap();
    { let mut p = d_inputs.as_cuda_mut_ptr(); p.memcpy(flat_inputs.as_ptr(),flat_inputs.len()).unwrap(); }

    // Upload loss batch
    let mut flat_loss: Vec<f64> = Vec::with_capacity(BATCH * NO);
    for b in loss.iter() {
        for &v in b.iter() { flat_loss.push(v); }
    }
    let mut d_loss = CudaVec::<f64,CudaTensor1dPtr<f64,A,NO>,A>::new(BATCH,alloc).unwrap();
    { let mut p = d_loss.as_cuda_mut_ptr(); p.memcpy(flat_loss.as_ptr(),flat_loss.len()).unwrap(); }

    // Build args
    let d_inputs = &d_inputs;

    // Output buffer
    let d_output = device_gpu.batch_backward_weight_gradient(d_inputs,(&d_loss).into()).unwrap();

    let gpu_out = d_output.read_to_vec().unwrap();

    let mut cpu_flat: Vec<f64> = Vec::with_capacity(BATCH * NO);
    for i in cpu_grad.iter() {
        for &v in i.iter() { cpu_flat.push(v); }
    }

    approx_eq_slice(&gpu_out,&cpu_flat,1e-2);
}
