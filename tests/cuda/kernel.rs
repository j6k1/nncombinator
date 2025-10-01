extern crate nncombinator;
extern crate rand;

use nncombinator::arr::{Arr, Arr2, IntoConverter, SerializedVec};
use nncombinator::cuda::{CudaTensor1dPtr,CudaTensor1dPtrView,CudaTensor2dPtr,CudaVec,CudaVecView,ReadMemory,WriteMemory,Kernel,AsCudaMutPtr,AsCudaView,CudaMutPtr};
use nncombinator::cuda::allocator::{CudaAllocator, DeviceAlloc, MemoryPoolAllocatorInstantiation};
use nncombinator::cuda::kernel::device::{ForwardLinearBatch,ForwardLinearBatchArgs,BackwardLinearBatch,BackwardLinearBatchArgs,LinearGradientBatch,LinearGradientBatchArgs};
use nncombinator::device::DeviceCpu;
use rand::Rng;
use nncombinator::device::linear::DeviceLinear;
use crate::common::SHARED_MEMORY_POOL;

const NI: usize = 500;
const NO: usize = 600;
const BATCH: usize = 400;

fn gen_inputs() -> (Arr<f32,NO>,Arr2<f32,NI,NO>,SerializedVec<f32,Arr<f32,NI>>) {
    let mut rng = rand::thread_rng();

    let mut bias = Arr::<f32,NO>::new();
    for b in bias.iter_mut() { *b = rng.gen::<f32>() * 1e-3; }

    let mut units = Arr2::<f32,NI,NO>::new();
    for i in 0..NI {
        for j in 0..NO { units[(i,j)] = rng.gen::<f32>(); }
    }

    let mut inputs_host: Vec<Arr<f32,NI>> = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let mut v = Arr::<f32,NI>::new();
        for x in v.iter_mut() { *x = rng.gen::<f32>(); }
        inputs_host.push(v);
    }

    (bias,units,inputs_host.into())
}

fn approx_eq_slice(a: &[f32],b: &[f32],eps: f32) {
    assert_eq!(a.len(),b.len());
    for (i,(x,y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (*x - *y).abs();
        assert!(d < eps,"diff[{}] = {} exceeds eps {} (x={},y={})",i,d,eps,x,y);
    }
}

fn upload_inputs_to_device<A: MemoryPoolAllocatorInstantiation<DeviceAlloc> + CudaAllocator>(alloc: &A,
    bias: &Arr<f32,NO>,units: &Arr2<f32,NI,NO>,batch_inputs: &SerializedVec<f32,Arr<f32,NI>>)
    -> (CudaTensor1dPtr<f32,A,NO>,CudaTensor2dPtr<f32,A,NI,NO>,CudaVec<f32,CudaTensor1dPtr<f32,A,NI>,A>)
    where CudaTensor1dPtr<f32,A,NO>: WriteMemory<f32>,
          CudaTensor2dPtr<f32,A,NI,NO>: WriteMemory<f32>,
          CudaVec<f32,CudaTensor1dPtr<f32,A,NI>,A>: AsCudaMutPtr<Pointee=f32,Allocator=A>,
          for<'a> CudaMutPtr<'a,f32,A>: WriteMemory<f32>,
          for<'a> &'a CudaVec<f32,CudaTensor1dPtr<f32,A,NI>,A>: AsCudaView<'a> {
    // Bias
    let mut d_bias = CudaTensor1dPtr::<f32,A,NO>::new(alloc).unwrap();
    d_bias.memcpy(bias.as_ptr(),NO).unwrap();

    // Units: flatten in (i,j) with leading dimension NO (calc_index(out=j,in=i,ld=NO) == i*NO+j)
    let mut flat_units: Vec<f32> = Vec::with_capacity(NI * NO);
    for i in 0..NI { for j in 0..NO { flat_units.push(units[(i,j)]); } }
    let mut d_units = CudaTensor2dPtr::<f32,A,NI,NO>::new(alloc).unwrap();
    d_units.memcpy(flat_units.as_ptr(),flat_units.len()).unwrap();

    // Inputs: layout is batch-major with leading dimension NI:
    // calc_index(x=i,y=batch_index,ld=NI) == batch_index*NI + i
    let mut flat_inputs: Vec<f32> = Vec::with_capacity(BATCH * NI);
    for b in batch_inputs.iter() {
        for &v in b.iter() { flat_inputs.push(v); }
    }
    let mut d_inputs = CudaVec::<f32,CudaTensor1dPtr<f32,A,NI>,A>::new(BATCH,alloc).unwrap();
    {
        let mut ptr = d_inputs.as_cuda_mut_ptr();
        ptr.memcpy(flat_inputs.as_ptr(),flat_inputs.len()).unwrap();
    }

    (d_bias,d_units,d_inputs)
}

#[test]
fn test_kernel_forward_linear_batch_matches_cpu() {
    let device = DeviceCpu::<f32>::new().unwrap();

    let (bias,units,inputs) = gen_inputs();

    // CPU result
    let cpu_out = device.batch_forward_linear(&bias,&units,&inputs).unwrap();

    // GPU kernel
    type A = nncombinator::cuda::allocator::MemoryPoolAllocator<DeviceAlloc>;
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
    type A = nncombinator::cuda::allocator::MemoryPoolAllocator<DeviceAlloc>;
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
    type A = nncombinator::cuda::allocator::MemoryPoolAllocator<DeviceAlloc>;
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
