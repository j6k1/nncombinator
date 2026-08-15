use std::fmt::Debug;
use lazy_static::lazy_static;
use rand::Rng;
use nncombinator::arr::{Arr, Arr2, SerializedVec};
use nncombinator::device::input::DeviceInput;
use nncombinator::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchLoss, BatchPreTrain, ContinueForward, ForwardAll, ForwardDiff, Loss, OnStep, PartialForward, PersistProgress, PreTrain, Step, UpdateWeight};
use nncombinator::cuda::allocator::{CudaAllocator, DeviceAlloc, MemoryPoolAllocator, MemoryPoolAllocatorInstantiation};
use nncombinator::cuda::{AsCudaMutPtr, AsCudaView, CudaMutPtr, CudaTensor1dPtr, CudaTensor2dPtr, CudaVec, WriteMemory};
use nncombinator::persistence::{Specialized, TextFilePersistence};

lazy_static! {
    pub static ref SHARED_MEMORY_POOL:MemoryPoolAllocator<DeviceAlloc> = MemoryPoolAllocator::with_size(8 * 1024 * 1024 * 1024,DeviceAlloc).unwrap();
}
pub fn assert_device<U,I,D: DeviceInput<U,I>>(_:&D)
    where U: Default + Clone + Copy + Debug + Send + Sync + 'static,
          I: BatchDataType + Debug + 'static,
          <I as BatchDataType>::Type: Debug + 'static {
}
pub fn assert_forward_all<T: ForwardAll>(_:&T) {

}
pub fn assert_pre_train<T: PreTrain>(_:&T) {

}
pub fn assert_backward_all<U: Clone + Copy + Debug,T: BackwardAll<U>>(_:&T) {

}
pub fn assert_loss<U: Clone + Copy + Debug,T: Loss<U>>(_:&T) {

}
pub fn assert_update_weight<T: UpdateWeight>(_:&T) {

}
pub fn assert_partial_forward<T: PartialForward>(_:&T) {}
pub fn assert_forward_diff<T: ForwardDiff>(_:&T) {}
pub fn assert_continue_forward<T: ContinueForward>(_:&T) {}
pub fn assert_batch_forward<T: BatchForward>(_:&T) {

}
pub fn assert_batch_pre_train<T: BatchPreTrain>(_:&T) {

}
pub fn assert_batch_backward<U: Clone + Copy + Debug,T: BatchBackward<U>>(_:&T) {

}
pub fn assert_batch_loss<U: Clone + Copy + Debug,T: BatchLoss<U>>(_:&T) {

}

pub fn assert_step<T: Step>(_:&T) {

}

pub fn assert_on_step<T: OnStep>(_:&T) {

}

pub fn assert_text_persist_progress<T: PersistProgress<TextFilePersistence,Specialized>>(_:&T) {}
const NI: usize = 500;
const NO: usize = 600;
const BATCH: usize = 400;

pub fn gen_inputs() -> (Arr<f64,NO>,Arr2<f64,NI,NO>,SerializedVec<f64,Arr<f64,NI>>) {
    let mut rng = rand::thread_rng();

    let mut bias = Arr::<f64,NO>::new();
    for b in bias.iter_mut() { *b = rng.gen::<f64>() * 1e-3; }

    let mut units = Arr2::<f64,NI,NO>::new();
    for i in 0..NI {
        for j in 0..NO { units[(i,j)] = rng.gen::<f64>(); }
    }

    let mut inputs_host: Vec<Arr<f64,NI>> = Vec::with_capacity(BATCH);
    for _ in 0..BATCH {
        let mut v = Arr::<f64,NI>::new();
        for x in v.iter_mut() { *x = rng.gen::<f64>(); }
        inputs_host.push(v);
    }

    (bias,units,inputs_host.into())
}

pub fn approx_eq_slice(a: &[f64],b: &[f64],eps: f64) {
    assert_eq!(a.len(),b.len());
    for (i,(x,y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (*x - *y).abs();
        assert!(d < eps,"diff[{}] = {} exceeds eps {} (x={},y={})",i,d,eps,x,y);
    }
}

pub fn upload_inputs_to_device<A: MemoryPoolAllocatorInstantiation<DeviceAlloc> + CudaAllocator + 'static>(
    alloc: &A,
    bias: &Arr<f64,NO>,units: &Arr2<f64,NI,NO>,
    batch_inputs: &SerializedVec<f64,Arr<f64,NI>>)
    -> (CudaTensor1dPtr<f64,A,NO>,CudaTensor2dPtr<f64,A,NI,NO>,CudaVec<f64,CudaTensor1dPtr<f64,A,NI>,A>)
    where CudaTensor1dPtr<f64,A,NO>: WriteMemory<f64>,
          CudaTensor2dPtr<f64,A,NI,NO>: WriteMemory<f64>,
          CudaVec<f64,CudaTensor1dPtr<f64,A,NI>,A>: AsCudaMutPtr<Pointee=f64,Allocator=A>,
          for<'a> CudaMutPtr<'a,f64,A>: WriteMemory<f64>,
          for<'a> &'a CudaVec<f64,CudaTensor1dPtr<f64,A,NI>,A>: AsCudaView<'a> {
    // Bias
    let mut d_bias = CudaTensor1dPtr::<f64,A,NO>::new(alloc).unwrap();
    d_bias.memcpy(bias.as_ptr(),NO).unwrap();

    // Units: flatten in (i,j) with leading dimension NO (calc_index(out=j,in=i,ld=NO) == i*NO+j)
    let mut flat_units: Vec<f64> = Vec::with_capacity(NI * NO);
    for i in 0..NI { for j in 0..NO { flat_units.push(units[(i,j)]); } }
    let mut d_units = CudaTensor2dPtr::<f64,A,NI,NO>::new(alloc).unwrap();
    d_units.memcpy(flat_units.as_ptr(),flat_units.len()).unwrap();

    // Inputs: layout is batch-major with leading dimension NI:
    // calc_index(x=i,y=batch_index,ld=NI) == batch_index*NI + i
    let mut flat_inputs: Vec<f64> = Vec::with_capacity(BATCH * NI);
    for b in batch_inputs.iter() {
        for &v in b.iter() { flat_inputs.push(v); }
    }
    let mut d_inputs = CudaVec::<f64,CudaTensor1dPtr<f64,A,NI>,A>::new(BATCH,alloc).unwrap();
    {
        let mut ptr = d_inputs.as_cuda_mut_ptr();
        ptr.memcpy(flat_inputs.as_ptr(),flat_inputs.len()).unwrap();
    }

    (d_bias,d_units,d_inputs)
}
