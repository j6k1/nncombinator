use nncombinator::cuda::{CudaPtr, ReadMemory};
use crate::common::SHARED_MEMORY_POOL;

mod kernel;
mod cublas;

#[test]
fn test_cudamemorypoolptr_with_initializer_zeros() {
    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let p:CudaPtr<f32,_> = CudaPtr::with_initializer(1200*1200,memory_pool,Default::default).unwrap();

    assert_eq!(vec![0f32;1200*1200],p.read_to_vec().unwrap());
}
#[test]
fn test_cudamemorypoolptr_with_initializer_ones() {
    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let p:CudaPtr<f32,_> = CudaPtr::with_initializer(1200*1200,memory_pool,|| 1.).unwrap();

    assert_eq!(vec![1f32;1200*1200],p.read_to_vec().unwrap());
}