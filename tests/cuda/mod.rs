use nncombinator::cuda::{CudaMemoryPoolPtr, ReadMemory};
use crate::common::SHARED_MEMORY_POOL;

#[test]
fn test_cudamemorypoolptr_with_initializer_zeros() {
    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let p:CudaMemoryPoolPtr<f32> = CudaMemoryPoolPtr::with_initializer(1200*1200,memory_pool,Default::default).unwrap();

    assert_eq!(vec![0f32;1200*1200],p.read_to_vec().unwrap());
}
#[test]
fn test_cudamemorypoolptr_with_initializer_ones() {
    let memory_pool = &SHARED_MEMORY_POOL.clone();

    let p:CudaMemoryPoolPtr<f32> = CudaMemoryPoolPtr::with_initializer(1200*1200,memory_pool,|| 1.).unwrap();

    assert_eq!(vec![1f32;1200*1200],p.read_to_vec().unwrap());
}