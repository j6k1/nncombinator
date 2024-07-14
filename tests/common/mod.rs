use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;
use nncombinator::cuda::mem::{Alloctype, MemoryPool};

lazy_static! {
    pub static ref SHARED_MEMORY_POOL:Arc<Mutex<MemoryPool>> = Arc::new(Mutex::new(MemoryPool::with_size(8 * 1024 * 1024 * 1024,Alloctype::Device).unwrap()));
}
