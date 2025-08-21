use core::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use libc::c_uint;
use crate::cuda::ffi;
use crate::cuda::mem::{Alloctype, MemoryPool};
use crate::error::CudaError;

pub trait CudaAllocator: Clone + Debug {
   /// Allocate memory from memory pool
    ///
    /// # Arguments
    /// * `size` - Size of memory to be allocated (number of elements, not bytes)
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    fn allocate<T>(&self, size: usize) -> Result<*mut T, CudaError>;

    /// Free up memory
    /// # Arguments
    /// * `ptr` - Pointer to the first address of the memory to be freed
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    fn deallocate<T>(&self, ptr:*mut T) -> Result<(),CudaError>;
}
#[derive(Clone)]
pub struct DeviceAllocator;
impl CudaAllocator for DeviceAllocator {
    fn allocate<T>(&self, size: usize) -> Result<*mut T, CudaError> {
        Ok(ffi::malloc(size)?)
    }

    fn deallocate<T>(&self, ptr: *mut T) -> Result<(), CudaError> {
        Ok(ffi::free(ptr)?)
    }
}
impl Debug for DeviceAllocator {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f,"DeviceAllocator")
    }
}
#[derive(Clone)]
pub struct HostAllocator {
    flags:c_uint
}
impl HostAllocator {
    pub fn new(flags:c_uint) -> HostAllocator {
        HostAllocator {
            flags
        }
    }
}
impl CudaAllocator for HostAllocator {
    fn allocate<T>(&self, size: usize) -> Result<*mut T, CudaError> {
        Ok(ffi::malloc_host(size,self.flags)?)
    }

    fn deallocate<T>(&self, ptr: *mut T) -> Result<(), CudaError> {
        Ok(ffi::free_host(ptr)?)
    }
}
impl Debug for HostAllocator {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f,"HostAllocator")
    }
}
pub struct MemoryPoolAllocator<A: CudaAllocator> {
    memory_pool:Arc<Mutex<MemoryPool>>,
    allocator:PhantomData<A>
}
impl MemoryPoolAllocator<DeviceAllocator> {
    pub fn new(_:DeviceAllocator) -> Result<MemoryPoolAllocator<DeviceAllocator>,CudaError> {
        Ok(MemoryPoolAllocator {
            memory_pool: Arc::new(Mutex::new(MemoryPool::new(Alloctype::Device)?)),
            allocator: PhantomData::<DeviceAllocator>
        })
    }
}
impl MemoryPoolAllocator<HostAllocator> {
    pub fn new(HostAllocator { flags }: HostAllocator) -> Result<MemoryPoolAllocator<HostAllocator>,CudaError> {
        Ok(MemoryPoolAllocator {
            memory_pool: Arc::new(Mutex::new(MemoryPool::new(Alloctype::Host(flags))?)),
            allocator: PhantomData::<HostAllocator>
        })
    }
}
impl<A: CudaAllocator> CudaAllocator for MemoryPoolAllocator<A> where Self: Debug {
    fn allocate<T>(&self, size: usize) -> Result<*mut T, CudaError> {
        let ptr:*mut T = match self.memory_pool.lock() {
            Ok(mut memory_pool) => {
                memory_pool.alloc_device(size)?
            },
            Err(_) => {
                return Err(CudaError::InvalidState(String::from(
                    "Failed to secure exclusive lock on memory pool."
                )));
            }
        };

        Ok(ptr)
    }

    fn deallocate<T>(&self, ptr: *mut T) -> Result<(), CudaError> {
        match self.memory_pool.lock() {
            Ok(mut memory_pool) => {
                memory_pool.deallocate(ptr as *const T).unwrap();
            },
            Err(_) => {
                panic!("Failed to secure exclusive lock on memory pool.");
            }
        }

        Ok(())
    }
}
impl Debug for MemoryPoolAllocator<DeviceAllocator> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f,"MemoryPoolAllocator<DeviceAllocator>")
    }
}
impl Debug for MemoryPoolAllocator<HostAllocator> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f,"MemoryPoolAllocator<HostAllocator>")
    }
}
impl<A: CudaAllocator> Clone for MemoryPoolAllocator<A> {
    fn clone(&self) -> Self {
        MemoryPoolAllocator {
            memory_pool: Arc::clone(&self.memory_pool)
        }
    }
}