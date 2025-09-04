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
impl DeviceAllocator  {
    pub fn new() -> DeviceAllocator {
        DeviceAllocator
    }
}
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
pub struct MemoryPoolAllocator<A> {
    memory_pool:Arc<Mutex<MemoryPool>>,
    allocator:PhantomData<A>
}
#[derive(Debug,Clone)]
pub struct DeviceAlloc;

impl DeviceAlloc {
    pub fn new() -> DeviceAlloc {
        DeviceAlloc
    }
}
#[derive(Debug,Clone)]
pub struct HostAlloc {
    flags:c_uint
}
impl HostAlloc {
    pub fn new(flags: c_uint) -> HostAlloc {
        HostAlloc {
            flags
        }
    }
}
impl MemoryPoolAllocator<DeviceAlloc> {
    pub fn new(_:DeviceAlloc) -> Result<MemoryPoolAllocator<DeviceAlloc>,CudaError> {
        Ok(MemoryPoolAllocator {
            memory_pool: Arc::new(Mutex::new(MemoryPool::new(Alloctype::Device)?)),
            allocator: PhantomData::<DeviceAlloc>
        })
    }

    pub fn with_size(size:usize, _:DeviceAlloc) -> Result<MemoryPoolAllocator<DeviceAlloc>,CudaError> {
        Ok(MemoryPoolAllocator {
            memory_pool: Arc::new(Mutex::new(MemoryPool::with_size(size,Alloctype::Device)?)),
            allocator: PhantomData::<DeviceAlloc>
        })
    }
}
impl MemoryPoolAllocator<HostAlloc> {
    pub fn new(HostAlloc { flags }: HostAlloc) -> Result<MemoryPoolAllocator<HostAlloc>,CudaError> {
        Ok(MemoryPoolAllocator {
            memory_pool: Arc::new(Mutex::new(MemoryPool::new(Alloctype::Host(flags))?)),
            allocator: PhantomData::<HostAlloc>

        })
    }

    pub fn with_size(size:usize, HostAlloc { flags }: HostAlloc) -> Result<MemoryPoolAllocator<HostAlloc>,CudaError> {
        Ok(MemoryPoolAllocator {
            memory_pool: Arc::new(Mutex::new(MemoryPool::with_size(size,Alloctype::Host(flags))?)),
            allocator: PhantomData::<HostAlloc>
        })
    }
}
impl<A> CudaAllocator for MemoryPoolAllocator<A> where Self: Debug {
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
impl Debug for MemoryPoolAllocator<DeviceAlloc> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f,"MemoryPoolAllocator<DeviceAlloc>")
    }
}
impl Debug for MemoryPoolAllocator<HostAlloc> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f,"MemoryPoolAllocator<HostAlloc>")
    }
}
impl<A> Clone for MemoryPoolAllocator<A> {
    fn clone(&self) -> Self {
        MemoryPoolAllocator {
            memory_pool: Arc::clone(&self.memory_pool),
            allocator: PhantomData::<A>
        }
    }
}