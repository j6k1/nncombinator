use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::size_of;
use std::ops::Deref;
use std::ptr::null_mut;
use std::rc::Rc;
use libc::c_void;
use rcudnn_sys::cudaError;
use crate::error::CudaError;
use crate::list::ListNode;

pub struct Usage {
    prev_key: Option<*mut c_void>,
    size: usize,
    allocated: bool
}
pub enum Alloctype {
    Device,
    Host(libc::c_uint)
}
pub struct MemoryPool {
    alloc_type:Alloctype,
    list:ListNode<Usage>,
    map: HashMap<*const c_void,Rc<RefCell<ListNode<Usage>>>>,
    pool: *mut c_void
}
unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}
impl MemoryPool {
    pub fn new(alloc_type:Alloctype) -> Result<MemoryPool,CudaError> {
        let size = 1024 * 1024 * 1024;

        Self::with_size(size,alloc_type)
    }

    pub fn with_size(size:usize,alloc_type:Alloctype) -> Result<MemoryPool,CudaError> {
        match alloc_type {
            Alloctype::Device => {
                Self::with_callback(alloc_type,size,|size,ptr| {
                    unsafe { rcudnn_sys::cudaMalloc(ptr,size) }
                })
            },
            Alloctype::Host(flags) => {
                Self::with_callback(alloc_type,size,|size,ptr| {
                    unsafe { rcudnn_sys::cudaHostAlloc(ptr,size,flags) }
                })
            }
        }
    }

    fn with_callback<F>(alloc_type:Alloctype,size:usize,f:F) -> Result<MemoryPool,CudaError> where F: FnOnce(usize,*mut *mut c_void) -> cudaError {
        let mut ptr: *mut c_void = null_mut();

        match f(size,&mut ptr as *mut *mut c_void) {
            cudaError::cudaSuccess => {
                assert_ne!(ptr,
                           null_mut(),
                           "cudaMalloc is succeeded, but returned null pointer!");

                let mut n = ListNode::new(Usage {
                    prev_key:None,
                    size: 0,
                    allocated: true
                });

                n.append(ListNode::new(Usage {
                    prev_key: None,
                    size: size,
                    allocated: false
                }));

                Ok(MemoryPool {
                    alloc_type:alloc_type,
                    list: n,
                    map: HashMap::new(),
                    pool: ptr
                })
            },
            cudaError::cudaErrorInvalidValue => {
                Err(CudaError::CudnnError(rcudnn::Error::InvalidValue("The range of one or more of the entered parameters is out of tolerance.")))
            },
            cudaError::cudaErrorMemoryAllocation => {
                Err(CudaError::CudnnError(rcudnn::Error::AllocFailed("Device memory allocation failed.")))
            },
            status => {
                Err(CudaError::CudnnError(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)))
            }
        }
    }

    pub fn alloc_device<T>(&mut self,size:usize) -> Result<*mut T,CudaError> {
        match self.alloc_type {
            Alloctype::Device => (),
            Alloctype::Host(_) => {
                return Err(CudaError::InvalidState(String::from(
                    "Attempted to allocate device memory from a memory pool not allocated as device memory."
                )));
            }
        }

        self.allocate(size)
    }

    pub fn alloc_host<T>(&mut self,size:usize) -> Result<*mut T,CudaError> {
        match self.alloc_type {
            Alloctype::Host(_) => (),
            Alloctype::Device => {
                return Err(CudaError::InvalidState(String::from(
                    "An attempt was made to allocate host memory from a memory pool that was not allocated as host memory."
                )));
            }
        }

        self.allocate(size)
    }

    pub fn allocate<T>(&mut self,size:usize) -> Result<*mut T,CudaError> {
        let size = size * size_of::<T>();

        let p = self.pool;

        let mut offset = 0;

        let n = &mut self.list;
        let mut n = n.next();

        while let Some(c) = n {
            let mut current = c.deref().borrow_mut();

            if current.value.allocated == false && current.value.size <= size {
                let remaining = current.value.size - size;

                current.split(Usage {
                    prev_key: Some(p),
                    size:remaining,
                    allocated:false
                });

                current.value.allocated = true;
                current.value.size = size;

                let p = unsafe { p.add(offset) };

                self.map.insert(p,Rc::clone(&c));

                return Ok(p as *mut T);
            }

            offset += current.value.size;

            n = c.deref().borrow().next();
        }

        Err(CudaError::AllocFailed(String::from("Memory allocation failed.")))
    }

    pub fn deallocate<T>(&mut self, ptr:*const T) -> Result<(),CudaError> {
        let mut n = self.map.get(&(ptr as *const c_void)).ok_or(CudaError::InvalidState(String::from(
            "An attempt was made to release an unregistered memory address."
        )))?.deref().borrow_mut();

        n.value.allocated = false;

        let size = n.value.size;
        let size = n.next().map(|n| n.deref().borrow().value.size + size).unwrap_or(size);

        n.value.size = size;

        if n.next().is_some() {
            n.merge_next();
        }

        let p = n.value.prev_key;

        if let Some(p) = p {
            let mut n = self.map.get(&(p as *const c_void)).ok_or(CudaError::LogicError(String::from(
                "Memory address is unregistered."
            )))?.deref().borrow_mut();

            if n.value.allocated == false {
                n.value.size += size;
                n.merge_next();
            }
        }

        Ok(())
    }
}