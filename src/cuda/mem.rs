use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use libc::c_void;
use crate::cuda::{CudaMemoryPoolPtr, CudaPtr, ffi, Memory};
use crate::error::{CudaError, InvalidStateError};
use crate::list::ListNode;
use crate::mem::AsRawSlice;

#[derive(Debug)]
pub struct Usage {
    prev_key: Option<*mut c_void>,
    size: usize,
    allocated: bool
}
#[derive(Debug)]
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
                Self::with_callback(alloc_type,size,|size| {
                    ffi::malloc(size)
                })
            },
            Alloctype::Host(flags) => {
                Self::with_callback(alloc_type,size,|size| {
                    ffi::malloc_host(size,flags)
                })
            }
        }
    }

    fn with_callback<F>(alloc_type:Alloctype,size:usize,f:F) -> Result<MemoryPool,CudaError>
        where F: FnOnce(usize) -> Result<*mut c_void,rcudnn::Error> {

        let ptr = f(size)?;

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
            {
                let mut current = c.deref().borrow_mut();

                if current.value.allocated == false && current.value.size >= size {
                    let remaining = current.value.size - size;

                    current.split(Usage {
                        prev_key: Some(p),
                        size: remaining,
                        allocated: false
                    });

                    current.value.allocated = true;
                    current.value.size = size;

                    let p = unsafe { p.add(offset) };

                    self.map.insert(p, Rc::clone(&c));

                    return Ok(p as *mut T);
                }

                offset += current.value.size;
            }

            n = c.deref().borrow().next();
        }

        Err(CudaError::AllocFailed(String::from("Memory allocation failed.")))
    }

    pub fn deallocate<T>(&mut self, ptr:*const T) -> Result<(),CudaError> {
        {
            let mut n = self.map.get(&(ptr as *const c_void)).ok_or(CudaError::InvalidState(String::from(
                "An attempt was made to release an unregistered memory address."
            )))?.deref().borrow_mut();

            n.value.allocated = false;

            let size = n.value.size;
            let size = n.next().map(|n| n.deref().borrow().value.size + size).unwrap_or(size);

            n.value.size = size;

            if n.next().map(|n| n.deref().borrow().value.allocated == false).unwrap_or(false) {
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
        }

        self.map.remove(&(ptr as *const c_void));

        Ok(())
    }
}
impl Drop for MemoryPool {
    fn drop(&mut self) {
        match self.alloc_type {
            Alloctype::Device => {
                ffi::free(self.pool).unwrap();
            },
            Alloctype::Host(_) => {
                ffi::free_host(self.pool).unwrap();
            }
        }
    }
}
pub struct ScopedMut<'a,U,T> where U: Debug + Default, T: AsRawSlice<U> {
    value: &'a mut T,
    ptr:Arc<Mutex<Option<CudaMemoryPoolPtr<U>>>>
}
impl<'a,U,T> ScopedMut<'a,U,T> where U: Debug + Default, T: AsRawSlice<U> {
    pub fn new(value:&'a mut T, ptr:Arc<Mutex<Option<CudaMemoryPoolPtr<U>>>>) -> ScopedMut<'a,U,T> {
        ScopedMut {
            value:value,
            ptr:ptr
        }
    }
}
impl<'a,U,T> Deref for ScopedMut<'a,U,T> where U: Debug + Default, T: AsRawSlice<U> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
impl<'a,U,T> DerefMut for ScopedMut<'a,U,T> where U: Debug + Default, T: AsRawSlice<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}
impl<'a,U,T> Drop for ScopedMut<'a,U,T> where U: Debug + Default, T: AsRawSlice<U> {
    fn drop(&mut self) {
        match self.ptr.lock() {
            Ok(mut ptr) => {
                if let Some(ptr) = ptr.as_mut() {
                    let len = self.value.as_raw_slice().len();

                    ptr.memcpy(self.value.as_raw_slice().as_ptr(),len).unwrap();
                }
            },
            Err(_) => {
                panic!("Failed to secure exclusive lock on memory pointer.");
            }
        }
    }
}
pub struct CachedTensor<U,T> where U: Debug + Default, T: AsRawSlice<U> {
    value:T,
    ptr:Arc<Mutex<Option<CudaMemoryPoolPtr<U>>>>,
    memory_pool:Arc<Mutex<MemoryPool>>
}
impl<U,T> CachedTensor<U,T> where U: Debug + Default, T: AsRawSlice<U> {
    pub fn new(value:T,memory_pool:&Arc<Mutex<MemoryPool>>) -> CachedTensor<U,T> {
        CachedTensor {
            value:value,
            ptr:Arc::new(Mutex::new(None)),
            memory_pool:Arc::clone(memory_pool)
        }
    }

    pub fn scoped_mut<'a>(&'a mut self) -> ScopedMut<'a,U,T> {
        ScopedMut {
            value:&mut self.value,
            ptr:Arc::clone(&self.ptr)
        }
    }

    pub fn and_then_with_lock<F,R,E>(&self,f:F) -> Result<R,E>
        where E: From<CudaError> + From<InvalidStateError> + From<rcudnn::Error>, F: Fn(&CudaMemoryPoolPtr<U>) -> Result<R,E> {

        match self.ptr.lock() {
            Ok(mut ptr) => {
                let p = match *ptr {
                    Some(ref p) => p,
                    ref mut p => {
                        let len = self.value.as_raw_slice().len();

                        let mut ptr = CudaMemoryPoolPtr::new(len,&self.memory_pool)?;
                        ptr.memcpy(self.value.as_raw_slice().as_ptr(),len)?;

                        p.get_or_insert(ptr)
                    }
                };

                f(p)
            },
            Err(_) => {
                Err(E::from(InvalidStateError(String::from(
                    "Failed to allocate exclusive lock on memory for gpu."
                ))))
            }
        }
    }
}
