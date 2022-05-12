use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};
use libc::c_void;
use crate::cuda::{AsPtr, CudaMemoryPoolPtr, ffi, Memory};
use crate::error::{CudaError};
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
    prev_map: HashMap<*const c_void,Rc<RefCell<ListNode<Usage>>>>,
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
            prev_map: HashMap::new(),
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

        let mut p = self.pool;
        let mut prev_key = None;

        let n = &mut self.list;
        let mut n = n.next();
        let mut found = false;

        while let Some(c) = n {
            {
                let mut current = c.deref().borrow_mut();

                if current.value.allocated == false && current.value.size >= size {
                    let remaining = current.value.size - size;

                    current.value.allocated = true;
                    current.value.size = size;

                    self.map.insert(p, Rc::clone(&c));

                    if remaining > 0 {
                        n = current.split(Usage {
                            prev_key: Some(p),
                            size: remaining,
                            allocated: false
                        });

                        prev_key = unsafe { Some(p.add(size)) };
                        self.prev_map.insert(p, Rc::clone(&c));
                    } else {
                        n = None;
                    }

                    found = true;

                    break;
                }

                unsafe { p = p.add(current.value.size); }
            }

            n = c.deref().borrow().next();
        }

        if found {
            n.map(|n| {
                let current = Rc::clone(&n);

                n.deref().borrow_mut().next().map(|n| {
                    n.borrow_mut().value.prev_key = prev_key;

                    if let Some(p) = prev_key {
                        self.prev_map.insert(p, current);
                    }
                });
            });

            Ok(p as *mut T)
        } else {
            Err(CudaError::AllocFailed(String::from("Memory allocation failed.")))
        }
    }

    pub fn deallocate<T>(&mut self, ptr:*const T) -> Result<(),CudaError> {
        let mut removes = vec![];

        let prev_key = ptr as *mut c_void;

        let (size,prev_key) = {
            let mut n = self.map.get(&(ptr as *const c_void)).ok_or(CudaError::InvalidState(String::from(
                "An attempt was made to release an unregistered memory address."
            )))?.deref().borrow_mut();

            if !n.value.allocated {
                return Err(CudaError::InvalidState(String::from(
                    "Attempted to release an area that has already been released."
                )));
            }

            n.value.allocated = false;

            let mut size = n.value.size;

            if n.next().map(|n| n.deref().borrow().value.allocated == false).unwrap_or(false) {
                size = n.next().map(|n| n.deref().borrow().value.size + size).unwrap_or(size);

                n.value.size = size;

                n.merge_next();
                n.next().map(|n| {
                    n.deref().borrow().value.prev_key.map(|ptr| {
                        removes.push(ptr as *const c_void);
                    });

                    n.deref().borrow_mut().value.prev_key = Some(prev_key)
                });
            }

            (size,n.value.prev_key)
        };

        if let Some(p) = prev_key {
            let prev_key = p as *mut c_void;

            let mut n = self.prev_map.get(&(p as *const c_void)).ok_or(CudaError::LogicError(String::from(
                "Memory address is unregistered."
            )))?.deref().borrow_mut();

            if n.value.allocated == false {
                n.value.size += size;
                n.merge_next();

                removes.push(ptr as *const c_void);

                n.next().map(|n| n.deref().borrow_mut().value.prev_key = Some(prev_key));
            }
        }

        self.map.remove(&(ptr as *const c_void));

        for r in removes {
            self.prev_map.remove(&r);
        }

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
    ptr:Arc<RwLock<Option<CudaMemoryPoolPtr<U>>>>
}
impl<'a,U,T> ScopedMut<'a,U,T> where U: Debug + Default, T: AsRawSlice<U> {
    pub fn new(value:&'a mut T, ptr:Arc<RwLock<Option<CudaMemoryPoolPtr<U>>>>) -> ScopedMut<'a,U,T> {
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
        match self.ptr.write() {
            Ok(mut ptr) => {
                if let Some(ptr) = ptr.as_mut() {
                    let len = self.value.as_raw_slice().len();

                    ptr.memcpy(self.value.as_raw_slice().as_ptr(),len).unwrap();
                }
            },
            Err(_) => {
                panic!("Failed to secure exclusive lock on memory pointer.")
            }
        }
    }
}
pub struct CachedTensor<U,T> where U: Debug + Default, T: AsRawSlice<U> {
    value:T,
    ptr:Arc<RwLock<Option<CudaMemoryPoolPtr<U>>>>,
    memory_pool:Option<Arc<Mutex<MemoryPool>>>
}
impl<U,T> CachedTensor<U,T> where U: Debug + Default, T: AsRawSlice<U> {
    pub fn new(value:T,memory_pool:Option<Arc<Mutex<MemoryPool>>>) -> CachedTensor<U,T> {
        CachedTensor {
            value:value,
            ptr:Arc::new(RwLock::new(None)),
            memory_pool:memory_pool.as_ref().map(|memory_pool| Arc::clone(memory_pool))
        }
    }

    pub fn scoped_mut<'a>(&'a mut self) -> ScopedMut<'a,U,T> {
        ScopedMut {
            value:&mut self.value,
            ptr:self.ptr.clone()
        }
    }

    pub fn get_or_insert_ptr(&self) -> Result<*const U,CudaError> {
        match self.ptr.read() {
            Ok(ptr) => {
                if let Some(ref p) = *ptr {
                    return Ok(p.as_ptr());
                }
            },
            Err(_) => {
                return Err(CudaError::InvalidState(String::from("Failed to allocate shared lock on memory for gpu.")))
            }
        };

        match self.ptr.write() {
            Ok(mut ptr) => {
                match *ptr {
                    ref mut p => {
                        let len = self.value.as_raw_slice().len();

                        if let Some(memory_pool) = self.memory_pool.as_ref() {
                            let mut ptr = CudaMemoryPoolPtr::new(len, memory_pool)?;

                            ptr.memcpy(self.value.as_raw_slice().as_ptr(),len)?;

                            Ok(p.get_or_insert(ptr).as_ptr())
                        } else {
                            return Err(CudaError::LogicError(String::from(
                                "Memory pool is not set."
                            )));
                        }
                    }
                }
            },
            Err(_) => {
                Err(CudaError::InvalidState(String::from("Failed to allocate exclusive lock on memory for gpu.")))
            }
        }
    }
}
impl<U,T> Deref for CachedTensor<U,T> where U: Debug + Default, T: AsRawSlice<U> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
unsafe impl<U,T> Send for CachedTensor<U,T> where U: Debug + Default, T: AsRawSlice<U> {}
unsafe impl<U,T> Sync for CachedTensor<U,T> where U: Debug + Default, T: AsRawSlice<U> {}
