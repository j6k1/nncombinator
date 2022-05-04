use std::fmt::Debug;
use rcudnn_sys::{cudaMemcpyKind, cudaStream_t};

pub mod ffi;

pub trait AsVoidPtr {
    fn as_void_ptr(&self) -> *const libc::c_void;
}
pub trait AsVoidMutPtr {
    fn as_void_mut_ptr(&mut self) -> *mut libc::c_void;
}
pub trait AsPtr<T> {
    fn as_ptr(&self) -> *const T;
}
pub trait AsMutPtr<T> {
    fn as_mut_ptr(&mut self) -> *mut T;
}
pub trait Memory<T: Default + Debug>: AsVoidMutPtr {
    fn memcpy(&mut self, p:*const T,len:usize) -> Result<usize,rcudnn::Error>;
    fn read_to_vec(&mut self) -> Result<Vec<T>,rcudnn::Error>;
    fn read_to_vec_with_size(&mut self,size:usize) -> Result<Vec<T>,rcudnn::Error>;
}
pub trait MemoryAsync<T: Default + Debug>: AsVoidMutPtr {
    fn memcpy_async(&mut self, p:*const T,len:usize,stream:cudaStream_t) -> Result<usize,rcudnn::Error>;
    fn read_to_vec_async(&mut self,stream:cudaStream_t) -> Result<Vec<T>,rcudnn::Error>;
    fn read_to_vec_with_size_async(&mut self,stream: cudaStream_t,size:usize) -> Result<Vec<T>,rcudnn::Error>;
}
pub struct CudaPtr<T> {
    ptr:*mut T,
    size:usize,
}
impl<T> CudaPtr<T> {
    pub fn new(size: usize) -> Result<CudaPtr<T>, rcudnn::Error> {
        let ptr: *mut T = ffi::malloc(size)?;

        Ok(CudaPtr {
            ptr: ptr,
            size: size
        })
    }
}
impl<T: Default + Debug> Memory<T> for CudaPtr<T> {
    fn memcpy(&mut self, p:*const T,len:usize) -> Result<usize,rcudnn::Error> {
        ffi::memcpy(self.ptr,
                         p,
                         len,
                         cudaMemcpyKind::cudaMemcpyHostToDevice)?;
        Ok(len)
    }

    fn read_to_vec(&mut self) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                         self.ptr,
                         self.size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(r)
    }

    fn read_to_vec_with_size(&mut self,size:usize) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(size);
        r.resize_with(size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                         self.ptr,
                         size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(r)
    }
}

impl<T> Drop for CudaPtr<T> {
    fn drop(&mut self) {
        ffi::free(self.ptr).unwrap();
    }
}
impl<T> AsVoidPtr for CudaPtr<T> {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self.ptr as *const T as *const libc::c_void
    }
}
impl<T> AsVoidMutPtr for CudaPtr<T> {
    fn as_void_mut_ptr(&mut self) -> *mut libc::c_void {
        self.ptr as *mut T as *mut libc::c_void
    }
}
impl<T> AsPtr<T> for CudaPtr<T> {
    fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }
}
impl<T> AsMutPtr<T> for CudaPtr<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
unsafe impl<T> Send for CudaPtr<T> where T: Send {}
pub struct CudaHostPtr<T> {
    ptr:*mut T,
    size:usize,
}
impl<T> CudaHostPtr<T> {
    pub fn new(size: usize,flags:libc::c_uint) -> Result<CudaHostPtr<T>, rcudnn::Error> {
        let ptr: *mut T = ffi::malloc_host(size,flags)?;
        Ok(CudaHostPtr {
            ptr: ptr,
            size: size
        })
    }
}
impl<T: Default + Debug> Memory<T> for CudaHostPtr<T> {
    fn memcpy(&mut self, p:*const T,len:usize) -> Result<usize,rcudnn::Error> {
        ffi::memcpy(self.ptr,
                         p,
                         len,
                         cudaMemcpyKind::cudaMemcpyHostToDevice)?;
        Ok(len)
    }

    fn read_to_vec(&mut self) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                         self.ptr,
                         self.size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(r)
    }

    fn read_to_vec_with_size(&mut self,size:usize) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(size);
        r.resize_with(size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                         self.ptr,
                         size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(r)
    }
}
impl<T: Default + Debug> MemoryAsync<T> for CudaHostPtr<T> {
    fn memcpy_async(&mut self, p:*const T,len:usize,stream:cudaStream_t) -> Result<usize,rcudnn::Error> {
        ffi::memcpy_async(self.ptr,
                               p,
                               len,
                               cudaMemcpyKind::cudaMemcpyHostToDevice,stream)?;
        Ok(len)
    }

    fn read_to_vec_async(&mut self, stream: cudaStream_t) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy_async(r.as_mut_ptr(),
                               self.ptr,
                               self.size,
                               cudaMemcpyKind::cudaMemcpyDeviceToHost,
                               stream)?;
        Ok(r)
    }

    fn read_to_vec_with_size_async(&mut self, stream: cudaStream_t, size:usize) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(size);
        r.resize_with(size,Default::default);

        ffi::memcpy_async(r.as_mut_ptr(),
                               self.ptr,
                               size,
                               cudaMemcpyKind::cudaMemcpyDeviceToHost,
                               stream)?;
        Ok(r)
    }
}

impl<T> Drop for CudaHostPtr<T> {
    fn drop(&mut self) {
        ffi::free_host(self.ptr).unwrap();
    }
}
impl<T> AsPtr<T> for CudaHostPtr<T> {
    fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }
}
impl<T> AsMutPtr<T> for CudaHostPtr<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
impl<T> AsVoidPtr for CudaHostPtr<T> {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self.ptr as *const T as *const libc::c_void
    }
}
impl<T> AsVoidMutPtr for CudaHostPtr<T> {
    fn as_void_mut_ptr(&mut self) -> *mut libc::c_void {
        self.ptr as *mut T as *mut libc::c_void
    }
}
