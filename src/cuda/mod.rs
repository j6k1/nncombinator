use std::fmt::Debug;
use rcudnn_sys::{cudaMemcpyKind, cudaStream_t};
use crate::error::CudaError;

pub mod ffi;

pub trait AsDoubleVoidPtr {
    fn as_double_void_ptr(&self) -> *const libc::c_void;
}
pub trait AsDoubleVoidMutPtr {
    fn as_double_void_mut_ptr(&mut self) -> *mut libc::c_void;
}
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
impl AsVoidMutPtr for i32 {
    fn as_void_mut_ptr(&mut self) -> *mut libc::c_void {
        self as *mut i32 as *mut libc::c_void
    }
}
impl AsVoidMutPtr for u32 {
    fn as_void_mut_ptr(&mut self) -> *mut libc::c_void {
        self as *mut u32 as *mut libc::c_void
    }
}
impl AsVoidMutPtr for i64 {
    fn as_void_mut_ptr(&mut self) -> *mut libc::c_void {
        self as *mut i64 as *mut libc::c_void
    }
}
impl AsVoidMutPtr for u64 {
    fn as_void_mut_ptr(&mut self) -> *mut libc::c_void {
        self as *mut u64 as *mut libc::c_void
    }
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
    pub fn new(size: usize) -> Result<CudaPtr<T>, CudaError> {
        let ptr: *mut T = ffi::malloc(size)?;

        Ok(CudaPtr {
            ptr: ptr,
            size: size,
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
impl<T> AsDoubleVoidPtr for CudaPtr<T> {
    fn as_double_void_ptr(&self) -> *const libc::c_void {
        &self.ptr as *const *mut T as *const libc::c_void
    }
}
impl<T> AsDoubleVoidMutPtr for CudaPtr<T> {
    fn as_double_void_mut_ptr(&mut self) -> *mut libc::c_void {
        &mut self.ptr as *mut *mut T as *mut libc::c_void
    }
}
impl<T> AsVoidPtr for CudaPtr<T> {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self.ptr as *const libc::c_void
    }
}
impl<T> AsVoidMutPtr for CudaPtr<T> {
    fn as_void_mut_ptr(&mut self) -> *mut libc::c_void {
        self.ptr as *mut libc::c_void
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
    pub fn new(size: usize, flags:libc::c_uint) -> Result<CudaHostPtr<T>, CudaError> {
        let ptr: *mut T = ffi::malloc_host(size,flags)?;

        Ok(CudaHostPtr {
            ptr: ptr,
            size: size,
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
        ffi::free_host(self.ptr).unwrap()
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
impl<T> AsDoubleVoidPtr for CudaHostPtr<T> {
    fn as_double_void_ptr(&self) -> *const libc::c_void {
        &self.ptr as *const *mut T as *const libc::c_void
    }
}
impl<T> AsDoubleVoidMutPtr for CudaHostPtr<T> {
    fn as_double_void_mut_ptr(&mut self) -> *mut libc::c_void {
        &mut self.ptr as *mut *mut T as *mut libc::c_void
    }
}
impl<T> AsVoidPtr for CudaHostPtr<T> {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self.ptr as *const libc::c_void
    }
}
impl<T> AsVoidMutPtr for CudaHostPtr<T> {
    fn as_void_mut_ptr(&mut self) -> *mut libc::c_void {
        self.ptr as *mut libc::c_void
    }
}
unsafe impl<T> Send for CudaHostPtr<T> where T: Send {}
pub struct ConstCudaPtr<T> where T: Default + Debug {
    ptr:CudaPtr<T>
}
impl<T> ConstCudaPtr<T> where T: Default + Debug {
    pub fn new(size:usize,ptr:*const T) -> Result<ConstCudaPtr<T>,CudaError> {
        let mut p = CudaPtr::new(size)?;

        p.memcpy(ptr,size)?;

        Ok(ConstCudaPtr {
            ptr: p
        })
    }
}
impl<T> AsVoidPtr for ConstCudaPtr<T> where T: Default + Debug {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self.as_void_ptr()
    }
}
impl<T> AsPtr<T> for ConstCudaPtr<T> where T: Default + Debug {
    fn as_ptr(&self) -> *const T {
        self.as_ptr()
    }
}
unsafe impl<T> Send for ConstCudaPtr<T> where T: Default + Debug {}
unsafe impl<T> Sync for ConstCudaPtr<T> where T: Default + Debug {}