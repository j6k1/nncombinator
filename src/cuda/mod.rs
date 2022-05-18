use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use cuda_runtime_sys::dim3;
use libc::c_void;
use rcudnn_sys::{cudaMemcpyKind, cudaStream_t};
use crate::cuda::mem::MemoryPool;
use crate::error::{CudaError, CudaRuntimeError};

pub mod ffi;
pub mod mem;

pub(crate) mod private {
    pub trait AsKernelPtrBase {
        fn as_kernel_ptr(&self) -> *const libc::c_void;
    }

    pub trait AsMutKernelPtrBase {
        fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void;
    }
}
pub trait AsKernelPtr: private::AsKernelPtrBase {
}
pub trait AsMutKernelPtr: private::AsMutKernelPtrBase {
}
impl<T> AsKernelPtr for T where T: private::AsKernelPtrBase {}
impl<T> AsMutKernelPtr for T where T: private::AsMutKernelPtrBase {}
pub trait AsVoidPtr {
    fn as_void_ptr(&self) -> *const libc::c_void;
}
pub trait AsMutVoidPtr {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void;
}
pub trait AsPtr<T> {
    fn as_ptr(&self) -> *const T;
}
pub trait AsMutPtr<T> {
    fn as_mut_ptr(&mut self) -> *mut T;
}
impl AsVoidPtr for i32 {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self as *const i32 as *const libc::c_void
    }
}
impl AsMutVoidPtr for i32 {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void {
        self as *mut i32 as *mut libc::c_void
    }
}
impl AsVoidPtr for u32 {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self as *const u32 as *const libc::c_void
    }
}
impl AsMutVoidPtr for u32 {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void {
        self as *mut u32 as *mut libc::c_void
    }
}
impl AsVoidPtr for i64 {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self as *const i64 as *const libc::c_void
    }
}
impl AsMutVoidPtr for i64 {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void {
        self as *mut i64 as *mut libc::c_void
    }
}
impl AsVoidPtr for u64 {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self as *const u64 as *const libc::c_void
    }
}
impl AsMutVoidPtr for u64 {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void {
        self as *mut u64 as *mut libc::c_void
    }
}
impl AsVoidPtr for usize {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self as *const usize as *const libc::c_void
    }
}
impl AsMutVoidPtr for usize {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void {
        self as *mut usize as *mut libc::c_void
    }
}
impl AsVoidPtr for f32 {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self as *const f32 as *const libc::c_void
    }
}
impl AsMutVoidPtr for f32 {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void {
        self as *mut f32 as *mut libc::c_void
    }
}
impl AsVoidPtr for f64 {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self as *const f64 as *const libc::c_void
    }
}
impl AsMutVoidPtr for f64 {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void {
        self as *mut f64 as *mut libc::c_void
    }
}
impl private::AsKernelPtrBase for i32 {
    fn as_kernel_ptr(&self) -> *const libc::c_void {
        self as *const i32 as *const libc::c_void
    }
}
impl private::AsMutKernelPtrBase for i32 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut i32 as *mut libc::c_void
    }
}
impl private::AsKernelPtrBase for u32 {
    fn as_kernel_ptr(&self) -> *const libc::c_void {
        self as *const u32 as *const libc::c_void
    }
}
impl private::AsMutKernelPtrBase for u32 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut u32 as *mut libc::c_void
    }
}
impl private::AsKernelPtrBase for i64 {
    fn as_kernel_ptr(&self) -> *const libc::c_void {
        self as *const i64 as *const libc::c_void
    }
}
impl private::AsMutKernelPtrBase for i64 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut i64 as *mut libc::c_void
    }
}
impl private::AsKernelPtrBase for u64 {
    fn as_kernel_ptr(&self) -> *const libc::c_void {
        self as *const u64 as *const libc::c_void
    }
}
impl private::AsMutKernelPtrBase for u64 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut u64 as *mut libc::c_void
    }
}
impl private::AsKernelPtrBase for usize {
    fn as_kernel_ptr(&self) -> *const libc::c_void {
        self as *const usize as *const libc::c_void
    }
}
impl private::AsMutKernelPtrBase for usize {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut usize as *mut libc::c_void
    }
}
impl private::AsKernelPtrBase for f32 {
    fn as_kernel_ptr(&self) -> *const libc::c_void {
        self as *const f32 as *const libc::c_void
    }
}
impl private::AsMutKernelPtrBase for f32 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
        self as *mut f32 as *mut libc::c_void
    }
}
impl private::AsKernelPtrBase for f64 {
    fn as_kernel_ptr(&self) -> *const libc::c_void {
        self as *const f64 as *const libc::c_void
    }
}
impl private::AsMutKernelPtrBase for f64 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
        self as *mut f64 as *mut libc::c_void
    }
}
pub trait Memory<T: Default + Debug>: AsMutVoidPtr {
    fn memcpy(&mut self, p:*const T,len:usize) -> Result<usize,rcudnn::Error>;
    fn read_to_vec(&mut self) -> Result<Vec<T>,rcudnn::Error>;
    fn read_to_vec_with_size(&mut self,size:usize) -> Result<Vec<T>,rcudnn::Error>;
}
pub trait MemoryAsync<T: Default + Debug>: AsMutVoidPtr {
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
impl<T> private::AsKernelPtrBase for CudaPtr<T> {
    fn as_kernel_ptr(&self) -> *const libc::c_void {
        &self.ptr as *const *mut T as *const libc::c_void
    }
}
impl<T> private::AsMutKernelPtrBase for CudaPtr<T> {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
        &mut self.ptr as *mut *mut T as *mut libc::c_void
    }
}
impl<T> AsVoidPtr for CudaPtr<T> {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self.ptr as *const libc::c_void
    }
}
impl<T> AsMutVoidPtr for CudaPtr<T> {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void {
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
impl<T> private::AsKernelPtrBase for CudaHostPtr<T> {
    fn as_kernel_ptr(&self) -> *const libc::c_void {
        &self.ptr as *const *mut T as *const libc::c_void
    }
}
impl<T> private::AsMutKernelPtrBase for CudaHostPtr<T> {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
        &mut self.ptr as *mut *mut T as *mut libc::c_void
    }
}
impl<T> AsVoidPtr for CudaHostPtr<T> {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self.ptr as *const libc::c_void
    }
}
impl<T> AsMutVoidPtr for CudaHostPtr<T> {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void {
        self.ptr as *mut libc::c_void
    }
}
pub struct CudaMemoryPoolPtr<T> {
    ptr:*mut T,
    size:usize,
    memory_pool:Arc<Mutex<MemoryPool>>
}
impl<T> CudaMemoryPoolPtr<T> {
    pub fn new(size: usize,memory_pool:&Arc<Mutex<MemoryPool>>) -> Result<CudaMemoryPoolPtr<T>, CudaError> {
        let ptr:* mut T = match memory_pool.lock() {
            Ok(mut memory_pool) => {
                memory_pool.alloc_device(size)?
            },
            Err(_) => {
                return Err(CudaError::InvalidState(String::from(
                    "Failed to secure exclusive lock on memory pool."
                )));
            }
        };

        Ok(CudaMemoryPoolPtr {
            ptr: ptr,
            size: size,
            memory_pool:Arc::clone(memory_pool)
        })
    }
}
impl<T: Default + Debug> Memory<T> for CudaMemoryPoolPtr<T> {
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
impl<T> Drop for CudaMemoryPoolPtr<T> {
    fn drop(&mut self) {
        match self.memory_pool.lock() {
            Ok(mut memory_pool) => {
                memory_pool.deallocate(self.ptr).unwrap();
            },
            Err(_) => {
                panic!("Failed to secure exclusive lock on memory pool.");
            }
        }
    }
}
impl<T> private::AsKernelPtrBase for CudaMemoryPoolPtr<T> {
    fn as_kernel_ptr(&self) -> *const libc::c_void {
        &self.ptr as *const *mut T as *const libc::c_void
    }
}
impl<T> private::AsMutKernelPtrBase for CudaMemoryPoolPtr<T> {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
        &mut self.ptr as *mut *mut T as *mut libc::c_void
    }
}
impl<T> AsVoidPtr for CudaMemoryPoolPtr<T> {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self.ptr as *const libc::c_void
    }
}
impl<T> AsMutVoidPtr for CudaMemoryPoolPtr<T> {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void {
        self.ptr as *mut libc::c_void
    }
}
impl<T> AsPtr<T> for CudaMemoryPoolPtr<T> {
    fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }
}
impl<T> AsMutPtr<T> for CudaMemoryPoolPtr<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
impl TryFrom<f32> for CudaPtr<f32> {
    type Error = CudaError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        let mut ptr:CudaPtr<f32> = CudaPtr::new(1)?;
        ptr.memcpy(&value as *const f32,1)?;
        Ok(ptr)
    }
}
impl TryFrom<f64> for CudaPtr<f64> {
    type Error = CudaError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        let mut ptr:CudaPtr<f64> = CudaPtr::new(1)?;
        ptr.memcpy(&value as *const f64,1)?;
        Ok(ptr)
    }
}
pub trait KernelArgs {
    fn as_vec(&mut self) ->  Vec<&mut dyn AsMutKernelPtr>;
}
pub trait Kernel {
    type Args: KernelArgs;

    const FUNC_PTR: *const c_void;

    fn launch(&mut self,grid_dim:dim3,block_dim:dim3,args:&mut Self::Args,shared_mem:usize) -> Result<(),CudaRuntimeError> {
        ffi::launch(Self::FUNC_PTR,
                     grid_dim,
                     block_dim,
                     &mut args.as_vec().into_iter()
                         .map(|p| p.as_mut_kernel_ptr())
                         .collect::<Vec<*mut c_void>>().as_mut_slice(),
                     shared_mem
        )
    }
}
