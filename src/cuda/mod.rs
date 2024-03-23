//! Function to wrap and handle cuda kernel

use std::fmt;
use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use cuda_runtime_sys::{cudaHostAllocDefault, dim3};
use libc::c_void;
use rcudnn::Error;
use rcudnn::utils::DataType;
use rcudnn_sys::{cudaMemcpyKind, cudaStream_t, cudnnDataType_t};
use crate::cuda::mem::{Alloctype, MemoryPool};
use crate::error::{CudaError, CudaRuntimeError};

pub mod ffi;
pub mod mem;
pub mod kernel;
pub mod cudnn;

/// Trait to associate a type with a cudnn type
pub trait DataTypeInfo {
    /// get cudnn data type
    fn cudnn_data_type() -> DataType;
    /// get cudnn raw data type
    fn cudnn_raw_data_type() -> cudnnDataType_t;
    /// get size
    fn size() -> usize;
}
impl DataTypeInfo for f32 {
    fn cudnn_data_type() -> DataType {
        DataType::Float
    }
    fn cudnn_raw_data_type() -> cudnnDataType_t {
        cudnnDataType_t::CUDNN_DATA_FLOAT
    }
    fn size() -> usize {
        4_usize
    }
}
impl DataTypeInfo for f64 {
    fn cudnn_data_type() -> DataType {
        DataType::Double
    }
    fn cudnn_raw_data_type() -> cudnnDataType_t {
        cudnnDataType_t::CUDNN_DATA_DOUBLE
    }
    fn size() -> usize {
        8_usize
    }
}
pub(crate) mod private {
    pub trait AsConstKernelPtrBase {
        fn as_const_kernel_ptr(&self) -> *mut libc::c_void;
    }

    pub trait AsMutKernelPtrBase {
        fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void;
    }

    pub trait AsKernelPtrBase {
        fn as_kernel_ptr(&mut self) -> *mut libc::c_void;
    }
}
/// Trait defining the conversion to an immutable pointer type passed to the cuda kernel
pub trait AsConstKernelPtr: private::AsConstKernelPtrBase {
}
/// Trait defining the conversion to an mutable pointer type passed to the cuda kernel
pub trait AsMutKernelPtr: private::AsMutKernelPtrBase {
}
/// Trait defining the conversion to an pointer type passed to the cuda kernel
pub trait AsKernelPtr: private::AsKernelPtrBase {
}
impl<T> AsConstKernelPtr for T where T: private::AsConstKernelPtrBase {}
impl<T> AsMutKernelPtr for T where T: private::AsMutKernelPtrBase {}
impl<T> private::AsKernelPtrBase for T where T: AsMutKernelPtr {
    fn as_kernel_ptr(&mut self) -> *mut c_void {
        self.as_mut_kernel_ptr()
    }
}
impl<T> AsKernelPtr for T where T: private::AsKernelPtrBase {

}
/// Obtaining an immutable void pointer
pub trait AsVoidPtr {
    fn as_void_ptr(&self) -> *const libc::c_void;
}
/// Obtaining an mutable void pointer
pub trait AsMutVoidPtr {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void;
}
/// Obtaining an immutable pointer
pub trait AsPtr<T> {
    fn as_ptr(&self) -> *const T;
}
/// Obtaining an mutable pointer
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
impl private::AsMutKernelPtrBase for i32 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut i32 as *mut libc::c_void
    }
}
impl private::AsMutKernelPtrBase for u32 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut u32 as *mut libc::c_void
    }
}
impl private::AsMutKernelPtrBase for i64 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut i64 as *mut libc::c_void
    }
}
impl private::AsMutKernelPtrBase for u64 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut u64 as *mut libc::c_void
    }
}
impl private::AsMutKernelPtrBase for usize {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut usize as *mut libc::c_void
    }
}
impl private::AsMutKernelPtrBase for f32 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
        self as *mut f32 as *mut libc::c_void
    }
}
impl private::AsMutKernelPtrBase for f64 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
        self as *mut f64 as *mut libc::c_void
    }
}
/// Trait defining cuda's synchronous memory operations
pub trait Memory<T: Default + Debug>: AsMutVoidPtr {
    /// Memory Copy
    /// # Arguments
    /// * `p` - Pointer to source memory
    /// * `len` - Number of elements of the value to be copied
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
    fn memcpy(&mut self, p:*const T,len:usize) -> Result<usize,rcudnn::Error>;
    /// Repeatedly copy the contents of memory
    /// # Arguments
    /// * `p` - Pointer to source memory
    /// * `len` - Number of elements of the value to be copied
    /// * `count` - Number of times to copy repeatedly
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
    fn memcpy_repeat(&mut self, p:*const T,len:usize,count:usize) -> Result<usize,rcudnn::Error>;
    /// Read memory as Vec
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
    fn read_to_vec(&self) -> Result<Vec<T>,rcudnn::Error>;
    /// Read memory as Vec with size specified
    /// # Arguments
    /// * `size` - Number of elements of the value to be read
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
    fn read_to_vec_with_size(&self,size:usize) -> Result<Vec<T>,rcudnn::Error>;
}
/// Trait defining cuda's asynchronous memory operations
pub trait MemoryAsync<T: Default + Debug>: AsMutVoidPtr {
    /// Memory Copy
    /// # Arguments
    /// * `p` - Pointer to source memory
    /// * `len` - Number of elements of the value to be copied
    /// * `stream` - cuda stream
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
    fn memcpy_async(&mut self, p:*const T,len:usize,stream:cudaStream_t) -> Result<usize,rcudnn::Error>;
    /// Repeatedly copy the contents of memory
    /// # Arguments
    /// * `p` - Pointer to source memory
    /// * `len` - Number of elements of the value to be copied
    /// * `count` - Number of times to copy repeatedly
    /// * `stream` - cuda stream
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    fn memcpy_async_repeat(&mut self, p:*const T,len:usize,count:usize,stream:cudaStream_t) -> Result<usize,rcudnn::Error>;
    ///
    /// Read memory as Vec
    /// # Arguments
    /// * `stream` - cuda stream
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
    fn read_to_vec_async(&self,stream:cudaStream_t) -> Result<Vec<T>,rcudnn::Error>;
    /// Read memory as Vec with size specified
    /// # Arguments
    /// * `stream` - cuda stream
    /// * `size` - Number of elements of the value to be read
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
    fn read_to_vec_with_size_async(&self,stream: cudaStream_t,size:usize) -> Result<Vec<T>,rcudnn::Error>;
}
/// Trait defining cuda's synchronous memory move to operations
pub trait MemoryMoveTo<T: Default + Debug,D: AsMutPtr<T>>: AsPtr<T> {
    /// Memory Copy To
    /// # Arguments
    /// * `dst` - Pointer to destination memory
    /// * `len` - Number of elements of the value to be copied
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
    fn memcpy_to(&self, dst:&mut D,len:usize) -> Result<usize,rcudnn::Error>;
}
/// Trait defining cuda's asynchronous memory move to operations
pub trait MemoryMoveToAsync<T: Default + Debug,D: AsMutPtr<T>>: AsPtr<T> {
    /// Memory Copy To
    /// # Arguments
    /// * `dst` - Pointer to destination memory
    /// * `len` - Number of elements of the value to be copied
    /// * `stream` - cuda stream
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
    fn memcpy_to_async(&self, dst:&mut D,len:usize,stream:cudaStream_t) -> Result<usize,rcudnn::Error>;
}
/// Wrapper to handle cuda device memory
#[derive(Debug)]
pub struct CudaPtr<T> {
    ptr:*mut T,
    size:usize,
}
impl<T> CudaPtr<T> {
    /// Create an instance of CudaPtr
    /// # Arguments
    /// * `size`- Number of value elements to be allocated
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
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

    fn memcpy_repeat(&mut self, p: *const T, len: usize, count: usize) -> Result<usize, Error> {
        for i in 0..count {
            unsafe {
                ffi::memcpy(self.ptr.add(i * len),
                            p,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToDevice)?;
            }
        }
        Ok(len * count)
    }

    fn read_to_vec(&self) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                         self.ptr,
                         self.size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(r)
    }

    fn read_to_vec_with_size(&self,size:usize) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(size);
        r.resize_with(size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                         self.ptr,
                         size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(r)
    }
}
impl<T: Default + Debug> MemoryMoveTo<T,CudaHostPtr<T>> for CudaPtr<T> {
    fn memcpy_to(&self, dst: &mut CudaHostPtr<T>, len: usize) -> Result<usize, Error> {
        ffi::memcpy(dst.as_mut_ptr(),
                    self.ptr,
                    len,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(len)
    }
}
impl<T: Default + Debug> MemoryMoveTo<T,CudaPtr<T>> for CudaPtr<T> {
    fn memcpy_to(&self, dst: &mut CudaPtr<T>, len: usize) -> Result<usize, Error> {
        ffi::memcpy(dst.as_mut_ptr(),
                    self.ptr,
                    len,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice)?;
        Ok(len)
    }
}
impl<T: Default + Debug> MemoryMoveTo<T,CudaMemoryPoolPtr<T>> for CudaPtr<T> {
    fn memcpy_to(&self, dst: &mut CudaMemoryPoolPtr<T>, len: usize) -> Result<usize, Error> {
        match dst.get_alloc_type() {
            Alloctype::Device => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice)?;
                Ok(len)
            },
            Alloctype::Host(_) => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
                Ok(len)
            }
        }
    }
}
impl<T> Drop for CudaPtr<T> {
    fn drop(&mut self) {
        ffi::free(self.ptr).unwrap();
    }
}
impl<T> private::AsConstKernelPtrBase for CudaPtr<T> {
    fn as_const_kernel_ptr(&self) -> *mut libc::c_void {
        &self.ptr as *const *mut T as *mut libc::c_void
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
/// Wrapper to handle cuda host memory
#[derive(Debug)]
pub struct CudaHostPtr<T> {
    ptr:*mut T,
    size:usize,
}
impl<T> CudaHostPtr<T> {
    /// Create an instance of CudaHostPtr
    /// # Arguments
    /// * `size`- Number of value elements to be allocated
    /// * `flags` - Requested properties of allocated memory
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
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

    fn memcpy_repeat(&mut self, p: *const T, len: usize, count: usize) -> Result<usize, Error> {
        for i in 0..count {
            unsafe {
                ffi::memcpy(self.ptr.add(i * len),
                            p,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToDevice)?;
            }
        }
        Ok(len * count)
    }

    fn read_to_vec(&self) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                         self.ptr,
                         self.size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(r)
    }

    fn read_to_vec_with_size(&self,size:usize) -> Result<Vec<T>,rcudnn::Error> {
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

    fn memcpy_async_repeat(&mut self, p: *const T, len: usize, count: usize, stream: cudaStream_t) -> Result<usize, Error> {
        for i in 0..count {
            unsafe {
                ffi::memcpy_async(self.ptr.add(i * len),
                            p,
                            len,
                                  cudaMemcpyKind::cudaMemcpyHostToDevice,stream)?;
            }
        }
        Ok(len * count)

    }

    fn read_to_vec_async(&self, stream: cudaStream_t) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy_async(r.as_mut_ptr(),
                               self.ptr,
                               self.size,
                               cudaMemcpyKind::cudaMemcpyDeviceToHost,
                               stream)?;
        Ok(r)
    }

    fn read_to_vec_with_size_async(&self, stream: cudaStream_t, size:usize) -> Result<Vec<T>,rcudnn::Error> {
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
impl<T: Default + Debug> MemoryMoveTo<T,CudaHostPtr<T>> for CudaHostPtr<T> {
    fn memcpy_to(&self, dst: &mut CudaHostPtr<T>, len: usize) -> Result<usize, Error> {
        ffi::memcpy(dst.as_mut_ptr(),
                    self.ptr,
                    len,
                    cudaMemcpyKind::cudaMemcpyHostToHost)?;
        Ok(len)
    }
}
impl<T: Default + Debug> MemoryMoveTo<T,CudaPtr<T>> for CudaHostPtr<T> {
    fn memcpy_to(&self, dst: &mut CudaPtr<T>, len: usize) -> Result<usize, Error> {
        ffi::memcpy(dst.as_mut_ptr(),
                    self.ptr,
                    len,
                    cudaMemcpyKind::cudaMemcpyHostToDevice)?;
        Ok(len)
    }
}
impl<T: Default + Debug> MemoryMoveTo<T,CudaMemoryPoolPtr<T>> for CudaHostPtr<T> {
    fn memcpy_to(&self, dst: &mut CudaMemoryPoolPtr<T>, len: usize) -> Result<usize, Error> {
        match dst.get_alloc_type() {
            Alloctype::Device => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToDevice)?;
                Ok(len)
            },
            Alloctype::Host(_) => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToHost)?;
                Ok(len)
            }
        }
    }
}
impl<T: Default + Debug> MemoryMoveToAsync<T,CudaHostPtr<T>> for CudaHostPtr<T> {
    fn memcpy_to_async(&self, dst: &mut CudaHostPtr<T>, len: usize,stream:cudaStream_t) -> Result<usize, Error> {
        ffi::memcpy_async(dst.as_mut_ptr(),
                    self.ptr,
                    len,
                    cudaMemcpyKind::cudaMemcpyHostToHost,stream)?;
        Ok(len)
    }
}
impl<T: Default + Debug> MemoryMoveToAsync<T,CudaPtr<T>> for CudaHostPtr<T> {
    fn memcpy_to_async(&self, dst: &mut CudaPtr<T>, len: usize,stream:cudaStream_t) -> Result<usize, Error> {
        ffi::memcpy_async(dst.as_mut_ptr(),
                    self.ptr,
                    len,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,stream)?;
        Ok(len)
    }
}
impl<T: Default + Debug> MemoryMoveToAsync<T,CudaMemoryPoolPtr<T>> for CudaHostPtr<T> {
    fn memcpy_to_async(&self, dst: &mut CudaMemoryPoolPtr<T>, len: usize,stream:cudaStream_t) -> Result<usize, Error> {
        match dst.get_alloc_type() {
            Alloctype::Device => {
                ffi::memcpy_async(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToDevice,stream)?;
                Ok(len)
            },
            Alloctype::Host(_) => {
                ffi::memcpy_async(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToHost,stream)?;
                Ok(len)
            }
        }
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
impl<T> private::AsConstKernelPtrBase for CudaHostPtr<T> {
    fn as_const_kernel_ptr(&self) -> *mut libc::c_void {
        &self.ptr as *const *mut T as *mut libc::c_void
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
/// Cuda memory object allocated from the memory pool
pub struct CudaMemoryPoolPtr<T> {
    ptr:*mut T,
    size:usize,
    memory_pool:Arc<Mutex<MemoryPool>>,
    alloc_type: Alloctype
}
impl<T> CudaMemoryPoolPtr<T> {
    /// Create an instance of CudaMemoryPoolPtr
    /// # Arguments
    /// * `size`- Number of value elements to be allocated
    /// * `memory_pool` - memory pool object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new(size: usize,memory_pool:&Arc<Mutex<MemoryPool>>) -> Result<CudaMemoryPoolPtr<T>, CudaError> {
        let (ptr,alloc_type):(*mut T,Alloctype) = match memory_pool.lock() {
            Ok(mut memory_pool) => {
                (memory_pool.alloc_device(size)?,memory_pool.get_alloc_type())
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
            memory_pool:Arc::clone(memory_pool),
            alloc_type: alloc_type
        })
    }

    pub fn get_alloc_type(&self) -> Alloctype {
        self.alloc_type
    }
}
impl<T> CudaMemoryPoolPtr<T> where T: Default + Debug {
    /// Create an instance of CudaMemoryPoolPtr
    /// # Arguments
    /// * `size`- Number of value elements to be allocated
    /// * `memory_pool` - memory pool object
    /// * `initializer` - Repeatedly called function to initialize each element
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn with_initializer<I: FnMut() -> T>(size: usize, memory_pool:&Arc<Mutex<MemoryPool>>, initializer: I) -> Result<CudaMemoryPoolPtr<T>, CudaError> {
        let mut ptr = Self::new(size,memory_pool)?;

        let mut src = Vec::with_capacity(size);

        src.resize_with(size,initializer);

        ptr.memcpy(src.into_boxed_slice().as_ptr(),size)?;

        Ok(ptr)
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

    fn memcpy_repeat(&mut self, p: *const T, len: usize, count: usize) -> Result<usize, Error> {
        for i in 0..count {
            unsafe {
                ffi::memcpy(self.ptr.add(i * len),
                            p,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToDevice)?;
            }
        }
        Ok(len * count)
    }

    fn read_to_vec(&self) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                    self.ptr,
                    self.size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(r)
    }

    fn read_to_vec_with_size(&self,size:usize) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(size);
        r.resize_with(size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                    self.ptr,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(r)
    }
}
impl<T: Default + Debug> MemoryMoveTo<T,CudaHostPtr<T>> for CudaMemoryPoolPtr<T> {
    fn memcpy_to(&self, dst: &mut CudaHostPtr<T>, len: usize) -> Result<usize, Error> {
        match self.alloc_type {
            Alloctype::Device => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
                Ok(len)
            },
            Alloctype::Host(_) => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToHost)?;
                Ok(len)

            }
        }
    }
}
impl<T: Default + Debug> MemoryMoveTo<T,CudaPtr<T>> for CudaMemoryPoolPtr<T> {
    fn memcpy_to(&self, dst: &mut CudaPtr<T>, len: usize) -> Result<usize, Error> {
        match self.alloc_type {
            Alloctype::Device => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice)?;
                Ok(len)
            },
            Alloctype::Host(_) => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToDevice)?;
                Ok(len)
            }
        }
    }
}
impl<T: Default + Debug> MemoryMoveTo<T,CudaMemoryPoolPtr<T>> for CudaMemoryPoolPtr<T> {
    fn memcpy_to(&self, dst: &mut CudaMemoryPoolPtr<T>, len: usize) -> Result<usize, Error> {
        match (self.get_alloc_type(),dst.get_alloc_type()) {
            (Alloctype::Device,Alloctype::Device) => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyDeviceToDevice)?;
                Ok(len)
            },
            (Alloctype::Host(_),Alloctype::Device) => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToDevice)?;
                Ok(len)
            },
            (Alloctype::Device,Alloctype::Host(_)) => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
                Ok(len)
            },
            (Alloctype::Host(_),Alloctype::Host(_)) => {
                ffi::memcpy(dst.as_mut_ptr(),
                            self.ptr,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToHost)?;
                Ok(len)
            }
        }
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
impl<T> private::AsConstKernelPtrBase for CudaMemoryPoolPtr<T> {
    fn as_const_kernel_ptr(&self) -> *mut libc::c_void {
        &self.ptr as *const *mut T as *mut libc::c_void
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
impl<T> Debug for CudaMemoryPoolPtr<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f,"CudaMemoryPoolPtr {{ ptr: {:?}, size {:?} }}",self.ptr,self.size)
    }
}
/// Type that represents a pointer of const type to be passed to Cuda
#[derive(Debug)]
pub struct CudaConstPtr<'a,T> where T: AsConstKernelPtr {
    ptr:&'a T
}
impl<'a,T> CudaConstPtr<'a,T> where T: AsConstKernelPtr {
    pub fn new(ptr: &'a T) -> CudaConstPtr<'a,T> {
        CudaConstPtr {
            ptr
        }
    }
}
impl<'a,T> private::AsKernelPtrBase for CudaConstPtr<'a,T> where T: AsConstKernelPtr {
    fn as_kernel_ptr(&mut self) -> *mut c_void {
        self.ptr.as_const_kernel_ptr()
    }
}
/// Cuda memory object representing a 1D array with dimension number as type parameter
#[derive(Debug)]
pub struct CudaTensor1dPtr<T,const N:usize> where T: Default + Debug {
    ptr:CudaMemoryPoolPtr<T>
}
impl<T,const N:usize> CudaTensor1dPtr<T,N> where T: Default + Debug {
    /// Create an instance of CudaTensor1dPtr
    /// # Arguments
    /// * `memory_pool` - memory pool object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new(memory_pool:&Arc<Mutex<MemoryPool>>) -> Result<CudaTensor1dPtr<T,N>, CudaError> {
        Ok(CudaTensor1dPtr {
            ptr:CudaMemoryPoolPtr::new(N,memory_pool)?
        })
    }

    /// Create an instance of CudaMemoryPoolPtr
    /// # Arguments
    /// * `memory_pool` - memory pool object
    /// * `initializer` - Repeatedly called function to initialize each element
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn with_initializer<I: FnMut() -> T>(memory_pool:&Arc<Mutex<MemoryPool>>, initializer: I) -> Result<CudaTensor1dPtr<T,N>, CudaError> {
        let mut ptr = CudaMemoryPoolPtr::new(N,memory_pool)?;

        let mut src = Vec::with_capacity(N);

        src.resize_with(N,initializer);

        ptr.memcpy(src.into_boxed_slice().as_ptr(),N)?;

        Ok(CudaTensor1dPtr {
            ptr: ptr
        })
    }
}
impl<T,const N:usize> private::AsConstKernelPtrBase for CudaTensor1dPtr<T,N> where T: Default + Debug {
    fn as_const_kernel_ptr(&self) -> *mut c_void {
        self.ptr.as_const_kernel_ptr()
    }
}
impl<T,const N:usize> Deref for CudaTensor1dPtr<T,N> where T: Default + Debug {
    type Target = CudaMemoryPoolPtr<T>;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}
impl<T,const N:usize> DerefMut for CudaTensor1dPtr<T,N> where T: Default + Debug {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr
    }   
}
impl<'a,T,const N:usize> From<&'a CudaTensor1dPtr<T,N>> for &'a CudaMemoryPoolPtr<T> where T: Default + Debug {
    fn from(value: &'a CudaTensor1dPtr<T,N>) -> Self {
        &value.ptr
    }
}
impl<'a,T,const N:usize> From<&'a mut CudaTensor1dPtr<T,N>> for &'a mut CudaMemoryPoolPtr<T> where T: Default + Debug {
    fn from(value: &'a mut CudaTensor1dPtr<T,N>) -> Self {
        &mut value.ptr
    }
}
/// Cuda memory object representing a 2D array with dimension number as type parameter
#[derive(Debug)]
pub struct CudaTensor2dPtr<T,const N1:usize,const N2:usize> where T: Default + Debug {
    ptr:CudaMemoryPoolPtr<T>
}
impl<T,const N1:usize,const N2:usize> CudaTensor2dPtr<T,N1,N2> where T: Default + Debug {
    /// Create an instance of CudaTensor1dPtr
    /// # Arguments
    /// * `memory_pool` - memory pool object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new(memory_pool:&Arc<Mutex<MemoryPool>>) -> Result<CudaTensor2dPtr<T,N1,N2>, CudaError> {
        Ok(CudaTensor2dPtr {
            ptr:CudaMemoryPoolPtr::new(N1*N2,memory_pool)?
        })
    }

    /// Create an instance of CudaMemoryPoolPtr
    /// # Arguments
    /// * `memory_pool` - memory pool object
    /// * `initializer` - Repeatedly called function to initialize each element
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn with_initializer<I: FnMut() -> T>(memory_pool:&Arc<Mutex<MemoryPool>>, initializer: I) -> Result<CudaTensor2dPtr<T,N1,N2>, CudaError> {
        let mut ptr = CudaMemoryPoolPtr::new(N1*N2,memory_pool)?;

        let mut src = Vec::with_capacity(N1*N2);

        src.resize_with(N1*N2,initializer);

        ptr.memcpy(src.into_boxed_slice().as_ptr(),N1*N2)?;

        Ok(CudaTensor2dPtr {
            ptr: ptr
        })
    }
}
impl<T,const N1:usize,const N2:usize> private::AsConstKernelPtrBase for CudaTensor2dPtr<T,N1,N2> where T: Default + Debug {
    fn as_const_kernel_ptr(&self) -> *mut c_void {
        self.ptr.as_const_kernel_ptr()
    }
}
impl<T,const N1:usize,const N2:usize> Deref for CudaTensor2dPtr<T,N1,N2> where T: Default + Debug {
    type Target = CudaMemoryPoolPtr<T>;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}
impl<T,const N1:usize,const N2:usize> DerefMut for CudaTensor2dPtr<T,N1,N2> where T: Default + Debug {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr
    }
}
impl<'a,T,const N1:usize,const N2:usize> From<&'a CudaTensor2dPtr<T,N1,N2>> for &'a CudaMemoryPoolPtr<T> where T: Default + Debug {
    fn from(value: &'a CudaTensor2dPtr<T,N1,N2>) -> Self {
        &value.ptr
    }
}
impl<'a,T,const N1:usize,const N2:usize> From<&'a mut CudaTensor2dPtr<T,N1,N2>> for &'a mut CudaMemoryPoolPtr<T> where T: Default + Debug {
    fn from(value: &'a mut CudaTensor2dPtr<T,N1,N2>) -> Self {
        &mut value.ptr
    }
}
/// Cuda memory object representing a 3D array with dimension number as type parameter
#[derive(Debug)]
pub struct CudaTensor3dPtr<T,const N1:usize,const N2:usize,const N3:usize> where T: Default + Debug {
    ptr:CudaMemoryPoolPtr<T>
}
impl<T,const N1:usize,const N2:usize,const N3:usize> CudaTensor3dPtr<T,N1,N2,N3> where T: Default + Debug {
    /// Create an instance of CudaTensor1dPtr
    /// # Arguments
    /// * `memory_pool` - memory pool object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new(memory_pool:&Arc<Mutex<MemoryPool>>) -> Result<CudaTensor3dPtr<T,N1,N2,N3>, CudaError> {
        Ok(CudaTensor3dPtr {
            ptr:CudaMemoryPoolPtr::new(N1*N2*N3,memory_pool)?
        })
    }

    /// Create an instance of CudaMemoryPoolPtr
    /// # Arguments
    /// * `memory_pool` - memory pool object
    /// * `initializer` - Repeatedly called function to initialize each element
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn with_initializer<I: FnMut() -> T>(memory_pool:&Arc<Mutex<MemoryPool>>, initializer: I) -> Result<CudaTensor3dPtr<T,N1,N2,N3>, CudaError> {
        let mut ptr = CudaMemoryPoolPtr::new(N1*N2*N3,memory_pool)?;

        let mut src = Vec::with_capacity(N1*N2*N3);

        src.resize_with(N1*N2*N3,initializer);

        ptr.memcpy(src.into_boxed_slice().as_ptr(),N1*N2*N3)?;

        Ok(CudaTensor3dPtr {
            ptr: ptr
        })
    }
}
impl<T,const N1:usize,const N2:usize,const N3:usize> private::AsConstKernelPtrBase for CudaTensor3dPtr<T,N1,N2,N3> where T: Default + Debug {
    fn as_const_kernel_ptr(&self) -> *mut c_void {
        self.ptr.as_const_kernel_ptr()
    }
}
impl<T,const N1:usize,const N2:usize,const N3:usize> Deref for CudaTensor3dPtr<T,N1,N2,N3> where T: Default + Debug {
    type Target = CudaMemoryPoolPtr<T>;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}
impl<T,const N1:usize,const N2:usize,const N3:usize> DerefMut for CudaTensor3dPtr<T,N1,N2,N3> where T: Default + Debug {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> From<&'a CudaTensor3dPtr<T,N1,N2,N3>> for &'a CudaMemoryPoolPtr<T> where T: Default + Debug {
    fn from(value: &'a CudaTensor3dPtr<T,N1,N2,N3>) -> Self {
        &value.ptr
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> From<&'a mut CudaTensor3dPtr<T,N1,N2,N3>> for &'a mut CudaMemoryPoolPtr<T> where T: Default + Debug {
    fn from(value: &'a mut CudaTensor3dPtr<T,N1,N2,N3>) -> Self {
        &mut value.ptr
    }
}
/// Cuda memory object representing a 4D array with dimension number as type parameter
#[derive(Debug)]
pub struct CudaTensor4dPtr<T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> where T: Default + Debug {
    ptr:CudaMemoryPoolPtr<T>
}
impl<T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> CudaTensor4dPtr<T,N1,N2,N3,N4> where T: Default + Debug {
    /// Create an instance of CudaTensor1dPtr
    /// # Arguments
    /// * `memory_pool` - memory pool object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new(memory_pool:&Arc<Mutex<MemoryPool>>) -> Result<CudaTensor4dPtr<T,N1,N2,N3,N4>, CudaError> {
        Ok(CudaTensor4dPtr {
            ptr:CudaMemoryPoolPtr::new(N1*N2*N3*N4,memory_pool)?
        })
    }

    /// Create an instance of CudaMemoryPoolPtr
    /// # Arguments
    /// * `memory_pool` - memory pool object
    /// * `initializer` - Repeatedly called function to initialize each element
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn with_initializer<I: FnMut() -> T>(memory_pool:&Arc<Mutex<MemoryPool>>, initializer: I) -> Result<CudaTensor4dPtr<T,N1,N2,N3,N4>, CudaError> {
        let mut ptr = CudaMemoryPoolPtr::new(N1*N2*N3*N4,memory_pool)?;

        let mut src = Vec::with_capacity(N1*N2*N3*N4);

        src.resize_with(N1*N2*N3*N4,initializer);

        ptr.memcpy(src.into_boxed_slice().as_ptr(),N1*N2*N3*N4)?;

        Ok(CudaTensor4dPtr {
            ptr: ptr
        })
    }
}
impl<T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> private::AsConstKernelPtrBase for CudaTensor4dPtr<T,N1,N2,N3,N4> where T: Default + Debug {
    fn as_const_kernel_ptr(&self) -> *mut c_void {
        self.ptr.as_const_kernel_ptr()
    }
}
impl<T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> Deref for CudaTensor4dPtr<T,N1,N2,N3,N4> where T: Default + Debug {
    type Target = CudaMemoryPoolPtr<T>;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}
impl<T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> DerefMut for CudaTensor4dPtr<T,N1,N2,N3,N4> where T: Default + Debug {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> From<&'a CudaTensor4dPtr<T,N1,N2,N3,N4>> for &'a CudaMemoryPoolPtr<T> where T: Default + Debug {
    fn from(value: &'a CudaTensor4dPtr<T,N1,N2,N3,N4>) -> Self {
        &value.ptr
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> From<&'a mut CudaTensor4dPtr<T,N1,N2,N3,N4>> for &'a mut CudaMemoryPoolPtr<T> where T: Default + Debug {
    fn from(value: &'a mut CudaTensor4dPtr<T,N1,N2,N3,N4>) -> Self {
        &mut value.ptr
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
impl TryFrom<f32> for CudaHostPtr<f32> {
    type Error = CudaError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        let mut ptr:CudaHostPtr<f32> = CudaHostPtr::new(1,cudaHostAllocDefault)?;
        ptr.memcpy(&value as *const f32,1)?;
        Ok(ptr)
    }
}
impl TryFrom<f64> for CudaHostPtr<f64> {
    type Error = CudaError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        let mut ptr:CudaHostPtr<f64> = CudaHostPtr::new(1,cudaHostAllocDefault)?;
        ptr.memcpy(&value as *const f64,1)?;
        Ok(ptr)
    }
}
/// Trait that defines arguments passed to cuda kernel functions
pub trait KernelArgs {
    /// Returns a Vec<&mut dyn AsMutKernelPtr> of the type implementing AsMutKernelPtr,
    /// which is converted to a data type that can be passed to the cuda kernel in subsequent processing.
    fn as_vec(&mut self) ->  Vec<&mut dyn AsKernelPtr>;
}
/// Trait defining cuda kernel functions
pub trait Kernel {
    /// Object to be converted into a list of arguments to be passed to the cuda kernel function
    type Args: KernelArgs;

    /// Pointer to cuda kernel function
    const FUNC_PTR: *const c_void;

    /// cuda kernel startup function
    /// # Arguments
    /// * `grid_dim` - Number of dims in grid
    /// * `block_dim` - Number of blocks in grid
    /// * `args` - List of arguments passed to cuda kernel functions
    /// * `shared_mem` - Size (in bytes) of shared memory to allocate for use within cuda kernel functions.
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaRuntimeError`]
    fn launch(&mut self,grid_dim:dim3,block_dim:dim3,args:&mut Self::Args,shared_mem:usize) -> Result<(),CudaRuntimeError> {
        ffi::launch(Self::FUNC_PTR,
                     grid_dim,
                     block_dim,
                     &mut args.as_vec().into_iter()
                         .map(|p| p.as_kernel_ptr())
                         .collect::<Vec<*mut c_void>>().as_mut_slice(),
                     shared_mem
        )
    }
}
