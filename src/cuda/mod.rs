//! Function to wrap and handle cuda kernel

use std::fmt;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use cuda_runtime_sys::{dim3};
use libc::{c_void};
use rcudnn::Error;
use rcudnn::utils::DataType;
use rcudnn_sys::{cudaMemcpyKind, cudaStream_t, cudnnDataType_t};
use crate::arr::{Arr, IntoConverter, MakeView, MakeViewMut, SerializedVec, SliceSize};
use crate::cuda::allocator::{CudaAllocator, DeviceAlloc, DeviceAllocator, HostAlloc, HostAllocator, MemoryPoolAllocator};
use crate::cuda::private::{AsConstKernelPtrBase, AsKernelPtrBase, AsMutKernelPtrBase};
use crate::device::{DeviceGpu};
use crate::error::{CudaError, CudaRuntimeError, SizeMismatchError, TypeConvertError};
use crate::layer::{BatchDataType, BatchSize};
use crate::mem::AsRawSlice;
use crate::ope::UnitValue;

pub mod ffi;
pub mod mem;
pub mod kernel;
pub mod cudnn;
pub mod allocator;

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
pub trait AsConstKernelPtr: AsConstKernelPtrBase {
}
/// Trait defining the conversion to an mutable pointer type passed to the cuda kernel
pub trait AsMutKernelPtr: AsMutKernelPtrBase {
}
/// Trait defining the conversion to an pointer type passed to the cuda kernel
pub trait AsKernelPtr: AsKernelPtrBase {
}
impl<'a,T> AsConstKernelPtr for T where T: AsConstKernelPtrBase {}
impl<'a,T> AsMutKernelPtr for T where T: AsMutKernelPtrBase {}
impl<'a,T> AsKernelPtrBase for T where T: AsMutKernelPtr {
    fn as_kernel_ptr(&mut self) -> *mut c_void {
        self.as_mut_kernel_ptr()
    }
}
impl<'a,T> AsKernelPtr for T where T: AsKernelPtrBase {

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
/// Trait that defines the ability to get a reference to a cuda read only pointer.
pub trait AsCudaReadOnlyPtr {
    /// The type of the elements in the referenced CudaPtr
    type Pointee;
    fn as_cuda_read_only_ptr<'a>(&'a self) -> CudaPtrRef<'a,Self::Pointee>;
}
/// Trait that defines the ability to get a reference to a cuda smart pointer.
pub trait AsCudaPtr<'a> {
    /// Returned const cuda pointer reference
    type Pointer: AsConstKernelPtr;
    fn as_cuda_ptr(&'a self) -> &'a Self::Pointer;
}
pub trait DeriveCudaConstPtr {}
pub trait TryClone: Sized {
    type Error;

    fn try_clone(&self) -> Result<Self,Self::Error>;
}
pub trait CudaView<'a> {
    type Type: 'a;
}
pub trait AsCudaView<'a>: CudaView<'a> {
    fn as_cuda_view(&self) -> Self::Type;
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
impl AsPtr<i32> for i32 {
    fn as_ptr(&self) -> *const i32 {
        self as *const i32
    }
}
impl AsMutPtr<i32> for i32 {
    fn as_mut_ptr(&mut self) -> *mut i32 {
        self as *mut i32
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
impl AsPtr<u32> for u32 {
    fn as_ptr(&self) -> *const u32 {
        self as *const u32
    }
}
impl AsMutPtr<u32> for u32 {
    fn as_mut_ptr(&mut self) -> *mut u32 {
        self as *mut u32
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
impl AsPtr<i64> for i64 {
    fn as_ptr(&self) -> *const i64 {
        self as *const i64
    }
}
impl AsMutPtr<i64> for i64 {
    fn as_mut_ptr(&mut self) -> *mut i64 {
        self as *mut i64
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
impl AsPtr<u64> for u64 {
    fn as_ptr(&self) -> *const u64 {
        self as *const u64
    }
}
impl AsMutPtr<u64> for u64 {
    fn as_mut_ptr(&mut self) -> *mut u64 {
        self as *mut u64
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
impl AsPtr<usize> for usize {
    fn as_ptr(&self) -> *const usize {
        self as *const usize
    }
}
impl AsMutPtr<usize> for usize {
    fn as_mut_ptr(&mut self) -> *mut usize {
        self as *mut usize
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
impl AsPtr<f32> for f32 {
    fn as_ptr(&self) -> *const f32 {
        self as *const f32
    }
}
impl AsMutPtr<f32> for f32 {
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self as *mut f32
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
impl AsPtr<f64> for f64 {
    fn as_ptr(&self) -> *const f64 {
        self as *const f64
    }
}
impl AsMutPtr<f64> for f64 {
    fn as_mut_ptr(&mut self) -> *mut f64 {
        self as *mut f64
    }
}
impl AsMutKernelPtrBase for i32 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut i32 as *mut libc::c_void
    }
}
impl AsMutKernelPtrBase for u32 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut u32 as *mut libc::c_void
    }
}
impl AsMutKernelPtrBase for i64 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut i64 as *mut libc::c_void
    }
}
impl AsMutKernelPtrBase for u64 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut u64 as *mut libc::c_void
    }
}
impl AsMutKernelPtrBase for usize {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
       self as *mut usize as *mut libc::c_void
    }
}
impl AsMutKernelPtrBase for f32 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
        self as *mut f32 as *mut libc::c_void
    }
}
impl AsMutKernelPtrBase for f64 {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
        self as *mut f64 as *mut libc::c_void
    }
}
/// Trait that defines the type of each element of a pointer
pub trait PointerElement {
    type Element: Default + Debug;
}
/// Trait to implement cuda synchronous memory read operations
pub trait ReadMemory<T: Default + Debug>: PointerElement {
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
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
/// Trait to implement cuda synchronous memory write operations
pub trait WriteMemory<T: Default + Debug>: AsMutPtr<T> {
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
}
/// Trait to implement cuda asynchronous memory read operations
pub trait ReadMemoryAsync<T: Default + Debug> {
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
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
/// Trait to implement cuda asynchronous memory write operations
pub trait WriteMemoryAsync<T: Default + Debug>: AsMutPtr<T> {
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
}
/// Cuda memory type (host or device)
pub trait MemoryType {
    type Type;
}
/// Functionality for implementing synchronized memory write operations in CUDA between Cuda smart pointers
pub(crate) trait WriteCudaMemory<T,S,D>
    where T: Debug + Default + AsPtr<T> + AsMutPtr<T> {
    /// Memory Copy
    /// # Arguments
    /// * `src` - Pointer to source memory
    /// * `dst` - - Pointer to destination memory
    /// * `len` - Number of elements of the value to be copied
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`rcudnn::Error`]
    ///
    fn memcpy(src:*const T, dst:*mut T, len:usize) -> Result<usize,rcudnn::Error>;
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
pub struct CudaPtr<T, A: CudaAllocator = DeviceAllocator> {
    ptr:*mut T,
    size:usize,
    allocator:A
}
impl<T,A> CudaPtr<T,A>
    where T: Debug + Default,
          A: CudaAllocator {
    /// Create an instance of CudaPtr
    /// # Arguments
    /// * `size`- Number of value elements to be allocated
    /// * `allocator` - Memory Allocator object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new(size: usize, allocator: &A) -> Result<CudaPtr<T, A>, CudaError> {
        let ptr: *mut T = allocator.allocate(size)?;

        Ok(CudaPtr {
            ptr: ptr,
            size: size,
            allocator: allocator.clone()
        })
    }
}
impl<T,A> CudaPtr<T,A>
    where T: Debug + Default,
          A: CudaAllocator,
          Self: WriteMemory<T> {
    /// Create an instance of CudaPtr
    /// # Arguments
    /// * `size`- Number of value elements to be allocated
    /// * `allocator` - Memory Allocator object
    /// * `initializer` - Repeatedly called function to initialize each element
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn with_initializer<I: FnMut() -> T>(size: usize, allocator:&A, initializer: I) -> Result<CudaPtr<T,A>, CudaError> {
        let mut ptr = Self::new(size,allocator)?;

        let mut src = Vec::with_capacity(size);

        src.resize_with(size,initializer);

        ptr.memcpy(src.into_boxed_slice().as_ptr(),size)?;

        Ok(ptr)
    }
}
impl<T,A> AsConstKernelPtrBase for CudaPtr<T,A> where A: CudaAllocator {
    fn as_const_kernel_ptr(&self) -> *mut libc::c_void {
        &self.ptr as *const *mut T as *mut libc::c_void
    }
}
impl<T,A> AsMutKernelPtrBase for CudaPtr<T,A> where A: CudaAllocator  {
    fn as_mut_kernel_ptr(&mut self) -> *mut libc::c_void {
        &mut self.ptr as *mut *mut T as *mut libc::c_void
    }
}
impl<T,A> AsVoidPtr for CudaPtr<T,A> where A: CudaAllocator  {
    fn as_void_ptr(&self) -> *const libc::c_void {
        self.ptr as *const libc::c_void
    }
}
impl<T,A> AsMutVoidPtr for CudaPtr<T,A> where A: CudaAllocator  {
    fn as_mut_void_ptr(&mut self) -> *mut libc::c_void {
        self.ptr as *mut libc::c_void
    }
}
impl<T,A> AsPtr<T> for CudaPtr<T,A> where A: CudaAllocator  {
    fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }
}
impl<T,A> AsMutPtr<T> for CudaPtr<T,A> where A: CudaAllocator  {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
impl<'a,T,A> AsCudaReadOnlyPtr for CudaPtr<T,A> where T: Default + Debug + 'a, A: CudaAllocator + 'a {
    type Pointee = T;
    fn as_cuda_read_only_ptr<'b>(&'b self) -> CudaPtrRef<'b,T> {
        CudaPtrRef::from(&self.ptr)
    }
}
impl<T: Default + Debug, A: CudaAllocator> PointerElement for CudaPtr<T,A> {
    type Element = T;
}
impl<T: Default + Debug> ReadMemory<T> for CudaPtr<T,DeviceAllocator> where Self: AsPtr<T> {
    fn read_to_vec(&self) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                         self.ptr,
                         self.size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(r)
    }

    fn read_to_vec_with_size(&self,size:usize) -> Result<Vec<T>,rcudnn::Error> where Self: AsPtr<T> {
        let mut r = Vec::with_capacity(size);
        r.resize_with(size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                         self.ptr,
                         size,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(r)
    }
}
impl<T: Default + Debug> ReadMemory<T> for CudaPtr<T,HostAllocator> where Self: AsPtr<T> {
    fn read_to_vec(&self) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                    self.ptr,
                    self.size,
                    cudaMemcpyKind::cudaMemcpyHostToHost)?;
        Ok(r)
    }

    fn read_to_vec_with_size(&self,size:usize) -> Result<Vec<T>,rcudnn::Error> where Self: AsPtr<T> {
        let mut r = Vec::with_capacity(size);
        r.resize_with(size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                    self.ptr,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToHost)?;
        Ok(r)
    }
}
impl<T: Default + Debug> ReadMemory<T> for CudaPtr<T,MemoryPoolAllocator<DeviceAlloc>> where Self: AsPtr<T> {
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
impl<T: Default + Debug> ReadMemory<T> for CudaPtr<T,MemoryPoolAllocator<HostAlloc>> where Self: AsPtr<T> {
    fn read_to_vec(&self) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                    self.ptr,
                    self.size,
                    cudaMemcpyKind::cudaMemcpyHostToHost)?;
        Ok(r)
    }

    fn read_to_vec_with_size(&self,size:usize) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(size);
        r.resize_with(size,Default::default);

        ffi::memcpy(r.as_mut_ptr(),
                    self.ptr,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToHost)?;
        Ok(r)
    }
}
impl<T: Default + Debug> WriteMemory<T> for CudaPtr<T,DeviceAllocator> where Self: AsPtr<T> + AsMutVoidPtr {
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
}
impl<T: Default + Debug> WriteMemory<T> for CudaPtr<T,HostAllocator> where Self: AsPtr<T> + AsMutVoidPtr {
    fn memcpy(&mut self, p:*const T,len:usize) -> Result<usize,rcudnn::Error> {
        ffi::memcpy(self.ptr,
                    p,
                    len,
                    cudaMemcpyKind::cudaMemcpyHostToHost)?;
        Ok(len)
    }

    fn memcpy_repeat(&mut self, p: *const T, len: usize, count: usize) -> Result<usize, Error> {
        for i in 0..count {
            unsafe {
                ffi::memcpy(self.ptr.add(i * len),
                            p,
                            len,
                            cudaMemcpyKind::cudaMemcpyHostToHost)?;
            }
        }
        Ok(len * count)
    }
}
impl<T: Default + Debug> WriteMemory<T> for CudaPtr<T,MemoryPoolAllocator<DeviceAlloc>>
    where Self: AsPtr<T> + AsMutVoidPtr {
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
}
impl<T: Default + Debug> WriteMemoryAsync<T> for CudaPtr<T,HostAllocator> where Self: AsPtr<T> {
    fn memcpy_async(&mut self, p:*const T,len:usize,stream:cudaStream_t) -> Result<usize,rcudnn::Error> {
        ffi::memcpy_async(self.ptr,
                          p,
                          len,
                          cudaMemcpyKind::cudaMemcpyHostToHost,stream)?;
        Ok(len)
    }

    fn memcpy_async_repeat(&mut self, p: *const T, len: usize, count: usize, stream: cudaStream_t) -> Result<usize, Error> {
        for i in 0..count {
            unsafe {
                ffi::memcpy_async(self.ptr.add(i * len),
                                  p,
                                  len,
                                  cudaMemcpyKind::cudaMemcpyHostToHost,stream)?;
            }
        }
        Ok(len * count)

    }
}
impl<T: Default + Debug> WriteMemoryAsync<T> for CudaPtr<T,MemoryPoolAllocator<HostAlloc>> where Self: AsPtr<T> {
    fn memcpy_async(&mut self, p:*const T,len:usize,stream:cudaStream_t) -> Result<usize,rcudnn::Error> {
        ffi::memcpy_async(self.ptr,
                          p,
                          len,
                          cudaMemcpyKind::cudaMemcpyHostToHost,stream)?;
        Ok(len)
    }

    fn memcpy_async_repeat(&mut self, p: *const T, len: usize, count: usize, stream: cudaStream_t) -> Result<usize, Error> {
        for i in 0..count {
            unsafe {
                ffi::memcpy_async(self.ptr.add(i * len),
                                  p,
                                  len,
                                  cudaMemcpyKind::cudaMemcpyHostToHost,stream)?;
            }
        }
        Ok(len * count)

    }
}
impl<T: Default + Debug> MemoryMoveToAsync<T,CudaPtr<T,HostAllocator>> for CudaPtr<T,HostAllocator>
    where Self: AsPtr<T> + AsMutPtr<T> {
    fn memcpy_to_async(&self, dst: &mut CudaPtr<T,HostAllocator>, len: usize,stream:cudaStream_t) -> Result<usize, Error> {
        ffi::memcpy_async(dst.as_mut_ptr(),
                          self.ptr,
                          len,
                          cudaMemcpyKind::cudaMemcpyHostToHost,stream)?;
        Ok(len)
    }
}
impl<T: Default + Debug> MemoryMoveToAsync<T,CudaPtr<T,MemoryPoolAllocator<HostAlloc>>> for CudaPtr<T,MemoryPoolAllocator<HostAlloc>>
    where Self: AsPtr<T> + AsMutPtr<T> {
    fn memcpy_to_async(&self, dst: &mut CudaPtr<T,MemoryPoolAllocator<HostAlloc>>, len: usize,stream:cudaStream_t) -> Result<usize, Error> {
        ffi::memcpy_async(dst.as_mut_ptr(),
                          self.ptr,
                          len,
                          cudaMemcpyKind::cudaMemcpyHostToHost,stream)?;
        Ok(len)
    }
}
impl<T: Default + Debug> MemoryMoveToAsync<T,CudaPtr<T,HostAllocator>> for CudaPtr<T,MemoryPoolAllocator<HostAlloc>>
    where Self: AsPtr<T>, CudaPtr<T,MemoryPoolAllocator<DeviceAlloc>>: AsMutPtr<T> {
    fn memcpy_to_async(&self, dst: &mut CudaPtr<T,HostAllocator>, len: usize,stream:cudaStream_t) -> Result<usize, Error> {
        ffi::memcpy_async(dst.as_mut_ptr(),
                          self.ptr,
                          len,
                          cudaMemcpyKind::cudaMemcpyHostToHost,stream)?;
        Ok(len)
    }
}
impl<T: Default + Debug> MemoryMoveToAsync<T,CudaPtr<T,MemoryPoolAllocator<HostAlloc>>> for CudaPtr<T,HostAllocator>
    where Self: AsPtr<T>, CudaPtr<T,MemoryPoolAllocator<DeviceAlloc>>: AsMutPtr<T> {
    fn memcpy_to_async(&self, dst: &mut CudaPtr<T,MemoryPoolAllocator<HostAlloc>>, len: usize,stream:cudaStream_t) -> Result<usize, Error> {
        ffi::memcpy_async(dst.as_mut_ptr(),
                          self.ptr,
                          len,
                          cudaMemcpyKind::cudaMemcpyHostToHost,stream)?;
        Ok(len)
    }
}
impl<T: Default + Debug> ReadMemoryAsync<T> for CudaPtr<T,HostAllocator> {
    fn read_to_vec_async(&self, stream: cudaStream_t) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy_async(r.as_mut_ptr(),
                          self.ptr,
                          self.size,
                          cudaMemcpyKind::cudaMemcpyHostToHost,
                          stream)?;
        Ok(r)
    }

    fn read_to_vec_with_size_async(&self, stream: cudaStream_t, size:usize) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(size);
        r.resize_with(size,Default::default);

        ffi::memcpy_async(r.as_mut_ptr(),
                          self.ptr,
                          size,
                          cudaMemcpyKind::cudaMemcpyHostToHost,
                          stream)?;
        Ok(r)
    }
}
impl<T: Default + Debug> ReadMemoryAsync<T> for CudaPtr<T,MemoryPoolAllocator<HostAlloc>> {
    fn read_to_vec_async(&self, stream: cudaStream_t) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(self.size);
        r.resize_with(self.size,Default::default);

        ffi::memcpy_async(r.as_mut_ptr(),
                          self.ptr,
                          self.size,
                          cudaMemcpyKind::cudaMemcpyHostToHost,
                          stream)?;
        Ok(r)
    }

    fn read_to_vec_with_size_async(&self, stream: cudaStream_t, size:usize) -> Result<Vec<T>,rcudnn::Error> {
        let mut r = Vec::with_capacity(size);
        r.resize_with(size,Default::default);

        ffi::memcpy_async(r.as_mut_ptr(),
                          self.ptr,
                          size,
                          cudaMemcpyKind::cudaMemcpyHostToHost,
                          stream)?;
        Ok(r)
    }
}
/// Implementation of memory write processing between Cuda smart points
pub struct MemoryWriter<T,S,D> where T: Debug + Default + AsPtr<T> + AsMutPtr<T> {
    t:PhantomData<T>,
    s:PhantomData<S>,
    d:PhantomData<D>
}

impl<T> WriteCudaMemory<T,DeviceAlloc,DeviceAlloc> for MemoryWriter<T,DeviceAlloc,DeviceAlloc>
    where T: Debug + Default + AsPtr<T> + AsMutPtr<T> {
    fn memcpy(src: *const T, dst: *mut T, len: usize) -> Result<usize, Error> {
        ffi::memcpy(dst,
                    src,
                    len,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice)?;
        Ok(len)
    }
}
impl<T> WriteCudaMemory<T,HostAlloc,HostAlloc> for MemoryWriter<T,HostAlloc,HostAlloc>
    where T: Debug + Default + AsPtr<T> + AsMutPtr<T> {
    fn memcpy(src: *const T, dst: *mut T, len: usize) -> Result<usize, Error> {
        ffi::memcpy(dst,
                    src,
                    len,
                    cudaMemcpyKind::cudaMemcpyHostToHost)?;
        Ok(len)
    }
}
impl<T> WriteCudaMemory<T,DeviceAlloc,HostAlloc> for MemoryWriter<T,DeviceAlloc,HostAlloc>
    where T: Debug + Default + AsPtr<T> + AsMutPtr<T> {
    fn memcpy(src: *const T, dst: *mut T, len: usize) -> Result<usize, Error> {
        ffi::memcpy(dst,
                    src,
                    len,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost)?;
        Ok(len)
    }
}
impl<T> WriteCudaMemory<T,HostAlloc,DeviceAlloc> for MemoryWriter<T,HostAlloc,DeviceAlloc>
    where T: Debug + Default + AsPtr<T> + AsMutPtr<T> {
    fn memcpy(src: *const T, dst: *mut T, len: usize) -> Result<usize, Error> {
        ffi::memcpy(dst,
                    src,
                    len,
                    cudaMemcpyKind::cudaMemcpyHostToDevice)?;
        Ok(len)
    }
}
impl MemoryType for DeviceAllocator {
    type Type = DeviceAlloc;
}
impl MemoryType for HostAllocator {
    type Type = HostAlloc;
}
impl MemoryType for MemoryPoolAllocator<DeviceAlloc> {
    type Type = DeviceAlloc;
}
impl MemoryType for MemoryPoolAllocator<HostAlloc> {
    type Type = HostAlloc;
}
impl<T,SA,DA> MemoryMoveTo<T,CudaPtr<T,DA>> for CudaPtr<T,SA>
    where T: Debug + Default + AsPtr<T> + AsMutPtr<T>,
          SA: MemoryType + CudaAllocator,
          DA: MemoryType + CudaAllocator,
          CudaPtr<T,SA>: AsPtr<T>,
          CudaPtr<T,DA>: AsMutPtr<T>,
          MemoryWriter<T,<SA as MemoryType>::Type,<DA as MemoryType>::Type>: WriteCudaMemory<T,<SA as MemoryType>::Type,<DA as MemoryType>::Type> {
    fn memcpy_to(&self, dst: &mut CudaPtr<T,DA>, len: usize) -> Result<usize, Error> {
        MemoryWriter::<T,<SA as MemoryType>::Type,<DA as MemoryType>::Type>::memcpy(self.as_ptr(),dst.as_mut_ptr(),len)
    }
}
impl<'a,T,SA,DA> MemoryMoveTo<T,CudaMutPtr<'a,T,DA>> for CudaPtr<T,SA>
    where T: Debug + Default + AsPtr<T> + AsMutPtr<T>,
          SA: MemoryType + CudaAllocator,
          DA: MemoryType + CudaAllocator,
          CudaPtr<T,SA>: AsPtr<T>,
          CudaMutPtr<'a,T,DA>: AsMutPtr<T>,
          MemoryWriter<T,<SA as MemoryType>::Type,<DA as MemoryType>::Type>: WriteCudaMemory<T,<SA as MemoryType>::Type,<DA as MemoryType>::Type> {
    fn memcpy_to(&self, dst: &mut CudaMutPtr<'a,T,DA>, len: usize) -> Result<usize, Error> {
        MemoryWriter::<T,<SA as MemoryType>::Type,<DA as MemoryType>::Type>::memcpy(self.as_ptr(),dst.as_mut_ptr(),len)
    }
}
#[derive(Debug,Copy)]
pub struct CudaPtrRef<'a,T> {
    ptr:&'a *mut T
}
impl<'a,T> Clone for CudaPtrRef<'a,T> {
    fn clone(&self) -> Self {
        CudaPtrRef {
            ptr: self.ptr
        }
    }
}
impl<'a,T> From<&'a *mut T> for CudaPtrRef<'a,T> {
    fn from(ptr: &'a *mut T) -> Self {
        CudaPtrRef {
            ptr
        }
    }
}
impl<'a,T> AsConstKernelPtrBase for CudaPtrRef<'a,T> {
    fn as_const_kernel_ptr(&self) -> *mut libc::c_void {
        self.ptr as *const *mut T as *mut libc::c_void
    }
}
impl<'a,T> AsVoidPtr for CudaPtrRef<'a,T> {
    fn as_void_ptr(&self) -> *const libc::c_void {
        *self.ptr as *const libc::c_void
    }
}
impl<'a,T> AsPtr<T> for CudaPtrRef<'a,T> {
    fn as_ptr(&self) -> *const T {
        *self.ptr as *const T
    }
}
impl<T,A: CudaAllocator> Drop for CudaPtr<T,A> {
    fn drop(&mut self) {
        self.allocator.deallocate(self.ptr).unwrap()
    }
}
impl<T,A: CudaAllocator> Debug for CudaPtr<T,A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f,"CudaPtr {{ ptr: {:?}, size {:?}, allocator {:?} }}",self.ptr,self.size,self.allocator)
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
impl<'a,T> AsKernelPtrBase for CudaConstPtr<'a,T> where T: AsConstKernelPtr {
    fn as_kernel_ptr(&mut self) -> *mut c_void {
        self.ptr.as_const_kernel_ptr()
    }
}
/// Cuda memory object representing a 1D array with dimension number as type parameter
#[derive(Debug)]
pub struct CudaTensor1dPtr<T,A,const N:usize>
    where T: Default + Debug,
          A: CudaAllocator {
    ptr:CudaPtr<T,A>
}
impl<T,A,const N:usize> CudaTensor1dPtr<T,A,N>
    where T: Default + Debug,
          A: CudaAllocator {
    /// Create an instance of CudaTensor1dPtr
    /// # Arguments
    /// * `allocator` - Memory Allocator object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new(allocator: &A) -> Result<CudaTensor1dPtr<T, A, N>, CudaError> {
        Ok(CudaTensor1dPtr {
            ptr: CudaPtr::new(N, allocator)?
        })
    }
}

impl<T,A,const N:usize> CudaTensor1dPtr<T,A,N>
    where T: Default + Debug,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T> {
    /// Create an instance of CudaMemoryPoolPtr
    /// # Arguments
    /// * `allocator` - Memory Allocator object
    /// * `initializer` - Repeatedly called function to initialize each element
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn with_initializer<I: FnMut() -> T>(allocator:&A, initializer: I) -> Result<CudaTensor1dPtr<T,A,N>, CudaError> {
        let mut ptr = CudaPtr::new(N,allocator)?;

        let mut src = Vec::with_capacity(N);

        src.resize_with(N,initializer);

        ptr.memcpy(src.into_boxed_slice().as_ptr(),N)?;

        Ok(CudaTensor1dPtr {
            ptr: ptr
        })
    }
}
impl<T,A,const N:usize> BatchDataType for CudaTensor1dPtr<T,A,N>
    where T: Default + Debug + UnitValue<T>,
          A: CudaAllocator {
    type Type = CudaVec<T,CudaTensor1dPtr<T,A,N>,A>;
}
impl<T,A,const N:usize> AsCudaReadOnlyPtr for CudaTensor1dPtr<T,A,N>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointee = T;

    #[inline]
    fn as_cuda_read_only_ptr<'a>(&'a self) -> CudaPtrRef<'a,T> {
        self.ptr.as_cuda_read_only_ptr()
    }
}
impl<T,A,const N:usize> AsCudaPtr<'_> for CudaTensor1dPtr<T,A,N>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointer = CudaPtr<T,A>;
    fn as_cuda_ptr(&self) -> &CudaPtr<T,A> {
        &self.ptr
    }
}
impl<T,A,const N:usize> AsCudaMutPtr for CudaTensor1dPtr<T,A,N>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointee = T;
    type Allocator = A;

    #[inline]
    fn as_cuda_mut_ptr<'a>(&'a mut self) -> CudaMutPtr<'a,T,A> {
        CudaMutPtr::new(&mut self.ptr)
    }
}
impl<'a,T,A,const N:usize> TryFrom<&'a CudaTensor1dPtr<T,A,N>> for CudaTensor1dPtr<T,A,N>
    where T: Default + Debug,
          A: CudaAllocator,
          Self: AsPtr<T> + AsCudaPtr<'a> + AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'b> <Self as AsCudaPtr<'a>>::Pointer: AsPtr<T> + MemoryMoveTo<T,CudaMutPtr<'b,T,A>>,
          for<'b> CudaMutPtr<'b,T,A>: AsMutPtr<T>,
          CudaPtr<T,A>: WriteMemory<T> {
    type Error = CudaError;
    fn try_from(src: &'a CudaTensor1dPtr<T,A,N>) -> Result<Self, Self::Error> {
        let mut dst = CudaTensor1dPtr::new(&src.ptr.allocator)?;

        src.as_cuda_ptr().memcpy_to(&mut dst.as_cuda_mut_ptr(),N)?;

        Ok(dst)
    }
}
impl<T,A,const N:usize> TryClone for CudaTensor1dPtr<T,A,N>
    where T: Default + Debug,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T>,
          for<'a> Self: AsPtr<T> + AsCudaPtr<'a> + AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'a> <Self as AsCudaPtr<'a>>::Pointer: AsPtr<T> + MemoryMoveTo<T,CudaMutPtr<'a,T,A>>,
          for<'a> CudaMutPtr<'a,T,A>: AsMutPtr<T> {
    type Error = CudaError;
    fn try_clone(&self) -> Result<Self,CudaError> {
        let mut dst = CudaTensor1dPtr::new(&self.ptr.allocator)?;

        self.as_cuda_ptr().memcpy_to(&mut dst.as_cuda_mut_ptr(),N)?;

        Ok(dst)
    }
}
impl<'a,T,A,const N:usize> From<&'a CudaTensor1dPtr<T,A,N>> for &'a CudaPtr<T,A>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a CudaTensor1dPtr<T,A,N>) -> Self {
        &value.ptr
    }
}
impl<'a,T,A,const N:usize> From<&'a mut CudaTensor1dPtr<T,A,N>> for &'a mut CudaPtr<T,A>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a mut CudaTensor1dPtr<T,A,N>) -> Self {
        &mut value.ptr
    }
}
impl<'a,T,A,const N:usize> ToHost<T> for CudaTensor1dPtr<T,A,N>
    where T: Default + Debug + Clone + Send + Sync + 'static,
          A: CudaAllocator,
          CudaPtr<T,A>: ReadMemory<T> {
    type Output = Arr<T,N>;
    fn to_host(self) -> Result<Self::Output,TypeConvertError> {
        Ok(self.ptr.read_to_vec()?.try_into()?)
    }
}
impl<T,A,const N:usize> MemorySize for CudaTensor1dPtr<T,A,N>
    where T: Default + Debug,
          A: CudaAllocator {
    #[inline]
    fn size() -> usize {
        N
    }
}
impl<T,A,const N:usize> PointerElement for CudaTensor1dPtr<T,A,N>
    where T: Default + Debug,
          A: CudaAllocator {
    type Element = T;
}
impl<T,A,const N:usize> DeriveCudaConstPtr for CudaTensor1dPtr<T,A,N>
    where T: Debug + Default,
          A: CudaAllocator {}
impl<'a,T,A,const N:usize> CudaView<'a> for CudaTensor1dPtr<T,A,N>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    type Type = CudaTensor1dPtrView<'a,T,N>;
}
impl<'a,T,A,const N:usize> CudaView<'a> for &'a CudaTensor1dPtr<T,A,N>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    type Type = CudaTensor1dPtrView<'a,T,N>;
}
impl<'a,T,A,const N:usize> AsCudaView<'a> for &'a CudaTensor1dPtr<T,A,N>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    fn as_cuda_view(&self) -> Self::Type {
        (*self).into()
    }
}
/// View into a Cuda memory object representing a 1D array with dimension number as a type parameter
#[derive(Debug)]
pub struct CudaTensor1dPtrView<'a,T,const N:usize>
    where T: Default + Debug {
    ptr:CudaPtrRef<'a,T>
}
impl<'a,T,A,const N:usize> From<&'a CudaTensor1dPtr<T,A,N>> for CudaTensor1dPtrView<'a,T,N>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a CudaTensor1dPtr<T,A,N>) -> Self {
        CudaTensor1dPtrView {
            ptr:value.as_cuda_read_only_ptr()
        }
    }
}
impl<'a,T,const N:usize> From<&'a CudaTensor1dPtrView<'a,T,N>> for CudaTensor1dPtrView<'a,T,N>
    where T: Default + Debug {
    fn from(value: &'a CudaTensor1dPtrView<'a,T,N>) -> Self {
        CudaTensor1dPtrView {
            ptr:value.ptr.clone()
        }
    }
}
impl<'a,T,const N:usize> PointerElement for CudaTensor1dPtrView<'a,T,N> where T: Default + Debug {
    type Element = T;
}
impl<'a,T,const N:usize> AsCudaReadOnlyPtr for CudaTensor1dPtrView<'a,T,N> where T: Default + Debug + 'a {
    type Pointee = T;

    #[inline]
    fn as_cuda_read_only_ptr<'b>(&'b self) -> CudaPtrRef<'b,T> {
        self.ptr.clone()
    }
}
impl<'a,T,const N:usize> MemorySize for CudaTensor1dPtrView<'a,T,N>
    where T: Default + Debug {
    #[inline]
    fn size() -> usize {
        N
    }
}
impl<'a,T,const N:usize> DeriveCudaConstPtr for CudaTensor1dPtrView<'a,T,N> where T: Debug + Default {}
/// Cuda memory object representing a 2D array with dimension number as type parameter
#[derive(Debug)]
pub struct CudaTensor2dPtr<T,A,const N1:usize,const N2:usize>
    where T: Default + Debug,
          A: CudaAllocator {
    ptr:CudaPtr<T,A>
}
impl<T,A,const N1:usize,const N2:usize> CudaTensor2dPtr<T,A,N1,N2>
    where T: Default + Debug,
          A: CudaAllocator {
    /// Create an instance of CudaTensor1dPtr
    /// # Arguments
    /// * `allocator` - Memory Allocator object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new(allocator: &A) -> Result<CudaTensor2dPtr<T, A, N1, N2>, CudaError> {
        Ok(CudaTensor2dPtr {
            ptr: CudaPtr::new(N1 * N2, allocator)?
        })
    }
}
impl<T,A,const N1:usize,const N2:usize> CudaTensor2dPtr<T,A,N1,N2>
    where T: Default + Debug,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T> {
    /// Create an instance of CudaMemoryPoolPtr
    /// # Arguments
    /// * `allocator` - Memory Allocator object
    /// * `initializer` - Repeatedly called function to initialize each element
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn with_initializer<I: FnMut() -> T>(allocator:&A, initializer: I) -> Result<CudaTensor2dPtr<T,A,N1,N2>, CudaError> {
        let mut ptr = CudaPtr::new(N1*N2,allocator)?;

        let mut src = Vec::with_capacity(N1*N2);

        src.resize_with(N1*N2,initializer);

        ptr.memcpy(src.into_boxed_slice().as_ptr(),N1*N2)?;

        Ok(CudaTensor2dPtr {
            ptr: ptr
        })
    }
}
impl<T,A,const N1:usize,const N2:usize> BatchDataType for CudaTensor2dPtr<T,A,N1,N2>
    where T: Default + Debug + UnitValue<T>,
          A: CudaAllocator {
    type Type = CudaVec<T,CudaTensor2dPtr<T,A,N1,N2>,A>;
}
impl<T,A,const N1:usize,const N2:usize> PointerElement for CudaTensor2dPtr<T,A,N1,N2>
    where T: Default + Debug,
          A: CudaAllocator {
    type Element = T;
}
impl<T,A,const N1:usize,const N2:usize> AsCudaReadOnlyPtr for CudaTensor2dPtr<T,A,N1,N2>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointee = T;
    #[inline]
    fn as_cuda_read_only_ptr<'a>(&'a self) -> CudaPtrRef<'a,T> {
        self.ptr.as_cuda_read_only_ptr()
    }
}
impl<T,A,const N1:usize,const N2:usize> AsCudaPtr<'_> for CudaTensor2dPtr<T,A,N1,N2>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointer = CudaPtr<T,A>;

    fn as_cuda_ptr(&self) -> &CudaPtr<T, A> {
        &self.ptr
    }
}
impl<T,A,const N1:usize,const N2:usize> AsCudaMutPtr for CudaTensor2dPtr<T,A,N1,N2>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointee = T;
    type Allocator = A;

    #[inline]
    fn as_cuda_mut_ptr<'a>(&'a mut self) -> CudaMutPtr<'a,T,A> {
        CudaMutPtr::new(&mut self.ptr)
    }
}
impl<T,A,const N1:usize,const N2:usize> TryClone for CudaTensor2dPtr<T,A,N1,N2>
    where T: Default + Debug,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T>,
          for<'a> CudaMutPtr<'a,T,A>: AsMutPtr<T>,
          for<'a> Self: AsPtr<T> + AsCudaPtr<'a> + AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'a> <Self as AsCudaPtr<'a>>::Pointer: AsPtr<T> + MemoryMoveTo<T,CudaMutPtr<'a,T,A>> {
    type Error = CudaError;
    fn try_clone(&self) -> Result<Self,CudaError> {
        let mut dst = CudaTensor2dPtr::new(&self.ptr.allocator)?;

        self.as_cuda_ptr().memcpy_to(&mut dst.as_cuda_mut_ptr(),N1*N2)?;

        Ok(dst)
    }
}
impl<'a,T,A,const N1:usize,const N2:usize> From<&'a CudaTensor2dPtr<T,A,N1,N2>> for &'a CudaPtr<T,A>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a CudaTensor2dPtr<T,A,N1,N2>) -> Self {
        &value.ptr
    }
}
impl<'a,T,A,const N1:usize,const N2:usize> From<&'a mut CudaTensor2dPtr<T,A,N1,N2>> for &'a mut CudaPtr<T,A>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a mut CudaTensor2dPtr<T,A,N1,N2>) -> Self {
        &mut value.ptr
    }
}
impl<T,A,const N1:usize,const N2:usize> MemorySize for CudaTensor2dPtr<T,A,N1,N2>
    where T: Default + Debug,
          A: CudaAllocator {
    #[inline]
    fn size() -> usize {
        N1 * N2
    }
}
impl<T,A,const N1:usize,const N2:usize> DeriveCudaConstPtr for CudaTensor2dPtr<T,A,N1,N2>
    where T: Debug + Default,
          A: CudaAllocator {}
impl<'a,T,A,const N1:usize,const N2:usize> CudaView<'a> for CudaTensor2dPtr<T,A,N1,N2>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    type Type = CudaTensor2dPtrView<'a,T,N1,N2>;
}
impl<'a,T,A,const N1:usize,const N2:usize> CudaView<'a> for &'a CudaTensor2dPtr<T,A,N1,N2>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    type Type = CudaTensor2dPtrView<'a,T,N1,N2>;
}
impl<'a,T,A,const N1:usize,const N2:usize> AsCudaView<'a> for &'a CudaTensor2dPtr<T,A,N1,N2>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    fn as_cuda_view(&self) -> Self::Type {
        (*self).into()
    }
}
/// View into a Cuda memory object representing a 2D array with dimension number as a type parameter
#[derive(Debug)]
pub struct CudaTensor2dPtrView<'a,T,const N1:usize,const N2:usize>
    where T: Default + Debug {
    ptr:CudaPtrRef<'a,T>
}
impl<'a,T,const N1:usize,const N2:usize> PointerElement for CudaTensor2dPtrView<'a,T,N1,N2> where T: Default + Debug {
    type Element = T;
}
impl<'a,T,const N1:usize,const N2:usize> AsCudaReadOnlyPtr for CudaTensor2dPtrView<'a,T,N1,N2> where T: Default + Debug + 'a {
    type Pointee = T;

    #[inline]
    fn as_cuda_read_only_ptr<'b>(&'b self) -> CudaPtrRef<'b,T> {
        self.ptr.clone()
    }
}
impl<'a,T,A,const N1:usize,const N2:usize> From<&'a CudaTensor2dPtr<T,A,N1,N2>> for CudaTensor2dPtrView<'a,T,N1,N2>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a CudaTensor2dPtr<T,A,N1,N2>) -> Self {
        CudaTensor2dPtrView {
            ptr: value.as_cuda_read_only_ptr()
        }
    }
}
impl<'a,T,const N1:usize,const N2:usize> From<&'a CudaTensor2dPtrView<'a,T,N1,N2>> for CudaTensor2dPtrView<'a,T,N1,N2>
    where T: Default + Debug {
    fn from(value: &'a CudaTensor2dPtrView<'a,T,N1,N2>) -> Self {
        CudaTensor2dPtrView {
            ptr:value.ptr.clone()
        }
    }
}
impl<'a,T,const N1:usize,const N2:usize> MemorySize for CudaTensor2dPtrView<'a,T,N1,N2>
    where T: Default + Debug {
    #[inline]
    fn size() -> usize {
        N1 * N2
    }
}
impl<'a,T,const N1:usize,const N2:usize> DeriveCudaConstPtr for CudaTensor2dPtrView<'a,T,N1,N2> where T: Debug + Default {}
/// Cuda memory object representing a 3D array with dimension number as type parameter
#[derive(Debug)]
pub struct CudaTensor3dPtr<T,A,const N1:usize,const N2:usize,const N3:usize>
    where T: Default + Debug,
          A: CudaAllocator {
    ptr:CudaPtr<T,A>
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize> CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Default + Debug,
          A: CudaAllocator {
    /// Create an instance of CudaTensor1dPtr
    /// # Arguments
    /// * `allocator` - Memory Allocator object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new(allocator: &A) -> Result<CudaTensor3dPtr<T, A, N1, N2, N3>, CudaError> {
        Ok(CudaTensor3dPtr {
            ptr: CudaPtr::new(N1 * N2 * N3, allocator)?
        })
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize> CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Default + Debug,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T> {
    /// Create an instance of CudaMemoryPoolPtr
    /// # Arguments
    /// * `allocator` - Memory Allocator object
    /// * `initializer` - Repeatedly called function to initialize each element
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn with_initializer<I: FnMut() -> T>(allocator:&A, initializer: I) -> Result<CudaTensor3dPtr<T,A,N1,N2,N3>, CudaError> {
        let mut ptr = CudaPtr::new(N1*N2*N3,allocator)?;

        let mut src = Vec::with_capacity(N1*N2*N3);

        src.resize_with(N1*N2*N3,initializer);

        ptr.memcpy(src.into_boxed_slice().as_ptr(),N1*N2*N3)?;

        Ok(CudaTensor3dPtr {
            ptr: ptr
        })
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize> BatchDataType for CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Default + Debug + UnitValue<T> ,
          A: CudaAllocator {
    type Type = CudaVec<T,CudaTensor3dPtr<T,A,N1,N2,N3>,A>;
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize> PointerElement for CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Default + Debug,
          A: CudaAllocator {
    type Element = T;
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize> AsCudaReadOnlyPtr for CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointee = T;

    #[inline]
    fn as_cuda_read_only_ptr<'a>(&'a self) -> CudaPtrRef<'a,T> {
        self.ptr.as_cuda_read_only_ptr()
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize> AsCudaPtr<'_> for CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointer = CudaPtr<T,A>;

    fn as_cuda_ptr(&self) -> &CudaPtr<T, A> {
        &self.ptr
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize> AsCudaMutPtr for CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointee = T;
    type Allocator = A;

    #[inline]
    fn as_cuda_mut_ptr<'a>(&'a mut self) -> CudaMutPtr<'a,T,A> {
        CudaMutPtr::new(&mut self.ptr)
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize> TryClone for CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Default + Debug,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T>,
          for<'a> Self: AsPtr<T> + AsCudaPtr<'a> + AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'a> <Self as AsCudaPtr<'a>>::Pointer: AsPtr<T> + MemoryMoveTo<T,CudaMutPtr<'a,T,A>>,
          for<'a> CudaMutPtr<'a,T,A>: AsMutPtr<T> {
    type Error = CudaError;
    fn try_clone(&self) -> Result<Self,CudaError> {
        let mut dst = CudaTensor3dPtr::new(&self.ptr.allocator)?;

        self.as_cuda_ptr().memcpy_to(&mut dst.as_cuda_mut_ptr(),N1*N2*N3)?;

        Ok(dst)
    }
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize> From<&'a CudaTensor3dPtr<T,A,N1,N2,N3>> for &'a CudaPtr<T,A>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a CudaTensor3dPtr<T,A,N1,N2,N3>) -> Self {
        &value.ptr
    }
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize> From<&'a mut CudaTensor3dPtr<T,A,N1,N2,N3>> for &'a mut CudaPtr<T,A>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a mut CudaTensor3dPtr<T,A,N1,N2,N3>) -> Self {
        &mut value.ptr
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize> MemorySize for CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Default + Debug,
          A: CudaAllocator {
    #[inline]
    fn size() -> usize {
        N1 * N2 * N3
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize> DeriveCudaConstPtr for CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Debug + Default,
          A: CudaAllocator {}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize> CudaView<'a> for CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    type Type = CudaTensor3dPtrView<'a,T,N1,N2,N3>;
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize> CudaView<'a> for &'a CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    type Type = CudaTensor3dPtrView<'a,T,N1,N2,N3>;
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize> AsCudaView<'a> for &'a CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    fn as_cuda_view(&self) -> Self::Type {
        (*self).into()
    }
}
/// View into a Cuda memory object representing a 3D array with dimension number as a type parameter
#[derive(Debug)]
pub struct CudaTensor3dPtrView<'a,T,const N1:usize,const N2:usize,const N3:usize>
    where T: Default + Debug {
    ptr: CudaPtrRef<'a,T>
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> PointerElement for CudaTensor3dPtrView<'a,T,N1,N2,N3>
    where T: Default + Debug {
    type Element = T;
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> AsCudaReadOnlyPtr for CudaTensor3dPtrView<'a,T,N1,N2,N3>
    where T: Default + Debug + 'a {
    type Pointee = T;

    #[inline]
    fn as_cuda_read_only_ptr<'b>(&'b self) -> CudaPtrRef<'b,T> {
        self.ptr.clone()
    }
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize> From<&'a CudaTensor3dPtr<T,A,N1,N2,N3>> for CudaTensor3dPtrView<'a,T,N1,N2,N3>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a CudaTensor3dPtr<T,A,N1,N2,N3>) -> Self {
        CudaTensor3dPtrView {
            ptr: value.as_cuda_read_only_ptr()
        }
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> From<&'a CudaTensor3dPtrView<'a,T,N1,N2,N3>> for CudaTensor3dPtrView<'a,T,N1,N2,N3>
    where T: Default + Debug {
    fn from(value: &'a CudaTensor3dPtrView<'a,T,N1,N2,N3>) -> Self {
        CudaTensor3dPtrView {
            ptr: value.ptr.clone()
        }
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> MemorySize for CudaTensor3dPtrView<'a,T,N1,N2,N3>
    where T: Default + Debug {
    #[inline]
    fn size() -> usize {
        N1 * N2 * N3
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize> DeriveCudaConstPtr for CudaTensor3dPtrView<'a,T,N1,N2,N3>
    where T: Debug + Default {}
/// Cuda memory object representing a 4D array with dimension number as type parameter
#[derive(Debug)]
pub struct CudaTensor4dPtr<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize>
    where T: Default + Debug,
          A: CudaAllocator {
    ptr:CudaPtr<T,A>
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Default + Debug,
          A: CudaAllocator {
    /// Create an instance of CudaTensor1dPtr
    /// # Arguments
    /// * `allocator` - Memory Allocator object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new(allocator: &A) -> Result<CudaTensor4dPtr<T, A, N1, N2, N3, N4>, CudaError> {
        Ok(CudaTensor4dPtr {
            ptr: CudaPtr::new(N1 * N2 * N3 * N4, allocator)?
        })
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Default + Debug,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T> {
    /// Create an instance of CudaMemoryPoolPtr
    /// # Arguments
    /// * `allocator` - Memory Allocator object
    /// * `initializer` - Repeatedly called function to initialize each element
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn with_initializer<I: FnMut() -> T>(allocator:&A, initializer: I) -> Result<CudaTensor4dPtr<T,A,N1,N2,N3,N4>, CudaError> {
        let mut ptr = CudaPtr::new(N1*N2*N3*N4,allocator)?;

        let mut src = Vec::with_capacity(N1*N2*N3*N4);

        src.resize_with(N1*N2*N3*N4,initializer);

        ptr.memcpy(src.into_boxed_slice().as_ptr(),N1*N2*N3*N4)?;

        Ok(CudaTensor4dPtr {
            ptr: ptr
        })
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> BatchDataType for CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Default + Debug + UnitValue<T>,
          A: CudaAllocator {
    type Type = CudaVec<T,CudaTensor4dPtr<T,A,N1,N2,N3,N4>,A>;
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> MemorySize for CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Default + Debug,
          A: CudaAllocator {
    #[inline]
    fn size() -> usize {
        N1 * N2 * N3 * N4
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> PointerElement for CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Default + Debug,
          A: CudaAllocator {
    type Element = T;
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> AsCudaReadOnlyPtr for CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointee = T;

    #[inline]
    fn as_cuda_read_only_ptr<'a>(&'a self) -> CudaPtrRef<'a,T> {
        self.ptr.as_cuda_read_only_ptr()
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> AsCudaPtr<'_> for CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointer = CudaPtr<T,A>;

    fn as_cuda_ptr(&self) -> &CudaPtr<T, A> {
        &self.ptr
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> AsCudaMutPtr for CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointee = T;
    type Allocator = A;

    #[inline]
    fn as_cuda_mut_ptr<'a>(&'a mut self) -> CudaMutPtr<'a,T,A> {
        CudaMutPtr::new(&mut self.ptr)
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> TryClone for CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Default + Debug,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T>,
          for<'a> Self: AsPtr<T> + AsCudaPtr<'a> + AsCudaMutPtr<Pointee=T,Allocator=A>,
          for<'a> <Self as AsCudaPtr<'a>>::Pointer: AsPtr<T> + MemoryMoveTo<T,CudaMutPtr<'a,T,A>>,
          for<'a> CudaMutPtr<'a,T,A>: AsMutPtr<T> {
    type Error = CudaError;
    fn try_clone(&self) -> Result<Self,CudaError> {
        let mut dst = CudaTensor4dPtr::new(&self.ptr.allocator)?;

        self.as_cuda_ptr().memcpy_to(&mut dst.as_cuda_mut_ptr(),N1*N2*N3*N4)?;

        Ok(dst)
    }
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> From<&'a CudaTensor4dPtr<T,A,N1,N2,N3,N4>>
    for &'a CudaPtr<T,A> where T: Default + Debug,
                               A: CudaAllocator {
    fn from(value: &'a CudaTensor4dPtr<T,A,N1,N2,N3,N4>) -> Self {
        &value.ptr
    }
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> From<&'a mut CudaTensor4dPtr<T,A,N1,N2,N3,N4>>
    for &'a mut CudaPtr<T,A> where T: Default + Debug,
                                   A: CudaAllocator {
    fn from(value: &'a mut CudaTensor4dPtr<T,A,N1,N2,N3,N4>) -> Self {
        &mut value.ptr
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> DeriveCudaConstPtr for CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Debug + Default,
          A: CudaAllocator {}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> CudaView<'a> for CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    type Type = CudaTensor4dPtrView<'a,T,N1,N2,N3,N4>;
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> CudaView<'a> for &'a CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    type Type = CudaTensor4dPtrView<'a,T,N1,N2,N3,N4>;
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> AsCudaView<'a> for &'a CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Debug + Default + 'a,
          A: CudaAllocator {
    fn as_cuda_view(&self) -> Self::Type {
        (*self).into()
    }
}
/// View into a Cuda memory object representing a 4D array with dimension number as a type parameter
#[derive(Debug)]
pub struct CudaTensor4dPtrView<'a,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize>
    where T: Default + Debug {
    ptr:CudaPtrRef<'a,T>
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> PointerElement for CudaTensor4dPtrView<'a,T,N1,N2,N3,N4>
    where T: Default + Debug{
    type Element = T;
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> AsCudaReadOnlyPtr for CudaTensor4dPtrView<'a,T,N1,N2,N3,N4>
    where T: Default + Debug + 'a {
    type Pointee = T;

    #[inline]
    fn as_cuda_read_only_ptr<'b>(&'b self) -> CudaPtrRef<'b,T> {
        self.ptr.clone()
    }
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> From<&'a CudaTensor4dPtr<T,A,N1,N2,N3,N4>>
    for CudaTensor4dPtrView<'a,T,N1,N2,N3,N4>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a CudaTensor4dPtr<T,A,N1,N2,N3,N4>) -> Self {
        CudaTensor4dPtrView {
            ptr:value.ptr.as_cuda_read_only_ptr()
        }
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> From<&'a CudaTensor4dPtrView<'a,T,N1,N2,N3,N4>>
    for CudaTensor4dPtrView<'a,T,N1,N2,N3,N4>
    where T: Default + Debug {
    fn from(value: &'a CudaTensor4dPtrView<'a,T,N1,N2,N3,N4>) -> Self {
        CudaTensor4dPtrView {
            ptr:value.as_cuda_read_only_ptr().clone()
        }
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> MemorySize for CudaTensor4dPtrView<'a,T,N1,N2,N3,N4>
    where T: Default + Debug {
    #[inline]
    fn size() -> usize {
        N1 * N2 * N3 * N4
    }
}
impl<'a,T,const N1:usize,const N2:usize,const N3:usize,const N4:usize> DeriveCudaConstPtr for CudaTensor4dPtrView<'a,T,N1,N2,N3,N4>
    where T: Debug + Default {}
/// Trait that returns the size of Cuda smart point type memory (returns the number of elements)
pub trait MemorySize {
    fn size() -> usize;
}
#[derive(Debug)]
pub struct CudaVec<U,T,A>
    where U: UnitValue<U>,
          A: CudaAllocator {
    len: usize,
    ptr:CudaPtr<U,A>,
    t:PhantomData<T>
}
impl<U,T,A> CudaVec<U,T,A>
    where U: UnitValue<U>,
          A: CudaAllocator + Debug,
          for<'a> T: AsConstKernelPtr + AsKernelPtr + MemorySize {
    /// Create an instance of CudaVec
    /// # Arguments
    /// * `size`- Number of value elements to be allocated
    /// * `allocator` - Memory Allocator object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn new(size: usize, allocator: &A) -> Result<CudaVec<U, T, A>, CudaError> {
        let ptr = CudaPtr::new(size * T::size(), allocator)?;

        Ok(CudaVec {
            len: size,
            ptr,
            t: PhantomData::<T>
        })
    }
}
impl<U,T,A> CudaVec<U,T,A>
    where U: UnitValue<U>,
          A: CudaAllocator + Debug,
          CudaPtr<U,A>: WriteMemory<U>,
          for<'a> T: AsConstKernelPtr + AsKernelPtr + MemorySize {
    /// Create an instance of CudaVec
    /// # Arguments
    /// * `size`- Number of value elements to be allocated
    /// * `allocator` - Memory Allocator object
    /// * `initializer` - Repeatedly called function to initialize each element
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaError`]
    pub fn with_initializer<I: FnMut() -> U>(size: usize, allocator:&A, initializer: I) -> Result<CudaVec<U,T,A>, CudaError> {
        let mut ptr = CudaPtr::new(size * T::size(),allocator)?;

        let mut src = Vec::with_capacity(size * T::size());

        src.resize_with(size * T::size(),initializer);

        ptr.memcpy(src.into_boxed_slice().as_ptr(),size * T::size())?;

        Ok(CudaVec {
            len:size,
            ptr,
            t:PhantomData::<T>
        })
    }
}
impl<U,T,A> BatchSize for CudaVec<U,T,A>
    where U: UnitValue<U>,
          A: CudaAllocator,
          for<'a> T: AsConstKernelPtr + AsKernelPtr + MemorySize {
    fn size(&self) -> usize {
        self.len
    }
}
impl<U,T,A> PointerElement for CudaVec<U,T,A>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          for<'a> T: AsConstKernelPtr + AsKernelPtr + MemorySize {
    type Element = U;
}
impl<U,T,A> AsCudaReadOnlyPtr for CudaVec<U,T,A>
    where U: UnitValue<U>,
          A: CudaAllocator {
    type Pointee = U;

    #[inline]
    fn as_cuda_read_only_ptr<'a>(&'a self) -> CudaPtrRef<'a,U> {
        CudaPtrRef::from(self.ptr.as_cuda_read_only_ptr())
    }
}
impl<U,T,A> AsCudaMutPtr for CudaVec<U,T,A>
    where U: UnitValue<U>,
          T: AsConstKernelPtr + AsKernelPtr + Debug,
          A: CudaAllocator {
    type Pointee = U;
    type Allocator = A;

    #[inline]
    fn as_cuda_mut_ptr<'a>(&'a mut self) -> CudaMutPtr<'a,U,A> {
        CudaMutPtr::new(&mut self.ptr)
    }
}
impl<U,T,A> TryClone for CudaVec<U,T,A>
    where U: Default + Debug + UnitValue<U>,
          A: CudaAllocator,
          T: Debug + AsConstKernelPtr + AsKernelPtr + MemorySize,
          CudaPtr<U,A>: WriteMemory<U>,
          for<'a> Self: AsPtr<U> + AsCudaPtr<'a> + AsCudaMutPtr<Pointee=U,Allocator=A>,
          for<'a> <Self as AsCudaPtr<'a>>::Pointer: AsPtr<U> + MemoryMoveTo<U,CudaMutPtr<'a,U,A>>,
          for<'a> CudaMutPtr<'a,U,A>: AsMutPtr<U> {
    type Error = CudaError;
    fn try_clone(&self) -> Result<Self,CudaError> {
        let mut dst = CudaVec::<U,T,A>::new(self.len * T::size(),&self.ptr.allocator)?;

        self.as_cuda_ptr().memcpy_to(&mut dst.as_cuda_mut_ptr(),self.len * T::size())?;

        Ok(dst)
    }
}
impl<'a,U,T,A> ToCuda<U,A> for &'a CudaVec<U,T,A>
    where U: UnitValue<U> + 'a,
          T: AsConstKernelPtr + AsKernelPtr + MemorySize +
             CudaView<'a> + 'a,
          A: CudaAllocator + 'a,
          CudaPtr<U,A>: WriteMemory<U>,
          CudaVecView<'a,U,<T as CudaView<'a>>::Type>: TryFrom<&'a CudaVec<U,T,A>,Error=TypeConvertError>,
          <T as CudaView<'a>>::Type: AsConstKernelPtr + AsKernelPtr + MemorySize + 'a {
    type Output = CudaVecView<'a,U,<T as CudaView<'a>>::Type>;

    fn to_cuda(self, _: &DeviceGpu<U,A>) -> Result<Self::Output, TypeConvertError> {
        Ok(self.try_into()?)
    }
}
impl<U,T,A> ToCuda<U,A> for CudaVec<U,T,A>
    where U: UnitValue<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: WriteMemory<U>,
          for<'a> T: AsConstKernelPtr + AsKernelPtr + MemorySize {
    type Output = CudaVec<U,T,A>;

    fn to_cuda(self, _: &DeviceGpu<U,A>) -> Result<Self::Output, TypeConvertError> {
        Ok(self)
    }
}
impl<'a,U,T,A> AsCudaPtr<'a> for CudaVec<U,T,A>
    where U: UnitValue<U>,
          A: CudaAllocator + Debug,
          T: AsConstKernelPtr + AsKernelPtr + MemorySize {
    type Pointer = CudaPtr<U,A>;

    fn as_cuda_ptr(&'a self) -> &'a Self::Pointer {
        &self.ptr
    } 
}
impl<U,T,A> DeriveCudaConstPtr for CudaVec<U,T,A> where U: UnitValue<U>, T: Debug, A: CudaAllocator {}
impl<'a,U,T,A> CudaView<'a> for CudaVec<U,T,A>
    where U: UnitValue<U>,
          T: Debug + CudaView<'a>,
          A: CudaAllocator,
          <T as CudaView<'a>>::Type: AsCudaReadOnlyPtr + DeriveCudaConstPtr {
    type Type = CudaVecView<'a,U,<T as CudaView<'a>>::Type>;
}
impl<'a,U,T,A> CudaView<'a> for &'a CudaVec<U,T,A>
    where U: UnitValue<U>,
          T: Debug + CudaView<'a>,
          A: CudaAllocator,
          <T as CudaView<'a>>::Type: AsCudaReadOnlyPtr + DeriveCudaConstPtr {
    type Type = CudaVecView<'a,U,<T as CudaView<'a>>::Type>;
}
impl<'a,U,T,A> AsCudaView<'a> for &'a CudaVec<U,T,A>
    where U: UnitValue<U> + 'a,
          T: Debug + CudaView<'a> + 'a,
          A: CudaAllocator,
          <T as CudaView<'a>>::Type: AsCudaReadOnlyPtr + DeriveCudaConstPtr {
    fn as_cuda_view(&self) -> Self::Type {
        CudaVecView {
            len: self.len,
            ptr: self.as_cuda_read_only_ptr(),
            t:PhantomData::<<T as CudaView<'a>>::Type>
        }
    }
}
#[derive(Debug)]
pub struct CudaVecView<'a,U,T>
    where U: UnitValue<U>,
          T: AsConstKernelPtrBase + AsConstKernelPtr {
    len: usize,
    ptr:CudaPtrRef<'a,U>,
    t:PhantomData<T>
}
impl<'a,U,T> BatchSize for CudaVecView<'a,U,T>
    where U: UnitValue<U>,
          T: AsConstKernelPtr + MemorySize {
    fn size(&self) -> usize {
        self.len
    }
}
impl<'a,U,T> PointerElement for CudaVecView<'a,U,T>
    where U: UnitValue<U>,
          T: AsConstKernelPtr + AsKernelPtr {
    type Element = U;
}
impl<'a,U,T> AsCudaReadOnlyPtr for CudaVecView<'a,U,T>
    where U: UnitValue<U> + 'a,
          T: AsConstKernelPtr + MemorySize {
    type Pointee = U;

    #[inline]
    fn as_cuda_read_only_ptr<'b>(&'b self) -> CudaPtrRef<'b,U> {
        self.ptr.clone()
    }
}
impl<'a,U,T,R,A> TryFrom<&'a CudaVec<U,T,A>> for CudaVecView<'a,U,R>
    where U: UnitValue<U> + Default + Clone + Send,
          A: CudaAllocator,
          T: MemorySize + AsKernelPtr + AsConstKernelPtr + CudaView<'a>,
          R: MemorySize + AsConstKernelPtr + From<<T as CudaView<'a>>::Type> {
    type Error = TypeConvertError;

    fn try_from(value: &'a CudaVec<U,T,A>) -> Result<Self, Self::Error> {
        if T::size() != R::size() {
            Err(TypeConvertError::SizeMismatchError(SizeMismatchError(T::size(),R::size())))
        } else {
            Ok(CudaVecView {
                len:value.size(),
                ptr: value.as_cuda_read_only_ptr(),
                t:PhantomData::<R>
            })
        }
    }
}
pub struct CudaVecViewConverter<'a,U,T>
    where U: UnitValue<U> + Default + Clone + Send,
          T: MemorySize + AsConstKernelPtr {
    len:usize,
    ptr:CudaPtrRef<'a,U>,
    t:PhantomData<T>
}
impl<'a,U,T> IntoConverter for CudaVecView<'a,U,T>
    where U: UnitValue<U> + Default + Clone + Send,
          T: MemorySize + AsConstKernelPtr {
    type Converter = CudaVecViewConverter<'a,U,T>;

    fn into_converter(self) -> Self::Converter {
        CudaVecViewConverter {
            len:self.len,
            ptr:self.ptr,
            t:PhantomData::<T>
        }
    }
}
impl<'a,U,T,R> TryFrom<CudaVecViewConverter<'a,U,T>> for CudaVecView<'a,U,R>
    where U: UnitValue<U> + Default + Clone + Send,
          T: MemorySize + AsConstKernelPtr,
          R: MemorySize + AsConstKernelPtr + From<T> {
    type Error = TypeConvertError;

    #[inline]
    fn try_from(value: CudaVecViewConverter<'a,U,T>) -> Result<Self, Self::Error> {
        if T::size() != R::size() {
            Err(TypeConvertError::SizeMismatchError(SizeMismatchError(T::size(),R::size())))
        } else {
            let len = value.len;

            Ok(CudaVecView {
                len:len,
                ptr: value.ptr,
                t:PhantomData::<R>
            })
        }
    }
}
impl<'a,U,T> DeriveCudaConstPtr for CudaVecView<'a,U,T>
    where U: UnitValue<U>,
          T: Debug + AsConstKernelPtrBase + AsConstKernelPtr +
             AsCudaReadOnlyPtr + DeriveCudaConstPtr {}
pub struct CudaVecConverter<U,T,A>
    where U: UnitValue<U> + Debug + Default + Clone + Send,
          A: CudaAllocator,
          for<'a> T: AsConstKernelPtr + AsKernelPtr + MemorySize {
    len:usize,
    ptr:CudaPtr<U,A>,
    u:PhantomData<U>,
    t:PhantomData<T>
}
impl<U,T,A> IntoConverter for CudaVec<U,T,A>
    where U: UnitValue<U> + Debug + Default + Clone + Send,
          A: CudaAllocator,
          for<'a> T: AsConstKernelPtr + AsKernelPtr + MemorySize {
    type Converter = CudaVecConverter<U,T,A>;

    fn into_converter(self) -> Self::Converter {
        CudaVecConverter {
            len:self.len,
            ptr:self.ptr,
            u:PhantomData::<U>,
            t:PhantomData::<T>
        }
    }
}
impl<U,T,A> BatchSize for CudaVecConverter<U,T,A>
    where U: UnitValue<U> + Default + Clone + Send,
          A: CudaAllocator,
          for<'a> T: AsConstKernelPtr + AsKernelPtr + MemorySize {
    fn size(&self) -> usize {
        self.len
    }
}
impl<U,T,A> From<CudaVecConverter<U,T,A>> for CudaPtr<U,A>
    where U: UnitValue<U> + Default + Clone + Send,
          A: CudaAllocator,
          for<'a> T: AsConstKernelPtr + AsKernelPtr + MemorySize {
    fn from(value: CudaVecConverter<U,T,A>) -> Self {
        value.ptr
    }
}
impl<U,T,R,A> TryFrom<CudaVecConverter<U,T,A>> for CudaVec<U,R,A>
    where U: UnitValue<U> + Default + Clone + Send,
          A: CudaAllocator,
          for<'a> T: MemorySize + AsKernelPtr + AsConstKernelPtr,
          for<'a> R: MemorySize + AsKernelPtr + AsConstKernelPtr + From<T> {
    type Error = TypeConvertError;

    #[inline]
    fn try_from(value: CudaVecConverter<U,T,A>) -> Result<Self, Self::Error> {
        if T::size() != R::size() {
            Err(TypeConvertError::SizeMismatchError(SizeMismatchError(T::size(),R::size())))
        } else {
            let len = value.size();

            Ok(CudaVec {
                len:len,
                ptr: value.into(),
                t:PhantomData::<R>
            })
        }
    }
}
impl<U,T,R,A> TryFrom<CudaVecConverter<U,T,A>> for SerializedVec<U,R>
    where U: Debug + Default + Clone + Copy + Send + UnitValue<U>,
          for<'a> T: MemorySize + AsKernelPtr + AsConstKernelPtr,
          for<'b> R: SliceSize + AsRawSlice<U> + MakeView<'b,U> + MakeViewMut<'b,U>,
          A: CudaAllocator,
          CudaPtr<U,A>: ReadMemory<U> {
    type Error = TypeConvertError;
    #[inline]
    fn try_from(value: CudaVecConverter<U,T,A>) -> Result<Self, Self::Error> {
        if T::size() != R::slice_size() {
            Err(TypeConvertError::SizeMismatchError(SizeMismatchError(T::size(),R::slice_size())))
        } else {
            Ok(value.ptr.read_to_vec()?.into_boxed_slice().try_into()?)
        }
    }
}
impl<U,T,A> ToHost<U> for CudaVec<U,T,A>
    where U: Debug + Default + Clone + Copy + Send + UnitValue<U>,
          SerializedVec<U,<T as ToHost<U>>::Output>: TryFrom<Box<[U]>,Error=TypeConvertError>,
          for<'a> <T as ToHost<U>>::Output: SliceSize + MakeView<'a,U>,
          for<'a> T: MemorySize + AsKernelPtr + AsConstKernelPtr + ToHost<U>,
          A: CudaAllocator,
          CudaPtr<U,A>: ReadMemory<U> {
    type Output = SerializedVec<U,<T as ToHost<U>>::Output>;
    #[inline]
    fn to_host(self) -> Result<Self::Output,TypeConvertError> {
        if T::size() != <T as ToHost<U>>::Output::slice_size() {
            Err(TypeConvertError::SizeMismatchError(SizeMismatchError(T::size(),<T as ToHost<U>>::Output::slice_size())))
        } else {
            Ok(self.ptr.read_to_vec()?.into_boxed_slice().try_into()?)
        }
    }
}
impl TryFrom<f32> for CudaPtr<f32,DeviceAllocator> {
    type Error = CudaError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        let mut ptr:CudaPtr<f32,DeviceAllocator> = CudaPtr::new(1,&DeviceAllocator::new())?;
        ptr.memcpy(&value as *const f32,1)?;
        Ok(ptr)
    }
}
impl TryFrom<f64> for CudaPtr<f64,DeviceAllocator> {
    type Error = CudaError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        let mut ptr:CudaPtr<f64,DeviceAllocator> = CudaPtr::new(1,&DeviceAllocator::new())?;
        ptr.memcpy(&value as *const f64,1)?;
        Ok(ptr)
    }
}
impl TryFrom<i32> for CudaPtr<i32,DeviceAllocator> {
    type Error = CudaError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let mut ptr:CudaPtr<i32,DeviceAllocator> = CudaPtr::new(1,&DeviceAllocator::new())?;
        ptr.memcpy(&value as *const i32,1)?;
        Ok(ptr)
    }
}
impl TryFrom<i64> for CudaPtr<i64,DeviceAllocator> {
    type Error = CudaError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        let mut ptr:CudaPtr<i64,DeviceAllocator> = CudaPtr::new(1,&DeviceAllocator::new())?;
        ptr.memcpy(&value as *const i64,1)?;
        Ok(ptr)
    }
}
/// Trait to convert value to Cuda smart pointer type
pub trait ToCuda<T,A> where T: UnitValue<T>, A: CudaAllocator {
    type Output;

    /// # Arguments
    /// * `device` - gpu device
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TypeConvertError`]
    ///
    fn to_cuda(self,device:&DeviceGpu<T,A>) -> Result<Self::Output,TypeConvertError>;
}
/// Trait for inverse conversion of value to host memory type
pub trait ToHost<T> where T: Default + Clone + Send {
    type Output;

    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TypeConvertError`]
    ///
    fn to_host(self) -> Result<Self::Output,TypeConvertError>;
}
impl<'a,T,A,const N:usize> ToCuda<T,A> for &'a CudaTensor1dPtr<T,A,N>
    where T :UnitValue<T>,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T> {
    type Output = CudaTensor1dPtrView<'a,T,N>;

    fn to_cuda(self, _: &DeviceGpu<T,A>) -> Result<Self::Output,TypeConvertError> {
        Ok(self.into())
    }
}
impl<T,A,const N:usize> ToCuda<T,A> for CudaTensor1dPtr<T,A,N>
    where T :UnitValue<T>,
          A: CudaAllocator {
    type Output = CudaTensor1dPtr<T,A,N>;

    fn to_cuda(self, _: &DeviceGpu<T,A>) -> Result<Self::Output,TypeConvertError> {
        Ok(self)
    }
}
impl<'a,T,A,const N1:usize,const N2:usize> ToCuda<T,A> for &'a CudaTensor2dPtr<T,A,N1,N2>
    where T :UnitValue<T>,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T> {
    type Output = CudaTensor2dPtrView<'a,T,N1,N2>;

    fn to_cuda(self, _: &DeviceGpu<T,A>) -> Result<Self::Output,TypeConvertError> {
        Ok(self.into())
    }
}
impl<T,A,const N1:usize,const N2:usize> ToCuda<T,A> for CudaTensor2dPtr<T,A,N1,N2>
    where T :UnitValue<T>,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T> {
    type Output = CudaTensor2dPtr<T,A,N1,N2>;

    fn to_cuda(self, _: &DeviceGpu<T,A>) -> Result<Self::Output,TypeConvertError> {
        Ok(self)
    }
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize> ToCuda<T,A> for &'a CudaTensor3dPtr<T,A,N1,N2,N3>
    where T :UnitValue<T>,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T> {
    type Output = CudaTensor3dPtrView<'a,T,N1,N2,N3>;

    fn to_cuda(self, _: &DeviceGpu<T,A>) -> Result<Self::Output,TypeConvertError> {
        Ok(self.into())
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize> ToCuda<T,A> for CudaTensor3dPtr<T,A,N1,N2,N3>
    where T :UnitValue<T>,
          A: CudaAllocator {
    type Output = CudaTensor3dPtr<T,A,N1,N2,N3>;

    fn to_cuda(self, _: &DeviceGpu<T,A>) -> Result<Self::Output,TypeConvertError> {
        Ok(self)
    }
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> ToCuda<T,A> for &'a CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T :UnitValue<T>,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T> {
    type Output = CudaTensor4dPtrView<'a,T,N1,N2,N3,N4>;

    fn to_cuda(self, _: &DeviceGpu<T,A>) -> Result<Self::Output,TypeConvertError> {
        Ok(self.into())
    }
}
impl<T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> ToCuda<T,A> for CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T :UnitValue<T>,
          A: CudaAllocator {
    type Output = CudaTensor4dPtr<T,A,N1,N2,N3,N4>;

    fn to_cuda(self, _: &DeviceGpu<T,A>) -> Result<Self::Output,TypeConvertError> {
        Ok(self)
    }
}
impl<'a,T,A,const N:usize> AsCudaReadOnlyPtr for &'a CudaTensor1dPtr<T,A,N>
    where T: Default + Debug,
          A: CudaAllocator {
    type Pointee = T;

    #[inline]
    fn as_cuda_read_only_ptr<'b>(&'b self) -> CudaPtrRef<'b,T> {
        self.ptr.as_cuda_read_only_ptr()
    }
}
impl<'a,T,A,const N1:usize,const N2:usize> AsCudaReadOnlyPtr for &'a CudaTensor2dPtr<T,A,N1,N2>
    where T: Default + Debug + 'a,
          A: CudaAllocator + 'a {
    type Pointee = T;

    #[inline]
    fn as_cuda_read_only_ptr<'b>(&'b self) -> CudaPtrRef<'b,T> {
        self.ptr.as_cuda_read_only_ptr()
    }
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize> AsCudaReadOnlyPtr for &'a CudaTensor3dPtr<T,A,N1,N2,N3>
    where T: Default + Debug + 'a,
          A: CudaAllocator + 'a,
          CudaPtr<T,A>: WriteMemory<T> {
    type Pointee = T;

    #[inline]
    fn as_cuda_read_only_ptr<'b>(&'b self) -> CudaPtrRef<'b,T> {
        self.ptr.as_cuda_read_only_ptr()
    }
}
impl<'a,T,A,const N1:usize,const N2:usize,const N3:usize,const N4:usize> AsCudaReadOnlyPtr for &'a CudaTensor4dPtr<T,A,N1,N2,N3,N4>
    where T: Default + Debug + 'a,
          A: CudaAllocator + 'a {

    type Pointee = T;
    #[inline]
    fn as_cuda_read_only_ptr<'b>(&'b self) -> CudaPtrRef<'b,T> {
        self.ptr.as_cuda_read_only_ptr()
    }
}
impl<CP> AsConstKernelPtrBase for CP
    where CP: AsCudaReadOnlyPtr + DeriveCudaConstPtr,
          for<'a> CudaPtrRef<'a,<CP as AsCudaReadOnlyPtr>::Pointee>: AsConstKernelPtr {
    #[inline]
    fn as_const_kernel_ptr(&self) -> *mut c_void {
        self.as_cuda_read_only_ptr().as_const_kernel_ptr()
    }
}
impl<CP> AsVoidPtr for CP
    where CP: AsCudaReadOnlyPtr + DeriveCudaConstPtr,
          for<'a> CudaPtrRef<'a,<CP as AsCudaReadOnlyPtr>::Pointee>: AsVoidPtr {
    #[inline]
    fn as_void_ptr(&self) -> *const c_void {
        self.as_cuda_read_only_ptr().as_void_ptr()
    }
}
impl<CP,T> AsPtr<T> for CP
    where CP: AsCudaReadOnlyPtr + DeriveCudaConstPtr,
          for<'a> CudaPtrRef<'a,<CP as AsCudaReadOnlyPtr>::Pointee>: AsPtr<T> {
    #[inline]
    fn as_ptr(&self) -> *const T {
        self.as_cuda_read_only_ptr().as_ptr()
    }
}
impl<CP,T> ReadMemory<T> for CP
    where for<'a> CP: AsCudaPtr<'a> + PointerElement<Element=T>,
          for<'a> <CP as AsCudaPtr<'a>>::Pointer: ReadMemory<T>,
          T: Debug + Default {
    #[inline]
    fn read_to_vec(&self) -> Result<Vec<T>, Error> {
        self.as_cuda_ptr().read_to_vec()
    }

    #[inline]
    fn read_to_vec_with_size(&self, size: usize) -> Result<Vec<T>, Error> {
        self.as_cuda_ptr().read_to_vec_with_size(size)
    }
}
impl<CP,T> ReadMemoryAsync<T> for CP
    where for<'a> CP: AsCudaPtr<'a> + PointerElement<Element=T>,
          for<'a> <CP as AsCudaPtr<'a>>::Pointer: ReadMemoryAsync<T>,
          T: Debug + Default {
    #[inline]
    fn read_to_vec_async(&self, stream: cudaStream_t) -> Result<Vec<T>, Error> {
        self.as_cuda_ptr().read_to_vec_async(stream)
    }
    #[inline]
    fn read_to_vec_with_size_async(&self, stream: cudaStream_t, size: usize) -> Result<Vec<T>, Error> {
        self.as_cuda_ptr().read_to_vec_with_size_async(stream, size)
    }
}
impl<CP,T,A> WriteMemory<T> for CP
    where CP: AsCudaMutPtr<Pointee=T,Allocator=A>,
          T: Debug + Default,
          A: CudaAllocator + Debug,
          for<'a> CudaMutPtr<'a,T,A>: WriteMemory<T> + AsMutPtr<T> {
    fn memcpy(&mut self, p: *const T, len: usize) -> Result<usize, Error> {
        self.as_cuda_mut_ptr().memcpy(p,len)
    }

    fn memcpy_repeat(&mut self, p: *const T, len: usize, count: usize) -> Result<usize, Error> {
        self.as_cuda_mut_ptr().memcpy_repeat(p,len,count)
    }
}
impl<CP,T,A> WriteMemoryAsync<T> for CP
    where CP: AsCudaMutPtr<Pointee=T,Allocator=A>,
          T: Debug + Default,
          A: CudaAllocator + Debug,
          for<'a> CudaMutPtr<'a,T,A>: WriteMemoryAsync<T> + AsMutPtr<T> {
    fn memcpy_async(&mut self, p: *const T, len: usize, stream: cudaStream_t) -> Result<usize, Error> {
        self.as_cuda_mut_ptr().memcpy_async(p,len,stream)
    }

    fn memcpy_async_repeat(&mut self, p: *const T, len: usize, count: usize, stream: cudaStream_t) -> Result<usize, Error> {
        self.as_cuda_mut_ptr().memcpy_async_repeat(p,len,count,stream)
    }
}
impl<'a,CP,T,D> MemoryMoveTo<T,D> for CP
    where T: Debug + Default,
          D: AsMutPtr<T> + AsCudaMutPtr<Pointee=T>,
          for<'b> CP: AsCudaPtr<'b> + AsPtr<T>,
          for<'b> <CP as AsCudaPtr<'b>>::Pointer: AsPtr<T> + MemoryMoveTo<T,CudaMutPtr<'b,T,<D as AsCudaMutPtr>::Allocator>>,
          for<'b> CudaMutPtr<'b,T,<D as AsCudaMutPtr>::Allocator>: AsMutPtr<T> {
    fn memcpy_to(&self, dst: &mut D, len: usize) -> Result<usize, Error> {
        self.as_cuda_ptr().memcpy_to(&mut dst.as_cuda_mut_ptr(),len)
    }
}
/// Proxy type to Cuda smart pointer type with write operation
pub struct CudaMutPtr<'a,T,A> where T: Debug, A: CudaAllocator + Debug {
    ptr:&'a mut CudaPtr<T,A>
}
impl<'a,T,A: CudaAllocator> CudaMutPtr<'a,T,A> where T: Debug, A: CudaAllocator + Debug {
    pub fn new(ptr:&'a mut CudaPtr<T,A>) -> CudaMutPtr<'a,T,A> {
        CudaMutPtr {
            ptr:ptr
        }
    }
}
impl<'a,T,A> AsMutKernelPtrBase for CudaMutPtr<'a,T,A>
    where T: Debug + Default,
          A: CudaAllocator {
    #[inline]
    fn as_mut_kernel_ptr(&mut self) -> *mut c_void {
        self.ptr.as_mut_kernel_ptr()
    }
}
impl<'a,T,A> AsMutVoidPtr for CudaMutPtr<'a,T,A>
    where T: Debug + Default,
          A: CudaAllocator {
    #[inline]
    fn as_mut_void_ptr(&mut self) -> *mut c_void {
        self.ptr.as_mut_void_ptr()
    }
}
impl<'a,T,A> AsMutPtr<T> for CudaMutPtr<'a,T,A>
    where T: Debug + Default,
          A: CudaAllocator {
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_mut_ptr()
    }
}
impl<'a,T,A> PointerElement for CudaMutPtr<'a,T,A>
    where T: Debug + Default,
          A: CudaAllocator {
    type Element = T;
}
impl<'a,T,A> WriteMemory<T> for CudaMutPtr<'a,T,A>
    where T: Debug + Default,
          A: CudaAllocator,
          CudaPtr<T,A>: WriteMemory<T> + AsMutPtr<T> {
    #[inline]
    fn memcpy(&mut self, p: *const T, len: usize) -> Result<usize, Error> {
        self.ptr.memcpy(p,len)
    }

    #[inline]
    fn memcpy_repeat(&mut self, p: *const T, len: usize, count: usize) -> Result<usize, Error> {
        self.ptr.memcpy_repeat(p,len,count)
    }
}
impl<'a,T,A> WriteMemoryAsync<T> for CudaMutPtr<'a,T,A>
    where T: Debug + Default,
          A: CudaAllocator,
          CudaMutPtr<'a,T,A>: AsMutPtr<T>,
          CudaPtr<T,A>: WriteMemoryAsync<T> {
    #[inline]
    fn memcpy_async(&mut self, p: *const T, len: usize, stream: cudaStream_t) -> Result<usize, Error> {
        self.ptr.memcpy_async(p,len,stream)
    }

    #[inline]
    fn memcpy_async_repeat(&mut self, p: *const T, len: usize, count: usize, stream: cudaStream_t) -> Result<usize, Error> {
        self.ptr.memcpy_async_repeat(p,len,count,stream)
    }
}
/// Characteristic that defines the ability to obtain a reference to a writable cuda smart pointer
pub trait AsCudaMutPtr {
    /// The type of the elements in the referenced CudaPtr
    type Pointee: Debug;
    /// The type of the CudaPtr allocator being referenced
    type Allocator: CudaAllocator;
    fn as_cuda_mut_ptr<'a>(&'a mut self) -> CudaMutPtr<'a,Self::Pointee,Self::Allocator>;
}
impl<CP> AsMutKernelPtrBase for CP
    where CP: AsCudaMutPtr,
          for<'a> CudaMutPtr<'a,<CP as AsCudaMutPtr>::Pointee,<CP as AsCudaMutPtr>::Allocator>: AsMutKernelPtrBase {
    #[inline]
    fn as_mut_kernel_ptr(&mut self) -> *mut c_void {
        self.as_cuda_mut_ptr().as_mut_kernel_ptr()
    }
}
impl<CP> AsMutVoidPtr for CP
    where CP: AsCudaMutPtr,
          for<'a> CudaMutPtr<'a,<CP as AsCudaMutPtr>::Pointee,<CP as AsCudaMutPtr>::Allocator>: AsMutVoidPtr {
    #[inline]
    fn as_mut_void_ptr(&mut self) -> *mut c_void {
        self.as_cuda_mut_ptr().as_mut_void_ptr()
    }
}
impl<CP,T> AsMutPtr<T> for CP
    where CP: AsCudaMutPtr,
          for<'a> CudaMutPtr<'a,<CP as AsCudaMutPtr>::Pointee,<CP as AsCudaMutPtr>::Allocator>: AsMutPtr<T> {
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_cuda_mut_ptr().as_mut_ptr()
    }
}
impl<'a,T,A,const N:usize> From<&'a mut CudaTensor1dPtr<T,A,N>> for CudaMutPtr<'a,T,A>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a mut CudaTensor1dPtr<T,A,N>) -> Self {
        value.as_cuda_mut_ptr()
    }
}
impl<'a,T,A,const N1:usize,const N2:usize> From<&'a mut CudaTensor2dPtr<T,A,N1,N2>> for CudaMutPtr<'a,T,A>
    where T: Default + Debug,
          A: CudaAllocator {
    fn from(value: &'a mut CudaTensor2dPtr<T,A,N1,N2>) -> Self {
        value.as_cuda_mut_ptr()
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

    /// Function that waits for the completion of the execution of the process passed to the Cuda kernel
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaRuntimeError`]
    fn device_synchronize(&self) -> Result<(),CudaRuntimeError> {
        ffi::device_synchronize()
    }
}
/// Trait defining cuda cooperative kernel functions
pub trait CooperativeKernel {
    /// Object to be converted into a list of arguments to be passed to the cuda kernel function
    type Args: KernelArgs;

    /// Pointer to cuda kernel function
    const FUNC_PTR: *const c_void;

    /// cuda kernel startup function
    /// Launches a device function where thread blocks can cooperate and synchronize as they execute.
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
        ffi::launch_cooperative(Self::FUNC_PTR,
                    grid_dim,
                    block_dim,
                    &mut args.as_vec().into_iter()
                        .map(|p| p.as_kernel_ptr())
                        .collect::<Vec<*mut c_void>>().as_mut_slice(),
                    shared_mem
        )
    }

    /// Function that waits for the completion of the execution of the process passed to the Cuda kernel
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`CudaRuntimeError`]
    fn device_synchronize(&self) -> Result<(),CudaRuntimeError> {
        ffi::device_synchronize()
    }
}
