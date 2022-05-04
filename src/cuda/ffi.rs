use std::{mem, result};
use std::os::raw;
use std::ptr::null_mut;
use rcudnn_sys::{cudaError, cudaMemcpyKind, cudaStream_t};

pub type Result<T> = result::Result<T, rcudnn::Error>;

pub fn malloc<T>(size: usize) -> Result<*mut T> {
    let size = mem::size_of::<T>() * size;
    let mut ptr: *mut T = null_mut();

    match unsafe { rcudnn_sys::cudaMalloc(&mut ptr as *mut *mut T as *mut *mut raw::c_void, size) } {
        cudaError::cudaSuccess => {
            assert_ne!(ptr,
                       null_mut(),
                       "cudaMalloc is succeeded, but returned null pointer!");
            Ok(ptr)
        },
        cudaError::cudaErrorInvalidValue => {
            Err(rcudnn::Error::InvalidValue("The range of one or more of the entered parameters is out of tolerance."))
        },
        cudaError::cudaErrorMemoryAllocation => {
            Err(rcudnn::Error::AllocFailed("Device memory allocation failed."))
        },
        status => {
            Err(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64))
        }
    }
}

pub fn malloc_host<T>(size: usize, flags:libc::c_uint) -> Result<*mut T> {
    let size = mem::size_of::<T>() * size;
    let mut ptr: *mut T = null_mut();

    match unsafe { rcudnn_sys::cudaHostAlloc(&mut ptr as *mut *mut T as *mut *mut raw::c_void, size, flags) } {
        cudaError::cudaSuccess => {
            assert_ne!(ptr,
                       null_mut(),
                       "cudaMalloc is succeeded, but returned null pointer!");
            Ok(ptr)
        },
        cudaError::cudaErrorInvalidValue => {
            Err(rcudnn::Error::InvalidValue("The range of one or more of the entered parameters is out of tolerance."))
        },
        cudaError::cudaErrorMemoryAllocation => {
            Err(rcudnn::Error::AllocFailed("Device memory allocation failed."))
        },
        status => {
            Err(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64))
        }
    }
}

pub fn memcpy<T>(dst: *mut T, src: *const T, size: usize, kind: cudaMemcpyKind) -> Result<()> {
    let size = mem::size_of::<T>() * size;

    match unsafe {
        rcudnn_sys::cudaMemcpy(dst as *mut raw::c_void, src as *mut raw::c_void, size, kind)
    } {
        cudaError::cudaSuccess => {
            Ok(())
        },
        cudaError::cudaErrorInvalidMemcpyDirection => {
            Err(rcudnn::Error::BadParam("Incorrect specification of memory transfer direction."))
        },
        status => {
            Err(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64))
        }
    }
}

pub fn memcpy_async<T>(dst: *mut T, src: *const T, size: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> Result<()> {
    let size = mem::size_of::<T>() * size;
    match unsafe {
        rcudnn_sys::cudaMemcpyAsync(dst as *mut raw::c_void, src as *mut raw::c_void, size, kind, stream)
    } {
        cudaError::cudaSuccess => {
            Ok(())
        },
        cudaError::cudaErrorInvalidMemcpyDirection => {
            Err(rcudnn::Error::BadParam("Incorrect specification of memory transfer direction."))
        },
        status => {
            Err(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64))
        }
    }
}

pub fn free<T>(devptr: *mut T) -> Result<()> {
    match unsafe { rcudnn_sys::cudaFree(devptr as *mut raw::c_void) } {
        cudaError::cudaSuccess => Ok(()),
        cudaError::cudaErrorInvalidValue => {
            Err(rcudnn::Error::InvalidValue("Invalid pointer passed as argument."))
        },
        status => {
            Err(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64))
        }
    }
}
pub fn free_host<T>(devptr: *mut T) -> Result<()> {
    match unsafe { rcudnn_sys::cudaFreeHost(devptr as *mut raw::c_void) } {
        cudaError::cudaSuccess => Ok(()),
        cudaError::cudaErrorInvalidValue => {
            Err(rcudnn::Error::InvalidValue("Invalid pointer passed as argument."))
        },
        status => {
            Err(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64))
        }
    }
}
