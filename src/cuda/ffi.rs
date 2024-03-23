//! FFI Module
use std::{mem};
use std::ptr::null_mut;
use cuda_runtime_sys::dim3;
use libc::c_void;
use rcudnn_sys::{cudaError, cudaMemcpyKind, cudaStream_t};
use crate::error::CudaRuntimeError;

/// Device Memory allocation
/// # Arguments
/// * `size` - memory size
///
/// # Errors
///
/// This function may return the following errors
/// * [`rcudnn::Error`]
///
/// note: The actual size of memory allocated is automatically calculated
///       by multiplying the size of the return type by the value of the argument passed.
///       There is no need for the caller to multiply the size of the return type.
pub fn malloc<T>(size: usize) -> Result<*mut T,rcudnn::Error> {
    let size = mem::size_of::<T>() * size;
    let mut ptr: *mut T = null_mut();

    match unsafe { rcudnn_sys::cudaMalloc(&mut ptr as *mut *mut T as *mut *mut libc::c_void, size) } {
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

/// Host Memory allocation
/// # Arguments
/// * `size` - memory size
///
/// # Errors
///
/// This function may return the following errors
/// * [`rcudnn::Error`]
///
/// note: The actual size of memory allocated is automatically calculated
///       by multiplying the size of the return type by the value of the argument passed.
///       There is no need for the caller to multiply the size of the return type.
pub fn malloc_host<T>(size: usize, flags:libc::c_uint) -> Result<*mut T,rcudnn::Error> {
    let size = mem::size_of::<T>() * size;
    let mut ptr: *mut T = null_mut();

    match unsafe { rcudnn_sys::cudaHostAlloc(&mut ptr as *mut *mut T as *mut *mut libc::c_void, size, flags) } {
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

/// Sync Copy memory
/// # Arguments
/// * `dst` - copy destination
/// * `src` - copy source
/// * `size` - Size to copy (number of elements, not bytes)
/// * `kind` - Type of memory copy defined by cuda
///
/// # Errors
///
/// This function may return the following errors
/// * [`rcudnn::Error`]
///
pub fn memcpy<T>(dst: *mut T, src: *const T, size: usize, kind: cudaMemcpyKind) -> Result<(),rcudnn::Error> {
    let size = mem::size_of::<T>() * size;

    match unsafe {
        rcudnn_sys::cudaMemcpy(dst as *mut libc::c_void, src as *mut libc::c_void, size, kind)
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

/// Async Copy memory
/// # Arguments
/// * `dst` - copy destination
/// * `src` - copy source
/// * `size` - Size to copy (number of elements, not bytes)
/// * `kind` - Type of memory copy defined by cuda
/// * `stream` - cuda stream
///
/// # Errors
///
/// This function may return the following errors
/// * [`rcudnn::Error`]
///
pub fn memcpy_async<T>(dst: *mut T, src: *const T, size: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> Result<(),rcudnn::Error> {
    let size = mem::size_of::<T>() * size;
    match unsafe {
        rcudnn_sys::cudaMemcpyAsync(dst as *mut libc::c_void, src as *mut libc::c_void, size, kind, stream)
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

/// free up memory
/// # Arguments
/// * `devptr` - Device Memory object to be free
///
/// # Errors
///
/// This function may return the following errors
/// * [`rcudnn::Error`]
///
pub fn free<T>(devptr: *mut T) -> Result<(),rcudnn::Error> {
    match unsafe { rcudnn_sys::cudaFree(devptr as *mut libc::c_void) } {
        cudaError::cudaSuccess => Ok(()),
        cudaError::cudaErrorInvalidValue => {
            Err(rcudnn::Error::InvalidValue("Invalid pointer passed as argument."))
        },
        status => {
            Err(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64))
        }
    }
}

/// free up memory
/// # Arguments
/// * `devptr` - Host Memory object to be free
///
/// # Errors
///
/// This function may return the following errors
/// * [`rcudnn::Error`]
///
pub fn free_host<T>(devptr: *mut T) -> Result<(),rcudnn::Error> {
    match unsafe { rcudnn_sys::cudaFreeHost(devptr as *mut libc::c_void) } {
        cudaError::cudaSuccess => Ok(()),
        cudaError::cudaErrorInvalidValue => {
            Err(rcudnn::Error::InvalidValue("Invalid pointer passed as argument."))
        },
        status => {
            Err(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64))
        }
    }
}
fn launch_with_stream(func: *const c_void,
                          grid_dim: dim3,
                          block_dim: dim3,
                          args: &mut [*mut c_void],
                          shared_mem: usize,
                          stream:cuda_runtime_sys::cudaStream_t)
                          -> Result<(),CudaRuntimeError> {
    let cuda_error = unsafe {
        cuda_runtime_sys::cudaLaunchKernel(func,
                                           grid_dim,
                                           block_dim,
                                           args.as_mut_ptr(),
                                           shared_mem,
                                           stream)
    };

    if cuda_error == cuda_runtime_sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(CudaRuntimeError::new(cuda_error))
    }
}
/// cuda kernel startup function
/// # Arguments
/// * `func` - Pointer to cuda kernel function
/// * `grid_dim` - Number of dims in grid
/// * `block_dim` - Number of blocks in grid
/// * `args` - List of arguments passed to cuda kernel functions
/// * `shared_mem` - Size (in bytes) of shared memory to allocate for use within cuda kernel functions.
///
/// # Errors
///
/// This function may return the following errors
/// * [`CudaRuntimeError`]
pub fn launch(func: *const c_void,
              grid_dim: dim3,
              block_dim: dim3,
              args: &mut [*mut c_void],
              shared_mem: usize)
              -> Result<(),CudaRuntimeError> {
    launch_with_stream(func,
                       grid_dim,
                       block_dim,
                       args,
                       shared_mem,
                       null_mut())
}
/// Function that waits for the completion of the execution of the process passed to the Cuda kernel
///
/// # Errors
///
/// This function may return the following errors
/// * [`CudaRuntimeError`]
pub fn device_synchronize() -> Result<(),CudaRuntimeError> {
    let cuda_error = unsafe { cuda_runtime_sys::cudaDeviceSynchronize() };

    if cuda_error == cuda_runtime_sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(CudaRuntimeError::new(cuda_error))
    }
}