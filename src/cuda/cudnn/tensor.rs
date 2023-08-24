use std::marker::PhantomData;
use rcudnn::API;
use rcudnn_sys::{cudnnSetTensor4dDescriptor, cudnnTensorDescriptor_t,cudnnStatus_t};
use rcudnn_sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
use crate::cuda::DataTypeInfo;

/// Wrapper for cudnnTensorDescriptor_t initialized by cudnnSetTensor4dDescriptors
pub struct CudnnTensor4dDescriptor<T> where T: DataTypeInfo {
    id: cudnnTensorDescriptor_t,
    t:PhantomData<T>
}
impl<T> CudnnTensor4dDescriptor<T> where T: DataTypeInfo {
    /// Create an instance of CudnnTensor4dDescriptor
    /// # Arguments
    ///
    /// * `n` - batch size
    /// * `c` - Number of Channels
    /// * `h` - height
    /// * `w` - width
    pub fn new(n:usize,c:usize,h:usize,w:usize) -> Result<CudnnTensor4dDescriptor<T>,rcudnn::Error> where T: DataTypeInfo {
        let desc = API::create_tensor_descriptor()?;

        unsafe {
            match cudnnSetTensor4dDescriptor(desc,CUDNN_TENSOR_NCHW,T::cudnn_raw_data_type(),
                                             n as libc::c_int,c as libc::c_int, h as libc::c_int, w as libc::c_int) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => (),
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(rcudnn::Error::BadParam("The parameter passed to the vs is invalid."));
                },
                status => {
                    return Err(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64));
                }
            }
        }

        Ok(CudnnTensor4dDescriptor {
            id: desc,
            t:PhantomData::<T>
        })
    }

    pub fn id_c(&self) -> &cudnnTensorDescriptor_t {
        &self.id
    }
}
impl<T> Drop for CudnnTensor4dDescriptor<T> where T: DataTypeInfo {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_tensor_descriptor(*self.id_c());
    }
}
