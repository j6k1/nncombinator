use std::collections::HashSet;
use std::marker::PhantomData;
use rcudnn::{ActivationDescriptor, API, Cudnn, cudnnActivationMode_t, cudnnSetActivationDescriptorSwishBeta, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, cudnnStatus_t, TensorDescriptor};
use crate::UnitValue;
use crate::arr::*;
use crate::cuda::{AsVoidMutPtr, Memory};
use crate::device::*;
use crate::error::{EvaluateError, TrainingError};
use crate::lossfunction::LossFunction;

pub trait Activation<U,T,D>: Send + Sync + 'static
    where U: UnitValue<U>, D: Device<U> {

    fn apply(&self, device:&D, input:&T) -> Result<T, EvaluateError>;
    fn derive(&self, device:&D, o:&T, loss:&T, u:&T) -> Result<T, TrainingError>;
    fn is_canonical_link<L: LossFunction<U>>(&self,l:&L) -> bool;
}
pub trait ActivationCommonBase<U,D>: Send + Sync + 'static where U: UnitValue<U>, D: Device<U> {
    fn apply_common_base(&self, cudnn:&Cudnn,
                         src_desc:&TensorDescriptor,
                         src_ptr:*const libc::c_void,
                         dest_desc:&TensorDescriptor,
                         dest_ptr:*mut libc::c_void) -> Result<(), EvaluateError>;

    fn derive_common_base(&self, cudnn:&Cudnn,
                          src_desc:&TensorDescriptor,
                          src_ptr:*const libc::c_void,
                          src_diff_desc:&TensorDescriptor,
                          src_diff_ptr:*const libc::c_void,
                          dest_desc:&TensorDescriptor,
                          dest_ptr:*const libc::c_void,
                          dest_diff_desc:&TensorDescriptor,
                          dest_diff_ptr:*mut libc::c_void) -> Result<(), TrainingError>;
}
pub trait ActivationCommon<U,T,D>: Send + Sync + 'static where U: UnitValue<U>, D: Device<U> {
    fn apply_common(&self, input:&T, device:&D) -> Result<T, EvaluateError>;
    fn derive_common(&self, o:&T, loss:&T, u:&T, device:&D) -> Result<T, TrainingError>;
    fn is_canonical_link<L: LossFunction<U>>(&self,l:&L) -> bool;
}
impl<U,A: ActivationCommon<U,Arr<U,N>,DeviceGpu<U>>,const N:usize> Activation<U,Arr<U,N>,DeviceGpu<U>> for A
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply(&self, device: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        self.apply_common(input,device)
    }

    fn derive(&self, device: &DeviceGpu<U>, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        self.derive_common(o,loss,u, device)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        <Self as ActivationCommon<U,Arr<U,N>,DeviceGpu<U>>>::is_canonical_link(self, l)
    }
}
pub struct Identity<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>,
    c:HashSet<&'static str>
}
impl<U,D> Identity<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(_:&D) -> Identity<U,D> {
        let mut c = HashSet::new();
        c.insert("mse");

        Identity {
            u: PhantomData::<U>,
            d:PhantomData::<D>,
            c:c
        }
    }
}
impl<U,const N:usize> ActivationCommon<U,Arr<U,N>,DeviceGpu<U>> for Identity<U,DeviceGpu<U>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    fn apply_common(&self, input: &Arr<U,N>, _: &DeviceGpu<U>) -> Result<Arr<U,N>, EvaluateError> {
        Ok((*input).clone())
    }

    fn derive_common(&self, _: &Arr<U,N>, loss: &Arr<U,N>, _: &Arr<U,N>, _: &DeviceGpu<U>) -> Result<Arr<U,N>, TrainingError> {
        Ok((*loss).clone())
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        Ok((*input).clone())
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, _: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        Ok(loss.clone())
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
pub struct Sigmoid<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>,
    c:HashSet<&'static str>
}
impl<U,D> Sigmoid<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(_:&D) -> Sigmoid<U,D> {
        let mut c = HashSet::new();
        c.insert("crossentropy");

        Sigmoid {
            u: PhantomData::<U>,
            d:PhantomData::<D>,
            c:c
        }
    }
}
impl<U> ActivationCommonBase<U,DeviceGpu<U>> for Sigmoid<U,DeviceGpu<U>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    fn apply_common_base(&self, cudnn:&Cudnn,
                         src_desc:&TensorDescriptor,
                         src_ptr:*const libc::c_void,
                         dest_desc:&TensorDescriptor,
                         dest_ptr:*mut libc::c_void) -> Result<(), EvaluateError> {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID)?;

        let alpha = U::one();
        let beta = U::default();

        API::activation_forward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr)?;

        Ok(())
    }

    fn derive_common_base(&self, cudnn:&Cudnn,
                          src_desc:&TensorDescriptor,
                          src_ptr:*const libc::c_void,
                          src_diff_desc:&TensorDescriptor,
                          src_diff_ptr:*const libc::c_void,
                          dest_desc:&TensorDescriptor,
                          dest_ptr:*const libc::c_void,
                          dest_diff_desc:&TensorDescriptor,
                          dest_diff_ptr:*mut libc::c_void) -> Result<(), TrainingError> {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID)?;

        let alpha = U::one();
        let beta = U::default();

        API::activation_backward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_ptr,
            *src_diff_desc.id_c(),
            src_diff_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr,
            *dest_diff_desc.id_c(),
            dest_diff_ptr,
        )?;

        Ok(())
    }
}
impl<U,const N:usize> ActivationCommon<U,Arr<U,N>,DeviceGpu<U>> for Sigmoid<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common(&self, input: &Arr<U,N>, device: &DeviceGpu<U>) -> Result<Arr<U,N>, EvaluateError> {
        device.linear_activation_forward(input,|cudnn,desc,mut input_output| {
            self.apply_common_base(cudnn,
                                   desc,
                                   input_output.as_void_mut_ptr(),
                                   desc,
                                   input_output.as_void_mut_ptr())?;
            Ok(input_output.read_to_vec()?.try_into()?)
        })
    }

    fn derive_common(&self, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>, device: &DeviceGpu<U>) -> Result<Arr<U,N>, TrainingError> {
        device.linear_activation_backward(o,loss,u,|cudnn,
                                                    o_desc,
                                                    mut o_ptr,
                                                    loss_desc,
                                                    mut loss_ptr,
                                                    u_desc,
                                                    mut u_ptr| {
            self.derive_common_base(cudnn,
                                    o_desc,
                                    o_ptr.as_void_mut_ptr(),
                                    loss_desc,
                                            loss_ptr.as_void_mut_ptr(),
                                    u_desc,
                                    u_ptr.as_void_mut_ptr(),
                                    loss_desc,
                                    loss_ptr.as_void_mut_ptr())?;
            Ok(loss_ptr.read_to_vec()?.try_into()?)
        })
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut r = Arr::new();

        for (r,&i) in r.iter_mut().zip(input.iter()) {
            *r = U::one() / (U::one() + (-i).exp());
        }

        Ok(r)
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(u.iter()) {
            let e = U::one() / (U::one() + (-*i).exp());
            *r = e * (U::one() - e);
        }

        for (r,&l) in r.iter_mut().zip(loss.iter()) {
            *r = *r * l;
        }

        Ok(r)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
pub struct ReLu<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> ReLu<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(_:&D) -> ReLu<U,D> {
        ReLu {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U> ActivationCommonBase<U,DeviceGpu<U>> for ReLu<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common_base(&self, cudnn:&Cudnn,
                         src_desc:&TensorDescriptor,
                         src_ptr:*const libc::c_void,
                         dest_desc:&TensorDescriptor,
                         dest_ptr:*mut libc::c_void) -> Result<(), EvaluateError> {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_RELU)?;

        let alpha = U::one();
        let beta = U::default();

        API::activation_forward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr)?;

        Ok(())
    }

    fn derive_common_base(&self, cudnn:&Cudnn,
                          src_desc:&TensorDescriptor,
                          src_ptr:*const libc::c_void,
                          src_diff_desc:&TensorDescriptor,
                          src_diff_ptr:*const libc::c_void,
                          dest_desc:&TensorDescriptor,
                          dest_ptr:*const libc::c_void,
                          dest_diff_desc:&TensorDescriptor,
                          dest_diff_ptr:*mut libc::c_void) -> Result<(), TrainingError> {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_RELU)?;

        let alpha = U::one();
        let beta = U::default();

        API::activation_backward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_ptr,
            *src_diff_desc.id_c(),
            src_diff_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr,
            *dest_diff_desc.id_c(),
            dest_diff_ptr,
        )?;

        Ok(())
    }
}
impl<U,const N:usize> ActivationCommon<U,Arr<U,N>,DeviceGpu<U>> for ReLu<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common(&self, input: &Arr<U,N>, device: &DeviceGpu<U>) -> Result<Arr<U,N>, EvaluateError> {
        device.linear_activation_forward(input,|cudnn,desc,mut input_output| {
            self.apply_common_base(cudnn,
                                   desc,
                                   input_output.as_void_mut_ptr(),
                                   desc,
                                   input_output.as_void_mut_ptr())?;
            Ok(input_output.read_to_vec()?.try_into()?)
        })
    }

    fn derive_common(&self, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>, device: &DeviceGpu<U>) -> Result<Arr<U,N>, TrainingError> {
        device.linear_activation_backward(o,loss,u,|cudnn,
                                                    o_desc,
                                                    mut o_ptr,
                                                    loss_desc,
                                                    mut loss_ptr,
                                                    u_desc,
                                                    mut u_ptr| {
            self.derive_common_base(cudnn,
                                    o_desc,
                                    o_ptr.as_void_mut_ptr(),
                                    loss_desc,
                                    loss_ptr.as_void_mut_ptr(),
                                    u_desc,
                                    u_ptr.as_void_mut_ptr(),
                                    loss_desc,
                                    loss_ptr.as_void_mut_ptr())?;
            Ok(loss_ptr.read_to_vec()?.try_into()?)
        })
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = if *i > U::default() || i.is_nan() {
                *i
            } else {
                U::default()
            };
        }
        Ok(r)
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(u.iter()) {
            if *i > U::default() {
                *r = U::one()
            } else {
                *r = U::default()
            };
        }

        for (r,&l) in r.iter_mut().zip(loss.iter()) {
            *r = *r * l;
        }

        Ok(r)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
pub struct Swish<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> Swish<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(_:&D) -> Swish<U,D> {
        Swish {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U> ActivationCommonBase<U,DeviceGpu<U>> for Swish<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common_base(&self, cudnn:&Cudnn,
                         src_desc:&TensorDescriptor,
                         src_ptr:*const libc::c_void,
                         dest_desc:&TensorDescriptor,
                         dest_ptr:*mut libc::c_void) -> Result<(), EvaluateError> {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_SWISH)?;

        unsafe {
            match cudnnSetActivationDescriptorSwishBeta(*activation_desc.id_c(), 1.) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => (),
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(EvaluateError::CudnnError(rcudnn::Error::BadParam("The activation descriptor is a NULL pointer.")));
                },
                status => {
                    return Err(EvaluateError::CudnnError(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }

        let alpha = U::one();
        let beta = U::default();

        API::activation_forward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr)?;

        Ok(())
    }

    fn derive_common_base(&self, cudnn:&Cudnn,
                          src_desc:&TensorDescriptor,
                          src_ptr:*const libc::c_void,
                          src_diff_desc:&TensorDescriptor,
                          src_diff_ptr:*const libc::c_void,
                          dest_desc:&TensorDescriptor,
                          dest_ptr:*const libc::c_void,
                          dest_diff_desc:&TensorDescriptor,
                          dest_diff_ptr:*mut libc::c_void) -> Result<(), TrainingError> {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_SWISH)?;

        unsafe {
            match cudnnSetActivationDescriptorSwishBeta(*activation_desc.id_c(), 1.) {
                cudnnStatus_t::CUDNN_STATUS_SUCCESS => (),
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                    return Err(TrainingError::CudnnError(rcudnn::Error::BadParam("The activation descriptor is a NULL pointer.")));
                },
                status => {
                    return Err(TrainingError::CudnnError(rcudnn::Error::Unknown("Unable to create the CUDA cuDNN context/resources.", status as i32 as u64)));
                }
            }
        }

        let alpha = U::one();
        let beta = U::default();

        API::activation_backward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_ptr,
            *src_diff_desc.id_c(),
            src_diff_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr,
            *dest_diff_desc.id_c(),
            dest_diff_ptr,
        )?;

        Ok(())
    }
}
impl<U,const N:usize> ActivationCommon<U,Arr<U,N>,DeviceGpu<U>> for Swish<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common(&self, input: &Arr<U,N>, device: &DeviceGpu<U>) -> Result<Arr<U,N>, EvaluateError> {
        device.linear_activation_forward(input,|cudnn,desc,mut input_output| {
            self.apply_common_base(cudnn,
                                   desc,
                                   input_output.as_void_mut_ptr(),
                                   desc,
                                   input_output.as_void_mut_ptr())?;
            Ok(input_output.read_to_vec()?.try_into()?)
        })
    }

    fn derive_common(&self, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>, device: &DeviceGpu<U>) -> Result<Arr<U,N>, TrainingError> {
        device.linear_activation_backward(o,loss,u,|cudnn,
                                                    o_desc,
                                                    mut o_ptr,
                                                    loss_desc,
                                                    mut loss_ptr,
                                                    u_desc,
                                                    mut u_ptr| {
            self.derive_common_base(cudnn,
                                    o_desc,
                                    o_ptr.as_void_mut_ptr(),
                                    loss_desc,
                                    loss_ptr.as_void_mut_ptr(),
                                    u_desc,
                                    u_ptr.as_void_mut_ptr(),
                                    loss_desc,
                                    loss_ptr.as_void_mut_ptr())?;
            Ok(loss_ptr.read_to_vec()?.try_into()?)
        })
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = *r * (U::one() / (U::one() + (-*i).exp()))
        }

        Ok(r)
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(u.iter()) {
            *r = *r * (U::one() / (U::one() + (-*i).exp())) +
                (U::one() / (U::one() + (-*i).exp())) * (U::one() - (*i * (U::one() / (U::one() + (-*i).exp()))))
        }

        for (r,&l) in r.iter_mut().zip(loss.iter()) {
            *r = *r * l;
        }

        Ok(r)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
pub struct Tanh<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>
}
impl<U,D> Tanh<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(_:&D) -> Tanh<U,D> {
        Tanh {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U> ActivationCommonBase<U,DeviceGpu<U>> for Tanh<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common_base(&self, cudnn:&Cudnn,
                         src_desc:&TensorDescriptor,
                         src_ptr:*const libc::c_void,
                         dest_desc:&TensorDescriptor,
                         dest_ptr:*mut libc::c_void) -> Result<(), EvaluateError> {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_TANH)?;

        let alpha = U::one();
        let beta = U::default();

        API::activation_forward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr)?;

        Ok(())
    }

    fn derive_common_base(&self, cudnn:&Cudnn,
                          src_desc:&TensorDescriptor,
                          src_ptr:*const libc::c_void,
                          src_diff_desc:&TensorDescriptor,
                          src_diff_ptr:*const libc::c_void,
                          dest_desc:&TensorDescriptor,
                          dest_ptr:*const libc::c_void,
                          dest_diff_desc:&TensorDescriptor,
                          dest_diff_ptr:*mut libc::c_void) -> Result<(), TrainingError> {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_TANH)?;

        let alpha = U::one();
        let beta = U::default();

        API::activation_backward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_ptr,
            *src_diff_desc.id_c(),
            src_diff_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr,
            *dest_diff_desc.id_c(),
            dest_diff_ptr,
        )?;

        Ok(())
    }
}
impl<U,const N:usize> ActivationCommon<U,Arr<U,N>,DeviceGpu<U>> for Tanh<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common(&self, input: &Arr<U,N>, device: &DeviceGpu<U>) -> Result<Arr<U,N>, EvaluateError> {
        device.linear_activation_forward(input,|cudnn,desc,mut input_output| {
            self.apply_common_base(cudnn,
                                   desc,
                                   input_output.as_void_mut_ptr(),
                                   desc,
                                   input_output.as_void_mut_ptr())?;
            Ok(input_output.read_to_vec()?.try_into()?)
        })
    }

    fn derive_common(&self, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>, device: &DeviceGpu<U>) -> Result<Arr<U,N>, TrainingError> {
        device.linear_activation_backward(o,loss,u,|cudnn,
                                                    o_desc,
                                                    mut o_ptr,
                                                    loss_desc,
                                                    mut loss_ptr,
                                                    u_desc,
                                                    mut u_ptr| {
            self.derive_common_base(cudnn,
                                    o_desc,
                                    o_ptr.as_void_mut_ptr(),
                                    loss_desc,
                                    loss_ptr.as_void_mut_ptr(),
                                    u_desc,
                                    u_ptr.as_void_mut_ptr(),
                                    loss_desc,
                                    loss_ptr.as_void_mut_ptr())?;
            Ok(loss_ptr.read_to_vec()?.try_into()?)
        })
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = i.tanh();
        }

        Ok(r)
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(u.iter()) {
            let e = i.tanh();
            *r = U::one() - e * e;
        }

        for (r,&l) in r.iter_mut().zip(loss.iter()) {
            *r = *r * l;
        }

        Ok(r)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
pub struct SoftMax<U,D> where U: UnitValue<U>, D: Device<U> {
    u:PhantomData<U>,
    d:PhantomData<D>,
    c:HashSet<&'static str>
}
impl<U,D> SoftMax<U,D> where U: UnitValue<U>, D: Device<U> {
    pub fn new(_:&D) -> SoftMax<U,D> {
        let mut c = HashSet::new();
        c.insert("crossentropymulticlass");

        SoftMax {
            u: PhantomData::<U>,
            d:PhantomData::<D>,
            c:c
        }
    }
}
impl<U> ActivationCommonBase<U,DeviceGpu<U>> for SoftMax<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common_base(&self, cudnn:&Cudnn,
                         src_desc:&TensorDescriptor,
                         src_ptr:*const libc::c_void,
                         dest_desc:&TensorDescriptor,
                         dest_ptr:*mut libc::c_void) -> Result<(), EvaluateError> {
        let alpha = U::one();
        let beta = U::default();

        API::softmax_forward(
            *cudnn.id_c(),
            cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST,
            cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr,
        )?;

        Ok(())
    }

    fn derive_common_base(&self, cudnn:&Cudnn,
                          src_desc:&TensorDescriptor,
                          src_ptr:*const libc::c_void,
                          src_diff_desc:&TensorDescriptor,
                          src_diff_ptr:*const libc::c_void,
                          _:&TensorDescriptor,
                          _:*const libc::c_void,
                          dest_diff_desc:&TensorDescriptor,
                          dest_diff_ptr:*mut libc::c_void) -> Result<(), TrainingError> {
        let alpha = U::one();
        let beta = U::default();

        API::softmax_backward(
            *cudnn.id_c(),
            cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST,
            cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_ptr,
            *src_diff_desc.id_c(),
            src_diff_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_diff_desc.id_c(),
            dest_diff_ptr
        )?;

        Ok(())
    }
}
impl<U,const N:usize> ActivationCommon<U,Arr<U,N>,DeviceGpu<U>> for SoftMax<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common(&self, input: &Arr<U,N>, device: &DeviceGpu<U>) -> Result<Arr<U,N>, EvaluateError> {
        device.linear_activation_forward(input,|cudnn,desc,mut input_output| {
            self.apply_common_base(cudnn,
                                   desc,
                                   input_output.as_void_mut_ptr(),
                                   desc,
                                   input_output.as_void_mut_ptr())?;
            Ok(input_output.read_to_vec()?.try_into()?)
        })
    }

    fn derive_common(&self, o: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>, device: &DeviceGpu<U>) -> Result<Arr<U,N>, TrainingError> {
        device.linear_activation_backward(o,loss,u,|cudnn,
                                                    o_desc,
                                                    mut o_ptr,
                                                    loss_desc,
                                                    mut loss_ptr,
                                                    u_desc,
                                                    mut u_ptr| {
            self.derive_common_base(cudnn,
                                    o_desc,
                                    o_ptr.as_void_mut_ptr(),
                                    loss_desc,
                                    loss_ptr.as_void_mut_ptr(),
                                    u_desc,
                                    u_ptr.as_void_mut_ptr(),
                                    loss_desc,
                                    loss_ptr.as_void_mut_ptr())?;
            Ok(loss_ptr.read_to_vec()?.try_into()?)
        })
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut r = Arr::new();

        let alpha = input.iter().fold(U::initial_max_value(), |m, &v| v.max(&m));
        let sum = input.iter().fold(U::default(),|acc, &x| acc + (x - alpha).exp());

        for (r,i) in r.iter_mut().zip(input.iter()) {
            let number = (*i - alpha).exp();
            *r = number / sum;
        }

        Ok(r)
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(u.iter()) {
            *r = *i * (U::one() - *i);
        }

        for (r,&l) in r.iter_mut().zip(loss.iter()) {
            *r = *r * l;
        }

        Ok(r)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
