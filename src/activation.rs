use std::collections::HashSet;
use std::marker::PhantomData;
use rcudnn::{ActivationDescriptor, API, Cudnn, cudnnActivationMode_t, cudnnSetActivationDescriptorSwishBeta, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, TensorDescriptor};
use crate::UnitValue;
use crate::arr::*;
use crate::device::*;
use crate::lossfunction::LossFunction;
use crate::mem::{AsRawSlice};

pub trait  ActivationImplType: Sync {}
pub enum Specialized {}
pub enum Common {}

impl ActivationImplType for Specialized {}
impl ActivationImplType  for Common {}

pub trait Activation<U,T,D,I>: Send + Sync + 'static
    where U: UnitValue<U>, D: Device<U>, I: ActivationImplType {

    fn apply(&self,device:&D,input:&T) -> T;
    fn derive(&self,device:&D,o:&T,loss:&T,u:&T) -> T;
    fn is_canonical_link<L: LossFunction<U>>(&self,l:&L) -> bool;
}
pub trait ActivationCommon<U,D>: Send + Sync + 'static where U: UnitValue<U>, D: Device<U> {
    fn apply_common(&self,cudnn:&Cudnn,
                        src_desc:&TensorDescriptor,
                        src_data:*const libc::c_void,
                        dest_desc:&TensorDescriptor,
                        dest_data:*mut libc::c_void);

    fn derive_common(&self,cudnn:&Cudnn,
                     src_desc:&TensorDescriptor,
                     src_data:*const libc::c_void,
                     src_diff_desc:&TensorDescriptor,
                     src_diff_data:*const libc::c_void,
                     dest_desc:&TensorDescriptor,
                     dest_data:*const libc::c_void,
                     dest_diff_desc:&TensorDescriptor,
                     dest_diff_ptr:*mut libc::c_void);
    fn is_canonical_link<L: LossFunction<U>>(&self,l:&L) -> bool;
}
impl<U,A: ActivationCommon<U,DeviceGpu<U>>,const N:usize> Activation<U,Arr<U,N>,DeviceGpu<U>,Common> for A
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply(&self, device: &DeviceGpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        device.linear_activation_forward(|cudnn, src_desc, dest_desc, dest_ptr| {
            self.apply_common(
                cudnn,
                src_desc,
                input.as_raw_slice().as_ptr() as *const ::libc::c_void,
                dest_desc,
                dest_ptr)
        })
    }

    fn derive(&self, device: &DeviceGpu<U>, o: &Arr<U, N>,loss: &Arr<U, N>, u:&Arr<U, N>)-> Arr<U, N> {
        device.linear_activation_backward(|cudnn,
                                           src_desc,
                                           src_diff_desc,
                                           dest_desc,
                                           dest_diff_desc,
                                           dest_diff_ptr| {

            self.derive_common(cudnn,
                               src_desc,
                               o.as_raw_slice().as_ptr() as *const libc::c_void,
                               src_diff_desc,
                               loss.as_raw_slice().as_ptr() as *const libc::c_void,
                               dest_desc,
                               u.as_raw_slice().as_ptr() as *const libc::c_void,
                               dest_diff_desc,
                               dest_diff_ptr
            )
        })
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        <Self as ActivationCommon<U,DeviceGpu<U>>>::is_canonical_link(self,l)
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
impl<U,D:Device<U>,const N:usize> Activation<U,Arr<U,N>,D,Specialized> for Identity<U,DeviceCpu<U>>
    where U: UnitValue<U>, D: Device<U> {

    fn apply(&self, _: &D, input: &Arr<U, N>) -> Arr<U, N> {
        (*input).clone()
    }

    fn derive(&self, _: &D, _: &Arr<U,N>,l: &Arr<U,N>,_: &Arr<U, N>) -> Arr<U, N> {
        l.clone()
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
    pub fn new(_:&DeviceCpu<U>) -> Sigmoid<U,D> {
        let mut c = HashSet::new();
        c.insert("crossentropy");

        Sigmoid {
            u: PhantomData::<U>,
            d:PhantomData::<D>,
            c:c
        }
    }
}
impl<U> ActivationCommon<U,DeviceGpu<U>> for Sigmoid<U,DeviceGpu<U>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    fn apply_common(&self,cudnn:&Cudnn,
                    src_desc:&TensorDescriptor,
                    src_data:*const libc::c_void,
                    dest_desc:&TensorDescriptor,
                    dest_ptr:*mut libc::c_void) {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID).unwrap();

        let alpha = U::one();
        let beta = U::default();

        API::activation_forward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_data,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr).unwrap()
    }

    fn derive_common(&self,cudnn:&Cudnn,
                     src_desc:&TensorDescriptor,
                     src_data:*const libc::c_void,
                     src_diff_desc:&TensorDescriptor,
                     src_diff_data:*const libc::c_void,
                     dest_desc:&TensorDescriptor,
                     dest_data:*const libc::c_void,
                     dest_diff_desc:&TensorDescriptor,
                     dest_diff_ptr:*mut libc::c_void) {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID).unwrap();

        let alpha = U::one();
        let beta = U::default();

        API::activation_backward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_data,
            *src_diff_desc.id_c(),
            src_diff_data,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_data,
            *dest_diff_desc.id_c(),
            dest_diff_ptr,
        ).unwrap()
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>,Common> for Sigmoid<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,&i) in r.iter_mut().zip(input.iter()) {
            *r = U::one() / (U::one() + (-i).exp());
        }

        r
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U, N>,l: &Arr<U, N>,u: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(u.iter()) {
            let e = U::one() / (U::one() + (-*i).exp());
            *r = e * (U::one() - e);
        }

        for (r,&l) in r.iter_mut().zip(l.iter()) {
            *r = *r * l;
        }

        r
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
    pub fn new(_:&DeviceCpu<U>) -> ReLu<U,D> {
        ReLu {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U> ActivationCommon<U,DeviceGpu<U>> for ReLu<U,DeviceGpu<U>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    fn apply_common(&self,cudnn:&Cudnn,
                        src_desc:&TensorDescriptor,
                        src_data:*const libc::c_void,
                        dest_desc:&TensorDescriptor,
                        dest_ptr:*mut libc::c_void) {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_RELU).unwrap();

        let alpha = U::one();
        let beta = U::default();

        API::activation_forward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_data,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr).unwrap()
    }

    fn derive_common(&self,cudnn:&Cudnn,
                     src_desc:&TensorDescriptor,
                     src_data:*const libc::c_void,
                     src_diff_desc:&TensorDescriptor,
                     src_diff_data:*const libc::c_void,
                     dest_desc:&TensorDescriptor,
                     dest_data:*const libc::c_void,
                     dest_diff_desc:&TensorDescriptor,
                     dest_diff_ptr:*mut libc::c_void) {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_RELU).unwrap();

        let alpha = U::one();
        let beta = U::default();

        API::activation_backward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_data,
            *src_diff_desc.id_c(),
            src_diff_data,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_data,
            *dest_diff_desc.id_c(),
            dest_diff_ptr,
        ).unwrap()
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>,Common> for ReLu<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = if *i > U::default() || i.is_nan() {
                *i
            } else {
                U::default()
            };
        }
        r
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U, N>,l: &Arr<U, N>,u: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(u.iter()) {
            if *i > U::default() {
                *r = U::one()
            } else {
                *r = U::default()
            };
        }

        for (r,&l) in r.iter_mut().zip(l.iter()) {
            *r = *r * l;
        }

        r
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
    pub fn new(_:&DeviceCpu<U>) -> Swish<U,D> {
        Swish {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U> ActivationCommon<U,DeviceGpu<U>> for Swish<U,DeviceGpu<U>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    fn apply_common(&self,cudnn:&Cudnn,
                    src_desc:&TensorDescriptor,
                    src_data:*const libc::c_void,
                    dest_desc:&TensorDescriptor,
                    dest_data:*mut libc::c_void) {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_SWISH).unwrap();

        unsafe {
            cudnnSetActivationDescriptorSwishBeta(*activation_desc.id_c(), 1.);
        }

        let alpha = U::one();
        let beta = U::default();

        API::activation_forward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_data,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_data).unwrap()
    }

    fn derive_common(&self,cudnn:&Cudnn,
                     src_desc:&TensorDescriptor,
                     src_data:*const libc::c_void,
                     src_diff_desc:&TensorDescriptor,
                     src_diff_data:*const libc::c_void,
                     dest_desc:&TensorDescriptor,
                     dest_data:*const libc::c_void,
                     dest_diff_desc:&TensorDescriptor,
                     dest_diff_ptr:*mut libc::c_void) {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_SWISH).unwrap();

        let alpha = U::one();
        let beta = U::default();

        API::activation_backward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_data,
            *src_diff_desc.id_c(),
            src_diff_data,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_data,
            *dest_diff_desc.id_c(),
            dest_diff_ptr,
        ).unwrap()
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>,Common> for Swish<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = *r * (U::one() / (U::one() + (-*i).exp()))
        }

        r
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U, N>,l: &Arr<U, N>,u: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(u.iter()) {
            *r = *r * (U::one() / (U::one() + (-*i).exp())) +
                (U::one() / (U::one() + (-*i).exp())) * (U::one() - (*i * (U::one() / (U::one() + (-*i).exp()))))
        }

        for (r,&l) in r.iter_mut().zip(l.iter()) {
            *r = *r * l;
        }

        r
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
    pub fn new(_:&DeviceCpu<U>) -> Tanh<U,D> {
        Tanh {
            u: PhantomData::<U>,
            d:PhantomData::<D>
        }
    }
}
impl<U> ActivationCommon<U,DeviceGpu<U>> for Tanh<U,DeviceGpu<U>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    fn apply_common(&self,cudnn:&Cudnn,
                    src_desc:&TensorDescriptor,
                    src_data:*const libc::c_void,
                    dest_desc:&TensorDescriptor,
                    dest_ptr:*mut libc::c_void) {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_TANH).unwrap();

        let alpha = U::one();
        let beta = U::default();

        API::activation_forward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_data,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr).unwrap()
    }

    fn derive_common(&self,cudnn:&Cudnn,
                     src_desc:&TensorDescriptor,
                     src_data:*const libc::c_void,
                     src_diff_desc:&TensorDescriptor,
                     src_diff_data:*const libc::c_void,
                     dest_desc:&TensorDescriptor,
                     dest_data:*const libc::c_void,
                     dest_diff_desc:&TensorDescriptor,
                     dest_diff_ptr:*mut libc::c_void) {
        let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_TANH).unwrap();

        let alpha = U::one();
        let beta = U::default();

        API::activation_backward(
            *cudnn.id_c(),
            *activation_desc.id_c(),
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_data,
            *src_diff_desc.id_c(),
            src_diff_data,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_data,
            *dest_diff_desc.id_c(),
            dest_diff_ptr,
        ).unwrap()
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>,Common> for Tanh<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(input.iter()) {
            *r = i.tanh();
        }

        r
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U, N>,l: &Arr<U, N>,u: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(u.iter()) {
            let e = i.tanh();
            *r = U::one() - e * e;
        }

        for (r,&l) in r.iter_mut().zip(l.iter()) {
            *r = *r * l;
        }

        r
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
    pub fn new(_:&DeviceCpu<U>) -> SoftMax<U,D> {
        let mut c = HashSet::new();
        c.insert("crossentropymulticlass");

        SoftMax {
            u: PhantomData::<U>,
            d:PhantomData::<D>,
            c:c
        }
    }
}
impl<U> ActivationCommon<U,DeviceGpu<U>> for SoftMax<U,DeviceGpu<U>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    fn apply_common(&self,cudnn:&Cudnn,
                    src_desc:&TensorDescriptor,
                    src_data:*const libc::c_void,
                    dest_desc:&TensorDescriptor,
                    dest_ptr:*mut libc::c_void) {
        let alpha = U::one();
        let beta = U::default();

        API::softmax_forward(
            *cudnn.id_c(),
            cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST,
            cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_data,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr,
        ).unwrap()
    }

    fn derive_common(&self,cudnn:&Cudnn,
                     src_desc:&TensorDescriptor,
                     src_data:*const libc::c_void,
                     src_diff_desc:&TensorDescriptor,
                     src_diff_data:*const libc::c_void,
                     _:&TensorDescriptor,
                     _:*const libc::c_void,
                     dest_diff_desc:&TensorDescriptor,
                     dest_diff_ptr:*mut libc::c_void) {
        let alpha = U::one();
        let beta = U::default();

        API::softmax_backward(
            *cudnn.id_c(),
            cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST,
            cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha as *const U as *const libc::c_void,
            *src_desc.id_c(),
            src_data,
            *src_diff_desc.id_c(),
            src_diff_data,
            &beta as *const U as *const libc::c_void,
            *dest_diff_desc.id_c(),
            dest_diff_ptr
        ).unwrap()
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>,Common> for SoftMax<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        let alpha = input.iter().fold(U::initial_max_value(), |m, &v| v.max(&m));
        let sum = input.iter().fold(U::default(),|acc, &x| acc + (x - alpha).exp());

        for (r,i) in r.iter_mut().zip(input.iter()) {
            let number = (*i - alpha).exp();
            *r = number / sum;
        }

        r
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U, N>,l: &Arr<U, N>,u: &Arr<U, N>) -> Arr<U, N> {
        let mut r = Arr::new();

        for (r,i) in r.iter_mut().zip(u.iter()) {
            *r = *i * (U::one() - *i);
        }

        for (r,&l) in r.iter_mut().zip(l.iter()) {
            *r = *r * l;
        }

        r
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
