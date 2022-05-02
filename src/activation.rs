use std::collections::HashSet;
use std::marker::PhantomData;
use rcudnn::{ActivationDescriptor, API, Cudnn, cudnnActivationMode_t, cudnnSetActivationDescriptorSwishBeta, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, TensorDescriptor};
use crate::UnitValue;
use crate::arr::*;
use crate::device::*;
use crate::lossfunction::LossFunction;
use crate::mem::{AsRawMutSlice};

pub trait Activation<U,T,D>: Send + Sync + 'static
    where U: UnitValue<U>, D: Device<U> {

    fn apply(&self,device:&D,input:&T) -> T;
    fn derive(&self,device:&D,o:&T,loss:&T,u:&T) -> T;
    fn is_canonical_link<L: LossFunction<U>>(&self,l:&L) -> bool;
}
pub trait ActivationCommonBase<U,D>: Send + Sync + 'static where U: UnitValue<U>, D: Device<U> {
    fn apply_common_base(&self, cudnn:&Cudnn,
                         src_desc:&TensorDescriptor,
                         src_ptr:*const libc::c_void,
                         dest_desc:&TensorDescriptor,
                         dest_ptr:*mut libc::c_void);

    fn derive_common_base(&self, cudnn:&Cudnn,
                          src_desc:&TensorDescriptor,
                          src_ptr:*const libc::c_void,
                          src_diff_desc:&TensorDescriptor,
                          src_diff_ptr:*const libc::c_void,
                          dest_desc:&TensorDescriptor,
                          dest_ptr:*const libc::c_void,
                          dest_diff_desc:&TensorDescriptor,
                          dest_diff_ptr:*mut libc::c_void);
}
pub trait ActivationCommon<U,T,D>: Send + Sync + 'static where U: UnitValue<U>, D: Device<U> {
    fn apply_common(&self, input:&T,device:&D) -> T;
    fn derive_common(&self, o:&T, loss:&T, u:&T,device:&D) -> T;
    fn is_canonical_link<L: LossFunction<U>>(&self,l:&L) -> bool;
}
impl<U,A: ActivationCommon<U,Arr<U,N>,DeviceGpu<U>>,const N:usize> Activation<U,Arr<U,N>,DeviceGpu<U>> for A
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply(&self, device: &DeviceGpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        self.apply_common(input,device)
    }

    fn derive(&self, device: &DeviceGpu<U>, o: &Arr<U, N>,loss: &Arr<U, N>, u:&Arr<U, N>)-> Arr<U, N> {
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
    fn apply_common(&self,input:&Arr<U,N>,_:&DeviceGpu<U>) -> Arr<U,N> {
        (*input).clone()
    }

    fn derive_common(&self, _:&Arr<U,N>, loss:&Arr<U,N>, _:&Arr<U,N>,_:&DeviceGpu<U>) -> Arr<U,N> {
        (*loss).clone()
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Identity<U,DeviceCpu<U>>
    where U: UnitValue<U> {

    fn apply(&self, _: &DeviceCpu<U>, input: &Arr<U, N>) -> Arr<U, N> {
        (*input).clone()
    }

    fn derive(&self, _: &DeviceCpu<U>, _: &Arr<U,N>,l: &Arr<U,N>,_: &Arr<U, N>) -> Arr<U, N> {
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
impl<U> ActivationCommonBase<U,DeviceGpu<U>> for Sigmoid<U,DeviceGpu<U>> where U: UnitValue<U>, DeviceGpu<U>: Device<U> {
    fn apply_common_base(&self, cudnn:&Cudnn,
                         src_desc:&TensorDescriptor,
                         src_ptr:*const libc::c_void,
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
            src_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr).unwrap()
    }

    fn derive_common_base(&self, cudnn:&Cudnn,
                          src_desc:&TensorDescriptor,
                          src_ptr:*const libc::c_void,
                          src_diff_desc:&TensorDescriptor,
                          src_diff_ptr:*const libc::c_void,
                          dest_desc:&TensorDescriptor,
                          dest_ptr:*const libc::c_void,
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
            src_ptr,
            *src_diff_desc.id_c(),
            src_diff_ptr,
            &beta as *const U as *const libc::c_void,
            *dest_desc.id_c(),
            dest_ptr,
            *dest_diff_desc.id_c(),
            dest_diff_ptr,
        ).unwrap()
    }
}
impl<U,const N:usize> ActivationCommon<U,Arr<U,N>,DeviceGpu<U>> for Sigmoid<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common(&self, input:&Arr<U,N>, device:&DeviceGpu<U>) -> Arr<U,N> {
        device.linear_activation_forward(input,|cudnn,src_desc,src_ptr,dest_desc,mut dest_data| {
            self.apply_common_base(cudnn,
                                   src_desc,
                                   src_ptr,
                                   dest_desc,
                                   dest_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
            dest_data
        })
    }

    fn derive_common(&self, o:&Arr<U,N>, loss:&Arr<U,N>, u:&Arr<U,N>, device:&DeviceGpu<U>) -> Arr<U,N> {
        device.linear_activation_backward(o,loss,u,|cudnn,
                                                            src_desc,
                                                            src_ptr,
                                                            src_diff_desc,
                                                            src_diff_ptr,
                                                            dest_desc,
                                                            dest_ptr,
                                                            dest_diff_desc,
                                                            mut dest_diff_data| {
            self.derive_common_base(cudnn,
                                    src_desc,
                                    src_ptr,
                                    src_diff_desc,
                                    src_diff_ptr,
                                    dest_desc,
                                    dest_ptr,
                                    dest_diff_desc,
                                    dest_diff_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
            dest_diff_data
        })
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Sigmoid<U,DeviceCpu<U>>
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
impl<U> ActivationCommonBase<U,DeviceGpu<U>> for ReLu<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common_base(&self, cudnn:&Cudnn,
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

    fn derive_common_base(&self, cudnn:&Cudnn,
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
}
impl<U,const N:usize> ActivationCommon<U,Arr<U,N>,DeviceGpu<U>> for ReLu<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common(&self, input:&Arr<U,N>, device:&DeviceGpu<U>) -> Arr<U,N> {
        device.linear_activation_forward(input,|cudnn,src_desc,src_ptr,dest_desc,mut dest_data| {
            self.apply_common_base(cudnn,
                                   src_desc,
                                   src_ptr,
                                   dest_desc,
                                   dest_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
            dest_data
        })
    }

    fn derive_common(&self, o:&Arr<U,N>, loss:&Arr<U,N>, u:&Arr<U,N>, device:&DeviceGpu<U>) -> Arr<U,N> {
        device.linear_activation_backward(o,loss,u,|cudnn,
                                                    src_desc,
                                                    src_ptr,
                                                    src_diff_desc,
                                                    src_diff_ptr,
                                                    dest_desc,
                                                    dest_ptr,
                                                    dest_diff_desc,
                                                    mut dest_diff_data| {
            self.derive_common_base(cudnn,
                                    src_desc,
                                    src_ptr,
                                    src_diff_desc,
                                    src_diff_ptr,
                                    dest_desc,
                                    dest_ptr,
                                    dest_diff_desc,
                                    dest_diff_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
            dest_diff_data
        })
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for ReLu<U,DeviceCpu<U>>
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
impl<U> ActivationCommonBase<U,DeviceGpu<U>> for Swish<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common_base(&self, cudnn:&Cudnn,
                         src_desc:&TensorDescriptor,
                         src_data:*const libc::c_void,
                         dest_desc:&TensorDescriptor,
                         dest_ptr:*mut libc::c_void) {
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
            dest_ptr).unwrap()
    }

    fn derive_common_base(&self, cudnn:&Cudnn,
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
}
impl<U,const N:usize> ActivationCommon<U,Arr<U,N>,DeviceGpu<U>> for Swish<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common(&self, input:&Arr<U,N>, device:&DeviceGpu<U>) -> Arr<U,N> {
        device.linear_activation_forward(input,|cudnn,src_desc,src_ptr,dest_desc,mut dest_data| {
            self.apply_common_base(cudnn,
                                   src_desc,
                                   src_ptr,
                                   dest_desc,
                                   dest_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
            dest_data
        })
    }

    fn derive_common(&self, o:&Arr<U,N>, loss:&Arr<U,N>, u:&Arr<U,N>, device:&DeviceGpu<U>) -> Arr<U,N> {
        device.linear_activation_backward(o,loss,u,|cudnn,
                                                    src_desc,
                                                    src_ptr,
                                                    src_diff_desc,
                                                    src_diff_ptr,
                                                    dest_desc,
                                                    dest_ptr,
                                                    dest_diff_desc,
                                                    mut dest_diff_data| {
            self.derive_common_base(cudnn,
                                    src_desc,
                                    src_ptr,
                                    src_diff_desc,
                                    src_diff_ptr,
                                    dest_desc,
                                    dest_ptr,
                                    dest_diff_desc,
                                    dest_diff_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
            dest_diff_data
        })
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Swish<U,DeviceCpu<U>>
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
impl<U> ActivationCommonBase<U,DeviceGpu<U>> for Tanh<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common_base(&self, cudnn:&Cudnn,
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

    fn derive_common_base(&self, cudnn:&Cudnn,
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
}
impl<U,const N:usize> ActivationCommon<U,Arr<U,N>,DeviceGpu<U>> for Tanh<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common(&self, input:&Arr<U,N>, device:&DeviceGpu<U>) -> Arr<U,N> {
        device.linear_activation_forward(input,|cudnn,src_desc,src_ptr,dest_desc,mut dest_data| {
            self.apply_common_base(cudnn,
                                   src_desc,
                                   src_ptr,
                                   dest_desc,
                                   dest_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
            dest_data
        })
    }

    fn derive_common(&self, o:&Arr<U,N>, loss:&Arr<U,N>, u:&Arr<U,N>, device:&DeviceGpu<U>) -> Arr<U,N> {
        device.linear_activation_backward(o,loss,u,|cudnn,
                                                    src_desc,
                                                    src_ptr,
                                                    src_diff_desc,
                                                    src_diff_ptr,
                                                    dest_desc,
                                                    dest_ptr,
                                                    dest_diff_desc,
                                                    mut dest_diff_data| {
            self.derive_common_base(cudnn,
                                    src_desc,
                                    src_ptr,
                                    src_diff_desc,
                                    src_diff_ptr,
                                    dest_desc,
                                    dest_ptr,
                                    dest_diff_desc,
                                    dest_diff_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
            dest_diff_data
        })
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, _: &L) -> bool {
        false
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for Tanh<U,DeviceCpu<U>>
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
impl<U> ActivationCommonBase<U,DeviceGpu<U>> for SoftMax<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common_base(&self, cudnn:&Cudnn,
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

    fn derive_common_base(&self, cudnn:&Cudnn,
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
}
impl<U,const N:usize> ActivationCommon<U,Arr<U,N>,DeviceGpu<U>> for SoftMax<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U> {

    fn apply_common(&self, input:&Arr<U,N>, device:&DeviceGpu<U>) -> Arr<U,N> {
        device.linear_activation_forward(input,|cudnn,src_desc,src_ptr,dest_desc,mut dest_data| {
            self.apply_common_base(cudnn,
                                   src_desc,
                                   src_ptr,
                                   dest_desc,
                                   dest_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
            dest_data
        })
    }

    fn derive_common(&self, o:&Arr<U,N>, loss:&Arr<U,N>, u:&Arr<U,N>, device:&DeviceGpu<U>) -> Arr<U,N> {
        device.linear_activation_backward(o,loss,u,|cudnn,
                                                    src_desc,
                                                    src_ptr,
                                                    src_diff_desc,
                                                    src_diff_ptr,
                                                    dest_desc,
                                                    dest_ptr,
                                                    dest_diff_desc,
                                                    mut dest_diff_data| {
            self.derive_common_base(cudnn,
                                    src_desc,
                                    src_ptr,
                                    src_diff_desc,
                                    src_diff_ptr,
                                    dest_desc,
                                    dest_ptr,
                                    dest_diff_desc,
                                    dest_diff_data.as_raw_mut_slice().as_mut_ptr() as *mut U as *mut libc::c_void);
            dest_diff_data
        })
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceCpu<U>> for SoftMax<U,DeviceCpu<U>>
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
