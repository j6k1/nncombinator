use std::collections::HashSet;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::c_uint;
use cuda_runtime_sys::dim3;
use crate::UnitValue;
use crate::arr::*;
use crate::cuda::{CudaPtr, DataTypeInfo, Kernel, KernelArgs, Memory};
use crate::cuda::kernel::activation::{
    ActivationBackwardArgs,
    ActivationForwardArgs,
    ReLuBackward,
    ReLuForward,
    SigmoidBackward,
    SigmoidForward,
    SoftMaxBackward,
    SoftMaxForward,
    SwishBackward,
    SwishForward,
    TanhBackward,
    TanhForward
};
use crate::device::*;
use crate::error::{CudaError, EvaluateError, TrainingError};
use crate::lossfunction::LossFunction;
use crate::mem::AsRawSlice;

pub trait Activation<U,T,D> where U: UnitValue<U>, D: Device<U> {
    fn apply(&self, device:&D, input:&T) -> Result<T, EvaluateError>;
    fn derive(&self, device:&D, o:&T, loss:&T, u:&T) -> Result<T, TrainingError>;
    fn is_canonical_link<L: LossFunction<U>>(&self,l:&L) -> bool;
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
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceGpu<U>> for Identity<U,DeviceGpu<U>>
    where U: UnitValue<U>, DeviceGpu<U>: Device<U> {

    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        Ok((*input).clone())
    }

    fn derive(&self, _: &DeviceGpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, _: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
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
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceGpu<U>> for Sigmoid<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
             SigmoidForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
             SigmoidBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = SigmoidForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(u_ptr, loss_ptr, N,1);

        let mut kernel = SigmoidBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
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
impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceGpu<U>> for ReLu<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          ReLuForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          ReLuBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {
    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {

        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = ReLuForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(u_ptr, loss_ptr, N,1);

        let mut kernel = ReLuBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
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
            // Todo +rが常に0の状態でrにかけてる。代りに*iと書くか、*rの値を*iで初期化する必要がある。
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
 impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceGpu<U>> for Swish<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
             SwishForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
             SwishBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = SwishForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(u_ptr, loss_ptr, N,1);

        let mut kernel = SwishBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
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
 impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceGpu<U>> for Tanh<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo, DeviceGpu<U>: Device<U>,
             TanhForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
             TanhBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N,1);

        let mut kernel = TanhForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(u_ptr, loss_ptr, N,1);

        let mut kernel = TanhBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
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
 impl<U,const N:usize> Activation<U,Arr<U,N>,DeviceGpu<U>> for SoftMax<U,DeviceGpu<U>>
    where U: UnitValue<U> + DataTypeInfo,
          DeviceGpu<U>: Device<U>,
          CudaPtr<U>: TryFrom<U,Error=CudaError>,
          SoftMaxForward<U>: Kernel<Args=ActivationForwardArgs<U>>,
          SoftMaxBackward<U>: Kernel<Args=ActivationBackwardArgs<U>> {

    fn apply(&self, _: &DeviceGpu<U>, input: &Arr<U,N>) -> Result<Arr<U,N>, EvaluateError> {
        let mut input_output: CudaPtr<U> = CudaPtr::new(N)?;
        input_output.memcpy(input.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationForwardArgs::new(input_output, N, 1);

        let mut kernel = SoftMaxForward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024 * 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 1024 * mem::size_of::<U>() * 2)?;

        Ok(args.input_output.read_to_vec()?.try_into()?)
    }

    fn derive(&self, _: &DeviceGpu<U>, _: &Arr<U,N>, loss: &Arr<U,N>, u: &Arr<U,N>) -> Result<Arr<U,N>, TrainingError> {
        let mut u_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        u_ptr.memcpy(u.as_raw_slice().as_ptr(), N)?;

        let mut loss_ptr: CudaPtr<U> = CudaPtr::new(N)?;
        loss_ptr.memcpy(loss.as_raw_slice().as_ptr(), N)?;

        let mut args = ActivationBackwardArgs::new(u_ptr, loss_ptr, N,1);

        let mut kernel = SoftMaxBackward::<U>::new();

        kernel.launch(dim3 { x: (N as c_uint + 1023) / 1024, y: 1, z: 1 },
                      dim3 { x: 1024, y: 1, z: 1 },
                      &mut args, 0).unwrap();

        Ok(args.loss.read_to_vec()?.try_into()?)
    }

    fn is_canonical_link<L: LossFunction<U>>(&self, l: &L) -> bool {
        self.c.contains(l.name())
    }
}
