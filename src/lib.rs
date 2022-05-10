extern crate libc;
extern crate cuda_runtime_sys;
extern crate rcublas_sys;
extern crate rcublas;
extern crate rcudnn;
extern crate rcudnn_sys;

use crate::ope::UnitValue;

pub mod error;
pub mod ope;
pub mod mem;
pub mod arr;
pub mod list;
pub mod optimizer;
pub mod lossfunction;
pub mod activation;
pub mod cuda;
pub mod device;
pub mod layer;
pub mod persistence;

pub trait Stack {
    type Remaining: Stack;
    type Head;

    fn pop(self) -> (Self::Remaining, Self::Head);
    fn map<F: FnOnce(&Self::Head) -> O,O>(&self,f:F) -> O;
    fn map_remaining<F: FnOnce(&Self::Remaining) -> O,O>(&self,f:F) -> O;
}
#[derive(Debug,Clone)]
pub struct Cons<R,T>(pub R,pub T) where R: Stack;

impl<R,T> Stack for Cons<R,T> where R: Stack {
    type Remaining = R;
    type Head = T;

    fn pop(self) -> (Self::Remaining, Self::Head) {
        match self {
            Cons(parent,head) => {
                (parent,head)
            }
        }
    }

    fn map<F: FnOnce(&Self::Head) -> O,O>(&self,f:F) -> O {
        f(&self.1)
    }
    fn map_remaining<F: FnOnce(&Self::Remaining) -> O,O>(&self,f:F) -> O { f(&self.0) }
}
#[derive(Debug,Clone)]
pub struct Nil;

impl Stack for Nil {
    type Remaining = Nil;
    type Head = ();
    fn pop(self) -> (Self::Remaining, Self::Head) {
        (Nil,())
    }

    fn map<F: FnOnce(&Self::Head) -> O,O>(&self,f:F) -> O {
        f(&())
    }
    fn map_remaining<F: FnOnce(&Self::Remaining) -> O, O>(&self, f: F) -> O {
        f(&Nil)
    }
}
#[cfg(test)]
mod tests {
    use crate::activation::ReLu;
    use crate::arr::Arr;
    use crate::device::DeviceCpu;
    use crate::layer::{ActivationLayer, AddLayer, AddLayerTrain, InputLayer, LinearLayer, LinearOutputLayer};

    #[test]
    fn build_layers() {
        let i:InputLayer<f32,Arr<f32,4>,_> = InputLayer::new();
        let device = DeviceCpu::new().unwrap();

        let _l = i.add_layer(|l| LinearLayer::<_,_,_,_,4,1>::new(l,&device, || 1., || 0.));
    }

    #[test]
    fn build_train_layers() {
        let i:InputLayer<f32,Arr<f32,4>,_> = InputLayer::new();
        let device = DeviceCpu::new().unwrap();

        let _l = i.add_layer(|l| {
            LinearLayer::<_,_,_,_,4,1>::new(l,&device,|| 1., || 0.)
        }).add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).add_layer_train(|l| LinearOutputLayer::new(l,&device));
    }
}
