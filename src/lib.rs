//! nncombinator is a neural network library that allows type-safe implementation.

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
pub mod collection;
pub mod list;
pub mod optimizer;
pub mod lossfunction;
pub mod activation;
pub mod cuda;
pub mod device;
pub mod computational_graph;
pub mod layer;
pub mod persistence;

/// Trait that defines a stack to store the results computed by forward propagation when training a neural network.
pub trait Stack {
    /// Stack containing elements that do not include the top element of the stack
    type Remaining: Stack;
    /// Top element of the stack
    type Head;

    /// Returns a tuple of the top item in the stack and the rest of the stack
    fn pop(self) -> (Self::Remaining, Self::Head);
    /// Returns Cons with items pushed to the stack
    /// # Arguments
    /// * `head` - Items to be added
    fn push<H>(self,head:H) -> Cons<Self,H> where Self: Sized;
    /// Returns the result of applying the callback function to the top element of the stack
    /// # Arguments
    /// * `f` - Applicable callbacks
    fn map<F: FnOnce(&Self::Head) -> O,O>(&self,f:F) -> O;
    /// Returns the result of applying the callback to a stack that does not contain the top element of the stack
    /// * `f` - Applicable callbacks
    fn map_remaining<F: FnOnce(&Self::Remaining) -> O,O>(&self,f:F) -> O;
    /// Returns the result of taking ownership of the first element of the stack and applying the callback function.
    /// # Arguments
    /// * `f` - Applicable callbacks
    fn take_map<F: FnOnce(Self::Head) -> (Self::Head,O),O>(self,f:F) -> (Self,O) where Self: Sized;
}
/// Stack containing elements
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

    fn push<H>(self,head:H) -> Cons<Self, H> where Self: Sized {
        Cons(self,head)
    }

    fn map<F: FnOnce(&Self::Head) -> O,O>(&self,f:F) -> O {
        f(&self.1)
    }
    fn map_remaining<F: FnOnce(&Self::Remaining) -> O,O>(&self,f:F) -> O { f(&self.0) }
    fn take_map<F: FnOnce(Self::Head) -> (Self::Head, O), O>(self, f: F) -> (Self, O) where Self: Sized {
        let (s,h) = self.pop();

        let (h,r) = f(h);

        (Cons(s,h),r)
    }
}
/// Empty stack, containing no elements
#[derive(Debug,Clone)]
pub struct Nil;

impl Stack for Nil {
    type Remaining = Nil;
    type Head = ();
    fn pop(self) -> (Self::Remaining, Self::Head) {
        (Nil,())
    }

    fn push<H>(self, head: H) -> Cons<Self, H> where Self: Sized {
        Cons(Nil,head)
    }

    fn map<F: FnOnce(&Self::Head) -> O,O>(&self,f:F) -> O {
        f(&())
    }
    fn map_remaining<F: FnOnce(&Self::Remaining) -> O, O>(&self, f: F) -> O {
        f(&Nil)
    }
    fn take_map<F: FnOnce(Self::Head) -> (Self::Head, O), O>(self, f: F) -> (Self, O) where Self: Sized {
        let (_,r) = f(());

        (Nil,r)
    }
}
#[cfg(test)]
mod tests {
    use crate::activation::ReLu;
    use crate::arr::Arr;
    use crate::device::DeviceCpu;
    use crate::layer::{AddLayer, AddLayerTrain};
    use crate::layer::activation::ActivationLayer;
    use crate::layer::input::InputLayer;
    use crate::layer::linear::LinearLayer;
    use crate::layer::output::LinearOutputLayer;

    #[test]
    fn build_layers() {
        let i:InputLayer<f32,Arr<f32,4>,_> = InputLayer::new();
        let device = DeviceCpu::new().unwrap();

        let _l = i.add_layer(|l| LinearLayer::<_,_,_,DeviceCpu<f32>,_,4,1>::new(l,&device, || 1., || 0.));
    }

    #[test]
    fn build_train_layers() {
        let i:InputLayer<f32,Arr<f32,4>,_> = InputLayer::new();
        let device = DeviceCpu::new().unwrap();

        let _l = i.add_layer(|l| {
            LinearLayer::<_,_,_,DeviceCpu<f32>,_,4,1>::new(l,&device,|| 1., || 0.)
        }).add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).add_layer_train(|l| LinearOutputLayer::new(l,&device));
    }
}
