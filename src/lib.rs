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
#[macro_use]
mod macros;
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
    fn take_map<F: FnOnce(Self::Head) -> Result<(Self::Head, O),E>, O,E>(self, f: F) -> Result<(Self, O),E> where Self: Sized;
    /// Pass a mutable reference to the top element of the stack to the callback function and return the result of executing it
    /// # Arguments
    /// * `f` - Applicable callbacks
    fn map_mut<F: FnOnce(&mut Self::Head) -> O,O>(&mut self,f:F) -> O;
}
/// Stack containing elements
#[derive(Debug,Clone)]
pub struct Cons<R,T>(pub R,pub T) where R: Stack;

impl<R,T> Cons<R,T> where R: Stack {
    /// Returns a reference to the remaining items in the stack, not including the top item in the stack.
    #[inline]
    pub fn get_remaining(&self) -> &R {
        match self {
            &Cons(ref parent,_) => {
                parent
            }
        }
    }

    /// Returns a reference to the top item on the stack
    #[inline]
    pub fn get_head(&self) -> &T {
        match self {
            &Cons(_, ref head) => {
                head
            }
        }
    }
}
impl<R,T> Stack for Cons<R,T> where R: Stack {
    type Remaining = R;
    type Head = T;

    #[inline]
    fn pop(self) -> (Self::Remaining, Self::Head) {
        match self {
            Cons(parent,head) => {
                (parent,head)
            }
        }
    }

    #[inline]
    fn push<H>(self,head:H) -> Cons<Self, H> where Self: Sized {
        Cons(self,head)
    }

    #[inline]
    fn map<F: FnOnce(&Self::Head) -> O,O>(&self,f:F) -> O {
        f(&self.1)
    }
    #[inline]
    fn map_remaining<F: FnOnce(&Self::Remaining) -> O,O>(&self,f:F) -> O { f(&self.0) }
    #[inline]
    fn take_map<F: FnOnce(Self::Head) -> Result<(Self::Head, O),E>, O,E>(self, f: F) -> Result<(Self, O),E> where Self: Sized {
        let (s,h) = self.pop();

        let (h,r) = f(h)?;

        Ok((Cons(s,h),r))
    }
    #[inline]
    fn map_mut<F: FnOnce(&mut Self::Head) -> O, O>(&mut self, f: F) -> O {
        f(&mut self.1)
    }
}
/// Empty stack, containing no elements
#[derive(Debug,Clone)]
pub struct Nil;

impl Stack for Nil {
    type Remaining = Nil;
    type Head = ();
    #[inline]
    fn pop(self) -> (Self::Remaining, Self::Head) {
        (Nil,())
    }

    #[inline]
    fn push<H>(self, head: H) -> Cons<Self, H> where Self: Sized {
        Cons(Nil,head)
    }

    #[inline]
    fn map<F: FnOnce(&Self::Head) -> O,O>(&self,f:F) -> O {
        f(&())
    }
    #[inline]
    fn map_remaining<F: FnOnce(&Self::Remaining) -> O, O>(&self, f: F) -> O {
        f(&Nil)
    }
    #[inline]
    fn take_map<F: FnOnce(Self::Head) -> Result<(Self::Head, O),E>, O,E>(self, f: F) -> Result<(Self, O),E> where Self: Sized {
        let (_,r) = f(())?;

        Ok((Nil,r))
    }
    #[inline]
    fn map_mut<F: FnOnce(&mut Self::Head) -> O, O>(&mut self, f: F) -> O {
        f(&mut ())
    }
}

#[cfg(test)]
mod tests {
    use crate::activation::ReLu;
    use crate::arr::Arr;
    use crate::device::DeviceCpu;
    use crate::layer::{AddLayer};
    use crate::layer::activation::ActivationLayer;
    use crate::layer::input::InputLayer;
    use crate::layer::linear::{LinearLayerBuilder};
    use crate::layer::output::LinearOutputLayer;
    use crate::optimizer::SGDBuilder;

    #[test]
    fn build_layers() {
        let device = DeviceCpu::new().unwrap();
        let i:InputLayer<f32,Arr<f32,4>,_,_> = InputLayer::new(&device);
        let optimizer_builder = SGDBuilder::new(&device).lr(0.01);

        let _l = i.add_layer(|l| LinearLayerBuilder::<4,1>::new().build(l,&device, || 1., || 0.,&optimizer_builder).unwrap());
    }

    #[test]
    fn build_train_layers() {
        let device = DeviceCpu::new().unwrap();
        let i:InputLayer<f32,Arr<f32,4>,_,_> = InputLayer::new(&device);
        let optimizer_builder = SGDBuilder::new(&device).lr(0.01);

        let _l = i.add_layer(|l| {
            LinearLayerBuilder::<4,1>::new().build(l,&device,|| 1., || 0.,&optimizer_builder).unwrap()
        }).add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).add_layer(|l| LinearOutputLayer::new(l,&device));
    }
}
