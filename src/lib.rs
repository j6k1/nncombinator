use std::marker::PhantomData;
use crate::ope::UnitValue;

pub mod error;
pub mod ope;
pub mod arr;
pub mod optimizer;
pub mod lossfunction;
pub mod activation;
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
#[derive(Debug)]
pub struct Cons<R,T>(pub R,pub T) where R: Stack;

pub trait Node<T> {
    type Index;
    fn get_index(&self) -> Self::Index;
}

impl Node<Nil> for Nil {
    type Index = Here;
    fn get_index(&self) -> Self::Index {
        Here
    }
}
impl<R,T> Node<Cons<R,T>> for Cons<R,T> where R: Node<R> + Stack {
    type Index = There<R::Index>;
    fn get_index(&self) -> Self::Index {
        There(self.0.get_index())
    }
}
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
#[derive(Debug)]
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

pub struct There<T>(T);
pub struct Here;

pub trait Same<T> {}

pub struct Pair<T1,T2>(PhantomData::<T1>,PhantomData::<T2>);

impl<T> Same<T> for Pair<T,T> {}

pub trait FindBase<T,P>: Stack {
    fn get_head(&self) -> &T;
    fn get_head_mut(&mut self) -> &mut T;
}
pub trait Find<T,P>: FindBase<T,P> {
    fn get(&self) -> &T {
        self.get_head()
    }

    fn get_mut(&mut self) -> &mut T {
        self.get_head_mut()
    }
}
impl<T,R,TailIndex> Find<T,Pair<TailIndex,TailIndex>> for Cons<R,T>
    where R: Find<T,Pair<TailIndex,TailIndex>>,
          Pair<TailIndex,TailIndex>: Same<TailIndex> {
    fn get(&self) -> &T {
        &self.1
    }

    fn get_mut(&mut self) -> &mut T {
        &mut self.1
    }
}
impl<T,R,TailIndex> FindBase<T,Pair<TailIndex,TailIndex>> for Cons<R,T>
    where R: Find<T,Pair<TailIndex,TailIndex>> {

    fn get_head(&self) -> &T {
        self.0.get()
    }
    fn get_head_mut(&mut self) -> &mut T {
        self.0.get_mut()
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
        let i:InputLayer<f32,Arr<f32,4>> = InputLayer::new();
        let device = DeviceCpu::new();

        let _l = i.add_layer(|l| LinearLayer::<_,_,_,_,4,1>::new(l,&device, || 1., || 0.));
    }

    #[test]
    fn build_train_layers() {
        let i:InputLayer<f32,Arr<f32,4>> = InputLayer::new();
        let device = DeviceCpu::new();

        let _l = i.add_layer(|l| {
            LinearLayer::<_,_,_,_,4,1>::new(l,&device,|| 1., || 0.)
        }).add_layer(|l| {
            ActivationLayer::new(l,ReLu::new(&device),&device)
        }).add_layer_train(|l| LinearOutputLayer::new(l,&device));
    }
}
