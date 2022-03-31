use crate::ope::UnitValue;

pub mod ope;
pub mod arr;
pub mod optimizer;
pub mod lossfunction;
pub mod activation;
pub mod device;
pub mod layer;

pub trait Stack {
    type Remaining: Stack;
    type Head;
    fn pop(self) -> (Self::Remaining, Option<Self::Head>);
    fn map<F: FnOnce(&Self::Head) -> O,O>(&self,f:F) -> O;
}
pub struct Cons<R,T>(pub R,pub T) where R: Stack;

impl<R,T> Stack for Cons<R,T> where R: Stack{
    type Remaining = R;
    type Head = T;

    fn pop(self) -> (Self::Remaining, Option<Self::Head>) {
        match self {
            Cons(parent,head) => {
                (parent,Some(head))
            }
        }
    }

    fn map<F: FnOnce(&Self::Head) -> O,O>(&self,f:F) -> O {
        f(&self.1)
    }
}
pub struct Nil;

impl Stack for Nil {
    type Remaining = Nil;
    type Head = ();
    fn pop(self) -> (Self::Remaining, Option<Self::Head>) {
        (Nil,None)
    }

    fn map<F: FnOnce(&Self::Head) -> O,O>(&self,f:F) -> O {
        f(&())
    }
}
#[cfg(test)]
mod tests {
    use crate::arr::Arr;
    use crate::device::DeviceCpu;
    use crate::layer::{AddLayer, AddLayerTrain, InputLayer, LinearLayer};

    #[test]
    fn build_layers() {
        let mut i:InputLayer<f32,Arr<f32,4>> = InputLayer::new();
        let device = DeviceCpu::new();

        let l = i.add_layer(|l| LinearLayer::<_,_,_,4,1>::new(l,&device, || 1., || 0.));
    }

    #[test]
    fn build_train_layers() {
        let mut i:InputLayer<f32,Arr<f32,4>> = InputLayer::new();
        let device = DeviceCpu::new();

        let l = i.add_layer_train(|l| LinearLayer::<_,_,_,4,1>::new(l,&device,|| 1., || 0.));
    }
}
