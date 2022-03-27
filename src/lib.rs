pub mod arr;
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
pub trait Forward<O> {
    type Input;
    fn forward(&self,input:&Self::Input) -> O;
}
pub trait ForwardAll {
    type Input;
    type Output;
    fn forward_all(&self,input:Self::Input) -> Self::Output;
}
pub trait Backward<I,U,OP: Optimizer<U>> {
    fn backward(&mut self,input:I,optimizer:&mut OP);
}
pub trait Train<U,OP>: ForwardAll where OP: Optimizer<U> {
    type OutStack: Stack<Head=Self::Output>;
    fn train(&mut self, input:Self::Input, optimizer:&mut OP) -> Self::OutStack;
}
pub trait Optimizer<U> {
    fn update(&mut self,w:&U,e:U) -> U;
}
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
