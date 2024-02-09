//! Computational graph implementation
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use num_traits::FromPrimitive;
use crate::arr::{AsView, MakeView, SerializedVec, SerializedVecView, SliceSize};
use crate::ope::{One, Sqrt, Sum};

/// Trait that defines a computational graph for calculating forward and back propagation of a neural network
pub trait GraphNode<FI,FO,BI,BO> {
    /// Forward propagation calculation
    /// # Arguments
    /// * `v` - forward input value.
    ///
    fn forward(&self,v:FI) -> FO;

    /// Back propagation calculation
    /// # Arguments
    /// * `d` - backward input value.
    ///
    fn backward(&self,d:BI) -> BO;
}
/// Implementation of additive nodes
pub struct AddNode<U> where U: Add<Output = U> + Clone {
    u:PhantomData<U>
}
impl<U> AddNode<U> where U: Add<Output = U> + Clone {
    pub fn new() -> AddNode<U> {
        AddNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<(U,U),U,U,(U,U)> for AddNode<U> where U: Add<Output = U> + Clone {
    fn forward(&self,(l,r):(U,U)) -> U {
        l + r
    }

    fn backward(&self,d:U) -> (U,U) {
        (d.clone(),d)
    }
}
/// Multiplication node implementation
pub struct MulNode<U> where U: Mul<Output = U> + Clone {
    u:PhantomData<U>
}
impl<U> MulNode<U> where U: Mul<Output = U> + Clone {
    pub fn new() -> MulNode<U> {
        MulNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<(U,U),U,(U,U,U),(U,U)> for MulNode<U> where U: Mul<Output = U> + Clone {
    fn forward(&self,(l,r):(U,U)) -> U {
        l * r
    }

    fn backward(&self,(l,r,d):(U,U,U)) -> (U,U) {
        (r * d.clone(), l * d)
    }
}
/// Branch node implementation
pub struct BranchNode<U> where U: Add<Output = U> + Clone {
    u:PhantomData<U>
}
impl<U> BranchNode<U> where U: Add<Output = U> + Clone {
    pub fn new() -> BranchNode<U> {
        BranchNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<U,(U,U),(U,U),U> for BranchNode<U> where U: Add<Output = U> + Clone {
    fn forward(&self,v:U) -> (U,U) {
        (v.clone(),v)
    }

    fn backward(&self,(d1,d2):(U,U)) -> U {
        d1 + d2
    }
}
/// Sum node implementation
pub struct SumNode<U,C> where U: Default + Clone + Send + Sync {
    u:PhantomData<U>,
    c:PhantomData<C>
}
impl<U,C> SumNode<U,C> where U: Default + Clone + Send + Sync {
    pub fn new() -> SumNode<U,C> {
        SumNode {
            u:PhantomData::<U>,
            c:PhantomData::<C>
        }
    }
}
impl<U,T> GraphNode<&SerializedVec<U,T>,T,(&T,usize),SerializedVec<U,T>> for SumNode<U,SerializedVec<U,T>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output=U> + 'static,
          for<'a> T: SliceSize + AsView<'a> + MakeView<'a,U> + Default + Clone + Send + Sync +
                  Add<Output=T> + Add<<T as AsView<'a>>::ViewType,Output=T>,
          for<'a> <T as AsView<'a>>::ViewType: Send,
          SerializedVec<U,T>: From<Vec<T>> {
    fn forward(&self,v: &SerializedVec<U,T>) -> T {
        v.sum()
    }

    fn backward(&self,(d,n): (&T,usize)) -> SerializedVec<U,T> {
        (0..n).map(|_| {
            d.clone().into()
        }).collect::<Vec<T>>().into()
    }
}
impl<'a,U,T> GraphNode<SerializedVecView<'a,U,T>,T,(&T,usize),SerializedVec<U,T>> for SumNode<U,SerializedVecView<'a,U,T>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output=U> + 'static,
          for<'b> T: SliceSize + AsView<'b> + MakeView<'b,U> +
                  Default + Clone + Send + Sync +
                  Add<Output=T> + Add<<T as AsView<'b>>::ViewType,Output=T>,
          for<'b> <T as AsView<'b>>::ViewType: Send,
          SerializedVec<U,T>: From<Vec<T>> {
    fn forward(&self,v: SerializedVecView<'a,U,T>) -> T {
        v.sum()
    }

    fn backward(&self,(d,n): (&T,usize)) -> SerializedVec<U,T> {
        (0..n).map(|_| {
            d.clone().into()
        }).collect::<Vec<T>>().into()
    }
}
/// Broadcast node implementation
pub struct BroadcastNode<U,C> where U: Default + Clone + Send + Sync {
    u:PhantomData<U>,
    c:PhantomData<C>
}
impl<U,C> BroadcastNode<U,C> where U: Default + Clone + Send + Sync {
    pub fn new() -> BroadcastNode<U,C> {
        BroadcastNode {
            u:PhantomData::<U>,
            c:PhantomData::<C>
        }
    }
}
impl<U,T> GraphNode<(&T,usize),SerializedVec<U,T>,&SerializedVec<U,T>,T> for BroadcastNode<U,&SerializedVec<U,T>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output=U> + 'static,
          for<'a> T: SliceSize + AsView<'a> + MakeView<'a,U> + Default + Clone + Send + Sync +
                  Add<Output=T> + Add<<T as AsView<'a>>::ViewType,Output=T>,
          for<'a> <T as AsView<'a>>::ViewType: Send,
          SerializedVec<U,T>: From<Vec<T>> {
    fn forward(&self,(v,n): (&T,usize)) -> SerializedVec<U,T> {
        (0..n).map(|_| v.clone()).collect::<Vec<_>>().into()
    }

    fn backward(&self,d: &SerializedVec<U,T>) -> T {
        d.sum()
    }
}
impl<'b,U,T> GraphNode<(&T,usize),SerializedVec<U,T>,SerializedVecView<'b,U,T>,T> for BroadcastNode<U,SerializedVecView<'b,U,T>>
    where U: Default + Clone + Copy + Send + Sync + Add<Output=U> + 'static,
          for<'a> T: SliceSize + AsView<'a> + MakeView<'a,U> + Default + Clone + Send + Sync +
                  Add<Output=T> + Add<<T as AsView<'a>>::ViewType,Output=T>,
          for<'a> <T as AsView<'a>>::ViewType: Send,
          SerializedVec<U,T>: From<Vec<T>> {
    fn forward(&self,(v,n): (&T,usize)) -> SerializedVec<U,T> {
        (0..n).map(|_| v.clone()).collect::<Vec<_>>().into()
    }

    fn backward(&self,d: SerializedVecView<'b,U,T>) -> T {
        d.sum()
    }
}
/// Implementation of reciprocal nodes
pub struct ReciprocalNode<U> where U: Div + Div<Output = U> + Mul + Mul<Output = U> + Neg {
    u:PhantomData<U>
}
impl<U> ReciprocalNode<U> where U: Div + Div<Output = U> + Mul + Mul<Output = U> + Neg {
    pub fn new() -> ReciprocalNode<U> {
        ReciprocalNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<U,U,U,U> for ReciprocalNode<U>
    where U: Div + Div<Output = U> + Neg + Neg<Output = U> + One + Mul + Mul<Output = U> + One + Clone + Copy {
    fn forward(&self,v: U) -> U {
        U::one() / v
    }

    fn backward(&self,d: U) -> U {
        -(U::one() / (d * d))
    }
}
/// Square root node implementation
pub struct SqrtNode<U> where U: Sqrt + Div + Div<Output = U> + FromPrimitive {
    u:PhantomData<U>
}
impl<U> SqrtNode<U> where U: Sqrt + Div + Div<Output = U> + FromPrimitive {
    pub fn new() -> SqrtNode<U> {
        SqrtNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<U,U,U,U> for SqrtNode<U>
    where U: Sqrt + Div + Div<Output = U> + Mul + Mul<Output = U> + One + FromPrimitive {

    fn forward(&self,v: U) -> U {
        v.sqrt()
    }

    fn backward(&self,d: U) -> U {
        U::one() / (U::from_f64(2.).expect("Error in type conversion from f64.") * d.sqrt())
    }
}
/// Squared node implementation
pub struct SquareNode<U> where U: FromPrimitive + Mul + Mul<Output = U> {
    u:PhantomData<U>
}
impl<U> SquareNode<U> where U: FromPrimitive + Mul + Mul<Output = U> {
    pub fn new() -> SquareNode<U> {
        SquareNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<U,U,(U,U),U> for SquareNode<U>
    where U: FromPrimitive + Mul + Mul<Output = U> + Clone + Copy {

    fn forward(&self,v: U) -> U {
        v * v
    }

    fn backward(&self,(i,d): (U, U)) -> U {
        U::from_f64(2.).expect("Error in type conversion from f64.") * i * d
    }
}
/// Implementation of negative additive nodes
pub struct SubNode<U> where U: Sub + Sub<Output = U> + Neg + Clone {
    u:PhantomData<U>
}
impl<U> SubNode<U> where U: Sub + Sub<Output = U> + Neg + Clone{
    pub fn new() -> SubNode<U> {
        SubNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<(U,U),U,U,(U,U)> for SubNode<U> where U: Sub + Sub<Output = U> + Neg + Neg<Output = U> + Clone {
    fn forward(&self,(l,r): (U, U)) -> U {
        l - r
    }

    fn backward(&self,d: U) -> (U,U) {
        (d.clone(),-d)
    }
}