//! Computational graph implementation
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use num_traits::FromPrimitive;
use rayon::prelude::{ParallelIterator, IntoParallelIterator};
use crate::arr::{Arr, SerializedVec};
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
pub struct SumNode<U> where U: Default + Clone + Copy + Send + Sync {
    u:PhantomData<U>
}
impl<U> SumNode<U> where U: Default + Clone + Copy + Send + Sync {
    pub fn new() -> SumNode<U> {
        SumNode {
            u:PhantomData::<U>
        }
    }
}
impl<U,const N:usize> GraphNode<&SerializedVec<U,Arr<U,N>>,Arr<U,N>,(&Arr<U,N>,usize),SerializedVec<U,Arr<U,N>>>
    for SumNode<U> where U: Add + Add<Output = U> + Default + Clone + Copy + Send + Sync + 'static,
                         for<'a> &'a Arr<U,N>: Add<&'a Arr<U,N>,Output = Arr<U,N>> {
    fn forward(&self,v: &SerializedVec<U, Arr<U, N>>) -> Arr<U, N> {
        v.sum()
    }

    fn backward(&self,(d,n): (&Arr<U, N>,usize)) -> SerializedVec<U, Arr<U,N>> {
        (0..n).into_par_iter().map(|_| {
            d.clone().into()
        }).collect::<Vec<Arr<U,N>>>().into()
    }
}
/// Broadcast node implementation
pub struct BroadcastNode<U> where U: Default + Clone + Send {
    u:PhantomData<U>
}
impl<U> BroadcastNode<U> where U: Default + Clone + Send {
    pub fn new() -> BroadcastNode<U> {
        BroadcastNode {
            u:PhantomData::<U>
        }
    }
}
impl<U,const N:usize> GraphNode<(&Arr<U,N>,usize),SerializedVec<U,Arr<U,N>>,&SerializedVec<U,Arr<U,N>>,Arr<U,N>>
    for BroadcastNode<U> where U: Add + Add<Output = U> + Default + Clone + Copy + Send + Sync + 'static {
    fn forward(&self,(v,n): (&Arr<U, N>,usize)) -> SerializedVec<U, Arr<U, N>> {
        (0..n).into_par_iter().map(|_| v.clone()).collect::<Vec<_>>().into()
    }

    fn backward(&self,d: &SerializedVec<U, Arr<U, N>>) -> Arr<U, N> {
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