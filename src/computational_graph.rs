use std::marker::PhantomData;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator, IndexedParallelIterator, IntoParallelIterator};
use crate::arr::{Arr, VecArr};
use crate::error::{SizeMismatchError, TrainingError};
use crate::ope::UnitValue;

pub trait GraphNode<FI,FO,BI,BO> {
    fn forward(v:FI) -> FO;

    fn backward(d:BI) -> BO;
}
pub struct AddNode<U> where U: UnitValue<U> {
    u:PhantomData<U>
}
impl<U> AddNode<U> where U: UnitValue<U> {
    pub fn new() -> AddNode<U> {
        AddNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<(U,U),U,U,(U,U)> for AddNode<U> where U: UnitValue<U> {
    fn forward((l,r):(U,U)) -> U {
        l + r
    }

    fn backward(d:U) -> (U,U) {
        (d,d)
    }
}
pub struct MulNode<U> where U: UnitValue<U> {
    u:PhantomData<U>
}
impl<U> MulNode<U> where U: UnitValue<U> {
    pub fn new() -> MulNode<U> {
        MulNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<(U,U),U,(U,U,U),(U,U)> for MulNode<U> where U: UnitValue<U> {
    fn forward((l,r):(U,U)) -> U {
        l * r
    }

    fn backward((l,r,d):(U,U,U)) -> (U,U) {
        (r * d, l * d)
    }
}
pub struct BranchNode<U> where U: UnitValue<U> {
    u:PhantomData<U>
}
impl<U> BranchNode<U> where U: UnitValue<U> {
    pub fn new() -> BranchNode<U> {
        BranchNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<U,(U,U),(U,U),U> for BranchNode<U> where U: UnitValue<U> {
    fn forward(v:U) -> (U,U) {
        (v,v)
    }

    fn backward((d1,d2):(U,U)) -> U {
        d1 + d2
    }
}
pub struct SumNode<U> where U: UnitValue<U> {
    u:PhantomData<U>
}
impl<U> SumNode<U> where U: UnitValue<U> {
    pub fn new() -> SumNode<U> {
        SumNode {
            u:PhantomData::<U>
        }
    }
}
impl<U,const N:usize> GraphNode<VecArr<U,Arr<U,N>>,Result<Arr<U,N>,SizeMismatchError>,(Arr<U,N>,usize),Result<VecArr<U,Arr<U,N>>,TrainingError>>
    for SumNode<U> where U: UnitValue<U> {

    fn forward(v: VecArr<U, Arr<U, N>>) -> Result<Arr<U, N>,SizeMismatchError> {
        v.par_iter().fold(|| vec![U::default();N],|acc,arr| {
            acc.par_iter().zip(arr.par_iter()).map(|(&a,&b)| a + b).collect::<Vec<U>>()
        }).reduce(|| vec![U::default();N],|acc,arr| {
            acc.par_iter().zip(arr.par_iter()).map(|(&a,&b)| a + b).collect::<Vec<U>>()
        }).try_into()
    }

    fn backward((d,n): (Arr<U, N>,usize)) -> Result<VecArr<U, Arr<U,N>>,TrainingError> {
        Ok((0..n).into_par_iter().map(|_| {
            d.par_iter().map(|&d| U::from_usize(n).ok_or(TrainingError::TypeCastError(
                String::from("An error occurred when casting the batch size data type")
            )).map(|n| d / n)).collect::<Result<Vec<U>,_>>().and_then(|r| {
                r.try_into().map_err(|e| TrainingError::from(e))
            })
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }
}
pub struct BroadcastNode<U> where U: UnitValue<U> {
    u:PhantomData<U>
}
impl<U> BroadcastNode<U> where U: UnitValue<U> {
    pub fn new() -> BroadcastNode<U> {
        BroadcastNode {
            u:PhantomData::<U>
        }
    }
}
impl<U,const N:usize> GraphNode<(Arr<U,N>,usize),VecArr<U,Arr<U,N>>,VecArr<U,Arr<U,N>>,Result<Arr<U,N>,SizeMismatchError>>
    for BroadcastNode<U> where U: UnitValue<U> {

    fn forward((v,n): (Arr<U, N>,usize)) -> VecArr<U, Arr<U, N>> {
        (0..n).into_par_iter().map(|_| v.clone()).collect::<Vec<_>>().into()
    }

    fn backward(d: VecArr<U, Arr<U, N>>) -> Result<Arr<U, N>,SizeMismatchError> {
        d.par_iter().fold(|| vec![U::default();N],|acc,arr| {
            acc.par_iter().zip(arr.par_iter()).map(|(&a,&b)| a + b).collect::<Vec<U>>()
        }).reduce(|| vec![U::default();N],|acc,arr| {
            acc.par_iter().zip(arr.par_iter()).map(|(&a,&b)| a + b).collect::<Vec<U>>()
        }).try_into()
    }
}
pub struct ReciprocalNode<U> where U: UnitValue<U> {
    u:PhantomData<U>
}
impl<U> ReciprocalNode<U> where U: UnitValue<U> {
    pub fn new() -> ReciprocalNode<U> {
        ReciprocalNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<U,U,U,U> for ReciprocalNode<U> where U: UnitValue<U> {
    fn forward(v: U) -> U {
        U::one() / v
    }

    fn backward(d: U) -> U {
        -(U::one() / (d * d))
    }
}
pub struct SqrtNode<U> where U: UnitValue<U> {
    u:PhantomData<U>
}
impl<U> SqrtNode<U> where U: UnitValue<U> {
    pub fn new() -> SqrtNode<U> {
        SqrtNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<U,U,U,U> for SqrtNode<U> where U: UnitValue<U> {
    fn forward(v: U) -> U {
        v.sqrt()
    }

    fn backward(d: U) -> U {
        U::one() / ((U::one() + U::one()) * d.sqrt())
    }
}
pub struct SquareNode<U> where U: UnitValue<U> {
    u:PhantomData<U>
}
impl<U> SquareNode<U> where U: UnitValue<U> {
    pub fn new() -> SquareNode<U> {
        SquareNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<U,U,(U,U),U> for SquareNode<U> where U: UnitValue<U> {
    fn forward(v: U) -> U {
        v * v
    }

    fn backward((i,d): (U, U)) -> U {
        (U::one() + U::one()) * i * d
    }
}
pub struct SubNode<U> where U: UnitValue<U> {
    u:PhantomData<U>
}
impl<U> SubNode<U> where U: UnitValue<U> {
    pub fn new() -> SubNode<U> {
        SubNode {
            u:PhantomData::<U>
        }
    }
}
impl<U> GraphNode<(U,U),U,U,(U,U)> for SubNode<U> where U: UnitValue<U> {
    fn forward((l,r): (U, U)) -> U {
        l - r
    }

    fn backward(d: U) -> (U,U) {
        (d,-d)
    }
}