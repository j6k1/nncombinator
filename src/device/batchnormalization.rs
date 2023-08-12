use rayon::prelude::{ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};

use crate::arr::{Arr, VecArr};
use crate::ope::Sum;
use crate::collection::Broadcast;
use crate::computational_graph::{BroadcastNode, GraphNode, SqrtNode, SquareNode, SumNode};
use crate::device::DeviceCpu;
use crate::error::{EvaluateError, TrainingError};
use crate::ope::UnitValue;

/// Features defining the implementation of the various computational processes in the batch normalization layer
pub trait DeviceBatchNorm<U,T,C,const N:usize>
    where U: UnitValue<U> {
    fn forward_batch_norm(&self, input: &Arr<U,N>, scale: &C, bias: &C,
                          estimated_mean: &C, estimated_variance: &C) -> Result<Arr<U,N>,EvaluateError>;
    fn forward_batch_norm_train(&self, input: &Arr<U,N>, scale: &C, bias: &C,
                                estimated_mean: &C, estimated_variance: &C) -> Result<(Arr<U,N>,T,T),EvaluateError>;
    fn batch_forward_batch_norm(&self, input: &VecArr<U,Arr<U,N>>, scale: &C , bias: &C,
                                estimated_mean: &C, estimated_variance: &C) -> Result<VecArr<U,Arr<U,N>>,EvaluateError>;
    fn batch_forward_batch_norm_train(&self, input: &VecArr<U,Arr<U,N>>, scale: &C, bias: &C,
                                      running_mean: &C, running_variance: &C, momentum: U)
                                      -> Result<(VecArr<U,Arr<U,N>>,T,T,Arr<U,N>,Arr<U,N>),TrainingError>;
    fn backward_batch_norm(&self, loss:&Arr<U,N>, input: &Arr<U,N>, scale: &C,
                           saved_mean: &T, saved_inv_variance: &T) -> Result<(Arr<U, N>,Arr<U,N>,Arr<U,N>), TrainingError>;
    fn batch_backward_batch_norm(&self, loss:&VecArr<U,Arr<U,N>>, input: &VecArr<U,Arr<U,N>>,
                                 scale: &C, saved_mean: &T, saved_inv_variance: &T) -> Result<(VecArr<U,Arr<U, N>>,Arr<U,N>,Arr<U,N>), TrainingError>;
}
impl<U,const N:usize> DeviceBatchNorm<U,Arr<U,N>,Arr<U,N>,N> for DeviceCpu<U>
    where U: UnitValue<U> {
    fn forward_batch_norm(&self, input: &Arr<U, N>, scale: &Arr<U, N>, bias: &Arr<U, N>,
                          estimated_mean: &Arr<U, N>, estimated_variance: &Arr<U, N>) -> Result<Arr<U, N>,EvaluateError> {
        let eps = U::from_f64(1e-6).ok_or(EvaluateError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        Ok(input.par_iter()
            .zip(scale.par_iter())
            .zip(bias.par_iter())
            .zip(estimated_mean.par_iter())
            .zip(estimated_variance.par_iter())
            .map(|((((&i,&scale),&bias),&mean),&variance)| {
                scale * ((i - mean) / (variance + eps)) + bias
            }).collect::<Vec<U>>().try_into()?)
    }

    fn forward_batch_norm_train(&self, input: &Arr<U, N>,
                                scale: &Arr<U, N>,
                                bias: &Arr<U, N>,
                                estimated_mean: &Arr<U, N>,
                                estimated_variance: &Arr<U, N>) -> Result<(Arr<U,N>,Arr<U,N>,Arr<U,N>),EvaluateError> {
        let eps = U::from_f64(1e-6).ok_or(EvaluateError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        Ok((input.par_iter()
                .zip(scale.par_iter())
                .zip(bias.par_iter())
                .zip(estimated_mean.par_iter())
                .zip(estimated_variance.par_iter())
                .map(|((((&i,&scale),&bias),&mean),&variance)| {
                    scale * ((i - mean) / SqrtNode::new().forward(variance + eps)) + bias
                }).collect::<Vec<U>>().try_into()?,
            estimated_mean.clone(),
            estimated_variance.par_iter().map(|&v| U::one() / SqrtNode::new().forward(v + eps)).collect::<Vec<U>>().try_into()?
        ))
    }

    fn batch_forward_batch_norm(&self, input: &VecArr<U, Arr<U, N>>, scale: &Arr<U, N>, bias: &Arr<U, N>,
                                estimated_mean: &Arr<U, N>, estimated_variance: &Arr<U, N>) -> Result<VecArr<U, Arr<U, N>>, EvaluateError> {

        let eps = U::from_f64(1e-6).ok_or(EvaluateError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        Ok(input.par_iter().map(|input| {
            input.par_iter()
                .zip(scale.par_iter())
                .zip(bias.par_iter())
                .zip(estimated_mean.par_iter())
                .zip(estimated_variance.par_iter())
                .map(|((((&i,&scale),&bias),&mean),&variance)| {
                    scale * (i - mean) / SqrtNode::new().forward(variance + eps) + bias
                }).collect::<Vec<U>>().try_into()
        }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into())
    }

    fn batch_forward_batch_norm_train(&self, input: &VecArr<U, Arr<U, N>>,
                                      scale: &Arr<U, N>, bias: &Arr<U, N>,
                                      running_mean: &Arr<U, N>, running_variance: &Arr<U, N>,
                                      momentum: U)
                                      -> Result<(VecArr<U,Arr<U,N>>,Arr<U,N>,Arr<U,N>,Arr<U,N>,Arr<U,N>), TrainingError> {

        let eps = U::from_f64(1e-6).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let n = input.len();
        let un = U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let mean:Arr<U,N> = SumNode::new().forward(input) / un;

        let variance:VecArr<U,Arr<U,N>> = (input - Broadcast::<Arr<U,N>>(mean.clone()))
            .par_iter()
            .map(|i| {
                i.par_iter().map(|&i| {
                    SquareNode::new().forward(i)
                }).collect::<Vec<U>>().try_into()
            }).collect::<Result<Vec<Arr<U,N>>,_>>()?.into();
        let variance = variance.sum() / un;

        let inv_variance:Arr<U,N> = variance.par_iter().map(|&v| U::one() / SqrtNode::new().forward(v + eps)).collect::<Vec<U>>().try_into()?;

        let o:VecArr<U,Arr<U,N>> = Broadcast(inv_variance.clone()) * (input - Broadcast(mean.clone()));

        let running_mean = running_mean * momentum + &mean * (U::one() - momentum);
        let running_variance = running_variance * momentum + variance * (U::one() - momentum);

        let o = (BroadcastNode::new().forward((scale,n)) * o) + Broadcast(bias.clone());

        Ok((o,mean,inv_variance,running_mean,running_variance))
    }

    fn backward_batch_norm(&self, loss: &Arr<U, N>, input: &Arr<U, N>,
                           scale: &Arr<U, N>, saved_mean: &Arr<U, N>, saved_inv_variance: &Arr<U, N>)
                           -> Result<(Arr<U, N>, Arr<U, N>, Arr<U, N>), TrainingError> {
        let b = loss.clone();

        let x = input - saved_mean;

        let s = (&x * saved_inv_variance) * loss;

        let dx1 = scale * loss;
        let dx2 = &dx1 * saved_inv_variance;
        let dx3 = &x * dx1;
        let dx4 =  -(saved_inv_variance * saved_inv_variance) * dx3;
        let dx5 = dx4 * (saved_inv_variance / U::from_f64(2.).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from f64.")
        ))?);
        let dx6 = &x * dx5 * U::from_usize(2).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;
        let dx7 = dx2 + dx6;
        let dx8 = &dx7;
        let dx9 = -&dx7;
        let dx = dx8 + dx9;

        Ok((dx,s,b))
    }

    fn batch_backward_batch_norm(&self, loss: &VecArr<U, Arr<U, N>>,
                                 input: &VecArr<U,Arr<U,N>>,
                                 scale: &Arr<U, N>,
                                 saved_mean: &Arr<U, N>, saved_inv_variance: &Arr<U, N>)
                                 -> Result<(VecArr<U, Arr<U, N>>, Arr<U, N>, Arr<U, N>), TrainingError> {
        let n = input.len();

        let un = U::from_usize(n).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;

        let b = BroadcastNode::new().backward(loss);

        let x = BroadcastNode::new().forward((saved_mean,n));
        let x2 = input - &x;
        let iv = BroadcastNode::new().forward((saved_inv_variance,n));

        let s = BroadcastNode::new().backward(&(&x2 * &iv * loss));

        let dx1 = Broadcast(scale.clone()) * loss;
        let dx2 = &dx1 * iv;
        let dx3 = BroadcastNode::new().backward(&(&x2 * dx1));
        let dx4 = -(saved_inv_variance * saved_inv_variance) * dx3;
        let dx5 = dx4 * (saved_inv_variance / U::from_f64(2.).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from f64.")
        ))?);
        let dx6 = SumNode::new().backward((&(dx5 / un),n));
        let dx7 = x2 * dx6 * U::from_usize(2).ok_or(TrainingError::TypeCastError(String::from(
            "Error in type conversion from usize."
        )))?;
        let dx8 = dx2 + dx7;
        let dx9 = &dx8;
        let dx10 = -&dx8;
        let dx11 = BroadcastNode::new().backward(&dx10);
        let dx12 = SumNode::new().backward((&dx11,n)) / un;

        let dx = dx9 + dx12;

        Ok((dx,s,b))
    }
}
