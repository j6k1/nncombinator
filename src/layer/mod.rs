//! The various layers that make up a neural network and the traits they implement

use std::fmt::Debug;
use crate::arr::*;
use crate::device::*;
use crate::{Stack};
use crate::error::{DeviceError, EvaluateError, TrainingError};
use crate::ope::UnitValue;
use crate::lossfunction::*;
use crate::optimizer::*;

pub mod input;
pub mod output;
pub mod linear;
pub mod activation;
pub mod bridge;
pub mod batchnormalization;
pub mod bias;

/// Differential input
#[derive(Debug)]
pub enum DiffInput<T,U,const NI:usize,const NO:usize>
    where U: UnitValue<U> + Clone + Copy + Debug, T: Debug {
    /// diff input
    Diff(T,Arr<U,NO>),
    /// fully input
    NotDiff(Arr<U,NI>)
}
/// Trait that defines the data type during batch training corresponding to the data type
pub trait BatchDataType {
    type Type;
}
/// Trait that defines the ability to get the size of a batch
pub trait BatchSize {
    fn size(&self) -> usize;
}
/// Trait defining the internal implementation of forward propagation of a neural network
pub trait Forward<I,O> {
    /// Forward propagation implementation
    /// # Arguments
    /// * `input` - input
    fn forward(&self,input:&I) -> O;
}
/// Trait defining the implementation of forward propagation of neural networks
pub trait ForwardAll {
    /// Input to this layer of the neural network
    type Input: Debug;
    /// Output from this layer of the neural network
    type Output: Debug + Send + Sync + 'static;
    /// Forward propagation
    /// # Arguments
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn forward_all(&self, input:Self::Input) -> Result<Self::Output, EvaluateError>;
}
/// Trait defining the implementation of error back propagation in neural networks
pub trait BackwardAll<U>: PreTrain<U> + UpdateWeight<U> where U: UnitValue<U> {
    /// Losses during neural network training
    type LossInput: Debug;
    /// Losses in the top layer during neural network training
    type LossOutput: Debug;

    /// Back propagation of errors
    /// # Arguments
    /// * `input` - loss
    /// * `stack` - Stack to store calculation results at upper layers
    /// * `lossf` - loss function
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_all<L: LossFunction<U>>(&mut self, input:Self::LossInput, stack:Self::OutStack, lossf:&L)
        -> Result<(<Self as BackwardAll<U>>::LossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError>;
    fn is_canonical_link<L: LossFunction<U>>(&self,_:&L) -> bool {
        false
    }
}
/// Trait defining the calculation of the error during error back propagation.
pub trait Loss<U>: BackwardAll<U> where U: UnitValue<U> {
    /// Error Calculation
    /// # Arguments
    /// * `loss` - Lower layer error
    /// * `_lossf` - loss function
    /// * `stack` - Stack to store calculation results at upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn loss<L: LossFunction<U>>(&mut self, loss:Self::LossInput, _lossf:&L, stack:Self::OutStack) -> Result<(Self::OutStack, Self::LossInput), TrainingError> {
        Ok((stack,loss))
    }
}
/// Characteristics defining the internal implementation of the error back propagation method in neural networks
pub trait Backward<U,I,O> where U: UnitValue<U> {
    /// Back propagation of errors
    /// # Arguments
    /// * `input` - loss
    fn backward(&mut self, input:I) -> O;
}
/// Trait that defines the process of forward propagation performed prior to the process of error back propagation.
pub trait PreTrain<U>: ForwardAll where U: UnitValue<U> {
    /// Type of object to keep the results of forward propagation needed to perform error back propagation.
    type OutStack: Stack<Head=Self::Output> + Debug + Sized;
    /// Perform forward propagation required to perform error back propagation
    /// # Arguments
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn pre_train(&self, input:Self::Input) -> Result<Self::OutStack, EvaluateError>;
}
/// Trait defining the implementation of updating weights process in a neural network
pub trait UpdateWeight<U> where U: UnitValue<U> {
    /// Type of object that holds the gradient needed to update the weights of the units in each layer.
    type GradientStack: Stack + Debug + Sized;
    /// Type of object that holds the gradient needed to update the unit weights.
    /// # Arguments
    /// * `stack` - Stack to store calculation results at upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn update_weight<OP: Optimizer<U>>(&mut self, stack:Self::GradientStack, optimizer:&mut OP) -> Result<(), TrainingError>;
}
/// Trait that defines the function of differential application of inputs in the process of forward propagation to neural networks.
pub trait ForwardDiff<U>: PreTrain<U> where U: UnitValue<U> {
    /// Forward propagation (differential application)
    /// # Arguments
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn forward_diff(&self, input:Self::Input) -> Result<Self::OutStack, EvaluateError>;
}
/// Trait that defines the learning process of a neural network.
pub trait Train<U>: PreTrain<U> where U: UnitValue<U> {
    /// Train neural networks.
    /// # Arguments
    /// * `expected` - expected value
    /// * `input` - loss
    /// * `optimizer` - Optimizer object that implements the algorithm used to update the weights
    /// * `lossf` - loss function
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn train<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, expected:Self::Output, input:Self::Input, optimizer:&mut OP, lossf:&L) -> Result<U, TrainingError>;
}
/// Trait that defines the function to query information to calculate the difference when applying the difference of neural networks.
pub trait AskDiffInput<U>: PreTrain<U> where U: UnitValue<U> {
    /// Diff Input to this layer of the neural network
    type DiffInput: Debug;
    /// Data inquiry for creating difference information
    /// # Arguments
    /// * `stack` - Stack to store calculation results at upper layers
    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput;
}
/// Trait defining the relevant type of implementation of forward propagation of neural networks by batch processing.
pub trait BatchForwardBase: ForwardAll {
    /// Input to this layer of the neural network for batch execution
    type BatchInput: Debug;
    /// Output from this layer of the neural network for batch execution
    type BatchOutput: Debug;
}
/// Trait defining the implementation of forward propagation of neural networks by batch processing.
pub trait BatchForward: BatchForwardBase {
    /// Forward propagation
    /// # Arguments
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_forward(&self,input:Self::BatchInput) -> Result<Self::BatchOutput, TrainingError>;
}
/// Trait defining an implementation of error back propagation for neural networks with batch processing.
pub trait BatchBackward<U>: BatchPreTrainBase<U> + UpdateWeight<U> where U: UnitValue<U> {
    /// Losses during neural network training for batch execution
    type BatchLossInput: Debug;
    /// Losses in the top layer during neural network training
    type BatchLossOutput: Debug;
    /// Back propagation of errors
    /// # Arguments
    /// * `input` - loss
    /// * `stack` - Stack to store calculation results at upper layers
    /// * `lossf` - loss function
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward<L: LossFunction<U>>(&mut self, input:Self::BatchLossInput, stack:Self::BatchOutStack, lossf:&L)
        -> Result<(<Self as BatchBackward<U>>::BatchLossOutput,<Self as UpdateWeight<U>>::GradientStack), TrainingError>;
}
/// Trait that defines the implementation of the process of calculating the loss during error back propagation of neural networks by batch processing.
pub trait BatchLoss<U>: BatchBackward<U> + Loss<U> where U: UnitValue<U> {
    /// Error Calculation
    /// # Arguments
    /// * `loss` - Lower layer error
    /// * `_lossf` - loss function
    /// * `stack` - Stack to store calculation results at upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_loss<L: LossFunction<U>>(&self, loss:Self::BatchLossInput, _lossf:&L, stack:Self::BatchOutStack) -> Result<(Self::BatchOutStack, Self::BatchLossInput), TrainingError> {
        Ok((stack,loss))
    }
}
/// Trait that defines the relevant type of implementation that
/// calculates the results of forward propagation prior to processing
/// the error back propagation of the neural network by batch processing.
pub trait BatchPreTrainBase<U>: BatchForwardBase + PreTrain<U> where U: UnitValue<U> {
    /// Type of object to keep the results of forward propagation
    /// needed to perform error back propagation for batch execution.
    type BatchOutStack: Stack<Head=Self::BatchOutput> + Sized + Debug;
}
/// Trait that defines an implementation that calculates
/// the results of forward propagation prior to
/// the error back propagation process of a neural network through batch processing.
pub trait BatchPreTrain<U>: BatchPreTrainBase<U> + BatchForwardBase + BatchForward + where U: UnitValue<U> {
    /// Perform forward propagation required to perform error back propagation
    /// # Arguments
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_pre_train(&self, input:Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError>;
}
/// Trait that defines the implementation of neural network training by batch processing.
pub trait BatchTrain<U,D>: BatchPreTrainBase<U> + BatchPreTrain<U> + BatchBackward<U> + PreTrain<U> where U: UnitValue<U>, D: Device<U> {
    /// Train neural networks.
    /// # Arguments
    /// * `expected` - expected value
    /// * `input` - loss
    /// * `optimizer` - Optimizer object that implements the algorithm used to update the weights
    /// * `lossf` - loss function
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_train<OP: Optimizer<U>,L: BatchLossFunction<U,D>>(&mut self, expected:Self::BatchOutput, input:Self::BatchInput, optimizer:&mut OP, lossf:&L) -> Result<U, TrainingError>;
}
/// Trait that defines the ability to add layers to a neural network.
pub trait AddLayer: ForwardAll where Self: Sized {
    /// Adding Layers
    /// # Arguments
    /// * `f` - Callback that takes itself and returns an object with an internally generated layer added
    fn add_layer<C,F>(self,f:F) -> C where C: ForwardAll, F: FnOnce(Self) -> C;
}
/// Trait that defines the ability to add a layer with learning capabilities to a neural network.
pub trait AddLayerTrain<U>: PreTrain<U> where Self: Sized, U: UnitValue<U> {
    /// Adding Layers
    /// # Arguments
    /// * `f` - Callback that takes itself and returns an object with an internally generated layer added
    fn add_layer_train<C,F>(self,f:F) -> C where C: Train<U>, F: FnOnce(Self) -> C;
}
impl<T> AddLayer for T where T: ForwardAll + Sized {
    fn add_layer<C, F>(self, f: F) -> C where C: ForwardAll, F: FnOnce(Self) -> C {
        f(self)
    }
}
impl<T,U> AddLayerTrain<U> for T where T: PreTrain<U> + Sized, U: UnitValue<U> {
    fn add_layer_train<C, F>(self, f: F) -> C where C: Train<U>, F: FnOnce(Self) -> C {
        f(self)
    }
}
/// Trait defined functionality that attempts to add layers to a neural network.
pub trait TryAddLayer: ForwardAll where Self: Sized {
    /// Adding Layers
    /// # Arguments
    /// * `f` - Callback that takes itself and returns an object of type Result with an internally generated layer added
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`DeviceError`]
    fn try_add_layer<C,F,E>(self,f:F) -> Result<C,E> where C: ForwardAll, F: FnOnce(Self) -> Result<C,E>;
}
/// Trait that defines a function that seeks to add a learnable layer to a neural network
pub trait TryAddLayerTrain<U>: PreTrain<U> where Self: Sized, U: UnitValue<U> {
    /// Adding Layers
    /// # Arguments
    /// * `f` - Callback that takes itself and returns an object of type Result with an internally generated layer added
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`DeviceError`]
    fn try_add_layer_train<C,F>(self,f:F) -> Result<C,DeviceError> where C: Train<U>, F: FnOnce(Self) -> Result<C,DeviceError>;
}
impl<T> TryAddLayer for T where T: ForwardAll + Sized {
    fn try_add_layer<C,F,E>(self, f: F) -> Result<C,E> where C: ForwardAll, F: FnOnce(Self) -> Result<C,E> {
        f(self)
    }
}
impl<T,U> TryAddLayerTrain<U> for T where T: PreTrain<U> + Sized, U: UnitValue<U> {
    fn try_add_layer_train<C, F>(self, f: F) -> Result<C,DeviceError> where C: Train<U>, F: FnOnce(Self) -> Result<C,DeviceError> {
        f(self)
    }
}
impl<T,U> ForwardDiff<U> for T where T: PreTrain<U> + Sized, U: UnitValue<U> {
    fn forward_diff(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        self.pre_train(input)
    }
}
