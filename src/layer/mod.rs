//! The various layers that make up a neural network and the traits they implement

use std::error::Error;
use std::fmt::Debug;
use std::ops::Deref;
use crate::device::*;
use crate::{Stack};
use crate::error::{EvaluateError, PersistenceError, TrainingError};
use crate::lossfunction::*;
#[cfg(feature = "cuda")]
use crate::error::{TypeConvertError};
#[cfg(feature = "cuda")]
use crate::cuda::allocator::CudaAllocator;
use crate::cuda::DataTypeInfo;
#[cfg(feature = "cuda")]
use crate::cuda::ToCuda;
use crate::mapper::{DataMapper};
use crate::persistence::PersistenceType;

pub mod input;
pub mod output;
pub mod linear;
pub mod activation;
pub mod bridge;
pub mod logging;
pub mod batchnormalization;
pub mod bias;
pub mod scale;
pub mod quantization;

/// Differential input
#[derive(Debug)]
pub struct DiffInput<'a,T,O>
    where T: Debug,
          O: Debug {
    /// diff input
    pub diff: T,
    pub output: &'a O
}
impl<'a,T,O> DiffInput<'a,T,O>
    where T: Debug + Clone,
          O: Debug{
    pub fn new(diff: T, output: &'a O) -> Self {
        DiffInput {
            diff,
            output
        }
    }
}
impl<'a,T,O> Clone for DiffInput<'a,T,O>
    where T: Debug + Clone,
          O: Debug {
    fn clone(&self) -> Self {
        DiffInput {
            diff: self.diff.clone(),
            output: self.output
        }
    }
}
/// Trait that defines the data type during batch training corresponding to the data type
pub trait BatchDataType {
    type Type;
}
impl BatchDataType for () {
    type Type = ();
}

impl<'a,T,O> BatchDataType for DiffInput<'a,T,O>
    where T: Debug,
          O: Debug + 'a {
    type Type = Vec<DiffInput<'a,T,O>>;
}
#[cfg(feature = "cuda")]
impl<'a,T,O,U,A> ToCuda<U,A> for DiffInput<'a,T,O>
    where U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          T: Debug,
          O: Debug + 'a,
          A: CudaAllocator + 'static {
    type Output = Self;

    fn to_cuda(self, _: &DeviceGpu<A>) -> Result<Self::Output, TypeConvertError> {
        Ok(self)
    }
}
#[cfg(feature = "cuda")]
impl<'a,T,O,U,A> ToCuda<U,A> for Vec<DiffInput<'a,T,O>>
    where U: Default + Clone + Copy + Debug + Send + Sync + DataTypeInfo + 'static,
          T: Debug,
          O: Debug + 'a,
          A: CudaAllocator + 'static {
    type Output = Self;

    fn to_cuda(self, _: &DeviceGpu<A>) -> Result<Self::Output, TypeConvertError> {
        Ok(self)
    }
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
    type Output: Debug + 'static;
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
pub trait BackwardAll<SO>: PreTrainBase + UpdateWeight
    where SO: Clone + Copy + Debug {
    /// Loss input scalar type
    type LossInputScalar: Clone + Copy + Debug;
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
    fn backward_all(&mut self, input:Self::LossInput, stack:Self::OutStack)
        -> Result<(<Self as BackwardAll<SO>>::LossOutput,<Self as UpdateWeight>::GradientStack), TrainingError>;
}
/// Trait defining the calculation of the error during error back propagation.
pub trait Loss<SO>: BackwardAll<SO>
    where SO: Clone + Copy + Debug {
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
    fn loss(&mut self, loss:Self::LossInput, stack:Self::OutStack) -> Result<(Self::OutStack, Self::LossInput), TrainingError> {
        Ok((stack,loss))
    }

    fn is_canonical_link<L: LossFunction<SO>>(&self,_:&L) -> bool {
        false
    }
}
/// Characteristics defining the internal implementation of the error back propagation method in neural networks
pub trait Backward<U,I,O> {
    /// Back propagation of errors
    /// # Arguments
    /// * `input` - loss
    fn backward(&mut self, input:I) -> O;
}
/// A trait that defines the relevant implementation types for calculating the results of forward propagation
/// before processing error backpropagation in a neural network.
/// Trait that defines the process of forward propagation performed prior to the process of error back propagation.
pub trait PreTrainBase {
    /// The type of output that is piled on the stack during the error back propagation process.
    type PreOutput: Debug + 'static;
    /// Type of object to keep the results of forward propagation needed to perform error back propagation.
    type OutStack: Stack<Head=Self::PreOutput> + Debug + Sized;
}
pub trait PreTrain: PreTrainBase + ForwardAll {
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
pub trait UpdateWeight {
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
    fn update_weight(&mut self, stack:Self::GradientStack, batch_size: usize) -> Result<(), TrainingError>;
}
/// Trait that defines the learning process of a neural network.
pub trait Train<U,L>: PreTrain {
    /// Train neural networks.
    /// # Arguments
    /// * `expected` - expected value
    /// * `input` - loss
    /// * `lossf` - loss function
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn train(&mut self, expected:Self::Output, input:Self::Input, lossf:&L) -> Result<U, TrainingError>;
}
/// Implementation of a function to return the intermediate results of forward propagation for difference calculation
pub trait PartialForward: ForwardAll {
    /// Input data types for intermediate results during forward propagation
    type PartialInput: Debug;
    /// Output data types for intermediate results during forward propagation
    type PartialOutput: Debug;
    /// Forward Propagation Differential Input Information
    type DiffInput: Debug;

    /// Returns the intermediate result during forward propagation
    /// # Arguments
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn partial_forward(&self, input:Self::Input) -> Result<Self::PartialOutput, EvaluateError>;
    /// Take a difference input as input and return the result of the forward propagation up to that point
    /// # Arguments
    /// * `input` - diff input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn partial_forward_by_diff(&self, input:Self::DiffInput, partial_input:&Self::PartialInput)
        -> Result<Self::PartialOutput, EvaluateError>;
}
/// Implementation of a process performing forward propagation calculations from differential input values
pub trait ForwardDiff: PartialForward {
    /// Perform forward propagation using the diff input
    /// # Arguments
    /// * `input` - diff input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn forward_diff(&self, input:Self::DiffInput, partial_input:&Self::PartialInput) -> Result<Self::Output, EvaluateError>;
}
/// Implementation of the process for performing forward propagation calculations from precomputed values
pub trait ContinueForward: PartialForward {
    /// Resume forward propagation using the precomputed output of this layer
    /// # Arguments
    /// * `input` - input
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn continue_forward(&self, input:&Self::PartialInput) -> Result<Self::Output, EvaluateError>;
}
/// Trait defining the relevant type of implementation of forward propagation of neural networks by batch processing.
pub trait BatchForwardBase: ForwardAll {
    /// Input to this layer of the neural network for batch execution
    type BatchInput: Debug;
    /// Output from this layer of the neural network for batch execution
    type BatchOutput: Debug + 'static;
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
pub trait BatchBackward<SO>: BackwardAll<SO> + BatchPreTrainBase + UpdateWeight
    where SO: Clone + Copy + Debug {
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
    fn batch_backward(&mut self, input:Self::BatchLossInput, stack:Self::BatchOutStack)
        -> Result<(<Self as BatchBackward<SO>>::BatchLossOutput,<Self as UpdateWeight>::GradientStack), TrainingError>;
}
/// Trait that defines the implementation of the process of calculating the loss during error back propagation of neural networks by batch processing.
pub trait BatchLoss<SO>: BatchBackward<SO> + Loss<SO>
    where SO: Clone + Copy + Debug {
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
    fn batch_loss(&self, loss:Self::BatchLossInput, stack:Self::BatchOutStack) -> Result<(Self::BatchOutStack, Self::BatchLossInput), TrainingError> {
        Ok((stack,loss))
    }
}
/// Trait that defines the relevant type of implementation that
/// calculates the results of forward propagation prior to processing
/// the error back propagation of the neural network by batch processing.
pub trait BatchPreTrainBase: PreTrainBase {
    /// The type of output that is piled on the stack during the back-propagation process for errors in a batch run.
    type BatchPreOutput: Debug + 'static;
    /// Type of object to keep the results of forward propagation
    /// needed to perform error back propagation for batch execution.
    type BatchOutStack: Stack<Head=Self::BatchPreOutput> + Sized + Debug;
}
/// Trait that defines an implementation that calculates
/// the results of forward propagation prior to
/// the error back propagation process of a neural network through batch processing.
pub trait BatchPreTrain: BatchPreTrainBase + BatchForwardBase + BatchForward {
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
pub trait BatchTrain<SO,D,L>: BatchPreTrainBase + BatchPreTrain + BatchBackward<SO> + PreTrain
    where SO: Default + Clone + Copy + Debug + Send + Sync + 'static + DataTypeInfo,
          D: Device<SO>,
          L: LossFunction<SO> {
    /// Train neural networks.
    /// # Arguments
    /// * `expected` - expected value
    /// * `input` - loss
    /// * `lossf` - loss function
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_train(&mut self, expected:Self::BatchOutput, input:Self::BatchInput, lossf:&L) -> Result<SO, TrainingError>;
}
/// Definition of a trait that notifies of progress during learning
pub trait Step {
    /// on step notification
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn step(&mut self) -> Result<(), TrainingError>;
    /// on frequently step notification
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn frequently_step(&mut self) -> Result<(), TrainingError>;
}
/// Definition of the feature to notify the number of learning progress steps
pub trait OnStep {
    /// on step notification with step count
    /// # Arguments
    /// * `step` - step count
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn on_step(&mut self, step:usize) -> Result<(), TrainingError>;

    /// on frequently step notification with step count
    /// # Arguments
    /// * `step` - step count
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn on_frequently_step(&mut self, step: usize, frequently_step: usize) -> Result<(), TrainingError>;
}
/// Trait that define the persistence of learning progress data
pub trait PersistProgress<P,K> where K: PersistenceType {
    /// Load train progress data
    /// # Arguments
    /// * `persistence` - train progress persistent object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn load_progress(&mut self, persistence:&mut P) -> Result<(),TrainingError>;
    /// Save train progress data
    /// # Arguments
    /// * `persistence` - train progress persistent object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`PersistenceError`]
    fn save_progress(&mut self, persistence:&mut P) -> Result<(), PersistenceError>;
}
/// A trait that converts the weights of a trained model into quantized weights
pub trait Quantization<P,K> where K: PersistenceType {
    /// Load train progress data
    /// # Arguments
    /// * `persistence` - model persistent object
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn quantization(&mut self, persistence:&mut P) -> Result<(),TrainingError>;
}
/// A trait that represents the input scale
pub trait InputScale {
    /// output mean scale
    fn scale_mean(&self) -> f32 {
        1.
    }
}
/// A trait that represents the output scale
pub trait OutputScale: ForwardAll
    where <Self as ForwardAll>::Output: BatchDataType + 'static,
          <Self::ScaledOutput as BatchDataType>::Type: Debug + BatchSize + 'static {
    /// Device used for scaling calculations
    type ScalingDevice: 'static;
    /// output scale.
    type Scale: Debug + 'static;
    /// scaled output.
    type ScaledOutput: Debug + BatchDataType + 'static;
    /// Data Mapper for Scaling
    type Mapper<'a>: DataMapper<'a,<Self as ForwardAll>::Output,Self::ScaledOutput,Self::ScalingDevice> where Self: Deref<Target=Self::ScaledOutput> + 'a;
    fn scaling_mapper<'a>(&'a self, input: &'a <Self as ForwardAll>::Output) -> Result<Self::Mapper<'a>,EvaluateError>
        where Self: Deref<Target=Self::ScaledOutput> + 'a;
}
/// A trait that represents the output scale
pub trait BatchOutputScale: OutputScale + BatchForwardBase
    where <Self as ForwardAll>::Output: BatchDataType + 'static,
          <Self::ScaledOutput as BatchDataType>::Type: Debug + BatchSize + 'static {
    /// batch scaled output.
    type BatchScaledOutput: Debug + 'static;
    /// Data Mapper for Scaling for batch execution
    type BatchMapper<'a>: DataMapper<'a,<Self as BatchForwardBase>::BatchOutput,Self::BatchScaledOutput,Self::ScalingDevice>
        where Self: Deref<Target=Self::BatchScaledOutput> + 'a;
    fn batch_scaling_mapper<'a>(&'a self, input: &'a <Self as BatchForwardBase>::BatchOutput) -> Result<Self::BatchMapper<'a>,EvaluateError>
        where Self: Deref<Target=Self::BatchScaledOutput> + 'a;
}
/// A trait that represents the maximum value of the input passed from this layer to the next layer
pub trait MaxInputValue {
    type Scalar: Clone + Copy + Debug;
    /// maximum value of the input passed from this layer to the next layer.
    fn max_input_value(&self) -> Self::Scalar;
}
/// A trait representing the weight type of inputs used in the implementation of various layers
pub trait InputTensorScalar {
    type Scalar: Clone + Copy + Debug;
}
/// A trait representing the weight type of outputs used in the implementation of various layers
pub trait OutputTensorScalar {
    type Scalar: Clone + Copy + Debug;
}
/// A characteristic that represents the size of the input weights used in the implementation of each layer
pub trait TensorSize {
    const SIZE: usize;
}
impl<const N: usize> TensorSize for [();N] {
    const SIZE: usize = N;
}
/// A feature representing the number of dimensions in the input from the upper layer
pub trait InputTensorSize<const N: usize> {
}
/// A characteristic that represents the number of dimensions in the input output by the upper layers
pub trait OutputTensorSize<const N: usize> {
}
/// Trait that defines the ability to add layers to a neural network.
pub trait AddLayer: ForwardAll where Self: Sized {
    /// Adding Layers
    /// # Arguments
    /// * `f` - Callback that takes itself and returns an object with an internally generated layer added
    fn add_layer<C,F>(self,f:F) -> C where C: ForwardAll, F: FnOnce(Self) -> C;
}
impl<T> AddLayer for T where T: ForwardAll + Sized {
    fn add_layer<C, F>(self, f: F) -> C where C: ForwardAll, F: FnOnce(Self) -> C {
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
    /// * `E`
    fn try_add_layer<C,F,E>(self,f:F) -> Result<C,E> where C: ForwardAll, F: FnOnce(Self) -> Result<C,E>;
}
impl<T> TryAddLayer for T where T: ForwardAll + Sized {
    fn try_add_layer<C,F,E>(self, f: F) -> Result<C,E> where C: ForwardAll, F: FnOnce(Self) -> Result<C,E> {
        f(self)
    }
}
