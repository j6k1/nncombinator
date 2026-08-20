//! Quantization related functions

use crate::layer::InputTensorScalar;

/// A trait that represents the type to be quantized
pub trait Quantizable<S> {
    /// Quantized type
    type Quantized: InputTensorScalar<Scalar=S>;
}