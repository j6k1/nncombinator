//! Definition of various errors
use std::{error, fmt, io};
use std::array::TryFromSliceError;
#[cfg(feature = "cuda")]
use std::ffi::CStr;
#[cfg(feature = "cuda")]
use std::fmt::{Formatter};
use std::fmt::{Debug};
use std::num::{ParseFloatError, ParseIntError};
#[cfg(feature = "cuda")]
use cuda_runtime_sys::cudaError_t;
use try_from_primitive::error::FromPrimitiveError;

/// Errors made during neural network training
#[derive(Debug)]
pub enum TrainingError {
    /// Tried to get two or more mutable references from Arc in multiple threads at the same time
    /// (this error is currently not used within crate)
    ReferenceCountError,
    /// Errors occurring within the launched external thread
    /// (this error is currently not used within crate)
    ThreadError(String),
    /// Error generating fixed-length collections from collections of different sizes
    SizeMismatchError(SizeMismatchError),
    /// Illegal input value
    InvalidInputError(String),
    /// Error during forward propagation of neural network
    EvaluateError(EvaluateError),
    /// The value is too large to convert.
    /// (this error is currently not used within crate)
    ToLargeInput(f64),
    /// Error in cuda processing
    #[cfg(feature = "cuda")]
    CudaError(CudaError),
    /// Error in cublas processing
    #[cfg(feature = "cuda")]
    CublasError(rcublas::error::Error),
    /// Error in cudnn processing
    #[cfg(feature = "cuda")]
    CudnnError(rcudnn::Error),
    /// Error in cuda runtime
    #[cfg(feature = "cuda")]
    CudaRuntimeError(CudaRuntimeError),
    /// Errors that occur when the internal state of a particular object or other object is abnormal.
    InvalidStateError(InvalidStateError),
    /// Error that occurs when calling a function that is not supported by the specification
    UnsupportedOperationError(UnsupportedOperationError),
    /// Error raised when cast of primitive type fails
    TypeCastError(String),
    /// Error generated when type conversion fails
    TypeConvertError(TypeConvertError),
    /// Error raised if cast to fixed-length array fails
    TryFromSliceError(TryFromSliceError),
    /// Error raised when the value is not a valid primitive type.
    FromPrimitiveError(FromPrimitiveError),
    /// Error when reading model
    ModelLoadError(ModelLoadError)
}
impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TrainingError::ReferenceCountError => write!(f, "There must be only one reference to Arc or Weak to itself when training is performed."),
            TrainingError::ThreadError(s) => write!(f, "An error occurred while processing the thread. cause = ({})",s),
            TrainingError::SizeMismatchError(e) => write!(f, "{}",e),
            TrainingError::InvalidInputError(s) => write!(f, "{}",s),
            TrainingError::EvaluateError(e) => write!(f, "{}",e),
            TrainingError::ToLargeInput(n) => write!(f, "The value is too large to convert. (Value = {})",n),
            #[cfg(feature = "cuda")]
            TrainingError::CudaError(e) => write!(f, "An error occurred in the process of cuda. ({})",e),
            #[cfg(feature = "cuda")]
            TrainingError::CublasError(e) => write!(f, "An error occurred during the execution of a process in cublas. ({})",e),
            #[cfg(feature = "cuda")]
            TrainingError::CudnnError(e) => write!(f, "An error occurred during the execution of a process in cudnn. ({})",e),
            #[cfg(feature = "cuda")]
            TrainingError::CudaRuntimeError(e) => write!(f,"{}",e),
            TrainingError::InvalidStateError(e) => write!(f,"Invalid state. ({})",e),
            TrainingError::UnsupportedOperationError(e) => write!(f,"unsupported operation. ({})",e),
            TrainingError::TypeCastError(s) => write!(f,"{}",s),
            TrainingError::TypeConvertError(e) => write!(f,"{}",e),
            TrainingError::TryFromSliceError(e) => write!(f,"{}",e),
            TrainingError::FromPrimitiveError(e) => write!(f,"{}",e),
            TrainingError::ModelLoadError(e) => write!(f,"{}",e),
        }
    }
}
impl error::Error for TrainingError {
    fn description(&self) -> &str {
        match self {
            TrainingError::ReferenceCountError => "There must be only one reference to Arc or Weak to itself when training is performed.",
            TrainingError::ThreadError(_) => "An error occurred while processing the thread.",
            TrainingError::SizeMismatchError(_) => "memory size does not match.",
            TrainingError::InvalidInputError(_) => "Incorrect input.",
            TrainingError::EvaluateError(_) => "An error occurred when running the neural network.",
            TrainingError::ToLargeInput(_) => "The value is too large to convert.",
            #[cfg(feature = "cuda")]
            TrainingError::CudaError(_) => "An error occurred in the process of cuda.",
            #[cfg(feature = "cuda")]
            TrainingError::CublasError(_) => "An error occurred during the execution of a process in cublas.",
            #[cfg(feature = "cuda")]
            TrainingError::CudnnError(_) => "An error occurred during the execution of a process in cudnn.",
            #[cfg(feature = "cuda")]
            TrainingError::CudaRuntimeError(_) => "An error occurred while running the Cuda kernel.",
            TrainingError::InvalidStateError(_) => "Invalid state.",
            TrainingError::UnsupportedOperationError(_) => "unsupported operation.",
            TrainingError::TypeCastError(_) => "Typecast failed.",
            TrainingError::TypeConvertError(_) => "Type convert failed.",
            TrainingError::TryFromSliceError(_) => "Conversion to fixed-length array failed.",
            TrainingError::FromPrimitiveError(_) => "Conversion from primitive type failed.",
            TrainingError::ModelLoadError(_) => "An error occurred when loading the model.",
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            TrainingError::ReferenceCountError => None,
            TrainingError::ThreadError(_) => None,
            TrainingError::SizeMismatchError(e) => Some(e),
            TrainingError::InvalidInputError(_) => None,
            TrainingError::EvaluateError(e) => Some(e),
            TrainingError::ToLargeInput(_) => None,
            #[cfg(feature = "cuda")]
            TrainingError::CudaError(e) => Some(e),
            #[cfg(feature = "cuda")]
            TrainingError::CublasError(e) => Some(e),
            #[cfg(feature = "cuda")]
            TrainingError::CudnnError(e) => Some(e),
            #[cfg(feature = "cuda")]
            TrainingError::CudaRuntimeError(_) => None,
            TrainingError::InvalidStateError(e) => Some(e),
            TrainingError::UnsupportedOperationError(e) => Some(e),
            TrainingError::TypeCastError(_) => None,
            TrainingError::TypeConvertError(e) => Some(e),
            TrainingError::TryFromSliceError(e) => Some(e),
            TrainingError::FromPrimitiveError(e) => Some(e),
            TrainingError::ModelLoadError(e) => Some(e),
        }
    }
}
/// Error when reading model
#[derive(Debug)]
pub enum ModelLoadError {
    /// IO Error
    IOError(io::Error),
    /// Errors that occur when the internal state of a particular object or other object is abnormal.
    InvalidState(String),
    /// Error when trying to parse a numeric string into numbers
    ParseFloatError(ParseFloatError),
    /// Error when trying to parse a numeric string into integers
    ParseIntError(ParseIntError),
    /// Error in cudnn processing
    #[cfg(feature = "cuda")]
    CudnnError(rcudnn::Error),
    /// Error in specialization
    SpecializationError(SpecializationError),
}
impl fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ModelLoadError::IOError(_) => write!(f, "Error occurred in file I/O."),
            ModelLoadError::InvalidState(ref s) => write!(f, "Configuration is invalid. ({})", s),
            ModelLoadError::ParseFloatError(_) => write!(f, "An error occurred when converting a string to a double value."),
            ModelLoadError::ParseIntError(_) => write!(f, "An error occurred when converting a string to an integer value."),
            #[cfg(feature = "cuda")]
            ModelLoadError::CudnnError(e) => write!(f, "An error occurred during the execution of a process in cudnn. ({})", e),
            ModelLoadError::SpecializationError(e) => write!(f, "An error occurred during specialization. ({})", e),
        }
    }
}
impl error::Error for ModelLoadError {
    fn description(&self) -> &str {
        match self {
            ModelLoadError::IOError(_) => "Error occurred in file I/O.",
            ModelLoadError::InvalidState(_) => "Configuration is invalid.",
            ModelLoadError::ParseFloatError(_) => "An error occurred when converting a string to a double value.",
            ModelLoadError::ParseIntError(_) => "An error occurred when converting a string to an integer value.",
            #[cfg(feature = "cuda")]
            ModelLoadError::CudnnError(_) => "An error occurred during the execution of a process in cudnn.",
            ModelLoadError::SpecializationError(_) => "An error occurred during specialization."
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            ModelLoadError::IOError(ref e) => Some(e),
            ModelLoadError::InvalidState(_) => None,
            ModelLoadError::ParseFloatError(ref e) => Some(e),
            ModelLoadError::ParseIntError(ref e) => Some(e),
            #[cfg(feature = "cuda")]
            ModelLoadError::CudnnError(ref e) => Some(e),
            ModelLoadError::SpecializationError(ref e) => Some(e)
        }
    }
}
impl From<SizeMismatchError> for TrainingError {
    fn from(err: SizeMismatchError) -> TrainingError {
        TrainingError::SizeMismatchError(err)
    }
}
impl From<TypeConvertError> for TrainingError {
    fn from(err: TypeConvertError) -> TrainingError {
        TrainingError::TypeConvertError(err)
    }
}
impl From<EvaluateError> for TrainingError {
    fn from(err: EvaluateError) -> TrainingError {
        TrainingError::EvaluateError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<CudaError> for TrainingError {
    fn from(err: CudaError) -> TrainingError {
        TrainingError::CudaError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcublas::error::Error> for TrainingError {
    fn from(err: rcublas::error::Error) -> TrainingError {
        TrainingError::CublasError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcudnn::Error> for TrainingError {
    fn from(err: rcudnn::Error) -> TrainingError {
        TrainingError::CudnnError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<CudaRuntimeError> for TrainingError {
    fn from(err: CudaRuntimeError) -> TrainingError {
        TrainingError::CudaRuntimeError(err)
    }
}
impl From<InvalidStateError> for TrainingError {
    fn from(err: InvalidStateError) -> TrainingError {
        TrainingError::InvalidStateError(err)
    }
}
impl From<UnsupportedOperationError> for TrainingError {
    fn from(err: UnsupportedOperationError) -> TrainingError {
        TrainingError::UnsupportedOperationError(err)
    }
}
impl From<ModelLoadError> for TrainingError {
    fn from(err: ModelLoadError) -> TrainingError {
        TrainingError::ModelLoadError(err)
    }
}
impl From<io::Error> for ModelLoadError {
    fn from(err: io::Error) -> ModelLoadError {
        ModelLoadError::IOError(err)
    }
}
impl From<ParseFloatError> for ModelLoadError {
    fn from(err: ParseFloatError) -> ModelLoadError {
        ModelLoadError::ParseFloatError(err)
    }
}
impl From<ParseIntError> for ModelLoadError {
    fn from(err: ParseIntError) -> ModelLoadError {
        ModelLoadError::ParseIntError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcudnn::Error> for ModelLoadError {
    fn from(err: rcudnn::Error) -> ModelLoadError {
        ModelLoadError::CudnnError(err)
    }
}
impl From<SpecializationError> for ModelLoadError {
    fn from(err: SpecializationError) -> ModelLoadError {
        ModelLoadError::SpecializationError(err)
    }
}
impl From<TryFromSliceError> for TrainingError {
    fn from(err: TryFromSliceError) -> TrainingError {
        TrainingError::TryFromSliceError(err)
    }
}
impl From<FromPrimitiveError> for TrainingError {
    fn from(err: FromPrimitiveError) -> TrainingError {
        TrainingError::FromPrimitiveError(err)
    }
}
/// Error generating fixed-length collections from collections of different sizes
#[derive(Debug)]
pub struct SizeMismatchError(pub usize, pub usize);
impl fmt::Display for SizeMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SizeMismatchError(from, to) => {
                write!(f, "memory size does not match. (from = {}, to = {})",from,to)
            }
        }
    }
}
impl error::Error for SizeMismatchError {
    fn description(&self) -> &str {
        "memory size does not match."
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}
/// An error that occurs when creating a fixed-length collection from collections
/// of different sizes, when the number of elements in the converted collection is not divisible
/// by the number of elements in the destination collection.
#[derive(Debug)]
pub struct IndivisibleError(pub usize, pub usize);
impl fmt::Display for IndivisibleError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IndivisibleError(from, to) => {
                write!(f, "memory size indivisible. (dividend = {}, divisor = {})",from,to)
            }
        }
    }
}
impl error::Error for IndivisibleError {
    fn description(&self) -> &str {
        "memory size indivisible."
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}
#[derive(Debug)]
pub enum TypeConvertError {
    /// Error generating fixed-length collections from collections of different sizes
    SizeMismatchError(SizeMismatchError),
    /// Error when creating a fixed length collection from collections of different sizes (left side not divisible by right side)
    IndivisibleError(IndivisibleError),
    /// Error in cuda processing
    #[cfg(feature = "cuda")]
    CudaError(CudaError),
    /// Error in cudnn processing
    #[cfg(feature = "cuda")]
    CudnnError(rcudnn::Error),
}
impl fmt::Display for TypeConvertError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TypeConvertError::SizeMismatchError(e) => write!(f,"{}",e),
            TypeConvertError::IndivisibleError(e) => write!(f,"{}",e),
            #[cfg(feature = "cuda")]
            TypeConvertError::CudaError(e) => write!(f, "An error occurred in the process of cuda. ({})",e),
            #[cfg(feature = "cuda")]
            TypeConvertError::CudnnError(e) => write!(f, "An error occurred during the execution of a process in cudnn. ({})",e),
        }
    }
}
impl error::Error for TypeConvertError {
    fn description(&self) -> &str {
        match *self {
            TypeConvertError::SizeMismatchError(_) => "memory size does not match.",
            TypeConvertError::IndivisibleError(_) => "memory size indivisible.",
            #[cfg(feature = "cuda")]
            TypeConvertError::CudaError(_) => "An error occurred in the process of cuda.",
            #[cfg(feature = "cuda")]
            TypeConvertError::CudnnError(_) => "An error occurred during the execution of a process in cudnn.",
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            TypeConvertError::SizeMismatchError(e) => Some(e),
            TypeConvertError::IndivisibleError(e) => Some(e),
            #[cfg(feature = "cuda")]
            TypeConvertError::CudaError(e) => Some(e),
            #[cfg(feature = "cuda")]
            TypeConvertError::CudnnError(e) => Some(e)
        }
    }
}
impl From<SizeMismatchError> for TypeConvertError {
    fn from(err: SizeMismatchError) -> TypeConvertError {
        TypeConvertError::SizeMismatchError(err)
    }
}
impl From<IndivisibleError> for TypeConvertError {
    fn from(err: IndivisibleError) -> TypeConvertError {
        TypeConvertError::IndivisibleError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<CudaError> for TypeConvertError {
    fn from(err: CudaError) -> TypeConvertError {
        TypeConvertError::CudaError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcudnn::Error> for TypeConvertError {
    fn from(err: rcudnn::Error) -> TypeConvertError {
        TypeConvertError::CudnnError(err)
    }
}
/// Error when accessing array out of range
#[derive(Debug)]
pub struct IndexOutBoundError {
    len:usize,
    index:usize
}
impl fmt::Display for IndexOutBoundError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IndexOutBoundError { len, index}  => {
                write!(f, "The value {} specified for the index is out of range; it exceeds {}.",index, len - 1)
            }
        }
    }
}
impl error::Error for IndexOutBoundError {
    fn description(&self) -> &str {
        "The value specified for the index is out of range."
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}
impl IndexOutBoundError {
    /// Creation of an instance of IndexOutBoundError
    /// # Arguments
    /// * `len` - max len
    /// * `index` - Specified index
    pub fn new(len:usize,index:usize) -> IndexOutBoundError {
        IndexOutBoundError {
            len:len,
            index:index
        }
    }
}
/// Error during forward propagation of neural network
#[derive(Debug)]
pub enum EvaluateError {
    /// Error in cuda processing
    #[cfg(feature = "cuda")]
    CudaError(CudaError),
    /// Error in cublas processing
    #[cfg(feature = "cuda")]
    CublasError(rcublas::error::Error),
    /// Error in cudnn processing
    #[cfg(feature = "cuda")]
    CudnnError(rcudnn::Error),
    /// Error in cuda runtime
    #[cfg(feature = "cuda")]
    CudaRuntimeError(CudaRuntimeError),
    /// Error generating fixed-length collections from collections of different sizes
    SizeMismatchError(SizeMismatchError),
    /// Errors that occur when the internal state of a particular object or other object is abnormal.
    InvalidStateError(InvalidStateError),
    /// Error raised when cast of primitive type fails
    TypeCastError(String),
    /// Error generated when type conversion fails
    TypeConvertError(TypeConvertError),
    /// Error raised if cast to fixed-length array fails
    TryFromSliceError(TryFromSliceError)
}
impl fmt::Display for EvaluateError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(feature = "cuda")]
            EvaluateError::CudaError(e) => write!(f, "An error occurred in the process of cuda. ({})", e),
            #[cfg(feature = "cuda")]
            EvaluateError::CublasError(e) => write!(f, "An error occurred during the execution of a process in cublas. ({})", e),
            #[cfg(feature = "cuda")]
            EvaluateError::CudnnError(e) => write!(f, "An error occurred during the execution of a process in cudnn. ({})", e),
            #[cfg(feature = "cuda")]
            EvaluateError::CudaRuntimeError(e) => write!(f,"{}",e),
            EvaluateError::SizeMismatchError(e) => write!(f,"{}",e),
            EvaluateError::InvalidStateError(e) => write!(f,"Invalid state. ({})",e),
            EvaluateError::TypeCastError(s) => write!(f,"{}",s),
            EvaluateError::TypeConvertError(e) => write!(f,"{}",e),
            EvaluateError::TryFromSliceError(e) => write!(f,"{}",e)
        }
    }
}
impl error::Error for EvaluateError {
    fn description(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            EvaluateError::CudaError(_) => "An error occurred in the process of cuda.",
            #[cfg(feature = "cuda")]
            EvaluateError::CublasError(_) => "An error occurred during the execution of a process in cublas.",
            #[cfg(feature = "cuda")]
            EvaluateError::CudnnError(_) => "An error occurred during the execution of a process in cudnn.",
            #[cfg(feature = "cuda")]
            EvaluateError::CudaRuntimeError(_) => "An error occurred while running the Cuda kernel.",
            EvaluateError::SizeMismatchError(_) => "memory size does not match.",
            EvaluateError::InvalidStateError(_) => "Invalid state.",
            EvaluateError::TypeCastError(_) => "Typecast failed.",
            EvaluateError::TypeConvertError(_) => "Type covert failed.",
            EvaluateError::TryFromSliceError(_) => "Conversion to fixed-length array failed."
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            #[cfg(feature = "cuda")]
            EvaluateError::CudaError(e) => Some(e),
            #[cfg(feature = "cuda")]
            EvaluateError::CublasError(e) => Some(e),
            #[cfg(feature = "cuda")]
            EvaluateError::CudnnError(e) => Some(e),
            #[cfg(feature = "cuda")]
            EvaluateError::CudaRuntimeError(_) => None,
            EvaluateError::SizeMismatchError(e) => Some(e),
            EvaluateError::InvalidStateError(e) => Some(e),
            EvaluateError::TypeCastError(_) => None,
            EvaluateError::TypeConvertError(e) => Some(e),
            EvaluateError::TryFromSliceError(e) => Some(e)
        }
    }
}
#[cfg(feature = "cuda")]
impl From<CudaError> for EvaluateError {
    fn from(err: CudaError) -> EvaluateError {
        EvaluateError::CudaError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcublas::error::Error> for EvaluateError {
    fn from(err: rcublas::error::Error) -> EvaluateError {
        EvaluateError::CublasError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcudnn::Error> for EvaluateError {
    fn from(err: rcudnn::Error) -> EvaluateError {
        EvaluateError::CudnnError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<CudaRuntimeError> for EvaluateError {
    fn from(err: CudaRuntimeError) -> EvaluateError {
        EvaluateError::CudaRuntimeError(err)
    }
}
impl From<SizeMismatchError> for EvaluateError {
    fn from(err: SizeMismatchError) -> EvaluateError {
        EvaluateError::SizeMismatchError(err)
    }
}
impl From<TypeConvertError> for EvaluateError {
    fn from(err: TypeConvertError) -> EvaluateError {
        EvaluateError::TypeConvertError(err)
    }
}
impl From<InvalidStateError> for EvaluateError {
    fn from(err: InvalidStateError) -> EvaluateError {
        EvaluateError::InvalidStateError(err)
    }
}
impl From<TryFromSliceError> for EvaluateError {
    fn from(err: TryFromSliceError) -> EvaluateError {
        EvaluateError::TryFromSliceError(err)
    }
}
#[derive(Debug)]
pub enum PersistenceError {
    /// Error in cudnn processing
    #[cfg(feature = "cuda")]
    CudnnError(rcudnn::Error),
    /// Errors caused by generating fixed-length arrays from different size Vecs
    SizeMismatchError(SizeMismatchError),
    /// Error generated when type conversion fails
    TypeConvertError(TypeConvertError),
    /// Error in generalization
    GeneralizationError(GeneralizationError),
}
impl fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(feature = "cuda")]
            PersistenceError::CudnnError(e) => write!(f, "An error occurred during the execution of a process in cudnn. ({})",e),
            PersistenceError::SizeMismatchError(e) => write!(f, "{}",e),
            PersistenceError::TypeConvertError(e) => write!(f,"{}",e),
            PersistenceError::GeneralizationError(e) => write!(f,"{}",e),
        }
    }
}
impl error::Error for PersistenceError {
    fn description(&self) -> &str {
        match *self {
            #[cfg(feature = "cuda")]
            PersistenceError::CudnnError(_) => "An error occurred during the execution of a process in cudnn.",
            PersistenceError::SizeMismatchError(_) => "memory size does not match.",
            PersistenceError::TypeConvertError(_) => "Type convert failed.",
            PersistenceError::GeneralizationError(_) => "An error occurred during generalization.",
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            #[cfg(feature = "cuda")]
            PersistenceError::CudnnError(ref e) => Some(e),
            PersistenceError::SizeMismatchError(ref e) => Some(e),
            PersistenceError::TypeConvertError(ref e) => Some(e),
            PersistenceError::GeneralizationError(ref e) => Some(e),
        }
    }
}
#[cfg(feature = "cuda")]
impl From<rcudnn::Error> for PersistenceError {
    fn from(err: rcudnn::Error) -> PersistenceError {
        PersistenceError::CudnnError(err)
    }
}
impl From<SizeMismatchError> for PersistenceError {
    fn from(err: SizeMismatchError) -> PersistenceError {
        PersistenceError::SizeMismatchError(err)
    }
}
impl From<TypeConvertError> for PersistenceError {
    fn from(err: TypeConvertError) -> PersistenceError {
        PersistenceError::TypeConvertError(err)
    }
}
impl From<GeneralizationError> for PersistenceError {
    fn from(err: GeneralizationError) -> PersistenceError {
        PersistenceError::GeneralizationError(err)
    }
}
/// Error in cuda processing
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub enum CudaError {
    /// Memory allocation failed.
    AllocFailed(String),
    /// Error in cudnn processing
    CudnnError(rcudnn::Error),
    /// Error in cublas processing
    CublasError(rcublas::error::Error),
    /// Errors that occur when the internal state of a particular object or other object is abnormal.
    InvalidState(String),
    /// There is a problem with the implementation logic of the program
    LogicError(String),
    /// Error that occurs when a specified argument or other setting value is invalid.
    InvalidConfigurationError(String),
}
#[cfg(feature = "cuda")]
impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CudaError::AllocFailed(s) => write!(f, "{}", s),
            CudaError::CudnnError(e) => write!(f, "An error occurred during the execution of a process in cudnn. ({})", e),
            CudaError::CublasError(e) => write!(f, "An error occurred during the execution of a process in cublas. ({})", e),
            CudaError::InvalidState(s) => write!(f, "{}", s),
            CudaError::LogicError(s) => write!(f,"{}",s),
            CudaError::InvalidConfigurationError(s) => write!(f,"{}",s),
        }
    }
}
#[cfg(feature = "cuda")]
impl error::Error for CudaError {
    fn description(&self) -> &str {
        match self {
            CudaError::AllocFailed(_) => "Memory allocation failed.",
            CudaError::CudnnError(_) => "An error occurred during the execution of a process in cudnn.",
            CudaError::CublasError(_) => "An error occurred during the execution of a process in cublas.",
            CudaError::InvalidState(_) => "Invalid state.s",
            CudaError::LogicError(_) => "Logic error.",
            CudaError::InvalidConfigurationError(_) => "Invalid configuration.",
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            CudaError::AllocFailed(_) => None,
            CudaError::CudnnError(e) => Some(e),
            CudaError::CublasError(e) => Some(e),
            CudaError::InvalidState(_) => None,
            CudaError::LogicError(_) => None,
            CudaError::InvalidConfigurationError(_) => None,
        }
    }
}
#[cfg(feature = "cuda")]
impl From<rcudnn::Error> for CudaError {
    fn from(err: rcudnn::Error) -> CudaError {
        CudaError::CudnnError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcublas::Error> for CudaError {
    fn from(err: rcublas::error::Error) -> CudaError {
        CudaError::CublasError(err)
    }
}
/// Errors that occur in processes implemented in objects that implement Device Traits
#[derive(Debug)]
pub enum DeviceError {
    /// Error in cuda processing
    #[cfg(feature = "cuda")]
    CudaError(CudaError),
    /// Error in cublas processing
    #[cfg(feature = "cuda")]
    CublasError(rcublas::error::Error),
    /// Error in cudnn processing
    #[cfg(feature = "cuda")]
    CudnnError(rcudnn::Error),
}
#[cfg(feature = "cuda")]
impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DeviceError::CudaError(e) => write!(f, "An error occurred in the process of cuda. ({})", e),
            DeviceError::CublasError(e) => write!(f, "An error occurred during the execution of a process in cublas. ({})", e),
            DeviceError::CudnnError(e) => write!(f, "An error occurred during the execution of a process in cudnn. ({})", e),
        }
    }
}
#[cfg(not(feature = "cuda"))]
impl fmt::Display for DeviceError {
    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
        unreachable!()
    }
}
#[cfg(feature = "cuda")]
impl error::Error for DeviceError {
    fn description(&self) -> &str {
        match self {
            DeviceError::CudaError(_) => "Asn error occurred in the process of cuda.",
            DeviceError::CublasError(_) => "An error occurred during the execution of a process in cublas.",
            DeviceError::CudnnError(_) => "An error occurred during the execution of a process in cudnn.",
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            DeviceError::CudaError(e) => Some(e),
            DeviceError::CublasError(e) => Some(e),
            DeviceError::CudnnError(e) => Some(e),
        }
    }
}
#[cfg(not(feature = "cuda"))]
impl error::Error for DeviceError {
    fn description(&self) -> &str {
        unreachable!()
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        unreachable!()
    }
}
#[cfg(feature = "cuda")]
impl From<CudaError> for DeviceError {
    fn from(err: CudaError) -> DeviceError {
        DeviceError::CudaError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcublas::error::Error> for DeviceError {
    fn from(err: rcublas::error::Error) -> DeviceError {
        DeviceError::CublasError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcudnn::Error> for DeviceError {
    fn from(err: rcudnn::Error) -> DeviceError {
        DeviceError::CudnnError(err)
    }
}
/// Errors that occur when the internal state of a particular object or other object is abnormal.
#[derive(Debug)]
pub struct InvalidStateError(pub String);
impl fmt::Display for InvalidStateError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InvalidStateError(s) => {
                write!(f,"{}",s)
            }
        }
    }
}
impl error::Error for InvalidStateError {
    fn description(&self) -> &str {
        match self {
            InvalidStateError(_) => "Invalid state."
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            InvalidStateError(_) => None,
        }
    }
}
/// Error that occurs when calling a function that is not supported by the specification
#[derive(Debug,PartialEq,Eq)]
pub struct UnsupportedOperationError(pub String);
impl fmt::Display for UnsupportedOperationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UnsupportedOperationError(s) => {
                write!(f,"{}",s)
            }
        }
    }
}
impl error::Error for UnsupportedOperationError {
    fn description(&self) -> &str {
        match self {
            UnsupportedOperationError(_) => "unsupported operation."
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            UnsupportedOperationError(_) => None,
        }
    }
}
/// Error in cuda runtime
#[cfg(feature = "cuda")]
pub struct CudaRuntimeError {
    raw: cudaError_t,
}
#[cfg(feature = "cuda")]
impl fmt::Debug for CudaRuntimeError {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f,"{}",unsafe { CStr::from_ptr(cuda_runtime_sys::cudaGetErrorString(self.raw)) }.to_string_lossy())
    }
}
#[cfg(feature = "cuda")]
impl fmt::Display for CudaRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,"An error occurred while running the Cuda kernel. cause = {}",
            unsafe { CStr::from_ptr(cuda_runtime_sys::cudaGetErrorString(self.raw)) }.to_string_lossy()
        )
    }
}
#[cfg(feature = "cuda")]
impl error::Error for CudaRuntimeError {
    fn description(&self) -> &str {
        "An error occurred while running the Cuda kernel."
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}
#[cfg(feature = "cuda")]
impl CudaRuntimeError {
    /// Creation of an instance of CudaRuntimeError
    pub fn new(raw:cudaError_t) -> CudaRuntimeError {
        CudaRuntimeError {
            raw: raw
        }
    }
}
/// Error when layer instantiation fails
#[derive(Debug)]
pub enum LayerInstantiationError {
    /// Error in cuda processing
    #[cfg(feature = "cuda")]
    CudaError(CudaError),
    /// Error in cudnn processing
    #[cfg(feature = "cuda")]
    CudnnError(rcudnn::Error),
    /// Error in build optimizer processing
    OptimizerBuildError(OptimizerBuildError),
    /// Errors caused by generating fixed-length arrays from different size Vecs
    SizeMismatchError(SizeMismatchError),
    /// Error generated when type conversion fails
    TypeConvertError(TypeConvertError),
    /// Error during data specialization
    SpecializationError(SpecializationError)

}
impl fmt::Display for LayerInstantiationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(feature = "cuda")]
            LayerInstantiationError::CudaError(e) => {
                write!(f,"An unexpected error occurred during layer instantiation ({})",e)
            },
            #[cfg(feature = "cuda")]
            LayerInstantiationError::CudnnError(e) => {
                write!(f,"An unexpected error occurred during layer instantiation ({})",e)
            },
            LayerInstantiationError::OptimizerBuildError(e) => {
                write!(f,"An unexpected error occurred during build optimizer ({})",e)
            },
            LayerInstantiationError::SizeMismatchError(e) => {
                write!(f,"An unexpected error occurred during layer instantiation ({})",e)
            },
            LayerInstantiationError::TypeConvertError(e) => {
                write!(f,"An unexpected error occurred during layer instantiation ({})",e)
            },
            LayerInstantiationError::SpecializationError(e) => {
                write!(f,"An unexpected error occurred during layer instantiation ({})",e)
            }
        }
    }
}
impl error::Error for LayerInstantiationError {
    fn description(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            LayerInstantiationError::CudaError(_) => {
                "An unexpected error occurred during layer instantiation (An error occurred in the process of cudas)."
            },
            #[cfg(feature = "cuda")]
            LayerInstantiationError::CudnnError(_) => {
                "An unexpected error occurred during layer instantiation (An error occurred in the process of cudnns)."
            },
            LayerInstantiationError::OptimizerBuildError(_) => {
                "An unexpected error occurred during build optimizer (An error occurred in the process of cudas)."
            },
            LayerInstantiationError::SizeMismatchError(_) => {
                "An unexpected error occurred during layer instantiation. (Error during conversion to fixed length array)."
            },
            LayerInstantiationError::TypeConvertError(_) => {
                "An error occurred during a type conversion operation within the layer object creation process."
            },
            LayerInstantiationError::SpecializationError(_) => {
                "An unexpected error occurred during layer instantiation. (Error during data specialization)"
            }
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            #[cfg(feature = "cuda")]
            LayerInstantiationError::CudaError(ref e) => Some(e),
            #[cfg(feature = "cuda")]
            LayerInstantiationError::CudnnError(ref e) => Some(e),
            LayerInstantiationError::OptimizerBuildError(ref e) => Some(e),
            LayerInstantiationError::SizeMismatchError(ref e) => Some(e),
            LayerInstantiationError::TypeConvertError(ref e) => Some(e),
            LayerInstantiationError::SpecializationError(ref e) => Some(e),
        }
    }
}
#[cfg(feature = "cuda")]
impl From<CudaError> for LayerInstantiationError {
    fn from(err: CudaError) -> LayerInstantiationError {
        LayerInstantiationError::CudaError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcudnn::Error> for LayerInstantiationError {
    fn from(err: rcudnn::Error) -> LayerInstantiationError {
        LayerInstantiationError::CudnnError(err)
    }
}
impl From<OptimizerBuildError> for LayerInstantiationError {
    fn from(err: OptimizerBuildError) -> LayerInstantiationError {
        LayerInstantiationError::OptimizerBuildError(err)
    }
}
impl From<SizeMismatchError> for LayerInstantiationError {
    fn from(err: SizeMismatchError) -> LayerInstantiationError {
        LayerInstantiationError::SizeMismatchError(err)
    }
}
impl From<TypeConvertError> for LayerInstantiationError {
    fn from(err: TypeConvertError) -> LayerInstantiationError {
        LayerInstantiationError::TypeConvertError(err)
    }
}
impl From<SpecializationError> for LayerInstantiationError {
    fn from(err: SpecializationError) -> LayerInstantiationError {
        LayerInstantiationError::SpecializationError(err)
    }
}
/// Error when layer instantiation fails
#[derive(Debug)]
pub enum OptimizerBuildError {
    /// Error in cuda processing
    #[cfg(feature = "cuda")]
    CudaError(CudaError),
    /// Error in cudnn processing
    #[cfg(feature = "cuda")]
    CudnnError(rcudnn::Error),
}
impl fmt::Display for OptimizerBuildError {
    #[cfg(feature = "cuda")]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OptimizerBuildError::CudaError(e) => {
                write!(f,"An unexpected error occurred during build optimizer ({})",e)
            },
            OptimizerBuildError::CudnnError(e) => {
                write!(f,"An unexpected error occurred during build optimizer ({})",e)
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
        unreachable!()
    }
}
impl error::Error for OptimizerBuildError {
    #[cfg(feature = "cuda")]
    fn description(&self) -> &str {
        match self {
            OptimizerBuildError::CudaError(_) => {
                "An unexpected error occurred during build optimizer (An error occurred in the process of cudas)."
            },
            OptimizerBuildError::CudnnError(_) => {
                "An unexpected error occurred during build optimizer (An error occurred in the process of cudnn)."
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            OptimizerBuildError::CudaError(ref e) => Some(e),
            OptimizerBuildError::CudnnError(ref e) => Some(e)
        }
    }
    #[cfg(not(feature = "cuda"))]
    fn description(&self) -> &str {
        unreachable!()
    }

    #[cfg(not(feature = "cuda"))]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        unreachable!()
    }
}
#[cfg(feature = "cuda")]
impl From<CudaError> for OptimizerBuildError {
    fn from(err: CudaError) -> OptimizerBuildError {
        OptimizerBuildError::CudaError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcudnn::Error> for OptimizerBuildError {
    fn from(err: rcudnn::Error) -> OptimizerBuildError {
        OptimizerBuildError::CudnnError(err)
    }
}
/// Errors when generalizing data
#[derive(Debug)]
pub enum GeneralizationError {
    /// Error in cuda processing
    #[cfg(feature = "cuda")]
    CudaError(CudaError),
    /// Error in cudnn processing
    #[cfg(feature = "cuda")]
    CudnnError(rcudnn::Error),
    /// Errors caused by generating fixed-length arrays from different size Vecs
    SizeMismatchError(SizeMismatchError),
    /// Error generated when type conversion fails
    TypeConvertError(TypeConvertError),
}
impl fmt::Display for GeneralizationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(feature = "cuda")]
            GeneralizationError::CudaError(e) => {
                write!(f,"An unexpected error occurred during data generalization ({})",e)
            },
            #[cfg(feature = "cuda")]
            GeneralizationError::CudnnError(e) => {
                write!(f,"An unexpected error occurred during data generalization ({})",e)
            },
            GeneralizationError::SizeMismatchError(e) => {
                write!(f,"An unexpected error occurred during data generalization ({})",e)
            },
            GeneralizationError::TypeConvertError(e) => {
                write!(f,"An unexpected error occurred during data generalization ({})",e)
            }
        }
    }
}
impl error::Error for GeneralizationError {
    fn description(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            GeneralizationError::CudaError(_) => {
                "An unexpected error occurred during data generalization (An error occurred in the process of cudas)."
            },
            #[cfg(feature = "cuda")]
            GeneralizationError::CudnnError(_) => {
                "An unexpected error occurred during data generalization (An error occurred in the process of cudnns)."
            },
            GeneralizationError::SizeMismatchError(_) => {
                "An unexpected error occurred during data generalization. (Error during conversion to fixed length array)."
            },
            GeneralizationError::TypeConvertError(_) => {
                "An error occurred during a type conversion operation within the layer object creation process."
            }
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            #[cfg(feature = "cuda")]
            GeneralizationError::CudaError(ref e) => Some(e),
            #[cfg(feature = "cuda")]
            GeneralizationError::CudnnError(ref e) => Some(e),
            GeneralizationError::SizeMismatchError(ref e) => Some(e),
            GeneralizationError::TypeConvertError(ref e) => Some(e)
        }
    }
}
#[cfg(feature = "cuda")]
impl From<CudaError> for GeneralizationError {
    fn from(err: CudaError) -> GeneralizationError {
        GeneralizationError::CudaError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcudnn::Error> for GeneralizationError {
    fn from(err: rcudnn::Error) -> GeneralizationError {
        GeneralizationError::CudnnError(err)
    }
}
impl From<SizeMismatchError> for GeneralizationError {
    fn from(err: SizeMismatchError) -> GeneralizationError {
        GeneralizationError::SizeMismatchError(err)
    }
}
impl From<TypeConvertError> for GeneralizationError {
    fn from(err: TypeConvertError) -> GeneralizationError {
        GeneralizationError::TypeConvertError(err)
    }
}
/// Error during data specialization
#[derive(Debug)]
pub enum SpecializationError {
    /// Error in cuda processing
    #[cfg(feature = "cuda")]
    CudaError(CudaError),
    /// Error in cudnn processing
    #[cfg(feature = "cuda")]
    CudnnError(rcudnn::Error),
    /// Errors caused by generating fixed-length arrays from different size Vecs
    SizeMismatchError(SizeMismatchError),
    /// Error generated when type conversion fails
    TypeConvertError(TypeConvertError),
}
impl fmt::Display for SpecializationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            #[cfg(feature = "cuda")]
            SpecializationError::CudaError(e) => {
                write!(f,"An unexpected error occurred during data specialization ({})",e)
            },
            #[cfg(feature = "cuda")]
            SpecializationError::CudnnError(e) => {
                write!(f,"An unexpected error occurred during data specialization ({})",e)
            },
            SpecializationError::SizeMismatchError(e) => {
                write!(f,"An unexpected error occurred during data specialization ({})",e)
            },
            SpecializationError::TypeConvertError(e) => {
                write!(f,"An unexpected error occurred during data specialization ({})",e)
            }
        }
    }
}
impl error::Error for SpecializationError {
    fn description(&self) -> &str {
        match self {
            #[cfg(feature = "cuda")]
            SpecializationError::CudaError(_) => {
                "An unexpected error occurred during data specialization (An error occurred in the process of cudas)."
            },
            #[cfg(feature = "cuda")]
            SpecializationError::CudnnError(_) => {
                "An unexpected error occurred during data specialization (An error occurred in the process of cudnns)."
            },
            SpecializationError::SizeMismatchError(_) => {
                "An unexpected error occurred during data specialization. (Error during conversion to fixed length array)."
            },
            SpecializationError::TypeConvertError(_) => {
                "An error occurred during a type conversion operation within the layer object creation process."
            }
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            #[cfg(feature = "cuda")]
            SpecializationError::CudaError(ref e) => Some(e),
            #[cfg(feature = "cuda")]
            SpecializationError::CudnnError(ref e) => Some(e),
            SpecializationError::SizeMismatchError(ref e) => Some(e),
            SpecializationError::TypeConvertError(ref e) => Some(e)
        }
    }
}
#[cfg(feature = "cuda")]
impl From<CudaError> for SpecializationError {
    fn from(err: CudaError) -> SpecializationError {
        SpecializationError::CudaError(err)
    }
}
#[cfg(feature = "cuda")]
impl From<rcudnn::Error> for SpecializationError {
    fn from(err: rcudnn::Error) -> SpecializationError {
        SpecializationError::CudnnError(err)
    }
}
impl From<SizeMismatchError> for SpecializationError {
    fn from(err: SizeMismatchError) -> SpecializationError {
        SpecializationError::SizeMismatchError(err)
    }
}
impl From<TypeConvertError> for SpecializationError {
    fn from(err: TypeConvertError) -> SpecializationError {
        SpecializationError::TypeConvertError(err)
    }
}
