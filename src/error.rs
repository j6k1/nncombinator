use std::{error, fmt, io};
use std::ffi::CStr;
use std::fmt::{Debug, Formatter, write};
use std::num::ParseFloatError;
use cuda_runtime_sys::cudaError_t;

#[derive(Debug)]
pub enum TrainingError {
    ReferenceCountError,
    ThreadError(String),
    SizeMismatchError(SizeMismatchError),
    InvalidInputError(String),
    EvaluateError(EvaluateError),
    ToLargeInput(f64),
    CudaError(CudaError),
    CublasError(rcublas::error::Error),
    CudnnError(rcudnn::Error),
    CudaRuntimeError(CudaRuntimeError),
    InvalidStateError(InvalidStateError)
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
            TrainingError::CudaError(e) => write!(f, "An error occurred in the process of cuda. ({})",e),
            TrainingError::CublasError(e) => write!(f, "An error occurred during the execution of a process in cublas. ({})",e),
            TrainingError::CudnnError(e) => write!(f, "An error occurred during the execution of a process in cudnn. ({})",e),
            TrainingError::CudaRuntimeError(e) => write!(f,"{}",e),
            TrainingError::InvalidStateError(e) => write!(f,"Invalid state. ({})",e)
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
            TrainingError::CudaError(_) => "An error occurred in the process of cuda.",
            TrainingError::CublasError(_) => "An error occurred during the execution of a process in cublas.",
            TrainingError::CudnnError(_) => "An error occurred during the execution of a process in cudnn.",
            TrainingError::CudaRuntimeError(_) => "An error occurred while running the Cuda kernel.",
            TrainingError::InvalidStateError(_) => "Invalid state."
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
            TrainingError::CudaError(e) => Some(e),
            TrainingError::CublasError(e) => Some(e),
            TrainingError::CudnnError(e) => Some(e),
            TrainingError::CudaRuntimeError(_) => None,
            TrainingError::InvalidStateError(e) => Some(e)
        }
    }
}
#[derive(Debug)]
pub enum ConfigReadError {
    IOError(io::Error),
    InvalidState(String),
    ParseFloatError(ParseFloatError)
}
impl fmt::Display for ConfigReadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ConfigReadError::IOError(_) => write!(f, "Error occurred in file I/O."),
            ConfigReadError::InvalidState(ref s) => write!(f, "Configuration is invalid. ({})",s),
            ConfigReadError::ParseFloatError(_) => write!(f, "An error occurred when converting a string to a double value."),
        }
    }
}
impl error::Error for ConfigReadError {
    fn description(&self) -> &str {
        match *self {
            ConfigReadError::IOError(_) => "Error occurred in file I/O.",
            ConfigReadError::InvalidState(_) => "Configuration is invalid.",
            ConfigReadError::ParseFloatError(_) => "An error occurred when converting a string to a double value."
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            ConfigReadError::IOError(ref e) => Some(e),
            ConfigReadError::InvalidState(_) => None,
            ConfigReadError::ParseFloatError(ref e) => Some(e),
        }
    }
}
impl From<SizeMismatchError> for TrainingError {
    fn from(err: SizeMismatchError) -> TrainingError {
        TrainingError::SizeMismatchError(err)
    }
}
impl From<EvaluateError> for TrainingError {
    fn from(err: EvaluateError) -> TrainingError {
        TrainingError::EvaluateError(err)
    }
}
impl From<CudaError> for TrainingError {
    fn from(err: CudaError) -> TrainingError {
        TrainingError::CudaError(err)
    }
}
impl From<rcublas::error::Error> for TrainingError {
    fn from(err: rcublas::error::Error) -> TrainingError {
        TrainingError::CublasError(err)
    }
}
impl From<rcudnn::Error> for TrainingError {
    fn from(err: rcudnn::Error) -> TrainingError {
        TrainingError::CudnnError(err)
    }
}
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
impl From<io::Error> for ConfigReadError {
    fn from(err: io::Error) -> ConfigReadError {
        ConfigReadError::IOError(err)
    }
}
impl From<ParseFloatError> for ConfigReadError {
    fn from(err: ParseFloatError) -> ConfigReadError {
        ConfigReadError::ParseFloatError(err)
    }
}
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
    pub fn new(len:usize,index:usize) -> IndexOutBoundError {
        IndexOutBoundError {
            len:len,
            index:index
        }
    }
}
#[derive(Debug)]
pub enum EvaluateError {
    CudaError(CudaError),
    CublasError(rcublas::error::Error),
    CudnnError(rcudnn::Error),
    CudaRuntimeError(CudaRuntimeError),
    SizeMismatchError(SizeMismatchError),
    InvalidStateError(InvalidStateError)
}
impl fmt::Display for EvaluateError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EvaluateError::CudaError(e) => write!(f, "An error occurred in the process of cuda. ({})", e),
            EvaluateError::CublasError(e) => write!(f, "An error occurred during the execution of a process in cublas. ({})", e),
            EvaluateError::CudnnError(e) => write!(f, "An error occurred during the execution of a process in cudnn. ({})", e),
            EvaluateError::CudaRuntimeError(e) => write!(f,"{}",e),
            EvaluateError::SizeMismatchError(e) => write!(f,"{}",e),
            EvaluateError::InvalidStateError(e) => write!(f,"Invalid state. ({})",e)
        }
    }
}
impl error::Error for EvaluateError {
    fn description(&self) -> &str {
        match self {
            EvaluateError::CudaError(_) => "An error occurred in the process of cuda.",
            EvaluateError::CublasError(_) => "An error occurred during the execution of a process in cublas.",
            EvaluateError::CudnnError(_) => "An error occurred during the execution of a process in cudnn.",
            EvaluateError::CudaRuntimeError(_) => "An error occurred while running the Cuda kernel.",
            EvaluateError::SizeMismatchError(_) => "memory size does not match.",
            EvaluateError::InvalidStateError(_) => "Invalid state."
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            EvaluateError::CudaError(e) => Some(e),
            EvaluateError::CublasError(e) => Some(e),
            EvaluateError::CudnnError(e) => Some(e),
            EvaluateError::CudaRuntimeError(_) => None,
            EvaluateError::SizeMismatchError(e) => Some(e),
            EvaluateError::InvalidStateError(e) => Some(e)
        }
    }
}
impl From<CudaError> for EvaluateError {
    fn from(err: CudaError) -> EvaluateError {
        EvaluateError::CudaError(err)
    }
}
impl From<rcublas::error::Error> for EvaluateError {
    fn from(err: rcublas::error::Error) -> EvaluateError {
        EvaluateError::CublasError(err)
    }
}
impl From<rcudnn::Error> for EvaluateError {
    fn from(err: rcudnn::Error) -> EvaluateError {
        EvaluateError::CudnnError(err)
    }
}
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
impl From<InvalidStateError> for EvaluateError {
    fn from(err: InvalidStateError) -> EvaluateError {
        EvaluateError::InvalidStateError(err)
    }
}
#[derive(Debug)]
pub enum PersistenceError {
}
impl fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "An error occurred when saving model information.")
    }
}
impl error::Error for PersistenceError {
    fn description(&self) -> &str {
        "An error occurred when saving model information."
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}
#[derive(Debug)]
pub enum CudaError {
    AllocFailed(String),
    CudnnError(rcudnn::Error),
    CublasError(rcublas::error::Error),
    InvalidState(String),
    LogicError(String)
}
impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CudaError::AllocFailed(s) => write!(f, "{}", s),
            CudaError::CudnnError(e) => write!(f, "An error occurred during the execution of a process in cudnn. ({})", e),
            CudaError::CublasError(e) => write!(f, "An error occurred during the execution of a process in cublas. ({})", e),
            CudaError::InvalidState(s) => write!(f, "{}", s),
            CudaError::LogicError(s) => write!(f,"{}",s)
        }
    }
}
impl error::Error for CudaError {
    fn description(&self) -> &str {
        match self {
            CudaError::AllocFailed(_) => "Memory allocation failed.",
            CudaError::CudnnError(_) => "An error occurred during the execution of a process in cudnn.",
            CudaError::CublasError(_) => "An error occurred during the execution of a process in cublas.",
            CudaError::InvalidState(_) => "Invalid state.s",
            CudaError::LogicError(_) => "Logic error."
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            CudaError::AllocFailed(_) => None,
            CudaError::CudnnError(e) => Some(e),
            CudaError::CublasError(e) => Some(e),
            CudaError::InvalidState(_) => None,
            CudaError::LogicError(_) => None,
        }
    }
}
impl From<rcudnn::Error> for CudaError {
    fn from(err: rcudnn::Error) -> CudaError {
        CudaError::CudnnError(err)
    }
}
impl From<rcublas::Error> for CudaError {
    fn from(err: rcublas::error::Error) -> CudaError {
        CudaError::CublasError(err)
    }
}
#[derive(Debug)]
pub enum DeviceError {
    CudaError(CudaError),
    CublasError(rcublas::error::Error),
    CudnnError(rcudnn::Error),
}
impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DeviceError::CudaError(e) => write!(f, "An error occurred in the process of cuda. ({})", e),
            DeviceError::CublasError(e) => write!(f, "An error occurred during the execution of a process in cublas. ({})", e),
            DeviceError::CudnnError(e) => write!(f, "An error occurred during the execution of a process in cudnn. ({})", e),
        }
    }
}
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
impl From<CudaError> for DeviceError {
    fn from(err: CudaError) -> DeviceError {
        DeviceError::CudaError(err)
    }
}
impl From<rcublas::error::Error> for DeviceError {
    fn from(err: rcublas::error::Error) -> DeviceError {
        DeviceError::CublasError(err)
    }
}
impl From<rcudnn::Error> for DeviceError {
    fn from(err: rcudnn::Error) -> DeviceError {
        DeviceError::CudnnError(err)
    }
}
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
            InvalidStateError(_) => "Invalid state,"
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            InvalidStateError(_) => None,
        }
    }
}
pub struct CudaRuntimeError {
    raw: cudaError_t,
}
impl fmt::Debug for CudaRuntimeError {
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        write!(f,"{}",unsafe { CStr::from_ptr(cuda_runtime_sys::cudaGetErrorString(self.raw)) }.to_string_lossy())
    }
}
impl fmt::Display for CudaRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,"An error occurred while running the Cuda kernel. cause = {}",
            unsafe { CStr::from_ptr(cuda_runtime_sys::cudaGetErrorString(self.raw)) }.to_string_lossy()
        )
    }
}
impl error::Error for CudaRuntimeError {
    fn description(&self) -> &str {
        "An error occurred while running the Cuda kernel."
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
}
impl CudaRuntimeError {
    pub fn new(raw:cudaError_t) -> CudaRuntimeError {
        CudaRuntimeError {
            raw: raw
        }
    }
}