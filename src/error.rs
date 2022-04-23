use std::{error, fmt, io};
use std::num::ParseFloatError;
#[derive(Debug)]
pub enum TrainingError {
    ReferenceCountError,
    ThreadError(String),
    SizeMismatchError(SizeMismatchError),
    InvalidInputError(String),
    EvaluateError(EvaluateError)
}
impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TrainingError::ReferenceCountError => write!(f, "There must be only one reference to Arc or Weak to itself when training is performed."),
            TrainingError::ThreadError(s) => write!(f, "An error occurred while processing the thread. cause = ({})",s),
            TrainingError::SizeMismatchError(e) => write!(f, "{}",e),
            TrainingError::InvalidInputError(s) => write!(f, "{}",s),
            TrainingError::EvaluateError(e) => write!(f, "{}",e),
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
            TrainingError::EvaluateError(_) => "An error occurred when running the neural network."
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            TrainingError::ReferenceCountError => None,
            TrainingError::ThreadError(_) => None,
            TrainingError::SizeMismatchError(e) => Some(e),
            TrainingError::InvalidInputError(_) => None,
            TrainingError::EvaluateError(e) => Some(e)
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
pub struct SizeMismatchError;
impl fmt::Display for SizeMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "memory size does not match.")
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

}
impl fmt::Display for EvaluateError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "An error occurred when running the neural network.")
    }
}
impl error::Error for EvaluateError {
    fn description(&self) -> &str {
        "An error occurred when running the neural network."
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
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
