use std::{error, fmt, io};
use std::num::ParseFloatError;
#[derive(Debug)]
pub enum TrainingError {
    ReferenceCountError,
    ThreadError(String),
}
impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TrainingError::ReferenceCountError => write!(f, "There must be only one reference to Arc or Weak to itself when training is performed."),
            TrainingError::ThreadError(s) => write!(f, "An error occurred while processing the thread. cause = ({})",s),
        }
    }
}
impl error::Error for TrainingError {
    fn description(&self) -> &str {
        match self {
            TrainingError::ReferenceCountError => "There must be only one reference to Arc or Weak to itself when training is performed.",
            TrainingError::ThreadError(_) => "An error occurred while processing the thread.",
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            TrainingError::ReferenceCountError => None,
            TrainingError::ThreadError(_) => None,
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
