use crate::error::TypeConvertError;

/// Trait for inverse conversion of value to host memory type
pub trait ToHost<T> where T: Default + Clone + Send {
    type Output;

    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TypeConvertError`]
    ///
    fn to_host(self) -> Result<Self::Output,TypeConvertError>;
}
