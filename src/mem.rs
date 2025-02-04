//! Functions related to cuda memory
use crate::arr::ShieldSlice;

/// Conversion to immutable slices
pub trait AsRawSlice<T> {
    fn as_raw_slice(&self) -> &[T];
}
/// Conversion to mmutable slices
pub trait AsRawMutSlice<'a,T> {
    fn as_raw_mut_slice(&'a mut self) -> ShieldSlice<'a,T>;
}
