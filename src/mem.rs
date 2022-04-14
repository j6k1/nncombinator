pub trait AsRawSlice<T> {
    fn as_raw_slice(&self) -> &[T];
}
pub trait AsRawMutSlice<'a,T> {
    fn as_raw_mut_slice(&'a mut self) -> &'a mut [T];
}
