use std::fmt::Debug;
use lazy_static::lazy_static;
use nncombinator::device::input::DeviceInput;
use nncombinator::layer::{BackwardAll, BatchBackward, BatchDataType, BatchForward, BatchLoss, BatchPreTrain, ForwardAll, Loss, PartialForward, PreTrain, UpdateWeight};
use nncombinator::ope::UnitValue;
use nncombinator::cuda::allocator::{DeviceAlloc, MemoryPoolAllocator, MemoryPoolAllocatorInstantiation};

lazy_static! {
    pub static ref SHARED_MEMORY_POOL:MemoryPoolAllocator<DeviceAlloc> = MemoryPoolAllocator::with_size(8 * 1024 * 1024 * 1024,DeviceAlloc).unwrap();
}
pub fn assert_device<U,I,D: DeviceInput<U,I>>(_:&D)
    where U: UnitValue<U>,
          I: BatchDataType + Debug + 'static,
          <I as BatchDataType>::Type: Debug + 'static {
}
pub fn assert_forward_all<T: ForwardAll>(_:&T) {

}
pub fn assert_pre_train<U: UnitValue<U>,T: PreTrain<U>>(_:&T) {

}
pub fn assert_backward_all<U: UnitValue<U>,T: BackwardAll<U>>(_:&T) {

}
pub fn assert_loss<U: UnitValue<U>,T: Loss<U>>(_:&T) {

}
pub fn assert_update_weight<U: UnitValue<U>,T: UpdateWeight<U>>(_:&T) {

}
pub fn assert_partial_forward<T: PartialForward>(_:&T) {}
pub fn assert_batch_forward<T: BatchForward>(_:&T) {

}
pub fn assert_batch_pre_train<U: UnitValue<U>,T: BatchPreTrain<U>>(_:&T) {

}
pub fn assert_batch_backward<U: UnitValue<U>,T: BatchBackward<U>>(_:&T) {

}
pub fn assert_batch_loss<U: UnitValue<U>,T: BatchLoss<U>>(_:&T) {

}
