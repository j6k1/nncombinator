use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;
use nncombinator::cuda::mem::{Alloctype, MemoryPool};
use nncombinator::layer::{AskDiffInput, BackwardAll, BatchBackward, BatchForward, BatchLoss, BatchPreTrain, ForwardAll, ForwardDiff, Loss, PreTrain, UpdateWeight};
use nncombinator::ope::UnitValue;

lazy_static! {
    pub static ref SHARED_MEMORY_POOL:Arc<Mutex<MemoryPool>> = Arc::new(Mutex::new(MemoryPool::with_size(8 * 1024 * 1024 * 1024,Alloctype::Device).unwrap()));
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
pub fn assert_foward_diff<U: UnitValue<U>,T: ForwardDiff<U>>(_:&T) {

}
pub fn assert_ask_diff_input<U: UnitValue<U>,T: AskDiffInput<U>>(_:&T) {

}
pub fn assert_batch_forward<T: BatchForward>(_:&T) {

}
pub fn assert_batch_pre_train<U: UnitValue<U>,T: BatchPreTrain<U>>(_:&T) {

}
pub fn assert_batch_backward<U: UnitValue<U>,T: BatchBackward<U>>(_:&T) {

}
pub fn assert_batch_loss<U: UnitValue<U>,T: BatchLoss<U>>(_:&T) {

}
