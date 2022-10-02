//! Implementation of a unidirectional list to be used for memory management in a memory pool

use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

/// Single-way chained list
#[derive(Debug)]
pub struct ListNode<T> {
    next:Option<Rc<RefCell<ListNode<T>>>>,
    pub value:T
}
impl<T> ListNode<T> {
    pub fn new(value:T) -> ListNode<T> {
        ListNode {
            next:None,
            value:value
        }
    }

    /// Returns Rc to the next element of the list wrapped in Option
    pub fn next(&self) -> Option<Rc<RefCell<ListNode<T>>>> {
        self.next.clone()
    }

    /// Add the next item in the list. (If it is already there, it will be overwritten.)
    /// # Arguments
    /// * `next` - next item
    pub fn append(&mut self,next:ListNode<T>) {
        self.next = Some(Rc::new(RefCell::new(next)));
    }

    /// Inserts a new item at the current next position in the list and returns the inserted item
    /// # Arguments
    /// * `value` - New Item Value
    pub fn split(&mut self,value:T) -> Option<Rc<RefCell<ListNode<T>>>> {
        let next = self.next.take();
        let mut n = ListNode::new(value);

        n.next = next;

        let n = Some(Rc::new(RefCell::new(n)));
        self.next = n.clone();

        n
    }

    /// Delete the current next item and rebuild the listing
    pub fn merge_next(&mut self) {
        let next = self.next.take();

        next.as_ref().map(|n| {
            self.next = n.deref().borrow().next.clone();
        });
    }
}