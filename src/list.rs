use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

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

    pub fn next(&self) -> Option<Rc<RefCell<ListNode<T>>>> {
        self.next.clone()
    }

    pub fn append(&mut self,next:ListNode<T>) {
        self.next = Some(Rc::new(RefCell::new(next)));
    }

    pub fn split(&mut self,value:T) -> Option<Rc<RefCell<ListNode<T>>>> {
        let next = self.next.take();
        let mut n = ListNode::new(value);

        n.next = next;

        let n = Some(Rc::new(RefCell::new(n)));
        self.next = n.clone();

        n
    }

    pub fn merge_next(&mut self) {
        let next = self.next.take();

        next.as_ref().map(|n| {
            self.next = n.deref().borrow().next.clone();
        });
    }
}