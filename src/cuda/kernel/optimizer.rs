//! Implementation of various optimizers using Cuda

use core::fmt::Debug;
use std::marker::PhantomData;
use libc::{size_t,c_void};
use crate::cuda::{AsKernelPtr, CudaConstPtr, CudaPtr, CudaMutPtr, Kernel, KernelArgs};
use crate::cuda::allocator::CudaAllocator;

extern "C" {
    fn update_with_sgd_float(weight: *mut f32, grad: *const f32, size: size_t, a: f32, weight_decay: f32);
    fn update_with_sgd_double(weight: *mut f64, grad: *const f64, size: size_t, a: f64, weight_decay: f64);
    fn update_with_momentum_sgd_float(weight: *mut f32, grad: *const f32, size: size_t, a: f32, mu: f32, weight_decay: f32, vt: *mut f32);
    fn update_with_momentum_sgd_double(weight: *mut f64, grad: *const f64, size: size_t, a: f64, mu: f64, weight_decay: f64, vt: *mut f64);
    fn update_with_adagrad_float(weight: *mut f32, grad: *const f32, size: size_t, a: f32, weight_decay: f32, eps: f32, gt: *mut f32);
    fn update_with_adagrad_double(weight: *mut f64, grad: *const f64, size: size_t, a: f64, weight_decay: f64, eps: f64, gt: *mut f64);
    fn update_with_rmsprop_float(weight: *mut f32, grad: *const f32, size: size_t, a: f32, alpha: f32, mu: f32, eps: f32, gt: *mut f32, bt: *mut f32);
    fn update_with_rmsprop_double(weight: *mut f64, grad: *const f64, size: size_t, a: f64, alpha: f64, mu: f64, eps: f64, gt: *mut f64, bt: *mut f64);
    fn update_with_adam_float(weight: *mut f32, grad: *const f32, size: size_t, a: f32, weight_decay: f32, eps: f32, mt: *mut f32, vt: *mut f32, b1: f32, b2: f32, b1t: f32, b2t: f32);
    fn update_with_adam_double(weight: *mut f64, grad: *const f64, size: size_t, a: f64, weight_decay: f64, eps: f64, mt: *mut f64, vt: *mut f64, b1: f64, b2: f64, b1t: f64, b2t: f64);
}
/// Defines the list passed to the cuda kernel function as arguments to the SGD optimizer.
pub struct SGDArgs<'a,T,A> where T: Debug, A: CudaAllocator {
    weight: &'a mut CudaMutPtr<'a,T,A>,
    grad: CudaConstPtr<'a,CudaPtr<T,A>>,
    size: usize,
    a: T,
    weight_decay: T
}
/// Create an instance of an object representing the argument list of the SGD optimizer.
impl<'a,T,A> SGDArgs<'a,T,A> where T: Debug, A: CudaAllocator {
    /// Create a SGDArgs instance
    /// # Arguments
    /// * `weight` - unit weight
    /// * `grad` - gradient
    /// * `size` - number of weights to be updated
    /// * `a` - learning rate
    /// * `weight_decay` - Weight decay
    pub fn new(weight: &'a mut CudaMutPtr<'a,T,A>, grad: &'a CudaPtr<T,A>, size: usize, a: T, weight_decay: T) -> SGDArgs<'a,T,A> {
        SGDArgs {
            weight,
            grad: CudaConstPtr::new(grad),
            size,
            a,
            weight_decay
        }
    }
}
impl<'a,T,A> KernelArgs for SGDArgs<'a,T,A> where T: AsKernelPtr + Debug, A: CudaAllocator {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            self.weight,
            &mut self.grad,
            &mut self.size,
            &mut self.a,
            &mut self.weight_decay
        ]
    }
}
/// Implementation SGD optimizer
pub struct SGD<'a,T,A> where T: Debug, A: CudaAllocator {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A> SGD<'a,T,A> where T: Debug, A: CudaAllocator {
    /// Create a SGD optimizer instance
    pub fn new() -> SGD<'a,T,A> {
        SGD {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A> Kernel for SGD<'a,f32,A> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = update_with_sgd_float as *const c_void;
    type Args = SGDArgs<'a,f32,A>;
}
impl<'a,A> Kernel for SGD<'a,f64,A> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = update_with_sgd_double as *const c_void;
    type Args = SGDArgs<'a,f64,A>;
}
/// Defines the list passed to the cuda kernel function as arguments to the Momentum SGD optimizer.
pub struct MomentumSGDArgs<'a,T,A> where T: Debug, A: CudaAllocator {
    weight: &'a mut CudaMutPtr<'a,T,A>,
    grad: CudaConstPtr<'a,CudaPtr<T,A>>,
    size: usize,
    a: T,
    mu: T,
    weight_decay: T,
    vt: &'a mut CudaPtr<T,A>
}
/// Create an instance of an object representing the argument list of the Momentum SGD optimizer.
impl<'a,T,A> MomentumSGDArgs<'a,T,A> where T: Debug, A: CudaAllocator {
    /// Create a MomentumSGDArgs instance
    /// # Arguments
    /// * `weight` - unit weight
    /// * `grad` - gradient
    /// * `size` - number of weights to be updated
    /// * `a` - learning rate
    /// * `mu` - mu
    /// * `weight_decay` - Weight decay
    /// * `vt` - vt
    pub fn new(weight: &'a mut CudaMutPtr<'a,T,A>, grad: &'a CudaPtr<T,A>,
               size: usize, a: T, mu: T, weight_decay: T,
               vt: &'a mut CudaPtr<T,A>) -> MomentumSGDArgs<'a,T,A> {
        MomentumSGDArgs {
            weight,
            grad: CudaConstPtr::new(grad),
            size,
            a,
            mu,
            weight_decay,
            vt
        }
    }
}
impl<'a,T,A> KernelArgs for MomentumSGDArgs<'a,T,A> where T: AsKernelPtr + Debug, A: CudaAllocator {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            self.weight,
            &mut self.grad,
            &mut self.size,
            &mut self.a,
            &mut self.mu,
            &mut self.weight_decay,
            self.vt
        ]
    }
}
/// Implementation Momentum SGD optimizer
pub struct MomentumSGD<'a,T,A> where T: Debug, A: CudaAllocator {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A> MomentumSGD<'a,T,A> where T: Debug, A: CudaAllocator {
    /// Create a Momentum SGD optimizer instance
    pub fn new() -> MomentumSGD<'a,T,A> {
        MomentumSGD {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A> Kernel for MomentumSGD<'a,f32,A> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = update_with_momentum_sgd_float as *const c_void;
    type Args = MomentumSGDArgs<'a,f32,A>;
}
impl<'a,A> Kernel for MomentumSGD<'a,f64,A> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = update_with_momentum_sgd_double as *const c_void;
    type Args = MomentumSGDArgs<'a,f64,A>;
}
/// Defines the list passed to the cuda kernel function as arguments to the Adagrad optimizer.
pub struct AdagradArgs<'a,T,A> where T: Debug, A: CudaAllocator {
    weight: &'a mut CudaMutPtr<'a,T,A>,
    grad: CudaConstPtr<'a,CudaPtr<T,A>>,
    size: usize,
    a: T,
    weight_decay: T,
    eps: T,
    gt: &'a mut CudaPtr<T,A>
}
/// Create an instance of an object representing the argument list of the Adagrad optimizer.
impl<'a,T,A> AdagradArgs<'a,T,A> where T: Debug, A: CudaAllocator {
    /// Create a AdagradArgs instance
    /// # Arguments
    /// * `weight` - unit weight
    /// * `grad` - gradient
    /// * `size` - number of weights to be updated
    /// * `a` - learning rate
    /// * `weight_decay` - Weight decay
    /// * `eps` - Correction value to prevent zero division
    /// * `gt` - gt
    pub fn new(weight: &'a mut CudaMutPtr<'a,T,A>, grad: &'a CudaPtr<T,A>,
               size: usize, a: T, weight_decay: T, eps: T,
               gt: &'a mut CudaPtr<T,A>) -> AdagradArgs<'a,T,A> {
        AdagradArgs {
            weight,
            grad: CudaConstPtr::new(grad),
            size,
            a,
            weight_decay,
            eps,
            gt
        }
    }
}
impl<'a,T,A> KernelArgs for AdagradArgs<'a,T,A> where T: AsKernelPtr + Debug, A: CudaAllocator {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            self.weight,
            &mut self.grad,
            &mut self.size,
            &mut self.a,
            &mut self.weight_decay,
            &mut self.eps,
            self.gt
        ]
    }
}
/// Implementation Adagrad optimizer
pub struct Adagrad<'a,T,A> where T: Debug, A: CudaAllocator {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A> Adagrad<'a,T,A> where T: Debug, A: CudaAllocator {
    /// Create a Adagrad optimizer instance
    pub fn new() -> Adagrad<'a,T,A> {
        Adagrad {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A> Kernel for Adagrad<'a,f32,A> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = update_with_adagrad_float as *const c_void;
    type Args = AdagradArgs<'a,f32,A>;
}
impl<'a,A> Kernel for Adagrad<'a,f64,A> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = update_with_adagrad_double as *const c_void;
    type Args = AdagradArgs<'a,f64,A>;
}
/// Defines the list passed to the cuda kernel function as arguments to the Rmsprop optimizer.
pub struct RMSpropArgs<'a,T,A> where T: Debug, A: CudaAllocator {
    weight: &'a mut CudaMutPtr<'a,T,A>,
    grad: CudaConstPtr<'a,CudaPtr<T,A>>,
    size: usize,
    lr: T,
    weight_decay: T,
    alpha:T,
    mu: T,
    eps: T,
    gt: &'a mut CudaPtr<T,A>,
    bt: &'a mut CudaPtr<T,A>
}
/// Create an instance of an object representing the argument list of the Rmsprop optimizer.
impl<'a,T,A> RMSpropArgs<'a,T,A> where T: Debug, A: CudaAllocator {
    /// Create a RmspropArgs instance
    /// # Arguments
    /// * `weight` - unit weight
    /// * `grad` - gradient
    /// * `size` - number of weights to be updated
    /// * `a` - learning rate
    /// * `mu` - mu
    /// * `weight_decay` - Weight Decay
    /// * `eps` - Correction value to prevent zero division
    /// * `gt` - gt
    /// * `bt` - bt
    pub fn new(weight: &'a mut CudaMutPtr<'a,T,A>, grad: &'a CudaPtr<T,A>,
               size: usize, lr: T, weight_decay: T, alpha: T, mu: T, eps: T,
               gt: &'a mut CudaPtr<T,A>,
               bt: &'a mut CudaPtr<T,A>) -> RMSpropArgs<'a,T,A> {
        RMSpropArgs {
            weight,
            grad: CudaConstPtr::new(grad),
            size,
            lr,
            weight_decay,
            alpha,
            mu,
            eps,
            gt,
            bt
        }
    }
}
impl<'a,T,A> KernelArgs for RMSpropArgs<'a,T,A> where T: AsKernelPtr + Debug, A: CudaAllocator {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            self.weight,
            &mut self.grad,
            &mut self.size,
            &mut self.lr,
            &mut self.weight_decay,
            &mut self.alpha,
            &mut self.mu,
            &mut self.eps,
            self.gt,
            self.bt
        ]
    }
}
/// Implementation Rmsprop optimizer
pub struct RMSprop<'a,T,A> where T: Debug, A: CudaAllocator {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A> RMSprop<'a,T,A> where T: Debug, A: CudaAllocator {
    /// Create a Rmsprop optimizer instance
    pub fn new() -> RMSprop<'a,T,A> {
        RMSprop {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A> Kernel for RMSprop<'a,f32,A> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = update_with_rmsprop_float as *const c_void;
    type Args = RMSpropArgs<'a,f32,A>;
}
impl<'a,A> Kernel for RMSprop<'a,f64,A> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = update_with_rmsprop_double as *const c_void;
    type Args = RMSpropArgs<'a,f64,A>;
}
/// Defines the list passed to the cuda kernel function as arguments to the Adam optimizer.
pub struct AdamArgs<'a,T,A> where T: Debug, A: CudaAllocator {
    weight: &'a mut CudaMutPtr<'a,T,A>,
    grad: CudaConstPtr<'a,CudaPtr<T,A>>,
    size: usize,
    a: T,
    weight_decay: T,
    eps: T,
    mt: &'a mut CudaPtr<T,A>,
    vt: &'a mut CudaPtr<T,A>,
    b1: T,
    b2: T,
    b1t: T,
    b2t: T
}
/// Create an instance of an object representing the argument list of the Adam optimizer.
impl<'a,T,A> AdamArgs<'a,T,A> where T: Debug, A: CudaAllocator {
    /// Create a AdamArgs instance
    /// # Arguments
    /// * `weight` - unit weight
    /// * `grad` - gradient
    /// * `size` - number of weights to be updated
    /// * `a` - learning rate
    /// * `weight_decay` - Weight Decay
    /// * `eps` - Correction value to prevent zero division
    /// * `mt` - mt
    /// * `vt` - vt
    /// * `b1` - b1
    /// * `b2` - b2
    /// * `b1t` - b1t
    /// * `b2t` - b2t
    pub fn new(weight: &'a mut CudaMutPtr<'a,T,A>, grad: &'a CudaPtr<T,A>,
               size: usize, a: T, weight_decay: T, eps: T,
               mt: &'a mut CudaPtr<T,A>,
               vt: &'a mut CudaPtr<T,A>,b1: T, b2: T, b1t: T, b2t: T) -> AdamArgs<'a,T,A> {
        AdamArgs {
            weight,
            grad: CudaConstPtr::new(grad),
            size,
            a,
            weight_decay,
            eps,
            mt,
            vt,
            b1,
            b2,
            b1t,
            b2t
        }
    }
}
impl<'a,T,A> KernelArgs for AdamArgs<'a,T,A> where T: AsKernelPtr + Debug, A: CudaAllocator {
    fn as_vec(&mut self) -> Vec<&mut dyn AsKernelPtr> {
        vec![
            self.weight,
            &mut self.grad,
            &mut self.size,
            &mut self.a,
            &mut self.weight_decay,
            &mut self.eps,
            self.mt,
            self.vt,
            &mut self.b1,
            &mut self.b2,
            &mut self.b1t,
            &mut self.b2t
        ]
    }
}
/// Implementation Adam optimizer
pub struct Adam<'a,T,A> where T: Debug, A: CudaAllocator {
    t:PhantomData<T>,
    a:PhantomData<A>,
    l:PhantomData<&'a ()>
}
impl<'a,T,A> Adam<'a,T,A> where T: Debug, A: CudaAllocator {
    /// Create a Adam optimizer instance
    pub fn new() -> Adam<'a,T,A> {
        Adam {
            t: PhantomData::<T>,
            a: PhantomData::<A>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a,A> Kernel for Adam<'a,f32,A> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = update_with_adam_float as *const c_void;
    type Args = AdamArgs<'a,f32,A>;
}
impl<'a,A> Kernel for Adam<'a,f64,A> where A: CudaAllocator {
    const FUNC_PTR: *const c_void = update_with_adam_double as *const c_void;
    type Args = AdamArgs<'a,f64,A>;
}
