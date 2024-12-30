//! Implementation of various optimizers using Cuda

use std::marker::PhantomData;
use libc::{size_t,c_void};
use crate::cuda::{AsKernelPtr, CudaConstPtr, CudaMemoryPoolPtr, CudaMutPtr, Kernel, KernelArgs};

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
pub struct SGDArgs<'a,T> {
    weight: &'a mut CudaMutPtr<'a,CudaMemoryPoolPtr<T>>,
    grad: CudaConstPtr<'a,CudaMemoryPoolPtr<T>>,
    size: usize,
    a: T,
    weight_decay: T
}
/// Create an instance of an object representing the argument list of the SGD optimizer.
impl<'a,T> SGDArgs<'a,T> {
    /// Create a SGDArgs instance
    /// # Arguments
    /// * `weight` - unit weight
    /// * `grad` - gradient
    /// * `size` - number of weights to be updated
    /// * `a` - learning rate
    /// * `weight_decay` - Weight decay
    pub fn new(weight: &'a mut CudaMutPtr<'a,CudaMemoryPoolPtr<T>>, grad: &'a CudaMemoryPoolPtr<T>, size: usize, a: T, weight_decay: T) -> SGDArgs<'a,T> {
        SGDArgs {
            weight,
            grad: CudaConstPtr::new(grad),
            size,
            a,
            weight_decay
        }
    }
}
impl<'a,T> KernelArgs for SGDArgs<'a,T> where T: AsKernelPtr {
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
pub struct SGD<'a,T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T> SGD<'a,T> {
    /// Create a SGD optimizer instance
    pub fn new() -> SGD<'a,T> {
        SGD {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a> Kernel for SGD<'a,f32> {
    const FUNC_PTR: *const c_void = update_with_sgd_float as *const c_void;
    type Args = SGDArgs<'a,f32>;
}
impl<'a> Kernel for SGD<'a,f64> {
    const FUNC_PTR: *const c_void = update_with_sgd_double as *const c_void;
    type Args = SGDArgs<'a,f64>;
}
/// Defines the list passed to the cuda kernel function as arguments to the Momentum SGD optimizer.
pub struct MomentumSGDArgs<'a,T> {
    weight: &'a mut CudaMutPtr<'a,CudaMemoryPoolPtr<T>>,
    grad: CudaConstPtr<'a,CudaMemoryPoolPtr<T>>,
    size: usize,
    a: T,
    mu: T,
    weight_decay: T,
    vt: &'a mut CudaMemoryPoolPtr<T>
}
/// Create an instance of an object representing the argument list of the Momentum SGD optimizer.
impl<'a,T> MomentumSGDArgs<'a,T> {
    /// Create a MomentumSGDArgs instance
    /// # Arguments
    /// * `weight` - unit weight
    /// * `grad` - gradient
    /// * `size` - number of weights to be updated
    /// * `a` - learning rate
    /// * `mu` - mu
    /// * `weight_decay` - Weight decay
    /// * `vt` - vt
    pub fn new(weight: &'a mut CudaMutPtr<'a,CudaMemoryPoolPtr<T>>, grad: &'a CudaMemoryPoolPtr<T>,
               size: usize, a: T, mu: T, weight_decay: T,
               vt: &'a mut CudaMemoryPoolPtr<T>) -> MomentumSGDArgs<'a,T> {
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
impl<'a,T> KernelArgs for MomentumSGDArgs<'a,T> where T: AsKernelPtr {
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
pub struct MomentumSGD<'a,T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T> MomentumSGD<'a,T> {
    /// Create a Momentum SGD optimizer instance
    pub fn new() -> MomentumSGD<'a,T> {
        MomentumSGD {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a> Kernel for MomentumSGD<'a,f32> {
    const FUNC_PTR: *const c_void = update_with_momentum_sgd_float as *const c_void;
    type Args = MomentumSGDArgs<'a,f32>;
}
impl<'a> Kernel for MomentumSGD<'a,f64> {
    const FUNC_PTR: *const c_void = update_with_momentum_sgd_double as *const c_void;
    type Args = MomentumSGDArgs<'a,f64>;
}
/// Defines the list passed to the cuda kernel function as arguments to the Adagrad optimizer.
pub struct AdagradArgs<'a,T> {
    weight: &'a mut CudaMutPtr<'a,CudaMemoryPoolPtr<T>>,
    grad: CudaConstPtr<'a,CudaMemoryPoolPtr<T>>,
    size: usize,
    a: T,
    weight_decay: T,
    eps: T,
    gt: &'a mut CudaMemoryPoolPtr<T>
}
/// Create an instance of an object representing the argument list of the Adagrad optimizer.
impl<'a,T> AdagradArgs<'a,T> {
    /// Create a AdagradArgs instance
    /// # Arguments
    /// * `weight` - unit weight
    /// * `grad` - gradient
    /// * `size` - number of weights to be updated
    /// * `a` - learning rate
    /// * `weight_decay` - Weight decay
    /// * `eps` - Correction value to prevent zero division
    /// * `gt` - gt
    pub fn new(weight: &'a mut CudaMutPtr<'a,CudaMemoryPoolPtr<T>>, grad: &'a CudaMemoryPoolPtr<T>,
               size: usize, a: T, weight_decay: T, eps: T,
               gt: &'a mut CudaMemoryPoolPtr<T>) -> AdagradArgs<'a,T> {
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
impl<'a,T> KernelArgs for AdagradArgs<'a,T> where T: AsKernelPtr {
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
pub struct Adagrad<'a,T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T> Adagrad<'a,T> {
    /// Create a Adagrad optimizer instance
    pub fn new() -> Adagrad<'a,T> {
        Adagrad {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a> Kernel for Adagrad<'a,f32> {
    const FUNC_PTR: *const c_void = update_with_adagrad_float as *const c_void;
    type Args = AdagradArgs<'a,f32>;
}
impl<'a> Kernel for Adagrad<'a,f64> {
    const FUNC_PTR: *const c_void = update_with_adagrad_double as *const c_void;
    type Args = AdagradArgs<'a,f64>;
}
/// Defines the list passed to the cuda kernel function as arguments to the Rmsprop optimizer.
pub struct RMSpropArgs<'a,T> {
    weight: &'a mut CudaMutPtr<'a,CudaMemoryPoolPtr<T>>,
    grad: CudaConstPtr<'a,CudaMemoryPoolPtr<T>>,
    size: usize,
    lr: T,
    weight_decay: T,
    alpha:T,
    mu: T,
    eps: T,
    gt: &'a mut CudaMemoryPoolPtr<T>,
    bt: &'a mut CudaMemoryPoolPtr<T>
}
/// Create an instance of an object representing the argument list of the Rmsprop optimizer.
impl<'a,T> RMSpropArgs<'a,T> {
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
    pub fn new(weight: &'a mut CudaMutPtr<'a,CudaMemoryPoolPtr<T>>, grad: &'a CudaMemoryPoolPtr<T>,
               size: usize, lr: T, weight_decay: T, alpha: T, mu: T, eps: T,
               gt: &'a mut CudaMemoryPoolPtr<T>,
               bt: &'a mut CudaMemoryPoolPtr<T>) -> RMSpropArgs<'a,T> {
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
impl<'a,T> KernelArgs for RMSpropArgs<'a,T> where T: AsKernelPtr {
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
pub struct RMSprop<'a,T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T> RMSprop<'a,T> {
    /// Create a Rmsprop optimizer instance
    pub fn new() -> RMSprop<'a,T> {
        RMSprop {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a> Kernel for RMSprop<'a,f32> {
    const FUNC_PTR: *const c_void = update_with_rmsprop_float as *const c_void;
    type Args = RMSpropArgs<'a,f32>;
}
impl<'a> Kernel for RMSprop<'a,f64> {
    const FUNC_PTR: *const c_void = update_with_rmsprop_double as *const c_void;
    type Args = RMSpropArgs<'a,f64>;
}
/// Defines the list passed to the cuda kernel function as arguments to the Adam optimizer.
pub struct AdamArgs<'a,T> {
    weight: &'a mut CudaMutPtr<'a,CudaMemoryPoolPtr<T>>,
    grad: CudaConstPtr<'a,CudaMemoryPoolPtr<T>>,
    size: usize,
    a: T,
    weight_decay: T,
    eps: T,
    mt: &'a mut CudaMemoryPoolPtr<T>,
    vt: &'a mut CudaMemoryPoolPtr<T>,
    b1: T,
    b2: T,
    b1t: T,
    b2t: T
}
/// Create an instance of an object representing the argument list of the Adam optimizer.
impl<'a,T> AdamArgs<'a,T> {
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
    pub fn new(weight: &'a mut CudaMutPtr<'a,CudaMemoryPoolPtr<T>>, grad: &'a CudaMemoryPoolPtr<T>,
               size: usize, a: T, weight_decay: T, eps: T,
               mt: &'a mut CudaMemoryPoolPtr<T>,
               vt: &'a mut CudaMemoryPoolPtr<T>,b1: T, b2: T, b1t: T, b2t: T) -> AdamArgs<'a,T> {
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
impl<'a,T> KernelArgs for AdamArgs<'a,T> where T: AsKernelPtr {
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
pub struct Adam<'a,T> {
    t:PhantomData<T>,
    l:PhantomData<&'a ()>
}
impl<'a,T> Adam<'a,T> {
    /// Create a Adam optimizer instance
    pub fn new() -> Adam<'a,T> {
        Adam {
            t: PhantomData::<T>,
            l: PhantomData::<&'a ()>
        }
    }
}
impl<'a> Kernel for Adam<'a,f32> {
    const FUNC_PTR: *const c_void = update_with_adam_float as *const c_void;
    type Args = AdamArgs<'a,f32>;
}
impl<'a> Kernel for Adam<'a,f64> {
    const FUNC_PTR: *const c_void = update_with_adam_double as *const c_void;
    type Args = AdamArgs<'a,f64>;
}
