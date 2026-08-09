Version 0.9.8 (2026-08-09)
===========================

Major changes
--------

- Added support for building in environments where CUDA is not installed.

Important bugfix
--------

- Fixed an issue in batch back propagation where errors propagated to parent layers
  were calculated from values that had already been divided by the batch size at
  the output layer. Errors are now propagated first and divided by the batch size
  in each layer as required, fixing incorrect training behavior.

Bugfix
--------

- Fixed cases where `TextPersistence` could not correctly handle comment lines
  in model files.
- Fixed missing boundary checks in the `Iterator` implementations for
  `Arr2Iter` and `Arr2IterMut`.

Function additions
--------

- Added learning rate scheduler support.
- Added the `CudaAllocator` trait and consolidated CUDA-related memory
  allocation behind its implementations.
- Added the `PartialForward`, `ForwardDiff`, and `ContinueForward` traits and
  implemented them for various layers.
- Added `LoggingLayer`.
- Added the `Step` and `OnStep` traits and implemented them for various layers.
- Added the `DeviceBatchAveraging` trait.
- Added `MemoryWriter`.
- Added `CudaPtrRef`, `AddBiasBatch`, and `AddBiasBatchArgs`.
- Added AdamW support.
- Added `DeviceAllocator`, `HostAllocator`, and `MemoryPoolAllocator`.
- Added the `AsCudaReadOnlyPtr`, `DeriveCudaConstPtr`, `CudaView`,
  `AsCudaView`, and `MemoryType` traits.
- Added `ModelLoadError` and `GeneralizationError`.
- Added `TrainingError::FromPrimitiveError` and
  `TrainingError::ModelLoadError`.
- Added `LayerInstantiationError::SpecializationError`.
- Added `From<ModelLoadError>` and `From<FromPrimitiveError>` implementations
  for `TrainingError`.
- Added a `From<SpecializationError>` implementation for
  `LayerInstantiationError`.
- Added `ShieldSlice::as_chunks_mut`.
- Added `From<[T; N]>` for `Arr`.
- Added `Index` and `IndexMut` implementations for `Arr2` that return `&[T]`
  and `&mut [T]`.
- Added the `VerifyEof` trait.
- Added the `Cos` trait and implemented it for `f32` and `f64`.
- Added the `TextRecord` type.
- Added `ClippedReLu` and `LeakyReLu` activation function implementations.
- Added `Kernel::launch_config`.
- Added `CudaPtr::with_initializer`.

Specification change
--------

- Changed the `Persistence::load` error type.
- Changed the `LinearPersistence::read` error type.
- Changed the error types returned by methods on `TextFilePersistence` and
  `BinFilePersistence`.
- Added a type parameter to the `SaveToFile` trait.
- Changed `UpdateWeight::update_weigth` to accept the batch size as its second
  argument.
- Changed the `DiffInput` specification.
- Removed the `AskDiffInput` trait.
- Changed `as_ptr` functions that accepted mutable references to accept
  immutable references instead.
- Changed `DeviceBias` batch execution to use the `AddBiasBatch`
  implementation.
- Implemented `DeviceBatchAveraging` for `DeviceCpu` and `DeviceGpu`.
- Removed the unsafe `Sync` implementation from `MemoryPool`.
- Updated signatures for activation-function-related traits and their
  implementations.
- Removed dimension-related arguments from `Kernel::launch`; dimension
  information is now returned from `Kernel::launch_config`.
- Moved the definition location of the `ToHost` trait.
- Changed `LossFunctionLinear` signatures.
- Added `on_step` and `on_frequently_step` methods to the `Optimizer` trait.

CUDA and memory API changes
--------

- Added `generalization_vars` and `specialization_vars` methods to the
  `DeviceBatchNorm` trait.
- Added `generalization_bias` and `specialization_bias` methods to the
  `DeviceBias` trait.
- Added `generalization_units`, `generalization_bias`, `specialization_units`,
  and `specialization_bias` methods to the `DeviceLinear` trait.
- Implemented `AsPtr<T>`, `AsMutPtr<T>`, and `AsMutKernelPtrBase` for `u32`,
  `i64`, `u64`, `usize`, `f32`, and `f64`.
- Implemented `MemorySize` for `CudaTensor1dPtrView`, `CudaTensor2dPtrView`,
  `CudaTensor3dPtrView`, and `CudaTensor4dPtrView`.
- Changed inherited traits for `WriteMemory` and `WriteMemoryAsync` from
  `AsMutVoidPtr` to `AsMutPtr<T>`.
- Implemented `AsCudaReadOnlyPtr` for `CudaPtr`.
- Removed `CudaMemoryPoolPtr`.
- Implemented `CudaView` for `CudaTensor1dPtr`, `CudaTensor2dPtr`,
  `CudaTensor3dPtr`, and `CudaTensor4dPtr`.
- Implemented `AsCudaReadOnlyPtr` and `DeriveCudaConstPtr` for
  `CudaTensor1dPtr`, `CudaTensor1dPtrView`, `CudaTensor2dPtr`,
  `CudaTensor2dPtrView`, `CudaTensor3dPtr`, `CudaTensor3dPtrView`,
  `CudaTensor4dPtr`, `CudaTensor4dPtrView`, `CudaVec`, and `CudaVecView`.
- Implemented `AsCudaReadOnlyPtr` for `&CudaTensor1dPtr`,
  `&CudaTensor2dPtr`, `&CudaTensor3dPtr`, and `&CudaTensor4dPtr`.
- Implemented `AsCudaView` for `&CudaTensor1dPtr`, `&CudaTensor2dPtr`,
  `&CudaTensor3dPtr`, and `&CudaTensor4dPtr`.
- Implemented `TryFrom` to convert references of CUDA tensor pointer types to
  their owned non-reference types.

Version 0.9.0 (2025-01-02)
===========================

Refactoring and functionality additions
--------

Type additions
- 
- ShieldSlice
- AsConstKernelPtrBase
- AsConstKernelPtr
- TryClone
- PointerElement
- ReadMemory
- WriteMemory
- ReadMemoryAsync
- WriteMemoryAsync
- CudaConstPtr
- CudaTensor1dPtr
- CudaTensor2dPtr
- CudaTensor3dPtr
- CudaTensor4dPtr
- CudaTensor1dPtrView
- CudaTensor2dPtrView
- CudaTensor3dPtrView
- CudaTensor4dPtrView
- CudaVec
- AsCudaPtrRef
- AsCudaMutPtr
- ToCuda
- BatchDataType
- LossFunctionLinear
- BatchLossFunctionLinear
- Product
- OptimizerBuilder
- OptimizerState


Remove Type
-
- Memory (Split into ReadMemory and WriteMemory and redefine)
- MemoryAsync (Split into ReadMemoryAsync and WriteMemoryAsync and redefine)

Add module
- cuda::kernel::optimizer
- device::activation
- device::input

Add Functions
-
- Add Arr::iter_mut,Arr::as_mut_ptr,Arr::index(trait),Arr::index_mut(trait)
- Add Arr2::as_ptr,Arr2::as_mut_ptr
- Add Arr3::as_ptr,Arr3::as_mut_ptr
- Add Arr4::as_ptr,Arr4::as_mut_ptr
- Add ArrViewMut::iter_mut
- Add Arr2View::as_ptr
- Add Arr2Iter::nth(trait)
- Add Arr2ViewMut::as_ptr,Arr2ViewMut::as_mut_ptr
- Add Arr2IterMut::nth(trait)
- Add Arr3View::as_ptr
- Add DiffArr::len
- Add SerializedVec::to_vec,SerializedVec::as_ptr,SerializedVec::as_mut_ptr
- Add SerializedVecView::as_ptr
- Add launch_cooperative,launch_cooperative_with_stream

specification change
-
- Remove DerefMut Trait implementation from Arr,ArrViewMut.
- A newly defined error is now returned in situations where
  SizeMismatchError was being returned.
- Changed the type of value returned by AsRawMutSlice to ShieldSlice
- Changed Optimizer Trait specifications
- Addition of From implementation for various collection types
- Partitioning of layer and device modules

Internal specification change
-
- When computing with the GPU version of the model,
  parameters are passed to and from subsequent layers directly in GPU memory.

Other
-
- Required WMMA API support for cuda kernel builds.

Bugfix
-
- Alignment is now taken into account when allocating memory from the memory pool.
- Fixed an issue where forward and back propagation calculations for linear layers were sometimes incorrect.

Version 0.8.0 (2023-12-16)
===========================

Refactoring and functionality additions
--------

Changed the type name
-
- Changed the type name of VecArr to SerializedVec
- Changed the type name of VecArrIter to SerializedVecIter
- Changed the type name of VecArrIterMut to VecArrIterMut

Type additions
- 
- SerializedVecView
- SliceSize
- MakeView
- MakeViewMut
- Arr3View
- Arr3ViewMut
- Arr3ParIter
- SerializedVecIterProducer
- SerializedVecParIter
- SerializedVecConverter
- Arr3ViewMut
- Arithmetic

- Partitioning of layer and device modules

Add module
- 
- collections
- computational_graph
- cuda::cudnn::tensor
- device::batchnormalization
- device::bias
- layer::batchnormalization
- layer::bias
- layer::bridge

Add Functions
-
- Add DataTypeInfo::cudnn_raw_data_type method
- Add Stack::take_map method

changed interface of layer,device implementation
-

Version 0.7.0 (2022-12-06)
===========================

BugFix
--------
- Corrected an error in the calculation process of derive and batch_derive in the activation function.
- Fixed a problem in which allocating memory from the memory pool by specifying size 0 did not result in an error, and also caused a panic in the drop after allocating memory by specifying size 0 multiple times
- Fixed typo in cuda kernel function name
- In the forward implementation of softmax in the cuda kernel, the process of multiplying the value of each element by scale before summing and dividing the result by scale has been removed.
- When updating the weights of units in a layer, the gradient is divided by the batch size.


Specification Change
--------
- Fixed batch_loss_linear_total to return results divided by batch size

Performance Tuning
--------
- Parallelization of the calculation of the CPU version of the activation function

Function addtion
--------
- The len method of VecArr can be used even if the type of T is not Arr<U,N>.
- TextFilePersistence, and BinFilePersistence constructors can be passed any type implementing AsRef<Path>.

Version 0.6.0 (2022-10-13)
===========================

BugFix
--------
- This version fixes a bug in which the calculation of error back propagation was not implemented correctly, which had continued to exist since earlier versions.
