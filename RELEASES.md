Version 0.8.0 (2023-12-16)
===========================

Refactoring and functionality additions
--------
Changed the type name of VecArr to SerializedVec
Changed the type name of VecArrIter to SerializedVecIter
Changed the type name of VecArrIterMut to VecArrIterMut

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

-
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
