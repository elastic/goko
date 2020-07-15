# Point Cloud

A dataset access layer that allows for metadata to be attached to points. Used for `goko`. Currently this accelerates distance calculations with a set of `packed_simd` accelerated norms and a `rayon` threadpool while abstracting the access of the datapoints across multiple data files. It's structured in such a way that adding formats should be easy. 

## Planned Features

#### Current work
* Benchmarks.
* PCA, & Gaussian calculators.

#### Near Future
* Cleanup of the metadata feature in `pointcloud`
* Sparse accessors and sparse databacking

#### Future
* Network interface for distributed datasets.
* Image file abstraction for applications like imagenet.
* Asynchronous access for the network and file accessors.