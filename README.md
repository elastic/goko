# Grandma, a KNN Library

This is a covertree library with some modifications to make it more sutible for real data. Currently it only implements the [fast covertree](http://proceedings.mlr.press/v37/izbicki15.pdf). There are plans to enable support for GMRA

## Project Layout 

Data Access is handled through the `pointcloud` library. Currently this supports memory maps (for NVMe backed datasets) and ram based data. We will support large file based datasets that cannot be loaded in memory concurrently and loading many formats into ram. The covertree implementation is inside the `grandma` library, and the `pygrandma` library is a python & numpy partial wrap around `grandma`.

## Unusual Choices

`grandma` stores the covertree as an array of layers, each with an arena style allocation of all nodes on that layer in a `evmap`. This is a pair of hashmaps where all the readers point at one constant map (so it can be safely, locklessly queried), while the writer edits the other map. When the writer is done it swaps the pointer. The readers then can see the updates, while the writer writes the same changes (and more) to the other hashmap. Plugins cannot control the update order, however the tree will be updated from leaf to root so you can use recursive logic. 

Children of a node that would only cover are stored in a singleton list. This saves a significant amount of memory (Trees built on Ember are 2-3x smaller) and you can chose not to query these nodes to speed up KNN & other queries. Considerations should be made when writing algorithms for this implementation for these unique children.

If you need to control update order you might be doing something awesome. Feel free to open an issue.

## Planned Features

#### Current work
* Plugins! Math gadgets attached to each node.
* Benchmarks

#### Near Future
* Nerve encoding for potential TDA, and making us of the ||isation
* Cleanup of the metadata feature in `pointcloud`

#### Future
* Network interface for the `pointcloud`


#### License

<sup>
Licensed under of <a href="LICENSE.txt">Apache License, Version 2.0</a>.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be licensed as above, without any additional terms or conditions.
</sub>