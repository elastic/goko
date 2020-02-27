# Grandma, a KNN Library

This is a covertree library with some modifications to make it more suitable for real data. Currently it only implements the [fast covertree](http://proceedings.mlr.press/v37/izbicki15.pdf), which is an extension of the original covertree [(pdf)](https://homes.cs.washington.edu/~sham/papers/ml/cover_tree.pdf). There are plans to enable support for full [geometric multi-resolution analysis](https://arxiv.org/pdf/1611.01179.pdf) (GMRA, where the library get it's name from) and [topological data analysis](https://arxiv.org/pdf/1602.06245.pdf). Help is welcome! We'd love to collaborate on more cool tricks to do with covertrees or coding up the large backlog of planned features to support the current known tricks.

## Project Layout & Documentation

Data Access is handled through the `pointcloud` library. See [here](https://docs.rs/pointcloud) for `pointcloud`'s documentation. This is meant to abstract many files and make them look like one, and due to this handles computations like adjacency matrices. The covertree implementation is inside the `grandma` library, it's the bread and butter of the library. See [here](https://docs.rs/grandma) for it's documentation. 

The `pygrandma` library is a python & numpy partial wrap around `grandma`. It can access the components of `grandma` for gathering statistics on your trees. Once we settle on how this is implemented we will publish the documentation somewhere.


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
