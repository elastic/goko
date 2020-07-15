# Goko

## Plugins

These are going to be the core of the library. Attaching math gadgets to each node of the covertree is the most interesting thing you can do with them and is the goal of this library. The objective is to have zero-cost plugins on the fastest covertree implementation possible. It should be impossible to write a faster implementation of your covertree based algorithm. This is the third near total rewrite of the library. The previous 2 implementations had foundational bottlenecks and details that hindered this goal. We feel that this latest architecture has the strongest foundation for future work.

Plugins are currently partially implemented. Expect them in the coming weeks!

## Read-Write heads

`goko` stores the covertree as an array of layers, each with an arena style allocation of all nodes on that layer in a `evmap`. This is a pair of hashmaps where all the readers point at one constant map (so it can be safely, locklessly queried), while the writer edits the other map. When the writer is done it swaps the pointer. The readers then can see the updates, while the writer writes the same changes (and more) to the other hashmap. Plugins cannot control the update order, however the tree will be updated from leaf to root so you can use recursive logic. 

## Singleton Children

Children of a node that would only cover are stored in a singleton list. This saves a significant amount of memory (Trees built on Ember are 2-3x smaller) and you can chose not to query these nodes to speed up KNN & other queries. Considerations should be made when writing algorithms for this implementation for these unique children.

## Planned Features

#### Current work
* Plugins! Math gadgets attached to each node.
* Benchmarks
* Inserts

#### Near Future
* Nerve encoding for potential TDA, and making us of the ||isation
* Plugin support for the nerve.

#### Future
* Move to either `stdasync` or `tokio` to handle coroutines to integrate with services more easily.