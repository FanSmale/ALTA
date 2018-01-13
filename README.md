# ALTA
In this paper, we propose an effective and adaptive algorithm that will be called active learning through two-stage clustering (ALTA).

The first stage is data preprocessing using the two-round-clustering algorithm.
Let $n$ be the number of instances and obtain $\sqrt{n}$ small blocks.
For each block, the closest instance of the center is selected as the sample.
The second stage is the active learning of sampling instances through density clustering.
This stage consists of a number of iterations of density clustering, labeling and classification.

In general, data preprocessing reduces the size of the data and the complexity of the algorithm.
The combination of distance vector clustering and density clustering makes the algorithm more adaptive.
