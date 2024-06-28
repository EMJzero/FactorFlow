# FactorFlow

## Setup

Requires python3.9 or newer. That's it, run `main.py`.

## Summary

The idea is simple: we exploit the fact that when addressing the mapping problem for the General Matrix Multiply (GEMM) operation, the factors/tiles can be solved with a greedy policy once the dataflow, parallelism strategy, and bypasses are all set. The same applies to the parallelism strategy, which can be greedily solved once dataflows are set.

Focusing on GEMMs enables a simplification and an improvement to the evaluation speed of the analytical model underlying the map-space exploration. The mapping problem for GEMMs is still relevant due to the large majority of operations in modern Neural Networks, such as Transformers and Large Language Models, being GEMMs. Addressing convolutions, of which GEMMs are a subset, could be more general but vastly complicates the task and could lead to ignoring approaches well-suited for GEMMs.

The proposed approach reaches (as of now) better or equivalent results than other mapping tools on GEMMs, while being consistently faster.

## Notation

Throughout FactorFlow the following compact notation for triplets of nested loops on a level is used:

![compact form like: M:2 K:24 N:64 (outermost to innermost loop)](/static/compact_notation.png)

## Contributions

- **Analytical Models and Map-Space Exploration (MSE):** 
  - Existing analytical models for spatial architectures have reached a sufficient evaluation speed for MSE, but techniques for MSE itself remain slow. We show that for GEMMs, the mapping problem can be broken down into loop permutations, tiling, and parallelism strategy, and addressed separately with suitable heuristics, achieving optimal mappings much faster.
  - **Loop Permutations:** Exhaustive search with a manageable number of permutations given typical memory hierarchy levels.
  - **Factors Allocation or Tiling:** Greedy heuristic that moves prime factors across nested loops, reminiscent of the Newton Method.
  - **Parallelism Strategy:** Maximized utilization by allocating prime factors to spatial dimensions shared with workload iterations, ensuring all possible strategies are explored.

- **Comparison with Random Exploration:**
  - Techniques like random search and genetic algorithms rarely reach optimal results on GEMM workloads. Our approach stops at local minima by navigating the map-space step-by-step.

- **Brute Force Techniques:**
  - Due to the speed and focus on GEMMs, brute force techniques are used on smaller parts of the mapping problem without catastrophic slowdowns.

**Note:** FactorFlow can be augmented with randomized starting points for its heuristics to find multiple local minima through random search.

**Note:** The proposed framework will be made available as open source once the article is published.

**Note:** Fixing permutations can dramatically reduce execution time to a few milliseconds if the developer has some knowledge of the workload and architecture.

**Note:** Batches are not modeled, but can always be achived by chaining multiple inputs along the L dimension.

## Relevant Traits

- **Analytical Model for GEMMs:**
  - FactorFlow introduces an analytical model focused on GEMMs, validated against Timeloop and showing complete matching results. The model computes energy-delay product (EDP) as the main optimization metric during MSE.

- **Lightweight Model:**
  - Built from the ground up for GEMMs, FactorFlow is lightweight compared to models focusing on convolutions.

- **MoveFactor Method:**
  - The `moveFactor` method allows prime factors of iterations to move around nested loops, enabling step-by-step navigation of the map-space.

- **Simple Interface:**
  - The exploration algorithm has a simple interface, making it easy to try multiple exploration heuristics without modifying the tool.

- **Compatibility with Timeloop:**
  - FactorFlow's architecture description interface is compatible with Timeloop and can rely on Accelergy for energy consumption and latency metrics.

## Results

### Validation

- The model’s accuracy has been validated against Timeloop on the Gemmini and EYERISS architectures.
- Future validation against other architectures from literature, such as NVDLA, Simba, and Google's TPU, is planned.
- RTL level simulations for Gemmini are planned as a golden reference.

### Comparative Analysis

- **Timeloop:**
  - Better results on Gemmini and EYERISS architectures.
  - Example: $1.5\times$ better energy-delay product on Gemmini obtained in $200ms$, compared to Timeloop's $20s$.

- **Maestro+Gamma, CoSA, ZigZag:**
  - Comparisons to be conducted.
  - ZigZag's SALSA scheduler uses Simulated Annealing but starts from a similar formulation of the map-space to FF, differing in optimization steps and metrics.

- **FLASH and Maestro-BLAS:**
  - Specialized for GEMMs but still leverage random search.
  - FactorFlow avoids formulating the entire map-space, moving step-by-step and starting from a known-valid mapping.

**Note:** All above tools, built for CNNs, were used by mimicking a GEMM through a specific scheme.