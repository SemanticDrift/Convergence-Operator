# The ★ Convergence Operator

### Prime-Normalized Spectral Projection onto Consistent Representations

*Series: Mathematical Foundations for Universal Systems*

**Author:** Carolina Johnson (CJ)
**Date:** October 2026
**DOI:** [10.5281/zenodo.18232257](https://doi.org/10.5281/zenodo.18232257)
**ORCID:** [0009-0002-8819-3347](https://orcid.org/0009-0002-8819-3347)
**License:** CC BY 4.0, attribution required

---

## What this is

Quantities are stored and computed in several representations at once: different bases, units, coordinate systems, and precisions. Each one carries its own artifacts, such as truncation, conversion residue, rounding, and mis-conversion.

The ⋆ Convergence Operator takes a state and returns the nearest state that satisfies every declared consistency constraint. It also reports how much inconsistency was present and where.

Use it when several readings, estimates, or representations of one quantity must agree and you can write down how they relate.

## The operator

For constraints $\delta x = b$ (stacked from any number of sources), the operator is

```
⋆ψ = ψ − δ⁺(δψ − b)
```

where $\delta^{+}$ is the Moore-Penrose pseudoinverse. For linear constraints this is the orthogonal projection onto the consistent set, computed by one singular value decomposition at cost O(mn·min(m, n)) for m constraints on n variables. For constraints of any smooth form $g(x) = b$, a Newton-type flow carries the state to a consistent point, with exact exponential decay of the residual under stated conditions.

A prime-exponent map sends every positive exact rational, however written, to one point of an integer lattice. This turns ratios, base changes, and scale-factor unit conversions into linear constraints.

## What it guarantees

- **Nearest consistent state.** In the linear case, ⋆ψ is the unique nearest point of the consistent set to ψ.
- **Error reduction.** When the true state satisfies the declared constraints, ⋆ never increases the error. For isotropic noise, the expected squared error falls from σ²n to σ²·dim ker δ.
- **Known noise covariance.** For unequal or correlated errors with a known covariance Σ, the same law holds in the covariance-weighted norm, using the weighted operator ⋆_Σ. The covariance is an input: ⋆ does not estimate it.
- **Spectral audit.** Kernel dimension, spectral gap, settling time, per-constraint residuals, and a feasibility certificate are reported with every projection.
- **Equivalence classes.** Two states map to the same point exactly when their difference lies in the row space of the constraints.
- **Adding constraints.** A new constraint never enlarges the invariant space and never reduces the artifact content.

The guarantees hold relative to the constraints and the noise model you declare. If the true state does not satisfy the constraints, ⋆ returns the nearest state that does, and the error can increase. The paper states this in its Scope section and demonstrates it in Experiment K.

## Quick start

```bash
pip install numpy scipy sympy
python star_replication.py
```

Every experiment prints its result and asserts its claim. A successful run ends with:

```
ALL CHECKS PASSED
```

The script reproduces every number in the paper.

## Results reproduced by the script

| Experiment | What it shows | Result |
|---|---|---|
| A | Four notations of 1/2 (decimal, binary, sexagesimal, Egyptian) | One point in the prime-exponent lattice |
| B | Mis-converted reading among three unit systems | Residuals [0, 1.4142, 1.4142] localize the faulty reading |
| C | Binary float 0.1 against the intended 1/10 | Exact gap vector, relative error 5.55e-17, below 2⁻⁵³ |
| D | Twelve fifths closed against seven octaves | Each fifth 701.955 → 700.000 cents, sum 8400 |
| F | Constraint that multiplies and adds in one equation (ab + c = d) | Residual decays as e⁻ᵗ from every start |
| G | Equivalence classes with redundant rows and infeasible targets | Verified to machine precision |
| H | Flow-network reconciliation (6 nodes, 10 edges) | Mean squared error 9.972 → 4.989 (law: 5) |
| I | Hierarchical forecast coherence (8 series) | Mean squared error 8.000 → 4.994 (law: 5) |
| J | Clock-offset loop closure (8 nodes, 12 links) | Mean squared error 12.063 → 7.041 (law: 7) |
| K | False constraint | Error 0 → 0.7071, the limit of the guarantee |
| L | Flow network with unequal and with correlated noise, known covariance | Weighted-norm error 10.011 → 5.010 and 9.965 → 4.957 (law: 10 → 5) |

Experiments H, I, J, and L use simulated noise on known true states. In each of them, ⋆ was never worse than the raw observation in any of 20,000 trials. For equal-variance errors, H and I coincide with classical data reconciliation and least-squares forecast reconciliation, and the paper cites both. Experiment L supplies the true covariance, so it shows the mechanism and does not estimate Σ from data.

## Classical and specific

Orthogonal projection onto an affine set, least squares, the pseudoinverse, the heat semigroup, alternating projections, and Newton-type flows are classical. The parts specific to this construction are the prime-exponent chart, the formulation of representation consistency as a declared constraint family with one operator for linear and nonlinear relations, the spectral audit as an engineering diagnostic, and the node-addition properties. Section 11 of the paper draws this line in detail.

## Files

| File | Contents |
|---|---|
| `The_Star_Convergence_Operator.pdf` | The paper |
| `The_Star_Convergence_Operator.tex` | LaTeX source (XeLaTeX, Latin Modern) |
| `star_replication.py` | Reference implementation and all experiments |

## Citation
```
Johnson, C. (2026). The ★ Convergence Operator: Prime-Normalized Spectral Projection onto Consistent Representations. Series: Mathematical Foundations for Universal Systems. SemanticDrift.
DOI: [10.5281/zenodo.18232257](https://doi.org/10.5281/zenodo.18232257)
```

## License

© 2026 Carolina Johnson (CJ)
Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
Attribution required. https://creativecommons.org/licenses/by/4.0/

Full publication list: https://www.SemanticDrift.net
