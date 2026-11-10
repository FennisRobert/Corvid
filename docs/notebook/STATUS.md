# CORVID research status (26 Oct 2026)

Notebook: progress.pdf (also https://claude.ai/artifact/DPeDptoDhxoQNqofXfTh68)

## Shipped
- corvid 0.4.0: sparse LDL^T vertex blocks (-55..60 % block memory) + soft-PML switch.
  Memory vs direct: horn with deltamax 2: 42 %; deltamax 8: 76 %; model2 31 %; model3 44 %.
- corvid 0.3.1: vertex blocks inside the exact level dropped. Horn memory 114% -> 85% of direct,
  iterations 24/29/32 -> 18/24/29. Next: sparse block factors (~60% of direct).
- corvid 0.3.0: exact rows on coarse-meshed metal walls (needs nodes + k0, optional).
  Horn p1/p4/p7: 41/53/93 -> 24/29/32 its (0.2.0: 57/77/134), up to 2.1x faster. Never worse.
- corvid 0.2.1: `Solver(pml_depth=2)` default. Exact PML level skips the first two vertex layers.
  Horn (model4) p1/p4/p7: 57/77/134 -> 41/53/93 its, 23-32% faster, smaller factor. Non-PML unchanged.

## Proven / established
- PML layer-splitting lemma a_s = s b_t + s^-1 b_n (deep layers decouple).
- Fourier model: injected coarse correction unbounded at the coarse resonance.
- Ideal coarse correction = K^-1 on the Whitney residual; truncations leak (why LOD-type coarse fails).
- Slow modes in the under-resolved regime are algebraically smooth vectors in the higher-order part of
  large elements (not near-resonant modes of K).
- Recycling: slow subspaces along a sweep form an analytic family (Kato) -> accumulate recycled vectors.
- Coarse inf-sup: enlarging an indefinite Galerkin coarse space can create coarse resonances; only
  invariant-subspace enrichments are safe.

## Measured but not shipped
- Recycling across a sweep: small box 61-73 -> 19-27 its; horn 61-89 -> 49-69 its (cap 60).

## Rejected (with reasons in the notebook)
LOD/Schur coarse, Chebyshev order, coarse complex shift, partial p-enrichment, k|s|h criterion,
constrained prolongation (LS), wave-ray/plane-wave coarse, local spectral (Dirichlet and Neumann/GenEO).

## Next
- Recycling in the package (needs state across frequencies; API decision for the user).
- Coarse enrichment with an explicit coarse inf-sup control.
