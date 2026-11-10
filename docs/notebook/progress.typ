// CORVID research notebook. Compile: typst compile --font-path fonts progress.typ
#set document(title: "On two-level preconditioners for time-harmonic Maxwell systems", author: "CORVID notebook")
#set page(paper: "a4", margin: (x: 2.6cm, y: 2.8cm), numbering: "1",
  header: context { if counter(page).get().first() > 1 [ #set text(size: 8.5pt, style: "italic"); CORVID research notebook #h(1fr) two-level preconditioners for Maxwell ] })
#set text(font: "EB Garamond", size: 11pt, lang: "en")
#show math.equation: set text(font: "Garamond-Math")
#set par(justify: true, leading: 0.62em, first-line-indent: 1.2em, spacing: 0.75em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(1.1em); set text(size: 12.5pt, weight: "semibold"); block(it); v(0.35em) }
#show heading.where(level: 2): it => { v(0.7em); set text(size: 11pt, weight: "semibold", style: "italic"); block(it); v(0.2em) }
#set math.equation(numbering: "(1)")
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.caption: set text(size: 9.5pt)
#set table(stroke: none, inset: (x: 5pt, y: 3pt))

// ---- theorem-like environments (shared numbering) ----
#let thmcounter = counter("thm")
#let env(name, italic: true) = (title: none, body) => {
  thmcounter.step()
  block(width: 100%, above: 0.9em, below: 0.9em, {
    [*#name #context thmcounter.display().*]
    if title != none [ (#title).]
    [ ]
    if italic { emph(body) } else { body }
  })
}
#let theorem = env("Theorem")
#let lemma = env("Lemma")
#let proposition = env("Proposition")
#let corollary = env("Corollary")
#let definition = env("Definition", italic: false)
#let remark = env("Remark", italic: false)
#let conjecture = env("Conjecture")
#let proof(body) = block(width: 100%, above: 0.5em, below: 0.9em, [_Proof._ #body #h(1fr) $square$])

#let Hcurl = $bold(H)(bold("curl"))$
#let curl = $bold("curl")$
#let ND = $cal(N)$
#let R = $bb(R)$
#let C = $bb(C)$

#align(center)[
  #v(1.5em)
  #text(size: 16pt, weight: "medium")[On Two-Level Preconditioners for Time-Harmonic Maxwell Systems: \ Layer Decoupling in Perfectly Matched Layers and Coarse-Level Dispersion]
  #v(0.8em)
  #text(size: 10.5pt, style: "italic")[CORVID research notebook --- working draft, begun 25 October 2026]
  #v(1.2em)
]

#block(inset: (x: 2.2em))[
  #set text(size: 9.8pt)
  #set par(first-line-indent: 0em)
  *Abstract.* CORVID preconditions the second-order Nédélec discretisation $K = S - k^2 M + B_Gamma$ of the time-harmonic Maxwell equations by a two-level method acting on $K$ itself: Chebyshev smoothing scaled by vertex-block (Arnold--Falk--Winther) additive Schwarz around an exact solve on the lowest-order (Whitney) unknowns. It converges in 7--14 GMRES iterations on resolved problems with absorbing boundaries, but needs an exact solve of every PML unknown when a perfectly matched layer is present, and degrades below about five edges per wavelength. We identify the mechanism of both failures. For the PML we prove an exact splitting of the stretched form, $a_s = s, b_t + s^(-1) b_n$, into an $x$-local tangential form and a normal coupling form; deep in the layer the normal coupling is suppressed by $|s|^(-2)$, the layer decomposes into weakly coupled two-dimensional problems, and the near-kernel acquires an arbitrary normal profile that the Whitney space cannot represent. For under-resolution we show, in a Fourier model, that the injected coarse operator amplifies the higher-order error by $|b slash a|$, which is unbounded at the _coarse_ resonance, whereas a Schur-complement (ideal-prolongation) coarse operator amplifies it by $|b slash c|$, which stays bounded. The second observation suggests a coarse operator in the spirit of localized orthogonal decomposition. This document is a notebook: results, conjectures and rejected ideas are recorded as they arise.
]

= Problem setting

Let $Omega subset R^3$ be a bounded Lipschitz polyhedron, $k > 0$ the free-space wavenumber, and $Gamma_A subset partial Omega$ the absorbing part of the boundary. In a perfectly matched layer $Omega_"P" subset Omega$ the material tensors are replaced by $Lambda epsilon$ and $Lambda mu$ with the stretching tensor of Bérenger type @berenger1994,
$ Lambda = "diag"(s_y s_z / s_x, s_x s_z / s_y, s_x s_y / s_z), quad s_j = 1 - i sigma_j (x_j), $ <eq:Lambda>
and EMerge uses $sigma(t) = delta_max t^(1.5)$ in the normalised depth $t in [0,1]$, with $delta_max = 8$. The variational problem is: find $bold(E) in Hcurl_0 (Omega)$ such that for all $bold(v)$
$ a(bold(E), bold(v)) := integral_Omega (Lambda mu)^(-1) curl bold(E) dot curl bold(v) - k^2 integral_Omega Lambda epsilon bold(E) dot bold(v) + i k integral_(Gamma_A) bold(E)_T dot bold(v)_T = ell(bold(v)). $ <eq:weak>
The form is bilinear (no complex conjugation), so its Galerkin matrix $K$ is complex symmetric. We discretise with the hierarchical second-order Nédélec space
$ ND_2 = ND_1 plus.o W, quad dim ND_1 = hash"edges", quad W = "span"{"second edge DOF"} plus.o "span"{"two face DOFs per face"}, $ <eq:split>
where $ND_1$ is the Whitney space. Write $K = mat(K_(c c), K_(c f); K_(f c), K_(f f))$ in the splitting @eq:split ($c$: Whitney, $f$: complement $W$), and $R : C^(n_c) -> C^n$ for the injection of the Whitney coefficients.

== The CORVID 0.2.0 preconditioner

#definition(title: "CORVID 0.2.0")[Let $V_v$ be the set of unknowns on edges and faces incident to the vertex $v$, $K_v = K|_(V_v times V_v)$, and $B = sum_v R_v K_v^(-1) R_v^T$ the vertex-block additive Schwarz operator. With $omega = 4 slash (3 dot 1.1 dot lambda_max (B K))$, the preconditioner is
$ M = (I - omega B K) R K_c^(-1) R^T (I - K omega B) + omega B + omega B - omega^2 B K B, quad K_c = R^T K R, $
i.e. one damped Schwarz step, an exact coarse correction on the Whitney rows ($K_c$ factored by SPARTA's $L D L^T$), and one damped Schwarz step. Its error propagator is
$ E = (I - omega B K)(I - Pi)(I - omega B K), quad Pi = R K_c^(-1) R^T K. $ <eq:E>
"exact = auto" replaces the Whitney rows by the Whitney rows together with all PML rows.]

Since $ND_1 subset ND_2$ (hierarchical basis), $K_c$ is the Whitney discretisation itself, and $Pi$ is the $a$-oblique projection onto $ND_1$. Nothing in @eq:E uses a shift: CORVID preconditions the indefinite operator directly, which is the reason it beats the shifted-Laplacian family @erlangga2004 on resolved problems (the shifted operator, even inverted exactly, needed 40 iterations on model2 against 9--10 here).

== The pathologies, measured

All test problems are generated with EMerge (an open-ended WR-10 waveguide radiating into a box, quarter model, 90 GHz). "none" is CORVID 0.2.0 with only the Whitney rows exact, "auto" adds all PML rows (GMRES iterations to $10^(-6)$, cap 300).

#figure(caption: [CORVID 0.2.0 on the small test family. epl: edges per wavelength, median / maximum edge.],
table(columns: 6, align: (left, right, right, right, right, right),
  table.hline(),
  [problem], [$n$], [PML rows], [epl], [none], [auto (+rows)],
  table.hline(stroke: 0.5pt),
  [ABC, res 0.15], [33 662], [0 %], [5.6 / 3.4], [12], [12],
  [PML $delta = 2$], [35 194], [77 %], [6.1 / 3.2], [23], [11],
  [PML $delta = 4$], [35 194], [77 %], [6.1 / 3.2], [64], [12],
  [PML $delta = 8$ (EMerge)], [35 194], [77 %], [6.1 / 3.2], [147], [12 (+22 428)],
  [PML phase only, $|s| = 1$], [35 194], [77 %], [6.1 / 3.2], [20], [11],
  [PML magnitude only, $s in R$], [35 194], [--], [6.1 / 3.2], [300+], [300+],
  [PML $delta = 8$, 8 mesh layers], [218 204], [89 %], [10.6 / 3.4], [209], [11],
  [PML $delta = 8$, box 5 mm], [87 824], [66 %], [5.6 / 3.3], [212], [13],
  [ABC, res 0.3], [6 340], [1 %], [3.6 / 1.7], [78], [80],
  table.hline(),
)) <tab:baseline>

Two separate failures are visible: the PML (driven by $|s|$, not by the phase of $s$, and not cured by refining the layer) and the ABC problem at 3.6 edges per wavelength.

= The perfectly matched layer

== An exact splitting of the stretched form

Consider a face layer normal to $x$, so $s_y = s_z = 1$, $s := s_x$, $Lambda = "diag"(s^(-1), s, s)$, and take $epsilon = mu = 1$ for brevity. Write $bold(u) = (u_x, bold(u)_t)$, $bold(u)_t = (u_y, u_z)$, $nabla_t = (partial_y, partial_z)$ and $"curl"_t bold(u)_t = partial_y u_z - partial_z u_y$.

#lemma(title: "layer splitting")[For $bold(u), bold(v) in Hcurl(Omega_"P")$ and any measurable $s = s(x) : Omega_"P" -> C without {0}$,
$ a_s (bold(u), bold(v)) = integral_(Omega_"P") s, beta_t (bold(u), bold(v)) + s^(-1) beta_n (bold(u), bold(v)), $ <eq:splitting>
where the real symmetric densities are
$ beta_t (bold(u), bold(v)) &= "curl"_t bold(u)_t "curl"_t bold(v)_t - k^2 bold(u)_t dot bold(v)_t, \
  beta_n (bold(u), bold(v)) &= (partial_x bold(u)_t - nabla_t u_x) dot (partial_x bold(v)_t - nabla_t v_x) - k^2 u_x v_x. $
In particular $beta_t$ contains no derivative in the normal direction and vanishes on normal fields $(u_x, 0, 0)$.]

#proof[$Lambda^(-1) = "diag"(s, s^(-1), s^(-1))$, and $(curl bold(u))_x = "curl"_t bold(u)_t$, $((curl bold(u))_y, (curl bold(u))_z) = (partial_z u_x - partial_x u_z, partial_x u_y - partial_y u_x)$, which is a rotation by $pi slash 2$ of $partial_x bold(u)_t - nabla_t u_x$; rotations preserve the dot product. Hence $Lambda^(-1) curl bold(u) dot curl bold(v) = s "curl"_t bold(u)_t "curl"_t bold(v)_t + s^(-1)(partial_x bold(u)_t - nabla_t u_x) dot (partial_x bold(v)_t - nabla_t v_x)$, and $Lambda bold(u) dot bold(v) = s^(-1) u_x v_x + s bold(u)_t dot bold(v)_t$. Collect the terms with $s$ and with $s^(-1)$.]

#corollary(title: "relative size of the coupling")[Dividing by $s$, $s^(-1) a_s = b_t + s^(-2) b_n$ with $|s^(-2)| = (1 + sigma^2)^(-1)$. For EMerge's profile at depth $t$, $|s|^(-2) = (1 + 64 t^3)^(-1)$: $0.94$ at $t = 1 slash 8$, $0.26$ at $t = 3 slash 8$, $0.06$ at $t = 5 slash 8$, $0.02$ at $t = 7 slash 8$.]

The form $b_t$ is a direct integral over $x$ of two-dimensional Maxwell forms on the planes $x = "const"$. Hence, as $sigma -> infinity$, the layer becomes a stack of _uncoupled_ two-dimensional problems; the coupling between neighbouring planes enters only through $s^(-2) b_n$, the only term with a normal derivative. This is the anisotropy picture of algebraic multigrid @ruge1987 transplanted to a complex medium: the strong couplings are tangential (weight $|s|$), the weak ones normal (weight $|s|^(-1)$).

#corollary(title: "no rotation makes the layer coercive")[For every $theta in R$ and $sigma >= 0$ there is $bold(u)$ with $"Re"(e^(i theta) a_s (bold(u), overline(bold(u)))) < 0$.]

#proof[The joint numerical range $J = {(b_t (bold(u), overline(bold(u))), b_n (bold(u), overline(bold(u))))}$ of the two Hermitian forms is a convex cone (Hausdorff--Toeplitz). It contains directions arbitrarily close to $(-1, 0)$ (tangential gradients $bold(u)_t = nabla_t phi$ with slow $x$-variation), $(1, 0)$ (tangential solenoidal fields of high tangential frequency, slow $x$-variation), $(0, 1)$ (normal fields with large $nabla_t u_x$) and $(0, -1)$ (normal fields depending on $x$ only). Its closure is therefore $R^2$, and the nonzero linear functional $(beta_t, beta_n) |-> "Re"(e^(i theta) s) beta_t + "Re"(e^(i theta) s^(-1)) beta_n$ takes negative values on it.]

#remark[In one dimension the opposite holds: $-i a_s (u, overline(u)) $ has real part $sigma (1 + sigma^2)^(-1) ||u'||^2 + k^2 sigma ||u||^2 > 0$. The three-dimensional obstruction is precisely the pair $("curl"_t bold(u)_t, u_x)$, whose weights $s$ and $s^(-1)$ sit on the "wrong" side after rotation. Field-of-values bounds for GMRES (Elman) are therefore not available inside the PML by a rotation argument.]

== Consequence for the two-level method

Let $Z_tau$ be the span of the eigenvectors of $K$ (restricted to the deep layer, $|s| >= s_0$) with $|lambda| <= tau |s| h^(-2)$, i.e. the near-kernel relative to the diagonal scale. By @eq:splitting these are, to $O(|s|^(-2))$, products of a near-resonant tangential field ($|kappa_t| approx k$, or any normal field) with an *arbitrary* profile in $x$.

#conjecture(title: "PML near-kernel")[(i) $Z_tau$ is not approximated by $ND_1$: its profiles vary within an element in the normal direction, which only $W$ resolves. (ii) $Z_tau$ is not reduced by vertex-block smoothing: its tangential extent is the wavelength, not the block. Hence the two-level error propagator @eq:E has $dim Z_tau$ eigenvalues of modulus $>= 1$.]

#remark(title: "heuristic count")[On a layer of area $A$ with $N_n$ sublayers of unknowns, a Weyl count of tangential modes with $abs(|kappa_t|^2 - k^2) <= tau h^(-2)$ gives $dim Z_tau approx N_n A tau slash (2 pi h^2) approx (tau slash 2 pi) n_"PML"$: a small _fraction_ of the PML unknowns, but not a bounded number. Global deflation with dense vectors is therefore excluded; whatever replaces the exact PML solve must be local in the tangential direction or exploit structure.]

The numerical evidence (box, $delta = 8$, $n = 35 194$) agrees with every part of the conjecture. Arnoldi on $E$ returned 200 eigenvalues of modulus $> 0.98$ (all requested; the largest $2.57$), every one with $100 %$ of its weight in the PML, mean depth $t approx 0.8$, and $70$--$95 %$ of its weight on the face unknowns. The exact solve needs both parts of $W$ and only the deep layer:

#figure(caption: [Which PML rows must be in the exact level (GMRES iterations; box $delta = 8$ / box 5 mm).],
table(columns: 4, align: (left, right, right, right),
  table.hline(),
  [exact rows besides Whitney], [share of $n$], [box], [box 5 mm],
  table.hline(stroke: 0.5pt),
  [none], [0 %], [147], [212],
  [PML face unknowns only], [50 % / 42 %], [126], [223],
  [PML second edge unknowns only], [14 % / 12 %], [134], [186],
  [PML complement $W$ (= auto)], [64 % / 54 %], [12], [13],
  [PML, depth $t > 0.25$], [61 % / 50 %], [14], [14],
  [PML, depth $t > 0.5$], [45 % / 36 %], [33], [45],
  [PML, depth $t < 0.5$], [33 % / 30 %], [145], [215],
  table.hline(),
))

== Idea A: sweeping through the depth (tested, rejected on cost)

Lemma 1 says that in the deep layer neighbouring planes couple only through $s^(-2) b_n$. Block Gauss--Seidel over depth slabs (exact solves of slab blocks; the slabs are BFS layers of the vertex graph measured from the PML interface, so the construction needs only the matrix and the mesh topology) should then contract at a rate $O(|s|^(-2))$, which is the classical cure for anisotropy: block relaxation along the strong (tangential) direction.

#figure(caption: [Depth-slab sweeps (Whitney exact, vertex Schwarz elsewhere). Factor entries from SPARTA $L D L^T$.],
table(columns: 5, align: (left, right, right, right, right),
  table.hline(),
  [box $delta = 8$, $n = 35 194$], [its], [coarse], [slab factors], [total],
  table.hline(stroke: 0.5pt),
  [CORVID 0.2.0 none], [100+], [0.61 M], [--], [0.61 M],
  [CORVID 0.2.0 auto], [12], [5.33 M], [--], [5.33 M],
  [slabs of 1 layer], [26], [0.61 M], [4.36 M], [4.97 M],
  [slabs of 2 layers, stride 1], [12], [0.61 M], [8.62 M], [9.23 M],
  [slabs of 2 layers, stride 2], [16], [0.61 M], [4.65 M], [5.26 M],
  table.hline(),
))

The prediction about convergence is confirmed (a depth sweep is as good as the exact PML solve), but the cost is not lower: EMerge's layers are only 6--8 vertex layers deep, so a slab is nearly as expensive to factor as the whole shell. The saving predicted by nested-dissection counting grows like the square of the number of sublayers and only appears for thick layers. *Rejected for EMerge's PMLs; kept as the fallback for thick layers.*

= Coarse-level dispersion

== A Fourier model of the p-two-level method

Take $-u'' - k^2 u$ on a uniform periodic mesh of width $h$, discretised with hierarchical $P_2 = P_1 plus.o {"element bubbles"}$. Per Fourier mode $theta$ the operator has the $2 times 2$ symbol $hat(K)(theta) = mat(a, b; overline(b), c)$ (vertex row, bubble row); with $t = k h$ (computed symbolically, `research/math/fourier1d.py`):
$ a(theta) = (6 - 2 t^2 - (6 + t^2) cos theta) / (3 h), quad b(theta) = -(k^2 h slash 3)(1 + e^(i theta)), quad c = 8 (10 - t^2) / (15 h). $ <eq:symbols>
The coarse space is $P_1$; injection gives the Galerkin coarse symbol $a$, while elimination of the bubble gives the Schur complement $S = a - |b|^2 slash c$.

#proposition(title: "amplification by the coarse correction")[In the model @eq:symbols,
$ norm(I - hat(Pi)_"inj" (theta)) = (1 + |b slash a|^2)^(1 slash 2), quad norm(I - hat(Pi)_"S" (theta)) = (1 + |b slash c|^2)^(1 slash 2), $
where $hat(Pi)_"inj"$ is the injected Galerkin coarse correction (CORVID 0.2.0) and $hat(Pi)_"S"$ the coarse correction with the ideal prolongation $P = (1, -overline(b) slash c)^T$ and coarse symbol $P^* hat(K) P = S$. For every $k h < sqrt(12)$ the first is unbounded in $theta$, the second bounded by $(1 + |b slash c|^2)^(1 slash 2)$ uniformly in $theta$ as long as $k h < sqrt(10)$.]

#proof[With $P = (1, 0)^T$: $P a^(-1) P^T hat(K) = mat(1, b slash a; 0, 0)$, so $I - hat(Pi)_"inj" = mat(0, -b slash a; 0, 1)$, whose norm is $(1 + |b slash a|^2)^(1 slash 2)$. With $P = (1, -overline(b) slash c)^T$: $P^T hat(K) = (a - |b|^2 slash c, 0) = (S, 0)$, hence $hat(Pi)_"S" = P S^(-1) (S, 0) = mat(1, 0; -overline(b) slash c, 0)$ and $I - hat(Pi)_"S" = mat(0, 0; overline(b) slash c, 1)$. Finally $a(theta) = 0$ iff $cos theta = (6 - 2t^2) slash (6 + t^2)$, which lies in $[-1, 1]$ iff $t^2 <= 12$; there $b eq.not 0$ unless $theta = pi$. And $c > 0$ iff $t^2 < 10$.]

The zero of $a$ is the _coarse_ resonance (the $P_1$ dispersion relation); the true resonance of the fine operator is the zero of $S$. The injected coarse correction divides by the coarse symbol exactly where it is wrong:

#figure(caption: [Fourier model. $theta_1$, $theta_2$: discrete resonances of $P_1$ (zero of $a$) and of $P_2$ (zero of $S$), relative to the exact $k h$; band: fraction of modes with $|b slash a| > 1$.],
table(columns: 7, align: right,
  table.hline(),
  [$k h$], [edges / $lambda$], [$theta_1 slash k h$], [$theta_2 slash k h$], [$max |b slash a|$], [$max |b slash c|$], [band],
  table.hline(stroke: 0.5pt),
  [0.25], [25.1], [0.9974], [1.00002], [$infinity$], [0.008], [0.057],
  [0.5], [12.6], [0.9899], [0.99995], [$infinity$], [0.032], [0.111],
  [1.0], [6.3], [0.9625], [0.99935], [$infinity$], [0.139], [0.211],
  [1.5], [4.2], [0.9253], [0.99689], [$infinity$], [0.363], [0.293],
  [2.0], [3.1], [0.8861], [0.99116], [$infinity$], [0.833], [0.357],
  [2.5], [2.5], [0.8521], [0.98140], [$infinity$], [2.083], [0.404],
  table.hline(),
))

This is the mechanism behind the Helmholtz multigrid literature @elman2001 @ernstgander2012, specialised to $p$-coarsening on a fixed mesh. In CORVID the smoother removes most of the bubble error before the coarse correction, which is why the method is robust at 6 edges per wavelength; the band of amplified modes grows with $k h$, which is the under-resolution failure of @tab:baseline.

== Idea B: a Schur-complement coarse operator (tested, rejected)

The proposition suggests replacing the pair (injection, $K_(c c)$) by (approximate ideal prolongation, approximate Schur complement):
$ P = mat(I; -X), quad X = K_(f f)^(-1) K_(f c), quad S = P^T K P = K_(c c) - K_(c f) K_(f f)^(-1) K_(f c), $ <eq:ideal>
with $X$ replaced by a _local_ approximation. In the language of localized orthogonal decomposition @peterseim2017 @gallistl2018, $P$ maps the Whitney basis to a corrected basis $phi - cal(C) phi$, where $cal(C)$ is the $a$-orthogonal projection onto the fine-scale space $W = ker R^T$; Peterseim shows that correctors localised to $O(log k)$ layers remove the pollution under $k H lt.tilde 1$. The construction needs only blocks of $K$, and it explains the PML failure in the same terms: LOD requires $a$ to be stable on $W$, and Lemma 1 shows that it is not in the deep PML.

*Experiments* (ABC box at 3.6 edges per wavelength, $n = 6340$; CORVID 0.2.0 needs 78 iterations). The columns of $X$ decay exponentially, by a factor 7--10 per vertex layer (relative maxima $1, 2.4 dot 10^(-1), 1.7 dot 10^(-2), 2.5 dot 10^(-3), dots, 4 dot 10^(-8)$ at distances $0, dots, 7$), and $K_(f f)$ is comfortably invertible (smallest $|lambda|$ is $1.4 %$ of the mean diagonal). Yet:

#figure(caption: [Coarse correction $e = P K_c^(-1) Q^T r$ with independently chosen restriction $Q$, operator $K_c$ and prolongation $P$ (GMRES iterations; 100 = stalled). LOD$ell$: correctors solved on the unknowns within $ell$ vertex layers of the edge.],
table(columns: 4, align: (left, left, left, right),
  table.hline(),
  [$Q$], [$K_c$], [$P$], [its],
  table.hline(stroke: 0.5pt),
  [injection], [$K_(c c)$], [injection], [78],
  [injection], [$S$ (exact)], [injection], [69],
  [ideal], [$S$], [injection], [64],
  [LOD1], [LOD1 Galerkin], [injection], [51],
  [ideal], [$K_(c c)$], [ideal], [100],
  [ideal], [LOD1 Galerkin ($||K_c - S|| = 4.5 % ||S||$)], [ideal], [100],
  [ideal], [$S$], [ideal], [10],
  [injection], [$S$], [ideal], [8],
  table.hline(),
))

#figure(caption: [Sensitivity to the prolongation: $Q$ = injection, $K_c = tilde(P)^T K tilde(P)$, with $tilde(X)$ the exact $X$ with small entries dropped, or LOD correctors.],
table(columns: 4, align: (left, right, right, right),
  table.hline(),
  [$tilde(X)$], [$||tilde(X) - X|| slash ||X||$], [nonzeros per column], [its],
  table.hline(stroke: 0.5pt),
  [drop $< 10^(-1) max$], [$2.5 dot 10^(-1)$], [51], [56],
  [drop $< 10^(-2) max$], [$4.8 dot 10^(-2)$], [332], [60],
  [drop $< 10^(-4) max$], [$7.3 dot 10^(-4)$], [2510], [51],
  [LOD, $ell = 1$], [$3.3 dot 10^(-1)$], [116], [49],
  [LOD, $ell = 3$], [$1.0 dot 10^(-2)$], [1732], [60],
  [exact $X$], [0], [all], [8],
  table.hline(),
))

A perturbation of $0.07 %$ costs as much as dropping the correction altogether. The explanation is algebraic.

#proposition(title: "the ideal coarse correction is the inverse")[With $P$ and $S$ as in @eq:ideal, $K P = R S$. Consequently $K^(-1) R g = P S^(-1) g$ for every $g in C^(n_c)$: the coarse correction $P S^(-1) R^T r$ applies $K^(-1)$ to the Whitney part of the residual. For a perturbed prolongation $tilde(P) = P + F delta$ ($F$ the injection of the complement),
$ K tilde(P) = R S + F K_(f f) delta, $ <eq:leak>
so a coarse correction $tilde(P) tilde(S)^(-1) R^T r$ leaves the residual $-F K_(f f) delta tilde(S)^(-1) R^T r$ in the complement rows.]

#proof[$(K P)_c = K_(c c) - K_(c f) X = S$ and $(K P)_f = K_(f c) - K_(f f) X = 0$. Hence $K P S^(-1) g = R g$. For @eq:leak add $K F delta$, whose $c$-rows $K_(c f) delta$ are absorbed into the coarse block and whose $f$-rows are $K_(f f) delta$.]

#remark(title: "why localisation fails here")[The leaked residual in @eq:leak is amplified by $||tilde(S)^(-1)||$, which is the reciprocal distance of the coarse spectrum from the origin. Below five edges per wavelength $S$ has many eigenvalues close to zero, and a truncation error $delta$ with a fixed relative size, however small, leaves $\#{mu in sigma(S) : |mu| lt.tilde ||K_(f f) delta||}$ amplified modes. Decay of $X$ is not enough; what is needed is an approximation whose error is proportional to the eigenvalue of the mode it approximates, which only the exact harmonic extension provides. LOD avoids the issue because its coarse mesh satisfies $k H lt.tilde 1$ with a constant small enough for the coarse problem to be stable; a $p$-coarse space on the same mesh at $k h approx 1.7$ is not in that regime. *Rejected.*]

Enriching the exact level with parts of $W$ fails in a related way. Adding only the second edge unknowns (gradients of edge bubbles), or only the face unknowns, makes things worse (100 iterations instead of 78, on both test boxes): neither subspace completes a discrete de Rham sequence, and the Galerkin operator on an incomplete space has its own spurious near-zero modes. Coarse spaces for the indefinite operator must be complete sequences, or the whole of $W$ locally (as in the PML).

= Further experiments on the two failures

== The under-resolved regime is a bulk phenomenon

Arnoldi on the error propagator @eq:E of the ABC box at 3.6 edges per wavelength gives 5 eigenvalues with $|mu| > 1$, 23 with $|mu| > 0.8$, and more than 150 with $|mu| > 0.5$ (at 5 edges per wavelength: none above 0.78). The degradation is a broad loss of contraction, not a handful of outliers, so deflation or Krylov recycling of a few vectors across a frequency sweep can only remove part of it.

#figure(caption: [Remedies tried for the under-resolved ABC box ($n = 6340$, 78 iterations) and their effect on a resolved one (5 edges per wavelength, 13 iterations).],
table(columns: 3, align: (left, right, right),
  table.hline(),
  [variant], [3.6 epl], [5.0 epl],
  table.hline(stroke: 0.5pt),
  [CORVID 0.2.0 (order-1 Chebyshev)], [78], [13],
  [Chebyshev order 2 / 3 / 4 / 6], [67 / 70 / 76 / 80], [9 / 10 / 16 / 24],
  [coarse shift $K_(c c) + i beta |"diag" K_(c c)|$, $beta = 0.1$], [63], [28],
  [coarse shift, $beta = -0.1$], [stalls], [stalls],
  [exact on rows with $k h > 1.6$ (82 % of a direct factor)], [26], [--],
  [exact on rows with $k h > 1.3$ (94 % of a direct factor)], [12], [--],
  [prolongation constrained to 30--200 eigenvectors of $K$ (least squares)], [stalls], [--],
  table.hline(),
))

Raising the Chebyshev order helps a little and then hurts. That is expected: the fourth-kind Chebyshev polynomial is optimal for a spectrum of $B K$ on $(0, lambda_max]$, and for the indefinite $K$ part of that spectrum lies off the positive axis, where higher degrees amplify. A complex coarse shift bounds the amplification of the proposition (the denominator $a + i beta m$ no longer vanishes), but stops correcting the modes it damps; on resolved meshes this costs more than it gains. Fitting the prolongation to near-kernel vectors by plain least squares produces rows with very large entries where the local constraint matrix is ill-conditioned; a regularised energy minimisation (Olson--Schroder--Tuminaro type) would be needed. This line is _inconclusive_, not rejected.

#remark(title: "a unifying criterion that does not work")[The PML stretches the local wavenumber to $k |s|$, and magnitude-only stretching fails while phase-only stretching converges, which suggests a single criterion "solve exactly where $k |s| h > kappa$". With $|s|$ estimated algebraically from each row ($rho = sum |"Im" K_(i j)| slash sum |K_(i j)| approx sigma slash sqrt(1 + sigma^2)$) this gives 37--89 iterations on the PML box against 12 for the full PML. The PML failure is not a resolution failure alone; Lemma 1 (decoupling of the layers) is the operative mechanism.]

== A cheaper exact level for the PML

The depth result of Section 2 can be made algebraic: the depth of a vertex is its distance in the mesh graph from the non-PML vertices (PML vertices are those whose block rows are mostly PML rows), and a complement row inherits the largest depth of its vertices. Only the mesh edges are needed.

#figure(caption: [Exact level = Whitney rows + PML complement rows with graph depth $>= L$. Factor entries of SPARTA's $L D L^T$; percentages relative to a direct factorisation of $K$.],
table(columns: 5, align: (left, right, right, right, right),
  table.hline(),
  [], [auto (all)], [depth $>= 2$], [depth $>= 3$], [depth $>= 4$],
  table.hline(stroke: 0.5pt),
  [small box ($n = 18 848$)], [77 %, 11 its], [75 %, 11], [52 %, 17], [29 %, 60],
  [box ($n = 35 194$)], [73 %, 12], [72 %, 12], [54 %, 15], [35 %, 33],
  [box 5 mm ($n = 87 824$)], [55 %, 13], [50 %, 12], [36 %, 17], [22 %, 53],
  [horn, res 0.15 ($n = 142 468$)], [50 %, 26], [46 %, 21], [32 %, 29], [20 %, 103],
  table.hline(),
))

Skipping the first two vertex layers of the PML removes a quarter to a third of the exact level's factor for three to five extra iterations; on the horn the factor shrinks by 36 % (20.1 M to 12.9 M entries) and the total time drops from 3.8 s to 3.1 s. This is the first change that is cheaper than CORVID 0.2.0 at nearly the same convergence; it is a refinement of "auto", not a new method.

#figure(caption: [Shipped as `Solver(pml_depth=2)` in corvid 0.2.1 (default). EMerge's model 4 horn, 3.6 edges per wavelength, $n = 101 814$; iterations, exact-level factor entries, total time.],
table(columns: 4, align: (left, right, right, right),
  table.hline(),
  [], [0.2.0 (`pml_depth=1`)], [`pml_depth=2`], [`pml_depth=3`],
  table.hline(stroke: 0.5pt),
  [p1], [57 its, 17.9 M, 4.8 s], [41, 17.0 M, 3.7 s], [50, 11.9 M, 3.4 s],
  [p4], [77, 5.7 s], [53, 4.1 s], [63, 3.7 s],
  [p7], [134, 9.3 s], [93, 6.3 s], [109, 5.8 s],
  table.hline(),
))

Removing the outer layers from the exact level _lowers_ the iteration count by about 30 %. In the light of the coarse inf-sup remark below this is expected: the shallow PML complement is an incomplete piece of $W$, and putting it into the exact level adds spurious coarse modes.

= The anatomy of the slow modes, and a correction

The analysis of Section 3 assumed that the modes CORVID fails on are the near-resonant modes of $K$, those with small $|lambda(K)|$. On the under-resolved ABC box this assumption is *false*, and the experiments of Section 4 are better read in the light of what the slow modes actually are.

#figure(caption: [Coarse-space enrichment of the under-resolved ABC box (78 iterations). Only the slow modes of the error propagator itself help.],
table(columns: 3, align: (left, right, right),
  table.hline(),
  [added to the Whitney level (Galerkin)], [vectors], [its],
  table.hline(stroke: 0.5pt),
  [eigenvectors of $K$ nearest 0], [10 / 60], [78 / 77],
  [same, localised by a partition of unity over 4--32 aggregates], [20--640], [68--88],
  [eigenvectors of $K_(f f)$ nearest 0], [30 / 150], [78 / 75],
  [dominant eigenvectors of $E$], [20 / 60 / 120], [24 / 17 / 15],
  [bootstrap vectors $E^q R$ ($m q$ = 64 / 128 / 192 setup applications)], [16 / 16 / 32], [42 / 34 / 25],
  table.hline(),
))

For the 40 dominant eigenvectors $v$ of $E$ we measured: Whitney share of $|v|^2$ about $2 %$; $||(I - Pi) v|| approx ||v||$ (the coarse correction does not see them); the Rayleigh quotient of $omega B K$ between $-0.06$ and $+0.16$ (the smoother does not reduce them, and amplifies the first few); $|v^* K v| slash (|v|^* |K| |v|) approx 10^(-2)$; half their energy on the 20 % of unknowns with the largest $k h$ (energy-weighted $k h = 2.57$ against a median of 2.02); and 90 % of their energy in about 30 % of the vertex blocks. Restricted additive and multiplicative vertex Schwarz do not remove them either (72 and 64 iterations).

#proposition(title: "what the slow modes are")[(empirical) Below about four edges per wavelength the slow modes of CORVID are _algebraically smooth_ in the sense of algebraic multigrid --- their residual is small relative to $|K|$, so no local exact solve reduces them --- and they live in the hierarchical complement $W$ of the largest elements. They are not eigenvectors of $K$ or of $K_(f f)$ with small eigenvalue.]

The mechanism is the classical one of algebraic multigrid @ruge1987, transposed to $p$-coarsening: a coarse space must contain the algebraically smooth vectors. On a resolved element a smooth wave is almost entirely Whitney plus a small hierarchical surplus; on an element with $k h approx 2.5$ its surplus is of the same size as its Whitney part, so the Whitney space cannot hold it. This also explains why completing the exact level by subsets of $W$ failed (Section 4): what is missing is a small number of particular vectors in $W$, not whole classes of unknowns.

== Recycling the slow modes across a frequency sweep

The slow modes need not be computed: GMRES finds them while it solves (the recycling idea of GCRO-DR @parks2006, used here inside the coarse space rather than in the Krylov method). After a solve with $j$ Arnoldi steps ($K M V_j = V_(j+1) underline(H)_j$, $Z_j = M V_j$), the harmonic Ritz vectors $V_j g$ with the smallest $|theta|$ approximate eigenvectors of $K M$, and $Z_j g$ the corresponding eigenvectors of $M K$, i.e. of $E = I - M K$. Adding them to the Whitney level of the _next_ frequency costs no preconditioner applications at all.

#figure(caption: [ABC box at 3.6 edges per wavelength, 5 frequencies from 88 to 92 GHz on one mesh. Iterations of CORVID 0.2.0 and with the Whitney level enriched by recycled harmonic Ritz vectors (kept per solve / cap).],
table(columns: 6, align: right,
  table.hline(),
  [], [88 GHz], [89], [90], [91], [92],
  table.hline(stroke: 0.5pt),
  [CORVID 0.2.0], [55], [61], [65], [68], [73],
  [recycle 10 / 20], [55], [28], [24], [58], [39],
  [recycle 20 / 40], [55], [26], [22], [58], [35],
  [recycle 30 / 60], [55], [25], [19], [24], [59],
  table.hline(),
))

The gain is large when it works and erratic when the previous solve was short: a fast solve has a short Krylov space and yields poor Ritz vectors, and the simple cap discards the older ones. Keeping the accumulated space fixes this:

#figure(caption: [Accumulated recycling (15 harmonic Ritz vectors per solve, at most 60 kept). Resolved cases with the exact PML level where present.],
table(columns: 6, align: right,
  table.hline(),
  [], [88 GHz], [89], [90], [91], [92],
  table.hline(stroke: 0.5pt),
  [ABC 3.6 epl: CORVID 0.2.0], [55], [61], [65], [68], [73],
  [ABC 3.6 epl: accumulated recycling], [55], [27], [22], [20], [19],
  [ABC 5.0 epl: 0.2.0 / recycling], [13 / 13], [13 / 13], [13 / 11], [13 / 13], [14 / 12],
  [PML 8.0 epl (auto): 0.2.0 / recycling], [11 / 11], [11 / 9], [11 / 10], [11 / 10], [11 / 10],
  table.hline(),
))

Why accumulation works is a statement from analytic perturbation theory.

#proposition(title: "recycling an analytic family of slow subspaces")[Let $E(k) = I - M(k) K(k)$ on a frequency interval $I = [k_-, k_+]$, where $K(k)$ is polynomial in $k$ and $M(k)$ is built from factorisations that stay nonsingular, so that $E$ extends holomorphically to a neighbourhood of $I$ containing the Bernstein ellipse $cal(E)_rho.alt$ of $I$, $rho.alt > 1$. Let $Gamma subset C$ be a closed contour enclosing the slow part ${|mu| >= rho}$ of $sigma(E(k))$ and no other eigenvalue, for every $k in cal(E)_rho.alt$, and $Pi(k) = (2 pi i)^(-1) integral.cont_Gamma (z - E(k))^(-1) dif z$ the spectral projector. If $k_0 < dots < k_N$ are Chebyshev points of $I$, then for every $k in I$
$ sup_(||v|| = 1) "dist"(Pi(k) v, thin "span" union.big_(j=0)^N "range" Pi(k_j)) <= (2 C_rho.alt (1 + Lambda_N)) / (rho.alt - 1) thin rho.alt^(-N), $
where $Lambda_N lt.tilde log N$ is the Lebesgue constant and $C_rho.alt = sup_(cal(E)_rho.alt) ||Pi||$.]

#proof[By Kato's theory @kato1995, $Pi$ is holomorphic wherever $Gamma$ stays in the resolvent set, hence on $cal(E)_rho.alt$. Let $p_N (k) = sum_j ell_j (k) Pi(k_j)$ be its Lagrange interpolant at the Chebyshev points. For holomorphic functions on $cal(E)_rho.alt$ the interpolation error satisfies $||Pi(k) - p_N (k)|| <= 2 C_rho.alt (1 + Lambda_N) rho.alt^(-N) slash (rho.alt - 1)$ (the operator-valued version of the classical estimate, applied entrywise in any basis). For $||v|| = 1$, $p_N (k) v in "span" union "range" Pi(k_j)$, so the distance of $Pi(k) v$ from that span is at most $||(Pi(k) - p_N (k)) v||$.]

#remark[In words: the slow subspaces along a sweep form a holomorphic family, and holomorphic families have exponentially small Kolmogorov widths. A recycled space assembled from a few earlier frequencies therefore contains the slow subspace of the current frequency to high accuracy, whereas the subspace of the single previous frequency is off by $O(|k - k_"prev"| slash "sep")$ (Davis--Kahan). The hypothesis fails when a slow eigenvalue crosses $Gamma$ inside the window; then the accumulated space has to grow by the crossing mode, which the next solve supplies. The harmonic Ritz vectors are only approximations of $"range" Pi(k_j)$; the experiment shows that this is enough.]

= Ideas from other fields

== The horn sweep

On EMerge's horn (model 4, $n = 101 814$, PML with the exact level, 3.6 edges per wavelength, frequency steps of $2.2 %$) accumulated recycling (15 per solve, at most 60) gives 49--69 iterations against 61--89 for CORVID 0.2.0 on frequencies 2--6. That is 20--40 %, far less than on the small box. The horn is 16 times larger, so by the Weyl-type count it has proportionally more slow modes than 60 vectors can hold, and its frequency step is twice as large.

== Twisting by characters (group theory) and Fermi-surface patches (renormalisation)

Two analogies point to the same construction. For a translation-invariant operator the near-kernel of the Laplacian sits at the trivial character, which survives restriction to a sublattice; the near-kernel of $K$ sits on the sphere $|theta| = k h$ of the dual torus, which does not. Multiplying by a character $e^(i k bold(d) dot bold(x))$ translates a point of that sphere to the identity, after which ordinary coarsening applies to the envelope (wave-ray multigrid, Brandt--Livshits; plane-wave coarse spaces of FETI-H). In the language of the renormalisation group, $-k^2$ is a relevant perturbation whose low-energy modes form a Fermi surface, and one coarse-grains toward the surface in angular patches (Shankar, Polchinski).

We built it: coarse functions $chi_a (bold(x)) bold(p) e^(i k bold(d) dot bold(x))$ with a smooth partition of unity over aggregates of $lambda slash 2$ or $lambda$, 6--20 directions on a Fibonacci sphere and two polarisations, interpolated into the second-order Nédélec space by least squares on EMerge's own point evaluator (a sparse matrix $Phi$ from coefficients to point values, obtained by probing the evaluator with the unit vectors of a colouring of the "shares a tetrahedron" graph; 26 colours). Result on the ABC box at 3.6 edges per wavelength: 69--86 iterations against 78. *Rejected*, for two reasons we measured:

+ A plane wave fitted with all unknowns has a point misfit of $4 %$ at wavenumber $k$ ($0.4 %$ at $k slash 3$); with the boundary conditions imposed (solved unknowns only) $20 %$, because in a box of 1.5 wavelengths most of the volume is near a wall, where true modes are incident--reflected pairs.
+ The interior residual $|K u| slash (|K| |u|)$ of a fitted plane wave is $3 dot 10^(-2)$ and does not depend on the wavenumber of the wave (flat from $0.85 k$ to $1.1 k$). The slow modes of CORVID have $10^(-2)$. At this resolution the discrete Fermi surface is blurred to the width of the discretisation error: the coarse functions are no smoother than the errors they should remove.

#remark(title: "when twisting can work")[Wave-ray and Fermi-surface renormalisation need a fine level that resolves the waves and a coarse level that does not. In $p$-coarsening on one mesh the regime "fine accurate, coarse inaccurate" is exactly where CORVID already converges (5 or more edges per wavelength). The construction is therefore a candidate for a future $h$-coarse level (CORVID's agglomerated hierarchy), not for the Whitney level.]

== Tangential twisting inside the PML

The experts' common suggestion was to apply the twist only where scale separation exists, and Lemma 1 says that this is the tangential direction in the PML. Coarse functions $chi_a (bold(x)_t) h_j (t) bold(p) e^(i k bold(d) dot bold(x)_t)$ per PML face (tangential cells of $lambda slash 2$ to $lambda$, 6--8 tangential directions, polarisations normal, tangential-perpendicular and tangential-parallel, 4--5 hat functions in depth), fitted into the second-order space through $Phi$ and added to the Whitney level. Small PML box ($n = 18 848$): Whitney only 102 iterations, exact PML level 11; the twisted space with 216 / 864 / 1440 / 3240 vectors gives 110 / 113 / 109 / 92. *Rejected.*

This is again Lemma 1: the near-kernel in the deep layer has an _arbitrary_ profile in the normal direction, down to the scale of the individual hierarchical unknowns (sublayers inside an element). Smooth profiles in depth cannot represent it, and refining them to that scale reproduces the whole complement, i.e. the exact solve. Twisting can at best compress the tangential direction, and on PMLs a few wavelengths across there is little to compress.

== Homological perturbation (category theory)

The ideal prolongation $P = mat(I; -K_(f f)^(-1) K_(f c))$ and the Schur complement are the corrected inclusion and the transferred differential of the homological perturbation lemma applied to the retract $"Whitney" arrow.r.l "N"_2$; the Neumann series of $K_(f f)^(-1)$ is the lemma's geometric series $sum (delta h)^n$. The lemma converges when $||delta h|| < 1$, which fails precisely where $K$ is unstable on the complement $W$ (deep PML, large elements). This is a reformulation of the failure of localised correctors, not a new remedy; it does say that the remedy must change the homotopy $h$, not truncate the series.

== Local spectral coarse spaces (local Hodge theory, Mayer--Vietoris gluing)

The topological reading of the twisting failure is that charts were chosen by hand; the robust remedy is to let the operator produce its local low-energy sections and glue them with a partition of unity (GenEO @spillane2014, DtN coarse spaces; for indefinite problems Bootland--Dolean--Graham--Ma--Scheichl). We implemented the algebraic version: vertex aggregates of 8--32 vertices, local domains with a one-ring overlap (1 100--2 100 unknowns at this resolution), the smallest $|lambda|$ of $D^(-1 slash 2) K_a D^(-1 slash 2)$ with $D = |"diag" K|$, a smooth vertex partition of unity, and pivoted QR to remove near-dependent vectors (the analogue of lattice basis reduction). Result on the ABC box: 81--107 iterations against 80, with 45--60 s of setup. *Rejected in this form.*

The reason is the one GenEO's theory insists on: the local eigenproblem must use the _Neumann_ (unassembled) local matrix $sum_(T subset Omega_a) K_T$. The assembled restriction $K_a$ is a Dirichlet matrix; it stiffens the aggregate boundary artificially, so its low modes are interior bubbles, while the modes the smoother misses cross aggregate boundaries. Neumann matrices require element matrices from EMerge, which is an interface change; it is the natural next step for this line.

*With Neumann matrices.* EMerge's element routine (`curlcurl._matrix_builder`) can be called directly; we exported the 20 x 20 element stiffness and mass matrices of the ABC box and verified that their sum reproduces $K$ exactly except on the absorbing-boundary and port rows (the surface terms, which we add back restricted to each aggregate). The local problem $K^N_a v = lambda D_a A^+_a D_a v$ with $A^+ = E + k^2 B$ (the H-GenEO choice) and the multiplicity partition of unity gives, for aggregates of 2--8 vertices and 2--8 vectors each ($5$--$10 %$ of $n$), stalls at a residual of $0.1$--$0.3$ or 136 iterations, against 78. *Rejected.*

#remark(title: "coarse inf-sup")[This is the third time that enlarging the coarse space made things worse (subsets of $W$, least-squares prolongations, local spectral vectors), and the mechanism is the same each time. For a coercive form any enlargement of a Galerkin coarse space is safe: the coarse operator inherits coercivity. For the indefinite $K$ the coarse operator $P^T K P$ of an enlarged space can have eigenvalues arbitrarily close to zero, i.e. a discrete inf-sup constant $gamma_c$ that is not controlled by the fine one, and the coarse correction is then amplified by $gamma_c^(-1)$. Local near-resonant vectors (Neumann eigenvectors with small $|lambda|$) glued by a partition of unity produce exactly such coarse resonances. H-GenEO's theory avoids this by assuming absorption (a complex shift that makes the form coercive); EMerge's problems at real $k$ have none in the air region. The global slow modes of $E$ and the recycled Ritz vectors are safe because they are (approximate) invariant subspaces of the iteration, which cannot introduce new coarse resonances. Any further coarse enrichment must come with a control of $gamma_c$.]

== Lessons from analogies that did not transfer

/ Sign problem (probability): for positive operators $K^(-1)$ is an expectation over random walks (Feynman--Kac); wave operators give complex weights whose sum cancels. Methods built on averaging (smoothers, aggregation) inherit the cancellation. Explains, does not fix.
/ Complex frequency (Wick rotation): preconditioning with $K(k(1 + i epsilon))$ is the shifted Laplacian; continuing solutions back to real $k$ is exponentially ill-conditioned except in rational form (model-order reduction).
/ Ray dynamics (dynamical systems): the resolvent of $K$ is $O(k)$ for non-trapping geometries and can grow very fast for trapping ones (cavities, filters). Prediction to test: CORVID's iteration counts follow the trapping of the device, and recycling matters most near high-$Q$ resonances.

= Ideas from antenna design

The problem was restated for an antenna designer: the error is a ghost field in the device; each iteration lets a carpet of electrically small absorbers (the vertex blocks) and one coarse, exact but low-order model (the Whitney level) act on it; the iteration count is the ring-down time of the slowest error resonances. Three of the designer's ideas were tested.

== Wire grids (line blocks)

"Span thin wires that short out the unwanted modes." A wire is thin (cheap) but long (it couples to waves along it, unlike a block much smaller than a wavelength). Blocks = all unknowns around a pencil of vertices of cross-section 2--3 mesh lengths running through the domain, factored with SPARTA.

#figure(caption: [Wire blocks. Additive with CORVID's damping is stable; multiplicative sweeps (wires or vertex blocks fired one after another) diverge on the horn even with damping.],
table(columns: 3, align: (left, right, right),
  table.hline(),
  [], [box, 3.6 epl], [horn p1],
  table.hline(stroke: 0.5pt),
  [CORVID (additive vertex blocks)], [78], [41],
  [additive wires, 3 cells, along x], [24], [48],
  [same, Chebyshev order 2], [17], [41],
  [wires along y / z (horn)], [--], [55 / 65],
  [pre-smooth x-wires, post-smooth y-wires ("polarisation twist")], [19], [46],
  [rotating arrangements per iteration (flexible GMRES)], [24--31], [62--85],
  table.hline(),
))

A factor 4.5 on the box, nothing on the horn. The multiplicative versions are the instructive failure: for an indefinite operator nothing guarantees that a sequence of exact local corrections contracts the error, and on the horn it does not.

== Where the horn's slow modes live

Arnoldi on the error propagator of 0.2.1 on the horn: the slowest modes ($|mu| = 0.97$--$0.98$) carry 60--96 % of their energy on unknowns touching a PEC vertex, 80--97 % inside the flare (7 % of the unknowns), 98 % on the hierarchical complement, and none in the PML or at the port. They are surface-wave-like patterns along the metal walls, where the vertex blocks are truncated by the boundary.

== Absorbers where the modes live

"Couple the modes into something that can absorb them." The exact level is a perfect absorber; we place it on the walls: every complement unknown touching a metal vertex whose longest edge has $k_0 h > 1$. Granularity matters again: a per-unknown threshold produced ragged sets and turned 21 iterations into 66 on the finer horn (an incomplete piece of $W$, the coarse inf-sup failure), while whole vertex neighbourhoods are robust.

#figure(caption: [Shipped as corvid 0.3.0: `Topology.nedelec2(..., nodes=)`, `factorize(K, k0=)`; without them the solver is 0.2.1.],
table(columns: 3, align: (left, right, right),
  table.hline(),
  [], [0.2.1], [0.3.0 (walls)],
  table.hline(stroke: 0.5pt),
  [horn p1 / p4 / p7], [41 / 53 / 93 its, 3.6 / 4.0 / 6.2 s], [24 / 29 / 32 its, 3.0 / 2.9 / 2.9 s],
  [horn, res 0.15], [21], [18],
  [ABC box 3.6 epl / 5.0 epl], [78 / 13], [60 / 11],
  [PML boxes], [12 / 11], [11 / 9],
  [model1 / model2 / model3], [12 / 12 / 14], [6 / 11 / 14],
  table.hline(),
))

The exact factor grows by about 10 % on the horn; finely meshed metal (model 2, 3) selects nothing.

== Memory: dropping redundant vertex blocks (corvid 0.3.1)

On a large horn 0.3.0 needed more memory than a direct factorisation. On model 4 the exact level (Whitney, deep PML, walls) takes 306 MB and the dense vertex-block inverses 203 MB, against 446 MB for a direct $L D L^T$ (114 %). Vertex blocks whose unknowns all lie in the exact level are redundant with it. Dropping them (961 of 3876 blocks kept) gives 378 MB (85 %) and _fewer_ iterations, because the block overlap falls and with it $lambda_max (B K)$, so the smoothing step on the remaining rows grows:

#figure(caption: [corvid 0.3.1. Memory relative to a direct $L D L^T$ of $K$ (SPARTA), iterations, time.],
table(columns: 3, align: (left, right, right),
  table.hline(),
  [], [0.3.0], [0.3.1],
  table.hline(stroke: 0.5pt),
  [horn p1 / p4 / p7], [114 %, 24 / 29 / 32 its, 3.0 s], [85 %, 18 / 24 / 29 its, 2.1--2.5 s],
  [horn res 0.15], [101 %, 18 its], [79 %, 13 its],
  [PML boxes], [11--12 its], [8--10 its],
  [model 2 / 3], [11 / 14], [12 / 14],
  table.hline(),
))

With `pml_depth=3` the horn needs 73 % of a direct factorisation (25--39 iterations); storing the remaining blocks as sparse SPARTA factors instead of dense inverses would bring it to about 60 %.

== A softer PML and sparse vertex blocks (corvid 0.4.0)

The literature on PMLs for iterative solvers (Shin and Fan @shinfan2012) makes the point that the absorbing layer should be designed for the solver. EMerge's profile $s = 1 - i delta_max t^(1.5)$ with $delta_max = 8$ absorbs a normally incident wave to $exp(-2 k d delta_max slash 2.5) approx 10^(-11)$ ($-210$ dB) on a 2 mm layer at 90 GHz, far below the discretisation error, and its steepness is what creates the near-kernel of Lemma 1. With $delta_max = 2$ ($-52$ dB) the field in the air region changes by $1.1 %$ relative to $delta_max = 8$ on the same mesh, and CORVID converges without any exact PML solve (box: 147 to 23 iterations without it). `exact="auto"` now estimates $|s|$ per row from the matrix ($rho = sum |"Im" K_(i j)| slash sum |K_(i j)| approx sigma slash |s|$, so $|s| approx (1 - rho^2)^(-1 slash 2)$); its 99th percentile is 1.8 / 3.4 / 7.0 for $delta_max$ = 2 / 4 / 8, and below 2.5 the PML is left to the vertex blocks.

With the exact PML gone, the vertex blocks are the whole memory bill. They are now stored as sparse $L D L^T$ factors after a minimum-degree ordering of each block (bitset implementation, unpivoted, with a fallback to the dense inverse on a tiny pivot that no block of any dataset needed): the same preconditioner to $10^(-13)$, 55--60 % less block memory, faster setup, the same apply time.

#figure(caption: [Total memory (exact level + vertex blocks) relative to a direct $L D L^T$ (SPARTA).],
table(columns: 4, align: (left, right, right, right),
  table.hline(),
  [], [0.3.1], [0.4.0], [its],
  table.hline(stroke: 0.5pt),
  [horn, $delta_max = 2$], [67 %], [42 %], [29],
  [horn, $delta_max = 2$, res 0.15], [66 %], [42 %], [15],
  [horn, $delta_max = 8$], [85 %], [76 %], [18],
  [model 2 / model 3], [65 % / 86 %], [31 % / 44 %], [12 / 14],
  table.hline(),
))

== Memory scaling (corvid 0.4.0, soft PML)

Air box with a waveguide feed and a $delta_max = 2$ PML, grown at fixed resolution (about 6 edges per wavelength):

#figure(caption: [Memory of corvid 0.4.0 against a direct $L D L^T$ (SPARTA). Fitted exponents over the four sizes: direct $n^(1.42)$, exact level $n^(1.24)$, vertex blocks $n^(1.05)$, total $n^(1.16)$.],
table(columns: 6, align: right,
  table.hline(),
  [$n$], [direct], [corvid], [ratio], [its], [corvid time],
  table.hline(stroke: 0.5pt),
  [62 302], [260 MB], [108 MB], [42 %], [14], [0.7 s],
  [151 244], [929 MB], [300 MB], [32 %], [14], [1.3 s],
  [400 656], [3.7 GB], [0.93 GB], [25 %], [16], [5.7 s],
  [1 041 474], [14.1 GB], [2.86 GB], [20 %], [15], [15.8 s],
  table.hline(),
))

The iteration count does not grow with the electrical size. Under mesh refinement of the 3 mm box (58k to 207k unknowns) the ratio falls from 41 % to 22 % and the iterations grow from 15 to 23. The vertex blocks are linear in $n$; the exact level (Whitney rows and metal walls) is a sparse factorisation with a smaller exponent and a much smaller constant than the direct one, and is the term that dominates at very large $n$.

== Artificial loss on the walls (rejected)

"Make the conductors lossy" in the preconditioner only: $K_"pc" = K + i beta |"diag" K|$ on the rows around metal vertices, the two-level method built from $K_"pc"$, GMRES on the true $K$. For $beta = 0.1$--$1$ every case stalls or degrades (horn p7 stalls, model 1 from 12 to 52--121); for $beta = 10^(-3)$--$10^(-2)$ it is neutral to harmful (horn p7 102--142 against 93). The trapped wall modes have a relative residual of about $10^(-2)$: a lossy coating perturbs the preconditioner's operator by more than that exactly where it must be accurate, so the modes are mis-solved rather than damped. This is the same principle as the failure of the soft-PML and shifted preconditioners: the preconditioner has to resolve the near-kernel of $K$ itself; it may only add _exact_ information there, never modify the operator. Together with 0.2.1 the horn at 3.6 edges per wavelength went from 57/77/134 iterations (0.2.0) to 24/29/32.

= Progress log

#set par(first-line-indent: 0em)
#let entry(d, body) = block(above: 0.5em, below: 0.5em)[*#d.* #body]
#entry[25 Oct 2026, 1][Set-up. Small EMerge generator `research/gen/gen_box.py` (waveguide into a box; PML or ABC; 2--10 s per dataset). Baseline @tab:baseline.]
#entry[25 Oct 2026, 2][Arnoldi on $E$: the PML failure is hundreds of amplified modes ($|mu|$ up to 2.6), deep ($t approx 0.8$), on $W$ (mostly face unknowns). The exact solve is needed for the whole complement $W$, but only for $t > 0.25$.]
#entry[25 Oct 2026, 3][Lemma 1 (layer splitting) and its corollaries; heuristic count $dim Z_tau approx (tau slash 2 pi) n_"PML"$ excludes global deflation.]
#entry[25 Oct 2026, 4][Idea A (depth-slab block Gauss--Seidel): convergence as predicted (12 its), no saving in factor storage for 6--8-layer PMLs. Rejected for EMerge PMLs.]
#entry[25 Oct 2026, 5][Research kernels in Rust (`src/research.rs`: SPARTA $L D L^T$ in double precision, block Schwarz, block sweeps): each experiment now takes seconds.]
#entry[25 Oct 2026, 6][Fourier model and the amplification proposition: the injected coarse correction is unbounded at the coarse resonance; the Schur coarse correction is bounded for $k h < sqrt(10)$.]
#entry[25 Oct 2026, 7][Idea B in 3D (ABC box, 3.6 edges per wavelength, 78 its). The exact ideal pair reaches 8--10 its, but it is algebraically $K^(-1)$ (Proposition on the ideal coarse correction). Every local approximation of $X$ (LOD $ell = 1, 2, 3$, thresholded $X$ down to $0.07 %$ error) gives 47--61 its; mixed pairs are worse than the baseline. Partial $p$-enrichment of the exact level (second edge unknowns only, faces only) stalls. Conclusion: in the $p$-two-level framework the under-resolved regime cannot be fixed by a better local coarse operator. Rejected.]
#entry[25 Oct 2026, 8][Under-resolution is a bulk loss of contraction (150+ modes with $|mu| > 0.5$). Chebyshev order, coarse complex shift, a $k|s|h$ exactness criterion and least-squares-constrained prolongations: none fixes it (table in Section 5). The only cure measured is an exact level of 80--95 % of a direct factor.]
#entry[25 Oct 2026, 9][PML exact level restricted to graph depth $>= 3$: 25--36 % smaller factor for 3--5 more iterations on three PML boxes; on the horn 36 % smaller and faster overall, and depth $>= 2$ is better than auto in both iterations and memory.]
#entry[25 Oct 2026, 10][Correction: the slow modes of the under-resolved case are not near-resonant modes of $K$ (60 exact eigenvectors of $K$ in the coarse space change nothing). They are algebraically smooth vectors in the complement $W$ of the largest elements. Enriching the Whitney level with 20 of them: 78 to 24 iterations.]
#entry[25 Oct 2026, 11][Recycling harmonic Ritz vectors across a 5-point sweep: 61--73 to 19--28 iterations on most frequencies, erratic on others.]
#entry[25 Oct 2026, 12][Accumulating the recycled space (15 per solve, cap 60): 61--73 to 19--27 on every later frequency; harmless on resolved ABC and PML sweeps. Proposition on analytic families of slow subspaces (Kato + Chebyshev interpolation) explains why accumulation is needed.]
#entry[25 Oct 2026, 13][Horn sweep with recycling: 61--89 to 49--69 iterations (20--40 %). The first attempt built the enriched coarse matrix densely and exhausted memory; replaced by a bordered solve.]
#entry[25 Oct 2026, 14][Character-twisted (wave-ray) coarse space from group theory and Fermi-surface renormalisation, built with EMerge's own evaluator: no gain (69--86 vs 78). Measured reason: at 3.6 edges per wavelength discrete plane waves have the same residual as the slow modes; the twisting needs a fine level that resolves the waves.]
#entry[25 Oct 2026, 15][Local spectral coarse space with Dirichlet local problems (assembled $K$ only): no gain, slow setup. GenEO needs Neumann matrices, i.e. element matrices from EMerge.]
#entry[25 Oct 2026, 16][Element matrices exported from EMerge and verified. GenEO with true Neumann local problems: stalls. Diagnosis: enlarged indefinite Galerkin coarse spaces have uncontrolled coarse inf-sup constants; only invariant-subspace enrichments (slow modes, recycling) are safe.]
#entry[25 Oct 2026, 17][corvid 0.2.1: `Solver(pml_depth=2)` (default). Model 4 horn p1/p4/p7: 57/77/134 to 41/53/93 iterations, 23--32 % faster, smaller exact factor; non-PML models identical; package tests pass.]
#entry[25 Oct 2026, 18][Tangentially twisted PML coarse space (216--3240 vectors): 92--113 iterations against 102 without and 11 with the exact level. The normal profile of the PML near-kernel is arbitrary at sub-element scale (Lemma 1), so no smooth construction across the layer can replace the exact solve.]
#entry[25 Oct 2026, 19][Experts' list: compatible relaxation 0.90 (additive) vs 0.54 (multiplicative) per sweep on the resolved box: the smoother limits there. Multiplicative sweeps best at equal work on boxes; ORAS rejected (vertex stars are too small for transmission conditions); Krylov smoothers not better at equal work.]
#entry[26 Oct 2026, 1][Antenna designer's wires: 78 to 17 on the box, no gain on the horn; multiplicative versions diverge on the horn. Horn slow modes located along the metal walls.]
#entry[26 Oct 2026, 2][corvid 0.3.0: exact rows on coarse-meshed metal walls. Horn p1/p4/p7 41/53/93 to 24/29/32 iterations, up to 2.1x faster; never worse on any dataset; package tests pass.]
#entry[26 Oct 2026, 3][Artificial loss on metal walls in the preconditioner: stalls for $beta >= 0.1$, harmful for $beta = 10^(-3)$--$10^(-2)$. Rejected: near-kernel modes must be solved, not damped.]
#entry[26 Oct 2026, 4][corvid 0.3.1: vertex blocks inside the exact level dropped. Horn memory 114 % to 85 % of a direct factorisation, iterations 24/29/32 to 18/24/29; COCG works (20/32); tests pass.]
#entry[26 Oct 2026, 5][corvid 0.4.0: soft-PML switch (exact PML only if the estimated $|s|$ is large) and sparse $L D L^T$ vertex blocks. Horn with $delta_max = 2$: 42 % of a direct factorisation's memory; model 2: 31 %.]
#entry[26 Oct 2026, 6][Memory scaling to $10^6$ unknowns: corvid $n^(1.16)$ against direct $n^(1.42)$; 20 % of the direct memory at 1.04 M unknowns (2.9 GB against 14.1 GB), 15 iterations.]
#entry[26 Oct 2026, 7][Bug report from the EMerge integration: the exact-row count climbs along a sweep. It is the metal-wall rule ($k_0 h > 1$ selects more vertices at higher frequency), not the PML detection; the rule helps at every frequency. Hardened anyway (corvid 0.4.1): imaginary round-off below $10^(-8)$ relative ignored (seen down to $10^(-27)$), fewer than 2 % complex-dominated rows count as no PML (ABC boundary rows reached an estimated $|s|$ of 2.5), separate PML and wall counters.]
#entry[26 Oct 2026, 8][Solve time climbing along a sweep (user: 0.9 to 4.3 s). Horn with $delta_max = 2$, one mesh, 90--110 GHz: 5.1 to 4.2 edges per wavelength, iterations 23 to 82 (without the wall rule 59 to 500), time 1.5 to 2.7 s; a direct solve takes 0.8 s at this size ($10^5$ unknowns). The climb is the under-resolution pathology. Chebyshev order 2 makes it worse (123), a lower wall threshold changes nothing at the top of the band, recycling across the sweep gives 68 to 47. Practical answer: mesh for the highest frequency of the sweep (at least 5 edges per wavelength everywhere), and use the direct solver below a few $10^5$ unknowns.]

#bibliography("refs.bib", title: "References", style: "ieee")
