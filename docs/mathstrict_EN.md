# An SCF Model for a Rotating Granular Body Based on a Cauchy Density, RTFP Gravity, and Semi-Lagrangian Transport

## Abstract

This paper gives a unified mathematical and computational framework for the granular body of a rotating small body [4][5]. The internal mass distribution is a spatially varying Cauchy density. A Feynman parametrization gives one integral representation of the potential, its gradient, and its Hessian. Complex free boundaries are handled by ray-traced finite-part (RTFP) geometry. These are coupled inside one self-consistent-field (SCF) iteration:

- a $\mu(I)$-type rheology;
- semi-Lagrangian transport on the surface;
- the geometric projection of frictional constraints.

Exterior classical derivatives, interior finite parts, and surface contact terms are kept distinct. Sufficient conditions are given under which the combined rheology reduces to the classical $\mu(I)$ law. Error and range of validity are stated for the geometry-adapted volume discretization and hyperbolic ray relaxation, for GPU ray intersection, and for the sequential convex QP/SOCP.

The reference model is the quasi-static viscoplastic body of Mode B, with the single stress field

$$
\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D,
$$

a $\mu(I)$ viscoplastic law regularized in the Bingham or Papanastasiou sense. A geological step proceeds as follows:

- **Creep solve.** The quasi-static Stokes creep equilibrium is solved implicitly at the prescribed spin.
- **Yield classification.** Yield is decided on that stress by the yield function $F$. Material connected to the unyielded rigid core gets no creep displacement. Only the yielded region, and sliding blocks separated from the core by a yielded band, are displaced by the creep velocity relative to the core.
- **Slope promotion (A-C4, Section 2.8c).** The loose surface layer also obeys an angle-of-repose proxy: the vertices of a face steeper than the proxy join the free set of the geometric subproblem even when the creep classification locks them. The hyperbolic ray solve resolves only wavelengths above its truncation and decides deep-seated and whole-body failure. Slope promotion is the subgrid closure of shallow failure.
- **Geometric correction.** An interior-point cone solver corrects only these increments, with the current shape as reference, never toward an equipotential.
- **Density transport.** Every Cauchy kernel is a passively transported material component. Its centre and metric are carried along the same masked slip velocity by exact 2-jet transport, then the set is closed in mass and, when a misfit indicator asks for it, refitted (Section 9.2). A dense core that is squeezed or sheared away takes its gravity with it.

By default the spin $\omega(t)$ is prescribed exogenously and is not coupled to the flow. The fluid mass level and equipotential of Mode A are kept only as a marked interface; they are not part of the granular loop.

Chapter 11 adds four named hypotheses, each with a switch whose off position recovers the model above exactly (Theorem 11.13):

- **A-Ξ:** rate-type dilatancy, with a material solid fraction that changes the friction and the density;
- **A-ϖ:** damage of the cohesion by plastic strain;
- **A-Θ:** a subgrid closure for the diurnal thermal fatigue of the surface skin, which weakens the shallow cohesion and produces loose material;
- **A-L:** a spin that follows the angular momentum and the evolving inertia.

The state variables of A-Ξ and A-ϖ are carried by Lagrangian markers and driven by a nonlocal plastic rate.

Chapter 12 follows a satellite past its failure (**A-B**): it breaks into the rigid blocks of its collapse mechanism, the fragments move about the primary and merge or break again, and the debris spreads as a viscous ring. Section 10.3.6 adds **A-O**, for a synchronous satellite: the tidal decay of the orbit follows the satellite's mass, which leaves through the necks of its Hill region. Section 10.3.7 adds **A-Λ**: the eccentricity, the forced libration, and the elastic cycle they impose, with its shakedown limit, ratchet, and fatigue. Section 2.17 gives a certified bracket of the limit load, and from it the orbit at which a frozen body can no longer stand.

Rigour here means only that formulae and acceptance conditions have a determinate meaning under the stated continuity, regularity, discretization, boundary, and constitutive assumptions. It does not mean an automatic error bound for an arbitrary material, mesh, or finite-precision computation. An example whose constitutive parameters or boundary data are missing is **model not closed** and must not be reported as passed.

## Problem

Given a rotation rate, a mass, Cauchy-density parameters, granular state variables, and a frictional law, construct a rigorous, conservation-checkable, GPU-implementable closed loop such that:

1. inside a complex free boundary, $U$, $\nabla U$, and $\nabla^2U$ have one verifiable integral representation;
2. the core of the loop is the Mode B quasi-static Stokes equilibrium with a single stress and the yield classification into rigid core and movable set. Mode C (yield-driven erosion and deposition) is auxiliary, and slope promotion adds the repose proxy of Section 10.2. Increments and slope relief are corrected relative to the current shape by sequential convex subproblems on frozen fields. The spin is exogenous unless A-L holds (Section 11.7), and Mode A is an interface only;
3. every reduction, regularization, and degeneration is stated with its hypotheses, error sources, and boundary of validity.

**Scope of the text.** The document is a mathematical specification: definitions, propositions with proofs, and algorithms in pseudocode. It contains no implementation status and no results of particular runs. Numerical values appear only as checks of formulas or as order-of-magnitude estimates from published body parameters.

## Notation and standing assumptions

- **Geometry.** $V$ is a bounded Lipschitz polyhedral solid. Its boundary $\Gamma$ is a closed, consistently oriented, embedded triangulation of any genus, possibly with sealed cavities. The origin (the centre of mass) need not lie in $V$. This is hypothesis (G) of Section 8.1, and it is the only geometric hypothesis of the model. The general formulation of Section 8.1 (geometry-adapted basis, ray-segment quadrature, normal increments) is normative throughout; Section 8.1 shows why a radial form cannot be the general theory.
- **Density.** $\rho_K\ge\rho_{\min}>0$ on $\mathcal V$. This makes the kinetic-energy operator $\mathbf M_\Gamma$ SPD on $\ell\ge1$. The density is frozen within a geological step and advanced between steps by Section 9.2 (under A-Ξ with the dilatational ratio of Section 11.4), after which $\rho_K\ge\rho_{\min}$ is rechecked at the volume nodes.
- **Constants and spin.** $G$ is the gravitational constant. $\omega$ is the spin about the body $z$-axis, which is the axis of maximum inertia (Section 10.3.4, step 8). By default it is prescribed and frozen at $\omega_n=\omega(t_n)$ within a step (Section 10.3.1); under A-L it is $\omega_n=L_n/I_1^{(n)}$ (Section 11.7), again frozen within the step.
- **Potential and derivatives.** $W=U+U_c$. Derivatives at exterior points are classical. Interior and boundary Hessians use the spherical finite-part convention.
- **Rheology.** The inertial rheology is used for $p>0$. At $p=0$ or $\dot\gamma=0$ the regularization or the static-yield branch is required. Under A-Ξ and A-ϖ the solid fraction $\nu$ and the damage $\varpi$ are material state variables, frozen within a step and advanced between steps (Chapter 11).
- **Status of statements.** A proposition, lemma, or theorem holds under its stated hypotheses and nothing more. Existence of the coupled velocity–pressure fixed point, convergence of the iterations that compute it, and the limit $\dot\gamma_\varepsilon\to0$ are open (PO-1, PO-11, PO-12, PO-19 of Appendix A.5). An algorithmic step that relies on them is a stopping rule judged by Section 2.15, not a consequence of a lemma. In particular Lemma 8.12 gives a unique velocity only for a frozen viscosity $\eta\ge\eta_{\rm lo}$, Proposition 2.17.1 only at frozen pressure, and Proposition 11.8 only at frozen pressure, state, and dilatation.

Potentials are positive ($U,U_c,W$). The conservative effective acceleration is $g_{\mathrm{eff}}=\nabla W$. Where $-\rho\nabla\Phi_{\mathrm{eff}}$ appears, $\Phi_{\mathrm{eff}}=-W$.

Chapter 2 sometimes drops boldface; the symbols are the same: stress $\boldsymbol\sigma$, direction $\hat{\boldsymbol\omega}$, angular velocity $\boldsymbol\Omega=\omega\mathbf e_z$.

# 1. Gravitational field, Hessian, and residuals of a polyhedral generalized Cauchy density

## Introduction: scope and logic

This chapter gives two rigorous integral representations of one potential, the exact range of the residual method, and a finite-degree polynomial approximation as a separate scheme with its own error. The two routes are:

- **Route A** defines the finite part by rays centred at the field point, with spherical exclusion.
- **Route B** turns the potential kernel, by a Feynman parametrization, into a radial kernel about a moving centre.

**Engineering choice.** Route A is the authoritative production implementation: it handles arbitrary non-star-shaped polyhedra by field-centred ray segments and explicit finite-part limits. Route B is an analytic cross-check and optional acceleration path; it must not be mixed into Route A during one evaluation.

They are two representations of the same potential and are never added. When a one-dimensional sub-integral is evaluated by a Carlson, Appell, or theta function, every outer integral not yet evaluated is retained. Each equality holds under the hypotheses stated next to it. "Analytic" means the identity introduces no transformation error. A strict numerical bound additionally needs separate bounds on truncation, quadrature, and rounding error.

## 1.1 Geometry, density, and potential

Let $V\subset\mathbb R^3$ be a bounded polyhedral solid with $V=\overline{\operatorname{int}V}$. Its boundary consists of finitely many nondegenerate planar triangles, consistently oriented and not self-intersecting. Several components and internal cavities are allowed; on a cavity boundary the outward normal of the solid is used. If the mesh does not define such a solid, the volume theorems below do not apply directly. Examples for small bodies are given by Hirata et al. [4].

For $k=1,\dots,K$ fix $w_k\in\mathbb R$, $\sigma_k>0$, $\alpha_k>0$, and $\mathbf c_k\in\mathbb R^3$. The density, smooth on all of space, is

$$
\rho_K(\mathbf y)
=
\sum_{k=1}^{K}
w_k
\left[
1+\sigma_k^2\|\mathbf y-\mathbf c_k\|^2
\right]^{-\alpha_k}.
$$

The identities allow signed weights. As a physical density one must check $\rho_K\ge0$ a.e. on $V$. This chapter needs nothing more. The standing assumption $\rho_K\ge\rho_{\min}>0$ is a hypothesis of the mechanics (Proposition 8.1.3, Section 8.3, and Lemma 8.12), not of the gravity identities.

**Constant matrix term.** The model adds a constant matrix density $\rho_b\ge0$:

$$
\rho_K(\mathbf y)=\rho_b+\sum_{k=1}^{K}w_k\left[1+\sigma_k^2\|\mathbf y-\mathbf c_k\|^2\right]^{-\alpha_k},
$$

with $K=0$ allowed (a uniform body). A constant is not a Cauchy kernel, but it needs none of the machinery below. Its potential, gradient, and Hessian are the closed-form field $\rho_b\Phi_V$ of the uniform polyhedron (Section 3.9), and every statement of this chapter holds for it with $\nabla\rho_b=\mathbf 0$. The kernels then describe the departure from the matrix, and their weights may be negative (a light core) provided $\rho_K\ge0$ on $V$.

**Anisotropic (sheared) cores.** The core transport of Section 9.2 deforms kernels by the local deformation gradient. The admissible class is therefore the generalized Cauchy kernel with an SPD metric $\mathbf S_k$ (dimension $\mathrm L^{-2}$):

$$
\rho_K(\mathbf y)
=
\sum_{k=1}^{K}
w_k
\left[
1+(\mathbf y-\mathbf c_k)^{\!\top}\mathbf S_k(\mathbf y-\mathbf c_k)
\right]^{-\alpha_k},
\qquad
\mathbf S_k=\mathbf S_k^{\!\top}\succ 0.
$$

The isotropic case is $\mathbf S_k=\sigma_k^2\mathbf I$. The point $\mathbf c_k$ is the unique critical point of the $k$-th summand, its value there is $w_k$, and its Hessian there is $-2\alpha_kw_k\mathbf S_k$. These three facts are what Section 9.2 transports. Sections 1.2, 1.3, 1.5, and 1.10–1.15 use only smoothness and boundedness of $\rho_K$ near $V$, so they hold for this class verbatim. Route A carries over through Lemma 1.4′ (one $3\times3$ product per ray and kernel), and Route B through the remark at the end of Section 1.7. Chapters 1–2 are written isotropically. Chapter 3 and Section 6.1 give the general formulae.

With positive potentials,

$$
U_K(\mathbf x)
=
G\int_V
\frac{\rho_K(\mathbf y)}
{\|\mathbf x-\mathbf y\|}
\,d^3\mathbf y,
$$

where $\mathbf x$ is the field point and $\mathbf y$ the source point. $U_K$ is finite everywhere because $V$ and $\rho_K$ are bounded and $1/\|\mathbf x-\mathbf y\|$ is locally integrable. We write $H_{ij}(\mathbf x)=\partial_i\partial_jU_K(\mathbf x)$, called the classical Hessian only where classical second derivatives exist. On edges and other exceptional points the finite part $H_{ij,\ell_0}^{\mathrm{FP}}$ is used.

## 1.2 Distributional derivatives and the spherical-exclusion convention

Let $\mathbf r=\mathbf x-\mathbf y$ and $R=\|\mathbf r\|>0$. Differentiating $R=(r_kr_k)^{1/2}$ and $R^{-1}$ twice gives

$$
\frac{\partial R}{\partial x_i}=\frac{r_i}{R},\qquad
\frac{\partial}{\partial x_i}\frac1R=-\frac{r_i}{R^3},\qquad
\frac{\partial^2}{\partial x_j\,\partial x_i}\frac1R
=
\frac{3r_i r_j-R^2\delta_{ij}}{R^5}.
$$

The numerator is traceless. The kernel is $O(R^{-3})$, not absolutely integrable at $R=0$, so the distributional derivative must record the spherical contact term. With spherical exclusion,

$$
\frac{\partial^2}{\partial x_j\,\partial x_i}\frac1R
=
\operatorname{p.v.}_{\mathrm{ball}}
\frac{3r_i r_j-R^2\delta_{ij}}{R^5}
-
\frac{4\pi}{3}\delta_{ij}\delta_0(\mathbf r),
$$

whose trace is $\Delta_{\mathbf x}R^{-1}=-4\pi\delta_0(\mathbf r)$ (the principal-value kernel is traceless). For $\mathbf x\in\operatorname{int}V$, where $\rho_K$ is smooth,

$$
\begin{aligned}
H_{ij}(\mathbf x)
={}&
G\lim_{\varepsilon\downarrow0}
\int_{V\setminus B_\varepsilon(\mathbf x)}
\rho_K(\mathbf y)
\frac{
3(x_i-y_i)(x_j-y_j)
-
\|\mathbf x-\mathbf y\|^2\delta_{ij}
}{
\|\mathbf x-\mathbf y\|^5
}
\,d^3\mathbf y
-
\frac{4\pi G}{3}
\rho_K(\mathbf x)\delta_{ij}.
\end{aligned}
$$

On the boundary $\rho_K\mathbf 1_V$ jumps, and the distributional identity alone gives the product of the contact distribution with the indicator no unique value. Boundary values below are a specified spherical-exclusion representative. They need not equal the interior or exterior limit.

## 1.3 Local tangent cone, logarithmic term, and finite part

This section gives one checkable pointwise convention for interior points, faces, edges, and vertices.

Fix a field point $\mathbf x$. Because the mesh is finite and planar, there is an $r_\ast(\mathbf x)>0$ such that $V\cap B_{r_\ast}(\mathbf x)$ coincides with a polyhedral cone at $\mathbf x$. Choose $0<r_0<r_\ast(\mathbf x)$. For an exterior point also require $r_0<\operatorname{dist}(\mathbf x,V)$; the local cone is then empty.

Rays are $\mathbf y=\mathbf x-R\mathbf u$ with $R>0$ and $\mathbf u\in\mathbb S^2$. Let $\Gamma_{\mathbf x}$ be the set of directions with $\mathbf x-R\mathbf u\in V$ for all small $R>0$; it is defined up to a null set of directions. Define

$$
T_{ij}(\mathbf u)=3u_i u_j-\delta_{ij},\qquad
C_{ij}(\mathbf x)=\int_{\Gamma_{\mathbf x}}T_{ij}(\mathbf u)\,d\Omega,\qquad
\vartheta_{\mathbf x}=\frac{1}{4\pi}\int_{\Gamma_{\mathbf x}}d\Omega .
$$

$\sum_iT_{ii}=3\|\mathbf u\|^2-3=0$, hence $\sum_iC_{ii}=0$. With the truncated integral

$$
J_{ij,\varepsilon}(\mathbf x)
=
G\int_{V\setminus B_\varepsilon(\mathbf x)}
\rho_K(\mathbf y)
\frac{
3(x_i-y_i)(x_j-y_j)
-
\|\mathbf x-\mathbf y\|^2\delta_{ij}
}{
\|\mathbf x-\mathbf y\|^5
}
\,d^3\mathbf y
$$

and a prescribed length $\ell_0>0$, define

$$
\begin{aligned}
H_{ij,\ell_0}^{\mathrm{FP}}(\mathbf x)
={}&
\lim_{\varepsilon\downarrow0}
\left[
J_{ij,\varepsilon}(\mathbf x)
-
G\rho_K(\mathbf x)C_{ij}(\mathbf x)
\log\frac{\ell_0}{\varepsilon}
\right]
-
\frac{4\pi G}{3}
\vartheta_{\mathbf x}\rho_K(\mathbf x)\delta_{ij}.
\end{aligned}
$$

The limit exists (Section 1.5). If $\rho_KC_{ij}\ne0$, the unsubtracted truncated integral diverges logarithmically. $H^{\mathrm{FP}}_{\ell_0}$ is therefore a finite part at scale $\ell_0$, not an ordinary limit. Its trace is

$$
\operatorname{Tr}H_{\ell_0}^{\mathrm{FP}}(\mathbf x)
=
-4\pi G\vartheta_{\mathbf x}\rho_K(\mathbf x),
$$

because the counterterm is traceless and the contact term contributes $-4\pi G\vartheta_{\mathbf x}\rho_K$. In the interior $\vartheta_{\mathbf x}=1$ (the Poisson value); on a face, $\vartheta_{\mathbf x}=1/2$.

Changing the scale shifts only the subtracted logarithm:

$$
H_{ij,\ell_1}^{\mathrm{FP}}(\mathbf x)
-
H_{ij,\ell_0}^{\mathrm{FP}}(\mathbf x)
=
G\rho_K(\mathbf x)C_{ij}(\mathbf x)
\log\frac{\ell_0}{\ell_1}.
$$

So $H^{\mathrm{FP}}$ is scale-free iff $\rho_KC=0$, i.e. $C(\mathbf x)=0$ when $\rho_K(\mathbf x)\ne0$. A full sphere gives $C=0$ by isotropy ($\int_{\mathbb S^2}u_iu_j\,d\Omega=\tfrac{4\pi}{3}\delta_{ij}$). Since $T(-\mathbf u)=T(\mathbf u)$, a hemisphere carries half of that, so $C=0$ on the relative interior of a face. Proposition 2.7.1 computes the hemispherical moment and uses the Feynman representation to show that the logarithmic RTFP singularity is absent at every face centre. Edges and vertices must be computed pointwise; there $C(\mathbf x)$ need not vanish.

## 1.4 Route A: spherical coordinates, intersection segments, and completion of the square

With $\mathbf y=\mathbf x-R\mathbf u$, $\mathbf u=(\sin\theta\cos\varphi,\sin\theta\sin\varphi,\cos\theta)$, $d\Omega=\sin\theta\,d\theta\,d\varphi$, and $d^3\mathbf y=R^2\,dR\,d\Omega$,

$$
\frac{
3(x_i-y_i)(x_j-y_j)
-
\|\mathbf x-\mathbf y\|^2\delta_{ij}
}{
\|\mathbf x-\mathbf y\|^5
}
\,d^3\mathbf y
=
T_{ij}(\mathbf u)\frac{dR}{R}\,d\Omega
$$

(numerator $T_{ij}R^2$, denominator $R^5$, volume element $R^2$). For almost every direction the ray intersection is a finite union of intervals:

$$
E_{\mathbf x}(\mathbf u)
=
\{R>0:\mathbf x-R\mathbf u\in V\}
=
\bigcup_{\ell=1}^{L(\mathbf u)}
[R^-_\ell(\mathbf u),R^+_\ell(\mathbf u)].
$$

For an interior field point the first segment is $(0,R_1]$. A discretization may use $[0,R_1]$, since $R=0$ contributes nothing to the potential and gradient integrals; the Hessian there follows Sections 1.2–1.3. Every nonempty interval enters the sum, including interval $\ell=1$ of an exterior ray. For the $k$-th kernel set

$$
\mathbf p_k=\mathbf x-\mathbf c_k,\qquad
b_k(\mathbf u)=\mathbf p_k\cdot\mathbf u,\qquad
a_k=\frac1{\sigma_k},\qquad
d_k^2(\mathbf u)=a_k^2+\|\mathbf p_k\|^2-b_k^2(\mathbf u).
$$

Completing the square in $\|\mathbf p_k-R\mathbf u\|^2=\|\mathbf p_k\|^2-2Rb_k+R^2$,

$$
1+\sigma_k^2
\|\mathbf x-R\mathbf u-\mathbf c_k\|^2
=
\sigma_k^2
\left[
[R-b_k(\mathbf u)]^2+d_k^2(\mathbf u)
\right].
$$

Since $\|\mathbf p_k\|^2-b_k^2=\|\mathbf p_k-b_k\mathbf u\|^2\ge0$, one has $d_k^2\ge a_k^2>0$: the Cauchy factor never vanishes on a ray.

**Lemma 1.4′ (Ray form of an anisotropic core).** For the anisotropic kernel of Section 1.1 set

$$
s_k(\mathbf u)=\mathbf u^{\!\top}\mathbf S_k\mathbf u,\qquad
b_k(\mathbf u)=\frac{\mathbf u^{\!\top}\mathbf S_k\mathbf p_k}{s_k(\mathbf u)},\qquad
d_k^2(\mathbf u)=\frac{1+\mathbf p_k^{\!\top}\mathbf S_k\mathbf p_k}{s_k(\mathbf u)}-b_k^2(\mathbf u).
$$

Then

$$
1+(\mathbf p_k-R\mathbf u)^{\!\top}\mathbf S_k(\mathbf p_k-R\mathbf u)
=
s_k(\mathbf u)\left[(R-b_k(\mathbf u))^2+d_k^2(\mathbf u)\right],
\qquad
d_k^2(\mathbf u)\ge\frac1{s_k(\mathbf u)}>0.
$$

*Proof.* Complete the square in $s_kR^2-2R\,\mathbf u^{\!\top}\mathbf S_k\mathbf p_k+1+\mathbf p_k^{\!\top}\mathbf S_k\mathbf p_k$. The bound follows from Cauchy–Schwarz in the $\mathbf S_k$ inner product, $(\mathbf u^{\!\top}\mathbf S_k\mathbf p_k)^2\le s_k\,\mathbf p_k^{\!\top}\mathbf S_k\mathbf p_k$. ∎

Every radial formula of Sections 1.5, 1.6, 1.11, and 3.1–3.3 therefore holds for an anisotropic core after the substitution $\sigma_k^{-2\alpha_k}\mapsto s_k(\mathbf u)^{-\alpha_k}$, with the direction-dependent $b_k,d_k$. The isotropic case $s_k=\sigma_k^2$ returns the definitions above. $d_k\ge\lambda_{\max}(\mathbf S_k)^{-1/2}$ never degenerates.

## 1.5 Route A: near field, far field, and the finite-part identity

With $r_0$ from Section 1.3, split $E_{\mathbf x}=E^{\mathrm{near}}_{\mathbf x}\mathbin{\dot\cup}E^{\mathrm{far}}_{\mathbf x}$ at $R=r_0$ (a shared endpoint contributes nothing). For almost every $\mathbf u$, $E^{\mathrm{near}}_{\mathbf x}(\mathbf u)=(0,r_0)$ if $\mathbf u\in\Gamma_{\mathbf x}$ and $\varnothing$ otherwise. For $0<\varepsilon<r_0$, adding and subtracting $\rho_K(\mathbf x)$ in the near field gives

$$
\begin{aligned}
J_{ij,\varepsilon}(\mathbf x)
={}&
G\rho_K(\mathbf x)
C_{ij}(\mathbf x)
\log\frac{r_0}{\varepsilon}
+
G\int_{\Gamma_{\mathbf x}}
T_{ij}(\mathbf u)
\int_\varepsilon^{r_0}
\frac{
\rho_K(\mathbf x-R\mathbf u)-\rho_K(\mathbf x)
}{R}
\,dR\,d\Omega\\
&+
G\int_{\mathbb S^2}
T_{ij}(\mathbf u)
\int_{E_{\mathbf x}^{\mathrm{far}}(\mathbf u)}
\frac{\rho_K(\mathbf x-R\mathbf u)}{R}
\,dR\,d\Omega.
\end{aligned}
$$

The density difference is bounded: $\rho_K(\mathbf x-R\mathbf u)-\rho_K(\mathbf x)=-R\int_0^1\nabla\rho_K(\mathbf x-tR\mathbf u)\cdot\mathbf u\,dt$, so the quotient is at most $M_1=\sup\|\nabla\rho_K\|$ on a compact set containing the near segments. The near-field difference therefore passes to $\varepsilon\downarrow0$. The far field converges absolutely because $R\ge r_0$ and $V$ is bounded. Substituting into Section 1.3 (the logarithms combine to $\log(r_0/\ell_0)$),

$$
\boxed{
\begin{aligned}
H_{ij,\ell_0}^{\mathrm{FP}}(\mathbf x)
={}&
G\rho_K(\mathbf x)C_{ij}(\mathbf x)
\log\frac{r_0}{\ell_0}
+
G\int_{\Gamma_{\mathbf x}}
T_{ij}(\mathbf u)
\int_0^{r_0}
\frac{
\rho_K(\mathbf x-R\mathbf u)-\rho_K(\mathbf x)
}{R}
\,dR\,d\Omega\\
&+
G\int_{\mathbb S^2}
T_{ij}(\mathbf u)
\int_{E_{\mathbf x}^{\mathrm{far}}(\mathbf u)}
\frac{\rho_K(\mathbf x-R\mathbf u)}{R}
\,dR\,d\Omega
-
\frac{4\pi G}{3}
\vartheta_{\mathbf x}\rho_K(\mathbf x)\delta_{ij}.
\end{aligned}
}
$$

The identity is independent of $r_0$. Moving to $r_1>r_0$ transfers $[r_0,r_1)$ from the far field to the near field, where the difference term plus the constant term equals the original density term pointwise. For an exterior point with $r_0<\operatorname{dist}(\mathbf x,V)$: $\Gamma_{\mathbf x}=\varnothing$ and $\vartheta_{\mathbf x}=0$, every segment is far field, and the identity is the classical Hessian with no counterterm. Near and far fields are complementary, so no segment is dropped or counted twice, including the first segment of an exterior ray.

## 1.6 The radial antiderivative for exponent one, and the zero endpoint

Fix $k$ and $\mathbf u$, and write $b,d$ for $b_k(\mathbf u),d_k(\mathbf u)$. Let

$$
A_0=1+\sigma_k^2\|\mathbf x-\mathbf c_k\|^2,\qquad
D(R)=1+\sigma_k^2\|\mathbf x-R\mathbf u-\mathbf c_k\|^2
=\sigma_k^2R^2-2\sigma_k^2bR+A_0
=\sigma_k^2[(R-b)^2+d^2].
$$

With $D'=2\sigma_k^2(R-b)$, the partial fractions are

$$
\frac1{R D(R)}
=
\frac1{A_0}
\left[
\frac1R
-
\frac{\sigma_k^2R-2\sigma_k^2b}{D(R)}
\right],\qquad
\sigma_k^2R-2\sigma_k^2b=\tfrac12D'(R)-\sigma_k^2b .
$$

Check: over the common denominator the numerator is $D-R(\sigma_k^2R-2\sigma_k^2b)=A_0$. With the continuous real arctangent branch for $R>0$, the antiderivative is

$$
P(R)
=
\frac1{A_0}
\left[
\log R
-
\frac12\log D(R)
+
\frac bd
\arctan\frac{R-b}{d}
\right],
$$

since $\frac{d}{dR}(-\tfrac12\log D)=-\sigma_k^2(R-b)/D$ and $\frac{d}{dR}\arctan\frac{R-b}{d}=\sigma_k^2d/D$, so $P'=[D-\sigma_k^2R(R-2b)]/(A_0RD)=1/(RD)$. Thus $\int_A^B dR/(RD)=P(B)-P(A)$ for $0<A<B$.

The radial finite part at the zero endpoint, for $L>0$, is

$$
J_{\ell_0}^{\mathrm{FP}}(0,L)
=
\lim_{\varepsilon\downarrow0}
\left[
P(L)-P(\varepsilon)
-
\frac1{A_0}
\log\frac{\ell_0}{\varepsilon}
\right].
$$

From $D(0)=A_0$ and $\arctan((\varepsilon-b)/d)=-\arctan(b/d)+O(\varepsilon)$, one gets $P(\varepsilon)=A_0^{-1}[\log\varepsilon-\tfrac12\log A_0-(b/d)\arctan(b/d)]+o(1)$. The counterterm cancels $\log\varepsilon$, and

$$
\boxed{
\begin{aligned}
J_{\ell_0}^{\mathrm{FP}}(0,L)
={}&
\frac1{A_0}
\Biggl[
\log\frac{L}{\ell_0}
-
\frac12\log\frac{D(L)}{A_0}
+
\frac bd
\left(
\arctan\frac{L-b}{d}
+
\arctan\frac bd
\right)
\Biggr].
\end{aligned}
}
$$

The two arctangents add because $-(b/d)\arctan((\varepsilon-b)/d)\to+(b/d)\arctan(b/d)$. Both logarithms have dimensionless arguments. This single-direction finite part is one way to evaluate the near field of Section 1.5; it is not an extra term.

Section 3.2 writes the same identity with $\sigma_k^2$ removed from $D$, because $\sigma_k^{-2\alpha_k}$ stands outside the radial integral. With $\widetilde D=(R-b)^2+d^2$, one has $D=\sigma_k^2\widetilde D$ and $\int dR/(RD)=\sigma_k^{-2}\int dR/(R\widetilde D)$. The symbol $\widetilde A_0$ of Section 3.2 is $b^2+d^2=\sigma_k^{-2}A_0$.

## 1.7 Route B: Feynman parametrization

Fix one kernel and drop $k$. Set $A(\mathbf y)=\|\mathbf y-\mathbf x\|^2$, $a=1/\sigma$, and $B(\mathbf y)=\|\mathbf y-\mathbf c\|^2+a^2$. Then $[1+\sigma^2\|\mathbf y-\mathbf c\|^2]^{-\alpha}=\sigma^{-2\alpha}B^{-\alpha}$. Write both factors as Gamma integrals, $Q^{-\nu}=\Gamma(\nu)^{-1}\int_0^\infty\lambda^{\nu-1}e^{-\lambda Q}d\lambda$ ($Q,\nu>0$), and substitute $s=u\lambda$, $t=(1-u)\lambda$ (Jacobian $\lambda$). The inner integral $\int_0^\infty\lambda^{\alpha-1/2}e^{-\lambda Q}d\lambda=\Gamma(\alpha+\tfrac12)Q^{-\alpha-1/2}$ then gives

$$
\boxed{
A^{-1/2}B^{-\alpha}
=
\frac{\Gamma\left(\alpha+\frac12\right)}
{\Gamma(1/2)\Gamma(\alpha)}
\int_0^1
u^{-1/2}(1-u)^{\alpha-1}
\left[uA+(1-u)B\right]^{-\alpha-1/2}
\,du.
}
$$

With

$$
\mathbf z(u)=u\mathbf x+(1-u)\mathbf c,\qquad
m^2(u)=(1-u)a^2+u(1-u)\|\mathbf x-\mathbf c\|^2,
$$

expanding the quadratic form gives $u\|\mathbf y-\mathbf x\|^2+(1-u)\|\mathbf y-\mathbf c\|^2=\|\mathbf y-\mathbf z\|^2+u(1-u)\|\mathbf x-\mathbf c\|^2$, hence

$$
\boxed{
uA(\mathbf y)+(1-u)B(\mathbf y)
=
\|\mathbf y-\mathbf z(u)\|^2+m^2(u).
}
$$

The centre $\mathbf z(u)$ moves along the segment from $\mathbf c$ to $\mathbf x$. The parameter mass $m^2=(1-u)[a^2+u\|\mathbf x-\mathbf c\|^2]>0$ for $0\le u<1$ has dimension $\mathrm L^2$. As $u\uparrow1$, $\mathbf z\to\mathbf x$ and $m\to0$: the near-field singularity sits at this endpoint, and the transformation does not remove it. The parametrization is an identity, not an approximation.

**Remark (anisotropic cores on Route B).** For $\mathbf S\succ0$ put $\bar\sigma=(\det\mathbf S)^{1/6}$, $\widehat{\mathbf S}=\mathbf S/\bar\sigma^2$ ($\det\widehat{\mathbf S}=1$, dimensionless), and $a=1/\bar\sigma$. Then $1+(\mathbf y-\mathbf c)^{\!\top}\mathbf S(\mathbf y-\mathbf c)=\bar\sigma^2\widehat B$ with $\widehat B=(\mathbf y-\mathbf c)^{\!\top}\widehat{\mathbf S}(\mathbf y-\mathbf c)+a^2$. The Feynman identity applies to $A^{-1/2}\widehat B^{-\alpha}$ with prefactor $\bar\sigma^{-2\alpha}$, and

$$
uA+(1-u)\widehat B=(\mathbf y-\mathbf z)^{\!\top}\mathbf M_u(\mathbf y-\mathbf z)+m^2,\qquad
\mathbf M_u=u\mathbf I+(1-u)\widehat{\mathbf S},
$$

$$
\mathbf z=\mathbf M_u^{-1}\big(u\mathbf x+(1-u)\widehat{\mathbf S}\mathbf c\big),\qquad
m^2=(1-u)a^2+u(1-u)(\mathbf x-\mathbf c)^{\!\top}\mathbf M_u^{-1}\widehat{\mathbf S}(\mathbf x-\mathbf c).
$$

The second term of $m^2$ is the parallel sum of $u\mathbf I$ and $(1-u)\widehat{\mathbf S}$. It is SPD, so $m^2\ge(1-u)a^2>0$ for $u<1$. The map $\mathbf y\mapsto\mathbf M_u^{1/2}\mathbf y$ sends polyhedra to polyhedra, so $S_p$ of Section 1.8 becomes $\det(\mathbf M_u)^{-1/2}S_p(\mathbf M_u^{1/2}\mathbf z,m;\mathbf M_u^{1/2}V)$, and the face reduction of Section 1.8 applies at each $u$. For $\widehat{\mathbf S}=\mathbf I$ this is the isotropic formula.

## 1.8 Parametric integral, dimensional reduction, and Hessian of the Feynman potential

Set $p_k=\alpha_k+\tfrac12$, $C_{\alpha_k}=\Gamma(\alpha_k+\tfrac12)/(\Gamma(\tfrac12)\Gamma(\alpha_k))$, and

$$
S_p(\mathbf z,m;V)
=
\int_V
\left[
\|\mathbf y-\mathbf z\|^2+m^2
\right]^{-p}
\,d^3\mathbf y.
$$

Tonelli's theorem exchanges the $u$ and $\mathbf y$ integrals for each nonnegative kernel, and signed channels are exchanged one by one. Hence

$$
\boxed{
U_K(\mathbf x)
=
G\sum_{k=1}^{K}
w_k\sigma_k^{-2\alpha_k}C_{\alpha_k}
\int_0^1
u^{-1/2}(1-u)^{\alpha_k-1}
S_{p_k}
\left(
\mathbf z_k(u),m_k(u);V
\right)
\,du,
}
$$

a one-dimensional parameter integral around a volume integral. This does not claim that $S_p$ has a closed form.

**Reduction to faces.** Fix $u<1$, let $\mathbf r=\mathbf y-\mathbf z$ and $r=\|\mathbf r\|$, and set

$$
A_p(r;m)=\int_0^r\frac{s^2}{(s^2+m^2)^p}\,ds,\qquad
\mathbf F_p(\mathbf r;m)=\frac{A_p(r;m)}{r^3}\mathbf r .
$$

By $\nabla\cdot(f(r)\mathbf r)=r^{-2}(r^3f)'$ one gets $\nabla_{\mathbf r}\cdot\mathbf F_p=r^{-2}A_p'=(r^2+m^2)^{-p}$. The divergence theorem then gives

$$
\boxed{
S_p(\mathbf z,m;V)
=
\sum_f
\int_{\Delta_f}
\mathbf F_p(\mathbf y-\mathbf z;m)
\cdot\mathbf n_f
\,dS.
}
$$

This needs $m>0$ (smooth $\mathbf F_p$), so the endpoint $u\to1$ is excluded. For $p=3/2$, $A_{3/2}(r;m)=\operatorname{arsinh}(r/m)-r/\sqrt{r^2+m^2}$; it vanishes at $r=0$, and its derivative is $r^2/(r^2+m^2)^{3/2}$.

**Reduction to edges.** On a face, let $\boldsymbol\xi$ be the in-plane coordinate and $\boldsymbol\xi_0$ the projection of $\mathbf z$, with $\rho=\|\boldsymbol\xi-\boldsymbol\xi_0\|$ and $\delta_f=(\mathbf y_f-\mathbf z)\cdot\mathbf n_f$ (independent of $\mathbf y_f$ in the plane). Then $\|\mathbf y-\mathbf z\|^2=\rho^2+\delta_f^2$, and the face integrand is

$$
g_{p,f}(\rho)=\delta_f\frac{A_p(\sqrt{\rho^2+\delta_f^2};m)}{(\rho^2+\delta_f^2)^{3/2}} .
$$

With $h_{p,f}(\rho)=\rho^{-2}\int_0^\rho s\,g_{p,f}(s)\,ds$, the field $\mathbf G_{p,f}=h_{p,f}(\rho)(\boldsymbol\xi-\boldsymbol\xi_0)$ satisfies $\nabla_{\boldsymbol\xi}\cdot\mathbf G_{p,f}=\rho^{-1}(\rho^2h)'=g_{p,f}$. With the outward in-plane conormal $\boldsymbol\nu_f$,

$$
\int_{\Delta_f}g_{p,f}\,dS
=
\oint_{\partial\Delta_f}
\mathbf G_{p,f}\cdot\boldsymbol\nu_f
\,ds .
$$

This needs $h_{p,f}$ locally integrable, a boundary trace, and the $\rho=0$ limit taken from the original integral. It turns a face integral into an edge integral, but it does not make the edge integral elementary or Carlson.

**Exterior Hessian.** For an exterior point, $d_V:=\operatorname{dist}(\mathbf x,V)>0$. Let $q_k=u\|\mathbf y-\mathbf x\|^2+(1-u)[\|\mathbf y-\mathbf c_k\|^2+a_k^2]$. Only the first term depends on $\mathbf x$, so $\partial_iq_k=2u(x_i-y_i)$ and $\partial_i\partial_jq_k=2u\delta_{ij}$. Hence

$$
\frac{\partial}{\partial x_i}q_k^{-p_k}
=
-2p_ku(x_i-y_i)q_k^{-p_k-1},
\qquad
\frac{\partial^2}{\partial x_j\partial x_i}q_k^{-p_k}
=
-2p_ku\delta_{ij}q_k^{-p_k-1}
+
4p_k(p_k+1)u^2
(x_i-y_i)(x_j-y_j)
q_k^{-p_k-2}.
$$

Since $\|\mathbf x-\mathbf y\|\ge d_V$, a uniform dominating function on an exterior neighbourhood justifies differentiating under both integrals. Interior and boundary points cannot use this estimate. What remains is the $u$ integral and the geometric boundary integrals.

## 1.9 The Carlson criterion on Route B

Suppose the face-to-edge reduction, with any integrations by parts, has produced

$$
I_e
=
\int_{t_0}^{t_1}
\frac{R(t,Y(t))}{Y(t)}
\,dt,
\qquad
Y(t)^2=P_4(t)=a_4t^4+a_3t^3+a_2t^2+a_1t+a_0,
$$

with $R$ rational in $(t,Y)$. Sufficient conditions for the Carlson channel are:

1. the quartic term is nondegenerate: $a_4\ne0$;
2. $P_4$ is square-free: $\gcd(P_4,P_4')=1$, equivalently $\operatorname{Disc}(P_4)\ne0$;
3. the path avoids the branch points $P_4=0$, or an integrable endpoint limit is specified;
4. no pole of $R$ lies inside the path; for a pole on the path, declare whether the integral is ordinary, a Cauchy principal value, or a Hadamard finite part;
5. $Y(t)$ is continued continuously along the path (pointwise principal roots do not guarantee this).

Then the smooth projective model of $\mathcal C:Y^2=P_4(t)$ has genus $1$. This applies only to a sub-integral proved to have this form. It is not a closed form for the original two-dimensional angular integral.

### 1.9.1 Hermite reduction

Find an algebraic $Q(t,Y)$ and constants $c_\nu$ such that

$$
\frac{R(t,Y)}{Y}\,dt
=
dQ(t,Y)
+
\sum_{\nu=1}^{N}
c_\nu\omega_\nu ,
$$

using $2Y\,dY=P_4'\,dt$. Since $d(t^jY)$ has numerator of degree $j+3$ over $Y$, every $t^n\,dt/Y$ with $n\ge3$ reduces modulo $dQ$ to the span of $t^0,t^1,t^2$ over $Y$. The reduced basis is

$$
\omega_0=\frac{dt}{Y}\ (\text{first kind}),\qquad
\omega_1=\frac{t\,dt}{Y}\ (\text{third kind: simple poles at }\infty_\pm,\ \text{residues }\mp a_4^{-1/2}),
$$

$$
\omega_2=\frac{t^2\,dt}{Y}\ (\text{double poles at }\infty_\pm),\qquad
\omega_p=\frac{dt}{(t-p)Y}\ (\text{third kind, for each simple pole }p\text{ that is not a branch point}).
$$

The residue-free combination $\omega_2+\tfrac{a_3}{2a_4}\omega_1$ is of the second kind, from $t^2/Y=a_4^{-1/2}(1-\tfrac{a_3}{2a_4t}+O(t^{-2}))$ at $\infty_+$. $\omega_2$ cannot be omitted: stopping at $\{\omega_0,\omega_1,\omega_p\}$ gives wrong coefficients whenever the numerator has degree $\ge2$. A pole $(t-p)^{-m}$ with $m\ge2$ needs a $dQ$ that removes it, which means actually solving for the coefficients.

### 1.9.2 Reduction from degree four to degree three

Let $e$ be a root of $P_4$ and use the Möbius map

$$
t=\frac{\alpha+\beta z}{\gamma+\delta z},
\qquad
\Delta:=\alpha\delta-\beta\gamma\neq0,
\qquad
\beta/\delta=e,
$$

which sends $z=\infty$ to $t=e$. Then $dt=-\Delta\,dz/(\gamma+\delta z)^2$ and $P_4(t)=Q_4(z)/(\gamma+\delta z)^4$. The $z^4$ coefficient of $Q_4(z)=\sum_ja_j(\alpha+\beta z)^j(\gamma+\delta z)^{4-j}$ is $\delta^4P_4(e)=0$. Define $W(z)=(\gamma+\delta z)^2Y(t(z))$, so $W^2=Q_3$ with $Q_3=Q_4$ cubic. The non-branch points $\infty_\pm$ go to the two sheets over $z_\infty=-\gamma/\delta$, where $Q_3(z_\infty)=a_4\Delta^4/\delta^4\ne0$. Since $t=e+\Delta/(\delta^2(z-z_\infty))$,

$$
\begin{aligned}
\omega_0&=-\Delta\,\frac{dz}{W},\\
\omega_1&=-\Delta e\,\frac{dz}{W}-\frac{\Delta^2}{\delta^2}\,\frac{dz}{(z-z_\infty)W},\\
\omega_2&=-\Delta e^2\,\frac{dz}{W}-\frac{2e\Delta^2}{\delta^2}\,\frac{dz}{(z-z_\infty)W}-\frac{\Delta^3}{\delta^4}\,\frac{dz}{(z-z_\infty)^2W},\\
\omega_p&=-\frac{\Delta\delta}{\beta-p\delta}\,\frac{dz}{W}+\frac{\Delta^2}{(\beta-p\delta)^2}\,\frac{dz}{(z-z_p)W},
\qquad z_p=-\frac{\alpha-p\gamma}{\beta-p\delta},
\end{aligned}
$$

with $\beta-p\delta\ne0$ ($p\ne e$) and $W(z_p)\ne0$. The image of $\omega_1$ is of the third kind over $z_\infty$, not of the form $(A_0+A_1z)\,dz/W$. The image of $\omega_2$ has a double pole over $z_\infty$, which one more Hermite step removes. With $h=z-z_\infty$ and $Q_3=q_0+q_1h+q_2h^2+q_3h^3$ ($q_0\ne0$),

$$
d\!\left(\frac{W}{h}\right)
=\left(-\frac{q_0}{h^2}-\frac{q_1}{2h}+\frac{q_3}{2}h\right)\frac{dz}{W},
\qquad
\frac{dz}{h^2W}=-\frac{1}{q_0}\,d\!\left(\frac{W}{h}\right)-\frac{q_1}{2q_0}\,\frac{dz}{hW}+\frac{q_3}{2q_0}\,\frac{(z-z_\infty)\,dz}{W}.
$$

The last term contains $z\,dz/W$, the second-kind differential of the cubic model, where infinity is now a single branch point. Every basis differential is then a combination of the following, with all coefficients fixed by the actual endpoints, branch points, and poles through $\alpha,\beta,\gamma,\delta$ and $Q_3$:

- $dz/W$ and $z\,dz/W$;
- third-kind terms $dz/[(z-q)W]$ with $q\in\{z_\infty,z_p,\dots\}$;
- the exact differential $d(W/h)$, which joins $dQ$.

### 1.9.3 Carlson symmetric integrals

For real parameters and a path avoiding zeros of the denominator,

$$
R_F(x,y,z)=\frac12\int_0^\infty\frac{ds}{\sqrt{(s+x)(s+y)(s+z)}},\qquad
R_D(x,y,z)=\frac32\int_0^\infty\frac{ds}{(s+z)\sqrt{(s+x)(s+y)(s+z)}},
$$

$$
R_J(x,y,z,p)=\frac32\int_0^\infty\frac{ds}{(s+p)\sqrt{(s+x)(s+y)(s+z)}},\qquad
R_C(x,y)=\frac12\int_0^\infty\frac{ds}{(s+y)\sqrt{s+x}}=R_F(x,y,y)
$$

(with agreeing branches; $R_C$ is the notation for a repeated root). After the cubic transformation, a rational substitution to a semi-infinite interval, and the standard decomposition, the elliptic sub-integrals of the first, second, and third kinds are finite combinations of these. The coefficients depend on the roots, endpoints, branch, and poles, and are generated element by element.

### 1.9.4 Duplication iteration and the error boundary

For $R_F$, start from $(x_0,y_0,z_0)=(x,y,z)$ and iterate

$$
\lambda_n=\sqrt{x_ny_n}+\sqrt{y_nz_n}+\sqrt{z_nx_n},\qquad
(x,y,z)_{n+1}=\tfrac14\big((x,y,z)_n+\lambda_n\big),
$$

using the homogeneity $R_F(tx,ty,tz)=t^{-1/2}R_F(x,y,z)$ ($t>0$) to expand the termination on normalized variables. $R_D$, $R_J$, and $R_C$ need their own recurrences and compensation terms. A spread threshold only decides convergence of the iteration. Claiming $|R_F^{\rm computed}-R_F^{\rm exact}|\le\varepsilon$ also needs bounds on the series remainder, the branch choice, and the rounding of the inputs and the recurrence. Without them, a Carlson threshold is an internal error estimate only.

Carlson evaluation is thus a conditional evaluator of one-dimensional sub-integrals on a square-free quartic curve. It is not a universal solver for arbitrary nonintegral powers, two-dimensional angular integrals, or residuals.

## 1.10 Strict definition of the residual

The residual must be strictly complementary to the analytic terms. It must not omit any exterior interval, and it must not repeat any complete element. For each open spherical element $\Omega_m$ and each nonempty far-field segment, define the index $q=(m,\ell,k,i,j)$: element, far segment, kernel, and Hessian component. Each index carries one complete far-field integral

$$
I_q
=
G w_k
\int_{\Omega_m}
T_{ij}(\mathbf u)
\int_{A_{m\ell}(\mathbf u)}^{B_{m\ell}(\mathbf u)}
\frac{
\left[
1+\sigma_k^2
\|\mathbf x-R\mathbf u-\mathbf c_k\|^2
\right]^{-\alpha_k}
}{R}
\,dR\,d\Omega .
$$

Near-field intervals are not residuals. Section 1.5 has already split them into a logarithm, an integrable difference, and a contact term. A deterministic classification $\chi(q)\in\{\mathrm{Carlson},\mathrm{Appell},\mathrm{homotopy},\mathrm{Theta},\mathrm{direct}\}$ must partition the far set disjointly:

$$
\mathcal Q_{\mathrm{far}}
=
\mathcal Q_{\mathrm{Carlson}}
\mathbin{\dot\cup}
\mathcal Q_{\mathrm{Appell}}
\mathbin{\dot\cup}
\mathcal Q_{\mathrm{homotopy}}
\mathbin{\dot\cup}
\mathcal Q_{\mathrm{Theta}}
\mathbin{\dot\cup}
\mathcal Q_{\mathrm{direct}} .
$$

Then $H^{\mathrm{Carlson}}_{ij}=\sum_{\mathcal Q_{\rm Carlson}}I_q$ is the analytically evaluated part, $H^{\mathrm{res}}_{ij}$ is the sum over the other four classes, and

$$
H_{ij}^{\mathrm{far}}
=
H_{ij}^{\mathrm{Carlson}}
+
H_{ij}^{\mathrm{res}} .
$$

To split one element pointwise, one must give $F_q=F_q^{\mathrm{ext}}+F_q^{\mathrm{rem}}$ a.e. and define $I_q^{\mathrm{ext}},I_q^{\mathrm{rem}}$ as their integrals over $D_q$. Otherwise $F_q^{\mathrm{rem}}$ is undefined. Regularity of a residual is piecewise only. Inside an open element with no tangency, visibility boundary, vertex, branch point, or path pole, it is locally $C^r$ when the kernel and endpoints are smooth enough. On the closed domain one can usually assert only $L^1$; $C^\infty$ on the whole domain needs a proof. The residual is thus an explicit complementary sum: not "the full integral minus an undefined analytic term", and not four channels added on the same element.

## 1.11 Hypergeometric radial channel: Appell is only a special degeneration

With $\mathbf p_k,b_k,d_k,a_k$ of Section 1.4, the radial integral at fixed direction is

$$
\mathcal R_{\alpha_k}(A,B;\mathbf u)
=
\sigma_k^{-2\alpha_k}
\int_A^B
\frac{dR}
{R
\left[
(R-b_k(\mathbf u))^2+d_k^2(\mathbf u)
\right]^{\alpha_k}} .
$$

With $\beta_k=b_k/d_k$ and $t=(R-b_k)/d_k$, one has $dR=d_k\,dt$, the quadratic becomes $d_k^2(1+t^2)$, and $1/R=1/[d_k(t+\beta_k)]$. So

$$
\mathcal R_{\alpha_k}(A,B;\mathbf u)
=
\sigma_k^{-2\alpha_k}
d_k(\mathbf u)^{-2\alpha_k}
\int_{t(A)}^{t(B)}
\frac{dt}
{\left[t+\beta_k(\mathbf u)\right](1+t^2)^{\alpha_k}} .
$$

The scale factor is $\sigma_k^{-2\alpha_k}d_k^{-2\alpha_k}$. After $(1+t^2)^{-\alpha_k}=(1-it)^{-\alpha_k}(1+it)^{-\alpha_k}$, the integrand has two linear power factors and one simple pole. In general it is an incomplete Lauricella $F_D$ integral. It reduces to Appell $F_1$, a Gauss hypergeometric function, a Beta function, or an elementary function only under a special degeneration of the pole, the exponent, or an endpoint. Each representation needs a continuous branch and a path that does not cross the pole. On a complete element the Appell channel supplies only the radial factor of

$$
\int_{\Omega_m}
T_{ij}(\mathbf u)\,
\mathcal R_{\alpha_k}
\left(
A_{m\ell}(\mathbf u),
B_{m\ell}(\mathbf u);
\mathbf u
\right)
\,d\Omega ,
$$

so the two-dimensional directional integral remains. It needs its own rigorous evaluator or error certificate. A nonintegral $\alpha_k$ does not mean the element is "completed by Appell".

## 1.12 Homotopy residual and direct quadrature

Let $D_m\subset\mathbb R^2$ be a directional parameter domain, star-shaped with respect to the origin, carrying the residual two-form $\eta_m=F_m\,d\xi\wedge d\eta$. With $A(\xi,\eta)=\int_0^1sF_m(s\xi,s\eta)\,ds$ define

$$
\mathcal H\eta_m
=
A(\xi,\eta)\left(\xi\,d\eta-\eta\,d\xi\right).
$$

For smooth $F_m$, $d(\mathcal H\eta_m)=(2A+\xi A_\xi+\eta A_\eta)\,d\xi\wedge d\eta$. Since $2A+\xi A_\xi+\eta A_\eta=\int_0^1\frac{d}{ds}[s^2F_m(s\xi,s\eta)]\,ds=F_m(\xi,\eta)$ (the boundary term at $s=0$ vanishes), Stokes' theorem gives

$$
\int_{D_m}
F_m(\xi,\eta)\,d\xi\,d\eta
=
\int_{\partial D_m}
\mathcal H\eta_m .
$$

A non-star-shaped $D_m$ is triangulated into star-shaped pieces; shared interior edges cancel by orientation. For a merely locally integrable $F_m$, the singular points must be excised, the limit taken, and the small boundary circles shown to vanish. The formula still keeps an integral in the homotopy parameter.

Elements that satisfy none of the Carlson, Appell, or homotopy conditions go to direct quadrature, $\mathcal Q_{\rm direct}$. A double-exponential halving test $|I_{h/2}-I_h|<\mathrm{tol}$ is an a-posteriori estimate only. A strict bound needs the endpoint singularity order, the analyticity strip width, the discrete tail, the outer quadrature error, and rounding. Direct quadrature is a legitimate fallback without an automatic certificate.

## 1.13 Conditional place of the higher-genus theta channel

The higher-genus channel is entered only when algebraization actually yields $Y^2=P_n(t)$ with $P_n$ square-free ($\gcd(P_n,P_n')=1$), of degree $n\ge5$, and the differential is genuinely abelian on that curve. For $n=2g+1$ or $2g+2$ the genus is $g=\lfloor(n-1)/2\rfloor$. Neither nonconvex geometry nor a nonintegral $\alpha_k$ implies $n\ge5$; the degree comes from the actual elimination.

Let $\tau=X+iY$ with $X=X^{\!\top}$ and $Y=Y^{\!\top}>0$ (here $Y$ is the imaginary part of the period matrix), and let $\mathbf z=\mathbf a+i\mathbf b$. Define

$$
\Theta(\mathbf z\mid\tau)
=
\sum_{\mathbf n\in\mathbb Z^g}
\exp\left(
\pi i\mathbf n^T\tau\mathbf n
+
2\pi i\mathbf n^T\mathbf z
\right).
$$

With $\lambda=\lambda_{\min}(Y)$, each term has modulus $\exp(-\pi\mathbf n^TY\mathbf n-2\pi\mathbf n^T\mathbf b)\le\exp(-\pi\lambda\|\mathbf n\|_2^2+2\pi\|\mathbf b\|_2\|\mathbf n\|_2)$. With $r_\Theta=\|\mathbf b\|_2/\lambda$ and $q\ge r_\Theta$, completing the square gives $-\pi\lambda q^2+2\pi\|\mathbf b\|_2q=\pi\|\mathbf b\|_2^2/\lambda-\pi\lambda(q-r_\Theta)^2$. The exponent $f(s)=-\pi\lambda s^2+2\pi\|\mathbf b\|_2s$ decreases for $s\ge r_\Theta$, and on the shell $\|\mathbf n\|_2\ge\|\mathbf n\|_\infty=q\ge r_\Theta$. So every term of the shell is bounded by $e^{f(q)}$, with $f$ taken as a whole; its two terms are not bounded separately, and the condition $q\ge r_\Theta$ is what makes the bound valid. The shell $\|\mathbf n\|_\infty=q$ has $N_g(q)=(2q+1)^g-(2q-1)^g$ points. If $N+1\ge r_\Theta$, the tail outside $\|\mathbf n\|_\infty\le N$ is bounded by

$$
\boxed{
B_g(N)
=
\exp\left(
\frac{\pi\|\mathbf b\|_2^2}{\lambda}
\right)
\sum_{q=N+1}^{\infty}
N_g(q)
\exp\left[
-\pi\lambda(q-r_\Theta)^2
\right].
}
$$

$B_g(N)\le\varepsilon_\Theta$ bounds only the series truncation. The errors of the period matrix, the Abel map, theta derivatives, the algebraic reduction, and floating point must be estimated separately.

## 1.14 Auxiliary route through a polynomial density

On compact $V$ each summand $\rho_k$ is real-analytic, so for a fixed degree $p$ one may fit $\rho_{k,p}(\mathbf y)=\sum_{|\boldsymbol\beta|\le p}c_{k,\boldsymbol\beta}(\mathbf y-\mathbf y_\ast)^{\boldsymbol\beta}$ (multi-index $\boldsymbol\beta$). This is a polynomial surrogate, not an identity for the Cauchy function. The potential, gradient, and Hessian of $\rho_{k,p}$ have term-by-term algebraic boundary recurrences on polyhedra, but that holds for $\rho_{k,p}$ only. Let $H^{(p)}_{ij,k}$ be the Hessian of the surrogate and $E^{(p)}_{ij,k}=H_{ij,k}-H^{(p)}_{ij,k}$. That is the kernel of Section 1.2 integrated against $\rho_k-\rho_{k,p}$. For an exterior point with $d_V=\operatorname{dist}(\mathbf x,V)>0$, $|3r_ir_j-R^2\delta_{ij}|\le2R^2$ gives a kernel bound of $2/d_V^3$. Hence, if $\sup_V|\rho_k-\rho_{k,p}|\le\varepsilon_{\rho,k,p}$,

$$
\boxed{
|E_{ij,k}^{(p)}(\mathbf x)|
\le
\frac{2G\,\operatorname{Vol}(V)}{d_V^3}
\varepsilon_{\rho,k,p}.
}
$$

For interior and boundary points and the finite-part Hessian, the local singular part must be separated first. Without $\sigma_k$, the centre locations, the box scale, and the norm, a degree-two or degree-three fit cannot promise $10^{-5}$ or $10^{-7}$. The rigorous form is $H=H^{(p)}+E^{(p)}$ together with a verifiable bound on $E^{(p)}$. The limit $p\to\infty$ needs convergence in a norm in which the gravity operator is continuous, and justification of the limit exchange. "Zero integration error" applies to the surrogate polynomial model only.

## 1.15 Self-check list and range of validity

Before a result is reported, verify:

1. the geometric input defines a finite, consistently oriented, non-self-intersecting polyhedral volume;
2. every ray segment is listed in $E_{\mathbf x}(\mathbf u)$, and $E_{\mathbf x}=E^{\mathrm{near}}_{\mathbf x}\mathbin{\dot\cup}E^{\mathrm{far}}_{\mathbf x}$;
3. the residual classes are disjoint, so no element is added by two channels;
4. every pointwise residual satisfies $F_q=F_q^{\mathrm{ext}}+F_q^{\mathrm{rem}}$; otherwise use complementary elements, not a formal subtraction;
5. the $\alpha=1$ finite part uses the limit of Section 1.6;
6. every Carlson element satisfies the square-free, branch, pole, and Hermite conditions;
7. Appell evaluates only the radial integral, and the outer spherical integral is still treated;
8. the residual is claimed only piecewise regular and integrable;
9. the error budget includes special-function truncation, inner and outer quadrature, endpoints, and rounding;
10. the polynomial route reports $E^{(p)}_{ij,k}$ and does not call the surrogate an identity.

What holds rigorously:

- RTFP has a complementary ray identity under the spherical finite-part convention;
- the Feynman parametrization is an exact moving-centre identity;
- at fixed parameter the spatial integral reduces to faces;
- an edge sub-integral enters the Carlson channel under the algebraic criteria;
- every remaining element goes to a mutually exclusive residual channel.

What does not hold in general:

- a global closed form in elementary and Carlson functions for an arbitrary Cauchy density, nonconvex polyhedron, or boundary point;
- an automatically zero residual;
- a strict absolute error from a fixed node count or a halving test;
- equality of a finite polynomial fit with the Cauchy model.

# 2. A unified integral representation of gravity

This chapter rewrites the kernel in the scale-parameter form used by the SCF iteration and derives the gradient, Hessian, and surface quantities from the volume integral. The two writings agree by factorization:

$$
\left[1+\sigma_k^2|y-c_k|^2\right]^{-\alpha_k}
=\sigma_k^{-2\alpha_k}\left[|y-c_k|^2+\sigma_k^{-2}\right]^{-\alpha_k}.
$$

Vectors are often written without boldface here; they are the objects of Chapter 1.

## 2.1 Three-dimensional volume integral and ray representation

With $\rho_k(y)=w_k\sigma_k^{-2\alpha_k}[|y-c_k|^2+a_k^2]^{-\alpha_k}$ and $a_k=\sigma_k^{-1}$,

$$
\nabla U(x)
=
-G\sum_{k=1}^K
w_k\sigma_k^{-2\alpha_k}
\int_V
\frac{x-y}{|x-y|^3}
\left[
|y-c_k|^2+a_k^2
\right]^{-\alpha_k}
dy ,
$$

where the sign comes from differentiating $|x-y|^{-1}$ in the field point. With $y=x-Ru$, the gradient kernel becomes $(x-y)|x-y|^{-3}dy=u\,dR\,d\Omega$, so

$$
\boxed{
\nabla U(x)
=
-G\sum_{k=1}^K
w_k\sigma_k^{-2\alpha_k}
\int_{S^2}
u
\sum_\ell
\int_{R_\ell^-(u)}^{R_\ell^+(u)}
\frac{dR}
{\left[(R-b_k(u))^2+d_k^2(u)\right]^{\alpha_k}}
\,d\Omega,
}
$$

with $p_k=x-c_k$, $b_k=p_k\cdot u$, and $d_k^2=a_k^2+|p_k|^2-b_k^2>0$. This is exactly the volume integral. The sum runs over every nonempty segment of Section 1.4. A segment starting at $R=0$ is harmless for the gradient (bounded integrand), but not for the Hessian.

## 2.2 Feynman parametrization

With $A(y)=|y-x|^2$ and $B_k(y)=|y-c_k|^2+a_k^2$, the $k$-th potential is $U_k=Gw_k\sigma_k^{-2\alpha_k}\int_VA^{-1/2}B_k^{-\alpha_k}dy$. By Section 1.7, with $p_k=\alpha_k+\tfrac12$ and $C_{\alpha_k}$ of Section 1.8,

$$
U_k(x)
=
G w_k\sigma_k^{-2\alpha_k}C_{\alpha_k}
\int_0^1
u^{-1/2}(1-u)^{\alpha_k-1}
S_{p_k}(x,u)\,du,
\qquad
S_{p_k}(x,u)=\int_VQ_k^{-p_k}\,dy,\qquad
Q_k=uA(y)+(1-u)B_k(y).
$$

## 2.3 Moving centre and effective screening scale

Completing the square as in Section 1.7,

$$
\boxed{
z_k(x,u)=ux+(1-u)c_k,
}\qquad
\boxed{
Q_k(x,y,u)
=
|y-z_k(x,u)|^2+m_k^2(x,u),
}\qquad
\boxed{
m_k^2(x,u)
=
(1-u)
\left[
a_k^2+u|x-c_k|^2
\right].
}
$$

$m_k^2>0$ for $0\le u<1$ and $m_k^2\to0$ as $u\to1$: the parametrization moves the source-point singularity to the endpoint $u=1$.

## 2.4 Unified gradient formula

From $\nabla_xz_k=uI$ and $\nabla_xm_k^2=2u(1-u)(x-c_k)$,

$$
\nabla_xQ_k=-2u(y-z_k)+2u(1-u)(x-c_k)=2u(x-y),
$$

since $-(y-z_k)+(1-u)(x-c_k)=x-y$. Hence $\nabla_xQ_k^{-p_k}=-2p_ku(x-y)Q_k^{-p_k-1}$, and

$$
\boxed{
\nabla U(x)
=
-2G
\sum_{k=1}^K
w_k\sigma_k^{-2\alpha_k}
C_{\alpha_k}p_k
\int_0^1
u^{1/2}(1-u)^{\alpha_k-1}
\int_V
(x-y)
\left[
|y-z_k|^2+m_k^2
\right]^{-p_k-1}
dy\,du,
}
$$

where $u^{1/2}$ is the Feynman weight $u^{-1/2}$ times the factor $u$ of $\nabla_xQ_k$. Differentiation and integration may be exchanged for exterior points. Interior and boundary points need a local singular analysis first.

## 2.5 Unified Hessian formula

From $\partial_j[(x_i-y_i)Q_k^{-p_k-1}]=\delta_{ij}Q_k^{-p_k-1}-2u(p_k+1)(x_i-y_i)(x_j-y_j)Q_k^{-p_k-2}$ and the gradient prefactor,

$$
\boxed{
H(x)
=
G\sum_{k=1}^K
w_k\sigma_k^{-2\alpha_k}
C_{\alpha_k}
\int_0^1
u^{-1/2}(1-u)^{\alpha_k-1}
\int_V
\Big[
4p_k(p_k+1)u^2
(x-y)(x-y)^T
Q_k^{-p_k-2}
-
2p_kuI\,Q_k^{-p_k-1}
\Big]dy\,du .
}
$$

For an exterior point this is a strict representation of the classical Hessian.

## 2.6 Consistency with the Newtonian Hessian

For $x\notin\overline V$ the potential can be differentiated directly:

$$
H_{ij}(x)
=
G\sum_{k=1}^K
w_k\sigma_k^{-2\alpha_k}
\int_V
\left[
\frac{3(x_i-y_i)(x_j-y_j)}{|x-y|^5}
-
\frac{\delta_{ij}}{|x-y|^3}
\right]
B_k(y)^{-\alpha_k}\,dy .
$$

The Feynman formula only rewrites $|x-y|^{-1}B_k^{-\alpha_k}$ pointwise as a parameter integral. With a positive distance bound, Fubini and dominated convergence allow both exchanges, so

$$
\boxed{
H_{\mathrm{Feynman}}(x)=H_{\mathrm{Newton}}(x)
\qquad(x\notin\overline V).
}
$$

---

## 2.7 Finite parts at interior points and at surface points

For $x\in V$ or $x\in\partial V$, the limit $u\to1$ forces $z_k\to x$ and $m_k\to0$, so the Feynman Hessian is not an absolutely convergent integral. The reference is the spherical finite part $H^{\mathrm{FP}}_{ij,\ell_0}$ of Section 1.3. The Feynman value may be said to compute it only after proving that $\lim_{\delta\downarrow0}[H^{(\delta)}_{\mathrm{Feynman}}-\text{the same logarithmic counterterm}]=H^{\mathrm{FP}}_{\ell_0}$, where $H^{(\delta)}_{\mathrm{Feynman}}$ is the parameter integral cut off at $u\le1-\delta$.

**Lemma 2.7.0 (The Feynman cut-off generates the contact term itself).** For every $\delta>0$, $H^{(\delta)}_{\mathrm{Feynman}}=\nabla^2U^\delta$ exactly, where $U^\delta$ is the potential of the truncated kernel $K^\delta=C_\alpha\int_0^{1-\delta}u^{-1/2}(1-u)^{\alpha-1}[uA+(1-u)B]^{-p}\,du$. Its integrand is smooth because $m^2\ge\delta a^2>0$. The missing part $K-K^\delta$ is concentrated at scale $m\sim\delta^{1/2}$ around $\mathbf y=\mathbf x$. After the rescaling of Proposition 2.7.1, and up to the centre shift $\eta=O(\delta^{1/2})$ controlled there, it is a radial $\kappa(|\boldsymbol\xi|)$ with $\kappa\sim\xi^{-1}$ at $0$ and rapid decay. For a cone $\Gamma$ with $C_{ij}=0$ and solid-angle fraction $\vartheta$, the spherical convention gives

$$
\int_\Gamma\partial_i\partial_j\kappa\,d\boldsymbol\xi
=\underbrace{\frac{4\pi\vartheta}{3}\delta_{ij}\Big[\xi^2\kappa'(\xi)\Big]_{0}^{\infty}}_{=+\frac{4\pi\vartheta}{3}\delta_{ij}}
\;\underbrace{-\;\frac{4\pi\vartheta}{3}\delta_{ij}}_{\text{contact}}
=0 .
$$

The left side is the spherical-convention value: the classical integral over $\Gamma\setminus\{\mathbf 0\}$ plus the contact term. For the classical part, $\partial_i\partial_j\kappa=\kappa''\hat\xi_i\hat\xi_j+\kappa'\xi^{-1}(\delta_{ij}-\hat\xi_i\hat\xi_j)$ and $\int_\Gamma\hat\xi_i\hat\xi_j\,d\Omega=\tfrac{4\pi\vartheta}{3}\delta_{ij}$ (that is, $C_{ij}=0$) give the angular integral $\tfrac{4\pi\vartheta}{3}\delta_{ij}(\kappa''+2\kappa'/\xi)=\tfrac{4\pi\vartheta}{3}\delta_{ij}\,\xi^{-2}(\xi^2\kappa')'$. The radial integral against $\xi^2d\xi$ is therefore $[\xi^2\kappa']_0^\infty$. Since $\kappa$ decays rapidly, $\xi^2\kappa'\to0$ at $\infty$, and $\xi^2\kappa'\to-1$ at $0$, so the bracket is $+1$. (A pure $\xi^{-1}$ without decay would give $\xi^2\kappa'\equiv-1$ and a zero bracket; the decay of $K-K^\delta$ is what makes it $+1$.) The contact term is the spherical-convention value of the $\xi^{-1}$ singularity on the cone, $-\tfrac{4\pi\vartheta}{3}\delta_{ij}$ (Section 1.3). So the missing part contributes nothing, and **the $\delta$-limit of the Feynman Hessian already contains the contact term**. It equals $\mathrm{p.v.}-\tfrac{4\pi G}{3}\rho_K\mathbf I$ at an interior point and $H^{\mathrm{FP}}$ where $C_{ij}=0$. The contact term must **never be added again** to a Feynman value. Where $C_{ij}\ne0$ (edges, vertices), the same computation leaves $C_{ij}\int\xi^{-1}d\xi$, the logarithm of Section 1.3, and that logarithm alone is the counterterm.

*Example.* Unit ball, one kernel with $\mathbf c=\mathbf 0$, $\sigma=\alpha=G=1$. (The ball is not a polyhedron, but its tangent cone at every surface point is a half-space, which is all the face-centre statements below use.) At $\mathbf x=(0.3,0,0)$ the truncated Feynman trace is $-11.15270,-11.52494,-11.52874,-11.52878$ for $\delta=10^{-2},10^{-4},10^{-6},10^{-8}$, against $-4\pi\rho(\mathbf x)=-11.52878$. At $(1,0,0)$, $H_{xx}\to2.245$ numerically. The exact value is the average $2.25194$ of the exterior value $2M=5.39353$ and the interior value $2M-4\pi\rho(1)=-0.88965$, where $M=4\pi-\pi^2=2.69677$ is the ball's mass. The $0.3\%$ gap is the quadrature error of the check.

$$
\boxed{
\begin{array}{ll}
x\notin\overline V:
&
H_{\mathrm{Feynman}}=H_{\mathrm{RTFP}}=H_{\mathrm{classical}};
\\[4pt]
x\in\operatorname{int}V:
&
\lim_{\delta\downarrow0}H^{(\delta)}_{\mathrm{Feynman}}=\mathrm{p.v.}-\tfrac{4\pi G}{3}\rho_K\mathbf I\ \text{(contact term already included; Lemma 2.7.0)};
\\[4pt]
x\in\partial V,\ C_{ij}(x)=0:
&
\lim_{\delta\downarrow0}H^{(\delta)}_{\mathrm{Feynman}}=H^{\mathrm{FP}}\ \text{(no counterterm, no added contact term)};
\\[4pt]
x\text{ on an edge or at a vertex}:
&
H_{\ell_0}^{\mathrm{FP}}\text{ depends on the local tangent cone and on }\ell_0 .
\end{array}
}
$$

For interior and boundary points, the primary path is the RTFP finite part of Section 1.3. The Feynman Hessian may replace it only after matching the same counterterm and the same $\ell_0$. At a triangular face centre the match is unconditional: Proposition 2.7.1 proves that the counterterm vanishes.

### Proposition 2.7.1 (Absence of the RTFP singularity at a triangular face centre)

Let $\Delta_f\subset\partial V$ be a nondegenerate triangle with outward normal $n$ and centroid $x_\star=\tfrac13(v_0+v_1+v_2)$, and let $\rho_K$ be as in Section 1.1. Then:

1. there is $r_0>0$ such that $V\cap B_{r_0}(x_\star)$ is the half-ball cut by the plane of $\Delta_f$;
2. $C_{ij}(x_\star)=0$ and $\vartheta_{x_\star}=\tfrac12$;
3. $J_{ij,\varepsilon}(x_\star)$ converges to a finite limit as $\varepsilon\downarrow0$, and

   $$
   H_{ij,\ell_0}^{\mathrm{FP}}(x_\star)
   =
   \lim_{\varepsilon\downarrow 0}J_{ij,\varepsilon}(x_\star)
   -
   \frac{2\pi G}{3}\rho_K(x_\star)\delta_{ij}
   $$

   is independent of $\ell_0$;
4. with $H^{\mathrm{F},\delta}_{ij}$ the Section 2.5 integral restricted to $u\in[0,1-\delta]$, $\lim_{\delta\downarrow0}H^{\mathrm{F},\delta}_{ij}(x_\star)$ exists and equals $H^{\mathrm{FP}}_{ij,\ell_0}(x_\star)$. The hemispherical contact term is already in the limit (Lemma 2.7.0), and no Hadamard subtraction is needed;
5. the one-sided classical limits exist and satisfy

   $$
   H^+=H^{\mathrm{FP}}+2\pi G\rho_K(x_\star)\,nn^{\!\top},
   \qquad
   H^-=H^{\mathrm{FP}}-2\pi G\rho_K(x_\star)\,nn^{\!\top},
   $$

   so their average is $H^{\mathrm{FP}}$, their jump is $H^+-H^-=4\pi G\rho_K\,nn^{\!\top}$, and $\operatorname{tr}H^--\operatorname{tr}H^+=-4\pi G\rho_K$.

Only the fact that $x_\star$ lies in the relative interior of $\Delta_f$ is used; the centroid is what Section 10.2 evaluates. Every nondegenerate triangle contains its centroid there, since the barycentric coordinates $(\tfrac13,\tfrac13,\tfrac13)$ are positive. Edge points and vertices are excluded: their cone is not a half-space, $C_{ij}$ need not vanish, and $H^{\mathrm{FP}}_{\ell_0}$ keeps the scale term.

*Proof of (1).* The centroid is at positive distance $d_f$ from $\partial\Delta_f$. Take $r_0$ below both $d_f$ and the cone radius of Section 1.3. The tangent cone is then the half-space $\{(y-x_\star)\cdot n\le0\}$.

*Proof of (2).* The inward directions form the hemisphere $\Gamma=\{u\cdot n\ge0\}$. With $n=e_3$, its area is $2\pi$, so $\vartheta=\tfrac12$, and $\int_\Gamma u_i^2\,d\Omega=2\pi/3$ for $i=1,2,3$. All off-diagonal moments vanish by the $\varphi$-integrals. Hence $\int_\Gamma(3u_iu_j-\delta_{ij})\,d\Omega=0$, and by Section 1.3 the finite part is scale-free once it exists.

*Proof of (3).* In the splitting of Section 1.5 the counterterm carries $C_{ij}=0$. The near difference is bounded by $\|\nabla\rho_K\|_{L^\infty(B_{r_0})}$, and the far field is absolutely integrable. So $\lim J_{ij,\varepsilon}$ exists as an ordinary limit, not a Hadamard part. The contact coefficient is $\tfrac{4\pi G}{3}\vartheta\rho_K=\tfrac{2\pi G}{3}\rho_K$.

*Proof of (4).* The ray factor $\int dR/R$ of $I_H$ looks singular at $R=0$. The Feynman form shows that this singularity is removable at a face centre. Fix one kernel. For $u<1$, $m^2\ge(1-u)a^2>0$, so the integrand is continuous, and only $\delta\downarrow0$ is at issue. Split $V$ into $V_{\rm near}=V\cap B_{r_0}(x_\star)$ and $V_{\rm far}$.

*Far field.* With $\delta_1=r_0/(2(1+|x_\star-c|))$ and $1-u\le\delta_1$, one has $|z(u)-x_\star|\le r_0/2$, so $Q_u\ge r_0^2/4$ on $V_{\rm far}$. The weight $u^{-1/2}(1-u)^{\alpha-1}$ is in $L^1(0,1)$ for $\alpha>0$, so the far part converges absolutely.

*Near part: no density split.* On $V_{\rm near}$ the Feynman integrand is used whole. $Q_u=|y-z(u)|^2+m^2(u)$ already contains the Cauchy factor through $z$ and $m$ (Section 1.7), so the rescaling below treats the full integrand, and the density enters only through $\mu$ and $\eta$. The split $\rho_K=\rho_K(x_\star)+(\rho_K-\rho_K(x_\star))$ belongs to the ray side, part (3); the Feynman representation acts on the whole product $A^{-1/2}B^{-\alpha}$ and has no separate transform of the two pieces.

*Rescaling.* Here the hemispherical cancellation enters. With $s=1-u$, $x_\star$ as origin, $m^2=s\mu^2$, and $\mu^2=a^2+(1-s)|c|^2\ge a^2$, the centre is displaced by $\eta_{\rm phys}=s(c-x_\star)=O(s)$. Substitute $y=m\xi$ ($dy=m^3d\xi$). The half-ball becomes $\{\xi\cdot n\le0,|\xi|<R_m\}$ with $R_m=r_0/m\to\infty$, and $Q=m^2(|\xi-\eta|^2+1)$ with $\eta=\eta_{\rm phys}/m=O(s^{1/2})$. The kernel becomes

$$
\int_{H}L_{ij}(y;u)\,dy
=
m^{-2\alpha}
\int_{\substack{\xi\cdot n\le 0\\|\xi|<R_m}}
\Lambda_{ij}(\xi;\eta,u)\,d\xi,
\qquad
\Lambda_{ij}
=
-2p\,u\,\delta_{ij}\bigl(|\xi-\eta|^2+1\bigr)^{-p-1}
+
4p(p+1)u^2\,\xi_i\xi_j\bigl(|\xi-\eta|^2+1\bigr)^{-p-2},
$$

with exponent $3-(2p+2)=-2\alpha$. As $s\downarrow0$, $\Lambda_{ij}\to\Lambda^0_{ij}=\partial_i\partial_j(|\xi|^2+1)^{-p}$. By the divergence theorem on $\Pi_-\cap B_R$, the disk term vanishes: $\partial_nf=0$ on $\xi\cdot n=0$, and the tangential derivatives are odd. The cap flux is $O(R^{1-2p})=O(R^{-2\alpha})$. Hence $\int_{\Pi_-}\Lambda^0=0$ and the finite half-ball integral is $O(R_m^{-2\alpha})=O(m^{2\alpha})$, which the prefactor $m^{-2\alpha}$ makes $O(1)$. This is the Feynman form of $C_{ij}=0$. For a general cone $\Gamma$ the same radial computation gives $\int_\Gamma\Lambda^0\,d\xi=\kappa_pC_{ij}$ with $\kappa_p=2p\int_0^\infty r^2(1+r^2)^{-p-1}dr>0$. A nonzero $C_{ij}$ would leave $m^{-2\alpha}\kappa_pC_{ij}\propto s^{-\alpha}$ in the integrand, hence $\int ds/s$ against the weight $s^{\alpha-1}$: the logarithm of Section 1.3.

*Remainders.* $\Lambda$ differs from $\Lambda^0$ through the shift $\eta$ and through the factors $u,u^2$. The $\eta$-derivatives of both terms of $\Lambda$ are $O(r^{-2\alpha-4})$ at large $r$, integrable against $r^2dr$ for every $\alpha>0$, so by the mean-value theorem the shift changes the half-space integral by at most $C|\eta|=O(s^{1/2})$. Replacing $u,u^2$ by $1$ changes the integrand by $O(s)$ times absolutely integrable functions. The same dominants bound the half-ball/half-space difference by $O(R_m^{-2\alpha})$. With the prefactor, the near part is $O(1)+O(s^{1/2-\alpha})+O(s^{1-\alpha})$.

*Parameter integral.* Against the weight $s^{\alpha-1}$ the three pieces give $O(s^{\alpha-1})$, $O(s^{-1/2})$, and $O(1)$, all integrable at $s=0$. The near part therefore converges as an ordinary improper integral, with no counterterm.

*Identification.* Far field and near part together show that $\lim_\delta H^{\mathrm F,\delta}(x_\star)$ exists. By Lemma 2.7.0, $H^{\mathrm F,\delta}=\nabla^2U^\delta$, and the Hessian of $U-U^\delta$ over the half-space cone tends to zero: its principal-value part $+\tfrac{2\pi}{3}\delta_{ij}$ and contact part $-\tfrac{2\pi}{3}\delta_{ij}$ (per unit $G\rho_K$) cancel. The limit is therefore the finite part of (3), with the contact term generated by the cut-off itself. The ray singularity $\int_0dR/R$ is absent at the face centre: its angular coefficient is zero.

*Proof of (5).* The local model is the constant-density **half-ball** $V_{\rm loc}=B_{r_0}(x_\star)\cap\Pi_-$, not the infinite half-space. The often-quoted half-space value $-2\pi G\rho_Knn^{\!\top}$ assumes slab-wise summation at infinity, which is not the spherical convention. Every inward ray of the half-ball has length $r_0$, so $J_{ij,\varepsilon}=G\rho_KC_{ij}\log(r_0/\varepsilon)=0$ and $H^{\mathrm{FP}}_{\rm loc}=-\tfrac{2\pi G}{3}\rho_K\mathbf I$, with trace $-4\pi G\vartheta\rho_K$. The classical jump relations for a density jump across a smooth surface are: continuity of $U$, of $\nabla U$, and of all tangential second derivatives, and $\partial_{nn}U^--\partial_{nn}U^+=-4\pi G\rho_K$. They give $H^+_{\rm loc}-H^-_{\rm loc}=4\pi G\rho_Knn^{\!\top}$. By Lemma 2.7.0, $H^{\mathrm{FP}}_{\rm loc}$ is the limit of Hessians of asymptotically radial mollifications. A symmetric mollification of a function whose Hessian jumps across a plane converges on the plane to the average. Hence

$$
H^\pm_{\mathrm{loc}}=H^{\mathrm{FP}}_{\mathrm{loc}}\pm 2\pi G\rho_K\,nn^{\!\top}
=-\frac{2\pi G}{3}\rho_K\mathbf I\pm 2\pi G\rho_K\,nn^{\!\top},
$$

with traces $0$ outside and $-4\pi G\rho_K$ inside.

*Check on a uniform ball of radius $a$.* At a surface point, $H^+=\tfrac{4\pi G\rho}{3}(3nn^{\!\top}-\mathbf I)$ and $H^-=-\tfrac{4\pi G\rho}{3}\mathbf I$. With chord length $2a\cos\theta$, the ray formula gives $\lim J_{nn}=2\pi G\rho\int_0^1(3\mu^2-1)\log\mu\,d\mu=\tfrac{4\pi G\rho}{3}$, so $H^{\mathrm{FP}}=\tfrac{2\pi G\rho}{3}(3nn^{\!\top}-\mathbf I)-\tfrac{2\pi G\rho}{3}\mathbf I$, which is the average.

For the actual polyhedron, the contribution from outside $B_{r_0}$ and the density fluctuation are continuous across the face. They add the same tensor to $H^\pm$ and $H^{\mathrm{FP}}$, so (5) survives. ∎

The local contact tensor $-\tfrac{2\pi G}{3}\rho_K\mathbf I\pm2\pi G\rho_Knn^{\!\top}$ is not a substitute for $H^\pm$. It omits the nonlocal integral over $V\setminus B_{r_0}$, which is finite at a face centre exactly because the logarithm vanishes, and whose deviatoric part belongs to the tidal field.

*Edge contrast.* On an edge the cone is a wedge with a lune section. Its moment is generally nonzero, the ray integral diverges as $C_{ij}\log(\ell_0/\varepsilon)$, and the Feynman weight does not cancel it. Face centres are exactly the boundary points where the coefficient provably vanishes. So every Hessian that enters $T_{\rm tide}$ at a face centre is a finite, scale-free RTFP value, and Section 10.2 excludes vertices and edges.

---

## 2.8 Granular balance equations

Three relations accompany granular balance, with different status:

$$
\begin{cases}
\Delta W
=
2\omega^2-4\pi G\rho_K\mathbf1_V,
\\[4pt]
|\nabla W\times n|^2
\le
\sin^2\phi(x)\,|\nabla W|^2,
\\[4pt]
\displaystyle\int_V\rho_K\,dx=M.
\end{cases}
$$

The first is an identity satisfied by $W=U+U_c$ inside any body; it determines nothing about the shape and serves as a check of the field. The second holds only on the loose faces of the slope set $\mathcal S_{\rm slope}$ (Section 2.8c); locked bedrock may be steeper than repose. The third is the mass constraint of the geometric step (Section 2.13). The system the loop solves is the quasi-static Stokes problem (B1) with the yield classification, followed by the cone program of Section 10.3.3.

The Poisson equation is an interior statement. There $\vartheta_x=1$, so $\operatorname{Tr}\nabla^2U=-4\pi G\rho_K$, and $U_c=\tfrac12\omega^2(x_1^2+x_2^2)$ has Laplacian $2\omega^2$. On the boundary $\vartheta_x\ne1$. With $W=U+U_c$,

$$
\nabla W=\nabla U+\omega^2(x_1,x_2,0),\qquad
\nabla^2W=\nabla^2U+\omega^2\operatorname{diag}(1,1,0),
$$

and at a surface node $g_i^{(n)}=\nabla U^{(n)}(x_i^{(n)})+\omega^2(x_{i1}^{(n)},x_{i2}^{(n)},0)$. The local slope constraint is

$$
\boxed{
|g_i^{(n)}\times n_i^{(n)}|^2
\le
\sin^2\phi_i^{(n)}\,|g_i^{(n)}|^2.
}
$$

For a unit normal and $|g_i|>0$ it reads $\sin^2\theta_i\le\sin^2\phi_i$, where $\theta_i$ is the angle between $-g_i$ and the outward normal, i.e. the slope relative to the local effective horizontal [6]. At $|g_i|=0$ the normalized form is undefined. Use the un-normalized contact or yield constraint, or mark the node degenerate; never divide by $|g_i|^2$.

**Mode B: quasi-static viscoplastic Stokes creep and yield classification (a single stress field).** The only stress is:

$$
\boxed{\ \boldsymbol\sigma=-p\,\mathbf I+\boldsymbol\tau,\qquad
\boldsymbol\tau=\mathbf T_{\rm tide}+\boldsymbol\tau_{\rm flow},\qquad
\boldsymbol\tau_{\rm flow}=2\eta(\tilde p,\mathbf D)\,\mathbf D,\qquad
\eta=\frac{c(\mathbf x)+\mu_d\,\tilde p}{\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}},\ }
$$

with $\dot\gamma=\sqrt{2\mathbf D:\mathbf D}$ and the unique regularized pressure $\tilde p:=\max([p]_+,p_{\min})$. By the rule of Section 4.13, the stress magnitude, the yield strength, and the denominator of $I_c$ all use this $\tilde p$, with $I_c$ at $p_*=\tilde p+p_c$; $\mu_d=\mu_d(I_c;\nu_0,Z_0)$ is given in Sections 4.5 and 10.2. Under A-Ξ (Section 11.3), $\mu_d=\mu_d(I_c,\nu)$, and $\mathbf D$ is replaced by its deviator $\mathbf D'$ in $\boldsymbol\tau_{\rm flow}$ and $\dot\gamma$; without A-Ξ, $\mathbf D'=\mathbf D$. Under A-ϖ, $c=(1-\mathcal R\varpi)c_{\rm intact}$ (Section 11.5). The Papanastasiou alternative $\eta=(c+\mu_d\tilde p)(1-e^{-\dot\gamma/\dot\gamma_\varepsilon})/\dot\gamma$ replaces $\eta$ and its derivatives in Sections 4.17, 8.3, and 10.3.2 together; the two are never mixed. At every discrete state, with a configured bound $\dot\gamma\le\dot\gamma_{\max}$ and $\mu_s\le\mu_d\le\mu_2$,

$$
0<\eta_{\rm lo}
:=\frac{c+\mu_s p_{\min}}{\sqrt{\dot\gamma_{\max}^2+\dot\gamma_\varepsilon^2}}
\le\eta\le
\eta_{\rm hi}
:=\frac{c+\mu_2\tilde p_{\max}}{\dot\gamma_\varepsilon},
$$

so the viscous stiffness is uniformly positive definite off the rigid-body constraints (Lemma 8.12). Under A-Ξ and A-ϖ the same bounds hold with $\mu_{\min}$ in place of $\mu_s$, $c\ge0$, and the upper friction of Proposition 11.6 (Propositions 11.6 and 11.9). **There is no** elastic strain, elastic modulus, hardening variable, return-mapping stress, or displacement unknown. The state variables $\nu$ and $\varpi$ of Chapter 11 are rate-driven internal variables of the viscoplastic law, frozen within a step, not hardening variables of an elastoplastic return map. Every stress that is needed — balance, yield, tide, traction, erosion, acceptance — is this one.

**(B1) Quasi-static Stokes balance.** At step $n$, on the frozen geometry $\mathcal V^{(n)}$, with $\rho_K(\cdot;\theta^{(n)})$ frozen (transported between steps, Section 9.2) and $\omega_n=\omega(t_n)$ ($\omega_n=L_n/I_1^{(n)}$ under A-L), solve

$$
\boxed{\
\begin{aligned}
&\nabla\cdot\boldsymbol\tau-\nabla p+\rho_K\,\mathbf g_{\rm eff}(\omega_n)=\mathbf 0,\qquad \nabla\cdot\mathbf v=0\quad\text{in }\mathcal V^{(n)},\\
&\boldsymbol\sigma\mathbf n=\mathbf t_{\rm ext}\quad\text{on }\Gamma^{(n)},\qquad
\int_{\mathcal V}\rho_K\mathbf v\,dV=\mathbf 0,\quad \int_{\mathcal V}\rho_K\,\mathbf x\times\mathbf v\,dV=\mathbf 0,
\end{aligned}\ }
$$

with $\mathbf g_{\rm eff}(\omega)=\nabla U+\omega^2\mathbf x_\perp$. Under A-Ξ the constraint is $\nabla\cdot\mathbf v=\dot\varepsilon_v$ of Section 11.3, with $p$ its multiplier, and the velocity is split as in Section 11.4. Inertia, convection, and Coriolis are omitted (the quasi-static limit). The indicator $\mathrm{Ek}^{-1}:=\max_{\mathcal Y}\rho_K\omega_nR_{\max}^2/\eta$ on the yielded region $\mathcal Y$ is reported. Above the threshold $\varepsilon_{\rm in}$ the step is marked **quasi-static assumption failed**. The geological step has no physical time step inside the creep solve. $\mathbf v$ is obtained as the stationary limit of the pseudo-time hyperbolic ray relaxation (8.3-H) and Section 10.3.2. The pseudo-time step obeys the characteristic CFL bound; it is a numerical solver parameter and is not added to geological time. The Euler force $-\dot\omega\mathbf e_z\times\mathbf x$ and the torque that drives $\dot\omega$ balance in the body frame. They are omitted together, and $\varepsilon_{\rm Euler}:=|\dot\omega|/\omega^2$ is reported (under A-L with the slow part of the change, Section 11.7). Omitting only one would violate the torque compatibility of Appendix A.3.

**(B2) Yield function and regularized yield criterion.** On the same stress,

$$
\tau_y:=c(\mathbf x)+\mu_y\,\tilde p,\qquad \mu_y:\equiv\mu_s(\nu_0,Z_0)=\lim_{I_c\to0}\mu_d,\qquad
F:=\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}-\tau_y=\|\mathbf s-\mathbf T_{\rm tide}\|_{\rm eq}-c-\mu_y\tilde p,
$$

with $\|\mathbf A\|_{\rm eq}=\sqrt{\mathbf A:\mathbf A/2}$ and deviatoric stress $\mathbf s=\boldsymbol\tau$. Both $\boldsymbol\tau_{\rm flow}$ and $\mathbf T_{\rm tide}$ are traceless: the first because it is $2\eta\mathbf D'$ ($\mathbf D'=\mathbf D$ under incompressibility), the second by construction. Under A-Ξ, $\mu_y=\mu_y(\nu)=\mu_s+K_\Xi(\nu-\nu_c^0)$ (Section 11.3), the same limit $\lim_{I_c\to0}\mu_d$. Using $\mu_s$ makes the unyielded viscous law and the yield function share one strength. Since $\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}=\eta\dot\gamma=(c+\mu_d\tilde p)\,r(\dot\gamma/\dot\gamma_\varepsilon)$ with $r(x)=x/\sqrt{1+x^2}$ (Bingham) or $1-e^{-x}$ (Papanastasiou), and $0\le r<1$, the regularized stress never reaches the strength when $\mu_d\to\mu_s$. The criterion therefore carries a tolerance $\delta_Y\in(0,1)$, default $0.05$:

$$
\boxed{\ F_\delta:=\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}-(1-\delta_Y)\,\tau_y,\qquad
\text{the point }\mathbf x\text{ has yielded}\iff F_\delta(\mathbf x)>0 .\ }
$$

With $\mu_d=\mu_s$, $F_\delta>0$ iff $\dot\gamma>\dot\gamma_Y$, where $\dot\gamma_Y=\dot\gamma_\varepsilon(1-\delta_Y)/\sqrt{2\delta_Y-\delta_Y^2}$ (Bingham, from $x/\sqrt{1+x^2}>1-\delta_Y$) or $\dot\gamma_\varepsilon\ln(1/\delta_Y)$ (Papanastasiou). Both are about $3\dot\gamma_\varepsilon$ at $\delta_Y=0.05$. This is a criterion on the stress, equivalently on a strain-rate invariant, not on the velocity magnitude: a rigid motion ($\mathbf D=\mathbf 0$) has $F_\delta=-(1-\delta_Y)\tau_y<0$ however fast it moves. A point has yielded when equilibrium needs at least the fraction $1-\delta_Y$ of the local strength. $\dot\gamma_\varepsilon$ only sets the scale of residual creep in the unyielded region, and the yielded set depends on it only through the convergence of the regularized solution as $\dot\gamma_\varepsilon\to0$ (PO-12). Spin and loads are frozen within a step, so no geological strain rate enters the test.

*Why (B2) cannot decide shallow failure.* Three properties of the model bias (B2) against failing a thin surface layer.

1. *Resolution.* A velocity basis of support scale $h_{\rm vol}$ and a quadrature of $N_q$ volume points resolve stresses only on wavelengths of order $h_{\rm vol}$ and on the scale of the point spacing. A failure surface at depth $h_s\ll h_{\rm vol}$ under one mesh face is below that resolution. A resolved solve would fail such a layer when it is steep enough: on an infinite slope of inclination $\theta$ the shear-to-normal ratio on parallel planes is $\tan\theta$.
2. *Stress state of the creep model.* For uniform density $\rho_K\nabla W$ is a gradient carried by the pressure. Deviatoric stress then comes only from the departure of $\Gamma$ from an equipotential and, for variable density, from the baroclinic source of Section 6.2. The incompressible creep state is the least-deviatoric state compatible with the geometry. Elastic–plastic analyses of a compressible body ([10] for Ryugu) carry more, so (B2) on the creep stress is biased toward no yield. A-Ξ (Section 11.3) does not remove this bias. Its volume change is driven by plastic shear in material that has already yielded, not by elastic compressibility.
3. *Strength.* With published values for Ryugu, $\rho_K|\mathbf g|\approx0.18\ \mathrm{Pa\,m^{-1}}$ and an interior cohesion $c_b\approx4$ Pa [10]. A failure plane parallel to a $45^\circ$ slope with $\mu_y=0.70$ needs a normal depth of about $c_b/(\rho_K|\mathbf g|(\sin45^\circ-\mu_y\cos45^\circ))\approx100$ m. The absence of interior yield on such a body is therefore not by itself a defect of the model; causes 1 and 2 only decide that a shallow layer cannot fail in the truncated solve.

The hyperbolic ray relaxation, at its stationary residual tolerance, decides deep-seated and whole-body failure. Shallow failure of the loose layer is the subgrid closure A-C4 (Section 2.8c).

**(B3) Rigid core and movable set.** Evaluate $F_\delta$ on the classification mesh of Section 8.1 (G7),

$$
\mathcal X_Y=\mathcal X_{\rm vol}\cup\{\mathbf X_i-\xi_q\,t_i^-\mathbf N_i:\ q=1,\dots,N_r\}\cup\{\mathbf X_i\},
$$

the volume nodes, probes along the inward vertex normals, and the surface vertices. $\mathbf v$, $\mathbf S$, and $p$ are synthesized pointwise by the characteristic ray solver of Section 8.3. The graph $\mathsf G_Y$ joins successive probes on a normal, probes of equal $q$ on mesh-adjacent vertices, and each point to its nearest volume nodes along segments inside $\mathcal V$. On the unyielded set $\mathcal U=\{F_\delta\le0\}$, the **rigid core** $\mathcal U_{\rm core}$ is the connected component of largest quadrature mass $\sum\rho_Kw$; ties go to the component with more volume nodes. $\mathcal U=\varnothing$ means global instability. The graph resolves a yielded band only if the band is at least one probe spacing thick. A thinner internal shear band leaves the unyielded set connected, and the block beyond it is classified with the core ($\chi_Y=0$). This is a resolution limit of the classification and part of PO-12. Shallow surface failure has its own closure (A-C4); a thin internal band has none. The movable indicator is

$$
\boxed{\ \chi_{Y,i}=\begin{cases}0,&\mathbf X_i\in\mathcal U_{\rm core}\quad\text{(solid connected to the rigid core; locked)},\\ 1,&\text{otherwise (the node itself has yielded, or it lies on a sliding block separated from the core by a yielded band)}.\end{cases}\ }
$$

*Star-shaped special case.* There the probes may be taken on the rays from $O$, $\mathcal X_Y=\{(\zeta_q,\hat{\boldsymbol\omega}_i)\}$ with radial Gauss–Jacobi nodes and $\zeta_{N_r+1}=1$, with all innermost points joined through a virtual centre node.

In the unregularized limit an unyielded component moves rigidly. A landslide block separated from the core by a yielded band slides as a whole, even if its surface points are themselves unyielded. That is why $F_\delta$ at the surface alone is not enough. The rigid motion of the core is the weighted least-squares fit

$$
(\mathbf a_c,\boldsymbol\theta_c)=\arg\min_{\mathbf a,\boldsymbol\theta}\sum_{\mathbf x\in\mathcal U_{\rm core}}w_{\mathbf x}\rho_K(\mathbf x)\,|\mathbf v(\mathbf x)-\mathbf a-\boldsymbol\theta\times\mathbf x|^2,
$$

a $6\times6$ SPD system unless the core is collinear. In that case the step is marked **core degeneracy** and $\mathbf a_c=\boldsymbol\theta_c=\mathbf 0$. Every geometric prediction uses the slip velocity

$$
\mathbf v_{\rm sl}:=\mathbf v-\mathbf a_c-\boldsymbol\theta_c\times\mathbf x .
$$

The core's own translation is absorbed by the realignment of Section 10.3.4, step 8. The residual creep inside the core comes from $\dot\gamma_\varepsilon>0$ and is removed exactly by $\chi_Y=0$. No state variable integrates $\mathbf D$ in time, so this creep accumulates nowhere.

**(B4) Surface predictor (the unique definition of $d_{\rm pred}$).** With $\rho_{\rm bulk}=\rho_p\nu_0$, the areal mass fluxes $J_{\rm dep},J_{\rm ero}$ (kg m⁻² s⁻¹), the thickness increment $\Delta h_C=\Delta t_n(\bar J_{\rm dep}-\bar J_{\rm ero})/\rho_{\rm bulk}$ with the step averages over $\tau_n=\lambda\Delta t_n$ of Section 2.8 (the update $\lambda d$ then lays down $\tau_n(\bar J_{\rm dep}-\bar J_{\rm ero})$, the same mass as the $\rho_s$ source), and its mass-equivalent

$$
\Delta h_C^*:=\frac{\rho_{\mathrm{bulk}}}{\rho_K|_\Gamma}\Delta h_C\qquad(\rho_{\rm bulk}\Delta h_C=\rho_K|_\Gamma\Delta h_C^*),
$$

the predicted normal increment of vertex $i$ is

$$
\boxed{\ d_{\mathrm{pred},i}^{(n)}=\chi_{Y,i}\,\Delta t_n\,\mathbf v_{{\rm sl},i}^{(n)}\!\cdot\mathbf N_i^{(n)}+\Delta h^*_{C,i}\ }\qquad(\mathbf X_i\leftarrow\mathbf X_i+d_i\mathbf N_i;\ \text{Section 8.1, G4}).
$$

The increment is normal, along the unit vertex normal $\mathbf N_i$ of (G4). Section 10.3.3 uses this same definition. There is no displacement field. The increment is the creep velocity relative to the core times the step, $\Delta t_n\mathbf v_{\rm sl}$.

The creep term carries $\chi_Y$. The node classes carry the geometric movability $\tilde\chi_i:=\max(\chi_{Y,i},\pi_i)$, where $\pi_i$ is the slope-promotion indicator of Section 2.8c ($\pi\equiv0$ without A-C4). A promoted node keeps $\chi_{Y,i}=0$, so its predictor has no creep. The classes are:

- **locked** $\mathcal L^{(n)}=\{\tilde\chi_i=0,\ \Delta h_{C,i}=0\}$, the solid substrate, with $d_i=0$;
- **pure deposition** $\mathcal D^{(n)}=\{\tilde\chi_i=0,\ \Delta h_{C,i}\ne0\}$, an immobile substrate receiving material, with $d_i=d_{{\rm pred},i}$ fixed;
- **movable** $\mathcal A^{(n)}=\{\tilde\chi_i=1\}$, the only free variables of the cone program: the yielded or sliding nodes plus the promoted ones.

Locking means the substrate does not flow. Deposition accumulates on it; locking the deposit too would break mass conservation. In the mass row $D\mathcal M^{(n)}d+s_M=M-\mathcal M^{(n)}$ every flux increment uses $\Delta h_C^*$. Otherwise $\rho_{\rm bulk}\ne\rho_K|_\Gamma$ creates a spurious drift $s_M\ne0$ even when $\int J_{\rm dep}=\int J_{\rm ero}$. Only the Mode C flux produces $\Delta h_C$, and no flow displacement is added twice. A solid free surface does not in general satisfy $W=\mathrm{const}$. The slope surrogate applies only at movable nodes of the loose layer where it has been declared applicable, with $g_{\rm eff}\cdot n<0$ checked separately; locked nodes do not receive it. Solid material without loose cover ($\rho_s=0$) may stand steeper than the repose angle, which is what retains real topography. A loose node on a supercritical face is promoted by A-C4 and relieved even though the creep classification locks it.

**(B5) Tensile truncation.** The hyperbolic pressure equation (8.3-H) has no pointwise inequality. Rheology and yield use $\tilde p=\max([p]_+,p_{\min})$, and the truncated part $\|[p]_-\|$ is reported each step as the **tensile fraction**. The truncated pressure does not satisfy the Poisson equation. In tension the strength degenerates to $\tau_y=c+\mu_yp_{\min}$ (Drucker–Prager tension cut-off); without cohesion such a region is ordinarily yielded.

**(B6) Plastic diagnostics (no feedback).** $\mathbf D_p:=\mathbf 1_{\{F_\delta>0\}}\mathbf D$, the dissipation $\Phi_p:=\int\mathbf 1_{\{F_\delta>0\}}\boldsymbol\tau_{\rm flow}:\mathbf D\,dV\ge0$ (since $\boldsymbol\tau_{\rm flow}:\mathbf D=2\eta|\mathbf D|^2$), and $\bar\varepsilon_p\leftarrow\bar\varepsilon_p+\lambda\Delta t_n\dot\gamma\mathbf 1_{\{F_\delta>0\}}$ on $\mathcal X_Y$ ($\lambda$ the under-relaxation factor of Section 2.14) are post-processing outputs only. They **do not enter** $\boldsymbol\sigma$, $\eta$, $F_\delta$, $d_{\rm pred}$, or any acceptance test. Under A-Ξ and A-ϖ the state variables $\nu$ and $\varpi$ are driven by the nonlocal form of the same plastic rate (Section 11.2), and they do feed back. $\bar\varepsilon_p$ itself stays a diagnostic.

**Mode C interface.** Let $\chi_{A,i}=\chi_{Y,i}\mathbf 1_{\{\rho_{s,i}>0\}}$ (movable and loose). On the moving surface $\Gamma(t)$, with surface measure $dS$, the loose layer obeys

$$
\partial^\bullet_t\rho_s+\rho_s\nabla_\Gamma\!\cdot\mathbf v_\Gamma=J_{\mathrm{dep}}-J_{\mathrm{ero}},\qquad
\mathbf v_\Gamma=\mathbf v_T+V_n\mathbf n,\qquad
\mathbf v_T=\chi_A\big(\mathbf v_{\rm sl}-(\mathbf v_{\rm sl}\cdot\mathbf n)\mathbf n\big)\big|_\Gamma,
$$

with $\partial^\bullet_t$ the time derivative following $\mathbf v_\Gamma$ and $V_n$ the normal speed of $\Gamma$ (Section 8.1, G5), and

$$
J_{\mathrm{ero}}=\chi_A\,k_e\rho_s\frac{[F_\delta]_+}{\sigma_0},\qquad
J_{\mathrm{dep}}(a)=\int_{\Gamma}K(a,a')J_{\mathrm{ero}}(a')\,dS(a'),\qquad
\int_{\Gamma}K(a,a')\,dS(a)=1,
$$

with $\rho_s$ the loose-layer areal density (initially $\rho_{\rm bulk}h_s$), $k_e\ge0$ in s⁻¹, $\sigma_0>0$, and $K\ge0$. In the reference model $K$ is the runout kernel of A-C3 (Section 2.8b); the isotropic kernel $G_w$ is its limit $X(a')=a'$. $F_\delta$ is the surface value of the unique stress, so a locked node has $J_{\rm ero}=0$. Integrating $J_{\rm dep}$ and using the normalization of $K$ gives the exact balance

$$
\int_{\Gamma}J_{\mathrm{dep}}\,dS
=\int_{\Gamma}J_{\mathrm{ero}}\,dS .
$$

An external source must be explicit, $J_{\rm ext}$, with $M^{n+1}=M^n+\Delta t_n\int_\Gamma J_{\rm ext}\,dS$, and never hidden in the deposit. The transport velocity is the active tangential slip $\mathbf v_T$ relative to the core; the unprojected 3-D velocity must not be used. $\rho_s$ is an accounting density: its tangential transport does not move the surface, which changes only through $J_{\rm dep}-J_{\rm ero}$ via $\Delta h_C^*$. Transport is the semi-Lagrangian scheme of Section 8.1 (G5) with the Jacobian correction. Positivity holds for every step. Let $\beta_i=\chi_{A,i}k_e[F_{\delta,i}]_+/\sigma_0$ be the frozen erosion rate, so $J_{\rm ero}=\beta\rho_s$, and $\tau_n=\lambda\Delta t_n$ the time the step advances (Section 10.3.1). $J_{\rm ero}$ is integrated exactly over $\tau_n$ through its step average $\bar J_{{\rm ero},i}=\rho_{s,i}(1-e^{-\beta_i\tau_n})/\tau_n$, and $\bar J_{\rm dep}$ is the routed and spread $\bar J_{\rm ero}$. The source update is $\rho_s\leftarrow\rho_s+\tau_n(\bar J_{\rm dep}-\bar J_{\rm ero})$, whose erosion part is exactly $\rho_s\mapsto\rho_se^{-\beta_i\tau_n}$, and $\Delta h^*_C$ is formed from the same averages. The average and the update must use the same interval: an average over $\Delta t_n$ applied for $\tau_n$ would give the factor $1-\lambda+\lambda e^{-\beta_i\Delta t_n}$, which is not the exponential for $\lambda<1$. The exponential is this update, not a second factor: applying both would remove the eroded mass twice. Forward Euler would need $\tau_n\beta_i<1$ even on an exhausted layer. Under hypothesis A-T1 (Section 10.3.1) the update is applied once per step, in the settling of step S (Section 10.3.4), with the $\tau_n$ of that step; for $\tau_n\gg1/\beta_i$ the layer of the eroding vertex is stripped and laid down where its packets come to rest. The remaining bound is $\Delta t_C$ of Section 10.3.1.

## 2.8b Rotating-frame runout of eroded regolith (hypothesis A-C3)

*Motivation.* Ryugu's equatorial ridge is attributed to a fast-spin phase (period near 3.5 h) in which the equator was the low of the rotating geopotential and loose material migrated to it from mid-latitudes [5]. The isotropic kernel returns eroded mass symmetrically about its source, and the tangential geopotential gradient and Coriolis act only through Mode B creep, so it cannot represent this. A-C3 lets each eroded parcel run out over $\Gamma$ in the rotating frame before the spread of Section 2.8.

**Hypothesis A-C3 (runout of eroded regolith).**

1. Material eroded at $a'$ in step $n$ leaves as a thin packet of thickness $h_s$ with broken bonds. Its residual cohesion is zero, and $p_c$ (Section 4.3) enters only its inertial number.
2. The packet moves over the frozen $\Gamma^{(n)}$ in the frozen $\mathbf g_{\rm eff}$ of step 1 (Section 10.3.4), under the Coriolis acceleration of $\boldsymbol\Omega=\omega_n\mathbf e_z$. The Euler force is omitted, as in (B1).
3. The runout is instantaneous geologically ($t_{\rm run}\ll\Delta t_n$), so $X(a')$ is evaluated on the frozen geometry.
4. Packets do not interact and do not change the surface.

**Eikonal–Fermat–GRIN reduction.** The following is the exact optical representation of the *reduced runout model* in which the packet speed is a prescribed positive scalar field on the frozen surface. Let $(\Gamma,g)$ be the piecewise-smooth surface with metric $g_{ab}$, and let $v_{\rm e}(x)>0$ be the frozen effective runout speed obtained from the local downslope acceleration, basal friction, layer depth, and pressure regularization. Define

$$
 n(x):=v_{\rm e}(x)^{-1}.
$$

For a launch point $a'$ the arrival-time function is the viscosity solution of

$$
\boxed{\quad g^{ab}(x)\,\partial_aT_{a'}(x)\,\partial_bT_{a'}(x)=n(x)^2,\qquad T_{a'}(a')=0,\quad T_{a'}\ge0.\quad}
$$

Equivalently, among all absolutely continuous surface curves from $a'$ to $a$,

$$
T_{a'}(a)=\inf_{\gamma(a')=a'}^{\gamma(1)=a}
\int_\gamma n(x)\,ds_g,
\qquad ds_g^2=g_{ab}\,dx^a dx^b.
$$

This is Fermat's principle. With optical parameter $s$, the characteristic Hamiltonian is

$$
H_0(x,p)=\frac12\big(g^{ab}p_ap_b-n(x)^2\big)=0,
$$

and its rays satisfy

$$
\frac{dx^a}{ds}=g^{ab}p_b,
\qquad
\frac{dp_a}{ds}=\frac12\partial_a n^2-\frac12(\partial_ag^{bc})p_bp_c.
$$

Thus the scalar GRIN medium is exactly equivalent to the Eikonal problem for this prescribed-speed model; the equality is between travel time and optical length, not between an arbitrary dissipative granular trajectory and a light ray.

**Rotation.** A Coriolis term is velocity-dependent and cannot, in general, be represented by a scalar refractive index. The exact optical analogue is a Randers/Finsler (magnetic) medium. Let $A=A_a(x)dx^a$ be a one-form whose curvature $B=dA$ is calibrated so that the magnetic force $g^{-1}(\iota_{\dot x}B)$ equals the tangential Coriolis force $-2\mathbf P_T(\boldsymbol\Omega\times\mathbf v)$ of the reduced packet model. Under the strong-convexity condition $|A|_g<n$, define

$$
F_R(x,\dot x)=n(x)\sqrt{g_{ab}\dot x^a\dot x^b}+A_a(x)\dot x^a,
$$

with Hamiltonian

$$
H_R(x,p)=\frac12g^{ab}(p_a-A_a)(p_b-A_b)-\frac12n(x)^2=0.
$$

The corresponding Hamilton equations give the GRIN bending plus the Coriolis magnetic deflection. If the calibrated Coriolis field is not representable by a position-only two-form, the full packet ODE below is the authoritative model and the scalar Eikonal/Randers reduction is not exact. This distinction is required for mathematical correctness.

**Ray/BVH algorithm.** On each frozen geometry and erosion step:

1. sample $v_{\rm e}$ and, when applicable, $A$ on face quadrature points and interpolate them on each triangle;
2. build/refit the surface BVH and launch one Hamilton ray per eroding or relief source;
3. integrate $H_0$ or $H_R$ with a structure-preserving step, querying the BVH after every segment to locate the next triangle, silhouette crossing, cavity opening, or neck;
4. stop at the viscosity-solution arrival condition, the static-friction condition, lift-off, escape, or $T_{\rm run}$;
5. deposit the packet using the terminal hit barycentric weights and normalize the resulting kernel so that $\int_\Gamma K(a,a')\,dS(a)=1$ for every non-escaped source.

RT hardware performs BVH traversal, triangle intersection, barycentric hit extraction, and segment visibility. The Hamilton update, Eikonal residual, friction evaluation, Coriolis calibration, and mass normalization remain compute/reduction work; RT hardware alone does not solve the global Eikonal PDE.

**Sliding.** With outward normal $\mathbf n$, tangential velocity $\mathbf v$, and normal curvature $\kappa_v$ along $\mathbf v$ (positive where convex),

$$
\boxed{
\dot{\mathbf v}=\mathbf P_n\big(\mathbf g_{\rm eff}-2\boldsymbol\Omega\times\mathbf v\big)-\mu(I)\,N\,\frac{\mathbf v}{|\mathbf v|}-\kappa_v|\mathbf v|^2\,\mathbf n,
\qquad
N=-\mathbf g_{\rm eff}\cdot\mathbf n+2(\boldsymbol\Omega\times\mathbf v)\cdot\mathbf n-\kappa_v|\mathbf v|^2,
}
$$

with $\mathbf P_n=\mathbf I-\mathbf n\mathbf n^{\!\top}$ and $N$ the normal reaction per unit mass. The last term keeps the packet on $\Gamma$ without changing $|\mathbf v|$. Basal friction is the $\mu(I)$ law for a depth-averaged Bagnold layer, with basal shear rate $5|\mathbf v|/(2h_s)$ and basal pressure $p_b=\rho_{\rm bulk}Nh_s$:

$$
\boxed{
I=\frac{d_p\,\dfrac{5|\mathbf v|}{2h_s}}{\sqrt{(p_b+p_c)/\rho_p}},
\qquad
\mu(I)=\mu_s+(\mu_2-\mu_s)\frac{I}{I_0+I},
}
$$

the same law and constants as (B1). Without $p_c$ the value is exact through the layer; with $p_c$ it is the basal value.

**Arrest.** The packet stops at the first instant with $|\mathbf v|\le v_{\rm stop}$ and $|\mathbf P_n\mathbf g_{\rm eff}|\le\mu_sN$ (static Coulomb at rest), with $v_{\rm stop}=\epsilon_{\rm stop}\sqrt{|\mathbf g_{\rm eff}|h_s}$; then $X(a')=\mathbf x$. A packet released where the static condition holds does not move.

**Flight.** Where $N\le0$ the packet lifts off:

$$
\boxed{
\dot{\mathbf v}=\mathbf g_{\rm eff}(\mathbf x)-2\boldsymbol\Omega\times\mathbf v,
\qquad
\mathbf g_{\rm eff}(\mathbf x)=\mathbf g_{\rm grav}(\mathbf x)+\omega_n^2\mathbf x_\perp+\mathbf g_{\rm ext}(\mathbf x).
}
$$

Centrifugal and tidal terms are exact. Self-gravity near the surface uses the first-order expansion $\mathbf g_{\rm grav}(\mathbf x)\approx\mathbf g_{\rm grav}(\mathbf x_\Gamma)+H^+(\mathbf x_\Gamma)(\mathbf x-\mathbf x_\Gamma)$, with $\mathbf x_\Gamma=\pi(\mathbf x)$ the closest point of $\Gamma$ and $H^+$ the exterior one-sided Hessian of the containing face (Proposition 2.7.1(5)); its error is $O((h/\bar R)^2)|\mathbf g|$ for hop height $h$. It uses no distance to the origin, which may lie outside $\mathcal V$ under (G). For $h$ above a fixed fraction of $\bar R$ (e.g. $0.1$) the field is evaluated directly at $\mathbf x\notin\overline V$ by the exterior route of Section 3.11, together with its classical Hessian $H(\mathbf x_0)$. Until the packet has moved $|\mathbf x-\mathbf x_0|\le0.1\,h$ from that evaluation point it uses $\mathbf g_{\rm grav}(\mathbf x_0)+H(\mathbf x_0)(\mathbf x-\mathbf x_0)$, whose error $O((|\mathbf x-\mathbf x_0|/h)^2)|\mathbf g|$ is within the same budget as the near-surface expansion; beyond that the field is evaluated again. Landing is perfectly inelastic normally ($e_n=0$): the tangential velocity is kept and sliding resumes. A packet above $r_{\rm esc}=3\max_\Gamma|\mathbf x|$, or above $1.05\max_\Gamma|\mathbf x|$ while moving outward under outward effective gravity, is classified **escaped**. This is a heuristic classification, not an orbital criterion. The motivation, that a packet beyond the synchronous radius $(GM/\omega_n^2)^{1/3}$ does not return, holds only for a spherical body, equatorial motion, and near-circular orbits, and the thresholds are not that radius; a trajectory with inward velocity can cross the synchronous radius and fall back. Under A-C3 an escaped packet is returned to its launch point, so escape is a count, not a mass loss; an explicit sink is not used. Under A-O (Section 10.3.6) the radius $1.05\max_\Gamma|\mathbf x|$ is not used. A packet in flight is escaped when it lies beyond a neck plane of its Hill region, or beyond $r_{\rm esc}$. Its mass is then lost: it is laid down nowhere, and the step's mass target falls by it. A packet still moving at $T_{\rm run}$ stops and is **timed out**. Both counts are reported.

**Runout kernel.** Let $X_R(a')$ be the terminal point of the Eikonal/Randers ray, or of the full packet ODE when the optical reduction hypotheses fail. With the normalized local spread $G_w$,

$$
\boxed{
K_R(a,a')=G_w\big(a,X_R(a')\big),
\qquad
\int_\Gamma K_R(a,a')\,dS(a)=1\ \text{for every non-escaped }a'.
}
$$

An escaped source contributes zero to deposited mass and is debited from the step mass target. Discretely, $J_{{\rm ero},i}A_i$ is routed to the terminal triangle by barycentric hit weights and then spread by $G_w$; the non-escaped sum is preserved to rounding.

**Reduction.** $X(a')=a'$ gives the isotropic kernel exactly (runout switched off). $\boldsymbol\Omega=\mathbf 0$ leaves the gravitational slope alone. $\mu_s\to\infty$ means no motion.

**Why Coriolis enters the runout and not Mode B.** The Coriolis-to-gravity ratio is $2\Omega U/|\mathbf g_{\rm eff}|$, and $\mathrm{Ro}=U/(2\Omega L)$. For Ryugu at 3.3 h ($|\mathbf g|\approx1.5\times10^{-4}$ m s⁻², $2\Omega\approx1.1\times10^{-3}$ s⁻¹), a packet accelerated at $10^{-5}$ m s⁻² over 50 m reaches $U\approx3\times10^{-2}$ m s⁻¹ in about $3\times10^3$ s. So $\mathrm{Ro}\approx0.6$ and $2\Omega t\approx3$, a turn of order one. In the rigid core the regularized creep is of order $\dot\gamma_YR\approx3\dot\gamma_\varepsilon R$ (Section 10.3.4); for $\dot\gamma_\varepsilon\le10^{-9}$ s⁻¹ and $R\lesssim1$ km this is $U\lesssim3\times10^{-6}$ m s⁻¹, a Coriolis-to-gravity ratio below $10^{-4}$. In a yielded region the Coriolis force per volume, $2\rho_K\Omega U$, is small against the viscous force $\sim\eta U/R^2$ exactly when $\mathrm{Ek}^{-1}\ll1$, the indicator of (B1). Under these conditions the frozen-coefficient momentum balance omits the Coriolis block without loss (Section 7.3). The same estimate gives $t_{\rm run}\sim10^3$–$10^4$ s against $\Delta t_n\gtrsim10^{10}$ s (item 3).

**Discretization.** On the polyhedron $V^{(n)}$, $\mathbf n$ and $\mathbf g_{\rm eff}$ are the piecewise-constant face values of step 1. Following Section 8.1 (G9):

- a packet lands at the first crossing of its drift segment with $\Gamma$;
- the surface point below $\mathbf x$ is the closest point $\pi(\mathbf x)$;
- the escape radii use $\max_\Gamma|\mathbf x|$;
- relief and deposits are normal thicknesses, with $(D\mathcal M)_i=\rho_K(\mathbf X_i)|\mathbf A_i|$.

One sliding substep of length $\tau$:

1. adds $\tau\mathbf P_n\mathbf g_{\rm eff}$ to $\mathbf v$;
2. turns $\mathbf v$ about $\mathbf n$ by $-2(\boldsymbol\Omega\cdot\mathbf n)\tau$ (the exact, unconditionally stable tangential Coriolis term);
3. lowers $|\mathbf v|$ by $\mu(I)N\tau$ without reversing it;
4. moves the packet to the surface point in the direction of $\mathbf x+\tfrac\tau2(\mathbf v_{\rm old}+\mathbf v)$ (trapezoidal drift, exact for constant deceleration), and projects $\mathbf v$ onto the new tangent plane at the same speed.

$\kappa_v$ is the signed turning of the smooth normal (area-weighted vertex normals blended barycentrically) along the motion, $\kappa_v\approx(\mathbf n_{\rm new}-\mathbf n_{\rm old})\cdot\hat{\mathbf v}/\Delta s$ over a substep of length $\Delta s$: positive on convex ground, negative in concave ground (crater floors, necks), where it raises $N$. An unsigned $|\Delta\mathbf n|/\Delta s$ would treat concave ground as convex and trigger spurious lift-off. Face normals would read facet edges as kinks. $N$ uses its last value. In flight the substep is velocity Verlet: $\mathbf v$ is turned about $\mathbf e_z$ by $-2\omega_n\tau$, the position moves by $\tau\mathbf v+\tfrac{\tau^2}{2}\mathbf g_{\rm eff}(\mathbf x)$ (the trapezoidal drift), and the velocity gets $\tfrac\tau2\big(\mathbf g_{\rm eff}(\mathbf x)+\mathbf g_{\rm eff}(\mathbf x')\big)$, the second half with the value at the end point $\mathbf x'$ that the next substep evaluates anyway. The flight is then second order in $\tau$ at one evaluation per substep: a kick with the start value alone is first order, and on a flight whose height is comparable with $\bar R$, where $\mathbf g_{\rm eff}$ changes by a fraction $\sim\theta_r$ over one substep, it lowers the apex by several per cent. A landing ends the substep with the start value, since the normal velocity is then removed. The substep is

$$
\tau=\theta_r\min\big(\ell/|\mathbf v|,\ \sqrt{\ell/|\dot{\mathbf v}|},\ |\mathbf v|/|\dot{\mathbf v}|,\ 1/(2\omega_n)\big),
$$

with $\theta_r=0.2$, and the last substep ends at $T_{\rm run}$. The length $\ell$ is the edge length while sliding and $\max(\text{edge length},h)$ in flight: the field varies on the scale of the height, and a drift of at most $\theta_rh$ cannot pass the ground before the landing test. The third bound resolves deceleration. The fourth resolves the Coriolis turn, which is exact in velocity but not in position. Packets are independent and traced in parallel, and only nodes with $J_{\rm ero}>0$ are traced.

**Routing of the slope relief (A-C3, second part).** Where Mode C does not erode ($F_\delta\le0$ at the surface), landslides are carried by the slope relief of Section 10.3.3 on the vertices promoted by A-C4. Without routing, the removed mass would be returned by the least-change correction of the mass row, without regard to where a slide goes. A-C3 sends it along the runout. With $d^1$ the first solution and $\mathcal V_{\rm slope}$ the movable vertices ($\in\mathcal A$) of the faces of $\mathcal S_{\rm slope}$, vertex $i\in\mathcal V_{\rm slope}$ loses

$$
\boxed{
r_i=\big[d_{{\rm pred},i}-d^{1}_i\big]_+,
\qquad
m^{\rm rel}_i=(D\mathcal M)_i\,r_i ,
}
$$

with $(D\mathcal M)_i$ the mass-row coefficient; $m^{\rm rel}_i$ is not the layer mass $m_i=\rho_{s,i}A_i$ of Section 8.1 (G5). These masses route the deposit. The loose-layer ledger of a promoted source is debited with the mass of the accepted step instead (Section 2.8c, item 3). A packet released at $a_i$ runs out to $X(a_i)$ and is spread by $K$, and vertex $j$ receives

$$
\boxed{
\delta_j=\frac{\sum_i K(a_j,X(a_i))\,m^{\rm rel}_i\,A_j}{(D\mathcal M)_j},
\qquad
\sum_j(D\mathcal M)_j\,\delta_j=\sum_i m^{\rm rel}_i .
}
$$

The subproblem is then solved once more, starting from the first solution, with $d'_{\rm pred}=d_{\rm pred}+\delta$:

- a locked vertex with $\delta_j>0$ joins $\mathcal D$, so the deposit rests on the core;
- a movable vertex has its target raised;
- the mass row is now met by the routed deposit.

One pass is made. The relative difference between the masses removed by the two solves is reported as the **routing gap**.

**Parameters:** the runout switch (for eroded packets and routed relief alike), $T_{\rm run}$ (s), and $\epsilon_{\rm stop}$; all part of step 0 of Section 10.3.4. **Report** per step, over all packets:

- the number of packets traced;
- the routed relief mass and the routing gap;
- the mean and largest runout distance, and the mass-weighted mean change of $|\text{latitude}|$ (negative is equatorward);
- the number of hops and the largest hop height;
- the escaped and timed-out counts.

A-C3 is a kinematic closure of the erosion–deposition interface. It changes neither the stress, nor the yield classification, nor any Mode B acceptance test.

## 2.8c Slope promotion of the loose layer (hypothesis A-C4)

*Motivation: a subgrid closure of shallow failure.* The loose layer (thickness $h_s$, of order metres on rubble-pile surfaces) lies far below the resolution of the truncated Stokes solve (Section 2.8, B2, cause 1). So (B2) cannot fail it; where the movable set of (B3) is empty at the surface, (B4) alone would never move the surface, even where Section 10.2 reports a loose layer steeper than repose. A-C4 decides shallow failure at mesh scale instead, by the infinite-slope Mohr–Coulomb balance at depth $h_s$. With normal stress $p_{\rm ref}=\rho_{\rm bulk}|\mathbf g\cdot\mathbf n|h_s$ and shear stress $\rho_{\rm bulk}|\mathbf g\times\mathbf n|h_s$, the layer holds when

$$
\frac{|\mathbf g\times\mathbf n|}{|\mathbf g\cdot\mathbf n|}\le\frac{c_s+\mu_{\rm proj}\,p_{\rm ref}}{p_{\rm ref}},
$$

with $c_s=c(d=0)$ (Section 4.3a). Under A-ϖ or A-Θ, $c_s$ is replaced here and below by the shallow cohesion $c_\Gamma$ of Section 11.6. The projection angle of Section 10.2, $\tan\phi=\mu_{\rm eff}=[c_s+\mu_{\rm proj}p_{\rm ref}-\|\mathbf T_{\rm tide}\|_F/\sqrt2]_+/p_{*,\rm ref}$, is this ratio with the tidal backstress subtracted and $p_c$ added to the confinement. A positive slack below therefore means the shallow layer under the face would yield in a resolved solve. (B2) on the truncated creep stress decides deep and whole-body failure. A-C4 on the mesh faces supplies shallow failure. The two share strength parameters, and A-C4 is a closure, not a consequence of (B2).

**Hypothesis A-C4 (slope promotion).** The repose slack of face $f$ at step $n$ is

$$
s_f:=\Big[\frac{|\mathbf g_f\times\mathbf n_f|}{|\mathbf g_f|}-\sin\phi_f\Big]_+ .
$$

1. Face $f$ is *supercritical* when $s_f>\tau_s$, with a configured tolerance $\tau_s\ge0$. $\tau_s$ is a tolerance on the difference of sines, not on the stress ratio of the Mohr–Coulomb balance above, and the two thresholds agree only at $\tau_s=0$. A face of slope $\theta$ is supercritical when $\sin\theta>\sin\phi_f+\tau_s$, which is the stress-ratio margin $\tan(\arcsin(\sin\phi_f+\tau_s))-\tan\phi_f$. $\mathbf g_f$, $\mathbf n_f$, and $\phi_f$ are face-centre values (Section 10.2) on the converged Stokes state of the step.
2. Vertex $i$ is *promoted*, $\pi_i=1$, when it belongs to a supercritical face, is not already movable ($\chi_{Y,i}=0$), and carries loose material ($\rho_{s,i}>0$).
3. A promoted vertex is a free variable of Section 10.3.3. Its predictor has no creep: $d_{{\rm pred},i}=\Delta h^*_{C,i}\ge0$, zero unless Mode C deposits there (it does not erode, since $\chi_{A,i}=0$). Its relief cannot cut below the loose layer it carries plus the deposit it receives in the step. Where $(D\mathcal M)_i>0$,
$$
d_i\ \ge\ \min(d_{{\rm pred},i},0)-\frac{\rho_{s,i}A_i}{(D\mathcal M)_i}\ \le\ 0,
$$
so $d=0$ stays feasible; where $(D\mathcal M)_i\le0$ there is no bound. For a full layer, $\rho_s=\rho_{\rm bulk}h_s$, the bound lies $(\rho_{\rm bulk}/\rho_K(\mathbf X_i))(A_i/|\mathbf A_i|)\,h_s$ below the current surface, with $A_i/|\mathbf A_i|\ge1$ and equality on a flat ring. The ledger uses the accepted step. With $d'_{\rm pred}=d_{\rm pred}+\delta$ the prediction of the second solve of Section 2.8b and $\lambda$ the accepted fraction of the step, the layer is debited by
$$
m^{\rm lay}_i=\lambda(D\mathcal M)_i\big[d'_{{\rm pred},i}-d_{*,i}\big]_+\ \le\ \rho_{s,i}A_i+\lambda(D\mathcal M)_i\big[d'_{{\rm pred},i}\big]_+,
$$
the inequality following from the bound above for $\lambda\le1$: a layer loses at most what it carries and what arrives on it in the step. Each $m^{\rm lay}_i$ is credited to $\rho_s$ along the runout of the first solve: it is moved to the landing points $X(a_i)$ of the packets of $i$ and spread by $K$, and an escaped packet lands at its source. The credit equals the debit, so the ledger $\sum_iA_i\rho_{s,i}$ is unchanged. $\sum_im^{\rm lay}_i$ differs from the routed $\sum_im^{\rm rel}_i$ by the factor $\lambda$ and by the change between the two solves, which the routing gap reports.
4. Promotion changes none of the following, all of which keep the (B3) classification: the stress; $\chi_Y$ and $F_\delta$; the Mode C fluxes ($\chi_A=\chi_Y\mathbf 1_{\{\rho_s>0\}}$ excludes promoted vertices); the eroded-packet sources of the runout (the relief packets of Section 2.8b are sources of their own, not a reclassification); the core advection $\chi_Y\mathbf v_{\rm sl}$ (Section 9.2.2); and the CFL bound.

The rule is local: a yielded vertex elsewhere does not switch it off.

**Slope rows.** The slope set of Section 10.3.3 is

$$
\mathcal S^{(n)}_{\rm slope}=\Big\{f\in\mathcal F:\ \sin\phi_f\ \text{finite},\ \textstyle\sum_{i\in f}\rho_{s,i}>0,\ f\text{ has at least one vertex in }\mathcal A^{(n)}\Big\},
$$

a set of faces ($\mathcal F$ the faces of $\Gamma$), with $\mathcal A=\{\tilde\chi=1\}$ of (B4) a set of vertices. Each face of $\mathcal S_{\rm slope}$ carries the second-order-cone slope row at its centre; its vertices outside $\mathcal A$ enter the linearized normal $\widehat{\mathbf n}_f(d)$ with their fixed increments ($0$ on $\mathcal L$, $d_{\rm pred}$ on $\mathcal D$). The set contains the supercritical faces, so every promoted vertex has a slope row, and it contains the subcritical loose faces around them, whose rows start inactive and keep the relief from pushing a neighbour past repose. Every face with a vertex in $\mathcal A$, loose or not, carries the anti-overhang row, except a face that already overhangs at $d=0$: item 2 of Section 2.13 removes its rows before the solve and counts it.

**What the relief does.** Where Mode C does not deposit, $d_{\rm pred}=0$ on promoted vertices, so the subproblem is the least-change projection of the current shape onto the repose cone,

$$
\min_d\ \tfrac12\sum_{i\in\mathcal A}w_i(d_i-d_{{\rm pred},i})^2+\varrho\sum_{f}t_f
\quad\text{s.t. slope rows, anti-overhang rows, mass row, trust box.}
$$

It removes material from supercritical faces and places it elsewhere in $\mathcal A$ or on $\mathcal D$, through the mass row or the routed deposit of Section 2.8b. The same program without the Stokes solve, with weights that protect selected regions (e.g. craters), is a stand-alone static relaxation of a given shape. In that relaxation the largest slack of the loose faces is the merit. A sweep that raises it is rejected like a geometric failure: $\Delta$ shrinks and the sweep is repeated. When $\Delta$ would fall below its floor, the relaxation stops at the last accepted shape and is reported as stalled. The largest slack is therefore non-increasing over the sweeps.

**Time scale.** Failure is instantaneous geologically. Under hypothesis A-T1 (Section 10.3.1) the relief is part of the fast operator: after the creep step, the surface settles in sweeps on the current shape (step S of Section 10.3.4), and settling advances no time. Each sweep re-evaluates the face gravity, promotes the vertices of the faces that are then supercritical, and solves the routed cone program. A sweep's relief is limited by the trust box $|d_i|\le\Delta=\Delta_0\bar R$ (configured fraction $\Delta_0$ of the mean radius $\bar R$), by the loose mass of each promoted vertex (item 3), and by the KKT tolerance. The second bound keeps the closure shallow: a promoted vertex never loses more than the layer it carries, and an exhausted layer ($\rho_{s,i}=0$) is not promoted again (item 2), so a shallow failure cannot excavate the body. Settling stops when no loose face is supercritical; when a sweep no longer removes a fixed fraction of the supercritical faces and does not lower the largest slack, the rest is unrelievable; and when no increment is admissible down to the trust floor. A configured number of sweeps bounds it, and what remains continues in the next step. Promoted vertices have no creep speed and do not enter $\Delta t_{\rm CFL}$. The onset of a landslide is physical to within one step; its duration is not modelled (instantaneous on the scale of $\Delta t_n$).

**Reports:**

- the number of movable vertices (promoted vertices included);
- the number of supercritical faces and the largest slack;
- the cone's residual slope slack $\sum t_f$;
- the largest displacement and the trust radius.

**Stage equilibrium.** Criterion (c) of Section 2.15 is a fixed-point test. With the whole body in the rigid core ($\chi_Y\equiv0$ on $\mathcal X_Y$) and no erosion ($J_{\rm ero}\equiv0$), a stage is equilibrated when the settling of step S leaves the surface unchanged: no loose face is supercritical, or the relief, converged within its KKT tolerance, cannot reduce the faces that are (for instance a face whose locked edge is itself steeper than repose). A *cone unresolved* sweep (Section 10.3.3) does not count. The step that reaches the fixed point ends the stage, and the remaining slack is reported as unrelievable. **Reduction.** $\tau_s\to\infty$ or $\rho_s\equiv0$ gives $\pi\equiv0$ and plain (B3)–(B4).

A-C4 is registered as PO-17. Its open questions:

- the instantaneous-failure assumption;
- the dependence of slide durations and per-step displacements on $\Delta$ and $\Delta t_{\max}$;
- agreement of the shallow-layer criterion with a solve that resolves the layer.

---

## 2.9 SCF iteration driven by the unified formulae

Let $V^{(n)}$ be the shape at geological step $n$. In the rest of this chapter $\omega$ is the frozen spin $\omega_n=\omega(t_n)$ (Section 10.3.1).

## 2.10 Forward evaluation

At each surface node, compute $U_i$, $\nabla U_i$, and $\nabla^2U_i$ from the unified representation, using spherical finite-part subtraction near the surface. Then

$$
W_i^{(n)}
=
U_i^{(n)}
+
\frac12\omega^2
\left[
(x_{i1}^{(n)})^2+(x_{i2}^{(n)})^2
\right],
\qquad
g_i^{(n)}
=
\nabla U_i^{(n)}
+
\omega^2
(x_{i1}^{(n)},x_{i2}^{(n)},0).
$$

---

## 2.11 Fluid mass-level interface (Mode A, retained but not enabled)

This and the next section define only the fluid interface of Mode A. Modes B and C impose $\mathcal M(V,\rho)=M$ directly on the material domain. They never select a boundary by a potential level, and they do not depend on these root-finds. Define $M^{(n)}(W_0)=\int_{\{W^{(n)}>W_0\}}\rho_K\,dx$ and solve $M^{(n)}(W_0^{(n)})=M$. Bisection is valid if $M^{(n)}$ is continuous and strictly decreasing on the target interval. That requires the region between the target levels to have positive volume and nonzero density.

---

## 2.12 Fluid equipotential interface (Mode A, retained but not enabled)

For each direction $u_i$ solve $W^{(n)}(R^{(n)}_{\mathrm{eq},i}u_i)=W_0^{(n)}$, giving $R_{\rm eq}^{(n)}=(R^{(n)}_{\mathrm{eq},1},\dots,R^{(n)}_{\mathrm{eq},N})$. This is executed only when a root exists in the search interval. Bisection needs strict monotonicity there, and with several roots the selection rule must be fixed in advance and the branch continued. The result is a fluid reference shape only, never a solid target of Modes B or C. A locally smooth equipotential follows from the implicit-function theorem only with sufficient regularity and $\nabla W\ne0$. At a polyhedral boundary or a critical point, global smoothness cannot be asserted.

---

## 2.13 Constraint correction after the mechanical prediction (Mode B primary, Mode C auxiliary)

Freeze $g^{(n)}$ and $H_W^{(n)}$. The flow prediction relative to the core comes from the Mode B creep solve and the yield classification (B1–B4). Only yielded surface points of the movable set are passed to Mode C, and the hand-off is explicit: no flow displacement or material flux is counted twice. The prediction $d_{\rm pred}^{(n)}$ (Section 2.8, B4) contains neither $R_{\rm eq}$ nor any equipotential blending.

**The current shape is the reference.** The only allowed changes are the predicted increment, which vanishes on the locked set, and the slope relief of the vertices promoted by A-C4. With the partition $\mathcal L,\mathcal D,\mathcal A$ of (B4), protected region $\mathcal P^{(n)}:=\mathcal L^{(n)}$, and weights $w_i>0$, the fidelity part of the objective is

$$
\frac12\sum_{i\in\mathcal A^{(n)}}w_i
\left(d_i-d_{{\mathrm{pred}},i}^{(n)}\right)^2,
$$

the weighted deviation from the current shape plus the mechanical increment. The slack penalties of the linearized subproblem below complete the objective. It **must not** contain $R_{\rm eq}$, $W_0$, or any equipotential term. An equipotential is the shape of a fluid without shear strength, and targeting it would erase unyielded topography into an ellipsoid. Non-movable nodes are fixed:

$$
d_i=0\qquad(i\in\mathcal L^{(n)}),\qquad
d_i=d_{\mathrm{pred},i}^{(n)}=\Delta h_{C,i}^*\qquad(i\in\mathcal D^{(n)}).
$$

**Outer nonlinear acceptance.** Write $\mathbf X(d)$ for the vertices $\mathbf X_i^{(n)}+d_i\mathbf N_i^{(n)}$. The true slope slack $[|\mathbf g_f\times\mathbf n_f(\mathbf X(d))|/|\mathbf g_f|-\sin\phi_f]_+$ is evaluated on the loose faces of $\mathcal S_{\rm slope}$ and reported; it does not reject the step, because a face that ends supercritical re-enters the next step through promotion. The mass test is $\mathcal M(\mathbf X(d),\rho_{\rm cand})=M$. Here $\rho_{\rm cand}=\rho_K(\cdot;\theta^{\rm pred})$ is the density predicted by the core transport of Section 9.2.3 for the same step $\tau_n=\lambda\Delta t_n$ ($\rho_K^{(n)}$ in the frozen-density variant A-ρ), and $D\mathcal M^{(n)}$ uses the same density. Under A-Ξ, $\rho_{\rm cand}$ also carries the dilatational deficit of the matrix, $-\rho_b(1-J_\nu)$ (Section 11.4).

**Linearized subproblem.** Rows are indexed by faces $f$ and unknowns by vertices $i$. With the linearized face normal $\widehat{\mathbf n}_f(d)=\mathbf n_f^{(n)}+D\mathbf n_f^{(n)}d$, the face-centre gravity $\mathbf g_f^{(n)}$, and $\widehat{\mathcal M}(d)=\mathcal M^{(n)}+D\mathcal M^{(n)}d$, inside the trust region $\|d\|\le\Delta$,

$$
\frac{\|\mathbf g_f^{(n)}\times\widehat{\mathbf n}_f(d)\|}{|\mathbf g_f^{(n)}|}
\le\sin\phi_f^{(n)}+t_f\quad(f\in\mathcal S_{\rm slope}),
\qquad
\mathbf g_f^{(n)}\cdot\widehat{\mathbf n}_f(d)\le-\epsilon_n|\mathbf g_f^{(n)}|\quad(f\ni\text{a vertex of }\mathcal A),
\qquad
D\mathcal M^{(n)}d+s_M=M-\mathcal M^{(n)},
$$

with elastic slacks $t_f\ge0$, $s_M\in\mathbb R$ (kg) and anti-overhang margin $\epsilon_n\in(0,1)$. The anti-overhang row is linear. The mass row is $\widehat{\mathcal M}(d)=M$ when $s_M=0$. It **must not** be written $\widehat{\mathcal M}(d)=M-\mathcal M^{(n)}$, which subtracts $\mathcal M^{(n)}$ twice and reverses the sign. With the relative mass slack $\sigma_M:=s_M/M$, the objective becomes

$$
\frac12\sum_{i\in\mathcal A}w_i(d_i-d_{\mathrm{pred},i})^2+\varrho\sum_{f\in\mathcal S_{\rm slope}}t_f+\varrho_M|\sigma_M|.
$$

*How large $\varrho$ must be.* The penalty is exact when $\varrho>\max_f|y_{t,f}|$, the slack multipliers. Relieving a slope moves nodes by about the node spacing $\bar h_\Gamma$, so $|y_t|\sim w\bar h_\Gamma^2$ ($d$ in metres), and a configuration takes $\varrho\ge10w\bar h_\Gamma^2$. A smaller value lets the program keep a supercritical slope and pay the penalty, and $\sum t_f$ stays positive although the geometry could relieve it. *Units of the mass penalty.* The fidelity term is $\mathrm L^2$, and $t_f$ is dimensionless, so $\varrho$ and $\varrho_M$ are $\mathrm L^2$. The mass slack must enter through the dimensionless $\sigma_M$: a penalty $\varrho|s_M|$ with $s_M$ in kg would add $\mathrm L^2$ to $\mathrm L^2\,\mathrm M$. The bound above concerns the slope rows only. The choice $\varrho_M=\varrho\hat M$ ($\hat M$ the numerical value of $M$ in kg) is a deliberate overweighting that makes the mass row effectively hard: $\sigma_M\ne0$ only when it cannot be met inside the trust box.

**Discrete form of the subproblem.** The program is assembled as follows; this is the normative discrete form of the rows above.

1. The rows sit at face centres, with the face-centre gravity and $D\mathbf n_f^{(n)}d=\sum_{k\in f}(\partial_{\mathbf X_k}\mathbf n_f)\mathbf N_k\,d_k$, where $d_k$ is the free increment for $k\in\mathcal A$ and the fixed one for $k\in\mathcal L\cup\mathcal D$ ($\phi_f$ is evaluated there too, Section 10.2). The slack $t_f$ belongs to the face.
2. The anti-overhang row is imposed only on faces that satisfy it at $d=0$. A violating face is not repaired: both of its rows (anti-overhang and slope) are dropped, and the face is counted and reported. This is the only treatment of such a face; a repair would move the geometry outside the mechanics.
3. The trust region is the box $|d_i|\le\Delta$ on the free nodes. On a promoted vertex ($\pi_i=1$) it is also bounded below by the loose mass the vertex carries, $d_i\ge\min(d_{{\rm pred},i},0)-\rho_{s,i}A_i/(D\mathcal M)_i$ (Section 2.8c). The linearization error is per node, and a ball would shrink the RMS displacement of a large patch by $\sqrt{|\mathcal A|}$.
4. Let $[a_-,a_+]$ be the range of $a\cdot x$ over the box, clearance, and loose-mass bounds of the free variables. When $b_M\notin[a_-,a_+]$, $\sigma_M$ cannot change sign, so the program assumes $\sigma_M$ has the sign of $b_M$ and folds $\varrho_M|b_M-a\cdot x|$ into the linear term (exact penalty of a sign-definite slack). Otherwise the target is reachable, the exact penalty keeps $\sigma_M=0$, and the hard equality is solved. If a folded solution flips the sign, the hard equality is solved instead.
5. The program is solved by a primal–dual interior-point method on its homogeneous self-dual embedding, with Nesterov–Todd scaling of the second-order cones and a sparse quasi-definite $LDL^{\top}$ factorization of the reduced KKT system [27]. A point is accepted when the relative primal residual, dual residual, and duality gap are each below $10^{-3}$ in scaled variables ($x=d/\bar R$, relative mass, and the objective divided by its largest coefficient, which leaves the minimizer unchanged: the folded mass penalty of order $\varrho M$ would otherwise stand beside a quadratic many orders of magnitude smaller and pass for an unbounded direction); the method reaches far smaller values in a few tens of iterations, independent of the conditioning that slows first-order methods. Otherwise the solve has failed, and the outer level halves $\Delta$.
6. Embeddedness is kept inside the program by the clearance rows $-\theta_gt_i^-\le d_i\le\theta_gt_i^+$ of Section 8.1 (G4). The outer test checks positive volume, no self-intersection, a valid closed, consistently oriented mesh (the hypothesis of Section 1.1), and the mass.

**Feasibility.** The clearance rows and the loose-mass bound of Section 2.8c contain $d=0$. If the anti-overhang rows hold at $d=0$ ($\mathbf g_f\cdot\mathbf n_f<-\epsilon_n|\mathbf g_f|$), then $d=0$, $s_M=M-\mathcal M^{(n)}$, $t_f=\max(0,\cdot)$ is feasible. It is not automatically a Slater point, because the positive part can sit on a cone boundary. A Slater certificate needs a margin $\delta_t>0$ on every cone slack and a check of every unrelaxed linear row. The anti-overhang row has no slack. If $d=0$ violates it on a face, $d=0$ would be infeasible. Item 2 removes that face's rows before the SOCP, which makes $d=0$ feasible again; the overhanging face stays in the geometry and in the count. $t_f=s_M=0$ means only that the slacks vanish; the outer test still checks the true constraints. These are genuine cone constraints, a QP only after further linearization. $\widehat n$ is not a unit normal, and shrinking its norm does not satisfy the physical constraint. A candidate is renormalized and rechecked against the outer nonlinear constraints, which alone use the true normal, mass, and yield condition. This correction does not replace solving stress balance and the constitutive law. A yield certificate at a frozen tensor is not a geometrically coupled yield constraint.

Fields are frozen inside the subproblem. After the update, geometry, density, and stress are re-evaluated by a forward gravity solve and checked again. If the state is balanced, feasible, and has $d_{\rm pred}=0$, then $d=0$ is the unique minimizer, so topography does not drift without cause.

**Unified trust-region policy.** The elastic program is always feasible on an overhang-free geometry after preprocessing, because $d=0$ meets the clearance rows and the loose-mass bound of Section 2.8c, both of which contain $0$, so "if infeasible, shrink and retry" is not used. If the current point itself violates a constraint, the feasible set empties as $\Delta\to0$, and that strategy cannot work. The inner SOCP never shrinks $\Delta^{(n)}$. The outer level shrinks it by a factor $\theta_\Delta\in(0,1)$ and regenerates the subproblem when the geometric test rejects the candidate (non-positive volume, self-intersection, invalid mesh) or the cone program misses its KKT tolerance (item 5). It is not shrunk for a mass miss (Section 2.14). Nonzero optimal $t_f$ or $s_M$ means the linear model holds only partially. The outer level then decides by a filter or merit function and may recompute the prediction. It must never release the protected region or fall back to a fluid target. If the protected region surrounds a violating node, or a signed $\rho_K$ makes $D\mathcal M$ degenerate, $s_M\ne0$ is unavoidable and is reported.

Freezing $g$ ignores $Dg\cdot d$. The iteration is sequential-convex (Picard), not SQP, and superlinear convergence is not claimed. The linearization error is $O(\Delta^2)$, and the outer test sets a tolerance. $D\mathbf n_f^{(n)}$ must be the Jacobian of the **normalized** face normal, so that $\mathbf n_f\cdot D\mathbf n_f\,d=0$ and $|\widehat{\mathbf n}_f|\ge1$. Hence $|\widehat{\mathbf n}_f|^2=1+|D\mathbf n_fd|^2$, and the slope row, written with the unnormalized $\widehat{\mathbf n}_f$, is tighter than the slope of the renormalized normal: it bounds $\sin\theta_f$ by $(\sin\phi_f+t_f)/|\widehat{\mathbf n}_f|$, with $|\widehat{\mathbf n}_f|=1+O(|D\mathbf n_fd|^2)$. The excess is second order in $d$, the order of the linearization error that the outer test already accepts. Dividing the row by $|\widehat{\mathbf n}_f|$ would make it nonconvex.

---

## 2.14 Under-relaxation

For $0<\lambda\le1$, $\mathbf X_i\leftarrow\mathbf X_i+\lambda d_{*,i}\mathbf N_i$ (Section 8.1, G4), and $V^{(n+1)}$ is rebuilt. Locked nodes do not move. The physical time advanced is $\lambda\Delta t_n$. Afterwards, gravity, the Stokes problem, and the pressure are recomputed. Stress is instantaneous, so there is no history to update. $\rho_s$ follows the Mode C law. Mass and yield constraints are not guaranteed along the segment. Mass must not be repaired by a global scaling or a uniform increment that moves the protected region; the correction stays inside $\mathcal A$. On a miss of $|\mathcal M-M|\le\varepsilon_MM$, one correction along the mass gradient is applied,

$$
\mathbf X_i\leftarrow\mathbf X_i+\alpha\,(D\mathcal M)_i\,\mathbf N_i\quad(i\in\mathcal A),\qquad
\alpha=\operatorname{clip}\Big(\frac{M-\mathcal M}{\sum_{i\in\mathcal A}(D\mathcal M)_i^2},\,-\frac{\Delta}{\max_{i\in\mathcal A}|(D\mathcal M)_i|},\,\frac{\Delta}{\max_{i\in\mathcal A}|(D\mathcal M)_i|}\Big),
$$

so that $\alpha$ ($\mathrm L^2\mathrm M^{-1}$) is bounded by a length divided by $(D\mathcal M)_i$ ($\mathrm M\,\mathrm L^{-1}$), and every increment $\alpha(D\mathcal M)_i$ stays inside the trust box $|\cdot|\le\Delta$. The candidate is accepted if it passes the geometric test. A residual still above $\varepsilon_M$ is reported as the mass residual, not rejected. Neither $\Delta t_n$ nor $\Delta$ is reduced for a mass miss.

---

## 2.15 Convergence criteria

There are three independent layers, not substitutes for one another (algorithm in Section 10.3.4).

**(a) Rheological hyperbolic relaxation.** On fixed geometry and frozen outer constitutive data, the ray solver of Section 10.3.2 stops only when

$$
\varepsilon_{\rm mom}^{\rm hyp}
:=\frac{\|\rho_\tau\partial_\tau\mathbf v-\nabla\cdot\mathbf S+\nabla p-\mathbf f\|}{\max\{\|\mathbf f\|,\varepsilon_f\}}
\le\varepsilon_{\rm hyp},
$$

$$
\varepsilon_{\rm con}^{\rm hyp}
:=\frac{\|\mathbf S-2\eta^{(m)}\mathbf D(\mathbf v)\|}{\max\{\|\mathbf S\|,\varepsilon_S\}}
\le\varepsilon_{\rm hyp},
$$

$$
\varepsilon_{\rm div}^{\rm hyp}
:=\frac{\|\nabla\cdot\mathbf v-g\|}{\max\{\|g\|,\varepsilon_g\}}
\le\varepsilon_{\rm hyp},
\qquad
\varepsilon_{\rm tr}^{\rm hyp}\le\varepsilon_{\rm hyp}.
$$

The last quantity is the normalized traction residual at the boundary. The ray CFL condition is checked on every sweep. After a pseudo-time solve, $\tilde p$, $\dot\gamma$, $\mu_d$, and $\eta$ are refreshed; the outer rheology iteration stops only when their relative changes and the velocity change are below the configured tolerances. Failure at the pseudo-time limit is **rheology not converged**. A smaller geological step does not repair a failed pseudo-time solve.

**(b) Mechanical balance (every step; hard).** On the converged $(\mathbf v,\mathbf S,p)$,

$$
\varepsilon_{\rm mom}^{\rm weak}:=\frac{\|\mathbf F-\mathbf K(\mathbf u;\tilde p)\mathbf u-\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda\|_2}{\max\{\|\mathbf F^{\rm abs}\|_2,\varepsilon_F\}}\le\varepsilon_{\rm mom},\qquad
\varepsilon_\lambda:=\frac{\|\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda\|_2}{\max\{\|\mathbf F^{\rm abs}\|_2,\varepsilon_F\}}\le\varepsilon_{\rm mom},
$$

with the absolute load scale $F^{\rm abs}_\alpha=\int(\rho_K|\mathbf g_{\rm eff}||\mathbf B_\alpha|+\|\mathbf T_{\rm tide}\|_F\|\mathbf D(\mathbf B_\alpha)\|_F)\,dV$. A load that cancels exactly would make a ratio to $\|\mathbf F\|$ meaningless. The residual is assembled from the projected velocity basis and the CellBVH stress samples as

$$r_\alpha=F_\alpha-\sum_i w_i\,\mathbf S_i^{\rm ray}:\mathbf D(\mathbf B_\alpha)(\mathbf x_i)-(\mathbf B_{\rm rb}^{\top}\boldsymbol\lambda)_\alpha.$$

The second indicator measures incompatibility of the load with the rigid modes, from a misaligned frame or a missing discrete rigid mode (Appendix A.3). Either failure ends the step. The strong stationary residual and the traction deviation $\delta_{na}$ are resolution indicators that converge with $K$ and $h_{\rm vol}$. They are reported each step. Above a threshold the step is marked **insufficient resolution**, which does not change decision (b): a truncated stress field can satisfy (b) and still be resolution-limited.

**(c) Stage equilibrium (constant spin on a stage).** The geometry is in static equilibrium iff

$$
\boxed{\ \chi_Y\equiv0\ \text{on }\mathcal X_Y,\quad J_{\rm ero}\equiv0,\quad\text{the settling of step S leaves the surface unchanged},\quad\text{and (b) holds}. \ }
$$

The first condition says that the whole body, not only its surface, belongs to the rigid core: an interior yielded pocket would move the cores even with a locked surface. Then:

- $J_{\rm ero}=J_{\rm dep}=0$, $\mathcal D=\varnothing$, and $d_{\rm pred}=0$ (promoted vertices have no creep);
- $\mathbf v_{\rm adv}\equiv\mathbf 0$, so the cores do not move ($\xi_n=0$, Section 9.2.2) and nothing is retraced (Section 3.15.2);
- the creep step has no free variable, and settling finds no promotable supercritical face, or its converged relief is zero. A zero from a cone solve that missed its KKT tolerance (*cone unresolved*, Section 10.3.3) does not count.

Positive slope slack may remain only on faces that the relief cannot reduce; it is reported as unrelievable slack.

This state does not depend on $\Delta t_n$ or $\lambda$. Since nothing integrates $\mathbf D$, it is a genuine fixed point, not a slowly drifting creep; the core's regularized creep is removed by $\chi_Y=0$. Apart from the exact zero above, the size of $|d_*|$ is not a convergence test, because the adaptive step makes each displacement about $\theta_{\rm CFL}h_\Gamma$. If the stage time $T_j$ runs out before the fixed point is reached, the stage ends **not equilibrated**, and the movable area fraction $\sum_{i\in\mathcal A}A_i/\sum_iA_i$ (vertex areas of G4) is reported. That is a physical result (the slide has not stopped), not a numerical failure. Under a continuously varying $\omega(t)$ there is no stage target: only (a) and (b) are applied, up to the terminal time. With the hypotheses of Chapter 11, the state above is a fixed point only under the additional conditions of Proposition 11.14 (no healing, no thermal fatigue or production, and, under A-L, no axial torque). Otherwise the slow state still evolves at rest, and (c) is a test at each step, not a shortcut to the stage end. Also report the largest true slope slack over the loose faces and check the mass,

$$
\max_{f\in\mathcal S^{(n)}_{\rm slope}}
\left[
\frac{|\mathbf g_f^{(n)}\times\mathbf n_f^{(n)}|}
{|\mathbf g_f^{(n)}|}
-\sin\phi_f^{(n)}
\right]_+ ,
\qquad
|\mathcal M(V^{(n)})-M|
<\varepsilon_M ;
$$

the slack is a diagnostic (positive values at a fixed point are unrelievable), the mass bound is a test.

**Final audit.**

1. Recompute $U$, $\nabla U$, and $\nabla^2U$ on the terminal geometry.
2. On that geometry, density, and $\omega$, recheck (b), the yield classification, and zero displacement on the locked set.

KKT passage of a subproblem is not passage of the nonlinear loop, and a step that is small only because $\Delta$ has contracted has not converged. The effective potential, gravity, and Hessian are emitted only after this audit passes.

---

## 2.16 The unified coupled system

The loop combines four parts:

- the Feynman potential of Section 1.8 (equivalently Section 2.2);
- $\nabla W$ and $\nabla^2W$ of Section 2.8;
- the mass constraint and the slope inequality on $\mathcal S_{\rm slope}$ displayed at the start of Section 2.8 (the Poisson identity there is a check of the field, not a condition on the shape);
- the Mode B balance (B1) with the unique stress $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D$, the traction condition, the yield classification and core locking (B2–B3), and slope promotion.

The slope inequality holds only at loose-layer faces whose vertices are movable (promotion included) and where the surrogate is declared applicable. It is not sufficient for balance of the solid. The spin is exogenous (or follows the angular momentum under A-L, Section 11.7), Mode C supplies the local conservative prediction, and Mode A is not in the loop. Section 2.17 restates the whole model in one place and records its convex structure.

The GPU algorithm that follows (Chapter 3) rests on these ideas:

- intersections use a bounding-volume hierarchy (BVH), with hardware ray tracing as an option;
- $U$, $\nabla U$, and $H$ are computed together on the same ray;
- one batch of rays shares all geometric results;
- $\alpha=1$ uses the analytic antiderivative, and other $\alpha$ the composite rule of Section 3.3;
- the constant part $\rho(x)\Phi_V$ is closed-form (Section 3.9), so the rays carry only the density difference and no finite part;
- the closed-form part is accelerated by the face tree of Section 3.10.

Hardware ray tracing only intersects rays with triangles. It evaluates no Feynman integral, and without it the same interface runs on a software BVH:

$$
\boxed{
\text{RTFP ray geometry}
+
\text{GPU radial integration}
+
\text{GPU reduction}
+
\text{closed-form uniform part on a face tree}.
}
$$

## 2.17 The unified rheological model on a general body

This section restates, in one place and without the radial parametrization, the model of Sections 2.8–2.13, Chapter 4, Sections 8.3, 9.2, and 10.2–10.3, on a body satisfying hypothesis (G) of Section 8.1. It adds no physics; the hypotheses of Chapter 11 enter only through (M10). It retains Proposition 2.17.1 and Remark 2.17.2 as auxiliary frozen-pressure and limit-load benchmarks; the normative solver is the hyperbolic characteristic ray relaxation of Section 10.3.2.

**State at $t_n$.**

- The surface $\Gamma^{(n)}$, with vertices $\mathbf X_i$.
- The core parameters $\theta^{(n)}$ of $\rho_K=\rho_b+\sum_kw_k[1+(\mathbf y-\mathbf c_k)^{\!\top}\mathbf S_k(\mathbf y-\mathbf c_k)]^{-\alpha_k}$.
- The surface density $\rho_s^{(n)}$.
- The spin $\omega_n$.
- With the hypotheses of Chapter 11: the markers with $(\nu,\varpi)$, the surface damage $\varpi_\Gamma$, and the spin angular momentum $L_n$.

The unknowns of the step are $(\mathbf v,p)$ in $\mathcal V^{(n)}$ and the increments $d$.

**(M1) Loads.** $W=U+\tfrac12\omega_n^2|\mathbf x_\perp|^2$ (plus $U_{\rm tide}$ under the orbital spin schedule of Section 10.3.1), with $U$ the RTFP potential of $\rho_K$. Then $\mathbf g_{\rm eff}=\nabla W$, $H_W=\nabla^2W$, and $\mathbf T_{\rm tide}=C_T\rho_p\nu_0d_p^2\operatorname{dev}H_W$.

**(M2) Constitutive law (one stress).**

$$
\boxed{
\begin{gathered}
\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D,\qquad
\eta=\frac{c(\mathbf x)+\mu_d(I_c)\,\tilde p}{\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}},\qquad
\dot\gamma=\sqrt{2\mathbf D:\mathbf D},\qquad\tilde p=\max([p]_+,p_{\min}),\\
I_c=\frac{d_p\dot\gamma}{\sqrt{(\tilde p+p_c)/\rho_p}},\qquad
\mu_d=\mu_s+(\mu_2-\mu_s)\frac{I_c}{I_0+I_c},\qquad
c(\mathbf x)=c_b+(c_s-c_b)\,e^{-\operatorname{dist}(\mathbf x,\Gamma)/h_c}.
\end{gathered}}
$$

**(M3) Quasi-static balance.**

$$
\nabla\cdot\boldsymbol\sigma+\rho_K\mathbf g_{\rm eff}=\mathbf 0,\quad\nabla\cdot\mathbf v=0\ \text{ in }\mathcal V;\qquad
\boldsymbol\sigma\mathbf n=\mathbf t_{\rm ext}\ \text{ on }\Gamma;\qquad
\int_{\mathcal V}\rho_K\mathbf v\,dV=\mathbf 0,\quad\int_{\mathcal V}\rho_K\,\mathbf x\times\mathbf v\,dV=\mathbf 0 .
$$

**(M4) Hyperbolic pressure and stress relaxation.** The rheology is advanced by (8.3-H) with $\rho_\tau>0$, $T_s>0$, $\kappa>0$, and $g=0$ (or $g=\dot\varepsilon_v$ under A-$\Xi$). The traction condition is $(-p\mathbf I+\mathbf S+\mathbf T_{\rm tide})\mathbf n=\mathbf t_{\rm ext}$. Its stationary limit is the quasi-static balance and incompressibility constraint; no independent elliptic pressure solve is part of the reference loop.

**(M5) Classification.** $F_\delta=\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}-(1-\delta_Y)(c+\mu_s\tilde p)$. The rigid core is the largest-mass component of $\{F_\delta\le0\}$ on the mesh of Section 8.1 (G7). The slip velocity is $\mathbf v_{\rm sl}=\mathbf v-\mathbf a_c-\boldsymbol\theta_c\times\mathbf x$. With $\chi_Y$ of (B3) and the promotion $\pi$ of A-C4, the movable set is $\mathcal A=\{\max(\chi_Y,\pi)=1\}$.

**(M6) Geometric step.** With $d_{{\rm pred},i}=\chi_{Y,i}\Delta t_n\mathbf v_{{\rm sl},i}\cdot\mathbf N_i+\Delta h^*_{C,i}$,

$$
d_*=\arg\min_{d,t\ge0,\sigma_M}\ \tfrac12\sum_{i\in\mathcal A}w_i(d_i-d_{{\rm pred},i})^2+\varrho\sum_ft_f+\varrho_M|\sigma_M|
$$

subject to:

- the locked and deposition rows;
- the slope cones on the faces of $\mathcal S_{\rm slope}$ and the anti-overhang rows on every face with a vertex in $\mathcal A$;
- the mass row $\sum_i(D\mathcal M)_id_i+s_M=M-\mathcal M^{(n)}$, with $\sigma_M:=s_M/M$;
- the trust box, the clearance rows, and the loose-mass bound of the promoted vertices (Section 2.8c, item 3).

The update is $\mathbf X_i\leftarrow\mathbf X_i+\lambda d_{*,i}\mathbf N_i$ (Sections 2.13 and 10.3.3, with Section 8.1, G4).

**(M7) Surface layer.** $\rho_s$ is transported along $\mathbf v_T$ (Section 8.1, G5) with the Mode C source $\tau_n(\bar J_{\rm dep}-\bar J_{\rm ero})$, $\tau_n=\lambda\Delta t_n$, with the step averages over $\tau_n$ of Section 2.8, and the runout of A-C3.

**(M8) Density.** The cores are transported along $\mathbf v_{\rm adv}=\chi_Y\mathbf v_{\rm sl}$ (Section 9.2).

**(M9) Frame.** A rigid realignment to the centre of mass and the principal axes (Section 8.1, G8).

**(M10) Evolving state (Chapter 11, each item only when its hypothesis is on).**

- *A-Ξ:* in (M2) $\mathbf D\to\mathbf D'$ and $\mu_d=\mu_d(I_c,\nu)$; in (M3) $\nabla\cdot\mathbf v=\dot\varepsilon_v=\Xi\dot\gamma_p^\star$; in (M5) $\mu_y(\nu)$ in place of $\mu_s$; in (M8) the dilatational ratio $J_\nu$.
- *A-ϖ:* in (M2) $c=(1-\mathcal R\varpi)c_{\rm intact}$.
- *A-Θ:* the shallow cohesion $c_\Gamma$ in (M6) and Section 10.2, and the source $J_\Theta$ in (M7).
- *A-L:* $\omega_n=L_n/I_1^{(n)}$ in (M1), with $L_{n+1}=L_n+\tau_nT_\parallel$.
- The state laws are integrated exactly over $\tau_n$ on markers and vertices (Propositions 11.5, 11.9, 11.11).

A step is therefore three nested problems. The frozen-coefficient hyperbolic ray relaxation is advanced to stationary residual tolerance, the outer constitutive state is refreshed as a fixed point, and the geometric program is convex.

**Proposition 2.17.1 (Variational structure at frozen pressure; limit load).** Fix the geometry. Freeze $\tilde p\ge p_{\min}$ (bounded above), $c$, $\mathbf T_{\rm tide}$, and the loads. Let

$$
\ell(\mathbf v):=\int_{\mathcal V}\rho_K\mathbf g_{\rm eff}\cdot\mathbf v\,dV-\int_{\mathcal V}\mathbf T_{\rm tide}:\mathbf D(\mathbf v)\,dV+\oint_\Gamma\mathbf t_{\rm ext}\cdot\mathbf v\,dS,
\qquad
\phi(s;\mathbf x):=\int_0^s\big(c+\mu_d(I_c(\varsigma))\,\tilde p\big)\frac{\varsigma}{\sqrt{\varsigma^2+\dot\gamma_\varepsilon^2}}\,d\varsigma ,
$$

and $J(\mathbf v):=\int_{\mathcal V}\phi(\dot\gamma(\mathbf v);\mathbf x)\,dV-\ell(\mathbf v)$ on a finite-dimensional $\mathbb V_h\subset H^1_\sigma(\mathcal V)$ containing the rigid motions, with $\mathbb K:=\ker\mathbf B_{\rm rb}\cap\mathbb V_h$.

1. $\phi$ is $C^1$, strictly convex, and increasing, with $\phi'(s)=s\eta(s)$ at the frozen pressure. A velocity $\mathbf v\in\mathbb K$ is a fixed point of the fixed-viscosity (Picard) map of Section 8.3 iff it minimizes $J$ on $\mathbb K$, and it is the frozen-pressure benchmark state against which the stationary hyperbolic ray relaxation may be checked.
2. $J$ is strictly convex on $\mathbb K$.
3. Let $\tau_{\max}:=c+\mu_2\tilde p$, let $N(\mathbf v):=\int_{\mathcal V}\tau_{\max}\dot\gamma(\mathbf v)\,dV$ (a norm on $\mathbb K$), and let $L^*:=\sup\{\ell(\mathbf v):\mathbf v\in\mathbb K,\ N(\mathbf v)=1\}$. If $L^*<1$, $J$ has a unique minimizer. If $L^*>1$, $J$ is unbounded below, and no quasi-static solution exists at this pressure. The borderline $L^*=1$ is not decided by this test.

*Proof.* (1) $\phi'(s)=(c+\mu_d\tilde p)\,r(s/\dot\gamma_\varepsilon)$ with $r(x)=x/\sqrt{1+x^2}$. At frozen pressure $c+\mu_d\tilde p\ge c+\mu_sp_{\min}>0$ is nondecreasing in $s$ and $r$ is strictly increasing, so $\phi'$ is strictly increasing with $\phi'(0)=0$. From $\partial\dot\gamma/\partial\mathbf D=2\mathbf D/\dot\gamma$ one gets $\partial_{\mathbf D}\phi(\dot\gamma)=2(\phi'/\dot\gamma)\mathbf D=2\eta\mathbf D$. The first variation of $J$ is therefore the frozen-coefficient momentum balance with $\eta$ at $\mathbf v$ itself, which is the fixed-point condition, and convexity makes stationary points minimizers.

(2) $\dot\gamma=\sqrt2|\mathbf D|$ is a norm. Take $\mathbf D_1\ne\mathbf D_2$. Unequal norms give strict midpoint convexity through $\phi$. Equal norms give a strictly smaller midpoint norm, and $\phi$ is strictly increasing. Two distinct elements of $\mathbb K$ differ in $\mathbf D$ on a set of positive measure, because a rigid motion in $\mathbb K$ is zero (Lemma 8.12).

(3) $\phi'(s)/\tau_{\max}\to1$ uniformly in $\mathbf x$, since $\tilde p$ is bounded and $I_c\ge k_{\min}s$. Hence for each $\delta\in(0,1)$, $\phi(s)\ge(1-\delta)\tau_{\max}s-C_\delta$, and $J\ge(1-\delta-L^*)N-C_\delta|\mathcal V|$. With $L^*<1$ and $\delta<1-L^*$, $J$ is coercive on finite-dimensional $\mathbb K$, so a minimizer exists, unique by (2). With $L^*>1$, take $\ell(\mathbf v)>N(\mathbf v)$. Since $\phi(ts)/t\uparrow\tau_{\max}s$, $J(t\mathbf v)/t\to N(\mathbf v)-\ell(\mathbf v)<0$. ∎

*Reading.* $L^*$ is the limit-load multiplier at the frozen pressure, with strength $c+\mu_2\tilde p$ ($\mu_2=\mu_s$ for Bingham). If $L^*>1$ there is no quasi-static creep state, and no iteration can converge: velocities grow while $\eta\to0$, and the hyperbolic ray residual cannot converge to a stationary state at that frozen load. This is a mechanical failure of the body or a block, outside the quasi-static route. Regularization does not remove it, because $\phi$ grows linearly for every $\dot\gamma_\varepsilon>0$. Discretely, $L^*$ is the dual norm of $\ell$:

$$
L^*=\min\Big\{\max_i\frac{|\boldsymbol\varsigma_i|}{\tau_{\max,i}}:\ \ell=\sum_iw_i\langle\boldsymbol\varsigma_i,\mathbf e_i(\cdot)\rangle\ \text{on }\mathbb K\Big\},
$$

with $\mathbf e_i$ the strain-rate samples of Section 10.3.2. This is a small SOCP (the static theorem of limit analysis) that can be reported each step. The fixed-viscosity (Kačanov) iteration decreases $J$ monotonically when $\eta(s)=\phi'(s)/s$ is nonincreasing. That holds on the Bingham branch. With rate dependence it fails at small rates, because $d\ln\eta/ds=\mu_d'\tilde p/(c+\mu_d\tilde p)-s/(s^2+\dot\gamma_\varepsilon^2)>0$ as $s\to0$. This is why the fixed-viscosity iteration is only an auxiliary benchmark: the normative inner solver is the hyperbolic ray relaxation of Section 10.3.2. The convexity holds per frozen pressure only.

**Remark 2.17.2 (The pressure coupling is not variational).** In (M2) the strength depends on the mean stress, while the flow is isochoric and co-axial with $\boldsymbol\tau_{\rm flow}$. That is a non-associated flow rule (zero dilatancy, positive friction), and such laws do not derive from a convex dissipation potential in the stress. The coupled $(\mathbf v,p)$ problem is therefore a fixed point, not a convex program, and its existence is PO-1. Appendix A.6 records a convex surrogate and a bracketing conjecture (Conjecture A.2, PO-21). Under A-Ξ the flow dilates, but at the rate $\Xi\dot\gamma$, with $\Xi\to0$ at the critical state while the friction stays positive. The rule therefore remains non-associated, and this remark still applies (Remark 11.6a).

**Certified bracket of the limit load.** Take the discrete setting of Proposition 2.17.1: strain-rate samples $\mathbf E_i:\mathbb R^{n_u}\to\mathbb R^6$ at the volume nodes, normalized so that $|\mathbf E_i\mathbf u|=\dot\gamma_i(\mathbf u)$; weights $w_i$; strengths $\tau_i=c_i+\mu\tilde p_i$ with $c^w_i:=w_i\tau_i$; the rigid rows $\mathbf B$; and the load vector $\boldsymbol\ell$. With $N(\mathbf u)=\sum_ic^w_i|\mathbf E_i\mathbf u|$,

$$
L^*=\sup\{\boldsymbol\ell^{\!\top}\mathbf u:\ \mathbf B\mathbf u=0,\ N(\mathbf u)=1\}=\Big(\min\{N(\mathbf u):\ \boldsymbol\ell^{\!\top}\mathbf u=1,\ \mathbf B\mathbf u=0\}\Big)^{-1}.
$$

**Proposition 2.17.3 (Certified bracket).** For any weights $q_i>0$ with $\sum_iq_i\mathbf E_i^{\!\top}\mathbf E_i$ positive definite on $\ker\mathbf B$, let $\mathbf u_0$ solve $\sum_iq_i\mathbf E_i^{\!\top}\mathbf E_i\mathbf u_0=\boldsymbol\ell+\mathbf B^{\!\top}\boldsymbol\mu$, $\mathbf B\mathbf u_0=0$. Then $\boldsymbol\ell^{\!\top}\mathbf u_0>0$ unless $\boldsymbol\ell$ vanishes on $\ker\mathbf B$ (then $L^*=0$), and

$$
L_{\rm lo}:=\frac{\boldsymbol\ell^{\!\top}\mathbf u_0}{N(\mathbf u_0)}\ \le\ L^*\ \le\ L_{\rm hi}:=\max_i\frac{q_i\,|\mathbf E_i\mathbf u_0|}{c^w_i}.
$$

*Proof.* $\boldsymbol\ell^{\!\top}\mathbf u_0=\sum_iq_i|\mathbf E_i\mathbf u_0|^2\ge0$. The kinematic bound holds because $\mathbf u_0\in\ker\mathbf B$ is admissible. For the static bound, take $\mathbf v\in\ker\mathbf B$:
$$
\boldsymbol\ell^{\!\top}\mathbf v=\sum_iq_i(\mathbf E_i\mathbf u_0)^{\!\top}\mathbf E_i\mathbf v\le\sum_iq_i|\mathbf E_i\mathbf u_0||\mathbf E_i\mathbf v|\le\max_i\frac{q_i|\mathbf E_i\mathbf u_0|}{c^w_i}\sum_ic^w_i|\mathbf E_i\mathbf v|=L_{\rm hi}N(\mathbf v).
$$
The stresses $\boldsymbol\varsigma_i=q_i\mathbf E_i\mathbf u_0/w_i$ are equilibrated with $\boldsymbol\ell$ on $\ker\mathbf B$, and $L_{\rm hi}$ is their largest ratio to the strength. ∎

The bracket holds for every choice of weights. It therefore does not depend on the convergence of the iteration that produces them. The weights $q_i=c^w_i/\sqrt{|\mathbf E_i\mathbf u|^2+\epsilon^2}$ are those of the stationarity condition of the smoothed problem $\min\sum_ic^w_i\sqrt{|\mathbf E_i\mathbf u|^2+\epsilon^2}$. At its minimizer, $L_{\rm hi}/L_{\rm lo}\le\big(1-\epsilon\sum_ic^w_i/N(\mathbf u)\big)^{-1}$. The iteration recomputes the weights from the last $\mathbf u_0$ with $\epsilon=\theta\,\overline{\dot\gamma}$, where $\overline{\dot\gamma}$ is the root-mean-square rate of $\mathbf u_0$ and $\theta$ halves on each pass from 1. Each pass is one weighted Gram matrix of the strain samples (the assembly of the metric of Section 10.3.2) and one dense solve of size $n_u$. The iteration keeps the largest $L_{\rm lo}$ and the smallest $L_{\rm hi}$ found [30].

- *Decision.* $L_{\rm lo}>1$ certifies failure: there is no quasi-static state at this pressure. $L_{\rm hi}<1$ certifies existence. Otherwise the bracket is undecided at its present width.
- *The frozen pressure.* $\tilde p_i=\max(p_{0,i},p_{\min})$, where $p_0$ is the pressure of Section 8.3 with $\boldsymbol\tau=\mathbf T_{\rm tide}$ and no flow stress, that is, the body at rest under its loads. Under the orbital form of Section 10.3.1 with $s:=n^2$, both $\boldsymbol\ell$ and $p_0$ are affine in $s$: the Hill terms are $s$ times fixed fields, $\mathbf T_{\rm tide}$ is affine in $H_W$, and the pressure problem is linear. Two evaluations therefore give them for every orbit.
- *Two strengths.* $\tau^{(2)}=c+\mu_2\tilde p$ gives $L^*_2$, the threshold of Proposition 2.17.1 and the end of the quasi-static route. $\tau^{(s)}=c+\mu_s\tilde p$ gives $L^*_s\ge L^*_2$, the onset of whole-body yield at rest.

**Algorithm 2.17.2 (Failure orbit of a frozen body).** A satellite of frozen shape and density, on a synchronous circular orbit about a primary $GM_p$:

1. Build the operators of Chapter 8 once. Form $\boldsymbol\ell=\boldsymbol\ell_0+s\boldsymbol\ell_1$ and $p_0=p_{00}+s\,p_{01}$ from two evaluations.
2. At an orbit $a$, set $s=GM_p/a^3$ and run the bracket until it is decided or its relative width is below $\varepsilon_L$.
3. Scan $a_j=a_{\max}(1-\delta)^j$ downward to the first certified failure. Then bisect in $\log a$ between the smallest certified-stable and the largest certified-failed orbit, down to a relative width $\varepsilon_a$. An undecided orbit ends the bisection: the failure orbit is then known only to the interval between the two certified ends.
4. Report:
   - the certified interval;
   - the bracket along the scan;
   - the normalized shear rates $\dot\gamma_i(\mathbf u_0)/N(\mathbf u_0)$ of the mechanism at the failed orbit. Their rigid blocks are the fragments of a breakup.

Monotonicity of $L^*$ in $a$ is not assumed. A crossing narrower than the scan step can be missed, and the step is reported. Repeating the algorithm with Davis' reduced strength $(c\cos\phi,\sin\phi)$, $\tan\phi=\mu$, gives the other end of the interval of Conjecture A.2.

In a run under A-O (Section 10.3.6), the bracket with $\tau^{(2)}$ is evaluated on the deep state every $N_L$ steps. A certified $L_{\rm lo}>1$ ends the run as a limit-load collapse; the bracket at the rest pressure $p_0$ thus warns of a failure before the iteration of Section 10.3.2 meets it at its own pressure. Under A-Λ the tidal cycle replaces $L^*$ by the peak multiplier $L_{\rm peak}$ and adds the shakedown multiplier $L_{\rm sd}\ge L_{\rm peak}$ (Proposition 10.3.7.5). The scan of Algorithm 2.17.2 then returns two orbits: where shakedown first fails, and where the peak load first collapses the body. Between them lies the ratchet band.

---

# 3. GPU ray integration and hybrid acceleration

## 3.1 The unified ray formula

For field point $x$ and direction $u_q$, set $y=x-Ru_q$ and

$$
D_k(R)=(R-b_k)^2+d_k^2,\qquad
b_k=(x-c_k)\cdot u_q,\qquad
d_k^2=a_k^2+|x-c_k|^2-b_k^2,\qquad a_k=\sigma_k^{-1}.
$$

The three radial integrals along one segment are

$$
I_U^{(k)}=\int_{R^-}^{R^+}R\,D_k^{-\alpha_k}\,dR,\qquad
I_g^{(k)}=\int_{R^-}^{R^+}D_k^{-\alpha_k}\,dR,\qquad
I_H^{(k)}=\int_{R^-}^{R^+}\frac{D_k^{-\alpha_k}}{R}\,dR,
$$

from $dy=R^2dR\,d\Omega$ and $x-y=Ru$: the potential gets $R^2/R$, the gradient $R^2/R^2$, and the Hessian $R^2/R^3$ after the angular factor $T(u)$ is removed. Hence

$$
\boxed{
U(x)=G\sum_{q,\ell,k}w_k\sigma_k^{-2\alpha_k}\omega_q I_{U,\ell}^{(k)},\qquad
\nabla U(x)=-G\sum_{q,\ell,k}w_k\sigma_k^{-2\alpha_k}\omega_q u_q I_{g,\ell}^{(k)},\qquad
H(x)=G\sum_{q,\ell,k}w_k\sigma_k^{-2\alpha_k}\omega_qT(u_q)I_{H,\ell}^{(k)},
}
$$

with $T(u)=3uu^T-I$ (Section 1.3), $\omega_q$ the angular quadrature weight (not the spin), and $\ell$ over every segment of the ray (Section 1.4). For an anisotropic core (Lemma 1.4′), replace $\sigma_k^{-2\alpha_k}$ by $s_k(u_q)^{-\alpha_k}$ and use the direction-dependent $b_k,d_k$. The three quantities share the direction, segment, coefficients, and density parameters, so they are never computed by three separate programs. The $I_H$ column applies as written only to segments with $R^->0$. A segment $[0,R_1]$ (every interior or boundary field point) contributes to the Hessian only through the subtraction of Section 3.9, under the route rule stated there.

---

## 3.2 Analytic antiderivatives for $\alpha=1$

With $D(R)=(R-b)^2+d^2$ and $D'=2(R-b)$,

$$
\boxed{
P(R)=\frac12\log D(R)+\frac{b}{d}\arctan\frac{R-b}{d}
}\quad(P'=R/D),\qquad
\boxed{
Q(R)=\frac1d\arctan\frac{R-b}{d}
}\quad(Q'=1/D).
$$

For the Hessian, with $\widetilde A_0=b^2+d^2$, the partial fractions are $\frac{1}{R D}=\frac{1}{\widetilde A_0R}+\frac{-R+2b}{\widetilde A_0D}$ (common numerator $\widetilde A_0$). Since $-R+2b=-(R-b)+b$,

$$
\boxed{
L(R)
=
\frac1{\widetilde A_0}\log R
-\frac1{2\widetilde A_0}\log\left((R-b)^2+d^2\right)
+
\frac{b}{\widetilde A_0d}
\arctan\frac{R-b}{d}.
}
$$

$L$ differs from the antiderivative of Section 1.6 by the constant factor that turns $1+\sigma^2|x-Ru-c|^2$ into $\sigma^2D$. For an ordinary segment, $I_U=[P]$, $I_g=[Q]$, and $I_H=[L]$, evaluated between $R^-$ and $R^+$: a few logarithms, arctangents, and multiply-adds per ray. A segment with $R^-=0$ must not use $[L]$ directly; it uses the finite part of Sections 1.6 and 3.9.

---

## 3.3 GPU quadrature for general $\alpha$

For $\alpha\ne1$ no adaptive integrator runs on the GPU. Two facts make a fixed rule uniformly accurate.

**(i) The potential integral is never quadratured.**

$$
\boxed{
I_U=\int_{R^-}^{R^+}R\,D^{-\alpha}\,dR
=b\,I_g+\frac{D(R^+)^{1-\alpha}-D(R^-)^{1-\alpha}}{2-2\alpha}
}
$$

(write $R=(R-b)+b$, with $\frac{d}{dR}D^{1-\alpha}=2(1-\alpha)(R-b)D^{-\alpha}$). For $\alpha=1$ the same splitting gives $P$. Also $I_g=d^{1-2\alpha}[F(t^+)-F(t^-)]$, with $t^\pm=(R^\pm-b)/d$ and $F(t)=t\,{}_2F_1(\tfrac12,\alpha;\tfrac32;-t^2)$, usable with a verified hypergeometric evaluator. Only $I_H$ really needs quadrature (it is the Lauricella/Appell integral of Section 1.11).

**(ii) Tangent substitution.** With $R=b+d\tan\theta$, $\theta^\pm=\arctan((R^\pm-b)/d)\in(-\tfrac\pi2,\tfrac\pi2)$, $dR=d\sec^2\theta\,d\theta$, and $D=d^2\sec^2\theta$,

$$
I_g=d^{1-2\alpha}\int_{\theta^-}^{\theta^+}\cos^{2\alpha-2}\theta\,d\theta,
\qquad
I_H=d^{1-2\alpha}\int_{\theta^-}^{\theta^+}\frac{\cos^{2\alpha-1}\theta}{b\cos\theta+d\sin\theta}\,d\theta,
$$

where $b\cos\theta+d\sin\theta=R\cos\theta>0$ for $R^->0$. The $I_g$ integrand no longer contains $d$ or $\sigma_k$, and the $I_H$ integrand only through $\cos\theta/R$, which is smooth away from $R=0$. The Lorentzian peak of width $d$ at $R=b$, which a fixed Gauss rule in $R$ misses when $d\ll R^+-R^-$ (a narrow dense core), becomes a smooth function of $\theta$. One $n_r$-point Gauss–Legendre rule in $\theta$ serves both $I_g$ and $I_H$, which share $\cos^{2\alpha-2}\theta_j$. For $\alpha\ge1$ the $I_g$ integrand is smooth on $[-\tfrac\pi2,\tfrac\pi2]$. For $\alpha<1$ it is singular at $\pm\tfrac\pi2$ (integrably only for $\alpha>\tfrac12$). The interval $[\theta^-,\theta^+]$ always lies strictly inside $(-\tfrac\pi2,\tfrac\pi2)$, but it comes close to the singularity when $|R^\pm-b|\gg d$.

The **composite rule**, valid for every $\alpha>0$, splits at $R=b\pm\kappa d$ (default $\kappa=4$). The inner part uses the tangent rule, with $\theta\in[-\arctan\kappa,\arctan\kappa]$. Each outer part, where $D$ has no peak, uses Gauss–Legendre in $\log|R-b|$, which maps the power behaviour to a smooth exponential and stays regular near small $R$. With 16 nodes per part the relative errors are $1.1\times10^{-8},2.3\times10^{-9},1.1\times10^{-10}$ in $I_g$ and $7.7\times10^{-8},3.4\times10^{-8},1.2\times10^{-8}$ in $I_H$ for $\alpha=0.3,0.6,0.9$. For the pure tangent rule ($n_r=16$, $b=0.7$, $d=0.05$, $[0.2,3]$) and $\alpha=0.6,1.3,2.5$:

- plain Gauss–Legendre in $R$ gives $6.8\times10^{-2},2.1\times10^{-1},1.7\times10^{-1}$ in $I_g$;
- the tangent rule gives $1.2\times10^{-3},1.6\times10^{-6},1\times10^{-15}$ in $I_g$ and $9.4\times10^{-4},6.2\times10^{-5},1.9\times10^{-7}$ in $I_H$.

These are a-posteriori observations, not certificates (Section 1.12). The cost per (segment, kernel) is one $\arctan$ pair, $n_r$ evaluations of $\cos^{2\alpha-2}$ with one division each, and a closed-form $I_U$. That is about two thirds of three independent quadratures, with accuracy independent of the core width.

These rules apply only to nonsingular segments with $R^->0$. A segment starting at $R^-=0$ (interior field point), or on a boundary or near-field singular branch, goes to the subtraction of Section 3.9. Whether the contact term appears explicitly depends on the route; the route rule of Section 3.9 fixes one route per evaluation.

Configuration:

- nodes and weights are precomputed for fixed $\alpha$;
- $\alpha=1$ takes the analytic branch;
- $n_r=8,16,32$ are three accuracy levels;
- only a few difficult rays are recomputed at higher order.

---

## 3.4 GPU BVH geometric pipeline (optional hardware ray-tracing backend)

Sections 3.5–3.8 describe the pipeline. Section 3.15 adds caching, incremental updates, and hardware ray tracing.

## 3.5 Preprocessing

Build a flat BVH for the triangle mesh (or an equivalent hardware acceleration structure). Store vertices, face normals, boxes, and edge data as one array per component (structure of arrays), so that neighbouring threads read neighbouring entries. Reuse the BVH while the shape is fixed.

---

## 3.6 Ray tasks

For field point $x_m$ and direction $u_q$, cast $r(t)=x_m-tu_q$, $t>0$. RayBackend returns all distances $0<R_1<\dots<R_L$. Exterior points usually give pairs. The first hit from an interior point is an exit, with $[0,R_1]$ inside. Nonconvex bodies give several segments. Intersections must therefore not be paired blindly as $[R_1,R_2],[R_3,R_4],\dots$. Fix the occupancy at $R=0$ from the interior/exterior status first:

$$
\begin{cases}
[R_1,R_2],[R_3,R_4],\ldots,&x\notin V,\\[2pt]
[0,R_1],[R_2,R_3],[R_4,R_5],\ldots,&x\in\operatorname{int}V.
\end{cases}
$$

A boundary point is first fixed to the spherical representative of Sections 1.2–1.3 (interior limit, exterior limit, or an explicit near-field branch), never forced into either rule.

**Orientation-based segment construction (mandatory when hits are out of order or duplicated, the normal case for hardware ray tracing).** Parity fails when a ray passes through a shared edge or vertex, because the hit is then reported twice or not at all. Record $\varsigma_j=\operatorname{sign}(\mathbf u\cdot\mathbf n_{f_j})$ per hit: since the ray direction is $-\mathbf u$, $+1$ means entering and $-1$ leaving.

1. Sort by $R$ (insertion sort in registers; $L$ is small).
2. Merge hits within $\epsilon_{\rm hit}=c\,\epsilon_{\rm mach}\max(R_j,h_\Gamma)$ **with equal sign** (a duplicate across an edge). Coincident hits of opposite sign are a grazing pair enclosing a zero-length segment, which is dropped.
3. Start from the field point's status and require alternation: $-,+,-,\dots$ from inside, $+,-,+,\dots$ from outside.
4. Two equal consecutive signs after merging mean a lost hit. Re-trace that ray in the software BVH in double precision, and count the event.

On a closed, consistently oriented, watertight mesh, alternation is an invariant (the winding number changes by $\pm1$ at each transversal crossing), so step 4 detects every inconsistency.

**Simplified construction (parity after merging).** Without orientation signs, sort the crossings and merge those closer than a relative tolerance $\epsilon_{\rm hit}(1+R_j/R_*)$. A ray ends outside the body, so the parity of the merged count fixes the occupancy at $R=0$: an odd count starts inside, $[0,R_1],[R_2,R_3],\dots$, and an even count gives $[R_1,R_2],\dots$. This is correct whenever each transversal crossing is reported exactly once. A lost or duplicated crossing (a watertightness failure at an edge or vertex) is not detected and mispairs the rest of the ray, so the simplified construction is admissible only together with a watertight intersection test; steps 1–4 remove that condition. Under the split of Section 3.9 only the density-difference rays are affected.

Threads must not allocate memory dynamically. The general scheme has three stages:

- **Stage A, counting:** each ray counts its intersections $L$.
- **Stage B, prefix sum:** a GPU prefix sum over the rays allocates one contiguous buffer.
- **Stage C, writing:** the query is run again, and the hits are written into the packed array.

This avoids wasting memory on fixed-length buffers.

---

## 3.7 Organization of the GPU kernels

One workgroup takes a batch of field points over a fixed set of directions, with shared batch parameters ($x$, $c_k$, weights) in workgroup memory. One invocation takes one direction and all its segments: it reads $[R^-,R^+]$, forms $u_q$ and $b_k,d_k^2$, computes $I_U,I_g,I_H$, and accumulates $U,\nabla U,H$. Partial sums are reduced hierarchically after a barrier, or by subgroup operations under the same error verification.

---

## 3.8 Fused kernel

One kernel computes all three fields; potential, gradient, and Hessian are never evaluated by separate passes:

```text
for each field point x:
    U_local = 0; g_local = (0,0,0); H_local = 0
    # x in closed V: the rays carry rho(y) - rho(x) (Section 3.9, route (i)); x outside V: rho(y)
    for each ray direction q:
        intersections = RT_query(x, u[q])
        for each segment [R0, R1]:
            for each density component k:
                compute b, d2
                if R0 == 0:                                          # interior or boundary point
                    IU, IG, IH = remainder_quotient(...)             # Section 3.9, whole quotient
                elif alpha[k] == 1:
                    IU, IG, IH = analytic(...)                       # Section 3.2
                else:
                    IG, IH = tangent_or_composite_rule(...)          # Section 3.3 (ii)
                    IU = b*IG + (D(R1)^(1-a) - D(R0)^(1-a))/(2-2a)  # Section 3.3 (i)
                U_local += coeff * weight[q] * IU
                g_local += -coeff * weight[q] * u[q] * IG
                H_local += coeff * weight[q] * T(u[q]) * IH
    if x in closed V:
        add rho(x) * (Phi_V, grad Phi_V, Hess Phi_V)(x)              # Section 3.9, route (i)
    write U_local, g_local, H_local
```

$T(u)=3uu^T-I$ needs only the six components $H_{xx},H_{yy},H_{zz},H_{xy},H_{xz},H_{yz}$. Symmetry follows from equal mixed derivatives in the exterior and from $T_{ij}=T_{ji}$ for the finite part.

---

## 3.9 Near-surface finite part

Every segment starting at $R=0$ is singular for $I_H$ (integrand $D^{-\alpha}/R$). This means every interior field point and every boundary point. For an interior point take $r_0=\min\{r_\ast(x),\operatorname{dist}(x,\partial V)\}$; then $\Gamma_x=\mathbb S^2$, $C=0$, and $\vartheta=1$. On a face, $C=0$ and $\vartheta=\tfrac12$ (Proposition 2.7.1). Only at edges and vertices does the logarithm survive. Two forms of the subtraction exist, and the route rule at the end of this section allows exactly one per evaluation.

**Local form (alternative).** Subtract at $r_0$ over the tangent-cone directions $\Gamma_x$:

$$
I_H^{\mathrm{near}}
=
\int_{\Gamma_x}T(u)\int_\varepsilon^{r_0}\frac{\rho(x-Ru)}{R}\,dR\,d\Omega
=
\rho(x)C(x)\log\frac{r_0}{\varepsilon}
+
\int_{\Gamma_x}T(u)\int_\varepsilon^{r_0}\frac{\rho(x-Ru)-\rho(x)}{R}\,dR\,d\Omega,
$$

the angular form of Section 1.5 before the factor $G$ and the contact term. On the GPU:

1. the part of a ray with $R<r_0$ enters the near kernel;
2. the density difference is computed and integrated by a fixed rule, using the limit $-\nabla\rho(x)\cdot u$ at $R=0$ (never $0/0$);
3. $G\rho(x)C(x)\log(r_0/\ell_0)$ and the contact term $-\frac{4\pi G}{3}\vartheta_x\rho(x)I$ are added explicitly.

$\log\varepsilon$ is never formed on the GPU.

**Global form of the subtraction (primary route).** The primary kernel subtracts over the whole body. With $\Phi_V(x)=\int_V|x-y|^{-1}dy$, the unit-density field of the polyhedron,

$$
U(x)=G\rho(x)\,\Phi_V(x)+G\int_V\frac{\rho(y)-\rho(x)}{|x-y|}\,dy,
$$

and likewise for $\nabla U$ and $H$, differentiating only the kernel. The constant part carries the whole singular structure at $x$: the contact term $-\tfrac{4\pi}{3}\vartheta_x\rho(x)\mathbf I$, the edge or vertex logarithm, and the jump across a face. $\Phi_V$, $\nabla\Phi_V$, and $\nabla\nabla\Phi_V$ are closed form (Werner and Scheeres [7]): sums over faces of solid angles $\omega_f$ and over edges of $L_e=\ln\frac{a+b+e}{a+b-e}$. For a point inside face $f$, the convention of Section 1.3 takes the mean of the one-sided limits, which sets $\omega_f=0$ for that face; then $\sum_f\omega_f=-\operatorname{tr}\nabla\nabla\Phi_V=4\pi\vartheta_x$. The remainder has integrands $(\rho(x-Ru)-\rho(x))R$, $\rho(x-Ru)-\rho(x)$, and $(\rho(x-Ru)-\rho(x))/R$, which are finite at $R=0$. Its ray integrals need no finite part, contact term, or logarithm. For the potential and gradient the two pieces separate on every segment: Section 3.3 integrates $\rho(\mathbf x-R\mathbf u)R$ and $\rho(\mathbf x-R\mathbf u)$, and $\rho(\mathbf x)R$, $\rho(\mathbf x)$ integrate in closed form. For the Hessian the pieces separate only on segments with $R^->0$: there $\int\rho(\mathbf x-R\mathbf u)R^{-1}dR$ follows Section 3.3 and $-\rho(\mathbf x)\log(R^+/R^-)$ is exact. On a segment $[0,R_1]$ each piece alone diverges, and the quotient $(\rho(\mathbf x-R\mathbf u)-\rho(\mathbf x))/R$ is integrated as a whole, kernel by kernel. With $D_k$ and $A_0=D_k(0)$ in the normalization of Section 1.6:

- $\alpha_k=1$: $\displaystyle\int_0^{R_1}\frac{D_k(R)^{-1}-A_0^{-1}}{R}\,dR=\frac1{A_0}\Big[-\frac12\log\frac{D_k(R_1)}{A_0}+\frac bd\Big(\arctan\frac{R_1-b}{d}+\arctan\frac bd\Big)\Big]$, the finite part of Section 1.6 with $\ell_0=R_1$;
- general $\alpha_k$: Gauss–Legendre in $R$ on the quotient, whose value at $R=0$ is the limit $-\nabla\rho_k(\mathbf x)\cdot\mathbf u$, with the split at $R=b\pm\kappa d$ of Section 3.3 when a narrow core lies on the segment (on the parts away from $R=0$ the pieces separate again).

The angular rule then sees only the smooth variation of $\rho$, not the kinks of constant-density chord lengths. With $\rho\equiv\rho(x)$, a degree-29 Lebedev rule alone leaves about $2\times10^{-2}$ relative error in $H$ at a cube's centre, and the split removes it exactly. A uniform density casts no rays.

**Route rule.** One evaluation of the Hessian at a point of $\overline V$ uses exactly one of three routes. (i) The global subtraction is the normative RTFP route: the contact term, the face jump, and the edge logarithms enter only through $\rho(\mathbf x)\nabla\nabla\Phi_V$, and nothing is added to the remainder. (ii) The local form adds $G\rho(\mathbf x)C(\mathbf x)\log(r_0/\ell_0)$ and the contact term once, and $\Phi_V$ is then not used. (iii) The Feynman route of Section 2.7 never adds the contact term (Lemma 2.7.0). Combining (i) or (iii) with an explicit contact term counts it twice. Applying the boxed $I_H$ of Section 3.1 to a segment $[0,R_1]$ outside these routes drops it, and diverges besides.

---

## 3.10 Far-field multipole acceleration

Rays cost $O(N_{\rm ray})$ per point, independent of the face count $F$. The closed-form uniform part costs $O(F)$ per point, and that part is accelerated by a tree in the spirit of the FMM [2]: a Barnes–Hut traversal with octupole expansions, without local expansions.

**Surface charges.** From $\nabla_y\cdot((y-x)/|y-x|)=2/|y-x|$, for any point $c$,

$$
\Phi_V(x)=\tfrac12\big(P(x)-(x-c)\cdot\mathbf S(x)\big),\qquad
\mathbf S(x)=\oint_{\partial V}\frac{\mathbf n}{|x-y|}\,dS_y,\qquad
P(x)=\oint_{\partial V}\frac{\mathbf n\cdot(y-c)}{|x-y|}\,dS_y,
$$

with $\nabla\Phi_V=-\mathbf S$ and $\nabla\nabla\Phi_V=-\nabla\mathbf S$. These are single layers of four charge densities, $n_1,n_2,n_3$ and $\mathbf n\cdot(y-c)$, each constant on a face.

**Tree.** Faces are split recursively at the median centroid along the longest axis, down to leaves of at most four faces. Each node stores its area-weighted centre $c_\nu$, its radius $h_\nu$, and, per charge type (type 3 about $c_\nu$), the moments $Q^{(k)}_{t,i_1\dots i_k}=\int_\nu q_t\,\delta_{i_1}\cdots\delta_{i_k}\,dS$, $\delta=y-c_\nu$, $k\le3$. Leaves integrate exactly with $\int_f\lambda_1^a\lambda_2^b\lambda_3^c\,dA=2A_f\,a!\,b!\,c!/(a+b+c+2)!$. Parents use the translation $\delta=\delta'+s$: $Q^{(1)}=Q'^{(1)}+sQ'^{(0)}$, $Q^{(2)}_{ab}=Q'^{(2)}_{ab}+s_aQ'^{(1)}_b+s_bQ'^{(1)}_a+s_as_bQ'^{(0)}$, and similarly for the octupole, after re-centring the type-3 density. The build is $O(F\log F)$ and the moments $O(F)$.

**Evaluation.** With $r=x-c_\nu$, $R=|r|$, $a_2=r^{\top}Q^{(2)}r$, $w=Q^{(3)}(\cdot,r,r)$, $v_k=Q^{(3)}_{iik}$, and $a_3=r\cdot w$,

$$
\phi=\frac{Q^{(0)}}{R}+\frac{r\cdot Q^{(1)}}{R^3}+\frac{3a_2-R^2\operatorname{tr}Q^{(2)}}{2R^5}+\frac{5a_3-3R^2\,v\cdot r}{2R^7},
$$

with the analytic gradient. A node with $R\theta>h_\nu$ contributes its expansion. Otherwise its children are visited, and leaf faces are summed in closed form with the own-face convention of Section 3.9. The truncation error of a node falls as $(h_\nu/R)^4<\theta^4$, and at fixed $\theta$ the number of interactions per point grows like $O(\log F)$ instead of $F$. Since $\operatorname{tr}\nabla\nabla\Phi_V=-\sum_f\omega_f$, a far node contributes $-\operatorname{tr}$ of its Hessian to the inside test's solid angle. The ray part and the tree share no data, and both parallelize over field points.

---

## 3.11 Hybrid scheduling

With $d(x,V)$ the distance to the geometry and thresholds $d_{\rm near}<d_{\rm far}$:

$$
\begin{cases}
d<d_{\mathrm{near}}:&\text{RTFP + finite part};\\
d_{\mathrm{near}}\le d\le d_{\mathrm{far}}:&\text{RTFP, full rays};\\
d>d_{\mathrm{far}}:&\text{GPU FMM/multipole}.
\end{cases}
$$

This beats Feynman face integrals at every point, because those still need a $u$-integral, a face integral, the $u\to1$ endpoint, and an inner quadrature. *Where the cost lies.* Every SCF field point (surface nodes, face centres, $\mathcal X_{\rm vol}$, $\mathcal X_Y$) lies in $\overline V$, so $d=0$ and the far branch is used only for exterior diagnostics. The production cost is RTFP, dominated by traversal, not radial integration (a few transcendentals per segment for $\alpha=1$). The effective accelerations therefore **reduce traversals**: reuse of segment tables across density updates and steps, and incremental retracing (Section 3.15).

---

## 3.12 GPU execution inside the SCF

**The only authoritative procedure is the nine-step loop of Section 10.3.4.** On the GPU, steps 1 and 9 compute $U$, $\nabla U$, and $H$ by batched RayBackend intersection and the fused kernel (Sections 3.6–3.8) on the current mesh and BVH. $W$, $\mathbf g_{\rm eff}$, and $H_{\rm eff}$ are assembled as in Section 2.8. Steps 2–8 (transport, mass recovery, SOCP, remapping) are specified in Chapters 9–10 and are not part of the RTFP kernel.

---

## 3.13 Complexity

With $N_x$ field points, $N_\Omega$ directions, $F$ triangles, $K$ kernels, $L$ segments per ray, and $n_r$ radial nodes, RTFP costs about

$$
\boxed{
O\left(N_xN_\Omega(\log F+L)Kn_r\right),
}
$$

with $n_r=1$ for analytic $\alpha=1$. The SCF total is $O(N_{\rm iter}N_xN_\Omega(\log F+L)K)$ plus the geometric update, mass calibration, and projection. These are planning estimates, not runtime certificates.

---

## 3.14 Algorithmic rules

$$
\boxed{
\text{Lebedev nodes}
+
\text{flat multi-hit BVH}
+
\text{RTFP analytic radial formulae}
+
\text{fused }U/\nabla U/H\text{ kernel}
+
\text{closed-form uniform part (3.9)}
+
\text{its face tree (3.10)}.
}
$$

The Feynman formula serves as an independent exterior channel, a face-integral evaluation, and a cross-check of RTFP; it is not the fastest primary algorithm. RayBackend does intersection, and the compute kernels do the radial formulae and tensor accumulation. Matrix hardware is used only for reductions with verified error, and the finite-part Hessian accumulation is never handed entirely to a low-precision path.

## 3.15 Data reuse, incremental retracing, and hardware ray tracing

This section is normative for efficiency only. It changes no value of Sections 3.1–3.9: each rule reuses an unchanged intermediate result or reorders the same arithmetic. The coarsest admissible form recasts every ray after a geometric update and reuses only results whose inputs are unchanged: the face-centre fields on a spin-only step, the Stokes operators and interior gravity while geometry and density are unchanged, and no rays at all for a uniform body (Section 3.9).

### 3.15.1 Factorization of the field evaluation

The evaluation at a point set $\mathcal X$ has two stages with disjoint inputs:

- **geometric stage:** from the mesh, $\mathcal X$, and $\{\mathbf u_q\}$, build the *segment table* $\mathcal T=\{(m,q,\ell,R^-_{mq\ell},R^+_{mq\ell},\text{flags})\}$ (Sections 3.6, 3.9). It uses no density parameter;
- **density stage:** from $\mathcal T$ and $\theta=\{w_k,\mathbf c_k,\mathbf S_k,\alpha_k\}$, evaluate the radial integrals and angular sums.

**Proposition 3.15.1 (Reuse).**

(a) If only $\theta$ changes, the fields are recomputed from $\mathcal T$ without traversal.

(b) The fields are affine in the weights: $U=\rho_b\Phi_V+\sum_kU_k$ and likewise for $\nabla U$, $H$. The subtraction of Section 3.9 keeps this split, since $\rho(\mathbf x)\Phi_V=\rho_b\Phi_V+\sum_k\rho_k(\mathbf x)\Phi_V$ and the density difference is a sum over kernels. With per-kernel partial sums $U_k,\nabla U_k,H_k$ stored, a weight change costs $O(|\mathcal X|K)$; the mass-closure scaling $w_k\mapsto(1+\xi)w_k$ of Section 9.2 multiplies the core part $\sum_kU_k$ by $1+\xi$ and leaves $\rho_b\Phi_V$ unchanged. Both statements concern fields on an unchanged geometry; after a geometric update the fields are re-evaluated (Proposition 3.15.2).

(c) $\mathbf S_k\mathbf u_q$ and $s_k(\mathbf u_q)$ depend only on $(q,k)$, and $\mathbf p_k^{\!\top}\mathbf S_k\mathbf p_k$ only on $(m,k)$. After $O((N_\Omega+|\mathcal X|)K)$ precomputation, $b_k$ and $d_k^2$ cost one dot product per (segment, kernel) for an anisotropic core, as for an isotropic one.

*Proof.* (a) and (c) follow from which inputs enter which formula (Sections 1.4, 3.1–3.3, Lemma 1.4′), and (b) from linearity in $\rho_K$. ∎

So the core transport of Section 9.2, which changes $\theta$ but not the mesh, never triggers a traversal by itself.

### 3.15.2 Incremental retracing after a geometric update

**Proposition 3.15.2 (Dirty-ray criterion).** Let $V,V'$ be closed meshes with the same connectivity, let $\mathcal F_{\rm chg}$ be the triangles with a moved vertex, and let $f,f'$ be the old and new positions. If a ray $\mathbf x-R\mathbf u$ from an unmoved field point meets no triangle of $\mathcal F_{\rm chg}\cup\mathcal F'_{\rm chg}$, then $E_{\mathbf x}(\mathbf u)$, including the status of $\mathbf x$, is the same for $V$ and $V'$.

*Proof.* Hits with unchanged triangles coincide, and there are no others. A change of status would put $\mathbf x$ in $V\triangle V'$, whose boundary lies in $\bigcup(f\cup f')$. A ray to infinity would then have to cross some $f$ or $f'$. ∎

*Algorithm* (after step 8a of Section 10.3.4):

1. mark as dirty every moved field point: vertices with $d_{*,i}\ne0$, their incident faces, and the volume nodes on rays from $O$ that the test marks (Section 8.1, G2);
2. build a small structure $\mathcal A_{\rm chg}$ over the $2|\mathcal F_{\rm chg}|$ old and new triangles;
3. trace every cached ray of a clean point against $\mathcal A_{\rm chg}$ with a terminate-on-first-hit query;
4. retrace fully only the rays of dirty points and the rays that hit.

Since $d_*=0$ on the locked set, $|\mathcal F_{\rm chg}|$ is typically a small fraction of the mesh and most of $\mathcal T$ is reused. In stage equilibrium nothing is retraced.

### 3.15.3 Acceleration-structure maintenance

Connectivity is fixed for the run; only vertex positions change, along their normals (Section 8.1, G4). The BVH, or the hardware structure, is **refitted** bottom-up in $O(F)$ after step 8a instead of rebuilt; a refit changes traversal cost, never hits. Rebuild when the surface-area-heuristic cost exceeds that of the last build by a configured factor (e.g. $1.5$), and after every realignment 8b, which moves every vertex.

### 3.15.4 Mapping onto ray-tracing hardware

The ray-tracing units (single-precision traversal and watertight tests) serve:

1. **multi-hit queries** for $\mathcal T$: every candidate (triangle id, distance) is recorded, none committed, in a per-ray slot of capacity $L_{\rm cap}$. Overflowing rays go to the count–scan–write scheme; otherwise one pass replaces stages A–C;
2. the first-hit dirty test of 3.15.2;
3. interior/exterior classification by one ray and the orientation rule;
4. the volume, mass, and inertia integrals of 8a–8b, and the clearances $t_i^\pm$ and volume segments of Section 8.1;
5. the self-intersection test of Section 2.13, restricted to edges of triangles in $\mathcal F_{\rm chg}$, as segment queries with $t_{\max}$ equal to the edge length.

**Hit refinement (precision decoupling).** From the triangle id, the integration kernel recomputes the distance by one double (or double-float) Möller–Trumbore evaluation. The endpoints, and so $U,\nabla U,H$, then do not depend on the RT unit's precision, which only selects candidates. Misses or spurious candidates are caught by the alternation test. **Antipodal pairing.** Lebedev rules are symmetric under $\mathbf u\mapsto-\mathbf u$ with equal weights, and $T(-\mathbf u)=T(\mathbf u)$. Trace the full line once, from $\mathbf x-\Lambda\mathbf u$ outside the box with $t\in(0,2\Lambda)$, and split the hits at $t=\Lambda$. This halves the traversals.

### 3.15.5 Wavefront organization of the kernels

| Pass | Work item | Content |
|---|---|---|
| W1 | ray (line) | generation in **direction-major order** (all points of one $\mathbf u_q$ contiguous, so neighbouring threads trace nearly parallel rays); dirty filtering |
| W2 | ray | traversal, orientation-based segments, hit refinement; writes $\mathcal T$ |
| W3 | segment | per kernel (parameters and the tables of Proposition 3.15.1(c) in constant/shared memory): $b_k,d_k$, radial integrals (analytic for $\alpha_k=1$; closed-form $I_U$ and tangent/composite rule otherwise), multiplication by $\omega_q,\mathbf u_q,T(\mathbf u_q)$; near-field branch for $R^-=0$ |
| W4 | field point | segmented reduction in a **fixed order** (bitwise reproducible); six Hessian components in double or compensated single |

W1–W2 run only for dirty rays, W3 after any change of $\theta$ or $\mathcal T$, and W4 always. The segment table ($|\mathcal X|N_\Omega\bar L$ entries, two endpoints each) stays in device memory between steps. If memory is short, it is kept only for the largest point set ($\mathcal X_{\rm vol}$).

### 3.15.6 Cost per geological step

With $N_x^{\rm dirty}$ moved points, $N_{\rm chg}=|\mathcal F_{\rm chg}|$, and $N_{\rm hit}$ clean points with a dirty ray,

$$
C_{\rm grav}
\approx
\underbrace{c_{\rm any}\,\tfrac12N_xN_\Omega\log N_{\rm chg}}_{\text{dirty test}}
+\underbrace{c_{\rm trav}\,\tfrac12\big(N_x^{\rm dirty}+N_{\rm hit}\big)N_\Omega(\log F+\bar L)}_{\text{retrace}}
+\underbrace{c_{\rm seg}\,N_xN_\Omega\bar LK}_{\text{radial integrals}},
$$

against $c_{\rm trav}N_xN_\Omega(\log F+\bar L)+c_{\rm seg}N_xN_\Omega\bar LK$ for a full re-evaluation (planning estimates, Section 3.13).

---

# 4. The combined rheological model and its classical limit

## 4.0 Normalization of the deviatoric law

Main result of this chapter: under the limit conditions listed explicitly below, the combined model reduces strictly and term by term to the classical $\mu(I)$ law. When those conditions fail, the two models differ.

With $\dot\gamma=\sqrt{2D':D'}$ and $\tau_{\rm eq}=\sqrt{\tfrac12\tau:\tau}$, the law must be written

$$
\boxed{
\tau=2\eta D',
\qquad
\eta=\frac{\mu p}{\dot\gamma},
}
$$

so that $\tau_{\rm eq}=\mu p$. Indeed, $\tau:\tau=4(\mu p/\dot\gamma)^2(D':D')=2\mu^2p^2$, because $D':D'=\dot\gamma^2/2$. Dropping the factor $2$ would give $\tau_{\rm eq}=\mu p/2$.

## 4.1 Constitutive convention and the classical $\mu(I)$ model

The law concerns the flow (deviatoric) stress of an incompressible velocity, $\nabla\cdot v=0$. The reference model (Section 2.8) uses only this viscoplastic stress, with no elastic strain and no equation of state. The pressure is the multiplier of incompressibility and is advanced by the artificial-compressibility equation of Section 8.3. Under the rate-type dilatancy of A-Ξ (Section 11.3) the law concerns the deviator $D'$ of a velocity with $\nabla\cdot v=\dot\varepsilon_v$. There is still no equation of state, and the pressure is the multiplier of that constraint. The classical inertial rheology [1][11] is

$$
\boxed{
I=\frac{d\dot\gamma}{\sqrt{p/\rho_p}}\quad(p>0),
\qquad
\mu_{\mathrm{cl}}(I)=\mu_s+(\mu_2-\mu_s)\frac{I}{I_0+I},
\qquad
\tau_{\mathrm{eq}}=\mu_{\mathrm{cl}}(I)p,
\qquad
\tau=2\mu_{\mathrm{cl}}(I)p\frac{D'}{\dot\gamma},
}
$$

with the factor $2$ fixed by Section 4.0.

---

## 4.2 The combined model

The combined model of this chapter is general: $c=c(\nu,Z)$, $\mu_y=\mu_y(\nu,Z)$, and the pressure $p$. The reference model of Sections 2.8 and 2.17 is its specialization: $c=c(\mathbf x)$ (Section 4.3a), $p\to\tilde p$ everywhere (the rule of Section 4.13), and the yield function $\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}-(1-\delta_Y)\tau_y$ with the classification tolerance $\delta_Y$ of (B2). The reduction statements of Sections 4.7–4.20 are made for the general form and hold for the specialization with these substitutions. The combined model has four extensions:

- a cohesive pressure;
- a modified inertial number;
- a state-dependent friction coefficient;
- a tidal backstress.

Chapter 11 specializes two of them further. Under A-Ξ the state variable $\nu$ evolves, and the friction becomes $\mu_d=\mu_{\rm cs}(I_c)+\Xi(\nu,I_c)$ (Section 11.3). Under A-ϖ the cohesion becomes $c=(1-\varpi)c_{\rm intact}(\mathbf x)$ (Section 11.5).

## 4.3 Cohesive pressure

$$
p_c=C_c\frac{Z\nu F_H}{d^2},\qquad
\boxed{p_*=p+p_c,}
$$

with coordination number $Z$ and single-contact cohesive force $F_H$.

## 4.3a Two-layer strength: the fragile surface regolith

The macroscopic cohesion $c$ of a rubble pile is not uniform. The Hayabusa2 crater on Ryugu grew in the gravity regime, which bounds the surface strength below about $1$ Pa ([9]; a tensile strength near $0.5$ Pa, [12]). The western bulge, by contrast, requires an interior cohesion of 4–10 Pa [10]. The cohesion is therefore a field of the depth below the surface,

$$
\boxed{
c(\mathbf x)=c_b+(c_s-c_b)\,e^{-d(\mathbf x)/h_c},\qquad
d(\mathbf x)=\operatorname{dist}(\mathbf x,\Gamma),
}
$$

with configured interior cohesion $c_b\ge0$, surface cohesion $c_s\ge0$, and transition depth $h_c>0$.

1. $c=c_s$ on $\Gamma$, and $c\to c_b$ with depth, monotonically, between the two constants.
2. $c$ is bounded and smooth away from the medial axis, so the $\eta$ bounds of Section 2.8 hold with $\min(c_s,c_b)$ in the lower bound and $\max(c_s,c_b)$ in the upper bound.
3. **Reduction.** If $c_s=c_b$, then $c\equiv c_b$ and the single-layer theory is recovered. Condition E (Section 4.12) then applies to both constants.

**Gradient** (needed through $\nabla\eta$ in Section 8.3):

$$
\nabla c=c'(d)\,\nabla d,\qquad c'(d)=-\frac{c_s-c_b}{h_c}\,e^{-d/h_c},\qquad
\nabla d=-\mathbf n\big(\pi(\mathbf x)\big)\quad\text{a.e.},
$$

with $\pi$ the closest point of $\Gamma$. $|\nabla d|=1$ off the medial axis, a null set on which $\nabla c$ jumps, so $\nabla c\in L^\infty$, which is all Section 8.3 needs. $d$ and $\pi$ are BVH closest-point queries (Section 8.1, G6). A radial depth $R(\hat{\boldsymbol\omega})-r$ is not used: it exists only on star-shaped bodies, and even there it only bounds $d$ from above.

**Where $c$ is evaluated.**

- **Viscosity and yield (B1, B2):** $c(\mathbf x)$ at each volume point and on $\mathcal X_Y$, with $d=\operatorname{dist}(\mathbf x,\Gamma)$.
- **Traction and the surface chain of Section 10.2:** $d=0$, so $c=c_s$.
- **Contact pressure $p_c$:** it is a grain-contact quantity and is not layered.

The profile is constitutive hypothesis **A-C2**. It is not derived from a contact model. Its constants are taken from published measurements of the body where available. Under A-ϖ (Section 11.5) the profile is the intact strength $c_{\rm intact}(\mathbf x)$, and the continuum uses $c=(1-\mathcal R\varpi)c_{\rm intact}$. Under A-ϖ or A-Θ (Section 11.6), the surface chain of Section 10.2 and A-C4 use the shallow cohesion $c_\Gamma=(1-\mathcal R\varpi)(1-\varpi_\Gamma)c_s$ in place of $c_s$.

## 4.4 Modified inertial number

$$
\boxed{
I_c=\frac{d\dot\gamma}{\sqrt{p_*/\rho_p}},
}
$$

the classical $I$ with $p$ replaced by $p_*$. It requires $p_*>0$.

## 4.5 State-dependent friction coefficient

$$
\boxed{
\mu_d(I_c,\nu,Z)
=
\mu_s(\nu,Z)
+
[\mu_2(\nu,Z)-\mu_s(\nu,Z)]
\frac{I_c}{I_0(\nu,Z)+I_c}.
}
$$

In the reference model $\nu=\nu_0$ and $Z=Z_0$ are frozen (Section 10.2). Under A-Ξ, $Z=Z_0$ stays frozen, and the $\nu$-dependence takes the stress–dilatancy form $\mu_d(I_c,\nu)=\mu_{\rm cs}(I_c)+K_\Xi(\nu-\nu_c(I_c))$. Here $\mu_{\rm cs}$ is the law above at $(\nu_0,Z_0)$, read as the critical-state friction (Section 11.3).

## 4.6 Tidal backstress

With $H_W=\nabla^2W$ and $\operatorname{dev}H_W=H_W-(\operatorname{tr}H_W)I/3$,

$$
\boxed{
T_{\mathrm{tide}}=C_T\rho_p\nu d^2\operatorname{dev}H_W,
\qquad
F_{\mathrm{comb}}
=
\sqrt{\tfrac12(s-T_{\mathrm{tide}}):(s-T_{\mathrm{tide}})}
-c(\nu,Z)-\mu_y(\nu,Z)p,
}
$$

with $s:=\operatorname{dev}\sigma$. For consistency with the dynamic law set $s=T_{\rm tide}+\tau_{\rm flow}$, so that $s-T_{\rm tide}=\tau_{\rm flow}$ identically, not merely in a limit. The flow branch is

$$
\boxed{
\tau_{{\mathrm{flow}},\mathrm{eq}}=c(\nu,Z)+\mu_d(I_c,\nu,Z)p,
\qquad
\tau_{\mathrm{flow}}=2\eta D',
}
$$

with $\eta$ chosen by Section 4.0 so that $\|2\eta D'\|_{\rm eq}$ equals the scalar target. Under the regularizations of Section 4.17 this equality holds only as $\dot\gamma_\varepsilon/\dot\gamma\to0$.

---

## 4.7 Conditions for reduction

Consider a sequence $(p_c^{(n)},H_W^{(n)},c^{(n)},\nu^{(n)},Z^{(n)})$ with $p>0$ and $\dot\gamma>0$ in the region considered. The index $n$ labels this parameter sequence, not an SCF iteration. In the reference model, $\nu$ and $Z$ are frozen constants (Sections 10.2, 10.3); the sequence is the hypothesis of the reduction theorem. Under A-Ξ, $\nu$ evolves (Section 11.3). Condition C then holds along a sequence with $K_\Xi\to0$, or at the critical state $\nu=\nu_c(I_c)$, where $\Xi=0$; under A-ϖ, Condition E is imposed on $c=(1-\varpi)c_{\rm intact}$.

## 4.8 Condition A: disappearance of cohesion in the pressure

$$
\boxed{\frac{p_c^{(n)}}p\longrightarrow0,}
$$

so $p_*^{(n)}\to p$, since $p$ stays positive (or has a positive limit, Section 4.20).

## 4.9 Condition B: disappearance of the tidal stress

$$
\boxed{
\frac{\rho_p\nu^{(n)}d^2\|\operatorname{dev}H_W^{(n)}\|}{p}\longrightarrow0,
}
$$

so $T^{(n)}_{\rm tide}\to0$, since $\|T^{(n)}_{\rm tide}\|=|C_T|\,p\times$(quotient) with $p$ bounded. Any norm equivalent to Frobenius may be used.

## 4.10 Condition C: frozen state variables

$$
\boxed{\nu^{(n)}\to\nu_0,\qquad Z^{(n)}\to Z_0.}
$$

## 4.11 Condition D: constitutive parameters reduce to constants

$$
\mu_s(\nu^{(n)},Z^{(n)})\to\mu_s,\qquad
\mu_2(\nu^{(n)},Z^{(n)})\to\mu_2,\qquad
I_0(\nu^{(n)},Z^{(n)})\to I_0,\qquad
\mu_y(\nu^{(n)},Z^{(n)})\to\mu_s .
$$

The last holds trivially in the reference model, where $\mu_y\equiv\mu_s(\nu_0,Z_0)$ (Section 2.8, B2).

## 4.12 Condition E: disappearance of cohesive strength

$$
\boxed{\frac{c(\nu^{(n)},Z^{(n)})}{p}\longrightarrow0.}
$$

Under Section 4.3a this is imposed on both constants, $\max(c_s^{(n)},c_b^{(n)})/p\to0$. Then $c(\mathbf x)/p\to0$ uniformly.

## 4.13 Condition F: disappearance of pressure regularization

With a pressure regularization $\tilde p=\max([p]_+,p_\varepsilon)\ge p$ (in the reference model $p_\varepsilon=p_{\min}$), require $\boxed{p_\varepsilon/p\to0}$. With a shear-rate regularization, require $\boxed{\dot\gamma_\varepsilon/\dot\gamma\to0}$. When the pressure regularization is used, $p$ is replaced by $\tilde p$ uniformly in the law, in every denominator and every stress magnitude; the principal formulae are written with $p$. The production rule (Section 2.8) is stricter than a limit statement. At every finite regularization, the stress magnitude, the yield strength, and the denominator of $I_c$ use the same $\tilde p$, with $I_c$ at $p_*=\tilde p+p_c$. The condition $p_\varepsilon/p\to0$ only states when the regularization disappears from the reduction. It is not a licence to regularize the denominator alone.

---

## 4.14 Limit of the inertial number

$I_c^{(n)}=I\,(p/(p+p_c^{(n)}))^{1/2}$, and the fraction tends to $1$ by Condition A. By continuity of the square root at $1$ (requiring $p>0$),

$$
\boxed{
I_c^{(n)}\longrightarrow\frac{d\dot\gamma}{\sqrt{p/\rho_p}}=I.
}
$$

On a free surface with $p=0$ the classical law is itself undefined, so there is no reduction in the ordinary sense.

---

## 4.15 Limit of the friction coefficient

By Conditions C and D and Section 4.14, $\mu_s^{(n)}\to\mu_s$, $\mu_2^{(n)}\to\mu_2$, $I_0^{(n)}\to I_0$, and $I_c^{(n)}\to I$. The map $(a,b,e,c)\mapsto a+(b-a)c/(e+c)$ is continuous on $e+c>0$. It needs four arguments, because $I_0$ also depends on $n$; freezing $I_0$ first is not enough. Hence

$$
\boxed{
\mu_d^{(n)}\longrightarrow\mu_s+(\mu_2-\mu_s)\frac{I}{I_0+I}=\mu_{\mathrm{cl}}(I).
}
$$

---

## 4.16 Limit of the tidal yield function

Since $s-T^{(n)}_{\rm tide}=\tau^{(n)}_{\rm flow}$ for every $n$, the effective stress $\tau^{(n)}_{\rm tide}=\sqrt{\tfrac12(s-T^{(n)}_{\rm tide}):(s-T^{(n)}_{\rm tide})}$ equals $\tau^{(n)}_{{\rm flow},\rm eq}$ identically and carries no independent information. If, in addition, a limit stress exists and $T^{(n)}_{\rm tide}\to0$, then

$$
\boxed{
\tau_{\mathrm{tide}}^{(n)}\longrightarrow\sqrt{\tfrac12 s:s}=\tau_{{\mathrm{flow}},\mathrm{eq}},
}
$$

the two members agreeing because $s-T_{\rm tide}\to s$. With $c^{(n)}/p\to0$ and $\mu_y^{(n)}\to\mu_s$,

$$
F^{(n)}_{\rm comb}=\tau^{(n)}_{\rm tide}-c^{(n)}-\mu^{(n)}_yp\longrightarrow\tau_{{\rm flow},\rm eq}-\mu_sp,
$$

the Drucker–Prager limit of the quasi-static theory, without cohesion or tide.

---

## 4.17 Limit of the dynamic stress

The scalar target $c^{(n)}+\mu_d^{(n)}p$ tends to $\mu_{\rm cl}(I)p$ by Condition E and Section 4.15. That is a target, not yet the equivalent stress of the regularized tensor. The regularized law, aligned with $D'$, is

$$
\boxed{
\tau_{\mathrm{flow}}^{(n)}=2\eta^{(n)}D',
\qquad
\eta^{(n)}=\frac{c^{(n)}+\mu_d^{(n)}p}{\sqrt{\dot\gamma^2+\dot\gamma_{\varepsilon}^{\,2}}},
}
$$

whose equivalent stress is $\eta^{(n)}\dot\gamma=(c^{(n)}+\mu_d^{(n)}p)\,\dot\gamma/\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}$, equal to the target only as $\dot\gamma_\varepsilon/\dot\gamma\to0$. Under $c^{(n)}/p\to0$, $\mu_d^{(n)}\to\mu_{\rm cl}(I)$, and $\dot\gamma_\varepsilon/\dot\gamma\to0$, one has $\eta^{(n)}\to\mu_{\rm cl}(I)p/\dot\gamma$. So

$$
\boxed{
\tau_{\mathrm{flow}}^{(n)}\longrightarrow2\mu_{\mathrm{cl}}(I)p\frac{D'}{\dot\gamma},
\qquad
\tau_{{\mathrm{flow}},\mathrm{eq}}^{(n)}\longrightarrow\mu_{\mathrm{cl}}(I)p,
}
$$

the second by continuity of $A\mapsto\sqrt{\tfrac12A:A}$ and $D':D'=\dot\gamma^2/2$. This is the classical $\mu(I)$ relation, and the factor $2$ is again indispensable.

---

## 4.18 Limit of the governing equations

The momentum equation of the combined model, with $g_{\rm eff}=\nabla W$, is

$$
\partial_t(\rho v)+\nabla\cdot(\rho v\otimes v)+2\rho\,\boldsymbol\Omega\times v=\nabla\cdot\sigma+\rho\nabla W.
$$

With a negative-potential convention, the sign must be rewritten throughout, not in this term alone. The Coriolis term is written in the same rotating-frame convention used here. The classical system is the inertial-frame case $\boldsymbol\Omega=\mathbf 0$. Since $\partial_t(\rho v)+\nabla\cdot(\rho v\otimes v)=\rho(\partial_tv+(v\cdot\nabla)v)+v(\partial_t\rho+\nabla\cdot(\rho v))$, the conservative and nonconservative forms agree on solutions of continuity, not as formal expressions.

The traditional incompressible model also fixes $\rho=\rho_0>0$, so continuity is equivalent to $\nabla\cdot v=0$. If $\nabla\cdot v=0$ is kept without fixing $\rho$, continuity becomes the passive transport $\partial_t\rho+v\cdot\nabla\rho=0$. That is the **inhomogeneous incompressible** extension: each particle keeps its density. An elastically compressible extension would need an equation of state, which this paper does not have. The rate-type dilatancy of A-Ξ (Section 11.3) is not one: $\nabla\cdot v=\dot\varepsilon_v$ is a kinematic constitutive constraint driven by plastic shear, $p$ remains its multiplier, and continuity becomes $D_t\rho=-\rho\,\dot\varepsilon_v$. Without A-Ξ the production model is of the inhomogeneous incompressible type, and Section 9.2 discretizes $\partial_t\rho_K+v\cdot\nabla\rho_K=0$. Neither form is the traditional $\mu(I)$ system unless $\rho=\rho_0$.

With $\sigma=-pI+T_{\rm tide}+\tau_{\rm flow}$, $\nabla\cdot\sigma$ contains $\nabla\cdot T_{\rm tide}$. That term vanishes in the limit only under the compactness hypotheses of Section 4.20, not by pointwise convergence. In the limit, every extension disappears:

- the solid fraction and the coordination number are fixed;
- the tide, the cohesion, and the cohesive pressure vanish;
- the parameters become constants;
- the regularizations vanish.

Then

$$
\boxed{
\begin{cases}
\rho=\rho_0,\qquad \nabla\cdot v=0,\\[4pt]
\partial_t(\rho v)
+\nabla\cdot(\rho v\otimes v)
+2\rho\,\boldsymbol\Omega\times v
=
\nabla\cdot
\left[
-pI+
2\mu_{\mathrm{cl}}(I)p
\dfrac{D'}{\dot\gamma}
\right]
+\rho\nabla W,\\[8pt]
\displaystyle
I=\frac{d\dot\gamma}{\sqrt{p/\rho_p}},\qquad
\mu_{\mathrm{cl}}(I)=\mu_s+(\mu_2-\mu_s)\frac{I}{I_0+I},
\end{cases}
}
$$

the traditional $\mu(I)$ system: in the rotating frame if $\boldsymbol\Omega$ is kept, inertial if $\boldsymbol\Omega=\mathbf 0$.

---

## 4.19 The reduction theorem

The results of Sections 4.14–4.18 are summarized as follows.

## 4.20 Theorem

Suppose a sequence of solutions of the combined model satisfies:

- $p_c/p\to0$, $\rho_pd^2\|\operatorname{dev}H_W\|/p\to0$, and $c/p\to0$;
- $\nu\to\nu_0$ and $Z\to Z_0$;
- $\mu_s,\mu_2,I_0\to$ constants, and $\mu_y\to\mu_s$;
- $p_\varepsilon/p\to0$ and $\dot\gamma_\varepsilon/\dot\gamma\to0$.

Suppose also that, along the chosen points, $p\to p_0>0$ and $\dot\gamma\to\dot\gamma_0>0$, or at least that both stay in a compact subset of $(0,\infty)$. In the latter case $I$ is formed from the same $p$ and $\dot\gamma$, and the limits below mean differences tending to zero. A positive lower bound alone is not enough for $T_{\rm tide}\to0$ and $\tau_{\rm flow,eq}-\mu_{\rm cl}(I)p\to0$, because the hypotheses control only ratios to $p$. Then

$$
I_c\to I,\qquad
\mu_d(I_c,\nu,Z)\to\mu_{\mathrm{cl}}(I),\qquad
T_{\mathrm{tide}}\to0,\qquad
\tau_{{\mathrm{flow}},\mathrm{eq}}\to\mu_{\mathrm{cl}}(I)p,
$$

with $I=d\dot\gamma_0/\sqrt{p_0/\rho_p}$ from the limit fields. Sections 4.14–4.17 are pointwise statements, about values at one point, not a sequence in a function space.

**The compactness chain for the momentum equation.** The scalar limit $\tau_{\rm eq}^{(n)}\to\mu_{\rm cl}p$ implies neither convergence of $D'^{(n)}/\dot\gamma^{(n)}$ nor convergence of the differentiated equation. The identity $s-T_{\rm tide}\equiv\tau_{\rm flow}$ (Section 4.16) holds by definition, and its limit adds nothing. After regularization, $\tau_{\rm flow,eq}\to c+\mu_dp$ only if $\dot\gamma_\varepsilon/\dot\gamma\to0$. The map of Section 4.15 has four arguments. The momentum equation converges in distributions only if, on a compact $K\Subset\{p>0\}$ in space-time, all of the following hold:

1. $p^{(n)}\ge p_{\min}>0$ and $\dot\gamma^{(n)}\ge\dot\gamma_{\min}>0$ uniformly, with conditions A–F uniform on $K$;
2. $v^{(n)}\to v$ strongly in $L^2(K)$, and $\nabla v^{(n)}\to\nabla v$, $p^{(n)}\to p$ a.e. (a subsequence suffices under convergence in measure);
3. $\sigma^{(n)}$ is bounded in $L^{1+\delta}(K)$ for some $\delta>0$, and $\nu^{(n)}\to\nu_0$ strongly.

Then $\sigma^{(n)}\to\sigma_{\rm cl}=-pI+2\mu_{\rm cl}(I)pD'/\dot\gamma$ in $L^1(K)$ by Vitali's theorem, so $\nabla\cdot\sigma^{(n)}\to\nabla\cdot\sigma_{\rm cl}$ in $\mathcal D'(K)$. The convective term converges by strong $L^2$ convergence ($\rho=\rho_0$), and $\nabla\cdot v=0$ is preserved.

**Limitations.**

1. Almost-everywhere convergence is a hypothesis, not a conclusion. The $\mu(I)$ stress–strain-rate relation is generally **not monotone**: it is ill-posed outside an intermediate range of the inertial number, at both small and large $I$ [8]. Minty's argument is therefore unavailable, and weak convergence does not identify the limit.
2. The result holds only inside $K\Subset\{p>0\}$. It excludes the free-surface condition and plug regions with $\dot\gamma=0$.
3. Mode B drops the inertial terms that this momentum equation keeps, and its regularized shear rate, although positive everywhere, has no uniform lower bound: in rigid regions $\dot\gamma\lesssim\dot\gamma_\varepsilon$. For Mode B the theorem therefore gives only the pointwise constitutive limits of Sections 4.14–4.17 in the yielded region; the momentum statement is the quasi-static hyperbolic balance.

---

## 4.21 Range of validity and the boundary of failure

The reduction must not be used in the following situations.

## 4.22 The low-pressure limit on the free surface, and the singularity at $p=0$

Normal balance on the free surface gives $p|_\Gamma=p_{\rm ext}+\mathbf n\cdot\boldsymbol\tau\mathbf n$ (Section 10.1.1). Without external pressure and deviatoric stress (the tideless, low-shear limit), $p\to0$, and $I=d\dot\gamma/\sqrt{p/\rho_p}$ divides by zero. An effective pressure $p_*=p+p_c>0$ or an explicit pressure regularization must then be kept.

## 4.23 The tidal stress is not negligible

If $\rho_pd^2\|\operatorname{dev}H_W\|\sim p$, Condition B fails, $T_{\rm tide}$ stays of the order of the pressure scale multiplying $\mu$, and there is no reduction.

*Order of magnitude for small bodies.* With $d=d_p$ and typical regolith values ($C_T=1$, grain density $\rho_p\approx2420$ kg m⁻³, packing $\nu_0\approx0.54$, $d_p\approx5$ cm), the prefactor is $C_T\rho_p\nu_0d_p^2\approx3.3$ kg m⁻¹. With published masses, radii, and spins, $\|\operatorname{dev}H_W\|\sim GM/R^3+\omega^2\lesssim10^{-6}$ s⁻² for Ryugu, Phobos, and Deimos. Hence $\|\mathbf T_{\rm tide}\|\sim10^{-6}$ Pa, against interior pressures of order $\rho_K|\mathbf g|R\gtrsim10$ Pa and loose-layer pressures $p_{\rm ref}=\rho_{\rm bulk}|\mathbf g|h_s\sim0.2$–$40$ Pa for $h_s$ of 1–5 m (Section 10.2). Condition B holds with at least five orders of magnitude to spare. (B2) compares the flow stress $\boldsymbol\tau_{\rm flow}=\mathbf s-\mathbf T_{\rm tide}$ with the strength, so the tide enters yield as the backstress that shifts $\mathbf s$, through the load $\nabla\cdot\mathbf T_{\rm tide}$ that $\boldsymbol\tau_{\rm flow}$ must balance, and in the repose proxy $\mu_{\rm eff}$ of Section 10.2, which subtracts $\|\mathbf T_{\rm tide}\|_F/\sqrt2$. Each contribution is of order $10^{-6}$ Pa and negligible here. For such bodies the tide acts through $\mathbf g_{\rm ext}$ in $\mathbf g_{\rm eff}$ (Section 10.3.1, orbital schedule), not through $\mathbf T_{\rm tide}$.

## 4.24 The state variables are still evolving

If $Z$ or $\nu$ varies appreciably in space, $\mu_d=\mu_d(I,\nu,Z)$ is not the traditional function of $I$ alone. Under A-Ξ this is the case away from the critical state.

## 4.25 Static regions

Where $\dot\gamma=0$, the dynamic formula is incomplete, and the static condition $\sqrt{\tfrac12(s-T_{\rm tide}):(s-T_{\rm tide})}\le c+\mu_yp$ applies. In the regularized reference model $\dot\gamma>0$ everywhere, so static regions are identified not by $\dot\gamma=0$ but by $F_\delta\le0$ on the same stress, together with connectivity to the rigid core (B2–B3), with $\mu_y\equiv\mu_s(\nu_0,Z_0)$.

---

**Conclusion.** The combined model is not automatically the classical $\mu(I)$ law. In the explicit limit where cohesion, tide, state evolution, and regularization all disappear, it reduces to that law term by term and continuously. This is a formal proposition under stated conditions, not an unconditional identity.

# 5. Surface transport and runout

This chapter exposes the surface-transport closure as an independent module. It uses the frozen-surface equations of Section 2.8b and does not introduce a second deposition law.

## 5.1 Fermat refractive-index field

Let $(\Gamma,g)$ be the frozen polyhedral surface with first fundamental form $g_{ab}$ and let $v_{\rm e}(x)>0$ be the prescribed effective runout speed. Define

$$n(x):=v_{\rm e}(x)^{-1}.$$

For a launch point $a'$, the arrival time is the viscosity solution

$$\boxed{g^{ab}(x)\,\partial_aT_{a'}(x)\,\partial_bT_{a'}(x)=n(x)^2,\qquad T_{a'}(a')=0,\quad T_{a'}\ge0.}$$

Equivalently,

$$T_{a'}(a)=\inf_{\gamma(a')=a',\,\gamma(1)=a}\int_\gamma n(x)\,ds_g,\qquad ds_g^2=g_{ab}dx^a dx^b.$$

Thus the prescribed-speed runout is exactly the Fermat problem for the scalar GRIN index $n$. The characteristic Hamiltonian is

$$H_0(x,p)=\tfrac12\big(g^{ab}(x)p_ap_b-n(x)^2\big)=0,$$

with

$$\dot x^a=\partial_{p_a}H_0=g^{ab}p_b,\qquad \dot p_a=-\partial_aH_0=-\tfrac12(\partial_ag^{bc})p_bp_c+n\partial_an.$$

The equality is exact for this reduced prescribed-speed model; it is not an assertion that arbitrary history-dependent granular dynamics has a scalar optical index.

## 5.2 Randers magnetic correction for rotation

Coriolis acceleration is velocity-dependent. When the rotating reduction admits a surface one-form $A=A_a dx^a$ with magnetic curvature $B=dA$ and strong-convexity condition $|A|_g<n$, use the Randers functional

$$\mathcal T_R[\gamma]=\int_\gamma\big(n(x)\,ds_g+A_a(x)dx^a\big).$$

Its Legendre-dual Hamiltonian can be written as

$$H_R(x,p)=\tfrac12\,g^{ab}(x)\big(p_a-A_a(x)\big)\big(p_b-A_b(x)\big)-\tfrac12n(x)^2=0,$$

so Hamilton's equations generate the corresponding magnetic geodesics and reproduce the Coriolis deflection represented by $B$. If no such position-only one-form represents the rotating packet law, Section 2.8b uses the full rotating packet ODE, which remains authoritative.

## 5.3 Caustics, deposition, and mass closure

Let a ray bundle be parameterized by launch coordinates $\xi$ and let $J_R=\det(\partial x/\partial\xi)$ be its surface Jacobian. The deposited density is accumulated weakly as

$$\int_\Gamma\varphi(a)J_{\rm dep}(a)\,dS(a)=\int_{\Gamma_{\rm src}}m(a')\,\varphi(X_R(a'))\,dS(a')$$

for every test function $\varphi$, with nonnegative packet mass $m$. Where $|J_R|$ becomes small, the ray family forms a caustic; subdivision, kernel spreading, or a bounded-density limiter is applied before accumulation. These operations redistribute mass and do not create it. After escaped or timed-out packets are debited, normalize the retained kernel so

$$K_R(a,a')\ge0,\qquad \int_\Gamma K_R(a,a')\,dS(a)=1,$$

which gives $\int_\Gamma J_{\rm dep}dS=\int J_{\rm ero}dS$ for retained packets. RT hardware performs BVH traversal and local hit queries only; the Eikonal/Randers integration, Jacobian estimates, packet weights, and global mass ledger remain explicit numerical operations.

# 6. Variable density: mass cross-coupling and the baroclinic source

$\rho=\rho_K$ (Section 1.1), transported by the flow (Section 9.2). On the quasi-static route this chapter is evaluated with the kernel parameters frozen at the current step. For anisotropic cores, $\sigma_k^2(\mathbf x-\mathbf c_k)$ is replaced by $\mathbf S_k(\mathbf x-\mathbf c_k)$.

---

## 6.1 Derivatives of the Cauchy density

With $q_k=1+(\mathbf x-\mathbf c_k)^{\!\top}\mathbf S_k(\mathbf x-\mathbf c_k)$ ($\mathbf S_k=\sigma_k^2\mathbf I$ isotropic), $\nabla q_k=2\mathbf S_k(\mathbf x-\mathbf c_k)$, $\nabla^2q_k=2\mathbf S_k$, and the chain rule,

$$
\nabla\rho_K=-2\sum_k\alpha_kw_k\,q_k^{-\alpha_k-1}\,\mathbf S_k(\mathbf x-\mathbf c_k),
$$

$$
\nabla^2\rho_K=\sum_kw_k\Big[-2\alpha_k q_k^{-\alpha_k-1}\mathbf S_k+4\alpha_k(\alpha_k+1)q_k^{-\alpha_k-2}\,\mathbf S_k(\mathbf x-\mathbf c_k)(\mathbf x-\mathbf c_k)^{\!\top}\mathbf S_k\Big],
$$

$$
\Delta\rho_K=\sum_kw_k\Big[-2\alpha_k q_k^{-\alpha_k-1}\operatorname{tr}\mathbf S_k+4\alpha_k(\alpha_k+1)q_k^{-\alpha_k-2}\,|\mathbf S_k(\mathbf x-\mathbf c_k)|^2\Big].
$$

In the isotropic case $\operatorname{tr}\mathbf S_k=3\sigma_k^2$ and $|\mathbf S_k(\mathbf x-\mathbf c_k)|^2=\sigma_k^4|\mathbf x-\mathbf c_k|^2$. The parameter derivatives used by the refit of Section 9.2 are

$$
\frac{\partial\rho_k}{\partial w_k}=q_k^{-\alpha_k},\qquad
\frac{\partial\rho_k}{\partial\mathbf c_k}=2\alpha_kw_k\,q_k^{-\alpha_k-1}\,\mathbf S_k(\mathbf x-\mathbf c_k),\qquad
\frac{\partial\rho_k}{\partial\mathbf S_k}=-\alpha_kw_k\,q_k^{-\alpha_k-1}\,(\mathbf x-\mathbf c_k)(\mathbf x-\mathbf c_k)^{\!\top}.
$$

The last is a symmetric matrix derivative (Frobenius product with $\delta\mathbf S$). At $\mathbf x=\mathbf c_k$ the $k$-th summand has $\nabla\rho_k=\mathbf 0$ and $\nabla^2\rho_k=-2\alpha_kw_k\mathbf S_k$.

---

## 6.2 The baroclinic source

**Proposition 6.2 (Baroclinic source in weak form).** For a smooth vector potential $\mathbf A$ and $\mathbf w=\nabla\times\mathbf A$,

$$
\boxed{
\int_{\mathcal V}\rho\,\mathbf g_0\cdot\nabla\times\mathbf A\,dV
=\int_{\mathcal V}\mathbf A\cdot\nabla\times(\rho\mathbf g_0)\,dV
+\oint_\Gamma\mathbf n\cdot(\mathbf A\times\rho\mathbf g_0)\,dS,}
$$

and $\nabla\times(\rho\nabla W)=\nabla\rho\times\nabla W$.

*Proof.* Integrate $\nabla\cdot(\mathbf A\times\mathbf F)=\mathbf F\cdot\nabla\times\mathbf A-\mathbf A\cdot\nabla\times\mathbf F$ with $\mathbf F=\rho\mathbf g_0$, and note $\nabla\times\nabla W=0$. ∎

**Non-structured source coupling.** On the actual polyhedron, any local body-force or baroclinic contribution is assembled as a volume source $\mathbf f_{\rm loc}(x)$ and inserted directly into the momentum equation of (8.3-H),

$$\rho_\tau\partial_\tau\mathbf v-\nabla\cdot\mathbf S+\nabla p=\mathbf f_{\rm grav}+\mathbf f_{\rm tide}+\mathbf f_{\rm loc}.$$

No spherical or radial expansion is required. The same source is sampled along characteristic rays and its discrete work is included in the residual and energy ledger. Rotation adds the Coriolis block described in Chapter 7; it does not alter the source insertion rule.



# 7. Rotation: Coriolis cross-coupling and the full nonlinear block system

## 7.1 The curl identity for the Coriolis term

**Lemma 7.1.** If $\nabla\cdot\mathbf v=0$ and $\boldsymbol\Omega=\omega\mathbf e_z$ is constant, then $\nabla\times(\boldsymbol\Omega\times\mathbf v)=-(\boldsymbol\Omega\cdot\nabla)\mathbf v=-\omega\partial_z\mathbf v$. This follows from $\nabla\times(\boldsymbol\Omega\times\mathbf v)=\boldsymbol\Omega(\nabla\cdot\mathbf v)-\mathbf v(\nabla\cdot\boldsymbol\Omega)+(\mathbf v\cdot\nabla)\boldsymbol\Omega-(\boldsymbol\Omega\cdot\nabla)\mathbf v$.

**Proposition 7.2 (Weak form of the Coriolis term).** For $\mathbf w=\nabla\times\mathbf A$,

$$
C(\mathbf w,\mathbf v)
:=2\int_{\mathcal V}\rho\,(\boldsymbol\Omega\times\mathbf v)\cdot\mathbf w\,dV
=\int_{\mathcal V}\mathbf A\cdot\big[2\nabla\rho\times(\boldsymbol\Omega\times\mathbf v)-2\omega\rho\,\partial_z\mathbf v\big]\,dV
+2\oint_\Gamma\mathbf n\cdot\big(\mathbf A\times\rho\,\boldsymbol\Omega\times\mathbf v\big)\,dS,
$$

and $C$ is skew: $C(\mathbf w,\mathbf v)=-C(\mathbf v,\mathbf w)$ and $C(\mathbf v,\mathbf v)=0$. (Apply Proposition 6.2 to $2\rho\boldsymbol\Omega\times\mathbf v$ with Lemma 7.1. Skewness is $(\boldsymbol\Omega\times\mathbf v)\cdot\mathbf w=-(\boldsymbol\Omega\times\mathbf w)\cdot\mathbf v$.) The Coriolis force does no work; it only redistributes energy among modes.

---

## 7.2 Rigid modes and solvability

**Proposition 7.3.** The viscous and pressure-coupling blocks vanish on the six rigid modes. Hence:

1. **Necessary condition.** The quasi-static problem (without $M\dot{\mathbf u}$ and $N$) needs zero external work on the rigid modes:

$$
\int_{\mathcal V}\rho\,\mathbf g_0\,dV+\oint_\Gamma\mathbf t_{\mathrm{ext}}\,dS=\mathbf 0,\qquad
\int_{\mathcal V}\rho\,\mathbf x\times\mathbf g_0\,dV+\oint_\Gamma\mathbf x\times\mathbf t_{\mathrm{ext}}\,dS=\mathbf 0.
$$

Self-gravity has zero resultant. The centrifugal force needs the centre of mass on the spin axis and that axis principal ($I_{xz}=I_{yz}=0$).

2. **Gauge.** The solution is fixed only up to a rigid mode, by $\int\rho\mathbf v=\mathbf 0$ and $\int\rho\mathbf x\times\mathbf v=\mathbf 0$.

(For a rigid test field the stress and pressure terms vanish. The rigid modes span the kernel of the viscous block.)

## 7.3 Range of validity

The block system is the complete dynamic weak form for $(\mathbf v,p,\rho,\Gamma)$. It places no geometric restriction on $\mathbf v$: both potentials are present, and the pressure is independent. Since $\mu(I)$ is not monotone in $\mathbf D$, no continuous existence or uniqueness is claimed. Discrete solvability depends on an inf–sup condition and on local well-posedness of the law (Chapter 8 on).

**The quasi-static route uses only the viscous and load blocks.** The inertial, convection, and Coriolis terms are omitted from this route; the Coriolis acceleration acts only on the far faster Mode C packets (Section 2.8b). The density is frozen within a step and transported between steps (A-ρL, Section 9.2). The route uses the hyperbolic relaxation (8.3-H).

At the continuous level the governing equations and natural boundary conditions are complete. Non-monotone rheology and discrete well-posedness are delimited by registered obligations (PO-1, PO-11, PO-12), and every dependent conclusion is marked.

---

# 8. Non-star-shaped volume discretization and characteristic ray solving

This chapter gives the finite-dimensional volume discretization and the characteristic-ray relaxation used on the current polyhedral domain. All fields use the geometry-adapted divergence-free basis and are integrated on ray segments inside the actual domain.

---

## 8.1 General geometry: removing the star-shaped hypothesis

This section states the geometric hypothesis of the whole model and the general route that every chapter uses. Proposition 8.1.1 shows that a radial parametrization cannot be stretched to a body that is not star-shaped, whatever the constants or origin. 8.1.2 records the routes that do not work. 8.1.3 gives the general route (G1)–(G10), which is normative in Chapters 2, 4, 5, 8, 9, and 10.

**Hypothesis (G).** $\mathcal V\subset\mathbb R^3$ is a bounded, connected, open polyhedral domain with Lipschitz boundary $\Gamma=\partial\mathcal V$, a closed, consistently oriented, embedded triangulation with vertices $\mathbf X_i$ and outward normal $\mathbf n$ (Section 1.1). It may have any genus. Its boundary $\Gamma$ may have several components, while $\mathcal V$ itself is connected; the inner components of $\Gamma$ bound sealed cavities $C_1,\dots,C_{N_c}$. The origin $O$ is the centre of mass (Section 10.3.4, 8b). $O\in\mathcal V$ is not assumed, only $O\notin\Gamma$. Write

$$
R_*:=\max_{\Gamma}|\mathbf x|,\qquad B_*:=\{|\mathbf x|<R_*\}\supset\mathcal V .
$$

### 8.1.1 Why the radial form cannot be extended

**Proposition 8.1.1 (Star-shapedness is the sign of $\mathbf n\cdot\mathbf e_r$).** Let $\mathcal V$ satisfy (G) and suppose it is not star-shaped about $O$: there are $\mathbf x\in\mathcal V$ and $t\in(0,1)$ with $t\mathbf x\notin\overline{\mathcal V}$ (if $O\notin\overline{\mathcal V}$, every $\mathbf x$ and small $t$ qualify). Then there is an open set $\Sigma\subset S^2$ of directions with

$$
\Gamma^-:=\{\mathbf y\in\Gamma:\ \mathbf n(\mathbf y)\cdot\mathbf e_r(\mathbf y)<0\},\qquad
\int_{\Gamma^-}\frac{|\mathbf n\cdot\mathbf e_r|}{|\mathbf y|^2}\,dS\ \ge\ |\Sigma|>0 ,
$$

so $\Gamma^-$ has positive area. Conversely, if $\mathbf n\cdot\mathbf e_r\ge c_0>0$ a.e., then $O\in\mathcal V$, every ray from $O$ crosses $\Gamma$ once, and $\mathcal V$ is star-shaped about $O$.

*Proof.* $E=\mathbb R^3\setminus\overline{\mathcal V}$ and $\mathcal V$ are open. So there is an open cone $\Sigma$ of directions whose rays have a point of $E$ at some $s_1$ and a point of $\mathcal V$ at some $s_2>s_1$; between them the ray enters $\mathcal V$ through $\Gamma$. Rays through edges or vertices, or lying in a face plane, form a null set. So for almost every $\hat{\boldsymbol\omega}\in\Sigma$ the entry is a transversal crossing of a face interior, where $\mathbf e_r$ points inward and $\mathbf n\cdot\mathbf e_r<0$. The radial projection maps $\Gamma^-$ onto almost all of $\Sigma$ with area Jacobian $|\mathbf n\cdot\mathbf e_r|/|\mathbf y|^2\le\operatorname{dist}(O,\Gamma)^{-2}$, which gives the bound and $|\Gamma^-|>0$. Conversely, crossings alternate between exit and entry, and $\mathbf n\cdot\mathbf e_r>0$ makes every crossing an exit. If $O\notin\overline{\mathcal V}$, a ray from $O$ that meets $\mathcal V$ enters it first, and by the first part such entries occupy a set of positive area with $\mathbf n\cdot\mathbf e_r<0$, which the hypothesis excludes. So $O\in\overline{\mathcal V}$, hence $O\in\mathcal V$ because $O\notin\Gamma$ by (G). A ray from an interior $O$ ends outside, so its crossings reduce to a single exit. ∎

### 8.1.2 Routes that do not work

**(N3) A union of star-shaped pieces.** Couple pieces $\mathcal V_m$, star-shaped about $\mathbf z_m$, across interior interfaces by mortar or Nitsche conditions. Feasible for a fixed partition. But the partition is not unique and must be rebuilt as $\Gamma$ evolves, and the core connectivity of (B3) and the surface transport cross the interfaces. **Not adopted.**

**(N4) A global map to the ball.** A diffeomorphism $F:B\to\mathcal V$ with the contravariant Piola transform $\mathbf v=(DF\hat{\mathbf w}/\det DF)\circ F^{-1}$ preserves solenoidality and normal flux exactly. However, such a map exists only for genus $0$ without cavities, its constants grow without bound across thin necks, and building $F$ for an evolving polyhedron has no closed form. **Correct within its class; not adopted.**

**(N5) A change of topology.** No fixed-triangulation route, including the adopted one, represents fission, cave closure, or merging. That needs an implicit surface (level set) and a different Chapter 3 pipeline. **Not covered:** a self-intersecting update is rejected, and persistent rejection ends the run with the status *geometry invalid* (Section 10.3.3).

### 8.1.3 The general route

**(G1) Geometry-adapted divergence-free basis.** Construct a finite basis directly from the current polyhedral geometry, without a radial parametrization or global angular expansion. Compactly supported scalar or vector kernels are centred at volume nodes and face quadrature points; a discrete curl, or an $H(\mathrm{div})$-compatible Raviart--Thomas/Nédélec reconstruction, supplies divergence-free trial fields. One independent flux mode is added for each sealed cavity. The basis is restricted to $\mathcal V$, and all volume and surface integrals use the actual mesh and BVH ray-segment decomposition. This construction is valid for arbitrary genus and for $O\notin\mathcal V$.

**Proposition 8.1.3 (Discrete independence and SPD mass block).** Let the retained basis functions be linearly independent after the cavity-flux constraints have been appended. If $0<\rho_{\min}\le\rho\le\rho_{\max}$ a.e. on the positive-volume domain $\mathcal V$, then

$$\mathbf M_\Gamma=\int_{\mathcal V}\rho\,\mathbf B_\alpha\cdot\mathbf B_\beta\,dV$$

 is symmetric positive definite. Rank is checked by the smallest singular value of the assembled basis matrix; a rank-deficient enrichment is rejected or orthogonalized before the solve.

*Proof.* For any coefficient vector $c$, $c^\top\mathbf M_\Gamma c=\int_{\mathcal V}\rho\,|\sum_\alpha c_\alpha\mathbf B_\alpha|^2dV$. Positivity of $\rho$ makes this zero only when the represented field vanishes a.e.; linear independence then gives $c=0$. ∎

**Lemma 8.12 (Frozen-viscosity uniqueness modulo rigid modes).** Let $\eta(\mathbf x)\ge\eta_{\rm lo}>0$ be frozen on a domain satisfying (G), let $\mathbb V_h^\sigma$ be the retained divergence-free trial space, and impose the six rigid-mode constraints. If $\mathbf v\in\mathbb V_h^\sigma$ has zero load, zero traction, and zero rigid coordinates, then $\mathbf v=0$. Consequently, the frozen viscous bilinear form

$$a_\eta(\mathbf u,\mathbf v)=2\int_{\mathcal V}\eta\,\mathbf D(\mathbf u):\mathbf D(\mathbf v)\,dV$$

 is coercive on the constrained space and the frozen velocity problem has at most one solution.

*Proof.* Testing the homogeneous weak equation with $\mathbf v$ gives $a_\eta(\mathbf v,\mathbf v)=0$. Since $\eta\ge\eta_{\rm lo}$, $\mathbf D(\mathbf v)=0$ a.e. Korn's lemma on a connected Lipschitz domain implies that $\mathbf v$ is a rigid translation plus rotation. The six constraints set those coordinates to zero. ∎

**Proposition 8.1.4 (Rigid modes and the rank of the constraint).** The six rigid translations and rotations are appended explicitly to the geometry-adapted basis. Their weighted $L^2(\rho)$ Gram matrix is positive definite on every positive-volume $\mathcal V$, so $\operatorname{rank}\mathbf C_{\rm rigid}=6$.

*Proof.* On rigid modes, the pairing of the rows of $\mathbf C_{\rm rigid}$ with a second rigid motion is $\int_{\mathcal V}\rho(\mathbf a'+\boldsymbol\theta'\times\mathbf x)\cdot(\mathbf a+\boldsymbol\theta\times\mathbf x)\,dV$. This is the $L^2(\rho)$ Gram matrix of the rigid motions, which is positive definite because a rigid motion vanishing on an open set is zero. ∎

The explicit rigid-mode enrichment contains the six rigid motions on every admissible domain.

**Proposition 8.1.5 (Completeness; sealed cavities).** Assume the geometry-adapted divergence-free family is dense in the chosen $H^1_\sigma(\mathcal V)$ trial space and that the appended cavity-flux modes span the independent cavity fluxes.

1. If $\Gamma$ is connected, the restrictions are dense in $H^1_\sigma(\mathcal V)$.
2. With sealed cavities, density holds after adding the flux modes $\mathbf v_j=\nabla|\mathbf x-\mathbf z_j|^{-1}$, $\mathbf z_j\in C_j$.

*Proof (sketch).* A field in the target solenoidal space is approximated directly by the geometry-adapted basis on $\mathcal V$; the cavity-flux modes supply the missing independent flux coordinates. The density statement is therefore an explicit approximation hypothesis of the chosen mesh/kernel family, tested by refinement. A connected $\Gamma$ has zero flux by the divergence theorem. Each $\mathbf v_j$ is harmonic on $\overline{\mathcal V}$ with flux $\pm4\pi$ through $\partial C_j$ and $0$ through the other cavities. Being divergence-free in $\mathcal V$, it carries the opposite flux through the outer component of $\Gamma$: the outer surface moves in by the volume the cavity loses, as incompressibility requires. Subtracting $\sum_jc_j\mathbf v_j$ with the matching $c_j$ removes every cavity flux, and then the outer flux vanishes too by the divergence theorem, so the geometry-adapted approximation applies. An incompressible body with a collapsing cavity carries such fluxes, but a field extended into the cavity cannot. ∎

*Conditioning.* The geometry-adapted family is independent but not uniformly stable. Near-dependent kernel fields form the numerical near-kernel, so $\operatorname{cond}\mathbf M_\Gamma$ and $\operatorname{cond}\mathbf K$ grow with basis resolution and with thin necks or strongly varying cell sizes. Remedy: orthonormalize on $\mathcal V$ by a Cholesky factor of the assembled $\eta$-free Gram matrix and report the condition number.

**(G2) The volume quadrature: ray segments from $O$.**

$$
\int_{\mathcal V}f\,dV=\int_{S^2}\sum_{s=1}^{J(\hat{\boldsymbol\omega})}\int_{r_s^-(\hat{\boldsymbol\omega})}^{r_s^+(\hat{\boldsymbol\omega})}f(r\hat{\boldsymbol\omega})\,r^2\,dr\,d\Omega,
$$

with the segments of the RTFP ray from the field point $O$ (Section 3.6), and $r_1^-=0$ iff $O\in\mathcal V$. The rule is the angular rule times radial nodes per segment: Gauss–Jacobi with weight $r^2$ when $r_s^-=0$, Gauss–Legendre on $fr^2$ otherwise. All weights are positive. The endpoints jump or behave like a square root at silhouette directions, so a fixed angular rule converges slowly. Subdivide the cut angular cells, or tetrahedralize $\mathcal V$. Polynomial integrands (the $\eta$-free matrices of the even basis) are exact by the exact polynomial volume rule.

**(G3) The pressure datum by Nitsche's method [17].** The elliptic Nitsche problem is retained as the static-pressure initializer $p_0$, as the stress-to-pressure affine operator used by limit-analysis bounds, and as an optional global correction every $N_p$ hyperbolic sweeps; it is not a second pressure solve in the reference loop. With $R_*$ the pressure functions do not vanish on $\Gamma$, so $p_\Gamma$ is imposed weakly. With $\mathbf f=\rho_K\mathbf g_{\rm eff}+\nabla\cdot\boldsymbol\tau$ and one global length $h_N=R_{\rm ref}$ (a global geometric scale $R_{\rm ref}$), seek $p_h\in\mathbb Q_h$ with $a_N(p_h,q)=\ell_N(q)$ for all $q\in\mathbb Q_h$:

$$
\begin{aligned}
a_N(p,q)&=\int_{\mathcal V}\nabla p\cdot\nabla q\,dV-\oint_\Gamma\big(\partial_np\,q+p\,\partial_nq\big)\,dS+\frac{\gamma_N}{h_N}\oint_\Gamma p\,q\,dS,\\
\ell_N(q)&=\int_{\mathcal V}\mathbf f\cdot\nabla q\,dV-\oint_\Gamma(\mathbf f\cdot\mathbf n)\,q\,dS-\oint_\Gamma p_\Gamma\,\partial_nq\,dS+\frac{\gamma_N}{h_N}\oint_\Gamma p_\Gamma\,q\,dS .
\end{aligned}
$$

*Consistency.* If $\Delta p=\nabla\cdot\mathbf f$ and $p=p_\Gamma$ on $\Gamma$, then $\int\nabla p\cdot\nabla q-\oint\partial_np\,q=\int\mathbf f\cdot\nabla q-\oint\mathbf f\cdot\mathbf n\,q$, and the other terms coincide. At equilibrium $\nabla p=\mathbf f$ and the two flux terms cancel. On $\Gamma$, $\mathbf f$ means its interior trace $\mathbf f^-=\rho_K\mathbf g_{\rm eff}^-+(\nabla\cdot\boldsymbol\tau)^-$, the limit from inside $\mathcal V$ at the relative interior of each face. The interior identities of Section 8.3 for $\nabla\cdot\mathbf T_{\rm tide}$ and $\nabla\cdot\boldsymbol\tau_{\rm flow}$ hold in this one-sided sense, and the retained basis has a well-defined trace across $\Gamma$ (Proposition 8.1.3), so $(\nabla\cdot\boldsymbol\tau_{\rm flow})^-=\eta\Delta\mathbf v+2\mathbf D\nabla\eta$ is evaluated directly at boundary quadrature points. Edges and vertices are a null set of $\Gamma$.

**Proposition 8.1.6 (Discrete coercivity).** Let $C_N:=\max\{h_N\|\partial_nq\|^2_{L^2(\Gamma)}/\|\nabla q\|^2_{L^2(\mathcal V)}:\ q\in\mathbb Q_h,\ \nabla q\ne\mathbf 0\}$, a generalized eigenvalue of size at most $\dim\mathbb Q_h$. (A $q$ with $\nabla q=\mathbf 0$ is constant on the connected $\mathcal V$ and has $\partial_nq=0$, so $h_N\|\partial_nq\|^2\le C_N\|\nabla q\|^2$ holds for every $q\in\mathbb Q_h$.) If $\gamma_N>2C_N$, then

$$
a_N(q,q)\ge\tfrac12\|\nabla q\|^2_{L^2(\mathcal V)}+\frac{\gamma_N-2C_N}{h_N}\|q\|^2_{L^2(\Gamma)},
$$

and the discrete problem is uniquely solvable.

*Proof.* $2ab\le\delta a^2+b^2/\delta$ with $a=\sqrt{h_N}\|\partial_nq\|$, $b=\|q\|/\sqrt{h_N}$, and $\delta=1/(2C_N)$ gives $2\oint|\partial_nq\,q|\le\tfrac12\|\nabla q\|^2+2C_N\|q\|^2_\Gamma/h_N$. The right-hand side is a norm on $H^1$ by Friedrichs' inequality. ∎

The continuous problem of Section 8.3 holds on every Lipschitz $\mathcal V$; only the discretization changes. The reference route uses the geometry-adapted basis on all admissible domains.

**(G4) The surface: vertices and normal increments.** $\Gamma$ is carried by its vertices, with fixed connectivity. Define

$$
\mathbf A_i:=\frac{\partial\,\mathrm{Vol}}{\partial\mathbf X_i}=\frac13\sum_{f\ni i}A_f\mathbf n_f,\qquad
\mathbf N_i:=\mathbf A_i/|\mathbf A_i|,\qquad A_i=\tfrac13\sum_{f\ni i}A_f .
$$

*Proof of the identity.* $\mathrm{Vol}=\tfrac16\sum_f\mathbf X_a\cdot(\mathbf X_b\times\mathbf X_c)$, so $\partial\mathrm{Vol}/\partial\mathbf X_i=\tfrac16\sum_{f\ni i}\mathbf X_b\times\mathbf X_c$ for oriented faces $(i,b,c)$. Also $A_f\mathbf n_f=\tfrac12(\mathbf X_b\times\mathbf X_c+\mathbf X_i\times\mathbf X_b+\mathbf X_c\times\mathbf X_i)$, and $\sum_{f\ni i}(\mathbf X_b-\mathbf X_c)=\mathbf 0$ around a closed ring. ∎

An update is $\mathbf X_i\leftarrow\mathbf X_i+d_i\mathbf N_i$. Then $\partial\mathrm{Vol}/\partial d_i=|\mathbf A_i|$ exactly, and

$$
(D\mathcal M)_i=\rho_K(\mathbf X_i)\,|\mathbf A_i| ,
$$

exact for constant density and first-order otherwise.

**Proposition 8.1.7 (Kinematics in normal form).** The material condition for $\Gamma(t)$ is $V_n=\mathbf v\cdot\mathbf n$, where $V_n$ is the normal speed; tangential motion of the parametrization is free. (A material surface moves with the material normally, and tangential motion is a relabelling. For $\mathbf X=R\hat{\boldsymbol\omega}$, $\dot{\mathbf X}\cdot\mathbf n=\partial_tR\,(\hat{\boldsymbol\omega}\cdot\mathbf n)$.)

*Clearance.* Normal offsets can collide across thin gaps or fold concave regions. With $t_i^\pm$ the distance along $\pm\mathbf N_i$ to the next crossing of $\Gamma$ (two first-hit queries; $t_i^+=\infty$ if unobstructed), the admissible increments satisfy

$$
-\theta_g\,t_i^-\le d_i\le\theta_g\,t_i^+,\qquad\theta_g\in(0,\tfrac12),
$$

linear rows of $\mathcal A_{\rm lin}$ (Section 10.3.3). They keep the update embedded, which is what the geometry needs; no condition on $\mathbf n\cdot\mathbf e_r$ enters. The self-intersection test is kept as the outer check.

**(G5) Surface transport of $\rho_s$.** The layer is carried as vertex masses $m_i=\rho_{s,i}A_i$, along the tangential slip $\mathbf v_T=\chi_A(\mathbf v_{\rm sl}-(\mathbf v_{\rm sl}\cdot\mathbf n)\mathbf n)|_\Gamma$.

- **Foot point:** trace the geodesic of $-\tau_n\mathbf v_T(\mathbf X_a)$ on the piecewise-flat surface, straight within a face and unfolded across edges.
- **Value:** barycentric interpolation of $\rho_s$ in the face that contains the foot point.
- **Jacobian:** $\mathcal J_a\approx1-\tau_n\nabla_\Gamma\!\cdot\mathbf v_T+\tau_n^2\det(\nabla_\Gamma\mathbf v_T)$, with $\nabla_\Gamma\mathbf v_T$ fitted on the vertex ring in a tangent frame.
- **After the normal update:** $\rho_{s,i}=m_i/A_i^{\rm new}$. This accounts for the area change $\rho_sHV_n$ ($H$ the mean curvature) in $\partial^\bullet_t\rho_s+\rho_s\nabla_\Gamma\!\cdot\mathbf v_\Gamma=J_{\rm dep}-J_{\rm ero}$.

Single-valuedness needs $\tau_n\|\nabla_\Gamma\mathbf v_T\|_\infty\le\theta_{\rm ch}$.

**(G6) The depth.** $d(\mathbf x)=\operatorname{dist}(\mathbf x,\Gamma)$, $\nabla d=-\mathbf n(\pi(\mathbf x))$ a.e., and $|\nabla d|=1$ off the medial axis (a null set), from BVH closest-point queries (Section 4.3a).

**(G7) The classification mesh.**

$$
\mathcal X_Y=\mathcal X_{\rm vol}\cup\{\mathbf X_i-\xi_q\,t_i^-\mathbf N_i:\ q=1,\dots,N_r\}\cup\{\mathbf X_i\},\qquad0<\xi_1<\dots<\xi_{N_r}<1 .
$$

Adjacency joins successive probes on a normal, probes of equal $q$ on mesh-adjacent vertices, and each point to its $k$ nearest points of $\mathcal X_{\rm vol}$ along segments inside $\mathcal V$ (any-hit query). The core, tie rule, and $\chi_{Y,i}$ are those of (B3), and the radial mesh of (B3) is the star-shaped instance.

*Evaluation by inheritance.* On a triangulation obtained by subdivision, every vertex of the finest level is the midpoint of an edge $(a,b)$ of the previous level. $\mathbf v$, $\mathbf D$, $\tilde p$, and $F_\delta$ are evaluated exactly on $\mathcal X_{\rm vol}$ and on the probes and vertices of the previous level. The point of layer $q$ of a midpoint vertex takes the mean of the layer-$q$ values of its parents when both have $F_\delta\le-\kappa_Y(1-\delta_Y)\tau_y$, $\kappa_Y=0.05$; otherwise it is evaluated exactly. An inherited point is unyielded and carries no yielded rate. $F_\delta$ of the exact field stays negative there whenever it varies by less than $\kappa_Y(1-\delta_Y)\tau_y$ over half an edge, which holds because the resolved field varies on lengths of order $h_{\rm vol}$, far above the edge length. Its velocity is the exact midpoint value for a rigid motion, the only motion an unyielded block has. The yield set, the core, and the slip velocity of every yielded or near-yield point are those of the exact evaluation. Without a subdivision hierarchy every point is evaluated exactly.

**(G8) The frame realignment.** Step 8b is a rigid motion of the vertices, $\mathbf X_i\leftarrow\mathbf Q(\mathbf X_i-\mathbf x_{\rm cm})$, and of the cores. There is no resampling, $\rho_s$ stays on its vertices, and the mass is unchanged to rounding, so $E_{\rm frame}=0$. The general route has no radial resampling and therefore no associated error term.

**(G9) The runout.** A packet lands at the first crossing of its drift segment with $\Gamma$. The surface point below $\mathbf x$ is $\pi(\mathbf x)$, and the escape radii use $\max_\Gamma|\mathbf x|$. Relief and deposits are normal thicknesses with $(D\mathcal M)_i$ of (G4).

**(G10) Time-step bounds.** $\Delta t_{\rm CFL}=\theta_{\rm CFL}h_{\Gamma,\min}/\max_{\{i:\chi_{Y,i}=1\}}|\mathbf v_{{\rm sl},i}\cdot\mathbf N_i|$ and $\Delta t_{\rm char}=\theta_{\rm ch}/\|\nabla_\Gamma\mathbf v_T\|_\infty$.

**What does not depend on star-shapedness:**

- the gravity of Chapters 1–3, which already allows nonconvex polyhedra with cavities (Section 1.1) and builds every ray's segments from its crossings (Section 3.6);
- the rheology of Chapter 4;
- the slope and anti-overhang rows (face normals, face gravity);
- the core transport of Section 9.2;
- the shear rate of Section 10.1.1 and the friction-angle chain of Section 10.2.

---

## 8.2 Free-surface kinematics and the mass constraint

The material condition for $\Gamma(t)$ is

$$
\boxed{\ V_n=\mathbf v\cdot\mathbf n\quad\text{on }\Gamma\ }
$$

(Proposition 8.1.7), discretized by the normal increments of (G4) with the velocity trace $\mathbf v(\mathbf X_i)$ of Section 10.1. The mass constraint $\int_{\mathcal V}\rho\,dV=M$ is evaluated by the ray-segment rule (G2), or exactly for the $\rho_b$ part by the exact polynomial volume rule, with $\partial\mathcal M/\partial d_i=\rho_K(\mathbf X_i)|\mathbf A_i|$. It fixes the mean normal increment, which the kinematics does not advance. The geological step is governed by Section 10.3.1.



## 8.3 Hyperbolic ray solver for the non-star-shaped rheology

The independent elliptic pressure solve is removed from the reference loop. On frozen geometry and frozen constitutive coefficients, the quasi-static Stokes problem is embedded in a dissipative hyperbolic relaxation system. The relaxation has the same stationary equations as the original problem, while its principal part is hyperbolic and can be advanced along characteristic rays.

Let $\mathbf v$ be the velocity, $\mathbf S$ the flow stress, and $p$ the pressure. With pseudo-time $\tau$, define

$$
\boxed{
\begin{aligned}
\rho_\tau\partial_\tau\mathbf v-\nabla\cdot\mathbf S+\nabla p&=\mathbf f,\\
T_s\partial_\tau\mathbf S+\mathbf S&=2\eta^{(m)}\mathbf D(\mathbf v),\\
\kappa\partial_\tau p+\nabla\cdot\mathbf v&=g.
\end{aligned}}
\tag{8.3-H}
$$

Here $\rho_\tau>0$, $T_s>0$, and $\kappa>0$ are numerical relaxation parameters, $g=0$ in the incompressible reference model and $g=\dot\varepsilon_v$ under A-$\Xi$. The viscosity $\eta^{(m)}=\eta(\tilde p^{(m)},\dot\gamma^{(m)})$ is frozen during outer rheology pass $m$. The tide is included by replacing $\mathbf S$ in the momentum flux by $\mathbf S+\mathbf T_{\rm tide}$.

At a stationary state, (8.3-H) gives

$$
-\nabla\cdot\mathbf S+\nabla p=\mathbf f,
\qquad
\mathbf S=2\eta^{(m)}\mathbf D(\mathbf v),
\qquad
\nabla\cdot\mathbf v=g,
$$

which is the frozen-viscosity quasi-static problem. Thus the hyperbolic system changes the solver path, not the stationary equations.

**Boundary condition.** The traction condition is imposed as

$$
(-p\mathbf I+\mathbf S+\mathbf T_{\rm tide})\mathbf n=\mathbf t_{\rm ext}
\quad\text{on }\Gamma.
$$

The incoming characteristic components are determined from this traction relation; outgoing components are obtained from interior ray propagation. The boundary condition is imposed only on incoming modes, so the hyperbolic problem is not over-constrained.

**Characteristic form and full eigensystem.** Let $\mathbf U=(\mathbf v,\mathbf S,p)\in\mathbb R^{10}$ (three velocity components, six independent symmetric stress components, and one pressure). For a unit propagation direction $\mathbf n$ completed to an orthonormal triad $\{\mathbf n,\mathbf t_1,\mathbf t_2\}$, define the principal symbol $A(\mathbf n)=\sum_j n_jA_j$ of (8.3-H). Its characteristic pairs $A(\mathbf n)r_\alpha=\lambda_\alpha r_\alpha$ split into three families for frozen positive $\rho_\tau$, $T_s$, $\kappa$, and $\eta$:

1. *Longitudinal (P-wave) family* ($\alpha=P,\pm$, multiplicity $1+1$):
   $$
   \lambda_{P,\pm}(x)=\pm c_{\rm hyp}(x),\qquad
   c_{\rm hyp}(x):=\sqrt{\frac{2\eta(x)}{\rho_\tau(x)T_s}+\frac{1}{\rho_\tau(x)\kappa}},
   $$
   with characteristic amplitudes $w_{P,\pm}=\rho_\tau c_{\rm hyp}\,(\mathbf v\cdot\mathbf n)\mp(\mathbf n\cdot\mathbf S\mathbf n-p)$.
2. *Transverse shear (S-wave) families* ($\alpha=(S,a,\pm)$ for $a\in\{1,2\}$, multiplicity $2+2$):
   $$
   \lambda_{S,a,\pm}(x)=\pm c_s(x),\qquad
   c_s(x):=\sqrt{\frac{\eta(x)}{\rho_\tau(x)T_s}},
   $$
   polarized along $\mathbf t_a\perp\mathbf n$, with characteristic amplitudes $w_{S,a,\pm}=\rho_\tau c_s\,(\mathbf v\cdot\mathbf t_a)\mp(\mathbf n\cdot\mathbf S\mathbf t_a)$.
3. *Stationary (non-propagating) relaxation modes* ($\lambda_0=0$, multiplicity $4$): the three in-plane stress components $\mathbf t_a\cdot\mathbf S\mathbf t_b$ ($1\le a\le b\le 2$) and the normal-stress/pressure invariant $\kappa^{-1}(\mathbf n\cdot\mathbf S\mathbf n)+(2\eta/T_s)p$, which do not propagate along $\mathbf n$ and relax in place at $x$.

At a boundary point with outward normal $\mathbf n$, the three incoming amplitudes $(w_{P,-},w_{S,1,-},w_{S,2,-})$ are uniquely determined from the three outgoing interior amplitudes $(w_{P,+},w_{S,1,+},w_{S,2,+})$ and the three components of the traction condition $(\mathbf S-p\mathbf I)\mathbf n=\mathbf t_{\rm ext}-\mathbf T_{\rm tide}\mathbf n$ (one normal component $\mathbf n\cdot\mathbf S\mathbf n-p$ and two shear components $\mathbf n\cdot\mathbf S\mathbf t_a$). Choosing $\rho_\tau=\rho_K$ makes the Cauchy density a propagation metric while leaving the stationary equations unchanged. The characteristic path is integrated in Hamiltonian form

$$
\frac{d x}{d\tau}=\nabla_k\omega_\alpha(x,k),
\qquad
\frac{d k}{d\tau}=-\nabla_x\omega_\alpha(x,k),
$$

with $\omega_\alpha(x,k)=\lambda_\alpha(x,\widehat k)\lVert k\rVert$. Within one BVH cell the path is approximated by a straight segment; at a cell boundary the local coefficients and characteristic direction are refreshed. This is the Cauchy-GRIN part of the method.

**Non-star-shaped ray geometry.** A surface BVH contains all boundary triangles and a cell BVH contains the volume cells or their AABBs. A ray is traced as an ordered list of segments. After a hit at $s_h$, the next query starts at $s_h+\varepsilon_{\rm geom}$, so concavities, cavities, thin necks, and multiple boundary crossings are handled without a radial function. Each segment returns its cell ID, length, triangle ID when present, and barycentric coordinates for interpolation.

**Ray update.** For every volume sample $x_m$ and characteristic family $\alpha$, backtrace a ray for one pseudo-time step. Interpolate $\mathbf U$ at its upstream point, project to the left characteristic vector, integrate the source term, and reconstruct with the right characteristic vectors. The stress relaxation is integrated exactly for frozen $\eta^{(m)}$:

$$
\mathbf S^{n+1}=e^{-\Delta\tau/T_s}\mathbf S^{\rm up}
+\bigl(1-e^{-\Delta\tau/T_s}\bigr)2\eta^{(m)}\mathbf D^{\rm up}.
$$

The momentum source and the A-$\Xi$ volume source use midpoint quadrature. At a boundary hit, outgoing modes are taken from the interior and incoming modes are solved from the traction equation.

**Energy check.** For constant frozen $\eta$, periodic or zero-power boundary data, define

$$
E=\frac{\rho_\tau}{2}\int|\mathbf v|^2dx
+\frac{T_s}{4\eta}\int|\mathbf S|^2dx
+\frac{\kappa}{2}\int p^2dx.
$$

Then

$$
\frac{dE}{d\tau}+\frac{1}{2\eta}\int|\mathbf S|^2dx
=\int\mathbf f\cdot\mathbf v\,dx+\int pg\,dx.
$$

This identity is the stability check for the frozen-coefficient solve. Full non-monotone $\mu(I)$ coupling remains an outer fixed point; no unconditional global existence or uniqueness is claimed.

**Pressure, incompressibility, and rigid modes.** The artificial-compressibility residual is

$$
\nabla\cdot\mathbf v-g=-\kappa\partial_\tau p.
$$

Pressure correction is applied every configured number of ray sweeps, and the mass-weighted translation and rotation modes are projected out using the $6\times6$ rigid-mode system of Appendix A.3. The step is accepted only when momentum, constitutive, divergence, traction, and rigid-mode residuals meet the tolerances of Section 2.15.

**Ray-solver algorithm.**

```text
Build or refit SurfaceBVH and CellBVH for the current non-star-shaped mesh.
For rheology pass m:
  evaluate p~, D(v), gamma_dot, mu_d, eta^(m), rho_tau=rho_K;
  compute local characteristic speeds and eigenvectors;
  repeat pseudo-time ray sweeps:
    for every volume sample and characteristic family:
      integrate the backward characteristic over Delta_tau;
      trace all crossed BVH segments and boundary hits;
      interpolate the upstream state;
      integrate source terms and exact stress relaxation;
      reconstruct (v,S,p);
    impose incoming traction modes at boundary hits;
    apply pressure correction and rigid-mode projection;
    test momentum, divergence, constitutive, traction, and energy residuals;
  update eta from the converged (v,p) and repeat the outer pass.
Stop when the outer rheology residuals meet Section 2.15.
```

The CFL condition is

$$
\Delta\tau\le C_{\rm CFL}\frac{h_{\min}}{\max_{x,\alpha}|\lambda_\alpha(x)|}.
$$

As $h\to0$, $\Delta\tau\to0$, and the ray sweeps converge, the stationary state tends to the frozen-viscosity Stokes solution. The geometric increment SOCP of Sections 2.13 and 10.3.3 is retained; the ray solver replaces the former elliptic rheology solver, not the geometric projection problem.

**Galerkin projection bridge.** CellBVH stores the piecewise-polynomial sample field $\mathbf U^{\rm ray}=(\mathbf v^{\rm ray},\mathbf S^{\rm ray},p^{\rm ray})$ used by characteristic tracing; it is not identified with the global modal coefficients. After a ray sweep reaches its stopping tolerance, define the projected velocity coefficients $u_\alpha$ by

$$\sum_\beta (\mathbf M_\Gamma)_{\alpha\beta}u_\beta=\int_{\mathcal V}\rho_K\,\mathbf B_\alpha\cdot(\mathbf v^{\rm ray}-\mathbf v_\varepsilon)\,dV,\qquad \mathbf v_h=\mathbf v_\varepsilon+\sum_\alpha u_\alpha\mathbf B_\alpha.$$

The integral is evaluated with the CellBVH quadrature and the left-hand matrix is the same constrained mass matrix used in Section 8.1. The six rigid coordinates are projected out with Appendix A.3. Surface traces, derivatives, marker advection, and the weak momentum residual use $\mathbf v_h$; the stress and pressure residuals use the sampled $\mathbf S^{\rm ray},p^{\rm ray}$ together with their CellBVH quadrature. The projection defect $e_{\rm proj}=\|\mathbf v^{\rm ray}-\mathbf v_h\|_{L^2(\rho_K)}$ is reported and included in PO-11/PO-19. Under A-Ξ, $\mathbf v_\varepsilon$ is included before projection, so the solenoidal basis represents only the remainder. This is the required connection between the characteristic sampler and $\mathbb V_h^\sigma$; it is not an assertion that the two discrete spaces are identical.

## 8.4 The internal gravitational load

With $\mathbf g=\nabla U=G\int\rho(\mathbf y)(\mathbf y-\mathbf x)|\mathbf x-\mathbf y|^{-3}d\mathbf y$, pointing toward the mass, the internal gravity depends on the internal density. It is not determined by the surface alone. A surface-only evaluation is an approximation, valid only for a known fixed density, a thin shell, a precomputed multipole expansion, or a separately corrected internal mass change. The load is

$$
\boxed{\ (\mathbf F_g)_\alpha=\int_{\mathcal V}\rho\,\mathbf g_{\rm RTFP}\cdot\mathbf B_\alpha\,dV\ }
$$

against every test of degree $\ell\ge1$. The same load can be written through surface fluxes.

**Proposition 8.4.1 (Loads as surface fluxes).** For $\mathbf v\in\mathbb V_h^\sigma$,

$$
\int_{\mathcal V}\rho_K\nabla W\cdot\mathbf v\,dV=\oint_\Gamma\rho_KW\,(\mathbf v\cdot\mathbf n)\,dS-\int_{\mathcal V}W\,\nabla\rho_K\cdot\mathbf v\,dV,
$$

$$
\int_{\mathcal V}\mathbf T_{\rm tide}:\mathbf D(\mathbf v)\,dV=C_T\rho_p\nu_0d_p^2\Big[\oint_\Gamma(H_W^-\mathbf n)\cdot\mathbf v\,dS+4\pi G\int_{\mathcal V}\nabla\rho_K\cdot\mathbf v\,dV\Big],
$$

where $H_W^-=H_W^{\rm FP}-2\pi G\rho_K\,\mathbf n\otimes\mathbf n$ is the interior trace on the relative interior of each face (Proposition 2.7.1(5)).

*Proof.* The first is the divergence theorem with $\nabla\cdot\mathbf v=0$. For the second, $\operatorname{dev}H_W:\mathbf D=H_W:\nabla\mathbf v$ because $\mathbf D$ is traceless and $H_W$ symmetric; integrating by parts with interior boundary traces gives $\oint(H_W^-\mathbf n)\cdot\mathbf v-\int\nabla\Delta W\cdot\mathbf v$, and $\nabla\Delta W=-4\pi G\nabla\rho_K$ in the interior (Section 8.3). ∎

Both surface integrands are defined almost everywhere on $\Gamma$: Proposition 2.7.1(5) holds at every point of a face's relative interior, not only at its centre, and edges and vertices are a null set of $\Gamma$. Near an edge the interior trace $H_W^-$ grows only logarithmically, which is integrable, and the integration by parts is valid because $\nabla H_W$ is $O(\operatorname{dist}^{-1})$ near edges and vertices, integrable in three dimensions. For a uniform body both loads are therefore surface integrals of face fields; interior gravity at $\mathcal X_{\rm vol}$ is then needed only as the scalar $W$ for the pressure, and hence the strength. A face-centre (one-point) rule integrates a log-singular trace without an a-priori error bound, so the surface form must be refined toward edges, or checked against the volume form, before it replaces it.

---

## 8.5 Rigour and the error report

The volume and ray discretizations are refinement parameters. Every run must report:

1. the hyperbolic ray-sweep count, the number of pressure corrections, and the residuals (a), (b) of Section 2.15;
2. $\varepsilon_\lambda$, $\operatorname{rank}\mathbf B_{\rm rb}$, and its smallest singular value;
3. the hyperbolic momentum, constitutive, divergence, traction, and energy residuals of Sections 2.15 and 8.3; the dynamic benchmark also reports a numerical inf–sup indicator (PO-1);
4. the tangential traction inconsistency $\delta_{na}$ (Section 10.1.1);
5. the RTFP gravity error;
6. the geometry of Section 8.1: the condition number of the $\eta$-free Gram matrix on $\mathcal V$, $C_N$ and $\gamma_N$, and the smallest clearance ratio $\min_i|d_i|/t_i^\pm$.

A smallest normalized singular value of $\mathbf B_{\rm rb}$ below a configured threshold means the rigid constraint is numerically rank-deficient, and the step stops.

The mixed two- and three-dimensional system is a reduced model under stated hypotheses and error control, not an unconditional zero-error reduction of the three-dimensional free-boundary problem.

---

# 9. Geological time stepping and material transport

## 9.1 Quasi-static time stepping

The reference loop advances geological time by $t_{n+1}=t_n+\lambda\Delta t_n$. The step is limited by the movable-region CFL, the erosional flux, and the spin table of Section 10.3.1. The velocity is recomputed by the non-star-shaped hyperbolic ray solver (8.3-H) on each step and is not carried as a dynamic surface field. The density cores are transported between steps by Section 9.2, while the surface is updated by the normal increment and the geometric SOCP correction of Section 10.3.3.


## 9.2 Lagrangian transport and adaptive refit of the Cauchy density cores (hypothesis A-ρL)

Hypothesis A-ρL (Lagrangian cores) applies to the quasi-static route. The alternative A-ρ, a density fixed in the body frame, is kept only as a comparison variant (Section 9.2.1 shows its defect). Two shape classes of cores are admitted:

- *anisotropic* cores, with a full SPD metric $\mathbf S_k$ (the reference choice);
- *isotropic* cores, $\mathbf S_k=\sigma_k^2\mathbf I$, with the projection of Section 9.2.3.

### 9.2.1 The requirement, and the size of the ghost-gravity error

With an exactly divergence-free velocity, mass conservation is the material transport $\partial_t\rho_K+\mathbf v\cdot\nabla\rho_K=0$ (Section 4.18). Under A-Ξ it is $D_t\rho_K=-\rho_K\dot\varepsilon_v$, and Section 11.4 states the changes to Levels 0–3. A density fixed in the body frame while material flows through it has a defect that grows secularly without bound, $\|\delta\rho\|_\infty\approx t\|\mathbf v_{\rm adv}\cdot\nabla\rho_K\|_\infty$: the dense rock has left, but its gravity stays. The following bound, used as the refit's acceptance metric, turns density defects into gravity defects.

**Proposition 9.2.1 (From density defect to gravity defect).** For $\delta\rho\in L^\infty(V)$ supported in $V$, $\delta\mathbf g=\nabla\delta U$ with $\delta U=G\int_V\delta\rho|\mathbf x-\mathbf y|^{-1}$, and $R_{\rm eq}=(3|V|/4\pi)^{1/3}$,

$$
\sup_{\mathbf x\in\mathbb R^3}|\delta\mathbf g(\mathbf x)|\le 4\pi G R_{\rm eq}\,\|\delta\rho\|_{L^\infty(V)},
\qquad
\|\delta\mathbf g\|_{L^2(\mathbb R^3)}\le 4\pi G\,C_S\,|V|^{1/3}\,\|\delta\rho\|_{L^2(V)},
$$

with the sharp Sobolev constant $C_S=3^{-1/2}(2/\pi)^{2/3}\approx0.4273$ of $\|f\|_{L^6}\le C_S\|\nabla f\|_{L^2}$.

*Proof.* (i) $|\delta\mathbf g|\le G\|\delta\rho\|_\infty\int_V|\mathbf x-\mathbf y|^{-2}$. By the bathtub principle this integral is largest for the ball of radius $R_{\rm eq}$ about $\mathbf x$, where it equals $4\pi R_{\rm eq}$. (ii) $-\Delta\delta U=4\pi G\delta\rho$ with the decay at infinity gives $\|\nabla\delta U\|_2^2=4\pi G\int\delta\rho\,\delta U\le4\pi G\|\delta\rho\|_{6/5}C_S\|\nabla\delta U\|_2$, and Hölder gives $\|\delta\rho\|_{6/5}\le|V|^{1/3}\|\delta\rho\|_2$. ∎

### 9.2.2 The advection velocity

The density moves with the **same masked slip velocity that moves the surface** (B4):

$$
\mathbf v_{\rm adv}(\mathbf x):=\chi_Y(\mathbf x)\,\mathbf v_{\rm sl}(\mathbf x),\qquad \tau_n:=\lambda\Delta t_n ,
$$

with $\chi_Y(\mathbf x)$ the movability of the classification cell containing $\mathbf x$ (0 on $\mathcal U_{\rm core}$), and $\mathbf v_{\rm sl}$, $\nabla\mathbf v_{\rm sl}$ synthesized from the geometry-adapted fields of Section 10.1, frozen over the step. The mask is mandatory, for three reasons:

1. density and topography move with one field;
2. the core's regularization creep, excluded from the geometry, does not accumulate in $\rho_K$;
3. the stage fixed point survives: $\chi_Y\equiv0$ on $\mathcal X_Y$ gives $\mathbf v_{\rm adv}\equiv\mathbf 0$, so 2.15(c) stays a true fixed point.

The mask breaks exact solenoidality only across the core boundary. The resulting mass discrepancy is closed in 9.2.4 and reported.

### 9.2.3 Level 0: exact 2-jet transport of every core

The transport is linear, so transporting each $\rho_k$ transports the sum. Each core is a material component, like a dyed fraction of the rock.

**Lemma 9.2.2 (Critical points and Hessians are transported).** For a $C^2$ diffeomorphism $\Phi$, $\tilde\rho=\rho\circ\Phi^{-1}$, $\nabla\rho(\mathbf c)=\mathbf 0$, $\mathbf c'=\Phi(\mathbf c)$, and $\mathbf F=\nabla\Phi(\mathbf c)$: $\tilde\rho(\mathbf c')=\rho(\mathbf c)$, $\nabla\tilde\rho(\mathbf c')=\mathbf 0$, and $\nabla^2\tilde\rho(\mathbf c')=\mathbf F^{-\top}\nabla^2\rho(\mathbf c)\mathbf F^{-1}$. (Chain rule with $\boldsymbol\psi=\Phi^{-1}$; the term with $\nabla\rho(\mathbf c)$ vanishes.)

With $\rho_k(\mathbf c_k)=w_k$ and $\nabla^2\rho_k(\mathbf c_k)=-2\alpha_kw_k\mathbf S_k$ (Section 6.1), matching value, critical point, and Hessian gives the **jet update**

$$
\boxed{\
\mathbf c_k'=\Phi_{\tau_n}(\mathbf c_k),\qquad
w_k'=w_k,\qquad
\mathbf S_k'=\mathbf F_k^{-\top}\mathbf S_k\mathbf F_k^{-1},\qquad
\alpha_k'=\alpha_k,\qquad
\mathbf F_k=\nabla\Phi_{\tau_n}(\mathbf c_k).\ }
$$

Each centre moves exactly with the material, and each metric is deformed exactly by the local deformation gradient.

**Proposition 9.2.3 (Properties of the jet update).**

1. *Local exactness.* The update agrees with $\rho_k\circ\Phi^{-1}_{\tau_n}$ to second order at $\mathbf c_k'$. The defect is $O(|\mathbf x-\mathbf c_k'|^3)$ and proportional to $\nabla^2\Phi_{\tau_n}$, since the kernel is even about its centre.
2. *Affine flows.* For $\mathbf v_{\rm adv}=\mathbf a+\mathbf L\mathbf x$ the update is exact, because $(\Phi^{-1}\mathbf x-\mathbf c)^{\!\top}\mathbf S(\Phi^{-1}\mathbf x-\mathbf c)=(\mathbf x-\mathbf c')^{\!\top}\mathbf F^{-\top}\mathbf S\mathbf F^{-1}(\mathbf x-\mathbf c')$.
3. *Volume and definiteness.* If $\operatorname{tr}\nabla\mathbf v_{\rm adv}=0$, then $\det\mathbf F_k=1$ (Liouville), so $\det\mathbf S_k'=\det\mathbf S_k$. Under A-Ξ, $\det\mathbf S_k'=J_{\nu,k}^2\det\mathbf S_k$ (Section 11.4). $\mathbf S_k'$ is SPD by congruence.
4. *Fixed point.* $\mathbf v_{\rm adv}\equiv\mathbf 0$ gives $\theta'=\theta$.

*Example.* A random traceless affine field gives a defect of $4\times10^{-16}$ and $\det\mathbf S'/\det\mathbf S=1$. For $\mathbf v=(\sin y+z^2,\tfrac12\cos z+xz,-0.3x^2+\sin y)$ and $\tau=0.2$, the maximum defect at $r=0.2,0.1,0.05,0.025$ is $8.2\times10^{-3}$, $8.2\times10^{-4}$, $1.2\times10^{-4}$, $1.4\times10^{-5}$: third order.

**Time discretization (second order for the frozen velocity).**

$$
\mathbf c_k^{\rm mid}=\mathbf c_k+\tfrac{\tau_n}{2}\mathbf v_{\rm adv}(\mathbf c_k),\qquad
\mathbf c_k'=\mathbf c_k+\tau_n\mathbf v_{\rm adv}(\mathbf c_k^{\rm mid}),\qquad
\mathbf F_k=\exp\!\big(\tau_n\mathbf L_k\big),\quad
\mathbf L_k=\operatorname{dev}\nabla\mathbf v_{\rm adv}(\mathbf c_k^{\rm mid}).
$$

The deviatoric projection makes $\det\mathbf F_k=\exp(\tau_n\operatorname{tr}\mathbf L_k)=1$ exactly. The exponential is required: for a trace-free $\mathbf L$ and $\mathbf A=\tfrac{\tau_n}{2}\mathbf L$, the Cayley form $(\mathbf I-\mathbf A)^{-1}(\mathbf I+\mathbf A)$ has determinant $(1+e_2(\mathbf A)+\det\mathbf A)/(1+e_2(\mathbf A)-\det\mathbf A)$ in 3-D ($e_2$ the second invariant), which is $1$ only if $\det\mathbf L=0$. The trace removed by $\operatorname{dev}$ is not physical compression. Without A-Ξ the flow is isochoric, and $\operatorname{tr}\nabla\mathbf v_{\rm adv}\ne0$ arises only where the mask $\chi_Y$ cuts the field at the core boundary. Under A-Ξ the physical dilatation is added back explicitly as $\tfrac13\bar\varepsilon_k\mathbf I$, from the exact ratio $J_{\nu,k}$ of Proposition 11.5, and $w_k'=J_{\nu,k}w_k$ (Section 11.4). Its mass effect is closed by Level 1 (Section 9.2.4) and measured locally by $e_\rho$, which triggers the local refit of Level 2 (Section 9.2.5). The single-valuedness condition $\tau_n\|\nabla\mathbf v_{\rm adv}\|\le\theta_{\rm ch}<1$ is enforced by $\Delta t_{\rm core}$ (Section 10.3.1).

**Anchored cores.** A core centred outside $\mathcal V^{(n)}$ (a tail only), or with $\chi_Y(\mathbf c_k)=0$, has an undefined or masked centre velocity. It is anchored ($\theta_k'=\theta_k$) and corrected only by Levels 1–2. **Isotropic option.** With isotropic cores, $\sigma_k'^2:=(\det\mathbf S'_k)^{1/3}=\sigma_k^2$ preserves volume, and the anisotropic part of the deformation becomes a model error measured by Level 2.

**Why moments are not matched.** For $\alpha_k=1$ a core's mass inside a body of size $L$ grows linearly with radius, so the core region holds only $O((\sigma_kL)^{-1})$ of the mass, and the second moment over $\mathbb R^3$ diverges for every $\alpha_k\le\tfrac52$. Moments over $V$ are dominated by the geometry, not by $\mathbf c_k,\mathbf S_k$, so matching them is ill-conditioned. Jet matching is local and exact.

### 9.2.4 Level 1: mass closure (always)

After step 8a, with the evaluator of the Section 2.13 mass row,

$$
\xi_n:=\frac{\mathsf M_{\rm tot}^{n+1}}{\mathcal M(V^{(n+1)},\theta')}-1,\qquad w_k''=(1+\xi_n)\,w_k'\quad(\text{all }k),
$$

which restores the mass exactly (linearity in the weights). With $\rho_b=0$ every core field scales by $1+\xi_n$; that reuse matters only when the geometry is unchanged, because after a geometric update the fields are re-evaluated anyway (Proposition 3.15.1(b)). $\mathsf M_{\rm tot}$ changes only through $J_{\rm ext}$. Nonzero $\xi_n$ comes from the mask at the core boundary, the Level 0 tail defect, the $O(\tau_n^2)$ jet/geometry splitting, and the SOCP slack. $|\xi_n|>\varepsilon_\xi$ (e.g. $10^{-3}$) triggers Level 2.

*With a matrix density $\rho_b>0$.* $\rho_b$ is kept constant, and the mass is closed on the cores:

$$
\xi_n:=\frac{\mathsf M_{\rm tot}^{n+1}-\rho_b|V^{(n+1)}|}{\mathcal M(V^{(n+1)},\theta')-\rho_b|V^{(n+1)}|}-1,\qquad w_k''=(1+\xi_n)w_k' .
$$

Consequences:

- $\xi_n$ is relative to the core mass, so a total-mass discrepancy is amplified by the total-to-core mass ratio. The closure is skipped below a core mass of $10^{-12}$ of the total.
- Only the core part of the fields scales by $1+\xi_n$; the matrix part $\rho_b\Phi_V$ does not. With per-kernel partial sums (Proposition 3.15.1(b)) the rescaled field still costs $O(|\mathcal X|K)$ on an unchanged geometry.

After a Level 2 refit the closure is applied once more. Under A-Ξ the matrix volume changes with the dilatation, and the closure is replaced by the mass-constrained two-scalar fit of Section 11.4, which may also rescale $\rho_b$.

### 9.2.5 Level 2: adaptive least-squares refit (triggered)

*Target.* On the volume nodes $\mathbf x_q\in\mathcal V^{(n+1)}$ (weights $\omega_q$), the transported density is evaluated semi-Lagrangian-wise and **exactly in space**:

$$
\tilde\rho_q:=\rho_K\big(\mathbf X_d(\mathbf x_q);\theta^{(n)}\big),\qquad
\mathbf X_d(\mathbf x)=\mathbf x-\tau_n\mathbf v_{\rm adv}\big(\mathbf x-\tfrac{\tau_n}{2}\mathbf v_{\rm adv}(\mathbf x)\big).
$$

The old density is analytic on $\mathbb R^3$, so the only error is the $O(\tau_n^3)$ trajectory error per step. Velocities at departure points outside $\mathcal V^{(n)}$ are taken at the closest point $\pi(\mathbf x)\in\Gamma^{(n)}$ (Section 8.1, G6). Under A-Ξ the target is multiplied by the dilatational ratio of the arriving material point, $\tilde\rho_q:=J_\nu(\mathbf x_q)\rho_K(\mathbf X_d(\mathbf x_q);\theta^{(n)})$. The non-amplification bound below then carries the factor $(\nu_{\rm hi}/\nu_{\rm lo})^{1/2}$ (Section 11.4).

*Misfit indicator* (always reported):

$$
e_{\rho,n}:=\frac{\big(\sum_q\omega_q(\rho_K(\mathbf x_q;\theta)-\tilde\rho_q)^2\big)^{1/2}}{\big(\sum_q\omega_q\tilde\rho_q^2\big)^{1/2}} ,
$$

which by Proposition 9.2.1 bounds the $L^2$ gravity defect, times $4\pi GC_S|V|^{1/3}\|\tilde\rho\|_{L^2}$, up to quadrature.

*Problem.* If $e_{\rho,n}>\varepsilon_\rho$ or $|\xi_n|>\varepsilon_\xi$, solve

$$
\min_{\theta}\ \tfrac12\sum_q\omega_q\big(\rho_K(\mathbf x_q;\theta)-\tilde\rho_q\big)^2+\tfrac{\mu_{\rm LM}}2\|\theta-\theta''\|_{\mathbf P}^2
\quad\text{s.t.}\quad\mathcal M_h(V^{(n+1)},\theta)=\mathsf M_{\rm tot}^{n+1},
$$

with:

- unknowns $\theta=\{w_k,\mathbf c_k,\mathbf S_k\}$, $\alpha_k$ fixed;
- $\mathbf S_k=\mathbf L_k\mathbf L_k^{\!\top}$ by a Cholesky factor with logarithmic diagonal, so every iterate is SPD;
- a diagonal unit scaling $\mathbf P$ and Levenberg–Marquardt damping $\mu_{\rm LM}$;
- Gauss–Newton steps with the analytic derivatives of Section 6.1. Each step assembles $\mathbf J^{\!\top}\mathbf W\mathbf J$ ($p=10K$ anisotropic, $5K$ isotropic) by one GPU reduction, and solves the $(p+1)$ KKT system with the linearized mass row.

A step is accepted only if the objective decreases; otherwise $\mu_{\rm LM}$ is increased. Stop at $e_\rho\le\varepsilon_\rho/2$ or a relative decrease below $10^{-3}$. Warm-starting from Level 0, already second-order exact at each core, keeps the iterations few.

*Non-amplification.* If the exact transport $\mathcal P_n:\rho\mapsto\rho\circ\Phi^{-1}_{\tau_n}$ is volume-preserving and maps the material domain onto itself, it is an $L^2$ isometry. With $e^n=\rho_K^n-\rho^n_{\rm true}$,

$$
\|e^{n+1}\|_{L^2}\le\|\rho_K^{n+1}-\mathcal P_n\rho_K^n\|_{L^2}+\|\mathcal P_n e^n\|_{L^2}
=\delta_n+\|e^n\|_{L^2},
\qquad
\|e^N\|_{L^2}\le\|e^0\|_{L^2}+\sum_{n<N}\delta_n ,
$$

with $\delta_n$ the per-step defect measured by $e_{\rho,n}$. The error accumulates at most additively, never exponentially, and not at all when nothing moves. The idealizations (exact trajectories, a material domain) fail through the trajectory error, the mask, and Mode C, which are reported through $\xi_n$.

### 9.2.6 Level 3: change of the number of cores (optional)

Without Level 3, $K$ is fixed for the run. $K$ Cauchy kernels cannot represent a core stretched into a long tongue. Level 3, with tolerances $\varepsilon_{\rm split}$, $\varepsilon_{\rm merge}$ and a cap $K_{\max}$, adapts $K$. With the partition $\psi_k=|\rho_k|/\sum_j|\rho_j|$ and per-core misfit $e_k=\|\psi_k(\rho_K-\tilde\rho)\|/\|\psi_k\tilde\rho\|$ after Level 2:

- **Split.** If $e_k>\varepsilon_{\rm split}$ and $K<K_{\max}$, replace core $k$ by two cores with the same $\alpha_k$:
  - centres $\mathbf c_k\pm\tfrac12\lambda_1^{-1/2}\mathbf e_1$, with $(\lambda_1,\mathbf e_1)$ the smallest eigenpair of $\mathbf S_k$ (the long axis);
  - metrics $\mathbf S_k+3\lambda_1\mathbf e_1\mathbf e_1^{\!\top}$ (half the width along $\mathbf e_1$);
  - a common weight $\bar w$ that preserves the core's mass inside $\mathcal V^{(n+1)}$ (linear in $\bar w$).

  Then apply Level 2 with $K+1$ cores.
- **Merge.** If $\alpha_i=\alpha_j$ and $(\mathbf c_i-\mathbf c_j)^{\!\top}(\mathbf S_i^{-1}+\mathbf S_j^{-1})^{-1}(\mathbf c_i-\mathbf c_j)<\varepsilon_{\rm merge}$, replace the pair by one core with weight-averaged centre and metric and the pair's mass. Then apply Level 2.
- **Acceptance.** A change is kept only if $e_\rho$ drops by the configured factor and $\rho_K\ge\rho_{\min}$ still holds at the nodes; otherwise it is reverted. Every change is logged. A change of $K$ invalidates the per-kernel partial sums of Proposition 3.15.1(b), not the segment table.
- **Spawn (optional, under A-Ξ).** A dilatational deficit far from every core cannot be fitted by Levels 1–2 with fixed $K$. Section 11.4 adds a signed core at the largest residual cluster, under the same acceptance rule.

### 9.2.7 Placement in the loop, frame realignment, and cost

- **Step 6b (prediction, after the classification and the step size).** Level 0 gives $\theta^{\rm pred}$. The SOCP of step 7 uses $\rho_{\rm cand}=\rho_K(\cdot;\theta^{\rm pred})$ in its mass row and $D\mathcal M$. With the old density, an outward flow of dense material would be offset by a spurious geometric correction. An outer retry with smaller $\lambda$ recomputes Level 0 ($O(K)$).
- **Step 8c (after 8a, before 8b).** Levels 1–3. The realignment then uses the **updated** density.
- **Step 8b(ii)/(iv).** $\mathbf c_k\leftarrow\mathbf Q(\mathbf c_k-\mathbf x_{\rm cm})$ **and** $\mathbf S_k\leftarrow\mathbf Q\mathbf S_k\mathbf Q^{\!\top}$; the metric must rotate with the frame.
- **Gravity.** Only $\theta$ changed in 8c, so the next step 1 re-evaluates from the cached segment table without traversal (Proposition 3.15.1). Retracing is driven by geometry alone (Proposition 3.15.2).
- **Mode C.** Loose-material relocation stays geometric (via $\Delta h_C^*$ and $\rho_K|_\Gamma$). The cores carry the bulk rock.

**Cost per step:**

- Level 0: $K$ pointwise syntheses of $\mathbf v,\nabla\mathbf v$ and $K$ $3\times3$ matrix exponentials;
- Level 1: one mass integral;
- Level 2, when triggered: $N_{\rm GN}$ reductions of $O(N_qp^2)$ and $(p+1)$-dimensional solves;
- field update: passes W3–W4 only.

All of this is small beside one outer rheology pass.

**Reported every step:**

- $|\xi_n|$ and $e_{\rho,n}$;
- the Gauss–Newton count and the number of anchored cores;
- the core displacements $|\mathbf c_k'-\mathbf c_k|$ and $\kappa(\mathbf S_k)$;
- $\min_q\rho_K(\mathbf x_q)/\rho_{\min}$;
- every split or merge.

---

This chapter contains the quasi-static geological step and the Cauchy-core transport used by the surface runout and geometric closure.

# 10. Geometric SCF coupling and hyperbolic ray relaxation

The chapter couples the general polyhedral volume discretization, the surface trace, the friction update, the geometric subproblem, and the hyperbolic ray relaxation. Contents:

- 10.1, the surface velocity trace, the surface shear rate, and the traction condition;
- 10.2, the friction angle from the shear rate;
- 10.3, the geometric subproblem and the SCF loop.

---

## 10.1 Free-surface velocity, the surface-invariant shear rate, and the traction condition

**General route.** The retained geometry-adapted basis is evaluated at surface quadrature points, so its trace and strain rate on $\Gamma$ are pointwise values. At a surface point $\mathbf x\in\Gamma$, evaluate the trace and first derivative by the retained geometry-adapted basis, $\mathbf v(\mathbf x)=\sum_\alpha c_\alpha\mathbf B_\alpha(\mathbf x)$ and $\nabla\mathbf v(\mathbf x)=\sum_\alpha c_\alpha\nabla\mathbf B_\alpha(\mathbf x)$. These values enter the predictor (B4), the shear rate of Section 10.1.1, and the friction chain of Section 10.2 at vertices and face centres.

### 10.1.1 The surface-invariant shear rate

Take local coordinates $x^a$ on $\Gamma$, with $\mathbf e_a=\partial_aX$, metric $g_{ab}$, unit normal $\mathbf n$, second fundamental form $K_{ab}:=\mathbf e_a\cdot\partial_b\mathbf n$ ($=g_{ab}/R>0$ on a sphere), and mean curvature $H=g^{ab}K_{ab}$ ($=2/R>0$ on a sphere with the outward normal). This sign is fixed throughout. A formula taken from a source with the opposite convention must have $K_{ab}$ and $H$ negated before use; otherwise $D_{nn}$ changes sign, and with it $\dot\gamma_\Gamma$ and $\mu_d$ of Section 10.2. Write $v_a=\mathbf v\cdot\mathbf e_a$ and $v_n=\mathbf v\cdot\mathbf n$ ($=V_n$). With the Levi-Civita derivative $\nabla_a$,

$$
\boxed{
D_{ab}=\nabla_{(a}v_{b)}+v_n\,K_{ab},\qquad
D_{nn}=-g^{ab}D_{ab}=-\big(\nabla_a v^a+H\,v_n\big),\qquad
D_{na}=\mathbf n\cdot\mathbf D\cdot\mathbf e_a.}
$$

*Derivation.* $D_{ab}=\tfrac12(\mathbf e_a\cdot\partial_b\mathbf v+\mathbf e_b\cdot\partial_a\mathbf v)$ involves only tangential derivatives. With $\mathbf v=v^c\mathbf e_c+v_n\mathbf n$ and Gauss–Weingarten ($\partial_a\mathbf e_b=\Gamma^c_{ab}\mathbf e_c-K_{ab}\mathbf n$, $\partial_a\mathbf n=K_a{}^c\mathbf e_c$), $\mathbf e_a\cdot\partial_b\mathbf v=\nabla_bv_a+v_nK_{ab}$. $D_{nn}$ contains a normal derivative, so it comes from $\operatorname{tr}\mathbf D=0$. ∎

Since $\dot\gamma^2=2\mathbf D:\mathbf D$,

$$
\boxed{
\dot\gamma_\Gamma^2
=2\Big[(g^{ab}D_{ab})^2+g^{ac}g^{bd}D_{ab}D_{cd}+2\,g^{ab}D_{na}D_{nb}\Big].}
$$

Under A-Ξ, $\operatorname{tr}\mathbf D=\dot\varepsilon_{v,h}$, so $D_{nn}=\dot\varepsilon_{v,h}-g^{ab}D_{ab}$, and $\dot\gamma^2=2\mathbf D':\mathbf D'$ gains the term $-\tfrac23\dot\varepsilon_{v,h}^2$ (Section 11.4).

**Traction condition.** On a free surface $\boldsymbol\sigma\mathbf n=-p_{\rm ext}\mathbf n$ with $\boldsymbol\sigma=-p\mathbf I+2\eta\mathbf D+\mathbf T_{\rm tide}$. The normal part gives $p|_\Gamma=p_{\rm ext}+\mathbf n\cdot\boldsymbol\tau\mathbf n$, and the tangential part

$$
\boxed{
D_{na}=-\frac{\mathbf n\cdot\mathbf T_{\rm tide}\cdot\mathbf e_a}{2\eta}\quad(\eta>0).}
$$

In the weak form this is natural and holds only in the Galerkin sense. It is also used as a **consistency relation**: with $\eta^{(n)}$ and $\mathbf T^{(n)}_{\rm tide}$ frozen it gives $D_{na}$, from which $\dot\gamma_\Gamma$ follows. Without a tide, $D_{na}=0$ and the third term drops. Since $\eta$ depends on $\dot\gamma$, freezing $\eta^{(n)}$ is an iterative approximation, not an identity. **Self-consistency report:** $\delta_{na}:=\|D^h_{na}-D^{\rm target}_{na}\|_{L^2(\Gamma)}$ is reported as $K$ and $h_{\rm vol}$ vary. It measures how well the reduced model meets the traction condition.

## 10.2 Closure of the endogenous, nonuniform, physical friction angle on the surface

$p_{\rm ref}=\rho_{\rm bulk}|\mathbf g\cdot\mathbf n|h_s$ is a representative pressure of the shallow flowing layer, and $\mu_{\rm eff}$ built from it is a nominal geometric projection angle. It is monotone in pressure without cohesion or at high confinement, but not conservative in general; true multiaxial yield is checked by the outer acceptance test. Two effective pressures are kept apart, and no unsubscripted $p_*$ is used:

- $p_{*,\rm phys}=p+p_c$, only for the dynamic rheology $\mu_d$;
- $p_{*,\rm ref}=p_{\rm ref}+p_c$, only for $\mu_{\rm eff}$.

Everything here lives on the surface ($d=0$), so $c=c_s$ (Section 4.3a); under A-ϖ or A-Θ, $c=c_{\Gamma,i}$ of Section 11.6.

**Material constants are frozen.** $\nu=\nu_0$ (e.g. $0.6$ [5][6]) and $Z=Z_0$ (e.g. $6$) are configuration constants with no evolution law. Hence $\rho_{\rm bulk}=\rho_p\nu_0$, $c$, $\mu_s$, $\mu_2$, $I_0$, $\mu_y\equiv\mu_s$, and $F_H$ are constants. Under A-Ξ (Section 11.3), the continuum solid fraction $\nu$ evolves on the markers and enters $\mu_d$ and $\mu_y$ below through $\mathcal R\nu(\mathbf X_i)$. $\rho_{\rm bulk}$, $p_c$, and $\mathbf T_{\rm tide}$ keep $\nu_0$. The $(n)$ of Sections 4.7–4.20 indexes a parameter sequence, not an SCF iteration. $\rho_{\rm bulk}$ (the packing of the loose layer) and $\rho_K$ (the continuum density) are independent, linked only through $\rho_{\rm bulk}/\rho_K|_\Gamma$, whose range is reported each step.

All quantities are evaluated on the frozen geometry, the frozen $g_i^{(n)}$, and the converged $\tilde p_i^{(n)}$ (Section 8.3), with

$$
p_{*,{\rm phys},i}^{(n)}:=\tilde p_i^{(n)}+p_c\ge p_{\min}>0,
\qquad
p_c=C_c\frac{Z_0\nu_0 F_H}{d_p^2},
$$

and $F_H=A_Ha/(6h_0^2)$ if a contact model supplies it. The floor $p_{\min}$ (e.g. $10^{-6}$ Pa; f32 results rechecked in f64) removes the singularities of $\sqrt{p_*/\rho_p}$ and of the $\mu_{\rm eff}$ denominator. $p_{*,\rm ref}$ is floored the same way. The floor is a numerical protection only. Per Section 4.13, if it is used in the denominator of $\mu_d$, the same regularized pressure is used in $c+\mu_dp$.

The nodal shear rate $\mathcal F_i:=\dot\gamma_\Gamma$ at the surface point $i$ (vertex or face centre) comes from Section 10.1 on the **converged** Stokes solution of the step. The yield test and the pressure use the same velocity.

**Reference pressure and the singularity at $p_*=0$.** At a surface node $p_i$ may vanish (Section 4.22), so the slope proxy uses a reference depth:

$$
p_{\mathrm{ref},i}=\rho_{\mathrm{bulk}}|g_i^{(n)}\cdot n_i^{(n)}|\,h_s,\qquad
p_{*,{\rm ref},i}:=\max(p_{\mathrm{ref},i}+p_c,\,p_{\min})>0 .
$$

The anti-overhang row with $|g_i|>0$ only ensures that the reference pressure is compressive. The shallow friction coefficient uses the physical effective pressure through

$$
\mu_{d,i}
=\mu_s+(\mu_2-\mu_s)\frac{d_p\,\mathcal F_i}{I_0\sqrt{p_{*,{\rm phys},i}/\rho_p}+d_p\,\mathcal F_i},
$$

which is nondecreasing in $\mathcal F_i$ with range $[\mu_s,\mu_2)$, and tends to $\mu_2$ as $p_{*,\rm phys}\downarrow0$ with $\mathcal F_i>0$. At $p_{*,\rm phys}=\mathcal F_i=0$ take the static branch $\mu_d:=\mu_s$. Under A-Ξ, $\Xi(\mathcal R\nu(\mathbf X_i),I_{c,i})$ is added, with the same $I_{c,i}$, and the static branch is $\mu_y(\mathcal R\nu(\mathbf X_i))$. The range is then that of Proposition 11.6.

**Rheological-chain switch (one of two modes, which must be chosen explicitly).**

- *dynamic*: $\mu_{{\rm proj},i}:=\mu_{d,i}$. The chain $\mathbf u^{(n)}\to\dot\gamma_i\to I_c\to\mu_d\to\mu_{\rm eff}\to\phi_i$ is **fully active**, so the trace and the shear rate set the projection angle directly.
- *static*: $\mu_{{\rm proj},i}:=\mu_y=\mu_s$ ($\mu_y(\mathcal R\nu(\mathbf X_i))$ under A-Ξ). $\phi_i$ is then a **static relaxation proxy**, independent of the shear rate.

Neither option changes the yield classification. $F_\delta$, the core, the movable set, and the Mode C flux always use the unique stress ($\mu_d$ for viscosity, $\mu_y$ for strength). $\phi_i$ is a nominal proxy for the SOCP, not an equivalent of multiaxial yield. Identifying $\phi=\arctan\mu$ with a repose angle is a modelling hypothesis between Mohr–Coulomb and Drucker–Prager, valid only for a declared loose layer.

**Tidal scalarization and the projection angle.** With $\|A\|_{\rm eq}=\|A\|_F/\sqrt2$, the condition $\|s-T\|_{\rm eq}\le c+\mu_yp$ implies only the necessary, non-conservative $\|s\|_{\rm eq}\le c+\mu_yp+\|T\|_{\rm eq}$. For example, $c=0$, $\mu_yp=1$, $T=\operatorname{diag}(0.6,-0.6,0)$, and $s=\operatorname{diag}(-0.5,0.5,0)$ give $\|s\|_{\rm eq}=0.5\le1.6$, yet $\|s-T\|_{\rm eq}=1.1>1$. A **sufficient** condition is $\|s\|_{\rm eq}\le c+\mu_yp-\|T\|_{\rm eq}$, so

$$
\mu_{\mathrm{eff},i}^{(n)}
:=\frac{\Big[c+\mu_{\mathrm{proj},i}\,p_{\mathrm{ref},i}-\|T_{\mathrm{tide},i}\|_F/\sqrt{2}\Big]_+}{p_{*,\mathrm{ref},i}}\ge 0,
\qquad
\boxed{\phi_i^{(n)}=\arctan\mu_{\mathrm{eff},i}^{(n)}},
\qquad
\sin^2\phi_i^{(n)}=\frac{(\mu_{\mathrm{eff},i}^{(n)})^2}{1+(\mu_{\mathrm{eff},i}^{(n)})^2}.
$$

**Monotonicity, and why $\phi_i$ is not absolutely conservative.** With $T:=\|T_{\rm tide}\|_{\rm eq}$, $\mu:=\mu_{\rm proj}$, and $\mu_{\rm eff}(p)=[c+\mu p-T]_+/(p+p_c)$,

$$
\frac{d}{dp}\,\frac{c+\mu p-T}{p+p_c}=\frac{\mu p_c-(c-T)}{(p+p_c)^2}.
$$

- If $\mu p_c\ge c-T$, $\mu_{\rm eff}$ is nondecreasing, so $p_{\rm ref}\le p$ biases the angle tight.
- If $c-T>\mu p_c$ (large cohesion), $\mu_{\rm eff}$ decreases, so $p_{\rm ref}<p$ biases it **loose**, and "$p_{\rm ref}\le p$ is conservative" is false.

So no absolute sufficient condition is claimed. $\phi_i$ is monotone without cohesion ($\mu p_c\ge-T$ automatically) or at high confinement. True yield, $\|s-T_{\rm tide}\|_{\rm eq}\le c+\mu_yp$, is checked by the outer test.

**Spurious effect of a vertex Hessian.** At vertices and edges $T_{\rm tide}\propto\operatorname{dev}H_W$ would take $H^{\rm FP}_{\ell_0}$ with $C_{ij}\ne0$, an $\ell_0$-dependent artefact of the discretization (Section 1.3). The Hessian for the stress is therefore taken at face centres, or as the mean of the one-sided limits. Proposition 2.7.1 shows that there the counterterm vanishes and the Feynman integral gives the same scale-free finite part, with $H^\pm=H^{\rm FP}\pm2\pi G\rho_K\mathbf n\mathbf n^{\!\top}$ ("$+$" exterior), mean $H^{\rm FP}$, and jump $4\pi G\rho_K\mathbf n\mathbf n^{\!\top}$. Retaining only the local contact tensor $-\tfrac{2\pi}{3}G\rho_K\mathbf I\pm2\pi G\rho_K\mathbf n\mathbf n^{\!\top}$ would discard the large-scale tidal tensor. Vertex values must never enter $T_{\rm tide}$ directly.

---

## 10.3 The geometric self-consistent-field closure with the preceding equations

### 10.3.1 Time levels, exogenous spin, and the geological step

| Level | Variables | Meaning and method |
|---|---|---|
| Exogenous spin table (default) | $t\mapsto\omega(t)$ | configured; **not solved**; no feedback from the flow, the inertia, or the angular momentum |
| Angular-momentum law (A-L, Section 11.7) | $L_n$, $T_\parallel$ | $\omega_n=L_n/I_1^{(n)}$, $L_{n+1}=L_n+\tau_nT_\parallel$; replaces the table |
| Geological step $n$ | $t_n$, $\Delta t_n$, $\lambda$ | geometric advance, $t_{n+1}=t_n+\lambda\Delta t_n$ |
| Hyperbolic ray relaxation | $(\mathbf v,\mathbf S,p)$ | pseudo-time stationary solve at fixed geometry and $\omega_n$; geological time is frozen |

The three meanings of time must not be mixed.

**Exogenous spin table (exactly one form).**

- *Stages*: $\{(\omega_j,T_j)\}$ with $T_j>0$; $\omega\equiv\omega_j$ on $[s_{j-1},s_j)$, with $s_j=s_{j-1}+T_j$ and $T_{\rm end}=s_J$.
- *Linear*: $\omega(t)=\omega_0+\dot\omega_{\rm ext}t$ (e.g. a mean YORP acceleration), with $\omega\ge0$ required.
- *Table*: nodes $(t_k,\omega_k)$ with increasing $t_k$; piecewise-linear $\omega$; $T_{\rm end}=t_{K_\omega}$.
- *Orbit*: synchronous rotation on a circular orbit about a primary $GM_p$ on the body $+x$ axis, with the constant-$Q$ law $a^{13/2}=a_0^{13/2}+\tfrac{13}{2}a_0^{11/2}\dot a_0t$ and $\omega=n=\sqrt{GM_p/a^3}$. The primary adds the Hill quadrupole

$$
U_{\rm tide}=\tfrac12 n^2\big(2x_1^2-x_2^2-x_3^2\big),\qquad
\mathbf g_{\rm ext}=n^2(2x_1,-x_2,-x_3),\qquad
\nabla^2U_{\rm tide}=n^2\operatorname{diag}(2,-1,-1),
$$

which enters $\mathbf g_0$, $H_W$, and $\mathbf T_{\rm tide}$. It is traceless, so $\Delta W$ and Section 8.3 are unchanged. After realignment the body $x$ axis is the long principal axis, the synchronous orientation. Under A-O (Section 10.3.6) the rate of the law follows the satellite's mass, and $a$ becomes a state of the run.

Each step freezes $\omega_n=\omega(t_n)$ (under the orbit form, also $n_n=\omega_n$). The spin enters only through

$$
\mathbf g_{\rm eff}=\nabla U+\omega_n^2\mathbf x_\perp+\mathbf g_{\rm ext},\qquad
H_W=\nabla^2U+\omega_n^2\operatorname{diag}(1,1,0)+\nabla^2U_{\rm tide},
$$

and hence $\mathbf T_{\rm tide}$; $\mathbf g_{\rm ext}=\mathbf 0$ except under the orbit form. **Hypothesis A-S (spin about a principal axis):** the spin axis is the body $z$-axis, the current axis of maximum inertia, as maintained by the realignment of step 8. Nutation damping is instantaneous relative to $\Delta t_n$. $\omega$ is never updated from the inertia or the angular momentum; that is the definition of an exogenous prescription. Under A-L the table is replaced by the angular-momentum law of Section 11.7. A-S is retained there, with its adiabatic justification and indicators (Proposition 11.12).

**Hypothesis A-T1 (time-scale separation).** Two groups of processes act on disjoint time scales. The slow operator $\mathcal S_{\rm creep}(\Delta t)$ is the creep of the yielded region ((B3)–(B4) with $\chi_Y$ alone), the transport of $\rho_s$ along its slip, the core transport of Section 9.2, and the state evolution of Chapter 11 (markers, $\varpi_\Gamma$, thermal production, and $L$). The fast operator $\mathcal S_{\rm surf}$ is the Mode C erosion of the yielded surface (e-folding time $\tau_C=\sigma_0/(k_e\max_i[F_{\delta,i}]_+)$), its runout ($T_{\rm run}$), and the shallow failure of A-C4. A-T1 assumes $\tau_C,\ T_{\rm run}\ll\tau_n=\lambda\Delta t_n$. Each step is then the Lie product

$$
\mathcal S_{\rm surf}\circ\mathcal S_{\rm creep}(\Delta t_n):
$$

the creep step on the deep state of $t_n$, followed by the quasi-static settling of the surface (step S of Section 10.3.4) on that state, frozen ($\chi_Y$ and $F_\delta$ per vertex), with the face gravity of the current shape. Settling advances no time. The split is first order in $\Delta t_n$; its defect is that the deep state seen by settling lags the settled shape by one step, and material laid on yielding ground leaves with the next step. The ratio $\tau_C/\tau_n$ is reported each step; where it is not small, A-T1 fails, and the coupled scheme (the surface processes inside the step, with the bound $\Delta t_C$ below) must be used.

**Geological step.** Under A-T1,

$$
\Delta t_n
=\min\{\Delta t_{\rm CFL},\ \Delta t_{\rm char},\ \Delta t_{\rm core},\ \Delta t_\omega,\ \Delta t_{\max},\ t_{\rm next}-t_n\}.
$$

Under A-Ξ, A-ϖ and A-Θ the minimum also includes $\Delta t_\Xi$ (Section 11.4), $\Delta t_\varpi$ (Section 11.5) and $\Delta t_\Theta$ (Section 11.6).

- **CFL of the movable region:** $\Delta t_{\rm CFL}=\theta_{\rm CFL}h_{\Gamma,\min}/\max_{\{i:\chi_{Y,i}=1\}}|\mathbf v_{{\rm sl},i}\cdot\mathbf N_i|$, with $\theta_{\rm CFL}\le0.5$; promoted vertices have no creep and do not enter (Section 8.1, G10). The value is $+\infty$ if $\chi_Y\equiv0$ on the surface.
- **Erosional flux (coupled scheme only):** $\Delta t_C=\theta_C\rho_{\rm bulk}h_s/\max_i|J_{\rm dep}-J_{\rm ero}|_i$, with $\theta_C<1$ and fluxes at $t_n$, so the layer thickness changes by at most $\theta_Ch_s$. Positivity needs no bound, since erosion is integrated exactly. On a full layer this bound is $\theta_C\sigma_0/(k_e[F_\delta]_+)=\theta_C\tau_C$, and it relaxes as the layer depletes. Under A-T1 it is not a bound: its scale $\tau_C$ is what A-T1 requires to be small.
- **$\rho_s$ characteristics:** $\Delta t_{\rm char}=\theta_{\rm ch}/\|\nabla_\Gamma\mathbf v_T\|_\infty$, with $\theta_{\rm ch}<1$ (Section 8.1, G5, G10).
- **Core jet transport:** $\Delta t_{\rm core}=\theta_{\rm ch}/(\lambda\max_{k\ \text{not anchored}}\|\operatorname{dev}\nabla\mathbf v_{\rm adv}(\mathbf c_k)\|_F)$, evaluated at the current centres, since the RK2 midpoint $\mathbf c_k^{\rm mid}$ itself depends on $\tau_n=\lambda\Delta t_n$. Step 6b then checks the same bound at $\mathbf c_k^{\rm mid}$ and, if it is violated, shortens $\Delta t_n$ by the violation ratio and repeats the $O(K)$ prediction; the bound is a smooth function of $\Delta t_n$ through $\mathbf c_k^{\rm mid}$, so a few repetitions suffice. This keeps $\mathbf F_k$ near the identity and the Level 2 backtracking single-valued. The Frobenius norm bounds the spectral norm. The value is $+\infty$ if every core is anchored.
- **Spin table:** none under the stage form (handled by $t_{\rm next}$). Under the linear or table form, $\Delta t_\omega=\varepsilon_\omega\max\{\omega(t_n),\omega_{\rm ref}\}/\max\{|\dot\omega(t_n)|,\epsilon\}$. Under A-L, $\Delta t_\omega$ is that of Section 11.7.
- **Caps:** $\Delta t_{\max}$ is configured, and $t_{\rm next}$ is the next stage boundary, table node, or $T_{\rm end}$.

**Why there is no viscous stiffness catastrophe.** The velocity comes implicitly from the frozen-coefficient momentum balance. $\Delta t_n$ appears nowhere in it and is independent of $\eta_{\rm hi}=(c+\mu_2\tilde p)/\dot\gamma_\varepsilon$. Core creep is removed by $\chi_Y=0$ and does not enter $\Delta t_{\rm CFL}$. The step count is about $T\max_{\mathcal A}|\mathbf v_{\rm sl}\cdot\mathbf n|/(\theta_{\rm CFL}h_{\Gamma,\min})$, set by the true landslide speed. Promoted vertices have no speed, so while only they move the step is $\Delta t_\omega$, $\Delta t_{\max}$, or $t_{\rm next}$. An equilibrated stage needs one step, which jumps to the stage end, provided the slow state of Chapter 11 is also at rest (Proposition 11.14). $\lambda\in(0,1]$ defaults to $1$ and is reduced only when the outer geometric test struggles; time accumulates as $\lambda\Delta t_n$. A step recomputed after a trust-region shrink (overhang, self-intersection) does not advance time. Stage equilibrium is 2.15(c). Its stability, and finite-step attainment, belong to PO-12.

### 10.3.2 Quasi-static rheology: non-star-shaped hyperbolic ray relaxation

The frozen-coefficient rheology is advanced by the hyperbolic ray solver of Section 8.3. On the frozen geometry, density, spin, body force, tide, and constitutive coefficients, the unknown is

$$
\mathbf U=(\mathbf v,\mathbf S,p),
$$

with $\mathbf S$ the flow stress. The pseudo-time equations are

$$
\begin{aligned}
\rho_\tau\partial_\tau\mathbf v-\nabla\cdot\mathbf S+\nabla p&=\rho_K\mathbf g_{\rm eff}-\nabla\cdot\mathbf T_{\rm tide},\\
T_s\partial_\tau\mathbf S+\mathbf S&=2\eta^{(m)}\mathbf D(\mathbf v),\\
\kappa\partial_\tau p+\nabla\cdot\mathbf v&=g.
\end{aligned}
\tag{10.3.2-H}
$$

The tide is equivalently included in the total stress $\mathbf S+\mathbf T_{\rm tide}$. The source $g$ is zero in the reference incompressible model and is $\dot\varepsilon_v$ under A-$\Xi$. The viscosity $\eta^{(m)}$ is frozen during outer rheology pass $m$ and recomputed from $(\tilde p,\dot\gamma)$ after the pseudo-time solve.

**Characteristic packets.** At every volume sample and for every local characteristic family $\alpha$ of Section 8.3, form $A(\mathbf n)$ from the principal part of (10.3.2-H), compute $(\lambda_\alpha,r_\alpha,\ell_\alpha)$, and backtrace the Hamiltonian characteristic

$$
\frac{dx}{d\tau}=\nabla_k\omega_\alpha(x,k),
\qquad
\frac{dk}{d\tau}=-\nabla_x\omega_\alpha(x,k),
\qquad
\omega_\alpha=\lambda_\alpha(x,\widehat k)\lVert k\rVert.
$$

With $\rho_\tau=\rho_K$, the Cauchy density enters both the longitudinal (P-wave) speed and the transverse shear (S-wave) speed,

$$
c_{\rm hyp}(x)=\sqrt{\frac{2\eta(x)}{\rho_K(x)T_s}+\frac{1}{\rho_K(x)\kappa}},
\qquad
c_s(x)=\sqrt{\frac{\eta(x)}{\rho_K(x)T_s}},
$$

so the propagating families have $\lambda_{P,\pm}=\pm c_{\rm hyp}(x)$ (one pair) and $\lambda_{S,a,\pm}=\pm c_s(x)$ ($a=1,2$, two orthogonal shear pairs), while the four stationary modes with $\lambda_0=0$ relax in place at $x$. The path of each propagating mode is approximated by straight segments inside BVH cells; coefficients and eigenvectors are refreshed at every segment boundary. This is a non-star-shaped construction and does not use $r=R(\hat\omega)$.

**RT hardware path query.** A SurfaceBVH stores the boundary triangles and a CellBVH stores volume-cell AABBs. For a backtraced characteristic, repeated ray queries return the ordered segment list

$$
\{(\Delta\tau_j,\,\text{cell}_j,\,\text{boundary}_j)\}_{j=1}^{J}.
$$

The next query begins at the previous hit plus $\varepsilon_{\rm geom}$. The list therefore handles cavities, concavities, thin necks, and multiple crossings. RT hardware supplies hit distance, primitive ID, normal, and barycentric coordinates; all state interpolation and constitutive arithmetic are done in compute kernels.

**Characteristic update.** At the upstream point of a segment, interpolate $\mathbf U$, project

$$
 a_\alpha=\ell_\alpha^\top\mathbf U,
$$

integrate $\ell_\alpha^\top\mathbf R$ along the segment, and reconstruct with the right eigenvectors. The frozen-viscosity stress source is integrated exactly:

$$
\mathbf S^{n+1}=e^{-\Delta\tau/T_s}\mathbf S^{\rm up}
+\left(1-e^{-\Delta\tau/T_s}\right)2\eta^{(m)}\mathbf D^{\rm up}.
$$

At a boundary hit, the three outgoing characteristic amplitudes $(w_{P,+},w_{S,1,+},w_{S,2,+})$ come from the interior ray and the three incoming amplitudes $(w_{P,-},w_{S,1,-},w_{S,2,-})$ are solved from the normal and tangential components of

$$
(-p\mathbf I+\mathbf S+\mathbf T_{\rm tide})\mathbf n=\mathbf t_{\rm ext}.
$$

No radial ordering is used.

**Pressure and rigid-mode correction.** Every $N_p$ ray sweeps, apply the artificial-compressibility correction implied by

$$
\nabla\cdot\mathbf v-g=-\kappa\partial_\tau p,
$$

(or the optional global Nitsche pressure projection of Section 8.1, G3), and project out the six mass-weighted rigid modes with the Appendix A.3 system. The pseudo-time solve stops only when the momentum, constitutive, divergence, traction, and rigid-mode residuals all satisfy Section 2.15.

**Modal projection and sample operators (connection to Sections 2.15, 9.2, and 10.1).** Once the outer rheology pass converges to the sample state $\mathbf U=(\mathbf v^{\rm ray},\mathbf S^{\rm ray},p^{\rm ray})$, the modal coefficient vector $\mathbf u=(u_\alpha)$ (written $c_\alpha\equiv u_\alpha$ in Section 10.1) in the geometry-adapted basis $\mathbb V_h^\sigma=\operatorname{span}\{\mathbf B_\alpha\}$ is obtained by the constrained $L^2(\rho_K)$ Galerkin projection of Section 8.3:

$$
\boxed{
\mathbf M_\Gamma\mathbf u+\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda_{\rm proj}
=\int_{\mathcal V}\rho_K\,\mathbf B^{\!\top}\!\big(\mathbf v^{\rm ray}-\mathbf v_\varepsilon\big)\,dV,
\qquad
\mathbf B_{\rm rb}\mathbf u=-\mathbf b_\varepsilon,
\qquad
\mathbf v_h(\mathbf x)=\mathbf v_\varepsilon(\mathbf x)+\sum_\alpha u_\alpha\mathbf B_\alpha(\mathbf x),
}
$$

where $\mathbf v_\varepsilon=\mathbf 0$ and $\mathbf b_\varepsilon=\mathbf 0$ unless A-$\Xi$ is on (Section 11.4). The Cholesky factor of $\mathbf M_\Gamma$ and the $6\times 6$ Schur complement of $\mathbf B_{\rm rb}$ are built once per geometry. All downstream operations that require continuous spatial derivatives—the surface velocity trace and surface-invariant shear rate of Section 10.1, the 2-jet core advection $\nabla\mathbf v_{\rm adv}$ of Section 9.2, and the marker transport of Section 11.1—evaluate $\mathbf v_h$ and $\nabla\mathbf v_h$ analytically from $\mathbf u$. At the volume quadrature nodes $\mathbf x_i$ (weights $w_i$), define the normalized strain-rate evaluation operator and dual stress samples

$$
\mathbf e_i(\mathbf u):=\sqrt{2}\,\mathbf D'(\mathbf v_h)(\mathbf x_i),
\qquad
\boldsymbol\varsigma_i:=\frac{1}{\sqrt{2}}\,\mathbf S^{\rm ray}(\mathbf x_i),
$$

so that $|\mathbf e_i(\mathbf u)|=\dot\gamma_i(\mathbf v_h)$, $|\boldsymbol\varsigma_i|=\|\mathbf S^{\rm ray}(\mathbf x_i)\|_{\rm eq}$, and $\sum_i w_i\mathbf e_i^{\!\top}\boldsymbol\varsigma_i$ is the discrete internal virtual work used in the weak mechanical balance test of Section 2.15(b), the limit-load dual program of Proposition 2.17.1, the fused variant of Section 10.3.5, and Conjecture A.2.

**Algorithm.**

```text
Input: current non-star-shaped mesh, rho_K, omega_n, gravity, tide, v, S, p.
Build or refit SurfaceBVH and CellBVH.
For outer rheology pass m:
  evaluate p~, D(v), gamma_dot, mu_d, eta^(m), rho_tau=rho_K;
  compute local characteristic speeds (c_hyp, c_s, 0) and eigenvectors;
  repeat ray sweeps until the hyperbolic residuals pass:
    for every cell sample and characteristic family:
      backtrace the characteristic over Delta_tau;
      query all BVH segments and boundary hits;
      interpolate the upstream state;
      integrate source terms and exact stress relaxation;
      reconstruct (v,S,p);
    impose incoming P and S traction modes at boundary hits;
    apply pressure correction and rigid-mode projection;
    evaluate momentum, constitutive, divergence, traction, and energy residuals;
  update eta and repeat the outer pass.
Project converged v onto the geometry-adapted basis:
  solve M_Gamma u + B_rb^T lambda = int rho_K B^T (v - v_eps) dV with B_rb u = -b_eps;
  set v_h = v_eps + sum_alpha u_alpha B_alpha and form (e_i(u), varsigma_i).
Output: converged (u, v_h, S, p), projection defect e_proj, and the yield/connectivity data used by B2--B4.
```

The local CFL restriction is

$$
\Delta\tau\le C_{\rm CFL}\frac{h_{\min}}{\max_{x,\alpha}|\lambda_\alpha(x)|}
=C_{\rm CFL}\frac{h_{\min}}{\max_{x}c_{\rm hyp}(x)},
$$

since $c_{\rm hyp}(x)>c_s(x)>0$ everywhere. For frozen positive coefficients the energy identity of Section 8.3 is the stability check. As $h\to0$, $\Delta\tau\to0$, and the residual tolerances tend to zero, every stationary limit satisfies the original frozen-viscosity Stokes equations. The outer $\mu(I)$ and pressure/state update remains a fixed-point problem; non-monotone rate dependence does not receive an unconditional existence or uniqueness claim.

The geometric correction following this solve is still the SOCP of Section 10.3.3. The hyperbolic ray solver replaces the former rheology solver only; it does not remove the mass, clearance, trust-region, or slope projection program.

### 10.3.3 The geometric increment subproblem

The correction is a **sequential convex increment subproblem after the mechanical prediction**, with the current geometry as reference (Section 2.13). The increments are normal (Section 8.1, G4). It uses the linearized face normals $\widehat{\mathbf n}_f(d)=\mathbf n_f^{(n)}+D\mathbf n_f^{(n)}d$ in those increments and $\widehat{\mathcal M}(d)=\mathcal M(\mathbf X^{(n)},\rho_{\rm cand})+D\mathcal M^{(n)}d$ with $(D\mathcal M)_i=\rho_K(\mathbf X_i)|\mathbf A_i|$, together with $d_{\rm pred}$ and the classes $\mathcal L,\mathcal D,\mathcal A$ of (B3)–(B4). $d_{\rm pred}$ uses the converged slip velocity (trace of Section 10.1) and the weight $\rho_{\rm bulk}/\rho_K$, so that flux increments are $\Delta h_C^*$. $\mathcal A_{\rm lin}$ contains the clearance rows of (G4). The interior-point method of Section 2.13 (item 5) solves

$$
\boxed{
\begin{aligned}
(d_*^{(n)},t_*,s_{M*})
=\arg\min_{d,\,t\ge 0,\,s_M}\;&
\frac12\sum_{i\in\mathcal A^{(n)}} w_i\left(d_i-d_{{\mathrm{pred}},i}^{(n)}\right)^2
+\varrho\sum_{f}t_f+\varrho_M\frac{|s_M|}{M}\\
\text{s.t. }&d_i=0\quad(i\in\mathcal L^{(n)}),\qquad
d_i=d_{\mathrm{pred},i}^{(n)}\quad(i\in\mathcal D^{(n)}),\\
&\frac{\|\mathbf g_f^{(n)}\times\widehat{\mathbf n}_f(d)\|}{|\mathbf g_f^{(n)}|}
\le\sin\phi_f^{(n)}+t_f\quad(f\in\mathcal S_{\mathrm{slope}}^{(n)}),\\
&\mathbf g_f^{(n)}\cdot\widehat{\mathbf n}_f(d)\le-\epsilon_n|\mathbf g_f^{(n)}|\quad(f\text{ with a vertex in }\mathcal A^{(n)}),\\
&D\mathcal M^{(n)}d+s_M=M-\mathcal M(\mathbf X^{(n)},\rho_{\rm cand}),\\
&d\in\mathcal A_{\mathrm{lin}}^{(n)},
\qquad \|d_{\mathcal A}\|\le\Delta^{(n)},\\
&d_i\ge\min(d_{\mathrm{pred},i}^{(n)},0)-\rho_{s,i}A_i/(D\mathcal M)_i\quad(\pi_i=1,\ (D\mathcal M)_i>0),
\end{aligned}}
$$

with the discrete form (per-face rows, box trust region, $\varrho_M|\sigma_M|$ with folding, KKT tolerance) given in Section 2.13, items 1–6.

**Further structure.**

- **Two solves under A-C3.** The mass removed from $\mathcal V_{\rm slope}$ by the first solution is routed by the runout and returned as $\delta$. The second solve uses $d_{\rm pred}+\delta$, with depositing locked vertices moved to $\mathcal D$; its solution is $d_*$. Both are the same convex program with different data.
- **The slope set.** $\mathcal S_{\rm slope}$ is the set of faces of Section 2.8c: loose faces with at least one vertex in $\mathcal A$, supercritical or not, with rows at face centres. Its fixed vertices enter $\widehat{\mathbf n}_f$ with their fixed increments. The trust region constrains only $d_{\mathcal A}$, with the loose-mass bound of Section 2.8c on promoted vertices. The objective has no $R_{\rm eq}$, $W_0$, or equipotential term.
- **Empty movable set.** If $\mathcal A=\varnothing$, then $d_*=d_{\rm pred}$ (zero on $\mathcal L$, the deposit on $\mathcal D$), and the cone program is not solved.
- **Feasibility and uniqueness.** The point $d_{\mathcal L}=0$, $d_{\mathcal D}=d_{\rm pred}$, $d_{\mathcal A}=0$ with $s_M$ and $t_f=[\cdot]_+$ meets the clearance rows and the loose-mass bound, and is feasible if the anti-overhang rows hold. Otherwise the violating faces lose their rows (item 2 of Section 2.13); they are counted, not repaired. This is not a Slater point (Section 2.13). The objective is strictly convex in $d_{\mathcal A}$ plus an $\ell_1$ term, so $d_{\mathcal A*}$ exists and is unique. True satisfaction is decided by $t_*,s_{M*}=0$ and by the outer test.
- **What is not convex.** $\mathcal A_{\rm lin}$ holds only affine or conic interior approximations (nodal order, positive volume, trust region). True positive volume and topology are checked by the outer test. The trust-region policy is that of Section 2.13. With the norm constraint on the normal kept, the program is an SOCP; it is a pure QP only after full affine approximation.

**Place of the tidal backstress and of yield.** $\sin\phi_f$ is the nominal proxy of Section 10.2 and constrains only the movable loose layer. $t_*=0$ only says that the proxy holds, and $t_*\ne0$ is not an assertion of tensorial yield. True yield is not an acceptance constraint; it drives the flow, and is recomputed next step on the new geometry, $\mathbf T_{\rm tide}$, and Stokes solution. The outer test checks only:

- positive volume;
- no self-intersection;
- the mass, corrected once and then recorded (Section 2.14).

Embeddedness inside the program is kept by the clearance rows. **Trust-region floor and outcomes.**

- No acceptable candidate: shrink the trust region, never fall back to a fluid target.
- The radius stops at $\Delta_{\min}=\Delta_{\min,0}\bar R$ (a configured fraction of the mean radius).
- A geometric rejection that persists at the floor ends the run with the status *geometry invalid*.
- A cone program that misses its KKT tolerance at every radius down to the floor keeps $d_*=\mathbf 0$. The geometry and its supercritical faces are carried unchanged, time advances by $\lambda\Delta t_n$, and the step is flagged *cone unresolved* and counted. $\rho_s$ transport and the core closure still run. Such a step leaves the slope unrelieved, as protected relief does, and does not move the body.

### 10.3.4 The main loop of the quasi-static evolution

The loop below is the reference closure. When a hypothesis of Chapter 11 is on, its steps carry the normative additions of Section 11.9, marked there by the same step numbers.

```text
0. Initialization: read the polyhedron (vertices X^(0), fixed connectivity; hypothesis (G) of
   Section 8.1 is checked), the density (rho_b and the Cauchy cores w_k, c_k, S_k, alpha_k; the
   core shape class and the tolerances eps_xi, eps_rho, gn_max of Section 9.2; eps_split, eps_merge,
   K_max if Level 3 is used), the material constants
   (nu_0, Z_0, c_b, c_s, h_c, mu_s, mu_2, I_0, d_p, rho_p, C_c, F_H, C_T, gd_eps, delta_Y, p_min),
   the Mode C parameters (k_e, sigma_0, kernel K, h_s, runout switch, T_run, eps_stop),
   the slope-promotion tolerance tau_s, the hyperbolic ray parameters of Section 10.3.2 (rho_tau=rho_K, T_s, kappa, C_CFL, Delta_tau, N_p,
   eps_hyp, k_max, continuation depth c), the spin table, and every tolerance.
   If any item is missing -> model not closed; refuse to run.
   rho_s^0 = rho_bulk h_s, eps_p_bar = 0, t_0 = 0.
   Perform the frame alignment of step 8b once (mandatory; the trigger condition is ignored).
For n = 0, 1, 2, ... until t_n >= T_end:
1. Forward evaluation: omega_n = omega(t_n). On V^(n) and rho_K(.; theta^(n)), compute by RTFP
   the face-centre Hessian (Section 10.2), the nodal g_i and H_W,i, and grad U, Hess U at
   X_vol (the assembly points of Section 8.1 (G2)) and on X_Y (Section 2.8, B3).
   Reuse the segment table and retrace only dirty rays (Section 3.15.2); a spin-only step
   reuses the fields.
   g_eff = grad U + omega_n^2 x_perp + g_ext,
   H_W = Hess U + omega_n^2 diag(1,1,0) + Hess U_tide   (g_ext, U_tide only under the orbit form),
   T_tide = C_T rho_p nu_0 d_p^2 dev H_W.
   Under A-O (Section 10.3.6): omega_n = n(a_n) from the orbit state, and the necks L_k, e_k of
   the Hill region by Newton's iteration; they are frozen for the step.
2. Hyperbolic characteristic ray relaxation (Section 10.3.2) -> (v^(n), S^(n), p^(n)).
   If criterion (a) or (b) of Section 2.15 fails, retry by continuation; if that fails, terminate and report.
3. Yield classification (Section 2.8, B2-B3): on X_Y synthesize v, D, p~; compute tau_flow, tau_y, F_delta;
   the unyielded component of largest mass on G_Y is U_core -> chi_Y,i;
   least-squares rigid fit (a_c, theta_c); v_sl = v - a_c - theta_c cross x.
   Update Phi_p and eps_p_bar (B6). Report the movable area fraction, the tensile fraction,
   Ek^{-1}, and eps_Euler.
   Under A-O, every N_L steps: the certified bracket of L* with tau^(2) (Proposition 2.17.3);
   L_lo > 1 ends the run as a limit-load collapse.
   Under A-Lambda (Section 10.3.7): the libration Gamma of the current inertia; the two elastic
   solves of the cycle at uniform viscosity mu_e; the steady cycle (Algorithm 10.3.7.6) on the
   creep stress -> ratchet velocity v_r, alternating range, fatigue rate r_F. Every N_L steps the
   bound of L_peak and the bracket of L_sd replace that of L*; a certified L_peak > 1 ends the run.
4. Friction chain (Section 10.2): surface velocity and strain rate at vertices and face centres
   (Section 10.1), gd (Section 10.1.1), then mu_d, mu_proj, phi, and the repose slack s_f of every face.
4+. Lag of the deep state. Steps 1-4 may instead reuse the deep state of the last step n0 that
   evaluated them (u, p, chi_Y, v_sl, the face chain, and g_f), when every vertex moved by at most
   theta_lag h_Gamma,min since n0 (theta_lag = 0.1), every core centre by at most the same length
   with weight and metric within 1e-3 relative, |omega_n - omega_n0| <= eps_omega |omega_n0| and the
   same for the tidal strength, there was no realignment (8b) since n0, and no hypothesis of
   Chapter 11 is on. The error is O(theta_lag h_Gamma,min |grad v|) in the creep velocity, of the
   order of the explicit geometric update itself under the CFL bound; with chi_Y = 0 on the
   surface the creep velocity moves nothing and the reuse is exact. The step is reported as lagged.
5. (Mode C, runout and slope promotion are the fast operator of A-T1: step S.)
6. Step size of the slow operator (Section 10.3.1, under A-T1), with Delta t_core from the gradients
   at the current centres c_k; A^(n) = {chi_Y = 1} on the surface.
   If the spin table has stages and chi_Y = 0 on all of X_Y: run step S on V^(n); if it leaves the
     surface unchanged and the slow state of Chapter 11 is at rest (Proposition 11.14), stage
     equilibrium (2.15(c)) -> emit the stage's terminal state, set t_{n+1} to the stage end,
     X^(n+1) = X^(n), go to step 1.
   Under A-O, Delta t_omega uses n-dot(a_n, M_n), and the time in which the orbit reaches the
   primary at the mass M_n bounds the step as well.
   If A^(n) is empty: Delta t_n = min(Delta t_omega, Delta t_core, Delta t_max, t_next - t_n),
     d_pred = 0; run 6b, then go to step 8 (no creep of the surface).
6b. Core prediction (Section 9.2.3, Level 0; skipped in the frozen-density variant A-rho):
   v_adv = chi_Y v_sl, tau_n = lambda Delta t_n; for each non-anchored core, RK2 for c_k,
   F_k = exp(tau_n dev grad v_adv(c_k^mid)), S_k' = F_k^{-T} S_k F_k^{-1}, w_k' = w_k.
   If tau_n ||dev grad v_adv(c_k^mid)||_F > theta_ch for some core, reduce Delta t_n by that ratio
   and repeat 6b (cost O(K)). -> theta^pred, rho_cand (recomputed whenever lambda or Delta t_n changes).
7. Creep increments: d_pred,i = Delta t_n v_sl,i . N_i on A^(n), the classes L / A (every other vertex
   locked); solve the SOCP of Section 10.3.3 with rho_cand, with the repose rows of the loose faces of
   A^(n), for d_*^(n). Under A-O the lobe cap bounds the movable vertices that would pass a neck
   plane, and the mass target falls by the mass M_cap carried beyond it.
8. Geometric update and frame realignment:
   8a. X_i <- X_i + lambda d_*,i N_i (Section 8.1, G4), with d_* = 0 on the locked set.
       Rebuild the polyhedron. Outer acceptance: positive volume, no self-intersection, a closed,
       consistently oriented mesh. On failure, or when the cone solve of step 7 failed, reduce
       Delta^(n), redo step 7, and do not advance time. At Delta_min: a geometric rejection ends
       the run (geometry invalid); a cone solve that still misses its KKT tolerance gives d_* = 0,
       the step is flagged cone unresolved, X^(n+1) = X^(n), and the step continues with the
       rho_s update and 8c, advancing time by lambda Delta t_n (Section 10.3.3). A miss of |M(V^(n+1), rho_cand) - M| <= eps_M M
       is corrected once on the movable set, then recorded (Section 2.14).
       rho_s: semi-Lagrangian transport along v_T on Gamma (Section 8.1, G5), step
       lambda Delta t_n, with the Jacobian correction and rho_s,i = m_i / A_i^new.
   8c. Core closure on V^(n+1) (Sections 9.2.4-9.2.6): Level 1 on the cores (rho_b fixed), the
       misfit e_rho, Level 2 Gauss-Newton refit if e_rho > eps_rho or |xi_n| > eps_xi, Level 3 if
       used, then Level 1 again; recheck rho_K >= rho_min at the volume nodes. 8c precedes 8b,
       which needs the updated density.
   8b. Frame realignment. Trigger: |x_cm| > eps_frame R_ref / 2, or max(|I_xz|,|I_yz|) > eps_frame I_1 / 2,
       or a violated ordering I_zz >= I_yy >= I_xx beyond eps_deg I_1; otherwise skip.
       At most three passes, each followed by a new gravity evaluation.
       (i)   x_cm = (1/M_V) int rho_K x dV with the 8c density, M_V = int rho_K dV.
       (ii)  X_i <- X_i - x_cm for every vertex; c_k <- c_k - x_cm.
       (iii) I = int rho_K (|x|^2 Id - x x^T) dV; eigenpairs I_1 >= I_2 >= I_3, a_1, a_2, a_3.
       (iv)  e_z' = a_1 (spin axis), e_x' = a_3, e_y' = e_z' x e_x', signs with a_1.e_z >= 0 and
             a_3.e_x >= 0 (Q closest to Id). Near degeneracy: if I_1 - I_2 <= eps_deg I_1, e_z' is
             the normalized projection of the old e_z onto span{a_1, a_2}; if I_2 - I_3 <= eps_deg I_1,
             e_x' is the normalized projection of the old e_x onto span{a_2, a_3} intersected with
             e_z'-perp; if all degenerate, Q = Id; a vanishing projection takes the lowest-index
             eigenvector. Rows of Q are (e_x', e_y', e_z'), det Q = +1.
             X_i <- Q X_i, c_k <- Q c_k, S_k <- Q S_k Q^T (cores must rotate with the mesh).
       (v)   No resampling (Section 8.1, G8): the moved vertices are the new surface, rho_s stays on
             its vertices, and the mass is unchanged to rounding, so E_frame = 0.
             Velocity is not carried: the next step initializes (v^0, S^0, p^0) = (0, 0, p_rest).
             omega is unchanged (exogenous; under A-L it is set in step 9+ from L and I_1);
             Omega = omega e_z'.
       (vi)  Recheck |x_cm'| <= eps_frame R_ref and max(|I_xz'|,|I_yz'|) <= eps_frame I_1; failure
             stops the run. After it passes, the compatibility conditions of Appendix A.3 hold.
S. Settling (fast operator, hypothesis A-T1), on V^(n+1) after 8b, with chi_Y and F_delta of step 3
   frozen per vertex and tau_n = lambda Delta t_n. Time does not advance. For sweeps k = 0, 1, ...:
   S1. Face gravity of the current shape (RTFP) and the friction chain at rest (v = 0, D = 0):
       g_eff,f, sin phi_f, repose slack s_f.
   S2. On sweep 0 only, Mode C (Section 2.8): chi_A = chi_Y 1[rho_s > 0]; trace each node with
       J_ero > 0 by the Eikonal/Randers ray or the full packet ODE (A-C3, Section 2.8b) to X_R; J_ero and J_dep = K_R * (moved
       source) as step averages over tau_n (the exact exponential); Delta h*_C = tau_n (J_dep - J_ero)
       / rho_K. Later sweeps: no erosion (material laid on yielding ground leaves with the next step).
       Under A-O the loose layer of a vertex with rho_s g_n > c_Gamma detaches (beta -> infinity);
       packets beyond a neck plane are lost, and the mass target of the sweep falls by their mass.
       Under A-Lambda the repose test takes the steepest phase of g_f + delta g, packets fly in the
       field of their launch phase, and the necks follow the phase.
   S3. Slope promotion (Section 2.8c): pi_i = 1 on vertices of faces with s_f > tau_s, chi_Y,i = 0,
       rho_s,i > 0. Stop if no vertex is promoted and nothing erodes (settled).
   S4. Classes: A = {pi = 1} with d_pred = Delta h*_C; D = other vertices with Delta h*_C != 0;
       L the rest. Solve the SOCP of Section 10.3.3 with the loose-mass bound of the promoted
       vertices; relief routing (A-C3): r_i = [d_pred,i - d_i]_+ on the movable vertices of the
       faces of S_slope, m_rel_i = (D M)_i r_i, run out to X_R(a_i), spread by K_R,
       delta_j = (K m_rel)_j A_j / (D M)_j; d_pred <- d_pred + delta; solve again for
       d_* (under A-O with the mass target less the routed mass lost through a neck). Outer acceptance as in 8a; when no increment is admissible down to Delta_min, stop and
       keep the last shape.
   S5. X_i <- X_i + d_*,i N_i; rho_s,i <- rho_s,i A_i / A_i^new + tau_n (J_dep - J_ero)_i + the relief
       ledger of Section 2.8c (debit (D M)_i [d'_pred,i - d_*,i]_+, credited along the routed
       runout) / A_i^new; clip at zero.
   S6. Stop when a sweep no longer removes a fixed fraction of the supercritical faces and does not
       lower the largest slack (the rest is unrelievable), or after the configured number of sweeps.
   Report the sweeps, the eroded and relieved masses, the slack before and after, tau_C / tau_n,
   and whether settling converged, stalled, or was limited by the geometry.
9. t_{n+1} = t_n + lambda Delta t_n. Under A-O: a_{n+1}^{13/2} = a_n^{13/2} - kappa M_n tau_n,
   M_{n+1} = M_n - M_esc,n; the run stops when a reaches the primary. Under A-Lambda:
   e_{n+1} = e_n (a_{n+1}/a_n)^(C_e + beta_n) (Proposition 10.3.7.1). Record X^(n+1), rho_s,
   theta^(n+1), the diagnostics. Return to step 1.
```

**Stage equilibrium and micro-creep.** 2.15(c) is a strict fixed point. With an empty movable set, $d_*=0$ exactly, independent of $\Delta t_n$ and $\lambda$. The regularized creep inside the core, $|\mathbf v|\sim\dot\gamma_YR$, is only a diagnostic, $\max_{\mathcal U_{\rm core}}|\mathbf v_{\rm sl}|$. It enters neither the geometry, nor the step, nor the flux, nor any state. There is no "velocity floor" test, and creep causes no geometric or historical drift. As $\dot\gamma_\varepsilon\to0$ the core creep vanishes, and the movable set tends to the yielded region and sliding blocks of the unregularized model (PO-12).

The quasi-static loop is

$$
\boxed{
\omega(t_n)
\xrightarrow{\text{RTFP forward evaluation}}
(\mathbf g_{\rm eff},\mathbf T_{\rm tide})
\xrightarrow{\text{hyperbolic characteristic ray relaxation}}
(\mathbf u,p;\ \boldsymbol\sigma)
\xrightarrow{F_\delta,\ \mathcal U_{\rm core},\ s_f}
(\chi_Y,\mathbf v_{\rm sl})
\xrightarrow{\Delta t_n}
\big(d_{\mathrm{pred}},\ \theta^{\rm pred}\big)
\xrightarrow{\mathrm{IPM}\ \text{elastic SOCP}\ (\rho_{\rm cand})}
d_*^{(n)}
\xrightarrow{\text{update}+\text{core closure}+\text{realignment}}
\big(\mathbf X^{(n+1)},\theta^{(n+1)}\big)
\xrightarrow{\text{settling (A-T1)}}
\mathbf X^{(n+1)}.
}
$$

It has one stress and an implicitly resolved velocity at every step, with an exogenous spin (or, under A-L, a spin from the angular momentum). The geometry changes only on the movable set (yielded, sliding, or promoted vertices), and the cores move only with yielded or sliding material. The Mode A interface $W_0/R_{\rm eq}$ is not in the loop; it is kept only as a defined interface for a fluid model.

### 10.3.5 Optional variants: the fused conic program and gravity feedback

Both variants reuse the operators of Sections 10.3.2–10.3.3. Neither is part of the reference loop of Section 10.3.4.

**Fused program (minimizing movement).** The split route (Stokes, then $d_{\rm pred}$, then the cone) can be fused into one convex program per step:

$$
\min_{\mathbf u\in\mathbb K,\,d}\ \Delta t_n\,J(\mathbf u)+\sum_i\frac{\kappa_i}2\big(d_i-\Delta t_n\,\mathbf v_i(\mathbf u)\cdot\mathbf N_i-\Delta h^*_{C,i}\big)^2+\varrho\sum_ft_f+\varrho_M|\sigma_M|
$$

subject to the slope cones, anti-overhang, mass, clearance, and trust-box rows of Section 2.13 in $d$. It is convex: $J$ is convex at frozen pressure, the coupling is a convex quadratic, and the rows are conic or linear. The tie stiffness $\kappa_i$ (force per length, so every term is an energy) controls the closure:

- $\kappa_i\to\infty$ gives $d_i=\Delta t_n\mathbf v_i\cdot\mathbf N_i+\Delta h^*_{C,i}$;
- $\kappa_i=0$ on promoted vertices leaves the relief free.

The slope multipliers act on $\mathbf u$ as nodal normal tractions, giving a slope-constrained creep in which relief and flow see each other within one step. This is a different closure from (B3)–(B4): $\chi_Y$ locking is replaced by exact Bingham rigidity, and the core motion is fixed by the gauge $\mathbb K$ rather than subtracted as $\mathbf v_{\rm sl}$. Its relation to (B3)–(B4) is open (PO-20). The fused program has the form of the cone program of Section 2.13 plus the nodal ball prox of Section 10.3.2:

- primal: $(\mathbf u,d)$;
- duals: $\boldsymbol\varsigma_i$, the face cone duals, the mass dual, and the box multipliers;
- operators: $\mathbf e$ (a ray pass), the trace $\mathbf u\mapsto(\mathbf v(\mathbf X_i)\cdot\mathbf N_i)$, the sparse face-normal Jacobian, and $D\mathcal M$.

**Gravity feedback.** Section 2.13 freezes $\mathbf g$ and ignores $D\mathbf g\cdot d$. The first variation $\delta\mathbf g(\mathbf x)=G\oint_\Gamma\rho_K\delta_n(\mathbf y)(\mathbf y-\mathbf x)|\mathbf y-\mathbf x|^{-3}dS$ is linear in $d$. For $\rho_b$ it is the vertex derivative of the closed-form field of Section 3.9, available on the face tree of Section 3.10. Linearizing $\mathbf g\times\widehat{\mathbf n}$ and $|\mathbf g|$ in both factors keeps the slope rows conic, which turns the frozen-$\mathbf g$ step into a Gauss–Newton step at one face-tree evaluation per linearization, in the split route and in the fused program alike. It matters when a slide moves enough mass to change its own gravity.

---

### 10.3.6 Orbital decay with mass loss (hypothesis A-O)

Under the orbital form of Section 10.3.1 the semi-major axis follows a closed law with a fixed rate. The tidal decay rate is proportional to the satellite's mass, and a satellite that fills its Hill region loses mass. A-O couples the two [29].

**Hypothesis A-O.**

1. *Constant-$Q$ tide in the primary.* $\dot a=-3\,(k_2/Q)_p\,(m/M_p)\,(R_p/a)^5\,n\,a$, with $(k_2/Q)_p$ and $R_p$ constant and the satellite inside the primary's synchronous orbit. In terms of the configured initial rate, $\dot a=\dot a_0\,(m/m_0)\,(a_0/a)^{11/2}$.
2. *Circular synchronous orbit in the Hill approximation* of Section 10.3.1: $\omega=n$, the second-degree tide only, and the primary's oblateness neglected. A-Λ (Section 10.3.7) adds the eccentricity and the libration at first order.
3. *Loss through the necks only.* Mass leaves by packets in flight that pass a neck plane of the Hill region, and by movable material carried beyond one (the lobe cap below). Lost mass does not return within the run.
4. *Prompt departure.* Material that leaves does so on the orbital time scale $1/n$, short compared with $\tau_n$ (as in A-T1). The necks are evaluated at the start of the step and frozen in it.
5. *Recoil neglected.* The momentum carried away is not fed back into the orbit; Remark 10.3.6.4 bounds its effect.

A-O keeps the spin synchronous and therefore excludes A-L.

**State and law.** The orbital state is $(a_n,M_n,M_{{\rm esc},n})$, with the ledger $M_n+M_{{\rm esc},n}=M_0$ exact. By item 1,

$$
\frac{d}{dt}a^{13/2}=-\kappa\,m(t),\qquad \kappa:=-\tfrac{13}{2}\,\frac{a_0^{11/2}\dot a_0}{m_0}>0 .
$$

The mass is constant within a step, because it is lost in the settling of step S at the step's end (A-T1). The update is therefore exact:

$$
\boxed{\ a_{n+1}^{13/2}=a_n^{13/2}-\kappa\,M_n\,\tau_n,\qquad n_n=\sqrt{GM_p/a_n^3},\qquad \omega_n=n_n,\qquad M_{n+1}=M_n-M_{{\rm esc},n}.\ }
$$

**Step bounds.**

- The table bound of Section 10.3.1 becomes $\Delta t_\omega=\varepsilon_\omega\max\{n_n,\omega_{\rm ref}\}/|\dot n_n|$, with $\dot n=-\tfrac32(n/a)\dot a$ and $\dot a_n=-\tfrac{2}{13}\kappa M_na_n^{-11/2}$.
- The time $(a_n^{13/2}-R_p^{13/2})/(\kappa M_n)$ in which the orbit reaches the primary at the mass $M_n$ also bounds the step. The run stops when $a$ reaches $R_p$.
- A loss of the whole mass, or a step past the primary, is outside the model.

**Proposition 10.3.6.1 (Reduction).** Without mass loss the update reproduces, at every step boundary, the closed law $a^{13/2}(t)=a_0^{13/2}+\tfrac{13}{2}a_0^{11/2}\dot a_0t$ of Section 10.3.1. With mass loss, $a(t)$ is that law evaluated at the mass-weighted time $\int_0^t m/m_0\,dt'$, exactly for a piecewise-constant $m$. A lighter satellite therefore decays more slowly.

*Proof.* The update telescopes: $a_n^{13/2}=a_0^{13/2}-\kappa\sum_{j<n}M_j\tau_j$. With $M_j=m_0$ this equals $a_0^{13/2}+\tfrac{13}{2}a_0^{11/2}\dot a_0t_n$. In general the sum is $\kappa m_0\int_0^{t_n}m/m_0\,dt'$. ∎

**The Hill potential.** In the frame of Section 10.3.1 with $\omega=n$, the centrifugal and tidal terms combine to

$$
W=U+\tfrac12n^2\big(3x_1^2-x_3^2\big),
$$

with the primary on the $+x_1$ axis.

**Definition 10.3.6.2 (Necks and the Jacobi energy).**

- *Saddles.* $L_k$ ($k=1$ toward the primary, $k=2$ away from it) are the critical points of $W$ on either side of the centroid $\mathbf x_c$, with $r_H=(GM_n/3n^2)^{1/3}$ as their point-mass distance. Along the axis $\mathbf x_c\pm t\mathbf e_1$, $\pm\partial_1W$ is negative inside a neck (self-gravity dominates) and positive beyond it (the tide dominates), whether the neck lies outside the body or, once it overflows, inside it. Beyond the neck the tide dominates all the way out, while density structure can add sign changes inside a body near its Roche limit. A scan of $t\in(0,2r_H]$ therefore brackets the outermost sign change, and bisection narrows it. If there is none, $\pm\partial_1W>0$ on the whole half-axis: the tide dominates the self-gravity even at the centroid, which happens once $3n^2$ exceeds $4\pi G\rho_K/3$ there (an underdense centre reaches this first). The neck is then at the centroid, and that half of the body stands by its strength alone. Newton's iteration $\mathbf x\leftarrow\mathbf x-H_W^{-1}\nabla W$, with $H_W=\nabla^2U+n^2\operatorname{diag}(3,0,-1)$ and the RTFP field at the iterate (the finite-part value inside $V$), then moves to the saddle itself, to a step of $10^{-6}r_H$. An asymmetric body puts the saddle off the axis. If the iteration does not reach a saddle within $r_H/4$ of the axial point, the axial point is used. A point-mass saddle has the eigenvalues $(9,-3,-4)n^2$ of $H_W$: $W$ is a minimum along one direction and a maximum in the other two. A critical point without exactly one positive eigenvalue is not a neck, and the step fails.
- *Neck plane.* $\Pi_k$ is the plane through $L_k$ normal to $\mathbf e_k$, the eigenvector of $H_W(L_k)$ with the positive eigenvalue, oriented away from the body. Its outer half-space is $\Pi_k^+$.
- *Jacobi energy.* A packet in flight has $E_J:=\tfrac12|\mathbf v|^2-W(\mathbf x)$. It is conserved in flight (the Coriolis force does no work, and $W$ is steady within the step) and decreases while sliding and at an inelastic landing.
- *Critical energy.* $E_{\rm crit}:=\min_k\{-W(L_k)\}$.

**Proposition 10.3.6.2 (Hill stability).** A packet that starts on the inner side of every neck plane with $E_J<E_{\rm crit}$ stays in the component of $\{W\ge-E_J\}$ that contains the body. It never reaches a neck plane.

*Proof.* $E_J$ never increases, so the packet stays in $\{W\ge-E_J\}$. Along $\mathbf e_k$, $W$ is a minimum at $L_k$; across it, $W$ is a maximum. For $E_J<E_{\rm crit}$ the set $\{W\ge-E_J\}$ therefore misses a neighbourhood of each $L_k$, and the component containing the body is separated from the outer region at both necks. The critical lobe touches $\Pi_k$ only at $L_k$. ∎

**Escape and the mass sink.** Under A-O a packet in flight is **escaped** when it lies in some $\Pi_k^+$ outside $\overline V$, or beyond the far radius $r_{\rm esc}$ of Section 2.8b.

- With $E_J\ge E_{\rm crit}$ this is a transit through the neck. Conley's analysis of the linearized neck region shows that such orbits leave the satellite's region [28]. A possible later recapture is excluded by item 3.
- With $E_J<E_{\rm crit}$ the packet was launched beyond the neck, in the outer region, and cannot return (Proposition 10.3.6.2, read from the outer side).

The mass of an escaped packet leaves the body. It is either eroded mass, $J_{\rm ero}A\tau_n$ (Section 2.8), or routed relief mass, $m_i$ (Section 2.8b).

- *Runout.* The runout kernel becomes sub-stochastic, and the exact balance of Section 2.8 becomes $\int_\Gamma(\bar J_{\rm dep}-\bar J_{\rm ero})\,dS=-M^C_{{\rm esc},n}/\tau_n$.
- *Relief.* The routed program of Section 10.3.3 holds the mass target less the routed mass lost.

**Detachment of the loose layer.** On a vertex where the outward pull exceeds the shallow cohesion,

$$
\rho_s\,g_n>c_\Gamma,\qquad g_n:=\frac{\sum_{f\ni i}A_f\,\mathbf g_{{\rm eff},f}\cdot\mathbf n_f}{\sum_{f\ni i}A_f},
$$

with $c_\Gamma$ the area-weighted mean of the face values of Section 11.6 ($c_s$ without A-ϖ and A-Θ), the layer detaches within the step: the exact erosion update of Section 2.8 is applied with $\beta_i\to\infty$. The layer is launched at rest in the frame and runs out like an eroded packet. A loose layer can stand only where the effective gravity presses it on, so the rule can act only near or beyond a neck, where the Hill terms turn $g_n$ outward.

**Lobe cap.** Movable material ($\tilde\chi=1$) carried into some $\Pi_k^+$ is in the outer region at rest and leaves. The geometric program of step 7 bounds the increment there:

- $d_i\le d^{\rm cap}_i$, where $\mathbf X_i+d^{\rm cap}_i\mathbf N_i\in\Pi_k$ for the first plane ahead along $\mathbf N_i$ (the bound is not set below the clearance row);
- the lost mass is $M^{\rm cap}_n=\sum_i(D\mathcal M)_i\,(d_{{\rm pred},i}-d^{\rm cap}_i)_+$.

Material of the rigid core beyond a neck is held by its strength. Whether it can be held is decided by $L^*$ (Proposition 2.17.3).

**Mass row.** $M_{{\rm esc},n}=M^C_{{\rm esc},n}+M^{\rm relief}_{{\rm esc},n}+M^{\rm cap}_n$, and the mass rows of steps 7 and S take the targets that fall by these masses as they occur.

**Remark 10.3.6.4 (Recoil).** Mass leaves the neck at relative speeds $v_{\rm rel}=O(n\,r_H)$. A loss $\Delta M$ changes the orbital speed by at most $\Delta M\,v_{\rm rel}/M$, so

$$
\frac{|\Delta a|}{a}\le2\,\frac{M_{\rm esc}}{M}\,\frac{v_{\rm rel}}{na}=O\!\Big(\frac{M_{\rm esc}}{M}\,\frac{r_H}{a}\Big).
$$

For a satellite with $r_H/a\sim10^{-3}$ this is negligible against the tidal decay. The bound $2(M_{\rm esc}/M)(r_H/a)$ is reported each step.

**Reports.** Each step reports:

- $a_n$, $n_n$ and $M_n$;
- the mass lost by runout, by relief and by the cap, and the total;
- $r_H$, and the vertices whose layer detached;
- the lobe margin $\mu_L:=\max_{\mathbf x\in\Gamma}\max_k(\mathbf x-L_k)\cdot\mathbf e_k/r_H$, which is positive when the body overflows;
- the recoil bound;
- every $N_L$ steps, the bracket of $L^*$ (Proposition 2.17.3).

**Reduction.** With A-O off, the model is the orbital form of Section 10.3.1 exactly. With A-O on and no face beyond a neck plane, no packet in a $\Pi_k^+$ and no detaching vertex, the only change is the exact law, which then coincides with the closed form (Proposition 10.3.6.1).

### 10.3.7 Eccentric orbit, forced libration, and the tidal cycle (hypothesis A-Λ)

Under the orbital form of Section 10.3.1 and A-O the load is constant in the body frame: the orbit is circular and the lock is exact. A satellite on an eccentric orbit sees the tide change in strength and in direction once per orbit, and its spin librates about the synchronous rate. A rigid–viscoplastic body has no stress amplitude below yield, so it cannot feel that cycle: there is no fatigue and no ratchet, at any resolution. A-Λ adds the cycle at first order in the eccentricity $e$, with the elastic response that carries it.

**Hypothesis A-Λ.**

1. *First order in $e$.* Every cyclic quantity is kept to first order in $e$, and the mean load is that of Section 10.3.1 at the semi-major axis $a$ (its correction $(1-e^2)^{-3/2}$ is second order).
2. *Constant-$Q$ tides in both bodies.* The primary's tide drives $a$ as in A-O and $e$ with the frequency signs of each tidal component [35]. The satellite's own tide damps $e$; its effect on $a$ is second order in $e$ and is dropped with the primary's. The orbit lies in the primary's equator, and the spin axis of the satellite is normal to it.
3. *Forced libration in longitude.* The spin rate is $n+\dot\gamma$, and $\gamma$ obeys the linear libration equation, damped by the satellite's $Q_s$. Latitude libration is neglected.
4. *Quasi-static elastic cycle.* The cyclic part of the load is carried by a homogeneous, isotropic, incompressible linear elastic response of shear modulus $\mu_e$, superposed on the creep state of the step. The gravity of the elastic deformation, inertia, and the anelastic lag are neglected; their indicators are reported.
5. *Elastic–perfectly plastic cycle.* The yield surface of the cycle is the Drucker–Prager surface of Proposition 2.17.1, with strength $c+\mu_2\tilde p$ and the pressure $\bar p+\delta p(M)$ of the step's creep state plus the elastic cycle. Flow is associated in the deviatoric space.
6. *Separation of rates.* The orbital period is short compared with $\tau_n$. The cycle reaches its steady state within a step, and its effects are rates averaged over a cycle.
7. *Fatigue.* Bonds fail under a cycle that stays elastic according to an S–N law with an endurance limit, and damage adds over cycles (Miner).

**Orbit.** The orbital state of A-O gains $e$. With $\Omega_p$ the primary's spin and $S_1=\operatorname{sgn}(\Omega_p-n)$, $S_2=\operatorname{sgn}(2\Omega_p-n)$, $S_3=\operatorname{sgn}(2\Omega_p-3n)$ the signs of the lags of the three second-degree terms that carry $e$,

$$
\frac{\dot e}{e}=C_e\,\frac{\dot a_p}{a}-\frac{21}{2}\Big(\frac{k_2}{Q}\Big)_s\frac{M_p}{m}\Big(\frac{R_s}{a}\Big)^5n,
\qquad
C_e=\frac{-\tfrac34S_1-\tfrac3{16}S_2+\tfrac{147}{16}S_3-\tfrac98}{3S_1},
$$

with $\dot a_p$ the rate of A-O, $R_s$ the satellite's volume-equivalent radius, and $k_{2,s}=\tfrac32/(1+\tilde\mu)$, $\tilde\mu=19\mu_e/(2\bar\rho\bar gR_s)$ for the same $\mu_e$ as item 4 [36].

**Proposition 10.3.7.1 (Eccentricity at constant $Q$).**

1. $C_e=\tfrac{19}{8}$ when $\Omega_p>\tfrac32n$ (the classical case), and $C_e=\tfrac{25}{8}$ when $\Omega_p<\tfrac12n$ (a satellite inside the synchronous orbit with every lag reversed; Phobos, with the coefficient of [40]).
2. Over a step at constant mass $M_n$ the two rates have the same dependence $a^{-13/2}$, so
$$
\boxed{\ e_{n+1}=e_n\Big(\frac{a_{n+1}}{a_n}\Big)^{C_e+\beta_n},\qquad
\beta_n=\frac{273}{4}\Big(\frac{k_2}{Q}\Big)_s\frac{M_pR_s^5\sqrt{GM_p}}{\kappa M_n^2},\ }
$$
exactly, with $a_{n+1}$ from A-O. Without the satellite's tide ($\beta=0$) and with a constant $C_e$, $e=e_0(a/a_0)^{C_e}$ for all time.

*Proof.* (1) In Kaula's expansion [35] the degree-2 terms of first order in $e$ are $(m,p,q)=(2,0,0),(2,0,\pm1),(0,1,\pm1)$, with inclination weights $\tfrac34$ and $\tfrac14$, Hansen factors $1$, $e/2$, $7e/2$ and $3e/2$, and lags of sign $\operatorname{sgn}(m\Omega_p-(2-2p+q)n)$. Lagrange's equations give $\dot a/a\propto\sum(2-2p+q)\,\epsilon$ and $\dot e\propto\tfrac1{2e}\sum[(2-2p+q)\sqrt{1-e^2}-(2-2p)]\,\epsilon$ over these terms. At first order the sums are $\tfrac32S_1$ and $e\big(-\tfrac34S_1-\tfrac3{16}S_2+\tfrac{147}{16}S_3-\tfrac98\big)/2$, which gives $C_e$; the two signed cases follow. (2) Both $\dot a_p/a=-\tfrac2{13}\kappa M_na^{-13/2}$ and the satellite term are proportional to $a^{-13/2}$, so $d\ln e/d\ln a=C_e+\beta_n$ is constant over the step. ∎

The step bound $\Delta t_\omega$ of A-O gains $\varepsilon_\omega\,a/(|C_e+\beta_n|\,|\dot a_n|)$, the time in which $e$ changes by the fraction $\varepsilon_\omega$.

**Forced libration.** In the body frame of Section 10.3.1, with the primary on the $+x$ axis, let $A\le B$ be the principal moments about $x$ and $y$, $C$ the moment about $z$, and $\chi:=3(B-A)/C$. The true anomaly is $f=M+2e\sin M$, and the direction of the primary in the body frame makes the angle $\psi=f-M-\gamma$ with the $x$ axis.

**Proposition 10.3.7.2 (Libration).** The linear libration equation

$$
\ddot\gamma+\frac{\sqrt\chi\,n}{Q_s}\dot\gamma+\chi n^2\big(\gamma-2e\sin M\big)=0
$$

has the periodic solution $\gamma=\gamma_s\sin M+\gamma_c\cos M$ with

$$
\gamma_s+i\gamma_c=\Gamma:=\frac{2e\chi}{\chi-1+i\sqrt\chi/Q_s}.
$$

Without damping $\gamma=-2e\chi\sin M/(1-\chi)$, and $\psi=2e\sin M/(1-\chi)$. The amplitude diverges at $\chi=1$, where the free libration is in resonance with the orbit. A-Λ requires $|\Gamma|$ and the amplitude of $\psi$ to stay below a configured $\psi_{\max}\ll1$; otherwise the linearization, and possibly the lock, fails, and the model is not closed.

*Proof.* The tidal torque about $z$ is $\tfrac32(GM_p/r^3)(B-A)\sin2\psi$. Linearized about $\psi=0$ with $r=a$ at this order, $C\ddot\gamma=-3n^2(B-A)(\gamma-2e\sin M)$. The damping term is the satellite's dissipation at the libration frequency $\sqrt\chi\,n$, at quality factor $Q_s$. Inserting $\gamma=\operatorname{Im}(\Gamma e^{iM})$ gives $(\chi-1+i\sqrt\chi/Q_s)\Gamma=2e\chi$. ∎

For a homogeneous ellipsoid $\chi=3(a_1^2-a_2^2)/(a_1^2+a_2^2)$, and resonance falls at $a_1/a_2=\sqrt2$, an elongation a satellite near its Roche limit can approach. For Phobos the undamped amplitude $2e\chi/(1-\chi)$ with the ellipsoid of its axes ($\chi\approx0.39$) and $e=0.0151$ is $1.1^\circ$, against the measured $1.14^\circ\pm0.03^\circ$ [37].

**The cyclic load.** Write $D_2=\operatorname{diag}(2,-1,-1)$, $P=\operatorname{diag}(1,1,0)$ and $S=\mathbf e_x\mathbf e_y^{\!\top}+\mathbf e_y\mathbf e_x^{\!\top}$. To first order the tidal Hessian is $n^2(a/r)^3(3\hat{\mathbf r}\hat{\mathbf r}^{\!\top}-\mathbf I)=n^2[(1+3e\cos M)D_2+3\psi S]$, the centrifugal Hessian is $(n+\dot\gamma)^2P$, and the Euler force per unit mass is $-\dot\Omega\,\mathbf e_z\times\mathbf x$ with $\dot\Omega=\ddot\gamma$. The cyclic body force per unit mass is therefore

$$
\boxed{\ \delta\mathbf g(\mathbf x,M)=\big(\mathbf H_c\cos M+\mathbf H_s\sin M\big)\mathbf x+\big(E_c\cos M+E_s\sin M\big)\,\mathbf e_z\times\mathbf x,\ }
$$

$$
\mathbf H_c=n^2\big(3eD_2-3\gamma_cS+2\gamma_sP\big),\quad
\mathbf H_s=n^2\big(3(2e-\gamma_s)S-2\gamma_cP\big),\quad
E_c=n^2\gamma_c,\quad E_s=n^2\gamma_s ,
$$

and the backstress of Section 4.6 gains $C_T\rho_p\nu d^2\operatorname{dev}(\mathbf H_c\cos M+\mathbf H_s\sin M)$.

**Proposition 10.3.7.3 (The cyclic load is self-equilibrated).** About the centre of mass, in the principal frame, $\int\rho_K\delta\mathbf g\,dV=\mathbf 0$ and the torque of $\rho_K\delta\mathbf g$ has no $x$ or $y$ component. Its $z$ component is $C\,\frac{\sqrt\chi n}{Q_s}\dot\gamma$, which vanishes without damping. The relative residual $\sqrt\chi/(Q_s|1-\chi|)$ is reported, and the solve below takes the load modulo the rigid modes.

*Proof.* $\delta\mathbf g$ is linear in $\mathbf x$ and $\int\rho_K\mathbf x=\mathbf 0$. With the second moment $\mathbf J=\int\rho_K\mathbf x\mathbf x^{\!\top}$ diagonal, the torque of $\rho_K\mathbf H\mathbf x$ has components $\epsilon_{ijk}H_{kl}J_{jl}=\epsilon_{ijk}H_{kj}J_{jj}$, so only the $xy$ entry of $\mathbf H$ gives one, along $z$, equal to $H_{xy}(J_{xx}-J_{yy})=H_{xy}(B-A)$. The Euler term gives $-C\dot\Omega$. Hence the $z$ torque is $3n^2(B-A)\psi-C\ddot\gamma$, which by the libration equation equals the damping term. ∎

**The elastic cycle.**

**Proposition 10.3.7.4 (The cyclic stress is independent of the modulus).** Let $\mathbf w$ and $q$ solve the incompressible, traction-free elastic problem $\mu_e\Delta\mathbf w-\nabla q+\rho_K\mathbf f=\mathbf 0$, $\nabla\cdot\mathbf w=0$, with the rigid rows of (B1). Then $\mathbf w=\mathbf w_1/\mu_e$, where $(\mathbf w_1,q)$ solves the Stokes problem of unit viscosity under the same load, and the stress $-q\mathbf I+2\mu_e\boldsymbol\varepsilon(\mathbf w)$ does not depend on $\mu_e$.

*Proof.* Substitute $\mathbf w=\mathbf w_1/\mu_e$: the equations and the boundary condition become those of unit viscosity with $q$ unchanged, and $2\mu_e\boldsymbol\varepsilon(\mathbf w)=2\boldsymbol\varepsilon(\mathbf w_1)$. ∎

The two solutions for $\mathbf f=\mathbf H_c\mathbf x+E_c\mathbf e_z\times\mathbf x$ and $\mathbf H_s\mathbf x+E_s\mathbf e_z\times\mathbf x$ give the cyclic deviatoric stress $\boldsymbol\tau^E_c\cos M+\boldsymbol\tau^E_s\sin M$ and the pressure $\delta p_c\cos M+\delta p_s\sin M$. The modulus enters only the strains (the ratchet below), $k_{2,s}$, and three reported indicators:

- inertia, $nR_s/\sqrt{\mu_e/\bar\rho}$;
- the gravity of the deformation, $1/\tilde\mu$;
- the lag, $1/Q_s$.

A compressible body carries more deviatoric stress than the incompressible one (Section 2.8, (B2), item 2), so the cycle of item 4 is a lower estimate.

**Peak and shakedown.** Use the notation of Proposition 2.17.3: $\mathbf E_i=\sqrt2\mathbf D_i$, $|\mathbf E_i\mathbf u|=\dot\gamma_i(\mathbf u)$, and a stress sample $\hat{\boldsymbol\sigma}_i$ whose Euclidean norm is the equivalent stress $\sqrt{\tfrac12\boldsymbol\tau:\boldsymbol\tau}$ (so $\hat{\boldsymbol\sigma}=\boldsymbol\tau/\sqrt2$ and the work is $\hat{\boldsymbol\sigma}_i\cdot\mathbf E_i\mathbf u$). Let $\hat{\boldsymbol\sigma}^E_i(M)=\hat{\boldsymbol\sigma}_{c,i}\cos M+\hat{\boldsymbol\sigma}_{s,i}\sin M$ be the elastic cycle, $\ell_c,\ell_s$ its load vectors, $|\delta p_i|=(\delta p_{c,i}^2+\delta p_{s,i}^2)^{1/2}$, and

$$
\tau_i^\pm:=c_i+\mu_2\max\big(p_{0,i}\pm|\delta p_i|,\ p_{\min}\big),\qquad
N^\pm(\mathbf u):=\sum_iw_i\tau_i^\pm\dot\gamma_i(\mathbf u),
$$

the strengths of the frozen-pressure convention of Proposition 2.17.3 at the extremes of the cycle. $L_{\rm peak}$ is the largest over $M$ of the limit multiplier under $\ell(M)=\bar\ell+\ell_c\cos M+\ell_s\sin M$ at the strength $c+\mu_2(p_0+\delta p(M))$. $L_{\rm sd}$ is the reciprocal of the largest factor on all loads at which the elastic–plastic body shakes down, with the strengths frozen.

**Proposition 10.3.7.5 (Brackets of the cycle).** For every admissible $\mathbf u$ ($\mathbf B_{\rm rb}\mathbf u=\mathbf 0$):

1. *Peak.* $L_{\rm peak}\ge\big(\bar\ell^{\!\top}\mathbf u+|(\ell_c^{\!\top}\mathbf u,\ \ell_s^{\!\top}\mathbf u)|\big)/N^+(\mathbf u)$.
2. *Shakedown, kinematic (Koiter [38]).* $L_{\rm sd}\ge\Big(\bar\ell^{\!\top}\mathbf u+\sum_iw_i\big|(\hat{\boldsymbol\sigma}_{c,i}\cdot\mathbf E_i\mathbf u,\ \hat{\boldsymbol\sigma}_{s,i}\cdot\mathbf E_i\mathbf u)\big|\Big)/N^+(\mathbf u)$.
3. *Shakedown, static (Melan [39]).* If $\hat{\boldsymbol\sigma}^*$ equilibrates $\bar\ell$ (for instance the static field $\hat{\boldsymbol\sigma}^*_i=q_i\mathbf E_i\mathbf u_q/w_i$ of Proposition 2.17.3), then
$$
L_{\rm sd}\le\max_i\frac{\max_M|\hat{\boldsymbol\sigma}^*_i+\hat{\boldsymbol\sigma}^E_i(M)|}{\tau_i^-},
$$
and over $N_M$ equispaced phases $\max_M\le\max_j+\tfrac\pi{N_M}\big(|\hat{\boldsymbol\sigma}_{c,i}|+|\hat{\boldsymbol\sigma}_{s,i}|\big)$.
4. *Order.* The bound of (2) is at least that of (1), and $L_{\rm sd}\ge L_{\rm peak}$.

*Proof.* (1) At the phase $M^*$ that maximizes $\ell(M)^{\!\top}\mathbf u$, the kinematic bound of Proposition 2.17.3 holds with the strength at $M^*$, which is at most $\tau^+$. (2) Take the plastic cycle that, at each sample $i$, puts the whole increment $\mathbf E_i\mathbf u$ at the phase $M_i^*$ maximizing $\hat{\boldsymbol\sigma}^E_i(M)\cdot\mathbf E_i\mathbf u$. Its accumulation over a cycle is the compatible $\mathbf E\mathbf u$. The work of the elastic stresses is $\bar\ell^{\!\top}\mathbf u$ for the mean (the mean elastic stress equilibrates $\bar\ell$) plus $\sum_iw_i\max_M\hat{\boldsymbol\sigma}^E_i\cdot\mathbf E_i\mathbf u$, which is the expression in the bound. The dissipation is at most $N^+(\mathbf u)$. Koiter's theorem says a load factor above work over dissipation cannot shake down. (3) $\boldsymbol\rho:=\hat{\boldsymbol\sigma}^*-\hat{\boldsymbol\sigma}^{\bar E}$, the static field less the mean elastic stress, is time-independent and self-equilibrated. At the load factor $\lambda$ the stress $\lambda\hat{\boldsymbol\sigma}^E(M)+\lambda\boldsymbol\rho=\lambda(\hat{\boldsymbol\sigma}^*+\hat{\boldsymbol\sigma}^E(M)-\hat{\boldsymbol\sigma}^{\bar E}+\hat{\boldsymbol\sigma}^{\bar E})$ is $\lambda(\hat{\boldsymbol\sigma}^*+\hat{\boldsymbol\sigma}^E(M))$ with the cyclic part only. It is admissible for all $M$ when $\lambda$ times the stated maximum is at most 1, and Melan's theorem then gives shakedown. The sampled maximum errs by at most the Lipschitz constant $|\hat{\boldsymbol\sigma}_c|+|\hat{\boldsymbol\sigma}_s|$ times half the spacing $\pi/N_M$. (4) $\sum_iw_i|(a_i,b_i)|\ge|(\sum_iw_ia_i,\sum_iw_ib_i)|$, and $\sum_iw_i\hat{\boldsymbol\sigma}_{c,i}\cdot\mathbf E_i\mathbf u=\ell_c^{\!\top}\mathbf u$ because the elastic cycle equilibrates $\ell_c$; likewise for $s$. Shakedown requires admissibility at every phase. ∎

Both theorems hold for the associated material with the strength $c+\mu_2\tilde p$ at frozen pressure, the same surrogate as Proposition 2.17.3. Shakedown of the non-associated granular law can be lower [16]. Between $L_{\rm sd}>1$ and $L_{\rm peak}<1$ the body never collapses at an instant, but it accumulates plastic strain cycle after cycle (ratchet) or alternates plastically. Only $L_{\rm peak}>1$ ends the quasi-static route (Chapter 12). The orbits at which $L_{\rm sd}$ and $L_{\rm peak}$ cross 1 bound the ratchet band of Algorithm 2.17.2.

**The steady cycle and the ratchet.** With the plastic strain $\hat{\boldsymbol\varepsilon}_i$ of the cycle in the same convention, its residual stress is $\hat{\boldsymbol\sigma}^R_i(\hat{\boldsymbol\varepsilon})=\mu_e(\mathbf E_i\mathbf u_{\hat\varepsilon}-\hat{\boldsymbol\varepsilon}_i)$, where $\mathbf u_{\hat\varepsilon}$ solves the elastic system with the load $\sum_iw_i\mu_e\mathbf E_i^{\!\top}\hat{\boldsymbol\varepsilon}_i$. The mean stress is that of the step's creep state, $\hat{\boldsymbol\sigma}^{\rm cr}_i=\eta_i\mathbf E_i\mathbf u$.

**Algorithm 10.3.7.6 (Steady cycle).**

```text
input:  sigma_cr, p_bar; the elastic cycle (sigma_c, sigma_s, dp_c, dp_s); phases M_j = 2 pi j / N_M
eps <- 0
for cycle = 1 .. K_max:
    eps_start <- eps;  range <- 0
    for j = 1 .. N_M:
        tau_ij <- c_i + mu_2 max(p_bar_i + dp_c,i cos M_j + dp_s,i sin M_j, p_min)
        repeat                                    # initial-stress iteration at fixed phase
            s_i <- sigma_cr,i + sigma_E,i(M_j) + sigma_R,i(eps)
            d_i <- [|s_i| - tau_ij]_+ / mu_e * s_i / |s_i|        # radial return
            eps <- eps + d;  range_i <- range_i + |d_i|
        until max_i [|s_i| - tau_ij]_+ <= tol tau_ij
    if |sigma_R(eps) - sigma_R(eps_start)| <= tol_cycle max_i tau_ij: steady; stop
ratchet per cycle:      d_eps <- eps - eps_start,   d_u <- u_{d_eps}
alternating plasticity: range (per sample)
```

**Proposition 10.3.7.7 (The cycle).**

1. If the trial stress $\hat{\boldsymbol\sigma}^{\rm cr}_i+\hat{\boldsymbol\sigma}^E_i(M_j)$ is admissible at every sample and phase, the algorithm returns $\hat{\boldsymbol\varepsilon}\equiv0$: no ratchet and no alternating plasticity.
2. At each phase the radial return is the closest-point projection onto the frozen yield surface in the energy norm $\mu_e^{-1}|\cdot|^2$, i.e. the backward Euler step of the elastic–perfectly plastic law [43]. The phase loop is therefore the incremental history sampled at $N_M$ phases.
3. For an elastic–perfectly plastic body under a periodic load the residual stress converges to a periodic state [41]. The plastic strain then grows by the same $\Delta\hat{\boldsymbol\varepsilon}$ each cycle (ratchet) or returns to its start (alternating plasticity, $\Delta\hat{\boldsymbol\varepsilon}=0$, range $>0$).
4. The ratchet strain per cycle scales as load over $\mu_e$: with every load and $c$ multiplied by $s$, $\Delta\hat{\boldsymbol\varepsilon}$ is multiplied by $s$. With $\mu_e$ multiplied by $s$, $\Delta\hat{\boldsymbol\varepsilon}$ is divided by $s$.

*Proof.* (1) $\hat{\boldsymbol\varepsilon}=0$ gives $\hat{\boldsymbol\sigma}^R=0$ and no excess. (2) For a ball of radius $\tau$ the closest point of $\mathbf s$ is $\tau\mathbf s/|\mathbf s|$, and the plastic increment that removes the excess at elastic stiffness $\mu_e$ is $(|\mathbf s|-\tau)\mathbf s/(\mu_e|\mathbf s|)$. (3) This is Frederick and Armstrong's convergence of the internal stresses [41]. (4) The return and the residual map are positively homogeneous of degree one in (stress, strength) and of degree $-1$ in $\mu_e$. ∎

The ratchet enters the slow operator as the velocity $\mathbf v_r=\frac{n}{2\pi}\Delta\mathbf u$, a whole-body field with zero momentum and angular momentum. The deformation it describes is real everywhere, so it is not masked by $\chi_Y$:

- the cores and markers are carried along $\mathbf v_{\rm adv}+\mathbf v_r$;
- the surface vertices move by $\tau_n(\mathbf v_r\cdot\mathbf N_i)\mathbf N_i$ before settling (the tangential part only relabels the surface, as for the creep predictor);
- the step is bounded by $\Delta t_r=\theta_{\rm CFL}h_{\Gamma,\min}/\max_i|\mathbf v_r(\mathbf X_i)\cdot\mathbf N_i|$.

The alternating plastic shear adds $\frac{n}{2\pi}\Delta\gamma_{{\rm alt},i}$ (range per cycle) to the plastic driving rate of A-ϖ (Section 11.2).

**High-cycle fatigue.** Where the steady cycle is elastic, the utilization of the cycle is $s_i:=\max_j|\hat{\boldsymbol\sigma}_i(M_j)|/\tau_{ij}$. With the S–N line $s=b_0-b_1\log_{10}N_f$ and the endurance limit $s_e$ [42],

$$
\boxed{\ r_{F,i}=\frac{n}{2\pi}\,10^{-(b_0-s_i)/b_1}\ \ (s_i>s_e),\qquad r_{F,i}=0\ \ (s_i\le s_e),\qquad
\dot\varpi=(1-\varpi)\Big(\frac{\dot\gamma_p^\star}{\gamma_\varpi}+r_F\Big)-h_\varpi\varpi .\ }
$$

Proposition 11.9 applies with $a=\dot\gamma_p^\star/\gamma_\varpi+r_F$, and $\Delta t_\varpi$ uses the same $a$. The shallow cohesion $c_\Gamma$ of Section 11.6 contains $\mathcal R\varpi$ and inherits the damage. The S–N line is that of rock in compression, transferred to the bonds of the matrix; this transfer is a hypothesis.

**The loose layer.** The settling of step S and Section 10.2 test repose against the face gravity of every phase, $\mathbf g_f+\delta\mathbf g(\mathbf x_f,M_j)$, and keep the steepest.

**Proposition 10.3.7.8 (Envelope).** A rate-independent loose layer that slips only when its slope exceeds repose, under a cycle slow compared with its own runout, comes to rest in the state where its slope is below repose at every phase. Thereafter it does not slip again. So the quasi-static settling against the steepest phase is the cycle-averaged behaviour of A-C4, and a ratchet of the layer needs hysteresis or noise that A-C4 does not contain.

*Proof.* Slip at a phase only reduces the excess of the slope over repose at that phase, and with the same static and dynamic friction it stops at repose. Once the slope is below repose at every phase, the condition of item 2 of A-C4 never holds again. ∎

**Escape on an eccentric orbit.** $W$ now depends on $M$, so $E_J$ is not conserved and Proposition 10.3.6.2 does not apply. A packet launched at phase $M_0$ flies in the mean field plus $\delta\mathbf g(\mathbf x,M_0+nt)$, with the Coriolis rate $n+\dot\gamma$. It is escaped when it passes a neck plane of the instantaneous Hill region. The point-mass Hill radius scales with $r$ and the primary lies along $\psi$, so the necks are

$$
L_k(M)=\mathbf x_c+(1-e\cos M)\,\mathbf R_z(\psi(M))\,(L_k-\mathbf x_c),\qquad \mathbf e_k(M)=\mathbf R_z(\psi(M))\,\mathbf e_k .
$$

Launch phases are spread uniformly over the packets (a low-discrepancy sequence in the source index), so the losses average the cycle.

**Reduction.** With $e=0$, $\Gamma=0$, every cyclic term vanishes, and the model is A-O exactly. With A-Λ off it is A-O.

**Reports.** Each step reports:

- $e$, $\chi$, $|\Gamma|$, and the amplitude of $\psi$;
- the torque residual of Proposition 10.3.7.3, and the three indicators of item 4;
- every $N_L$ steps, the bound of $L_{\rm peak}$ and the bracket of $L_{\rm sd}$;
- the cycles to the steady state, the volume fractions that ratchet and that alternate, the largest ratchet speed, and the largest fatigue rate.

## 10.4 Computational realization: dependency graph, GPU and ray-tracing residency, and ECS organization

Like Section 3.15, this changes no computed value; it only reuses unchanged results or reorders independent work.

### 10.4.1 Data dependencies of one geological step

| Stage (10.3.4) | Reads | Writes | Device | Independent of |
|---|---|---|---|---|
| 1 gravity | mesh, AS, segment table $\mathcal T$, $\theta$, point sets | $U,\nabla U,H$ on all point sets | RT units + compute (Section 3.15) | — |
| 2 hyperbolic rays | fields of 1, $\theta$, mesh, SurfaceBVH, CellBVH | $\mathbf v,\mathbf S,p$ and ray residuals | RT units + compute (Sections 8.3, 10.3.2) | — |
| 3 yield | $\mathbf v,\mathbf S,p$ | $F_\delta$, $\chi_Y$, $\mathbf v_{\rm sl}$, core fit | compute; graph components on CPU | — |
| 4 friction chain | $\mathbf v$, $p$, fields | $\phi_i$ | compute | 5 |
| 5 Mode C | $F_\delta$, $\chi_Y$, $\rho_s$ | $J_{\rm ero},J_{\rm dep}$ | compute | 4 |
| 6 step size | extrema of 3–5, $\nabla\mathbf v_{\rm adv}(\mathbf c_k)$ | $\Delta t_n$ | reductions | — |
| 6b core prediction | $\mathbf v_{\rm sl}$, $\chi_Y$, $\theta$, $\Delta t_n$ | $\theta^{\rm pred}$ (may shorten $\Delta t_n$) | CPU or compute ($K$ items) | — |
| 7 SOCP | $d_{\rm pred}$, $\phi$, $\rho_{\rm cand}$ | $d_*$ | CPU (or GPU first-order solver) | — |
| 8a geometry | $d_*$ | mesh, $\rho_s$ | compute | — |
| 8a′ AS refit and dirty test | old and new mesh, $\mathcal T$ | dirty-ray list | RT units | 8c |
| 8c core closure | new mesh, $\theta^{\rm pred}$, $\theta^{(n)}$, $\mathbf v_{\rm adv}$ | $\theta^{(n+1)}$ | compute (reductions) | 8a′ |
| 8b realignment (if triggered) | mesh, $\theta^{(n+1)}$ | mesh, $\theta$ | compute (volume and inertia integrals) | — |

Stages in the last column read the same inputs and write disjoint data, so they may run concurrently. The critical path is 1 → 2 → 3 → 6 → 6b → 7 → 8a → (8a′ ∥ 8c) → 1, dominated by 2. The mass integral of 8c and 8a needs no new traversal: it is the exact surface formula of the exact polynomial volume rule for $\rho_b$ plus the ray-segment rule (G2) for the cores, with the segments from $O$ reused until the geometry changes. The stages of Chapter 11 attach to this graph as listed in Section 11.9. The marker update and the thermal solve depend only on the outputs of steps 3 and 1 respectively, and the spin update needs only the settled inertia.

### 10.4.2 Residency and transfers

All per-point and per-node arrays stay in device memory for the run: mesh, acceleration structure, segment table, point sets, fields, modal coefficients, $\eta$ and $\tilde p$ at the nodes, $\chi_Y$, $\rho_s$, and the cores. Only *reductions* return to the host:

- the hyperbolic momentum, constitutive, divergence, traction, energy, and ray residuals;
- the $6\times6$ rigid-mode Gram matrix;
- the extrema for $\Delta t_n$;
- the Gauss–Newton matrix;
- $\mathcal M$, $\mathbf x_{\rm cm}$, and $\mathbf I$;
- the SOCP data on $\mathcal A$.

The components of step 3 need only one byte of $\chi$ per point. Each stage is one command submission.

### 10.4.3 Precision policy

| Quantity | Arithmetic | Reason |
|---|---|---|
| traversal, candidate hits | single precision on RT units | selects triangles only |
| hit distances, $b_k,d_k$, radial integrals | double, or double-float, after hit refinement (3.15.4) | endpoints enter every integral |
| accumulation of $U,\nabla U,H$ (six components) | double, or compensated single, fixed reduction order | reproducibility; finite-part cancellations |
| characteristic state reconstruction and source accumulation | single for ray-local work, double for residuals and reductions | pseudo-time iteration |
| projected modal state $\mathbf u$, $\mathbf M_\Gamma$ factor, residuals of (a)(b), $\boldsymbol\lambda$ | double | acceptance decisions |
| core parameters, $\mathbf F_k$, Gauss–Newton solve | double | $K$ is small |

### 10.4.4 Organization in an entity–component–system architecture

The rules below concern data layout and scheduling only.

1. **Coarse entities.** One entity is the body: mesh (vertices, fixed connectivity), surface fields, modal state, frame state, and cores $\theta$. One entity per point-set role (surface nodes, face centres, $\mathcal X_{\rm vol}$, $\mathcal X_Y$) holds its points, field mask, results, and segment table. Points, nodes, triangles, and cores are **never** entities. Components hold structure-of-arrays storage laid out exactly as the device buffers, so uploads are copies and accesses are coalesced.
2. **Systems are the stages of 10.4.1,** each declaring its read and write sets. The scheduler derives the concurrency of 10.4.1 from the absence of write conflicts, with no further ordering.
3. **Change detection drives invalidation.** This is what realizes Propositions 3.15.1–3.15.2:
   - a mesh write marks changed directions, and only the rays of Proposition 3.15.2 are retraced;
   - a $\theta$-only write triggers W3–W4;
   - a common weight factor (Level 1) scales only the stored per-kernel parts, on an unchanged geometry;
   - a change of $K$ invalidates the per-kernel partial sums;
   - a realignment invalidates everything.
4. **Long stages are asynchronous tasks** (ray sweeps, outer constitutive refresh, SOCP, GPU submissions). Results are written back at completion, so readers see only step-boundary states.
5. **Determinism switch.** With fixed reduction trees (W4, the deterministic reduction trees and Gauss–Newton assembly), runs on one device are bitwise identical.

### 10.4.5 Cost per geological step

$$
C_{\rm step}\approx C_{\rm grav}+N_{\rm ray}C_{\rm ray}+C_{\rm yield}+C_{\rm SOCP}(|\mathcal A|)+C_{\rm core}+C_{\rm run},
\qquad
C_{\rm core}=O(K)+N_{\rm GN}\,O(N_qp^2),
\qquad
C_{\rm run}=O\big(N_{\rm src}\,\bar N_\tau\big),
$$

with $N_{\rm ray}$ hyperbolic ray sweeps of cost $C_{\rm ray}$ each (Section 10.4.6), $N_{\rm src}$ eroding nodes and relieved vertices, and $\bar N_\tau$ mean runout substeps, each a constant-time face lookup. Relief routing adds one warm-started second solve on geometry steps. Against a full rebuild, full retrace, and assembled $\mathbf K(\eta)$ per round, the dominant terms fall:

- traversal goes from $O(N_xN_\Omega(\log F+\bar L))$ to the dirty fraction, which is zero in equilibrium and for density-only updates;
- each hyperbolic ray sweep costs $C_{\rm ray}=O(PR\log N_{\rm BVH}+PR\bar L+P)$ of Section 10.4.6, with no assembled global stiffness matrix.

With the hypotheses of Chapter 11, $C_{\rm step}$ additionally gains the terms of Section 11.9: the marker transport and reconstruction, the nonlocal rate and $\chi_h$ at each pressure update, the thermal solve on dirty faces, and an $O(1)$ spin update. The unknowns and the characteristic CFL condition of the ray relaxation replace the former primal–dual step condition. These are planning estimates.

### 10.4.6 Kernels of the hyperbolic characteristic ray solver

The rheology kernels are divided between RT hardware and compute shaders:

1. **Characteristic packet generation.** Compute shaders evaluate the local eigenvalues, eigenvectors, Cauchy density, viscosity, and Hamiltonian ray direction.
2. **BVH traversal.** RT hardware traces the backtraced characteristic segments through the surface and cell BVHs. Repeated queries after $s_h+\varepsilon_{\rm geom}$ return all crossings of a non-star-shaped body.
3. **State gathering.** Compute shaders use the returned cell IDs and barycentric coordinates to interpolate $(\mathbf v,\mathbf S,p)$ at departure points.
4. **Source and stress update.** The source terms are integrated along each segment; the frozen-viscosity stress equation uses the exact exponential update in Section 10.3.2.
5. **Boundary characteristic solve.** Incoming modes are reconstructed from the traction condition; outgoing modes are retained from the interior ray.
6. **Global reductions.** Compute shaders reduce momentum, constitutive, divergence, traction, energy, and rigid-mode residuals. The $6\times6$ rigid projection is solved in double precision.

The per-sweep work is

$$
C_{\rm ray}=O\!\left(PR\log N_{\rm BVH}+PR\bar L+P\right),
$$

where $P$ is the number of volume samples, $R$ the number of characteristic families, and $\bar L$ the mean number of crossed BVH cells per packet. The geometric SOCP remains a separate stage with cost $C_{\rm SOCP}(|\mathcal A|)$.

# 11. Evolving material state and angular momentum (hypotheses A-Ξ, A-ϖ, A-Θ, A-L)

Chapters 2–10 freeze the solid fraction, the cohesion, and the strength of the surface skin, and they prescribe the spin. This chapter adds four named hypotheses. Each has a switch, and each acts only through quantities the loop of Section 10.3.4 already has. With a switch off, every formula of Chapters 2–10 holds unchanged in that respect (Theorem 11.13).

| Hypothesis | State it adds | Acts on | Switched off by |
|---|---|---|---|
| A-Ξ, dilatancy (11.3–11.4) | solid fraction $\nu$ on markers; its density effect on the cores | $\nabla\cdot\mathbf v$, $\mu_d$, $\mu_y$, $\rho_K$ | $K_\Xi=0$ |
| A-ϖ, damage of the cohesion (11.5) | damage $\varpi$ on markers | $c$ | $\varpi\equiv0$ |
| A-Θ, thermal fatigue of the surface (11.6) | surface damage $\varpi_\Gamma$ on vertices; a source of $\rho_s$ | the shallow cohesion $c_\Gamma$; the loose-layer mass | $r_\Theta\equiv0$ |
| A-L, spin from angular momentum (11.7) | spin angular momentum $L$ | $\omega$ | the exogenous table of Section 10.3.1 |

Two devices are shared: Lagrangian state markers (Section 11.1), and a nonlocal plastic driving rate (Section 11.2). All four processes belong to the slow operator of hypothesis A-T1 (Section 10.3.1). Their rates are frozen over $\tau_n=\lambda\Delta t_n$ and integrated exactly over it. The settling of step S advances no time and changes none of these states. The one exception is the spin rate, which follows the inertia at constant angular momentum (Section 11.7). Placement in the loop is Section 11.9.

## 11.1 Lagrangian state markers

The cores carry the density (Section 9.2). The solid fraction $\nu$ and the damage $\varpi$ are intensive material properties. They obey no balance law, but they must move with the material and stay in their admissible ranges. The volume nodes of (G2) are attached to rays from $O$, and every geometric update and realignment moves them, so they cannot carry such a state.

**Definition 11.1 (Markers).** A marker set is $\mathcal Y=\{(\mathbf y_j,\nu_j,\varpi_j)\}$ with $\mathbf y_j\in\mathcal V$. The admissible sets are $\mathfrak A_\nu=[\nu_{\rm lo},\nu_{\rm hi}]$ (Section 11.3) and $\mathfrak A_\varpi=[0,1]$.

1. *Seeding.* At step 0, one marker sits at each volume node of (G2) on $\mathcal V^{(0)}$, with state $(\nu^{(0)},\varpi^{(0)})$ there.
2. *Transport.* With the advection velocity $\mathbf v_{\rm adv}$ of Section 9.2.2 (under A-Ξ including the dilatational field, Section 11.4), frozen over the step, use the RK2 of the core centres: $\mathbf y_j^{\rm mid}=\mathbf y_j+\tfrac{\tau_n}2\mathbf v_{\rm adv}(\mathbf y_j)$, $\mathbf y_j'=\mathbf y_j+\tau_n\mathbf v_{\rm adv}(\mathbf y_j^{\rm mid})$. A velocity at a point outside $\mathcal V^{(n)}$ is taken at the closest point $\pi(\mathbf y)$ (G6), as in Section 9.2.5.
3. *State update.* At $\mathbf y_j^{\rm mid}$, with the rates of Sections 11.3 and 11.5 frozen, apply the exact step formulas (Propositions 11.5 and 11.9).
4. *Realignment.* $\mathbf y_j\leftarrow\mathbf Q(\mathbf y_j-\mathbf x_{\rm cm})$ in step 8b, like the core centres.
5. *Maintenance (on the settled geometry, Section 11.9).*
   - *Deletion.* A marker with $\mathbf y_j\notin\mathcal V^{(n+1)}$ is deleted.
   - *Coverage.* Every point of $\mathcal X_{\rm vol}$ must see at least $n_{\min}$ markers within the radius $h_R$ along segments inside $\mathcal V$ (any-hit query, as in G7). Otherwise a marker is inserted at that point. Its state is the deposit state $(\nu_{\rm dep},\varpi_{\rm dep})$ if the point lies in $\mathcal V^{(n+1)}\setminus\mathcal V^{(n)}$ (the old body carried by the rigid motion of 8b) and its closest surface vertex is in $\mathcal D^{(n)}$, or received a relief or routed deposit in the step. Otherwise the state is the Shepard mean of the visible markers within $2h_R$, or the deposit state if there are none.
   - *Merging.* Where more than $n_{\max}$ markers lie in a ball of radius $h_R/4$, they are replaced by one marker at their mean position, with their mean state.

**Reconstruction.** At a point $\mathbf x$, with the visible markers within $h_R$ and weights $W_j=w(|\mathbf x-\mathbf y_j|/h_R)$, $w(s)=(1-s^2)_+^4$, the linear moving least-squares fit of a marker field $z$ is

$$
(a,\mathbf b)=\arg\min_{a,\mathbf b}\sum_jW_j\big(a+\mathbf b\cdot(\mathbf y_j-\mathbf x)-z_j\big)^2,\qquad
\mathcal Rz(\mathbf x):=\operatorname{clip}_{\mathfrak A}(a),\qquad
\nabla\mathcal Rz(\mathbf x):=\begin{cases}\mathbf b,&a\in\operatorname{int}\mathfrak A,\\ \mathbf 0,&\text{otherwise}.\end{cases}
$$

If the weighted design is singular (fewer than four visible markers not in a plane), the Shepard mean $\sum_jW_jz_j/\sum_jW_j$ with zero gradient is used, and the point is counted. With no visible marker within $h_R$, the value of the nearest marker is used, with zero gradient, and the point is counted.

**Lemma 11.2 (Reconstruction).**

1. $\mathcal Rz(\mathbf x)\in\mathfrak A$ for every $\mathbf x$ and every admissible marker state.
2. If $z$ is affine on the support and the design is nonsingular, then $\mathcal Rz=z$, and $\nabla\mathcal Rz=\nabla z$ wherever $z(\mathbf x)\in\operatorname{int}\mathfrak A$.
3. If no marker moves and no state changes, $\mathcal R$ is unchanged.

*Proof.* (1) A clipped value, or a convex combination of admissible values. (2) The affine function fits with zero residual, and a nonsingular design makes the minimizer unique. (3) The fit depends only on positions and states. ∎

$\mathcal R$ is evaluated at the volume nodes, on $\mathcal X_Y$, at the surface vertices, at the boundary quadrature points of (G3), at the core centres, and at departure points. The gradient is needed only at the boundary quadrature points, through $\nabla\eta$ (Section 8.3). Clipping makes $\nabla\mathcal Rz$ jump at the clip boundary, but it stays bounded, and $L^\infty$ is all Section 8.3 needs. Visibility keeps markers across a gap of a nonconvex body (a neck, a crater wall) out of the fit. The reconstruction error and the effect of the maintenance rules are PO-25.

## 11.2 The plastic driving rate and its nonlocal regularization

Both A-Ξ and A-ϖ soften. Under A-Ξ the friction of a dense packing relaxes from its peak to the critical-state value; under A-ϖ the cohesion falls with plastic strain. A softening law driven by the local plastic rate localizes into bands, and in a local model the band width is set by the discretization. That adds a resolution dependence to the ill-posedness of PO-10. The normative driving rate is therefore nonlocal.

**Definition 11.3 (Driving rate).** The local plastic rate is $\dot\gamma_p:=\mathbf 1_{\{F_\delta>0\}}\dot\gamma$, as in (B6), with $F_\delta$ evaluated on $\mathcal X_Y$ and extended to a point by its classification cell (Section 9.2.2). With a length $\ell_p$ and the kernel $k(s)=(1-s^2)_+^2$,

$$
\bar{\dot\gamma}_p(\mathbf x):=\frac{\int_{\mathcal V}k(|\mathbf x-\mathbf y|/\ell_p)\,\mathbf 1[\overline{\mathbf x\mathbf y}\subset\mathcal V]\,\dot\gamma_p(\mathbf y)\,dV_{\mathbf y}}{\int_{\mathcal V}k(|\mathbf x-\mathbf y|/\ell_p)\,\mathbf 1[\overline{\mathbf x\mathbf y}\subset\mathcal V]\,dV_{\mathbf y}} .
$$

The driving rate is masked where $F_\delta$ is computed, on the volume nodes $\mathbf x_q\in\mathcal X_{\rm vol}\subset\mathcal X_Y$:

$$
\boxed{\ \dot\gamma^\star_{p,q}:=\mathbf 1_{\{F_\delta(\mathbf x_q)>0\}}\,\bar{\dot\gamma}_p(\mathbf x_q),\ }
$$

and it is carried to any other point by the visible Shepard mean over the nodes within the reconstruction radius $h_R$ of Section 11.1,

$$
\dot\gamma_p^\star(\mathbf x):=\frac{\sum_qW_q(\mathbf x)\,\dot\gamma^\star_{p,q}}{\sum_qW_q(\mathbf x)},\qquad
W_q(\mathbf x):=w(|\mathbf x-\mathbf x_q|/h_R)\,\mathbf 1[\overline{\mathbf x\mathbf x_q}\subset\mathcal V],
$$

with $w(s)=(1-s^2)_+^4$, and the value of the nearest node if no visible node lies within $h_R$, as for $\mathcal R$. The inertial number $I_c$ at a point is carried in the same way from its node values. The stress solve uses the node values $\dot\gamma^\star_{p,q}$ (Section 11.4), and the material points receive the same field. A pointwise mask read at an arbitrary point by its classification cell would be zero almost everywhere once the yielded set is a scatter of single nodes. The solve would then dilate there while no material point saw the yielding.

The integral is discretized on the volume nodes of (G2). The nodes are aggregated into cubic cells of side at most $\ell_p/4$, each carrying its weight $\sum_qw_q$, its weighted centroid, and the weighted mean of $\dot\gamma_p$. This is a quadrature of the same integral with relative error $O((h_{\rm cell}/\ell_p)^2)$. Visibility is decided by one any-hit query per pair, except that a segment no longer than the depth of its starting point is inside without a query. Only nodes with $F_\delta>0$ are evaluated, since the mask removes the rest. The length obeys $\ell_p\ge c_\ell\,h_{\rm vol}$, with $h_{\rm vol}:=\text{the maximum local CellBVH support diameter}$ (half the shortest resolved wavelength) and a configured $c_\ell\ge1$.

**Proposition 11.4 (Properties of the driving rate).**

1. $0\le\dot\gamma_p^\star\le\operatorname{ess\,sup}\dot\gamma_p$. $\dot\gamma^\star_{p,q}=0$ at every node with $F_\delta\le0$, and $\dot\gamma_p^\star(\mathbf x)=0$ wherever no visible node within $h_R$ of $\mathbf x$ (or, failing one, the nearest node) has $F_\delta>0$. The largest value over all points is attained on $\mathcal X_{\rm vol}$.
2. $\dot\gamma_p\equiv0$ implies $\dot\gamma_p^\star\equiv0$.
3. As $\ell_p\to0$, $\bar{\dot\gamma}_p(\mathbf x)\to\dot\gamma_p(\mathbf x)$ at every Lebesgue point of $\dot\gamma_p$ in the interior of $\mathcal V$ (the local law).
4. $|\bar{\dot\gamma}_p(\mathbf x)-\bar{\dot\gamma}_p'(\mathbf x)|\le\|\dot\gamma_p-\dot\gamma_p'\|_{L^1(\mathcal V)}/m_\ell(\mathbf x)$, with $m_\ell(\mathbf x)>0$ the denominator above and $\|k\|_\infty=1$.

*Proof.* (1) $\bar{\dot\gamma}_p$ is a weighted average with nonnegative weights, the mask multiplies it by $0$ or $1$, and the Shepard mean is a convex combination of node values. (2) Immediate. (3) For $\ell_p<\operatorname{dist}(\mathbf x,\Gamma)$ the visibility indicator equals $1$, and averaging against a bounded, nonnegative, radial kernel of shrinking support converges at Lebesgue points (Lebesgue differentiation theorem). (4) The numerators differ by at most $\int_{\mathcal V}|\dot\gamma_p-\dot\gamma_p'|$. ∎

*Reading.* The mask keeps the loading condition local to the resolution of $\mathcal X_{\rm vol}$. Unyielded material farther than $h_R$ from any yielded node, including the locked core and rigidly sliding blocks, does not evolve. The intensity is averaged over $\ell_p$. Band-scale state fields are then averages over $\ell_p$, and $\ell_p$ is a model length, not a resolved band width. Convergence as the numerical resolution grows at fixed $\ell_p$, and the meaning of the limit $\ell_p\to0$, are PO-24.

## 11.3 Dilatancy and the solid fraction (hypothesis A-Ξ)

Sheared dense granular material dilates and loose material contracts, both toward a critical solid fraction [18][19]. The friction in excess of the critical-state value equals the rate of dilation (stress–dilatancy). The reference model fixes $\nu=\nu_0$ and $\nabla\cdot\mathbf v=0$, which is zero dilatancy (Remark 2.17.2). Isotropic compaction under load is not included. Interior pressures are of order $\tfrac{2\pi}{3}G\rho_K^2R^2\approx40$ Pa for Ryugu [5], far below grain-crushing stresses, so A-Ξ is purely shear-driven.

**Hypothesis A-Ξ (rate-type dilatancy).**

1. *Material solid fraction.* $\nu(\cdot,t)$ is carried by the material, with a given initial field $\nu^{(0)}$; the uniform $\nu^{(0)}\equiv\nu_0$ is a special case. The grain-scale density $\rho_g:=\rho_K/\nu$ is materially conserved. With mass conservation this gives
$$
\frac{D_t\rho_K}{\rho_K}=\frac{D_t\nu}{\nu}=-\nabla\cdot\mathbf v .
$$
2. *Dilatancy law.*
$$
\boxed{\ \nabla\cdot\mathbf v=\dot\varepsilon_v:=\Xi(\nu,I_c)\,\dot\gamma_p^\star,\qquad
\Xi=K_\Xi\big(\nu-\nu_c(I_c)\big),\qquad
\nu_c(I_c)=\nu_c^0-\Delta\nu_c\frac{I_c}{I_\nu+I_c},\ }
$$
with $K_\Xi\ge0$, $\Delta\nu_c\ge0$, and $I_\nu>0$. $\Xi$ plays the role of $\tan\psi$, the tangent of the dilatancy angle, in the notation of [18].
3. *Stress–dilatancy.* The friction of Section 4.5 is read as the critical-state friction $\mu_{\rm cs}(I_c):=\mu_s+(\mu_2-\mu_s)I_c/(I_0+I_c)$, and
$$
\boxed{\ \mu_d(I_c,\nu)=\mu_{\rm cs}(I_c)+\Xi(\nu,I_c),\qquad
\mu_y(\nu):=\lim_{I_c\to0}\mu_d=\mu_s+K_\Xi(\nu-\nu_c^0).\ }
$$
$\mu_y(\nu)$ replaces $\mu_s(\nu_0,Z_0)$ in (B2), in (M5), and in the strength of (B5).
4. *Admissibility.* With $\nu_{\rm lo}:=\min(\operatorname{ess\,inf}\nu^{(0)},\nu_c^0-\Delta\nu_c,\nu_{\rm dep})$ and $\nu_{\rm hi}:=\max(\operatorname{ess\,sup}\nu^{(0)},\nu_c^0,\nu_{\rm dep})$, require $\nu_{\rm hi}$ below random close packing and
$$
\mu_{\min}:=\mu_s-K_\Xi(\nu_c^0-\nu_{\rm lo})>0 .
$$
5. *Where $\nu$ enters.* Only the friction and the density. The contact-scale prefactors $p_c=C_cZ_0\nu_0F_H/d_p^2$ and $\mathbf T_{\rm tide}=C_T\rho_p\nu_0d_p^2\operatorname{dev}H_W$ keep $\nu_0$, and so does the loose layer, $\rho_{\rm bulk}=\rho_p\nu_0$ (Section 10.2). For $\mathbf T_{\rm tide}$ the error is negligible (Section 4.23). For $p_c$ the relative error is at most $\max|\nu/\nu_0-1|$, which is reported.
6. *No equation of state.* There is no elastic compressibility. $p$ remains the multiplier of the constraint $\nabla\cdot\mathbf v=\dot\varepsilon_v$ (Section 11.4), not a function of $\nu$.

**Proposition 11.5 (Exact step for the solid fraction).** At a material point, freeze $I_c$ and $\dot\gamma_p^\star$ over $\tau_n$, and let $\Delta\gamma_p:=\tau_n\dot\gamma_p^\star$ and $\nu_c=\nu_c(I_c)$. Then $D_t\nu=-\nu\dot\varepsilon_v=-K_\Xi\nu(\nu-\nu_c)\dot\gamma_p^\star$ has the solution

$$
\boxed{\ \nu'=\frac{\nu_c}{1-(1-\nu_c/\nu)\,e^{-K_\Xi\nu_c\Delta\gamma_p}},\qquad J_\nu:=\frac{\nu'}{\nu}, \ }
$$

and:

1. $\nu'$ lies between $\nu$ and $\nu_c$, is monotone in $\Delta\gamma_p$, and tends to $\nu_c$ as $\Delta\gamma_p\to\infty$; it never crosses $\nu_c$;
2. $\mathfrak A_\nu$ is invariant, because $\nu_c\in[\nu_c^0-\Delta\nu_c,\nu_c^0]\subset\mathfrak A_\nu$;
3. $J_\nu$ is the density ratio $\rho_K'/\rho_K$ of the material point, and $-\ln J_\nu$ its volumetric strain over the step;
4. $\Delta\gamma_p=0$ or $\nu=\nu_c$ gives $\nu'=\nu$.

*Proof.* With $u=1/\nu$ the equation becomes the linear $du/d\gamma=K_\Xi(1-\nu_cu)$, so $u(\Delta\gamma_p)=1/\nu_c+(1/\nu-1/\nu_c)e^{-K_\Xi\nu_c\Delta\gamma_p}$, which is the formula. $u$ moves monotonically from $1/\nu$ toward $1/\nu_c$ without reaching it in finite strain, which gives (1), (2), and (4). (3) is item 1 of A-Ξ. ∎

**Proposition 11.6 (Constitutive bounds under A-Ξ).** Write $\mu_2^\Xi(\nu):=\mu_2+K_\Xi(\nu-\nu_c^0+\Delta\nu_c)$. For every admissible state:

1. $\mu_{\min}\le\mu_d(I_c,\nu)<\mu_2^\Xi(\nu)$, and $\mu_d\to\mu_2^\Xi(\nu)$ as $I_c\to\infty$;
2. $\mu_d$ is nondecreasing in $I_c$;
3. the viscosity bounds of Section 2.8 hold with $\mu_s$ replaced by $\mu_{\min}$ in $\eta_{\rm lo}$ and $\mu_2$ by $\max_{\mathfrak A_\nu}\mu_2^\Xi$ in $\eta_{\rm hi}$. Proposition 2.17.1 holds at frozen $\nu$ with the pointwise $\tau_{\max}(\mathbf x):=c+\mu_2^\Xi(\nu(\mathbf x))\tilde p$, and with the dilatational shift of Proposition 11.8.

*Proof.* (1) $\mu_{\rm cs}\in[\mu_s,\mu_2)$, increasing to $\mu_2$. $\nu_c(I_c)\in(\nu_c^0-\Delta\nu_c,\nu_c^0]$ decreases to $\nu_c^0-\Delta\nu_c$, so $\Xi\ge K_\Xi(\nu_{\rm lo}-\nu_c^0)$, and the limits add. (2) Both $\mu_{\rm cs}$ and $-K_\Xi\nu_c(I_c)$ are nondecreasing. (3) The proof of Proposition 2.17.1 uses only that $c+\mu_d\tilde p$ is positive, nondecreasing in the shear rate, and tends to $\tau_{\max}$ uniformly as the shear rate grows; all three hold pointwise here. ∎

*Remark 11.6a (Still non-associated).* An associated Drucker–Prager rule would dilate at the rate $\mu_d\dot\gamma$: the cone of Conjecture A.2. Here $\Xi\to0$ at the critical state while $\mu_d\to\mu_{\rm cs}>0$, so the flow rule stays non-associated. Remark 2.17.2 still applies, and the coupled problem stays a fixed point. *Remark 11.6b (Peak and residual friction).* A dense packing has $\mu_y(\nu)>\mu_s$ and loses the excess as $\nu\to\nu_c$. This is the friction softening of the model, so A-ϖ acts on the cohesion only (Section 11.5).

## 11.4 The dilatational velocity, the Stokes problem, and the density under A-Ξ

The geometry-adapted solenoidal basis is exactly divergence-free, so it cannot carry $\dot\varepsilon_v$. The velocity is split into a particular dilatational field and a solenoidal remainder:

$$
\mathbf v=\mathbf v_\varepsilon+\mathbf v_\sigma,\qquad \mathbf v_\sigma\in\mathbb V_h^\sigma,\qquad \mathbf v_\varepsilon:=\nabla\chi_h,\quad \Delta\chi_h\approx\dot\varepsilon_v .
$$

**Proposition 11.7 (The particular field does not matter).** Let $\dot\varepsilon_v\in L^2(\mathcal V)$ on a domain satisfying (G).

1. A particular field exists: $\mathbf v^\circ=\nabla\mathcal N$, with $\mathcal N(\mathbf x)=-\frac1{4\pi}\int_{\mathcal V}\dot\varepsilon_v(\mathbf y)|\mathbf x-\mathbf y|^{-1}dV_{\mathbf y}$, lies in $H^1(\mathcal V)$ and has $\nabla\cdot\mathbf v^\circ=\dot\varepsilon_v$.
2. If $\mathbf v_1,\mathbf v_2\in H^1(\mathcal V)$ both have divergence $\dot\varepsilon_v$, then $\mathbf v_1+H^1_\sigma(\mathcal V)=\mathbf v_2+H^1_\sigma(\mathcal V)$. The set of admissible velocities, and hence the minimizer of the frozen problem below, does not depend on the choice.

*Proof.* (1) $\Delta\mathcal N=\dot\varepsilon_v\mathbf 1_{\mathcal V}$ in $\mathbb R^3$, and Plancherel gives $\|\nabla^2\mathcal N\|_{L^2(\mathbb R^3)}=\|\Delta\mathcal N\|_{L^2(\mathbb R^3)}=\|\dot\varepsilon_v\|_{L^2(\mathcal V)}$; $\nabla\mathcal N\in L^2_{\rm loc}$ by the Hardy–Littlewood–Sobolev inequality. (2) $\mathbf v_1-\mathbf v_2$ is divergence-free. ∎

A Dirichlet choice $\chi|_\Gamma=0$ is not used, because on a nonconvex polyhedron it loses $H^2$ regularity at reentrant edges.

**The discrete particular field.** Let $\mathbb Q_h^0$ be the polynomials of degree at most $d_\chi$ without the constant, written as $\chi=R_*^2\sum_ia_iq_i(\mathbf x/R_*)$ with monomials $q_i$. The even-radial pressure functions of the exact polynomial volume rule are polynomials of this kind. Every derivative that the coupling needs ($\nabla\chi$, $\nabla\nabla\chi$, $\nabla\Delta\chi$, $\nabla\nabla\nabla\chi$) is then analytic. With $\varepsilon_\chi:=\hat\varepsilon_\chi R_*^{-2}$ and a small configured $\hat\varepsilon_\chi>0$,

$$
\boxed{\ \chi_h=\arg\min_{\chi\in\mathbb Q_h^0}\ \tfrac12\|\Delta\chi-\dot\varepsilon_v\|^2_{L^2(\mathcal V)}+\tfrac{\varepsilon_\chi}2\|\nabla\chi\|^2_{L^2(\mathcal V)},\qquad
(\mathbf G_\Delta+\varepsilon_\chi\mathbf G_\nabla)\,\mathbf x_\chi=\mathbf b_\chi ,\ }
$$

with $(\mathbf G_\Delta)_{ab}=\int_{\mathcal V}\Delta q_a\Delta q_b$, $(\mathbf G_\nabla)_{ab}=\int_{\mathcal V}\nabla q_a\cdot\nabla q_b$, and $(\mathbf b_\chi)_a=\int_{\mathcal V}\dot\varepsilon_v\Delta q_a$.

- $\mathbf G_\nabla$ is SPD on $\mathbb Q_h^0$ ($\nabla q=\mathbf 0$ only for constants on the connected $\mathcal V$), so $\chi_h$ is unique.
- As $\hat\varepsilon_\chi\to0$, $\chi_h$ tends to the least-squares solution of smallest $\|\nabla\chi\|$.
- Both matrices are $\eta$-free polynomial integrals, assembled on the volume rule (exact by the exact polynomial volume rule when that rule is replaced by the surface formula) and factored once per geometry. The right-hand side uses the volume nodes.

The *realized dilatation* is $\dot\varepsilon_{v,h}:=\Delta\chi_h$, and the projection defect $e_\varepsilon:=\|\Delta\chi_h-\dot\varepsilon_v\|_{L^2}/\max\{\|\dot\varepsilon_v\|_{L^2},\varepsilon_e\}$ is reported. A localized $\dot\varepsilon_v$ is resolved only to the degree of $\mathbb Q_h$.

**The frozen problem.** Freeze $\tilde p$, the state $(\nu,\varpi)$, and $\dot\varepsilon_v$, hence $\mathbf v_\varepsilon$. With $\mathbf D':=\mathbf D-\tfrac13(\operatorname{tr}\mathbf D)\mathbf I$ and $\dot\gamma:=\sqrt{2\mathbf D':\mathbf D'}$,

$$
\min_{\mathbf v_\sigma\in\mathbb V_h^\sigma}\ J_\varepsilon(\mathbf v_\sigma):=\int_{\mathcal V}\phi\big(\dot\gamma(\mathbf v_\varepsilon+\mathbf v_\sigma);\mathbf x\big)\,dV-\ell(\mathbf v_\sigma)
\quad\text{s.t.}\quad \mathbf B_{\rm rb}\mathbf u=-\mathbf b_\varepsilon,\qquad
\mathbf b_\varepsilon:=\Big(\int_{\mathcal V}\rho_K\mathbf v_\varepsilon\,dV,\ \int_{\mathcal V}\rho_K\,\mathbf x\times\mathbf v_\varepsilon\,dV\Big),
$$

with $\mathbf u$ the coefficients of $\mathbf v_\sigma$, and $\phi$ and $\ell$ those of Proposition 2.17.1. The rigid rows now act on the total velocity, so the gauge of (B1) is kept.

**Proposition 11.8 (Variational structure with dilatancy).**

1. $J_\varepsilon$ is strictly convex on the affine set $\mathbb K_\varepsilon:=\{\mathbf u:\mathbf B_{\rm rb}\mathbf u=-\mathbf b_\varepsilon\}$, which is nonempty.
2. With $L^*$ of Proposition 2.17.1 (pointwise $\tau_{\max}$ of Proposition 11.6), $L^*<1$ gives a unique minimizer, and $L^*>1$ gives none.
3. The minimizer satisfies $\int_{\mathcal V}2\eta\,\mathbf D'(\mathbf v):\mathbf D(\mathbf w)\,dV+\boldsymbol\lambda^{\!\top}\mathbf B_{\rm rb}\mathbf w=\ell(\mathbf w)$ for all $\mathbf w\in\mathbb V_h^\sigma$. This is the Galerkin momentum equation tested with solenoidal functions, so $p$ drops out exactly as in the frozen-coefficient momentum balance.
4. With $\dot\varepsilon_v\equiv0$, $\chi_h=0$, and the problem is that of Proposition 2.17.1.

*Proof.* (1) $\mathbf D(\mathbf v_\sigma)$ is traceless, so $\dot\gamma$ is the Euclidean norm of the affine map $\mathbf v_\sigma\mapsto\sqrt2(\mathbf D'(\mathbf v_\varepsilon)+\mathbf D(\mathbf v_\sigma))$. Composed with the convex, strictly increasing $\phi$, it is convex. Strictness follows as in Proposition 2.17.1(2): two points of $\mathbb K_\varepsilon$ differ by a nonzero element of $\ker\mathbf B_{\rm rb}$, whose strain is nonzero on a set of positive measure (Lemma 8.12). $\operatorname{rank}\mathbf B_{\rm rb}=6$ makes $\mathbb K_\varepsilon$ nonempty. (2) $\phi(|\mathbf a+t\mathbf b|)/t\to\tau_{\max}|\mathbf b|$ as $t\to\infty$ for fixed $\mathbf a$. So the fixed shift leaves the recession function, and with it the argument of Proposition 2.17.1(3), unchanged. (3) The derivative of $\phi(\dot\gamma)$ in the direction $\mathbf w$ is $2\eta\mathbf D'(\mathbf v):\mathbf D'(\mathbf w)$, and $\mathbf D'(\mathbf w)=\mathbf D(\mathbf w)$ for solenoidal $\mathbf w$. (4) $\mathbf b_\chi=\mathbf 0$ gives $\mathbf x_\chi=\mathbf 0$. ∎

**Surface shear rate (changes to Section 10.1.1).** $\operatorname{tr}\mathbf D=\dot\varepsilon_{v,h}$ gives

$$
D_{nn}=\dot\varepsilon_{v,h}-g^{ab}D_{ab},\qquad
\dot\gamma_\Gamma^2=2\Big[(\dot\varepsilon_{v,h}-g^{ab}D_{ab})^2+g^{ac}g^{bd}D_{ab}D_{cd}+2g^{ab}D_{na}D_{nb}-\tfrac13\dot\varepsilon_{v,h}^2\Big].
$$

The traction condition is unchanged, since $D'_{na}=D_{na}$.

**Geometry and density.** The predictor (B4) uses the slip velocity of the total $\mathbf v$, so dilatation raises the movable surface. The density is advanced as follows.

- *Level 0 (Section 9.2.3).* $\mathbf L_k:=\operatorname{dev}\nabla\mathbf v_{\rm adv}(\mathbf c_k^{\rm mid})+\tfrac13\bar\varepsilon_k\mathbf I$, with $\bar\varepsilon_k:=-\ln J_{\nu,k}/\tau_n$ and $J_{\nu,k}$ from Proposition 11.5 at $\mathbf c_k^{\rm mid}$ (with $\mathcal R\nu$, $\dot\gamma_p^\star$, and $I_c$ there). Then $\mathbf F_k=\exp(\tau_n\mathbf L_k)$, $\det\mathbf F_k=1/J_{\nu,k}$, $w_k'=J_{\nu,k}w_k$, $\mathbf S_k'=\mathbf F_k^{-\top}\mathbf S_k\mathbf F_k^{-1}$, and $\det\mathbf S_k'=J_{\nu,k}^2\det\mathbf S_k$. For an affine flow with uniform dilatation this is exact, since then $\rho'(\mathbf x)=\rho(\Phi^{-1}\mathbf x)/\det\mathbf F$. Anchored cores are not updated; their deficit enters the Level 2 target.
- *The matrix deficit.* $\rho_b$ is not a material component, so Level 0 does not carry the deficit $-\rho_b(1-J_\nu)$. With $J_\nu(\mathbf x)$ the ratio of Proposition 11.5 for the material point that arrives at $\mathbf x$, the candidate density of the cone program (Section 2.13) is
$$
\rho_{\rm cand}:=\rho_K(\cdot;\theta^{\rm pred})-\rho_b\big(1-J_\nu\big),\qquad(D\mathcal M)_i=\rho_{\rm cand}(\mathbf X_i)|\mathbf A_i| .
$$
In the mass integral the $\rho_b$ part is exact (the exact polynomial volume rule) and the cores use RTFP. The deficit term uses (G2), and is supported where $\dot\gamma_p^\star>0$.
- *Level 1 (replaces Section 9.2.4 under A-Ξ).* With $\rho_c:=\sum_k\rho_k$, solve the mass-constrained two-scalar fit
$$
(s_b,s_c)=\arg\min\sum_q\omega_q\big(s_b\rho_b+s_c\rho_c(\mathbf x_q;\theta')-\tilde\rho_q\big)^2\quad\text{s.t.}\quad s_b\rho_b|\mathcal V^{(n+1)}|+s_c\,\mathcal M_c(\mathcal V^{(n+1)},\theta')=\mathsf M_{\rm tot},
$$
a $3\times3$ KKT system, then set $\rho_b\leftarrow s_b\rho_b$ and $w_k\leftarrow s_cw_k$. If $\rho_b=0$, the core mass is below the threshold of Section 9.2.4, or the two fields are numerically dependent on the nodes, the closure falls back to one scalar on the other part. The mass is exact in every case. The fallback and $(s_b,s_c)$ are reported. Under A-Ξ, $\rho_b$ therefore changes; without A-Ξ, Section 9.2.4 applies unchanged.
- *Level 2 target (Section 9.2.5).* $\tilde\rho_q:=J_\nu(\mathbf x_q)\,\rho_K(\mathbf X_d(\mathbf x_q);\theta^{(n)})$: the exact transported density times the ratio of the arriving material point. The misfit $e_\rho$ and Proposition 9.2.1 apply unchanged. The non-amplification bound becomes
$$
\|e^N\|_{L^2}\le\Big(\frac{\nu_{\rm hi}}{\nu_{\rm lo}}\Big)^{1/2}\Big(\|e^0\|_{L^2}+\sum_{n<N}\delta_n\Big),
$$
because the exact transport $\rho\mapsto J_\nu\cdot\rho\circ\Phi^{-1}$ satisfies $\|\mathcal P_ne\|^2_{L^2}=\int J_\nu e^2\,dV$. Composed along a trajectory, the factors multiply to $\nu_{\rm end}/\nu_{\rm start}\le\nu_{\rm hi}/\nu_{\rm lo}$, with the same idealizations as Section 9.2.5.
- *Level 3 spawn (optional).* After Level 2, if $e_\rho>\varepsilon_\rho$ and $K<K_{\max}$:
  1. let $r_q:=\tilde\rho_q-\rho_K(\mathbf x_q;\theta)$, and let $\mathcal C$ be the component, on the adjacency of $\mathcal X_Y$, of $\{|r_q|\ge\tfrac12\max|r|\}$ that contains the maximum;
  2. add a core with configured exponent $\alpha_{\rm new}$, centre $\mathbf c=\sum_{\mathcal C}\omega_q|r_q|\mathbf x_q/\sum_{\mathcal C}\omega_q|r_q|$, metric $\mathbf S=s_\alpha(\mathbf C+h_\Gamma^2\mathbf I)^{-1}$ with $\mathbf C$ the $|r|$-weighted covariance of $\mathcal C$ and a configured $s_\alpha$, and signed weight $w$ equal to $r_q$ at the node of $\mathcal C$ nearest to $\mathbf c$; Section 1.1 allows negative weights;
  3. apply Level 2 with $K+1$ cores, with the acceptance rule of Section 9.2.6.

  A spawned core is a material component like any other.

**Mass consistency.** The kinematic dilatation ($\Delta\chi_h$, masked by $\chi_Y$ in $\mathbf v_{\rm adv}$) and the density ratio $J_\nu$ need not give the same volume change. The first is a Galerkin projection, the second is exact per material point. The mass row of the cone program, with $\rho_{\rm cand}$, decides the volume, so the mass is exact to the cone tolerance. The difference appears in $s_M$ and in the mean normal correction, and both are reported, together with the kinematic volume change $\tau_n\int\dot\varepsilon_{v,h}$ and the density-implied one $-\int\ln J_\nu$.

**Step bound.**

$$
\Delta t_\Xi:=\frac{\theta_\Xi}{K_\Xi\,\nu_{\rm hi}\max_{\mathbf x}\dot\gamma_p^\star(\mathbf x)}\quad(+\infty\text{ if nothing yields}),\qquad\theta_\Xi\le1 .
$$

Along the exact solution $(\nu-\nu_c)$ decays as $\exp(-K_\Xi\int\nu\,d\gamma)$, so over a step the relative change of $\Xi$, and hence of $\dot\varepsilon_v$, is at most $1-e^{-\theta_\Xi}\le\theta_\Xi$. The frozen kinematic dilatation $\tau_n\dot\varepsilon_v$ and the exact $-\ln J_\nu$ then agree to first order in $\theta_\Xi$.

## 11.5 Damage of the cohesion (hypothesis A-ϖ)

The cohesion of a rubble pile comes from fine-grain bonds, and shear breaks them. A-C2 fixes the cohesion profile. A-ϖ lets it fall with the plastic strain the material has undergone.

**Hypothesis A-ϖ.**

1. *Material damage.* $\varpi\in[0,1]$ is carried by the markers, with initial field $\varpi^{(0)}$ and deposit value $\varpi_{\rm dep}$ (default $1$: a deposit carries no bonds).
2. *Cohesion only.*
$$
\boxed{\ c(\mathbf x)=\big(1-\mathcal R\varpi(\mathbf x)\big)\,c_{\rm intact}(\mathbf x),\ }
$$
with $c_{\rm intact}$ the intact profile of A-C2 (Section 4.3a). The friction is not damaged; its softening is that of A-Ξ (Remark 11.6b).
3. *Evolution.*
$$
\boxed{\ \dot\varpi=(1-\varpi)\,\frac{\dot\gamma_p^\star}{\gamma_\varpi}-h_\varpi\,\varpi,\ }
$$
with $\gamma_\varpi>0$ the plastic strain scale of bond breakage and $h_\varpi\ge0$ a healing rate (default $0$).

**Proposition 11.9 (Exact step, bounds, and mechanics under A-ϖ).** Freeze $a:=\dot\gamma_p^\star/\gamma_\varpi$ and $h_\varpi$ at a material point over $\tau_n$. If $a+h_\varpi>0$,

$$
\varpi'=\varpi_\infty+(\varpi-\varpi_\infty)\,e^{-(a+h_\varpi)\tau_n},\qquad\varpi_\infty:=\frac a{a+h_\varpi},
$$

and $\varpi'=\varpi$ otherwise. Then:

1. $\varpi'\in[0,1]$, between $\varpi$ and $\varpi_\infty$;
2. $0\le c\le\max(c_s,c_b)$. The viscosity bounds of Section 2.8 hold with $c\ge0$, $\eta_{\rm lo}=\mu_{\min}p_{\min}/\sqrt{\dot\gamma_{\max}^2+\dot\gamma_\varepsilon^2}$ ($\mu_{\min}=\mu_s$ without A-Ξ). So Lemma 8.12 and Propositions 2.17.1 and 11.8 hold at frozen $\varpi$;
3. at the boundary quadrature points $\nabla c=(1-\mathcal R\varpi)\nabla c_{\rm intact}-c_{\rm intact}\nabla\mathcal R\varpi\in L^\infty$, which is what Section 8.3 needs.

*Proof.* (1) The equation is linear with constant coefficients over the step, and $\varpi_\infty\in[0,1]$. The solution is a convex combination of $\varpi$ and $\varpi_\infty$ with weight $e^{-(a+h_\varpi)\tau_n}\in(0,1]$. (2) $0\le1-\mathcal R\varpi\le1$ (Lemma 11.2) and $0\le c_{\rm intact}\le\max(c_s,c_b)$. $\eta_{\rm lo}$ uses only $c\ge0$ and $\mu\ge\mu_{\min}$. (3) Product rule with Lemma 11.2. ∎

**Remarks.**

- *Why not $(1-\varpi)\eta$.* Scaling the whole viscosity would scale the frictional strength to zero as $\varpi\to1$. That violates $\eta_{\rm lo}>0$ (Lemma 8.12), and broken bonds do not remove grain friction.
- *Relation to A-C2.* With $c_s=c_b$, a weak surface layer can emerge from $\varpi$, through deposits and shallow shear; A-C2 remains the intact profile.
- *Relation to (B6).* $\bar\varepsilon_p$ stays a diagnostic. $\varpi$ is driven by the same plastic rate, made nonlocal, and it does feed back. It is the only feedback of plastic strain in the model. It is a softening internal variable of the viscoplastic law, not the hardening variable of an elastoplastic return map; the model still has no elastic strain.
- *Localization.* Softening with plastic strain localizes; the nonlocal driving of Section 11.2 and PO-24 apply.

**Step bound.** The stress solve sees $c$ only at the step start, and Proposition 11.9 integrates $\varpi$ exactly with the rates frozen. Over a step with large rates, $\varpi$, and with it $c/c_{\rm intact}$, can therefore change by up to $1$ without the stress solve ever seeing the weaker material. Let

$$
\Delta t_\varpi:=\frac{\theta_\varpi}{\max_{\mathbf x\in\mathcal X_{\rm vol}}\Big|\big(1-\mathcal R\varpi\big)\,\dot\gamma_p^\star/\gamma_\varpi-h_\varpi\,\mathcal R\varpi\Big|}\quad(+\infty\text{ if the denominator vanishes}),\qquad\theta_\varpi\le1 .
$$

By Proposition 11.9, $\varpi'-\varpi=(\varpi_\infty-\varpi)\big(1-e^{-(a+h_\varpi)\tau_n}\big)$, so

$$
|\varpi'-\varpi|\le|\varpi_\infty-\varpi|\,(a+h_\varpi)\,\tau_n=\big|(1-\varpi)a-h_\varpi\varpi\big|\,\tau_n ,
$$

and with the rates frozen, $\varpi$ and $c/c_{\rm intact}$ change by at most $\theta_\varpi$ in one step. The bound uses the signed rate, not $a+h_\varpi$. Material that is fully damaged ($\varpi\to1$) or at the healing steady state ($\varpi=\varpi_\infty$) does not hold the step down; only material that is still softening does. The bound resolves the timing of the softening. It does not prevent the softening. As with $\Delta t_\Xi$, the rates are evaluated on $\mathcal X_{\rm vol}$ at the step start, while the markers are updated at their midpoints.

## 11.6 Thermal cycling and fatigue of the surface (hypothesis A-Θ)

Diurnal thermal cycling breaks exposed rock on small bodies faster than micrometeoroid impacts and is a main source of fine regolith [21]. On Ryugu, a thermal inertia of about 200–300 J m⁻² K⁻¹ s⁻¹ᐟ² [25] gives a diurnal skin depth of order centimetres, several orders of magnitude below the continuum resolution $h_{\rm vol}$ of any practical truncation. A three-dimensional heat equation coupled to the creep would resolve none of it. In a model without an elastic modulus a thermal stress is not even defined. An isotropic thermal stress added to the incompressible balance would enter as a gradient, which the pressure absorbs apart from its effect on $\tilde p$. Thermal cycling is therefore a subgrid closure of the surface skin.

**Hypothesis A-Θ.**

1. *Scale separation.* The diurnal skin depth $\delta_1:=\sqrt{2\kappa_{\rm th}/\omega}$ is small compared with every face and with $h_{\rm vol}$. Each face is a one-dimensional half-space with constant thermal effusivity $\Upsilon=\sqrt{k_{\rm th}(\rho c)_{\rm th}}$ and diffusivity $\kappa_{\rm th}=\Upsilon^2/(\rho c)_{\rm th}^2$. Lateral conduction is neglected.
2. *Quasi-steady diurnal state.* At each orbital sample the surface temperature is the time-periodic diurnal solution ($P_{\rm rot}\ll P_{\rm orb}$), with zero mean conductive flux; seasonal heat storage is neglected. $P_{\rm rot}/P_{\rm orb}$ and the seasonal skin depth are reported.
3. *Radiation.* Absorption with Bond albedo $A_B$, gray emission with emissivity $\epsilon_{\rm th}$, and direct insolation with geometric self-shadowing; for a satellite, also the eclipse by its primary and the primary's light and heat (below). Mutual radiative exchange (self-heating) and multiple scattering are neglected. The occluded fraction of each face's sky hemisphere is reported as the indicator of that error.
4. *Fatigue.* Rock in the skin fails after $N_f(\Delta T)=N_\Theta(\Delta T_\Theta/\Delta T)^{m_\Theta}$ cycles of diurnal temperature range $\Delta T$ (Basquin), since the thermal stress amplitude in exposed rock scales with $\Delta T$ [21]. Damage adds linearly over cycles (Miner), at $\omega/2\pi$ cycles per unit time.
5. *Shielding.* A loose cover of thickness $h_{\rm reg}:=\rho_s/\rho_{\rm bulk}$ damps the diurnal wave as $e^{-z/\delta_1^{\rm reg}}$, with $\delta_1^{\rm reg}$ from the cover's own properties. The fatigue rate of the rock beneath is multiplied by
$$
S_\Theta(h_{\rm reg}):=\exp\big(-m_\Theta h_{\rm reg}/\delta_1^{\rm reg}\big).
$$
The surface temperature itself is computed with one effective $\Upsilon$.
6. *Scope of the effects.* A-Θ changes only the shallow subgrid layer, through two outputs. The surface damage $\varpi_\Gamma$ at the vertices weakens the shallow cohesion of A-C4 and Section 10.2. The production $J_\Theta$ of loose material adds to $\rho_s$. The continuum stress, the yield classification, and the Mode C fluxes are not changed.
7. *Exogenous orbit and pole.* The heliocentric orbit and the inertial direction $\hat{\mathbf p}$ of the spin axis are given as an orbit-and-pole schedule. Only $\omega$ may evolve (A-L).

**Insolation.** Take $N_{\rm orb}$ orbital samples $j$ at equally spaced mean anomalies, so that orbit averages are uniform in time and the periodic trapezoidal rule is spectrally accurate. Each sample has heliocentric distance $r_{\odot,j}$, solar flux $F_{\odot,j}=F_{\odot}(1\,\mathrm{AU})\,(1\,\mathrm{AU}/r_{\odot,j})^2$, and subsolar latitude $\delta_{\odot,j}=\arcsin(\hat{\mathbf p}\cdot\hat{\mathbf s}_j^{\rm in})$, with $\hat{\mathbf s}_j^{\rm in}$ the inertial direction to the Sun. In the body frame, at solar hour angle $\varphi_\odot=\omega t$, the Sun is at $\hat{\mathbf s}_j(\varphi_\odot)=(\cos\delta_{\odot,j}\cos\varphi_\odot,\ -\cos\delta_{\odot,j}\sin\varphi_\odot,\ \sin\delta_{\odot,j})$. For a body spinning in the positive sense about $\mathbf e_z$, the Sun moves toward decreasing body longitude. This sense fixes the sign of the thermal lag, and with it the sign of the radiative torque below. The origin of $\varphi_\odot$ is irrelevant to rotation averages. With the pole at obliquity $\epsilon_{\rm ob}$ and longitude $\lambda_p$ (measured from perihelion in the orbit plane), $\delta_{\odot,j}=-\arcsin\big(\sin\epsilon_{\rm ob}\cos(f_j-\lambda_p)\big)$ at true anomaly $f_j$. The absorbed flux of face $f$ is

$$
S_{fj}(\varphi_\odot)=(1-A_B)\,F_{\odot,j}\,[\mathbf n_f\cdot\hat{\mathbf s}_j]_+\,\mathsf V_f(\hat{\mathbf s}_j),
$$

with $\mathsf V_f\in[0,1]$ the unshadowed fraction, from first-hit queries toward $\hat{\mathbf s}_j$ from $N_{\rm vis}$ points of the face offset along $\mathbf n_f$.

**A satellite's primary.** Under the orbital form of Section 10.3.1 the primary, of radius $R_p$, lies on the body $+x$ axis at the distance $a$, and the satellite's solar day is its orbital period. The hour angle $\varphi_\odot=0$ puts the Sun behind the primary. Two terms enter.

*Eclipse.* The Sun's disk, of angular radius $\alpha_\odot=R_\odot/r_{\odot,j}$, is partly hidden by the primary's, of angular radius $\alpha_p=\arcsin(R_p/a)$, at the angular separation $\vartheta=\arccos(\hat{\mathbf s}_j\cdot\mathbf e_x)$. The visible fraction is

$$
\mathsf E(\vartheta)=1-\frac{\mathcal A(\alpha_\odot,\alpha_p,\vartheta)}{\pi\alpha_\odot^2},
$$

with $\mathcal A$ the overlap area of two disks of these radii at this separation, taken in the tangent plane of the Sun's direction. The error is $O(\alpha_\odot\alpha_p)$ relative to the penumbra width. $\mathsf E$ multiplies the direct insolation of every face, since $R_s\ll a$. There is no eclipse while the subsolar latitude exceeds $\alpha_p+\alpha_\odot$.

*Light and heat of the primary.* The primary is a sphere of uniform effective temperature $T_p$ and Bond albedo $A_p$ (for Mars both are tabulated in [43]). A face of normal $\mathbf n_f$ whose horizon does not cut that sphere receives its thermal emission exactly as

$$
\sigma_{\rm SB}T_p^4\Big(\frac{R_p}{a}\Big)^2[\mathbf n_f\cdot\mathbf e_x]_+
$$

(the view factor of a sphere). It also receives the sunlight reflected by a Lambert sphere,

$$
\tfrac23A_pF_{\odot,j}\Big(\frac{R_p}{a}\Big)^2\Phi(\alpha)\,[\mathbf n_f\cdot\mathbf e_x]_+,\qquad
\Phi(\alpha)=\frac{\sin\alpha+(\pi-\alpha)\cos\alpha}{\pi},\qquad \cos\alpha=-\hat{\mathbf s}_j\cdot\mathbf e_x,
$$

in the point-source form, with relative error $O((R_p/a)^2)$ for the reflected part. Both are absorbed with the face's $(1-A_B)$ and add to $S_{fj}$. Self-shadowing of the primary's light and the faces whose horizon cuts the primary are neglected, and the fraction of such faces is reported. The pole of a synchronous satellite is the primary's, so the schedule of item 7 is the primary's heliocentric orbit and obliquity.

**The periodic surface problem.** Find a $2\pi$-periodic $T(\varphi_\odot)$ with

$$
\boxed{\ \epsilon_{\rm th}\sigma_{\rm SB}\,T|T|^3+\mathcal Q_\omega T=S_{fj},\qquad
\widehat{(\mathcal Q_\omega T)}_k=\Upsilon\sqrt{|k|\omega/2}\,(1+i\operatorname{sgn}k)\,\hat T_k .\ }
$$

$\mathcal Q_\omega$ maps the surface temperature to the conductive flux into the ground. $T|T|^3$ equals $T^4$ for $T\ge0$, and it makes the problem monotone.

**Proposition 11.10 (Diurnal surface temperature).**

1. For a $2\pi$-periodic $T\in H^{1/2}$, the bounded time-periodic solution of $\partial_t\theta=\kappa_{\rm th}\partial_{zz}\theta$ in $z>0$ with $\theta(0,t)=T(\omega t)$ is unique, and its flux into the ground is $-k_{\rm th}\partial_z\theta(0,t)=(\mathcal Q_\omega T)(\omega t)$.
2. For real $T$, $\frac1{2\pi}\int_0^{2\pi}(\mathcal Q_\omega T)\,T\,d\varphi_\odot=\Upsilon\sum_k\sqrt{|k|\omega/2}\,|\hat T_k|^2\ge0$, and $\mathcal Q_\omega T$ has zero mean.
3. The periodic surface problem has at most one solution. A solution with $S\ge0$ satisfies $T\ge0$ and the mean balance $\epsilon_{\rm th}\sigma_{\rm SB}\langle T^4\rangle=\langle S\rangle$.
4. *Discrete problem.* On $N_\varphi$ equispaced phases, with $\mathbf Q$ the circulant matrix of the symbol (real part only at the Nyquist index), $\mathbf G(\mathbf T):=\epsilon_{\rm th}\sigma_{\rm SB}\mathbf T\circ|\mathbf T|^3+\mathbf Q\mathbf T-\mathbf S$ is continuous, strictly monotone, and coercive on $\mathbb R^{N_\varphi}$, so $\mathbf G(\mathbf T)=\mathbf 0$ has exactly one solution. The Jacobian $\operatorname{diag}(4\epsilon_{\rm th}\sigma_{\rm SB}|\mathbf T|^3)+\mathbf Q$ has a positive semidefinite symmetric part. That part is positive definite where no component of $\mathbf T$ vanishes, and there the damped Newton direction is a descent direction of $\tfrac12|\mathbf G|^2$.

*Proof.* (1) In Fourier series in time, $\theta_k(z)=\hat T_ke^{-\lambda_kz}$ with $\lambda_k^2=ik\omega/\kappa_{\rm th}$. Boundedness selects $\lambda_k=\sqrt{|k|\omega/(2\kappa_{\rm th})}\,(1+i\operatorname{sgn}k)$, and $k=0$ gives a constant. The flux is $k_{\rm th}\lambda_k\hat T_k$, and $k_{\rm th}/\sqrt{\kappa_{\rm th}}=\Upsilon$. (2) For real $T$, $\hat T_{-k}=\overline{\hat T_k}$. The imaginary part of the symbol is odd in $k$ and cancels in the sum, and the $k=0$ symbol vanishes. (3) For two solutions, subtract and pair with $T_1-T_2$. Both terms are nonnegative, and the first, from the strictly increasing $t\mapsto t|t|^3$, is positive unless $T_1=T_2$ a.e. For positivity: the bounded periodic solution is the periodized convolution of $T$ with the half-line Dirichlet heat kernel $z(4\pi\kappa_{\rm th}s^3)^{-1/2}e^{-z^2/(4\kappa_{\rm th}s)}$. That kernel is nonnegative with unit integral in $s$, so $\theta(z,t)\ge\min T$. At a time of minimal surface temperature, $\partial_z\theta(0,\cdot)\ge0$, so the flux into the ground is $\le0$ and $\epsilon_{\rm th}\sigma_{\rm SB}T|T|^3=S-\mathcal Q_\omega T\ge0$ there. Hence $\min T\ge0$. The mean balance is the $k=0$ component. (4) $\mathbf T\mapsto\mathbf T\circ|\mathbf T|^3$ is strictly monotone, and $\mathbf Q$ is monotone by (2). A continuous, strictly monotone, coercive map of $\mathbb R^N$ is a bijection (Minty–Browder). The symmetric part of $\mathbf Q$ has the real symbol $\Upsilon\sqrt{|k|\omega/2}\ge0$. ∎

The discrete solution is not guaranteed positive. $\min T$ is reported, and $N_\varphi$ is refined if it is negative.

**Fatigue rate, surface damage, and production.** With $\Delta T_{fj}:=\max_{\varphi_\odot}T_{fj}-\min_{\varphi_\odot}T_{fj}$ over the samples,

$$
r_{\Theta,f}:=\frac{\omega}{2\pi N_\Theta}\,\frac1{N_{\rm orb}}\sum_j\Big(\frac{\Delta T_{fj}}{\Delta T_\Theta}\Big)^{m_\Theta},\qquad
r_{\Theta,i}:=\frac{\sum_{f\ni i}A_fr_{\Theta,f}}{\sum_{f\ni i}A_f},
$$

and at each vertex, with $\ell_\Theta$ the fatigue-active depth,

$$
\boxed{\ \dot\varpi_\Gamma=(1-\varpi_\Gamma)\,r_\Theta\,S_\Theta(h_{\rm reg})-h_\varpi\varpi_\Gamma,\qquad
\dot\rho_s\big|_\Theta=J_\Theta:=\rho_{\rm bulk}\,\ell_\Theta\,r_\Theta\,S_\Theta(\rho_s/\rho_{\rm bulk}).\ }
$$

**Proposition 11.11 (Exact thermal sources).** Freeze $r_\Theta$ over $\tau_n$, and write $\delta_r:=\delta_1^{\rm reg}$ and $h:=h_{\rm reg}$.

1. $\varpi_\Gamma$ follows Proposition 11.9 with $a=r_\Theta S_\Theta(h)$, where $h$ is the cover at the start of the step (first-order split), so $\varpi_\Gamma'\in[0,1]$.
2. The production integrates exactly to
$$
h'=\frac{\delta_r}{m_\Theta}\ln\Big(e^{m_\Theta h/\delta_r}+\frac{m_\Theta\,\ell_\Theta r_\Theta\tau_n}{\delta_r}\Big),\qquad\rho_s'=\rho_{\rm bulk}h',
$$
and $h\le h'\le h+\ell_\Theta r_\Theta\tau_n$: production is never negative, and it never exceeds the unshielded production.
3. *Ledger.* Production moves mass from the rock to the $\rho_s$ sub-ledger inside the same volume. $\Gamma$, $\mathsf M_{\rm tot}$, and $\rho_K$ are unchanged. The loose-mass bound of A-C4 (Section 2.8c, item 3) then admits relief into the produced layer.
4. $r_\Theta\equiv0$ gives $\rho_s'=\rho_s$ and $\varpi_\Gamma'=\varpi_\Gamma e^{-h_\varpi\tau_n}$.

*Proof.* (2) $dh/dt=\ell_\Theta r_\Theta e^{-m_\Theta h/\delta_r}$ separates to $d(e^{m_\Theta h/\delta_r})/dt=m_\Theta\ell_\Theta r_\Theta/\delta_r$. For $x,y\ge0$, $\ln(e^x+y)\le x+ye^{-x}$. With $x=m_\Theta h/\delta_r$, $y=m_\Theta\ell_\Theta r_\Theta\tau_n/\delta_r$, and the factor $\delta_r/m_\Theta$, this gives $h'\le h+\ell_\Theta r_\Theta\tau_ne^{-x}\le h+\ell_\Theta r_\Theta\tau_n$; the lower bound holds because $y\ge0$. (3) $\rho_s$ is an accounting density inside $\mathcal V$ (Section 2.8). (1) and (4) are immediate. ∎

**Shallow cohesion.** With $c_s$ the surface value of A-C2,

$$
\boxed{\ c_{\Gamma,i}:=\big(1-\mathcal R\varpi(\mathbf X_i)\big)\big(1-\varpi_{\Gamma,i}\big)\,c_s,\ }
$$

with face values taken as the mean over the face's vertices. $c_\Gamma$ replaces $c_s$ in the Mohr–Coulomb balance and the slack of A-C4 (Section 2.8c) and in $\mu_{\rm eff}$ of Section 10.2. The two damage mechanisms are independent, so their surviving bond fractions multiply. The continuum keeps $c=(1-\mathcal R\varpi)c_{\rm intact}$ up to $\Gamma$: the skin that $\varpi_\Gamma$ describes is thinner than the continuum resolution by assumption 1. Where the rock surface slips ($\chi_Y=1$), $\varpi_\Gamma$ moves with it: it is transported semi-Lagrangian along $\chi_Y(\mathbf v_{\rm sl}-(\mathbf v_{\rm sl}\cdot\mathbf n)\mathbf n)$ as in (G5), without the Jacobian correction, since it is an intensive quantity. Elsewhere it stays on its vertex.

**Radiative torque (input to A-L, law L2).** From the same solution, with the incident momentum, Lambertian reflection, and Lambertian thermal emission [22],

$$
T_\parallel^{\rm rad}:=\frac1{N_{\rm orb}}\sum_j\frac1{2\pi}\int_0^{2\pi}\sum_f\mathbf e_z\cdot\big(\mathbf x_f\times\mathbf f_{fj}(\varphi_\odot)\big)\,d\varphi_\odot,
$$

$$
\mathbf f_{fj}=-\frac{A_fF_{\odot,j}[\mathbf n_f\cdot\hat{\mathbf s}_j]_+\mathsf V_f}{c_{\rm light}}\Big(\hat{\mathbf s}_j+\tfrac23A_B\mathbf n_f\Big)-\frac23\,\frac{A_f\epsilon_{\rm th}\sigma_{\rm SB}T_{fj}^4}{c_{\rm light}}\,\mathbf n_f ,
$$

with $\mathbf x_f$ the face centre relative to the centre of mass and $c_{\rm light}$ the speed of light. The same neglect of self-heating and multiple scattering applies. The torque is known to be very sensitive to unresolved topography [23], which is why law L2 calibrates it (Section 11.7).

**Evaluation, reuse, and step bound.** A full evaluation costs $N_FN_{\rm orb}$ times ($N_\varphi N_{\rm vis}$ shadow rays plus $N_{\rm Newton}$ Krylov solves of $O(N_\varphi\log N_\varphi)$ each, with $\mathbf Q$ applied by FFT). A face is re-evaluated when its normal changes, when one of its sun rays is dirty in the sense of Proposition 3.15.2 (the shadow rays are cached), or when $\omega$ or the orbit-and-pole schedule changes. Reuse within a configured tolerance on $|\Delta\omega|/\omega$ is an approximation, and that tolerance is reported. The step bound

$$
\Delta t_\Theta:=\frac{\theta_\Theta}{\max_i\big(r_{\Theta,i}S_{\Theta,i}+h_\varpi\big)}
$$

limits the change of $\varpi_\Gamma$ in one step to $\theta_\Theta$, since $|\varpi_\Gamma'-\varpi_\Gamma|\le(a+h_\varpi)\tau_n$. The settling of step S sees $c_\Gamma$ only at step ends, so this bound resolves the timing of delayed failure.

## 11.7 Spin from angular momentum (hypothesis A-L)

Under the exogenous table $\omega$ does not respond to the mass redistribution it causes. Material moving toward the equator raises $I_1$, which at fixed angular momentum lowers $\omega$ and the centrifugal load. That negative feedback matters near the critical spin [5][10].

**Hypothesis A-L.**

1. *Tisserand frame.* The rotating frame is the one in which the creep velocity has zero momentum and zero angular momentum: $\int\rho_K\mathbf v=\mathbf 0$ and $\int\rho_K\mathbf x\times\mathbf v=\mathbf 0$, which are the rigid rows of (B1). The angular momentum about the centre of mass is therefore $\mathbf L=\mathbf I\boldsymbol\Omega$.
2. *Axial torque law.* The component $T_\parallel$ of the rotation-averaged external torque along the spin axis is given by law L1 or L2 below. Torque perpendicular to the axis (precession, change of obliquity) is not modelled; the pole direction is exogenous, as in A-Θ item 7.
3. *Principal-axis rotation (A-S retained).* The body spins about its axis of maximum inertia $\mathbf a_1$.
   - *During the slow operator* this rests on adiabatic invariance: $\mathbf a_1$ must move slowly compared with the free-wobble frequency $\omega_w:=\omega\sqrt{(I_1-I_2)(I_1-I_3)/(I_2I_3)}$. The indicator $\varepsilon_{{\rm ad},n}:=\angle\big(\mathbf a_1^{(n)},\mathbf a_1^{(n),\rm creep}\big)/(\tau_n\omega_w)$ is reported, where the second axis is that after the creep update and before settling.
   - *After the instantaneous settling of step S,* any wobble it excites is assumed damped within the step [24]. The misalignment and the energy to be dissipated are reported (Proposition 11.12).

**State and law.** The scalar $L_n\ge0$ is initialized after the mandatory initial realignment as $L_0=I_1^{(0)}\omega_0$. Then

$$
\boxed{\ \omega_n=\frac{L_n}{I_1^{(n)}},\qquad L_{n+1}=L_n+\tau_n\,T_{\parallel,n},\ }
$$

with $I_1^{(n+1)}$ the largest principal moment about the centre of mass of $\rho_K$ on the settled $\mathcal V^{(n+1)}$, after step S and the core closure. $L_{n+1}\le0$ is outside the model and ends the run (*model not closed*). The table bound $\Delta t_\omega$ of Section 10.3.1 becomes

$$
\Delta t_\omega:=\varepsilon_\omega\,\frac{\max\{\omega_n,\omega_{\rm ref}\}\,I_1^{(n)}}{\max\{|T_{\parallel,n}|,\epsilon_T\}} .
$$

**Torque laws.**

- **L1, table.** $T_\parallel(t)$ is given in the forms of the spin table (stages, linear, table), together with $\omega_0$. A measured spin change $\dot\omega_{\rm obs}$ of a body with principal moment $I_{\rm obs}$ corresponds to $T_\parallel=I_{\rm obs}\dot\omega_{\rm obs}$, up to the inertia change during the measurement interval.
- **L2, radiative.** $T_\parallel=\kappa_YT_\parallel^{\rm rad}$, with $T_\parallel^{\rm rad}$ from Section 11.6 on the current shape at $\omega_n$. The factor is fixed once, at a declared calibration state (default step 0): $\kappa_Y:=T_\parallel^{\rm cal}/T_\parallel^{\rm rad}({\rm cal})$, with $T_\parallel^{\rm cal}$ the configured torque of that state. If $|T_\parallel^{\rm rad}({\rm cal})|$ is below a configured fraction of $|T_\parallel^{\rm cal}|$, the model is not closed. L2 assumes that the ratio of the true torque to the resolved-shape torque stays constant along the evolution, i.e. that the unresolved topography to which the torque is sensitive [23] is statistically unchanged. The model does not test this. L2 needs the thermal solution of A-Θ, but not its fatigue law.
- **Synchronous orbit schedule.** Under the orbital form of Section 10.3.1 the spin is tidally locked, $\omega=n$, and A-L is not applied.

Under A-L the stage form of the table becomes a sequence of torque stages $\{(T_{\parallel,j},T_j)\}$.

**Proposition 11.12 (Spin at conserved angular momentum).**

1. For a given $\mathbf L$ and SPD inertia $\mathbf I$ with largest eigenvalue $I_1$ and eigenvector $\mathbf a_1$, the rotational energy $E=\tfrac12\mathbf L^{\!\top}\mathbf I^{-1}\mathbf L$ satisfies $E\ge|\mathbf L|^2/(2I_1)$, with equality iff $\mathbf L\parallel\mathbf a_1$. Internal dissipation at constant $\mathbf L$ therefore ends in rotation about $\mathbf a_1$ at $\omega=|\mathbf L|/I_1$.
2. A sudden change $\mathbf I\to\mathbf I'$ at constant $\mathbf L$ (step S) gives $E'=\tfrac12\mathbf L^{\!\top}\mathbf I'^{-1}\mathbf L$. Damping to the state of (1) dissipates $\Delta E=E'-|\mathbf L|^2/(2I_1')\ge0$ and turns the body through $\vartheta_L:=\angle(\mathbf L,\mathbf a_1')$, where $\mathbf L$ lies along the spin axis $\mathbf e_z$ of the step. In the body frame this reorientation is exactly the realignment of step 8b, $\mathbf e_z'=\mathbf a_1'$ (true polar wander).
3. *Reduction.*
   - If $T_\parallel=I_1\dot\omega_{\rm ext}$ with $I_1$ constant and $\dot\omega_{\rm ext}$ constant on each step (true for the linear and table forms, whose nodes are step boundaries through $t_{\rm next}$), then $\omega_n=\omega(t_n)$ of the exogenous schedule with $\omega(0)=\omega_0$.
   - $T_\parallel\equiv0$ with constant $\mathbf I$ gives constant $\omega$.
4. *Frame residual.* After realignment $|I_{xz}|,|I_{yz}|\le\varepsilon_{\rm frame}I_1$, so $|\mathbf L|=\omega(I_{zz}^2+I_{xz}^2+I_{yz}^2)^{1/2}$, and $I_1-I_{zz}=O(\varepsilon_{\rm frame}^2I_1/\hat g)$ with the relative gap $\hat g:=(I_1-I_2)/I_1$. So $\omega=L/I_1$ agrees with the spin of the body frame to $O(\varepsilon_{\rm frame}^2/\hat g)$. For $\hat g\le\varepsilon_{\rm deg}$ the axis is not determined (step 8b(iv)) and $\omega_w\to0$, which the indicator reports.

*Proof.* (1) The eigenvalues of $\mathbf I^{-1}$ are $1/I_k\ge1/I_1$, so $\mathbf L^{\!\top}\mathbf I^{-1}\mathbf L\ge|\mathbf L|^2/I_1$, with equality iff $\mathbf L$ lies in the eigenspace of $1/I_1$. Without external torque, dissipation conserves $\mathbf L$ and lowers $E$, and the minimum is attained only there. (2) The definition, with $\Delta E\ge0$ by (1). (3) $L_{n+1}=L_n+\tau_nI_1\dot\omega_{\rm ext}$; dividing by the constant $I_1$ gives $\omega_{n+1}=\omega_n+\tau_n\dot\omega_{\rm ext}$, which is exact for a rate constant on the step. (4) Second-order eigenvalue perturbation of $\mathbf I$ in its off-diagonal entries. ∎

*The Euler force.* (B1) omits the Euler force together with its driving torque. Under A-L, $\varepsilon_{\rm Euler}$ is evaluated with the slow part of the change, $|L_{n+1}/I_1^{(n),\rm creep}-\omega_n|/(\tau_n\omega_n^2)$. The jump of $\omega$ at settling is instantaneous by A-T1 and lies outside the quasi-static balance; it is reported through $\vartheta_L$ and $\Delta E$. The rigid rate $\boldsymbol\theta_c$ of the core relative to the Tisserand frame (B3) is reported as $|\boldsymbol\theta_c|/\omega$.

## 11.8 Reduction and the stage fixed point

**Theorem 11.13 (Exact reduction).** Suppose:

- (a) $K_\Xi=0$ (A-Ξ off);
- (b) $\varpi^{(0)}\equiv0$, $\varpi_{\rm dep}=0$, and $\gamma_\varpi=\infty$ (A-ϖ off);
- (c) $r_\Theta\equiv0$ and $\varpi_\Gamma^{(0)}\equiv0$ (A-Θ off, $N_\Theta=\infty$);
- (d) the spin table of Section 10.3.1 (A-L off).

Then every quantity of the loop of Sections 10.3.4 and 11.9 equals, step by step, that of Section 10.3.4 without this chapter. Each switch may be turned off alone; the other extensions then run with the reference closure in that respect.

*Proof.*

- (a) $\Xi\equiv0$ gives $\dot\varepsilon_v\equiv0$, so $\mathbf b_\chi=\mathbf 0$ and, the normal equations being SPD, $\chi_h=0$. Hence $\mathbf v_\varepsilon=\mathbf 0$ and $\mathbf b_\varepsilon=\mathbf 0$, which leaves Section 8.3 and Section 10.3.2 unchanged. Also $\mu_d=\mu_{\rm cs}$ and $\mu_y=\mu_s$. $J_\nu\equiv1$ by Proposition 11.5(4), which leaves Level 0, $\rho_{\rm cand}$, and the Level 2 target unchanged. Level 1 is that of Section 9.2.4.
- (b) Proposition 11.9 with $a=0$ and $\varpi=0$ keeps $\varpi\equiv0$, so $c=c_{\rm intact}$.
- (c) $\varpi_\Gamma\equiv0$ and $J_\Theta=0$ (Proposition 11.11(4)), so $c_\Gamma=(1-\mathcal R\varpi)c_s$ and $\rho_s$ has no new source.
- (d) By definition.

The markers then carry constant states that enter nothing, and $\Delta t_\Xi=\Delta t_\varpi=\Delta t_\Theta=\infty$. ∎

**Proposition 11.14 (Stage fixed point).** Let a state satisfy 2.15(c): $\chi_Y\equiv0$ on $\mathcal X_Y$, $J_{\rm ero}\equiv0$, an inactive settling, and (b).

1. Then $\dot\gamma_p^\star\equiv0$. So $\dot\varepsilon_v\equiv0$, $\mathbf v_\varepsilon=\mathbf 0$, $\nu'=\nu$, and $\dot\varpi=-h_\varpi\varpi$. Also $\mathbf v_{\rm adv}\equiv\mathbf 0$, so the markers do not move.
2. The state is a fixed point of the extended loop iff in addition:
   - $h_\varpi\varpi\equiv0$ on the markers and $h_\varpi\varpi_\Gamma\equiv0$;
   - $r_\Theta S_\Theta\equiv0$;
   - under A-L, $T_\parallel\equiv0$.

Otherwise the slow state evolves at rest. That delayed weakening, or change of spin, is physical, and the stage end is not jumped to. The steps are then bounded by $\Delta t_\Theta$, $\Delta t_\omega$, and $\Delta t_{\max}$, and 2.15(c) is applied at each step as a test, not as a shortcut.

*Proof.* (1) $\chi_Y\equiv0$ on $\mathcal X_Y$ gives $F_\delta\le0$ on $\mathcal X_Y$, so the mask of Definition 11.3 vanishes and $\dot\gamma_p^\star\equiv0$. The rest follows from Propositions 11.5 and 11.9, and from Section 9.2.2 for $\mathbf v_{\rm adv}$. (2) What remains are the listed rates. ∎

## 11.9 Placement in the loop and cost

The additions to Section 10.3.4, step by step (normative; a bracketed label means "when that hypothesis is on"):

```text
0+. [A-Ξ, A-ϖ] Seed markers on X_vol of V^(0) with nu^(0), varpi^(0); check the admissibility of
    Section 11.3 (mu_min > 0, nu_hi below random close packing).
    [A-Θ] varpi_Γ^(0), the orbit-and-pole schedule, the thermal and fatigue constants.
    [A-L] the torque law; L_0 = I_1 omega_0 after the initial realignment 8b; kappa_Y by calibration (L2).
    Any missing item -> model not closed.
1+. omega_n = L_n / I_1^(n) [A-L], else omega(t_n).
    Reconstruct nu and varpi (Section 11.1) at X_vol, on X_Y, at the vertices, and at the boundary
    quadrature points.
    [A-Θ or L2] thermal solve on the dirty faces (Section 11.6): S_fj, T_fj, Delta T_fj, r_Θ, T_par^rad.
2+. Hyperbolic characteristic ray relaxation (Sections 10.3.2, 11.4) with c = (1 - R varpi) c_{\rm intact} and mu_d(I_c, nu).
    Every N_p iterations, together with p~: F_delta on X_Y, gd_p, gd_p* (Section 11.2),
    eps_v = Xi gd_p*, chi_h, e^eps, b_eps.
3.  Classification as before, on the total velocity.
4+. Friction chain with mu_d(I_c, nu) and the shallow cohesion c_Γ (Section 11.6).
6+. Delta t_n is also bounded by Delta t_Xi, by Delta t_varpi, by Delta t_Θ, and, under A-L, by the
    Delta t_omega of Section 11.7.
6b+. Level 0 with J_nu,k (Section 11.4).
7+. Cone program with rho_cand = rho_K(.; theta^pred) - rho_b (1 - J_nu).
8a+. Markers: RK2 transport; state update by Propositions 11.5 and 11.9 at the midpoints.
     [A-Θ] varpi_Γ and the production J_Θ over tau_n (Proposition 11.11); transport of varpi_Γ on
     slipping vertices.
8c+. [A-Ξ] Level 1 by the two-scalar closure; Level 2 with the target J_nu rho_K(X_d); optional spawn.
8b+. Markers: y_j <- Q (y_j - x_cm).
S+.  Settling with c_Γ; no state of this chapter evolves.
S7.  Marker maintenance on the settled geometry (Section 11.1).
9+. [A-L] I_1^(n+1) on the settled shape; L_{n+1} = L_n + tau_n T_par,n;
    omega_{n+1} = L_{n+1} / I_1^(n+1); report epsilon_ad, theta_L, Delta E.
```

Under A-L the frame realignment keeps $\omega$ as $L/I_1$ instead of leaving it unchanged (step 8b(v)), and the stage shortcut of step 6 requires the conditions of Proposition 11.14(2).

**Cost per step (planning estimates).**

| Item | Cost |
|---|---|
| marker transport and state update | $N_{\rm mk}$ point syntheses of $\mathbf v$ and $\nabla\mathbf v$, as Level 0 with $K\to N_{\rm mk}$ |
| reconstruction | $N_{\rm eval}\,n_{\rm nb}$ neighbour queries and $4\times4$ solves |
| nonlocal rate | one sparse product, with $n_\ell$ entries per volume node, per pressure update; the visibility pattern is built once per geometry |
| $\chi_h$ | $\mathbf G_\Delta$, $\mathbf G_\nabla$ exact once per geometry (the exact polynomial volume rule) and factored; per pressure update one right-hand side $O(N_{\rm vol}\dim\mathbb Q_h)$, two triangular solves, and the Hessian of $\chi_h$ at the nodes |
| thermal | $N_FN_{\rm orb}$ times ($N_\varphi N_{\rm vis}$ shadow rays plus $N_{\rm Newton}$ FFT-based Krylov solves), on dirty faces only |
| spin | $O(1)$; $\mathbf I$ comes from step 8b |

The velocity problem keeps its unknowns, because $\mathbf v_\varepsilon$ is a fixed shift, and its step condition, because the characteristic CFL condition of Section 10.3.2 is unchanged. The largest addition is the nonlocal rate when $\ell_p$ spans many nodes.

## 11.10 Reports

Each step, for the hypotheses that are on:

- **A-Ξ:**
  - the range of $\nu$ and $\max|\Xi|$;
  - $e_\varepsilon$, and the kinematic against the density-implied volume change;
  - $\Delta t_\Xi$ and $(s_b,s_c)$;
  - spawned cores;
  - $\max|\nu/\nu_0-1|$, the error of the frozen contact prefactors.
- **A-ϖ:** the largest and mean $\varpi$ over the yielded set and over the body; $\min c/c_{\rm intact}$; $\Delta t_\varpi$.
- **Markers:** the count and the insertions, deletions, merges, and Shepard fallbacks.
- **Nonlocal rate:** $\ell_p/h_{\rm vol}$.
- **A-Θ:**
  - the largest $\Delta T$ and the mean and largest $r_\Theta$;
  - the produced mass and the largest $\varpi_\Gamma$;
  - $\min T$ (discrete positivity);
  - $\delta_1$ over the smallest face size;
  - the occluded sky fraction and $P_{\rm rot}/P_{\rm orb}$.
- **A-L:**
  - $\omega_n$, $L_n$, and $T_\parallel$;
  - $\varepsilon_{\rm ad}$, $\omega_w$, and $\hat g$;
  - $\vartheta_L$ and $\Delta E$;
  - $|\boldsymbol\theta_c|/\omega$.

---

# 12. Breakup, fragments, and the debris ring

Proposition 2.17.3 and Section 10.3.6 end the quasi-static route at a certified $L^*>1$. This chapter follows the body past that point:

- it breaks along its collapse mechanism (Sections 12.1–12.2);
- the fragments move as rigid bodies about the primary and about one another on the orbital time scale (Section 12.3);
- they merge when they meet and break again when their own limit load is exceeded (Section 12.4);
- pieces below a strength scale join a ring that spreads viscously between the primary and the Roche limit (Section 12.5).

The quasi-static loop is not used in this phase. The minutes in which blocks separate along their bands are replaced by Hypothesis A-B.

**Hypothesis A-B (breakup and its aftermath).**

1. *Rigid blocks.* A body whose certified $L^*$ exceeds 1 (strength $c+\mu_2\tilde p$ at the rest pressure, Proposition 2.17.3) breaks into the blocks of the piecewise-rigid fit of its collapse mechanism (Section 12.1). The material of a shear band goes to the block whose motion fits it best.
2. *Quasi-static onset.* The blocks start with the parent's rigid motion: its orbit and its spin. The mechanism gives only the direction in which they separate; the tide and the self-gravity then drive them.
3. *Rigid fragments.* Between breakups a fragment is a rigid body of frozen shape and density.
4. *Sticky contact.* Fragments that touch while approaching merge perfectly inelastically.
5. *Dispersal.* A fragment whose certified collapse is not piecewise rigid (one block) is a granular flow, and it disperses into the ring. So does a fragment smaller than the debris diameter $D_{\rm debris}$.
6. *Passive ring.* The ring is axisymmetric and does not act on the fragments. Matter that spreads past the Roche limit, or falls on the primary, leaves it; moonlets that form from it are a ledger only, with no migration and no reaccretion.

A-B does not return to the quasi-static loop: a fragment that stands is followed as a rigid body to the end of the dynamic phase.

## 12.1 The cut: a piecewise-rigid fit of the mechanism

The kinematic bound of Proposition 2.17.3 comes with a velocity field $\mathbf u_0\in\ker\mathbf B$: the collapse mechanism. In limit analysis a mechanism is made of rigid blocks separated by velocity discontinuities. The Galerkin field smooths the discontinuities into bands, but it keeps the blocks.

**Definition 12.1 (Piecewise-rigid fit).** Let $\mathbf v_i$ be the mechanism velocity at the volume nodes $\mathbf x_i$ with masses $m_i=w_i\rho_K(\mathbf x_i)$. For a labelling $\beta:i\mapsto\{1,\dots,K\}$, block $b$ has the weighted least-squares rigid motion $\mathbf v_b(\mathbf x)=\mathbf a_b+\boldsymbol\theta_b\times(\mathbf x-\mathbf c_b)$, where $\mathbf c_b$ is its centre of mass. The misfit is

$$
\mathcal E_K=\frac{\sum_im_i|\mathbf v_i-\mathbf v_{\beta(i)}(\mathbf x_i)|^2}{\sum_im_i|\mathbf v_i-\bar{\mathbf v}|^2}.
$$

The fit alternates two steps until the labels no longer change:

- assign each node to the motion that fits it best;
- refit each block.

After each assignment every block is reduced to its largest connected component on the graph of the ten nearest neighbours of each node, and the other components join the block they touch most. A block below a fraction $f_{\min}$ of the mass is not admitted.

Block $K+1$ is seeded with the nodes nearest to the worst-fitted node. It is kept if $\mathcal E_{K+1}\le(1-\gamma)\mathcal E_K$, so each further block must lower the misfit by the fraction $\gamma$; at most $K_{\max}$ blocks are formed.

**Proposition 12.1.** If $\mathbf v$ is exactly piecewise rigid on $K$ connected blocks with distinct motions, then the labelling by blocks has $\mathcal E_K=0$, and it is a fixed point of the alternation.

*Proof.* Each block's own motion fits its nodes exactly, so every node is assigned to its own block, and the refit reproduces the motions. ∎

## 12.2 The pieces

Each block becomes a body that satisfies hypothesis (G).

**Voxel labels.** On a grid of spacing $h_v$ over the parent, each voxel centre inside $V$ takes the label of the nearest volume node. Each label is then reduced to its largest face-connected set of voxels; smaller islands join the label they touch most, or leave the body if they touch none.

**Surface.** Marching tetrahedra on the indicator of a label extract its surface. Each cube of voxel centres is cut into the six tetrahedra along its main diagonal, and each tetrahedron is cut at the midpoints of its edges whose ends differ. Neighbouring cubes cut their common faces alike, and the indicator is never $\tfrac12$ at a vertex. The surface is therefore a closed, oriented 2-manifold.

**Body.** The extracted closed 2-manifold surface directly forms the piece's boundary without radial spherical projection, strictly preserving non-star-shaped geometry, concavities, and narrow necks. To eliminate sliver triangles produced by marching tetrahedra while preserving the manifold topology, the surface undergoes topology-preserving isotropic remeshing (alternating edge collapse, edge split, and surface-projected tangential relaxation) until the mean edge length matches the parent's. The resulting non-star-shaped polyhedron directly satisfies hypothesis (G) of Section 8.1: its internal gravity and multi-crossing chord intervals are resolved via its own SurfaceBVH and CellBVH (Section 3.15 and Section 8.3). Each Cauchy kernel goes to the piece whose label holds its centre.

**Mass.** The bulk density of a piece is set so that its mass equals the parent density integrated over its voxels, with the block masses scaled to the parent's total. The relative error of the pieces' total mass is reported. Each piece is then moved to its principal frame (Section 8.1, G8).

## 12.3 Dynamics of the fragments

**State.** Each fragment $k$ carries:

- its centre of mass $\mathbf x_k$ and velocity $\mathbf v_k$ in a frame centred on the primary;
- the rotation $\mathbf R_k$ from its principal frame;
- its spin angular momentum $\mathbf L_k$ about its centre of mass;
- its mass $m_k$ and principal moments $\mathbf I_k$.

Its angular velocity is $\boldsymbol\omega_k=\mathbf R_k\mathbf I_k^{-1}\mathbf R_k^{\!\top}\mathbf L_k$.

**The primary.** It is a point mass $GM_p$ with zonal harmonic $J_2$ about $\mathbf e_z$:

$$
\Phi_p=-\frac{GM_p}{r}\Big[1-J_2\Big(\frac{R_p}{r}\Big)^2P_2\Big(\frac zr\Big)\Big].
$$

Its coupling to a fragment's own quadrupole adds the potential $-GM_p\,Q_k(\mathbf x_k)$, with

$$
Q(\mathbf r)=\frac{\operatorname{tr}\mathbf I-3\hat{\mathbf r}^{\!\top}\mathbf I\hat{\mathbf r}}{2r^3},
$$

and the gravity-gradient torque $3GM_p\,\boldsymbol\rho\times\mathbf I\boldsymbol\rho/\rho^5$, with $\boldsymbol\rho=-\mathbf x_k$ the vector from the fragment to the primary. Here $\mathbf I$ is the fragment's inertia tensor in space.

**Pairs far apart.** A pair farther apart than three times the sum of the bounding radii interacts through MacCullagh's expansion [29]:

$$
V_{jk}=-\frac{Gm_jm_k}{r}-Gm_kQ_j(\mathbf r)-Gm_jQ_k(\mathbf r),\qquad\mathbf r=\mathbf x_k-\mathbf x_j .
$$

It acts by the forces $\mp\nabla_{\mathbf r}V_{jk}$ and by the gravity-gradient torques of each body in the other's monopole. Its relative error is $O((R/r)^2)$.

**Pairs close together.** A closer pair interacts through the RTFP field $\mathbf g_h$ of the heavier body, evaluated at the volume nodes $\mathbf y_i$ of the lighter. The force $m_i\mathbf g_h(\mathbf y_i)$ acts at each node, and its reaction acts on the heavier body at the same point:

$$
\mathbf F=\sum_im_i\mathbf g_h(\mathbf y_i),\qquad
\boldsymbol\tau_{\rm light}=\sum_i(\mathbf y_i-\mathbf x_{\rm light})\times m_i\mathbf g_h(\mathbf y_i),\qquad
\boldsymbol\tau_{\rm heavy}=-\sum_i(\mathbf y_i-\mathbf x_{\rm heavy})\times m_i\mathbf g_h(\mathbf y_i).
$$

**Proposition 12.2 (Conservation by the pair forces).** Both pair forms conserve momentum and angular momentum exactly: the forces are opposite, and $\boldsymbol\tau_j+\boldsymbol\tau_k+\mathbf r\times\mathbf F_k=\mathbf 0$. With $J_2=0$ the coupling to the primary conserves the total angular momentum about it; with $J_2\ne0$ it conserves the component along $\mathbf e_z$.

*Proof.*
- *Close pairs.* Summing the torques gives $\sum_i(\mathbf x_{\rm heavy}-\mathbf x_{\rm light})\times\mathbf F_i=-(\mathbf x_{\rm light}-\mathbf x_{\rm heavy})\times\mathbf F$, which cancels the orbital torque of the pair.
- *Far pairs.* $V_{jk}$ is invariant under a common rotation of $\mathbf r$ and both inertia tensors. Its generator gives exactly the sum of the gravity-gradient torques and the orbital torque.
- *The primary.* The same invariance holds for $-GM_pQ_k(\mathbf x_k)$ and the point mass. $\Phi_p$ is invariant under rotations about $\mathbf e_z$. ∎

**Integrator.** The step is kick–drift–kick:

1. A half kick $\mathbf v\mathrel{+}=\tfrac{\tau}{2}\mathbf F/m$, $\mathbf L\mathrel{+}=\tfrac{\tau}{2}\boldsymbol\tau$.
2. A drift $\mathbf x\mathrel{+}=\tau\mathbf v$ together with the free rigid body over $\tau$.
3. A half kick with the new forces.

The free rigid body is split into its principal-axis flows, $x(\tfrac\tau2)\,y(\tfrac\tau2)\,z(\tau)\,y(\tfrac\tau2)\,x(\tfrac\tau2)$. Each flow is an exact rotation about one body axis, by the angle $\pi_a\tau/I_a$ with $\boldsymbol\pi=\mathbf R^{\!\top}\mathbf L$ [31].

**Proposition 12.3.** The step is symplectic and second order. It conserves the total angular momentum to rounding whenever the forces conserve it (Proposition 12.2).

*Proof.* The kicks change the total by $\tfrac\tau2\big(\sum\mathbf x\times\mathbf F+\sum\boldsymbol\tau\big)=\mathbf 0$. The drift keeps $\mathbf x\times\mathbf v$, because $\mathbf x$ moves along $\mathbf v$, and it keeps every $\mathbf L_k$. Each split flow is the exact flow of a Hamiltonian, and the symmetric composition is second order. ∎

**Step size.** The step is the smallest of:
- an orbital period divided by a fixed number of substeps;
- one fiftieth of the mutual free-fall time $\sqrt{r^3/G(m_j+m_k)}$ of every close pair;
- one fiftieth of every spin period.

## 12.4 Mergers and further breakups

**Contact.** Two fragments touch when a vertex of one lies inside the other. They merge if they touch while approaching, $(\mathbf v_k-\mathbf v_j)\cdot(\mathbf x_k-\mathbf x_j)<0$. Pieces of one cut are born touching along their cut; they merge only after they have once come apart.

**Merger.**
- *Geometry.* The merged body is the union of the two solids, voxelized and extracted as in Section 12.2, with both sets of kernels.
- *Density and mass.* Its bulk density is the mass-weighted mean of the two, adjusted so that its mass is $m_j+m_k$.
- *Motion.* Its centre of mass and velocity are the parts' mass-weighted means. Its spin angular momentum is
$$
\mathbf L=\mathbf L_j+\mathbf L_k+m_j(\mathbf x_j-\mathbf x)\times(\mathbf v_j-\mathbf v)+m_k(\mathbf x_k-\mathbf x)\times(\mathbf v_k-\mathbf v).
$$
  Mass, momentum and angular momentum are therefore conserved exactly; the kinetic energy that the merger dissipates is reported.

**Cascade.** At fixed intervals each fragment is checked as follows.

- *Impact.* It hits the primary when its periapsis lies below $R_p$.
- *Escape.* It leaves the primary when its orbital energy is non-negative.
- *Debris.* It joins the ring at its semi-major axis when its volume-equivalent diameter is below $D_{\rm debris}$. The scale is set by the strength: a body of size $D$ in the tide $n^2$ carries stresses of order $\tfrac38\rho n^2D^2$, which cohesion $c$ holds only below $D_c=\sqrt{8c/(3\rho n^2)}$. Friction alone does not stop the cascade. The tidal load and the frictional strength, through the lithostatic pressure, both scale with $D^2$, so $L^*$ of similar shapes does not depend on size, and a piece of a body that failed by friction fails again until cohesion takes over near $D_c$. The number of pieces then grows as $(D/D_c)^3$. A $D_{\rm debris}$ above $D_c$ stops resolving the cascade at that size: pieces smaller than it are taken to disintegrate and join the ring unresolved.
- *Limit load.* Every other fragment is bracketed, but only when its tide $n^2=GM_p/|\mathbf x|^3$ has grown by more than 5 % since its last check. The load is that of a synchronous body with its long axis toward the primary, the orientation in which the tide stretches it most.
  - If $L_{\rm lo}>1$, it is cut (Section 12.1). If the cut finds a single block, it disperses into the ring (A-B item 5).
  - The pieces start with the parent's rigid motion: at the arm $\mathbf d$ of a piece, $\mathbf v+\boldsymbol\omega\times\mathbf d$, with the parent's angular velocity $\boldsymbol\omega$.

## 12.5 The debris ring

The ring is the surface density $\Sigma(R,t)$ of an axisymmetric disk of particles of radius $r_p$ and density $\rho_p$ about the primary. It occupies the annulus from $R_p$ to the fluid Roche limit $R_R=2.456\,R_p(\rho_{\rm primary}/\rho_p)^{1/3}$. Viscous spreading obeys [33]

$$
\frac{\partial\Sigma}{\partial t}=\frac3R\frac{\partial}{\partial R}\Big[\sqrt R\,\frac{\partial}{\partial R}\big(\nu\Sigma\sqrt R\big)\Big].
$$

**Viscosity.** The viscosity of a dense ring with self-gravity wakes is [32][33]

$$
\nu=\nu_{\rm trans}+\nu_{\rm coll}+\nu_{\rm grav},\qquad
\nu_{\rm coll}=r_p^2\Omega\tau,\qquad
\nu_{\rm trans}+\nu_{\rm grav}=\begin{cases}\dfrac{\sigma_r^2}{2\Omega}\dfrac{0.46\,\tau}{1+\tau^2},&Q>2,\\[2mm] 26\,r_h^{*5}\,\dfrac{G^2\Sigma^2}{\Omega^3},&Q<2,\end{cases}
$$

with:
- the optical depth $\tau=3\Sigma/(4r_p\rho_p)$;
- the Hill parameter $r_h^*=(2m_p/3M_p)^{1/3}R/(2r_p)$;
- the velocity dispersion $\sigma_r=\sqrt{Gm_p/r_p}$ for $r_h^*>\tfrac12$ and $2r_p\Omega$ below;
- the Toomre parameter $Q=\Omega\sigma_r/(3.36\,G\Sigma)$.

**Discretization.** Cells of width $\Delta R$ hold the masses $m_i=2\pi R_i\Delta R\,\Sigma_i$. The inward mass flux at a face is

$$
\dot M=6\pi\sqrt R\,\frac{\partial}{\partial R}\big(\nu\Sigma\sqrt R\big),
$$

differenced across the face, with an empty cell beyond each edge. A cell gains the flux through its outer face and loses that through its inner face. The flux through the inner edge falls on the primary, and the flux through the Roche limit leaves to form moonlets [34].

Deposits follow the same rule: one inside $R_p$ goes to the primary, one beyond $R_R$ to the moonlets, and any other joins its cell.

**Step.** The diffusivity of $\nu\Sigma$ is at most $3\nu(1+\partial\ln\nu/\partial\ln\Sigma)\le9\nu$ in the wake regime. The explicit step $0.2\,\Delta R^2/(9\nu_{\max})$ therefore keeps the update stable.

**Proposition 12.4 (Ledger).** The update conserves exactly the ring's mass plus the masses fallen on the primary and gone to the moonlets.

*Proof.* Each interior face flux leaves one cell and enters its neighbour, and the edge fluxes are booked to the two sinks. The sum telescopes. ∎

**Reports.**
- *Cut.* The misfit of each accepted number of blocks, and the pieces' mass error.
- *Fragments.* At recorded times: their masses, diameters, orbital elements and spins.
- *Events.* Cuts, mergers (with the dissipated energy), dispersals, impacts, escapes and debris.
- *Ledgers.* The ring, the primary and the moonlets.
- *Conservation.* The total momentum, angular momentum and energy. The energy uses the quadrupole form also for close pairs, so it is a diagnostic there.

---

# 13. Conclusion

The paper gives a closed loop whose core is the Mode B quasi-static viscoplastic Stokes equilibrium, with Mode C erosion and deposition as an auxiliary process. It couples:

- the gravity of a Cauchy density;
- conservative surface transport;
- an incremental SOCP.

**Gravity.** Exterior fields are cross-checked by the Feynman and Newtonian representations. Interior and boundary Hessians are defined by explicit finite-part and contact conventions.

**Rheology.** There is one stress, $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D$. At each step, under the prescribed spin (or the spin of A-L), the non-star-shaped hyperbolic ray system (8.3-H) is solved on the current BVH geometry; under A-Ξ its source is $g=\dot\varepsilon_v$. Pressure is advanced by the artificial-compressibility equation in (8.3-H) and is accepted only after the divergence and traction residuals pass. The frozen-coefficient rheology is advanced by the non-star-shaped hyperbolic characteristic ray solver of Sections 8.3 and 10.3.2. Its stationary residual is checked against the original quasi-static balance; the geometric correction remains the incremental SOCP. On the same stress, $F_\delta$ and connectivity split the body into a rigid core and a movable set. The core gets no creep displacement, and only yielded and sliding regions produce a prediction, from the slip velocity relative to the core.

**Surface.** Slope promotion (A-C4) relieves the loose layer toward repose. Where the truncated hyperbolic ray solve yields no surface node, it is the only mechanism that moves the surface, because shallow failure lies below that solve's resolution (Section 2.8, B2). The cone program corrects only predicted normal increments, relative to the current shape, never toward an equipotential. The step is limited by the CFL of the true landslide speed and by the erosional flux, with no viscous stiffness.

**Evolving state (Chapter 11).** Four optional hypotheses add history to the model, each with an exact reduction to the closure above (Theorem 11.13).

- *A-Ξ* lets sheared material dilate or contract toward a critical solid fraction. The velocity gains a fixed dilatational field, the friction the stress–dilatancy term, and the cores the dilatational ratio. The frozen problem stays strictly convex (Proposition 11.8).
- *A-ϖ* weakens the cohesion with nonlocal plastic strain.
- *A-Θ* closes the diurnal thermal fatigue of the surface skin: an exact periodic half-space problem per face, a Basquin–Miner damage of the shallow cohesion, and a shielded production of loose material.
- *A-L* evolves the spin from the angular momentum and the settled inertia, with principal-axis rotation kept by adiabaticity and dissipation.

The state laws are integrated exactly over each step and preserve their admissible ranges.

**The tidal cycle (Section 10.3.7).** An eccentric orbit and the forced libration it drives load a synchronous satellite once per orbit. The cyclic stress comes from an incompressible elastic response whose stress does not depend on the modulus. It gives a peak limit load and a shakedown limit, bracketed by the theorems of Koiter and Melan. The steady cycle by phase stepping gives a ratchet velocity that scales as load over modulus. Where the cycle stays elastic, an S–N law with an endurance limit damages the bonds. At constant $Q$ both tides damp the eccentricity, as a power of the semi-major axis, so the cycle weakens as the orbit decays.

**Breakup (Chapter 12).** Past a certified limit load, the body breaks into the rigid blocks of its collapse mechanism. The fragments move about the primary under forces that conserve momentum and angular momentum exactly. They merge when they meet and break again when their own limit load is exceeded, and pieces below the strength scale spread as a viscous ring between the primary and the Roche limit.

**Discretization and geometry.**

- Mode A is a marked interface only.
- The density obeys the material transport of incompressibility, under A-Ξ times the exact dilatational ratio: each core is transported exactly to second order, then mass-closed and adaptively refitted with a misfit $e_\rho$ that bounds the gravity defect.
- The geometry is a general polyhedron of any genus, possibly with cavities and with the centre of mass outside the body (Section 8.1). The velocity basis is built on the actual polyhedral domain, volume integrals follow ray segments, the pressure datum is imposed by Nitsche's method, and the surface moves by normal increments with clearance rows. Star-shaped formulas are a special case only.
- Segment-table reuse, incremental retracing, and the matrix-free hyperbolic ray operator reduce the cost without changing any value (Sections 3.15, 8.3, and 10.3.2).

Rigour rests on:

- positive pressure and integrability;
- the finite-part convention;
- the constitutive regularization;
- under Chapter 11, the nonlocal length $\ell_p$, the marker reconstruction, and the stated subgrid hypotheses of A-Θ and A-L.


# Appendix A. Normative conventions and the register of proof obligations (single source of truth)

This appendix records the remaining proof obligations and notation for the general non-star-shaped route; Chapters 1–4 are its inputs. "Must" and "must not" are mandatory. Every conclusion called a "theorem" must name the proof obligation (A.5) it depends on. Anything unproved may be called only a "hypothesis" or a "numerical observation".

## A.1 Domains, traces, and orientations

All volume integrals use the Lipschitz polyhedral domain $\mathcal V$ of (G); cavity boundaries carry the outward normal of the solid. Surface traces are one-sided interior traces, and equalities on edges or vertices are understood in the almost-everywhere or quadrature sense.

## A.2 Discrete spaces and rank tests

A retained basis is accepted only after the mass and constraint blocks pass the stated singular-value tests. Refinement changes the discrete space and therefore requires the corresponding residual and conditioning reports.

## A.3 Compatibility and rigid modes

The six rigid translations and rotations are constrained explicitly. The load must satisfy the six rigid-mode compatibility rows to the configured tolerance; failure terminates the step as an incompatible load, rather than being hidden in a pressure datum.

## A.4 Error reporting

Every step reports geometric, mass, hyperbolic residual, ray/runout, density-fit, and conservation indicators named in the relevant chapter. A value outside its tolerance is recorded as a failed obligation and does not silently become a theorem.

## A.5 Register of proof obligations (PO)

| Number | Content | Status |
|---|---|---|
| PO-1 | Local solvability and convergence of the hyperbolic ray relaxation (8.3-H), including the frozen-viscosity stationary limit and the outer constitutive refresh | frozen positive-coefficient system is defined; nonlinear outer convergence is not proved |
| PO-9 | Consistency and stability, as $\Delta t_n\to0$, of the geometric update (an RTFP–Stokes–SOCP sequence explicit in the geometry); a bound on $E_{\rm proj}$ | not proved (depends on the CFL, the under-relaxation, and the trust region) |
| PO-10 | Short-wave ill-posedness of $\mu(I)$ outside an intermediate range of the inertial number (small and large $I$ [8]) (the short-wave symbol analysis, with the pressure coupling $\mathbb K_p$), and its localization and mesh dependence under $\dot\gamma_\varepsilon$ and $p_{\min}$. Softening under A-Ξ and A-ϖ adds a further localization, treated by the nonlocal rate of Section 11.2 (PO-24). Compressibility is known to regularize parts of the ill-posed range of $\mu(I)$ [8]; whether A-Ξ does so here is open | not proved |
| PO-11 | Convergence of the hyperbolic ray relaxation and its non-star-shaped BVH discretization | judged by the hyperbolic residuals of Section 2.15(a); nonlinear outer convergence is not proved |
| PO-12 | Convergence, as $\dot\gamma_\varepsilon\to0$, of the regularized yield surface and of the core and movable set to the yielded region and sliding blocks of the unregularized model; sufficiency of the step rule of Section 10.3.1; finite-step attainment of stage equilibrium at constant $\omega_j$ (under Chapter 11, only when the slow state is at rest, Proposition 11.14) | not proved (otherwise report "stage not equilibrated", criterion (c) of Section 2.15) |
| PO-13 | The frame realignment. On the general route it is an exact rigid motion of vertices and cores, so $E_{\rm frame}=0$ (Section 8.1, G8). Spin evolution is not an obligation without A-L: $\omega(t)$ is exogenous, and principal-axis spin is hypothesis A-S. Under A-L it is PO-27 | closed on the general route |
| PO-14 | Hypothesis A-ρL (Section 9.2). Proved: exactness of the jet update for affine flows, preservation of $\det\mathbf S_k$ and definiteness, the stage fixed point, Proposition 9.2.1, and additive accumulation of projection defects under the stated idealizations. Not proved: an a-priori bound on $\delta_n$ for a general flow, and the effect of the mask at the core boundary. Under A-Ξ: exactness for affine flows with uniform dilatation, $\det\mathbf S_k'=J_\nu^2\det\mathbf S_k$, exact mass by the two-scalar closure, and the non-amplification factor $(\nu_{\rm hi}/\nu_{\rm lo})^{1/2}$ are proved; the fit of the matrix deficit is measured by $e_\rho$ | partially closed; reported through $e_{\rho,n}$, $|\xi_n|$, $s_{M*}$, the range of $\rho_{\rm bulk}/\rho_K\rvert_\Gamma$, and $\min\rho_K/\rho_{\min}$. In the frozen-density variant A-ρ the ghost gravity is not measured |
| PO-15 | Well-posedness of the Level 2 refit (rank of $\mathbf J^{\!\top}\mathbf W\mathbf J$ for strongly overlapping cores or cores mostly outside $V$) and termination of the Level 3 rules | not proved; rank deficiency is detected by the LM solve and reported; termination of Level 3 is open |
| PO-16 | Hypothesis A-C3: convergence of $X(a')$ as $\theta_r\to0$, the near-surface gravity error of a hop, and convergence of the relief routing (one pass; the routing gap measures its defect). Proved: exact mass balance of the runout kernel and of the routed relief in the linear mass row, the reductions to the isotropic kernel, and unconditional stability of the exact Coriolis rotation | partially closed; escaped and timed-out counts and the largest hop are reported each step; arrival points are checked against halved substeps |
| PO-17 | Hypothesis A-C4. Open: instantaneous failure, the dependence of the settled shape on the trust radius and the number of sweeps, convergence of the relieved shape as the trust radius shrinks, and agreement with a solve resolving the layer. Proved: reduction to (B3)–(B4) for $\tau_s\to\infty$ or $\rho_s\equiv0$, and that promotion leaves the stress, the classification, and the Mode C fluxes unchanged | open; movable vertices, supercritical faces, residual slope slack, largest displacement, and trust radius are reported each step |
| PO-18 | The general route of Section 8.1 and the characteristic-ray structure. Proved: Proposition 8.1.1; independence and the rigid modes of the geometry-adapted basis, with the cavity flux modes (Propositions 8.1.3 and 8.1.4); discrete Nitsche coercivity (Proposition 8.1.6); the vertex-area identity and normal kinematics (G4, Proposition 8.1.7); the ray-segment and surface-flux propositions; Proposition 2.17.1. Sketched: density of the restricted basis with cavity flux modes (Proposition 8.1.5). Open: a rate for the conditioning growth with basis resolution; convergence of the ray-segment rule across silhouettes; sufficiency of the clearance rows for embeddedness; forward stability of the moment recurrence | partially closed |
| PO-19 | Convergence and parameter choice for the characteristic ray relaxation, artificial-compressibility correction, and pressure/constitutive refresh interval $N_p$ | not proved; judged by the hyperbolic residuals of Section 2.15(a) |
| PO-20 | The fused variant of Section 10.3.5: its relation to the closure (B3)–(B4), and the limit $\kappa_i\to\infty$ | not proved; optional variant, not in the reference loop |
| PO-21 | Conjecture A.2 (Appendix A.6): the two associated cone programs bracket the frozen-pressure fixed point and $L^*$ | open |
| PO-22 | Hypothesis A-T1: the error of the Lie split (first order in $\Delta t_n$; the deep state lags the settled shape by one step) as $\tau_C/\tau_n\to0$, and its agreement with the coupled scheme bounded by $\Delta t_C$ | open; $\tau_C/\tau_n$, the settling sweeps, and the settled slack are reported each step |
| PO-23 | Hypothesis A-Ξ, kinematics. Open: convergence of the inexact fixed point in which $\dot\varepsilon_v$ (with $\tilde p$) is frozen and updated every $N_p$ iterations; convergence of $\Delta\chi_h\to\dot\varepsilon_v$ as $\mathbb Q_h$ grows on a nonconvex polyhedron; a bound on the difference between the kinematic and the density-implied volume change, which the mass row closes. Proved: Propositions 11.5–11.8 (exact solid-fraction step, constitutive bounds, independence of the particular field, strict convexity of the frozen problem), the step bound $\Delta t_\Xi$, and the reduction (Theorem 11.13) | partially closed; $e_\varepsilon$, the two volume changes, and $s_M$ are reported each step |
| PO-24 | Softening under A-Ξ and A-ϖ: well-posedness of the evolution driven by the nonlocal rate of Section 11.2, convergence as the numerical resolution grows at fixed $\ell_p$, and the meaning of $\ell_p\to0$ (expected to inherit PO-10). Proved: Proposition 11.4, the exact bounded steps (Propositions 11.5, 11.9), and the preservation of the viscosity bounds and of Proposition 2.17.1 at frozen state | open; $\ell_p/h_{\rm vol}$ and the state ranges are reported each step |
| PO-25 | The state markers of Section 11.1: an error bound for the reconstruction under transport, and the effect of insertion, deletion, and merging. Proved: Lemma 11.2 (admissible values, affine exactness, invariance at rest) and the stage fixed point | partially closed; marker counts, insertions, deletions, merges, and Shepard fallbacks are reported each step |
| PO-26 | Hypothesis A-Θ: the errors of the quasi-steady diurnal state, the neglected seasonal storage, lateral conduction, and mutual radiation, and of one effective effusivity with the shielding factor; the fatigue law and its constants are constitutive inputs. Proved: Proposition 11.10 (exact half-space flux map, monotonicity, uniqueness, positivity, mean balance, discrete existence and uniqueness) and Proposition 11.11 (exact, bounded sources and the ledger) | partially closed; the indicators of Section 11.10 are reported each step |
| PO-27 | Hypothesis A-L: a quantitative adiabatic bound for principal-axis rotation under slowly changing inertia, the damping of the wobble excited by settling within a step, and the constancy of $\kappa_Y$ under law L2. Proved: Proposition 11.12 (energy minimum at fixed $\mathbf L$, true polar wander, reduction to the exogenous table, frame residual) | partially closed; $\varepsilon_{\rm ad}$, $\omega_w$, $\hat g$, $\vartheta_L$, $\Delta E$, and $|\boldsymbol\theta_c|/\omega$ are reported each step |
| PO-28 | Hypothesis A-O. Proved: Proposition 10.3.6.1 (exact update and reduction), Proposition 10.3.6.2 (Hill stability), the exact mass ledger. Open: recapture after a transit; the prompt-departure assumption and the frozen necks within a step; the error of the lobe cap; the neglected recoil, bounded by Remark 10.3.6.4 | partially closed; $a_n$, $M_n$, the lost masses by route, $r_H$, the lobe margin, and the recoil bound are reported each step |
| PO-29 | The failure orbit (Algorithm 2.17.2). Proved: the bracket of Proposition 2.17.3 for any weights. Open: monotonicity of $L^*$ in $a$; the meaning of the frozen pressure $p_0$ at failure, which reduces to PO-21 | partially closed; the certified interval and the bracket along the scan are reported |
| PO-30 | Hypothesis A-B, the cut. Proved: Proposition 12.1 (an exactly piecewise-rigid mechanism is a fixed point of the fit with zero misfit), the closed manifold of the extracted surface, the mass of the pieces. Open: the physical validity of rigid blocks for a granular mechanism smoothed by the Galerkin basis; the dependence of the cut on the gain, the voxel spacing, and the outermost-crossing re-triangulation | partially closed; the misfits and the mass error are reported |
| PO-31 | The fragments' dynamics. Proved: Propositions 12.2 and 12.3 (exact conservation of momentum and angular momentum, second order). Open: the error of the quadrupole expansion at the switching distance; sticky contact in place of frictional contact; the synchronous, tide-aligned load of the cascade as the worst case | partially closed; momentum, angular momentum, energy, and the energy dissipated in mergers are reported |
| PO-32 | The debris ring. Proved: Proposition 12.4 (the ledger). Open: positivity of the explicit nonlinear update at the stated step; the range of validity of the wake viscosity; moonlet formation and migration, and the ring's action on fragments, which are not modelled | partially closed; the ring, primary, and moonlet masses are reported |
| PO-33 | Hypothesis A-Λ. Proved: Propositions 10.3.7.1 (eccentricity at constant $Q$, exact per step), 10.3.7.2 (libration), 10.3.7.3 (self-equilibrium of the cyclic load), 10.3.7.4 (modulus independence of the cyclic stress), 10.3.7.5 (peak and shakedown brackets for the associated surrogate at frozen strength), 10.3.7.7 (1, 2, 4) and 10.3.7.8. Open: convergence of the sampled phase loop to the continuous cycle as $N_M\to\infty$; shakedown of the non-associated law; the error of the incompressible elastic cycle; the transfer of the rock S–N line to the bonds of a rubble pile | partially closed; $|\Gamma|$, $\psi$, the torque residual, the three indicators, the brackets, and the cycle counts are reported |

## A.6 Conjectures

**Conjecture A.2 (A convex surrogate of the pressure coupling brackets the fixed point; PO-21).** The coupled problem is non-associated (Remark 2.17.2), but the elliptic Nitsche pressure equation of Section 8.1 (G3), obtained from the divergence of momentum balance, is affine in the stress, $p=p_0+\mathcal P\boldsymbol\tau$. Impose the Drucker–Prager cone

$$
|\boldsymbol\varsigma_i|\le c_i+\mu\,(p_0+\mathcal P\boldsymbol\tau)_i
$$

on the dual samples of Section 10.3.2, without the tension truncation of (B5); the apex cuts tension at $p=-c/\mu$. Each constraint is a second-order cone with an affine radius, so the admissible stresses form a convex set and the min–max of Section 10.3.2 over it is convex–concave. Its velocity obeys the associated flow rule of the extended space, including the $\mathcal P^{\!\top}$ term, not the co-axial isochoric rule of Section 2.8.

In limit analysis, a non-associated Mohr–Coulomb material has its limit load bracketed by two associated materials: one with the original strength [15], one with Davis's reduced strength [16], $c^*=c\cos\phi$ and $\tan\phi^*=\sin\phi$ for zero dilatancy (plane strain). The conjecture is that the two analogous cone programs, with $(c,\mu)$ and with $(c\cos\phi,\sin\phi)$, bracket the frozen-pressure fixed point of Section 10.3.2 and its limit-load multiplier $L^*$. If it holds, it gives rigorous bounds on $L^*$ and a warm start for the outer constitutive ray relaxation.

---

# References

[1] Forterre, Y., & Pouliquen, O. (2008). Flows of dense granular media. *Annual Review of Fluid Mechanics*, 40, 1–24. https://doi.org/10.1146/annurev.fluid.40.111406.102142

[2] Greengard, L., & Rokhlin, V. (1987). A fast algorithm for particle simulations. *Journal of Computational Physics*, 73(2), 325–348. https://doi.org/10.1016/0021-9991(87)90140-9

[3] Staniforth, A., & Côté, J. (1991). Semi-Lagrangian integration schemes for atmospheric models—A review. *Monthly Weather Review*, 119(9), 2206–2223. https://doi.org/10.1175/1520-0493(1991)119<2206:SLISFA>2.0.CO;2

[4] Hirata, N., Abe, S., Ueno, M., Barnouin-Jha, O. S., Gaskell, R. W., & Miyamoto, H. (2009). Pole and global shape of 25143 Itokawa. *Icarus*, 200(2), 480–497. https://doi.org/10.1016/j.icarus.2008.10.018

[5] Watanabe, S., Hirabayashi, M., Hirata, N., et al. (2019). Hayabusa2 arrives at the carbonaceous asteroid 162173 Ryugu—A spinning top–shaped rubble pile. *Science*, 364(6437), 268–272. https://doi.org/10.1126/science.aav8032

[6] Miyamoto, H., Yano, H., Scheeres, D. J., Abe, S., Barnouin-Jha, O. S., Cheng, A. F., Demura, H., Gaskell, R. W., Hirata, N., et al. (2007). Regolith migration and sorting on asteroid Itokawa. *Science*, 316(5827), 1011–1014. https://doi.org/10.1126/science.1134390

[7] Werner, R. A., & Scheeres, D. J. (1997). Exterior gravitation of a polyhedron derived and compared with harmonic and mascon gravitation representations of asteroid 4769 Castalia. *Celestial Mechanics and Dynamical Astronomy*, 65(3), 313–344. https://doi.org/10.1007/BF00053511

[8] Barker, T., Schaeffer, D. G., Bohórquez, P., & Gray, J. M. N. T. (2015). Well-posed and ill-posed behaviour of the μ(I)-rheology for granular flow. *Journal of Fluid Mechanics*, 779, 794–818. https://doi.org/10.1017/jfm.2015.412

[9] Arakawa, M., Saiki, T., Wada, K., et al. (2020). An artificial impact on the asteroid (162173) Ryugu formed a crater in the gravity-dominated regime. *Science*, 368(6486), 67–71. https://doi.org/10.1126/science.aaz1701

[10] Hirabayashi, M., Tatsumi, E., Miyamoto, H., et al. (2019). The western bulge of 162173 Ryugu formed as a result of a rotationally driven deformation process. *The Astrophysical Journal Letters*, 874(1), L10. https://doi.org/10.2847/2041-8213/ab0e8b

[11] Jop, P., Forterre, Y., & Pouliquen, O. (2006). A constitutive law for dense granular flows. *Nature*, 441(7094), 727–730. https://doi.org/10.1038/nature04801

[12] Sugimura, R., Arakawa, M., & Yasui, M. (2026). Crater size scaling in the strength-dominated regime for weak porous targets: Implications for the tensile strength of asteroid surfaces. *Icarus*, in press (available online 27 May 2026).

[13] Galdi, G. P. (2011). *An Introduction to the Mathematical Theory of the Navier–Stokes Equations: Steady-State Problems* (2nd ed.). Springer.


[14] Radenkovic, D. (1961). Théorèmes limites pour un matériau de Coulomb à dilatation non standardisée. *Comptes Rendus de l'Académie des Sciences, Paris*, 252.

[15] Davis, E. H. (1968). Theories of plasticity and the failure of soil masses. In I. K. Lee (Ed.), *Soil Mechanics: Selected Topics* (pp. 341–380). Butterworths.

[16] Nitsche, J. (1971). Über ein Variationsprinzip zur Lösung von Dirichlet-Problemen bei Verwendung von Teilräumen, die keinen Randbedingungen unterworfen sind. *Abhandlungen aus dem Mathematischen Seminar der Universität Hamburg*, 36, 9–15.

[17] Pailha, M., & Pouliquen, O. (2009). A two-phase flow description of the initiation of underwater granular avalanches. *Journal of Fluid Mechanics*, 633, 115–135. https://doi.org/10.1017/S0022112009007460

[18] Roux, J.-N., & Radjai, F. (1998). Texture-dependent rigid-plastic behavior. In H. J. Herrmann, J.-P. Hovi, & S. Luding (Eds.), *Physics of Dry Granular Media* (NATO ASI Series E, Vol. 350, pp. 229–236). Kluwer.

[19] Pijaudier-Cabot, G., & Bažant, Z. P. (1987). Nonlocal damage theory. *Journal of Engineering Mechanics*, 113(10), 1512–1533. https://doi.org/10.1061/(ASCE)0733-9399(1987)113:10(1512)

[20] Delbo, M., Libourel, G., Wilkerson, J., Murdoch, N., Michel, P., Ramesh, K. T., Ganino, C., Verati, C., & Marchi, S. (2014). Thermal fatigue as the origin of regolith on small asteroids. *Nature*, 508(7495), 233–236. https://doi.org/10.1038/nature13153

[21] Rubincam, D. P. (2000). Radiative spin-up and spin-down of small asteroids. *Icarus*, 148(1), 2–11. https://doi.org/10.1006/icar.2000.6485

[22] Statler, T. S. (2009). Extreme sensitivity of the YORP effect to small-scale topography. *Icarus*, 202(2), 502–513. https://doi.org/10.1016/j.icarus.2009.03.003

[23] Burns, J. A., & Safronov, V. S. (1973). Asteroid nutation angles. *Monthly Notices of the Royal Astronomical Society*, 165(4), 403–411. https://doi.org/10.1093/mnras/165.4.403

[24] Okada, T., Fukuhara, T., Tanaka, S., et al. (2020). Highly porous nature of a primitive asteroid revealed by thermal imaging. *Nature*, 579(7800), 518–522. https://doi.org/10.1038/s41586-020-2102-6

[25] Heyman, J., Delannay, R., Tabuteau, H., & Valance, A. (2017). Compressibility regularizes the μ(I)-rheology for dense granular flows. *Journal of Fluid Mechanics*, 830, 553–568. https://doi.org/10.1017/jfm.2017.612

[26] Goulart, P. J., & Chen, Y. (2024). Clarabel: An interior-point solver for conic programs with quadratic objectives. *arXiv:2405.12762*.

[27] Conley, C. C. (1968). Low energy transit orbits in the restricted three-body problems. *SIAM Journal on Applied Mathematics*, 16(4), 732–746.

[28] Murray, C. D., & Dermott, S. F. (1999). *Solar System Dynamics*. Cambridge University Press.

[29] Andersen, K. D., Christiansen, E., Conn, A. R., & Overton, M. L. (2000). An efficient primal–dual interior-point method for minimizing a sum of Euclidean norms. *SIAM Journal on Scientific Computing*, 22(1), 243–262.

[30] Dullweber, A., Leimkuhler, B., & McLachlan, R. (1997). Symplectic splitting methods for rigid body molecular dynamics. *The Journal of Chemical Physics*, 107(15), 5840–5851.

[31] Daisaka, H., Tanaka, H., & Ida, S. (2001). Viscosity in a dense planetary ring with self-gravitating particles. *Icarus*, 154(2), 296–312.

[32] Salmon, J., Charnoz, S., Crida, A., & Brahic, A. (2010). Long-term and large-scale viscous evolution of dense planetary rings. *Icarus*, 209(2), 771–785.

[33] Hesselbrock, A. J., & Minton, D. A. (2017). An ongoing satellite–ring cycle of Mars and the origins of Phobos and Deimos. *Nature Geoscience*, 10(4), 266–269.

[34] Kaula, W. M. (1964). Tidal dissipation by solid friction and the resulting orbital evolution. *Reviews of Geophysics*, 2(4), 661–685.

[35] Goldreich, P., & Sari, R. (2009). Tidal evolution of rubble piles. *The Astrophysical Journal*, 691(1), 54–60.

[36] Burmeister, S., Willner, K., Schmidt, V., & Oberst, J. (2018). Determination of Phobos' rotational parameters by an inertial frame bundle block adjustment. *Journal of Geodesy*. https://doi.org/10.1007/s00190-018-1112-8

[37] Koiter, W. T. (1960). General theorems for elastic–plastic solids. In I. N. Sneddon & R. Hill (Eds.), *Progress in Solid Mechanics*, Vol. 1 (pp. 165–221). North-Holland.

[38] Melan, E. (1938). Zur Plastizität des räumlichen Kontinuums. *Ingenieur-Archiv*, 9, 116–126.

[39] Ćuk, M., Anand, K. P., & Minton, D. A. (2025). Two possible orbital histories of Phobos. arXiv:2503.12691.

[40] Frederick, C. O., & Armstrong, P. J. (1966). Convergent internal stresses and steady cyclic states of stress. *Journal of Strain Analysis*, 1(2), 154–159.

[41] Cerfontaine, B., & Collin, F. (2018). Cyclic and fatigue behaviour of rock materials: review, interpretation and research perspectives. *Rock Mechanics and Rock Engineering*, 51(2), 391–414.

[42] Williams, D. R. Mars fact sheet. NASA Goddard Space Flight Center, National Space Science Data Center.

[43] Simo, J. C., & Hughes, T. J. R. (1998). *Computational Inelasticity*. Springer.
