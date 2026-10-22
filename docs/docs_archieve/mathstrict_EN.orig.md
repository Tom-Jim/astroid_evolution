# An SCF Model for a Rotating Granular Body Based on a Cauchy Density, RTFP Gravity, and Semi-Lagrangian Transport

## Abstract

This paper establishes a unified mathematical and computational framework for the granular body of a rotating small body [5][6]. The internal mass distribution is a spatially varying Cauchy density. A Feynman parametrization supplies a single integral representation of the potential, its gradient, and its Hessian. Complex free boundaries are treated by ray-traced finite-part (RTFP) ray geometry. A variable-density poloidal–toroidal double-potential flow, a $\mu(I)$-type rheology, semi-Lagrangian transport on the surface, and the geometric projection of frictional constraints are coupled inside one self-consistent-field (SCF) iteration. Exterior classical derivatives, interior finite parts, and surface contact terms are distinguished. Sufficient conditions are given under which the combined rheological model reduces to the classical $\mu(I)$ law. The error and the range of validity are stated for a finite radial Galerkin truncation, for GPU ray intersection, and for a sequential convex quadratic-objective quadratic program or second-order-cone program (QP/SOCP).

The production default is the quasi-static viscoplastic granular body of Mode B. There is a single stress field,

$$
\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D,
$$

a $\mu(I)$ viscoplastic law regularized in the Bingham or Papanastasiou sense. At each geological step the quasi-static Stokes creep equilibrium is solved implicitly at the prescribed spin. Yield is then decided on that stress by the yield function $F$. Solid material connected to the unyielded rigid core is strictly locked ($d_i=0$). Only the yielded region, and those sliding blocks that are separated from the core by a yielded band, acquire a landslide displacement from the creep velocity relative to the core. Metal-PDHG corrects only this mechanically predicted increment, using the current shape as the reference, and never takes an equipotential as its target. The spin rate $\omega(t)$ is an exogenously prescribed piecewise constant, or an exogenously prescribed function of time. It is not coupled to the internal flow by angular-momentum feedback. The fluid mass level and the fluid equipotential of Mode A are retained only as an explicitly marked interface for later use. They do not enter the default granular closed loop.

Rigour, in this paper, means only that the formulae and the acceptance conditions have a determinate meaning under the continuity, regularity, discretization, boundary, and constitutive assumptions that are listed. It does not mean that an absolute error bound is obtained automatically for an arbitrary material, an arbitrary mesh, or an arbitrary finite-precision computation. Any numerical example whose constitutive parameters or boundary data are not supplied is judged **model not closed** and must not be reported as passed.

## Problem

Given a rotation rate, a mass, Cauchy-density parameters, granular state variables, and a frictional constitutive law, construct a mathematically rigorous, conservation-checkable, GPU-implementable closed-loop algorithm such that the following four requirements hold.

1. Inside a complex free boundary, $U$, $\nabla U$, and $\nabla^2U$ admit one integral representation that can be verified.
2. Radial modal changes produced by semi-Lagrangian surface transport pass, conditionally, to the surface shear rate, the inertial number, and the nodal friction angle, under the assumptions of model hypothesis A-M, under compatibility of the recovery operator, and under the full definition of $D'$. A radial difference, when used, is marked explicitly as a local approximation.
3. The core of the loop is the quasi-static viscoplastic Stokes equilibrium of Mode B, with a single stress field, together with the classification into a rigid core and a movable set by the yield function. Mode C, local yield-driven erosion and deposition, is auxiliary. Mechanically predicted increments are corrected, relative to the current shape, by sequential convex subproblems on frozen fields. The spin is prescribed exogenously. The fluid equipotential solve of Mode A is retained only as an interface and does not enter the granular closed loop.
4. Every reduction, regularization, and degeneration is stated together with its hypotheses, its sources of error, and the boundary of its validity.

## Notation and standing assumptions

Unless otherwise stated, $V$ is a bounded polyhedral solid and $\rho_K$ is a nonnegative mass density. In the physical production configurations considered here, the spatial Cauchy density satisfies the global strict lower bound $\rho_K(\mathbf x)\ge\rho_{\min}>0$. This bound makes the kinetic-energy mass operator $\mathbf M_\Gamma$ symmetric positive definite on the subspace $\ell\ge 1$. The constant $G$ is the gravitational constant. The scalar $\omega$ is the exogenously prescribed angular velocity about the body-frame $z$-axis, which is the principal axis of maximum inertia (step 8 of Section 10.4.4). Within each geological step it is frozen at the constant value $\omega_n=\omega(t_n)$ (Section 10.4.1). The effective potential is $W=U+U_c$. Star-shapedness is an additional assumption, used only in the sections on the radial Galerkin method and on the free-surface parametrization. Derivatives at exterior points are classical. The Hessian at interior points and at boundary points is defined by the spherical finite-part convention. The inertial rheology is used when the granular pressure satisfies $p>0$. When $p=0$ or $\dot\gamma=0$, the corresponding regularization, or the static-yield branch, is required.

The paper adopts the positive-potential convention for $U$, $U_c$, and $W$, and defines the conservative effective acceleration by

$$
g_{\mathrm{eff}}=\nabla W.
$$

If the standard incompressible-fluid expression $-\rho\nabla\Phi_{\mathrm{eff}}$ appears, the convention is $\Phi_{\mathrm{eff}}=-W$. More generally, $-\nabla\Phi_{\mathrm{eff}}$ is the conservative acceleration adopted in that expression. This correspondence is maintained throughout the document.

Chapter 2 sometimes writes the same vectors without boldface. The symbols denote the same objects. From Chapter 5 onward the notation of Appendix A.1 is mandatory: the stress is the bold tensor $\boldsymbol\sigma$, a direction is $\hat{\boldsymbol\omega}$, and the angular-velocity vector is $\boldsymbol\Omega=\omega\mathbf e_z$.

# 1. Gravitational field, Hessian, and residuals of a polyhedral generalized Cauchy density

## Introduction: scope and logic

**Objective of this section.** For one and the same model, establish two rigorous integral representations, state the exact range of the original residual method, and treat a polynomial approximation of finite degree as a separate evaluation scheme that carries its own error.

The logical order is the following. Geometry, notation, and the pointwise convention on the boundary are fixed first. Route A defines the finite part by rays centred at the field point and by spherical exclusion. Route B converts the potential kernel, by a Feynman parametrization, into a radial kernel about a moving centre. The two routes are different representations of the same potential. They are not added. When a one-dimensional sub-integral is evaluated by a Carlson, Appell, or theta function, every outer integral that has not yet been evaluated is retained.

Every equality holds under the hypotheses stated in the adjacent text. An analytic representation means only that the identity itself introduces no transformation error. A strict numerical error bound requires, in addition, a computable upper bound on the truncation error, on the quadrature error, and on the rounding error, each separately.

**Conclusion of this section.** The mathematical objects, the algorithmic channels, and the error statements used below have one common interpretation.

## 1.1 Geometry, density, and potential

**Objective of this section.** Give simultaneous and explicit conditions of applicability to ray-intersection segments, to the divergence theorem, and to the local tangent cone.

Let $V\subset\mathbb R^3$ be a bounded polyhedral solid. Assume $V=\overline{\operatorname{int}V}$. The boundary consists of finitely many nondegenerate planar triangles. The triangles are consistently oriented and do not intersect themselves. Finitely many connected components, and internal cavities, are allowed. On the boundary of an interior-oriented cavity, the outward unit normal of the solid is used. If the input mesh does not determine such a solid, the theorems below that refer to a volume domain do not apply directly. Concrete examples of such meshes for small solar-system bodies are given by Hirata et al. [5].

Let $K$ be a positive integer. For each $k\in\{1,\ldots,K\}$, fix $w_k\in\mathbb R$, $\sigma_k>0$, $\alpha_k>0$, and $\mathbf c_k\in\mathbb R^3$. Define the smooth density extended to all of space by

$$
\rho_K(\mathbf y)
=
\sum_{k=1}^{K}
w_k
\left[
1+\sigma_k^2\|\mathbf y-\mathbf c_k\|^2
\right]^{-\alpha_k}.
$$

If this function is used as a physical mass density, one must check separately that $\rho_K(\mathbf y)\ge 0$ for almost every $\mathbf y\in V$. The mathematical identities allow signed weights.

With the positive-potential sign convention,

$$
U_K(\mathbf x)
=
G\int_V
\frac{\rho_K(\mathbf y)}
{\|\mathbf x-\mathbf y\|}
\,d^3\mathbf y,
$$

where $G>0$, $\mathbf x$ is the field point, and $\mathbf y$ is the source point. The potential is finite at every $\mathbf x\in\mathbb R^3$: $V$ is bounded, the density is bounded, and $1/\|\mathbf x-\mathbf y\|$ is locally integrable in three dimensions near $\mathbf y=\mathbf x$.

The notation

$$
H_{ij}(\mathbf x)
=
\frac{\partial^2 U_K(\mathbf x)}
{\partial x_i\,\partial x_j}
$$

is used, but $H_{ij}(\mathbf x)$ is called the classical Hessian only where the classical second derivatives exist. On edges and at other exceptional locations the separately defined finite part $H_{ij,\ell_0}^{\mathrm{FP}}$ is used.

**Conclusion of this section.** The ranges of validity of $V$, $\rho_K$, $U_K$, and the classical Hessian are fixed.

## 1.2 Distributional derivatives and the spherical-exclusion convention

**Objective of this section.** Derive the nonlocal kernel of the Hessian, and restore the contact term at coincident source and field points to the complete formula.

Let $\mathbf r=\mathbf x-\mathbf y$ and $R=\|\mathbf r\|>0$. For $i,j\in\{1,2,3\}$,

$$
\frac{\partial R}{\partial x_i}
=
\frac{r_i}{R},
$$

$$
\frac{\partial}{\partial x_i}\frac1R
=
-\frac{r_i}{R^3},
$$

$$
\frac{\partial^2}{\partial x_j\,\partial x_i}\frac1R
=
-\frac{\delta_{ij}}{R^3}
+
\frac{3r_i r_j}{R^5}.
$$

*Intermediate differentiation.* The first identity is the derivative of $R=(r_k r_k)^{1/2}$. Differentiating $R^{-1}$ gives $-R^{-2}\partial_{x_i}R=-r_i R^{-3}$. Differentiating again,

$$
\partial_{x_j}(-r_i R^{-3})
=
-\delta_{ij}R^{-3}-r_i(-3)R^{-4}\partial_{x_j}R
=
-\delta_{ij}R^{-3}+3r_i r_j R^{-5},
$$

which is the displayed second derivative. Equivalently,

$$
\partial_{x_j}\partial_{x_i}R^{-1}
=
\frac{3r_i r_j-R^2\delta_{ij}}{R^5}.
$$

The numerator is traceless: $\sum_i(3r_i^2-R^2)=0$. The classical pointwise formula is $O(R^{-3})$, hence not absolutely locally integrable near $R=0$, and the distributional derivative must record the spherical contact contribution.

For the distribution defined by spherical exclusion, the correct identity is

$$
\frac{\partial^2}{\partial x_j\,\partial x_i}\frac1R
=
\operatorname{p.v.}_{\mathrm{ball}}
\frac{3r_i r_j-R^2\delta_{ij}}{R^5}
-
\frac{4\pi}{3}\delta_{ij}\delta_0(\mathbf r).
$$

Summing the three diagonal components gives

$$
\Delta_{\mathbf x}\frac1R
=
-4\pi\delta_0(\mathbf r).
$$

*Trace check.* The principal-value kernel has trace zero, and $-\sum_i(4\pi/3)\delta_{ii}\delta_0=-4\pi\delta_0$.

When $\mathbf x\in\operatorname{int}V$, the density $\rho_K$ is smooth near $\mathbf x$, so the classical Hessian and the spherical-exclusion formula satisfy

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

The factor $G$ multiplies both the principal-value integral of the kernel and the contact density $-(4\pi/3)\rho_K(\mathbf x)\delta_{ij}$.

On the boundary, $\rho_K\mathbf 1_V$ jumps. The pointwise product of the contact distribution with the indicator is not assigned a unique numerical value by the distributional identity alone. Boundary values used below are a specified spherical-exclusion representative. They do not automatically equal the interior limit or the exterior limit.

**Conclusion of this section.** The signs and the locations of applicability of the nonlocal kernel, of the spherical principal value, and of the interior contact term are fixed.

## 1.3 Local tangent cone, logarithmic term, and finite part

**Objective of this section.** Give one pointwise convention, which can be checked, for interior points, faces, edges, and vertices.

Fix a field point $\mathbf x$. A finite planar mesh guarantees an $r_\ast(\mathbf x)>0$ such that, after translation to $\mathbf x$, the intersection $V\cap B_{r_\ast(\mathbf x)}(\mathbf x)$ coincides inside that ball with the intersection of a polyhedral cone with the ball. Choose $0<r_0<r_\ast(\mathbf x)$. For an exterior point, require in addition $0<r_0<\operatorname{dist}(\mathbf x,V)$, in which case the local cone is empty.

Adopt the ray convention

$$
\mathbf y=\mathbf x-R\mathbf u,
\qquad
R>0,
\qquad
\mathbf u\in\mathbb S^2.
$$

Let $\Gamma_{\mathbf x}$ be the set of directions for which $\mathbf x-R\mathbf u\in V$ for every sufficiently small $R>0$. Directions on the boundary of this set may be altered on a subset of spherical measure zero. Define

$$
T_{ij}(\mathbf u)=3u_i u_j-\delta_{ij},
$$

$$
C_{ij}(\mathbf x)
=
\int_{\Gamma_{\mathbf x}}
T_{ij}(\mathbf u)
\,d\Omega,
$$

$$
\vartheta_{\mathbf x}
=
\frac{1}{4\pi}
\int_{\Gamma_{\mathbf x}}
1\,d\Omega.
$$

The trace identity used repeatedly below is

$$
\sum_{i=1}^3 T_{ii}(\mathbf u)
=
\sum_{i=1}^3(3u_i^2-1)
=
3\|\mathbf u\|^2-3
=
0.
$$

Hence $\sum_i C_{ii}(\mathbf x)=0$.

Define the nonlocal spherically truncated integral

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
\,d^3\mathbf y.
$$

For a prescribed positive length $\ell_0>0$, define

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

Section 1.5 proves that the limit exists. If $\rho_K(\mathbf x)C_{ij}(\mathbf x)\ne 0$, the corresponding truncated integral, before the subtraction, diverges logarithmically. The number $H_{ij,\ell_0}^{\mathrm{FP}}(\mathbf x)$ is therefore the finite part in the spherical-exclusion convention at the prescribed scale $\ell_0$. It is not the ordinary limit of a truncated integral.

From $\sum_i T_{ii}(\mathbf u)=0$,

$$
\operatorname{Tr}H_{\ell_0}^{\mathrm{FP}}(\mathbf x)
=
-4\pi G\vartheta_{\mathbf x}\rho_K(\mathbf x).
$$

*Derivation of the trace.* The logarithmic counterterm has trace zero because $\sum_i C_{ii}=0$. The contact term contributes

$$
\sum_i\Bigl(-\frac{4\pi G}{3}\vartheta_{\mathbf x}\rho_K\delta_{ii}\Bigr)
=
-4\pi G\vartheta_{\mathbf x}\rho_K.
$$

In the interior, $\Gamma_{\mathbf x}=\mathbb S^2$ up to a null set, so $\vartheta_{\mathbf x}=1$ and the trace is the classical Poisson value $-4\pi G\rho_K(\mathbf x)$. On a face whose interior solid angle is a hemisphere, $\vartheta_{\mathbf x}=1/2$.

For $\ell_0,\ell_1>0$, the change of scale is

$$
H_{ij,\ell_1}^{\mathrm{FP}}(\mathbf x)
-
H_{ij,\ell_0}^{\mathrm{FP}}(\mathbf x)
=
G\rho_K(\mathbf x)C_{ij}(\mathbf x)
\log\frac{\ell_0}{\ell_1}.
$$

*Derivation.* Replacing $\ell_0$ by $\ell_1$ changes only the subtracted logarithm, by

$$
G\rho_K C_{ij}\Bigl(\log\frac{\ell_1}{\varepsilon}-\log\frac{\ell_0}{\varepsilon}\Bigr)
=
G\rho_K C_{ij}\log\frac{\ell_1}{\ell_0}
=
-G\rho_K C_{ij}\log\frac{\ell_0}{\ell_1}
$$

in the subtracted term of the limit that defines $H_{\ell_1}^{\mathrm{FP}}$. Because that term is subtracted, $H_{\ell_1}^{\mathrm{FP}}-H_{\ell_0}^{\mathrm{FP}}$ has the opposite sign, which is the displayed identity. The whole matrix is independent of $\ell_0$ if and only if $\rho_K(\mathbf x)C(\mathbf x)=0$. When $\rho_K(\mathbf x)\ne 0$, this condition reduces to $C(\mathbf x)=0$. The spherical second-moment tensor of a full ball vanishes by isotropy: $\int_{\mathbb S^2}u_iu_j\,d\Omega=\tfrac{4\pi}{3}\delta_{ij}$, so $\int_{\mathbb S^2}T_{ij}\,d\Omega=0$. Since $T_{ij}(-\mathbf u)=T_{ij}(\mathbf u)$, a hemisphere carries exactly half of the full-sphere value. The hemisphere attached to the relative interior of a face also gives $C(\mathbf x)=0$. Proposition 2.7.1 computes the hemispherical moment and then uses the Feynman representation to prove that the logarithmic RTFP singularity is absent at every triangular face centre. Edges and vertices must be computed pointwise, and on those sets $C(\mathbf x)$ need not vanish.

**Conclusion of this section.** The classical Hessian, the truncated integral, and the finite part that depends on the spherical convention and on $\ell_0$ are distinguished.

## 1.4 Route A: spherical coordinates, intersection segments, and completion of the square

**Objective of this section.** Write the nonlocal volume integral exactly as an integral in spherical angle and as a radial integral over every ray-intersection segment.

For $\mathbf y=\mathbf x-R\mathbf u$, in the usual spherical coordinates $(R,\theta,\varphi)$,

$$
\mathbf u
=
(\sin\theta\cos\varphi,\sin\theta\sin\varphi,\cos\theta),
$$

$$
d\Omega
=
\sin\theta\,d\theta\,d\varphi,
$$

$$
d^3\mathbf y
=
R^2\,dR\,d\Omega.
$$

Since $\mathbf x-\mathbf y=R\mathbf u$,

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
T_{ij}(\mathbf u)\frac{dR}{R}\,d\Omega.
$$

*Jacobian check.* The numerator is $T_{ij}(\mathbf u)R^2$, the denominator is $R^5$, and the volume element contributes $R^2\,dR\,d\Omega$, so the product is $T_{ij}(\mathbf u)R^{-1}\,dR\,d\Omega$.

Define the ray intersection

$$
E_{\mathbf x}(\mathbf u)
=
\{R>0:\mathbf x-R\mathbf u\in V\}.
$$

Finite polyhedrality implies that, for almost every direction, this set is a finite union of closed intervals:

$$
E_{\mathbf x}(\mathbf u)
=
\bigcup_{\ell=1}^{L(\mathbf u)}
[R^-_\ell(\mathbf u),R^+_\ell(\mathbf u)].
$$

Here “closed interval” is understood in the relative topology of $R>0$. If the field point lies in the interior of the solid, the first segment is actually $(0,R_1]$. An implementation may use its closure $[0,R_1]$, because the endpoint $R=0$ makes no volumetric contribution to the ordinary radial integrals of the potential and of the gradient, whereas the Hessian at that endpoint must be treated by the finite-part rules of Sections 1.2–1.3.

Every nonempty interval enters the sum, in particular the interval $\ell=1$ of a ray from an exterior point. For the $k$-th kernel set

$$
\mathbf p_k=\mathbf x-\mathbf c_k,
$$

$$
b_k(\mathbf u)=\mathbf p_k\cdot\mathbf u,
$$

$$
a_k=\frac1{\sigma_k},
$$

$$
d_k^2(\mathbf u)
=
a_k^2+\|\mathbf p_k\|^2-b_k^2(\mathbf u)
>0.
$$

Expanding term by term,

$$
\|\mathbf x-R\mathbf u-\mathbf c_k\|^2
=
\|\mathbf p_k-R\mathbf u\|^2
=
\|\mathbf p_k\|^2-2Rb_k(\mathbf u)+R^2,
$$

$$
\|\mathbf p_k\|^2-2Rb_k(\mathbf u)+R^2
=
[R-b_k(\mathbf u)]^2
+
\|\mathbf p_k\|^2-b_k^2(\mathbf u),
$$

$$
1+\sigma_k^2
\|\mathbf x-R\mathbf u-\mathbf c_k\|^2
=
\sigma_k^2
\left[
[R-b_k(\mathbf u)]^2+d_k^2(\mathbf u)
\right].
$$

*Positivity.* The identity $\|\mathbf p_k\|^2-b_k^2=\|\mathbf p_k-b_k\mathbf u\|^2\ge 0$ and $a_k>0$ give $d_k^2(\mathbf u)\ge a_k^2>0$, so the quadratic in $R$ has no real root and the Cauchy factor never vanishes on the ray.

**Conclusion of this section.** Every ray-intersection segment has a unique geometric index, and the real radial quadratic form of each kernel is strictly positive.

## 1.5 Route A: near field, far field, and the finite-part identity

**Objective of this section.** Prove that no segment is omitted or integrated twice, and prove that the finite-part limit of Section 1.3 exists.

With the radius $r_0$ chosen in Section 1.3, define for each direction

$$
E_{\mathbf x}^{\mathrm{near}}(\mathbf u)
=
E_{\mathbf x}(\mathbf u)\cap(0,r_0),
$$

$$
E_{\mathbf x}^{\mathrm{far}}(\mathbf u)
=
E_{\mathbf x}(\mathbf u)\cap[r_0,\infty).
$$

Set operations give

$$
E_{\mathbf x}(\mathbf u)
=
E_{\mathbf x}^{\mathrm{near}}(\mathbf u)
\mathbin{\dot\cup}
E_{\mathbf x}^{\mathrm{far}}(\mathbf u),
$$

where a shared endpoint contributes nothing to a radial integral. For almost every $\mathbf u$,

$$
E_{\mathbf x}^{\mathrm{near}}(\mathbf u)
=
\begin{cases}
(0,r_0),&\mathbf u\in\Gamma_{\mathbf x},\\
\varnothing,&\mathbf u\notin\Gamma_{\mathbf x}.
\end{cases}
$$

Let $0<\varepsilon<r_0$. The Jacobian of Section 1.4 yields

$$
\begin{aligned}
J_{ij,\varepsilon}(\mathbf x)
={}&
G\int_{\Gamma_{\mathbf x}}
T_{ij}(\mathbf u)
\int_\varepsilon^{r_0}
\frac{\rho_K(\mathbf x-R\mathbf u)}{R}
\,dR\,d\Omega\\
&+
G\int_{\mathbb S^2}
T_{ij}(\mathbf u)
\int_{E_{\mathbf x}^{\mathrm{far}}(\mathbf u)}
\frac{\rho_K(\mathbf x-R\mathbf u)}{R}
\,dR\,d\Omega.
\end{aligned}
$$

Adding and subtracting $\rho_K(\mathbf x)$ in the near field,

$$
\begin{aligned}
J_{ij,\varepsilon}(\mathbf x)
={}&
G\rho_K(\mathbf x)
C_{ij}(\mathbf x)
\log\frac{r_0}{\varepsilon}\\
&+
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

*Origin of the logarithm.* On $\Gamma_{\mathbf x}$,

$$
\int_\varepsilon^{r_0}\frac{\rho_K(\mathbf x)}{R}\,dR
=
\rho_K(\mathbf x)\log\frac{r_0}{\varepsilon},
$$

and integration against $T_{ij}$ produces $C_{ij}$.

On a compact set containing every segment $\mathbf x-R\mathbf u$ with $0\le R\le r_0$, there is an $M_1<\infty$ such that $\|\nabla\rho_K\|\le M_1$. The fundamental theorem of calculus gives

$$
\rho_K(\mathbf x-R\mathbf u)-\rho_K(\mathbf x)
=
-R\int_0^1
\nabla\rho_K(\mathbf x-tR\mathbf u)
\cdot\mathbf u
\,dt,
$$

and therefore

$$
\left|
\frac{
\rho_K(\mathbf x-R\mathbf u)-\rho_K(\mathbf x)
}{R}
\right|
\le M_1.
$$

The spherical measure is finite, so the near-field difference may be passed to the limit $\varepsilon\downarrow 0$. In the far field, $R\ge r_0$ and $V$ is bounded, so the far-field integral converges absolutely.

Substitution into the definition of Section 1.3 produces

$$
\boxed{
\begin{aligned}
H_{ij,\ell_0}^{\mathrm{FP}}(\mathbf x)
={}&
G\rho_K(\mathbf x)C_{ij}(\mathbf x)
\log\frac{r_0}{\ell_0}\\
&+
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
\,dR\,d\Omega\\
&-
\frac{4\pi G}{3}
\vartheta_{\mathbf x}\rho_K(\mathbf x)\delta_{ij}.
\end{aligned}
}
$$

*Assembly of the logarithm.* The truncated integral contains $G\rho_K C_{ij}\log(r_0/\varepsilon)$. The definition subtracts $G\rho_K C_{ij}\log(\ell_0/\varepsilon)$. Their difference tends to $G\rho_K C_{ij}\log(r_0/\ell_0)$.

To check independence of $r_0$, take another admissible radius $r_1>r_0$. In the same truncated integral, the interval $[r_0,r_1)$ moves from the old far field into the new near field. On that interval the density-difference term and the constant term add pointwise to the original density term, so the sum is unchanged.

For an exterior field point one may take $r_0<\operatorname{dist}(\mathbf x,V)$. Then $\Gamma_{\mathbf x}=\varnothing$, every intersection segment remains in the far field, and the resulting identity is the classical Hessian. The contact coefficient vanishes because $\vartheta_{\mathbf x}=0$, and there is no logarithmic counterterm.

**Conclusion of this section.** The near field of the fixed tangent cone and the complete far field are strictly complementary. The finite-part limit of Section 1.3 exists. The first intersection segment of an exterior ray is not omitted.

## 1.6 The radial antiderivative for exponent one, and the zero endpoint

**Objective of this section.** For $\alpha_k=1$, give the complete antiderivative, the reverse differentiation, and the finite-part value at a zero endpoint.

Fix a kernel index $k$ and a direction $\mathbf u$, and write $b_k(\mathbf u)$ and $d_k(\mathbf u)$ temporarily as $b$ and $d$. Define

$$
A_0=1+\sigma_k^2\|\mathbf x-\mathbf c_k\|^2,
$$

$$
D(R)=1+\sigma_k^2\|\mathbf x-R\mathbf u-\mathbf c_k\|^2,
$$

$$
D(R)
=
\sigma_k^2R^2-2\sigma_k^2bR+A_0
=
\sigma_k^2[(R-b)^2+d^2].
$$

Since $D'(R)=2\sigma_k^2(R-b)$, the partial-fraction decomposition is

$$
\frac1{R D(R)}
=
\frac1{A_0}
\left[
\frac1R
-
\frac{\sigma_k^2R-2\sigma_k^2b}{D(R)}
\right],
$$

$$
\sigma_k^2R-2\sigma_k^2b
=
\frac12D'(R)-\sigma_k^2b.
$$

*Verification of the decomposition.* The proposed right-hand side, cleared over the common denominator $A_0 R D$, has numerator

$$
D(R)-R(\sigma_k^2R-2\sigma_k^2 b)
=
\bigl(\sigma_k^2R^2-2\sigma_k^2 b R+A_0\bigr)-\sigma_k^2 R^2+2\sigma_k^2 b R
=
A_0.
$$

Division by $A_0 R D$ recovers $1/(R D)$.

Take the continuous real branch of the arctangent for $R>0$, and define

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
\right].
$$

Differentiate term by term. First,

$$
\frac{d}{dR}\log R=\frac1R,
\qquad
\frac{d}{dR}\Bigl(-\frac12\log D\Bigr)
=
-\frac{\sigma_k^2(R-b)}{D(R)},
$$

because $D'=2\sigma_k^2(R-b)$. Second,

$$
\frac{d}{dR}\arctan\frac{R-b}{d}
=
\frac{1/d}{1+\bigl((R-b)/d\bigr)^2}
=
\frac{d}{(R-b)^2+d^2}
=
\frac{\sigma_k^2 d}{D(R)},
$$

where the last step uses $D=\sigma_k^2[(R-b)^2+d^2]$. Multiplication by $b/d$ therefore contributes $\sigma_k^2 b/D(R)$. Combining these derivatives,

$$
\begin{aligned}
P'(R)
={}&
\frac1{A_0}
\left[
\frac1R
-
\frac{\sigma_k^2(R-b)}{D(R)}
+
\frac{\sigma_k^2b}{D(R)}
\right]\\
={}&
\frac{
D(R)-\sigma_k^2R(R-2b)
}{
A_0 R D(R)
}\\
={}&
\frac{
D(R)-\sigma_k^2R^2+2\sigma_k^2 b R
}{
A_0 R D(R)
}\\
={}&
\frac{A_0}{A_0 R D(R)}
=
\frac1{R D(R)}.
\end{aligned}
$$

The numerator identity is the same cancellation used in the partial-fraction check. Thus, for $0<A<B$,

$$
\int_A^B
\frac{dR}{R D(R)}
=
P(B)-P(A).
$$

For $L>0$, the radial finite part at the zero endpoint is

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

Because $D(0)=A_0$ and

$$
\arctan\frac{\varepsilon-b}{d}
=
\arctan\Bigl(-\frac{b}{d}\Bigr)+O(\varepsilon)
=
-\arctan\frac{b}{d}+O(\varepsilon),
$$

one has the expansion

$$
\begin{aligned}
P(\varepsilon)
={}&
\frac1{A_0}
\left[
\log\varepsilon
-
\frac12\log D(\varepsilon)
+
\frac{b}{d}\arctan\frac{\varepsilon-b}{d}
\right]\\
={}&
\frac1{A_0}
\left[
\log\varepsilon
-
\frac12\log A_0
-
\frac{b}{d}\arctan\frac{b}{d}
\right]
+
o(1).
\end{aligned}
$$

The logarithm $\log\varepsilon$ is cancelled by the counterterm $\log(\ell_0/\varepsilon)$, and

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

*Sign of the second arctangent.* The term $-(b/d)\arctan((\varepsilon-b)/d)$ tends to $+(b/d)\arctan(b/d)$, which is why the two arctangents are added. The arguments of both logarithms are dimensionless: $L/\ell_0$ is a ratio of lengths, and $D(L)/A_0$ is a quotient of two dimensionless quantities. The finite part along a single direction is one way of evaluating the near field of Section 1.5. It is not an extra term to be superimposed again.

The antiderivative of Section 3.2 is the same identity after the constant factor $\sigma_k^2$ has been removed from $D$. If $\widetilde D(R)=(R-b)^2+d^2$, then $D=\sigma_k^2\widetilde D$ and $A_0=\sigma_k^2(b^2+d^2)$ on the ray through the present normalization, so

$$
\int\frac{dR}{R D(R)}
=
\sigma_k^{-2}\int\frac{dR}{R\widetilde D(R)}.
$$

Section 3.2 records the integral against $\widetilde D$, because the factor $\sigma_k^{-2\alpha_k}$ stands outside the radial integral. The symbol $A_0$ of Section 3.2 denotes $b^2+d^2=\sigma_k^{-2}A_0$, with $A_0$ as defined in the present section.

**Conclusion of this section.** The radial antiderivative, and the sign of the finite-part value at the zero endpoint, have been checked term by term.

## 1.7 Route B: Feynman parametrization

**Objective of this section.** Fuse the two centres of the Newtonian factor and of the generalized Cauchy factor into one moving centre, and retain the correct scale factor.

Fix one kernel channel and suppress the subscript $k$ temporarily. Set

$$
A(\mathbf y)
=
\|\mathbf y-\mathbf x\|^2,
$$

$$
a=\frac1{\sigma},
$$

$$
B(\mathbf y)
=
\|\mathbf y-\mathbf c\|^2+a^2.
$$

Then

$$
\left[
1+\sigma^2\|\mathbf y-\mathbf c\|^2
\right]^{-\alpha}
=
\sigma^{-2\alpha}B(\mathbf y)^{-\alpha},
$$

because $1+\sigma^2\|\mathbf y-\mathbf c\|^2=\sigma^2(\|\mathbf y-\mathbf c\|^2+\sigma^{-2})=\sigma^2 B(\mathbf y)$.

For $\alpha>0$, $A(\mathbf y)>0$, and $B(\mathbf y)>0$, the Gamma integral gives

$$
A^{-1/2}
=
\frac1{\Gamma(1/2)}
\int_0^\infty
s^{-1/2}e^{-sA}\,ds,
$$

$$
B^{-\alpha}
=
\frac1{\Gamma(\alpha)}
\int_0^\infty
t^{\alpha-1}e^{-tB}\,dt.
$$

Both formulae are the standard representation $Q^{-\nu}=\Gamma(\nu)^{-1}\int_0^\infty\lambda^{\nu-1}e^{-\lambda Q}\,d\lambda$ for $Q>0$ and $\nu>0$. Set

$$
s=u\lambda,
\qquad
t=(1-u)\lambda,
\qquad
0<u<1,
\qquad
\lambda>0.
$$

The Jacobian is

$$
\left|
\frac{\partial(s,t)}{\partial(u,\lambda)}
\right|
=
\lambda,
$$

because $\partial(s,t)/\partial(u,\lambda)$ has rows $(\lambda,u)$ and $(-\lambda,1-u)$, whose determinant is $\lambda(1-u)+\lambda u=\lambda$. Hence $ds\,dt=\lambda\,du\,d\lambda$, and

$$
\begin{aligned}
A^{-1/2}B^{-\alpha}
={}&
\frac1{\Gamma(1/2)\Gamma(\alpha)}
\int_0^1
u^{-1/2}(1-u)^{\alpha-1}
\,du\\
&\times
\int_0^\infty
\lambda^{\alpha-1/2}
\exp\left(
-\lambda[uA+(1-u)B]
\right)
\,d\lambda.
\end{aligned}
$$

The powers of $\lambda$ are $s^{-1/2}t^{\alpha-1}\lambda=(u\lambda)^{-1/2}((1-u)\lambda)^{\alpha-1}\lambda=\lambda^{\alpha-1/2}u^{-1/2}(1-u)^{\alpha-1}$. The inner Gamma integral is

$$
\int_0^\infty
\lambda^{\alpha-1/2}e^{-\lambda Q}
\,d\lambda
=
\Gamma\left(\alpha+\frac12\right)
Q^{-\alpha-1/2},
\qquad
Q>0.
$$

Therefore

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

Define

$$
\mathbf z(u)
=
u\mathbf x+(1-u)\mathbf c,
$$

$$
m^2(u)
=
(1-u)a^2
+
u(1-u)\|\mathbf x-\mathbf c\|^2.
$$

Expand the quadratic form:

$$
\begin{aligned}
u\|\mathbf y-\mathbf x\|^2
&+
(1-u)\|\mathbf y-\mathbf c\|^2\\
={}&
\|\mathbf y\|^2
-2\mathbf y\cdot
\left[u\mathbf x+(1-u)\mathbf c\right]\\
&+
u\|\mathbf x\|^2
+
(1-u)\|\mathbf c\|^2\\
={}&
\|\mathbf y-\mathbf z(u)\|^2
+
u(1-u)\|\mathbf x-\mathbf c\|^2.
\end{aligned}
$$

The cross term that produces $u(1-u)\|\mathbf x-\mathbf c\|^2$ is

$$
u\|\mathbf x\|^2+(1-u)\|\mathbf c\|^2-\|\mathbf z\|^2
=
u(1-u)\|\mathbf x\|^2+u(1-u)\|\mathbf c\|^2-2u(1-u)\mathbf x\cdot\mathbf c.
$$

Adding $(1-u)a^2$ gives

$$
\boxed{
uA(\mathbf y)+(1-u)B(\mathbf y)
=
\|\mathbf y-\mathbf z(u)\|^2+m^2(u).
}
$$

Thus $\mathbf z(u)$ travels along the segment from $\mathbf c$ to $\mathbf x$. For $0\le u<1$,

$$
m^2(u)
=
(1-u)
\left[
a^2+u\|\mathbf x-\mathbf c\|^2
\right]
>0,
$$

because $a>0$ and $1-u>0$. As $u\uparrow 1$, one has $\mathbf z(u)\to\mathbf x$ and $m(u)\to 0$. This is the parametric endpoint at which the near-field singularity sits. The Feynman transformation does not remove that singularity.

**Conclusion of this section.** The dimensionally inconsistent combination $u+(1-u)\sigma^2$ that appears in some reference expressions is replaced by the dimensionally correct $m^2(u)$. The Feynman parametrization is an identity, not an approximation.

## 1.8 Parametric integral, dimensional reduction, and Hessian of the Feynman potential

**Objective of this section.** State which integrals remain after the parametrization, and give a differentiation formula for the exterior Hessian in which differentiation and integration may be exchanged.

Set

$$
p_k=\alpha_k+\frac12,
$$

$$
C_{\alpha_k}
=
\frac{\Gamma\left(\alpha_k+\frac12\right)}
{\Gamma(1/2)\Gamma(\alpha_k)}.
$$

Define

$$
S_p(\mathbf z,m;V)
=
\int_V
\left[
\|\mathbf y-\mathbf z\|^2+m^2
\right]^{-p}
\,d^3\mathbf y.
$$

For a nonnegative single kernel, Tonelli's theorem permits the exchange of the integrals in $u$ and in $\mathbf y$. Finitely many channels with signed weights may be exchanged channel by channel and then summed. Therefore

$$
\boxed{
\begin{aligned}
U_K(\mathbf x)
={}&
G\sum_{k=1}^{K}
w_k\sigma_k^{-2\alpha_k}C_{\alpha_k}
\\
&\times
\int_0^1
u^{-1/2}(1-u)^{\alpha_k-1}
S_{p_k}
\left(
\mathbf z_k(u),m_k(u);V
\right)
\,du.
\end{aligned}
}
$$

This is a one-dimensional outer parameter integral wrapped around a spatial volume integral. It does not prove that $S_p$ is available in closed form.

Fix $u<1$ and set

$$
\mathbf r=\mathbf y-\mathbf z,
\qquad
r=\|\mathbf r\|,
$$

$$
A_p(r;m)
=
\int_0^r
\frac{s^2}{(s^2+m^2)^p}
\,ds,
$$

$$
\mathbf F_p(\mathbf r;m)
=
\frac{A_p(r;m)}{r^3}\mathbf r.
$$

Since

$$
\frac{dA_p(r;m)}{dr}
=
\frac{r^2}{(r^2+m^2)^p},
$$

the radial divergence formula $\nabla_{\mathbf r}\cdot(f(r)\mathbf r)=r^{-2}(r^3 f)'$ gives

$$
\begin{aligned}
\nabla_{\mathbf r}\cdot\mathbf F_p
&=
\frac1{r^2}
\frac{d}{dr}
\left[
r^3\frac{A_p(r;m)}{r^3}
\right]\\
&=
\frac1{r^2}\frac{dA_p(r;m)}{dr}\\
&=
(r^2+m^2)^{-p}.
\end{aligned}
$$

The divergence theorem on the polyhedron, with the outward unit normal $\mathbf n_f$ of each face $\Delta_f$, therefore yields

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

The identity requires $m>0$, so that $\mathbf F_p$ is smooth. The endpoint $u\to 1$, where $m\to 0$, is excluded from this undifferentiated application.

When $p=3/2$,

$$
A_{3/2}(r;m)
=
\operatorname{arsinh}\left(\frac rm\right)
-
\frac r{\sqrt{r^2+m^2}}.
$$

*Check.* Differentiating the right-hand side gives

$$
\frac{1}{\sqrt{r^2+m^2}}
-
\frac{m^2}{(r^2+m^2)^{3/2}}
=
\frac{r^2}{(r^2+m^2)^{3/2}},
$$

which is the integrand of $A_{3/2}$, and the value at $r=0$ is zero.

Fix one planar face. Let $\boldsymbol\xi$ be the in-plane coordinate and let $\boldsymbol\xi_0$ be the in-plane projection of $\mathbf z$. Set

$$
\rho=\|\boldsymbol\xi-\boldsymbol\xi_0\|,
$$

$$
\delta_f
=
(\mathbf y_f-\mathbf z)\cdot\mathbf n_f,
$$

where $\mathbf y_f$ is any point of the plane of $\Delta_f$; $\delta_f$ does not depend on that choice.
The in-plane distance satisfies

$$
\|\mathbf y-\mathbf z\|^2
=
\rho^2+\delta_f^2.
$$

The integrand of the face integral is

$$
g_{p,f}(\rho)
=
\delta_f
\frac{
A_p\left(\sqrt{\rho^2+\delta_f^2};m\right)
}{
\left(\rho^2+\delta_f^2\right)^{3/2}
}.
$$

If one defines

$$
h_{p,f}(\rho)
=
\frac1{\rho^2}
\int_0^\rho
s\,g_{p,f}(s)\,ds,
\qquad
\rho>0,
$$

then

$$
\mathbf G_{p,f}(\boldsymbol\xi)
=
h_{p,f}(\rho)
(\boldsymbol\xi-\boldsymbol\xi_0)
$$

satisfies

$$
\nabla_{\boldsymbol\xi}\cdot\mathbf G_{p,f}
=
g_{p,f}(\rho),
$$

by the two-dimensional analogue of the radial divergence formula used above: $\nabla_{\boldsymbol\xi}\cdot\bigl(h(\rho)(\boldsymbol\xi-\boldsymbol\xi_0)\bigr)=\rho^{-1}\bigl(\rho^2h\bigr)'=\rho^{-1}\cdot\rho\,g_{p,f}(\rho)=g_{p,f}(\rho)$. Here $\boldsymbol\nu_f$ below is the outward in-plane unit conormal of $\partial\Delta_f$. Therefore

$$
\int_{\Delta_f}g_{p,f}\,dS
=
\oint_{\partial\Delta_f}
\mathbf G_{p,f}\cdot\boldsymbol\nu_f
\,ds.
$$

This step requires $h_{p,f}$ to be locally integrable in the face, the boundary trace to exist, and the limit at $\rho=0$ to be taken from the original integral definition. It writes the face integral as an edge integral. It does not guarantee that the edge integral is elementary or a Carlson integral.

For an exterior field point, let $d=\operatorname{dist}(\mathbf x,V)>0$. For

$$
q_k(u,\mathbf y;\mathbf x)
=
u\|\mathbf y-\mathbf x\|^2
+
(1-u)\left[
\|\mathbf y-\mathbf c_k\|^2+a_k^2
\right]
$$

one has

$$
\frac{\partial q_k}{\partial x_i}
=
2u(x_i-y_i),
$$

$$
\frac{\partial^2q_k}
{\partial x_j\partial x_i}
=
2u\delta_{ij}.
$$

*Differentiation.* Only $A(\mathbf y)=\|\mathbf y-\mathbf x\|^2$ depends on $\mathbf x$, and $\partial_{x_i}A=2(x_i-y_i)$, $\partial_{x_j}\partial_{x_i}A=2\delta_{ij}$. The moving-centre form of Section 1.7 produces the same first derivative, as recomputed in Section 2.4. With $p_k=\alpha_k+1/2$,

$$
\frac{\partial}{\partial x_i}q_k^{-p_k}
=
-2p_ku(x_i-y_i)q_k^{-p_k-1},
$$

$$
\begin{aligned}
\frac{\partial^2}{\partial x_j\partial x_i}q_k^{-p_k}
={}&
-2p_ku\delta_{ij}q_k^{-p_k-1}\\
&+
4p_k(p_k+1)u^2
(x_i-y_i)(x_j-y_j)
q_k^{-p_k-2}.
\end{aligned}
$$

The second line is the product rule applied to $-2p_k u(x_i-y_i)q_k^{-p_k-1}$: the derivative hits either the linear factor or the power, and $\partial_{x_j}q_k^{-p_k-1}=-(p_k+1)q_k^{-p_k-2}\partial_{x_j}q_k$.

Since $\|\mathbf x-\mathbf y\|\ge d$, a uniform dominant function on an exterior neighbourhood justifies exchanging differentiation with the volume integral and with the parameter integral. Interior and boundary points cannot use this exterior estimate directly.

**Conclusion of this section.** The Feynman route reduces rigorously to faces, and, under further conditions, to edge integrals. The differentiation formula for the exterior Hessian is complete. The outer integral in $u$ and the geometric boundary integrals remain.

## 1.9 The Carlson criterion on Route B

**Objective of this section.** Prescribe when an edge integral is genuinely an algebraic integral of genus $1$, and when it must be passed to another channel.

Suppose that the reduction from a face to an edge, together with any integration by parts that has been performed, has written a one-dimensional integral as

$$
I_e
=
\int_{t_0}^{t_1}
\frac{R(t,Y(t))}{Y(t)}
\,dt,
$$

$$
Y(t)^2=P_4(t),
$$

where $R(t,Y)$ is rational in $t$ and $Y$, and

$$
P_4(t)
=
a_4t^4+a_3t^3+a_2t^2+a_1t+a_0,
\qquad
a_4\neq0.
$$

The sufficient criterion for the Carlson channel consists of the following five conditions.

First, the quartic term is genuinely nondegenerate: $a_4\neq 0$.

Second, $P_4$ is square-free:

$$
\gcd(P_4,P_4')=1,
$$

or equivalently

$$
\operatorname{Disc}(P_4)\neq0.
$$

Third, the path of integration does not pass through a branch point $P_4(t)=0$, or an integrable endpoint limit has been specified explicitly.

Fourth, no pole of $R(t,Y)$ lies in the interior of the path. If a pole lies on the path, one must declare separately whether the integral is ordinary, a Cauchy principal value, or a Hadamard finite part.

Fifth, the square root $Y(t)$ is chosen continuously along the path. Calling the principal square root independently at each point does not guarantee a continuous branch on the path.

Under these conditions, the smooth projective compactification of the algebraic curve

$$
\mathcal C:\quad Y^2=P_4(t)
$$

is a curve of genus $1$. This conclusion applies only to a sub-integral that has been proved to be a one-dimensional differential of the form above. It is not an automatic global closed form for the original two-dimensional spherical integral.

### 1.9.1 Hermite reduction

**Objective of this section.** Reduce a one-dimensional differential with high-order poles and a high-degree numerator to finitely many standard abelian differentials.

One must find an algebraic function $Q(t,Y)$ and a finite family of basis differentials $\omega_\nu$ such that

$$
\frac{R(t,Y)}{Y}\,dt
=
dQ(t,Y)
+
\sum_{\nu=1}^{N}
c_\nu\omega_\nu.
$$

Because

$$
2Y\,dY=P_4'(t)\,dt,
$$

the differential $dQ$ can be rewritten as a rational combination of $dt$ and $dt/Y$.

When the path contains no pole, the usual basis differentials include

$$
\omega_0=\frac{dt}{Y},
$$

$$
\omega_1=\frac{t\,dt}{Y},
$$

and, for each simple pole $a$ that is not a branch point, the differential of the third kind

$$
\omega_a=\frac{dt}{(t-a)Y}.
$$

If the original expression contains $(t-a)^{-m}$ with $m\ge 2$, the corresponding $dQ$ must be constructed so as to remove the higher-order pole. This step requires the actual solution of algebraic equations for the coefficients. Reduction is not established merely by the shape of the integrand.

### 1.9.2 Reduction from degree four to degree three

**Objective of this section.** Record the substitution required to enter the cubic standard model of Carlson.

Let $e$ be a root of $P_4$. Choose a Möbius transformation

$$
t=\frac{a+bz}{c+dz},
\qquad
ad-bc\neq0,
$$

that sends one branch point to infinity. After substitution and clearing of the denominator, define

$$
W(z)=(c+dz)^2Y(t(z)).
$$

One obtains

$$
W(z)^2=Q_3(z),
$$

where $Q_3$ is a cubic polynomial. Each of $dt/Y$, $t\,dt/Y$, and $dt/[(t-a)Y]$ converts, term by term, into

$$
\frac{A_0+A_1z}{W(z)}\,dz
$$

or

$$
\frac{dz}{(z-p)W(z)},
$$

together with the rational differential already integrated in $dQ$. The coefficients $A_0$, $A_1$, and $p$ are computed from the actual endpoints, branch points, and poles. There is no coefficient that is fixed independently of the geometric input.

### 1.9.3 Carlson symmetric integrals

**Objective of this section.** Give the correct definitions of $R_F$, $R_D$, $R_J$, and $R_C$, and keep definitions, degenerations, and numerical error separate.

For real parameters, and for a path of integration that does not meet a zero of the denominator, define

$$
R_F(x,y,z)
=
\frac12
\int_0^\infty
\frac{ds}
\sqrt{(s+x)(s+y)(s+z)}.
$$

Define

$$
R_D(x,y,z)
=
\frac32
\int_0^\infty
\frac{ds}
{(s+z)\sqrt{(s+x)(s+y)(s+z)}}.
$$

Define

$$
R_J(x,y,z,p)
=
\frac32
\int_0^\infty
\frac{ds}
{(s+p)\sqrt{(s+x)(s+y)(s+z)}}.
$$

Define the degenerate form

$$
R_C(x,y)
=
\frac12
\int_0^\infty
\frac{ds}
{(s+y)\sqrt{s+x}}.
$$

When the branches agree, the definition of $R_F$ gives directly

$$
R_C(x,y)=R_F(x,y,y),
$$

because the two factors $s+y$ under the square root, together with the explicit factor $s+y$ in the denominator of $R_C$, reproduce the integrand of $R_F(x,y,y)$. The two writings are not contradictory definitions. The symbol $R_C$ is the special notation for the case of a repeated root.

After the transformation to a cubic curve, a rational substitution from a finite interval to a semi-infinite interval, and the standard decomposition into differentials, the elliptic sub-integrals of the first, second, and third kinds can be written as finite linear combinations of the Carlson symmetric integrals above. The coefficients depend on the input roots, the endpoints, the branch along the path, and the locations of the poles. They must be generated term by term on each element.

### 1.9.4 Duplication iteration and the error boundary

**Objective of this section.** Give the core iteration of a Carlson evaluation, and state explicitly that an internal stopping test is not a proof of a strict absolute error.

For $R_F(x,y,z)$, set

$$
x_0=x,\qquad
y_0=y,\qquad
z_0=z.
$$

At step $n$, define

$$
\lambda_n
=
\sqrt{x_ny_n}
+
\sqrt{y_nz_n}
+
\sqrt{z_nx_n},
$$

and set

$$
x_{n+1}
=
\frac{x_n+\lambda_n}{4},
\qquad
y_{n+1}
=
\frac{y_n+\lambda_n}{4},
\qquad
z_{n+1}
=
\frac{z_n+\lambda_n}{4}.
$$

These quantities draw together. Homogeneity,

$$
R_F(\lambda x,\lambda y,\lambda z)
=
\lambda^{-1/2}R_F(x,y,z),
\qquad
\lambda>0,
$$

is used to expand the termination on normalized variables. The integrals $R_D$, $R_J$, and $R_C$ each require their own recurrence and compensation terms.

A threshold on the relative spread only decides whether the iteration has drawn together. To claim

$$
|R_F^{\mathrm{computed}}-R_F^{\mathrm{exact}}|
\le\varepsilon
$$

one must also give a combined bound on the remainder of the terminating series, on the choice of complex branch, on the rounding error of the input, and on the rounding error of the recurrence. This paper therefore calls a Carlson stopping threshold an internal error estimate, unless those bounds are actually computed in the implementation.

**Conclusion of this section.** Carlson evaluation is a conditional evaluator of one-dimensional sub-integrals on a square-free quartic algebraic curve. It is not a universal solver for an arbitrary nonintegral power, an arbitrary two-dimensional angular integral, or an arbitrary residual.

## 1.10 Strict definition of the residual

**Objective of this section.** Recover the residual idea of the original reference, while keeping the residual strictly complementary to the analytic terms, omitting no single exterior interval, and never repeating one and the same complete element.

For every open spherical element $\Omega_m$ and every nonempty far-field intersection segment, define the index

$$
q=(m,\ell,k,i,j),
$$

where $m$ is the index of the directional element, $\ell$ is the index of the far-field segment, $k$ is the kernel channel, and $i,j$ are the Hessian components. Each $q$ corresponds to one complete far-field integral

$$
\begin{aligned}
I_q
={}&
G w_k
\int_{\Omega_m}
T_{ij}(\mathbf u)
\\
&\times
\int_{A_{m\ell}(\mathbf u)}^{B_{m\ell}(\mathbf u)}
\frac{
\left[
1+\sigma_k^2
\|\mathbf x-R\mathbf u-\mathbf c_k\|^2
\right]^{-\alpha_k}
}{R}
\,dR\,d\Omega.
\end{aligned}
$$

Near-field intervals do not belong to this residual set. They have already been split, in the finite-part identity of Section 1.5, into an explicit local logarithm, an absolutely integrable difference, and a contact term.

Define a deterministic classification

$$
\chi(q)
\in
\{
\mathrm{Carlson},
\mathrm{Appell},
\mathrm{homotopy},
\mathrm{Theta},
\mathrm{direct}
\}.
$$

Require the disjoint decomposition

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
\mathcal Q_{\mathrm{direct}}.
$$

Then

$$
\begin{aligned}
H_{ij}^{\mathrm{res}}(\mathbf x)
={}&
\sum_{q\in\mathcal Q_{\mathrm{Appell}}}I_q
+
\sum_{q\in\mathcal Q_{\mathrm{homotopy}}}I_q
+
\sum_{q\in\mathcal Q_{\mathrm{Theta}}}I_q
+
\sum_{q\in\mathcal Q_{\mathrm{direct}}}I_q,
\end{aligned}
$$

while the Carlson subset belongs to the analytically evaluated part:

$$
H_{ij}^{\mathrm{Carlson}}(\mathbf x)
=
\sum_{q\in\mathcal Q_{\mathrm{Carlson}}}I_q.
$$

The global far field satisfies

$$
H_{ij}^{\mathrm{far}}(\mathbf x)
=
H_{ij}^{\mathrm{Carlson}}(\mathbf x)
+
H_{ij}^{\mathrm{res}}(\mathbf x).
$$

If a complete element is to be split pointwise into an extracted function and a remainder, one must give

$$
F_q(\xi,\eta)
=
F_q^{\mathrm{ext}}(\xi,\eta)
+
F_q^{\mathrm{rem}}(\xi,\eta)
\quad\text{almost everywhere},
$$

and define

$$
I_q^{\mathrm{ext}}
=
\int_{D_q}F_q^{\mathrm{ext}}(\xi,\eta)\,d\xi\,d\eta,
$$

$$
I_q^{\mathrm{rem}}
=
\int_{D_q}F_q^{\mathrm{rem}}(\xi,\eta)\,d\xi\,d\eta.
$$

Otherwise $F_q^{\mathrm{rem}}$ has no mathematical definition.

Regularity of a residual function can be asserted only piecewise. In the interior of an open element that contains no tangency, no visibility boundary, no vertex, no branch point, and no pole of the path, the residual is locally of class $C^r$ if the kernel and the endpoint functions are sufficiently smooth. On a global closed domain one can usually assert only piecewise integrability, or membership in $L^1$. Without a separate proof one cannot call the residual a $C^\infty$ function on the whole closed domain.

**Conclusion of this section.** The residual is now an explicit complementary sum. It is not “the full integral minus an undefined analytic term”, and it is not the repeated addition of four channels on the same element.

## 1.11 Hypergeometric radial channel: Appell is only a special degeneration

**Objective of this section.** Describe the general hypergeometric type of a radial integral at fixed direction, state that the Appell function appears only under a special degeneration, and correct the scale and the outer integral.

Fix $\mathbf u$ and set

$$
\mathbf p_k=\mathbf x-\mathbf c_k,
$$

$$
b_k(\mathbf u)
=
\mathbf p_k\cdot\mathbf u,
$$

$$
d_k^2(\mathbf u)
=
a_k^2+\|\mathbf p_k\|^2-b_k^2(\mathbf u),
\qquad
a_k=\frac1{\sigma_k}.
$$

Then

$$
1+\sigma_k^2
\|\mathbf x-R\mathbf u-\mathbf c_k\|^2
=
\sigma_k^2
\left[
(R-b_k(\mathbf u))^2+d_k^2(\mathbf u)
\right].
$$

The radial integral at fixed direction is therefore

$$
\mathcal R_{\alpha_k}(A,B;\mathbf u)
=
\sigma_k^{-2\alpha_k}
\int_A^B
\frac{dR}
{R
\left[
(R-b_k(\mathbf u))^2+d_k^2(\mathbf u)
\right]^{\alpha_k}}.
$$

Set

$$
\beta_k(\mathbf u)
=
\frac{b_k(\mathbf u)}{d_k(\mathbf u)},
$$

and

$$
t=\frac{R-b_k(\mathbf u)}{d_k(\mathbf u)}.
$$

Then

$$
R=d_k(\mathbf u)t+b_k(\mathbf u),
$$

$$
dR=d_k(\mathbf u)\,dt,
$$

$$
(R-b_k(\mathbf u))^2+d_k^2(\mathbf u)
=
d_k^2(\mathbf u)(1+t^2),
$$

and consequently

$$
\mathcal R_{\alpha_k}(A,B;\mathbf u)
=
\sigma_k^{-2\alpha_k}
d_k(\mathbf u)^{-2\alpha_k}
\int_{t(A)}^{t(B)}
\frac{dt}
{\left[t+\beta_k(\mathbf u)\right](1+t^2)^{\alpha_k}}.
$$

*Scale count.* The factor $\sigma_k^{-2\alpha_k}$ comes from the Cauchy normalization. The substitution contributes $d_k\cdot d_k^{-2\alpha_k}$ from $dR$ and from the denominator, hence one overall factor $d_k^{-2\alpha_k}$, together with the dimensionless denominator $(t+\beta_k)(1+t^2)^{\alpha_k}$.

After the factorization

$$
(1+t^2)^{-\alpha_k}
=
(1-it)^{-\alpha_k}(1+it)^{-\alpha_k},
$$

the integral generally contains two linear power factors and one simple polar factor. Under a linear or fractional change of variables it is, in general, an incomplete integral of Lauricella type $F_D$ in several variables. It reduces to an Appell function $F_1$, a Gaussian hypergeometric function, a Beta function, or an elementary function only when the polar factor, the exponent, or an endpoint undergoes a special degeneration. Every such representation requires a continuous branch and a path that does not cross a pole.

On a complete directional element, the Appell channel supplies only

$$
\int_{\Omega_m}
T_{ij}(\mathbf u)
\left[
\mathcal R_{\alpha_k}
\left(
A_{m\ell}(\mathbf u),
B_{m\ell}(\mathbf u);
\mathbf u
\right)
\right]
\,d\Omega.
$$

The two-dimensional directional integral therefore remains. The Appell channel is a computable scheme for a complete element only when the outer directional integral has a separate rigorous evaluator or an error certificate.

**Conclusion of this section.** A nonintegral value of $\alpha_k$ does not mean that “the whole element is completed by Appell”. The general radial expression may even require a Lauricella function $F_D$. The correct scale factor is $\sigma_k^{-2\alpha_k}d_k^{-2\alpha_k}$, and the outer angular integral must be retained.

## 1.12 Homotopy residual and direct quadrature

**Objective of this section.** Recover the Poincaré homotopy method, and restrict its geometric domain and its regularity hypotheses.

Let a directional parameter domain $D_m\subset\mathbb R^2$ be parametrized by planar coordinates, and assume that $D_m$ is star-shaped with respect to the origin. Let the residual two-form be

$$
\eta_m
=
F_m(\xi,\eta)
\,d\xi\wedge d\eta.
$$

Define

$$
\mathcal H\eta_m
=
\left[
\int_0^1
sF_m(s\xi,s\eta)
\,ds
\right]
\left(
\xi\,d\eta-\eta\,d\xi
\right).
$$

When $F_m$ is sufficiently smooth, compute the exterior derivative directly. Set

$$
A(\xi,\eta)
=
\int_0^1sF_m(s\xi,s\eta)\,ds.
$$

Then

$$
\mathcal H\eta_m
=
A(\xi,\eta)\xi\,d\eta
-
A(\xi,\eta)\eta\,d\xi,
$$

and

$$
d(\mathcal H\eta_m)
=
\left[
\frac{\partial(A\xi)}{\partial\xi}
+
\frac{\partial(A\eta)}{\partial\eta}
\right]
d\xi\wedge d\eta.
$$

The coefficient is

$$
\frac{\partial(A\xi)}{\partial\xi}
+
\frac{\partial(A\eta)}{\partial\eta}
=
2A+\xi\frac{\partial A}{\partial\xi}
+
\eta\frac{\partial A}{\partial\eta}.
$$

Substitute $A$ and differentiate in $s$:

$$
\frac{d}{ds}
\left[
s^2F_m(s\xi,s\eta)
\right]
=
2sF_m(s\xi,s\eta)
+
s^2
\left[
\xi\partial_\xi F_m(s\xi,s\eta)
+
\eta\partial_\eta F_m(s\xi,s\eta)
\right].
$$

Integrating in $s$ and using $\partial_\xi A=\int_0^1 s\cdot s\,\partial_\xi F_m(s\xi,s\eta)\,ds$, and the analogous formula for $\partial_\eta A$, gives

$$
2A+\xi A_\xi+\eta A_\eta
=
\int_0^1
\frac{d}{ds}
\left[
s^2F_m(s\xi,s\eta)
\right]
\,ds
=
F_m(\xi,\eta),
$$

because $s^2F_m(s\xi,s\eta)\to 0$ as $s\downarrow 0$ under the smoothness assumption that makes the boundary term at $s=0$ vanish. Hence

$$
d(\mathcal H\eta_m)
=
F_m(\xi,\eta)
\,d\xi\wedge d\eta.
$$

Stokes' theorem gives

$$
\int_{D_m}
F_m(\xi,\eta)\,d\xi\,d\eta
=
\int_{\partial D_m}
\mathcal H\eta_m.
$$

If $D_m$ is not star-shaped, it must be triangulated into finitely many star-shaped subdomains, and contributions on shared interior edges cancel by opposite orientation. If $F_m$ is merely locally integrable, one must delete the singular points, pass to the limit, and prove that the contribution of the small boundary circles tends to zero. The smooth version cannot be quoted directly.

For elements that satisfy neither the Carlson, nor the Appell, nor the homotopy condition, define the direct-quadrature set $\mathcal Q_{\mathrm{direct}}$. A double-exponential step-halving test

$$
|I_{h/2}-I_h|<\mathrm{tol}
$$

is only an a-posteriori estimate. A strict error bound requires the order of the endpoint singularity, the width of the complex-analytic strip, the discrete tail, the error of the outer quadrature, and the rounding error.

**Conclusion of this section.** The homotopy formula has been verified term by term, but it retains an integral in the homotopy parameter. Direct quadrature is a legitimate fallback. It does not automatically carry a certificate of absolute error.

## 1.13 Conditional place of the higher-genus theta channel

**Objective of this section.** State when a hyperelliptic integral enters the residual, and give a checkable upper bound on the truncation error of a theta series.

The higher-genus channel is entered only when algebraization has actually produced

$$
Y^2=P_n(t),
$$

where $P_n$ is a square-free polynomial of degree $n\ge 5$, and the differential to be integrated is genuinely an abelian differential on that curve. Square-freeness requires

$$
\gcd(P_n,P_n')=1.
$$

If $n=2g+1$ or $n=2g+2$, the genus of the smooth projective model is

$$
g=\left\lfloor\frac{n-1}{2}\right\rfloor.
$$

Nonconvex geometry does not by itself imply $n\ge 5$, and a nonintegral $\alpha_k$ does not by itself imply a hyperelliptic curve. The degree of the curve must be obtained term by term from the actual elimination.

Let

$$
\tau=X+iY,
$$

where $X=X^T$ and $Y=Y^T>0$, and let

$$
\mathbf z=\mathbf a+i\mathbf b.
$$

Define the Riemann theta series of genus $g$ by

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

Let $\lambda=\lambda_{\min}(Y)>0$. The absolute value of a general term satisfies

$$
\left|
\exp\left(
\pi i\mathbf n^T\tau\mathbf n
+
2\pi i\mathbf n^T\mathbf z
\right)
\right|
=
\exp\left(
-\pi\mathbf n^TY\mathbf n
-
2\pi\mathbf n^T\mathbf b
\right)
$$

and

$$
\left|
\exp\left(
\pi i\mathbf n^T\tau\mathbf n
+
2\pi i\mathbf n^T\mathbf z
\right)
\right|
\le
\exp\left(
-\pi\lambda\|\mathbf n\|_2^2
+
2\pi\|\mathbf b\|_2\|\mathbf n\|_2
\right).
$$

The second bound follows from $\mathbf n^TY\mathbf n\ge\lambda\|\mathbf n\|_2^2$ and $-\mathbf n^T\mathbf b\le\|\mathbf b\|_2\|\mathbf n\|_2$. Set

$$
r_\Theta
=
\frac{\|\mathbf b\|_2}{\lambda}.
$$

For $q\ge r_\Theta$,

$$
-\pi\lambda q^2+2\pi\|\mathbf b\|_2 q
=
\frac{\pi\|\mathbf b\|_2^2}{\lambda}
-
\pi\lambda(q-r_\Theta)^2,
$$

which is the completion of the square. The shell in the infinity norm,

$$
\{\mathbf n\in\mathbb Z^g:\|\mathbf n\|_\infty=q\},
$$

contains

$$
N_g(q)
=(2q+1)^g-(2q-1)^g
$$

lattice points, the difference of the two cubes of odd integer side length. If $N+1\ge r_\Theta$, the tail outside the truncation region $\|\mathbf n\|_\infty\le N$ satisfies

$$
\boxed{
\begin{aligned}
B_g(N)
={}&
\exp\left(
\frac{\pi\|\mathbf b\|_2^2}{\lambda}
\right)
\sum_{q=N+1}^{\infty}
N_g(q)
\exp\left[
-\pi\lambda(q-r_\Theta)^2
\right].
\end{aligned}
}
$$

Thus, if some $N$ has been computed for which

$$
B_g(N)\le\varepsilon_\Theta,
$$

one may assert only that the truncation error of the series does not exceed $\varepsilon_\Theta$. The errors of the period matrix, of the Abel map, of theta derivatives, of the algebraic reduction, and of floating-point arithmetic must still be estimated separately.

**Conclusion of this section.** The theta function is a conditional channel for specified square-free abelian sub-integrals of higher genus. The tail bound is explicit. Nonconvexity and a nonintegral exponent are no longer miswritten as automatic evidence of higher genus.

## 1.14 Auxiliary route through a polynomial density

**Objective of this section.** Decide independently whether a polynomial fit of finite degree can turn the original Cauchy model into a strict closed form.

On the compact set $V$, each summand $\rho_k$ is real analytic. For a fixed degree $p$ one may construct

$$
\rho_{k,p}(\mathbf y)
=
\sum_{|\boldsymbol\beta|\le p}
c_{k,\boldsymbol\beta}
(\mathbf y-\mathbf y_\ast)^{\boldsymbol\beta},
$$

where

$$
\boldsymbol\beta=(\beta_1,\beta_2,\beta_3),
\qquad
|\boldsymbol\beta|=\beta_1+\beta_2+\beta_3,
$$

$$
(\mathbf y-\mathbf y_\ast)^{\boldsymbol\beta}
=
(y_1-y_{\ast,1})^{\beta_1}
(y_2-y_{\ast,2})^{\beta_2}
(y_3-y_{\ast,3})^{\beta_3}.
$$

This expression is a polynomial surrogate. It is not an identity for the original Cauchy function at finite degree. By the theory of polynomial densities on polyhedra, the potential, the gradient, and the Hessian of $\rho_{k,p}$ can be written term by term, through a boundary recurrence, as algebraic expressions in vertices, faces, and edges. That conclusion applies to $\rho_{k,p}$. It does not apply automatically to $\rho_k$.

Define the surrogate Hessian

$$
\begin{aligned}
H_{ij,k}^{(p)}(\mathbf x)
={}&
G\int_V
\rho_{k,p}(\mathbf y)
\\
&\times
\frac{
3(x_i-y_i)(x_j-y_j)
-
\|\mathbf x-\mathbf y\|^2\delta_{ij}
}{
\|\mathbf x-\mathbf y\|^5
}
\,d^3\mathbf y.
\end{aligned}
$$

The difference between the original target and the surrogate is

$$
\begin{aligned}
E_{ij,k}^{(p)}(\mathbf x)
={}&
H_{ij,k}(\mathbf x)-H_{ij,k}^{(p)}(\mathbf x)\\
={}&
G\int_V
\left[
\rho_k(\mathbf y)-\rho_{k,p}(\mathbf y)
\right]
\\
&\times
\frac{
3(x_i-y_i)(x_j-y_j)
-
\|\mathbf x-\mathbf y\|^2\delta_{ij}
}{
\|\mathbf x-\mathbf y\|^5
}
\,d^3\mathbf y.
\end{aligned}
$$

Let $\mathbf x$ be an exterior point, and set

$$
d=\operatorname{dist}(\mathbf x,V)>0.
$$

The numerator estimate

$$
\left|
3(x_i-y_i)(x_j-y_j)
-
\|\mathbf x-\mathbf y\|^2\delta_{ij}
\right|
\le
3\|\mathbf x-\mathbf y\|^2
+
\|\mathbf x-\mathbf y\|^2
=
4\|\mathbf x-\mathbf y\|^2
$$

uses $|r_i r_j|\le R^2$. Division by $R^5$ and the lower bound $R\ge d$ give

$$
\left|
\frac{
3(x_i-y_i)(x_j-y_j)
-
\|\mathbf x-\mathbf y\|^2\delta_{ij}
}{
\|\mathbf x-\mathbf y\|^5
}
\right|
\le
\frac4{d^3}.
$$

If

$$
\sup_{\mathbf y\in V}
\left|
\rho_k(\mathbf y)-\rho_{k,p}(\mathbf y)
\right|
\le
\varepsilon_{\rho,k,p},
$$

then

$$
\boxed{
|E_{ij,k}^{(p)}(\mathbf x)|
\le
\frac{4G\,\operatorname{Vol}(V)}{d^3}
\varepsilon_{\rho,k,p}.
}
$$

The bound is valid for exterior points. For interior points, boundary points, and the finite-part Hessian, the kernel does not admit the same uniform majorant near $\mathbf y=\mathbf x$. The local singular part must be separated before the density-approximation error is estimated.

A fit of degree two or three cannot promise $10^{-5}$ or $10^{-7}$ in general without $\sigma_k$, the locations of the kernel centres, the scale of the bounding box, and the norm in which the error is measured. The rigorous writing of this route is

$$
H_{ij,k}(\mathbf x)
=
H_{ij,k}^{(p)}(\mathbf x)
+
E_{ij,k}^{(p)}(\mathbf x),
$$

together with a verifiable upper bound on $E_{ij,k}^{(p)}$. If one lets $p\to\infty$, one must still prove that the approximating sequence converges in a suitable norm, that the gravitational integral operator is continuous in that norm, and that the limit may be exchanged with the integral.

**Conclusion of this section.** The polynomial auxiliary route is useful, but it is an analytic approximation that carries a model error. The phrase “zero numerical-integration error” belongs only to the surrogate polynomial model, not to a Cauchy fit of finite order.

## 1.15 Self-check list and range of validity

**Objective of this section.** Before a formal expository file is generated, make a final check of logical closure, notation, boundaries, and error statements.

First, verify that the geometric input defines a finite, polyhedral, consistently oriented volume domain without self-intersection.

Second, verify that every ray-intersection segment is listed in $E_{\mathbf x}(\mathbf u)$, and that the near and far fields satisfy

$$
E_{\mathbf x}(\mathbf u)
=
E_{\mathbf x}^{\mathrm{near}}(\mathbf u)
\mathbin{\dot\cup}
E_{\mathbf x}^{\mathrm{far}}(\mathbf u).
$$

Third, verify that the residual classification is a union of disjoint sets. One and the same complete element must not be added simultaneously by Carlson, Appell, homotopy, and direct quadrature.

Fourth, verify that every pointwise residual satisfies the explicit definition

$$
F_q=F_q^{\mathrm{ext}}+F_q^{\mathrm{rem}}.
$$

Otherwise use a complementary collection of elements, not a formal subtraction.

Fifth, verify that the finite part for $\alpha=1$ uses

$$
\lim_{\varepsilon\downarrow0}
\left[
P(L)-P(\varepsilon)
-
\frac1{A_0}\log\frac{\ell_0}{\varepsilon}
\right].
$$

Sixth, verify that every Carlson element satisfies the conditions of a square-free quartic, of the branch along the path, of the poles, and of Hermite reduction.

Seventh, verify that Appell evaluates only the radial integral at fixed direction, and that the outer spherical integral is still treated.

Eighth, verify that the global residual is declared only to be piecewise locally regular and integrable. Unless a separate proof is given, do not claim that it is smooth on the whole closed domain.

Ninth, verify that the numerical error budget includes truncation of special functions, outer and inner quadrature, treatment of endpoints, and rounding.

Tenth, verify that the polynomial route reports $E_{ij,k}^{(p)}$ explicitly, and does not call a surrogate of finite degree an identity for the original function.

The conclusions that can be retained rigorously at present are these. RTFP has a complementary ray identity under the spherical finite-part convention. The Feynman parametrization has a strict moving-centre identity. A spatial integral at fixed parameter reduces rigorously to faces. A one-dimensional edge sub-integral may enter the Carlson channel when the additional algebraic criteria are satisfied. Every remaining element may be evaluated through a mutually exclusive residual channel.

The following universal assertions do not hold. An arbitrary generalized Cauchy density, an arbitrary nonconvex polyhedron, and an arbitrary boundary point do not automatically possess a global closed form composed of finitely many elementary functions and Carlson functions. Not every residual is automatically zero. No fixed number of nodes, and no step-halving test, automatically produces a strict absolute error. A polynomial fit of finite order is not the original Cauchy model.

**Conclusion of this section.** The residual analysis, the Carlson conditions of Route B, and the polynomial auxiliary route are placed under one complementary definition. The final conclusions are restricted strictly to the geometric, branch, regularization, and error conditions that have been listed.

# 2. A unified integral representation of gravity

Chapter 1 fixed the geometric convention, the distributional Hessian, the finite part, the representations of Route A and Route B, and the residual criteria for a polyhedral generalized Cauchy density. This chapter rewrites the density kernel in the scale-parameter form used by the present SCF implementation, and derives the gradient, the Hessian, and the surface coupling quantities step by step from the three-dimensional volume integral. The two writings of the density are identical:

$$
\left[1+\sigma_k^2|y-c_k|^2\right]^{-\alpha_k}
=\sigma_k^{-2\alpha_k}\left[|y-c_k|^2+\sigma_k^{-2}\right]^{-\alpha_k}.
$$

The identity is the factorization $1+\sigma_k^2|y-c_k|^2=\sigma_k^2(|y-c_k|^2+\sigma_k^{-2})$. Throughout this chapter the same vectors are often written without boldface; they are the objects of Chapter 1.

## 2.1 Three-dimensional volume integral and ray representation

For the $k$-th Cauchy kernel,

$$
\rho_k(y)
=
w_k\sigma_k^{-2\alpha_k}
\left[
|y-c_k|^2+a_k^2
\right]^{-\alpha_k},
\qquad
a_k=\sigma_k^{-1}.
$$

Therefore

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
dy.
$$

The minus sign is the derivative of $|x-y|^{-1}$ with respect to the field point, as in Section 1.2. Set

$$
y=x-Ru,\qquad u\in S^2.
$$

Then

$$
x-y=Ru,\qquad dy=R^2\,dR\,d\Omega,
$$

and

$$
\frac{x-y}{|x-y|^3}\,dy
=
\frac{Ru}{R^3}R^2\,dR\,d\Omega
=
u\,dR\,d\Omega.
$$

Hence

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
\,d\Omega.
}
$$

Here

$$
p_k=x-c_k,
\qquad
b_k(u)=p_k\cdot u,
$$

$$
d_k^2(u)
=
a_k^2+|p_k|^2-b_k^2(u)>0.
$$

The formula is exactly equivalent to the three-dimensional volume integral. The sum over $\ell$ runs through every nonempty intersection segment of Section 1.4. If a segment begins at $R=0$, the gradient integral is nonsingular, because the integrand in $R$ is bounded, whereas the Hessian integral of the next sections is not.

---

## 2.2 Feynman parametrization

Define

$$
A(y)=|y-x|^2,
$$

$$
B_k(y)=|y-c_k|^2+a_k^2.
$$

The $k$-th term of the original potential is

$$
U_k(x)
=
G w_k\sigma_k^{-2\alpha_k}
\int_V A(y)^{-1/2}B_k(y)^{-\alpha_k}\,dy.
$$

Use the Feynman identity of Section 1.7,

$$
A^{-1/2}B^{-\alpha}
=
C_\alpha
\int_0^1
u^{-1/2}(1-u)^{\alpha-1}
[uA+(1-u)B]^{-p}\,du,
$$

where

$$
p=\alpha+\frac12,
$$

$$
C_\alpha
=
\frac{\Gamma(\alpha+\frac12)}
{\Gamma(\frac12)\Gamma(\alpha)}.
$$

Therefore

$$
U_k(x)
=
G w_k\sigma_k^{-2\alpha_k}C_{\alpha_k}
\int_0^1
u^{-1/2}(1-u)^{\alpha_k-1}
S_{p_k}(x,u)\,du,
$$

with

$$
p_k=\alpha_k+\frac12
$$

and

$$
S_{p_k}(x,u)
=
\int_V
Q_k(x,y,u)^{-p_k}\,dy,
$$

$$
Q_k(x,y,u)
=
uA(y)+(1-u)B_k(y).
$$

---

## 2.3 Moving centre and effective screening scale

Expand $Q_k$:

$$
Q_k
=
u|y-x|^2
+
(1-u)\left(|y-c_k|^2+a_k^2\right).
$$

Define the moving centre

$$
\boxed{
z_k(x,u)=ux+(1-u)c_k.
}
$$

Completion of the square, as in Section 1.7, gives

$$
\boxed{
Q_k(x,y,u)
=
|y-z_k(x,u)|^2+m_k^2(x,u),
}
$$

where

$$
\boxed{
m_k^2(x,u)
=
(1-u)
\left[
a_k^2+u|x-c_k|^2
\right].
}
$$

Since $a_k>0$,

$$
m_k^2(x,u)>0\qquad(0\le u<1),
$$

but

$$
m_k^2(x,u)\to0\qquad(u\to1).
$$

The Feynman parametrization therefore moves the source-point singularity to the parametric endpoint $u=1$.

---

## 2.4 Unified gradient formula

Differentiation in $x$ begins with the derivatives of $Q_k$. From

$$
z_k=ux+(1-u)c_k
$$

one has

$$
\nabla_x z_k=uI.
$$

From

$$
m_k^2
=
(1-u)\left[a_k^2+u|x-c_k|^2\right]
$$

one has

$$
\nabla_xm_k^2
=
2u(1-u)(x-c_k).
$$

On the other hand,

$$
\nabla_x|y-z_k|^2
=
-2u(y-z_k),
$$

because $\nabla_x z_k=uI$. Adding the two contributions,

$$
\begin{aligned}
\nabla_xQ_k
&=
-2u(y-z_k)
+
2u(1-u)(x-c_k)\\
&=
2u\bigl(x-y\bigr).
\end{aligned}
$$

The cancellation uses $z_k-u(1-u)(x-c_k)$ rearranged as follows:

$$
-(y-z_k)+(1-u)(x-c_k)
=
-y+ux+(1-u)c_k+(1-u)x-(1-u)c_k
=
x-y.
$$

Therefore

$$
\nabla_x Q_k^{-p_k}
=
-p_kQ_k^{-p_k-1}\nabla_xQ_k
=
-2p_k u(x-y)Q_k^{-p_k-1}.
$$

The unified gradient is

$$
\boxed{
\begin{aligned}
\nabla U(x)
={}&
-2G
\sum_{k=1}^K
w_k\sigma_k^{-2\alpha_k}
C_{\alpha_k}p_k\\
&\times
\int_0^1
u^{1/2}(1-u)^{\alpha_k-1}
\int_V
(x-y)
\left[
|y-z_k|^2+m_k^2
\right]^{-p_k-1}
dy\,du.
\end{aligned}
}
$$

The power of $u$ is $u^{-1/2}\cdot u=u^{1/2}$, coming from the Feynman weight and from the factor $u$ in $\nabla_x Q_k$. Here

$$
z_k=z_k(x,u),
\qquad
m_k=m_k(x,u).
$$

The formula is a unified representation of the gradient.

For an exterior point $x\notin\overline V$, differentiation in $x$ may be exchanged with integration. For an interior or boundary point, a local singular analysis is required first. That exchange cannot be quoted directly.

---

## 2.5 Unified Hessian formula

Differentiate the gradient integrand:

$$
\partial_{x_j}
\left[
(x_i-y_i)Q_k^{-p_k-1}
\right].
$$

The first contribution is

$$
\partial_{x_j}(x_i-y_i)
=
\delta_{ij}.
$$

The second is

$$
\partial_{x_j}Q_k^{-p_k-1}
=
-(p_k+1)Q_k^{-p_k-2}
\partial_{x_j}Q_k,
$$

and

$$
\partial_{x_j}Q_k
=
2u(x_j-y_j),
$$

so

$$
\begin{aligned}
\partial_{x_j}
\left[
(x_i-y_i)Q_k^{-p_k-1}
\right]
={}&
\delta_{ij}Q_k^{-p_k-1}\\
&-
2u(p_k+1)
(x_i-y_i)(x_j-y_j)
Q_k^{-p_k-2}.
\end{aligned}
$$

Insert this expression into the gradient formula. The prefactor $-2G\sum w_k\sigma_k^{-2\alpha_k}C_{\alpha_k}p_k$, multiplied by the two terms and by the extra factor $u$ already present in the gradient weight $u^{1/2}=u\cdot u^{-1/2}$, produces the coefficients $4p_k(p_k+1)u^2$ and $-2p_k u$. The unified Hessian is

$$
\boxed{
\begin{aligned}
H_{ij}(x)
={}&
G\sum_{k=1}^K
w_k\sigma_k^{-2\alpha_k}
C_{\alpha_k}\\
&\times
\int_0^1
u^{-1/2}(1-u)^{\alpha_k-1}
\Bigg[
4p_k(p_k+1)u^2\\
&\qquad\qquad\times
\int_V
\frac{(x_i-y_i)(x_j-y_j)}
{\left(|y-z_k|^2+m_k^2\right)^{p_k+2}}
\,dy\\
&\qquad
-
2p_ku
\delta_{ij}
\int_V
\frac{dy}
{\left(|y-z_k|^2+m_k^2\right)^{p_k+1}}
\Bigg]du.
\end{aligned}
}
$$

In matrix form,

$$
\boxed{
\begin{aligned}
H(x)
={}&
G\sum_{k=1}^K
w_k\sigma_k^{-2\alpha_k}
C_{\alpha_k}\\
&\times\int_0^1
u^{-1/2}(1-u)^{\alpha_k-1}
\int_V
\Big[
4p_k(p_k+1)u^2
(x-y)(x-y)^T\\
&\qquad\qquad\qquad\qquad\cdot
Q_k^{-p_k-2}
-
2p_kuI\,Q_k^{-p_k-1}
\Big]dy\,du.
\end{aligned}
}
$$

For an exterior point this formula is a strict representation of the classical Hessian.

---

## 2.6 Consistency with the Newtonian Hessian

When $x\notin\overline V$, the original potential may be differentiated twice directly:

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
B_k(y)^{-\alpha_k}\,dy.
$$

The Feynman formula only rewrites

$$
|x-y|^{-1}B_k(y)^{-\alpha_k}
$$

as a parameter integral. The Feynman identity of Section 1.7 holds pointwise, and an exterior point has a positive lower bound on distance, so Fubini's theorem and the dominated-convergence theorem permit the exchange of the order of integration and the order of differentiation. Therefore

$$
\boxed{
H_{\mathrm{Feynman}}(x)=H_{\mathrm{Newton}}(x)
\qquad(x\notin\overline V).
}
$$

The Feynman representation and the Newtonian representation of the exterior field agree.

---

## 2.7 Finite parts at interior points and at surface points

When $x\in V$ or $x\in\partial V$, the limit $u\to 1$ forces

$$
z_k(x,u)\to x,
\qquad
m_k(x,u)\to 0.
$$

The Feynman formula for the Hessian cannot then be read as an ordinary absolutely convergent integral.

The spherical finite part must be introduced first:

$$
H^{\mathrm{FP}}_{ij,\ell_0}(x)
=
\lim_{\varepsilon\downarrow0}
\left[
J_{ij,\varepsilon}(x)
-
G\rho_K(x)C_{ij}(x)
\log\frac{\ell_0}{\varepsilon}
\right]
-
\frac{4\pi G}{3}
\vartheta_x\rho_K(x)\delta_{ij}.
$$

Only after one has proved

$$
\lim_{\delta\downarrow0}
\left[
H_{\mathrm{Feynman}}^{(\delta)}(x)
-
\text{the same spherical local counterterm}
\right]
=
H^{\mathrm{FP}}_{\ell_0}(x)
$$

may one say that the Feynman representation has computed that same finite part. The symbol $H_{\mathrm{Feynman}}^{(\delta)}$ denotes the parameter integral cut off at $u\le 1-\delta$, or an equivalent excision that removes the same ball.

The strict conclusion, once the representations have been identified, is

$$
\boxed{
\begin{array}{ll}
x\notin\overline V:
&
H_{\mathrm{Feynman}}=H_{\mathrm{RTFP}}=H_{\mathrm{classical}};
\\[4pt]
x\in\operatorname{int}V:
&
H_{\mathrm{Feynman}}\text{ requires the principal value and the contact term};
\\[4pt]
x\in\partial V:
&
H_{\mathrm{Feynman}}\text{ must be matched to the specified spherical finite part};
\\[4pt]
x\text{ on an edge or at a vertex}:
&
H_{\ell_0}^{\mathrm{FP}}\text{ depends on the local tangent cone and on the reference scale}.
\end{array}
}
$$

In an implementation, the primary path for interior points and for boundary points uses the RTFP finite part defined in Section 1.3. The Feynman Hessian may replace that path only after the match to the same local counterterm and to the same scale $\ell_0$ has been verified. At a triangular face centre that match is unconditional: Proposition 2.7.1 proves that the logarithmic counterterm is identically zero.

### Proposition 2.7.1 (Absence of the RTFP singularity at a triangular face centre)

Let $\Delta_f\subset\partial V$ be a nondegenerate planar triangle, with outward unit normal $n$ and centroid

$$
x_\star=\frac13(v_0+v_1+v_2).
$$

Let $\rho_K$ be a generalized Cauchy density of Section 1.1. Then the RTFP Hessian at $x_\star$ has no logarithmic singularity. Precisely:

1. There exists $r_0>0$ such that $V\cap B_{r_0}(x_\star)$ is the closed half-ball cut by the plane of $\Delta_f$.
2. The inward spherical moment vanishes, $C_{ij}(x_\star)=0$, and the solid-angle fraction is $\vartheta_{x_\star}=\tfrac12$.
3. The truncated integral $J_{ij,\varepsilon}(x_\star)$ remains bounded as $\varepsilon\downarrow 0$ and converges to a finite limit. Consequently

   $$
   H_{ij,\ell_0}^{\mathrm{FP}}(x_\star)
   =
   \lim_{\varepsilon\downarrow 0}J_{ij,\varepsilon}(x_\star)
   -
   \frac{2\pi G}{3}\rho_K(x_\star)\delta_{ij}
   $$

   is independent of the reference length $\ell_0$.
4. The same finiteness is obtained from the Feynman integral of Section 2.5: if $H_{ij}^{\mathrm{F},\delta}(x_\star)$ denotes that integral with the parameter restricted to $u\in[0,1-\delta]$, then $\lim_{\delta\downarrow 0}H_{ij}^{\mathrm{F},\delta}(x_\star)$ exists in $\mathbb R$ after the hemispherical contact term is restored, and the limit equals $H_{ij,\ell_0}^{\mathrm{FP}}(x_\star)$. No Hadamard subtraction in $\ell_0$ is required to reach it.
5. The interior and exterior limits of the classical Hessian exist and satisfy

   $$
   H^+=H^{\mathrm{FP}}+2\pi G\rho_K(x_\star)\,nn^{\!\top},
   \qquad
   H^-=H^{\mathrm{FP}}-2\pi G\rho_K(x_\star)\,nn^{\!\top}.
   $$

   Their average is $H^{\mathrm{FP}}$. Their jump is $H^+-H^-=4\pi G\rho_K(x_\star)\,nn^{\!\top}$, and

   $$
   \operatorname{tr}H^--\operatorname{tr}H^+=-4\pi G\rho_K(x_\star).
   $$

The centroid is essential only as the evaluation point used for the tidal tensor in Section 10.3. The argument uses solely that $x_\star$ lies in the relative interior of $\Delta_f$. Every nondegenerate triangle has its centroid in that relative interior, because the barycentric coordinates $\bigl(\tfrac13,\tfrac13,\tfrac13\bigr)$ are strictly positive. A point of an edge, or a vertex, is excluded: the tangent cone is then not a half-space, $C_{ij}$ need not vanish, and $H^{\mathrm{FP}}_{\ell_0}$ retains the scale term of Section 1.3.

*Proof of (1).* Nondegeneracy means that the three vertices are affinely independent, so $\Delta_f$ has positive area and a well-defined unit normal. The centroid lies at a positive distance $d_f$ from $\partial\Delta_f$. Section 1.3 supplies a radius inside which $V$ coincides with its tangent cone. Take $r_0$ smaller than both that radius and $d_f$. Then $B_{r_0}(x_\star)$ meets $\partial V$ only in the plane of $\Delta_f$, and the tangent cone is the half-space $\{y:(y-x_\star)\cdot n\le 0\}$.

*Proof of (2).* The inward cone of directions is the hemisphere

$$
\Gamma_{x_\star}=\{u\in\mathbb S^2:u\cdot n\ge 0\}.
$$

A ray $x_\star-Ru$ enters the solid for every small $R>0$ if and only if $-u\cdot n\le 0$. Rotate coordinates so that $n=e_3$. On this hemisphere $\theta\in[0,\pi/2]$ and $\varphi\in[0,2\pi)$, with $d\Omega=\sin\theta\,d\theta\,d\varphi$. The area is

$$
\int_{\Gamma}1\,d\Omega
=
\int_0^{2\pi}\!d\varphi\int_0^{\pi/2}\sin\theta\,d\theta
=
2\pi,
$$

so $\vartheta_{x_\star}=(1/(4\pi))\cdot 2\pi=\tfrac12$. The second moments are

$$
\int_{\Gamma}u_3^2\,d\Omega
=
\int_0^{2\pi}\!d\varphi\int_0^{\pi/2}\cos^2\theta\sin\theta\,d\theta
=
2\pi\int_0^1\mu^2\,d\mu
=
\frac{2\pi}{3},
$$

where $\mu=\cos\theta$, and

$$
\int_{\Gamma}u_1^2\,d\Omega
=
\Bigl(\int_0^{2\pi}\cos^2\varphi\,d\varphi\Bigr)
\Bigl(\int_0^{\pi/2}\sin^3\theta\,d\theta\Bigr)
=
\pi\int_0^1(1-\mu^2)\,d\mu
=
\pi\cdot\frac23
=
\frac{2\pi}{3}.
$$

The same value is obtained for $u_2^2$ by replacing $\cos^2\varphi$ with $\sin^2\varphi$. Every off-diagonal moment contains a factor $\int_0^{2\pi}\cos\varphi\,d\varphi$, $\int_0^{2\pi}\sin\varphi\,d\varphi$, or $\int_0^{2\pi}\sin\varphi\cos\varphi\,d\varphi$, each of which is zero. Therefore

$$
\int_{\Gamma}(3u_iu_j-\delta_{ij})\,d\Omega
=
3\cdot\frac{2\pi}{3}\delta_{ij}-2\pi\delta_{ij}
=
0.
$$

Hence $C_{ij}(x_\star)=0$. By the scale identity of Section 1.3 the finite part is independent of $\ell_0$ as soon as the limit that defines it exists.

*Proof of (3), from the ray decomposition.* Apply the near/far splitting of Section 1.5 at this $x_\star$. The logarithmic counterterm carries the factor $C_{ij}(x_\star)$ and is identically zero. On the near hemisphere the density difference satisfies

$$
\Bigl|\frac{\rho_K(x_\star-Ru)-\rho_K(x_\star)}{R}\Bigr|\le\|\nabla\rho_K\|_{L^\infty(B_{r_0})},
$$

so the difference integral extends continuously down to $R=0$. The far-field integrand is $O(R^{-1})$ on a set bounded away from the origin and is absolutely integrable. Therefore $\lim_{\varepsilon\downarrow 0}J_{ij,\varepsilon}(x_\star)$ exists and is finite. The contact coefficient reduces to

$$
\frac{4\pi G}{3}\vartheta_{x_\star}\rho_K(x_\star)
=
\frac{2\pi G}{3}\rho_K(x_\star).
$$

This is an ordinary limit of an excised integral, not a Hadamard finite part.

*Proof of (4): the Feynman integral removes the apparent ray singularity.* The ray formula for $I_H$ contains $\int dR/R$. At a surface point that factor looks singular at $R=0$. The Feynman representation proves that, at a face centre, the singularity is removable and that the improper parameter integral converges to the finite part already identified in (3).

Fix one kernel and suppress its index. For $0\le u<1$,

$$
m^2(u)=(1-u)\bigl(a^2+u|x_\star-c|^2\bigr)\ge (1-u)a^2>0,
$$

so $Q_u(y)=|y-z(u)|^2+m^2(u)\ge m^2(u)>0$ on the compact set $V$. The integrand of Section 2.5 is continuous for each such $u$. Write $H_{ij}^{\mathrm{F},\delta}$ for the integral stopped at $u=1-\delta$. The limit $\delta\downarrow 0$ is the only point at issue.

Split $V=V_{\mathrm{near}}\cup V_{\mathrm{far}}$ with $V_{\mathrm{near}}=V\cap B_{r_0}(x_\star)$. Set

$$
\delta_1=\frac{r_0}{2\bigl(1+|x_\star-c|\bigr)}.
$$

If $1-u\le\delta_1$, then $|z(u)-x_\star|=(1-u)|c-x_\star|\le r_0/2$. Every far source point therefore satisfies $|y-z(u)|\ge r_0/2$ and $Q_u(y)\ge r_0^2/4$. The spatial integrand is bounded uniformly in $u$. The weight $u^{-1/2}(1-u)^{\alpha-1}$ lies in $L^1(0,1)$ for $\alpha>0$: the exponent $-\tfrac12$ is integrable at $u=0$, and the exponent $\alpha-1>-1$ is integrable at $u=1$. The far contribution converges absolutely.

On the half-ball, split $\rho_K(y)=\rho_K(x_\star)+\bigl(\rho_K(y)-\rho_K(x_\star)\bigr)$. The fluctuation is Lipschitz and vanishes at $x_\star$. Against the Newtonian Hessian kernel,

$$
\Bigl|\frac{3r_ir_j-R^2\delta_{ij}}{R^5}\Bigr|\,|\rho_K(y)-\rho_K(x_\star)|
\le
\frac{4}{R^3}\cdot M_1 R
=
\frac{4M_1}{R^2}.
$$

The majorant $R^{-2}$ is integrable on a ball in three dimensions, because $\int_0^{r_0}R^{-2}\cdot R^2\,dR<\infty$. Absolute convergence permits differentiation under the integral sign up to the boundary point. The identity of Section 1.7, applied to the potential of the fluctuation, equates this finite Newtonian integral with its Feynman transform. The fluctuation produces no singularity.

The constant-density piece is estimated by scaling, which is where the hemispherical cancellation enters the Feynman integral. Set $s=1-u$, $x_\star$ as the origin, and

$$
m^2(s)=s\,\mu^2(s),\qquad
\mu^2(s)=a^2+(1-s)|c|^2\ge a^2>0.
$$

Then $m(s)=\sqrt{s}\,\mu(s)$ and the moving centre is displaced from the plane by

$$
\eta_{\mathrm{phys}}=z(u)-x_\star=s(c-x_\star),\qquad |\eta_{\mathrm{phys}}|=O(s).
$$

Change variables by $y=m\xi$, so $dy=m^3\,d\xi$. The half-ball $|y|<r_0$ becomes $|\xi|<R_m$ with $R_m=r_0/m\to\infty$, and the half-space constraint is $\xi\cdot n\le 0$. The screening identity

$$
Q=|m\xi-\eta_{\mathrm{phys}}|^2+m^2=m^2\bigl(|\xi-\eta|^2+1\bigr),
\qquad
\eta=\eta_{\mathrm{phys}}/m=O(m)=O(s^{1/2}),
$$

converts the Section 2.5 kernel into

$$
\int_{H}L_{ij}(y;u)\,dy
=
m^{-2\alpha}
\int_{\substack{\xi\cdot n\le 0\\|\xi|<R_m}}
\Lambda_{ij}(\xi;\eta,u)\,d\xi.
$$

The exponent is $3-(2p+2)=1-2p=-2\alpha$, since $p=\alpha+\tfrac12$. The rescaled density is

$$
\begin{aligned}
\Lambda_{ij}(\xi;\eta,u)
={}&
-2p\,u\,\delta_{ij}\bigl(|\xi-\eta|^2+1\bigr)^{-p-1}\\
&+
4p(p+1)u^2\,\xi_i\xi_j\bigl(|\xi-\eta|^2+1\bigr)^{-p-2}.
\end{aligned}
$$

As $s\downarrow 0$ one has $u\to 1$ and $\eta\to 0$, and $\Lambda_{ij}$ converges pointwise to

$$
\Lambda_{ij}^0(\xi)
=
\partial_{\xi_i}\partial_{\xi_j}\bigl(|\xi|^2+1\bigr)^{-p}.
$$

Integrate $\Lambda^0$ over the half-space by the divergence theorem. For $f=(|\xi|^2+1)^{-p}$,

$$
\int_{\Pi_-\cap B_R}\partial_i\partial_j f\,d\xi
=
\int_{D_R}\partial_i f\,n_j\,dA
+
\int_{\mathrm{cap}_R}\partial_i f\,\nu_j\,dS.
$$

On the disk, $\xi\cdot n=0$, so $\partial_n f=-2p(\xi\cdot n)(|\xi|^2+1)^{-p-1}=0$. Each tangential derivative is odd about the origin and integrates to zero. On the cap, $|\nabla f|\le C R^{-2p-1}$ and the area is $O(R^2)$, so the flux is $O(R^{1-2p})=O(R^{-2\alpha})$. Therefore

$$
\int_{\Pi_-}\Lambda_{ij}^0\,d\xi=0,
$$

and the integral over the finite half-ball of radius $R_m$ equals the negative of the tail:

$$
\int_{|\xi|<R_m,\,\xi\cdot n\le 0}\Lambda_{ij}^0\,d\xi
=
O(R_m^{-2\alpha})
=
O(m^{2\alpha}).
$$

Multiplication by the prefactor $m^{-2\alpha}$ produces a contribution $O(1)$, uniformly as $m\downarrow 0$. This is the Feynman form of the statement $C_{ij}(x_\star)=0$: had the hemispherical moment been nonzero, the cap flux would have been logarithmic in $R_m$ and the prefactor $m^{-2\alpha}$ would not have cancelled it.

The displacement $\eta$ is estimated by the mean-value theorem. Differentiating $(|\xi-t\eta|^2+1)^{-q}$ in the shift produces a factor of order $|\xi|(|\xi|^2+1)^{-q-1}$. In polar coordinates the radial integrand is of order $r^{1-2q}$. For every exponent $q\in\{p+1,p+2\}$ one has $q=\alpha+\tfrac32$ or $q=\alpha+\tfrac52$, hence $q>1$, and $\int_1^\infty r^{1-2q}\,dr$ converges. The resulting dominant is integrable on the half-space for every $\alpha>0$, with no restriction $\alpha>\tfrac12$. Therefore, for $|\eta|\le 1$,

$$
\int_{\Pi_-}\bigl|\Lambda_{ij}(\xi;\eta,u)-\Lambda_{ij}^0(\xi)\bigr|\,d\xi
\le
C|\eta|
=
O(m).
$$

Multiplication by $m^{-2\alpha}$ produces $O(m^{1-2\alpha})=O(s^{(1-2\alpha)/2})$. The symmetric difference between the scaled half-ball and the half-space is the tail already shown to be $O(m^{2\alpha})$ in the unscaled $\Lambda$-integral, and the same integrable dominant controls the tail of the difference $\Lambda-\Lambda^0$.

Return to the parameter weight $(1-u)^{\alpha-1}=s^{\alpha-1}$. The cancelled leading piece contributes

$$
s^{\alpha-1}\cdot O(1),
$$

which is integrable near $s=0$ because $\alpha-1>-1$. The centre-shift piece contributes

$$
s^{\alpha-1}\cdot O\bigl(s^{(1-2\alpha)/2}\bigr)
=
O(s^{-1/2}),
$$

and $\int_0^{1/2}s^{-1/2}\,ds<\infty$. Therefore $\int_0^1$ of the near, constant-density contribution converges as an ordinary improper integral. No logarithmic counterterm is subtracted.

Adding the far field, the fluctuation, and this near-field limit shows that $\lim_{\delta\downarrow 0}H_{ij}^{\mathrm{F},\delta}(x_\star)$ exists and is finite. For an exterior point the same integral equals the Newtonian Hessian by Section 2.6. Approaching $x_\star$ from the interior of the half-ball restores the contact term of solid angle $2\pi$, namely $-\tfrac{2\pi G}{3}\rho_K(x_\star)\delta_{ij}$, by the trace convention of Section 1.2 and the value $\vartheta=\tfrac12$ from (2). The sum equals $H_{ij,\ell_0}^{\mathrm{FP}}(x_\star)$. Independence of $\ell_0$ is the statement $C_{ij}(x_\star)=0$, already used as the vanishing of the half-space flux. The ray singularity $\int_0 dR/R$ is therefore absent at the face centre: its angular coefficient is zero, and the Feynman integral converges to the same finite number.

*Proof of (5).* The local model is the constant-density half-space $\Pi_-$, whose potential depends only on the normal coordinate once second derivatives are taken. Poisson’s equation gives $\partial_{nn}U=-4\pi G\rho_K$ on the interior side and $\partial_{nn}U=0$ on the exterior side, while every tangential second derivative of this one-dimensional profile vanishes. Hence the local jump is

$$
H^+_{\mathrm{loc}}-H^-_{\mathrm{loc}}=4\pi G\rho_K\,nn^{\!\top}.
$$

The trace form of the same jump is $\operatorname{tr}H^-_{\mathrm{loc}}-\operatorname{tr}H^+_{\mathrm{loc}}=-4\pi G\rho_K$. The spherical finite part of the half-space, with $\vartheta=\tfrac12$ and $C=0$, is the average of the two limits,

$$
H^{\mathrm{FP}}_{\mathrm{loc}}=-2\pi G\rho_K\,nn^{\!\top},
$$

which has trace $-2\pi G\rho_K=-4\pi G\vartheta\rho_K$ and agrees with Section 1.3. Equivalently,

$$
H^\pm_{\mathrm{loc}}=H^{\mathrm{FP}}_{\mathrm{loc}}\pm 2\pi G\rho_K\,nn^{\!\top}.
$$

For the genuine polyhedron, the integrand outside $B_{r_0}(x_\star)$ is nonsingular as the field point approaches the face, and the density fluctuation was shown above to be absolutely integrable. Both contributions are continuous across the face. They add the same finite tensor to $H^+$, to $H^-$, and to $H^{\mathrm{FP}}$. The jump relations of the half-space therefore survive unchanged, which is (5).

The local contact tensor $-\tfrac{2\pi G}{3}\rho_K I\pm 2\pi G\rho_K\,nn^{\!\top}$ is not a substitute for $H^\pm$. It omits the nonlocal integral over $V\setminus B_{r_0}(x_\star)$. That integral is finite at the face centre precisely because the logarithm has been shown to vanish, and its deviatoric part is part of the tidal field. Discarding it would delete a long-range contribution that the Feynman representation retains.

*Edge contrast, not used at face centres.* If the field point lies on an edge, the tangent cone is a dihedral wedge whose spherical section is a lune rather than a hemisphere. The moment $\int(3u_iu_j-\delta_{ij})\,d\Omega$ over that lune is not zero in general. The ray integral then diverges as $C_{ij}\log(\ell_0/\varepsilon)$, and $H^{\mathrm{FP}}_{\ell_0}$ depends on $\ell_0$. The Feynman weight does not cancel that coefficient. Face centres are exactly the boundary points at which the coefficient is proved to vanish.

**Conclusion of the proposition.** Every Hessian that enters $T_{\mathrm{tide}}$ through a triangular face centre is a finite, scale-free RTFP value. The Feynman integral is the instrument that isolates the local half-space, exhibits the vanishing of its flux, and identifies the resulting finite number with the ray-traced finite part. Vertices and edge samples remain scale-dependent and are excluded from that evaluation by Section 10.3.

---

## 2.8 Granular balance equations

The following conditions on the potential, on the slope, and on the mass are only part of granular balance:

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

*Origin of the Poisson equation.* In the interior, the spherical convention of Section 1.3 gives $\vartheta_x=1$ and $\operatorname{Tr}\nabla^2 U=-4\pi G\rho_K$. The centrifugal potential $U_c=\tfrac12\omega^2(x_1^2+x_2^2)$ has Laplacian $2\omega^2$. Hence $\Delta W=\Delta U+2\omega^2=2\omega^2-4\pi G\rho_K$ in $\operatorname{int}V$. On the boundary, $\vartheta_x$ is not identically $1$, so this Poisson equation is an interior statement. Here

$$
W(x)=U(x)+U_c(x),
$$

the unified gradient is

$$
\nabla W(x)
=
\nabla U(x)
+
\omega^2(x_1,x_2,0),
$$

and the unified Hessian is

$$
\nabla^2W(x)
=
\nabla^2U(x)
+
\omega^2
\begin{pmatrix}
1&0&0\\
0&1&0\\
0&0&0
\end{pmatrix}.
$$

Both follow by differentiating $U_c$ twice. At a surface node $x_i$,

$$
g_i^{(n)}
=
\nabla U^{(n)}(x_i^{(n)})
+
\omega^2(x_{i1}^{(n)},x_{i2}^{(n)},0).
$$

The local slope constraint is

$$
\boxed{
|g_i^{(n)}\times n_i^{(n)}|^2
\le
\sin^2\phi_i^{(n)}\,|g_i^{(n)}|^2.
}
$$

*Geometric meaning.* The identity $|g\times n|^2=|g|^2|n|^2-(g\cdot n)^2$ rewrites the inequality, for a unit normal and for $|g_i^{(n)}|>0$, as

$$
\sin^2\theta_i\le\sin^2\phi_i^{(n)},
$$

where $\theta_i$ is the angle between $-g_i^{(n)}$ and the outward normal, equivalently the slope angle of the surface relative to the local effective horizontal [7]. The normalized constraint requires $|g_i^{(n)}|>0$. If $|g_i^{(n)}|=0$ at a node, the displayed formula is undefined. One must then use the un-normalized contact or yield constraint, or mark the node as degenerate. The implementation must not divide by $|g_i^{(n)}|^2$.

**Mode B: quasi-static viscoplastic Stokes creep and physical yield classification (a single stress field).** The mechanical route of this paper has **one** stress field, the viscoplastic fluid stress of Definition 5.1 in Section 5.1:

$$
\boxed{\ \boldsymbol\sigma=-p\,\mathbf I+\boldsymbol\tau,\qquad
\boldsymbol\tau=\mathbf T_{\rm tide}+\boldsymbol\tau_{\rm flow},\qquad
\boldsymbol\tau_{\rm flow}=2\eta(\tilde p,\mathbf D)\,\mathbf D,\qquad
\eta=\frac{c+\mu_d\,\tilde p}{\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}},\ }
$$

where $\dot\gamma=\sqrt{2\mathbf D:\mathbf D}$ and $\tilde p:=\max([p]_+,p_{\min})$ is the unique regularized pressure. The mandatory rule of Section 4.13 is that the stress magnitude, the yield strength, and the denominator of $I_c$ use this same $\tilde p$, with $I_c$ evaluated at $p_*=\tilde p+p_c$. The coefficient $\mu_d=\mu_d(I_c;\nu_0,Z_0)$ is given in Sections 4.5 and 10.3. If the Papanastasiou regularization

$$
\eta=(c+\mu_d\tilde p)\bigl(1-e^{-\dot\gamma/\dot\gamma_\varepsilon}\bigr)/\dot\gamma
$$

is used instead, the viscosity $\eta$ and its derivatives in Sections 4.17, 5.1, and 5.3 must be replaced together. The two regularizations must not be mixed. From $p_{\min}>0$ and the boundedness of the discrete velocity,

$$
0<\eta_{\rm lo}
:=\frac{c+\mu_s p_{\min}}{\sqrt{\dot\gamma_{\max}^2+\dot\gamma_\varepsilon^2}}
\le\eta\le
\eta_{\rm hi}
:=\frac{c+\mu_2\tilde p_{\max}}{\dot\gamma_\varepsilon}
$$

at every discrete state. The upper bound uses $0<\mu_d\le\mu_2$ and $\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}\ge\dot\gamma_\varepsilon$. The lower bound uses $\mu_d\ge\mu_s$ and a configured bound $\dot\gamma\le\dot\gamma_{\max}$ on the discrete velocity gradient. The viscous stiffness is therefore uniformly positive definite on the complement of the rigid-body constraints (Lemma 8.12 of Section 8.4). **This paper contains no** elastic strain, elastic modulus, hardening variable, return-mapping stress, or independent displacement unknown. Wherever a stress is required — balance, yield, tide, traction, erosional flux, or acceptance — the expression above is used.

**(B1) Quasi-static Stokes balance.** At geological step $n$, on the frozen geometry $\mathcal V^{(n)}$, with the frozen Cauchy density $\rho_K$ and the exogenous spin $\omega_n:=\omega(t_n)$ (Section 10.4.1), solve the time-independent creep balance

$$
\boxed{\ 
\begin{aligned}
&\nabla\cdot\boldsymbol\tau-\nabla p+\rho_K\,\mathbf g_{\rm eff}(\omega_n)=\mathbf 0,\qquad \nabla\cdot\mathbf v=0\quad\text{in }\mathcal V^{(n)},\\
&\boldsymbol\sigma\mathbf n=\mathbf t_{\rm ext}\quad\text{on }\Gamma^{(n)},\qquad
\int_{\mathcal V}\rho_K\mathbf v\,dV=\mathbf 0,\quad \int_{\mathcal V}\rho_K\,\mathbf x\times\mathbf v\,dV=\mathbf 0,
\end{aligned}\ }
$$

with $\mathbf g_{\rm eff}(\omega)=\nabla U+\omega^2\mathbf x_\perp$. The inertia $\rho\partial_t\mathbf v$, the convection $\rho(\mathbf v\cdot\nabla)\mathbf v$, and the Coriolis acceleration $2\rho\boldsymbol\Omega\times\mathbf v$ are all omitted. This is the quasi-static limit. The indicator of applicability

$$
\mathrm{Ek}^{-1}:=\max_{\mathcal Y}\rho_K\omega_n R_{\max}^2/\eta
$$

is reported at each step on the yielded region $\mathcal Y$. If it exceeds the configured threshold $\varepsilon_{\rm in}$, the step is marked **quasi-static assumption failed**. The problem **contains no time-step size**. The field $\mathbf v$ is the creep velocity in instantaneous balance with the current load. It is obtained implicitly from the discrete Stokes system (8.4′) of Section 8.4, from the pressure Poisson problem of Section 8.7, and from the Picard iteration of Section 10.4.2. There is no explicit momentum relaxation, and there is no CFL restriction of viscous-diffusion type.

The Euler inertial force $-\dot\omega\,\mathbf e_z\times\mathbf x$ associated with an exogenous spin rate $\dot\omega$, and the external torque that drives the change of spin, balance each other in the body frame. Their relative contribution to the internal stress is

$$
\varepsilon_{\rm Euler}:=|\dot\omega|/\omega^2.
$$

Both are omitted together, and $\varepsilon_{\rm Euler}$ is reported. Omitting only one of them would violate the torque-compatibility condition of Section A.3.

**(B2) Yield function and regularized yield criterion.** The static strength and the yield function are evaluated on the unique stress field:

$$
\tau_y:=c+\mu_y\,\tilde p,\qquad \mu_y:\equiv\mu_s(\nu_0,Z_0)=\lim_{I_c\to0}\mu_d,\qquad
F:=\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}-\tau_y=\|\mathbf s-\mathbf T_{\rm tide}\|_{\rm eq}-c-\mu_y\tilde p,
$$

where $\|\mathbf A\|_{\rm eq}=\sqrt{\mathbf A:\mathbf A/2}$ and $\mathbf s=\boldsymbol\tau$ is the deviatoric stress. The identity $\operatorname{tr}\boldsymbol\tau_{\rm flow}=0$ follows from $\nabla\cdot\mathbf v=0$, because $\operatorname{tr}\mathbf D=0$. The identity $\operatorname{tr}\mathbf T_{\rm tide}=0$ follows from the deviatoric projection in the definition of $\mathbf T_{\rm tide}$. The static limit of $\mu_d$ is used for $\mu_y$, so that the viscous law in the unyielded region and the yield function use the same strength.

From $\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}=\eta\dot\gamma=(c+\mu_d\tilde p)\,r(\dot\gamma/\dot\gamma_\varepsilon)$, with $r(x)=x/\sqrt{1+x^2}$ in the Bingham regularization, or $r(x)=1-e^{-x}$ in the Papanastasiou regularization, one has $0\le r<1$. *Verification for the Bingham branch:*

$$
\eta\dot\gamma
=
(c+\mu_d\tilde p)\frac{\dot\gamma}{\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}}
=
(c+\mu_d\tilde p)\,r(\dot\gamma/\dot\gamma_\varepsilon).
$$

In the quasi-static regime, where $\mu_d\to\mu_s$, the regularized stress is **strictly smaller** than the strength, and $F>0$ is never triggered. The yield criterion must therefore carry a tolerance $\delta_Y\in(0,1)$, a configured value whose default is $0.05$:

$$
\boxed{\ F_\delta:=\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}-(1-\delta_Y)\,\tau_y,\qquad
\text{the point }\mathbf x\text{ has yielded}\iff F_\delta(\mathbf x)>0 .\ }
$$

When $\mu_d=\mu_s$, one has $F_\delta>0$ if and only if $\dot\gamma>\dot\gamma_Y$, with

$$
\dot\gamma_Y=\dot\gamma_\varepsilon\frac{1-\delta_Y}{\sqrt{2\delta_Y-\delta_Y^2}}
$$

for the Bingham branch, and

$$
\dot\gamma_Y=\dot\gamma_\varepsilon\ln(1/\delta_Y)
$$

for the Papanastasiou branch. *Bingham threshold.* The inequality $r(\dot\gamma/\dot\gamma_\varepsilon)>1-\delta_Y$ becomes

$$
\frac{x}{\sqrt{1+x^2}}>1-\delta_Y,\qquad x=\dot\gamma/\dot\gamma_\varepsilon,
$$

hence $x>(1-\delta_Y)/\sqrt{1-(1-\delta_Y)^2}=(1-\delta_Y)/\sqrt{2\delta_Y-\delta_Y^2}$. *Papanastasiou threshold.* The inequality $1-e^{-x}>1-\delta_Y$ is $e^{-x}<\delta_Y$, hence $x>\ln(1/\delta_Y)$. At $\delta_Y=0.05$ both thresholds are approximately $3\dot\gamma_\varepsilon$. This is a **criterion on the stress, or equivalently on an invariant of the strain rate, not a criterion on the magnitude of the velocity**. A rigid translation or rotation, for which $\mathbf D=\mathbf 0$, satisfies $F_\delta=-(1-\delta_Y)\tau_y<0$ however large the velocity may be. Convergence of the yield surface of the regularized solution, at fixed $\delta_Y$, to the yield surface of the unregularized Bingham or $\mu(I)$ law as $\dot\gamma_\varepsilon\to 0$ is proof obligation PO-12.

**(B3) Rigid core and movable set (the physical definition of the dead zone).** Evaluate $F_\delta$ on the classification mesh

$$
\mathcal X_Y:=\{(\zeta_q,\hat{\boldsymbol\omega}_i):\ q=1,\dots,N_r+1,\ i=1,\dots,N_S\},
$$

where $\hat{\boldsymbol\omega}_i$ is the direction of a surface-mesh vertex, $\zeta_1<\dots<\zeta_{N_r}$ are the radial Gauss–Jacobi nodes, and $\zeta_{N_r+1}=1$ is the surface. The velocity $\mathbf v$ is synthesized pointwise from the modal coefficients by Proposition 10.2 and Section 5.2. The pressure $p$ is synthesized pointwise from the pressure basis of Section 8.7. Both are defined at an arbitrary point. Adjacency in the graph $\mathsf G_Y$ is as follows: successive radial points on the same ray; at a fixed $\zeta_q$, the rays that meet along an edge of the surface mesh; and all points at $\zeta_1$, joined through one virtual centre node. Let $\mathcal U=\{F_\delta\le 0\}$ be the unyielded set. Compute the connected components of $\mathsf G_Y$ restricted to $\mathcal U$. The **rigid core** $\mathcal U_{\rm core}$ is the component of largest quadrature mass $\sum\rho_K w$. In the event of a tie, choose the component that contains the largest number of points on the layer $\zeta_1$. If $\mathcal U=\varnothing$, then $\mathcal U_{\rm core}=\varnothing$, which is global instability. The movable indicator of a surface node is

$$
\boxed{\ \chi_{Y,i}=\begin{cases}0,&(1,\hat{\boldsymbol\omega}_i)\in\mathcal U_{\rm core}\quad\text{(solid connected to the rigid core; locked)},\\ 1,&\text{otherwise (the node itself has yielded, or it lies on a sliding block separated from the core by a yielded band)}.\end{cases}\ }
$$

In the unregularized limit, an unyielded connected component moves as a rigid body. Solid material connected to the core moves with the core. A sliding block separated from the core by a yielded shear band — a landslide block — slides as a whole relative to the core. Surface points of that block may themselves be unyielded, but they must remain movable. This is why the surface value of $F_\delta$ alone is not a sufficient test.

The rigid motion of the core is the least-squares fit

$$
(\mathbf a_c,\boldsymbol\theta_c)=\arg\min_{\mathbf a,\boldsymbol\theta}\sum_{\mathbf x\in\mathcal U_{\rm core}}w_{\mathbf x}\rho_K(\mathbf x)\,|\mathbf v(\mathbf x)-\mathbf a-\boldsymbol\theta\times\mathbf x|^2.
$$

The normal equation is $6\times 6$ and symmetric positive definite when the weighted points are not collinear. If $\mathcal U_{\rm core}$ degenerates to a collinear set, the step is marked **core degeneracy** and one sets $\mathbf a_c=\boldsymbol\theta_c=\mathbf 0$. Every geometric prediction uses the **velocity relative to the core**,

$$
\mathbf v_{\rm sl}:=\mathbf v-\mathbf a_c-\boldsymbol\theta_c\times\mathbf x.
$$

The net translation of the core itself is absorbed by the centre-of-mass realignment in step 8 of Section 10.4.4. The residual $\mathbf v_{\rm sl}$ inside the core is a numerical creep produced by the regularization $\dot\gamma_\varepsilon>0$. It is removed exactly by $\chi_{Y,i}=0$. This paper contains no state variable obtained by integrating $\mathbf D$ in time, so this creep neither enters the geometry nor accumulates in any history variable.

**(B4) Surface predictor (the unique definition of $d_{\rm pred}$).**

$$
\boxed{\ d_{\mathrm{pred},i}^{(n)}=\frac{\chi_{Y,i}\,\Delta t_n\,\mathbf v_{{\rm sl},i}^{(n)}\!\cdot n_i^{(n)}+\dfrac{\rho_{\mathrm{bulk}}}{\rho_K(\mathbf x_i)}\,\Delta h_{C,i}}{n_i^{(n)}\cdot e_r},\ }
\qquad
\Delta h_C=\Delta t_n\,\frac{J_{\mathrm{dep}}-J_{\mathrm{ero}}}{\rho_{\mathrm{bulk}}},
\qquad
\rho_{\mathrm{bulk}}=\rho_p\nu_0,
$$

where $\Delta t_n$ is the geological step of Section 10.4.1. Surface nodes then fall into three classes.

- The **locked set** $\mathcal L^{(n)}=\{i:\chi_{Y,i}=0,\ \Delta h_{C,i}=0\}$ is the solid substrate, with $d_i=0$.
- The **pure-deposition set** $\mathcal D^{(n)}=\{i:\chi_{Y,i}=0,\ \Delta h_{C,i}\ne 0\}$ has an immobile substrate and only an external accumulation of material, with $d_i=d_{\mathrm{pred},i}$ held fixed.
- The **movable set** $\mathcal A^{(n)}=\{i:\chi_{Y,i}=1\}$ is the only set of free variables of Metal-PDHG.

“Locked in the unyielded region” means that the solid substrate undergoes no flow displacement. Deposition is the accumulation of landslide material upon that substrate. It is not a motion of the substrate. Locking the deposition increment as well would destroy conservation of mass.

The symbol $\Delta R_C$ denotes $\Delta h_C^*/(n\cdot e_r)$, where

$$
\Delta h_C^*:=\frac{\rho_{\mathrm{bulk}}}{\rho_K|_\Gamma}\Delta h_C.
$$

The formula above is the **unique definition** of $d_{\mathrm{pred}}$, and it is the same definition as in Section 10.4.3. This paper does not use a displacement field $u$. The incremental displacement is the instantaneous Stokes creep velocity relative to the rigid core, multiplied by the geological step: $\Delta t_n\,\mathbf v_{\rm sl}$.

**Mass-Jacobian weight.** The increment $\Delta h_C$ converts the bulk density $\rho_{\mathrm{bulk}}$ into a thickness. When that thickness is converted into a geometric displacement expressed in the Cauchy density $\rho_K$ of the solid, it must be multiplied by the local density weight $\rho_{\mathrm{bulk}}/\rho_K|_\Gamma$, so that the mass per unit area is conserved:

$$
\rho_{\mathrm{bulk}}\Delta h_C=\rho_K|_\Gamma\Delta h_C^*.
$$

In the Metal-PDHG mass row $D\mathcal M^{(n)}d+s_M=M-\mathcal M^{(n)}$, every geometric increment that comes from a flux uses $\Delta h_C^*$. Otherwise, even if $\int J_{\mathrm{dep}}=\int J_{\mathrm{ero}}$, the inequality $\rho_{\mathrm{bulk}}\ne\rho_K|_\Gamma$ produces a spurious mass drift $s_M\ne 0$.

The unknown $d$ is a radial increment, whereas $\mathbf v_{\rm sl}\cdot n$ and $\Delta h_C^*$ are normal increments. The first-order geometric relation $\delta R\,(n\cdot e_r)=\delta_n$ requires the surface to be star-shaped with respect to the origin and non-grazing, $n\cdot e_r\ge c_0>0$. Otherwise that node must not be parametrized radially. The fluxes $J_{\mathrm{dep}}$ and $J_{\mathrm{ero}}$ are areal mass fluxes, in $\mathrm{kg\,m^{-2}\,s^{-1}}$, so $\Delta h_C$ has the dimension of length. The increment $\Delta R_C$ can come only from the material flux of Mode C. The same flow displacement must not be added a second time. A solid free surface does not in general satisfy $W=\mathrm{const}$. The slope surrogate is used only at those movable nodes of the loose surface layer for which its applicability has been declared, and one must check separately that $g_{\mathrm{eff}}\cdot n<0$. Locked nodes do not receive the slope surrogate. Solid material connected to the core may be steeper than the angle of repose. That is the mechanism that retains real topography.

**(B5) Tensile truncation.** The pressure Poisson problem of Section 8.7 carries no pointwise inequality constraint. The pressure used by the rheology and by the yield test is the truncated regularized value $\tilde p=\max([p]_+,p_{\min})$. The truncated tensile part $\|[p]_-\|$ must be reported at each step as the diagnostic **tensile fraction**. One must not claim that the truncated pressure still satisfies the Poisson equation. In a tensile region the strength degenerates to $\tau_y=c+\mu_y p_{\min}$, the Drucker–Prager tensile truncation. In the absence of cohesion, that region is ordinarily yielded according to $F_\delta$.

**(B6) Plastic diagnostics (no feedback).** The plastic strain rate $\mathbf D_p:=\mathbf 1_{\{F_\delta>0\}}\mathbf D$ in the yielded region, the plastic dissipation

$$
\Phi_p:=\int_{\mathcal V}\mathbf 1_{\{F_\delta>0\}}\boldsymbol\tau_{\rm flow}:\mathbf D\,dV\ge 0,
$$

and the accumulated equivalent plastic strain on $\mathcal X_Y$,

$$
\bar\varepsilon_p\leftarrow\bar\varepsilon_p+\lambda\Delta t_n\,\dot\gamma\,\mathbf 1_{\{F_\delta>0\}},
$$

are diagnostic outputs of post-processing only. They **do not enter** $\boldsymbol\sigma$, $\eta$, $F_\delta$, $d_{\rm pred}$, or any acceptance criterion. Nonnegativity of $\Phi_p$ follows from $\boldsymbol\tau_{\rm flow}=2\eta\mathbf D$ and $\eta\ge 0$, because $\boldsymbol\tau_{\rm flow}:\mathbf D=2\eta|\mathbf D|^2\ge 0$.

The closed interface of Mode C is as follows. Set $\chi_{A,i}=\chi_{Y,i}\cdot\mathbf 1_{\{\rho_{s,i}>0\}}$, the indicator of a node that is movable and carries loose material. The erosional flux is driven by the yield excess at the surface point:

$$
\partial_t(J_\Gamma\rho_s)+\partial_a(J_\Gamma\rho_s c^a)
=J_\Gamma(J_{\mathrm{dep}}-J_{\mathrm{ero}}).
$$

The erosional flux uses the configured erosion coefficient $k_e\ge 0$:

$$
J_{\mathrm{ero}}=\chi_A\,k_e\rho_s\frac{[F_\delta]_+}{\sigma_0}\quad(\rho_s\ \text{is the loose-layer areal density in kg·m}^{-2},\ \text{with initial value }\rho_s^{0}=\rho_{\mathrm{bulk}}h_s,\ k_e\ \text{in s}^{-1}),
$$

$$
J_{\mathrm{dep}}(a)=\int_{\Gamma}K(a,a')J_{\mathrm{ero}}(a')J_\Gamma(a')\,da',
$$

where $\sigma_0>0$, $K\ge 0$, and

$$
\int_{\Gamma}K(a,a')J_\Gamma(a)\,da=1.
$$

The value of $F_\delta$ is the value computed from the unique stress field at the surface point $(1,\hat{\boldsymbol\omega}_i)$, so a locked node has $J_{\rm ero}=0$. In the absence of an external material source one therefore has the exact balance

$$
\int_{\Gamma}J_{\mathrm{dep}}J_\Gamma\,da
=\int_{\Gamma}J_{\mathrm{ero}}J_\Gamma\,da.
$$

*Proof of the balance.* Integrate the definition of $J_{\mathrm{dep}}$ and apply the normalization of $K$ in the variable $a$. An external source must be added explicitly as $J_{\mathrm{ext}}$ and included in the total mass budget. It must not be hidden inside the deposition term. The mass target is then updated synchronously by

$$
M^{n+1}=M^n+\Delta t_n\int_{\Gamma}J_{\mathrm{ext}}J_\Gamma\,da.
$$

Only this one conservation equation produces $\Delta R_C$. The flow displacement of Mode B and the flux of Mode C must not be counted twice. On the active region the coordinate velocity on the surface is

$$
c^a:=\chi_A\,(\mathbf v_{\rm sl}|_\Gamma)_S^a/R,
$$

the restriction to the active set of the surface parameter velocity $\mathbf c=\mathbf v_S/R$ of Proposition 9.1 in Section 9.1, taken relative to the rigid core. The unprojected three-dimensional velocity must not be substituted directly into the surface conservation law. The field $\rho_s$ is only an areal density of the loose layer, kept for accounting. Its tangential transport does not itself change $R$. The radius $R$ is updated only by the net flux $J_{\rm dep}-J_{\rm ero}$, through $\Delta h_C^*$. Transport of $\rho_s$ uses the semi-Lagrangian scheme of Section 9.2, including the Jacobian correction. Positivity is guaranteed by the step restriction

$$
\Delta t_n k_e\max[F_\delta]_+/\sigma_0\le\theta_C<1
$$

of Section 10.4.1. Under the parametrization by $S^2$, one has $J_\Gamma\,d\theta\,d\varphi=j\,d\Omega$, so the conservation equation is equivalent to a form with density $j$. The implementation should use $j$, which avoids the coordinate singularity at the poles.

---

## 2.9 SCF iteration driven by the unified formulae

Let the shape at geological step $n$ be $V^{(n)}$, and let the exogenous spin be $\omega=\omega_n=\omega(t_n)$ (Section 10.4.1). In every subsequent formula of this chapter, $\omega$ means that frozen value.

## 2.10 Forward evaluation

At each surface node $x_i^{(n)}$, compute

$$
U_i^{(n)},
\qquad
\nabla U_i^{(n)},
\qquad
\nabla^2U_i^{(n)}
$$

from the unified Feynman representation. Near the surface, use subtraction of the spherical finite part. Then compute

$$
W_i^{(n)}
=
U_i^{(n)}
+
\frac12\omega^2
\left[
(x_{i1}^{(n)})^2+(x_{i2}^{(n)})^2
\right],
$$

$$
g_i^{(n)}
=
\nabla U_i^{(n)}
+
\omega^2
(x_{i1}^{(n)},x_{i2}^{(n)},0).
$$

---

## 2.11 Fluid mass-level interface (Mode A, retained but not enabled)

<!-- RESERVED_FLUID_INTERFACE: The mass-level interface and its mathematical definition are retained for a future fluid simulation that enables them explicitly. Modes B and C do not call them. An implementation or a cleanup must not delete them merely because they are unused at present. -->

The formulae below define only the fluid interface of Mode A. Modes B and C constrain $\mathcal M(V,\rho)=M$ directly on the actual material domain. They do not select a solid boundary by a level of the potential, and they do not depend on successful root-finding for this interface.

Define

$$
M^{(n)}(W_0)
=
\int_{\{x:W^{(n)}(x)>W_0\}}
\rho_K(x)\,dx.
$$

Solve

$$
M^{(n)}(W_0^{(n)})=M.
$$

If $M^{(n)}$ is continuous and strictly decreasing on the target interval, bisection may be used. Strict decrease requires the region between the target levels to have positive volume and the density on that region to be nonzero.

---

## 2.12 Fluid equipotential interface (Mode A, retained but not enabled)

<!-- RESERVED_FLUID_INTERFACE: R_eq is an interface output for a future fluid simulation. The present granular objective and iteration must not refer to it. The interface is retained and is not deleted as unused code. -->

For each direction $u_i$, solve

$$
W^{(n)}(R_{\mathrm{eq},i}^{(n)}u_i)
=
W_0^{(n)}.
$$

The equation may be executed only when a root exists in the chosen radial search interval. Monotone bisection requires, in addition, that $R\mapsto W^{(n)}(Ru_i)$ be strictly monotone on that interval. If there are several roots, the selection rule must be fixed in advance, and the branch must be continued. The resulting candidate equipotential radius is

$$
R_{\mathrm{eq}}^{(n)}
=
\left(
R_{\mathrm{eq},1}^{(n)},\ldots,R_{\mathrm{eq},N}^{(n)}
\right).
$$

This step produces only a fluid reference shape. It is not a solid target of Mode B or Mode C. A locally smooth equipotential follows from the implicit-function theorem only when the potential has the corresponding regularity and $\nabla W\ne 0$. At a polyhedral boundary, and at a critical point, one cannot assert unconditionally that the surface is globally of class $C^\infty$ or analytic.

---

## 2.13 Constraint correction after the mechanical prediction (Mode B primary, Mode C auxiliary)

Freeze $g^{(n)}$ and $H_W^{(n)}$. First obtain the flow prediction relative to the rigid core from the quasi-static Stokes creep solution of Mode B and from the physical yield classification (B1–B4 of Section 2.8). Only yielded surface points in the movable set are passed to Mode C for conservative erosion and deposition. The hand-off must be explicit. The same flow displacement, or the same material flux, must not be counted twice. Write $d_{\mathrm{pred}}^{(n)}$ for the predicted radial increment, and set $R_{\mathrm{pred}}=R^{(n)}+d_{\mathrm{pred}}^{(n)}$. This prediction contains neither $R_{\mathrm{eq}}$ nor a global equipotential blending term.

**Geometric reference: the current shape is held.** The reference of the subproblem is the **current geometry** $R^{(n)}$ itself. The only change that is allowed is the mechanically predicted increment $d_{\rm pred}^{(n)}$, which vanishes identically on the locked set. Partition the nodes, by (B4) of Section 2.8, into the locked set $\mathcal L^{(n)}$, the pure-deposition set $\mathcal D^{(n)}$, and the movable set $\mathcal A^{(n)}$. The protected region is $\mathcal P^{(n)}:=\mathcal L^{(n)}$. Given weights $w_i>0$, the Metal-PDHG subproblem has the objective

$$
d_*^{(n)}=\arg\min_d\frac12\sum_{i\in\mathcal A^{(n)}}w_i
\left(d_i-d_{{\mathrm{pred}},i}^{(n)}\right)^2,
\qquad R_*^{(n)}=R^{(n)}+d_*^{(n)}.
$$

This is the weighted deviation

$$
\|(R^{(n)}+d)-(R^{(n)}+d_{\rm pred})\|_w^2
$$

from the current shape plus the mechanical increment, with the current shape as the origin. The objective **must not** contain $R_{\mathrm{eq}}$, $W_0$, or any equipotential term. An equipotential is the shape of a fluid with no shear strength. Using it as a target would erase unyielded topography into a smooth ellipsoid. Nodes that are not movable are fixed by hard constraints:

$$
d_i=0\qquad(i\in\mathcal L^{(n)}),\qquad
d_i=d_{\mathrm{pred},i}^{(n)}=\frac{\Delta h_{C,i}^*}{n_i^{(n)}\cdot e_r}\qquad(i\in\mathcal D^{(n)}).
$$

The outer nonlinear acceptance test uses

$$
\frac{|g_i^{(n)}\times n_i(R^{(n)}+d)|^2}{|g_i^{(n)}|^2}
\le \sin^2\phi_i^{(n)},
$$

only at those nodes of the movable set $\mathcal A^{(n)}$ for which the loose-layer surrogate has been declared applicable. Locked nodes are not constrained by the slope surrogate. It also uses

$$
\mathcal M(R^{(n)}+d,\rho_{\mathrm{cand}})=M.
$$

Metal-PDHG receives only the first-order model at the current iterate. Let $n_i^{(n)}$ be the outward unit normal, and define

$$
\widehat n_i(d)=n_i^{(n)}+Dn_i^{(n)}d,
\qquad
\widehat{\mathcal M}(d)=\mathcal M^{(n)}+D\mathcal M^{(n)}d.
$$

Inside the trust region $\|d\|\le\Delta$, the subproblem uses

$$
\frac{\|g_i^{(n)}\times\widehat n_i(d)\|}{|g_i^{(n)}|}
\le\sin\phi_i^{(n)}+t_i,
\qquad
g_i^{(n)}\cdot\widehat n_i(d)\le-\epsilon_n|g_i^{(n)}|,
\qquad
D\mathcal M^{(n)}d+s_M=M-\mathcal M^{(n)},
$$

where $t_i\ge 0$ and $s_M\in\mathbb R$ are elastic slack variables, and $\epsilon_n\in(0,1)$ is the anti-overhang margin. The anti-overhang row is linear and does not destroy convexity. The mass constraint is equivalent to $\widehat{\mathcal M}(d)=M$ when $s_M=0$. It **must not** be written $\widehat{\mathcal M}(d)=M-\mathcal M^{(n)}$, which would subtract $\mathcal M^{(n)}$ twice and reverse the sign of the increment. The objective is changed accordingly to

$$
\frac12\sum_{i\in\mathcal A}w_i(d_i-d_{\mathrm{pred},i})^2+\varrho\Big(\sum_it_i+|s_M|\Big),
$$

where the exact-penalty factor $\varrho>0$ is taken large enough.

If the anti-overhang row holds at $d=0$, that is, if $g_i\cdot n_i<-\epsilon_n|g_i|$, then $d=0$, $s_M=M-\mathcal M^{(n)}$, and $t_i=\max(0,\cdot)$ give a feasible point. This point does not automatically provide a Slater strictly feasible interior point, because the positive-part construction may place a second-order-cone constraint on its boundary. If a Slater certificate is required, every second-order-cone slack must be given an additional prescribed positive margin $\delta_t>0$, one must take $t_i=\max(0,\cdot)+\delta_t$, and every linear constraint that has not been relaxed must be checked at the same time. The anti-overhang row has no slack variable. If $d=0$ violates that row, then $d=0$ is infeasible. The point must be repaired, along a geometrically verifiable direction, or removed, in the outer geometric preprocessing, before the SOCP is entered. The statement $t_i=s_M=0$ means only that the slack variables vanish. The outer acceptance test must still check the true constraints.

These are genuine second-order-cone constraints. They become a quadratic program only if the norm constraints are linearized further. The vector $\widehat n$ is not treated as a unit normal. Shrinking its norm is not a way to satisfy the physical constraint. A candidate solution must have its true normal renormalized, and it must be checked again against the outer nonlinear constraints. The original normal $n(R+d)$, the true mass, and the true yield condition belong only to the outer acceptance test. This geometric correction does not replace the solution of stress balance and of the constitutive law. A tensorial yield certificate must change with the candidate mechanical state. Passage or rejection at a frozen constant tensor cannot be called a geometrically coupled yield constraint.

Fields are frozen inside the subproblem. After the outer update, the candidate geometry, density, and stress are re-evaluated by a forward gravitational solve and checked again. If the current state is already balanced, satisfies the constraints, and has $d_{\mathrm{pred}}=0$, then $d=0$ is the unique minimizer of a strictly positive-weight objective. Topography therefore does not drift in the absence of an external cause.

Because the elastic form above is always feasible on an overhang-free geometry after preprocessing, the logic “if infeasible, shrink the trust region and retry” is not used. When the current point itself violates a constraint, $d=0$ is infeasible, and as $\Delta\to 0$ the feasible set tends to empty, so that strategy necessarily fails. **Unified trust-region policy.** The inner SOCP never shrinks $\Delta^{(n)}$. The outer level shrinks $\Delta^{(n)}$, and regenerates the subproblem, only when the outer nonlinear geometric acceptance test finds an overhang or a self-intersection of the mesh. Every flux increment in the mass row is taken to be $\Delta h_C^*=(\rho_{\mathrm{bulk}}/\rho_K|_\Gamma)\Delta h_C$ (Section 2.8). If the optimal $t_i$ or $s_M$ is nonzero, the linearized model can satisfy the constraints only partially. The outer level decides acceptance by a filter or a merit function, and it may recompute the mechanical prediction. It must not release the protected region or fall back to a fluid target. If the protected region completely surrounds a violating node, or if a signed $\rho_K$ makes $D\mathcal M$ degenerate, then $s_M\neq 0$ is unavoidable and must be reported as such.

Freezing $g$ ignores $Dg\cdot d$. The iteration is therefore of sequential-convex (Picard) type, not a strict SQP iteration, and local superlinear convergence is not claimed. The linearization error is $O(\Delta^2)$, and the outer acceptance test must set a tolerance. The derivative $Dn_i^{(n)}$ in $\widehat n_i$ must be the Jacobian of the **normalized** unit normal. Then $n\cdot Dn\,d=0$ and $|\widehat n|\ge 1$.

---

## 2.14 Under-relaxation

Take $0<\lambda\le 1$ and define

$$
R_i^{(n+1)}
=
(1-\lambda)R_i^{(n)}
+
\lambda R_{*,i}^{(n)}.
$$

Rebuild the new polyhedron $V^{(n+1)}$ from $R^{(n+1)}$. On the locked set, $d_{*,i}=0$, so under-relaxation does not move a locked node. The physical time advanced by under-relaxation is $\lambda\Delta t_n$ (Section 10.4.1). After under-relaxation one must recompute gravity on the new geometry and resolve the quasi-static Stokes problem and the pressure. Stress is an instantaneous equilibrium quantity; there is no material history to update. The loose-layer density $\rho_s$ is updated by the conservation law of Mode C. Nonlinear mass and yield constraints are not guaranteed to hold along the line segment. Mass must not be repaired by a global scaling of the radius, or by a uniform increment, in such a way that the protected region moves. The conservation correction is performed inside the movable set. If it fails, the step is rejected and $\Delta t_n$ is reduced. The trust region is reduced, according to Section 2.13, only when the outer geometric acceptance test finds an overhang or a self-intersection.

---

## 2.15 Convergence criteria

The quasi-static route has three layers of criteria. They are independent and must not be substituted for one another. The algorithm is given in Section 10.4.4.

**(a) Convergence of the rheological nonlinearity (Picard, level $k$).** On a fixed geometry and at a fixed $\omega_n$, the Stokes–pressure Picard iteration of Section 10.4.2 stops if and only if

$$
\frac{\|\mathbf u^{k+1}-\mathbf u^{k}\|_{\mathbf K}}{\max\{\|\mathbf u^{k+1}\|_{\mathbf K},\varepsilon_u\}}\le\varepsilon_{\rm pic},\qquad
\frac{\|p^{k+1}-p^{k}\|_{L^2}}{\max\{\|p^{k+1}\|_{L^2},\varepsilon_p\}}\le\varepsilon_{\rm pic},
$$

where $\|\mathbf u\|_{\mathbf K}^2=\mathbf u^{\!\top}\mathbf K(\eta^{k})\mathbf u$. If the iteration limit $k_{\max}$ is reached without satisfaction of the test, retry by the continuation in $\dot\gamma_\varepsilon$ and by the viscosity relaxation of Section 10.4.2. That solve is independent of $\Delta t_n$; reducing the step does not help. If the retry still fails, the step is judged **rheology not converged** and is terminated. A passing status must not be emitted.

**(b) Acceptance of mechanical balance (each geological step; a hard criterion).** On the converged pair $(\mathbf u,p)$,

$$
\varepsilon_{\rm mom}^{\rm weak}:=\frac{\|\mathbf F-\mathbf K(\mathbf u;\tilde p)\mathbf u-\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda\|_2}{\max\{\|\mathbf F\|_2,\varepsilon_F\}}\le\varepsilon_{\rm mom},\qquad
\varepsilon_\lambda:=\frac{\|\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda\|_2}{\max\{\|\mathbf F\|_2,\varepsilon_F\}}\le\varepsilon_{\lambda}^{\rm tol}.
$$

The first quantity is the Galerkin momentum residual of the true fluid system, with its unique stress field. The second measures incompatibility of the external load with the rigid-body modes, caused by a misaligned frame or by a missing discrete rigid mode (Section A.3). The relative norm $\|\mathbf r_{\rm rot}\|_{L^2}/\max\{\|\rho_K\mathbf g_{\rm eff}\|_{L^2},\varepsilon_f\}$ of the strong residual $\mathbf r_{\rm rot}=\mathbf f-\nabla p$, and the tangential traction deviation $\delta_{na}$, are indicators of discrete resolution. They converge as $K$ and $L_{\max}$ increase, and they must be reported at each step. If they exceed a configured threshold, the step is marked **insufficient resolution**. That mark does not rewrite the decision (b).

**(c) Stage equilibrium (when the exogenous spin is constant on a stage).** During a stage of constant $\omega_j$, the geometry has reached static equilibrium if and only if

$$
\boxed{\ \mathcal A^{(n)}=\varnothing\quad(\text{every surface point belongs to the rigid core }\mathcal U_{\rm core}),\qquad\text{and (b) holds}. \ }
$$

Then $F_\delta\le 0$ at every surface point, so $J_{\rm ero}=0$, $J_{\rm dep}=0$, $\mathcal D^{(n)}=\varnothing$, and $d_{\rm pred}=0$. The SOCP has no free variable, and $d_*=0$ holds exactly. This state is independent of $\Delta t_n$ and of $\lambda$. Because no state variable integrates $\mathbf D$, the state is a genuine fixed point, not a slowly drifting creep state. Regularized creep still exists inside the core, but it is removed exactly by $\chi_Y=0$. When $\mathcal A^{(n)}\ne\varnothing$, so that a genuine yielded flow exists, the size of $d_*$ is not a meaningful convergence test: the adaptive step $\Delta t_n$ makes the geometric displacement of each step about $\theta_{\rm CFL}h_\Gamma$. If the stage duration $T_j$ is exhausted while $\mathcal A^{(n)}\ne\varnothing$, the stage ends as **not equilibrated**, and the movable area fraction

$$
\sum_{i\in\mathcal A}j_iw_i\Big/\sum_ij_iw_i
$$

is reported. That outcome is a physical result — the landslide has not stopped in the allotted time — and not a numerical failure. Under a continuously varying exogenous spin $\omega(t)$ there is no stage-equilibrium target. One performs only (a) and (b), and advances to the terminal time.

At the same time, check the slope residual of the applicable surface layer, at movable nodes only,

$$
\max_{i\in\mathcal A^{(n)}}
\left[
\frac{|g_i^{(n)}\times n_i^{(n)}|}
{|g_i^{(n)}|}
-\sin\phi_i^{(n)}
\right]_+
<\varepsilon_{\mathrm{slope}},
$$

and the mass residual

$$
|\mathcal M(V^{(n)})-M|
<\varepsilon_M.
$$

Finally recompute

$$
U,\qquad
\nabla U,\qquad
\nabla^2U
$$

on the terminal geometry, and, on that same final geometry, density, and $\omega$, recheck (b), the yield classification, and the vanishing of the displacement on the locked set. Passage of the KKT conditions of the subproblem is not passage of the nonlinear mechanical loop. A step that has become small because the trust region has contracted is not convergence. Only after the final audit has passed are the effective potential, the effective gravity, and the Hessian emitted.

---

## 2.16 The unified coupled system

After the unified computational formulae enter the granular SCF, the final mathematical loop is

$$
\boxed{
\begin{aligned}
U(x)
={}&
G\sum_{k=1}^K
w_k\sigma_k^{-2\alpha_k}C_{\alpha_k}
\int_0^1
u^{-1/2}(1-u)^{\alpha_k-1}\\
&\qquad\times
\int_V
\left[
|y-z_k(x,u)|^2+m_k^2(x,u)
\right]^{-\alpha_k-\frac12}
\,dy\,du,
\end{aligned}
}
$$

$$
\boxed{
\nabla W=\nabla U+\omega^2(x_1,x_2,0),
}
$$

$$
\boxed{
\nabla^2W
=
\nabla^2U
+
\omega^2
\begin{pmatrix}
1&0&0\\
0&1&0\\
0&0&0
\end{pmatrix},
}
$$

The field equation of Mode B, the applicable surface slope surrogate, and the mass constraint may be written

$$
\boxed{
\begin{cases}
\Delta W
=
2\omega^2-4\pi G\rho_K\mathbf1_V,
\\[4pt]
|\nabla W\times n|^2
\le
\sin^2\phi_i^{(n)}\,|\nabla W|^2,
\\[4pt]
\displaystyle\int_V\rho_K\,dx=M
\end{cases}
}
$$

The slope inequality in this box holds only at those nodes of the loose surface layer that lie in the movable set and for which the surrogate has been declared applicable. It is not a sufficient condition for balance of the solid body. Together with the quasi-static Stokes balance of (B1) in Section 2.8 — the unique stress $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D$ — with the free-surface traction, and with the yield classification and the locking of the rigid core in (B2–B3), it forms the coupled system solved in Mode B. The spin $\omega$ is an exogenous datum (Section 10.4.1), not an unknown. Mode C supplies the local conservative flow prediction. The Mode A interface does not enter this loop.

A fast algorithm aimed at the GPU, at RayBackend, and at batches of field points is given below. Its central idea is the following.

- Geometric intersections use a WGSL bounding-volume hierarchy, with hardware ray tracing as an option.
- The quantities $U$, $\nabla U$, and $H$ are computed together on the same ray.
- One batch of rays shares all geometric results.
- The case $\alpha=1$ uses an analytic radial antiderivative.
- Every other $\alpha$ uses Gauss–Jacobi quadrature with a fixed set of nodes.
- The near-surface region uses subtraction of the finite part.
- The far field may switch to a GPU multipole expansion, which reduces the number of rays.

Hardware ray tracing is responsible only for ray–triangle intersection. It does not evaluate a Feynman parameter integral. In the absence of hardware ray tracing, the same interface uses a WGSL bounding-volume hierarchy. The most effective structure is therefore

$$
\boxed{
\text{RTFP ray geometry}
+
\text{GPU radial integration}
+
\text{GPU reduction}
+
\text{far-field FMM/multipole acceleration}.
}
$$

# 3. GPU ray integration and hybrid acceleration

## 3.1 The unified ray formula

For each field point $x$ and each spherical direction $u_q$, set

$$
y=x-Ru_q.
$$

The $k$-th density kernel is

$$
D_k(R)
=
(R-b_k)^2+d_k^2,
$$

where

$$
b_k=(x-c_k)\cdot u_q,
$$

$$
d_k^2
=
a_k^2+|x-c_k|^2-b_k^2,
\qquad
a_k=\sigma_k^{-1}.
$$

The three basic radial integrals along one ray are

$$
I_U^{(k)}
=
\int_{R^-}^{R^+}
R\,D_k(R)^{-\alpha_k}\,dR,
$$

$$
I_g^{(k)}
=
\int_{R^-}^{R^+}
D_k(R)^{-\alpha_k}\,dR,
$$

$$
I_H^{(k)}
=
\int_{R^-}^{R^+}
\frac{D_k(R)^{-\alpha_k}}{R}\,dR.
$$

They arise by inserting $dy=R^2\,dR\,d\Omega$ and $x-y=Ru$ into the Newtonian kernels. The potential picks up one factor of $R$ from $R^2/R$, the gradient picks up none from $R^2/R^2$, and the Hessian picks up $R^{-1}$ from $R^2/R^3$ after the angular factor $T(u)$ has been removed. Consequently

$$
\boxed{
U(x)
=
G\sum_{q,k}
w_k\sigma_k^{-2\alpha_k}
\omega_q I_U^{(k)}.
}
$$

The gradient is

$$
\boxed{
\nabla U(x)
=
-G\sum_{q,k}
w_k\sigma_k^{-2\alpha_k}
\omega_q u_q I_g^{(k)}.
}
$$

The Hessian is

$$
\boxed{
H(x)
=
G\sum_{q,k}
w_k\sigma_k^{-2\alpha_k}
\omega_q
T(u_q)I_H^{(k)},
}
$$

where

$$
T(u)=3uu^T-I
$$

is the matrix whose entries are $T_{ij}(u)$ of Section 1.3, and $\omega_q$ is the spherical quadrature weight of direction $u_q$, not the spin. The three quantities share

- the same direction $u_q$;
- the same ray-intersection segment $[R^-,R^+]$;
- the same coefficients $b_k,d_k$;
- the same density parameters.

Three separate integration programs should therefore not be run.

---

## 3.2 Analytic antiderivatives for $\alpha=1$

When $\alpha_k=1$, define

$$
D(R)=(R-b)^2+d^2.
$$

The potential integral has the antiderivative

$$
\boxed{
P(R)
=
\frac12\log D(R)
+
\frac{b}{d}
\arctan\frac{R-b}{d}.
}
$$

*Differentiation.* One has $D'=2(R-b)$, so

$$
P'(R)
=
\frac{R-b}{D(R)}
+
\frac{b}{(R-b)^2+d^2}
=
\frac{R-b+b}{D(R)}
=
\frac{R}{D(R)}.
$$

Thus $\int R\,D^{-1}\,dR=P(R)$ up to a constant. The gradient integral has the antiderivative

$$
\boxed{
Q(R)
=
\frac1d\arctan\frac{R-b}{d},
}
$$

and $Q'(R)=1/D(R)$. The radial integral of the Hessian,

$$
L(R)=\int \frac{dR}{R[(R-b)^2+d^2]},
$$

is computed by partial fractions. Set

$$
A_0=b^2+d^2.
$$

Then

$$
\frac1{R[(R-b)^2+d^2]}
=
\frac1{A_0}\frac1R
+
\frac{-R+2b}{A_0[(R-b)^2+d^2]}.
$$

*Verification.* The common-denominator numerator is

$$
(R-b)^2+d^2+R(-R+2b)
=
R^2-2bR+b^2+d^2-R^2+2bR
=
A_0.
$$

Integration of the second term uses

$$
\int\frac{-R+2b}{(R-b)^2+d^2}\,dR
=
-\frac12\log D(R)
+
\frac{b}{d}\arctan\frac{R-b}{d},
$$

because $-R+2b=-(R-b)+b$ and $\int(R-b)/D\,dR=\tfrac12\log D$. Therefore

$$
\boxed{
\begin{aligned}
L(R)
={}&
\frac1{A_0}\log R
-\frac1{2A_0}\log\left((R-b)^2+d^2\right)\\
&+
\frac{b}{A_0d}
\arctan\frac{R-b}{d}.
\end{aligned}
}
$$

As recorded in Section 1.6, this $L$ differs from the antiderivative written there by the constant factor that converts $1+\sigma^2|x-Ru-c|^2$ into $\sigma^2 D(R)$. For an ordinary exterior ray,

$$
I_U=P(R^+)-P(R^-),
$$

$$
I_g=Q(R^+)-Q(R^-),
$$

$$
I_H=L(R^+)-L(R^-).
$$

The dominant cost of each ray is then a small number of logarithms, arctangents, and multiply-adds. A segment with $R^-=0$ must not use $L(R^+)-L(R^-)$ directly. It uses the finite-part limit of Sections 1.6 and 3.9.

---

## 3.3 GPU quadrature for general $\alpha$

When $\alpha\neq 1$, do not run an adaptive integrator on the GPU. Use Gauss–Jacobi or Gauss–Legendre quadrature with a fixed set of nodes. Set

$$
R(t)=\frac{R^++R^-}{2}
+
\frac{R^+-R^-}{2}t,
\qquad -1\le t\le 1.
$$

At the $n_r$ fixed nodes $t_j$, compute simultaneously

$$
D_j=(R(t_j)-b)^2+d^2.
$$

Then

$$
I_U
\approx
\frac{R^+-R^-}{2}
\sum_{j=1}^{n_r}
w_j
R_jD_j^{-\alpha},
$$

$$
I_g
\approx
\frac{R^+-R^-}{2}
\sum_{j=1}^{n_r}
w_jD_j^{-\alpha},
$$

$$
I_H
\approx
\frac{R^+-R^-}{2}
\sum_{j=1}^{n_r}
w_j\frac{D_j^{-\alpha}}{R_j}.
$$

These ordinary quadrature formulae apply only to a nonsingular segment with $R^->0$. If the field point lies inside the solid and the segment starts at $R^-=0$, or if the segment belongs to a boundary or near-field singular branch, then $I_H$ must not be computed by ordinary Gauss quadrature. Control passes to the finite-part subtraction of Section 3.9, and the contact term is added explicitly.

The implementation is configured as follows.

- When $\alpha$ is fixed, nodes and weights are precomputed.
- The case $\alpha=1$ takes the analytic branch.
- The node counts $n_r=8,16,32$ are three levels of accuracy.
- Only a small number of difficult rays are recomputed at the higher-order nodes.

---

## 3.4 GPU BVH geometric pipeline (optional hardware ray-tracing backend)

## 3.5 Preprocessing

Build a bounding-volume hierarchy for the polyhedral triangle mesh.

1. Store triangle vertices in a structure-of-arrays layout.
2. Precompute face normals, bounding boxes, and edge data.
3. Construct a flat bounding-volume hierarchy that can be uploaded to WGSL. If hardware ray tracing is available, build an equivalent acceleration structure.
4. Reuse the hierarchy while the shape of the body is fixed.

Triangle data may be stored as
```text
v0x[], v0y[], v0z[]
v1x[], v1y[], v1z[]
v2x[], v2y[], v2z[]
normal_x[], normal_y[], normal_z[]
```
An array-of-structures layout, which produces uncoalesced memory access, is avoided.

---

## 3.6 Ray tasks

For each field point $x_m$ and each spherical direction $u_q$, cast the ray

$$
r(t)=x_m-tu_q,\qquad t>0.
$$

RayBackend returns every intersection distance

$$
0<R_1<R_2<\cdots<R_L.
$$

For a closed polyhedron:

- an exterior point usually produces intersections in pairs;
- the first intersection of an interior point is an exit, and the interval from $R=0$ to that first intersection also belongs to the solid;
- a nonconvex polyhedron may have several intersection segments.

One must therefore not pair intersections as $[R_1,R_2],[R_3,R_4],\ldots$ for every field point. The occupancy at $R=0$ is determined first from the interior/exterior classification of the field point, and the intervals are then generated by switching state:

$$
\begin{cases}
[R_1,R_2],[R_3,R_4],\ldots,&x\notin V,\\[2pt]
[0,R_1],[R_2,R_3],[R_4,R_5],\ldots,&x\in\operatorname{int}V.
\end{cases}
$$

A boundary point must first be fixed to the spherical representative convention of Sections 1.2–1.3. An implementation may use the interior limit, the exterior limit, or an explicit near-field branch at $R=0$. A boundary point must not be forced unconditionally into either pairing rule.

Do not let each thread allocate memory dynamically. The recommended scheme has two stages, followed by a write.

### Stage A: counting

Each ray only counts the number $L$ of intersections.

### Stage B: prefix sum

A GPU prefix sum over all rays allocates a contiguous intersection buffer.

### Stage C: writing

The ray query is executed again, and the intersections are written into the packed array.

This avoids wasting a large amount of device memory on a buffer of fixed length.

---

## 3.7 Organization of the GPU kernels

The recommended thread mapping is
```text
one WGSL workgroup
    → one batch of field points
    → a fixed number of spherical directions

one invocation
    → one direction
    → every intersection segment of that direction

one workgroup
    → shared parameters of the batch, including x, c_k, and the quadrature weights
```
For each directional thread:

1. Read the segment $[R^-,R^+]$.
2. Compute $u_q$.
3. For each $k$, compute $b_k$ and $d_k^2$.
4. Compute $I_U$, $I_g$, and $I_H$.
5. Accumulate $U$, $\nabla U$, and $H$.

Local accumulations are written to workgroup-local storage and reduced hierarchically after a workgroup barrier. If the device supports subgroup operations, the reduction may be replaced under the same error verification.

---

## 3.8 Fused kernel

Do not call separately
```text
compute_potential()
compute_gradient()
compute_hessian()
```
Use one fused kernel:
```text
for each field point x:
    U_local = 0
    g_local = (0,0,0)
    H_local = zero_matrix

    for each ray direction q:
        intersections = RT_query(x, u[q])

        for each segment [R0, R1]:
            for each density component k:
                compute b, d2

                if alpha[k] == 1:
                    IU = analytic_potential(...)
                    IG = analytic_gradient(...)
                    IH = analytic_hessian(...)
                else:
                    IU, IG, IH = fixed_quadrature(...)

                U_local += coeff * weight[q] * IU
                g_local += -coeff * weight[q] * u[q] * IG
                H_local += coeff * weight[q] * T(u[q]) * IH

    write U_local, g_local, H_local
```
Here

$$
T(u)=3uu^T-I
$$

contains only six independent Hessian components,

$$
H_{xx},H_{yy},H_{zz},H_{xy},H_{xz},H_{yz}.
$$

The nine entries of the matrix need not be computed separately. Symmetry $H_{ji}=H_{ij}$ follows from equality of mixed classical derivatives in the exterior, and from the symmetry $T_{ij}=T_{ji}$ of the finite-part kernel.

---

## 3.9 Near-surface finite part

If the field point is close to the surface, a direct evaluation of $I_H$ produces a logarithmic divergence. Let $\Gamma_x$ be the set of directions of the local tangent cone. Subtract at the near-field radius $r_0$:

$$
\begin{aligned}
I_H^{\mathrm{near}}
={}&
\int_{\Gamma_x}
T(u)
\int_\varepsilon^{r_0}
\frac{\rho(x-Ru)}{R}\,dR\,d\Omega\\
={}&
\rho(x)C(x)\log\frac{r_0}{\varepsilon}\\
&+
\int_{\Gamma_x}T(u)
\int_\varepsilon^{r_0}
\frac{\rho(x-Ru)-\rho(x)}{R}\,dR\,d\Omega.
\end{aligned}
$$

This is the angular form of Section 1.5, before multiplication by $G$ and before the contact term. In the GPU implementation:

1. The part of a ray whose intersection distance satisfies $R<r_0$ enters the near kernel.
2. Compute the density difference $\rho(x-Ru)-\rho(x)$.
3. Apply fixed quadrature to the difference. At a node with $R=0$, use the limit $-\nabla\rho(x)\cdot u$. Do not evaluate $0/0$. The limit is the integrand identity of Section 1.5.
4. Add explicitly

   $$
   G\rho(x)C(x)\log(r_0/\ell_0).
   $$

5. Add the contact term

   $$
   -\frac{4\pi G}{3}\vartheta_x\rho(x)I.
   $$

The divergent term $\log\varepsilon$ is then never computed on the GPU. The matrix $C(x)$ is the spherical moment of $T(u)$ over $\Gamma_x$, and $\vartheta_x$ is the corresponding solid-angle fraction, both as in Section 1.3.

---

## 3.10 Far-field multipole acceleration

When a field point satisfies

$$
\operatorname{dist}(x,V)>r_{\mathrm{switch}},
$$

rays need not be cast at every face. A GPU fast multipole method, or an octree multipole expansion, may be used instead.

Partition the source volume into tree nodes. For a field point far from a node, use a truncated expansion

$$
U(x)
\approx
G\sum_{|\ell|\le p}
\frac{M_\ell}{|x-c_{\mathrm{node}}|^{\ell+1}}.
$$

This formula is only a scalar shorthand for a statement about complexity. A genuine three-dimensional multipole expansion must specify the multi-index, the normalization of the basis functions, the translation operators of the tree, and a remainder bound at truncation order $p$. Without those definitions, the formula is not an implementation formula for the numerical kernel, and the complexity statements below cannot be read as a strict error bound or as a strict runtime guarantee.

The gradient and the Hessian are obtained by recurrence from the same multipole coefficients:

$$
\nabla U
=
\nabla \mathcal M_p,
\qquad
H
=
\nabla^2\mathcal M_p.
$$

Near-field nodes still use RayBackend ray intersection. Far-field nodes use the multipole expansion.

Only under a balanced tree, a fixed acceptance criterion, a fixed expansion order $p$, and a bounded capacity of each leaf, may the total complexity be estimated for planning purposes as follows. The direct method,

$$
O(N_x F),
$$

is reduced to the approximation

$$
\boxed{
O(N_x\log F+N_{\mathrm{source}}p^2)
}
$$

or to the standard fast-multipole estimate

$$
\boxed{
O((N_x+N_{\mathrm{source}})p^2).
}
$$

The default scheme is a hybrid of a GPU bounding-volume hierarchy and a fast multipole method. Hardware ray tracing is only an optional backend:

$$
\boxed{
\text{near field: WGSL BVH (hardware ray tracing optional)}
\qquad
\text{far field: GPU-FMM}
}
$$

---

## 3.11 Hybrid scheduling

For each field point $x$, let $d(x,V)$ be the distance to the nearest geometry. Choose two thresholds

$$
d_{\mathrm{near}}<d_{\mathrm{far}}.
$$

Use

$$
\begin{cases}
d<d_{\mathrm{near}}:&
\text{RTFP + finite part};\\
d_{\mathrm{near}}\le d\le d_{\mathrm{far}}:&
\text{RTFP, full rays};\\
d>d_{\mathrm{far}}:&
\text{GPU FMM/multipole expansion}.
\end{cases}
$$

This is faster than using Feynman face integrals at every field point, because the Feynman route usually still requires

- a parameter integral in $u$;
- a face integral;
- treatment of the endpoint $u\to 1$;
- an additional inner quadrature.

---

## 3.12 GPU execution inside the SCF

**The only authoritative execution procedure of the SCF algorithm is the nine-step loop of Section 10.4.4. This section no longer retains an obsolete description that omitted modal transport and remapping.** It records only the part of that loop which the GPU carries.

- In step 1 of Section 10.4.4, and in the recomputation of step 9, the fields $U$, $\nabla U$, and $H$ are computed by a batched RayBackend intersection together with the fused kernel of Sections 3.6–3.8. The input is the triangle mesh of the current $R^{(n)}$ and the updated bounding-volume hierarchy.
- The effective quantities $W=U+U_c$, $\mathbf g_{\rm eff}$, and $H_{\rm eff}$ are assembled as in Section 2.8.
- The execution requirements of modal transport, mass recovery, the SOCP, and remapping in steps 2–8 are given in Chapters 9 and 10. They are not part of the RTFP kernel.

<!-- RESERVED_FLUID_INTERFACE: W0 and R_eq are reserved for a future Mode A fluid simulation. The present steps do not call them, and they must not be deleted by mistake. -->

---

## 3.13 Complexity

Let

- $N_x$ be the number of field points;
- $N_\Omega$ the number of spherical directions;
- $F$ the number of triangles;
- $K$ the number of density kernels;
- $L$ the average number of intersection segments per ray;
- $n_r$ the number of radial quadrature nodes.

After the bounding-volume hierarchy is used, the near-field complexity of RTFP is approximately

$$
\boxed{
O\left(
N_xN_\Omega
\bigl(\log F+L\bigr)
K n_r
\right).
}
$$

When $\alpha=1$ and the analytic antiderivative is used, $n_r$ may be treated as the constant $1$, and the estimate becomes

$$
\boxed{
O\left(
N_xN_\Omega
(\log F+L)K
\right).
}
$$

The total complexity of the SCF is

$$
\boxed{
O\left(
N_{\mathrm{iter}}
N_xN_\Omega
(\log F+L)K
\right)
}
$$

plus the cost of the geometric update, of the mass calibration, and of the constraint projection. These are planning estimates under the hypotheses stated with the hierarchy and the quadrature. They are not strict runtime certificates.

---

## 3.14 Implementation rules

The fastest practical implementation should use

$$
\boxed{
\text{Lebedev spherical nodes}
+
\text{flat WGSL BVH with multiple intersections}
+
\text{RTFP radial analytic formulae}
+
\text{a fused }U/\nabla U/H\text{ kernel}
+
\text{the near-surface finite part}
+
\text{a far-field GPU-FMM}.
}
$$

The unified Feynman formula is appropriate as

- an independent channel for the exterior field;
- an implementation by face integrals;
- a cross-check of the RTFP result.

It is not appropriate as the fastest primary algorithm at every point.

RayBackend is responsible for geometric intersection. The WGSL compute kernel is responsible for the radial formulae and for accumulation of the tensors. Optional matrix hardware is used only for a reduction whose error has been verified. The critical accumulation of the finite-part Hessian must not be handed entirely to a low-precision path.

The strict conclusion, carried forward into Chapter 4, is the following.

> Under the limit conditions that are listed explicitly, the combined model reduces, strictly and term by term, to the classical $\mu(I)$ rheological model.

If those conditions are not satisfied, the combined model is not equal to the classical $\mu(I)$ law.

A coefficient must be corrected before that reduction is used. If one defines

$$
\dot\gamma=\sqrt{2D':D'},
\qquad
\tau_{\mathrm{eq}}=\sqrt{\frac12\tau:\tau},
$$

then the constitutive relation must be written

$$
\boxed{
\tau
=
2\frac{\mu p}{\dot\gamma}D'
}
$$

in order to obtain

$$
\tau_{\mathrm{eq}}=\mu p,
$$

or equivalently

$$
\boxed{
\tau=2\eta D',
\qquad
\eta=\frac{\mu p}{\dot\gamma}.
}
$$

*Expansion of the coefficient.* With $\tau=2(\mu p/\dot\gamma)D'$,

$$
\tau:\tau
=
4\frac{\mu^2 p^2}{\dot\gamma^2}(D':D')
=
4\frac{\mu^2 p^2}{\dot\gamma^2}\cdot\frac{\dot\gamma^2}{2}
=
2\mu^2 p^2,
$$

because $D':D'=\dot\gamma^2/2$ by the definition of $\dot\gamma$. Therefore

$$
\tau_{\mathrm{eq}}
=
\sqrt{\tfrac12\tau:\tau}
=
\mu p.
$$

The conventions in force are $\tau_{\mathrm{eq}}=\sqrt{\tfrac12\tau:\tau}$ and $\dot\gamma=\sqrt{2D':D'}$. If the factor $2$ in the constitutive relation is omitted, the equivalent stress becomes $\mu p/2$: the same calculation with $\tau=(\mu p/\dot\gamma)D'$ produces $\tau:\tau=\mu^2 p^2/2$ and $\tau_{\mathrm{eq}}=\mu p/2$.

# 4. The combined rheological model and its classical limit

## 4.1 Constitutive convention and the classical $\mu(I)$ model

*Range of application.* The $\mu(I)$ law of this chapter, and the argument that it is recovered in the limit, concern the flow (deviatoric) stress and an incompressible flow velocity ($\nabla\cdot v=0$). The production model (Section 2.8) uses only the viscoplastic stress given in this chapter. It introduces neither an elastic strain nor an equation of state. Pressure is the Lagrange multiplier of the incompressibility constraint, and it is determined by the Poisson problem of Section 8.7.

The classical inertial rheology defines

$$
\dot\gamma=\sqrt{2D':D'},
$$

$$
\boxed{
I=
\frac{d\dot\gamma}{\sqrt{p/\rho_p}},
}
\qquad p>0.
$$

The friction coefficient is

$$
\boxed{
\mu_{\mathrm{cl}}(I)
=
\mu_s+
(\mu_2-\mu_s)
\frac{I}{I_0+I}.
}
$$

In a state of sustained flow,

$$
\boxed{
\tau_{\mathrm{eq}}
=
\mu_{\mathrm{cl}}(I)p.
}
$$

The corresponding deviatoric stress is

$$
\boxed{
\tau
=
2\mu_{\mathrm{cl}}(I)p
\frac{D'}{\dot\gamma}.
}
$$

The coefficient $2$ is required by the normalization fixed at the end of Section 3.14. Explicitly,

$$
\frac12
\left(
2\mu p\frac{D'}{\dot\gamma}
\right):
\left(
2\mu p\frac{D'}{\dot\gamma}
\right)
=
\frac{4\mu^2p^2}{2\dot\gamma^2}
(D':D'),
$$

and $D':D'=\dot\gamma^2/2$, so the right-hand side equals $\mu^2 p^2$. Therefore

$$
\tau_{\mathrm{eq}}=\mu p.
$$

---

## 4.2 The combined model

The combined model contains four extensions: a cohesive pressure, a modified inertial number, a state-dependent friction coefficient, and a tidal backstress.

## 4.3 Cohesive pressure

Define

$$
p_c
=
C_c\frac{Z\nu F_H}{d^2}.
$$

Here $Z$ is the coordination number and $F_H$ is the cohesive force of a single contact. Define the effective pressure

$$
\boxed{
p_*=p+p_c.
}
$$

## 4.4 Modified inertial number

$$
\boxed{
I_c
=
\frac{d\dot\gamma}
{\sqrt{p_*/\rho_p}}.
}
$$

The formula is the classical inertial number with $p$ replaced by $p_*$. It requires $p_*>0$.

## 4.5 State-dependent friction coefficient

The dynamic friction coefficient is

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

## 4.6 Tidal backstress

Let

$$
H_W=\nabla^2W,
$$

and define

$$
\boxed{
T_{\mathrm{tide}}
=
C_T\rho_p\nu d^2
\operatorname{dev}H_W.
}
$$

The deviatoric projection $\operatorname{dev}H_W=H_W-(\operatorname{tr}H_W)I/3$ makes $T_{\mathrm{tide}}$ traceless. The combined yield function is

$$
\boxed{
F_{\mathrm{comb}}
=
\sqrt{
\frac12
(s-T_{\mathrm{tide}}):(s-T_{\mathrm{tide}})
}
-
c(\nu,Z)
-
\mu_y(\nu,Z)p.
}
$$

Here $s:=\operatorname{dev}\sigma$ is the deviator of the total stress. In order that the tidal backstress be consistent with the dynamic constitutive law, define the flow deviator $\tau_{\mathrm{flow}}$ and set

$$
s=T_{\mathrm{tide}}+\tau_{\mathrm{flow}}.
$$

The effective stress in the yield function is therefore

$$
s-T_{\mathrm{tide}}=\tau_{\mathrm{flow}}
$$

identically, not merely in a limit. The dynamic flow branch uses

$$
\boxed{
\tau_{{\mathrm{flow}},\mathrm{eq}}
=
c(\nu,Z)+\mu_d(I_c,\nu,Z)p,
\qquad
\tau_{\mathrm{flow}}=2\eta D'.
}
$$

The scalar target and the tensor law are related by the normalization of Section 3.14 once $\eta$ is chosen so that $\|2\eta D'\|_{\mathrm{eq}}$ equals the scalar target. Under the regularizations of Section 4.17 that equality holds only in the limit $\dot\gamma_\varepsilon/\dot\gamma\to 0$.

---

## 4.7 Conditions for reduction

Consider a family of parameters and states of the combined model,

$$
\left(
p_c^{(n)},
H_W^{(n)},
c^{(n)},
\nu^{(n)},
Z^{(n)}
\right).
$$

Assume that $p>0$ and $\dot\gamma>0$ in the region under consideration, and that the following conditions hold. The index $n$ is the index of this parameter sequence. It is not the index of an SCF iteration. In the production configuration of Sections 10.3 and 10.4, the solid fraction and the coordination number are frozen material constants; the sequence below is the hypothesis of the reduction theorem, not an evolution inside the SCF.

## 4.8 Condition A: disappearance of cohesion in the pressure

$$
\boxed{
\frac{p_c^{(n)}}p\longrightarrow0.
}
$$

Then

$$
p_*^{(n)}=p+p_c^{(n)}
\longrightarrow p,
$$

because $p_c^{(n)}=p\cdot(p_c^{(n)}/p)$ and $p$ is held at a positive value along the sequence, or tends to a positive limit as required in Section 4.20.

## 4.9 Condition B: disappearance of the tidal stress

$$
\boxed{
\frac{
\rho_p\nu^{(n)}d^2
\|\operatorname{dev}H_W^{(n)}\|
}{p}
\longrightarrow0.
}
$$

Then

$$
T_{\mathrm{tide}}^{(n)}
=
C_T\rho_p\nu^{(n)}d^2\operatorname{dev}H_W^{(n)}
\longrightarrow 0,
$$

since $\|T_{\mathrm{tide}}^{(n)}\|=|C_T|\,p\cdot\bigl(\text{displayed quotient}\bigr)$, where $C_T$ is a fixed coefficient, the quotient tends to zero, and $p$ is bounded. Any matrix norm equivalent to the Frobenius norm may be used; the limit zero is independent of that choice.

## 4.10 Condition C: frozen state variables

$$
\boxed{
\nu^{(n)}\to\nu_0,
\qquad
Z^{(n)}\to Z_0.
}
$$

## 4.11 Condition D: constitutive parameters reduce to constants

Require

$$
\mu_s(\nu^{(n)},Z^{(n)})
\to\mu_s,
$$

$$
\mu_2(\nu^{(n)},Z^{(n)})
\to\mu_2,
$$

$$
I_0(\nu^{(n)},Z^{(n)})
\to I_0.
$$

The yield coefficient used in Section 4.16 and in the theorem of Section 4.20 is required to satisfy

$$
\mu_y(\nu^{(n)},Z^{(n)})\to\mu_s
$$

(in production $\mu_y\equiv\mu_s(\nu_0,Z_0)$, Section 2.8, B2, and this holds trivially).

## 4.12 Condition E: disappearance of cohesive strength

$$
\boxed{
\frac{c(\nu^{(n)},Z^{(n)})}{p}
\longrightarrow0.
}
$$

## 4.13 Condition F: disappearance of pressure regularization

If a pressure regularization with threshold $p_\varepsilon$ is used, so that the regularized pressure is $\tilde p=\max([p]_+,p_\varepsilon)\ge p$ (in production $p_\varepsilon=p_{\min}$), require

$$
\boxed{
\frac{p_\varepsilon}{p}\longrightarrow0.
}
$$

When that pressure regularization is adopted, the implementation must replace the corresponding $p$ by $\tilde p$ uniformly in the constitutive law, including every denominator and every stress magnitude. The principal formulae of this paper, in which regularization has not been enabled, continue to be written with $p$.

If a shear-rate regularization $\dot\gamma_\varepsilon$ is used, require

$$
\boxed{
\frac{\dot\gamma_\varepsilon}{\dot\gamma}
\longrightarrow0.
}
$$

The production rule of Section 2.8 is stricter than a limit statement. At every finite regularization, the stress magnitude, the yield strength, and the denominator of $I_c$ use one and the same $\tilde p=\max([p]_+,p_{\min})$, and $I_c$ uses $p_*=\tilde p+p_c$. The limit $p_\varepsilon/p\to 0$ is the condition under which that regularization disappears from the classical reduction. It is not a licence to regularize only the denominator.

---

## 4.14 Limit of the inertial number

From

$$
I_c^{(n)}
=
\frac{d\dot\gamma}
{\sqrt{(p+p_c^{(n)})/\rho_p}}
$$

and

$$
\frac{p_c^{(n)}}p\to0,
$$

one has $p+p_c^{(n)}\to p$. The map $x\mapsto x^{-1/2}$ is continuous on a neighbourhood of any $p>0$, so

$$
\boxed{
I_c^{(n)}
\longrightarrow
\frac{d\dot\gamma}{\sqrt{p/\rho_p}}
=
I.
}
$$

*Expanded limit.* Write

$$
I_c^{(n)}
=
I\cdot
\left(\frac{p}{p+p_c^{(n)}}\right)^{1/2}.
$$

The fraction in parentheses tends to $1$, and the square root is continuous at $1$. This step requires $p>0$. On a free surface where $p=0$, the classical $\mu(I)$ law is itself undefined, so there is no reduction in the ordinary sense.

---

## 4.15 Limit of the friction coefficient

The dynamic friction coefficient of the combined model is

$$
\mu_d^{(n)}
=
\mu_s(\nu^{(n)},Z^{(n)})
+
\left[
\mu_2(\nu^{(n)},Z^{(n)})
-
\mu_s(\nu^{(n)},Z^{(n)})
\right]
\frac{I_c^{(n)}}
{I_0(\nu^{(n)},Z^{(n)})+I_c^{(n)}}.
$$

Conditions C and D, together with the conclusion of the previous section, give

$$
\nu^{(n)}\to\nu_0,
\qquad
Z^{(n)}\to Z_0,
\qquad
I_c^{(n)}\to I,
$$

and therefore

$$
\mu_s(\nu^{(n)},Z^{(n)})\to\mu_s,
\qquad
\mu_2(\nu^{(n)},Z^{(n)})\to\mu_2,
\qquad
I_0(\nu^{(n)},Z^{(n)})\to I_0.
$$

The correct continuous mapping has four arguments, because $I_0$ also depends on $n$:

$$
(a,b,e,c)\mapsto a+(b-a)\frac{c}{e+c},
$$

which is continuous on the set $e+c>0$. Substituting

$$
a=\mu_s^{(n)},\quad
b=\mu_2^{(n)},\quad
e=I_0^{(n)},\quad
c=I_c^{(n)}
$$

and passing to the limit produces

$$
\boxed{
\mu_d^{(n)}
\longrightarrow
\mu_s+
(\mu_2-\mu_s)
\frac{I}{I_0+I}
=
\mu_{\mathrm{cl}}(I).
}
$$

A three-argument map that freezes $I_0$ before the limit is not sufficient for this conclusion. The same four-argument statement is the one required again in Section 4.20.

---

## 4.16 Limit of the tidal yield function

The effective shear stress in the combined yield function is

$$
\tau_{\mathrm{tide}}^{(n)}
=
\sqrt{
\frac12
(s-T_{\mathrm{tide}}^{(n)}):(s-T_{\mathrm{tide}}^{(n)})
}.
$$

By the definition $s=T_{\mathrm{tide}}+\tau_{\mathrm{flow}}$ one has the identity

$$
s-T_{\mathrm{tide}}^{(n)}=\tau_{\mathrm{flow}}^{(n)}
$$

at every $n$, and therefore

$$
\tau_{\mathrm{tide}}^{(n)}
=
\|\tau_{\mathrm{flow}}^{(n)}\|_{\mathrm{eq}}
=
\tau_{{\mathrm{flow}},\mathrm{eq}}^{(n)}
$$

identically. The limit of $\tau_{\mathrm{tide}}^{(n)}$ is the limit of $\tau_{{\mathrm{flow}},\mathrm{eq}}^{(n)}$ and contains no independent information. This is the observation recorded again in Section 4.20.

If, in addition, a limit stress $s$ exists and $T_{\mathrm{tide}}^{(n)}\to 0$, continuity of the Frobenius norm gives the further identification

$$
\boxed{
\tau_{\mathrm{tide}}^{(n)}
\longrightarrow
\sqrt{\frac12 s:s}
=
\tau_{{\mathrm{flow}},\mathrm{eq}}.
}
$$

The two members on the right agree because $s-T_{\mathrm{tide}}\to s$ and $s-T_{\mathrm{tide}}=\tau_{\mathrm{flow}}$ together force $\|s\|_{\mathrm{eq}}=\|\tau_{\mathrm{flow}}\|_{\mathrm{eq}}$ in the limit. From

$$
\frac{c^{(n)}}p\to0,
\qquad
\mu_y(\nu^{(n)},Z^{(n)})\to\mu_s
$$

the static yield function has the limit

$$
F_{\mathrm{comb}}^{(n)}
\longrightarrow
\tau_{{\mathrm{flow}},\mathrm{eq}}-\mu_s p.
$$

*Algebra.* One has $F_{\mathrm{comb}}^{(n)}=\tau_{\mathrm{tide}}^{(n)}-c^{(n)}-\mu_y^{(n)}p$, the cohesive term is $p\cdot(c^{(n)}/p)\to 0$, and $\mu_y^{(n)}p\to\mu_s p$. This is the Drucker–Prager limit of the quasi-static theory, without cohesion and without the tidal extension.

---

## 4.17 Limit of the dynamic stress

On the flow branch of the combined model, the scalar target is

$$
\tau_{{\mathrm{flow}},\mathrm{eq}}^{(n)}
=
c^{(n)}
+
\mu_d^{(n)}p.
$$

From

$$
\frac{c^{(n)}}p\to0,
\qquad
\mu_d^{(n)}\to\mu_{\mathrm{cl}}(I),
$$

one obtains

$$
\boxed{
\tau_{{\mathrm{flow}},\mathrm{eq}}^{(n)}
\longrightarrow
\mu_{\mathrm{cl}}(I)p.
}
$$

The classical $\mu(I)$ stress ratio is thereby recovered as a target. This line is not yet the equivalent stress of the regularized tensor.

Because the deviator is aligned with $D'$, the corrected regularized constitutive law is written

$$
\boxed{
\tau_{\mathrm{flow}}^{(n)}
=
2\eta^{(n)}D',
}
$$

where

$$
\eta^{(n)}
=
\frac{
c^{(n)}+\mu_d^{(n)}p
}{\sqrt{\dot\gamma^2+\dot\gamma_{\varepsilon}^{\,2}}}.
$$

The equivalent stress of this tensor is not $c^{(n)}+\mu_d^{(n)}p$. By the calculation of Section 3.14,

$$
\|\tau_{\mathrm{flow}}^{(n)}\|_{\mathrm{eq}}
=
\eta^{(n)}\dot\gamma
=
\bigl(c^{(n)}+\mu_d^{(n)}p\bigr)
\frac{\dot\gamma}{\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}}.
$$

It tends to $c^{(n)}+\mu_d^{(n)}p$ only when $\dot\gamma_\varepsilon/\dot\gamma\to 0$. Under the joint hypotheses

$$
\frac{c^{(n)}}p\to0,
\qquad
\mu_d^{(n)}\to\mu_{\mathrm{cl}}(I),
\qquad
\dot\gamma_\varepsilon/\dot\gamma\to0,
$$

one has

$$
\eta^{(n)}
\longrightarrow
\frac{\mu_{\mathrm{cl}}(I)p}{\dot\gamma},
$$

and therefore

$$
\boxed{
\tau_{\mathrm{flow}}^{(n)}
\longrightarrow
2\mu_{\mathrm{cl}}(I)p
\frac{D'}{\dot\gamma}.
}
$$

Taking the equivalent shear stress,

$$
\begin{aligned}
\tau_{{\mathrm{flow}},\mathrm{eq}}^{(n)}
&=
\sqrt{
\frac12
\tau_{\mathrm{flow}}^{(n)}:\tau_{\mathrm{flow}}^{(n)}
}\\
&\longrightarrow
2\mu_{\mathrm{cl}}(I)p
\sqrt{
\frac{D':D'}{2\dot\gamma^2}
}.
\end{aligned}
$$

The passage of the limit under the square root uses continuity of $A\mapsto\sqrt{\tfrac12 A:A}$ and the tensor limit already obtained. Since

$$
D':D'=\frac{\dot\gamma^2}{2},
$$

the expression under the square root equals $1/4$, its square root equals $1/2$, and

$$
2\mu_{\mathrm{cl}}(I)p\cdot\frac12
=
\mu_{\mathrm{cl}}(I)p.
$$

Hence

$$
\boxed{
\tau_{{\mathrm{flow}},\mathrm{eq}}^{(n)}
\longrightarrow
\mu_{\mathrm{cl}}(I)p.
}
$$

This is the classical $\mu(I)$ relation. The same algebra shows why the factor $2$ in $\tau=2\eta D'$ is indispensable.

---

## 4.18 Limit of the governing equations

The momentum equation of the combined model is

$$
\partial_t(\rho v)
+
\nabla\cdot(\rho v\otimes v)
+2\rho\,\boldsymbol\Omega\times v
=
\nabla\cdot\sigma
+
\rho\nabla W.
$$

The positive-potential convention $g_{\mathrm{eff}}=\nabla W$ of this paper is in force. If a traditional negative gravitational potential is used instead, the sign of the potential gradient must be rewritten throughout. It is not enough to change this one term. The Coriolis term $2\rho\boldsymbol\Omega\times v$ in the rotating frame agrees with Proposition 5.2. The classical $\mu(I)$ system corresponds to an inertial frame, $\boldsymbol\Omega=\mathbf 0$, in which that term is absent.

*Relation to the nonconservative form.* Expanding the left-hand side gives

$$
\partial_t(\rho v)+\nabla\cdot(\rho v\otimes v)
=
\rho\bigl(\partial_t v+(v\cdot\nabla)v\bigr)
+
v\bigl(\partial_t\rho+\nabla\cdot(\rho v)\bigr).
$$

The two forms therefore agree on solutions of the continuity equation $\partial_t\rho+\nabla\cdot(\rho v)=0$, which is the form used in Section 5.1. They are not identical as formal differential expressions.

In the limit above, a traditional incompressible dense granular model must also fix $\rho=\rho_0>0$. The continuity equation is then equivalent to $\nabla\cdot v=0$. If the density is not fixed, the system below can be called only a compressible variable-density extension. It cannot be called the traditional incompressible $\mu(I)$ system directly.

In that incompressible limit the total stress of the combined model is

$$
\sigma=-pI+T_{\mathrm{tide}}+\tau_{\mathrm{flow}}.
$$

The divergence $\nabla\cdot\sigma$ in the momentum equation therefore contains $\nabla\cdot T_{\mathrm{tide}}$. In the limit of this section, $T_{\mathrm{tide}}\to 0$, so that term disappears. Passage of the divergence to the limit is not implied by the pointwise limit alone; the additional compactness hypotheses are stated in Section 4.20.

In the incompressible limit:

- the solid fraction is fixed;
- the coordination number is fixed;
- the tidal backstress disappears;
- cohesion disappears;
- the cohesive pressure disappears;
- the state-dependent parameters become constants;
- the pressure and shear-rate regularizations disappear (Condition F).

The constitutive term therefore becomes

$$
\sigma
=
-pI
+
2\mu_{\mathrm{cl}}(I)p
\frac{D'}{\dot\gamma}.
$$

The governing equations reduce strictly to

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
I=
\frac{d\dot\gamma}{\sqrt{p/\rho_p}},\\[8pt]
\displaystyle
\mu_{\mathrm{cl}}(I)
=
\mu_s+
(\mu_2-\mu_s)\frac{I}{I_0+I}.
\end{cases}
}
$$

This is the traditional $\mu(I)$ system of granular rheology, in the rotating frame if $\boldsymbol\Omega$ is retained and in an inertial frame if $\boldsymbol\Omega=\mathbf 0$.

---

## 4.19 The reduction theorem

The preceding results may be summarized as follows.

## 4.20 Theorem

Suppose that a sequence of solutions of the combined model satisfies

$$
\frac{p_c}{p}\to0,
$$

$$
\frac{\rho_pd^2\|\operatorname{dev}H_W\|}{p}\to0,
$$

$$
\frac{c}{p}\to0,
$$

$$
\nu\to\nu_0,
\qquad
Z\to Z_0,
$$

$$
\mu_s(\nu,Z)\to\mu_s,
$$

$$
\mu_2(\nu,Z)\to\mu_2,
$$

$$
I_0(\nu,Z)\to I_0,
$$

$$
\mu_y(\nu,Z)\to\mu_s,
$$

$$
p_\varepsilon/p\to0,
\qquad
\dot\gamma_\varepsilon/\dot\gamma\to0,
$$

and that, along the chosen sequence of points, $p\to p_0>0$ and $\dot\gamma\to\dot\gamma_0>0$, or at least that these quantities remain in a compact subset of $(0,\infty)$. In the latter case $I$ is the classical inertial number formed from the same $p$ and $\dot\gamma$, and the convergences below are understood as differences tending to zero. A positive lower bound alone does not suffice for the absolute limits $T_{\mathrm{tide}}\to0$ and $\tau_{{\mathrm{flow}},\mathrm{eq}}-\mu_{\mathrm{cl}}(I)p\to0$, because the hypotheses on $\operatorname{dev}H_W$ and $c$ control only the ratios to $p$. Then the combined model satisfies

$$
I_c\to I,
$$

$$
\mu_d(I_c,\nu,Z)\to\mu_{\mathrm{cl}}(I),
$$

$$
T_{\mathrm{tide}}\to0,
$$

$$
\tau_{{\mathrm{flow}},\mathrm{eq}}\to\mu_{\mathrm{cl}}(I)p.
$$

Here $I=d\dot\gamma_\infty/\sqrt{p_\infty/\rho_p}$ is formed from the limit fields. Sections 4.14–4.17 are pointwise statements: $p$, $\dot\gamma$, and $D'$ are values at the same point, not a sequence of solutions in a function space.

**Strict conditions for convergence of the momentum equation (the compactness chain).** The pointwise constitutive limit $\tau^{(n)}_{\mathrm{eq}}\to\mu_{\mathrm{cl}}p$ is only a scalar convergence. It implies neither convergence of the tensor $D'^{(n)}/\dot\gamma^{(n)}$ nor convergence of the differentiated momentum equation. In Section 4.16 the identity $s-T_{\mathrm{tide}}\equiv\tau_{\mathrm{flow}}$ holds by the definition of $s$. Its limit is a corollary of that identity and contains no independent information. After the regularization of Section 4.17,

$$
\tau_{\mathrm{flow,eq}}=(c+\mu_d p)\frac{\dot\gamma}{\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}},
$$

which tends to $c+\mu_d p$ only when $\dot\gamma_\varepsilon/\dot\gamma\to 0$. The continuous mapping used in Section 4.15 is the four-argument map $(a,b,e,c)\mapsto a+(b-a)\,c/(e+c)$, in which $I_0$ also changes with $n$ and $e+c>0$.

The momentum equation converges in the sense of distributions only under **all** of the following additional hypotheses. Let $K\Subset\{p>0\}$ be a compact set in the space-time region, and assume:

1. $p^{(n)}\ge p_{\min}>0$ and $\dot\gamma^{(n)}\ge\dot\gamma_{\min}>0$ hold uniformly on $K$, and conditions A–F hold uniformly on $K$;
2. $v^{(n)}\to v$ strongly in $L^2(K)$, and $\nabla v^{(n)}\to\nabla v$ together with $p^{(n)}\to p$ almost everywhere on $K$ (a subsequence suffices if the convergence holds in measure);
3. $\sigma^{(n)}$ is uniformly bounded in $L^{1+\delta}(K)$ for some $\delta>0$, and $\nu^{(n)}\to\nu_0$ strongly.

Then

$$
\sigma^{(n)}\to\sigma_{\mathrm{cl}}(\nabla v,p)=-pI+2\mu_{\mathrm{cl}}(I)p\,D'/\dot\gamma
$$

in $L^1(K)$, by Vitali's theorem: almost-everywhere convergence together with uniform integrability. Hence $\nabla\cdot\sigma^{(n)}\to\nabla\cdot\sigma_{\mathrm{cl}}$ in $\mathcal D'(K)$. The convective term converges by the strong $L^2$ convergence alone ($v^{(n)}\otimes v^{(n)}\to v\otimes v$ in $L^1(K)$, with $\rho=\rho_0$ fixed as in Section 4.18), and $\nabla\cdot v=0$ is preserved.

**Limitations.** (a) Almost-everywhere convergence is a hypothesis, not a conclusion. The stress–strain-rate relation of $\mu(I)$ is in general **not monotone** — there is a region of ill-posedness at small Froude number (Barker et al., 2015) — so the Minty monotonicity argument is unavailable, and the limit cannot be identified from weak convergence alone. (b) The conclusion is restricted to the interior of a compact set $K\Subset\{p>0\}$. It does not cover the free-surface boundary condition or a plug region where $\dot\gamma=0$. (c) Mode B, which is quasi-static and formally sits at $\dot\gamma=0$, lies outside the hypothesis $\dot\gamma_{\min}>0$. The theorem is therefore vacuous for the production default. It applies only to the active flow region of Mode C.

---

## 4.21 Range of validity and the boundary of failure

The reduction above must not be used in the following situations.

## 4.22 The low-pressure limit on the free surface, and the singularity at $p=0$

Normal momentum balance on the free surface gives

$$
p|_\Gamma=p_{\rm ext}+\mathbf n\cdot\boldsymbol\tau\,\mathbf n
$$

(Corollary 5.3 of Section 5.1). In the absence of external pressure and of a deviatoric stress, or in the tideless low-shear limit, one has $p\to 0$. In that limit the classical inertial number

$$
I=\frac{d\dot\gamma}{\sqrt{p/\rho_p}}
$$

has a division by zero in the denominator. One must then retain an effective pressure $p_*=p+p_c>0$, or adopt an explicit pressure regularization.

## 4.23 The tidal stress is not negligible

If

$$
\rho_pd^2\|\operatorname{dev}H_W\|
\sim p,
$$

the combined model does not reduce to the classical $\mu(I)$ law. Condition B fails, and $T_{\mathrm{tide}}$ remains of the same order as the pressure scale that multiplies $\mu$.

## 4.24 The state variables are still evolving

If $Z$ or $\nu$ varies appreciably in space, then

$$
\mu_d=\mu_d(I,\nu,Z)
$$

is not the traditional function of the single variable $I$.

## 4.25 Static regions

If

$$
\dot\gamma=0,
$$

the dynamic $\mu(I)$ formula is not a complete description. One should then use

$$
\sqrt{\frac12(s-T_{\mathrm{tide}}):(s-T_{\mathrm{tide}})}
\le c+\mu_y p.
$$

In the regularized viscoplastic production model (Section 2.8), a static region is not identified by $\dot\gamma=0$, because the regularization makes $\dot\gamma>0$ everywhere. It is identified by the yield function $F_\delta\le 0$, evaluated on the same stress field, together with connectivity to the rigid core (B2–B3 of Section 2.8), with $\mu_y\equiv\mu_s(\nu_0,Z_0)$.

---

The final strict conclusion is

$$
\boxed{
\text{the combined model is not automatically equal to the classical }\mu(I)\text{ law;}
}
$$

$$
\boxed{
\text{in the explicit limit in which cohesion, tide, state evolution, and regularization all disappear,}
}
$$

$$
\boxed{
\text{it reduces, term by term and continuously, to the classical }\mu(I)\text{ rheological equation.}
}
$$

This is a reduction that can be written as a formal mathematical proposition. It is not an identity that is absolutely correct without conditions.

# 5. The mixed weak form and the poloidal–toroidal representation of velocity

This chapter constructs the mixed weak form of momentum, incompressibility, and mass transport on the star-shaped domain $\mathcal V=\{r<R(\hat{\boldsymbol\omega},t)\}$, and writes the velocity through a poloidal potential and a toroidal potential. The notation obeys the glossary of Appendix A.1: the stress is the bold tensor $\boldsymbol\sigma$, a direction is $\hat{\boldsymbol\omega}$, and the angular-velocity vector is $\boldsymbol\Omega=\omega\mathbf e_z$. The basic unknowns are $(\mathbf v,p,\rho,R)$.

---

## 5.1 Governing equations and the natural boundary term

**Definition 5.1 (Total stress and body acceleration).** Take

$$
\boldsymbol\sigma=-p\,\mathbf I+\boldsymbol\tau,\qquad
\boldsymbol\tau(\mathbf v,p)=\mathbf T_{\mathrm{tide}}+2\eta(p,\mathbf D)\,\mathbf D,\qquad
\mathbf D=\tfrac12(\nabla\mathbf v+\nabla\mathbf v^{\!\top}),
$$

where $\eta$ is given by Chapter 4. In the combined model,

$$
\eta=\frac{c+\mu_d(I_c)\,\tilde p}{\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}},
$$

with $\tilde p=\max([p]_+,p_{\min})$ the unique regularized pressure of (B1) in Section 2.8 and with $I_c$ evaluated at $p_*=\tilde p+p_c$. In the classical branch, the formal limit $c=0$ and $\dot\gamma_\varepsilon\to 0$ is $\eta=\mu(I)p/\dot\gamma$, so

$$
\boldsymbol\tau=2\mu(I)\,p\,\mathbf D'/\dot\gamma
$$

(and $\mathbf D'=\mathbf D$ when the flow is incompressible). The body acceleration is

$$
\mathbf g_0=\mathbf g_{\mathrm{RTFP}}-\boldsymbol\Omega\times(\boldsymbol\Omega\times\mathbf x)+\mathbf g_{\mathrm{ext}}
=\nabla W+\mathbf g_{\mathrm{ext}},\qquad
W=U+\tfrac12\omega^2(x_1^2+x_2^2).
$$

The centrifugal acceleration is $-\boldsymbol\Omega\times(\boldsymbol\Omega\times\mathbf x)=\omega^2\mathbf x_\perp$, so the minus sign in $\mathbf g_0$ is required by the positive-potential convention of Section 2.8. The field $\mathbf g_{\mathrm{ext}}$ does not contain the gravitational or centrifugal contributions already accounted for. If $\dot{\boldsymbol\Omega}\ne 0$, the Euler acceleration $-\dot{\boldsymbol\Omega}\times\mathbf x$ is added. That term is not in general a gradient. The weak form of this chapter and the source formulae of Section 6.3 are written through $\nabla\times(\rho\mathbf g_0)$, and they apply to the Euler acceleration in the same way. On the quasi-static route, $\omega(t)$ is prescribed exogenously. The Euler term and the external torque that drives it are omitted as a pair (Section 2.8, B1), and the ratio $|\dot\omega|/\omega^2$ is reported.

**Proposition 5.2 (Mixed weak form and the natural boundary term).** Suppose $(\mathbf v,p)$ is smooth enough to satisfy the strong form

$$
\rho\big(\partial_t\mathbf v+(\mathbf v\cdot\nabla)\mathbf v+2\boldsymbol\Omega\times\mathbf v\big)=\nabla\cdot\boldsymbol\sigma+\rho\mathbf g_0,\qquad
\nabla\cdot\mathbf v=0\quad\text{in }\mathcal V,\qquad
\boldsymbol\sigma\mathbf n=\mathbf t_{\mathrm{ext}}\quad\text{on }\Gamma.
$$

Then, for every $(\mathbf w,q)\in H^1(\mathcal V;\mathbb R^3)\times L^2(\mathcal V)$,

$$
\boxed{
\begin{aligned}
&\int_{\mathcal V}\rho\big(\partial_t\mathbf v+(\mathbf v\cdot\nabla)\mathbf v+2\boldsymbol\Omega\times\mathbf v\big)\cdot\mathbf w\,dV
+\int_{\mathcal V}\boldsymbol\tau(\mathbf v,p):\mathbf D(\mathbf w)\,dV\\
&\qquad-\int_{\mathcal V}p\,(\nabla\cdot\mathbf w)\,dV
+\int_{\mathcal V}q\,(\nabla\cdot\mathbf v)\,dV
-\oint_\Gamma\mathbf t_{\mathrm{ext}}\cdot\mathbf w\,dS
=\int_{\mathcal V}\rho\,\mathbf g_0\cdot\mathbf w\,dV.
\end{aligned}}
$$

Conversely, a smooth pair $(\mathbf v,p)$ that satisfies this identity satisfies the strong form, including the boundary condition.

*Proof.* Multiply the momentum equation by $\mathbf w$ and integrate. The divergence theorem gives

$$
\int_{\mathcal V}(\nabla\cdot\boldsymbol\sigma)\cdot\mathbf w\,dV
=\oint_\Gamma\boldsymbol\sigma\mathbf n\cdot\mathbf w\,dS
-\int_{\mathcal V}\boldsymbol\sigma:\nabla\mathbf w\,dV.
$$

Symmetry of $\boldsymbol\sigma$ replaces $\nabla\mathbf w$ by its symmetric part: $\boldsymbol\sigma:\nabla\mathbf w=\boldsymbol\sigma:\mathbf D(\mathbf w)$. Split $\boldsymbol\sigma=-p\mathbf I+\boldsymbol\tau$. Then

$$
\boldsymbol\sigma:\mathbf D(\mathbf w)=-p\,(\nabla\cdot\mathbf w)+\boldsymbol\tau:\mathbf D(\mathbf w),
$$

because $\mathbf I:\mathbf D(\mathbf w)=\nabla\cdot\mathbf w$. Substituting $\boldsymbol\sigma\mathbf n=\mathbf t_{\mathrm{ext}}$ produces the momentum terms of the boxed identity. Multiplying the incompressibility equation by $q$ and integrating produces the term in $q$.

In the converse direction, first restrict $\mathbf w$ to be compactly supported in $\mathcal V$. The boundary integral drops, and the fundamental lemma of the calculus of variations returns the momentum equation pointwise. With the momentum equation restored, the remaining identity reduces to

$$
\oint_\Gamma(\boldsymbol\sigma\mathbf n-\mathbf t_{\mathrm{ext}})\cdot\mathbf w\,dS=0
$$

for every trace of $\mathbf w$ on $\Gamma$. Hence $\boldsymbol\sigma\mathbf n=\mathbf t_{\mathrm{ext}}$. ∎

**Corollary 5.3 (Natural realization of the boundary condition).** The free-surface traction condition of Chapter 4 is not an essential constraint on $\mathbf v$. It is the surface term $-\oint_\Gamma\mathbf t_{\mathrm{ext}}\cdot\mathbf w\,dS$ of the identity above.

1. On a free surface, $\mathbf t_{\mathrm{ext}}=\mathbf 0$. The surface term vanishes, and $\boldsymbol\sigma\mathbf n=\mathbf 0$ holds automatically as a natural condition.
2. On a contact surface, $\mathbf t_{\mathrm{ext}}=-p_n\mathbf n+\mathbf t_T$. The normal contact reaction $p_n\ge 0$ and the tangential friction $\mathbf t_T$, subject to $|\mathbf t_T|\le\mu_w p_n$, are supplied by the contact model and enter only through the surface term.
3. The natural condition splits into normal and tangential components:

$$
\mathbf n\cdot\boldsymbol\tau\mathbf n-p=\mathbf n\cdot\mathbf t_{\mathrm{ext}},\qquad
\boldsymbol\tau\mathbf n-(\mathbf n\cdot\boldsymbol\tau\mathbf n)\mathbf n
=\mathbf t_{\mathrm{ext}}-(\mathbf n\cdot\mathbf t_{\mathrm{ext}})\mathbf n.
$$

On a free surface this is $p=\mathbf n\cdot\boldsymbol\tau\mathbf n$. The limit $\boldsymbol\tau\to\mathbf 0$ therefore gives $p\to 0$, in agreement with Section 4.22, and no algebraic substitution is required.

**Supplementary equations.** Mass transport and the free-surface kinematics are

$$
\partial_t\rho+\nabla\cdot(\rho\mathbf v)=0,\qquad
\partial_tR=v_r-\frac1R\,\nabla_{S^2}R\cdot\mathbf v_S\Big|_{r=R}.
$$

The kinematic relation is the material condition $\partial_t F+\mathbf v\cdot\nabla F=0$ for $F=r-R(\hat{\boldsymbol\omega},t)$. Here $\mathbf v_S$ is the physical tangential component and $\nabla_{S^2}$ is the gradient on the unit sphere. Explicitly, at fixed time,

$$
\nabla F=\mathbf e_r-R^{-1}\nabla_{S^2}R,\qquad \partial_t F=-\partial_t R,
$$

so

$$
-\partial_t R+v_r-R^{-1}\nabla_{S^2}R\cdot\mathbf v_S=0,
$$

which is the displayed formula. The outward unit normal of the graph $r=R$ is

$$
\mathbf n=\frac{R\mathbf e_r-\nabla_{S^2}R}{\sqrt{R^2+|\nabla_{S^2}R|^2}},\qquad
\mathbf n\cdot\mathbf e_r=\frac{R}{\sqrt{R^2+|\nabla_{S^2}R|^2}}.
$$

Therefore

$$
\mathbf v\cdot\mathbf n=\frac{R v_r-\nabla_{S^2}R\cdot\mathbf v_S}{\sqrt{R^2+|\nabla_{S^2}R|^2}},
$$

and division by $\mathbf n\cdot\mathbf e_r$ cancels the common square root and the factor $R$:

$$
\frac{\mathbf v\cdot\mathbf n}{\mathbf n\cdot\mathbf e_r}
=v_r-\frac1R\nabla_{S^2}R\cdot\mathbf v_S.
$$

The quasi-static route does not advance the continuity equation: $\rho_K$ is spatially fixed in the body frame (model hypothesis A-ρ, PO-14). The radius $R$ is obtained from $\partial_t R=\mathbf v\cdot\mathbf n/(\mathbf n\cdot\mathbf e_r)$, classified by yield, and discretized over a geological step as $d_{\rm pred}$. It is then updated by the geometric subproblem of Section 10.4.3 (see Sections 9.9 and 10.4.1).

---

## 5.2 The poloidal–toroidal representation of velocity

**Definition 5.4.** For scalar potentials $\Phi$ and $\Psi$, write

$$
\mathcal P[\Phi]=\nabla\times\nabla\times(\Phi\,\mathbf r),\qquad
\mathcal T[\Psi]=\nabla\times(\Psi\,\mathbf r)=\nabla\Psi\times\mathbf r,\qquad
\mathbf v=\mathcal P[\Phi]+\mathcal T[\Psi],\qquad \mathbf r=r\mathbf e_r.
$$

**Lemma 5.5 (Component formulae).** Let $\Phi=g(r)Y_{\ell m}$ and $\Psi=f(r)Y_{\ell m}$, with $L^2 Y_{\ell m}=\Lambda_\ell Y_{\ell m}$ and $\Lambda_\ell=\ell(\ell+1)$. Then

$$
\mathcal P[\Phi]=\nabla\big(\partial_r(r\Phi)\big)-\mathbf r\,\Delta\Phi,\qquad
\nabla\cdot\mathcal P[\Phi]=0,\qquad \nabla\cdot\mathcal T[\Psi]=0,
$$

$$
v_r^{\mathcal P}=\frac{L^2\Phi}{r}=\frac{\Lambda_\ell g}{r}Y_{\ell m},\qquad
\mathbf v_S^{\mathcal P}=\frac1r\,(rg)'\,\nabla_{S^2}Y_{\ell m},\qquad
v_r^{\mathcal T}=0,\qquad
\mathbf v^{\mathcal T}=f\,\nabla_{S^2}Y_{\ell m}\times\mathbf e_r.
$$

In spherical components, $v^{\mathcal T}_\theta=\dfrac1{\sin\theta}\partial_\varphi\Psi$ and $v^{\mathcal T}_\varphi=-\partial_\theta\Psi$.

*Proof.* The vector identity $\nabla\times\nabla\times\mathbf F=\nabla(\nabla\cdot\mathbf F)-\Delta\mathbf F$ is applied to $\mathbf F=\Phi\mathbf r$. The divergence is

$$
\nabla\cdot(\Phi\mathbf r)=3\Phi+r\partial_r\Phi,
$$

because $\nabla\cdot\mathbf r=3$ and $\nabla\Phi\cdot\mathbf r=r\partial_r\Phi$. The Laplacian of the vector field satisfies $\Delta(\Phi\mathbf r)=\mathbf r\,\Delta\Phi+2\nabla\Phi$. Subtraction therefore gives

$$
\begin{aligned}
\nabla\times\nabla\times(\Phi\mathbf r)
&=\nabla(3\Phi+r\partial_r\Phi)-\mathbf r\,\Delta\Phi-2\nabla\Phi\\
&=\nabla(\Phi+r\partial_r\Phi)-\mathbf r\,\Delta\Phi,
\end{aligned}
$$

which is the first identity, since $\partial_r(r\Phi)=\Phi+r\partial_r\Phi$.

The radial component is computed in spherical coordinates. The radial part of $\nabla(\Phi+r\Phi_r)$ is

$$
\partial_r(\Phi+r\Phi_r)=2\Phi_r+r\Phi_{rr}.
$$

The spherical Laplacian is

$$
\Delta\Phi=\Phi_{rr}+\frac{2}{r}\Phi_r-\frac{L^2\Phi}{r^2},
$$

so the radial component of $\mathbf r\,\Delta\Phi$ is

$$
r\Delta\Phi=r\Phi_{rr}+2\Phi_r-\frac{L^2\Phi}{r}.
$$

Subtracting these two expressions cancels $r\Phi_{rr}$ and $2\Phi_r$, and leaves

$$
v_r^{\mathcal P}=\frac{L^2\Phi}{r}.
$$

If $\Phi=g(r)Y_{\ell m}$, then $L^2\Phi=\Lambda_\ell g\,Y_{\ell m}$. The tangential part of $\nabla(\partial_r(r\Phi))$ is $r^{-1}\nabla_{S^2}\partial_r(r\Phi)$. For a separated potential, $\partial_r(r\Phi)=(rg)'Y_{\ell m}$, which is the formula for $\mathbf v_S^{\mathcal P}$.

For the toroidal field, $\nabla\Psi=\Psi_r\mathbf e_r+r^{-1}\nabla_{S^2}\Psi$. The cross product with $\mathbf r=r\mathbf e_r$ kills the radial term and leaves $\nabla_{S^2}\Psi\times\mathbf e_r$. In components this is $v_\theta=(1/\sin\theta)\partial_\varphi\Psi$ and $v_\varphi=-\partial_\theta\Psi$, so the radial component vanishes. Both divergences vanish because $\nabla\cdot\nabla\times=0$.

The rigid rotation $\Psi=\omega r\cos\theta$ is a check: $v_\varphi=-\partial_\theta\Psi=\omega r\sin\theta$, which is the azimuthal velocity of rotation at rate $\omega$ about $\mathbf e_z$. ∎

**Proposition 5.6 (Uniqueness, the degree $\ell=0$, and the rigid modes).**

1. For $\ell\ge 1$, the coefficient $\Phi_{\ell m}$ is uniquely determined by $v_r$, and the coefficient $\Psi_{\ell m}$ is uniquely determined by $\mathbf r\cdot\nabla\times\mathbf v$.
2. A purely radial potential $\Phi=f(r)$ or $\Psi=h(r)$, both of degree $\ell=0$, satisfies $\nabla\times(f\mathbf r)=\nabla f\times\mathbf r=\mathbf 0$ and contributes nothing to the velocity. Velocity degrees of freedom therefore contain only $\ell\ge 1$. On an incompressible star-shaped domain, a radial breathing mode of degree $\ell=0$ is forbidden by $\nabla\cdot\mathbf v=0$.
3. A translation $\mathbf v=\mathbf e$ corresponds to $\Phi=\tfrac12\mathbf e\cdot\mathbf x$, because

$$
\nabla\times\nabla\times\big(\tfrac12(\mathbf e\cdot\mathbf x)\,\mathbf r\big)
=\nabla\times\big(\tfrac12\mathbf e\times\mathbf x\big)=\mathbf e.
$$

A rotation $\mathbf v=\boldsymbol\theta\times\mathbf x$ corresponds to $\Psi=\boldsymbol\theta\cdot\mathbf x$. Both families lie at degree $\ell=1$, with radial profile proportional to $r$. Together they are six modes, and they exhaust the solutions of $\mathbf D(\mathbf v)=\mathbf 0$.

*Proof of (1).* By Lemma 5.5, $v_r=L^2\Phi/r$. On each harmonic of degree $\ell\ge 1$, $L^2$ acts by the nonzero factor $\Lambda_\ell$, so

$$
\Phi_{\ell m}=\frac{r}{\Lambda_\ell}(v_r)_{\ell m}.
$$

For the toroidal potential, the radial component of the curl is computed from $v_r=0$, $v_\theta=(\sin\theta)^{-1}\partial_\varphi\Psi$, and $v_\varphi=-\partial_\theta\Psi$:

$$
\begin{aligned}
(\nabla\times\mathbf v)_r
&=\frac{1}{r\sin\theta}\Big(\partial_\theta(\sin\theta\, v_\varphi)-\partial_\varphi v_\theta\Big)\\
&=-\frac{1}{r}\Delta_{S^2}\Psi
=\frac{L^2\Psi}{r}.
\end{aligned}
$$

Hence $\mathbf r\cdot\nabla\times\mathbf v=L^2\Psi$, and $\Psi_{\ell m}$ is likewise unique for $\ell\ge 1$. A poloidal field has vanishing radial curl on each spherical harmonic, so the two reconstructions do not interfere.

*Proof of (2).* If $\Phi=f(r)$, then $\nabla f=f'\mathbf e_r$ is parallel to $\mathbf r$, so $\nabla f\times\mathbf r=\mathbf 0$ and both curls vanish. The same holds for $\Psi=h(r)$. A radial field $\mathbf v=v_r(r)\mathbf e_r$ has

$$
\nabla\cdot\mathbf v=\frac{1}{r^2}\partial_r(r^2 v_r).
$$

Incompressibility forces $r^2 v_r$ to be constant. Boundedness at the origin, which lies in the star-shaped domain, forces the constant to vanish, so $v_r=0$.

*Proof of (3).* Let $\Phi=\tfrac12(\mathbf e\cdot\mathbf x)$ with $\mathbf e$ constant, and write $\mathbf x$ for $\mathbf r$. Then

$$
\nabla\times\big((\mathbf e\cdot\mathbf x)\,\mathbf x\big)
=\nabla(\mathbf e\cdot\mathbf x)\times\mathbf x
=\mathbf e\times\mathbf x,
$$

because $\nabla\times\mathbf x=\mathbf 0$. Hence $\nabla\times(\Phi\mathbf x)=\tfrac12\mathbf e\times\mathbf x$. The second curl of a constant vector crossed into $\mathbf x$ is

$$
\nabla\times(\mathbf e\times\mathbf x)
=\mathbf e\,(\nabla\cdot\mathbf x)-(\mathbf e\cdot\nabla)\mathbf x
=3\mathbf e-\mathbf e=2\mathbf e,
$$

so $\nabla\times(\tfrac12\mathbf e\times\mathbf x)=\mathbf e$. For the rotation, $\nabla(\boldsymbol\theta\cdot\mathbf x)=\boldsymbol\theta$, and Definition 5.4 gives $\mathcal T[\boldsymbol\theta\cdot\mathbf x]=\boldsymbol\theta\times\mathbf x$.

Both potentials are linear in $\mathbf x$. Linear functions on the sphere are spherical harmonics of degree $\ell=1$, and the radial factor is proportional to $r$, which is the profile $\psi_{\ell 0}(\zeta)\propto\zeta$ of Section 8.1. It remains to see that these six fields exhaust $\{\mathbf v:\mathbf D(\mathbf v)=\mathbf 0\}$ on a connected domain. The condition $D_{ij}=\tfrac12(\partial_i v_j+\partial_j v_i)=0$ implies, by differentiation,

$$
\partial_k\partial_i v_j
=\partial_i D_{jk}+\partial_j D_{ik}-\partial_k D_{ij}=0.
$$

Thus $\nabla\mathbf v$ is constant. Symmetry of $\mathbf D$ forces that constant tensor to be skew, so there is a vector $\boldsymbol\theta$ with $\nabla\mathbf v\,\mathbf h=\boldsymbol\theta\times\mathbf h$. Integration on a connected domain yields $\mathbf v=\mathbf a+\boldsymbol\theta\times\mathbf x$. These are exactly the six modes already represented. ∎

**Proposition 5.7 (Block structure under a spherically symmetric weight).** Let $R$ be constant and let $\rho=\rho(r)$. For $\Phi_a=g_a Y_{\ell m}$ and $\Psi_a=f_a Y_{\ell m}$,

$$
\int_{\mathcal V}\rho\,\mathcal P[\Phi_1]\cdot\mathcal P[\Phi_2]\,dV
=\int_0^R\rho\,\big[\Lambda_\ell^2 g_1 g_2+\Lambda_\ell\,(rg_1)'(rg_2)'\big]\,dr,
$$

$$
\int_{\mathcal V}\rho\,\mathcal T[\Psi_1]\cdot\mathcal T[\Psi_2]\,dV
=\int_0^R\rho\,\Lambda_\ell\,f_1 f_2\,r^2\,dr,
$$

and the poloidal–toroidal cross term vanishes: $\int_{\mathcal V}\rho\,\mathcal P[\Phi]\cdot\mathcal T[\Psi]\,dV=0$.

*Proof.* Insert Lemma 5.5 and integrate in angle. The identity $\int_{S^2}|\nabla_{S^2}Y_{\ell m}|^2\,d\Omega=\Lambda_\ell$, together with $\int_{S^2}Y_{\ell m}^2\,d\Omega=1$, produces the two diagonal formulae after the radial measure $r^2\,dr$ is restored: the radial–radial contribution carries $\Lambda_\ell^2 g_1 g_2/r^2\cdot r^2=\Lambda_\ell^2 g_1 g_2$, and the tangential contribution carries $\Lambda_\ell(rg_1)'(rg_2)'$. For the cross term the integrand reduces to

$$
\frac1r(rg)'f\ \nabla_{S^2}Y\cdot(\nabla_{S^2}Y'\times\mathbf e_r)
=\frac1r(rg)'f\ \mathbf e_r\cdot(\nabla_{S^2}Y\times\nabla_{S^2}Y').
$$

The angular integral is the integral over the closed sphere of $\mathbf e_r\cdot\nabla\times(Y\nabla_{S^2}Y')$, which is the integral of a surface divergence and therefore vanishes. ∎

---

## 5.3 The Newton tangent of the rheological law

For the classical branch $\boldsymbol\tau=2\mu(I)p\,\mathbf D/\dot\gamma$, with $I=d\dot\gamma\sqrt{\rho_p/p}$ ($p>0$) and

$$
\mu(I)=\mu_s+(\mu_2-\mu_s)\frac{I}{I_0+I},
$$

the first variation at a base state $(\bar p,\bar{\mathbf D},\bar{\dot\gamma}=\sqrt{2\bar{\mathbf D}:\bar{\mathbf D}},\bar I,\bar\mu=\mu(\bar I))$ is

$$
\delta I=\bar I\Big(\frac{\delta\dot\gamma}{\bar{\dot\gamma}}-\frac{\delta p}{2\bar p}\Big),\qquad
\delta\dot\gamma=\frac{2\bar{\mathbf D}:\delta\mathbf D}{\bar{\dot\gamma}},\qquad
\mu'(\bar I)=\frac{(\mu_2-\mu_s)I_0}{(I_0+\bar I)^2},
$$

$$
\boxed{
\delta\boldsymbol\tau
=2\bar\mu\bar p\Big(\frac{\delta\mathbf D}{\bar{\dot\gamma}}-\frac{\bar{\mathbf D}\,\delta\dot\gamma}{\bar{\dot\gamma}^2}\Big)
+\frac{2\bar{\mathbf D}}{\bar{\dot\gamma}}
\Big[\bar p\,\mu'(\bar I)\bar I\Big(\frac{\delta\dot\gamma}{\bar{\dot\gamma}}-\frac{\delta p}{2\bar p}\Big)+\bar\mu\,\delta p\Big].}
$$

*Derivation.* Write $\boldsymbol\tau=2(\mu p/\dot\gamma)\mathbf D$. The product rule gives

$$
\delta\boldsymbol\tau
=2\,\delta\Big(\frac{\mu p}{\dot\gamma}\Big)\,\bar{\mathbf D}
+2\frac{\bar\mu\bar p}{\bar{\dot\gamma}}\,\delta\mathbf D.
$$

The second summand, rewritten by adding and subtracting the projection along $\bar{\mathbf D}$, is the first group of the boxed formula together with the part of the second group that does not contain $\delta\mu$. For the scalar prefactor,

$$
\delta\Big(\frac{\mu p}{\dot\gamma}\Big)
=\frac{p}{\dot\gamma}\,\delta\mu+\frac{\mu}{\dot\gamma}\,\delta p-\frac{\mu p}{\dot\gamma^2}\,\delta\dot\gamma,
$$

evaluated on the base state. The chain rule $\delta\mu=\mu'(\bar I)\,\delta I$ and the logarithmic differential of $I\propto\dot\gamma\,p^{-1/2}$,

$$
\frac{\delta I}{\bar I}=\frac{\delta\dot\gamma}{\bar{\dot\gamma}}-\frac{\delta p}{2\bar p},
$$

produce $\delta I$. Differentiating $\dot\gamma^2=2\mathbf D:\mathbf D$ gives $2\dot\gamma\,\delta\dot\gamma=4\mathbf D:\delta\mathbf D$, hence $\delta\dot\gamma=2\mathbf D:\delta\mathbf D/\dot\gamma$. Substituting these three differentials and collecting coefficients of $\delta\mathbf D$ and of $\delta p$ yields the boxed formula. The derivative of the classical friction law is the quotient rule:

$$
\mu'(I)=\frac{(\mu_2-\mu_s)I_0}{(I_0+I)^2}.
$$

∎

The tangent operator therefore depends on both velocity and pressure. The derivative $\partial\boldsymbol\tau/\partial\mathbf D$ supplies the Newton matrix of the viscous block. The pressure coupling in the velocity equation is

$$
\frac{\partial\boldsymbol\tau}{\partial p}
=\frac{2\bar{\mathbf D}}{\bar{\dot\gamma}}\Big(\bar\mu-\tfrac12\mu'(\bar I)\bar I\Big),
$$

obtained by retaining only $\delta p$ in the boxed formula: the coefficient of $\delta p$ inside the bracket is $\bar\mu-\tfrac12\mu'(\bar I)\bar I$, and it is multiplied by $2\bar{\mathbf D}/\bar{\dot\gamma}$.

In a local simple shear, where $\bar D_{r\theta}>0$ is the only nonzero entry of $\bar{\mathbf D}$ (and likewise $\bar D_{\theta r}$), one has $\bar{\mathbf D}:\bar{\mathbf D}=2\bar D_{r\theta}^2$, so $\bar{\dot\gamma}=2\bar D_{r\theta}$. Vary $\delta D_{r\theta}$ and $\delta p$, leaving every other component of $\delta\mathbf D$ equal to zero. Then $\bar{\mathbf D}:\delta\mathbf D=2\bar D_{r\theta}\,\delta D_{r\theta}$ and

$$
\delta\dot\gamma=\frac{2\cdot 2\bar D_{r\theta}\,\delta D_{r\theta}}{\bar{\dot\gamma}}=2\,\delta D_{r\theta}.
$$

The first group of the boxed formula, evaluated on the $r\theta$ entry, is

$$
2\bar\mu\bar p\Big(\frac{\delta D_{r\theta}}{\bar{\dot\gamma}}-\frac{\bar D_{r\theta}\cdot 2\delta D_{r\theta}}{\bar{\dot\gamma}^2}\Big)
=2\bar\mu\bar p\Big(\frac{\delta D_{r\theta}}{\bar{\dot\gamma}}-\frac{\delta D_{r\theta}}{\bar{\dot\gamma}}\Big)=0,
$$

because $\bar D_{r\theta}=\bar{\dot\gamma}/2$. The projection terms cancel. The second group contributes its full scalar bracket, since $2\bar D_{r\theta}/\bar{\dot\gamma}=1$:

$$
\begin{aligned}
\delta\tau_{r\theta}
&=\bar p\,\mu'(\bar I)\bar I\Big(\frac{2\delta D_{r\theta}}{\bar{\dot\gamma}}-\frac{\delta p}{2\bar p}\Big)+\bar\mu\,\delta p\\
&=2\eta_t\,\delta D_{r\theta}+\mathcal K_p\,\delta p,
\end{aligned}
$$

with

$$
\eta_t=\bar p\,\mu'(\bar I)\bar I/\bar{\dot\gamma},\qquad
\mathcal K_p=\bar\mu-\tfrac12\mu'(\bar I)\bar I.
$$

The tangent viscosity $\eta_t$ and the secant viscosity $\eta_0=\bar\mu\bar p/\bar{\dot\gamma}$ are different quantities. The secant law still returns $\tau_{r\theta}=2\eta_0\bar D_{r\theta}=\bar\mu\bar p$, so the equivalent stress remains $\tau_{\mathrm{eq}}=\bar\mu\bar p$.

The tangent of the combined model is obtained in the same way from the derivatives of $\eta(p,\mathbf D)$. The pressure $p_*$ enters only through $I_c$. If pressure regularization is enabled, the same regularized pressure must be used in every corresponding denominator. Every term is assembled from one and the same differentiation.

**Proposition 5.8 (A local short-wave stability criterion).** For a uniform base state and a plane wave $\delta\mathbf v=\hat{\mathbf v}\,e^{\lambda t+i\mathbf k\cdot\mathbf x}$ with $\mathbf k\cdot\hat{\mathbf v}=0$, let the tangential acoustic tensor be $\mathsf A(\mathbf k)=k_j\mathbb C_{ijkl}k_l$, where $\mathbb C=\partial\boldsymbol\tau/\partial\nabla\mathbf v$. Under incompressibility the pressure disturbance is determined by the momentum equation contracted with $\mathbf k$, and it is not an independent unknown. Then $\lambda$ equals $-1/\rho$ times an eigenvalue of $\mathbb P(\mathbf k)\mathsf A(\mathbf k)$ on $\operatorname{Ran}\mathbb P(\mathbf k)$, where $\mathbb P=\mathbf I-\mathbf k\mathbf k^{\!\top}/|\mathbf k|^2$. A Newtonian fluid, for which $\mathbb C$ is the constant viscosity $\eta_0$, gives $\lambda=-\eta_0|\mathbf k|^2/\rho<0$. On a non-Newtonian base state, $\operatorname{Ran}\mathbb P$ is two-dimensional and the characteristic equation is of degree at most two. The criterion holds only in the local plane-wave approximation. A spherical-shell boundary and the curvature of the surface change the spectrum.

---

## 5.4 Weak projection blocks under poloidal and toroidal test functions

Fix the velocity expansion $\mathbf v=\sum_\alpha u_\alpha\mathbf B_\alpha$, where $\mathbf B_\alpha\in\{\mathcal P[\phi_\alpha],\,\mathcal T[\psi_\alpha]\}$. The potentials $\phi_\alpha$ and $\psi_\alpha$ are given basis functions of degree $\ell\ge 1$, and the index $\alpha$ records the type, the pair $(\ell,m)$, and the radial index.

**Proposition 5.9 (Test spaces and the block decomposition).** Test functions are taken in three classes.

- Poloidal tests $\mathbf w_{\mathrm{pol}}=\mathcal P[\phi_t]$ and toroidal tests $\mathbf w_{\mathrm{tor}}=\mathcal T[\psi_t]$. Both are divergence-free, so $\int p\,\nabla\cdot\mathbf w=0$.
- Gradient tests $\mathbf w=\nabla\chi$ with $\chi\in H^2(\mathcal V)$ and $\chi|_\Gamma=0$. Then $\nabla\chi|_\Gamma=(\partial_n\chi)\mathbf n$.

The weak form of Proposition 5.2 splits as follows.

**(i) Poloidal equation** (one equation for each $\phi_t$)

$$
\sum_\beta\Big[M_{P\beta}\dot u_\beta+C_{P\beta}u_\beta\Big]+N_P(\mathbf v;\mathbf v)+K_P(\mathbf v,p)
=F_P.
$$

**(ii) Toroidal equation** (one equation for each $\psi_t$)

$$
\sum_\beta\Big[M_{T\beta}\dot u_\beta+C_{T\beta}u_\beta\Big]+N_T(\mathbf v;\mathbf v)+K_T(\mathbf v,p)
=F_T.
$$

**(iii) Pressure equation** (one equation for each $\chi$)

$$
\begin{aligned}
&\int_{\mathcal V}\rho\big(\partial_t\mathbf v+(\mathbf v\cdot\nabla)\mathbf v+2\boldsymbol\Omega\times\mathbf v\big)\cdot\nabla\chi\,dV
+\int_{\mathcal V}\boldsymbol\tau:\nabla\nabla\chi\,dV
-\int_{\mathcal V}p\,\Delta\chi\,dV\\
&\qquad=\int_{\mathcal V}\rho\,\mathbf g_0\cdot\nabla\chi\,dV
+\oint_\Gamma(\mathbf t_{\mathrm{ext}}\cdot\mathbf n)\,\partial_n\chi\,dS.
\end{aligned}
$$

For $X\in\{P,T\}$, the test block $\mathbf B_X$ is $\mathcal P[\phi_t]$ or $\mathcal T[\psi_t]$, and

$$
M_{X\beta}=\int_{\mathcal V}\rho\,\mathbf B_X\cdot\mathbf B_\beta\,dV,\qquad
C_{X\beta}=2\int_{\mathcal V}\rho\,(\boldsymbol\Omega\times\mathbf B_\beta)\cdot\mathbf B_X\,dV,
$$

$$
N_X=\int_{\mathcal V}\rho\,(\mathbf v\cdot\nabla\mathbf v)\cdot\mathbf B_X\,dV,\qquad
K_X(\mathbf v,p)=\int_{\mathcal V}\boldsymbol\tau(\mathbf v,p):\mathbf D(\mathbf B_X)\,dV,
$$

$$
F_X=\int_{\mathcal V}\rho\,\mathbf g_0\cdot\mathbf B_X\,dV+\oint_\Gamma\mathbf t_{\mathrm{ext}}\cdot\mathbf B_X\,dS.
$$

The pressure $p\in L^2$ appears in (i) and (ii) only through $\boldsymbol\tau(\mathbf v,p)$, and in (iii) also through $-\int p\,\Delta\chi$. It is an independent unknown, solved simultaneously with $\mathbf v$. The pressure test $q$ in the mixed weak form enforces the constraint. There is no closed scalar equation that contains only the velocity potentials.

**Proposition 5.10 (The gradient component vanishes).** Let the velocity space be $\mathbb V_\sigma\oplus\nabla X$, where $X=\{\chi\in H^2:\chi|_\Gamma=0\}$, and assume that the pressure test space satisfies $Q_h\supseteq\Delta X$. The constraint $\int q\,\nabla\cdot\mathbf v=0$, applied to $q=\Delta\chi_v$ for the splitting $\mathbf v=\mathbf v_\sigma+\nabla\chi_v$, gives $\|\Delta\chi_v\|_{L^2}^2=0$, hence $\Delta\chi_v=0$. Together with $\chi_v|_\Gamma=0$, this forces $\chi_v\equiv 0$. The velocity of the solution has no gradient component.

*Proof.* $\nabla\cdot\mathbf v=\Delta\chi_v$. The choice $q=\Delta\chi_v$ yields the norm identity. A function that is harmonic and vanishes on the boundary vanishes identically. ∎

---

# 6. Variable density: mass cross-coupling and the baroclinic source

The density is $\rho=\rho_K$ (Section 1.1). If $\rho$ is transported by the flow, it obeys the transport equation of Section 5.1. This chapter writes every channel through which a variable density couples the poloidal equation to the toroidal equation, as an explicit integral.

---

## 6.1 Derivatives of the Cauchy density

Let $q_k(\mathbf x)=1+\sigma_k^2|\mathbf x-\mathbf c_k|^2$ and $\rho_K=\sum_k w_k q_k^{-\alpha_k}$. Differentiation of one summand begins from $\nabla q_k=2\sigma_k^2(\mathbf x-\mathbf c_k)$. The chain rule gives

$$
\nabla(q_k^{-\alpha_k})=-\alpha_k q_k^{-\alpha_k-1}\nabla q_k=-2\alpha_k\sigma_k^2(\mathbf x-\mathbf c_k)\,q_k^{-\alpha_k-1},
$$

and therefore

$$
\nabla\rho_K=-2\sum_k\alpha_k w_k\sigma_k^2(\mathbf x-\mathbf c_k)\,q_k^{-\alpha_k-1}.
$$

The Hessian follows by the product rule on $(\mathbf x-\mathbf c_k)\,q_k^{-\alpha_k-1}$. The derivative of the vector factor is $\mathbf I$. The derivative of the scalar factor is

$$
\nabla(q_k^{-\alpha_k-1})=-(\alpha_k+1)q_k^{-\alpha_k-2}\nabla q_k=-2(\alpha_k+1)\sigma_k^2(\mathbf x-\mathbf c_k)\,q_k^{-\alpha_k-2}.
$$

Collecting the coefficient $-2\alpha_k w_k\sigma_k^2$ already present in the gradient produces

$$
\nabla^2\rho_K
=\sum_k w_k\Big[-2\alpha_k\sigma_k^2 q_k^{-\alpha_k-1}\mathbf I
+4\alpha_k(\alpha_k+1)\sigma_k^4(\mathbf x-\mathbf c_k)(\mathbf x-\mathbf c_k)^{\!\top}q_k^{-\alpha_k-2}\Big].
$$

The Laplacian is the trace. The identity contributes a factor $3$, and the rank-one term contributes $|\mathbf x-\mathbf c_k|^2$, so

$$
\Delta\rho_K
=\sum_k w_k\Big[-6\alpha_k\sigma_k^2 q_k^{-\alpha_k-1}
+4\alpha_k(\alpha_k+1)\sigma_k^4|\mathbf x-\mathbf c_k|^2 q_k^{-\alpha_k-2}\Big].
$$

---

## 6.2 Poloidal–toroidal cross-coupling in the mass block and in the viscous block

**Proposition 6.1 (Selection rule).** Expand the weight, either $\rho$ or $\eta$, as $\sum_{L,M}a_{LM}(r)Y_{LM}$. The bilinear forms between a poloidal field $\mathcal P_{\ell m}$ and a toroidal field $\mathcal T_{\ell'm'}$,

$$
M_{PT}=\int\rho\,\mathcal P\cdot\mathcal T,\qquad
K_{PT}=\int 2\eta\,\mathbf D(\mathcal P):\mathbf D(\mathcal T),
$$

can be nonzero only when the following necessary conditions hold simultaneously:

$$
\ell+\ell'+L\ \text{is odd},\qquad
|\ell-\ell'|\le L\le\ell+\ell',\qquad
m'=m+M.
$$

Under a spherically symmetric weight, whose only component has $L=0$, one has $M_{PT}=K_{PT}=0$, and the poloidal and toroidal blocks separate diagonally.

*Proof.* Under a spherically symmetric weight, $\mathcal P_{\ell m}$ and $\mathcal T_{\ell m}$ belong to the same representation of total angular momentum $J=\ell$. Their parities are $(-1)^{\ell+1}$ and $(-1)^\ell$, which are opposite, so they are orthogonal in an inner product that conserves $J$. Proposition 5.7 is the direct verification for the mass block. When the weight contains $Y_{LM}$, the parity factor $(-1)^L$ and the triangular rule for Gaunt coefficients give the conditions above, and conservation of $M$ gives $m'=m+M$. ∎

**Explicit cross expression for the mass block.** For $\Phi=g Y_{\ell m}$ and $\Psi=f Y_{\ell'm'}$,

$$
M_{PT}
=\int_0^R r\,(rg)'\,f
\Big[\int_{S^2}\rho\ \mathbf e_r\cdot\big(\nabla_{S^2}Y_{\ell m}\times\nabla_{S^2}Y_{\ell'm'}\big)\,d\hat{\boldsymbol\omega}\Big]dr.
$$

The bracket is the integral of the angular structure of $\rho$ against the Poisson bracket of two spherical gradients. Consequently $M_{PT}\ne 0$ holds in general whenever the material has a dipole or a higher angular asymmetry ($L\ge 1$), or whenever the star-shaped boundary $R(\hat{\boldsymbol\omega})$ is not spherically symmetric, so that the domain of integration depends on direction. The poloidal and toroidal equations are then coupled through the mass operator.

---

## 6.3 The baroclinic source

**Proposition 6.2 (Baroclinic source in weak form).** For an arbitrary smooth vector potential $\mathbf A$ and a test velocity $\mathbf w=\nabla\times\mathbf A$,

$$
\boxed{
\int_{\mathcal V}\rho\,\mathbf g_0\cdot\nabla\times\mathbf A\,dV
=\int_{\mathcal V}\mathbf A\cdot\nabla\times(\rho\mathbf g_0)\,dV
+\oint_\Gamma\mathbf n\cdot(\mathbf A\times\rho\mathbf g_0)\,dS.}
$$

When $\mathbf g_0=\nabla W$, one has $\nabla\times(\rho\nabla W)=\nabla\rho\times\nabla W$.

*Proof.* The pointwise identity $\nabla\cdot(\mathbf A\times\mathbf F)=\mathbf F\cdot(\nabla\times\mathbf A)-\mathbf A\cdot(\nabla\times\mathbf F)$, applied to $\mathbf F=\rho\mathbf g_0$ and integrated, produces the volume terms. The divergence theorem converts $\int\nabla\cdot(\mathbf A\times\mathbf F)$ into the stated boundary integral. If $\mathbf g_0=\nabla W$, then $\nabla\times(\rho\nabla W)=\nabla\rho\times\nabla W+\rho\nabla\times\nabla W$, and the second summand vanishes. ∎

(Numerical check: for polynomial fields, the two sides agree on the unit ball to a relative error below $10^{-15}$.)

**Corollary 6.3 (Poloidal and toroidal sources).** Take $\mathbf A_{\mathrm{pol}}=\nabla\phi_t\times\mathbf r$, so that $\nabla\times\mathbf A_{\mathrm{pol}}=\mathcal P[\phi_t]$, and $\mathbf A_{\mathrm{tor}}=\psi_t\mathbf r$, so that $\nabla\times\mathbf A_{\mathrm{tor}}=\mathcal T[\psi_t]$. With $\mathbf g_0=\nabla W$,

$$
\begin{aligned}
F^{\mathrm{bc}}_{\mathrm{pol}}[\phi_t]
&=\int_{\mathcal V}\nabla\phi_t\cdot\big[\nabla\rho\,(\mathbf r\cdot\nabla W)-\nabla W\,(\mathbf r\cdot\nabla\rho)\big]\,dV\\
&\qquad+\oint_\Gamma\rho\,(\nabla\phi_t\times\mathbf r)\cdot(\nabla W\times\mathbf n)\,dS,
\end{aligned}
$$

$$
F^{\mathrm{bc}}_{\mathrm{tor}}[\psi_t]
=\int_{\mathcal V}\psi_t\ \mathbf r\cdot(\nabla\rho\times\nabla W)\,dV
+\oint_\Gamma\rho\,\psi_t\ \mathbf r\cdot(\nabla W\times\mathbf n)\,dS.
$$

In spherical coordinates,

$$
\mathbf r\cdot(\nabla\rho\times\nabla W)=\frac1r\,\mathbf e_r\cdot(\nabla_{S^2}\rho\times\nabla_{S^2}W).
$$

Only a failure of the angular gradients of $\rho$ and of $W$ to be parallel drives a toroidal flow. The first poloidal formula follows from the vector identity

$$
\mathbf r\times(\nabla\rho\times\nabla W)=\nabla\rho\,(\mathbf r\cdot\nabla W)-\nabla W\,(\mathbf r\cdot\nabla\rho),
$$

contracted against $\nabla\phi_t$ after the integration by parts of Proposition 6.2 has been written with $\mathbf A_{\mathrm{pol}}=\nabla\phi_t\times\mathbf r$.

**Proposition 6.4 (Modal selection rule for the baroclinic source).** If the angular harmonics of $\rho$ and of $W$ have degrees $L_1$ and $L_2$, a toroidal source at degree $\ell'$ can be nonzero only when $L_1+L_2+\ell'$ is odd. This is the antisymmetry, under parity, of $\mathbf e_r\cdot(\nabla_{S^2}Y\times\nabla_{S^2}Y')$. When $\rho=\rho(W)$, so that isopycnals and equipotentials coincide, $\nabla\rho\times\nabla W=\mathbf 0$. The volume source vanishes and only the surface term remains.

---

# 7. Rotation: Coriolis cross-coupling and the full nonlinear block system

---

## 7.1 The curl identity for the Coriolis term

**Lemma 7.1.** If $\nabla\cdot\mathbf v=0$ and $\boldsymbol\Omega=\omega\mathbf e_z$ is a constant vector, then

$$
\nabla\times(\boldsymbol\Omega\times\mathbf v)=-(\boldsymbol\Omega\cdot\nabla)\mathbf v=-\omega\,\partial_z\mathbf v.
$$

*Proof.* The product rule for the curl of a cross product is

$$
\nabla\times(\boldsymbol\Omega\times\mathbf v)
=\boldsymbol\Omega\,(\nabla\cdot\mathbf v)-\mathbf v\,(\nabla\cdot\boldsymbol\Omega)+(\mathbf v\cdot\nabla)\boldsymbol\Omega-(\boldsymbol\Omega\cdot\nabla)\mathbf v.
$$

The first term vanishes by incompressibility. The second and third vanish because $\boldsymbol\Omega$ is constant. The remaining term is $-\omega\partial_z\mathbf v$. ∎

(The sign has been checked on a poloidal field.)

**Proposition 7.2 (Weak form of the Coriolis term).** For $\mathbf w=\nabla\times\mathbf A$,

$$
\begin{aligned}
C(\mathbf w,\mathbf v)
&:=2\int_{\mathcal V}\rho\,(\boldsymbol\Omega\times\mathbf v)\cdot\mathbf w\,dV\\
&=\int_{\mathcal V}\mathbf A\cdot\big[2\nabla\rho\times(\boldsymbol\Omega\times\mathbf v)-2\omega\rho\,\partial_z\mathbf v\big]\,dV
+2\oint_\Gamma\mathbf n\cdot\big(\mathbf A\times\rho\,\boldsymbol\Omega\times\mathbf v\big)\,dS,
\end{aligned}
$$

and $C$ is skew in $(\mathbf w,\mathbf v)$: $C(\mathbf w,\mathbf v)=-C(\mathbf v,\mathbf w)$ and $C(\mathbf v,\mathbf v)=0$.

*Proof.* Apply Proposition 6.2 to $\mathbf F=2\rho\,\boldsymbol\Omega\times\mathbf v$. The curl splits as

$$
\nabla\times(\rho\mathbf F_0)=\nabla\rho\times\mathbf F_0+\rho\nabla\times\mathbf F_0
$$

with $\mathbf F_0=\boldsymbol\Omega\times\mathbf v$, and Lemma 7.1 replaces $\nabla\times\mathbf F_0$ by $-\omega\partial_z\mathbf v$. Skew-symmetry is the scalar identity $(\boldsymbol\Omega\times\mathbf v)\cdot\mathbf w=-(\boldsymbol\Omega\times\mathbf w)\cdot\mathbf v$. ∎

The Coriolis force therefore does no work. It does not change the energy budget. It only redistributes energy among modes.

---

## 7.2 Selection rules and explicit cross coefficients

**Proposition 7.3 (Selection rule).** Let $R$ be constant and let $\rho=\rho(r)$. In a spherical-harmonic expansion, the form $C$ satisfies the following.

1. The order $m$ is conserved, because $\boldsymbol\Omega\times$ is invariant under rotations about the $z$-axis.
2. Poloidal–poloidal blocks and toroidal–toroidal blocks connect only a single degree $\ell$.
3. A block between a poloidal field of degree $\ell$ and a toroidal field of degree $\ell'$ can be nonzero only under the **necessary** selection rule $\ell'=\ell\pm 1$. The actual coefficient depends on an angular overlap integral and may still vanish.

*Proof.* The map $\boldsymbol\Omega\times$ is a vector operator of rank one. By the Wigner–Eckart theorem the change in total angular momentum $J$ lies in $\{0,\pm 1\}$, and both the poloidal and the toroidal field have $J=\ell$. The parity of a poloidal field is $(-1)^{\ell+1}$ and the parity of a toroidal field is $(-1)^\ell$, while $\boldsymbol\Omega\times$ preserves parity. Fields of the same type are therefore coupled only at $\Delta J=0$, and fields of opposite type only at $\Delta J=\pm 1$. ∎

**Proposition 7.4 (Explicit cross coefficients).** Let $\mathcal V$ be the ball $r<R$ with $R$ constant, and let $\rho=\rho(r)$. Let the poloidal potential be $\Phi=g(r)Y_{\ell m}$ and the toroidal test potential be $\psi_t=f(r)Y_{\ell'm}$, with $\ell'=\ell\pm 1$. Set

$$
h=\Lambda_\ell g-(rg)',\quad
\mathcal L^{+}_\ell h=h'-\frac{\ell h}{r},\quad
\mathcal L^{-}_\ell h=h'+\frac{(\ell+1)h}{r},\quad
\Delta_\ell g=g''+\frac{2g'}{r}-\frac{\Lambda_\ell g}{r^2},\quad
a_{\ell m}=\sqrt{\frac{\ell^2-m^2}{4\ell^2-1}}.
$$

Then

$$
\boxed{
C_{TP}=-2\omega\int_0^R\rho\,f\ \kappa\,\big[\mathcal L^{\pm}_\ell h+r\,\Delta_\ell g\big]\,r^2\,dr,}\qquad
\kappa=\begin{cases}
a_{\ell+1,m},&\ell'=\ell+1\ (\mathcal L^{+}),\\[2pt]
a_{\ell m},&\ell'=\ell-1\ (\mathcal L^{-}),
\end{cases}
$$

and $C_{PT}=-C_{TP}^{\!\top}$.

*Derivation.* Take $\mathbf A=\psi_t\mathbf r$ and $\mathbf F=\mathcal P[\Phi]$. Under spherical symmetry of $\rho$, one has $\nabla\rho\parallel\mathbf r$, so the volume term $\mathbf r\cdot(\nabla\rho\times\,\cdot\,)$ vanishes. On the sphere, $\mathbf n\parallel\mathbf r$, so the surface term $\mathbf n\cdot(\psi_t\mathbf r\times\,\cdot\,)$ vanishes as well. Proposition 7.2 therefore reduces to

$$
C_{TP}=-2\omega\int\rho\,\psi_t\,\mathbf r\cdot\partial_z\mathbf v\,dV.
$$

The radial contraction of a derivative splits as $\mathbf r\cdot\partial_z\mathbf v=\partial_z(\mathbf r\cdot\mathbf v)-v_z$. For a poloidal field, Lemma 5.5 gives $\mathbf r\cdot\mathbf v=L^2\Phi$ and

$$
\mathbf v=\nabla\big(\partial_r(r\Phi)\big)-\mathbf r\,\Delta\Phi,
$$

so the Cartesian component along $\mathbf e_z$ is $v_z=\partial_z\partial_r(r\Phi)-z\Delta\Phi$. Combining these identities,

$$
\begin{aligned}
\mathbf r\cdot\partial_z\mathbf v
&=\partial_z(L^2\Phi)-\partial_z\partial_r(r\Phi)+z\Delta\Phi\\
&=\partial_z\big(L^2\Phi-\partial_r(r\Phi)\big)+z\Delta\Phi.
\end{aligned}
$$

With $\Phi=g(r)Y_{\ell m}$, one has $L^2\Phi=\Lambda_\ell g\,Y_{\ell m}$ and $\partial_r(r\Phi)=(rg)'Y_{\ell m}$, hence

$$
L^2\Phi-\partial_r(r\Phi)=h Y_{\ell m}.
$$

The radial Laplacian of a separated harmonic is $\Delta\Phi=(\Delta_\ell g)\,Y_{\ell m}$, and $z=r\cos\theta$, so

$$
\mathbf r\cdot\partial_z\mathbf v
=\partial_z(h Y_{\ell m})+r\cos\theta\,(\Delta_\ell g)\,Y_{\ell m}.
$$

The standard recurrences

$$
\partial_z\big(h\,Y_{\ell m}\big)
=a_{\ell+1,m}\,\mathcal L^{+}_\ell h\ Y_{\ell+1,m}
+a_{\ell m}\,\mathcal L^{-}_\ell h\ Y_{\ell-1,m},
$$

$$
\cos\theta\,Y_{\ell m}=a_{\ell+1,m}Y_{\ell+1,m}+a_{\ell m}Y_{\ell-1,m}
$$

project onto $Y_{\ell\pm 1,m}$ and give the boxed coefficient. The second recurrence is consistent with the gradient formula for the solid harmonic $r^\ell Y_{\ell m}$: if $h=r^\ell$, then $\mathcal L^{+}_\ell h=\ell r^{\ell-1}-\ell r^{\ell-1}=0$. ∎

**Numerical check.** Take $\rho=1+r^2$, $R=1$, and $m=0$.

- From $\ell=1$ to $\ell'=2$, with $g=r^2(1+r)$ and $f=r^3$: a direct triple integral gives $-2.1811658591$, and the formula gives $-2.1811658591$.
- From $\ell=2$ to $\ell'=1$, with $g=r^3(1+r)$ and $f=r^2$: the direct value is $-9.6013101430$, and the formula gives $-9.6013101430$.
- From $\ell=1$ to $\ell'=3$, and the same-degree poloidal–poloidal block from $\ell=1$ to $\ell'=1$, both are $O(10^{-16})$, in agreement with the selection rule.

**Corollary 7.5 (The toroidal component cannot be omitted).** If $\omega\ne 0$ and if $\mathcal L^{\pm}_\ell h+r\Delta_\ell g$ is not identically zero, the term $C_{TP}\,u_P$ produced in the toroidal equation by the poloidal coefficients is nonzero. The residual of the toroidal equation cannot be cancelled by poloidal coefficients, so the toroidal coefficient $s_{\ell'km}$ must be retained as an independent unknown. The same conclusion applies to the baroclinic volume source $F^{\mathrm{bc}}_{\mathrm{tor}}\ne 0$ of Section 6.3, which is nonzero when the angular gradients of $\rho$ and of $W$ are not parallel: only a toroidal coefficient can carry it.

---

## 7.3 The full nonlinear block system

Write the velocity-coefficient vector as $\mathbf u=(\mathbf q,\mathbf s)$, poloidal then toroidal. Items (i) and (ii) of Proposition 5.9 combine into the $2\times 2$ block system

$$
\boxed{
\begin{pmatrix}
M_{PP}&M_{PT}\\
M_{TP}&M_{TT}
\end{pmatrix}
\begin{pmatrix}\dot{\mathbf q}\\ \dot{\mathbf s}\end{pmatrix}
+\begin{pmatrix}
C_{PP}&C_{PT}\\
C_{TP}&C_{TT}
\end{pmatrix}
\begin{pmatrix}\mathbf q\\ \mathbf s\end{pmatrix}
+\begin{pmatrix}N_P\\ N_T\end{pmatrix}
+\begin{pmatrix}K_P(\mathbf u,p)\\ K_T(\mathbf u,p)\end{pmatrix}
=\begin{pmatrix}F_P\\ F_T\end{pmatrix}.}
$$

The integrals that define the blocks are those of Proposition 5.9, and they satisfy the following.

| Block | Origin | Symmetry and coupling |
|---|---|---|
| $M_{XY}$ | inertia | symmetric positive definite on $\ell\ge 1$; $M_{PT}\ne 0$ comes from an angular asymmetry of the density or of the boundary (Proposition 6.1) |
| $C_{XY}$ | Coriolis | skew as a whole; $C_{PT}$ and $C_{TP}$ connect $\ell\leftrightarrow\ell\pm 1$ (Propositions 7.3 and 7.4); $C_{PP}$ and $C_{TT}$ stay at the same degree |
| $K_X$ | viscosity | depends on $(\mathbf u,p)$; $K_{PT}\ne 0$ comes from an angular inhomogeneity of $\eta$ (Proposition 6.1); the Newton tangent is Section 5.3 |
| $N_X$ | convection | trilinear; poloidal and toroidal components couple to each other |
| $F_X$ | gravity, centrifugation, baroclinicity, traction | the baroclinic part $F^{\mathrm{bc}}_X$ is Corollary 6.3; the surface traction is Corollary 5.3 |

The system is solved together with the pressure equation (iii), the constraint $\int q\,\nabla\cdot\mathbf v=0$ (Proposition 5.10), density transport, and the boundary kinematics of Section 5.1. Every basic unknown $(\mathbf v,p,\rho,R)$ appears, and the pressure is an independent unknown.

**Geometric connection of moving basis functions, and the ALE weak form.** When the boundary $R(\hat{\boldsymbol\omega},t)$ deforms in time, the basis fields $\mathbf B_\alpha$ depend on $R(t)$ through $\zeta=r/R$ and through the factors $R^2$ and $R$. Their partial time derivative $\partial_t\mathbf B_\alpha$ at fixed $\mathbf x$ induces, inside the inertial term, the volume matrix

$$
(\mathbf G)_{\beta\alpha}=\int_{\mathcal V}\rho\,\mathbf B_\beta\cdot\partial_t\mathbf B_\alpha\,dV,\qquad
\partial_t\mathbf M_\Gamma=\mathbf G+\mathbf G^{\!\top}+\mathbf M_{\partial\mathcal V}+\mathbf M_{\partial_t\rho},
$$

where $\mathbf M_{\partial\mathcal V}$ is the Reynolds boundary term of Section 9.1. The matrix $\mathbf G$ is **not** a term that can be absorbed into the surface velocity $\mathbf c$. Its algebraic place depends on the mode.

- **Quasi-static route (Mode B).** The inertial block $\mathbf M$, the connection $\mathbf G$, and the convection $\mathbf N$ are all absent (Section 2.8, B1). At each geological step the Stokes system (8.4′) is reassembled and solved on the currently frozen geometry $R^{(n)}$. Velocity is not carried from one step to the next, so a jump in the geometry requires no modal remapping.
- **Dynamic free surface.** Take the ALE mesh velocity $\mathbf w_{\rm mesh}(\mathbf x)=(r/R)(\partial_t R)\,\mathbf e_r$, the radial stretch that holds $\hat{\boldsymbol\omega}$ and $\zeta$ fixed. On the boundary, $\mathbf w_{\rm mesh}\cdot\mathbf n=\partial_t R\,(\mathbf n\cdot\mathbf e_r)$, in agreement with Proposition 9.1. The source terms of this paper use the Eulerian weak form in fixed spatial coordinates: $\mathbf G$ is retained as written above, and the inertial convection still uses the physical velocity $\mathbf v$. If an implementation switches to an ALE time derivative, it must at the same time replace $\partial_t|_{\mathbf x}$ by $\partial_t|_{\rm ALE}$ and replace the convective velocity by $\mathbf v-\mathbf w_{\rm mesh}$. That alternative form must not be added again to the fixed-space matrix $\mathbf G$. Because $\nabla\cdot\mathbf w_{\rm mesh}\ne 0$, the Reynolds boundary term, $\mathbf G$, and the density term must still be assembled explicitly and included in the source $\boldsymbol{\mathcal S}$ of Section 9.1.

**Remark (geometry of the boundary terms).** When $R=R(\hat{\boldsymbol\omega})$, the analytic reductions of Section 7.4 no longer apply. The domain of integration and the normal $\mathbf n$ depend on direction, and every block must be integrated directly on $\mathcal V(R)$ by the integrals of Proposition 5.9.

---

## 7.4 Rigid modes and solvability

**Proposition 7.6.** On the six rigid modes of Proposition 5.6, the viscous block and the pressure-coupling block both vanish. Consequently:

1. A necessary condition for the quasi-static problem (with $M\dot{\mathbf u}$ and $N$ omitted) to have a solution is that the external force do no work on the rigid modes:

$$
\int_{\mathcal V}\rho\,\mathbf g_0\,dV+\oint_\Gamma\mathbf t_{\mathrm{ext}}\,dS=\mathbf 0,\qquad
\int_{\mathcal V}\rho\,\mathbf x\times\mathbf g_0\,dV+\oint_\Gamma\mathbf x\times\mathbf t_{\mathrm{ext}}\,dS=\mathbf 0.
$$

The resultant of self-gravity vanishes identically. The centrifugal force requires the centre of mass to lie on the axis of rotation and that axis to be a principal axis of inertia ($I_{xz}=I_{yz}=0$).

2. The solution is determined only up to a rigid mode. It is fixed by the six constraints $\int_{\mathcal V}\rho\,\mathbf v\,dV=\mathbf 0$ and $\int_{\mathcal V}\rho\,\mathbf x\times\mathbf v\,dV=\mathbf 0$.

*Proof.* For item 1, take $\mathbf w$ to be a rigid mode. Then $\mathbf D(\mathbf w)=\mathbf 0$ and $\nabla\cdot\mathbf w=0$, so the stress term and the pressure term of Proposition 5.2 both vanish. Only the external-force terms remain, and they must vanish. For item 2, the rigid modes are the kernel of the viscous block, and additional constraints are required to determine the solution. ∎

## 7.5 Range of validity

The block system of this chapter is the complete dynamic weak form for $(\mathbf v,p,\rho,R)$. It imposes no geometric restriction on $\mathbf v$: the poloidal potential and the toroidal potential are present simultaneously, and the pressure enters as an independent unknown through $\boldsymbol\tau(\mathbf v,p)$ and through $-p\,\nabla\cdot\mathbf w$. The map $\mu(I)$ is not monotone in $\mathbf D$, so this chapter does not assert existence or uniqueness for the continuous problem. Discrete solvability of the block system depends on an inf–sup condition for the discrete spaces and on local well-posedness of the constitutive law. Those questions are treated separately in the discrete model from Chapter 8 onward. **The production route uses only the viscous block and the load block of this chapter.** The quasi-static system (8.4′) drops $M\dot{\mathbf u}$, the convective block, and the Coriolis block. The density is fixed (hypothesis A-ρ). The pressure is supplied independently by the Poisson problem of Section 8.7. The full block system is used only in the dynamic mode (Section 9.9).

**Summary.** At the continuous level the model states, in full, the governing equations and the natural boundary conditions for a poloidal–toroidal double potential, a variable density, Coriolis coupling, and an independent pressure. At the discrete level, both the quasi-static route and the dynamic mode are given an algebraic form under Track A (a global solve) and under Track B (a spherical-harmonic block-diagonal surrogate; Appendix A.4). The deeper analytic consequences of a non-monotone rheology, and discrete well-posedness, are delimited by explicitly registered proof obligations (PO-1, PO-11, and PO-12; the dynamic mode adds PO-2 and PO-4; Track B is PO-8). Those obligations are not proved, and every conclusion that depends on them is marked. Within the scientific scope of a computational self-consistent field, the model closes. Its self-consistency is limited to the hypotheses and proof obligations that have been registered.

---

# 8. Radial Galerkin reduction and spatial assembly

This chapter projects the continuous mixed weak form of Chapters 5–7 (Propositions 5.2, 5.9, and 5.10, and the full block system of Section 7.3) onto a finite-dimensional space $\mathbb V_h\times\mathbb Q_h$. The radial direction uses $K=4$ Jacobi modes and the angular direction uses a spherical-harmonic expansion. Both are Galerkin truncations. The resulting system is a reduced model of the three-dimensional problem, not an identity transformation of it. Symbols, spaces, and trace spaces obey Appendix A of this paper (A.1 for disambiguation of symbols, A.2 for function spaces and trace spaces).

---

## 8.1 The star-shaped domain and the Jacobi radial basis

**Definition 8.1 (Geometry).** Let $R\in C^3(S^2\times[0,T])$ satisfy $0<R_{\min}\le R\le R_{\max}$. The material domain is the star-shaped set

$$
\mathcal V(R)=\{(r,\hat{\boldsymbol\omega}):0\le r<R(\hat{\boldsymbol\omega},t),\ \hat{\boldsymbol\omega}\in S^2\},\qquad
\zeta=\frac{r}{R(\hat{\boldsymbol\omega},t)}\in[0,1).
$$

**Definition 8.2 (Radial basis).** For $\ell\ge 1$ and $k=0,1,2,3$,

$$
\boxed{\ \psi_{\ell k}(\zeta)=\zeta^{\ell}\,P_k^{(0,\,2\ell+2)}(2\zeta-1),\qquad \Lambda_\ell=\ell(\ell+1),\ }
$$

where $P_k^{(\alpha,\beta)}$ is the standard Jacobi polynomial.

**Proposition 8.3 (Orthogonality, endpoint values, and the power structure).**

1. $\displaystyle\int_0^1\psi_{\ell j}\,\psi_{\ell k}\,\zeta^2\,d\zeta=\dfrac{\delta_{jk}}{2k+2\ell+3}$.
2. $\psi_{\ell k}(1)=1$, $\psi_{\ell k}'(1)=\ell+k(k+2\ell+3)$, and $\psi_{\ell k}''(1)$ is the quantity $\kappa_{\ell k}$ tabulated in Section 10.2.1.
3. $\psi_{\ell k}(\zeta)=\sum_{j=0}^{k}a_{\ell kj}\,\zeta^{\ell+j}$, with

$$
a_{\ell k0}=(-1)^k\binom{k+2\ell+2}{k},\qquad
a_{\ell kk}=\binom{2k+2\ell+2}{k}\ne 0.
$$

The constant term equals $(-1)^k$ only for $k=0$. The binomial factor is required by the standard normalization of $P_k^{(0,2\ell+2)}$ and by the explicit polynomials of Section 10.1. For $\ell=1$ the constant terms are $1,-5,15,-35$ and the leading coefficients are $1,6,28,120$.

*Proof of (1).* Set $x=2\zeta-1$, so $\zeta=(1+x)/2$ and $d\zeta=dx/2$. Then

$$
\psi_{\ell j}\psi_{\ell k}\zeta^2\,d\zeta
=2^{-(2\ell+3)}(1+x)^{2\ell+2}P_j^{(0,2\ell+2)}(x)P_k^{(0,2\ell+2)}(x)\,dx,
$$

because $\zeta^{2\ell+2}=2^{-(2\ell+2)}(1+x)^{2\ell+2}$ and the extra factor $\tfrac12$ comes from $d\zeta$. The weight $(1-x)^{\alpha}(1+x)^{\beta}$ with $\alpha=0$ and $\beta=2\ell+2$ is exactly the Jacobi weight of these polynomials. Its squared norm is

$$
h_k
=\frac{2^{\alpha+\beta+1}}{2k+\alpha+\beta+1}
\cdot
\frac{\Gamma(k+\alpha+1)\Gamma(k+\beta+1)}{k!\,\Gamma(k+\alpha+\beta+1)}.
$$

With $\alpha=0$, one has $\Gamma(k+1)=k!$ and $\Gamma(k+\beta+1)=\Gamma(k+\alpha+\beta+1)$, so the Gamma ratio equals $1$. The remaining factor is $h_k=2^{2\ell+3}/(2k+2\ell+3)$. Multiplication by $2^{-(2\ell+3)}$ produces $1/(2k+2\ell+3)$.

*Proof of (2).* The endpoint value of the Jacobi polynomial is $P_k^{(\alpha,\beta)}(1)=\binom{k+\alpha}{k}$. For $\alpha=0$ this equals $1$, so $\psi_{\ell k}(1)=1$. The derivative identity

$$
\frac{d}{dx}P_k^{(\alpha,\beta)}(x)=\frac12(k+\alpha+\beta+1)P_{k-1}^{(\alpha+1,\beta+1)}(x)
$$

and $P_{k-1}^{(1,\beta+1)}(1)=\binom{k}{k-1}=k$ give, at $x=1$ and $\alpha=0$,

$$
P_k'(1)=\frac12 k(k+\beta+1).
$$

Differentiating $\psi(\zeta)=\zeta^\ell P(2\zeta-1)$ therefore yields

$$
\psi'(1)=\ell\cdot P(1)+2P'(1)=\ell+k(k+\beta+1).
$$

The parameter $\beta=2\ell+2$ converts the second summand into $k(k+2\ell+3)$.

*Proof of (3).* The evaluation $P_k^{(\alpha,\beta)}(-1)=(-1)^k\binom{k+\beta}{k}$ follows by retaining, in the binomial expansion, the unique term that does not vanish at $x=-1$. With $\alpha=0$ and $\beta=2\ell+2$,

$$
P_k^{(0,2\ell+2)}(-1)=(-1)^k\binom{k+2\ell+2}{k}.
$$

The change of argument $x=2\zeta-1$ does not alter the constant term, and multiplication by $\zeta^\ell$ shifts every power by $\ell$, so the lowest coefficient is $a_{\ell k0}=P_k^{(0,2\ell+2)}(-1)$. The leading coefficient of $P_k^{(\alpha,\beta)}$ is $2^{-k}\binom{2k+\alpha+\beta}{k}$. Substituting $x=2\zeta-1$, whose leading term is $2\zeta$, multiplies by $2^k$ and cancels $2^{-k}$. The leading coefficient of $\psi_{\ell k}$ is therefore $\binom{2k+2\ell+2}{k}$, which is nonzero. These endpoint values have also been checked by exact rational arithmetic. ∎

The opposite parameter order $(2\ell+2,0)$ produces the weight $(1-x)^{2\ell+2}$. That weight does not match the weight $\zeta^{2\ell+2}$ obtained from the spherical volume element $\zeta^2\,d\zeta$ multiplied by $\zeta^{2\ell}$. It is not used.

**Proposition 8.4 (Regularity at the centre; proof obligation PO-3, see Appendix A.5).** [Weak integrability of the rheology at the centre: when $\ell=1$ and $R$ is not constant, $\nabla\mathbf v=O(r^{-1})$ fails to lie in $L^\infty$. If $\mathbf D\sim r^{-1}$, then $\dot\gamma\sim r^{-1}$, the viscosity satisfies $\eta\propto 1/\dot\gamma\sim r$, the deviatoric stress $\boldsymbol\tau=2\eta\mathbf D$ is $O(1)$, and the strain-energy density $\boldsymbol\tau:\mathbf D(\mathbf w)$ is $O(r^{-1})$. Against the volume measure $r^2\,dr$ one has $\int_0^\varepsilon r\,dr<\infty$, so the strain energy of the weak form is locally weakly integrable in a neighbourhood of the origin and the algebraic matrix entries can be assembled. This is not an existence or uniqueness theorem for a weak solution of the continuous or discrete problem. If a strong rheological solution is required, one may switch to the optional even basis $\psi^{\rm ev}$ (Appendix A.2; valid only when $R$ is constant).] For $\ell\ge 1$, $k\le 3$, and $R\in C^3(S^2)$, the basis fields

$$
\mathbf V_{\ell km}=\nabla\times\nabla\times\big(R^2\psi_{\ell k}Y_{\ell m}\,\mathbf r\big),\qquad
\mathbf W_{\ell km}=\nabla\times\big(R\,\psi_{\ell k}Y_{\ell m}\,\mathbf r\big)
$$

belong to $H^1(\mathcal V)$. The toroidal field $\mathbf W_{\ell km}$ is Lipschitz near the origin. The poloidal field $\mathbf V_{\ell km}$ is Lipschitz near the origin when $R$ is constant or when $\ell\ge 2$. Only when $\ell=1$ and $R$ is not constant is it merely bounded in a neighbourhood of the origin, with gradient of order $O(r^{-1})$ (the gradient lies in $L^2$ and not in $L^\infty$).

*Proof.* By Proposition 8.3(3), inside the ball $r<R_{\min}$,

$$
R^2\psi_{\ell k}(r/R)Y_{\ell m}=\sum_j a_{\ell kj}R^{2-n}r^{n}Y_{\ell m},\qquad n=\ell+j\ge 1.
$$

Each term is a homogeneous function $\Phi_n=r^n h_n(\hat{\boldsymbol\omega})$ of degree $n$, with $h_n=a_{\ell kj}R^{2-n}Y_{\ell m}\in C^3$. Lemma 5.5 gives

$$
\mathcal P[\Phi_n]=\nabla(\partial_r(r\Phi_n))-\mathbf r\,\Delta\Phi_n=(n+1)\nabla\Phi_n-\mathbf r\,\Delta\Phi_n,
$$

which is homogeneous of degree $n-1$. Its gradient is homogeneous of degree $n-2$, so $|\nabla\mathcal P[\Phi_n]|\le C r^{n-2}$. Then

$$
\int_0^\varepsilon r^{2(n-2)}r^2\,dr=\int_0^\varepsilon r^{2n-2}\,dr<\infty
$$

because $n\ge 1$, and the field lies in $H^1$. For $n\ge 2$ the gradient is bounded, so the field is Lipschitz. The case $n=1$ occurs only for $\ell=1$ and $j=0$. Then $\Phi_1=a\,R(\hat{\boldsymbol\omega})\,r\,Y_{1m}$. If $R$ is constant this is a linear function and $\mathcal P[\Phi_1]$ is a constant vector. If $R$ is not constant, $\mathcal P[\Phi_1]$ is homogeneous of degree zero and depends on direction, so its gradient is $O(r^{-1})$. For the toroidal field, $\Psi_n=r^n R^{1-n}Y_{\ell m}a$ gives $\mathcal T[\Psi_n]=\nabla\Psi_n\times\mathbf r$, of degree $n\ge 1$, with bounded gradient. ∎

The factor $\zeta^\ell$ makes the lowest power of $\Phi$ exactly $n=\ell$. When $R$ is constant, $r^\ell Y_{\ell m}$ is a harmonic polynomial in $\mathbf x$ and is therefore smooth. Every remaining term, of degree $n>\ell$, may contain an odd power and is only finitely smooth. Proposition 8.4 is precisely the $H^1$ conclusion at that degree of smoothness. The law $\mu(I)$ needs a finite $\dot\gamma$ at the centre. Except for the case $\ell=1$ with nonconstant $R$ already stated, one has $\mathbf D\in L^\infty$.

---

## 8.2 The double-potential expansion and the degrees of freedom

**Definition 8.5 (Velocity expansion).**

$$
\boxed{
\begin{aligned}
\Phi(\mathbf x,t)&=R^2(\hat{\boldsymbol\omega},t)\sum_{\ell=1}^{L_{\max}}\sum_{m=-\ell}^{\ell}\sum_{k=0}^{3}q_{\ell km}(t)\,\psi_{\ell k}(\zeta)\,Y_{\ell m}(\hat{\boldsymbol\omega}),\\
\Psi(\mathbf x,t)&=R(\hat{\boldsymbol\omega},t)\sum_{\ell,m,k}s_{\ell km}(t)\,\psi_{\ell k}(\zeta)\,Y_{\ell m}(\hat{\boldsymbol\omega}),
\end{aligned}}
$$

$$
\mathbf v=\mathcal P[\Phi]+\mathcal T[\Psi]=\sum_{\ell,m,k}\big(q_{\ell km}\mathbf V_{\ell km}+s_{\ell km}\mathbf W_{\ell km}\big).
$$

The vector of velocity degrees of freedom is $\mathbf u=(\mathbf q,\mathbf s)$, with $[q]=[s]=\mathrm T^{-1}$. **The meaning of $K=4$** is that each pair $(\ell,m)$ and each potential keeps four radial modes, so each pair $(\ell,m)$ has eight velocity degrees of freedom. The total number is

$$
N_u=8\sum_{\ell=1}^{L_{\max}}(2\ell+1)=8L_{\max}(L_{\max}+2).
$$

Degree $\ell=0$ carries no velocity degree of freedom (Proposition 5.6(2)). Every matrix inversion is restricted to $\ell\ge 1$. The fields $\mathbf V_{\ell km}$ and $\mathbf W_{\ell km}$ are exactly divergence-free (Lemma 5.5). Their dependence on $R$ enters every integral and is not frozen at the nodes.

---

## 8.3 The global kinetic-energy mass operator

**Definition 8.6.** $(\mathbf M_\Gamma)_{\alpha\beta}=\int_{\mathcal V(R)}\rho\,\mathbf B_\alpha\cdot\mathbf B_\beta\,dV$, where $\mathbf B_\alpha\in\{\mathbf V_{\ell km},\mathbf W_{\ell km}\}$.

**Proposition 8.7 (Symmetric positive definiteness)** [complete proof; independent of any proof obligation]. Let $0<\rho_{\min}\le\rho\le\rho_{\max}$ and let $R\in C^3$. Then $\mathbf M_\Gamma$ is symmetric, and for every $\mathbf u\ne\mathbf 0$,

$$
\mathbf u^{\!\top}\mathbf M_\Gamma\mathbf u=\int_{\mathcal V}\rho|\mathbf v|^2\,dV\ge\rho_{\min}\|\mathbf v\|_{L^2}^2>0.
$$

*Proof.* Symmetry and positive semi-definiteness are immediate from the definition. Suppose $\mathbf v=\mathbf 0$. Proposition 5.6(1) gives $L^2\Phi=r v_r=0$ and $L^2\Psi=0$, so inside the ball $r<R_{\min}$ the potentials $\Phi=\Phi_0(r)$ and $\Psi=\Psi_0(r)$ depend only on $r$. Equating coefficients of $r^n$ in $\Phi$, one has $R^{2-n}H_n(\hat{\boldsymbol\omega})=c_n$, where

$$
H_n=\sum_{\ell\le n,\,m}\Big(\sum_k a_{\ell k,\,n-\ell}q_{\ell km}\Big)Y_{\ell m}
$$

contains no component of degree $\ell=0$. Hence $H_n=c_n R^{n-2}$. The spherical mean of both sides, using $R>0$ so that $\langle R^{n-2}\rangle>0$, forces $c_n=0$, and therefore $H_n=0$. Spherical harmonics are linearly independent, so for every $(\ell,m)$ and every $n\ge\ell$,

$$
\sum_k a_{\ell k,\,n-\ell}q_{\ell km}=0.
$$

That is, the polynomial $\sum_k q_{\ell km}\psi_{\ell k}(\zeta)$ vanishes identically. The functions $\psi_{\ell 0},\ldots,\psi_{\ell 3}$ have distinct degrees and are linearly independent, so $q_{\ell km}=0$. The same argument with $R^{1-n}$ in place of $R^{2-n}$ gives $s_{\ell km}=0$. ∎

**Proposition 8.8 (Spherically symmetric reference geometry; diagonal blocks)** [complete proof; independent of any proof obligation]. Let $R\equiv R_0$ be constant and let $\rho=\rho(r)$. Then $\mathbf M_\Gamma$ is block diagonal in $(\ell,m)$, the poloidal and toroidal fields do not couple, and, by Proposition 5.7,

$$
\boxed{
\begin{aligned}
(\mathbf M^P_\ell)_{ij}
&=\Lambda_\ell^2\!\int_0^{R_0}\!\rho\,g_i g_j\,dr
+\Lambda_\ell\!\int_0^{R_0}\!\rho\,(rg_i)'(rg_j)'\,dr,\\
(\mathbf M^T_\ell)_{ij}
&=\Lambda_\ell\!\int_0^{R_0}\!\rho\,f_i f_j\,r^2\,dr,
\end{aligned}}
$$

where $g_i=R_0^2\psi_{\ell i}(r/R_0)$ and $f_i=R_0\psi_{\ell i}(r/R_0)$. Each block is $4\times 4$. At constant density $\rho=\rho_0$,

$$
\mathbf M^P_\ell=\rho_0 R_0^5\,\Lambda_\ell\big(\Lambda_\ell\tilde{\mathbf N}_\ell+\tilde{\mathbf P}_\ell\big),\qquad
\mathbf M^T_\ell=\rho_0 R_0^5\,\Lambda_\ell\,\mathrm{diag}\Big(\frac1{2k+2\ell+3}\Big),
$$

$$
(\tilde{\mathbf N}_\ell)_{ij}=\int_0^1\psi_{\ell i}\psi_{\ell j}\,d\zeta,\qquad
(\tilde{\mathbf P}_\ell)_{ij}=\int_0^1(\zeta\psi_{\ell i})'(\zeta\psi_{\ell j})'\,d\zeta.
$$

The toroidal block is strictly diagonal in the Jacobi basis (Proposition 8.3(1)). The poloidal block is not diagonal, but $\tilde{\mathbf N}_\ell$ and $\tilde{\mathbf P}_\ell$ are symmetric positive definite, because the families $\psi_{\ell i}$ and $(\zeta\psi_{\ell i})'$ are linearly independent. Numerical values are given in Section 10.1. Every block entry is an $L^2$ polynomial (quadratic for the poloidal block, linear for the toroidal block). This is why $\mathbf M_\Gamma$ is an angular differential operator and not a nodal local constant.

**Proposition 8.9 (A general star-shaped surface; Gaunt coupling).** When $R=R(\hat{\boldsymbol\omega})$, or when $\rho$ depends on angle, the angular integral in $(\mathbf M_\Gamma)_{\alpha\beta}$ contains the product of a weight $w(\hat{\boldsymbol\omega})$ with $Y_{\ell m}Y_{\ell'm'}$ and with their surface gradients. If $w=\sum_{L,M}w_{LM}Y_{LM}$, a block between $(\ell,m)$ and $(\ell',m')$ can be nonzero only under the Gaunt conditions

$$
\int Y_{\ell m}Y_{\ell'm'}Y_{LM}\,d\Omega\ne 0
\ \Rightarrow\
|\ell-\ell'|\le L\le\ell+\ell',\quad
\ell+\ell'+L\ \text{even},\quad
M=-m-m'.
$$

If the weight is band-limited ($w_{LM}=0$ for $L>L_w$), then $\mathbf M_\Gamma$ is a strictly banded sparse block matrix, with block bandwidth $|\ell-\ell'|\le L_w$. For a general smooth $R$ and $\rho$, the coefficients $w_{LM}$ decay in $L$ and may be truncated to a numerical bandwidth at a tolerance $\varepsilon_w$. Cross blocks between poloidal and toroidal fields obey the selection rule of Proposition 6.1 (parity: $\ell+\ell'+L$ odd).

---

## 8.4 The discrete mixed weak form and the saddle-point system

**Discrete spaces.**

$$
\mathbb V_h^{\sigma}=\operatorname{span}\{\mathbf V_{\ell km},\mathbf W_{\ell km}\},\qquad
\mathbb V_h^{g}=\{\nabla\chi_h:\chi_h\in X_h\},\quad
X_h\subset\{\chi\in H^2:\chi|_\Gamma=0\},
$$

$$
\mathbb Q_h=\operatorname{span}\{\psi_{\ell k}(\zeta)Y_{\ell m}\},\qquad
0\le\ell\le L_p,\quad 0\le k\le K_p-1.
$$

The pressure space includes $\ell=0$, the hydrostatic component. The pressure layer extends the notation by $\psi_{0k}(\zeta):=P_k^{(0,2)}(2\zeta-1)$. These modes of degree $\ell=0$ are used only for pressure and do not enter the velocity basis. The remaining truncations satisfy $L_p\le L_{\max}$ and $K_p\le K$. Matching $X_h$ with $\mathbb Q_h$ is the inf–sup proof obligation PO-2, which this paper does not claim to have proved. Every function in $\mathbb V_h^\sigma$ is exactly divergence-free, so $b(\mathbf w,q)\equiv 0$ for $\mathbf w\in\mathbb V_h^\sigma$.

**Geometric connection of moving basis functions.** The quasi-static route has no inertial block, and $\mathbf G$ does not appear. In the dynamic mode, $\mathbf G$ and the ALE convection $\mathbf N_{\rm ALE}$ of the relative velocity $\mathbf v_{\rm rel}=\mathbf v-\mathbf w_{\rm mesh}$ are those of Section 7.3. They must be assembled explicitly inside $\mathbf N(\mathbf u)$ and inside the inertial block of $\mathbf A_{\sigma\sigma}$.

**Projection onto test functions.** Testing the weak form of Proposition 5.2 with $\mathbf w=\mathbf V_{\ell km}$, with $\mathbf W_{\ell km}$, and with $\nabla\chi_h$ produces the three groups of Proposition 5.9: poloidal (i), toroidal (ii), and pressure (iii). Write the full velocity coefficient as $\tilde{\mathbf u}=(\mathbf u,\mathbf x)$, where $\mathbf x$ holds the coefficients of $\chi_h$ and $\mathbf p$ holds the pressure coefficients.

**The global system** [discrete solvability holds only if the proof obligations PO-1 and PO-2 hold]. The blocks below are defined by the integrals of Proposition 5.9: $M_{X\beta}$ (inertia), $C_{X\beta}$ (Coriolis), $N_X$ (convection), $K_X(\mathbf v,p)$ (viscosity, including $\boldsymbol\tau(\mathbf v,p)=2\eta(p,\mathbf D)\mathbf D$), and $F_X$ (gravity, centrifugation, baroclinicity, traction, and the load $\mathbf T_{\rm tide}$).

$$
\boxed{
\begin{pmatrix}
\mathbf A(\tilde{\mathbf u},\mathbf p)&-\mathbf B^{\!\top}\\
\mathbf B&\mathbf 0
\end{pmatrix}
\begin{pmatrix}\tilde{\mathbf u}\\ \mathbf p\end{pmatrix}
=\begin{pmatrix}\mathbf f_{\rm ext}\\ \mathbf 0\end{pmatrix},}
$$

$$
\mathbf A=\begin{pmatrix}\mathbf A_{\sigma\sigma}&\mathbf A_{\sigma g}\\ \mathbf A_{g\sigma}&\mathbf A_{gg}\end{pmatrix},\qquad
\mathbf A_{\sigma\sigma}
=\begin{pmatrix}
\mathbf M_{PP}/\Delta t+\mathbf C_{PP}+\mathbf K_{PP}&
\mathbf M_{PT}/\Delta t+\mathbf C_{PT}+\mathbf K_{PT}\\
\mathbf M_{TP}/\Delta t+\mathbf C_{TP}+\mathbf K_{TP}&
\mathbf M_{TT}/\Delta t+\mathbf C_{TT}+\mathbf K_{TT}
\end{pmatrix}
+\mathbf N(\mathbf u),
$$

$$
\mathbf B=\begin{pmatrix}\mathbf 0&\mathbf B_g\end{pmatrix},\quad
(\mathbf B_g)_{q\chi}=\int_{\mathcal V}q\,\Delta\chi\,dV,\qquad
\mathbf f_{\rm ext}=\big(\mathbf F_P,\mathbf F_T,\mathbf F_g\big).
$$

The quotient $\mathbf M_{XY}/\Delta t$ is the time-discrete inertial term. It is set to zero on the quasi-static route; see (8.4′). The dependence of $\mathbf K_{XY}$ on $(\mathbf u,p)$ is linearized by the Newton tangent of Section 5.3. The structure of the blocks is as follows.

| Block | Structure and origin |
|---|---|
| $\mathbf M_{PP},\mathbf M_{TT}$ | diagonal in $(\ell,m)$ on a spherically symmetric geometry (Proposition 8.8); a Gaunt band on a general geometry (Proposition 8.9) |
| $\mathbf M_{PT}$ | zero on a spherically symmetric geometry; nonzero when $\rho$ or $R$ is angularly inhomogeneous (Proposition 6.1) |
| $\mathbf C_{PT},\mathbf C_{TP}$ | connect $\ell\leftrightarrow\ell\pm 1$ at fixed $m$ (Proposition 7.3); coefficients in Proposition 7.4; the Coriolis form is skew as a whole |
| $\mathbf K_{XY}$ | viscous blocks depending on $(\mathbf u,p)$; $\mathbf K_{PT}\ne 0$ comes from an angular inhomogeneity of $\eta$ |
| $\mathbf F_X$ | baroclinic terms in Proposition 6.2, Corollary 6.3, and Proposition 6.4; surface traction is a natural boundary term (Corollary 5.3) |

The Jacobian of the Newton iteration is

$$
\begin{pmatrix}
\mathbf A_T&-\mathbf B^{\!\top}+\mathbf K_p\\
\mathbf B&\mathbf 0
\end{pmatrix},\qquad
\mathbf K_p=\frac{\partial\mathbf K}{\partial\mathbf p}.
$$

The dependence $\eta=\eta(p,\mathbf D)$ makes $\mathbf K_p\ne 0$. Pressure appears in both $\mathbf A$ and $\mathbf B$, so the Jacobian is not in general a symmetric saddle-point matrix.

**The quasi-static production system (8.4′).** The quasi-static route (Section 2.8, B1) restricts the system above by $\mathbf M=\mathbf N=\mathbf C=\mathbf 0$ and $\mathbf G=\mathbf 0$, and takes the velocity space to be the exactly divergence-free space $\mathbb V_h^\sigma$ directly. The conclusion $\mathbf x=\mathbf 0$ of Proposition 8.10 is adopted here as a choice of space, not as a hypothesis of a theorem. Testing the weak form of Proposition 5.2 with $\mathbf w=\mathbf B_\alpha\in\mathbb V_h^\sigma$ makes $\int p\,\nabla\cdot\mathbf w$ vanish exactly, and one obtains

$$
\boxed{
\begin{pmatrix}
\mathbf K(\eta)&\mathbf B_{\rm rb}^{\!\top}\\
\mathbf B_{\rm rb}&\mathbf 0
\end{pmatrix}
\begin{pmatrix}\mathbf u\\ \boldsymbol\lambda\end{pmatrix}
=\begin{pmatrix}\mathbf F\\ \mathbf 0\end{pmatrix},
\qquad
\begin{aligned}
K_{\alpha\beta}(\eta)&=\int_{\mathcal V}2\eta\,\mathbf D(\mathbf B_\beta):\mathbf D(\mathbf B_\alpha)\,dV,\\
F_\alpha&=\int_{\mathcal V}\rho_K\,\mathbf g_{\rm eff}(\omega_n)\cdot\mathbf B_\alpha\,dV
-\int_{\mathcal V}\mathbf T_{\rm tide}:\mathbf D(\mathbf B_\alpha)\,dV
+\oint_\Gamma\mathbf t_{\rm ext}\cdot\mathbf B_\alpha\,dS.
\end{aligned}}
$$

The matrix $\mathbf B_{\rm rb}$ is that of Section 8.5 and of Appendix A.3. Its rows are $\int\rho_K\mathbf B_\alpha\,dV$ and $\int\rho_K\mathbf x\times\mathbf B_\alpha\,dV$. Pressure enters $\mathbf K$ only through $\eta(\tilde p,\mathbf D)$, and it is determined by the Poisson problem of Section 8.7. The nonlinear coupling of the two is closed by the Picard iteration of Section 10.4.2. Because the velocity space is exactly divergence-free, this system **does not need** a velocity–pressure inf–sup condition. The obligation PO-2 constrains only the fully implicit benchmark below. Integrals use the angular quadrature of Section 9.4 and Gauss–Jacobi quadrature in radius for the weight $\zeta^{2\ell+2}$. The tensor $\mathbf D(\mathbf B_\alpha)$ is evaluated analytically, pointwise, from Lemma 5.5.

**Lemma 8.12 (Well-posedness of the quasi-static system).** Assume $\rho_K\ge\rho_{\min}>0$, $\eta\ge\eta_{\rm lo}>0$ almost everywhere (Section 2.8), that $\mathcal V$ is connected, and that $\operatorname{rank}\mathbf B_{\rm rb}=6$. Then $\mathbf K(\eta)$ is symmetric positive definite on $\ker\mathbf B_{\rm rb}$, and (8.4′) has a unique solution. For every $\gamma_{\rm AL}>0$, the matrix $\mathbf K_\gamma:=\mathbf K+\gamma_{\rm AL}\mathbf B_{\rm rb}^{\!\top}\mathbf B_{\rm rb}$ is symmetric positive definite on the whole space, and replacing $\mathbf K$ by $\mathbf K_\gamma$ does not change the solution of (8.4′).

*Proof.* Symmetry follows from the definition. The energy identity

$$
\mathbf u^{\!\top}\mathbf K\mathbf u=\int 2\eta\,|\mathbf D(\mathbf v)|^2\,dV\ge 2\eta_{\rm lo}\|\mathbf D(\mathbf v)\|_{L^2}^2
$$

shows that a vanishing energy forces $\mathbf D(\mathbf v)=\mathbf 0$. On a connected domain, $\mathbf v=\mathbf a+\boldsymbol\theta\times\mathbf x$. If in addition $\mathbf B_{\rm rb}\mathbf u=\mathbf 0$, then

$$
\int\rho_K|\mathbf v|^2\,dV
=\mathbf a\cdot\int\rho_K\mathbf v\,dV+\boldsymbol\theta\cdot\int\rho_K\mathbf x\times\mathbf v\,dV=0.
$$

The bound $\rho_K\ge\rho_{\min}$ forces $\mathbf v=\mathbf 0$, and linear independence of the basis forces $\mathbf u=\mathbf 0$. Thus $\mathbf K$ is positive definite on $\ker\mathbf B$. Combined with the full row rank of $\mathbf B$, the saddle-point system is nonsingular. This is the standard conclusion. If $\mathbf u^{\!\top}\mathbf K_\gamma\mathbf u=0$, then $\mathbf B\mathbf u=\mathbf 0$ and $\mathbf u^{\!\top}\mathbf K\mathbf u=0$, so $\mathbf u=\mathbf 0$. A solution of (8.4′) satisfies $\mathbf B\mathbf u=\mathbf 0$, hence $\gamma_{\rm AL}\mathbf B^{\!\top}\mathbf B\mathbf u=\mathbf 0$, and the solution is unchanged. ∎

**Solution by the range-space method, exact up to the iteration tolerance.** Set

$$
\gamma_{\rm AL}:=\|\operatorname{diag}\mathbf K\|_\infty\big/\|\operatorname{diag}(\mathbf B_{\rm rb}^{\!\top}\mathbf B_{\rm rb})\|_\infty.
$$

Apply seven iterations of PCG to $\mathbf K_\gamma$, preconditioned by the block-diagonal inverse $\mathbf K_{\gamma,B1}^{-1}$ (the inverse of one $8\times 8$ principal sub-block for each $(\ell,m)$; Appendix A.4). This produces $\mathbf y_0=\mathbf K_\gamma^{-1}\mathbf F$ and $\mathbf Y=\mathbf K_\gamma^{-1}\mathbf B_{\rm rb}^{\!\top}\in\mathbb R^{N_u\times 6}$. Solve the $6\times 6$ symmetric positive-definite system $(\mathbf B_{\rm rb}\mathbf Y)\boldsymbol\lambda=\mathbf B_{\rm rb}\mathbf y_0$, and set $\mathbf u=\mathbf y_0-\mathbf Y\boldsymbol\lambda$. The spatial contrast $\eta_{\rm hi}/\eta_{\rm lo}$ determines the number of PCG iterations and must be reported together with that number. Newton acceleration, using the tangent of Section 5.3, is optional. The default is Picard.

**Relation to the fully implicit system.** The global system that retains inertia, convection, Coriolis coupling, and an independent pressure is the **theoretical benchmark** of the dynamic free-surface mode. The quasi-static route does not solve it. The system (8.4′) is its exact restriction in the quasi-static limit. The production model contains neither an elastic strain nor an equation of state. Pressure is the multiplier of the incompressibility constraint.

**Proposition 8.10 (Conditional discrete pressure-recovery theorem; conditional on the hypothesis $\mathbb Q_h\supseteq\Delta X_h$ and on the proof obligation PO-2)** [depends on PO-2: the inf–sup matching $\mathbb Q_h\supseteq\Delta X_h$]. If $\mathbb Q_h\supseteq\Delta X_h$ (the hypothesis of Proposition 5.10), then $\mathbf x=\mathbf 0$ and the velocity contains only components from $\mathbb V_h^\sigma$. If in addition $\mathbb Q_h=\Delta X_h$, then $\mathbf B_g$ is square and nonsingular. The pressure coefficients are then determined by the discrete form of (iii), $\mathbf B_g^{\!\top}\mathbf p=\mathbf A_{g\sigma}\mathbf u-\mathbf F_g$, and they return to the velocity equation through $\eta(p,\mathbf D)$. Pressure is an unknown solved simultaneously with velocity. The two form one nonlinear system.

*Proof.* The constraint $\mathbf B\tilde{\mathbf u}=\mathbf 0$ means $\int q\,\Delta\chi_h=0$ for every $q\in\mathbb Q_h$. The choice $q=\Delta\chi_h$ gives $\Delta\chi_h=0$, and $\chi_h|_\Gamma=0$ then gives $\chi_h=0$ (Proposition 5.10). The entries of $\mathbf B_g$ are $\int q\,\Delta\chi$. When $\mathbb Q_h=\Delta X_h$, this matrix is the Gram matrix of $\Delta X_h$. It is positive definite, hence nonsingular. ∎

---

## 8.5 The rigid-body kernel and its algebraic constraints

A rigid motion, whether a translation or a rotation, satisfies $\mathbf D(\mathbf w)=\mathbf 0$ (Proposition 5.6(3)). For a discrete velocity of degree $\ell\ge 1$, the six linear functionals

$$
\boxed{
\begin{aligned}
(\mathbf C_{\rm rigid})_{a\alpha}&=\int_{\mathcal V}\rho\,\mathbf B_\alpha\cdot\mathbf e_a\,dV,\\
(\mathbf C_{\rm rigid})_{3+a,\alpha}&=\int_{\mathcal V}\rho\,(\mathbf x\times\mathbf B_\alpha)\cdot\mathbf e_a\,dV,
\end{aligned}
\quad a=1,2,3}
$$

enforce zero momentum and zero angular momentum through $\mathbf C_{\rm rigid}\mathbf u=\mathbf 0$. With a Lagrange multiplier $\boldsymbol\lambda=(\boldsymbol\lambda_P,\boldsymbol\lambda_L)\in\mathbb R^6$, the full system is

$$
\begin{pmatrix}
\mathbf A(\tilde{\mathbf u},\mathbf p)&-\mathbf B^{\!\top}&\mathbf C_{\rm rigid}^{\!\top}\\
\mathbf B&\mathbf 0&\mathbf 0\\
\mathbf C_{\rm rigid}&\mathbf 0&\mathbf 0
\end{pmatrix}
\begin{pmatrix}\tilde{\mathbf u}\\ \mathbf p\\ \boldsymbol\lambda\end{pmatrix}
=\begin{pmatrix}\mathbf f_{\rm ext}\\ \mathbf 0\\ \mathbf 0\end{pmatrix}.
$$

**Proposition 8.11 (The constraint matrix on a spherically symmetric reference geometry)** [the matrix entries are exact integrals; solvability of the global system that contains this constraint depends on PO-1]. Let $R\equiv R_0$ and $\rho=\rho(r)$, and take the real spherical harmonics $Y_{1,a}=\sqrt{3/4\pi}\,\hat\omega_a$ for $a=x,y,z$. Only modes of degree $\ell=1$ appear, and

$$
C_{{\rm P},a}\big|_{q_{1ka}}
=\frac{4\sqrt{3\pi}}{3}\int_0^{R_0}\rho\,(2rg_k+r^2 g_k')\,dr,\qquad
C_{{\rm L},a}\big|_{s_{1ka}}
=\frac{4\sqrt{3\pi}}{3}\int_0^{R_0}\rho\,f_k\,r^3\,dr,
$$

where $g_k=R_0^2\psi_{1k}(r/R_0)$ and $f_k=R_0\psi_{1k}(r/R_0)$. Every other entry vanishes. At constant density $\rho_0$,

$$
C_{{\rm P},a}\big|_{q_{1ka}}
=\frac{4\sqrt{3\pi}}{3}\,\rho_0 R_0^4
\quad(k=0,1,2,3),\qquad
C_{{\rm L},a}\big|_{s_{1ka}}
=\frac{4\sqrt{3\pi}}{15}\,\rho_0 R_0^5\,\delta_{k0}.
$$

Hence $\mathrm{rank}\,\mathbf C_{\rm rigid}=6$. The angular-momentum row contains only $k=0$, by Jacobi orthogonality:

$$
\int_0^1\psi_{1k}\zeta^3\,d\zeta
=\int_0^1\zeta^4 P_k^{(0,4)}(2\zeta-1)\,d\zeta
=\delta_{k0}/5.
$$

*Proof.* For the poloidal mode $g(r)Y_{1,z}$ of degree $\ell=1$, write $c=\sqrt{3/(4\pi)}$ and $Y_{1,z}=c\cos\theta$. Lemma 5.5 gives $v_r=2(g/r)Y_{1,z}$ and $\mathbf v_S=r^{-1}(rg)'\nabla_{S^2}Y_{1,z}$. Since $\nabla_{S^2}(\cos\theta)=-\sin\theta\,\mathbf e_\theta$,

$$
v_\theta=-\frac{(rg)'}{r}\,c\sin\theta,\qquad v_\varphi=0.
$$

The Cartesian component is $v_z=v_r\cos\theta-v_\theta\sin\theta$, so

$$
v_z=c\Big[2\frac{g}{r}\cos^2\theta+\frac{(rg)'}{r}\sin^2\theta\Big].
$$

The angular integrals $\int_{S^2}\cos^2\theta\,d\Omega=4\pi/3$ and $\int_{S^2}\sin^2\theta\,d\Omega=8\pi/3$, together with $(rg)'/r=g/r+g'$, collapse the bracket to

$$
\int_{S^2}v_z\,d\Omega=c\cdot\frac{8\pi}{3}\Big(2\frac{g}{r}+g'\Big).
$$

The prefactor simplifies by $\sqrt{3/(4\pi)}\cdot 8\pi/3=4\sqrt{3\pi}/3$. Multiplication by the radial measure $\rho r^2\,dr$ produces the momentum row. At constant density the integrand $2rg+r^2 g'$ is $(gr^2)'$, and

$$
\int_0^{R_0}(gr^2)'\,dr=g(R_0)R_0^2.
$$

Every basis function satisfies $\psi_{1k}(1)=1$, so $g_k(R_0)=R_0^2$ and $g_k(R_0)R_0^2=R_0^4$, independently of $k$.

For the toroidal mode $f(r)Y_{1,z}$, the velocity is a rotation about the $z$-axis, $v_\varphi=cf\sin\theta$. The angular-momentum density contributes

$$
\int_{S^2}(\mathbf x\times\mathbf v)\cdot\mathbf e_z\,d\Omega
=c\cdot\frac{8\pi}{3}\,f r,
$$

and the radial integral against $\rho r^2\,dr$ is the angular-momentum row. At constant density, $f_k=R_0\psi_{1k}(r/R_0)$ and $r=R_0\zeta$ give

$$
\int_0^{R_0}f_k r^3\,dr=R_0^5\int_0^1\psi_{1k}\zeta^3\,d\zeta.
$$

The mode $\psi_{10}(\zeta)=\zeta$ is the Jacobi function of index $k=0$ at $\ell=1$, and Proposition 8.3(1) with weight $\zeta^2$ yields

$$
\int_0^1\psi_{1k}\zeta^3\,d\zeta
=\int_0^1\psi_{1k}\psi_{10}\zeta^2\,d\zeta
=\frac{\delta_{k0}}{5},
$$

because the normalization constant at $\ell=1$, $k=0$ is $2\cdot 0+2\cdot 1+3=5$. Multiplication by $4\sqrt{3\pi}/3$ produces the factor $4\sqrt{3\pi}/15$.

Every other degree, and every field of the wrong type, integrates to zero by orthogonality of spherical harmonics against the constant vectors $\mathbf e_a$. Momentum projects only onto the poloidal fields of degree $\ell=1$. Under a spherically symmetric density, a toroidal field of degree $\ell=1$ has no net momentum. A poloidal mode has no $e_\varphi$ component and no angular momentum. The integrals have been checked by exact arithmetic for $k=0,1,2,3$. ∎

**A general star-shaped surface.** When $R$ is not constant, the mode $\Phi=R\,r\,Y_{1m}$ of indices $\ell=1$, $k=0$ is no longer a linear function. The six rigid motions need not lie in $\mathbb V_h^\sigma$. They coincide with the six modes of indices $\ell=1$, $k=0$ only when $R$ is constant. The definition of $\mathbf C_{\rm rigid}$, as an integral against discrete functions, is unchanged. Whether it has full rank, and the smallest singular value of the viscous block, must be reported at every outer step (Appendix A.3). The alternative fixing (b), namely $q_{1m0}=s_{1m0}=0$, is equivalent to (a) only when $R$ is constant and $\rho=\rho(r)$. In every other case one uses (a).

**Solvability (Proposition 7.6).** The quasi-static problem requires compatibility of the resultant force and of the resultant torque: $\int\rho\mathbf g_0\,dV+\oint\mathbf t_{\rm ext}\,dS=\mathbf 0$, together with the corresponding torque condition. If the condition fails, the problem has no static solution. The incompatibility must be reported. It cannot be absorbed by the multiplier $\boldsymbol\lambda$.

---

## 8.6 Free-surface kinematics and the mass constraint

The surface is defined by $F=r-R(\hat{\boldsymbol\omega},t)=0$. The kinematic condition is $\partial_t F+\mathbf v\cdot\nabla F=0$, with $\nabla F=\mathbf e_r-R^{-1}\nabla_{S^2}R$ and with $\nabla_{S^2}$ the gradient on the unit sphere. At $r=R$ one obtains, as in Proposition 9.1 and as already expanded in Section 5.1,

$$
\boxed{\ 
\partial_t R
=v_r-\frac1R\nabla_{S^2}R\cdot\mathbf v_S\Big|_{r=R}
=\frac{\mathbf v\cdot\mathbf n}{\mathbf n\cdot\mathbf e_r}.\ }
$$

The reduction $\partial_t R\approx v_r$ is legitimate only on a sphere, or when tangential transport is neglected. The surface velocities $v_r$ and $\mathbf v_S$ are recovered from $\mathbf u$ by the trace formulae of Section 10.2.2. The mass constraint

$$
\int_{S^2}\int_0^{R(\hat{\boldsymbol\omega},t)}\rho\,r^2\,dr\,d\Omega=M
$$

determines the component of degree $\ell=0$ in $R$ (Appendix A.3). That component is not advanced by the kinematic equation. The meanings of time in the dynamic and static regimes are those of Sections 9.9 and 10.4.1. They must not be mixed.

---

## 8.7 Pressure recovery: an independent scalar pressure-Poisson weak form

**Physical basis.** The quasi-static momentum equation $\nabla\cdot\boldsymbol\sigma+\rho_K\mathbf g_{\rm eff}=\mathbf 0$, with $\boldsymbol\sigma=-p\mathbf I+\boldsymbol\tau$, gives $\nabla p=\mathbf f$, where

$$
\mathbf f:=\rho_K\mathbf g_{\rm eff}+\nabla\cdot\boldsymbol\tau,\qquad
\boldsymbol\tau=\mathbf T_{\rm tide}+\boldsymbol\tau_{\rm flow}.
$$

In general $\mathbf f$ is not a gradient: $\nabla\times\mathbf f=\nabla\rho_K\times\nabla W+\nabla\times(\nabla\cdot\boldsymbol\tau)\ne\mathbf 0$, and the failure is pronounced for a rotating asteroid. The irrotational part is carried by the pressure. The rotational part of the load $\rho_K\mathbf g_{\rm eff}$, which is the baroclinic source of Section 6.3, drives the double-potential flow and is balanced by the rotational part of $\nabla\cdot\boldsymbol\tau$. That balance is exactly the content of (8.4′), so at an exact equilibrium $\mathbf f$ is a gradient. The pressure is therefore taken to be the $L^2$ projection of $\mathbf f$ onto gradient fields, which is the scalar Poisson problem below. The problem is **independent of the velocity–pressure saddle-point system**. At every round of the Picard iteration in Section 10.4.2 it supplies a continuous pressure $p^{k+1}$ on the whole field.

**Strong form and boundary data.** The normal component of the free-surface traction $\boldsymbol\sigma\mathbf n=\mathbf t_{\rm ext}$ supplies the Dirichlet datum

$$
p|_\Gamma=p_\Gamma
:=\mathbf n\cdot\boldsymbol\tau\,\mathbf n-\mathbf n\cdot\mathbf t_{\rm ext}
\ \overset{\mathbf t_{\rm ext}=-p_{\rm ext}\mathbf n}{=}\
p_{\rm ext}+\mathbf n\cdot\mathbf T_{\rm tide}\mathbf n+\mathbf n\cdot\boldsymbol\tau_{\rm flow}\mathbf n.
$$

(The tangential component of the traction is a consistency condition on $\mathbf v$; see Section 10.2.3. It does not constrain $p$.)

**Weak form.** Write $p=p_b+p_0$, where $p_b$ is any $H^1$ lifting of $p_\Gamma$. Seek $p_0\in H^1_0(\mathcal V)$ such that

$$
\boxed{\ 
\int_{\mathcal V}\nabla p_0\cdot\nabla q\,dV
=\int_{\mathcal V}\big(\rho_K\mathbf g_{\rm eff}+\nabla\cdot\boldsymbol\tau\big)\cdot\nabla q\,dV
-\int_{\mathcal V}\nabla p_b\cdot\nabla q\,dV
\qquad\forall q\in H^1_0(\mathcal V).\ }
$$

Because $q|_\Gamma=0$, no traction integral $\oint_\Gamma(\ldots)\,q\,dS$ appears on the right-hand side. Traction enters through the Dirichlet datum $p_\Gamma$. Writing the same identity for a general $q\in H^1$ as a “boundary-integral term” would introduce the dimensionally inconsistent integral $\oint\mathbf t_{\rm ext}\cdot\mathbf n\,q\,dS$ and would require the unknown $\partial_n p$. That form is not used. The left-hand side is the inner product of $H^1_0$. The Poincaré inequality gives coercivity, so the Lax–Milgram theorem guarantees that **the solution exists and is unique**. There is no free constant, in contrast with the mixed system of Section 8.4.

**Discretization (shifted Jacobi and spherical-harmonic scalar basis, with a vanishing trace).**

$$
\mathbb Q_h^0
=\operatorname{span}\{\,(1-\zeta)\,\psi_{\ell k}(\zeta)\,Y_{\ell m}\,\},
\quad 0\le\ell\le L_p,\ 0\le k\le K_p-1,
\qquad
\psi_{0k}(\zeta):=P_k^{(0,2)}(2\zeta-1).
$$

The factor $(1-\zeta)$ makes every basis function vanish on $\Gamma$. For $\ell\ge 1$, $\psi_{\ell k}$ is the function of Section 8.1. Degree $\ell=0$ is reserved for pressure. The boundary lifting is $p_b=\sum_{\ell,m}\hat p_{\ell m}\,\zeta^{\ell}Y_{\ell m}$, where the coefficients $\hat p_{\ell m}$ are the spherical-harmonic analysis of the nodal values $p_\Gamma(\hat{\boldsymbol\omega}_a)$, using the quadrature of step 1 in Section 9.4. The stiffness matrix $(\mathbf K_p)_{ab}=\int_{\mathcal V}\nabla\phi_a\cdot\nabla\phi_b\,dV$ is symmetric positive definite and is solved by PCG with a block-diagonal preconditioner. The quadrature is the same as the assembly of $\mathbf M_\Gamma$: Gauss–Jacobi in radius and spherical quadrature in angle.

**Interior data for the right-hand side (every field is evaluated at interior quadrature points).**

- The effective gravity $\mathbf g_{\rm eff}=\nabla U+\omega^2\mathbf x_\perp$ is the interior RTFP gradient. The gradient converges classically and does not require a finite-part Hessian. Step 1 of Section 10.4.4 must therefore compute $\nabla U$ on the additional interior quadrature set $\mathcal X_{\rm vol}$.
- For $\nabla\cdot\mathbf T_{\rm tide}$, the interior density $\rho_K$ is smooth. The deviator satisfies

$$
\big(\nabla\cdot\operatorname{dev} H_W\big)_i
=\partial_j\partial_i\partial_j W-\frac13\partial_i(\Delta W)
=\partial_i\Delta W-\frac13\partial_i\Delta W
=\frac23\partial_i\Delta W,
$$

so $\nabla\cdot\operatorname{dev} H_W=\tfrac23\nabla\Delta W$. In the interior, where the solid angle is the full sphere ($\vartheta=1$), $\Delta U=-4\pi G\rho_K$ and the centrifugal potential $\tfrac12\omega^2(x_1^2+x_2^2)$ has Laplacian $2\omega^2$. Hence $\Delta W=2\omega^2-4\pi G\rho_K$. The centrifugal Hessian is constant, so it contributes nothing to $\nabla\Delta W$, and $\nabla\Delta W=-4\pi G\nabla\rho_K$. Therefore

$$
\nabla\cdot\operatorname{dev} H_W=-\frac{8\pi}{3}G\nabla\rho_K,
$$

and, with $\mathbf T_{\rm tide}=C_T\rho_p\nu_0 d_p^2\operatorname{dev} H_W$,

$$
\nabla\cdot\mathbf T_{\rm tide}=-\frac{8\pi}{3}G\,C_T\rho_p\nu_0 d_p^2\,\nabla\rho_K.
$$

The gradient $\nabla\rho_K$ is that of Section 6.1. No third derivative of the Hessian is required. This identity is an interior identity. On the boundary, $\vartheta\ne 1$, and the face-centre evaluation of Section 10.3 is used for the stress itself.
- For the flow stress, incompressibility gives $\nabla\cdot(2\mathbf D)=\Delta\mathbf v$. Indeed,

$$
\big(\nabla\cdot(\nabla\mathbf v+\nabla\mathbf v^{\!\top})\big)_i
=\Delta v_i+\partial_i(\nabla\cdot\mathbf v)=\Delta v_i.
$$

If $\boldsymbol\tau_{\rm flow}=2\eta\mathbf D$, the product rule yields

$$
\nabla\cdot\boldsymbol\tau_{\rm flow}=\eta\,\Delta\mathbf v+2\mathbf D\,\nabla\eta.
$$

The velocity is the Stokes solution $\mathbf u^k$ of the previous Picard round. At the start of a geological step the converged coefficients of the previous geological step are reused, without change, on the new geometry. The first geological step takes $\mathbf u^0=\mathbf 0$, hence $\boldsymbol\tau_{\rm flow}=\mathbf 0$, and $p$ is then the generalized hydrostatic pressure that includes the tide. The viscosity $\eta$ and its gradient $\nabla\eta$ are frozen at $(\tilde p^k,\mathbf D(\mathbf u^k))$ and are computed from $\eta(\tilde p,\mathbf D)$ by the chain rule, with $\nabla\tilde p=\mathbf 1_{\{p>p_{\min}\}}\nabla p$. The initial value affects only the number of Picard iterations. It does not affect the converged solution.

**Coupling to the velocity (Picard iteration).** Inside one geological step, iterate $k=0,1,\ldots$: from the frozen pair $(\mathbf u^k,\tilde p^k)$, solve the present problem for $p^{k+1}$; from $\tilde p^{k+1}$ and $\mathbf D(\mathbf u^k)$, assemble $\mathbf K(\eta^{k+1/2})$ and solve (8.4′) for $\mathbf u^{k+1}$; stop when criterion (a) of Section 2.15 holds. The algorithm is Section 10.4.2. At convergence, $(\mathbf u,p)$ satisfies both the divergence-free Galerkin momentum equation and the present Poisson problem, which is the quasi-static equilibrium of one and the same stress. Convergence of this fixed-point iteration is not proved (PO-11). Failure to converge is judged by criterion (a) of Section 2.15. The pressure passed downstream is $\tilde p=\max([p]_+,p_{\min})$ (Section 2.8, B5), and the fraction of the domain in tension must be reported. The strong residual $\mathbf r_{\rm rot}:=\mathbf f-\nabla p$ vanishes at a continuous exact solution. On a discrete solution it measures the rotational part of $\mathbf f$ that has not been absorbed by the Galerkin projection onto $\mathbb V_h^\sigma$. It converges as $K$ and $L_{\max}$ increase, and it is reported as a resolution indicator under criterion (b) of Section 2.15.

**Relation to Section 8.4.** On the quasi-static route the velocity is given by (8.4′) and the pressure is given by the present section. The Picard iteration closes the two as one nonlinear system. The fully implicit saddle-point system of Section 8.4, which retains inertia and an independent pressure, and Proposition 8.10, are only the theoretical benchmark of the dynamic mode. They are not solved inside the quasi-static loop.

---

## 8.8 The internal gravitational load

The gravitational field

$$
\mathbf g(\mathbf x)=G\int_{\mathcal V}\rho(\mathbf y)\frac{\mathbf x-\mathbf y}{|\mathbf x-\mathbf y|^3}\,d\mathbf y
$$

depends on the internal density. If only the surface $R(\hat{\boldsymbol\omega})$ is known and $\rho$ is not, the internal gravity is not unique. A surface RTFP evaluation is an approximation only when the density is known and fixed, when the mass is concentrated in a thin shell, when a precomputed multipole expansion is used, or when a change of the internal mass is corrected separately. The entries of the load vector are

$$
\boxed{\ (\mathbf F_g)_\alpha=\int_{\mathcal V}\rho\,\mathbf g_{\rm RTFP}\cdot\mathbf B_\alpha\,dV,\ }
$$

integrated over $\mathcal V(R)$ and against every test function of degree $\ell\ge 1$.

---

## 8.9 Rigour and the error report

The radial Jacobi truncation ($K=4$) and the angular spherical-harmonic truncation ($L_{\max}$) are both Galerkin truncations. Convergence to the three-dimensional problem is possible only as $K\to\infty$ and $L_{\max}\to\infty$, and only if the basis is complete. A finite $K$ is not equivalent to the original problem. Every run must report the following.

1. Convergence in $K$, through $E_{4\to 8}=\|\mathbf u_8-\mathbf u_4\|/\|\mathbf u_8\|$, and convergence in $L_{\max}$.
2. On the quasi-static route: the viscosity contrast $\eta_{\rm hi}/\eta_{\rm lo}$, the number of preconditioned PCG iterations, the number of Picard iterations, and the residuals of criteria (a) and (b) in Section 2.15. In the dynamic mode: $\mathrm{cond}(\mathbf M_\Gamma|_{\ell\ge 1})$ (Proposition 8.7) and the condition number of each block in $\ell$.
3. The relative rigid-body multiplier $\varepsilon_\lambda$ of criterion (b) in Section 2.15, together with $\operatorname{rank}\mathbf B_{\rm rb}$ and its smallest singular value.
4. The strong residual $\|\mathbf r_{\rm rot}\|$ (Section 8.7). The dynamic fully implicit benchmark also reports a numerical inf–sup indicator of the pressure space (PO-2, not proved).
5. The tangential traction inconsistency $\delta_{na}$ (Section 10.2.3).
6. The RTFP gravitational error.
7. When Track B is used (Appendix A.4), the quantities $\varepsilon_B$ and $\mathbf r_B$.

The resulting mixed two-dimensional and three-dimensional discrete system is a reduced model subject to stated hypotheses and to error control. It is not an unconditional, zero-error reduction of the original three-dimensional free-boundary problem.

---

# 9. Conservative transport on the surface and the semi-Lagrangian advance

This chapter obeys the normative conventions of Appendix A (A.1–A.4). The velocity degrees of freedom are the poloidal and toroidal coefficients $\mathbf u=(\mathbf q,\mathbf s)$ (Section 7.3). The conserved variable $\mathbf m$ is tied to the velocity coefficients by the global mass operator $\mathbf M_\Gamma(R,\rho)$. Track A is the reference. Track B is a local approximation whose truncation error is stated explicitly.

**Range of application (mandatory).** Transport of the modal momentum $\mathbf m$ (Section 9.1), the explicit source update (Section 9.3), and mass recovery (Sections 9.4–9.5) **belong only to the dynamic free-surface mode, which retains inertia**. The quasi-static route (Sections 2.8 and 10.4) does not carry, transport, or remap $\mathbf m$. At each geological step the velocity is recomputed implicitly from (8.4′), so the route is not subject to the explicit source-step restriction of Section 9.3. The quasi-static route uses only Section 9.0 of this chapter (surface geometry and the parametric surface velocity $\mathbf c$) and Section 9.2 (semi-Lagrangian backtracking and the Jacobian conservation correction), and it uses them for the surface density $\rho_s$ of Mode C (Section 2.8). In that use the eight fields of Section 9.2 are replaced by the single scalar field $\rho_s$, and $\mathbf c$ is the velocity of the active region relative to the rigid core, $\chi_A\mathbf v_{{\rm sl},S}/R$.

---

## 9.0 Surface geometry and the material velocity on the surface

The free surface $\Gamma$ is $X(\hat{\boldsymbol\omega},t)=R(\hat{\boldsymbol\omega},t)\,\hat{\boldsymbol\omega}$, with $\hat{\boldsymbol\omega}\in S^2$. Star-shapedness makes $\hat{\boldsymbol\omega}$ a global parameter of $\Gamma$. No atlas is required. If $\nabla_{S^2}$ is the gradient on the unit sphere and $\mathbf e_r=\hat{\boldsymbol\omega}$, then

$$
\mathbf n=\frac{R\,\mathbf e_r-\nabla_{S^2}R}{\sqrt{R^2+|\nabla_{S^2}R|^2}},\qquad
\mathbf n\cdot\mathbf e_r=\frac{R}{\sqrt{R^2+|\nabla_{S^2}R|^2}}\ \ge c_0>0.
$$

In spherical coordinates $(\theta,\varphi)$ the induced metric is

$$
g_{\theta\theta}=R^2+R_\theta^2,\quad
g_{\varphi\varphi}=R^2\sin^2\theta+R_\varphi^2,\quad
g_{\theta\varphi}=R_\theta R_\varphi.
$$

The area element is $dS=J_\Gamma\,d\theta\,d\varphi=j\,d\Omega$, where the area density relative to solid angle is

$$
j(\hat{\boldsymbol\omega},t)=R\sqrt{R^2+|\nabla_{S^2}R|^2}=\frac{R^2}{\mathbf n\cdot\mathbf e_r},\qquad
J_\Gamma=j\sin\theta.
$$

The vanishing of $J_\Gamma$ at $\theta=0,\pi$ is only a coordinate singularity. The discretization represents points by the unit vector $\hat{\boldsymbol\omega}$ and uses $j$ as the conserved density.

**Proposition 9.1 (Parametric velocity of a material point on the surface).** A material point of the surface satisfies the kinematic condition of Section 5.1. The velocity of its parametric trajectory $\hat{\boldsymbol\omega}(t)$ is

$$
\boxed{\ 
\mathbf c:=\dot{\hat{\boldsymbol\omega}}=\frac1R\,\mathbf v_S\big|_{r=R},\qquad
\partial_t R+\mathbf c\cdot\nabla_{S^2}R=v_r\big|_{r=R},\ }
$$

where $\mathbf v_S$ is the physical tangential component of $\mathbf v|_\Gamma$ inside $\hat{\boldsymbol\omega}^{\perp}$. Equivalently, $\partial_t R=v_r-R^{-1}\nabla_{S^2}R\cdot\mathbf v_S$. In terms of the physical surface gradient $\nabla_S=R^{-1}\nabla_{S^2}$, this is $\partial_t R+\mathbf v_S\cdot\nabla_S R=v_r$. Furthermore,

$$
\partial_t R=\frac{\mathbf v\cdot\mathbf n}{\mathbf n\cdot\mathbf e_r}.
$$

*Proof.* A material point is $X(t)=R(\hat{\boldsymbol\omega}(t),t)\,\hat{\boldsymbol\omega}(t)$. Differentiation gives

$$
\mathbf v=(\partial_t R+\nabla_{S^2}R\cdot\dot{\hat{\boldsymbol\omega}})\,\hat{\boldsymbol\omega}+R\dot{\hat{\boldsymbol\omega}}.
$$

The radial component is the kinematic condition. The tangential component gives $\mathbf c=\mathbf v_S/R$. The last identity follows by writing

$$
\mathbf v\cdot\mathbf n=\frac{R v_r-\nabla_{S^2}R\cdot\mathbf v_S}{\sqrt{R^2+|\nabla_{S^2}R|^2}}
$$

and dividing by the expression for $\mathbf n\cdot\mathbf e_r$, as in Section 5.1. ∎

The last identity is the same geometric relation as $d_{\rm pred}=(\mathbf u\cdot\mathbf n+\Delta h_C)/(\mathbf n\cdot\mathbf e_r)$ in Section 2.8. It converts a normal displacement into a radial increment.

---

## 9.1 Conserved variables and the transport equation

**Definition 9.2 (Conserved variables).** Take the radial basis of Section 8.1 and the expansion of Section 8.2, with $\mathbf u=(q_{\ell km},s_{\ell km})$ and $1\le\ell\le L_{\max}$. The mass operator is

$$
(\mathbf M_\Gamma)_{\alpha\beta}=\int_{\mathcal V(R)}\rho\,\mathbf B_\alpha\cdot\mathbf B_\beta\,dV,\qquad
\mathbf m:=\mathbf M_\Gamma(R,\rho)\,\mathbf u.
$$

**Conversion operators (defined as matrices, and thereby made unique).** Let $\{\hat{\boldsymbol\omega}_a\}_{a=1}^{N_S}$ be the surface quadrature nodes, with positive weights $w_a$. The single-field weight matrix is $\mathbf W_1=\operatorname{diag}(w_a)$. The eight-field nodal weight matrix is fixed as $\mathbf W_8=\mathbf W_1\otimes\mathbf I_8$. The spherical-harmonic synthesis matrix $\mathbf S_1\in\mathbb R^{N_S\times N_{\ell m}}$ has entries $(\mathbf S_1)_{ai}=Y_i(\hat{\boldsymbol\omega}_a)$, where $i=(\ell,m)$ and $1\le\ell\le L_{\max}$. The eight-field synthesis matrix is $\mathbf S=\mathbf S_1\otimes\mathbf I_8$, and the coefficient vector is fixed in the order “spherical-harmonic index first, the eight components $(X,k)$ contiguous”. If an implementation uses field-major order, it must switch to the equivalent $\mathbf I_8\otimes\mathbf S_1$ and rewrite every block-diagonal factor and every analysis matrix in the same way. The quadrature analysis matrix is $\mathbf S^*_{\rm quad}:=\mathbf S^{\!\top}\mathbf W_8$. When the quadrature is exact for polynomials of degree $2L_{\max}$, one has $\mathbf S_1^{\top}\mathbf W_1\mathbf S_1=\mathbf I$, hence $\mathbf S^*_{\rm quad}\mathbf S=\mathbf I$. Everywhere below, $\mathbf W$ means the definite matrix $\mathbf W_8$.

**Hypothesis (A-M; PO-4 is not proved).** Treating $\mathbf M_\Gamma\mathbf u$ as a variable transported along surface characteristics is a modelling hypothesis based on operator splitting. It is not a lossless transformation of the three-dimensional weak form.

On the subspace $\ell\ge 1$, $\mathbf M_\Gamma$ is symmetric positive definite. It is defined and inverted only on the velocity-mode space of degree $\ell\ge 1$ (Section 8.2 and Definition 8.6). The double potential of degree $\ell=0$ contributes nothing to the velocity (Proposition 5.6(2)), so $\mathbf M_\Gamma$ has no row or column of degree $\ell=0$, and there is no inversion of a singular row. When $R$ is spherically symmetric and $\rho=\rho(r)$, the operator is block diagonal in $(\ell,m)$ and the poloidal and toroidal parts decouple (Proposition 5.7). In general it is a banded sparse block matrix, whose bandwidth is fixed by the angular harmonic degrees of $R$ and of $\rho$ (Proposition 6.1). For each type $X\in\{P,T\}$ and each radial index $k\in\{0,1,2,3\}$ — eight surface fields in all — the standard spherical-harmonic transform is $m_{X,k}(\hat{\boldsymbol\omega})=\sum_{\ell m}m_{X,\ell km}Y_{\ell m}(\hat{\boldsymbol\omega})$, with $m_{X,\ell km}=\oint_{S^2}m_{X,k}Y_{\ell m}\,d\Omega$. By orthonormality of the real spherical harmonics, this map is a two-sided isometric isomorphism of the band-limited space $L^2(S^2)$ (Parseval’s identity). It is not to be confused with conservation of physical kinetic energy. The surface fields are

$$
m_{X,k}(\hat{\boldsymbol\omega})=\sum_{\ell\ge 1,\,|m|\le\ell}m_{X,\ell km}\,Y_{\ell m}(\hat{\boldsymbol\omega}),
$$

eight fields in all.

**Transport equation.** Along a material point of the surface, the conservation form is

$$
\boxed{\ 
\partial_t(j\,m_{X,k})+\nabla_{S^2}\cdot(j\,\mathbf c\,m_{X,k})=j\,\mathcal S_{X,k}.\ }
$$

The source is obtained by differentiating the block system of Section 7.3 in time, at $\mathbf m=\mathbf M_\Gamma\mathbf u$:

$$
\boldsymbol{\mathcal S}
=(\partial_t\mathbf M_\Gamma-\mathbf G)\,\mathbf u
-\mathbf C\mathbf u
-\mathbf K(\mathbf u,p)
-\mathbf N(\mathbf u,\mathbf u)
+\mathbf F.
$$

The derivative $\partial_t\mathbf M_\Gamma$ comes from the time dependence of $R$ and of $\rho$. It contains the Reynolds boundary term

$$
\oint_\Gamma\rho\,(\mathbf B_\alpha\cdot\mathbf B_\beta)\,(\partial_t R)\,(\mathbf n\cdot\mathbf e_r)\,dS
=\int_{S^2}\rho\,(\mathbf B_\alpha\cdot\mathbf B_\beta)\,R^2\partial_t R\,d\Omega,
$$

the basis-deformation term $\int_{\mathcal V}\rho\,(\partial_t\mathbf B_\alpha\cdot\mathbf B_\beta+\mathbf B_\alpha\cdot\partial_t\mathbf B_\beta)\,dV$, and the term in $\partial_t\rho$. Define $(\mathbf G)_{\beta\alpha}=\int_{\mathcal V}\rho\,\mathbf B_\beta\cdot\partial_t\mathbf B_\alpha\,dV$. The basis-connection term in the coefficient equation is therefore $\mathbf G\mathbf u$, while the geometric source in the equation for the conserved variable must be $(\partial_t\mathbf M_\Gamma-\mathbf G)\mathbf u$. One must not count both $\partial_t\mathbf M_\Gamma\mathbf u$ and $\mathbf G\mathbf u$. In the fixed-space Eulerian form used here, $\mathbf N$ uses $\mathbf v$. One must not substitute $\mathbf v-\mathbf w_{\rm mesh}$ into the same term. A switch to the ALE form must switch the time derivative and the basis derivatives together with it. In a static SCF, $\mathbf G\equiv\mathbf 0$. The definitions of $\mathbf C$, $\mathbf K$, $\mathbf N$, and $\mathbf F$ are those of Sections 5.4 and 7.3.

**Model hypothesis A-M (operator splitting).** The equation $\mathbf M_\Gamma\dot{\mathbf u}+\cdots=\mathbf F$ of Section 7.3 takes the Eulerian derivative at fixed $\hat{\boldsymbol\omega}$. This chapter uses its split form: first translate $\mathbf m$ along surface material points (pure transport, $\mathcal S=0$), then accumulate $\boldsymbol{\mathcal S}$ at the arrival point. **Convection is not counted twice.** Translation of $\mathbf m$ along surface material points at velocity $\mathbf c$ already carries the tangential convection on the surface. (The quasi-static route does not transport $\mathbf m$. This paragraph concerns only the dynamic mode.) In the dynamic free-surface mode, the nonlinear convective term $\mathbf N$ inside $\boldsymbol{\mathcal S}$ must be the residual relative convection left after the tangential surface convection has been removed. Write it $\mathbf N_{\rm rel}$, so that $\mathbf N=\mathbf N_{\rm rel}+\mathcal A(\mathbf c)\mathbf m$, where $\mathcal A(\mathbf c)\mathbf m$ is the part already carried by the transport step. Otherwise the same convective effect is counted twice. After that subtraction, the splitting is consistent of order one (Lie) or of order two (Strang) in the time step. **The statement holds only after the repeated convection has been removed.** Without the subtraction there is a model error of order $O(1)$. The splitting error $E_{\rm split}$ is entered in the error budget of Section 9.8. It is a modelling choice, not a consequence of the Galerkin residual.

---

## 9.2 Semi-Lagrangian backtracking

For a grid point $\hat{\boldsymbol\omega}_a$, integrate the backward characteristic $\dot{\hat{\boldsymbol\omega}}=\mathbf c(\hat{\boldsymbol\omega},t)$ on the sphere, with $\hat{\boldsymbol\omega}(t^{n+1})=\hat{\boldsymbol\omega}_a$, preserving $|\hat{\boldsymbol\omega}|=1$:

$$
\hat{\boldsymbol\omega}_m
=\mathcal N\!\Big[\hat{\boldsymbol\omega}_a-\tfrac{\Delta t}{2}\mathbf c(\hat{\boldsymbol\omega}_a,t^n)\Big],\qquad
\hat{\boldsymbol\omega}_d
=\mathcal N\!\Big[\hat{\boldsymbol\omega}_a-\Delta t\,\mathbf c(\hat{\boldsymbol\omega}_m,t^{n+1/2})\Big],
$$

where $\mathcal N(\mathbf x)=\mathbf x/|\mathbf x|$. If only the velocity at $t^n$ is available, the scheme is a first-order approximation and must not be claimed to be of second order. Ray intersection in the RayBackend is used only to locate the triangle that contains $\hat{\boldsymbol\omega}_d$. A surface projection on that element is then required before $\hat{\boldsymbol\omega}_d$ is a valid departure point.

Inside the triangle $\{\hat{\boldsymbol\omega}_1,\hat{\boldsymbol\omega}_2,\hat{\boldsymbol\omega}_3\}$, take barycentric coordinates $\lambda_i$ and interpolate all eight fields at once:

$$
m^d_{X,k}=\sum_{i=1}^3\lambda_i\,m_{X,k}(\hat{\boldsymbol\omega}_i).
$$

The conservation correction, which secures discrete conservation of $\int j\,m\,d\Omega$, uses the Jacobian of the characteristic map,

$$
\mathcal J_a=\Big|\det\frac{\partial\hat{\boldsymbol\omega}_d}{\partial\hat{\boldsymbol\omega}}\Big|_{\hat{\boldsymbol\omega}_a},
$$

with the determinant taken on the tangent space of $S^2$. A determinant of the embedded vectors has no meaning. The characteristic map is required to be single-valued and free of folds at the mesh scale in use, in the sense that the Lipschitz constant satisfies $\Delta t\,\|\nabla\mathbf c\|_\infty<1$. Otherwise the departure point is not unique and $\Delta t$ must be reduced. The source-free update is

$$
j^{n+1}_a\,m^{n+1,*}_a=\mathcal J_a\,j^{n}_d\,m^d.
$$

---

## 9.3 Source update and splitting

The explicit source update (Lie splitting) is

$$
j^{n+1}_a\,m_a^{n+1}
=\mathcal J_a\,j^{n}_d\,m^d_a
+\Delta t\,j^{n+1}_a\,\boldsymbol{\mathcal S}_a^{n}.
$$

Every term of $\boldsymbol{\mathcal S}_a^n$ is assembled as in Section 9.1. The velocity $\mathbf u^n$ is the one recovered globally at the previous step (Section 9.4). If the source is stiff, in the sense that $\Delta t\,\|\partial\boldsymbol{\mathcal S}/\partial\mathbf m\|$ is not small, one must reduce $\Delta t$ or subcycle. This paper does not introduce a locally implicit source update. Such an update would approximate $\mathbf M_\Gamma^{-1}$ locally, which is an undeclared Track B.

---

## 9.4 Global mass recovery (the production route, Track A)

After transport and the source update, one has a set of nodal fields $\{m_{X,k}(\hat{\boldsymbol\omega}_a)\}$. The velocity coefficients must be recovered by the following global procedure.

**Step 1 (analysis).** Perform a spherical-harmonic analysis with a quadrature that is exact, on the sphere, for polynomials of degree $2L_{\max}$ (Gauss–Legendre in latitude times a uniform grid in $\varphi$, or a Lebedev rule):

$$
m_{X,\ell km}
=\sum_a w_a\,m_{X,k}(\hat{\boldsymbol\omega}_a)\,Y_{\ell m}(\hat{\boldsymbol\omega}_a),\qquad
1\le\ell\le L_{\max}.
$$

**Step 2 (gauge of the potential at $\ell=0$).** The potentials $\Phi$ and $\Psi$ of degree $\ell=0$ do not contribute to the velocity (Proposition 5.6). The operator $\mathbf M_\Gamma$ itself contains only rows and columns of degree $\ell\ge 1$ (Definition 8.6) and does not contain $\ell=0$. The gauge is

$$
q_{0km}=s_{0km}=0,\qquad m_{X,0km}:=0.
$$

Discarding degree $\ell=0$ is, on the dual space, the orthogonal projection $\mathcal P_{\ge 1}$. It is the mean-removal (direct-current removal) orthogonal projection of the standard $L^2(S^2)$ on the sphere, not a low-pass filter. Cross-degree coupling induced by a non-spherical $R$ can leak a small component of degree $\ell=0$ into the transported surface field. The quantity $\|(I-\mathcal P_{\ge 1})\mathbf m\|$ is the modal truncation error induced by geometric asphericity. The content $m^{\rm spur}_{X,0k}$ of degree $\ell=0$ produced by the analysis is discarded. Its norm $\|\mathbf m^{\rm spur}\|/\max\{\|\mathbf m\|,\varepsilon_m\}$ is a diagnostic of transport self-consistency. If $\|\mathbf m\|=0$ and the numerator is zero, the diagnostic is taken to be $0$; otherwise the step is judged to have failed. The diagnostic must lie below the configured tolerance, or the transport step is judged to have failed.

**Step 3 (global solve).** Solve the saddle-point system with the rigid-body constraint (Proposition 5.6, Proposition 8.11, and Appendix A.3),

$$
\boxed{
\begin{pmatrix}
\mathbf M_\Gamma(R,\rho)&\mathbf B_{\rm rb}^{\!\top}\\
\mathbf B_{\rm rb}&\mathbf 0
\end{pmatrix}
\begin{pmatrix}\mathbf u\\ \boldsymbol\lambda\end{pmatrix}
=\begin{pmatrix}\mathbf m\\ \mathbf 0\end{pmatrix}.}
$$

The rows of $\mathbf B_{\rm rb}\in\mathbb R^{6\times N_u}$ are $\int_{\mathcal V}\rho\,\mathbf B_\alpha\,dV$ and $\int_{\mathcal V}\rho\,\mathbf x\times\mathbf B_\alpha\,dV$. The operator $\mathbf M_\Gamma$ is symmetric positive definite. The saddle-point system is well posed if and only if the rigid-body block has full algebraic rank, $\operatorname{rank}(\mathbf B_{\rm rb})=6$. When $R$ is constant and $\rho=\rho(r)$, Proposition 8.11 guarantees that rank. When $R$ is not spherical, the rank is determined by the geometry and by the weight. It must be checked numerically at every outer step, together with the smallest singular value of $\mathbf B_{\rm rb}$. One does not assume in advance that rigid motions lie in the discrete space. The multiplier

$$
\boldsymbol\lambda
=(\mathbf B_{\rm rb}\mathbf M_\Gamma^{-1}\mathbf B_{\rm rb}^{\!\top})^{-1}
\mathbf B_{\rm rb}\mathbf M_\Gamma^{-1}\mathbf m
$$

is the **projection multiplier** that orthogonally projects $\mathbf m$, in the $\mathbf M_\Gamma^{-1}$ inner product, onto $\ker(\mathbf B_{\rm rb}\mathbf M_\Gamma^{-1})$. It is not a residual that “must vanish”. A nonzero $\boldsymbol\lambda$ means only that $\mathbf m$ contains a rigid momentum or angular momentum that the projection can remove. By itself it is not a failure criterion. The monitored quantity is the relative size of the projection correction (the rigid-body correction residual)

$$
E_{\rm rb}
:=\frac{\|\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda\|_{\mathbf M_\Gamma^{-1}}}
{\max\{\|\mathbf m\|_{\mathbf M_\Gamma^{-1}},\varepsilon_{\rm rb}\}}
\in[0,1],
$$

where $\|\mathbf a\|_{\mathbf M_\Gamma^{-1}}^2=\mathbf a^{\!\top}\mathbf M_\Gamma^{-1}\mathbf a$. The identity

$$
(\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda)^{\!\top}\mathbf M_\Gamma^{-1}(\mathbf m-\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda)
=\boldsymbol\lambda^{\!\top}\mathbf B_{\rm rb}\mathbf u=0
$$

gives the Pythagorean relation $\|\mathbf m\|^2=\|\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda\|^2+\|\mathbf m_{\rm rb}\|^2$, all norms being $\mathbf M_\Gamma^{-1}$ norms, so $E_{\rm rb}\le 1$. The same zero-denominator regularization must be used in any other norm, and the norm must be declared. The step is judged abnormal only when $E_{\rm rb}$ exceeds the configured tolerance, that is, when the rigid component dominates and the transport or the source has produced a non-negligible parasitic rigid momentum. After recovery, the synchronous momentum that corresponds exactly to the physical velocity is $\mathbf m_{\rm rb}:=\mathbf M_\Gamma\mathbf u=\mathbf m-\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda$. In the dynamic mode, every recovery must be followed by the state synchronization $\mathbf m^{(n+1/2)}\leftarrow\mathbf m_{\rm rb}^{(n+1/2)}$. Subsequent transport and multi-step cycles carry the momentum $\mathbf m_{\rm rb}$ from which the parasitic rigid momentum has been removed, so that numerical rigid drift does not accumulate without bound.

**Meaning of the constrained recovery.** The system defines the constrained orthogonal projection

$$
\mathbf u=\mathcal R_{\rm rb}(\mathbf m)=\mathbf M_\Gamma^{-1}(\mathbf m-\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda).
$$

When $\boldsymbol\lambda\ne\mathbf 0$, the recovered $\mathbf u$ does not satisfy $\mathbf m=\mathbf M_\Gamma\mathbf u$. It applies to the conserved variable an $\mathbf M_\Gamma^{-1}$-orthogonal correction that removes parasitic rigid momentum. The momentum correction $\Delta\mathbf m_{\rm rb}=\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda$ is entered in the error budget of Section 9.8 through $E_{\rm rb}$.

The system is solved by preconditioned conjugate gradients on the Schur complement. The preconditioner is the block-diagonal inverse $\mathbf M_{B1}^{-1}$ of Track B1 (Section 9.5). The number of iterations after preconditioning is controlled by $\|\mathbf M_{B1}^{-1}\mathbf E\|$. The solution is exact up to the CG tolerance. It is not an approximation.

**Assembly of $\mathbf M_\Gamma$.** In the blocks $(\ell,m)$ and $(\ell',m')$,

$$
(\mathbf M_\Gamma)_{\alpha\beta}
=\int_{S^2}\!\int_0^{R(\hat{\boldsymbol\omega})}\rho\,\mathbf B_\alpha\cdot\mathbf B_\beta\,r^2\,dr\,d\Omega.
$$

The radial integral uses Gauss–Jacobi quadrature for the Jacobi weight corresponding to $\zeta^{2\ell+2}$. The angular integral uses the same quadrature as step 1. The degrees of the spherical-harmonic expansions of $R$ and of $\rho$ determine the bandwidth. The selection rule of Proposition 6.1 locates the nonzero blocks.

(`tests/verify_closure.py` is an independent test of abstract linear-algebraic identities only. It checks the algebraic identities of the saddle-point system and of the constrained decomposition. It is not a verification of the Galerkin assembly, of the physical model, or of a numerical result.)

---

## 9.5 Track B: the block-diagonal surrogate and its truncation error

**Definition 9.3 (Track B is the B1 block-diagonal surrogate).** Take

$$
\mathbf M_B:=\mathbf M_{B1}=\operatorname{blockdiag}_{(\ell,m)}\mathbf M_{\ell m,\ell m},\qquad
\mathbf u_B:=\mathbf M_B^{-1}\mathbf m,
$$

where, for each $(\ell,m)$, the block $\mathbf M_{\ell m,\ell m}\in\mathbb R^{8\times 8}$ is the diagonal principal sub-block of $\mathbf M_\Gamma$ (four poloidal radial modes and four toroidal radial modes; on a general geometry it includes the intra-block cross entries of $\mathbf M_{PT}$). The principal sub-block inherits the symmetric positive definiteness of $\mathbf M_\Gamma$, so $\mathbf M_B$ is symmetric positive definite and invertible block by block. The residual operator $\mathbf E:=\mathbf M_\Gamma-\mathbf M_B$ contains only the Gaunt coupling blocks that cross between distinct pairs $(\ell,m)$. This paper defines no nodal (pointwise) approximation; see PO-8. An implementation must not retain such a code path.

**Properties.** (i) If $R$ is constant and $\rho=\rho(r)$, then $\mathbf E=\mathbf 0$, and the unconstrained solution of Track B agrees with that of Track A to the last bit (Proposition 8.8). (ii) On a general geometry, $\mathbf M_B^{-1}\mathbf M_\Gamma$ is similar to the symmetric positive-definite matrix

$$
\mathbf M_B^{-1/2}\mathbf M_\Gamma\mathbf M_B^{-1/2}
=\mathbf I+\mathbf M_B^{-1/2}\mathbf E\mathbf M_B^{-1/2},
$$

so its spectrum consists of **positive real numbers**. There is no divergence by a change of sign. The size of the departure from $1$ is controlled by the computable quantity $\|\mathbf M_B^{-1/2}\mathbf E\mathbf M_B^{-1/2}\|$. The dependence of that quantity on the degree of asphericity is not proved (PO-8). The implementation must report an estimate of it, for example by power iteration, at every outer step. (iii) The inverse $\mathbf M_B^{-1}$ is at the same time the preconditioner of the PCG applied to $\mathbf M_\Gamma$ in Section 9.4. The preconditioned condition number is $\kappa(\mathbf I+\mathbf M_B^{-1/2}\mathbf E\mathbf M_B^{-1/2})$.

**Theorem 9.4 (Truncation error)** [the identity and the a-posteriori bound are proved; an empirical upper bound for $\varepsilon_B$, and the domain of applicability, belong to PO-8, on which this theorem does not depend]. Define the modal truncation operator

$$
\mathbf E_B^{\rm solve}:=\mathbf M_B^{-1}-\mathbf M_\Gamma^{-1},
$$

and write $\mathbf u_B:=\mathbf M_B^{-1}\mathbf m$ for the unconstrained Track B recovery. The constrained Track A solution is $\mathbf u=\mathcal R_{\rm rb}(\mathbf m)=\mathbf M_\Gamma^{-1}(\mathbf m-\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda)$. The **constrained orthogonal decomposition** holds exactly:

$$
\boxed{
\mathbf u_B-\mathbf u
=\mathbf E_B^{\rm solve}\,\mathbf m+\mathbf M_\Gamma^{-1}\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda
=\big(\mathbf M_B^{-1}-\mathbf M_\Gamma^{-1}\big)\mathbf m
+\mathbf M_\Gamma^{-1}\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda.}
$$

The first term is the truncation error of the mass operator. The second is the difference induced by the rigid-body projection, and it is measured by $E_{\rm rb}$. Let $\mathbf u_0:=\mathbf M_\Gamma^{-1}\mathbf m$ be the unconstrained Track A solution. For the **unconstrained** part, the following identity and a-posteriori bound hold:

$$
\boxed{
\mathbf u_B-\mathbf u_0
=\mathbf M_B^{-1}\mathbf E\,\mathbf u_0
=\mathbf M_\Gamma^{-1}\mathbf E\,\mathbf u_B,\qquad
\|\mathbf u_B-\mathbf u_0\|
\le\frac{\|\mathbf m-\mathbf M_\Gamma\mathbf u_B\|}{\lambda_{\min}(\mathbf M_\Gamma|_{\ell\ge 1})}.}
$$

*Proof.* The two factorizations

$$
\mathbf M_B^{-1}-\mathbf M_\Gamma^{-1}
=\mathbf M_B^{-1}(\mathbf M_\Gamma-\mathbf M_B)\mathbf M_\Gamma^{-1}
=\mathbf M_\Gamma^{-1}(\mathbf M_\Gamma-\mathbf M_B)\mathbf M_B^{-1}
$$

applied to $\mathbf m$ give the identity. The second form of the difference gives $\mathbf M_\Gamma(\mathbf u_B-\mathbf u_0)=\mathbf M_\Gamma\mathbf u_B-\mathbf m$. The Euclidean norm and the smallest eigenvalue of $\mathbf M_\Gamma$ on $\ell\ge 1$ give the bound. ∎

(In the unconstrained case, $\mathbf u_B-\mathbf u_0=\mathbf E_B^{\rm solve}\mathbf m$.) The computable a-posteriori quantity is $\mathbf r_B=\mathbf m-\mathbf M_\Gamma\mathbf u_B$, one sparse matrix–vector product. Corollary: $\mathbf u_B=\mathbf u_0$ if and only if $\mathbf E\mathbf u_0=\mathbf 0$. Whenever Track B is used, one must report $\varepsilon_B:=\|\mathbf r_B\|/\max\{\|\mathbf m\|,\varepsilon_m\}$ (the zero-momentum case is handled as in Appendix A.4) and its distribution in $\ell$. The result must not be called “exact” or “equivalent”, except on a spherically symmetric geometry, as stated above.

**Rules of use (the Track B red line).**

1. The domain of applicability of Track B is decided by the measured $\varepsilon_B$ and by $\|\mathbf M_B^{-1/2}\mathbf E\mathbf M_B^{-1/2}\|$ against configured thresholds. If a threshold is exceeded, that outer step must fall back to Track A.
2. Any convergence conclusion drawn under Track B must be recovered again under Track A and accepted there (Section 10.4.2).
3. The production reference is Track A.

---

## 9.6 The role of the RayBackend

The RayBackend locates surface elements and computes intersections for the backward characteristics. It does not compute transport fluxes, source matrices, or the mass operator. Once a ray has returned one face, all eight fields are read from that face. The number of ray intersections is unchanged. The cost of interpolation and of the local source grows linearly with the number of fields.

---

## 9.7 Computational complexity

Let $N_S$ be the number of surface nodes, $B$ the number of triangles in the BVH, $N_{\rm ray}$ the number of rays per node, $N_{\rm hit}$ the average number of hit faces, and $N_{\rm src}$ the number of local Newton iterations.

**Local part** (intersection, interpolation, source), eight fields at each node:

$$
C_{\rm local}
=O\!\big[N_S N_{\rm ray}(\log B+N_{\rm hit})+(c_{\rm int}+c_{\rm src}N_{\rm src})N_S\big].
$$

**Comparison with a single mode** (local part only). If the local cost of one step is recorded as $4+16N_{\rm src}$ for the four-mode source update at $K=4$, and as $1$ for a single mode, then

$$
\frac{C_{K=4}}{C_{K=1}}
\approx
1+\frac{3+16N_{\rm src}}{N_{\rm ray}(\log B+N_{\rm hit})}.
$$

When ray intersection dominates, $K=4$ costs about $1.1$ to $1.5$ times as much. When the source dominates, the ratio approaches four.

**Global part** (peculiar to Track A): spherical-harmonic analysis and synthesis cost $O(N_S L_{\max}^2)$ (a fast transform reduces this to $O(L_{\max}^2\log L_{\max})$ per field). A sparse matrix–vector product with $\mathbf M_\Gamma$ costs $O(N_u\cdot b)$, where $b$ is the block bandwidth. With $N_{\rm it}$ iterations of PCG, the total is $O(N_{\rm it} N_u b)$. Track B (the B1 block-diagonal surrogate) removes this term, at the price of the error in Theorem 9.4.

---

## 9.8 The error budget

The error components have different dimensions. **It is forbidden** to add absolute residuals directly. The total error budget is a sum of dimensionless relative quantities,

$$
\operatorname{rel}_{\varepsilon}(a;b):=\|a\|/\max\{\|b\|,\varepsilon_b\},
$$

where each $\varepsilon_b$ has the dimension of the corresponding denominator, and the numerator and the denominator of one term use the same norm. The budget remains defined at zero load and at zero velocity.

**Quasi-static route (each geological step):**

$$
\begin{aligned}
E_{\rm total}^{\rm qs}
&:=\varepsilon_{\rm mom}^{\rm weak}
+\varepsilon_\lambda
+\varepsilon_{\rm pic}^{\rm achieved}
+\operatorname{rel}_{\varepsilon}(\mathbf r_{\rm rot};\rho_K\mathbf g_{\rm eff})
+\operatorname{rel}_{\varepsilon}(E_{\rm frame};R)\\
&\qquad+\operatorname{rel}_{\varepsilon}(\mathbf u_B-\mathbf u;\mathbf u)
+\frac{\delta_{na}}{\max\{D_{\rm ref},\varepsilon_{na}\}}.
\end{aligned}
$$

- $\varepsilon_{\rm mom}^{\rm weak}$ and $\varepsilon_\lambda$ are the Galerkin momentum residual and the rigid-body incompatibility of criterion (b) in Section 2.15, for the unique stress field.
- $\varepsilon_{\rm pic}^{\rm achieved}$ is the relative increment actually attained at the last Picard round (criterion (a) of Section 2.15).
- $\mathbf r_{\rm rot}$ is the strong residual of Section 8.7, a resolution indicator.
- $E_{\rm frame}$ is the error of radius resampling, and of the interpolation of $\rho_s$ and $\bar\varepsilon_p$, in the frame realignment of step 8 in Section 10.4.4. It is defined as $\|R'-R'_{\rm ref}\|_\infty$, the difference between the realigned radius and a reference resampled by fine-grid rays. If the reference is absent, the quantity is registered as PO-13.
- $\mathbf u_B-\mathbf u$ is nonzero only when (8.4′) is solved approximately by Track B (Appendix A.4). On Track A it is defined to be zero.
- $\delta_{na}$ is the traction inconsistency of Section 10.2.3, and $D_{\rm ref}>0$ is a configured reference scale.

The following are reported separately and are not added to the expression above: the geometric projection departure $E_{\rm proj}:=\|d_*-d_{\rm pred}\|_{w,\mathcal A}/\max\{\|d_{\rm pred}\|_{w,\mathcal A},\varepsilon_d\}$ (the part of the SOCP correction that has no corresponding flow; it is a model error of the sequential convex projection), the mass slack $|s_{M*}|/M$, the fraction in tension, the movable area fraction, $\mathrm{Ek}^{-1}$, and $\varepsilon_{\rm Euler}$ (Section 2.8, B1).

**Dynamic free-surface mode (each physical step):**

$$
\begin{aligned}
E_{\rm total}^{\rm dyn}
&:=\operatorname{rel}_{\varepsilon}(\mathbf r_{\rm mom};\mathbf f_{\rm ext})
+\operatorname{rel}_{\varepsilon}(\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda;\mathbf m)
+\operatorname{rel}_{\varepsilon}(E_{\ell=0};\mathbf m)\\
&\qquad+\operatorname{rel}_{\varepsilon}(\mathbf u_B-\mathbf u;\mathbf u)
+\operatorname{rel}_{\varepsilon}(E_{\rm split};\mathbf m)
+\operatorname{rel}_{\varepsilon}(E_{\rm remap};\mathbf m)
+\frac{\delta_{na}}{\max\{D_{\rm ref},\varepsilon_{na}\}}.
\end{aligned}
$$

In the $\mathbf M_\Gamma^{-1}$ norm, the rigid-body term equals $E_{\rm rb}$. On Track A, the term $\mathbf u_B-\mathbf u$ is defined to be zero.

- $\mathbf r_{\rm mom}$ is the momentum (equilibrium) residual of the global system in Section 8.4, and $\mathbf f_{\rm ext}$ is its load.
- $E_{\ell=0}$ is the mean-truncation error induced by geometric asphericity. The recovery $\mathcal R_{\rm rb}$ is defined only on the space $\ell\ge 1$, so a quantity that contains $\ell=0$ is **not** substituted into it. The error $E_{\ell=0}$ is defined by the Euclidean norm on the dual space of modal coefficients:

$$
E_{\ell=0}:=\|(\mathbf I-\mathcal P_{\ge 1})\mathbf m\|_2
=\sqrt{\sum_{X,k}|m_{X,0k0}|^2}.
$$

- $E_{\rm split}$ is the operator-splitting error of hypothesis A-M, defined as the difference $\|\mathbf m^{\rm split}-\mathbf m^{\rm ref}\|_2$ between one transport-plus-source step and a reference computed with refined substeps (Lie of order $O(\Delta t)$, Strang of order $O(\Delta t^2)$; PO-4).
- $E_{\rm remap}$ is the error, in the dynamic mode, of the modal remapping $\mathbf m^{\rm new}=\int_{\mathcal V^{\rm new}}\rho\,\mathbf B_\alpha^{\rm new}\cdot\mathbf v^{\rm old}\,dV$ after a change of geometry. On $\mathcal V^{\rm new}\setminus\mathcal V^{\rm old}$, the old velocity $\mathbf v^{\rm old}$ is evaluated by analytic continuation of its modal polynomial to $\zeta>1$. The error is $\|\mathbf m^{\rm remap}-\mathbf m^{\rm cons,new}\|_2$. The reference is a refined remapping. If it is absent, the quantity is registered as PO-4.
- Discrete truncation errors are not added to the expression above. They are $E_{\rm radial}$, the radial truncation at $K=4$; $E_{\rm angular}$, the truncation at $L_{\max}$; $E_{\rm char}=O(\Delta t^p)$; $E_{\rm interp}=O(h_\Gamma^2)$; the aliasing $E_{\rm alias}$; and $E_{\rm RTFP}$. Each is reported separately by a refinement experiment in mesh or in order, through the relative difference $\|\mathbf u_{\rm fine}-\mathbf u\|/\|\mathbf u_{\rm fine}\|$, for example $E_{4\to 8}$.

At fixed $K=4$ and $L_{\max}$, one may say only that the discrete solution converges to that reduced model. It does not converge to the full three-dimensional solution. Convergence must be tested by comparison with $K=8$ and with a higher $L_{\max}$, through $E_{4\to 8}=\|\mathbf u_8-\mathbf u_4\|/\|\mathbf u_8\|$.

---

## 9.9 Decoupling of the meanings of time

| | Dynamic free surface | Quasi-static route (Modes B and C) |
|---|---|---|
| Time variable | physical time $t$; the step is limited by inertia and by the explicit source of Section 9.3 | geological time $t_n$, with $t_{n+1}=t_n+\lambda\Delta t_n$; $\Delta t_n$ is limited only by the geometric CFL of the movable region, by the erosional flux, and by the resolution of the spin table (Section 10.4.1) |
| Velocity | recovered from the transported modal momentum $\mathbf m$ (Section 9.4) | the instantaneous Stokes creep velocity, computed implicitly at each step by (8.4′) and Section 8.7; not carried across steps |
| Inertial terms | retained | omitted entirely ($\mathbf M=\mathbf N=\mathbf C=\mathbf 0$) |
| Evolution of $R$ | $\partial_t R+\mathbf c\cdot\nabla_{S^2}R=v_r$ (Proposition 9.1) | $\partial_t R=\chi_Y\,\mathbf v_{\rm sl}\cdot\mathbf n/(\mathbf n\cdot\mathbf e_r)$ plus the Mode C flux, discretized over $\Delta t_n$ as $d_{\rm pred}$, then corrected by the SOCP |
| Update of $R$ | integrated with $\partial_t R$ | $R^{(n+1)}=R^{(n)}+\lambda\mathbf d_*$ (on the locked set, $d_*=0$) |
| Transport | eight modal fields $\mathbf m$ | the surface density $\rho_s$ only (Section 9.2) |
| Spin | exogenously prescribed $\omega(t)$ | exogenously prescribed $\omega(t)$, frozen at $\omega_n$ on each step |

The two modes must not be mixed inside one loop. The stage-equilibrium condition of the quasi-static route is that the movable set is empty and that the mechanical acceptance test holds (criterion (c) of Section 2.15).

---

# 10. The $K=4$ closure and the geometric SCF coupling

The radial Galerkin truncation is locked at order $K=4$. The meaning is that each spherical-harmonic index $(\ell,m)$ and each velocity potential — the poloidal potential $\Phi$ and the toroidal potential $\Psi$ — keeps four radial modes. Each pair $(\ell,m)$ therefore has eight velocity degrees of freedom, and the velocity-coefficient vector is $\mathbf u=(\mathbf q,\mathbf s)$ (Section 8.2). The radial basis is uniformly the shifted Jacobi family that depends on $\ell$,

$$
\psi_{\ell k}(\zeta)=\zeta^{\ell}P_k^{(0,\,2\ell+2)}(2\zeta-1),\qquad
\zeta=\frac{r}{R(\hat{\boldsymbol\omega},t)},\quad k=0,1,2,3.
$$

The potentials expand as

$$
\Phi=R^2\sum_{\ell\ge 1,\,m,k}q_{\ell km}\,\psi_{\ell k}(\zeta)\,Y_{\ell m},\qquad
\Psi=R\sum_{\ell\ge 1,\,m,k}s_{\ell km}\,\psi_{\ell k}(\zeta)\,Y_{\ell m}.
$$

This chapter and Chapter 8 use the same coefficients $(q_{\ell km},s_{\ell km})$. There is no further rescaling. Derivative terms produced when $R$ changes in time or in direction are accounted for explicitly by the source $\boldsymbol{\mathcal S}$ of Chapter 9 and by the trace formulae of Section 10.2.

The contents are as follows. Section 10.1 gives the analytic structure and the condition numbers of the mass blocks on a fixed reference geometry. Section 10.2 gives the velocity trace on a general star-shaped surface, the surface-invariant shear rate, and the traction condition. Section 10.3 generates the friction angle from the shear rate. Section 10.4 gives the geometric increment subproblem and the SCF loop.

---

## 10.1 Diagonal blocks and basis orthogonality on a fixed reference geometry

This section takes the fixed reference sphere $R\equiv R_0$ and the constant density $\rho\equiv\rho_0$, and on that reference geometry it makes the mass blocks of Section 8.3 explicit. They are used for unit tests and for the construction of preconditioners. A production problem with variable geometry or variable density must be reassembled from the general definition of Section 8.3 (Propositions 8.7 and 8.9), and symmetric positive definiteness must be rechecked at every outer step.

**Proposition 10.1 (Diagonal blocks and basis orthogonality on the reference geometry) [conditional closure of the fixed-reference block in PO-6].** Let $R\equiv R_0$, $\rho\equiv\rho_0$, $\ell\ge 1$, and $\Lambda=\ell(\ell+1)$.

1. **Orthogonality.**

$$
\int_0^1\psi_{\ell j}\,\psi_{\ell k}\,\zeta^2\,d\zeta=\frac{\delta_{jk}}{2k+2\ell+3}.
$$

2. **The toroidal block is strictly diagonal.** The toroidal kinetic-energy mass block of a single pair $(\ell,m)$ is

$$
\mathbf M^T_\ell
=\rho_0 R_0^5\,\Lambda\,
\operatorname{diag}\!\Big(\frac1{2\ell+3},\ \frac1{2\ell+5},\ \frac1{2\ell+7},\ \frac1{2\ell+9}\Big).
$$

3. **The poloidal block.**

$$
\mathbf M^P_\ell
=\rho_0 R_0^5\,\Lambda\big(\Lambda\,\tilde{\mathbf N}_\ell+\tilde{\mathbf P}_\ell\big),\qquad
(\tilde{\mathbf N}_\ell)_{ij}=\int_0^1\psi_{\ell i}\psi_{\ell j}\,d\zeta,\quad
(\tilde{\mathbf P}_\ell)_{ij}=\int_0^1(\zeta\psi_{\ell i})'(\zeta\psi_{\ell j})'\,d\zeta.
$$

The matrices $\tilde{\mathbf N}_\ell$ and $\tilde{\mathbf P}_\ell$ are symmetric positive definite, so $\mathbf M^P_\ell$ is symmetric positive definite.
4. **Decoupling between blocks.** Mass blocks between distinct pairs $(\ell,m)$, and between poloidal and toroidal fields, vanish (Proposition 5.7). Hence $\mathbf M_\Gamma=\operatorname{blockdiag}(\mathbf M^P_{\ell},\mathbf M^T_{\ell})$, and the smallest and largest eigenvalues of $\mathbf M_\Gamma$ on $\ell\ge 1$ are determined by the individual blocks.
5. The block of degree $\ell=0$ vanishes identically and takes part in no inversion.

*Proof.* Item 1 is Proposition 8.3(1). For item 2, the toroidal entry is $\Lambda\rho_0\int f_i f_j r^2\,dr$ (Proposition 5.7). With $f_i=R_0\psi_{\ell i}(r/R_0)$ and $r=R_0\zeta$, this is $\rho_0 R_0^5\Lambda\int\psi_{\ell i}\psi_{\ell j}\zeta^2\,d\zeta$, which is diagonal by item 1. The four denominators are $2k+2\ell+3$ for $k=0,1,2,3$. For item 3, the poloidal block of Proposition 5.7 is $\Lambda^2\int\rho\,g_i g_j\,dr+\Lambda\int\rho\,(rg_i)'(rg_j)'\,dr$. With $g_i=R_0^2\psi_{\ell i}(r/R_0)$, one has $rg_i=R_0^3\zeta\psi_{\ell i}$ and $(rg_i)'=R_0^2(\zeta\psi_{\ell i})'$. The change of variables produces the displayed formula. The matrix $\tilde{\mathbf N}_\ell$ is the Gram matrix of the linearly independent functions $\psi_{\ell i}$. The matrix $\tilde{\mathbf P}_\ell$ is the Gram matrix of $(\zeta\psi_{\ell i})'$. The condition $(\zeta\psi)'=0$ means that $\zeta\psi$ is constant, hence $\psi=0$ as a polynomial, so the family is linearly independent. Item 4 is Proposition 5.7. ∎

**Analytic matrices at $\ell=1$.** From $\psi_{10}=\zeta$, $\psi_{11}=6\zeta^2-5\zeta$, $\psi_{12}=28\zeta^3-42\zeta^2+15\zeta$, and $\psi_{13}=120\zeta^4-252\zeta^3+168\zeta^2-35\zeta$,

$$
\tilde{\mathbf N}_1
=\begin{pmatrix}
\frac13&-\frac16&\frac1{10}&-\frac1{15}\\[0.4ex]
-\frac16&\frac8{15}&-\frac25&\frac{61}{210}\\[0.4ex]
\frac1{10}&-\frac25&\frac45&-\frac7{10}\\[0.4ex]
-\frac1{15}&\frac{61}{210}&-\frac7{10}&\frac{17}{15}
\end{pmatrix},\qquad
\tilde{\mathbf P}_1
=\begin{pmatrix}
\frac43&\frac73&\frac95&\frac{32}{15}\\[0.4ex]
\frac73&\frac{122}{15}&\frac{42}5&\frac{797}{105}\\[0.4ex]
\frac95&\frac{42}5&\frac{86}5&\frac{81}5\\[0.4ex]
\frac{32}{15}&\frac{797}{105}&\frac{81}5&\frac{428}{15}
\end{pmatrix},
$$

$$
\mathbf M^T_1
=2\rho_0 R_0^5\operatorname{diag}\!\Big(\frac15,\frac17,\frac19,\frac1{11}\Big),\qquad
\mathbf M^P_1
=2\rho_0 R_0^5\big(2\tilde{\mathbf N}_1+\tilde{\mathbf P}_1\big).
$$

(The toroidal block at $\ell=2$ is $6\rho_0 R_0^5\operatorname{diag}(\tfrac17,\tfrac19,\tfrac1{11},\tfrac1{13})$. The poloidal matrix is computed from the same integral definition. Every entry has been checked by exact rational arithmetic.)

**Condition numbers ($4\times 4$ blocks).** The table below gives $2$-norm condition numbers computed from the exact rational Gram matrix after conversion to float64. The “monomial comparison basis” is used only for the condition-number comparison (the powers $\zeta^2,\ldots,\zeta^5$, passed through the same formulae). It takes part in no assembly.

$$
\begin{array}{c|ccccc}
\ell&1&2&3&5&10\\\hline
\operatorname{cond}(\mathbf M^P_\ell),\ \text{Jacobi}&30.1&27.3&26.1&25.1&24.5\\
\operatorname{cond}(\mathbf M^P_\ell),\ \text{after diagonal scaling}&6.6&8.6&10.2&12.5&15.8\\
\operatorname{cond}(\mathbf M^T_\ell),\ \text{Jacobi}&2.2&1.86&1.67&1.46&1.26\\
\operatorname{cond}(\mathbf M^P_\ell),\ \text{monomial comparison basis}&4.0\times 10^5&3.4\times 10^5&3.2\times 10^5&3.2\times 10^5&3.8\times 10^5
\end{array}
$$

Previously, the $L^2$ mass matrix $\big(1/(i+j+5)\big)_{i,j=1}^4$ in the monomial basis had condition number $2.35\times 10^6$. In the Jacobi basis the corresponding $L^2$ mass matrix $\int\psi_i\psi_j\zeta^2\,d\zeta$ is diagonal, with condition number $(2\ell+9)/(2\ell+3)\le 11/5$. The condition number of the kinetic-energy mass block falls from about $10^5$ to about $25$–$30$ (and to at most $16$ after diagonal scaling), a reduction of about four orders of magnitude.

The improvement is **inside a block**. The eigenvalue scale of blocks at different $\ell$ grows like $\Lambda_\ell^2\sim\ell^4$, so the global condition number $\operatorname{cond}(\mathbf M_\Gamma|_{\ell\ge 1})$ still grows with $L_{\max}$. That growth is handled by the block-diagonal preconditioner of Section 9.4, whose preconditioner is $\mathbf M_{B1}^{-1}$, the inverse of each $(\ell,m)$ block.

**Link to Section 10.2.** The next section needs the value and the derivatives of $\psi_{\ell k}$ at $\zeta=1$. They come from Proposition 8.3(2): $\psi_{\ell k}(1)=1$, $\sigma_{\ell k}=\psi_{\ell k}'(1)=\ell+k(k+2\ell+3)$, and the second derivative $\kappa_{\ell k}=\psi_{\ell k}''(1)$ is tabulated in Section 10.2.1. The symbols $\Lambda_\ell$, $Y_{\ell m}$, $R(\hat{\boldsymbol\omega},t)$, and $\nabla=\nabla_{S^2}$ keep the meanings fixed in this section and in Chapter 8.

---

## 10.2 Free-surface velocity, the surface-invariant shear rate, and the traction condition

The velocity coefficients, the radial basis, and the expansion used in this section are those of Sections 8.1–8.2: $\Phi=R^2\sum q_{\ell km}\psi_{\ell k}(r/R)Y_{\ell m}$ and $\Psi=R\sum s_{\ell km}\psi_{\ell k}(r/R)Y_{\ell m}$, with star-shaped boundary $R=R(\hat{\boldsymbol\omega},t)$. Write $\Lambda=\ell(\ell+1)$, and write $\nabla:=\nabla_{S^2}$ and $\Delta:=\Delta_{S^2}$ for the operators on the unit sphere.

### 10.2.1 Surface trace of the radial basis

For $\psi_{\ell k}(\zeta)=\zeta^\ell P_k^{(0,2\ell+2)}(2\zeta-1)$, write $\sigma_{\ell k}:=\psi'_{\ell k}(1)$ and $\kappa_{\ell k}:=\psi''_{\ell k}(1)$. Symbolic computation has verified $\psi_{\ell k}(1)=1$, and

$$
\boxed{\ 
\sigma_{\ell k}=\ell+k(k+2\ell+3),\qquad
\begin{array}{c|cccc}
k&0&1&2&3\\\hline
\sigma_{\ell k}&\ell&3\ell+4&5(\ell+2)&7\ell+18\\
\kappa_{\ell k}&\ell(\ell-1)&\ell(5\ell+7)&(\ell+2)(13\ell+15)&(\ell+2)(25\ell+63)
\end{array}\ }
$$

The first-derivative row is the specialization of Proposition 8.3(2). The second-derivative row follows by differentiating $\psi(\zeta)=\zeta^\ell P(2\zeta-1)$ twice and using

$$
P'(1)=\frac12 k(k+2\ell+3),\qquad
P''(1)=\frac18 k(k-1)(k+2\ell+3)(k+2\ell+4),
$$

which are the endpoint values of the Jacobi derivative recurrence. One obtains

$$
\kappa_{\ell k}
=\ell(\ell-1)+2\ell k(k+2\ell+3)+\frac12 k(k-1)(k+2\ell+3)(k+2\ell+4).
$$

Substitution of $k=0,1,2,3$ reproduces the table: $\ell(\ell-1)$, $\ell(5\ell+7)$, $(\ell+2)(13\ell+15)$, and $(\ell+2)(25\ell+63)$.

### 10.2.2 The velocity trace on a general star-shaped surface

**Proposition 10.2.** For $\Phi$ and $\Psi$ expanded as above, the contribution of each mode $(\ell,k,m)$ at $r=R(\hat{\boldsymbol\omega})$, using $\nabla$ and $\Delta$ only on $Y_{\ell m}$ and on $R$, is

$$
\begin{aligned}
v_r\big|_\Gamma
&=\sum q_{\ell km}\,\frac1R\Big[
R^2\Lambda Y
-2R(2-\sigma)\,\nabla R\cdot\nabla Y
-Y\big(R(2-\sigma)\Delta R+(2-2\sigma+\kappa)|\nabla R|^2\big)
\Big],\\
\mathbf v_S\big|_\Gamma
&=\sum q_{\ell km}\Big[R(1+\sigma)\nabla Y+(2-\kappa)\,Y\nabla R\Big],\\
\mathbf v^T\big|_\Gamma
&=\sum s_{\ell km}\Big[R\nabla Y+(1-\sigma)\,Y\nabla R\Big]\times\mathbf e_r,
\end{aligned}
$$

where $\sigma=\sigma_{\ell k}$, $\kappa=\kappa_{\ell k}$, and $Y=Y_{\ell m}(\hat{\boldsymbol\omega})$. The surface velocity is $\mathbf v|_\Gamma=v_r\mathbf e_r+\mathbf v_S+\mathbf v^T$.

*Proof.* Lemma 5.5 gives $v_r=L^2\Phi/r$, $\mathbf v_S=r^{-1}\nabla\,\partial_r(r\Phi)$, and $\mathbf v^T=\nabla\Psi\times\mathbf e_r$, with $\nabla$ acting at fixed $r$. For a single mode write $A(\hat{\boldsymbol\omega};r)=R^2\psi(r/R)$ and $B=R\psi(r/R)$, and set $\zeta=r/R$. Differentiation at fixed $r$ uses $\partial_R\zeta=-\zeta/R$. Then

$$
\begin{aligned}
A_R
&=2R\psi+R^2\psi'(-\zeta/R)
=R(2\psi-\zeta\psi'),\\
A_{RR}
&=2\psi-2\zeta\psi'+\zeta^2\psi''.
\end{aligned}
$$

At $\zeta=1$, where $\psi=1$, $\psi'=\sigma$, and $\psi''=\kappa$,

$$
A=R^2,\qquad A_R=R(2-\sigma),\qquad A_{RR}=2-2\sigma+\kappa.
$$

The radial factor of $\partial_r(rA)$ is $C=R^2(\psi+\zeta\psi')$. Its derivative in $R$ at fixed $r$ is

$$
C_R=2R(\psi+\zeta\psi')-R\zeta(2\psi'+\zeta\psi''),
$$

so

$$
C\big|_{\zeta=1}=R^2(1+\sigma),\qquad C_R\big|_{\zeta=1}=R(2-\kappa).
$$

For the toroidal factor, $B_R=\psi-\zeta\psi'$, and $B_R|_{\zeta=1}=1-\sigma$, while $B|_{\zeta=1}=R$.

At fixed $r$, $\nabla A=A_R\nabla R$ and

$$
\Delta A=A_R\Delta R+A_{RR}|\nabla R|^2.
$$

The spherical operator on a product is

$$
L^2(AY)=A\Lambda Y-2\nabla A\cdot\nabla Y-Y\Delta A.
$$

Division by $r=R$ and substitution of the endpoint values produce $v_r$. Likewise,

$$
\mathbf v_S=R^{-1}\nabla(CY)=R^{-1}\big(C\nabla Y+Y C_R\nabla R\big)
$$

becomes $R(1+\sigma)\nabla Y+(2-\kappa)Y\nabla R$, and

$$
\mathbf v^T=\nabla(BY)\times\mathbf e_r=\big(B\nabla Y+Y B_R\nabla R\big)\times\mathbf e_r
$$

becomes the toroidal formula. ∎

(The three formulae have been compared, at one point, with direct differentiation on the non-spherical radius $R=2+0.3\sin\theta\cos\phi+0.2\cos\theta$, at $\ell=2$ and $k=2$. They agree to about $10^{-14}$.)

When $\nabla R=0$, they reduce to $v_r=R\Lambda q Y$, $\mathbf v_S=R(1+\sigma)q\nabla Y$, and $\mathbf v^T=Rs\,\nabla Y\times\mathbf e_r$.

### 10.2.3 The surface-invariant shear rate

Let $x^a$ ($a=1,2$) be local coordinates on $\Gamma$, with tangent basis $\mathbf e_a=\partial_a X$ and metric $g_{ab}=\mathbf e_a\cdot\mathbf e_b$ (Section 9.0). Let $\mathbf n$ be the unit normal, let $K_{ab}:=\mathbf e_a\cdot\partial_b\mathbf n=-\mathbf n\cdot\partial_a\partial_b X$ be the second fundamental form (on a sphere with outward normal, $K_{ab}=g_{ab}/R>0$), and let $H=g^{ab}K_{ab}$ be the mean curvature. Decompose $\mathbf v|_\Gamma$ as

$$
v_a=\mathbf v\cdot\mathbf e_a,\qquad
v_n=\mathbf v\cdot\mathbf n=(\mathbf n\cdot\mathbf e_r)\,\partial_t R.
$$

Let $\nabla_a$ be the Levi-Civita covariant derivative of $g_{ab}$. The components of $\mathbf D=\operatorname{sym}\nabla\mathbf v$ in the surface frame are

$$
\boxed{
D_{ab}=\nabla_{(a}v_{b)}+v_n\,K_{ab},\qquad
D_{nn}=-g^{ab}D_{ab}=-\big(\nabla_a v^a+H\,v_n\big),\qquad
D_{na}=\mathbf n\cdot\mathbf D\cdot\mathbf e_a.}
$$

*Derivation.* The tangential components

$$
D_{ab}=\mathbf e_a\cdot\mathbf D\cdot\mathbf e_b
=\tfrac12(\mathbf e_a\cdot\partial_b\mathbf v+\mathbf e_b\cdot\partial_a\mathbf v)
$$

involve only tangential derivatives. Substitute $\mathbf v=v^c\mathbf e_c+v_n\mathbf n$ and use the Gauss–Weingarten formulae $\partial_a\mathbf e_b=\Gamma^c_{ab}\mathbf e_c-K_{ab}\mathbf n$ and $\partial_a\mathbf n=K_a{}^c\mathbf e_c$. The tangential projection is $\mathbf e_a\cdot\partial_b\mathbf v=\nabla_b v_a+v_n K_{ab}$, and symmetrization produces $D_{ab}$. The normal–normal component $D_{nn}$ contains a normal derivative and cannot be read from surface values alone. Incompressibility, $\operatorname{tr}\mathbf D=D_{nn}+g^{ab}D_{ab}=0$, supplies the displayed formula. ∎

The tensor $\mathbf D$ is traceless, and $\dot\gamma^2=2\mathbf D:\mathbf D$. In the surface frame, with inverse metric $g^{ab}$,

$$
\boxed{
\dot\gamma_\Gamma^2
=2\Big[(g^{ab}D_{ab})^2+g^{ac}g^{bd}D_{ab}D_{cd}+2\,g^{ab}D_{na}D_{nb}\Big].}
$$

**Traction condition.** On a free surface, $\boldsymbol\sigma\mathbf n=-p_{\rm ext}\mathbf n$. With $\boldsymbol\sigma=-p\mathbf I+2\eta\mathbf D+\mathbf T_{\rm tide}$, the normal component gives $p|_\Gamma=p_{\rm ext}+\mathbf n\cdot\boldsymbol\tau\cdot\mathbf n$ (Section 4.22), and the tangential component gives

$$
\boxed{
D_{na}=-\frac{\mathbf n\cdot\mathbf T_{\rm tide}\cdot\mathbf e_a}{2\eta}\quad(\eta>0).}
$$

In the mixed weak form this condition is a natural boundary condition. It holds only in the Galerkin sense. This paper also uses it as a **consistency relation** for the closure of the surface shear rate: after $\eta^{(n)}$ and $\mathbf T^{(n)}_{\rm tide}$ have been frozen, the formula supplies $D_{na}$, and $\dot\gamma_\Gamma$ is then computed. In the absence of a tide, $D_{na}=0$, the third term drops, and

$$
\dot\gamma_\Gamma^2\big|_{\mathbf T_{\rm tide}=0}
=2\Big[(g^{ab}D_{ab})^2+g^{ac}g^{bd}D_{ab}D_{cd}\Big].
$$

The viscosity $\eta$ itself depends on $\dot\gamma$, so the dependence is implicit. In the SCF one freezes the viscosity $\eta^{(n)}$ of the previous iteration. This is an iterative approximation, not an algebraic identity.

**Self-consistency report.** The $D_{na}$ of the actual Galerkin solution does not equal the formula exactly. Their difference

$$
\delta_{na}:=\|D_{na}^{h}-D_{na}^{\rm target}\|_{L^2(\Gamma)}
$$

must be reported as $K$ and $L_{\max}$ vary. It measures the extent to which the reduced model satisfies the traction condition.

### 10.2.4 A special case: the fixed reference sphere

**Proposition 10.3 (The reduced condition $D_{rS}=0$).** Only under (i) a constant radius $R\equiv R_0$, so that the surface is a fixed reference sphere, $\mathbf n=\mathbf e_r$, and $K_{ab}=g_{ab}/R_0$; (ii) a velocity that contains only a single spherical-harmonic degree $(\ell,m)$, with distinct pairs $(\ell,m)$ decoupled; and (iii) $\mathbf T_{\rm tide}=\mathbf 0$, do Proposition 10.2 and the traction condition above reduce to

$$
\begin{aligned}
v_r&=R_0\Lambda\sum_k q_k Y,\\
D_{rr}&=\Lambda\sum_k(\sigma_{\ell k}-1)q_k Y,\\
2D_{rS}
&=\Big[\sum_k(\Lambda+\kappa_{\ell k}-2)q_k\Big]\nabla Y
+\Big[\sum_k(\sigma_{\ell k}-1)s_k\Big]\nabla Y\times\mathbf e_r,
\end{aligned}
$$

$$
D_{SS}
=\sum_k\Big[
(1+\sigma_{\ell k})q_k\operatorname{Hess}_S Y
+\Lambda q_k Y\,\mathbf I_S
+s_k\operatorname{sym}\nabla\big(\nabla Y\times\mathbf e_r\big)
\Big].
$$

One has $\operatorname{tr} D_{SS}=-\Lambda\sum_k(\sigma_{\ell k}-1)q_k Y=-D_{rr}$, in agreement with incompressibility. Because $\nabla Y$ and $\nabla Y\times\mathbf e_r$ are pointwise orthogonal, the condition $D_{rS}=0$ is equivalent to two independent scalar constraints

$$
\boxed{
\sum_{k=0}^3 c^{P}_{\ell k}q_{\ell km}=0,\ \ c^P_{\ell k}=\Lambda_\ell+\kappa_{\ell k}-2;\qquad
\sum_{k=0}^3 c^{T}_{\ell k}s_{\ell km}=0,\ \ c^T_{\ell k}=\sigma_{\ell k}-1.}
$$

$$
\begin{array}{c|cccc}
k&0&1&2&3\\\hline
c^P_{\ell k}&2(\ell-1)(\ell+1)&2(3\ell^2+4\ell-1)&14(\ell+1)(\ell+2)&2(\ell+2)(13\ell+31)\\
c^T_{\ell k}&\ell-1&3(\ell+1)&5\ell+9&7\ell+17
\end{array}
$$

The table is the substitution of Section 10.2.1 into $c^P=\Lambda+\kappa-2$ and $c^T=\sigma-1$. For $k=0$, $\sigma=\ell$ and $\kappa=\ell(\ell-1)$ give $c^P=2(\ell^2-1)=2(\ell-1)(\ell+1)$ and $c^T=\ell-1$. For $k=1$, $\sigma=3\ell+4$ and $\kappa=\ell(5\ell+7)$ give $c^P=2(3\ell^2+4\ell-1)$ and $c^T=3(\ell+1)$. For $k=2$, $\sigma=5(\ell+2)$ and $\kappa=(\ell+2)(13\ell+15)$ give $c^P=14(\ell+1)(\ell+2)$ and $c^T=5\ell+9$. For $k=3$, $\sigma=7\ell+18$ and $\kappa=(\ell+2)(25\ell+63)$ give $c^P=2(\ell+2)(13\ell+31)$ and $c^T=7\ell+17$.

Consistency check: the translational and rotational modes at $\ell=1$, $k=0$ have $c^P_{10}=c^T_{10}=0$. A rigid motion produces no surface shear.

In this case $\dot\gamma_\Gamma^2=2\big[D_{rr}^2+D_{SS}:D_{SS}\big]$. The whole set of formulae applies **only** to the special case just stated. For a moving non-spherical radius $R(\theta,\varphi)$, for cross-degree modes that include toroidal coupling, or in the presence of a tidal backstress, one must use the general formulae of Section 10.2.3. One must not set $D_{rS}=0$ again. If one further sets $s_{\ell km}\equiv 0$, so that there is no toroidal flow, only the poloidal constraint remains.

---

## 10.3 Closure of the endogenous, nonuniform, physical friction angle on the surface

Remark. The pressure $p_{\rm ref}=\rho_{\rm bulk}|\mathbf g\cdot\mathbf n|h_s$ is a representative reference pressure of the shallow flowing layer. The coefficient $\mu_{\rm eff}$ constructed from it is a nominal geometric projection angle based on that representative surface pressure $p_{\rm ref}$. (A nominal geometric projection angle is monotone in pressure when there is no cohesion, or at high confining pressure. In general it is not guaranteed to be conservative. The true multiaxial yield is checked separately by the outer acceptance test.) The physical effective pressure $p_{*,\mathrm{phys}}=p+p_c$ is used only for the dynamic rheology $\mu_d$. The reference effective pressure $p_{*,\mathrm{ref}}=p_{\mathrm{ref}}+p_c$ is used only for the geometric projection angle $\mu_{\rm eff}$. This section does not use an unsubscripted $p_*$.

**Material constants are frozen.** The solid fraction $\nu=\nu_0$ (for example $\nu_0=0.6$ for a standard rubble pile [6][7]) and the coordination number $Z=Z_0$ (for example $Z_0=6$) are **material constants given by the configuration**. They are not updated with the SCF iteration or with the relaxation, and they have no evolution law. Consequently $\rho_{\rm bulk}=\rho_p\nu_0$, $c=c(\nu_0,Z_0)$, $\mu_s=\mu_s(\nu_0,Z_0)$, $\mu_2$, $I_0$, $\mu_y\equiv\mu_s$ (Section 2.8, B2), and $F_H$ are all constants. The symbols below keep their original form and no longer carry a nodal or an iteration subscript. The superscript $(n)$ on $\nu^{(n)}$ and $Z^{(n)}$ in Sections 4.7–4.20 is the **index of a parameter sequence** in the reduction theorem. It has nothing to do with the SCF iteration. The density $\rho_{\rm bulk}$ is the packing density of the loose surface layer. The density $\rho_K$ is the internal Cauchy continuum density. They are two independent configuration quantities, linked only through the density weight $\rho_{\rm bulk}/\rho_K|_\Gamma$ (Section 2.8). The range of the ratio $\rho_{\rm bulk}/\rho_K|_\Gamma$ should be reported at each step.

Every quantity below is evaluated on the frozen geometry $R^{(n)}$ of geological step $n$, on the frozen effective gravity $g_i^{(n)}$, and on the converged pressure $p_i^{(n)}$ of Section 8.7, taken at its regularized value $\tilde p_i^{(n)}=\max([p_i^{(n)}]_+,p_{\min})$ (Section 2.8). Define first

$$
p_{*,{\rm phys},i}^{(n)}:=\tilde p_i^{(n)}+p_c\ge p_{\min}>0,
\qquad
p_c=C_c\frac{Z_0\nu_0 F_H}{d_p^2}.
$$

If a particular contact model supplies $F_H=A_H a/(6h_0^2)$, that expression is substituted. The numerical lower bound $p_{\min}>0$ (for example $10^{-6}\,\mathrm{Pa}$; an f32 implementation must be rechecked in f64) removes the division-by-zero singularity of $\sqrt{p_*/\rho_p}$ and of the denominator of $\mu_{\rm eff}$ at $p=0$ and $p_c=0$. The reference pressure $p_{*,{\rm ref}}$ is likewise replaced by $\max(\cdot,p_{\min})$. The lower bound is only a numerical protection. It must not be used only in a denominator. By the mandatory rule of Section 4.13, if a lower-bound pressure is used in the denominator of $\mu_d$, then the pressure inside the stress amplitude $c+\mu_d p$ must be the same regularized pressure $\max(p,p_{\min})$.

The surface shear rate is $\dot\gamma=\sqrt{2\mathbf D':\mathbf D'}$. Its nodal value is $\mathcal F_i:=\dot\gamma_i^{(n)}=\dot\gamma_\Gamma(\hat{\boldsymbol\omega}_i)$, computed from the general formula of Section 10.2.3 on the **converged** quasi-static Stokes solution $\mathbf u^{(n)}$ of step $n$ (system (8.4′) and Section 10.4.2). The yield test of Section 2.8 and the pressure of Section 8.7 use the same velocity field. The fixed-sphere formulae of Section 10.2.4 are used only as a check of that special case.

**Reference pressure, and the treatment of the singularity at $p_*=0$.** At a free-surface node, $p_i=p_{\rm ext}+\mathbf n\cdot\boldsymbol\tau\cdot\mathbf n$ may vanish (Section 4.22), and the denominators of $I_c$ and of $\mu_{\rm eff}$ may vanish. The slope proxy uses the effective pressure at a **reference depth**,

$$
p_{\mathrm{ref},i}=\rho_{\mathrm{bulk}}|g_i^{(n)}\cdot n_i^{(n)}|\,h_s,\qquad
p_{*,{\rm ref},i}:=\max(p_{\mathrm{ref},i}+p_c,\,p_{\min})>0,
$$

where $h_s>0$ is the configured thickness of the active surface layer. The bound $p_{*,{\rm ref},i}>0$ is guaranteed by the explicit floor $p_{\min}>0$. The anti-overhang constraint $g_i\cdot\widehat n_i\le-\epsilon|g_i|$, together with $|g_i|>0$, only guarantees that the reference pressure points in the physical compressive direction. To avoid division by zero or underflow in $\sqrt{p_{*,{\rm phys}}/\rho_p}$, the shallow dynamic friction coefficient $\mu_{d,i}$ uses the physical effective pressure $p_{*,{\rm phys},i}$, already defined and regularized, through the rational expression

$$
\mu_{d,i}
=\mu_s+(\mu_2-\mu_s)\frac{d_p\,\mathcal F_i}{I_0\sqrt{p_{*,{\rm phys},i}/\rho_p}+d_p\,\mathcal F_i}.
$$

It is monotone nondecreasing in $\mathcal F_i\ge 0$, with range $[\mu_s,\mu_2)$. As $p_{*,{\rm phys}}\downarrow 0$ with $\mathcal F_i>0$, it tends to $\mu_2$. The expression is a genuine indeterminate form only when $p_{*,{\rm phys}}=\mathcal F_i=0$, and in that case one takes the static branch $\mu_d:=\mu_s$. An f32 implementation must place a floor $p_{\min}>0$ under $p_{*,{\rm phys}}$ and must recheck the result in f64.

**Rheological-chain switch (the configuration item `mu_proj_mode`, which must be given explicitly).**

- `dynamic`: $\mu_{\mathrm{proj},i}:=\mu_{d,i}$. The shear-rate chain $\mathbf u^{(n)}\to\dot\gamma_i\to I_c\to\mu_d\to\mu_{\rm eff}\to\phi_i$ is then **fully active**. The surface velocity trace (Proposition 10.2) and the surface-invariant shear rate (Section 10.2.3) determine the geometric projection angle of a movable node directly.
- `static`: $\mu_{\mathrm{proj},i}:=\mu_y=\mu_s$. The angle $\phi_i$ is a **static geometric relaxation proxy**, independent of the shear rate.

Neither option changes the yield classification. The tolerance $F_\delta$ of Section 2.8, the rigid core, the movable set, and the Mode C flux are all computed on the unique stress field, using $\mu_d$ for the viscosity and $\mu_y$ for the strength, independently of `mu_proj_mode`.

Under either option, $\phi_i$ is only a nominal geometric proxy for the SOCP. It is not an equivalent rewriting of the true multiaxial yield. Identifying $\phi=\arctan\mu$ with an angle of repose is a modelling hypothesis that sits between Mohr–Coulomb and Drucker–Prager. It holds only for a loose surface layer whose applicability has been declared.

**Scalarization of the tidal backstress, and the nominal projection angle (based on the representative surface pressure $p_{\rm ref}$; not absolutely conservative).** Write $\|A\|_{\mathrm{eq}}=\|A\|_F/\sqrt{2}$. The yield condition $\|s-T\|_{\mathrm{eq}}\le c+\mu_y p$ implies only $\|s\|_{\mathrm{eq}}\le c+\mu_y p+\|T\|_{\mathrm{eq}}$. This is a **necessary** condition, and it is not conservative. For example, if $c=0$, $\mu_y p=1$, $T=\mathrm{diag}(0.6,-0.6,0)$, and $s=\mathrm{diag}(-0.5,0.5,0)$, then $\|s\|_{\mathrm{eq}}=0.5\le 1.6$, but $\|s-T\|_{\mathrm{eq}}=1.1>1$, so the material has already yielded. A **sufficient** condition at a given pressure $p$ follows from $\|s-T\|_{\mathrm{eq}}\le\|s\|_{\mathrm{eq}}+\|T\|_{\mathrm{eq}}$: if $\|s\|_{\mathrm{eq}}\le c+\mu_y p-\|T\|_{\mathrm{eq}}$, then $\|s-T\|_{\mathrm{eq}}\le c+\mu_y p$. One therefore takes

$$
\mu_{\mathrm{eff},i}^{(n)}
:=\frac{\Big[c+\mu_{\mathrm{proj},i}\,p_{\mathrm{ref},i}-\|T_{\mathrm{tide},i}\|_F/\sqrt{2}\Big]_+}{p_{*,\mathrm{ref},i}}\ge 0,
\qquad
\boxed{\phi_i^{(n)}=\arctan\mu_{\mathrm{eff},i}^{(n)}}.
$$

**Monotonicity, and a counterexample ($\phi_i$ is not an absolutely conservative angle).** Write $T:=\|T_{\rm tide}\|_{\rm eq}$ and $\mu:=\mu_{\rm proj}$, and regard the proxy angle as a function of pressure, $\mu_{\rm eff}(p)=[c+\mu p-T]_+/(p+p_c)$. Whenever $c+\mu p-T>0$,

$$
\begin{aligned}
\frac{d}{dp}\,\frac{c+\mu p-T}{p+p_c}
&=\frac{\mu(p+p_c)-(c+\mu p-T)}{(p+p_c)^2}
=\frac{\mu p_c-(c-T)}{(p+p_c)^2}.
\end{aligned}
$$

Thus $\mu_{\rm eff}$ is nondecreasing in $p$ if and only if $\mu p_c\ge c-T$. In that case $p_{\rm ref}\le p$ implies $\mu_{\rm eff}(p_{\rm ref})\le\mu_{\rm eff}(p)$, and the proxy angle does not exceed the value at the true pressure (it is biased tight). Conversely, if $c-T>\mu p_c$ (cohesion is large), then $\mu_{\rm eff}$ is strictly decreasing in $p$. When $p_{\rm ref}<p$, the proxy angle is **instead too large** (biased loose), and the slogan “$p_{\rm ref}\le p$ is conservative” is false. For a general granular body one therefore makes no claim of an “absolutely conservative sufficient condition”. The angle $\phi_i$ is only a nominal geometric projection angle based on the representative surface pressure $p_{\rm ref}$. It is monotone when there is no cohesion, since $c=0$ and $T\ge 0$ make $\mu p_c\ge -T$ automatic, and it is monotone at high confining pressure. It is a geometric proxy for the SOCP, not an algebraic equivalent. In general, the true multiaxial yield is checked separately by the tensorial yield condition $\|s-T_{\rm tide}\|_{\rm eq}\le c+\mu_y p$ in the outer acceptance test.

Since $\mu_{\mathrm{eff}}=\tan\phi$, the projection uses

$$
\sin^2\phi_i^{(n)}
=\frac{(\mu_{\mathrm{eff},i}^{(n)})^2}{1+(\mu_{\mathrm{eff},i}^{(n)})^2}.
$$

**Spurious effect of a vertex Hessian.** The stress $T_{\mathrm{tide}}\propto\operatorname{dev} H_W$ takes the value $H^{\mathrm{FP}}_{\ell_0}$ at mesh vertices and along edges. There $C_{ij}\ne 0$, so the result depends on an arbitrary reference length $\ell_0$ (Section 1.3: $\Delta H=G\rho C\log(\ell_0/\ell_1)$). It is an artifact of the polyhedral discretization. For use in the stress, the Hessian must be evaluated in the interior of a face, at the face centre, or one must take the average of the one-sided limits on the two sides of the face. Proposition 2.7.1 proves that at every point of the relative interior of a triangular face, and in particular at the centroid, the hemispherical moment vanishes, the logarithmic RTFP singularity is absent, and the Feynman integral converges to the same scale-independent finite part. For a point in the interior of a planar face,

$$
H^{\pm}=H^{\mathrm{FP}}\pm 2\pi G\rho_K\,\mathbf n\mathbf n^{\!\top},
$$

where “$+$” is the exterior limit and “$-$” is the interior limit. The finite part $H^{\rm FP}$ contains the whole nonlocal finite contribution. At an interior point of the face, $C_{ij}=0$, so the value is independent of $\ell_0$. The average of the two sides is exactly $H^{\mathrm{FP}}$. The jump $H^+-H^-=4\pi G\rho_K\mathbf n\mathbf n^{\!\top}$ agrees with $\operatorname{tr} H^--\operatorname{tr} H^+=-4\pi G\rho_K$. One must not discard the nonlocal part and retain only the local contact tensor $-\tfrac{2\pi}{3}G\rho_K\mathbf I\pm 2\pi G\rho_K\mathbf n\mathbf n^{\!\top}$. Doing so would omit the large-scale tidal tensor. Vertex values must not enter $T_{\mathrm{tide}}$ directly.

---

## 10.4 The geometric self-consistent-field closure with the preceding equations

### 10.4.1 Time levels, exogenous spin, and the geological step

The quasi-static route has three nested levels. The meaning of “time” is different at each level, and the levels must not be mixed.

| Level | Variables | Meaning and method of solution |
|---|---|---|
| Exogenous spin table | $t\mapsto\omega(t)$ | given by the configuration; **not solved**; no feedback from the internal flow, from a change of inertia, or from angular momentum |
| Geological step $n$ | $t_n$, $\Delta t_n$, $\lambda$ | geometric advance $R^{(n)}\to R^{(n+1)}$, with $t_{n+1}=t_n+\lambda\Delta t_n$ |
| Picard iteration $k$ | $(\mathbf u^k,p^k)$ | nonlinear solution of the instantaneous Stokes equilibrium at fixed geometry and fixed $\omega_n$; **contains no time** |

**Exogenous spin table (the configuration item `spin_schedule`; exactly one form must be given).**

- `stages`: $\{(\omega_j,T_j)\}_{j=1}^{J}$ with $T_j>0$. Stage $j$ occupies $[s_{j-1},s_j)$, where $s_0=0$ and $s_j=s_{j-1}+T_j$, and on that interval $\omega(t)\equiv\omega_j$. The terminal time is $T_{\rm end}=s_J$.
- `linear`: $\omega(t)=\omega_0+\dot\omega_{\rm ext}\,t$ for $t\in[0,T_{\rm end}]$, where $\dot\omega_{\rm ext}$ is a given constant (for example a mean YORP acceleration), and one requires $\omega(t)\ge 0$.
- `table`: nodes $(t_k,\omega_k)_{k=0}^{K_\omega}$, with $t_k$ strictly increasing. The spin $\omega$ is the piecewise-linear interpolant, and $T_{\rm end}=t_{K_\omega}$.

Each geological step freezes $\omega_n:=\omega(t_n)$. The spin enters the model only through $\mathbf g_{\rm eff}=\nabla U+\omega_n^2\mathbf x_\perp$ and through $H_W=\nabla^2 U+\omega_n^2\operatorname{diag}(1,1,0)$, and hence through $\mathbf T_{\rm tide}$. **Hypothesis A-S (spin about a principal axis):** the spin axis is always the body-frame $z$-axis, that is, the current principal axis of maximum inertia, as guaranteed by the realignment of step 8. Nutation damping is regarded as instantaneous relative to $\Delta t_n$. The value of $\omega$ is never updated from the inertia tensor or from the angular momentum. A deformation that changes $I_1$ does not feed back into $\omega$. That is the definition of an exogenous prescription, not an omission.

**Geological step.**

$$
\Delta t_n
=\min\{\Delta t_{\rm CFL},\ \Delta t_C,\ \Delta t_{\rm char},\ \Delta t_\omega,\ \Delta t_{\max},\ t_{\rm next}-t_n\}.
$$

- Geometric CFL of the movable region:

$$
\Delta t_{\rm CFL}
=\theta_{\rm CFL}\,h_{\Gamma,\min}
\Big/\max_{i\in\mathcal A^{(n)}}\big(|\mathbf v_{{\rm sl},i}\cdot\mathbf n_i|/(\mathbf n_i\cdot\mathbf e_r)\big),
$$

with $\theta_{\rm CFL}\le 0.5$. If $\mathcal A^{(n)}=\varnothing$, the value is $+\infty$.
- Erosional flux:

$$
\Delta t_C
=\theta_C\min\Big\{
\frac{\sigma_0}{k_e\max_i[F_{\delta,i}]_+},\ 
\frac{\rho_{\rm bulk}h_s}{\max_i|J_{\rm dep}-J_{\rm ero}|_i}
\Big\},
$$

with $\theta_C<1$, so that $\rho_s\ge 0$ and the change of layer thickness in one step does not exceed $\theta_C h_s$. If there is no flux, the value is $+\infty$.
- Single-valuedness of the $\rho_s$ characteristics: $\Delta t_{\rm char}=\theta_{\rm ch}/\|\nabla_{S^2}\mathbf c\|_\infty$, with $\theta_{\rm ch}<1$ and $\mathbf c=\chi_A\mathbf v_{{\rm sl},S}/R$ (Section 9.2).
- Resolution of the spin table: under `stages` there is no separate restriction (the quantity $t_{\rm next}$ handles it). Under `linear` or `table`,

$$
\Delta t_\omega
=\varepsilon_\omega\max\{\omega(t_n),\omega_{\rm ref}\}/\max\{|\dot\omega(t_n)|,\epsilon\},
$$

so that the relative change of spin in one step does not exceed $\varepsilon_\omega$.
- The quantity $\Delta t_{\max}$ is a configured upper bound. The time $t_{\rm next}$ is the next stage boundary, table node, or $T_{\rm end}$.

**Why there is no viscous stiffness catastrophe.** The velocity is computed implicitly from (8.4′). The step $\Delta t_n$ does not appear in the velocity equation, and it is independent of $\eta_{\rm hi}=(c+\mu_2\tilde p)/\dot\gamma_\varepsilon$. Regularized creep inside the rigid core is removed by $\chi_Y=0$ and does not enter $\Delta t_{\rm CFL}$. The number of steps required is therefore about $T\cdot\max_{\mathcal A}|\mathbf v_{\rm sl}\cdot\mathbf n|/(\theta_{\rm CFL}h_{\Gamma,\min})$. It is fixed by the true landslide speed. There is no CFL restriction of viscous-diffusion type. An equilibrated stage whose movable set is empty needs only one step, which jumps to the end of the stage. The factor $\lambda\in(0,1]$ defaults to $1$ and is reduced only when the outer geometric acceptance test is difficult. Accumulated time is measured by $\lambda\Delta t_n$. A step that is recomputed after the trust region has been shrunk because of an overhang or a self-intersection does not advance time. Stage equilibrium and the termination condition are criterion (c) of Section 2.15. Stability of that criterion (sufficiency of the step-size rule), and the conditions under which stage equilibrium is reached in finitely many steps, belong to PO-12.

### 10.4.2 Quasi-static Stokes–pressure solution at fixed geometry and fixed $\omega_n$

Geological step $n$ is executed on the frozen data $R^{(n)}$, $\rho_K$, $\omega_n$, $\mathbf g_{\rm eff}$, and $\mathbf T_{\rm tide}$:

```text
Input: initial value u^0 (the converged coefficients of the previous geological step,
       reused without change; u^0 = 0 on the first step, or immediately after an 8b realignment),
       p~^0 (the converged pressure coefficients of the previous geological step, synthesized
       on the new geometry and replaced by max([.]_+, p_min); p_min on the first step).
for k = 0, 1, ..., k_max-1:
  (P1) Pressure: from the frozen pair (u^k, p~^k), compute
       f = rho_K g_eff + div T_tide + div tau_flow
       and the Dirichlet datum
       p_Gamma = p_ext + n.T_tide.n + n.tau_flow.n.
       Solve the Poisson problem of Section 8.7 for p^{k+1};
       p~^{k+1} = max([p^{k+1}]_+, p_min).
  (P2) Viscosity: at the interior quadrature points X_vol,
       gd^k = sqrt(2 D(u^k):D(u^k)),
       I_c = d_p gd^k / sqrt((p~^{k+1} + p_c)/rho_p),
       mu_d = mu_d(I_c) (Section 4.5),
       eta^{k+1/2} = (c + mu_d p~^{k+1}) / sqrt((gd^k)^2 + gd_eps^2).
       Optional relaxation:
       eta <- theta_eta eta^{k+1/2} + (1 - theta_eta) eta^{k-1/2},
       with theta_eta in (0,1].
  (P3) Velocity: assemble K(eta^{k+1/2}) and F, and solve (8.4')
       by the range-space method of Section 8.4, obtaining (u^{k+1}, lambda^{k+1}).
  (P4) Stop if criterion (a) of Section 2.15 holds.
If the iteration has not converged within k_max: continue in gd_eps
(first solve at 10 gd_eps, then reduce level by level to the target value,
each level starting from the solution of the previous level) and retry with a smaller theta_eta.
If the retry still fails, terminate and report "rheology not converged".
A passing status must not be emitted.
Output: u^(n), p^(n), lambda^(n), eta^(n); then apply criterion (b) of Section 2.15.
```

The identities $\nabla\cdot\boldsymbol\tau_{\rm flow}=\eta\Delta\mathbf v+2\mathbf D\nabla\eta$ and $\nabla\cdot\mathbf T_{\rm tide}=-\tfrac{8\pi}{3}G C_T\rho_p\nu_0 d_p^2\nabla\rho_K$, used in (P1), are those of Section 8.7. The converged pair $(\mathbf u^{(n)},p^{(n)})$ satisfies both the divergence-free Galerkin momentum equation (8.4′) and the pressure Poisson equation. That is the quasi-static equilibrium of the unique stress $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D$. The yield classification of Section 2.8, the friction-angle chain of Section 10.3, and the Mode C flux are all computed on this same solution.

**Track B (Appendix A.4).** If, in (P3), the block-diagonal inverse $\mathbf K_{\gamma,B1}^{-1}$ replaces $\mathbf K_\gamma^{-1}$ (a truncated model), the resulting $\mathbf u_B$ must still have the residual $\varepsilon_{\rm mom}^{\rm weak}$ of criterion (b) in Section 2.15 computed with the **full** matrix $\mathbf K$. That residual is the a-posteriori measure of the Track B truncation. If it exceeds the threshold, the step must be resolved on Track A.

### 10.4.3 The geometric increment subproblem

Geometric correction is a **sequential convex increment subproblem after the mechanical prediction**. The reference is the current geometry $R^{(n)}$ (Section 2.13). Linearize to first order at the current iteration point, and write

$$
\widehat n_i(d)=n_i^{(n)}+D n_i^{(n)}d,
\qquad
\widehat{\mathcal M}(d)=\mathcal M(R^{(n)},\rho^{(n)})+D\mathcal M^{(n)}d.
$$

The mechanical prediction $d_{\mathrm{pred}}^{(n)}$ and the three-way nodal partition into the locked set $\mathcal L^{(n)}$, the pure-deposition set $\mathcal D^{(n)}$, and the movable set $\mathcal A^{(n)}$ are given uniquely by Section 2.8 (B3–B4):

$$
d_{\mathrm{pred},i}^{(n)}
=\frac{\chi_{Y,i}\,\Delta t_n\,\mathbf v_{{\rm sl},i}^{(n)}\!\cdot n_i^{(n)}
+\dfrac{\rho_{\rm bulk}}{\rho_K(\mathbf x_i)}\,\Delta h_{C,i}}
{n_i^{(n)}\cdot\mathbf e_r}
\qquad(n\cdot e_r\ge c_0>0).
$$

Here $\mathbf v_{\rm sl}$ is the slip velocity of the converged quasi-static Stokes solution of step $n$, relative to the rigid core (the surface trace is synthesized by Proposition 10.2). The indicator $\chi_{Y,i}$ is the movability indicator based on the yield function $F_\delta$ of the unique stress field and on connectivity to the rigid core. The Mode C increment is $\Delta h_C=\Delta t_n(J_{\rm dep}-J_{\rm ero})/\rho_{\rm bulk}$. The density weight $\rho_{\rm bulk}/\rho_K$ makes the flux increment in the mass row equal to $\Delta h_C^*$. Metal-PDHG solves

$$
\boxed{
\begin{aligned}
(d_*^{(n)},t_*,s_{M*})
=\arg\min_{d,\,t\ge 0,\,s_M}\;&
\frac12\sum_{i\in\mathcal A^{(n)}} w_i\left(d_i-d_{{\mathrm{pred}},i}^{(n)}\right)^2
+\varrho\Big(\sum_{i}t_i+|s_M|\Big)\\
\text{s.t. }&d_i=0\quad(i\in\mathcal L^{(n)}),\qquad
d_i=d_{\mathrm{pred},i}^{(n)}\quad(i\in\mathcal D^{(n)}),\\
&\frac{\|g_i^{(n)}\times\widehat n_i(d)\|}{|g_i^{(n)}|}
\le\sin\phi_i^{(n)}+t_i\quad(i\in\mathcal S_{\mathrm{slope}}^{(n)}),\\
&g_i^{(n)}\cdot\widehat n_i(d)\le-\epsilon_n|g_i^{(n)}|\quad(i\in\mathcal S_{\mathrm{slope}}^{(n)}),\\
&D\mathcal M^{(n)}d+s_M=M-\mathcal M(R^{(n)},\rho^{(n)}),\\
&d\in\mathcal A_{\mathrm{lin}}^{(n)},
\qquad \|d_{\mathcal A}\|\le\Delta^{(n)}.
\end{aligned}}
$$

The set $\mathcal S_{\rm slope}^{(n)}\subseteq\mathcal A^{(n)}$ contains only those nodes of the movable set that belong to the loose surface layer to which the angle-of-repose proxy applies. The vector $d_{\mathcal A}$ is the restriction of $d$ to the movable set. The trust region constrains only the free variables. The objective **does not** contain $R_{\rm eq}$, $W_0$, or any equipotential term. If $\mathcal A^{(n)}=\varnothing$, the subproblem has no free variable, $d_*=d_{\rm pred}$ (zero on the locked set, and equal to the deposited amount on the deposition set), and Metal-PDHG is not called.

The mass row is equivalent to $\widehat{\mathcal M}(d)=M-s_M$. A feasible point is constructed by taking $d_{\mathcal L}=0$, $d_{\mathcal D}=d_{\rm pred,\mathcal D}$, $d_{\mathcal A}=0$, $s_M=M-\widehat{\mathcal M}(d)$, and $t_i=\big[\|g_i\times\widehat n_i(d)\|/|g_i|-\sin\phi_i\big]_+$. If the anti-overhang row is satisfied at that point, the point is feasible. The anti-overhang row has no slack. If it is violated, the outer level must repair the point by a geometric preprocessing in the direction $-g$, or the point must be removed when $\mathcal S_{\rm slope}$ is defined. This construction is only a feasibility construction. It is not a Slater certificate of a strict interior point. A strict interior point, if one is required, must be given a positive margin in the second-order cone and must satisfy every unrelaxed linear constraint strictly, as stated in Section 2.13. The objective is strictly convex in $d_{\mathcal A}$ (because $w_i>0$) plus an $\ell_1$ term, so a solution exists and $d_{\mathcal A*}$ is unique. The second-order-cone constraints, the affine constraints, and the trust ball make the problem a convex SOCP. Whether the constraints are truly satisfied is decided by whether $t_*$ and $s_{M*}$ vanish, and by the outer nonlinear geometric acceptance test.

The set $\mathcal A_{\mathrm{lin}}^{(n)}$ contains only the affine or conic interior approximations of nodal order, positive volume, and the trust region. True positive volume and the topological conditions are not convex constraints of Metal-PDHG. They are checked by the outer acceptance test. If they fail, the trust region is reduced and the subproblem is resolved. **Unified trust-region policy.** The inner Metal-PDHG SOCP is convex-feasible because of the elastic slack variables $t_i$ and $s_M$. The inner level never reduces the trust region. Only when the outer nonlinear geometric acceptance test finds a mesh overhang or a self-intersection, including a failure of positive volume or of a topological condition, does the outer level reduce the trust-region radius $\Delta^{(n)}$ and regenerate the subproblem; see Section 2.13. If the constraint on the normal norm is retained, the problem is an SOCP with a quadratic objective. It may be called a pure QP only after every constraint has been affinely approximated.

**Place of the tidal backstress and of yield inside the subproblem.** The quantity $\sin\phi_i^{(n)}$ in the subproblem comes from the nominal geometric projection proxy of Section 10.3, based on $p_{\rm ref}$. It constrains only the geometry of the movable loose surface layer. Vanishing of $t_*$ says only that the proxy constraint holds. A nonzero $t_*$ is not, by itself, an assertion of tensorial yield. True tensorial yield is not a constraint of the geometric acceptance test. It is the physical state that drives the flow. On the next geological step, $F_\delta$ is recomputed and the classification is repeated, on the new geometry, the new $\mathbf T_{\rm tide}$, and the new Stokes solution (Section 2.8). The outer geometric acceptance test checks only positive volume, absence of self-intersection, star-shapedness $n\cdot e_r\ge c_0$, and the mass residual. If no feasible candidate exists, the trust region is reduced. A fluid equipotential is not used as a fallback target.

### 10.4.4 The main loop of the quasi-static evolution

```text
0. Initialization: read the star-shaped polyhedron R^(0), the Cauchy density parameters
   (including the kernel centres c_k), the material constants
   (nu_0, Z_0, c, mu_s, mu_2, I_0, d_p, rho_p, C_c, F_H, C_T, gd_eps, delta_Y, p_min),
   the Mode C parameters (k_e, sigma_0, kernel K, h_s), spin_schedule, and every tolerance.
   If any item is missing -> ModelNotClosed, and refuse to run.
   rho_s^0 = rho_bulk h_s, eps_p_bar = 0, t_0 = 0.
   Perform the frame alignment of step 8b once (mandatory; the trigger condition is ignored).
For n = 0, 1, 2, ... until t_n >= T_end:
1. Forward evaluation: omega_n = omega(t_n). On V^(n) and rho_K, compute by the unified
   RTFP/Feynman representation the face-centre Hessian (Section 10.3), the nodal fields
   g_i and H_W,i, and grad U and Hess U at the interior quadrature points X_vol
   (the assembly points of (8.4')) and on the classification mesh X_Y (Section 2.8, B3).
   Interior points converge classically.
   g_eff = grad U + omega_n^2 x_perp,
   H_W = Hess U + omega_n^2 diag(1,1,0),
   T_tide = C_T rho_p nu_0 d_p^2 dev H_W.
2. Quasi-static Stokes–pressure Picard iteration (Section 10.4.2), producing
   (u^(n), p^(n), lambda^(n), eta^(n)).
   If either criterion (a) or (b) of Section 2.15 fails, retry by the continuation
   of Section 10.4.2. If the retry still fails, terminate and report.
3. Yield classification (Section 2.8, B2–B3): on X_Y, synthesize v, D, and p~,
   and compute tau_flow, tau_y, and F_delta.
   On the graph G_Y, find the unyielded connected components and take the component
   of largest mass as the rigid core U_core. This produces chi_Y,i.
   Fit the rigid motion (a_c, theta_c) of the core by least squares, and set
   v_sl = v - a_c - theta_c x x.
   Update the diagnostics eps_p_bar and Phi_p.
   Report the movable area fraction, the fraction in tension, Ek^{-1}, and eps_Euler.
4. Friction-angle chain (Section 10.3): from u^(n), synthesize the surface velocity
   by Proposition 10.2, compute gd_i by Section 10.2.3, and obtain mu_d,i, mu_proj,i,
   and phi_i (required only at movable nodes in S_slope).
5. Mode C (Section 2.8): chi_A = chi_Y * 1[rho_s > 0], and J_ero, J_dep
   (the kernel K is normalized; conservation is strict).
6. Step size (Section 10.4.1): Delta t_n.
   If spin_schedule = stages and A^(n) is empty and J_ero = 0 (hence J_dep = 0):
     stage equilibrium (criterion (c) of Section 2.15) -> emit the terminal state of the stage,
     set t_{n+1} to the end s_j of the current stage, set R^(n+1) = R^(n), and go to step 1.
   If A^(n) is empty and the schedule is not stages:
     Delta t_n = min(Delta t_omega, Delta t_max, t_next - t_n),
     d_pred = 0, and go directly to step 8 (the geometry is unchanged; only omega advances).
7. Form d_pred (Section 2.8, B4) and the nodal classes L / D / A.
   Solve the Metal-PDHG SOCP of Section 10.4.3 for d_*^(n).
8. Geometric update and frame realignment:
   8a. R^(n+1) = R^(n) + lambda d_*^(n) (on the locked set, d_* = 0). Rebuild the polyhedron.
       Outer geometric acceptance: positive volume, no self-intersection, n.e_r >= c_0,
       and |M(V^(n+1)) - M| <= eps_M. On failure, reduce Delta^(n), resolve step 7,
       and do not advance time.
       For rho_s: transport the single field by the semi-Lagrangian scheme of Section 9.2,
       along c = chi_A v_sl,S / R, with step lambda Delta t_n, including the Jacobian
       correction, then add lambda Delta t_n (J_dep - J_ero).
   8b. Standard frame realignment. Trigger: |x_cm| > eps_frame R_ref / 2, or
       max(|I_xz|, |I_yz|) > eps_frame I_1 / 2. If the trigger is not met, skip 8b,
       so that repeated resampling does not cause numerical diffusion.
       (i)   Centre of mass x_cm = (1/M_V) int_{V(R^(n+1))} rho_K x dV,
             with M_V = int rho_K dV.
       (ii)  Translation: every mesh point x <- x - x_cm; every kernel centre c_k <- c_k - x_cm.
       (iii) Inertia tensor I = int rho_K (|x|^2 Id - x x^T) dV. Eigendecomposition
             I_1 >= I_2 >= I_3, with unit eigenvectors a_1, a_2, a_3.
       (iv)  New frame: e_z' = a_1 (the axis of the largest principal moment I_1, which is
             the spin axis), e_x' = a_3 (the axis of the smallest principal moment I_3),
             e_y' = e_z' x e_x'. Signs are chosen so that a_1.e_z >= 0 and a_3.e_x >= 0,
             which makes Q as close to the identity as possible.
             Near degeneracy: if I_1 - I_2 <= eps_deg I_1, take e_z' to be the normalization
             of the projection of the old e_z onto span{a_1, a_2}. If I_2 - I_3 <= eps_deg I_1,
             take e_x' to be the normalization of the projection of the old e_x onto the
             intersection of span{a_2, a_3} with the orthogonal complement of e_z'.
             If all three moments are degenerate, Q = Id. If a projection vanishes, take
             the eigenvector of least index in that subspace.
             The rows of the rotation matrix Q are (e_x', e_y', e_z'), and det Q = +1.
             Then x <- Q x and c_k <- Q c_k. The kernel centres must rotate with the mesh;
             otherwise the physical density field changes.
       (v)   From the new origin, cast a ray along each fixed nodal direction w_i,
             intersect the transformed polyhedron, and resample R'(w_i) (RayBackend).
             One must have n.e_r >= c_0; otherwise reject the step, reduce Delta^(n),
             and repeat.
       (vi)  Scalar barycentric interpolation of the surface density rho_s and of the
             diagnostic eps_p_bar on the classification mesh. Velocity is not carried
             across the step. The next Picard iteration takes u^0 = 0. The spin omega
             is not changed by the realignment, because it is prescribed exogenously.
             In the new frame, Omega = omega e_z'.
       (vii) Recheck |x_cm'| <= eps_frame R_ref and max(|I_xz'|, |I_yz'|) <= eps_frame I_1.
             After this check passes, the rigid-body compatibility conditions of
             Appendix A.3 hold by construction. The resampling error is entered in E_frame.
9. Set t_{n+1} = t_n + lambda Delta t_n. Record R^(n+1), rho_s, the diagnostics, and the
   error budget of Section 9.8. Return to step 1.
```

**Stage equilibrium and micro-creep.** Criterion (c) of Section 2.15 is a strict fixed point. When the movable set is empty, $d_*=0$ holds exactly, independently of $\Delta t_n$ and of $\lambda$. The regularization $\dot\gamma_\varepsilon>0$ leaves a numerical creep of size $|\mathbf v|\sim\dot\gamma_Y R$ inside the rigid core. It is reported only as a diagnostic, $\max_{\mathcal U_{\rm core}}|\mathbf v_{\rm sl}|$. It does not enter the geometry, the step size, the flux, or any state variable. There is therefore no convergence test of “velocity floor” type, and creep causes neither a geometric drift nor a historical drift. As $\dot\gamma_\varepsilon\to 0$, the creep inside the core tends to zero, and the movable set converges to the yielded region and the sliding blocks of the unregularized model (PO-12).

**The dynamic free-surface mode** does not use this loop for its time advance. At each physical step, $R$ is integrated by $\partial_t R+\mathbf v_S\cdot\nabla_S R=v_r$. The velocity is given by the modal-momentum transport and the mass recovery of Chapter 9, and the geometric subproblem acts as a constraint correction. Its error budget is $E_{\rm total}^{\rm dyn}$ of Section 9.8.

The closed loop of the quasi-static route is

$$
\boxed{
\omega(t_n)
\xrightarrow{\text{RTFP forward evaluation}}
(\mathbf g_{\rm eff},\mathbf T_{\rm tide})
\xrightarrow{\text{Stokes–Poisson Picard}}
(\mathbf u,p;\ \boldsymbol\sigma)
\xrightarrow{F_\delta,\ \mathcal U_{\rm core}}
(\chi_Y,\mathbf v_{\rm sl},J)
\xrightarrow{\Delta t_n}
d_{\mathrm{pred}}
\xrightarrow{\mathrm{Metal-PDHG}\ \text{elastic SOCP}}
d_*^{(n)}
\xrightarrow{\text{update}+\text{realignment}}
R^{(n+1)}.
}
$$

The loop contains only one stress field $\boldsymbol\sigma$. The velocity is resolved implicitly at every step. The spin is prescribed exogenously. The geometry changes only on the movable set. The interface $W_0/R_{\mathrm{eq}}$ of Mode A does not enter this loop. Its interface comments must be retained, so that a future explicit fluid simulation can call them. They must not be deleted by mistake in an implementation.

---

# 11. Conclusion

This paper gives a closed loop whose core is the quasi-static viscoplastic Stokes equilibrium of Mode B, with local yield-driven erosion and deposition of Mode C as an auxiliary process. The loop couples the gravity of a Cauchy density, a variable-density poloidal–toroidal double-potential flow, conservative transport on the surface, and an incremental SOCP. Exterior fields are cross-checked by a Feynman parametrization and by the Newtonian representation. Interior and boundary Hessians are defined explicitly by the finite-part and contact conventions. Mechanically there is only one stress field, $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D$. At each geological step, under the exogenously prescribed spin $\omega(t_n)$, the quasi-static Stokes system (8.4′) is solved implicitly in an exactly divergence-free shifted Jacobi–spherical-harmonic double-potential basis. The pressure is given by an independent scalar Poisson problem. The two are closed by a Picard iteration. On that same stress, the yield function $F_\delta$ with its tolerance, together with connectivity of the unyielded region, partitions the body into a rigid core and a movable set. Topography on the core is strictly locked. Only the yielded region and the sliding blocks produce a geometric prediction, from the slip velocity relative to the core. Metal-PDHG corrects only that predicted increment, with the current shape as the reference, and never takes an equipotential as its target. The geological step is limited only by the geometric CFL of the true landslide speed and by the erosional flux. There is no viscous-diffusion stiffness. Track A (a global solve) is the reference. Track B (a spherical-harmonic block-diagonal surrogate) is used only as a preconditioner, or as an approximation that carries an a-posteriori residual. Mode A is retained only as an explicitly marked interface for a future fluid simulation.

The rigour of the model depends on positive pressure, on integrability, on the finite-part convention, on the radial modal truncation, and on constitutive regularization. Numerical results at $K=4$ should be checked for convergence against a higher-order radial basis and against a three-dimensional reference solution. The framework therefore provides an implementable reduced model whose error budget can be built term by term. It is not an unconditional closed-form equivalent of the three-dimensional problem.

# Appendix A. Normative conventions and the register of proof obligations (single source of truth)

This appendix is the normative convention for Chapters 5–10. It replaces a contract file that once stood alone. The words “must” and “must not” are mandatory. Chapters 1–4 are inputs to this appendix and are only cited. Every conclusion marked “theorem” must name the proof obligation (A.5) on which it depends. What has not been proved may be called only a “hypothesis” or a “numerical observation”.

## A.1 Glossary for disambiguation

The basic unknowns of the quasi-static route are $(\mathbf v,p,R)$, together with the surface density $\rho_s$ of Mode C. The density $\rho_K$ is spatially fixed in the body frame and is not advanced by a continuity equation (hypothesis A-ρ, PO-14). The spin $\omega(t)$ is prescribed exogenously (Section 10.4.1).

| Symbol | Meaning | Dimension / convention |
|---|---|---|
| $\mathbf x,\ r,\ \mathbf e_r$ | position in the rotating frame (origin at the centre of mass), radial coordinate, and unit vector | L; $\boldsymbol\Omega=\omega\mathbf e_z$ |
| $\hat{\boldsymbol\omega}=(\theta,\varphi)\in S^2$ | direction (**always written $\hat{\boldsymbol\omega}$**, so that it is not confused with the angular speed $\omega$) | 1 |
| $R(\hat{\boldsymbol\omega},t)$, $\zeta=r/R$ | star-shaped boundary radius, normalized radius | L, 1; $0<R_{\min}\le R\le R_{\max}$ |
| $\mathcal V(R)$, $\Gamma=\partial\mathcal V$, $\mathbf n$ | material domain, free surface, outward normal | $\mathbf n\cdot\mathbf e_r\ge c_0>0$ |
| $\ell,m,k$ | spherical-harmonic degree and order (velocity only for $\ell\ge 1$); radial mode $k=0..3$ ($K=4$) | 1 |
| $\Lambda_\ell=\ell(\ell+1)$, $L^2=-\Delta_{S^2}$ | eigenvalue of the spherical Laplacian, and the squared angular momentum | 1 |
| $\mathbf v,\ p$ | velocity, and pressure $p=-\operatorname{tr}\boldsymbol\sigma/3$ | L T⁻¹; M L⁻¹ T⁻² |
| $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+\boldsymbol\tau_{\rm flow}$ | Cauchy stress (**bold**, to be distinguished from the kernel width $\sigma_k$) | M L⁻¹ T⁻² |
| $\Phi,\Psi$; $q_{\ell km},s_{\ell km}$ | poloidal and toroidal potentials, and their modal coefficients | L² T⁻¹, L T⁻¹; coefficients T⁻¹ |
| $\mathbf u=(\mathbf q,\mathbf s)$ | vector of velocity degrees of freedom, $N_u=8N_{\ell m}$, $N_{\ell m}=\#\{(\ell,m):1\le\ell\le L_{\max}\}$ | T⁻¹ |
| $\mathbf M_\Gamma$, $\mathbf m=\mathbf M_\Gamma\mathbf u$ | global kinetic-energy mass operator, and the conserved modal variable (dynamic mode only) | $\sim\rho R^5$ |
| $\mathbf K(\eta)$, $\mathbf F$ | quasi-static viscous stiffness and load (8.4′) | $\sim\eta R^3$; $\sim\rho g R^4$ |
| $\eta,\ \tilde p$ | regularized viscosity, and the unique regularized pressure $\max([p]_+,p_{\min})$ | M L⁻¹ T⁻¹; M L⁻¹ T⁻² |
| $\tau_y,\ F,\ F_\delta,\ \delta_Y$ | static strength $c+\mu_y\tilde p$, yield function, yield function with tolerance, and the tolerance (Section 2.8, B2) | M L⁻¹ T⁻²; $\delta_Y$ dimensionless |
| $\mathcal U_{\rm core}$, $\chi_{Y,i}$, $\mathcal L,\mathcal D,\mathcal A$ | rigid core, movability indicator, and the locked, pure-deposition, and movable nodal sets (Section 2.8, B3–B4) | — |
| $\mathbf v_{\rm sl}$ | slip velocity relative to the rigid core, $\mathbf v-\mathbf a_c-\boldsymbol\theta_c\times\mathbf x$ | L T⁻¹ |
| $\omega(t)$, $\omega_n$ | exogenously prescribed spin table, and the value frozen at step $n$ (Section 10.4.1) | T⁻¹ |

Time notation: $t_n$ is geological time, $\Delta t_n$ is the geological step (Section 10.4.1), and $t_{n+1}=t_n+\lambda\Delta t_n$. The Picard index $k$ does not represent time. This paper does not use a pseudo-time, a step $\Delta\tau$, or a physical relaxation step $\Delta t_{\rm phys}$. The under-relaxation factor $\lambda$ (a scalar) is distinct from the rigid-body multiplier $\boldsymbol\lambda\in\mathbb R^6$ (bold). The vectors $\mathbf x_{\rm cm}$ and $\mathbf Q$ are the centre of mass and the rotation of the frame realignment (step 8b). This paper defines no elastic modulus, no hardening variable, and no return-mapping stress.

Mandatory disambiguation. The Feynman parameter remains $u$, and only in Chapters 1–2. The degree-of-freedom vector $\mathbf u$ is written only in bold. A displacement increment is $\mathbf d$, with radial components $d_i$. Distinguish the stress $\boldsymbol\sigma$, the kernel width $\sigma_k$, and the growth rate $\lambda_{\rm gr}$. Distinguish the total mass $\mathsf M_{\rm tot}$ from the mass operator $\mathbf M_\Gamma$. Distinguish the domain $\mathcal V$ from the angular-velocity vector $\boldsymbol\Omega$. The kernel scale is $a_k=1/\sigma_k$. A nodal index $a$ is used only for Track B. The grain diameter is $d_p$. Index convention: $\sum_{\ell,m,k}$ runs over $1\le\ell\le L_{\max}$, $|m|\le\ell$, and $0\le k\le 3$. The default implementation uses a real orthonormal spherical-harmonic basis. The selection rules for $m$ in Sections 6.1, 7.3–7.4, and 8.9 are written in the notation of complex spherical harmonics. In a real basis, the corresponding $m$ and $-m$ must be merged into an equivalent two-dimensional real block. The relation $m'=m+M$ must not be treated as an index identity column by column in the real basis. Cartesian indices are $i,j,l$.

## A.2 Specification of the function spaces and the trace spaces

**Geometric hypothesis (mandatory).** One has $R\in W^{1,\infty}(S^2)$, $R\ge R_{\min}>0$, and $\mathbf n\cdot\mathbf e_r\ge c_0>0$ almost everywhere. The domain $\mathcal V$ is therefore a Lipschitz star-shaped domain, and the trace operator, Korn’s inequality, and inequalities of Poincaré type are available.

$$
\mathbb V=H^1(\mathcal V;\mathbb R^3),\qquad
\mathbb Q=L^2(\mathcal V),\qquad
\mathbb V_\sigma=\{\mathbf w\in\mathbb V:\nabla\cdot\mathbf w=0\}.
$$

The trace $\gamma_0:\mathbb V\to H^{1/2}(\Gamma;\mathbb R^3)$ is continuous and surjective. The traction $\mathbf t=\boldsymbol\sigma\mathbf n$ lies in $H^{-1/2}(\Gamma;\mathbb R^3)$, and the pairing is the duality of $H^{-1/2}$ with $H^{1/2}$. The free surface is a natural boundary (by default, $\mathbf t_{\rm ext}=0$ enters the surface term on the right-hand side). On a star-shaped domain, the part of degree $\ell\ge 1$ of a field $\mathbf w\in\mathbb V_\sigma$ is written uniquely as $\mathbf v=\nabla\times\nabla\times(\Phi\mathbf r)+\nabla\times(\Psi\mathbf r)$, with $\mathbf r=r\mathbf e_r$ (Section 5.2). At the continuous level, $p\in\mathbb Q$ has no free constant under a natural traction condition on the whole surface. The discrete level is Section 8.4.

**Function-theoretic hypotheses of the nonlinear constitutive law.** The map $\mu(I)$ is not monotone in $\mathbf D$. This paper **does not claim** existence or uniqueness for the continuous problem. A “solution” on Track A is defined to be a Picard fixed point, at a given $(L_{\max},K=4,\text{quadrature},\dot\gamma_\varepsilon,p_{\min})$, of the quasi-static discrete system (8.4′) together with the pressure Poisson problem of Section 8.7 (Section 10.4.2). Linear solvability at fixed viscosity is guaranteed by Lemma 8.12. Existence of the nonlinear fixed point, and convergence of Picard iteration, belong to PO-1 and PO-11. The fully implicit dynamic benchmark also involves PO-2. The constitutive denominator requires $p_*\ge p_{c,\min}>0$, or an explicit pressure regularization.

**Optional even basis.** If one requires $C^\infty$ smoothness at the centre when $R$ is constant, one may switch to $\psi^{\rm ev}_{\ell k}=\zeta^\ell P_k^{(0,\ell+1/2)}(2\zeta^2-1)$, which is orthogonal for the weight $\zeta^{2\ell+2}$. When $R$ is not constant, every term of the potential contains a direction-dependent factor $R(\hat{\boldsymbol\omega})^{2-n}$. Smoothness at the centre is then decided by that factor and is independent of the parity of the radial basis. The exception $\ell=1$ of Proposition 8.4 is unchanged. The default basis is Definition 8.2. Its radial parameters are fixed as $(0,2\ell+2)$, with argument $2\zeta-1$. One must not use $P_k^{(2\ell+2,0)}(2\zeta-1)$. The symbol $K=4$ means four radial modes for each $(\ell,m)$ and each potential, hence eight velocity degrees of freedom for each $(\ell,m)$. It does not mean “four scalar fields globally”.

## A.3 The rigid-body kernel and the constraint matrix

**The singularity at $\ell=0$ (mandatory).** For a function $f$ that depends only on $r$, one has $\nabla\times(f\mathbf r)=\mathbf 0$. The potentials $\Phi$ and $\Psi$ of degree $\ell=0$ therefore contribute nothing to $\mathbf v$. Velocity degrees of freedom contain only $\ell\ge 1$. The matrices $\mathbf K$, $\mathbf K_\gamma$, and $\mathbf M_\Gamma$ are assembled and inverted only on the subspace $\ell\ge 1$. No implementation may invert a matrix that contains $\ell=0$. The component of degree $\ell=0$ in $R$ is determined by the total-mass constraint $\int_{\mathcal V}\rho\,dV=\mathsf M_{\rm tot}$. It is not advanced by the kinematic equation.

**The rigid-body space.** One has $\mathbf D(\mathbf w)=\mathbf 0$ if and only if $\mathbf w$ is a rigid motion. A translation $\mathbf w=\mathbf e$ corresponds to $\Phi=\tfrac12\mathbf e\cdot\mathbf x$. A rotation $\mathbf w=\boldsymbol\theta\times\mathbf x$ corresponds to $\Psi=\boldsymbol\theta\cdot\mathbf x$, with radial profile $\zeta^1$. When $R$ is constant, these are exactly the six coefficients of indices $\ell=1$, $k=0$. When $R$ is not constant, they need not lie in $\mathbb V_h^\sigma$, and the rank must be checked at each step.

**Constraint matrix (mandatory; the recommended form).** The mass-weighted constraints of zero momentum and zero angular momentum, $\int_{\mathcal V}\rho\mathbf v\,dV=\mathbf 0$ and $\int_{\mathcal V}\rho\,\mathbf x\times\mathbf v\,dV=\mathbf 0$, become $\mathbf B_{\rm rb}\mathbf u=\mathbf 0$, with $\mathbf B_{\rm rb}\in\mathbb R^{6\times N_u}$. The rows are $\int_{\mathcal V}\rho\,\mathbf B_\alpha\,dV$ and $\int_{\mathcal V}\rho\,\mathbf x\times\mathbf B_\alpha\,dV$, and one requires $\operatorname{rank}(\mathbf B_{\rm rb})=6$. The meaning of the multiplier $\boldsymbol\lambda$, and the monitor $E_{\rm rb}$, are those of Section 9.4. An alternative is to fix $q_{1m0}=s_{1m0}=0$ directly. That alternative is allowed only when the two forms are equivalent, and the difference must be reported.

**Compatibility conditions for solvability (mandatory check).** A necessary condition for the quasi-static problem to have a solution is that the external force do no work on the rigid modes:

$$
\int\rho\mathbf g_{\rm eff}\,dV+\oint_\Gamma\mathbf t_{\rm ext}\,dS=\mathbf 0,
\qquad
\int\rho\,\mathbf x\times\mathbf g_{\rm eff}\,dV+\oint_\Gamma\mathbf x\times\mathbf t_{\rm ext}\,dS=\mathbf 0.
$$

The resultant of self-gravity vanishes identically. The centrifugal force requires the centre of mass to lie on the axis of rotation and that axis to be a principal axis of inertia ($I_{xz}=I_{yz}=0$). If the condition fails, one must report “incompatible”. The failure must not be absorbed by the multiplier. On the quasi-static route these two conditions are guaranteed by construction through the **standard frame realignment** of step 8b in Section 10.4.4. After a geometric update, if the mismatch exceeds the trigger threshold, the origin is translated back to the centre of mass and the principal axis of the largest principal moment $I_1$ is rotated onto the spin axis $z$. The centre of mass then lies on the axis of rotation and $I_{xz}=I_{yz}=0$. Spin about the axis of largest principal moment is also the stable rotating state (hypothesis A-S). Every Cauchy kernel centre $\mathbf c_k$ is translated and rotated in the same way. After the realignment one must still recheck the residuals $|\mathbf x_{\rm cm}|\le\varepsilon_{\rm frame}R_{\rm ref}$ and $|I_{xz}|,|I_{yz}|\le\varepsilon_{\rm frame}I_1$. Only if that check fails does one report “incompatible”. The Euler force of an exogenous $\dot\omega$, and the external torque, are omitted as a pair (Section 2.8, B1) and do not break torque compatibility. For every $\ell\ge 1$ one must report the smallest singular value of the global stiffness matrix.

## A.4 Definitions of Track A and Track B, and Theorem A.1

The two tracks are defined for one and the same symmetric positive-definite operator $\mathbf A$. On the quasi-static route, $\mathbf A=\mathbf K_\gamma$ (the augmented viscous stiffness of (8.4′), with right-hand side $\mathbf b=\mathbf F$). In the dynamic mode, $\mathbf A=\mathbf M_\Gamma$ (Section 9.4, with right-hand side $\mathbf b=\mathbf m$). **Track A (the reference):** solve the full operator $\mathbf A$ exactly, up to the iteration tolerance. **Track B (a fast diagonal surrogate; a truncated model):** take $\mathbf A_B:=\operatorname{blockdiag}_{(\ell,m)}\mathbf A_{\ell m,\ell m}$ (one $8\times 8$ principal sub-block for each $(\ell,m)$; for $\mathbf M_\Gamma$ this is $\mathbf M_{B1}$ of Section 9.5), and set $\mathbf u_B=\mathbf A_B^{-1}\mathbf b$. The residual operator is $\mathbf E:=\mathbf A-\mathbf A_B$. On a spherically symmetric geometry, $\mathbf E=\mathbf 0$ and Track B agrees with Track A. Track B means only the B1 block-diagonal surrogate (PO-8). The inverse $\mathbf A_B^{-1}$ is at the same time the PCG preconditioner inside Track A.

**Theorem A.1 (Exact error identity).** If $\mathbf A_B$ and $\mathbf A$ are both invertible, then for the same $\mathbf b$,

$$
\mathbf u_B-\mathbf u_0=\mathbf A_B^{-1}\mathbf E\,\mathbf u_0,\qquad \mathbf u_0=\mathbf A^{-1}\mathbf b.
$$

*Proof.* The identity $\mathbf A_B^{-1}-\mathbf A^{-1}=\mathbf A_B^{-1}(\mathbf A-\mathbf A_B)\mathbf A^{-1}$, applied to $\mathbf b$, gives the result. ∎ Corollary: $\mathbf u_B=\mathbf u_0$ if and only if $\mathbf E\mathbf u_0=\mathbf 0$. Track B is therefore not an exact equivalent. The complete form that includes the rigid-body constraint in the dynamic mode is Theorem 9.4. On the quasi-static route, the a-posteriori quantity of Track B is the residual $\varepsilon_{\rm mom}^{\rm weak}$ computed with the full matrix $\mathbf K$ (Section 10.4.2).

**Mandatory report.** Every run of Track B must give $\varepsilon_B:=\|\mathbf r_B\|/\max\{\|\mathbf b\|,\varepsilon_b\}$, where $\mathbf r_B=\mathbf b-\mathbf A\mathbf u_B$. If $\|\mathbf b\|=0$, then $\varepsilon_B=0$ when $\mathbf r_B=0$; otherwise the step is judged undefined and has failed. The distribution of $\varepsilon_B$ in $\ell$ must be given, together with an estimate of $\mathbf A_B^{-1}-\mathbf A^{-1}$. Track B must not be called “exact”, “equivalent”, or “rigorous”. **Red line.** Track B may be used only when the measured $\varepsilon_B$ lies below the configured threshold, and a convergence conclusion drawn from it must be checked again on Track A (Section 9.5, PO-8). The production reference is Track A. The two tracks share the elastic SOCP geometric correction of Sections 2.13 and 10.4.3. They differ only in the linear solve.

## A.5 Register of proof obligations (PO)

| Number | Content | Status |
|---|---|---|
| PO-1 | Local solvability, under continuation in $\dot\gamma_\varepsilon$, of the discrete nonlinear system (8.4′) together with Section 8.7, including $p_*\ge p_{c,\min}$. The linear subproblem at fixed $\eta$ is closed by Lemma 8.12 | linear subproblem closed; the nonlinear problem is not proved |
| PO-2 | The inf–sup condition, and the concrete matching of $\mathbb V_h^g$ with $\mathbb Q_h$. This concerns only the fully implicit dynamic benchmark. The quasi-static route has an exactly divergence-free velocity space and does not need it | not proved |
| PO-3 | Compatibility of the basis at the centre, $\mathbf v\in H^1$, including $\ell=1$ | closed: Proposition 8.4 |
| PO-4 | Hypothesis A-M (transport of $\mathbf m$ along the surface), a bound on the model error of geometric remapping, and the complete expansion of the source. Dynamic mode only; the quasi-static route does not transport $\mathbf m$ | not proved |
| PO-5 | Explicit modal formulae for the free-surface shear rate in the presence of $\Psi$, and for the two tangential traction constraints | conditionally closed: Propositions 10.2 and 10.3 (a consistency residual $\delta_{na}$ is allowed) |
| PO-6 | Block entries, bandwidth, and condition numbers of $\mathbf M_\Gamma$ in the Jacobi basis | conditionally closed: Proposition 10.1 (fixed reference sphere only) |
| PO-7 | Assembly formulae for every term of the velocity equation in the basis | conditionally closed: Propositions 6.2, 6.4, 7.3, and 7.4 (Coriolis block only for a spherically symmetric radial density) |
| PO-8 | The abolished B2 (a nodally local model that freezes $\Lambda_\star$: numerical observation shows that the spectral radius of its iteration matrix $\mathbf I-\widetilde{\mathbf M}_{B2}^{-1}\mathbf M_\Gamma$ is much larger than $1$ on high modes, so the iteration diverges or converges spuriously). Track B means only the B1 block-diagonal surrogate. What remains unproved is the a-priori dependence of $\|\mathbf M_B^{-1/2}\mathbf E\mathbf M_B^{-1/2}\|$ on the degree of asphericity, and an a-priori upper bound for $\varepsilon_B$ | B2 abolished; the a-priori bound for B1 is not proved |
| PO-9 | Consistency and stability, as $\Delta t_n\to 0$, of the geological geometric update, which is an RTFP–Stokes–SOCP sequence explicit in the geometry; an upper bound for the SOCP projection departure $E_{\rm proj}$ | not proved (depends on the CFL restriction, on under-relaxation, and on the trust region) |
| PO-10 | Short-wave ill-posedness of $\mu(I)$ viscoplasticity at high inertial number (Proposition 5.8), and the strain localization and mesh dependence of that ill-posedness under the regularizations $\dot\gamma_\varepsilon$ and $p_{\min}$ | not proved |
| PO-11 | Convergence of the Picard iteration that couples the pressure Poisson problem (Section 8.7) to the quasi-static Stokes problem (8.4′); stability of $\nabla\cdot\boldsymbol\tau_{\rm flow}$ with respect to the frozen pair $(\mathbf u^k,\tilde p^k)$ | not proved (failure to converge is judged by criterion (a) of Section 2.15) |
| PO-12 | Convergence, as $\dot\gamma_\varepsilon\to 0$, of the regularized yield surface ($F_\delta=0$ at fixed $\delta_Y$) and of the rigid core and the movable set, toward the yielded region and the sliding blocks of the unregularized Bingham or $\mu(I)$ model; sufficiency of the step-size rule in Section 10.4.1; conditions under which, at constant $\omega_j$, the movable set becomes empty in finitely many steps (stage equilibrium is attainable) | not proved (if it is not attained, report “stage not equilibrated”, criterion (c) of Section 2.15) |
| PO-13 | A bound $E_{\rm frame}$ on the resampling error of the frame realignment, and preservation of star-shapedness about the new origin. Evolution of the spin is not a proof obligation: $\omega(t)$ is prescribed exogenously, and spin about a principal axis is the explicit model hypothesis A-S (Section 10.4.1) | partially closed (the error bound is not proved) |
| PO-14 | Hypothesis A-ρ: $\rho_K$ is spatially fixed in the body frame and is not transported by a continuity equation; a bound on the error in the mass distribution when landslide material crosses a density gradient | a model hypothesis; reported at each step through $s_{M*}$, the range of $\rho_{\rm bulk}/\rho_K\rvert_\Gamma$, and the mass residual |

---

# References

[1] Forterre, Y., & Pouliquen, O. (2008). An experimental study of quasi-static granular flows. *Journal of Fluid Mechanics*, 616, 163–188. https://doi.org/10.1017/S0022112008000491

[2] Gidaspow, D. (1994). *Multiphase Flow and Fluidization: Continuum and Kinetic Theory Descriptions*. Academic Press.

[3] Greengard, L., & Rokhlin, V. (1987). A fast algorithm for particle simulations. *Journal of Computational Physics*, 73(2), 325–348. https://doi.org/10.1016/0021-9991(87)90140-9

[4] Staniforth, A., & Côté, J. (1991). Semi-Lagrangian integration schemes for atmospheric models—A review. *Monthly Weather Review*, 119(9), 2206–2223. https://doi.org/10.1175/1520-0493(1991)119<2206:SLISFA>2.0.CO;2

[5] Hirata, N., Abe, S., Ueno, M., Barnouin-Jha, O. S., Gaskell, R. W., & Miyamoto, H. (2009). Pole and global shape of 25143 Itokawa. *Icarus*, 200(2), 480–497. https://doi.org/10.1016/j.icarus.2008.10.018

[6] Watanabe, S., Hirabayashi, M., Hirata, N., et al. (2019). Hayabusa2 arrives at the carbonaceous asteroid 162173 Ryugu—A spinning top–shaped rubble pile. *Science*, 364(6437), 268–272. https://doi.org/10.1126/science.aav8032

[7] Miyamoto, H., Yano, H., Scheeres, D. J., Abe, S., Barnouin-Jha, O. S., Cheng, A. F., Demura, H., Gaskell, R. W., Hirata, N., et al. (2007). Regolith migration and sorting on asteroid Itokawa. *Science*, 316(5827), 1011–1014. https://doi.org/10.1126/science.1134390
