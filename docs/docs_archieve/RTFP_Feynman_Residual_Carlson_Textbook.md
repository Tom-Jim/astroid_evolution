# 多面体广义 Cauchy 密度的牛顿引力与 Hessian

## 阅读规则

**预期解决的目标。** 对同一个模型建立两条严格的积分表示，给出原残差方法的准确作用范围，并把有限多项式近似作为另一个带误差的求值方案。

本文使用如下逻辑顺序：先固定几何、符号与边界点值约定；主线 A 用以场点为中心的射线和球形排异定义有限部分；主线 B 用 Feynman 参数化把势核变成移动中心的径向核。两条主线是同一势的不同表示，不能相加。对一维子积分使用 Carlson、Appell 或 Theta 时，必须保留尚未计算的外层积分。

所有数学等式均在相邻文字列出的假设下成立。“解析表示”只意味着无变换误差；“严格数值误差界”还要求截断、求积和舍入误差分别得到可计算上界。

**本节实际解决的总结。** 后文的数学对象、算法通道和误差声明有了统一的解释。

## 第一节　几何、密度和势

**预期解决的目标。** 让射线交段、散度定理和局部切锥同时具备明确的适用条件。

令 $V\subset\mathbb R^3$ 是一个有界多面体实体。假设 $V=\overline{\operatorname{int}V}$，边界由有限个非退化平面三角面片组成；面片定向一致，无自交；允许有限个连通分量以及内部空腔。对内部定向的空腔边界，应使用作为实体边界的外法向量。若输入网格不能确定一个这样的实体，则下文关于体积域的定理不直接适用。

令 $K$ 为正整数。对每个 $k\in\{1,\ldots,K\}$，定义 $w_k\in\mathbb R$、$\sigma_k>0$、$\alpha_k>0$ 和 $\mathbf c_k\in\mathbb R^3$。定义全空间延拓的光滑密度

$$
\rho_K(\mathbf y)
=
\sum_{k=1}^{K}
w_k
\left[
1+\sigma_k^2\|\mathbf y-\mathbf c_k\|^2
\right]^{-\alpha_k}.
$$

若该函数用于物理质量密度，还应另行检查 $\rho_K(\mathbf y)\geq0$ 对几乎所有 $\mathbf y\in V$ 成立；数学恒等式允许带符号权重。

取正势号约定，令

$$
U_K(\mathbf x)
=
G\int_V
\frac{\rho_K(\mathbf y)}
{\|\mathbf x-\mathbf y\|}
\,d^3\mathbf y,
$$

其中 $G>0$，$\mathbf x$ 是场点，$\mathbf y$ 是源点。该势对每个 $\mathbf x\in\mathbb R^3$ 都有限，因为 $V$ 有界、密度有界，而且三维空间中的 $1/\|\mathbf x-\mathbf y\|$ 在 $\mathbf y=\mathbf x$ 附近局部可积。

本文使用

$$
H_{ij}(\mathbf x)
=
\frac{\partial^2 U_K(\mathbf x)}
{\partial x_i\,\partial x_j}
$$

的记号，但只有在经典二阶导数确实存在时才把它称为经典 Hessian。在边棱等位置使用另行定义的 $H_{ij,\ell_0}^{\mathrm{FP}}$。

**本节实际解决的总结。** $V$、$\rho_K$、$U_K$ 及经典 Hessian 的适用范围已经固定。

## 第二节　分布导数与球形排异约定

**预期解决的目标。** 推导 Hessian 的非局部核，并把源点重合处的接触项放回完整公式。

令 $\mathbf r=\mathbf x-\mathbf y$、$R=\|\mathbf r\|>0$。对 $i,j\in\{1,2,3\}$，

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

对球形排异定义的分布，正确的等式为

$$
\frac{\partial^2}{\partial x_j\,\partial x_i}\frac1R
=
\operatorname{p.v.}_{\mathrm{ball}}
\frac{3r_i r_j-R^2\delta_{ij}}{R^5}
-
\frac{4\pi}{3}\delta_{ij}\delta_0(\mathbf r).
$$

从三个对角分量求和得到

$$
\Delta_{\mathbf x}\frac1R
=
-4\pi\delta_0(\mathbf r).
$$

在 $\mathbf x\in\operatorname{int}V$ 时，$\rho_K$ 在 $\mathbf x$ 附近光滑，因此经典 Hessian 与球形排异式满足

$$
H_{ij}(\mathbf x)
=
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
$$

边界上的 $\rho_K\mathbf1_V$ 跳跃，接触分布与指示函数的逐点乘积没有由分布等式唯一确定的数值。以后使用的边界值是一种明确指定的球形排异代表，不自动等于内侧或外侧极限。

**本节实际解决的总结。** 非局部核、球形主值及内部接触项的符号和适用位置已经明确。

## 第三节　局部切锥、对数项和有限部分

**预期解决的目标。** 给内部、面片、棱边和顶点一个统一且可检验的点值约定。

固定场点 $\mathbf x$。有限平面网格保证存在 $r_\ast(\mathbf x)>0$，使 $V\cap B_{r_\ast(\mathbf x)}(\mathbf x)$ 在平移到 $\mathbf x$ 后与一个多面体锥在该球内的交集相同。取 $0<r_0<r_\ast(\mathbf x)$。对外部点额外要求 $0<r_0<\operatorname{dist}(\mathbf x,V)$，且其局部锥为空集。

取射线约定

$$
\mathbf y=\mathbf x-R\mathbf u,
\qquad
R>0,
\qquad
\mathbf u\in\mathbb S^2.
$$

令 $\Gamma_{\mathbf x}$ 是满足 $\mathbf x-R\mathbf u\in V$ 对所有充分小的 $R>0$ 成立的方向集合。其边界方向可忽略球面测度零的差异。定义

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

定义非局部球形截断积分

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

给定正长度 $\ell_0>0$，定义

$$
H_{ij,\ell_0}^{\mathrm{FP}}(\mathbf x)
=
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
$$

下面第五节证明极限存在。若 $\rho_K(\mathbf x)C_{ij}(\mathbf x)\neq0$，相应的未经扣除截断积分有对数发散，因此这个数是指定 $\ell_0$ 和球形排异规范下的有限部分，不是有限的普通截断积分极限。

由 $\sum_{i=1}^3T_{ii}(\mathbf u)=0$ 得

$$
\operatorname{Tr}H_{\ell_0}^{\mathrm{FP}}(\mathbf x)
=
-4\pi G\vartheta_{\mathbf x}\rho_K(\mathbf x).
$$

对 $\ell_0,\ell_1>0$，尺度变换为

$$
H_{ij,\ell_1}^{\mathrm{FP}}(\mathbf x)
-
H_{ij,\ell_0}^{\mathrm{FP}}(\mathbf x)
=
G\rho_K(\mathbf x)C_{ij}(\mathbf x)
\log\frac{\ell_0}{\ell_1}.
$$

整个矩阵与 $\ell_0$ 无关的充要条件是 $\rho_K(\mathbf x)C(\mathbf x)=0$。当 $\rho_K(\mathbf x)\neq0$ 时，才可以简化为 $C(\mathbf x)=0$。内部整球及面片相对内部半球的球面二阶矩均给出 $C(\mathbf x)=0$；棱边和顶点须逐点计算。

**本节实际解决的总结。** 明确区分经典 Hessian、截断积分以及依赖球形规范与 $\ell_0$ 的有限部分。

## 第四节　主线 A：球坐标、交段和距离配方

**预期解决的目标。** 把非局部体积分精确写成球面角向积分与全部射线交段的径向积分。

对 $\mathbf y=\mathbf x-R\mathbf u$，在通常的球坐标 $(R,\theta,\varphi)$ 中，

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

因 $\mathbf x-\mathbf y=R\mathbf u$，

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

定义射线交集

$$
E_{\mathbf x}(\mathbf u)
=
\{R>0:\mathbf x-R\mathbf u\in V\}.
$$

有限多面体性保证对几乎所有方向，此集合是有限个闭区间之并：

$$
E_{\mathbf x}(\mathbf u)
=
\bigcup_{\ell=1}^{L(\mathbf u)}
[R^-_\ell(\mathbf u),R^+_\ell(\mathbf u)].
$$

所有非空区间都必须参与求和，特别是外部点射线的 $\ell=1$ 区间。对于第 $k$ 个核令

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

逐项展开：

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

**本节实际解决的总结。** 全部射线交段有了唯一的几何索引；核的实径向二次式严格为正。

## 第五节　主线 A：近场、远场和有限部分恒等式

**预期解决的目标。** 证明不存在漏段或重复积分，并证明第三节的有限部分极限存在。

按照第三节选定的 $r_0$，对每个方向定义

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

由集合运算直接得到

$$
E_{\mathbf x}(\mathbf u)
=
E_{\mathbf x}^{\mathrm{near}}(\mathbf u)
\mathbin{\dot\cup}
E_{\mathbf x}^{\mathrm{far}}(\mathbf u),
$$

其中公共端点对径向积分贡献为零。对几乎所有 $\mathbf u$，

$$
E_{\mathbf x}^{\mathrm{near}}(\mathbf u)
=
\begin{cases}
(0,r_0),&\mathbf u\in\Gamma_{\mathbf x},\\
\varnothing,&\mathbf u\notin\Gamma_{\mathbf x}.
\end{cases}
$$

设 $0<\varepsilon<r_0$。由第四节的雅可比变换，

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

在近场内加减 $\rho_K(\mathbf x)$：

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

在包含所有线段 $\mathbf x-R\mathbf u$ 且 $0\le R\le r_0$ 的紧集中，存在 $M_1<\infty$ 满足 $\|\nabla\rho_K\|\le M_1$。由微积分基本定理，

$$
\rho_K(\mathbf x-R\mathbf u)-\rho_K(\mathbf x)
=
-R\int_0^1
\nabla\rho_K(\mathbf x-tR\mathbf u)
\cdot\mathbf u
\,dt,
$$

因此

$$
\left|
\frac{
\rho_K(\mathbf x-R\mathbf u)-\rho_K(\mathbf x)
}{R}
\right|
\le M_1.
$$

球面测度有限，因此近场差分项可以令 $\varepsilon\downarrow0$。远场的 $R\ge r_0$、$V$ 有界，故远场积分绝对收敛。

把上述分解代入第三节定义，得到

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

为验证 $r_0$ 的独立性，取另一允许半径 $r_1>r_0$。在同一截断积分中，区间 $[r_0,r_1)$ 从旧远场移入新近场；密度差分项与常数项在该区间逐点相加，等于原密度项。故总和不变。

对于外部场点，可取 $r_0<\operatorname{dist}(\mathbf x,V)$，于是 $\Gamma_{\mathbf x}=\varnothing$，全部交段留在远场，所得等式就是经典 Hessian。

**本节实际解决的总结。** 固定切锥近场和完整远场严格互补；第三节的有限部分极限存在；外部第一射线交段不会遗漏。

## 第六节　指数一径向原函数与零端点

**预期解决的目标。** 在 $\alpha_k=1$ 时给出完整原函数、反向求导和有限部分下限。

固定核编号 $k$ 与方向 $\mathbf u$，并暂将 $b_k(\mathbf u)$ 和 $d_k(\mathbf u)$ 写成 $b$ 和 $d$。定义

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

由 $D'(R)=2\sigma_k^2(R-b)$，作部分分式：

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

取 $R>0$ 的连续实反正切分支，定义

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

直接求导：

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
D(R)-\sigma_k^2R^2+2\sigma_k^2bR
}{
A_0R D(R)
}\\
={}&
\frac1{R D(R)}.
\end{aligned}
$$

因此对 $0<A<B$，

$$
\int_A^B
\frac{dR}{R D(R)}
=
P(B)-P(A).
$$

对 $L>0$，零端点的径向有限部分定义为

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

因为 $D(0)=A_0$ 且

$$
P(\varepsilon)
=
\frac1{A_0}
\left[
\log\varepsilon
-
\frac12\log A_0
-
\frac bd\arctan\frac bd
\right]
+
o(1),
$$

所以

$$
\boxed{
J_{\ell_0}^{\mathrm{FP}}(0,L)
=
\frac1{A_0}
\left[
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
\right].
}
$$

对数的量纲由两个长度比和 $D(L)/A_0$ 消去。单条方向上的有限部分只是第五节近场的一种求值方式，不能当作额外加项再次叠加。

**本节实际解决的总结。** 径向原函数和正确的有限部分下限符号都已逐步核验。

## 第七节　主线 B：Feynman 参数化

**预期解决的目标。** 把 Newton 因子与广义 Cauchy 因子的两个中心严格熔合成一个移动中心，并保留正确的尺度因子。

固定一个核通道，暂省下标 $k$。令

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

则

$$
\left[
1+\sigma^2\|\mathbf y-\mathbf c\|^2
\right]^{-\alpha}
=
\sigma^{-2\alpha}B(\mathbf y)^{-\alpha}.
$$

对于 $\alpha>0$、$A(\mathbf y)>0$ 和 $B(\mathbf y)>0$，Gamma 积分给出

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

令

$$
s=u\lambda,
\qquad
t=(1-u)\lambda,
\qquad
0<u<1,
\qquad
\lambda>0.
$$

Jacobian 为

$$
\left|
\frac{\partial(s,t)}{\partial(u,\lambda)}
\right|
=
\lambda.
$$

于是

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

对 $\lambda$ 使用 Gamma 积分，

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

故

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

定义

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

展开二次型：

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

加入 $(1-u)a^2$ 后，

$$
\boxed{
uA(\mathbf y)+(1-u)B(\mathbf y)
=
\|\mathbf y-\mathbf z(u)\|^2+m^2(u).
}
$$

因此 $\mathbf z(u)$ 沿着 $\mathbf c$ 到 $\mathbf x$ 的线段移动。对 $0\le u<1$，

$$
m^2(u)
=
(1-u)
\left[
a^2+u\|\mathbf x-\mathbf c\|^2
\right]
>0.
$$

当 $u\uparrow1$ 时，$\mathbf z(u)\to\mathbf x$ 且 $m(u)\to0$；这正是近场奇异性的参数端点，而不是被 Feynman 变换消灭的奇异性。

**本节实际解决的总结。** 参考表达式中不具量纲的 $u+(1-u)\sigma^2$ 被替换为量纲正确的 $m^2(u)$；Feynman 参数化是严格恒等式，不是近似。

## 第八节　Feynman 势的参数积分、空间降维和 Hessian

**预期解决的目标。** 明确参数化之后还剩哪些积分，并给出外部 Hessian 的可交换求导公式。

令

$$
p_k=\alpha_k+\frac12,
$$

$$
C_{\alpha_k}
=
\frac{\Gamma\left(\alpha_k+\frac12\right)}
{\Gamma(1/2)\Gamma(\alpha_k)}.
$$

定义

$$
S_p(\mathbf z,m;V)
=
\int_V
\left[
\|\mathbf y-\mathbf z\|^2+m^2
\right]^{-p}
\,d^3\mathbf y.
$$

对非负单个核，Tonelli 定理允许交换 $u$ 和 $\mathbf y$ 的积分；有限个带符号权重通道可以逐通道交换后再求和。因此

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

这只是一个一维外参数积分包着一个空间体积分；它没有证明 $S_p$ 已经是闭式。

固定 $u<1$，令

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

因为

$$
\frac{dA_p(r;m)}{dr}
=
\frac{r^2}{(r^2+m^2)^p},
$$

所以

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

故

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

当 $p=3/2$，

$$
A_{3/2}(r;m)
=
\operatorname{arsinh}\left(\frac rm\right)
-
\frac r{\sqrt{r^2+m^2}}.
$$

固定一个平面面片。令 $\boldsymbol\xi$ 是面内坐标，$\boldsymbol\xi_0$ 是 $\mathbf z$ 的面内投影，令

$$
\rho=\|\boldsymbol\xi-\boldsymbol\xi_0\|,
$$

$$
\delta_f
=
(\mathbf y_f-\mathbf z)\cdot\mathbf n_f,
$$

则面内距离满足

$$
\|\mathbf y-\mathbf z\|^2
=
\rho^2+\delta_f^2.
$$

面片积分的被积函数是

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

若定义

$$
h_{p,f}(\rho)
=
\frac1{\rho^2}
\int_0^\rho
s\,g_{p,f}(s)\,ds,
\qquad
\rho>0,
$$

则

$$
\mathbf G_{p,f}(\boldsymbol\xi)
=
h_{p,f}(\rho)
(\boldsymbol\xi-\boldsymbol\xi_0)
$$

满足

$$
\nabla_{\boldsymbol\xi}\cdot\mathbf G_{p,f}
=
g_{p,f}(\rho),
$$

因此

$$
\int_{\Delta_f}g_{p,f}\,dS
=
\oint_{\partial\Delta_f}
\mathbf G_{p,f}\cdot\boldsymbol\nu_f
\,ds.
$$

这一步需要 $h_{p,f}$ 在面片内局部可积、边界迹存在，并且在 $\rho=0$ 处使用原积分定义取极限。它把面片积分写成棱边积分，但没有保证棱边积分是初等函数或 Carlson 积分。

对外部场点，设 $d=\operatorname{dist}(\mathbf x,V)>0$。对

$$
q_k(u,\mathbf y;\mathbf x)
=
u\|\mathbf y-\mathbf x\|^2
+
(1-u)\left[
\|\mathbf y-\mathbf c_k\|^2+a_k^2
\right]
$$

有

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

令 $p_k=\alpha_k+1/2$，则

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

由于 $\|\mathbf x-\mathbf y\|\ge d$，可以在外部邻域使用统一主导函数证明求导与体积分、参数积分交换。内部和边界点不能直接套用这个外部估计。

**本节实际解决的总结。** Feynman 路线严格降到面片；在进一步满足条件时可写成棱边积分；外部 Hessian 的求导公式完整给出。外层 $u$ 积分和几何边界积分仍然存在。

## 第九节　主线 B 的 Carlson 判据

**预期解决的目标。** 规定什么时候棱边积分确实属于 genus $1$ 的代数积分，什么时候只能交给其他通道。

设某个面片到棱边的过程，以及可能的分部积分，已经把一维积分写为

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

其中 $R(t,Y)$ 是关于 $t$ 和 $Y$ 的有理函数，并且

$$
P_4(t)
=
a_4t^4+a_3t^3+a_2t^2+a_1t+a_0,
\qquad
a_4\neq0.
$$

Carlson 通道的充分判据包括以下五项。

第一，四次项确实不退化，即 $a_4\neq0$。

第二，$P_4$ 平方自由：

$$
\gcd(P_4,P_4')=1,
$$

等价地，

$$
\operatorname{Disc}(P_4)\neq0.
$$

第三，积分路径不穿过分支点 $P_4(t)=0$，或者已经明确规定可积端点极限。

第四，$R(t,Y)$ 的极点不位于路径内部；如果极点在路径上，必须单独声明普通积分、Cauchy 主值或 Hadamard 有限部分。

第五，沿路径连续选择平方根 $Y(t)$。逐点调用主值平方根不能保证路径上的分支连续。

在这些条件下，代数曲线

$$
\mathcal C:\quad Y^2=P_4(t)
$$

的光滑射影紧化是 genus $1$ 曲线。这个结论只适用于已经被证明为上述一维微分的子积分，不适用于原始球面二维积分的自动整体闭式。

### 9.1 Hermite 约化

**预期解决的目标。** 把含高阶极点和高次分子的一维微分化为有限个标准 Abel 微分。

要求求出代数函数 $Q(t,Y)$ 和有限组基微分 $\omega_\nu$，使

$$
\frac{R(t,Y)}{Y}\,dt
=
dQ(t,Y)
+
\sum_{\nu=1}^{N}
c_\nu\omega_\nu.
$$

因为

$$
2Y\,dY=P_4'(t)\,dt,
$$

$dQ$ 可化为 $dt$ 与 $dt/Y$ 的有理组合。

不含路径极点时，常用基微分包括

$$
\omega_0=\frac{dt}{Y},
$$

$$
\omega_1=\frac{t\,dt}{Y},
$$

以及每个不等于分支点的简单极点 $a$ 对应的第三类微分

$$
\omega_a=\frac{dt}{(t-a)Y}.
$$

若原式含有 $(t-a)^{-m}$、$m\ge2$，应构造相应的 $dQ$ 消除高阶极点。这个步骤需要实际解代数系数方程，不能只凭积分外形宣称约化已经完成。

### 9.2 四次到三次

**预期解决的目标。** 说明进入 Carlson 三次标准模型所需的代换。

设 $e$ 是 $P_4$ 的一个根。取一个 Möbius 变换

$$
t=\frac{a+bz}{c+dz},
\qquad
ad-bc\neq0,
$$

把一个分支点映到无穷远。代入并清除分母后，定义

$$
W(z)=(c+dz)^2Y(t(z)),
$$

得到

$$
W(z)^2=Q_3(z),
$$

其中 $Q_3$ 是三次多项式。$dt/Y$、$t\,dt/Y$ 和 $dt/[(t-a)Y]$ 都可逐项转换为

$$
\frac{A_0+A_1z}{W(z)}\,dz,
$$

或

$$
\frac{dz}{(z-p)W(z)},
$$

加上已经由 $dQ$ 积出的有理微分。这里的 $A_0,A_1,p$ 由实际端点、分支点和极点计算，不存在与几何输入无关的固定系数。

### 9.3 Carlson 对称积分

**预期解决的目标。** 给出正确的 $R_F,R_D,R_J,R_C$ 定义，并避免把定义、退化关系和数值误差混为一谈。

在实参数且积分路径不碰分母零点时，定义

$$
R_F(x,y,z)
=
\frac12
\int_0^\infty
\frac{ds}
\sqrt{(s+x)(s+y)(s+z)}.
$$

定义

$$
R_D(x,y,z)
=
\frac32
\int_0^\infty
\frac{ds}
(s+z)\sqrt{(s+x)(s+y)(s+z)}.
$$

定义

$$
R_J(x,y,z,p)
=
\frac32
\int_0^\infty
\frac{ds}
(s+p)\sqrt{(s+x)(s+y)(s+z)}.
$$

定义退化形式

$$
R_C(x,y)
=
\frac12
\int_0^\infty
\frac{ds}
(s+y)\sqrt{s+x}.
$$

在分支一致时，直接由 $R_F$ 的定义得到

$$
R_C(x,y)=R_F(x,y,y).
$$

因此这两个写法不是互相矛盾的定义；$R_C$ 是单根退化情形的专用记号。

经过三次曲线变换、有限区间到半无限区间的有理代换以及标准微分分解，第一类、第二类和第三类椭圆子积分可以写成上述 Carlson 对称积分的有限线性组合。组合系数依赖输入根、端点、路径分支和极点位置，必须在每个单元逐项生成。

### 9.4 倍增迭代和误差边界

**预期解决的目标。** 给出 Carlson 计算的核心迭代，同时明确内部停止判据不是严格绝对误差证明。

对 $R_F(x,y,z)$，令

$$
x_0=x,\qquad
y_0=y,\qquad
z_0=z.
$$

第 $n$ 步定义

$$
\lambda_n
=
\sqrt{x_ny_n}
+
\sqrt{y_nz_n}
+
\sqrt{z_nx_n},
$$

并令

$$
x_{n+1}
=
\frac{x_n+\lambda_n}{4},
$$

$$
y_{n+1}
=
\frac{y_n+\lambda_n}{4},
$$

$$
z_{n+1}
=
\frac{z_n+\lambda_n}{4}.
$$

这些量逐步聚拢。利用齐次性

$$
R_F(\lambda x,\lambda y,\lambda z)
=
\lambda^{-1/2}R_F(x,y,z),
\qquad
\lambda>0,
$$

在归一化变量上作终止展开。$R_D$、$R_J$ 和 $R_C$ 需要各自的递推和补偿项。

一个相对离差阈值只能判断迭代是否聚拢。要声称

$$
|R_F^{\mathrm{computed}}-R_F^{\mathrm{exact}}|
\le\varepsilon
$$

还必须给出终止级数余项、复分支选择、输入舍入误差和递推舍入误差的总界。因而本文把 Carlson 停止阈值称为内部误差估计，除非这些界已经在实现中实际计算。

**本节实际解决的总结。** Carlson 是四次平方自由代数曲线一维子积分的条件性求值器；它不是任意非整数幂次、任意二维角向积分或任意残差的总求解器。

## 第十节　残差的严格定义

**预期解决的目标。** 恢复原参考文本的残差思想，但保证残差与解析项严格互补，不遗漏单段外部区间，也不重复同一个完整单元。

对每个球面开单元 $\Omega_m$ 和所有非空远场交段，定义索引

$$
q=(m,\ell,k,i,j),
$$

其中 $m$ 是方向单元编号，$\ell$ 是远场交段编号，$k$ 是核通道，$i,j$ 是 Hessian 分量。每个 $q$ 对应一个完整远场积分

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

近场区间不属于这里的残差集合；它已经在第五节的有限部分恒等式中被分成显式局部对数项、绝对可积差分项和接触项。

定义确定性的分类函数

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

要求

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

于是定义

$$
H_{ij}^{\mathrm{res}}(\mathbf x)
=
\sum_{q\in\mathcal Q_{\mathrm{Appell}}}I_q
+
\sum_{q\in\mathcal Q_{\mathrm{homotopy}}}I_q
+
\sum_{q\in\mathcal Q_{\mathrm{Theta}}}I_q
+
\sum_{q\in\mathcal Q_{\mathrm{direct}}}I_q,
$$

而 Carlson 子集属于解析求值部分：

$$
H_{ij}^{\mathrm{Carlson}}(\mathbf x)
=
\sum_{q\in\mathcal Q_{\mathrm{Carlson}}}I_q.
$$

全局远场满足

$$
H_{ij}^{\mathrm{far}}(\mathbf x)
=
H_{ij}^{\mathrm{Carlson}}(\mathbf x)
+
H_{ij}^{\mathrm{res}}(\mathbf x).
$$

如果需要把某个完整单元逐点拆为已提取函数和剩余函数，必须给出

$$
F_q(\xi,\eta)
=
F_q^{\mathrm{ext}}(\xi,\eta)
+
F_q^{\mathrm{rem}}(\xi,\eta)
\quad\text{几乎处处成立},
$$

并定义

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

否则 $F_q^{\mathrm{rem}}$ 没有数学定义。

残差函数的正则性只能按分片声明：在不含切触、可见性边界、顶点、分支点和路径极点的开单元内部，如果核和端点函数足够光滑，则它是局部 $C^r$；在全局闭域上，通常只能声明分片可积或 $L^1$，不能未经证明称为整个闭域上的 $C^\infty$ 函数。

**本节实际解决的总结。** 残差现在是一个明确的互补求和，不是“全积分减去一个未定义解析项”，也不是四个通道对同一单元的重复相加。

## 第十一节　超几何径向通道：Appell 只是特殊退化

**预期解决的目标。** 说明固定方向径向积分的一般超几何类型，明确 Appell 只在特殊退化时出现，并修正尺度和外层积分问题。

固定 $\mathbf u$，令

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

则

$$
1+\sigma_k^2
\|\mathbf x-R\mathbf u-\mathbf c_k\|^2
=
\sigma_k^2
\left[
(R-b_k(\mathbf u))^2+d_k^2(\mathbf u)
\right].
$$

因此固定方向的径向积分是

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

令

$$
\beta_k(\mathbf u)
=
\frac{b_k(\mathbf u)}{d_k(\mathbf u)}.
$$

再令

$$
t=\frac{R-b_k(\mathbf u)}{d_k(\mathbf u)},
$$

则

$$
R=d_k(\mathbf u)t+b_k(\mathbf u),
$$

$$
dR=d_k(\mathbf u)\,dt,
$$

$$
(R-b_k(\mathbf u))^2+d_k^2(\mathbf u)
=
d_k^2(\mathbf u)(1+t^2).
$$

从而

$$
\mathcal R_{\alpha_k}(A,B;\mathbf u)
=
\sigma_k^{-2\alpha_k}
d_k(\mathbf u)^{-2\alpha_k}
\int_{t(A)}^{t(B)}
\frac{dt}
{\left[t+\beta_k(\mathbf u)\right](1+t^2)^{\alpha_k}}.
$$

把 $(1+t^2)^{-\alpha_k}$ 分解为

$$
(1+t^2)^{-\alpha_k}
=
(1-it)^{-\alpha_k}(1+it)^{-\alpha_k}
$$

后，积分一般含有两个线性幂因子和一个简单极点因子。经线性或分式变量变换，它一般属于多变量 Lauricella $F_D$ 型不完全积分；只有在极点因子、指数或端点发生特殊退化时，才可降为 Appell $F_1$、Gauss 超几何、Beta 函数或初等函数。所有这些表示都需要连续分支和不穿越极点的路径。

对完整方向单元，Appell 只给出

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

因此二维方向积分仍然存在。Appell 通道只能在外层方向积分另有严格求值器或误差证书时，才构成完整单元的可计算方案。

**本节实际解决的总结。** 非整数 $\alpha_k$ 不等于“整个单元由 Appell 完成”；一般径向表达甚至可能需要 Lauricella $F_D$，正确的尺度因子是 $\sigma_k^{-2\alpha_k}d_k^{-2\alpha_k}$，外层角向积分必须保留。

## 第十二节　同伦残差和直接求积

**预期解决的目标。** 恢复 Poincare 同伦方法，同时限制它的几何域和正则性条件。

设一个方向参数域 $D_m\subset\mathbb R^2$ 经过平面坐标参数化，且 $D_m$ 关于原点星形。设残差二形式为

$$
\eta_m
=
F_m(\xi,\eta)
\,d\xi\wedge d\eta.
$$

定义

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

对 $F_m$ 足够光滑的情形，直接计算外微分。令

$$
A(\xi,\eta)
=
\int_0^1sF_m(s\xi,s\eta)\,ds.
$$

则

$$
\mathcal H\eta_m
=
A(\xi,\eta)\xi\,d\eta
-
A(\xi,\eta)\eta\,d\xi.
$$

于是

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

由于

$$
\frac{\partial(A\xi)}{\partial\xi}
+
\frac{\partial(A\eta)}{\partial\eta}
=
2A+\xi\frac{\partial A}{\partial\xi}
+
\eta\frac{\partial A}{\partial\eta},
$$

代入 $A$ 并对 $s$ 求导：

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

因此

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

因为 $s^2F_m(s\xi,s\eta)\to0$ 当 $s\downarrow0$。故

$$
d(\mathcal H\eta_m)
=
F_m(\xi,\eta)
\,d\xi\wedge d\eta.
$$

Stokes 定理给出

$$
\int_{D_m}
F_m(\xi,\eta)\,d\xi\,d\eta
=
\int_{\partial D_m}
\mathcal H\eta_m.
$$

如果 $D_m$ 不是星形域，必须三角剖分成有限个星形子域，并在内部公共边上按相反方向相消。若 $F_m$ 仅局部可积，则需要在删去奇异点后取极限，并证明边界小圆贡献趋于零；不能直接套用光滑版本。

对不满足 Carlson、Appell 或同伦条件的单元，定义直接求积集合 $\mathcal Q_{\mathrm{direct}}$。双指数步长折半

$$
|I_{h/2}-I_h|<\mathrm{tol}
$$

只是后验估计。严格误差需要给出端点奇异阶数、复解析带宽、离散尾项、外层求积误差和舍入误差。

**本节实际解决的总结。** 同伦公式被逐项验证，但它保留一个同伦参数积分；直接求积是合法兜底，不自动带有绝对误差证书。

## 第十三节　高亏格 Theta 通道的条件性位置

**预期解决的目标。** 说明什么时候超椭圆积分进入残差，以及如何给 Theta 级数的截断误差一个可检查的上界。

只有在代数化后实际得到

$$
Y^2=P_n(t),
$$

其中 $P_n$ 是次数 $n\ge5$ 的平方自由多项式，且待积微分确实是该曲线上的 Abel 微分时，才进入高亏格通道。平方自由要求

$$
\gcd(P_n,P_n')=1.
$$

若 $n=2g+1$ 或 $n=2g+2$，光滑射影模型的亏格为

$$
g=\left\lfloor\frac{n-1}{2}\right\rfloor.
$$

非凸几何本身并不推出 $n\ge5$，非整数 $\alpha_k$ 本身也不推出超椭圆曲线。曲线次数必须由实际消元逐项得到。

令

$$
\tau=X+iY,
$$

其中 $X=X^T$、$Y=Y^T>0$，并令

$$
\mathbf z=\mathbf a+i\mathbf b.
$$

定义 genus $g$ 的 Riemann Theta 级数

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

令 $\lambda=\lambda_{\min}(Y)>0$。单项绝对值满足

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

以及

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

令

$$
r_\Theta
=
\frac{\|\mathbf b\|_2}{\lambda}.
$$

当 $q\ge r_\Theta$ 时，

$$
-\pi\lambda q^2+2\pi\|\mathbf b\|_2q
=
\frac{\pi\|\mathbf b\|_2^2}{\lambda}
-
\pi\lambda(q-r_\Theta)^2.
$$

无穷范数壳层

$$
\{\mathbf n\in\mathbb Z^g:\|\mathbf n\|_\infty=q\}
$$

的点数为

$$
N_g(q)
=(2q+1)^g-(2q-1)^g.
$$

若 $N+1\ge r_\Theta$，则截断区域 $\|\mathbf n\|_\infty\le N$ 之外的尾项满足

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

因此若计算出某个 $N$ 使

$$
B_g(N)\le\varepsilon_\Theta,
$$

只能声明级数截断误差不超过 $\varepsilon_\Theta$。周期矩阵、Abel 映射、Theta 导数、代数约化和浮点运算的误差仍须单独估计。

**本节实际解决的总结。** Theta 是特定平方自由高亏格 Abel 子积分的条件性通道；尾项界被明确，非凸性和非整数指数不再被误写成自动高亏格证据。

## 第十四节　多项式密度辅助路线

**预期解决的目标。** 独立判断有限阶多项式拟合是否能把原始 Cauchy 模型变成严格闭式。

在紧集 $V$ 上，$\rho_k$ 是实解析函数。对固定次数 $p$，可以构造

$$
\rho_{k,p}(\mathbf y)
=
\sum_{|\boldsymbol\beta|\le p}
c_{k,\boldsymbol\beta}
(\mathbf y-\mathbf y_\ast)^{\boldsymbol\beta},
$$

其中

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

上式是多项式替代模型，而不是有限次数下的原 Cauchy 函数恒等式。由多面体多项式密度理论，势、梯度和 Hessian 可以逐项通过边界递推写成顶点、面和棱的代数表达；这个结论适用于 $\rho_{k,p}$，不自动适用于 $\rho_k$。

定义替代 Hessian

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

原目标与替代目标之差为

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

设 $\mathbf x$ 是外部点，且

$$
d=\operatorname{dist}(\mathbf x,V)>0.
$$

因为

$$
\left|
3(x_i-y_i)(x_j-y_j)
-
\|\mathbf x-\mathbf y\|^2\delta_{ij}
\right|
\le
4\|\mathbf x-\mathbf y\|^2,
$$

所以

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

若

$$
\sup_{\mathbf y\in V}
\left|
\rho_k(\mathbf y)-\rho_{k,p}(\mathbf y)
\right|
\le
\varepsilon_{\rho,k,p},
$$

则

$$
\boxed{
|E_{ij,k}^{(p)}(\mathbf x)|
\le
\frac{4G\,\operatorname{Vol}(V)}{d^3}
\varepsilon_{\rho,k,p}.
}
$$

该界对外部点有效。内部点、边界点和有限部分 Hessian 的核在 $\mathbf y=\mathbf x$ 附近不具有相同的统一上界，必须先分离局部奇异项，再估计密度逼近误差。

二阶或三阶拟合不能在没有 $\sigma_k$、核中心位置、包围盒尺度和误差范数的情况下普遍承诺 $10^{-5}$ 或 $10^{-7}$。这条路线的严格写法是

$$
H_{ij,k}(\mathbf x)
=
H_{ij,k}^{(p)}(\mathbf x)
+
E_{ij,k}^{(p)}(\mathbf x),
$$

并同时报告 $E_{ij,k}^{(p)}$ 的可验证上界。若令 $p\to\infty$，还需证明逼近序列在适当范数中收敛、引力积分算子连续，并允许极限与积分交换。

**本节实际解决的总结。** 多项式辅助路线有价值，但它是带模型误差的解析近似；“零数值积分误差”只属于替代的多项式模型，不属于原始有限阶 Cauchy 拟合。

## 第十五节　自检清单与不能声称的结论

**预期解决的目标。** 在生成正式教材文件之前，对逻辑闭环、符号、边界和误差声明做最后核查。

第一，验证几何输入确实定义了一个有限、多面体、定向一致且无自交的体积域。

第二，验证所有射线交段均列入 $E_{\mathbf x}(\mathbf u)$，并且近场与远场满足

$$
E_{\mathbf x}(\mathbf u)
=
E_{\mathbf x}^{\mathrm{near}}(\mathbf u)
\mathbin{\dot\cup}
E_{\mathbf x}^{\mathrm{far}}(\mathbf u).
$$

第三，验证残差分类满足互不相交并集；同一个完整单元不得同时被 Carlson、Appell、同伦和直接求积相加。

第四，验证任何逐点残差都满足

$$
F_q=F_q^{\mathrm{ext}}+F_q^{\mathrm{rem}}
$$

的明确定义；否则使用互补单元集合而不使用形式减法。

第五，验证 $\alpha=1$ 的有限部分使用

$$
\lim_{\varepsilon\downarrow0}
\left[
P(L)-P(\varepsilon)
-
\frac1{A_0}\log\frac{\ell_0}{\varepsilon}
\right].
$$

第六，验证 Carlson 单元满足平方自由四次曲线、路径分支、极点和 Hermite 约化条件。

第七，验证 Appell 只计算固定方向的径向积分，外层球面积分仍被处理。

第八，验证全局残差只被声明为分片局部正则和可积，除非另有证明，不声称其在整个闭域上光滑。

第九，验证数值误差预算包含特殊函数截断、外层和内层求积、端点处理以及舍入误差。

第十，验证多项式路线明确报告 $E_{ij,k}^{(p)}$，不把有限次替代称作原函数恒等式。

因此当前可以严格保留的结论是：RTFP 有球形有限部分约定下的互补射线恒等式；Feynman 参数化有严格移动中心恒等式；固定参数空间积分可严格降到面片；在额外代数判据满足时，一维棱边子积分可进入 Carlson；其余单元可以按互斥残差通道求值。

不能保留的结论是：任意广义 Cauchy 密度、任意非凸多面体和任意边界点都自动拥有有限个初等函数与 Carlson 函数组成的全局闭式；所有残差都自动为零；任何固定节点数或步长折半都自动给出严格绝对误差；有限阶多项式拟合就是原始 Cauchy 模型。

**本节实际解决的总结。** 本教材已经把原残差分析、主线 B Carlson 条件和多项式辅助路线放在同一套互补定义下；最终结论严格限定在已列出的几何、分支、正则化和误差条件内。
