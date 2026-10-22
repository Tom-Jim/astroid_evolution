# 基于 Cauchy 密度、RTFP 引力与半拉格朗日输运的旋转颗粒体 SCF 模型

## 摘要

本文建立一个用于旋转小天体颗粒体的统一数学与计算框架。模型以空间变 Cauchy 密度描述内部质量分布，以 Feynman 参数化统一势、梯度和 Hessian 的积分表示，以 RTFP/射线几何处理复杂自由边界，并将变密度极向—环向双势流、$\mu(I)$ 型流变、表面半拉格朗日输运和摩擦约束几何投影耦合到同一 SCF（self-consistent field）迭代中。文中明确区分外部经典导数、内部有限部分和表面接触项，给出联合流变模型退化到经典 $\mu(I)$ 的充分条件，并说明有限径向 Galerkin 截断、GPU 射线求交和顺序凸二次目标 QP/SOCP 的误差与适用范围。

生产默认解释为模式 B 的准静态黏塑性颗粒体：系统只有一个应力场 $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D$（Bingham/Papanastasiou 正则化的 $\mu(I)$ 黏塑性），每个地质步在给定自转下隐式求解准静态 Stokes 蠕变平衡；在该应力上按屈服函数 $F$ 判定屈服，与未屈服刚性核相连的固体严格锁死（$d_i=0$），只有屈服区及被屈服带隔开的滑块按相对核的蠕变速度产生滑坡位移，Clarabel 以当前形状为基准只校正该力学预测增量，从不以等势面为目标。自转角速度 $\omega(t)$ 是外生给定的分阶段常数或时间函数，不与内部流动做角动量反馈。模式 A 的流体质量能级/等势面仅保留为显式标记的未来接口；它不参与本文默认颗粒体闭环。

本文的“严格”只表示：在明确列出的连续性、正则性、离散、边界和本构假设下，公式与验收条件有确定含义。它不表示任意材料、任意网格或任意有限精度计算都自动得到绝对误差界。任何未列出本构参数或边界数据的算例都必须判为“模型未闭合”，不得输出通过状态。

## 研究问题

在给定旋转率、质量、Cauchy 密度参数、颗粒状态变量和摩擦本构的条件下，如何构造一个数学严格、守恒可检查、可由 GPU 实现的闭环算法，使得

1. 复杂自由边界内的 $U,\nabla U,\nabla^2U$ 具有统一且可验证的积分表示；
2. 半拉格朗日表面输运产生的径向模态变化能够在 A-M 模型假设与恢复算子协调下、完整 $D'$ 定义下条件性传递到表面剪切率、惯性数和节点摩擦角；采用径向差分时明确标记为局部近似；
3. 以模式 B 的准静态黏塑性 Stokes 平衡（单一应力场）与基于屈服函数的刚性核/可动集分类为核心，以模式 C 的局部屈服侵蚀/沉积为辅助，通过冻结场的顺序凸子问题在保持当前形状的基准上校正力学预测增量；自转由外生给定；模式 A 的流体等势求解仅保留接口，不参与颗粒体闭环；
4. 所有降阶、正则化和退化结论都明确给出假设、误差来源与失效边界。

## 符号与基本假设

除非另有说明，$V$ 为有界多面体实体，$\rho_K$ 为非负质量密度（在所考虑的物理生产配置下，假定空间 Cauchy 密度满足全局严格正下界 $\rho_K(\mathbf x)\ge\rho_{\min}>0$，保证动能质量算子 $\mathbf M_\Gamma$ 在 $\ell\ge1$ 上对称正定），$G$ 为引力常数，$\omega$ 为绕体坐标 $z$ 轴（最大惯量主轴，第 10.4.4 节步骤 8）的外生给定角速度（每个地质步内冻结为常数 $\omega_n=\omega(t_n)$，第 10.4.1 节），$W=U+U_c$ 为有效势。星形性只在径向 Galerkin 和自由表面参数化章节中作为额外假设。外部点导数按经典意义理解；内部点和边界点的 Hessian 按球形有限部分规范定义。颗粒压力满足 $p>0$ 时使用惯性流变；$p=0$ 或 $\dot\gamma=0$ 时必须使用相应的正则化或静态屈服分支。

本文对 $U$、$U_c$ 和 $W$ 采用正势约定，保守有效加速度定义为
$g_{\mathrm{eff}}=\nabla W$。若在不可压缩流体的标准写法中出现
$-\rho\nabla\Phi_{\mathrm{eff}}$，则约定 $\Phi_{\mathrm{eff}}=-W$（或更一般地，
$-\nabla\Phi_{\mathrm{eff}}$ 就是所采用的保守加速度）；整个文档必须保持这一对应关系。

# 1. 多面体广义 Cauchy 密度的引力、Hessian 与残差基础

## 导言：方法范围与逻辑

**本节目标：** 对同一个模型建立两条严格的积分表示，给出原残差方法的准确作用范围，并把有限多项式近似作为另一个带误差的求值方案。

本文使用如下逻辑顺序：先固定几何、符号与边界点值约定；主线 A 用以场点为中心的射线和球形排异定义有限部分；主线 B 用 Feynman 参数化把势核变成移动中心的径向核。两条主线是同一势的不同表示，不能相加。对一维子积分使用 Carlson、Appell 或 Theta 时，必须保留尚未计算的外层积分。

所有数学等式均在相邻文字列出的假设下成立。“解析表示”只意味着无变换误差；“严格数值误差界”还要求截断、求积和舍入误差分别得到可计算上界。

**本节结论：** 后文的数学对象、算法通道和误差声明有了统一的解释。

## 1.1 几何、密度和势

**本节目标：** 让射线交段、散度定理和局部切锥同时具备明确的适用条件。

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

**本节结论：** $V$、$\rho_K$、$U_K$ 及经典 Hessian 的适用范围已经固定。

## 1.2 分布导数与球形排异约定

**本节目标：** 推导 Hessian 的非局部核，并把源点重合处的接触项放回完整公式。

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

**本节结论：** 非局部核、球形主值及内部接触项的符号和适用位置已经明确。

## 1.3 局部切锥、对数项和有限部分

**本节目标：** 给内部、面片、棱边和顶点一个统一且可检验的点值约定。

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

**本节结论：** 明确区分经典 Hessian、截断积分以及依赖球形规范与 $\ell_0$ 的有限部分。

## 1.4 主线 A：球坐标、交段和距离配方

**本节目标：** 把非局部体积分精确写成球面角向积分与全部射线交段的径向积分。

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

这里的“闭区间”按 $R>0$ 的相对拓扑理解；若场点在实体内部，首段实际为 $(0,R_1]$，实现时可用其闭包 $[0,R_1]$，因为端点 $R=0$ 对势和梯度的普通径向积分没有体积贡献，而对 Hessian 必须按第 1.2--1.3 节的有限部分规则处理。

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

**本节结论：** 全部射线交段有了唯一的几何索引；核的实径向二次式严格为正。

## 1.5 主线 A：近场、远场和有限部分恒等式

**本节目标：** 证明不存在漏段或重复积分，并证明第三节的有限部分极限存在。

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

**本节结论：** 固定切锥近场和完整远场严格互补；第三节的有限部分极限存在；外部第一射线交段不会遗漏。

## 1.6 指数一径向原函数与零端点

**本节目标：** 在 $\alpha_k=1$ 时给出完整原函数、反向求导和有限部分下限。

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

**本节结论：** 径向原函数和正确的有限部分下限符号都已逐步核验。

## 1.7 主线 B：Feynman 参数化

**本节目标：** 把 Newton 因子与广义 Cauchy 因子的两个中心严格熔合成一个移动中心，并保留正确的尺度因子。

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

**本节结论：** 参考表达式中不具量纲的 $u+(1-u)\sigma^2$ 被替换为量纲正确的 $m^2(u)$；Feynman 参数化是严格恒等式，不是近似。

## 1.8 Feynman 势的参数积分、空间降维和 Hessian

**本节目标：** 明确参数化之后还剩哪些积分，并给出外部 Hessian 的可交换求导公式。

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

**本节结论：** Feynman 路线严格降到面片；在进一步满足条件时可写成棱边积分；外部 Hessian 的求导公式完整给出。外层 $u$ 积分和几何边界积分仍然存在。

## 1.9 主线 B 的 Carlson 判据

**本节目标：** 规定什么时候棱边积分确实属于 genus $1$ 的代数积分，什么时候只能交给其他通道。

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

### 1.9.1 Hermite 约化

**本节目标：** 把含高阶极点和高次分子的一维微分化为有限个标准 Abel 微分。

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

### 1.9.2 四次到三次

**本节目标：** 说明进入 Carlson 三次标准模型所需的代换。

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

### 1.9.3 Carlson 对称积分

**本节目标：** 给出正确的 $R_F,R_D,R_J,R_C$ 定义，并避免把定义、退化关系和数值误差混为一谈。

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

### 1.9.4 倍增迭代和误差边界

**本节目标：** 给出 Carlson 计算的核心迭代，同时明确内部停止判据不是严格绝对误差证明。

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

**本节结论：** Carlson 是四次平方自由代数曲线一维子积分的条件性求值器；它不是任意非整数幂次、任意二维角向积分或任意残差的总求解器。

## 1.10 残差的严格定义

**本节目标：** 恢复原参考文本的残差思想，但保证残差与解析项严格互补，不遗漏单段外部区间，也不重复同一个完整单元。

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

**本节结论：** 残差现在是一个明确的互补求和，不是“全积分减去一个未定义解析项”，也不是四个通道对同一单元的重复相加。

## 1.11 超几何径向通道：Appell 只是特殊退化

**本节目标：** 说明固定方向径向积分的一般超几何类型，明确 Appell 只在特殊退化时出现，并修正尺度和外层积分问题。

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

**本节结论：** 非整数 $\alpha_k$ 不等于“整个单元由 Appell 完成”；一般径向表达甚至可能需要 Lauricella $F_D$，正确的尺度因子是 $\sigma_k^{-2\alpha_k}d_k^{-2\alpha_k}$，外层角向积分必须保留。

## 1.12 同伦残差和直接求积

**本节目标：** 恢复 Poincare 同伦方法，同时限制它的几何域和正则性条件。

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

**本节结论：** 同伦公式被逐项验证，但它保留一个同伦参数积分；直接求积是合法兜底，不自动带有绝对误差证书。

## 1.13 高亏格 Theta 通道的条件性位置

**本节目标：** 说明什么时候超椭圆积分进入残差，以及如何给 Theta 级数的截断误差一个可检查的上界。

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

**本节结论：** Theta 是特定平方自由高亏格 Abel 子积分的条件性通道；尾项界被明确，非凸性和非整数指数不再被误写成自动高亏格证据。

## 1.14 多项式密度辅助路线

**本节目标：** 独立判断有限阶多项式拟合是否能把原始 Cauchy 模型变成严格闭式。

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

**本节结论：** 多项式辅助路线有价值，但它是带模型误差的解析近似；“零数值积分误差”只属于替代的多项式模型，不属于原始有限阶 Cauchy 拟合。

## 1.15 自检清单与适用范围

**本节目标：** 在生成正式教材文件之前，对逻辑闭环、符号、边界和误差声明做最后核查。

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

不成立的普遍性断言包括：任意广义 Cauchy 密度、任意非凸多面体和任意边界点都自动拥有有限个初等函数与 Carlson 函数组成的全局闭式；所有残差都自动为零；任何固定节点数或步长折半都自动给出严格绝对误差；有限阶多项式拟合就是原始 Cauchy 模型。

**本节结论：** 本文将残差分析、主线 B Carlson 条件和多项式辅助路线置于同一套互补定义下；最终结论严格限定在已列出的几何、分支、正则化和误差条件内。

# 2. 统一引力积分表示

第 1 章给出了多面体广义 Cauchy 密度的几何约定、分布 Hessian、有限部分、主线 A/主线 B 表示以及残差判据。本章将其中的密度核写成当前 SCF 实现所使用的尺度参数形式，并由三维体积分逐步推导梯度、Hessian 和表面耦合量。两种密度写法完全等价：

$$
\left[1+\sigma_k^2|y-c_k|^2\right]^{-\alpha_k}
=\sigma_k^{-2\alpha_k}\left[|y-c_k|^2+\sigma_k^{-2}\right]^{-\alpha_k}.
$$


## 2.1 三维体积分与射线表示

对第 $k$ 个 Cauchy 核有

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

因此

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

令

$$
y=x-Ru,\qquad u\in S^2,
$$

则

$$
x-y=Ru,\qquad dy=R^2\,dR\,d\Omega.
$$

于是

$$
\frac{x-y}{|x-y|^3}\,dy
=
\frac{Ru}{R^3}R^2\,dR\,d\Omega
=
u\,dR\,d\Omega.
$$

从而

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

其中

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

这个公式与三维体积分完全等价。

---

## 2.2 Feynman 参数化

定义

$$
A(y)=|y-x|^2,
$$

$$
B_k(y)=|y-c_k|^2+a_k^2.
$$

原势的第 $k$ 项为

$$
U_k(x)
=
G w_k\sigma_k^{-2\alpha_k}
\int_V A(y)^{-1/2}B_k(y)^{-\alpha_k}\,dy.
$$

使用 Feynman 恒等式

$$
A^{-1/2}B^{-\alpha}
=
C_\alpha
\int_0^1
u^{-1/2}(1-u)^{\alpha-1}
[uA+(1-u)B]^{-p}\,du,
$$

其中

$$
p=\alpha+\frac12,
$$

$$
C_\alpha
=
\frac{\Gamma(\alpha+\frac12)}
{\Gamma(\frac12)\Gamma(\alpha)}.
$$

因此

$$
U_k(x)
=
G w_k\sigma_k^{-2\alpha_k}C_{\alpha_k}
\int_0^1
u^{-1/2}(1-u)^{\alpha_k-1}
S_{p_k}(x,u)\,du,
$$

其中

$$
p_k=\alpha_k+\frac12
$$

以及

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

## 2.3 移动中心与有效屏蔽尺度

展开 $Q_k$：

$$
Q_k
=
u|y-x|^2
+
(1-u)\left(|y-c_k|^2+a_k^2\right).
$$

定义移动中心

$$
\boxed{
z_k(x,u)=ux+(1-u)c_k.
}
$$

配方得到

$$
\boxed{
Q_k(x,y,u)
=
|y-z_k(x,u)|^2+m_k^2(x,u),
}
$$

其中

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

由于 $a_k>0$，

$$
m_k^2(x,u)>0\qquad(0\le u<1),
$$

但

$$
m_k^2(x,u)\to0\qquad(u\to1).
$$

因此 Feynman 参数化将源点奇异性移动到了参数端点 $u=1$。

---

## 2.4 统一梯度公式

对 $x$ 求导时，先计算 $Q_k$ 的导数。

由

$$
z_k=ux+(1-u)c_k,
$$

有

$$
\nabla_x z_k=uI.
$$

又因为

$$
m_k^2
=
(1-u)\left[a_k^2+u|x-c_k|^2\right],
$$

所以

$$
\nabla_xm_k^2
=
2u(1-u)(x-c_k).
$$

另一方面，

$$
\nabla_x|y-z_k|^2
=
-2u(y-z_k).
$$

两项相加：

$$
\begin{aligned}
\nabla_xQ_k
&=
-2u(y-z_k)
+
2u(1-u)(x-c_k)\\
&=
2u(x-y).
\end{aligned}
$$

因此

$$
\nabla_x Q_k^{-p_k}
=
-p_kQ_k^{-p_k-1}\nabla_xQ_k,
$$

即

$$
\nabla_x Q_k^{-p_k}
=
-2p_k u(x-y)Q_k^{-p_k-1}.
$$

于是统一后的梯度为

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

这里

$$
z_k=z_k(x,u),
\qquad
m_k=m_k(x,u).
$$

该式给出统一的梯度表示。

对外部点 $x\notin\overline V$，可以交换 $x$-微分与积分。对内部或边界点，需要先进行局部奇异性分析，不能直接套用这个交换步骤。

---

## 2.5 统一 Hessian 公式

对梯度 integrand 求导：

$$
\partial_{x_j}
\left[
(x_i-y_i)Q_k^{-p_k-1}
\right].
$$

第一项为

$$
\partial_{x_j}(x_i-y_i)
=
\delta_{ij}.
$$

第二项为

$$
\partial_{x_j}Q_k^{-p_k-1}
=
-(p_k+1)Q_k^{-p_k-2}
\partial_{x_j}Q_k.
$$

因为

$$
\partial_{x_j}Q_k
=
2u(x_j-y_j),
$$

所以

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

代入梯度公式：

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

矩阵形式更简洁：

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

这个公式在外部点是经典 Hessian 的严格表示。

---

## 2.6 与 Newton Hessian 的一致性

当 $x\notin\overline V$ 时，可以对原始势直接求二阶导数：

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

Feynman 公式只是将

$$
|x-y|^{-1}B_k(y)^{-\alpha_k}
$$

改写为参数积分。由于前面的 Feynman 恒等式是逐点恒等式，且外部点有正距离下界，所以可以使用 Fubini 和支配收敛定理交换积分顺序和导数次序。因此：

$$
\boxed{
H_{\mathrm{Feynman}}(x)=H_{\mathrm{Newton}}(x)
\qquad(x\notin\overline V).
}
$$

因此，外部场的 Feynman 表示与 Newton 表示一致。

---

## 2.7 内部点与表面点的有限部分

当 $x\in V$ 或 $x\in\partial V$ 时，$u\to1$ 导致

$$
z_k(x,u)\to x,
\qquad
m_k(x,u)\to0.
$$

此时 Feynman Hessian 公式不能直接理解为普通绝对收敛积分。

必须先引入球形有限部分：

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

然后只有在证明

$$
\lim_{\delta\downarrow0}
\left[
H_{\mathrm{Feynman}}^{(\delta)}(x)
-
\text{同一球形局部反项}
\right]
=
H^{\mathrm{FP}}_{\ell_0}(x)
$$

之后，才能说 Feynman 表示计算出了同一个有限部分。

因此统一后的严格结论是：

$$
\boxed{
\begin{array}{ll}
x\notin\overline V:
&
H_{\mathrm{Feynman}}=H_{\mathrm{RTFP}}=H_{\mathrm{classical}};
\\[4pt]
x\in\operatorname{int}V:
&
H_{\mathrm{Feynman}}\text{ 需采用主值和接触项};
\\[4pt]
x\in\partial V:
&
H_{\mathrm{Feynman}}\text{ 需与指定的球形有限部分匹配};
\\[4pt]
x\text{ 在棱或顶点}:
&
H_{\ell_0}^{\mathrm{FP}}\text{ 依赖局部切锥和参考尺度}.
\end{array}
}
$$

实现时，内部点和边界点的主路径应直接使用第 1.3 节定义的 RTFP
有限部分；Feynman Hessian 只有在同一局部反项、同一 $\ell_0$ 规范已经
完成匹配验证后，才能替代该路径。

---

## 2.8 颗粒平衡方程

以下势场、坡度和质量条件只是颗粒平衡的部分条件：

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

其中

$$
W(x)=U(x)+U_c(x).
$$

统一后的梯度为

$$
\nabla W(x)
=
\nabla U(x)
+
\omega^2(x_1,x_2,0).
$$

统一后的 Hessian 为

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

在表面节点 $x_i$ 上，定义

$$
g_i^{(n)}
=
\nabla U^{(n)}(x_i^{(n)})
+
\omega^2(x_{i1}^{(n)},x_{i2}^{(n)},0).
$$

局部坡度约束为

$$
\boxed{
|g_i^{(n)}\times n_i^{(n)}|^2
\le
\sin^2\phi_i^{(n)}\,|g_i^{(n)}|^2.
}
$$

该归一化坡度约束要求 $|g_i^{(n)}|>0$。若某节点出现 $|g_i^{(n)}|=0$，上式不定义，必须改用未归一化的接触/屈服约束或将该节点标记为退化点，不能在程序中直接除以 $|g_i^{(n)}|^2$。

**模式 B：准静态黏塑性 Stokes 蠕变与物理屈服分类（单一应力场）。** 本文力学主线只有**一个**应力场，即第 5.1 节定义 5.1 的黏塑性流体应力

$$
\boxed{\ \boldsymbol\sigma=-p\,\mathbf I+\boldsymbol\tau,\qquad
\boldsymbol\tau=\mathbf T_{\rm tide}+\boldsymbol\tau_{\rm flow},\qquad
\boldsymbol\tau_{\rm flow}=2\eta(\tilde p,\mathbf D)\,\mathbf D,\qquad
\eta=\frac{c+\mu_d\,\tilde p}{\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}},\ }
$$

其中 $\dot\gamma=\sqrt{2\mathbf D:\mathbf D}$，$\tilde p:=\max([p]_+,p_{\min})$ 是唯一的正则化压力（第 4.13 节强制规则：应力幅值、屈服强度与 $I_c$ 的分母使用同一个 $\tilde p$，$I_c$ 取 $p_*=\tilde p+p_c$），$\mu_d=\mu_d(I_c;\nu_0,Z_0)$ 见第 4.5、10.3 节。若改用 Papanastasiou 正则化 $\eta=(c+\mu_d\tilde p)(1-e^{-\dot\gamma/\dot\gamma_\varepsilon})/\dot\gamma$，必须同步替换第 4.17、5.1、5.3 节中的 $\eta$ 及其导数，不得混用两种正则化。由 $p_{\min}>0$ 与离散速度的有界性，每个离散状态上 $0<\eta_{\rm lo}:=\dfrac{c+\mu_s p_{\min}}{\sqrt{\dot\gamma_{\max}^2+\dot\gamma_\varepsilon^2}}\le\eta\le\eta_{\rm hi}:=\dfrac{c+\mu_2\tilde p_{\max}}{\dot\gamma_\varepsilon}$，故黏性刚度在刚体约束补空间上一致正定（第 8.4 节引理 8.12）。**本文不存在**弹性应变、弹性模量、硬化变量、返回映射应力或独立位移自由度；凡需要应力之处（平衡、屈服、潮汐、牵引、侵蚀通量、验收）一律取上式。

**（B1）准静态 Stokes 平衡。** 第 $n$ 个地质步在冻结几何 $\mathcal V^{(n)}$、冻结 Cauchy 密度 $\rho_K$ 与外生自转 $\omega_n:=\omega(t_n)$（第 10.4.1 节）下求解与时间无关的蠕变平衡

$$
\boxed{\ 
\begin{aligned}
&\nabla\cdot\boldsymbol\tau-\nabla p+\rho_K\,\mathbf g_{\rm eff}(\omega_n)=\mathbf 0,\qquad \nabla\cdot\mathbf v=0\quad\text{in }\mathcal V^{(n)},\\
&\boldsymbol\sigma\mathbf n=\mathbf t_{\rm ext}\quad\text{on }\Gamma^{(n)},\qquad
\int_{\mathcal V}\rho_K\mathbf v\,dV=\mathbf 0,\quad \int_{\mathcal V}\rho_K\,\mathbf x\times\mathbf v\,dV=\mathbf 0,
\end{aligned}\ }
$$

$\mathbf g_{\rm eff}(\omega)=\nabla U+\omega^2\mathbf x_\perp$。惯性 $\rho\partial_t\mathbf v$、对流 $\rho(\mathbf v\cdot\nabla)\mathbf v$ 与 Coriolis $2\rho\boldsymbol\Omega\times\mathbf v$ 全部略去（准静态极限；适用性指标 $\mathrm{Ek}^{-1}:=\max_{\mathcal Y}\rho_K\omega_nR_{\max}^2/\eta$ 在屈服区 $\mathcal Y$ 上逐步报告，超过配置阈值 $\varepsilon_{\rm in}$ 时该步标记“准静态假设失效”）。该问题**不含时间步长**：$\mathbf v$ 是与当前载荷瞬时平衡的蠕变速度场，由第 8.4 节的离散 Stokes 系统（8.4′）、第 8.7 节的压力 Poisson 问题与第 10.4.2 节的 Picard 迭代隐式求出；不存在显式动量松弛，也不存在黏性扩散型 CFL 限制。外生自转率 $\dot\omega$ 的 Euler 惯性力 $-\dot\omega\,\mathbf e_z\times\mathbf x$ 与驱动自转变化的外力矩在体坐标系中成对平衡，其对内部应力的相对贡献为 $\varepsilon_{\rm Euler}:=|\dot\omega|/\omega^2$，二者一并略去并报告 $\varepsilon_{\rm Euler}$（只略去其一会破坏第 A.3 节的力矩相容条件）。

**（B2）屈服函数与正则化屈服判据。** 静态强度与屈服函数取在唯一应力场上：

$$
\tau_y:=c+\mu_y\,\tilde p,\qquad \mu_y:\equiv\mu_s(\nu_0,Z_0)=\lim_{I_c\to0}\mu_d,\qquad
F:=\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}-\tau_y=\|\mathbf s-\mathbf T_{\rm tide}\|_{\rm eq}-c-\mu_y\tilde p,
$$

$\|\mathbf A\|_{\rm eq}=\sqrt{\mathbf A:\mathbf A/2}$，$\mathbf s=\boldsymbol\tau$ 为偏应力（$\operatorname{tr}\boldsymbol\tau_{\rm flow}=0$ 由 $\nabla\cdot\mathbf v=0$，$\operatorname{tr}\mathbf T_{\rm tide}=0$ 由 $\operatorname{dev}$）。$\mu_y$ 取 $\mu_d$ 的静态极限，使未屈服区的黏性律与屈服函数使用同一强度。由 $\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}=\eta\dot\gamma=(c+\mu_d\tilde p)\,r(\dot\gamma/\dot\gamma_\varepsilon)$，$r(x)=x/\sqrt{1+x^2}$（Bingham）或 $r(x)=1-e^{-x}$（Papanastasiou），$0\le r<1$：在准静态区（$\mu_d\to\mu_s$）正则化应力**严格小于**强度，$F>0$ 永不触发。因此屈服判据必须带容差 $\delta_Y\in(0,1)$（配置，缺省 $0.05$）：

$$
\boxed{\ F_\delta:=\|\boldsymbol\tau_{\rm flow}\|_{\rm eq}-(1-\delta_Y)\,\tau_y,\qquad
\text{点 }\mathbf x\text{ 屈服}\iff F_\delta(\mathbf x)>0 .\ }
$$

$\mu_d=\mu_s$ 时 $F_\delta>0\iff\dot\gamma>\dot\gamma_Y$，$\dot\gamma_Y=\dot\gamma_\varepsilon(1-\delta_Y)/\sqrt{2\delta_Y-\delta_Y^2}$（Bingham）或 $\dot\gamma_Y=\dot\gamma_\varepsilon\ln(1/\delta_Y)$（Papanastasiou）；$\delta_Y=0.05$ 时二者均约为 $3\dot\gamma_\varepsilon$。这是**应力（等价地，应变率不变量）判据，不是速度大小判据**：刚体平移/转动（$\mathbf D=0$）无论速度多大都有 $F_\delta=-(1-\delta_Y)\tau_y<0$。正则化解的屈服面在 $\dot\gamma_\varepsilon\to0$（固定 $\delta_Y$）时向未正则化 Bingham/$\mu(I)$ 屈服面的收敛属 PO-12。

**（B3）刚性核与可动集（死区的物理定义）。** 在分类网格 $\mathcal X_Y:=\{(\zeta_q,\hat{\boldsymbol\omega}_i):\ q=1,\dots,N_r+1,\ i=1,\dots,N_S\}$ 上求 $F_\delta$，其中 $\hat{\boldsymbol\omega}_i$ 为表面网格顶点方向，$\zeta_1<\dots<\zeta_{N_r}$ 为径向 Gauss–Jacobi 节点，$\zeta_{N_r+1}=1$ 为表面点；$\mathbf v$ 由模态系数按命题 10.2 与第 5.2 节逐点合成，$p$ 由第 8.7 节压力基逐点合成，二者在任意点有定义。图 $\mathsf G_Y$ 的邻接为：同一射线上的相邻径向点；同一 $\zeta_q$ 上沿表面网格边相邻的两条射线；全部 $\zeta_1$ 点经一个虚拟中心结点相连。记未屈服点集 $\mathcal U=\{F_\delta\le0\}$，在 $\mathsf G_Y|_{\mathcal U}$ 上求连通分量，**刚性核** $\mathcal U_{\rm core}$ 取求积质量 $\sum\rho_K w$ 最大的分量（并列时取含 $\zeta_1$ 层点最多者）；$\mathcal U=\varnothing$ 时 $\mathcal U_{\rm core}=\varnothing$（整体失稳）。表面节点的可动指示为

$$
\boxed{\ \chi_{Y,i}=\begin{cases}0,&(1,\hat{\boldsymbol\omega}_i)\in\mathcal U_{\rm core}\quad\text{（与刚性核相连的固体，锁死）},\\ 1,&\text{否则（自身屈服，或位于被屈服带与核隔开的滑块上）}.\end{cases}\ }
$$

在未正则化极限中，未屈服连通分量作刚体运动；与核相连的固体随核运动，被屈服剪切带隔开的滑块（滑坡块体）相对核整体滑动——后者表面点本身未屈服但必须可动，这正是不能只用表面点 $F_\delta$ 判定的原因。核的刚体运动由最小二乘拟合

$$
(\mathbf a_c,\boldsymbol\theta_c)=\arg\min_{\mathbf a,\boldsymbol\theta}\sum_{\mathbf x\in\mathcal U_{\rm core}}w_{\mathbf x}\rho_K(\mathbf x)\,|\mathbf v(\mathbf x)-\mathbf a-\boldsymbol\theta\times\mathbf x|^2
$$

（$6\times6$ 对称正定法方程；$\mathcal U_{\rm core}$ 退化为共线点集时判“核退化”并取 $\mathbf a_c=\boldsymbol\theta_c=\mathbf 0$）给出，几何预测一律使用**相对核的速度** $\mathbf v_{\rm sl}:=\mathbf v-\mathbf a_c-\boldsymbol\theta_c\times\mathbf x$；核本身的整体平移由第 10.4.4 节步骤 8 的质心重对齐吸收。核内的残余 $\mathbf v_{\rm sl}$ 是正则化 $\dot\gamma_\varepsilon>0$ 产生的数值蠕变，由 $\chi_{Y,i}=0$ 精确剔除；本文不存在任何对 $\mathbf D$ 作时间积分的状态量，故该蠕变既不进入几何，也不在任何历史量中累积。

**（B4）表面预测量（$d_{\rm pred}$ 的唯一定义）。**

$$
\boxed{\ d_{\mathrm{pred},i}^{(n)}=\frac{\chi_{Y,i}\,\Delta t_n\,\mathbf v_{{\rm sl},i}^{(n)}\!\cdot n_i^{(n)}+\dfrac{\rho_{\mathrm{bulk}}}{\rho_K(\mathbf x_i)}\,\Delta h_{C,i}}{n_i^{(n)}\cdot e_r},\ }
\qquad
\Delta h_C=\Delta t_n\,\frac{J_{\mathrm{dep}}-J_{\mathrm{ero}}}{\rho_{\mathrm{bulk}}},
\qquad
\rho_{\mathrm{bulk}}=\rho_p\nu_0,
$$

$\Delta t_n$ 为第 10.4.1 节的地质步长。表面节点据此分为三类：**锁死集** $\mathcal L^{(n)}=\{i:\chi_{Y,i}=0,\ \Delta h_{C,i}=0\}$（固体基底，$d_i=0$）；**纯沉积集** $\mathcal D^{(n)}=\{i:\chi_{Y,i}=0,\ \Delta h_{C,i}\ne0\}$（基底不动，仅外加物料堆积，$d_i=d_{\mathrm{pred},i}$ 固定）；**可动集** $\mathcal A^{(n)}=\{i:\chi_{Y,i}=1\}$（Clarabel 的唯一自由变量）。“未屈服区锁死”指固体基底不发生流动位移；沉积是滑坡物料在其上的堆积，不是基底运动，若同时锁死沉积量将破坏质量守恒。

组成（$\Delta R_C$ 记号即此 $\Delta h_C^*/(n\cdot e_r)$，其中 $\Delta h_C^*:=\dfrac{\rho_{\mathrm{bulk}}}{\rho_K|_\Gamma}\Delta h_C$）。**$d_{\mathrm{pred}}$ 的唯一定义**即上式，与第 10.4.3 节相同：本文不使用位移场 $u$；增量位移就是瞬时 Stokes 蠕变速度（相对刚性核）乘以地质步长 $\Delta t_n\,\mathbf v_{\rm sl}$。**质量雅可比加权：** $\Delta h_C$ 由堆积密度 $\rho_{\mathrm{bulk}}$ 换算为厚度；当该厚度折算为以实体 Cauchy 密度 $\rho_K$ 表示的几何位移时，必须乘局部密度权 $\rho_{\mathrm{bulk}}/\rho_K|_\Gamma$，使单位面积质量 $\rho_{\mathrm{bulk}}\Delta h_C=\rho_K|_\Gamma\Delta h_C^*$ 守恒；Clarabel 质量行 $D\mathcal M^{(n)}d+s_M=M-\mathcal M^{(n)}$ 中凡涉及通量的几何增量一律使用 $\Delta h_C^*$。否则即使 $\int J_{\mathrm{dep}}=\int J_{\mathrm{ero}}$，只要 $\rho_{\mathrm{bulk}}\ne\rho_K|_\Gamma$ 就会产生虚假质量漂移 $s_M\ne0$。$d$ 是径向增量而 $\mathbf v_{\rm sl}\cdot n$、$\Delta h_C^*$ 是法向量，一阶几何关系 $\delta R\,(n\cdot e_r)=\delta_n$ 要求先决条件：表面相对原点星形且非掠射，$n\cdot e_r\ge c_0>0$；否则该节点不得用径向参数化。$J_{\mathrm{dep}},J_{\mathrm{ero}}$ 为面质量通量（kg·m$^{-2}$·s$^{-1}$），故 $\Delta h_C$ 单位为 m；$\Delta R_C$ 只能来自模式 C 的物料通量，不能把同一流动位移重复加入。固态自由表面一般不满足 $W=\mathrm{const}$。坡度代理仅用于已声明适用的松散表层可动节点，须另检查 $g_{\mathrm{eff}}\cdot n<0$；锁死节点不施加坡度代理（与核相连的固体可以陡于安息角，这正是保留真实地貌的机制）。

**（B5）受拉截断。** 第 8.7 节的压力 Poisson 问题不带点态不等式约束；流变与屈服使用的压力为截断正则化值 $\tilde p=\max([p]_+,p_{\min})$，被截断的受拉部分 $\|[p]_-\|$ 必须作为“受拉比例”诊断量逐步报告，且不得宣称截断后的压力仍满足 Poisson 方程。受拉区的强度退化为 $\tau_y=c+\mu_yp_{\min}$（Drucker–Prager 张力截断），无内聚时该区按 $F_\delta$ 判定通常屈服。

**（B6）塑性诊断量（不反馈）。** 屈服区的塑性应变率 $\mathbf D_p:=\mathbf 1_{\{F_\delta>0\}}\mathbf D$、塑性耗散率 $\Phi_p:=\int_{\mathcal V}\mathbf 1_{\{F_\delta>0\}}\boldsymbol\tau_{\rm flow}:\mathbf D\,dV\ge0$ 与 $\mathcal X_Y$ 上的累计等效塑性应变 $\bar\varepsilon_p\leftarrow\bar\varepsilon_p+\lambda\Delta t_n\,\dot\gamma\,\mathbf 1_{\{F_\delta>0\}}$ 只作后处理诊断输出，**不进入** $\boldsymbol\sigma$、$\eta$、$F_\delta$、$d_{\rm pred}$ 或任何验收判据。

模式 C 的闭合接口如下。令 $\chi_{A,i}=\chi_{Y,i}\cdot\mathbf 1_{\{\rho_{s,i}>0\}}$（可动且有松散物料）。侵蚀通量由表面点的屈服超额驱动：

$$
\partial_t(J_\Gamma\rho_s)+\partial_a(J_\Gamma\rho_s c^a)
=J_\Gamma(J_{\mathrm{dep}}-J_{\mathrm{ero}}).
$$

$J_{\mathrm{ero}}$ 采用配置的侵蚀系数 $k_e\ge0$：

$$
J_{\mathrm{ero}}=\chi_A\,k_e\rho_s\frac{[F_\delta]_+}{\sigma_0}\quad(\rho_s\ \text{为松散层面密度 kg·m}^{-2},\ \text{初值 }\rho_s^{0}=\rho_{\mathrm{bulk}}h_s,\ k_e\ \text{单位 s}^{-1}),
\qquad
J_{\mathrm{dep}}(a)=\int_{\Gamma}K(a,a')J_{\mathrm{ero}}(a')J_\Gamma(a')\,da',
$$

其中 $\sigma_0>0$，$K\ge0$ 且
$\int_{\Gamma}K(a,a')J_\Gamma(a)\,da=1$。$F_\delta$ 取表面点 $(1,\hat{\boldsymbol\omega}_i)$ 处由唯一应力场算得的值，故锁死节点 $J_{\rm ero}=0$。因此无外部物料源时严格有

$$
\int_{\Gamma}J_{\mathrm{dep}}J_\Gamma\,da
=\int_{\Gamma}J_{\mathrm{ero}}J_\Gamma\,da.
$$

外部源必须显式增加 $J_{\mathrm{ext}}$ 并计入总质量预算，不能隐藏在沉积项中；此时质量目标同步更新为

$$
M^{n+1}=M^n+\Delta t_n\int_{\Gamma}J_{\mathrm{ext}}J_\Gamma\,da.
$$

只有这一个守恒方程产生 $\Delta R_C$；模式 B 的流动位移和模式 C 的通量不得重复计数。活动区的曲面坐标速度取 $c^a:=\chi_A\,(\mathbf v_{\rm sl}|_\Gamma)_S^a/R$，即第 9.1 节命题 9.1 的表面参数速度 $\mathbf c=\mathbf v_S/R$（相对刚性核）在活动集上的限制，不得用未投影的三维速度直接代入曲面守恒式。$\rho_s$ 仅是记账用的松散层面密度场，其切向输运本身不改变 $R$，$R$ 只由净通量 $J_{\rm dep}-J_{\rm ero}$ 经 $\Delta h_C^*$ 更新；$\rho_s$ 的输运用第 9.2 节的半拉格朗日格式（含 Jacobian 修正），正性由步长条件 $\Delta t_n k_e\max[F_\delta]_+/\sigma_0\le\theta_C<1$ 保证（第 10.4.1 节）。在 $S^2$ 参数化下 $J_\Gamma\,d\theta d\varphi=j\,d\Omega$，上述守恒方程等价于以 $j$ 为密度的形式，实现应使用 $j$（避免极点坐标奇异）。

---

## 2.9 统一公式驱动的 SCF 迭代

设第 $n$ 个地质步的形状为 $V^{(n)}$，外生自转为 $\omega=\omega_n=\omega(t_n)$（第 10.4.1 节）；本章其余各式中的 $\omega$ 均指该步冻结值。

## 2.10 正演

对每个表面节点 $x_i^{(n)}$，使用统一 Feynman 表示计算

$$
U_i^{(n)},
\qquad
\nabla U_i^{(n)},
\qquad
\nabla^2U_i^{(n)}.
$$

靠近表面时采用球形有限部分减法。

然后计算

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

## 2.11 流体质量能级接口（模式 A，保留但不启用）

<!-- RESERVED_FLUID_INTERFACE: 保留质量能级接口及其数学定义，供未来显式启用的流体模拟使用；模式 B/C 不调用，代码实施和清理时不得因当前未使用而误删。 -->

下式仅定义模式 A 的流体接口。模式 B/C 直接约束实际物质域的 $\mathcal M(V,\rho)=M$，不以势能级选取固态边界，也不依赖此接口的求根成功。

定义

$$
M^{(n)}(W_0)
=
\int_{\{x:W^{(n)}(x)>W_0\}}
\rho_K(x)\,dx.
$$

求解

$$
M^{(n)}(W_0^{(n)})=M.
$$

若 $M^{(n)}$ 在目标区间连续且严格递减，则可使用二分法。严格递减需要保证目标能级之间的区域具有正体积且密度不为零。

---

## 2.12 流体等势接口（模式 A，保留但不启用）

<!-- RESERVED_FLUID_INTERFACE: R_eq 是未来流体模拟的接口输出，当前颗粒体目标函数和迭代流程不得引用；保留接口，不作为无用代码删除。 -->

对每个方向 $u_i$，求解

$$
W^{(n)}(R_{\mathrm{eq},i}^{(n)}u_i)
=
W_0^{(n)}.
$$

该方程只有在所选径向搜索区间内存在根时才可执行；若要用单调二分法，还必须假设 $R\mapsto W^{(n)}(Ru_i)$ 在该区间严格单调。多根情形必须预先规定选取哪一个根，并进行连续分支跟踪。

得到等势候选半径

$$
R_{\mathrm{eq}}^{(n)}
=
\left(
R_{\mathrm{eq},1}^{(n)},\ldots,R_{\mathrm{eq},N}^{(n)}
\right).
$$

这一步只产生流体参考形状，不是模式 B/C 的固态目标。势场具有相应正则性且 $\nabla W\ne0$ 时才可由隐函数定理得到局部光滑等势面；多面体边界和临界点处不能无条件声称其为全局 $C^\infty$ 或解析曲面。

---

## 2.13 力学预测后的约束校正（模式 B 为主，模式 C 为辅）

冻结 $g^{(n)}$、$H_W^{(n)}$，先由模式 B 的准静态 Stokes 蠕变解与物理屈服分类（2.8 节 B1–B4）得到相对刚性核的流动预测；只有可动集上的屈服表面点才由模式 C 求解守恒侵蚀/沉积。二者须明确交接，不能重复计入同一流动位移或物料通量。记预测半径增量为 $d_{\mathrm{pred}}^{(n)}$，且 $R_{\mathrm{pred}}=R^{(n)}+d_{\mathrm{pred}}^{(n)}$。它不含 $R_{\mathrm{eq}}$ 或全局等势混合项。

**几何基准：保持当前形状。** 子问题的基准是**当前几何** $R^{(n)}$ 本身；唯一允许的改变是力学预测增量 $d_{\rm pred}^{(n)}$（锁死集上恒为零）。按 2.8 节（B4）把节点分成锁死集 $\mathcal L^{(n)}$、纯沉积集 $\mathcal D^{(n)}$ 与可动集 $\mathcal A^{(n)}$，保护区域定义为 $\mathcal P^{(n)}:=\mathcal L^{(n)}$。给定 $w_i>0$，Clarabel 子问题的目标为

$$
d_*^{(n)}=\arg\min_d\frac12\sum_{i\in\mathcal A^{(n)}}w_i
\left(d_i-d_{{\mathrm{pred}},i}^{(n)}\right)^2,
\qquad R_*^{(n)}=R^{(n)}+d_*^{(n)},
$$

它等于以当前形状为原点、对“当前形状 + 力学增量”的加权偏离 $\|(R^{(n)}+d)-(R^{(n)}+d_{\rm pred})\|_w^2$；目标中**不得**出现 $R_{\mathrm{eq}}$、$W_0$ 或任何等势面项（等势面是无剪切强度流体的形状，以它为目标会把未屈服地形抹平成光滑椭球）。非可动节点以硬约束固定：

$$
d_i=0\qquad(i\in\mathcal L^{(n)}),\qquad
d_i=d_{\mathrm{pred},i}^{(n)}=\frac{\Delta h_{C,i}^*}{n_i^{(n)}\cdot e_r}\qquad(i\in\mathcal D^{(n)}),
$$

外层非线性验收使用

$$
\frac{|g_i^{(n)}\times n_i(R^{(n)}+d)|^2}{|g_i^{(n)}|^2}
\le \sin^2\phi_i^{(n)},
$$

（仅限可动集 $\mathcal A^{(n)}$ 中已声明适用的松散表层代理节点；锁死节点不受坡度代理约束）以及

$$
\mathcal M(R^{(n)}+d,\rho_{\mathrm{cand}})=M.
$$

Clarabel 实际只接收当前迭代点的一阶模型。令 $n_i^{(n)}$ 为单位外法向，并定义

$$
\widehat n_i(d)=n_i^{(n)}+Dn_i^{(n)}d,
\qquad
\widehat{\mathcal M}(d)=\mathcal M^{(n)}+D\mathcal M^{(n)}d.
$$

在信赖域 $\|d\|\le\Delta$ 内，子问题使用

$$
\frac{\|g_i^{(n)}\times\widehat n_i(d)\|}{|g_i^{(n)}|}
\le\sin\phi_i^{(n)}+t_i,
\qquad
g_i^{(n)}\cdot\widehat n_i(d)\le-\epsilon_n|g_i^{(n)}|,
\qquad
D\mathcal M^{(n)}d+s_M=M-\mathcal M^{(n)},
$$

其中 $t_i\ge0$、$s_M\in\mathbb R$ 为弹性松弛变量，$\epsilon_n\in(0,1)$ 为防倒悬裕度（线性约束，不破坏凸性）。质量约束等价于 $\widehat{\mathcal M}(d)=M$（当 $s_M=0$），**不得**写成 $\widehat{\mathcal M}(d)=M-\mathcal M^{(n)}$（会重复减去 $\mathcal M^{(n)}$ 而使增量反号）。目标函数相应改为
$\tfrac12\sum_{i\in\mathcal A}w_i(d_i-d_{\mathrm{pred},i})^2+\varrho\big(\sum_it_i+|s_M|\big)$，$\varrho>0$ 取为足够大的精确罚因子。若防倒悬行在 $d=0$ 处满足（$g_i\cdot n_i<-\epsilon_n|g_i|$），则 $d=0$、$s_M=M-\mathcal M^{(n)}$ 和 $t_i=\max(0,\cdot)$ 给出一个可行点；它不自动给出 Slater 严格内点，因为正部取值可能使 SOC 约束取等号。若需要 Slater 证书，必须在每个 SOC 松弛中另加给定的正裕度 $\delta_t>0$，取 $t_i=\max(0,\cdot)+\delta_t$，并同时检查所有未松弛的线性约束。防倒悬行没有松弛变量，若 $d=0$ 违反该行，则 $d=0$ 不可行，须在外层几何预处理中沿可验证的几何方向修复或剔除该点后再进入 SOCP；$t_i=s_M=0$ 才表示松弛变量为零，外层验收仍须检查真实约束。

这是真正的 SOCP 约束；若再把范数约束线性化，才是 QP。$\widehat n$ 不被当作单位法向，不能用缩小其范数来满足物理约束；候选解必须重新归一化真实法向并用外层非线性约束复核。原始 $n(R+d)$、真实质量和真实屈服条件只属于外层验收。该几何校正不替代应力平衡与本构求解；张量屈服证书必须随候选力学状态变化，冻结常数张量的通过/拒绝不能称为几何耦合屈服约束。

子问题内冻结场；候选几何、密度和应力在外层更新后重新正演并复核。若当前状态已平衡、满足约束且 $d_{\mathrm{pred}}=0$，则 $d=0$ 是严格正权目标的唯一最小解，避免无外因时的地形漂移。由于上述弹性形式在预处理后的无倒悬几何上恒可行，不再采用“不可行时缩小信赖域重试”的逻辑（当前点本身违反约束时 $d=0$ 不可行，且 $\Delta\to0$ 时可行集趋于空，该策略必然失败）。**统一的信赖域策略：** 内层 SOCP 绝不缩小 $\Delta^{(n)}$；仅当外层非线性几何验收判定网格倒悬或自交时，外层才缩小 $\Delta^{(n)}$ 并重新生成子问题。质量行中的通量几何增量一律取 $\Delta h_C^*=(\rho_{\mathrm{bulk}}/\rho_K|_\Gamma)\Delta h_C$（2.8 节）。若最优解的 $t_i$ 或 $s_M$ 非零，表示线性化模型下约束只能部分满足：外层应以 filter/价值函数判据决定接受与否，并可重新求解力学预测；不能解除保护区或回退到流体目标。保护区全部包围某违反节点、或 $\rho_K$ 带符号使 $D\mathcal M$ 退化时，$s_M\neq0$ 是不可避免的，须如实报告。冻结 $g$ 忽略了 $Dg\cdot d$，故该迭代是顺序凸（Picard）型而非严格 SQP，不声称局部超线性收敛；线性化误差为 $O(\Delta^2)$，外层验收须设容差。$\widehat n_i$ 中的 $Dn_i^{(n)}$ 必须是**归一化**单位法向的 Jacobian（此时 $n\cdot Dn\,d=0$，$|\widehat n|\ge1$）。

---

## 2.14 欠松弛

取 $0<\lambda\le1$，定义

$$
R_i^{(n+1)}
=
(1-\lambda)R_i^{(n)}
+
\lambda R_{*,i}^{(n)}.
$$

然后由 $R^{(n+1)}$ 重建新多面体 $V^{(n+1)}$。

锁死集上 $d_{*,i}=0$，故欠松弛不移动锁死节点。欠松弛对应的物理时间推进为 $\lambda\Delta t_n$（第 10.4.1 节）。欠松弛后须在新几何上重新正演引力、重新求解准静态 Stokes 与压力（应力是瞬时平衡量，没有需要“更新”的材料历史），并按模式 C 守恒更新 $\rho_s$；非线性质量与屈服约束不保证沿线段成立。不得通过全局半径缩放或均匀增量修复质量而移动保护区；应在可动集内做守恒校正，失败则回退并减小 $\Delta t_n$；信赖域仅在外层几何验收出现倒悬或自交时按 2.13 节缩小。

---

## 2.15 收敛判据

准静态主线有三层判据，各自独立，不得互相替代（算法见第 10.4.4 节）。

**（a）流变非线性（Picard，层级 $k$）收敛。** 在固定几何与固定 $\omega_n$ 上，第 10.4.2 节的 Stokes–压力 Picard 迭代停止当且仅当

$$
\frac{\|\mathbf u^{k+1}-\mathbf u^{k}\|_{\mathbf K}}{\max\{\|\mathbf u^{k+1}\|_{\mathbf K},\varepsilon_u\}}\le\varepsilon_{\rm pic},\qquad
\frac{\|p^{k+1}-p^{k}\|_{L^2}}{\max\{\|p^{k+1}\|_{L^2},\varepsilon_p\}}\le\varepsilon_{\rm pic},
$$

（$\|\mathbf u\|_{\mathbf K}^2=\mathbf u^{\!\top}\mathbf K(\eta^{k})\mathbf u$）。达到迭代上限 $k_{\max}$ 仍未满足时，按第 10.4.2 节的 $\dot\gamma_\varepsilon$ 连续化与黏度松弛重试（该求解与 $\Delta t_n$ 无关，减小步长无济于事）；仍失败则判“流变未收敛”并终止，不得输出通过状态。

**（b）力学平衡验收（每个地质步，硬判据）。** 在收敛的 $(\mathbf u,p)$ 上

$$
\varepsilon_{\rm mom}^{\rm weak}:=\frac{\|\mathbf F-\mathbf K(\mathbf u;\tilde p)\mathbf u-\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda\|_2}{\max\{\|\mathbf F\|_2,\varepsilon_F\}}\le\varepsilon_{\rm mom},\qquad
\varepsilon_\lambda:=\frac{\|\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda\|_2}{\max\{\|\mathbf F\|_2,\varepsilon_F\}}\le\varepsilon_{\lambda}^{\rm tol},
$$

前者是真实流体系统（唯一应力场）的 Galerkin 动量残差，后者度量外载对刚体模式的不相容（参考系未对齐或离散刚体模态缺失，A.3）。强形式残差 $\mathbf r_{\rm rot}=\mathbf f-\nabla p$ 的相对范数 $\|\mathbf r_{\rm rot}\|_{L^2}/\max\{\|\rho_K\mathbf g_{\rm eff}\|_{L^2},\varepsilon_f\}$ 与切向牵引偏差 $\delta_{na}$ 是离散分辨率指标（随 $K,L_{\max}$ 收敛），必须逐步报告；超过配置阈值时标记“分辨率不足”，但不改写 (b) 的判定。

**（c）阶段平衡（外生自转分阶段恒定时）。** 在恒定 $\omega_j$ 的阶段内，几何达到静力平衡当且仅当

$$
\boxed{\ \mathcal A^{(n)}=\varnothing\quad(\text{全部表面点属于刚性核 }\mathcal U_{\rm core}),\qquad\text{且 (b) 成立}. \ }
$$

此时 $F_\delta\le0$ 在全部表面点成立，故 $J_{\rm ero}=0$、$J_{\rm dep}=0$、$\mathcal D^{(n)}=\varnothing$，$d_{\rm pred}=0$，SOCP 无自由变量而 $d_*=0$ 严格成立；该状态与 $\Delta t_n$、$\lambda$ 无关，且由于不存在对 $\mathbf D$ 积分的状态量，它是真正的不动点而不是缓慢漂移的蠕变态——正则化蠕变仍存在于核内，但被 $\chi_Y=0$ 精确剔除。$\mathcal A^{(n)}\ne\varnothing$ 时（存在真实屈服流动），用 $d_*$ 的大小判收敛没有意义：自适应步长 $\Delta t_n$ 使每步几何位移约为 $\theta_{\rm CFL}h_\Gamma$。阶段时长 $T_j$ 用尽而 $\mathcal A^{(n)}\ne\varnothing$ 时，阶段以“未平衡”结束并报告可动面积分数 $\sum_{i\in\mathcal A}j_iw_i/\sum_ij_iw_i$，这是物理结果（给定时间内滑坡尚未停止），不是数值失败。连续外生自转 $\omega(t)$ 下没有阶段平衡目标，只做 (a)(b) 并推进到终止时刻。

同时检查适用表层的坡度残差（仅可动节点）

$$
\max_{i\in\mathcal A^{(n)}}
\left[
\frac{|g_i^{(n)}\times n_i^{(n)}|}
{|g_i^{(n)}|}
-\sin\phi_i^{(n)}
\right]_+
<\varepsilon_{\mathrm{slope}},
$$

以及质量残差

$$
|\mathcal M(V^{(n)})-M|
<\varepsilon_M.
$$

最后在终态几何上重新计算

$$
U,\qquad
\nabla U,\qquad
\nabla^2U,
$$

并在同一最终几何、密度与 $\omega$ 上复核 (b)、屈服分类与锁死集位移为零。子问题 KKT 通过不等于非线性力学闭环通过，步长因信赖域收缩而变小也不构成收敛。最终审计通过后才输出有效势、有效重力和 Hessian。

---

## 2.16 统一耦合系统

统一计算公式进入颗粒 SCF 后，最终数学闭环为

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

模式 B 的场方程、适用表层坡度代理和质量约束可记为

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

该框中的坡度不等式只对可动集中适用的松散表层节点成立，不是固态实体的充分平衡条件。它必须与第 2.8 节（B1）的准静态 Stokes 平衡（唯一应力场 $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D$）、自由表面牵引、（B2–B3）的屈服分类与刚性核锁死共同组成模式 B 的耦合求解系统；$\omega$ 是外生给定量（第 10.4.1 节），不作为未知量；模式 C 提供局部守恒流动预测，模式 A 接口不进入此闭环。

下面给出一个面向 GPU、RayBackend 和批量场点的快速算法。核心思想是：

- 几何交点使用 WGSL BVH（硬件 RT 可选）；
- $U,\nabla U,H$ 在同一条射线上同时计算；
- 同一批射线共享所有几何结果；
- $\alpha=1$ 使用解析径向原函数；
- 其他 $\alpha$ 使用固定节点 Gauss–Jacobi 求积；
- 近表面使用有限部分减法；
- 远场可切换到 GPU 多极展开，减少射线数量。

硬件 RT 只负责射线—三角形求交，不能直接计算 Feynman 参数积分；没有硬件 RT 时使用同一接口的 WGSL BVH。因此最有效的结构是：

$$
\boxed{
\text{RTFP 射线几何}
+
\text{GPU 径向积分}
+
\text{GPU 归约}
+
\text{远场 FMM/多极加速}.
}
$$

---

# 3. GPU 射线积分与混合加速

## 3.1 统一射线公式

对每个场点 $x$ 和球面方向 $u_q$，取

$$
y=x-Ru_q.
$$

第 $k$ 个密度核定义为

$$
D_k(R)
=
(R-b_k)^2+d_k^2,
$$

其中

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

沿一条射线的三个基本径向积分是

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

于是

$$
\boxed{
U(x)
=
G\sum_{q,k}
w_k\sigma_k^{-2\alpha_k}
\omega_q I_U^{(k)}.
}
$$

梯度为

$$
\boxed{
\nabla U(x)
=
-G\sum_{q,k}
w_k\sigma_k^{-2\alpha_k}
\omega_q u_q I_g^{(k)}.
}
$$

Hessian 为

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

其中

$$
T(u)=3uu^T-I.
$$

这三个量共享：

- 相同的方向 $u_q$；
- 相同的射线交段 $[R^-,R^+]$；
- 相同的 $b_k,d_k$；
- 相同的密度参数。

因此不应分别运行三套积分程序。

---

## 3.2 $\alpha=1$ 时的解析原函数

当 $\alpha_k=1$ 时，定义

$$
D(R)=(R-b)^2+d^2.
$$

势积分：

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

梯度积分：

$$
\boxed{
Q(R)
=
\frac1d\arctan\frac{R-b}{d}.
}
$$

Hessian 的径向积分

$$
L(R)=\int \frac{dR}{R[(R-b)^2+d^2]}
$$

可通过部分分式分解计算。令

$$
A_0=b^2+d^2,
$$

则

$$
\frac1{R[(R-b)^2+d^2]}
=
\frac1{A_0}\frac1R
+
\frac{-R+2b}{A_0[(R-b)^2+d^2]}.
$$

因此

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

对普通外部射线：

$$
I_U=P(R^+)-P(R^-),
$$

$$
I_g=Q(R^+)-Q(R^-),
$$

$$
I_H=L(R^+)-L(R^-).
$$

这会把每个射线的主要成本降到少量 $\log$、$\arctan$ 和乘加操作。

---

## 3.3 一般 $\alpha$ 的 GPU 求积

当 $\alpha\neq1$ 时，不要在 GPU 中执行自适应积分。使用固定节点 Gauss–Jacobi 或 Gauss–Legendre 求积。

令

$$
R(t)=\frac{R^++R^-}{2}
+
\frac{R^+-R^-}{2}t,
\qquad -1\le t\le1.
$$

在 $n_r$ 个固定节点 $t_j$ 上同时计算

$$
D_j=(R(t_j)-b)^2+d^2.
$$

然后

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

上述普通求积公式只适用于 $R^->0$ 的非奇异交段。若场点位于实体内部而交段从 $R^-=0$ 开始，或该段属于边界/近场奇异分支，则 $I_H$ 不能直接用普通 Gauss 求积；必须转入第 3.9 节的有限部分减法，并显式加入接触项。

实现配置如下：

- $\alpha$ 固定时预计算节点和权重；
- $\alpha=1$ 走解析分支；
- $n_r=8,16,32$ 分三级精度；
- 只对少数困难射线重新计算高阶节点。

---

## 3.4 GPU BVH 几何管线（可选硬件 RT 后端）

## 3.5 预处理阶段

对多面体三角网格建立 BVH：

1. 三角形顶点使用 Structure of Arrays 布局；
2. 预计算面法向、包围盒和边数据；
3. 构造可上传到 WGSL 的扁平 BVH；支持硬件 RT 时可建立等价加速结构；
4. 对固定天体形状重复使用 BVH。

三角形数据可存储为：

```text
v0x[], v0y[], v0z[]
v1x[], v1y[], v1z[]
v2x[], v2y[], v2z[]
normal_x[], normal_y[], normal_z[]
```

避免 Array of Structures 导致非合并访存。

---

## 3.6 射线任务

对每个场点 $x_m$ 和球面方向 $u_q$，发射射线

$$
r(t)=x_m-tu_q,\qquad t>0.
$$

RayBackend 返回全部交点距离

$$
0<R_1<R_2<\cdots<R_L.
$$

对封闭多面体：

- 外部点通常交点成对出现；
- 内部点的第一个交点为退出点；此时从 $R=0$ 到第一个交点的区间也属于实体；
- 非凸多面体可能有多个交段。

因此不能对所有场点都简单地把交点两两配成
$[R_1,R_2],[R_3,R_4],\ldots$。应先根据场点的内外分类确定 $R=0$
处的占据状态，再按状态切换生成区间：

$$
\begin{cases}
[R_1,R_2],[R_3,R_4],\ldots,&x\notin V,\\[2pt]
[0,R_1],[R_2,R_3],[R_4,R_5],\ldots,&x\in\operatorname{int}V.
\end{cases}
$$

边界点必须先固定本文第 1.2--1.3 节的球形代表约定；实现中可用内侧/外侧极限或显式的 $R=0$
近场分支，不能把边界点无条件套入任一配对规则。

不要让每条线程动态分配内存。推荐两阶段方案：

### 阶段 A：统计

每条射线只统计交点数量 $L$。

### 阶段 B：前缀和

对所有射线做 GPU prefix-sum，分配连续交点缓冲区。

### 阶段 C：写入

重新执行射线查询，将交点写入压缩数组。

这样可以避免固定长度缓冲区浪费大量显存。

---

## 3.7 GPU Kernel 组织

推荐使用以下线程映射：

```text
一个 WGSL workgroup
    → 一个场点批次
    → 固定数量的球面方向

一个 invocation
    → 一个方向
    → 处理该方向的全部交段

一个 workgroup
    → 共享该批次的 x、c_k、ω 等参数
```

对于每个方向线程：

1. 读取交段 $[R^-,R^+]$；
2. 计算 $u_q$；
3. 对每个 $k$ 计算 $b_k,d_k^2$；
4. 计算 $I_U,I_g,I_H$；
5. 累积 $U,\nabla U,H$。

局部累积写入 workgroup-local storage，再用 workgroup barrier 做分层归约；若设备支持 subgroup 操作，可在同一误差验证下替换归约步骤。

---

## 3.8 融合 Kernel

不要分别调用：

```text
compute_potential()
compute_gradient()
compute_hessian()
```

而应使用一个融合 Kernel：

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

其中

$$
T(u)=3uu^T-I
$$

只包含六个独立 Hessian 分量：

$$
H_{xx},H_{yy},H_{zz},H_{xy},H_{xz},H_{yz}.
$$

因此不必计算九个分量。

---

## 3.9 近表面有限部分

若场点靠近表面，直接计算 $I_H$ 会产生对数发散。

设局部切锥方向集合为 $\Gamma_x$。对近场半径 $r_0$ 做减法：

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
\frac{\rho(x-Ru)-\rho(x)}R\,dR\,d\Omega.
\end{aligned}
$$

GPU 实现中：

1. 射线交点距离 $R<r_0$ 的部分进入 near kernel；
2. 计算密度差 $\rho(x-Ru)-\rho(x)$；
3. 对差分项使用固定求积；
   在 $R=0$ 的节点使用极限 $-\nabla\rho(x)\cdot u$，不要执行 $0/0$；
4. 显式加入
   $$
   G\rho(x)C(x)\log(r_0/\ell_0);
   $$
5. 加入接触项
   $$
   -\frac{4\pi G}{3}\vartheta_x\rho(x)I.
   $$

这样 GPU 中不会直接计算发散的 $\log\varepsilon$ 项。

---

## 3.10 远场多极加速

当场点满足

$$
\operatorname{dist}(x,V)>r_{\mathrm{switch}},
$$

可不再对所有面片发射射线，而使用 GPU FMM 或八叉树多极展开。

将源体积划分为树节点。对远离节点的场点，用截断展开：

$$
U(x)
\approx
G\sum_{|\ell|\le p}
\frac{M_\ell}{|x-c_{\mathrm{node}}|^{\ell+1}}.
$$

该式只是复杂度说明的标量简写；实际三维多极展开必须明确多重指标、基函数归一化、树节点平移算子和截断阶 $p$ 的余项界。没有这些定义时，它不能作为数值核的实现公式，也不能把下面的复杂度写成严格误差或严格运行时间保证。

梯度和 Hessian 由同一组多极系数递推得到：

$$
\nabla U
=
\nabla \mathcal M_p,
\qquad
H
=
\nabla^2\mathcal M_p.
$$

近场节点仍然使用 RayBackend 射线求交，远场节点使用多极展开。

在树平衡、接受准则、展开阶 $p$ 固定且每个叶节点容量有界的条件下，总复杂度才可作如下规划估计。直接法为

$$
O(N_xF)
$$

降为近似

$$
\boxed{
O(N_x\log F+N_{\mathrm{source}}p^2)
}
$$

或标准 FMM 的

$$
\boxed{
O((N_x+N_{\mathrm{source}})p^2).
}
$$

GPU BVH 和 FMM 的混合是默认方案；硬件 RT 仅作为可选后端：

$$
\boxed{
\text{近场：WGSL BVH（硬件 RT 可选）}
\qquad
\text{远场：GPU-FMM}
}
$$

---

## 3.11 混合调度

对每个场点 $x$，设最近几何距离为 $d(x,V)$。

选择两个阈值：

$$
d_{\mathrm{near}}<d_{\mathrm{far}}.
$$

采用：

$$
\begin{cases}
d<d_{\mathrm{near}}:&
\text{RTFP + 有限部分};\\
d_{\mathrm{near}}\le d\le d_{\mathrm{far}}:&
\text{RTFP 全射线};\\
d>d_{\mathrm{far}}:&
\text{GPU FMM/多极展开}.
\end{cases}
$$

这比对所有场点统一使用 Feynman 面片积分更快，因为 Feynman 路线通常还需要：

- $u$ 参数积分；
- 面片积分；
- 端点 $u\to1$ 处理；
- 额外的内层求积。

---

## 3.12 SCF 的 GPU 执行流程

**SCF 算法的唯一权威执行流程见第 10.4.4 节（9 步闭环）；本节不再保留未包含模态输运与重映射的过时流程描述。** 本节仅说明 GPU 在该闭环中承担的部分：

- 10.4.4 步骤 1（及步骤 9 的重算）中的 $U,\nabla U,H$ 由 RayBackend 批量求交加融合 Kernel（第 3.6–3.8 节）计算，输入为当前 $R^{(n)}$ 对应的三角网格与更新后的 BVH；
- 有效量 $W=U+U_c$、$\mathbf g_{\rm eff}$、$H_{\rm eff}$ 的合成同 2.8 节；
- 步骤 2–8 的模态输运、质量恢复、SOCP 与重映射的执行要求见第 9、10 章，不属于 RTFP 内核。

<!-- RESERVED_FLUID_INTERFACE: W0/R_eq 仅供未来模式 A 流体模拟，当前步骤不调用，不误删。 -->

---

## 3.13 复杂度

设：

- $N_x$：场点数；
- $N_\Omega$：球面方向数；
- $F$：三角面片数；
- $K$：密度核数量；
- $L$：平均每条射线交段数；
- $n_r$：径向求积节点数。

使用 BVH 后，RTFP 近场复杂度近似为

$$
\boxed{
O\left(
N_xN_\Omega
\bigl(\log F+L\bigr)
K n_r
\right).
}
$$

当 $\alpha=1$ 使用解析原函数时，$n_r$ 可近似视为常数 $1$，所以变为

$$
\boxed{
O\left(
N_xN_\Omega
(\log F+L)K
\right).
}
$$

SCF 总复杂度为

$$
\boxed{
O\left(
N_{\mathrm{iter}}
N_xN_\Omega
(\log F+L)K
\right)
}
$$

加上几何更新、质量标定和约束投影成本。

---

## 3.14 实现准则

最快的实际实现应采用：

$$
\boxed{
\text{Lebedev 球面节点}
+
\text{WGSL 扁平 BVH 多交点查询}
+
\text{RTFP 径向解析式}
+
\text{融合 }U/\nabla U/H\text{ Kernel}
+
\text{近表面有限部分}
+
\text{远场 GPU-FMM}.
}
$$

Feynman 统一公式适合作为：

- 外部场的独立计算通道；
- 面片积分实现；
- RTFP 结果的交叉验证；
- 不适合直接作为所有点的最快主算法。

RayBackend 负责几何求交，WGSL compute kernel 负责径向公式和张量累积；可选的矩阵硬件只用于经过误差验证的归约，有限部分 Hessian 的关键累积不得完全交给低精度路径。

严格结论为：

> 在明确列出的极限条件下，联合模型严格、逐项地退化为传统 $\mu(I)$ 流变模型。

如果这些条件不满足，联合模型就不会等于传统 $\mu(I)$。

另外需要先修正一个系数问题：若定义

$$
\dot\gamma=\sqrt{2D':D'},
\qquad
\tau_{\mathrm{eq}}=\sqrt{\frac12\tau:\tau},
$$

则为了得到

$$
\tau_{\mathrm{eq}}=\mu p,
$$

本构关系必须写成

$$
\boxed{
\tau
=
2\frac{\mu p}{\dot\gamma}D'
}
$$

或等价地

$$
\boxed{
\tau=2\eta D',
\qquad
\eta=\frac{\mu p}{\dot\gamma}.
}
$$

这里使用的是
$\tau_{\mathrm{eq}}=\sqrt{\tfrac12\tau:\tau}$ 和
$\dot\gamma=\sqrt{2D':D'}$；若省略上式中的因子 $2$，所得等效应力会变成
$\mu p/2$。

---

# 4. 联合流变模型及其经典极限

## 4.1 本构约定与经典 $\mu(I)$ 模型

*适用说明：* 本章的 $\mu(I)$ 及其退化论证针对流动（偏）应力与不可压缩流动速度场（$\nabla\cdot v=0$）。生产模型（2.8 节）只使用本章给出的黏塑性应力，不引入弹性应变或状态方程；压力是不可压缩约束的 Lagrange 乘子，由第 8.7 节的 Poisson 问题确定。

传统惯性流变模型定义

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

摩擦系数为

$$
\boxed{
\mu_{\mathrm{cl}}(I)
=
\mu_s+
(\mu_2-\mu_s)
\frac{I}{I_0+I}.
}
$$

在持续流动状态下，

$$
\boxed{
\tau_{\mathrm{eq}}
=
\mu_{\mathrm{cl}}(I)p.
}
$$

对应偏应力张量为

$$
\boxed{
\tau
=
2\mu_{\mathrm{cl}}(I)p
\frac{D'}{\dot\gamma}.
}
$$

因为

$$
\frac12
\left(
2\mu p\frac{D'}{\dot\gamma}
\right):
\left(
2\mu p\frac{D'}{\dot\gamma}
\right)
={}
\frac{4\mu^2p^2}{2\dot\gamma^2}
(D':D'),
$$

且

$$
D':D'=\frac{\dot\gamma^2}{2},
$$

所以

$$
\tau_{\mathrm{eq}}=\mu p.
$$

---

## 4.2 联合模型

联合模型包含四个扩展项。

## 4.3 内聚压力

定义

$$
p_c
=
C_c\frac{Z\nu F_H}{d^2}.
$$

其中 $Z$ 是配位数，$F_H$ 是单个接触的内聚力。

定义有效压力

$$
\boxed{
p_*=p+p_c.
}
$$

## 4.4 修正惯性数

$$
\boxed{
I_c
=
\frac{d\dot\gamma}
{\sqrt{p_*/\rho_p}}.
}
$$

## 4.5 状态依赖摩擦系数

动态摩擦系数为

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

## 4.6 潮汐背应力

令

$$
H_W=\nabla^2W,
$$

定义

$$
\boxed{
T_{\mathrm{tide}}
=
C_T\rho_p\nu d^2
\operatorname{dev}H_W.
}
$$

联合屈服函数为

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

其中 $s:=\operatorname{dev}\sigma$ 是总应力的偏应力。为使潮汐背应力与动态本构一致，定义流动偏应力 $\tau_{\mathrm{flow}}$，并取
$$
s=T_{\mathrm{tide}}+\tau_{\mathrm{flow}}.
$$
屈服函数中的有效应力因此是 $s-T_{\mathrm{tide}}=\tau_{\mathrm{flow}}$。

动态流动分支使用

$$
\boxed{
\tau_{{\mathrm{flow}},\mathrm{eq}}
=
c(\nu,Z)+\mu_d(I_c,\nu,Z)p,
\qquad
\tau_{\mathrm{flow}}=2\eta D'.
}
$$

---

## 4.7 退化条件

考虑一族联合模型参数和状态：

$$
\left(
p_c^{(n)},
H_W^{(n)},
c^{(n)},
\nu^{(n)},
Z^{(n)}
\right).
$$

假定在所考虑的区域内 $p>0$，$\dot\gamma>0$，并满足以下条件。

## 4.8 条件 A：内聚效应消失

$$
\boxed{
\frac{p_c^{(n)}}p\longrightarrow0.
}
$$

于是

$$
p_*^{(n)}=p+p_c^{(n)}
\longrightarrow p.
$$

## 4.9 条件 B：潮汐应力消失

$$
\boxed{
\frac{
\rho_p\nu^{(n)}d^2
\|\operatorname{dev}H_W^{(n)}\|
}{p}
\longrightarrow0.
}
$$

于是

$$
T_{\mathrm{tide}}^{(n)}\longrightarrow0.
$$

## 4.10 条件 C：状态变量冻结

$$
\boxed{
\nu^{(n)}\to\nu_0,
\qquad
Z^{(n)}\to Z_0.
}
$$

## 4.11 条件 D：本构参数退化为常数

要求

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

## 4.12 条件 E：黏聚强度消失

$$
\boxed{
\frac{c(\nu^{(n)},Z^{(n)})}{p}
\longrightarrow0.
}
$$

## 4.13 条件 F：压力正则化消失

若使用 $p_\varepsilon$，要求

$$
\boxed{
\frac{p_\varepsilon}{p}\longrightarrow0.
}
$$

采用该压力正则化时，程序必须在本构中把相应的 $p$ 统一替换为
$p_\varepsilon$（包括分母和应力幅值）；本文未启用正则化的主公式仍写作 $p$。

若使用剪切率正则化 $\dot\gamma_\varepsilon$，要求

$$
\boxed{
\frac{\dot\gamma_\varepsilon}{\dot\gamma}
\longrightarrow0.
}
$$

---

## 4.14 惯性数极限

由

$$
I_c^{(n)}
=
\frac{d\dot\gamma}
{\sqrt{(p+p_c^{(n)})/\rho_p}},
$$

以及

$$
\frac{p_c^{(n)}}p\to0,
$$

有

$$
p+p_c^{(n)}\to p.
$$

由于函数

$$
x\mapsto x^{-1/2}
$$

在 $p>0$ 邻域连续，所以

$$
\boxed{
I_c^{(n)}
\longrightarrow
\frac{d\dot\gamma}{\sqrt{p/\rho_p}}
=
I.
}
$$

这一步要求 $p>0$。在 $p=0$ 的自由表面，传统 $\mu(I)$ 本身没有定义，因此不存在通常意义下的退化。

---

## 4.15 摩擦系数极限

联合动态摩擦系数为

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

根据条件 C、D 和上一节的结论：

$$
\nu^{(n)}\to\nu_0,
\qquad
Z^{(n)}\to Z_0,
\qquad
I_c^{(n)}\to I.
$$

因此

$$
\mu_s(\nu^{(n)},Z^{(n)})\to\mu_s,
$$

$$
\mu_2(\nu^{(n)},Z^{(n)})\to\mu_2,
$$

$$
I_0(\nu^{(n)},Z^{(n)})\to I_0.
$$

由于有理函数

$$
(a,b,c)\mapsto a+(b-a)\frac{c}{I_0+c}
$$

在 $I_0+c>0$ 时连续，所以

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

---

## 4.16 潮汐屈服函数极限

联合屈服函数中的有效剪应力是

$$
\tau_{\mathrm{tide}}^{(n)}
=
\sqrt{
\frac12
(s-T_{\mathrm{tide}}^{(n)}):(s-T_{\mathrm{tide}}^{(n)})
}.
$$

由

$$
T_{\mathrm{tide}}^{(n)}\to0
$$

以及 Frobenius 范数的连续性，

$$
\boxed{
\tau_{\mathrm{tide}}^{(n)}
\longrightarrow
\sqrt{\frac12s:s}
=
\tau_{{\mathrm{flow}},\mathrm{eq}}.
}
$$

又由

$$
\frac{c^{(n)}}p\to0,
\qquad
\mu_y(\nu^{(n)},Z^{(n)})\to\mu_s
$$

可得静态屈服函数极限：

$$
F_{\mathrm{comb}}^{(n)}
\longrightarrow
\tau_{{\mathrm{flow}},\mathrm{eq}}-\mu_s p.
$$

这正是准静态、无黏聚、无潮汐扩展的 Drucker–Prager 型极限。

---

## 4.17 动态应力极限

在联合模型的流动分支中，

$$
\tau_{{\mathrm{flow}},\mathrm{eq}}^{(n)}
=
c^{(n)}
+
\mu_d^{(n)}p.
$$

由

$$
\frac{c^{(n)}}p\to0,
\qquad
\mu_d^{(n)}\to\mu_{\mathrm{cl}}(I),
$$

得到

$$
\boxed{
\tau_{{\mathrm{flow}},\mathrm{eq}}^{(n)}
\longrightarrow
\mu_{\mathrm{cl}}(I)p.
}
$$

因此传统 $\mu(I)$ 应力比例严格恢复。

由于偏应力与 $D'$ 同向，修正后的正则化本构写成

$$
\boxed{
\tau_{\mathrm{flow}}^{(n)}
=
2\eta^{(n)}D',
}
$$

其中

$$
\eta^{(n)}
=
\frac{
c^{(n)}+\mu_d^{(n)}p
}{\sqrt{\dot\gamma^2+\dot\gamma_{\varepsilon}^{\,2}}}.
$$

当

$$
\frac{c^{(n)}}p\to0,
\qquad
\mu_d^{(n)}\to\mu_{\mathrm{cl}}(I),
\qquad
\dot\gamma_\varepsilon/\dot\gamma\to0,
$$

有

$$
\eta^{(n)}
\longrightarrow
\frac{\mu_{\mathrm{cl}}(I)p}{\dot\gamma}.
$$

于是

$$
\boxed{
\tau_{\mathrm{flow}}^{(n)}
\longrightarrow
2\mu_{\mathrm{cl}}(I)p
\frac{D'}{\dot\gamma}.
}
$$

取等效剪应力：

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

由于

$$
D':D'=\frac{\dot\gamma^2}{2},
$$

得到

$$
\boxed{
\tau_{{\mathrm{flow}},\mathrm{eq}}^{(n)}
\longrightarrow
\mu_{\mathrm{cl}}(I)p.
}
$$

这就是传统 $\mu(I)$ 关系。

---

## 4.18 控制方程极限

联合模型的动量方程为

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

这里沿用本文的正势约定 $g_{\mathrm{eff}}=\nabla W$；若改用传统的负引力势，则必须同时把势梯度符号整体改写，不能只改动这一项。旋转系中的 Coriolis 项 $2\rho\boldsymbol\Omega\times v$ 与命题 5.2 一致；经典 $\mu(I)$ 系统对应惯性系（$\boldsymbol\Omega=\mathbf0$），此时该项消失。

在上述极限下，若采用传统不可压缩密实颗粒模型，还必须固定
$\rho=\rho_0>0$；连续方程随后等价于 $\nabla\cdot v=0$。若不固定密度，
下面只能称为可压缩变密度扩展，不能直接称为传统不可压缩 $\mu(I)$ 系统。

在该不可压缩极限下，联合模型的总应力应写为
$$
\sigma=-pI+T_{\mathrm{tide}}+\tau_{\mathrm{flow}}.
$$
动量方程中的 $\nabla\cdot\sigma$ 因而包含 $\nabla\cdot T_{\mathrm{tide}}$；在本节极限中
$T_{\mathrm{tide}}\to0$，所以该项消失。

在该不可压缩极限下：

- 密实度固定；
- 配位数固定；
- 潮汐背应力消失；
- 黏聚力消失；
- 内聚压力消失；
- 状态依赖参数变为常数。

因此本构项变为

$$
\sigma
=
-pI
+
2\mu_{\mathrm{cl}}(I)p
\frac{D'}{\dot\gamma}.
$$

所以控制方程严格退化为

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

这就是传统 $\mu(I)$ 颗粒流变学系统。

---

## 4.19 退化定理

可以将上述结果总结为：

## 4.20 定理

设联合模型解序列满足：

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

且沿所取点列 $p\to p_0>0$、$\dot\gamma\to\dot\gamma_0>0$（或至少在正下界上有统一界）。则联合模型满足

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
\tau_{{\mathrm{flow}},\mathrm{eq}}\to\mu_{\mathrm{cl}}(I)p,
$$

**动量方程收敛的严格条件（紧性链）。** 点态本构极限 $\tau^{(n)}_{\mathrm{eq}}\to\mu_{\mathrm{cl}}p$ 只是标量收敛，既推不出张量 $D'^{(n)}/\dot\gamma^{(n)}$ 收敛，也推不出微分后的动量方程收敛。第 4.14–4.17 节为逐点量，其中 $p,\dot\gamma,D'$ 是同一点上的数值，并非解序列；结论 $I_c\to I$ 中的 $I$ 指 $I=d\dot\gamma_\infty/\sqrt{p_\infty/\rho_p}$（极限场）。第 4.16 节中 $s-T_{\mathrm{tide}}\equiv\tau_{\mathrm{flow}}$ 由 $s$ 的定义恒成立，其极限是该恒等式的推论，不含独立信息。4.17 节正则化后 $\tau_{\mathrm{flow,eq}}=(c+\mu_dp)\dot\gamma/\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}$，仅 $\dot\gamma_\varepsilon/\dot\gamma\to0$ 时趋于 $c+\mu_dp$；4.15 节所用连续映射应为四变量 $(a,b,e,c)\mapsto a+(b-a)\,c/(e+c)$（$I_0$ 亦随 $n$ 变化，$e+c>0$）。

在以下**全部**附加假设下，动量方程才在分布意义下收敛：设 $K\Subset\{p>0\}$ 为紧集（空间时间区域），且

1. $p^{(n)}\ge p_{\min}>0$、$\dot\gamma^{(n)}\ge\dot\gamma_{\min}>0$ 在 $K$ 上一致，条件 A–F 在 $K$ 上一致成立；
2. $v^{(n)}\to v$ 在 $L^2(K)$ 中强收敛，$\nabla v^{(n)}\to\nabla v$ 与 $p^{(n)}\to p$ 在 $K$ 上几乎处处收敛（可由 Egorov/子列得到）；
3. $\sigma^{(n)}$ 在 $L^{1+\delta}(K)$（某 $\delta>0$）中一致有界，$\nu^{(n)}\to\nu_0$ 强收敛。

则 $\sigma^{(n)}\to\sigma_{\mathrm{cl}}(\nabla v,p)=-pI+2\mu_{\mathrm{cl}}(I)p\,D'/\dot\gamma$ 在 $L^1(K)$ 中收敛（Vitali 定理：几乎处处收敛 + 一致可积），故 $\nabla\cdot\sigma^{(n)}\to\nabla\cdot\sigma_{\mathrm{cl}}$ 在 $\mathcal D'(K)$ 中，对流项由 $L^2$ 强收敛与 $L^{1+\delta}$ 界收敛，$\nabla\cdot v=0$ 保持。**局限**：（a）几乎处处收敛是假设而非推论——$\mu(I)$ 的应力—应变率关系一般**非单调**（存在 $\mathrm{Fr}$ 小时的不适定区，Barker 等 2015），Minty 单调性技巧不可用，故无法仅由弱收敛识别极限；（b）结论仅限 $K\Subset\{p>0\}$ 内部，不覆盖自由表面边界条件与 $\dot\gamma=0$ 的塞流区；（c）模式 B（准静态，$\dot\gamma=0$）落在 $\dot\gamma_{\min}>0$ 之外，因此本定理对生产默认模式是空条件，仅适用于模式 C 的活动流动区。

---

## 4.21 适用范围与失效边界

以下情形不能使用上述退化结论：

## 4.22 自由表面低压极限与 $p=0$ 奇点

自由表面处法向动量平衡满足 $p|_\Gamma=p_{\rm ext}+\mathbf n\cdot\boldsymbol\tau\,\mathbf n$（第 5.1 节推论 5.3）。在无外压且无偏应力（或无潮汐、低剪切极限）下 $p\to0$。在此极限下，传统惯性数

$$
I=\frac{d\dot\gamma}{\sqrt{p/\rho_p}}
$$

存在分母除零奇点，因此必须保留有效压力 $p_*=p+p_c>0$ 或采用显式压力正则化。

## 4.23 潮汐应力不可忽略

若

$$
\rho_pd^2\|\operatorname{dev}H_W\|
\sim p,
$$

则联合模型不会退化为传统 $\mu(I)$。

## 4.24 状态变量仍在演化

若 $Z$ 或 $\nu$ 空间变化明显，则

$$
\mu_d=\mu_d(I,\nu,Z)
$$

不是传统的单变量 $\mu(I)$。

## 4.25 静态区域

若

$$
\dot\gamma=0,
$$

则动态 $\mu(I)$ 公式不是完整描述。此时应使用

$$
\sqrt{\frac12(s-T_{\mathrm{tide}}):(s-T_{\mathrm{tide}})}
\le c+\mu_y p.
$$

在正则化黏塑性生产模型中（2.8 节），静态区域不由 $\dot\gamma=0$ 识别（正则化下 $\dot\gamma>0$ 处处成立），而由同一应力场上的带容差屈服函数 $F_\delta\le0$ 及其与刚性核的连通性识别（2.8 节 B2–B3），$\mu_y\equiv\mu_s(\nu_0,Z_0)$。

---

最终严格结论是：

$$
\boxed{
\text{联合模型不是自动等于传统 }\mu(I)\text{；}
}
$$

$$
\boxed{
\text{在内聚、潮汐、状态演化和正则化都消失的明确极限下，}
}
$$

$$
\boxed{
\text{它逐项、连续地退化为传统 }\mu(I)\text{ 流变方程。}
}
$$

这是一条可以写成正式数学命题的退化关系，而不是“无条件绝对正确”的普适等式。

# 5. 混合弱式与速度双势表示

本章在星形域 $\mathcal V=\{r<R(\hat{\boldsymbol\omega},t)\}$ 上建立动量—不可压缩—质量输运的混合弱式，并把速度写成极向与环向两个势。记号服从附录 A.1 的符号表：应力 $\boldsymbol\sigma$（粗体）、方向 $\hat{\boldsymbol\omega}$、角速度向量 $\boldsymbol\Omega=\omega\mathbf e_z$。基本未知量为 $(\mathbf v,p,\rho,R)$。

---

## 5.1 控制方程与自然边界项

**定义 5.1（总应力与体加速度）。** 取

$$
\boldsymbol\sigma=-p\,\mathbf I+\boldsymbol\tau,\qquad
\boldsymbol\tau(\mathbf v,p)=\mathbf T_{\mathrm{tide}}+2\eta(p,\mathbf D)\,\mathbf D,\qquad
\mathbf D=\tfrac12(\nabla\mathbf v+\nabla\mathbf v^{\!\top}),
$$

其中 $\eta$ 由第 4 章给出：联合模型 $\eta=\dfrac{c+\mu_d(I_c)\,\tilde p}{\sqrt{\dot\gamma^2+\dot\gamma_\varepsilon^2}}$（$\tilde p=\max([p]_+,p_{\min})$ 为 2.8 节 B1 的唯一正则化压力，$I_c$ 取 $p_*=\tilde p+p_c$），经典分支（$c=0$、$\dot\gamma_\varepsilon\to0$ 的形式极限）$\eta=\mu(I)p/\dot\gamma$，因而 $\boldsymbol\tau=2\mu(I)\,p\,\mathbf D'/\dot\gamma$（不可压缩时 $\mathbf D'=\mathbf D$）。体加速度

$$
\mathbf g_0=\mathbf g_{\mathrm{RTFP}}-\boldsymbol\Omega\times(\boldsymbol\Omega\times\mathbf x)+\mathbf g_{\mathrm{ext}}
=\nabla W+\mathbf g_{\mathrm{ext}},\qquad W=U+\tfrac12\omega^2(x_1^2+x_2^2).
$$

离心加速度是 $-\boldsymbol\Omega\times(\boldsymbol\Omega\times\mathbf x)=\omega^2\mathbf x_\perp$，所以 $\mathbf g_0$ 中取负号（与第 2.8 节的正势约定一致）；$\mathbf g_{\mathrm{ext}}$ 不含已计入的引力与离心项。若 $\dot{\boldsymbol\Omega}\neq0$，再加 $-\dot{\boldsymbol\Omega}\times\mathbf x$，该项一般非梯度，本章的弱式与源项公式（5.6、6.3 节）以 $\nabla\times(\rho\mathbf g_0)$ 的形式书写，对其同样成立。准静态主线中 $\omega(t)$ 外生给定，Euler 项与驱动它的外力矩成对略去（2.8 节 B1，报告 $|\dot\omega|/\omega^2$）。

**命题 5.2（混合弱式与自然边界项）。** 设 $(\mathbf v,p)$ 足够光滑并满足强形式

$$
\rho\big(\partial_t\mathbf v+(\mathbf v\cdot\nabla)\mathbf v+2\boldsymbol\Omega\times\mathbf v\big)=\nabla\cdot\boldsymbol\sigma+\rho\mathbf g_0,\qquad
\nabla\cdot\mathbf v=0\ \text{in }\mathcal V,\qquad
\boldsymbol\sigma\mathbf n=\mathbf t_{\mathrm{ext}}\ \text{on }\Gamma .
$$

则对所有 $(\mathbf w,q)\in H^1(\mathcal V;\mathbb R^3)\times L^2(\mathcal V)$，

$$
\boxed{
\begin{aligned}
&\int_{\mathcal V}\rho\big(\partial_t\mathbf v+(\mathbf v\cdot\nabla)\mathbf v+2\boldsymbol\Omega\times\mathbf v\big)\cdot\mathbf w\,dV
+\int_{\mathcal V}\boldsymbol\tau(\mathbf v,p):\mathbf D(\mathbf w)\,dV\\
&\qquad-\int_{\mathcal V}p\,(\nabla\cdot\mathbf w)\,dV
+\int_{\mathcal V}q\,(\nabla\cdot\mathbf v)\,dV
-\oint_\Gamma\mathbf t_{\mathrm{ext}}\cdot\mathbf w\,dS
=\int_{\mathcal V}\rho\,\mathbf g_0\cdot\mathbf w\,dV .
\end{aligned}}
$$

反之，满足上式的光滑 $(\mathbf v,p)$ 满足强形式，包括边界条件。

*证明。* 用 $\mathbf w$ 乘动量方程并积分，散度定理给出
$\int_{\mathcal V}(\nabla\cdot\boldsymbol\sigma)\cdot\mathbf w=\oint_\Gamma\boldsymbol\sigma\mathbf n\cdot\mathbf w-\int_{\mathcal V}\boldsymbol\sigma:\nabla\mathbf w$。$\boldsymbol\sigma$ 对称，故 $\boldsymbol\sigma:\nabla\mathbf w=\boldsymbol\sigma:\mathbf D(\mathbf w)=-p\nabla\cdot\mathbf w+\boldsymbol\tau:\mathbf D(\mathbf w)$。把 $\boldsymbol\sigma\mathbf n$ 换成 $\mathbf t_{\mathrm{ext}}$ 即得第一式；不可压缩方程乘 $q$ 积分即得 $q$ 项。反向由 $\mathbf w$ 在 $\mathcal V$ 内的紧支撑任意性得到动量方程，再由 $\mathbf w|_\Gamma$ 的任意性得到 $\boldsymbol\sigma\mathbf n=\mathbf t_{\mathrm{ext}}$。∎

**推论 5.3（边界条件的自然实现）。** 第 4 章的自由表面牵引条件不是对 $\mathbf v$ 的本质约束，而是上式中的表面项 $-\oint_\Gamma\mathbf t_{\mathrm{ext}}\cdot\mathbf w\,dS$：

1. 自由表面：$\mathbf t_{\mathrm{ext}}=\mathbf 0$，表面项为零，$\boldsymbol\sigma\mathbf n=\mathbf 0$ 作为自然条件自动成立；
2. 接触面：$\mathbf t_{\mathrm{ext}}=-p_n\mathbf n+\mathbf t_T$，法向接触反力 $p_n\ge0$ 与切向摩擦 $\mathbf t_T$（$|\mathbf t_T|\le\mu_wp_n$）由接触模型给出，同样只进入表面项；
3. 自然条件隐含法向与切向分量：$\mathbf n\cdot\boldsymbol\tau\mathbf n-p=\mathbf n\cdot\mathbf t_{\mathrm{ext}}$，$\boldsymbol\tau\mathbf n-(\mathbf n\cdot\boldsymbol\tau\mathbf n)\mathbf n=\mathbf t_{\mathrm{ext}}-(\mathbf n\cdot\mathbf t_{\mathrm{ext}})\mathbf n$。自由表面上 $p=\mathbf n\cdot\boldsymbol\tau\mathbf n$，在 $\boldsymbol\tau\to0$ 时给出 $p\to0$（与第 4.22 节一致），无需任何代数替换。

**补充方程。** 质量输运与自由表面运动学为

$$
\partial_t\rho+\nabla\cdot(\rho\mathbf v)=0,\qquad
\partial_tR=v_r-\frac1R\,\nabla_{S^2}R\cdot\mathbf v_S\Big|_{r=R},
$$

后者来自 $F=r-R(\hat{\boldsymbol\omega},t)$ 满足 $\partial_tF+\mathbf v\cdot\nabla F=0$，其中 $\mathbf v_S$ 为物理切向分量、$\nabla_{S^2}$ 为单位球面梯度。准静态主线不推进连续方程（$\rho_K$ 在体坐标系中空间固定，模型假设 A-ρ，PO-14）；$R$ 由运动学关系 $\partial_tR=\mathbf v\cdot\mathbf n/(\mathbf n\cdot\mathbf e_r)$ 经屈服分类与地质步长离散为 $d_{\rm pred}$ 后，再由第 10.4.3 节的几何子问题更新（见第 9.9 节与 10.4.1 节）。

---

## 5.2 速度双势表示

**定义 5.4。** 对标量势 $\Phi,\Psi$，记

$$
\mathcal P[\Phi]=\nabla\times\nabla\times(\Phi\,\mathbf r),\qquad
\mathcal T[\Psi]=\nabla\times(\Psi\,\mathbf r)=\nabla\Psi\times\mathbf r,\qquad
\mathbf v=\mathcal P[\Phi]+\mathcal T[\Psi],\ \ \mathbf r=r\mathbf e_r .
$$

**引理 5.5（分量公式）。** 设 $\Phi=g(r)Y_{\ell m}$，$\Psi=f(r)Y_{\ell m}$，$L^2Y_{\ell m}=\Lambda_\ell Y_{\ell m}$，$\Lambda_\ell=\ell(\ell+1)$。则

$$
\mathcal P[\Phi]=\nabla\big(\partial_r(r\Phi)\big)-\mathbf r\,\Delta\Phi,\qquad
\nabla\cdot\mathcal P[\Phi]=0,\qquad \nabla\cdot\mathcal T[\Psi]=0,
$$

$$
v_r^{\mathcal P}=\frac{L^2\Phi}{r}=\frac{\Lambda_\ell g}{r}Y_{\ell m},\qquad
\mathbf v_S^{\mathcal P}=\frac1r\,(rg)'\,\nabla_{S^2}Y_{\ell m},\qquad
v_r^{\mathcal T}=0,\qquad
\mathbf v^{\mathcal T}=f\,\nabla_{S^2}Y_{\ell m}\times\mathbf e_r .
$$

球坐标下 $v^{\mathcal T}_\theta=\dfrac1{\sin\theta}\partial_\varphi\Psi$，$v^{\mathcal T}_\varphi=-\partial_\theta\Psi$。

*证明。* $\nabla\times\nabla\times\mathbf F=\nabla(\nabla\cdot\mathbf F)-\Delta\mathbf F$；对 $\mathbf F=\Phi\mathbf r$：$\nabla\cdot\mathbf F=3\Phi+r\Phi_r$，$\Delta(\Phi\mathbf r)=\mathbf r\Delta\Phi+2\nabla\Phi$，相减得 $\nabla(\Phi+r\Phi_r)-\mathbf r\Delta\Phi$。径向分量 $\partial_r(\Phi+r\Phi_r)-r\Delta\Phi=\Phi_r+\Phi_r+r\Phi_{rr}-r\Phi_{rr}-2\Phi_r+L^2\Phi/r=L^2\Phi/r$。切向分量即 $\frac1r\nabla_{S^2}\partial_r(r\Phi)$。环向：$\nabla\Psi=\Psi_r\mathbf e_r+\frac1r\nabla_{S^2}\Psi$，与 $r\mathbf e_r$ 叉乘得 $\nabla_{S^2}\Psi\times\mathbf e_r$。散度由 $\nabla\cdot\nabla\times=0$。以刚体转动 $\Psi=\omega r\cos\theta$ 检验：$v_\varphi=-\partial_\theta\Psi=\omega r\sin\theta$。∎

**命题 5.6（唯一性、$\ell=0$ 与刚体模式）。**

1. 对 $\ell\ge1$，$\Phi_{\ell m}$ 由 $v_r$ 唯一确定，$\Psi_{\ell m}$ 由 $\mathbf r\cdot\nabla\times\mathbf v$ 唯一确定；
2. $\ell=0$ 的 $\Phi=f(r)$ 与 $\Psi=h(r)$ 给出 $\nabla\times(f\mathbf r)=\nabla f\times\mathbf r=\mathbf 0$，对速度无贡献，因此速度自由度只含 $\ell\ge1$；不可压缩的星形域上径向呼吸 $\ell=0$ 的 $v_r$ 被 $\nabla\cdot\mathbf v=0$ 禁止；
3. 平移 $\mathbf v=\mathbf e$ 对应 $\Phi=\tfrac12\mathbf e\cdot\mathbf x$（因 $\nabla\times\nabla\times(\tfrac12(\mathbf e\cdot\mathbf x)\mathbf r)=\nabla\times(\tfrac12\mathbf e\times\mathbf x)=\mathbf e$），转动 $\mathbf v=\boldsymbol\theta\times\mathbf x$ 对应 $\Psi=\boldsymbol\theta\cdot\mathbf x$，二者都在 $\ell=1$、径向剖面 $\propto r$，共 6 个模式，构成 $\mathbf D(\mathbf v)=0$ 的全部解。

**命题 5.7（球对称权下的块结构）。** 设 $R$ 为常数、$\rho=\rho(r)$。则对 $\Phi_a=g_aY_{\ell m}$、$\Psi_a=f_aY_{\ell m}$，

$$
\int_{\mathcal V}\rho\,\mathcal P[\Phi_1]\cdot\mathcal P[\Phi_2]\,dV=\int_0^R\rho\,\big[\Lambda_\ell^2g_1g_2+\Lambda_\ell\,(rg_1)'(rg_2)'\big]dr,\qquad
\int_{\mathcal V}\rho\,\mathcal T[\Psi_1]\cdot\mathcal T[\Psi_2]\,dV=\int_0^R\rho\,\Lambda_\ell\,f_1f_2\,r^2dr,
$$

而极向—环向交叉项为零：$\int_{\mathcal V}\rho\,\mathcal P[\Phi]\cdot\mathcal T[\Psi]\,dV=0$。

*证明。* 用 $\int_{S^2}|\nabla_{S^2}Y|^2=\Lambda_\ell$ 与引理 5.5。交叉项被积函数为 $\tfrac1r(rg)'f\,\nabla_{S^2}Y\cdot(\nabla_{S^2}Y'\times\mathbf e_r)=\tfrac1r(rg)'f\,\mathbf e_r\cdot(\nabla_{S^2}Y\times\nabla_{S^2}Y')$，其球面积分是 $\mathbf e_r\cdot\nabla\times(Y\nabla_{S^2}Y')$ 的积分，为闭曲面上的散度，等于零。∎

---

## 5.3 流变本构的 Newton 切线

对经典分支 $\boldsymbol\tau=2\mu(I)p\,\mathbf D/\dot\gamma$，$I=d\dot\gamma\sqrt{\rho_p/p}$（$p>0$），$\mu(I)=\mu_s+(\mu_2-\mu_s)\dfrac{I}{I_0+I}$。在基态 $(\bar p,\bar{\mathbf D},\bar{\dot\gamma}=\sqrt{2\bar{\mathbf D}:\bar{\mathbf D}},\bar I,\bar\mu=\mu(\bar I))$ 处的一阶变分为

$$
\delta I=\bar I\Big(\frac{\delta\dot\gamma}{\bar{\dot\gamma}}-\frac{\delta p}{2\bar p}\Big),\quad
\delta\dot\gamma=\frac{2\bar{\mathbf D}:\delta\mathbf D}{\bar{\dot\gamma}},\quad
\mu'(\bar I)=\frac{(\mu_2-\mu_s)I_0}{(I_0+\bar I)^2},
$$

$$
\boxed{
\delta\boldsymbol\tau=2\bar\mu\bar p\Big(\frac{\delta\mathbf D}{\bar{\dot\gamma}}-\frac{\bar{\mathbf D}\,\delta\dot\gamma}{\bar{\dot\gamma}^2}\Big)
+\frac{2\bar{\mathbf D}}{\bar{\dot\gamma}}\Big[\bar p\,\mu'(\bar I)\bar I\Big(\frac{\delta\dot\gamma}{\bar{\dot\gamma}}-\frac{\delta p}{2\bar p}\Big)+\bar\mu\,\delta p\Big].}
$$

*推导。* 对 $\boldsymbol\tau=2\mu p\mathbf D/\dot\gamma$ 取全微分并代入 $\delta\mu=\mu'(\bar I)\delta I$。∎

由此，切线算子对速度和压力都有依赖：$\partial\boldsymbol\tau/\partial\mathbf D$ 给出黏性块的 Newton 矩阵，$\partial\boldsymbol\tau/\partial p=\dfrac{2\bar{\mathbf D}}{\bar{\dot\gamma}}\big(\bar\mu-\tfrac12\mu'(\bar I)\bar I\big)$ 给出速度方程对压力的耦合块。在局部简单剪切（$\bar D_{r\theta}>0$ 为唯一非零分量）下，$\delta\tau_{r\theta}=2\eta_t\delta D_{r\theta}+\mathcal K_p\delta p$，$\eta_t=\bar p\,\mu'(\bar I)\bar I/\bar{\dot\gamma}$，$\mathcal K_p=\bar\mu-\tfrac12\mu'(\bar I)\bar I$；$\eta_t$ 与割线黏度 $\eta_0=\bar\mu\bar p/\bar{\dot\gamma}$ 是不同的量。联合模型的切线同样由 $\eta(p,\mathbf D)$ 的导数给出；$p_*$ 只通过 $I_c$ 进入，若启用压力正则化则同一正则化压力必须在所有对应分母中一致使用。各项在装配时用同一求导实现。

**命题 5.8（局部短波稳定性判据）。** 对均匀基态与平面波 $\delta\mathbf v=\hat{\mathbf v}e^{\lambda t+i\mathbf k\cdot\mathbf x}$，$\mathbf k\cdot\hat{\mathbf v}=0$，设切线声学张量为 $\mathsf A(\mathbf k)=k_j\mathbb C_{ijkl}k_l$，$\mathbb C=\partial\boldsymbol\tau/\partial\nabla\mathbf v$（不可压缩下压力扰动由 $\mathbf k\cdot$ 动量方程确定，不独立）。则 $\lambda$ 是 $\mathbb P(\mathbf k)\mathsf A(\mathbf k)$ 在 $\operatorname{Ran}\mathbb P(\mathbf k)$ 上的特征值的 $-1/\rho$ 倍，$\mathbb P=\mathbf I-\mathbf k\mathbf k^{\!\top}/|\mathbf k|^2$。Newton 流体（$\mathbb C$ 对应常黏度 $\eta_0$）得 $\lambda=-\eta_0|\mathbf k|^2/\rho<0$。非 Newton 基态下 $\operatorname{Ran}\mathbb P$ 是二维空间，特征方程至多为二次；该判据只在局部平面波近似下成立，球壳边界与曲率会改变谱。

---

## 5.4 极向与环向试验函数下的弱投影块

固定速度展开 $\mathbf v=\sum_\alpha u_\alpha\mathbf B_\alpha$，其中 $\mathbf B_\alpha\in\{\mathcal P[\phi_\alpha],\ \mathcal T[\psi_\alpha]\}$（$\phi_\alpha,\psi_\alpha$ 为给定的 $\ell\ge1$ 基函数，$\alpha$ 标记类型、$(\ell,m)$ 与径向序号）。

**命题 5.9（测试空间与方程分块）。** 取试验函数分三类：

- 极向 $\mathbf w_{\mathrm{pol}}=\mathcal P[\phi_t]$，环向 $\mathbf w_{\mathrm{tor}}=\mathcal T[\psi_t]$，二者无散，$\int p\,\nabla\cdot\mathbf w=0$；
- 梯度型 $\mathbf w=\nabla\chi$，$\chi\in H^2(\mathcal V)$，$\chi|_\Gamma=0$（此时 $\nabla\chi|_\Gamma=(\partial_n\chi)\mathbf n$）。

则命题 5.2 的弱式分解为：

**(i) 极向方程**（对每个 $\phi_t$）

$$
\sum_\beta\Big[M_{P\beta}\dot u_\beta+C_{P\beta}u_\beta\Big]+N_P(\mathbf v;\mathbf v)+K_P(\mathbf v,p)
=F_P,
$$

**(ii) 环向方程**（对每个 $\psi_t$）

$$
\sum_\beta\Big[M_{T\beta}\dot u_\beta+C_{T\beta}u_\beta\Big]+N_T(\mathbf v;\mathbf v)+K_T(\mathbf v,p)
=F_T,
$$

**(iii) 压力方程**（对每个 $\chi$）

$$
\int_{\mathcal V}\rho\big(\partial_t\mathbf v+(\mathbf v\cdot\nabla)\mathbf v+2\boldsymbol\Omega\times\mathbf v\big)\cdot\nabla\chi\,dV
+\int_{\mathcal V}\boldsymbol\tau:\nabla\nabla\chi\,dV-\int_{\mathcal V}p\,\Delta\chi\,dV
=\int_{\mathcal V}\rho\,\mathbf g_0\cdot\nabla\chi\,dV+\oint_\Gamma(\mathbf t_{\mathrm{ext}}\cdot\mathbf n)\,\partial_n\chi\,dS .
$$

其中对 $X\in\{P,T\}$，试验块 $\mathbf B_X$ 取 $\mathcal P[\phi_t]$ 或 $\mathcal T[\psi_t]$，

$$
M_{X\beta}=\int_{\mathcal V}\rho\,\mathbf B_X\cdot\mathbf B_\beta\,dV,\qquad
C_{X\beta}=2\int_{\mathcal V}\rho\,(\boldsymbol\Omega\times\mathbf B_\beta)\cdot\mathbf B_X\,dV,
$$

$$
N_X=\int_{\mathcal V}\rho\,(\mathbf v\cdot\nabla\mathbf v)\cdot\mathbf B_X\,dV,\qquad
K_X(\mathbf v,p)=\int_{\mathcal V}\boldsymbol\tau(\mathbf v,p):\mathbf D(\mathbf B_X)\,dV,
$$

$$
F_X=\int_{\mathcal V}\rho\,\mathbf g_0\cdot\mathbf B_X\,dV+\oint_\Gamma\mathbf t_{\mathrm{ext}}\cdot\mathbf B_X\,dS .
$$

压力 $p\in L^2$ 在 (i)(ii) 中仅通过 $\boldsymbol\tau(\mathbf v,p)$ 出现，在 (iii) 中还通过 $-\int p\Delta\chi$ 出现；它是和 $\mathbf v$ 联立的独立未知量，混合弱式中的试探函数 $q$ 给出约束，不存在只含速度势的闭合标量方程。

**命题 5.10（梯度分量为零）。** 设速度空间为 $\mathbb V_\sigma\oplus\nabla X$，$X=\{\chi\in H^2:\chi|_\Gamma=0\}$，并设压力试探空间 $Q_h\supseteq\Delta X$。则约束 $\int q\,\nabla\cdot\mathbf v=0$ 对 $q=\Delta\chi_v$（$\mathbf v=\mathbf v_\sigma+\nabla\chi_v$）给出 $\|\Delta\chi_v\|_{L^2}^2=0$，从而 $\Delta\chi_v=0$；结合 $\chi_v|_\Gamma=0$ 得 $\chi_v\equiv0$，即解的速度无梯度分量。

*证明。* $\nabla\cdot\mathbf v=\Delta\chi_v$。取 $q=\Delta\chi_v$ 即得；调和且边界为零的函数恒为零。∎

---

# 6. 变密度：质量交叉耦合与斜压源

密度取 $\rho=\rho_K$（第 1.1 节）；若 $\rho$ 随流动输运，则服从 5.1 节的输运方程。本章把变密度在极向与环向方程之间的全部耦合渠道写成显式积分。

---

## 6.1 Cauchy 密度导数

令 $q_k(\mathbf x)=1+\sigma_k^2|\mathbf x-\mathbf c_k|^2$，$\rho_K=\sum_kw_kq_k^{-\alpha_k}$，则

$$
\nabla\rho_K=-2\sum_k\alpha_kw_k\sigma_k^2(\mathbf x-\mathbf c_k)\,q_k^{-\alpha_k-1},
$$

$$
\nabla^2\rho_K=\sum_kw_k\Big[-2\alpha_k\sigma_k^2q_k^{-\alpha_k-1}\mathbf I+4\alpha_k(\alpha_k+1)\sigma_k^4(\mathbf x-\mathbf c_k)(\mathbf x-\mathbf c_k)^{\!\top}q_k^{-\alpha_k-2}\Big],
$$

$$
\Delta\rho_K=\sum_kw_k\Big[-6\alpha_k\sigma_k^2q_k^{-\alpha_k-1}+4\alpha_k(\alpha_k+1)\sigma_k^4|\mathbf x-\mathbf c_k|^2q_k^{-\alpha_k-2}\Big].
$$

---

## 6.2 质量块与黏性块的极向—环向交叉耦合

**命题 6.1（选择规则）。** 设权函数（$\rho$ 或 $\eta$）展开为 $\sum_{L,M}a_{LM}(r)Y_{LM}$。极向 $\mathcal P_{\ell m}$ 与环向 $\mathcal T_{\ell'm'}$ 的双线性型
$M_{PT}=\int\rho\,\mathcal P\cdot\mathcal T$，$K_{PT}=\int2\eta\,\mathbf D(\mathcal P):\mathbf D(\mathcal T)$
只在下列必要条件同时成立时才可能非零：

$$
\ell+\ell'+L\ \text{为奇数},\qquad |\ell-\ell'|\le L\le\ell+\ell',\qquad m'=m+M .
$$

球对称权（$L=0$ 的唯一分量）下 $M_{PT}=K_{PT}=0$，极向与环向块对角分离。

*证明。* 球对称权下，$\mathcal P_{\ell m}$ 与 $\mathcal T_{\ell m}$ 属于总角动量 $J=\ell$ 的同一表示，宇称分别为 $(-1)^{\ell+1}$ 与 $(-1)^\ell$，相反，因此在 $J$ 守恒的内积下正交（命题 5.7 给出了质量块的直接验证）。权中含 $Y_{LM}$ 时，宇称因子 $(-1)^L$ 与 Gaunt 三角规则给出上述条件，$M$ 守恒给出 $m'=m+M$。∎

**质量块的显式交叉表达式。** 对 $\Phi=gY_{\ell m}$，$\Psi=fY_{\ell'm'}$，

$$
M_{PT}=\int_0^R r\,(rg)'\,f\ \Big[\int_{S^2}\rho\ \mathbf e_r\cdot\big(\nabla_{S^2}Y_{\ell m}\times\nabla_{S^2}Y_{\ell'm'}\big)\,d\hat{\boldsymbol\omega}\Big]dr ,
$$

方括号是 $\rho$ 的角向结构与两个球谐梯度的 Poisson 括号的积分。因此物质有偶极及更高阶角向不对称（$L\ge1$）、或星形边界 $R(\hat{\boldsymbol\omega})$ 不球对称（积分域随方向变化）时，$M_{PT}\neq0$ 一般成立，极向与环向方程通过质量算子相耦合。

---

## 6.3 斜压源项

**命题 6.2（弱形式斜压源）。** 对任意光滑向量势 $\mathbf A$ 与测试速度 $\mathbf w=\nabla\times\mathbf A$，

$$
\boxed{
\int_{\mathcal V}\rho\,\mathbf g_0\cdot\nabla\times\mathbf A\,dV=\int_{\mathcal V}\mathbf A\cdot\nabla\times(\rho\mathbf g_0)\,dV+\oint_\Gamma\mathbf n\cdot(\mathbf A\times\rho\mathbf g_0)\,dS .}
$$

当 $\mathbf g_0=\nabla W$ 时 $\nabla\times(\rho\nabla W)=\nabla\rho\times\nabla W$。

*证明。* 由 $\nabla\cdot(\mathbf A\times\mathbf F)=\mathbf F\cdot(\nabla\times\mathbf A)-\mathbf A\cdot(\nabla\times\mathbf F)$，取 $\mathbf F=\rho\mathbf g_0$ 并积分。∎（数值复核：取多项式场，两边在单位球上相对误差 $<10^{-15}$。）

**推论 6.3（极向与环向源）。** 取 $\mathbf A_{\mathrm{pol}}=\nabla\phi_t\times\mathbf r$（此时 $\nabla\times\mathbf A_{\mathrm{pol}}=\mathcal P[\phi_t]$），$\mathbf A_{\mathrm{tor}}=\psi_t\mathbf r$（$\nabla\times\mathbf A_{\mathrm{tor}}=\mathcal T[\psi_t]$），$\mathbf g_0=\nabla W$：

$$
F^{\mathrm{bc}}_{\mathrm{pol}}[\phi_t]=\int_{\mathcal V}\nabla\phi_t\cdot\big[\nabla\rho\,(\mathbf r\cdot\nabla W)-\nabla W\,(\mathbf r\cdot\nabla\rho)\big]dV+\oint_\Gamma\rho\,(\nabla\phi_t\times\mathbf r)\cdot(\nabla W\times\mathbf n)\,dS,
$$

$$
F^{\mathrm{bc}}_{\mathrm{tor}}[\psi_t]=\int_{\mathcal V}\psi_t\ \mathbf r\cdot(\nabla\rho\times\nabla W)\,dV+\oint_\Gamma\rho\,\psi_t\ \mathbf r\cdot(\nabla W\times\mathbf n)\,dS .
$$

球坐标下 $\mathbf r\cdot(\nabla\rho\times\nabla W)=\dfrac1r\,\mathbf e_r\cdot(\nabla_{S^2}\rho\times\nabla_{S^2}W)$，即只有 $\rho$ 与 $W$ 的角向梯度不平行才驱动环向流。极向的第一式用 $\mathbf r\times(\nabla\rho\times\nabla W)=\nabla\rho(\mathbf r\cdot\nabla W)-\nabla W(\mathbf r\cdot\nabla\rho)$ 得到。

**命题 6.4（斜压源的模态选择规则）。** 若 $\rho$ 与 $W$ 的角向谐波阶数分别为 $L_1,L_2$，则环向源在阶数 $\ell'$ 上非零须满足 $L_1+L_2+\ell'$ 为奇数；这来自 $\mathbf e_r\cdot(\nabla_{S^2}Y\times\nabla_{S^2}Y')$ 在宇称下的反对称性。当 $\rho=\rho(W)$（等密度面与等势面重合）时 $\nabla\rho\times\nabla W=\mathbf 0$，体积源为零，仅余表面项。

---

# 7. 旋转：Coriolis 交叉耦合与完整非线性块系统

---

## 7.1 Coriolis 项的旋度恒等式

**引理 7.1。** 对 $\nabla\cdot\mathbf v=0$ 与常向量 $\boldsymbol\Omega=\omega\mathbf e_z$，

$$
\nabla\times(\boldsymbol\Omega\times\mathbf v)=-(\boldsymbol\Omega\cdot\nabla)\mathbf v=-\omega\,\partial_z\mathbf v .
$$

*证明。* $\nabla\times(\boldsymbol\Omega\times\mathbf v)=\boldsymbol\Omega(\nabla\cdot\mathbf v)-\mathbf v(\nabla\cdot\boldsymbol\Omega)+(\mathbf v\cdot\nabla)\boldsymbol\Omega-(\boldsymbol\Omega\cdot\nabla)\mathbf v$，前三项为零。∎（已对极向场符号复核。）

**命题 7.2（Coriolis 的弱形式）。** 对 $\mathbf w=\nabla\times\mathbf A$，

$$
C(\mathbf w,\mathbf v):=2\int_{\mathcal V}\rho\,(\boldsymbol\Omega\times\mathbf v)\cdot\mathbf w\,dV
=\int_{\mathcal V}\mathbf A\cdot\big[2\nabla\rho\times(\boldsymbol\Omega\times\mathbf v)-2\omega\rho\,\partial_z\mathbf v\big]dV+2\oint_\Gamma\mathbf n\cdot\big(\mathbf A\times\rho\,\boldsymbol\Omega\times\mathbf v\big)dS,
$$

并且 $C$ 关于 $(\mathbf w,\mathbf v)$ 反对称：$C(\mathbf w,\mathbf v)=-C(\mathbf v,\mathbf w)$，$C(\mathbf v,\mathbf v)=0$。

*证明。* 命题 6.2 取 $\mathbf F=2\rho\,\boldsymbol\Omega\times\mathbf v$，再由 $\nabla\times(\rho\mathbf F_0)=\nabla\rho\times\mathbf F_0+\rho\nabla\times\mathbf F_0$ 与引理 7.1。反对称来自 $(\boldsymbol\Omega\times\mathbf v)\cdot\mathbf w=-(\boldsymbol\Omega\times\mathbf w)\cdot\mathbf v$。∎

所以 Coriolis 力不做功，不改变能量收支，只改变能量在模态之间的分配。

---

## 7.2 极向与环向的选择规则与显式交叉系数

**命题 7.3（选择规则）。** 取 $R$ 为常数、$\rho=\rho(r)$。则 $C$ 在 $(\ell,m)$ 球谐展开下满足：

1. $m$ 守恒（$\boldsymbol\Omega\times$ 对绕 $z$ 轴旋转不变）；
2. 极向—极向与环向—环向块只连接同一 $\ell$；
3. 极向 $\ell$ 与环向 $\ell'$ 之间的块非零的**必要**选择规则为 $\ell'=\ell\pm1$（具体系数取决于角向重叠积分，可能因之为零）。

*证明。* $\boldsymbol\Omega\times$ 是秩 1 的矢量算子，由 Wigner–Eckart 定理总角动量 $J$ 的改变量 $\Delta J\in\{0,\pm1\}$，其中极向与环向场的 $J=\ell$。极向场宇称为 $(-1)^{\ell+1}$，环向为 $(-1)^\ell$，而 $\boldsymbol\Omega\times$ 保持宇称。因此同型之间 $\Delta J=0$，异型之间 $\Delta J=\pm1$。∎

**命题 7.4（显式交叉系数）。** 设 $\mathcal V$ 为球 $r<R$（$R$ 常数）、$\rho=\rho(r)$，极向势 $\Phi=g(r)Y_{\ell m}$，环向测试势 $\psi_t=f(r)Y_{\ell'm}$，$\ell'=\ell\pm1$。令

$$
h=\Lambda_\ell g-(rg)',\quad
\mathcal L^{+}_\ell h=h'-\frac{\ell h}{r},\quad
\mathcal L^{-}_\ell h=h'+\frac{(\ell+1)h}{r},\quad
\Delta_\ell g=g''+\frac{2g'}{r}-\frac{\Lambda_\ell g}{r^2},\quad
a_{\ell m}=\sqrt{\frac{\ell^2-m^2}{4\ell^2-1}} .
$$

则

$$
\boxed{
C_{TP}=-2\omega\int_0^R\rho\,f\ \kappa\,\big[\mathcal L^{\pm}_\ell h+r\,\Delta_\ell g\big]\,r^2dr,}\qquad
\kappa=\begin{cases}a_{\ell+1,m},&\ell'=\ell+1\ (\mathcal L^{+}),\\[2pt] a_{\ell m},&\ell'=\ell-1\ (\mathcal L^{-}),\end{cases}
$$

而 $C_{PT}=-C_{TP}^{\!\top}$。

*推导。* 取 $\mathbf A=\psi_t\mathbf r$，$\mathbf F=\mathcal P[\Phi]$。球对称 $\rho$ 下 $\nabla\rho\parallel\mathbf r$，故 $\mathbf r\cdot(\nabla\rho\times\cdot)=0$；表面项中 $\mathbf n\parallel\mathbf r$，$\mathbf n\cdot(\psi_t\mathbf r\times\cdot)=0$。命题 7.2 化为 $C_{TP}=-2\omega\int\rho\,\psi_t\,\mathbf r\cdot\partial_z\mathbf v\,dV$。而 $\mathbf r\cdot\partial_z\mathbf v=\partial_z(\mathbf r\cdot\mathbf v)-v_z$，$\mathbf r\cdot\mathbf v=L^2\Phi$，$v_z=\partial_z\partial_r(r\Phi)-z\Delta\Phi$（引理 5.5），所以 $\mathbf r\cdot\partial_z\mathbf v=\partial_z(L^2\Phi-\partial_r(r\Phi))+z\Delta\Phi=\partial_z(hY_{\ell m})+r\cos\theta\,\Delta_\ell g\,Y_{\ell m}$。再用

$$
\partial_z\big(h\,Y_{\ell m}\big)=a_{\ell+1,m}\,\mathcal L^{+}_\ell h\ Y_{\ell+1,m}+a_{\ell m}\,\mathcal L^{-}_\ell h\ Y_{\ell-1,m},\qquad
\cos\theta\,Y_{\ell m}=a_{\ell+1,m}Y_{\ell+1,m}+a_{\ell m}Y_{\ell-1,m},
$$

对 $Y_{\ell\pm1,m}$ 投影即得（第二式由 $r^\ell Y_{\ell m}$ 的梯度公式与 $\cos\theta$ 递推确认：$h=r^\ell$ 时 $\mathcal L^{+}h=0$）。∎

**数值复核。** 取 $\rho=1+r^2$，$R=1$，$m=0$：

- $\ell=1\to\ell'=2$（$g=r^2(1+r)$，$f=r^3$）：三重积分直接计算 $-2.1811658591$，公式 $-2.1811658591$；
- $\ell=2\to\ell'=1$（$g=r^3(1+r)$，$f=r^2$）：直接 $-9.6013101430$，公式 $-9.6013101430$；
- $\ell=1\to\ell'=3$ 与 $\ell=1\to\ell'=1$（同阶极向—极向）：均为 $O(10^{-16})$，符合选择规则。

**推论 7.5（环向分量不可省略）。** 当 $\omega\neq0$ 且 $[\mathcal L^{\pm}_\ell h+r\Delta_\ell g]$ 不恒为零时，环向方程中仅由极向系数产生的项 $C_{TP}\,u_P$ 非零，环向方程的残差无法被极向系数抵消，所以环向系数 $s_{\ell'km}$ 必须作为独立未知量保留。同理，6.3 节的斜压体积源 $F^{\mathrm{bc}}_{\mathrm{tor}}\neq0$（$\rho$ 与 $W$ 的角向梯度不平行）也只能由环向系数承担。

---

## 7.3 完整非线性块系统

记速度系数向量 $\mathbf u=(\mathbf q,\mathbf s)$（极向、环向），则命题 5.9 的 (i)(ii) 合写为 $2\times2$ 块系统：

$$
\boxed{
\begin{pmatrix}M_{PP}&M_{PT}\\M_{TP}&M_{TT}\end{pmatrix}\!
\begin{pmatrix}\dot{\mathbf q}\\\dot{\mathbf s}\end{pmatrix}
+\begin{pmatrix}C_{PP}&C_{PT}\\C_{TP}&C_{TT}\end{pmatrix}\!
\begin{pmatrix}\mathbf q\\\mathbf s\end{pmatrix}
+\begin{pmatrix}N_P\\N_T\end{pmatrix}
+\begin{pmatrix}K_P(\mathbf u,p)\\K_T(\mathbf u,p)\end{pmatrix}
=\begin{pmatrix}F_P\\F_T\end{pmatrix},}
$$

其中各块的积分由 5.9 给出，并满足：

| 块 | 来源 | 对称性与耦合渠道 |
|---|---|---|
| $M_{XY}$ | 惯性 | 对称正定（$\ell\ge1$）；$M_{PT}\neq0$ 来自密度或边界的角向不对称（命题 6.1） |
| $C_{XY}$ | Coriolis | 整体反对称；$C_{PT},C_{TP}$ 连接 $\ell\leftrightarrow\ell\pm1$（命题 7.3、7.4）；$C_{PP},C_{TT}$ 同阶 |
| $K_X$ | 黏性 | 依赖 $(\mathbf u,p)$；$K_{PT}\neq0$ 来自 $\eta$ 的角向不均匀（命题 6.1）；Newton 切线见 5.3 节 |
| $N_X$ | 对流 | 三线性，极向、环向分量互相耦合 |
| $F_X$ | 引力、离心、斜压、牵引 | 斜压项 $F^{\mathrm{bc}}_X$ 见推论 6.3；表面牵引项见推论 5.3 |

并与压力方程 (iii) 及约束 $\int q\,\nabla\cdot\mathbf v=0$（命题 5.10）、密度输运、边界运动学（5.1 节）联立。基本未知量 $(\mathbf v,p,\rho,R)$ 都出现，压力是独立未知量。

**移动基函数的几何连接项与 ALE 弱式。** 当边界 $R(\hat{\boldsymbol\omega},t)$ 随时间变形时，基函数 $\mathbf B_\alpha$ 经 $\zeta=r/R$ 与 $R^2,R$ 因子依赖 $R(t)$，其内部时间导数 $\partial_t\mathbf B_\alpha$（固定 $\mathbf x$）在惯性项中诱导体积分矩阵

$$
(\mathbf G)_{\beta\alpha}=\int_{\mathcal V}\rho\,\mathbf B_\beta\cdot\partial_t\mathbf B_\alpha\,dV,\qquad
\partial_t\mathbf M_\Gamma=\mathbf G+\mathbf G^{\!\top}+\mathbf M_{\partial\mathcal V}+\mathbf M_{\partial_t\rho},
$$

其中 $\mathbf M_{\partial\mathcal V}$ 为 Reynolds 边界项（9.1 节）。$\mathbf G$ **不是**可以并入表面速度 $\mathbf c$ 的项，其代数结构按模式区分：

- **准静态主线（模式 B）：** 惯性块 $\mathbf M$、$\mathbf G$、$\mathbf N$ 全部不出现（2.8 节 B1）；每个地质步在当前冻结几何 $R^{(n)}$ 上重新装配并求解 Stokes 系统（8.4′），速度不跨步携带，故几何跳变后无需任何模态重映射。
- **动态自由表面：** 取 ALE 网格速度场 $\mathbf w_{\rm mesh}(\mathbf x)=\dfrac rR(\partial_tR)\,\mathbf e_r$（保持 $\hat{\boldsymbol\omega}$、$\zeta$ 不变的径向拉伸，边界上 $\mathbf w_{\rm mesh}\cdot\mathbf n=\partial_tR\,(\mathbf n\cdot\mathbf e_r)$ 与命题 9.1 一致）。本文源项采用固定空间坐标的 Euler 弱式：$\mathbf G$ 按上式保留，惯性对流项仍使用物理速度 $\mathbf v$。若实现改用 ALE 时间导数，则必须同时把 $\partial_t|_{\mathbf x}$ 改成 $\partial_t|_{\rm ALE}$ 并以 $\mathbf v-\mathbf w_{\rm mesh}$ 替换对流速度；该替代形式不得与固定空间的 $\mathbf G$ 再次叠加。因 $\nabla\cdot\mathbf w_{\rm mesh}\ne0$，Reynolds 边界项、$\mathbf G$ 和密度项仍须显式装配并计入源项 $\boldsymbol{\mathcal S}$（9.1 节）。

**注（边界项的几何）。** 当 $R=R(\hat{\boldsymbol\omega})$ 时，7.4 节的解析简化不再适用，积分域与 $\mathbf n$ 随方向变化，各块须直接按 5.9 的积分在 $\mathcal V(R)$ 上求积。

---

## 7.4 刚体模式与可解性

**命题 7.6。** 在命题 5.6 的 6 个刚体模式上，黏性块与压力耦合块均为零。因此：

1. 准静态问题（略去 $M\dot{\mathbf u}$、$N$）存在解的必要条件是外力对刚体模式不做功：

$$
\int_{\mathcal V}\rho\,\mathbf g_0\,dV+\oint_\Gamma\mathbf t_{\mathrm{ext}}\,dS=\mathbf 0,\qquad
\int_{\mathcal V}\rho\,\mathbf x\times\mathbf g_0\,dV+\oint_\Gamma\mathbf x\times\mathbf t_{\mathrm{ext}}\,dS=\mathbf 0 .
$$

自引力的合力恒为零；离心力要求质心在转轴上、转轴是惯量主轴（$I_{xz}=I_{yz}=0$）；

2. 解仅确定到刚体模式，固定方式取 $\int_{\mathcal V}\rho\,\mathbf v\,dV=\mathbf 0$、$\int_{\mathcal V}\rho\,\mathbf x\times\mathbf v\,dV=\mathbf 0$（6 个约束）；

*证明。* 1：取 $\mathbf w$ 为刚体模式，$\mathbf D(\mathbf w)=0$、$\nabla\cdot\mathbf w=0$，命题 5.2 中应力项与压力项均为零，只剩外力项，其必须为零。2：刚体模式是黏性块的零空间，需额外约束定解。∎

## 7.5 适用范围

本章的块系统是 $(\mathbf v,p,\rho,R)$ 的完整动态弱式，不引入任何关于 $\mathbf v$ 的几何限制：极向与环向两个势同时存在，压力作为独立未知量通过 $\boldsymbol\tau(\mathbf v,p)$ 与 $-p\,\nabla\cdot\mathbf w$ 进入。$\mu(I)$ 关系关于 $\mathbf D$ 非单调，所以本章不断言连续问题的存在唯一性；块系统的离散可解性取决于离散空间的 inf–sup 条件与本构的局部适定性，在第 8 章以后的离散模型中单独处理。**生产主线只使用本章的黏性块与载荷块：** 准静态系统（8.4′）去掉 $\mathbf M\dot{\mathbf u}$、对流与 Coriolis 块，密度固定（假设 A-ρ），压力由第 8.7 节 Poisson 问题独立给出；完整块系统仅用于动态模式（第 9.9 节）。

**总结说明。** 本模型在连续层面上完整给出了包含极向—环向双势、变密度、Coriolis 耦合与独立压力的全套物理控制方程与自然边界条件；在离散层面上，准静态主线与动态模式在 Track A（全局求解）与 Track B（球谐块对角代理，附录 A.4）下均给出了代数形式。对于非单调流变引起的深层分析性质，以及离散适定性，已通过显式登记证明义务（PO-1、PO-11、PO-12；动态模式另有 PO-2、PO-4；Track B 为 PO-8）划定理论边界，这些义务未被证明，文中依赖它们的结论均已标注。模型在计算几何自洽场（SCF）的科学范畴内给出了闭环，其自洽性限于所登记的假设与证明义务之内。

# 8. 径向 Galerkin 降阶与空间离散装配

本章把第 5–7 章的连续混合弱式（命题 5.2、5.9、5.10 及 7.3 节的完整块系统）投影到有限维空间 $\mathbb V_h\times\mathbb Q_h$。径向方向采用 $K=4$ 个 Jacobi 模式，角向方向采用球谐展开，两者都是 Galerkin 截断：所得系统是三维问题的降阶模型，不是它的恒等变换。符号、空间与迹空间服从本文附录 A（A.1 符号消歧、A.2 函数空间与迹空间）。

---

## 8.1 星形域与 Jacobi 径向基

**定义 8.1（几何）。** 设 $R\in C^3(S^2\times[0,T])$，$0<R_{\min}\le R\le R_{\max}$，物质域为星形域

$$
\mathcal V(R)=\{(r,\hat{\boldsymbol\omega}):0\le r<R(\hat{\boldsymbol\omega},t),\ \hat{\boldsymbol\omega}\in S^2\},\qquad
\zeta=\frac{r}{R(\hat{\boldsymbol\omega},t)}\in[0,1).
$$

**定义 8.2（径向基）。** 对 $\ell\ge1$、$k=0,1,2,3$，

$$
\boxed{\ \psi_{\ell k}(\zeta)=\zeta^{\ell}\,P_k^{(0,\,2\ell+2)}(2\zeta-1),\qquad \Lambda_\ell=\ell(\ell+1),\ }
$$

其中 $P_k^{(\alpha,\beta)}$ 是标准 Jacobi 多项式。

**命题 8.3（正交性、端点值与幂次结构）。**

1. $\displaystyle\int_0^1\psi_{\ell j}\,\psi_{\ell k}\,\zeta^2\,d\zeta=\frac{\delta_{jk}}{2k+2\ell+3}$；
2. $\psi_{\ell k}(1)=1$，$\psi_{\ell k}'(1)=\ell+k(k+2\ell+3)$，$\psi_{\ell k}''(1)$ 见第 10.2.1 节的表；
3. $\psi_{\ell k}(\zeta)=\sum_{j=0}^{k}a_{\ell kj}\,\zeta^{\ell+j}$，且 $a_{\ell k0}=(-1)^k$，$a_{\ell kk}\ne0$。

*证明。* 1：令 $x=2\zeta-1$，$\zeta=(1+x)/2$，则 $\psi_{\ell j}\psi_{\ell k}\zeta^2d\zeta=2^{-(2\ell+3)}(1+x)^{2\ell+2}P_j^{(0,2\ell+2)}P_k^{(0,2\ell+2)}dx$。这正是 Jacobi 权 $(1-x)^{\alpha}(1+x)^{\beta}$（$\alpha=0$，$\beta=2\ell+2$）。其范数为 $2^{\alpha+\beta+1}/(2k+\alpha+\beta+1)$（$\alpha=0$ 时 Gamma 因子相消），乘以 $2^{-(2\ell+3)}$ 即得。2、3：直接代入 $P_k^{(0,\beta)}(-1)=(-1)^k$ 与首项系数非零；端点值已用符号计算复核。∎

参数次序 $(2\ell+2,0)$ 对应权 $(1-x)^{2\ell+2}$，与球坐标体积元 $\zeta^2d\zeta$ 乘以 $\zeta^{2\ell}$ 得到的权 $\zeta^{2\ell+2}$ 不匹配，不再使用。

**命题 8.4（中心正则性；证明义务 PO-3，见附录 A.5）。**[中心流变弱可积性：$\ell=1$、非常数 $R$ 时 $\nabla\mathbf v=O(r^{-1})$ 非 $L^\infty$，但当 $\mathbf D\sim r^{-1}$ 时 $\dot\gamma\sim r^{-1}$，黏度 $\eta\propto1/\dot\gamma\sim r$，偏应力 $\boldsymbol\tau=2\eta\mathbf D\sim O(1)$，应变能密度 $\boldsymbol\tau:\mathbf D(\mathbf w)\sim r^{-1}$，在体积测度 $r^2dr$ 下 $\int_0^\varepsilon r\,dr<\infty$，弱形式应变能在原点邻域局部弱可积，代数矩阵元可装配（这不是连续或离散问题弱解的存在唯一性结论）；若需流变强解，可切换备选偶次基 $\psi^{\rm ev}$（附录 A.2；仅 $R$ 为常数时有效）。] 对 $\ell\ge1$、$k\le3$ 及 $R\in C^3(S^2)$，基函数

$$
\mathbf V_{\ell km}=\nabla\times\nabla\times\big(R^2\psi_{\ell k}Y_{\ell m}\,\mathbf r\big),\qquad
\mathbf W_{\ell km}=\nabla\times\big(R\,\psi_{\ell k}Y_{\ell m}\,\mathbf r\big)
$$

属于 $H^1(\mathcal V)$。其中 $\mathbf W_{\ell km}$ 在原点附近 Lipschitz；$\mathbf V_{\ell km}$ 当 $R$ 为常数，或 $\ell\ge2$ 时在原点附近 Lipschitz；仅当 $\ell=1$ 且 $R$ 非常数时，它在原点的邻域内只保证有界，其梯度按 $O(r^{-1})$ 增长（属于 $L^2$，不属于 $L^\infty$）。

*证明。* 由命题 8.3(3)，在 $r<R_{\min}$ 内，$R^2\psi_{\ell k}(r/R)Y_{\ell m}=\sum_ja_{\ell kj}R^{2-n}\,r^{n}Y_{\ell m}$，$n=\ell+j\ge1$。每一项是 $r$ 的 $n$ 次齐次函数 $\Phi_n=r^nh_n(\hat{\boldsymbol\omega})$，$h_n=a_{\ell kj}R^{2-n}Y_{\ell m}\in C^3$。由引理 5.5，$\mathcal P[\Phi_n]=\nabla(\partial_r(r\Phi_n))-\mathbf r\Delta\Phi_n=(n+1)\nabla\Phi_n-\mathbf r\Delta\Phi_n$，故它是 $n-1$ 次齐次函数，其梯度是 $n-2$ 次齐次函数：$|\nabla\mathcal P[\Phi_n]|\le Cr^{n-2}$。$\int_0^\varepsilon r^{2(n-2)}r^2dr=\int_0^\varepsilon r^{2n-2}dr<\infty$（$n\ge1$），故属于 $H^1$。$n\ge2$ 时梯度有界，即 Lipschitz；$n=1$ 仅发生在 $\ell=1,j=0$，此时 $\Phi_1=a\,R(\hat{\boldsymbol\omega})\,r\,Y_{1m}$，$R$ 为常数时它是线性函数，$\mathcal P[\Phi_1]$ 为常向量；$R$ 非常数时 $\mathcal P[\Phi_1]$ 是 $0$ 次齐次、依赖方向，梯度 $O(r^{-1})$。环向：$\Psi_n=r^nR^{1-n}Y_{\ell m}a$ 给出 $\mathcal T[\Psi_n]=\nabla\Psi_n\times\mathbf r$，次数为 $n\ge1$，梯度有界。∎

$\zeta^\ell$ 因子使 $\Phi$ 的最低幂次恰为 $n=\ell$：$R$ 为常数时 $r^\ell Y_{\ell m}$ 是 $\mathbf x$ 的调和多项式，因而光滑；其余项 $n>\ell$ 含奇次幂，仅有限光滑，命题 8.4 给出的正是这一光滑度下的 $H^1$ 结论。$\mu(I)$ 在中心处需要有限的 $\dot\gamma$：除上述 $\ell=1$、非常数 $R$ 的特例外，$\mathbf D\in L^\infty$。

---

## 8.2 双势展开与自由度

**定义 8.5（速度展开）。**

$$
\boxed{\ 
\Phi(\mathbf x,t)=R^2(\hat{\boldsymbol\omega},t)\sum_{\ell=1}^{L_{\max}}\sum_{m=-\ell}^{\ell}\sum_{k=0}^{3}q_{\ell km}(t)\,\psi_{\ell k}(\zeta)\,Y_{\ell m}(\hat{\boldsymbol\omega}),\qquad
\Psi(\mathbf x,t)=R(\hat{\boldsymbol\omega},t)\sum_{\ell,m,k}s_{\ell km}(t)\,\psi_{\ell k}(\zeta)\,Y_{\ell m}(\hat{\boldsymbol\omega}),\ }
$$

$$
\mathbf v=\mathcal P[\Phi]+\mathcal T[\Psi]=\sum_{\ell,m,k}\big(q_{\ell km}\mathbf V_{\ell km}+s_{\ell km}\mathbf W_{\ell km}\big).
$$

速度自由度向量为 $\mathbf u=(\mathbf q,\mathbf s)$，$[q]=[s]=\mathrm T^{-1}$。**$K=4$ 的含义**：每个 $(\ell,m)$、每个势各保留 4 个径向模式，所以每个 $(\ell,m)$ 有 $8$ 个速度自由度；总自由度数

$$
N_u=8\sum_{\ell=1}^{L_{\max}}(2\ell+1)=8L_{\max}(L_{\max}+2).
$$

$\ell=0$ 不含速度自由度（命题 5.6(2)）；所有矩阵求逆限于 $\ell\ge1$。$\mathbf V_{\ell km},\mathbf W_{\ell km}$ 精确无散（引理 5.5），$R$ 对它们的依赖进入所有积分，不在节点处冻结。

---

## 8.3 全局动能质量算子

**定义 8.6。** $\displaystyle(\mathbf M_\Gamma)_{\alpha\beta}=\int_{\mathcal V(R)}\rho\,\mathbf B_\alpha\cdot\mathbf B_\beta\,dV$，$\mathbf B_\alpha\in\{\mathbf V_{\ell km},\mathbf W_{\ell km}\}$。

**命题 8.7（对称正定性）**[完整证明，不依赖 PO]。 设 $0<\rho_{\min}\le\rho\le\rho_{\max}$，$R\in C^3$，则 $\mathbf M_\Gamma$ 对称，并且对任何 $\mathbf u\ne\mathbf 0$ 有 $\mathbf u^{\!\top}\mathbf M_\Gamma\mathbf u=\int_{\mathcal V}\rho|\mathbf v|^2dV\ge\rho_{\min}\|\mathbf v\|_{L^2}^2>0$。

*证明。* 对称性与半正定性来自定义。设 $\mathbf v=\mathbf 0$。命题 5.6(1) 给出 $L^2\Phi=r v_r=0$ 与 $L^2\Psi=0$，所以在球 $r<R_{\min}$ 内 $\Phi=\Phi_0(r)$，$\Psi=\Psi_0(r)$ 只依赖 $r$。取 $\Phi$ 的 $r^n$ 系数：$R^{2-n}H_n(\hat{\boldsymbol\omega})=c_n$，$H_n=\sum_{\ell\le n,m}\big(\sum_ka_{\ell k,n-\ell}q_{\ell km}\big)Y_{\ell m}$ 不含 $\ell=0$ 分量。故 $H_n=c_nR^{n-2}$，对两边取球面平均，因 $R>0$ 而 $\langle R^{n-2}\rangle>0$，推出 $c_n=0$，即 $H_n=0$。球谐函数线性无关，故对每个 $(\ell,m)$ 和每个 $n\ge\ell$ 有 $\sum_ka_{\ell k,n-\ell}q_{\ell km}=0$，即多项式 $\sum_kq_{\ell km}\psi_{\ell k}(\zeta)\equiv0$。$\psi_{\ell 0},\dots,\psi_{\ell 3}$ 次数各异，线性无关，所以 $q_{\ell km}=0$。对 $\Psi$（以 $R^{1-n}$ 代替 $R^{2-n}$）同理 $s_{\ell km}=0$。∎

**命题 8.8（球对称参考几何，对角块）**[完整证明，不依赖 PO]。 设 $R\equiv R_0$ 为常数，$\rho=\rho(r)$。则 $\mathbf M_\Gamma$ 对 $(\ell,m)$ 块对角，极向与环向互不耦合，并且（命题 5.7）

$$
\boxed{\ 
(\mathbf M^P_\ell)_{ij}=\Lambda_\ell^2\!\int_0^{R_0}\!\rho\,g_ig_j\,dr+\Lambda_\ell\!\int_0^{R_0}\!\rho\,(rg_i)'(rg_j)'\,dr,\qquad
(\mathbf M^T_\ell)_{ij}=\Lambda_\ell\!\int_0^{R_0}\!\rho\,f_if_j\,r^2dr,\ }
$$

其中 $g_i=R_0^2\psi_{\ell i}(r/R_0)$，$f_i=R_0\psi_{\ell i}(r/R_0)$，每块 $4\times4$。常数密度 $\rho=\rho_0$ 时

$$
\mathbf M^P_\ell=\rho_0R_0^5\,\Lambda_\ell\big(\Lambda_\ell\tilde{\mathbf N}_\ell+\tilde{\mathbf P}_\ell\big),\qquad
\mathbf M^T_\ell=\rho_0R_0^5\,\Lambda_\ell\,\mathrm{diag}\Big(\frac1{2k+2\ell+3}\Big),
$$

$$
(\tilde{\mathbf N}_\ell)_{ij}=\int_0^1\psi_{\ell i}\psi_{\ell j}\,d\zeta,\qquad
(\tilde{\mathbf P}_\ell)_{ij}=\int_0^1(\zeta\psi_{\ell i})'(\zeta\psi_{\ell j})'\,d\zeta .
$$

环向块在 Jacobi 基下严格对角（命题 8.3(1)）；极向块不是对角，但 $\tilde{\mathbf N}_\ell,\tilde{\mathbf P}_\ell$ 对称正定（$\psi_{\ell i}$ 与 $(\zeta\psi_{\ell i})'$ 线性无关）。数值见第 10.1 节。所有块元素都是 $L^2$ 的多项式（极向为二次，环向为一次）——这就是 $\mathbf M_\Gamma$ 是角向微分算子、不是节点局部常数的原因。

**命题 8.9（一般星形表面，Gaunt 耦合）。** 当 $R=R(\hat{\boldsymbol\omega})$ 或 $\rho$ 有角向依赖时，$(\mathbf M_\Gamma)_{\alpha\beta}$ 的角向积分含权函数 $w(\hat{\boldsymbol\omega})$ 与 $Y_{\ell m}Y_{\ell'm'}$（及其表面梯度）的乘积。设 $w=\sum_{L,M}w_{LM}Y_{LM}$，则 $(\ell,m)$ 与 $(\ell',m')$ 之间的块非零须满足 Gaunt 条件

$$
\int Y_{\ell m}Y_{\ell'm'}Y_{LM}\,d\Omega\ne0\ \Rightarrow\ |\ell-\ell'|\le L\le\ell+\ell',\ \ \ell+\ell'+L\ \text{为偶数},\ \ M=-m-m' .
$$

权函数带限（$w_{LM}=0,\ L>L_w$）时 $\mathbf M_\Gamma$ 为严格带状稀疏块矩阵，块带宽 $|\ell-\ell'|\le L_w$；一般光滑的 $R$ 与 $\rho$ 下 $w_{LM}$ 随 $L$ 衰减，可按容差 $\varepsilon_w$ 截断为数值带宽。极向与环向之间的交叉块遵循命题 6.1 的选择规则（宇称：$\ell+\ell'+L$ 为奇数）。

---

## 8.4 离散混合弱式与鞍点系统

**离散空间。**

$$
\mathbb V_h^{\sigma}=\operatorname{span}\{\mathbf V_{\ell km},\mathbf W_{\ell km}\},\qquad
\mathbb V_h^{g}=\{\nabla\chi_h:\chi_h\in X_h\},\ \ X_h\subset\{\chi\in H^2:\chi|_\Gamma=0\},
$$

$$
\mathbb Q_h=\operatorname{span}\{\psi_{\ell k}(\zeta)Y_{\ell m}\},\ \ 0\le\ell\le L_p,\ 0\le k\le K_p-1,
$$

其中压力空间包含 $\ell=0$（静水分量）。压力层把记号扩展为 $\psi_{0k}(\zeta):=P_k^{(0,2)}(2\zeta-1)$；这些 $\ell=0$ 模式只用于压力，不进入速度基。其余取 $L_p\le L_{\max}$，$K_p\le K$；$X_h$ 与 $\mathbb Q_h$ 的匹配是 inf–sup 证明义务（PO-2），本文不声称已证。$\mathbb V_h^\sigma$ 中的函数精确无散，所以 $b(\mathbf w,q)\equiv0$ 对 $\mathbf w\in\mathbb V_h^\sigma$ 成立。

**移动基函数的几何连接项。** 准静态主线无惯性块，$\mathbf G$ 不出现；动态模式的 $\mathbf G$ 与 ALE 相对速度对流项 $\mathbf N_{\rm ALE}$（$\mathbf v_{\rm rel}=\mathbf v-\mathbf w_{\rm mesh}$）见 7.3 节，须在 $\mathbf A_{\sigma\sigma}$ 的 $\mathbf N(\mathbf u)$ 与惯性块中显式装配。

**测试函数投影。** 将命题 5.2 的弱式分别用 $\mathbf w=\mathbf V_{\ell km}$、$\mathbf W_{\ell km}$、$\nabla\chi_h$ 检验，得到命题 5.9 的三组方程：极向 (i)、环向 (ii) 和压力 (iii)。记全速度系数 $\tilde{\mathbf u}=(\mathbf u,\mathbf x)$，$\mathbf x$ 是 $\chi_h$ 的系数，$\mathbf p$ 是压力系数。

**整体系统**[在证明义务 PO-1、PO-2 成立的前提下，其离散可解性才成立]。以下各块由命题 5.9 的积分定义：$M_{X\beta}$（惯性）、$C_{X\beta}$（Coriolis）、$N_X$（对流）、$K_X(\mathbf v,p)$（黏性，含 $\boldsymbol\tau(\mathbf v,p)=2\eta(p,\mathbf D)\mathbf D$）、$F_X$（引力、离心、斜压、牵引及 $\mathbf T_{\rm tide}$ 载荷）。

$$
\boxed{\ 
\begin{pmatrix}\mathbf A(\tilde{\mathbf u},\mathbf p)&-\mathbf B^{\!\top}\\ \mathbf B&\mathbf 0\end{pmatrix}
\begin{pmatrix}\tilde{\mathbf u}\\ \mathbf p\end{pmatrix}
=\begin{pmatrix}\mathbf f_{\rm ext}\\ \mathbf 0\end{pmatrix},\ }
$$

$$
\mathbf A=\begin{pmatrix}\mathbf A_{\sigma\sigma}&\mathbf A_{\sigma g}\\ \mathbf A_{g\sigma}&\mathbf A_{gg}\end{pmatrix},\qquad
\mathbf A_{\sigma\sigma}=\begin{pmatrix}\mathbf M_{PP}/\Delta t+\mathbf C_{PP}+\mathbf K_{PP}&\mathbf M_{PT}/\Delta t+\mathbf C_{PT}+\mathbf K_{PT}\\ \mathbf M_{TP}/\Delta t+\mathbf C_{TP}+\mathbf K_{TP}&\mathbf M_{TT}/\Delta t+\mathbf C_{TT}+\mathbf K_{TT}\end{pmatrix}+\mathbf N(\mathbf u),
$$

$$
\mathbf B=\begin{pmatrix}\mathbf 0&\mathbf B_g\end{pmatrix},\quad (\mathbf B_g)_{q\chi}=\int_{\mathcal V}q\,\Delta\chi\,dV,\qquad
\mathbf f_{\rm ext}=\big(\mathbf F_P,\mathbf F_T,\mathbf F_g\big),
$$

其中 $\mathbf M_{XY}/\Delta t$ 表示时间离散的惯性项（准静态主线取零，见（8.4′）），$\mathbf K_{XY}$ 的 $(\mathbf u,p)$ 依赖通过 5.3 节的 Newton 切线线性化。块的结构：

| 块 | 结构与来源 |
|---|---|
| $\mathbf M_{PP},\mathbf M_{TT}$ | 球对称几何下对 $(\ell,m)$ 对角（命题 8.8），一般几何下为 Gaunt 带状（命题 8.9） |
| $\mathbf M_{PT}$ | 球对称几何下为零；非零由 $\rho$ 或 $R$ 的角向不均匀产生（命题 6.1） |
| $\mathbf C_{PT},\mathbf C_{TP}$ | 连接 $\ell\leftrightarrow\ell\pm1$、$m$ 不变（命题 7.3），系数见命题 7.4；Coriolis 整体反对称 |
| $\mathbf K_{XY}$ | 依赖 $(\mathbf u,p)$ 的黏性块；$\mathbf K_{PT}\ne0$ 来自 $\eta$ 的角向不均匀 |
| $\mathbf F_X$ | 斜压项见命题 6.2、推论 6.3 与命题 6.4；表面牵引是自然边界项（推论 5.3） |

Newton 迭代的 Jacobian 为

$$
\begin{pmatrix}\mathbf A_T&-\mathbf B^{\!\top}+\mathbf K_p\\ \mathbf B&\mathbf 0\end{pmatrix},\qquad
\mathbf K_p=\frac{\partial\mathbf K}{\partial\mathbf p},
$$

$\eta=\eta(p,\mathbf D)$ 使 $\mathbf K_p\ne0$，压力同时出现在 $\mathbf A$ 与 $\mathbf B$ 中，因此 Jacobian 一般不是对称鞍点矩阵。

**准静态生产系统（8.4′）。** 准静态主线（2.8 节 B1）取上述系统在 $\mathbf M=\mathbf N=\mathbf C=\mathbf 0$、$\mathbf G=\mathbf 0$ 下的限制，并把速度空间直接取为精确无散的 $\mathbb V_h^\sigma$（即命题 8.10 的结论 $\mathbf x=\mathbf 0$ 作为空间选择而非定理前提）。命题 5.2 的弱式以 $\mathbf w=\mathbf B_\alpha\in\mathbb V_h^\sigma$ 检验，$\int p\,\nabla\cdot\mathbf w=0$ 精确消失，得

$$
\boxed{\ 
\begin{pmatrix}\mathbf K(\eta)&\mathbf B_{\rm rb}^{\!\top}\\ \mathbf B_{\rm rb}&\mathbf 0\end{pmatrix}
\begin{pmatrix}\mathbf u\\ \boldsymbol\lambda\end{pmatrix}
=\begin{pmatrix}\mathbf F\\ \mathbf 0\end{pmatrix},\qquad
\begin{aligned}
K_{\alpha\beta}(\eta)&=\int_{\mathcal V}2\eta\,\mathbf D(\mathbf B_\beta):\mathbf D(\mathbf B_\alpha)\,dV,\\
F_\alpha&=\int_{\mathcal V}\rho_K\,\mathbf g_{\rm eff}(\omega_n)\cdot\mathbf B_\alpha\,dV-\int_{\mathcal V}\mathbf T_{\rm tide}:\mathbf D(\mathbf B_\alpha)\,dV+\oint_\Gamma\mathbf t_{\rm ext}\cdot\mathbf B_\alpha\,dS,
\end{aligned}\ }
$$

$\mathbf B_{\rm rb}$ 见第 8.5 节与附录 A.3（行 $\int\rho_K\mathbf B_\alpha dV$、$\int\rho_K\mathbf x\times\mathbf B_\alpha dV$）。压力只经 $\eta(\tilde p,\mathbf D)$ 进入 $\mathbf K$，由第 8.7 节的 Poisson 问题确定；二者的非线性耦合由第 10.4.2 节的 Picard 迭代闭合。因速度空间精确无散，本系统**不需要**速度–压力 inf–sup 条件（PO-2 只约束下述全隐式基准）。积分用第 9.4 节的角向求积与权 $\zeta^{2\ell+2}$ 的径向 Gauss–Jacobi 求积；$\mathbf D(\mathbf B_\alpha)$ 由引理 5.5 逐点解析求出。

**引理 8.12（准静态系统的适定性）。** 设 $\rho_K\ge\rho_{\min}>0$，$\eta\ge\eta_{\rm lo}>0$ 几乎处处（2.8 节），$\mathcal V$ 连通，且 $\operatorname{rank}\mathbf B_{\rm rb}=6$。则 $\mathbf K(\eta)$ 在 $\ker\mathbf B_{\rm rb}$ 上对称正定，（8.4′）有唯一解；对任意 $\gamma_{\rm AL}>0$，$\mathbf K_\gamma:=\mathbf K+\gamma_{\rm AL}\mathbf B_{\rm rb}^{\!\top}\mathbf B_{\rm rb}$ 在全空间对称正定，且以 $\mathbf K_\gamma$ 替换 $\mathbf K$ 不改变（8.4′）的解。

*证明。* 对称性由定义。$\mathbf u^{\!\top}\mathbf K\mathbf u=\int2\eta|\mathbf D(\mathbf v)|^2\ge2\eta_{\rm lo}\|\mathbf D(\mathbf v)\|_{L^2}^2$；若为零则 $\mathbf D(\mathbf v)=0$，连通域上 $\mathbf v=\mathbf a+\boldsymbol\theta\times\mathbf x$。若又有 $\mathbf B_{\rm rb}\mathbf u=\mathbf 0$，则 $\int\rho_K|\mathbf v|^2=\mathbf a\cdot\int\rho_K\mathbf v+\boldsymbol\theta\cdot\int\rho_K\mathbf x\times\mathbf v=0$，由 $\rho_K\ge\rho_{\min}$ 得 $\mathbf v=0$，基线性无关给出 $\mathbf u=\mathbf 0$。$\mathbf K$ 在 $\ker\mathbf B$ 上正定且 $\mathbf B$ 行满秩，鞍点系统非奇异（标准结论）。$\mathbf u^{\!\top}\mathbf K_\gamma\mathbf u=0$ 迫使 $\mathbf B\mathbf u=0$ 且 $\mathbf u^{\!\top}\mathbf K\mathbf u=0$，故 $\mathbf u=0$；解满足 $\mathbf B\mathbf u=0$，所以 $\gamma_{\rm AL}\mathbf B^{\!\top}\mathbf B\mathbf u=0$，解不变。∎

**求解（值域空间法，精确到迭代容差）。** 取 $\gamma_{\rm AL}:=\|\operatorname{diag}\mathbf K\|_\infty/\|\operatorname{diag}(\mathbf B_{\rm rb}^{\!\top}\mathbf B_{\rm rb})\|_\infty$。以块对角预条件子 $\mathbf K_{\gamma,B1}^{-1}$（每个 $(\ell,m)$ 一个 $8\times8$ 主子块的逆，附录 A.4）对 $\mathbf K_\gamma$ 作 7 次 PCG：$\mathbf y_0=\mathbf K_\gamma^{-1}\mathbf F$，$\mathbf Y=\mathbf K_\gamma^{-1}\mathbf B_{\rm rb}^{\!\top}\in\mathbb R^{N_u\times6}$；再解 $6\times6$ 对称正定系统 $(\mathbf B_{\rm rb}\mathbf Y)\boldsymbol\lambda=\mathbf B_{\rm rb}\mathbf y_0$，得 $\mathbf u=\mathbf y_0-\mathbf Y\boldsymbol\lambda$。$\eta$ 的空间对比度 $\eta_{\rm hi}/\eta_{\rm lo}$ 决定 PCG 迭代数，须与迭代数一并报告；Newton 加速（用 5.3 节切线）可选，缺省为 Picard。

**与全隐式系统的关系。** 上面的含惯性、对流、Coriolis 与独立压力的整体系统是动态自由表面模式的**理论基准**，准静态主线不求解它；（8.4′）是它在准静态极限下的精确限制。生产模型不含弹性应变或状态方程，压力是不可压缩约束的乘子。

**命题 8.10（条件性离散压力恢复定理；以假设 $\mathbb Q_h\supseteq\Delta X_h$ 及证明义务 PO-2 成立为前提）**[依赖 PO-2：$\mathbb Q_h\supseteq\Delta X_h$ 的 inf–sup 匹配]。 设 $\mathbb Q_h\supseteq\Delta X_h$（命题 5.10 的条件），则 $\mathbf x=\mathbf 0$，速度仅含 $\mathbb V_h^\sigma$ 分量。若进一步取 $\mathbb Q_h=\Delta X_h$，则 $\mathbf B_g$ 是方阵且非奇异。此时压力系数由 (iii) 的离散形式 $\mathbf B_g^{\!\top}\mathbf p=\mathbf A_{g\sigma}\mathbf u-\mathbf F_g$ 确定，再通过 $\eta(p,\mathbf D)$ 回到速度方程；压力是与速度联立的未知量，二者构成同一个非线性系统。

*证明。* $\mathbf B\tilde{\mathbf u}=\mathbf 0$ 即 $\int q\Delta\chi_h=0$ 对所有 $q\in\mathbb Q_h$ 成立；取 $q=\Delta\chi_h$ 得 $\Delta\chi_h=0$，加上 $\chi_h|_\Gamma=0$ 得 $\chi_h=0$（命题 5.10）。$\mathbf B_g$ 的矩阵元为 $\int q\Delta\chi$，当 $\mathbb Q_h=\Delta X_h$ 时它是 $\Delta X_h$ 上的 Gram 矩阵，正定，故非奇异。∎

---

## 8.5 刚体零空间与约束代数化

刚体运动（平移、转动）满足 $\mathbf D(\mathbf w)=\mathbf 0$（命题 5.6(3)）。对 $\ell\ge1$ 的离散速度，6 个线性泛函

$$
\boxed{\ 
(\mathbf C_{\rm rigid})_{a\alpha}=\int_{\mathcal V}\rho\,\mathbf B_\alpha\cdot\mathbf e_a\,dV,\qquad
(\mathbf C_{\rm rigid})_{3+a,\alpha}=\int_{\mathcal V}\rho\,(\mathbf x\times\mathbf B_\alpha)\cdot\mathbf e_a\,dV,\quad a=1,2,3\ }
$$

给出零动量与零角动量约束 $\mathbf C_{\rm rigid}\mathbf u=\mathbf 0$。带 Lagrange 乘子 $\boldsymbol\lambda=(\boldsymbol\lambda_P,\boldsymbol\lambda_L)\in\mathbb R^6$ 的完整系统为

$$
\begin{pmatrix}\mathbf A(\tilde{\mathbf u},\mathbf p)&-\mathbf B^{\!\top}&\mathbf C_{\rm rigid}^{\!\top}\\ \mathbf B&\mathbf 0&\mathbf 0\\ \mathbf C_{\rm rigid}&\mathbf 0&\mathbf 0\end{pmatrix}
\begin{pmatrix}\tilde{\mathbf u}\\ \mathbf p\\ \boldsymbol\lambda\end{pmatrix}
=\begin{pmatrix}\mathbf f_{\rm ext}\\ \mathbf 0\\ \mathbf 0\end{pmatrix}.
$$

**命题 8.11（球对称参考几何下的约束矩阵）**[矩阵元为精确积分；含该约束的整体系统的可解性依赖 PO-1]。 设 $R\equiv R_0$、$\rho=\rho(r)$，取 $Y_{1,a}=\sqrt{3/4\pi}\,\hat\omega_a$（实球谐，$a=x,y,z$）。则只有 $\ell=1$ 的模式出现，且

$$
C_{{\rm P},a}\big|_{q_{1ka}}=\frac{4\sqrt{3\pi}}{3}\int_0^{R_0}\rho\,(2rg_k+r^2g_k')\,dr,\qquad
C_{{\rm L},a}\big|_{s_{1ka}}=\frac{4\sqrt{3\pi}}{3}\int_0^{R_0}\rho\,f_k\,r^3\,dr,
$$

（$g_k=R_0^2\psi_{1k}(r/R_0)$，$f_k=R_0\psi_{1k}(r/R_0)$），其余元素为零。常数密度 $\rho_0$ 时

$$
C_{{\rm P},a}\big|_{q_{1ka}}=\frac{4\sqrt{3\pi}}{3}\,\rho_0R_0^4\quad(k=0,1,2,3),\qquad
C_{{\rm L},a}\big|_{s_{1ka}}=\frac{4\sqrt{3\pi}}{15}\,\rho_0R_0^5\,\delta_{k0}.
$$

于是 $\mathrm{rank}\,\mathbf C_{\rm rigid}=6$，并且角动量行只含 $k=0$（Jacobi 正交性：$\int_0^1\psi_{1k}\zeta^3d\zeta=\int_0^1\zeta^4P_k^{(0,4)}(2\zeta-1)d\zeta=\delta_{k0}/5$）。

*证明。* 对 $\ell=1$ 极向模式 $g(r)Y_{1,z}$，$v_z=c[2(g/r)\cos^2\theta+((rg)'/r)\sin^2\theta]$，$c=\sqrt{3/4\pi}$；对 $\Omega$ 积分得 $c\cdot\frac{8\pi}{3}(2g/r+g')$，再乘 $\rho r^2dr$ 积分。常数 $\rho$ 时被积函数是 $(gr^2)'$，积分为 $g(R_0)R_0^2=R_0^4$（$\psi(1)=1$）。环向模式 $f(r)Y_{1,z}$ 为绕 $z$ 轴的转动 $v_\varphi=cf\sin\theta$，$L_z=c\,\frac{8\pi}{3}\int\rho fr^3dr$。其余 $\ell$ 与不同类型的贡献由球谐正交性和 $\mathbf e_a$ 为常向量而为零：动量只投影到 $\ell=1$ 的极向，球对称 $\rho$ 下 $\ell=1$ 环向无净动量；极向模式无 $e_\varphi$ 分量，无角动量。以上积分已在 $k=0,1,2,3$ 上用符号计算复核。∎

**一般星形表面。** $R$ 非常数时，$\ell=1,k=0$ 的模式 $\Phi=R\,r\,Y_{1m}$ 一般不再是线性函数，六个刚体运动不一定落在 $\mathbb V_h^\sigma$ 内，只有 $R$ 为常数时才恰为 $\ell=1,k=0$ 的 6 个模式。此时 $\mathbf C_{\rm rigid}$ 的定义（对离散函数积分）不变，是否满秩以及黏性块的最小奇异值须在每个外层步报告（附录 A.3）；固定条件(b) $q_{1m0}=s_{1m0}=0$ 仅当 $R$ 为常数、$\rho=\rho(r)$ 时与(a)等价，其他情形一律用(a)。

**可解性（命题 7.6）。** 准静态问题要求合力与合力矩相容：$\int\rho\mathbf g_0dV+\oint\mathbf t_{\rm ext}dS=\mathbf 0$ 与对应的力矩条件。不满足时问题无静态解，须报告不相容，不能由乘子 $\boldsymbol\lambda$ 吸收。

---

## 8.6 自由表面运动学与质量约束

表面由 $F=r-R(\hat{\boldsymbol\omega},t)=0$ 定义，运动学条件为 $\partial_tF+\mathbf v\cdot\nabla F=0$，$\nabla F=\mathbf e_r-R^{-1}\nabla_{S^2}R$（$\nabla_{S^2}$ 为单位球面梯度），在 $r=R$ 处得（命题 9.1）：

$$
\boxed{\ \partial_tR=v_r-\frac1R\nabla_{S^2}R\cdot\mathbf v_S\Big|_{r=R}=\frac{\mathbf v\cdot\mathbf n}{\mathbf n\cdot\mathbf e_r}.\ }
$$

只有球面或忽略切向输运时才可近似为 $\partial_tR\approx v_r$。表面速度 $v_r,\mathbf v_S$ 由 $\mathbf u$ 经第 10.2.2 节的迹公式给出。质量约束

$$
\int_{S^2}\int_0^{R(\hat{\boldsymbol\omega},t)}\rho\,r^2\,dr\,d\Omega=M
$$

决定 $R$ 的 $\ell=0$ 分量（附录 A.3），它不由运动学方程推进。动态与静态的时间含义见第 9.9 节与第 10.4.1 节，二者不得混用。

---

## 8.7 压力恢复：独立标量压力 Poisson 弱形式

**物理依据。** 准静态动量方程 $\nabla\cdot\boldsymbol\sigma+\rho_K\mathbf g_{\rm eff}=\mathbf 0$、$\boldsymbol\sigma=-p\mathbf I+\boldsymbol\tau$ 给出 $\nabla p=\mathbf f$，其中

$$
\mathbf f:=\rho_K\mathbf g_{\rm eff}+\nabla\cdot\boldsymbol\tau,\qquad \boldsymbol\tau=\mathbf T_{\rm tide}+\boldsymbol\tau_{\rm flow}.
$$

$\mathbf f$ 一般不是梯度场（$\nabla\times\mathbf f=\nabla\rho_K\times\nabla W+\nabla\times(\nabla\cdot\boldsymbol\tau)\neq\mathbf 0$，旋转小行星尤甚）；无旋部分由压力承担；载荷 $\rho_K\mathbf g_{\rm eff}$ 的有旋部分（6.3 节斜压源）驱动双势流动，并由 $\nabla\cdot\boldsymbol\tau$ 的有旋部分平衡——这正是（8.4′）的内容，故精确平衡处 $\mathbf f$ 恰为梯度场。因此压力取为 $\mathbf f$ 在梯度场上的 $L^2$ 投影，即下述标量 Poisson 问题。该问题**独立于速度–压力鞍点系统**，它在第 10.4.2 节 Picard 迭代的每一轮给出全场连续压力 $p^{k+1}$。

**强形式与边界数据。** 自由表面牵引 $\boldsymbol\sigma\mathbf n=\mathbf t_{\rm ext}$ 的法向分量给出 Dirichlet 数据

$$
p|_\Gamma=p_\Gamma:=\mathbf n\cdot\boldsymbol\tau\,\mathbf n-\mathbf n\cdot\mathbf t_{\rm ext}\ \overset{\mathbf t_{\rm ext}=-p_{\rm ext}\mathbf n}{=}\ p_{\rm ext}+\mathbf n\cdot\mathbf T_{\rm tide}\mathbf n+\mathbf n\cdot\boldsymbol\tau_{\rm flow}\mathbf n .
$$

（牵引的切向分量是对 $\mathbf v$ 的一致性条件，见第 10.2.3 节，不约束 $p$。）

**弱形式。** 令 $p=p_b+p_0$，$p_b$ 是 $p_\Gamma$ 的任一 $H^1$ 提升，求 $p_0\in H^1_0(\mathcal V)$ 使

$$
\boxed{\ \int_{\mathcal V}\nabla p_0\cdot\nabla q\,dV=\int_{\mathcal V}\big(\rho_K\mathbf g_{\rm eff}+\nabla\cdot\boldsymbol\tau\big)\cdot\nabla q\,dV-\int_{\mathcal V}\nabla p_b\cdot\nabla q\,dV\qquad\forall q\in H^1_0(\mathcal V).\ }
$$

因 $q|_\Gamma=0$，牵引项 $\oint_\Gamma(\ldots)q\,dS$ 不出现在右端：牵引通过 Dirichlet 数据 $p_\Gamma$ 进入（把它写成对一般 $q\in H^1$ 的“边界积分项”会引入量纲不一致的 $\oint\mathbf t_{\rm ext}\!\cdot\!\mathbf n\,q\,dS$，且需要未知的 $\partial_np$，故不采用）。左端是 $H^1_0$ 上的内积，Poincaré 不等式给出强制性，故 Lax–Milgram 保证**解存在唯一**，不存在自由常数（对比第 8.4 节混合系统）。

**离散（正交移位 Jacobi/球谐标量基）。**

$$
\mathbb Q_h^0=\operatorname{span}\{\,(1-\zeta)\,\psi_{\ell k}(\zeta)\,Y_{\ell m}\,\},\quad 0\le\ell\le L_p,\ 0\le k\le K_p-1,\qquad \psi_{0k}(\zeta):=P_k^{(0,2)}(2\zeta-1),
$$

（因子 $(1-\zeta)$ 使基函数在 $\Gamma$ 上为零，$\ell\ge1$ 的 $\psi_{\ell k}$ 同 8.1 节；$\ell=0$ 为压力专用）。边界提升取 $p_b=\sum_{\ell,m}\hat p_{\ell m}\,\zeta^{\ell}Y_{\ell m}$，其中 $\hat p_{\ell m}$ 由节点值 $p_\Gamma(\hat{\boldsymbol\omega}_a)$ 作球谐分析（同 9.4 节步骤 1 的求积）得到。刚度矩阵 $(\mathbf K_p)_{ab}=\int_{\mathcal V}\nabla\phi_a\cdot\nabla\phi_b\,dV$ 对称正定，用 PCG（块对角预条件）求解，求积同 $\mathbf M_\Gamma$ 的装配（径向 Gauss–Jacobi、角向球面求积）。

**右端的体内数据（均须在体内求积点上取值）。**
- $\mathbf g_{\rm eff}=\nabla U+\omega^2\mathbf x_\perp$ 由 RTFP 内部点梯度给出（经典收敛，无需有限部分 Hessian）；因此 10.4.4 步骤 1 必须在体内求积点集 $\mathcal X_{\rm vol}$ 上额外计算 $\nabla U$。
- $\nabla\cdot\mathbf T_{\rm tide}$：内部 $\rho_K$ 光滑，$\nabla\cdot\operatorname{dev}H_W=\tfrac23\nabla\Delta W=-\tfrac{8\pi}{3}G\nabla\rho_K$（离心项为常 Hessian，无贡献），故 $\nabla\cdot\mathbf T_{\rm tide}=-\tfrac{8\pi}{3}G\,C_T\rho_p\nu_0d_p^2\,\nabla\rho_K$，$\nabla\rho_K$ 见第 6.1 节，无需 Hessian 的三阶导数。
- $\nabla\cdot\boldsymbol\tau_{\rm flow}=\eta\,\Delta\mathbf v+2\mathbf D\,\nabla\eta$（因 $\nabla\cdot\mathbf v=0$ 有 $\nabla\cdot(2\mathbf D)=\Delta\mathbf v$），其中 $\mathbf v$ 取 Picard 上一轮的 Stokes 解 $\mathbf u^{k}$（每个地质步的初值取上一地质步的收敛解在新几何上按系数原样使用；首个地质步取 $\mathbf u^0=\mathbf 0$，即 $\boldsymbol\tau_{\rm flow}=\mathbf 0$，此时 $p$ 为含潮汐的广义静水压），$\eta$ 与 $\nabla\eta$ 用 $(\tilde p^{k},\mathbf D(\mathbf u^k))$ 冻结并按链式法则由 $\eta(\tilde p,\mathbf D)$ 计算（$\nabla\tilde p=\mathbf 1_{\{p>p_{\min}\}}\nabla p$）。初值只影响 Picard 迭代数，不影响收敛解。

**与速度的耦合（Picard 迭代）。** 一个地质步内迭代 $k=0,1,\dots$：由冻结的 $(\mathbf u^{k},\tilde p^{k})$ 解本问题得 $p^{k+1}$，再由 $\tilde p^{k+1}$ 与 $\mathbf D(\mathbf u^k)$ 装配 $\mathbf K(\eta^{k+1/2})$ 并解（8.4′）得 $\mathbf u^{k+1}$，直到第 2.15 节判据 (a) 成立（算法见第 10.4.2 节）。收敛时 $(\mathbf u,p)$ 同时满足无散 Galerkin 动量方程与本 Poisson 问题，即同一应力场的准静态平衡；该不动点迭代的收敛性未证明（PO-11），未收敛按 2.15 (a) 判失败。下游使用的压力为 $\tilde p=\max([p]_+,p_{\min})$（2.8 节 B5），受拉比例须报告。强形式残差 $\mathbf r_{\rm rot}:=\mathbf f-\nabla p$ 在连续精确解处为零；离散解中它度量 $\mathbf f$ 未被 $\mathbb V_h^\sigma$ Galerkin 投影吸收的有旋部分（随 $K,L_{\max}$ 收敛），按 2.15 (b) 作为分辨率指标报告。

**与第 8.4 节的关系。** 准静态主线的速度由（8.4′）给出、压力由本节给出，二者经 Picard 迭代闭合为同一非线性系统；第 8.4 节含惯性与独立压力的全隐式鞍点系统及命题 8.10 只作动态模式的理论基准，不在准静态循环内求解。

---

## 8.8 内部引力载荷

引力 $\mathbf g(\mathbf x)=G\int_{\mathcal V}\rho(\mathbf y)\frac{\mathbf x-\mathbf y}{|\mathbf x-\mathbf y|^3}d\mathbf y$ 依赖内部密度。仅知表面 $R(\hat{\boldsymbol\omega})$ 而不知 $\rho$ 时内部引力不唯一，表面 RTFP 只在密度已知且固定、质量集中于薄壳、使用预先构造的多极展开、或内部质量变化被单独修正时才可作近似。载荷向量的元素为

$$
\boxed{\ (\mathbf F_g)_\alpha=\int_{\mathcal V}\rho\,\mathbf g_{\rm RTFP}\cdot\mathbf B_\alpha\,dV,\ }
$$

积分域是 $\mathcal V(R)$，包含全部 $\ell\ge1$ 的测试函数。

---

## 8.9 严格性与误差报告

径向 Jacobi 截断（$K=4$）和角向球谐截断（$L_{\max}$）都是 Galerkin 截断：只有 $K\to\infty$、$L_{\max}\to\infty$ 且基完备时才可能收敛到三维问题，有限 $K$ 并不等价于原问题。每次运行必须报告：

1. $K$ 的收敛性，$E_{4\to8}=\|\mathbf u_8-\mathbf u_4\|/\|\mathbf u_8\|$，以及 $L_{\max}$ 的收敛性；
2. 准静态主线：黏度对比度 $\eta_{\rm hi}/\eta_{\rm lo}$、预条件后 PCG 迭代数、Picard 迭代数与 2.15 (a)(b) 的残差；动态模式：$\mathrm{cond}(\mathbf M_\Gamma|_{\ell\ge1})$（命题 8.7）与每个 $\ell$ 块的条件数；
3. 刚体约束的相对乘子 $\varepsilon_\lambda$（2.15 (b)）与 $\operatorname{rank}\mathbf B_{\rm rb}$、其最小奇异值；
4. 强形式残差 $\|\mathbf r_{\rm rot}\|$（8.7 节）；动态全隐式基准另报压力空间的 inf–sup 数值指标（PO-2，未证）；
5. 切向牵引一致性偏差 $\delta_{na}$（第 10.2.3 节）；
6. RTFP 引力误差；
7. 使用 Track B（附录 A.4）时的 $\varepsilon_B$ 与 $\mathbf r_B$。

所得二维—三维混合离散系统是经过假设与误差控制的降阶模型，不是对原三维自由边界问题的无条件、零误差降维。

---

# 9. 曲面守恒输运与半拉格朗日推进

本章服从附录 A 的规范约定（A.1–A.4）：速度自由度为极向与环向系数 $\mathbf u=(\mathbf q,\mathbf s)$（第 7.3 节），守恒变量 $\mathbf m$ 与速度系数通过全局质量算子 $\mathbf M_\Gamma(R,\rho)$ 相联。Track A 为基准，Track B 是显式给出截断误差的局部近似。

**适用范围（强制）。** 模态动量 $\mathbf m$ 的输运（9.1 节）、显式源项更新（9.3 节）与质量恢复（9.4–9.5 节）**只属于保留惯性的动态自由表面模式**。准静态主线（第 2.8、10.4 节）不携带、不输运、不重映射 $\mathbf m$，速度每个地质步由（8.4′）重新隐式求解，因而不受 9.3 节显式源项步长限制。准静态主线只使用本章的 9.0 节（曲面几何与表面参数速度 $\mathbf c$）与 9.2 节（半拉格朗日反向追踪与 Jacobian 守恒修正），用于模式 C 的面密度 $\rho_s$ 输运（2.8 节），此时 9.2 节中的 8 个场替换为单个标量场 $\rho_s$、$\mathbf c$ 取相对刚性核的活动区速度 $\chi_A\mathbf v_{{\rm sl},S}/R$。

---

## 9.0 曲面几何与表面物质速度

自由表面 $\Gamma$：$X(\hat{\boldsymbol\omega},t)=R(\hat{\boldsymbol\omega},t)\,\hat{\boldsymbol\omega}$，$\hat{\boldsymbol\omega}\in S^2$。星形性使 $\hat{\boldsymbol\omega}$ 成为 $\Gamma$ 的全局参数，不需要多图册。设 $\nabla_{S^2}$ 为单位球面梯度，$\mathbf e_r=\hat{\boldsymbol\omega}$，则

$$
\mathbf n=\frac{R\,\mathbf e_r-\nabla_{S^2}R}{\sqrt{R^2+|\nabla_{S^2}R|^2}},\qquad
\mathbf n\cdot\mathbf e_r=\frac{R}{\sqrt{R^2+|\nabla_{S^2}R|^2}}\ \ge c_0>0 .
$$

诱导度量在球坐标 $(\theta,\varphi)$ 下为

$$
g_{\theta\theta}=R^2+R_\theta^2,\quad g_{\varphi\varphi}=R^2\sin^2\theta+R_\varphi^2,\quad g_{\theta\varphi}=R_\theta R_\varphi,
$$

面积元 $dS=J_\Gamma\,d\theta\,d\varphi=j\,d\Omega$，其中相对立体角的面积密度

$$
j(\hat{\boldsymbol\omega},t)=R\sqrt{R^2+|\nabla_{S^2}R|^2}=\frac{R^2}{\mathbf n\cdot\mathbf e_r},\qquad J_\Gamma=j\sin\theta .
$$

$\theta=0,\pi$ 处的 $J_\Gamma=0$ 只是坐标奇异性；离散一律使用单位向量 $\hat{\boldsymbol\omega}$ 表示点位并用 $j$ 作为守恒密度。

**命题 9.1（表面物质点的参数速度）。** 表面物质点满足运动学条件（第 5.1 节），其参数轨迹 $\hat{\boldsymbol\omega}(t)$ 的速度为

$$
\boxed{\ \mathbf c:=\dot{\hat{\boldsymbol\omega}}=\frac1R\,\mathbf v_S\big|_{r=R},\qquad
\partial_tR+\mathbf c\cdot\nabla_{S^2}R=v_r\big|_{r=R},\ }
$$

其中 $\mathbf v_S$ 是 $\mathbf v|_\Gamma$ 在 $\hat{\boldsymbol\omega}^{\perp}$ 内的物理切向分量。等价地 $\partial_tR=v_r-\tfrac1R\nabla_{S^2}R\cdot\mathbf v_S$，即用表面物理梯度 $\nabla_S=R^{-1}\nabla_{S^2}$ 时 $\partial_tR+\mathbf v_S\cdot\nabla_SR=v_r$。并且

$$
\partial_tR=\frac{\mathbf v\cdot\mathbf n}{\mathbf n\cdot\mathbf e_r}.
$$

*证明。* 物质点 $X(t)=R(\hat{\boldsymbol\omega}(t),t)\hat{\boldsymbol\omega}(t)$，求导得 $\mathbf v=(\partial_tR+\nabla_{S^2}R\cdot\dot{\hat{\boldsymbol\omega}})\hat{\boldsymbol\omega}+R\dot{\hat{\boldsymbol\omega}}$。径向分量即运动学条件，切向分量给出 $\mathbf c=\mathbf v_S/R$。最后一式由 $\mathbf v\cdot\mathbf n=(Rv_r-\nabla_{S^2}R\cdot\mathbf v_S)/\sqrt{R^2+|\nabla_{S^2}R|^2}$ 与 $\mathbf n\cdot\mathbf e_r$ 的表达式得到。∎

最后一式与第 2.8 节 $d_{\rm pred}=(\mathbf u\cdot\mathbf n+\Delta h_C)/(\mathbf n\cdot\mathbf e_r)$ 是同一个几何关系，把法向位移变为径向增量。

---

## 9.1 守恒变量与输运方程

**定义 9.2（守恒变量）。** 取第 8.1 节的径向基与第 8.2 节的展开，$\mathbf u=(q_{\ell km},s_{\ell km})$，$1\le\ell\le L_{\max}$。质量算子

$$
(\mathbf M_\Gamma)_{\alpha\beta}=\int_{\mathcal V(R)}\rho\,\mathbf B_\alpha\cdot\mathbf B_\beta\,dV,\qquad
\mathbf m:=\mathbf M_\Gamma(R,\rho)\,\mathbf u .
$$

**转换算子（矩阵定义，唯一化）。** 设表面求积点 $\{\hat{\boldsymbol\omega}_a\}_{a=1}^{N_S}$ 与求积权 $w_a>0$，单场权矩阵为 $\mathbf W_1=\operatorname{diag}(w_a)$，8 场节点权矩阵固定定义为 $\mathbf W_8=\mathbf W_1\otimes\mathbf I_8$。球谐合成矩阵 $\mathbf S_1\in\mathbb R^{N_S\times N_{\ell m}}$，$({\mathbf S}_1)_{ai}=Y_i(\hat{\boldsymbol\omega}_a)$（$i=(\ell,m)$，$1\le\ell\le L_{\max}$）；8 场合成矩阵定义为 $\mathbf S=\mathbf S_1\otimes\mathbf I_8$，并固定系数向量为“球谐指标优先、8 个 $(X,k)$ 分量连续”的顺序；若实现采用场优先顺序，必须改用等价的 $\mathbf I_8\otimes\mathbf S_1$ 并同步改写所有 blockdiag 与分析矩阵。求积分析矩阵定义为 $\mathbf S^*_{\rm quad}:=\mathbf S^{\!\top}\mathbf W_8$。求积对 $2L_{\max}$ 次多项式精确时 $\mathbf S_1^{\top}\mathbf W_1\mathbf S_1=\mathbf I$，故 $\mathbf S^*_{\rm quad}\mathbf S=\mathbf I$；后文 $\mathbf W$ 一律指明确的 $\mathbf W_8$。

**假设（A-M，PO-4 未证）。** 将 $\mathbf M_\Gamma\mathbf u$ 作为沿表面特征线输运的变量，是基于算子分裂的模型假设，而非三维弱形式的直接无损变换。

$\mathbf M_\Gamma$ 在 $\ell\ge1$ 子空间上对称正定，仅定义并求逆于 $\ell\ge1$ 的速度模态空间（第 8.2 节、定义 8.6）：$\ell=0$ 的双势对速度场的贡献恒为零（命题 5.6(2)），故 $\mathbf M_\Gamma$ 不含 $\ell=0$ 的行列，也不存在对奇异行列求逆的问题。$R$ 球对称、$\rho=\rho(r)$ 时它对 $(\ell,m)$ 块对角，极向与环向解耦（命题 5.7）；一般情形是带状稀疏块矩阵，带宽由 $R$ 与 $\rho$ 的角向谐波阶数决定（命题 6.1）。对每个类型 $X\in\{P,T\}$ 与径向序号 $k\in\{0,1,2,3\}$（共 8 个表面输运场），标准球谐变换为 $m_{X,k}(\hat{\boldsymbol\omega})=\sum_{\ell m}m_{X,\ell km}Y_{\ell m}(\hat{\boldsymbol\omega})$，$m_{X,\ell km}=\oint_{S^2}m_{X,k}Y_{\ell m}\,d\Omega$；由实球谐基的标准正交性，该映射在带限 $L^2(S^2)$ 空间上为双向等距同构（Parseval 恒等式），不与物理动能守恒混淆。定义表面场

$$
m_{X,k}(\hat{\boldsymbol\omega})=\sum_{\ell\ge1,|m|\le\ell}m_{X,\ell km}\,Y_{\ell m}(\hat{\boldsymbol\omega}),
$$

共 $8$ 个场。

**输运方程。** 沿表面物质点，守恒形式为

$$
\boxed{\ \partial_t(j\,m_{X,k})+\nabla_{S^2}\cdot(j\,\mathbf c\,m_{X,k})=j\,\mathcal S_{X,k},\ }
$$

源项由第 7.3 节块系统对 $\mathbf m=\mathbf M_\Gamma\mathbf u$ 求时间导数得到：

$$
\boldsymbol{\mathcal S}=(\partial_t\mathbf M_\Gamma-\mathbf G)\,\mathbf u-\mathbf C\mathbf u-\mathbf K(\mathbf u,p)-\mathbf N(\mathbf u,\mathbf u)+\mathbf F ,
$$

其中 $\partial_t\mathbf M_\Gamma$ 来自 $R,\rho$ 的时间变化（$\partial_t\mathbf M_\Gamma$ 含 Reynolds 边界项 $\oint_\Gamma\rho(\mathbf B_\alpha\cdot\mathbf B_\beta)(\partial_tR)(\mathbf n\cdot\mathbf e_r)\,dS=\int_{S^2}\rho(\mathbf B_\alpha\cdot\mathbf B_\beta)R^2\partial_tR\,d\Omega$、基函数形变项 $\int_{\mathcal V}\rho(\partial_t\mathbf B_\alpha\cdot\mathbf B_\beta+\mathbf B_\alpha\cdot\partial_t\mathbf B_\beta)dV$ 及 $\partial_t\rho$ 项）。定义 $(\mathbf G)_{\beta\alpha}=\int_{\mathcal V}\rho\,\mathbf B_\beta\cdot\partial_t\mathbf B_\alpha\,dV$；因此系数方程中的基函数连接项是 $\mathbf G\mathbf u$，而守恒变量方程的几何源项必须为 $(\partial_t\mathbf M_\Gamma-\mathbf G)\mathbf u$，不是把 $\partial_t\mathbf M_\Gamma\mathbf u$ 与 $\mathbf G\mathbf u$ 同时计入。本文固定空间 Euler 形式的 $\mathbf N$ 使用 $\mathbf v$，不得再把 $\mathbf v-\mathbf w_{\rm mesh}$ 代入同一项。若切换到 ALE 形式，必须连同时间导数和基函数导数一起切换。静态 SCF 中 $\mathbf G\equiv\mathbf 0$。$\mathbf C,\mathbf K,\mathbf N,\mathbf F$ 的定义见第 5.4 节与第 7.3 节。

**模型假设 A-M（算子分裂）。** 7.3 节的 $\mathbf M_\Gamma\dot{\mathbf u}+\cdots=\mathbf F$ 在固定 $\hat{\boldsymbol\omega}$ 取 Euler 导数；本章采用的是它的分裂形式：先沿表面物质点平移 $\mathbf m$（纯输运，$\mathcal S=0$），再在到达点上累加 $\boldsymbol{\mathcal S}$。**对流去重：** 沿表面物质点以 $\mathbf c$ 平移 $\mathbf m$ 已经承担了表面切向对流。（准静态主线不输运 $\mathbf m$，本段只针对动态模式。）动态自由表面模式下，源项 $\boldsymbol{\mathcal S}$ 中的非线性对流项 $\mathbf N$ 必须取为扣除表面切向对流后的剩余相对对流项（记 $\mathbf N_{\rm rel}$，使 $\mathbf N=\mathbf N_{\rm rel}+\mathcal A(\mathbf c)\mathbf m$，其中 $\mathcal A(\mathbf c)\mathbf m$ 为已由输运步承担的部分），否则同一对流效应被计两次。该分裂对时间步长是一阶（Lie）或二阶（Strang）相容的，**仅在扣除该重复对流后成立**；未扣除时存在 $O(1)$ 的模型误差。分裂误差 $E_{\rm split}$ 列入第 9.8 节误差预算；它是建模选择，不是 Galerkin 残差的推论。

---

## 9.2 半拉格朗日反向追踪

对网格点 $\hat{\boldsymbol\omega}_a$ 求反向特征线 $\dot{\hat{\boldsymbol\omega}}=\mathbf c(\hat{\boldsymbol\omega},t)$，$\hat{\boldsymbol\omega}(t^{n+1})=\hat{\boldsymbol\omega}_a$，并在球面上积分（保持 $|\hat{\boldsymbol\omega}|=1$）：

$$
\hat{\boldsymbol\omega}_m=\mathcal N\!\Big[\hat{\boldsymbol\omega}_a-\tfrac{\Delta t}{2}\mathbf c(\hat{\boldsymbol\omega}_a,t^n)\Big],\qquad
\hat{\boldsymbol\omega}_d=\mathcal N\!\Big[\hat{\boldsymbol\omega}_a-\Delta t\,\mathbf c(\hat{\boldsymbol\omega}_m,t^{n+1/2})\Big],
$$

$\mathcal N(\mathbf x)=\mathbf x/|\mathbf x|$。若仅有 $t^n$ 的速度，则只是一阶近似，不得声称二阶精度。RayBackend 对射线求交只用于定位含 $\hat{\boldsymbol\omega}_d$ 的三角形，随后必须在该单元上做曲面投影才得到有效的 $\hat{\boldsymbol\omega}_d$。

在三角形 $\{\hat{\boldsymbol\omega}_1,\hat{\boldsymbol\omega}_2,\hat{\boldsymbol\omega}_3\}$ 内取重心坐标 $\lambda_i$，对 8 个场同时插值：

$$
m^d_{X,k}=\sum_{i=1}^3\lambda_i\,m_{X,k}(\hat{\boldsymbol\omega}_i).
$$

守恒修正（保证 $\int j\,m\,d\Omega$ 的离散守恒）取特征映射的 Jacobian

$$
\mathcal J_a=\Big|\det\frac{\partial\hat{\boldsymbol\omega}_d}{\partial\hat{\boldsymbol\omega}}\Big|_{\hat{\boldsymbol\omega}_a}
$$

（对 $S^2$ 的切空间取行列式；直接对嵌入向量取行列式没有意义）。要求特征映射在所取网格尺度上是单值、不折叠的（Lipschitz 常数 $\Delta t\,\|\nabla\mathbf c\|_\infty<1$），否则反向出发点不唯一，须缩小 $\Delta t$。无源更新为

$$
j^{n+1}_a\,m^{n+1,*}_a=\mathcal J_a\,j^{n}_d\,m^d .
$$

---

## 9.3 源项更新与分裂

显式源项更新（Lie 分裂）：

$$
j^{n+1}_a\,m_a^{n+1}=\mathcal J_a\,j^{n}_d\,m^d_a+\Delta t\,j^{n+1}_a\,\boldsymbol{\mathcal S}_a^{n},
$$

其中 $\boldsymbol{\mathcal S}_a^n$ 的各项由 9.1 节装配，所用 $\mathbf u^n$ 由上一步的全局恢复（9.4 节）得到。源项刚性（$\Delta t\,\|\partial\boldsymbol{\mathcal S}/\partial\mathbf m\|$ 不小）时须缩小 $\Delta t$ 或子循环；本文不引入局部隐式源项更新，因为它要对 $\mathbf M_\Gamma^{-1}$ 作局部近似，等于未声明的 Track B。

---

## 9.4 全局质量恢复（Track A 生产主线）

输运与源项更新之后得到的是一组网格点场 $\{m_{X,k}(\hat{\boldsymbol\omega}_a)\}$。速度系数必须由下述全局过程恢复。

**步骤 1（分析）。** 用在球面上对 $2L_{\max}$ 次多项式精确的求积（Gauss–Legendre $\times$ 等距 $\varphi$ 网格，或 Lebedev）作球谐分析：

$$
m_{X,\ell km}=\sum_aw_a\,m_{X,k}(\hat{\boldsymbol\omega}_a)\,Y_{\ell m}(\hat{\boldsymbol\omega}_a),\qquad 1\le\ell\le L_{\max}.
$$

**步骤 2（$\ell=0$ 势规范）。** $\ell=0$ 的 $\Phi,\Psi$ 对速度无贡献（命题 5.6），$\mathbf M_\Gamma$ 本身只含 $\ell\ge1$ 的行列（定义 8.6），不含 $\ell=0$。规范取

$$
q_{0km}=s_{0km}=0,\qquad m_{X,0km}:=0,
$$

丢弃 $\ell=0$ 在对偶空间上是正交投影 $\mathcal P_{\ge1}$，即球面标准 $L^2(S^2)$ 上的去均值（去直流）正交投影算子，不是低通滤波；非球形 $R$ 的跨阶耦合可使表面输运场产生微小 $\ell=0$ 泄漏，$\|(I-\mathcal P_{\ge1})\mathbf m\|$ 即几何非球形诱导的模态截断误差。分析得到的 $\ell=0$ 内容 $m^{\rm spur}_{X,0k}$ 被丢弃，其范数 $\|\mathbf m^{\rm spur}\|/\max\{\|\mathbf m\|,\varepsilon_m\}$ 作为输运自洽性诊断量（$\|\mathbf m\|=0$ 且分子为零时取 0，否则判该步失败），须小于配置容差，否则判该步输运失败。

**步骤 3（全局求解）。** 求解带刚体约束的鞍点系统（命题 5.6、命题 8.11 与附录 A.3）

$$
\boxed{\ 
\begin{pmatrix}\mathbf M_\Gamma(R,\rho)&\mathbf B_{\rm rb}^{\!\top}\\ \mathbf B_{\rm rb}&\mathbf 0\end{pmatrix}
\begin{pmatrix}\mathbf u\\ \boldsymbol\lambda\end{pmatrix}
=\begin{pmatrix}\mathbf m\\ \mathbf 0\end{pmatrix},\ }
$$

其中 $\mathbf B_{\rm rb}\in\mathbb R^{6\times N_u}$ 的行为 $\int_{\mathcal V}\rho\,\mathbf B_\alpha\,dV$ 与 $\int_{\mathcal V}\rho\,\mathbf x\times\mathbf B_\alpha\,dV$。$\mathbf M_\Gamma$ 对称正定，鞍点系统适定当且仅当刚体约束块满足代数满秩条件 $\operatorname{rank}(\mathbf B_{\rm rb})=6$。$R$ 为常数、$\rho=\rho(r)$ 时该秩条件由命题 8.11 保证；$R$ 非球形时它由几何与权函数决定，须在每个外层步数值核验该秩（以及 $\mathbf B_{\rm rb}$ 的最小奇异值），不预设刚体运动落在离散空间内。$\boldsymbol\lambda=(\mathbf B_{\rm rb}\mathbf M_\Gamma^{-1}\mathbf B_{\rm rb}^{\!\top})^{-1}\mathbf B_{\rm rb}\mathbf M_\Gamma^{-1}\mathbf m$ 是 $\mathbf M_\Gamma^{-1}$ 内积下把 $\mathbf m$ 正交投影到 $\ker(\mathbf B_{\rm rb}\mathbf M_\Gamma^{-1})$ 的**投影乘子**，不是“必须为零”的残差：$\boldsymbol\lambda\ne\mathbf0$ 仅表示 $\mathbf m$ 含有可被投影掉的刚体动量/角动量分量，本身不构成失败判据。监控量为投影相对修正量（刚体修正残差）

$$
E_{\rm rb}:=\frac{\|\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda\|_{\mathbf M_\Gamma^{-1}}}{\max\{\|\mathbf m\|_{\mathbf M_\Gamma^{-1}},\varepsilon_{\rm rb}\}}\in[0,1],
$$

（$\|\mathbf a\|_{\mathbf M_\Gamma^{-1}}^2=\mathbf a^{\!\top}\mathbf M_\Gamma^{-1}\mathbf a$；由 $(\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda)^{\!\top}\mathbf M_\Gamma^{-1}(\mathbf m-\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda)=\boldsymbol\lambda^{\!\top}\mathbf B_{\rm rb}\mathbf u=0$ 得 Pythagoras 关系 $\|\mathbf m\|^2=\|\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda\|^2+\|\mathbf m_{\rm rb}\|^2$（均为 $\mathbf M_\Gamma^{-1}$ 范数），故 $E_{\rm rb}\le1$，在其它范数下也须使用相同的零分母正则化约定，并声明范数）。只有 $E_{\rm rb}$ 超过配置容差（刚体成分占主导，说明输运或源项产生了不可忽略的寄生刚体动量）时才判该步异常。恢复后与物理速度场严格对应的同步动量为 $\mathbf m_{\rm rb}:=\mathbf M_\Gamma\mathbf u=\mathbf m-\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda$；动态模式每次恢复之后必须执行状态同步赋值 $\mathbf m^{(n+1/2)}\leftarrow\mathbf m_{\rm rb}^{(n+1/2)}$，后续输运与多步循环传递的是已消除寄生刚体动量的 $\mathbf m_{\rm rb}$，以避免数值刚体漂移无界累积。

**受约束恢复的含义。** 该系统定义受约束正交投影解 $\mathbf u=\mathcal R_{\rm rb}(\mathbf m)=\mathbf M_\Gamma^{-1}(\mathbf m-\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda)$。当 $\boldsymbol\lambda\neq\mathbf0$ 时，恢复的 $\mathbf u$ 不满足 $\mathbf m=\mathbf M_\Gamma\mathbf u$，而是对守恒变量施加了消除寄生刚体动量的 $\mathbf M_\Gamma^{-1}$-正交投影修正；动量修正量 $\Delta\mathbf m_{\rm rb}=\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda$ 以 $E_{\rm rb}$ 计入第 9.8 节误差预算。

该系统用预条件共轭梯度（对 Schur 补）求解，预条件子取 Track B1 的块对角逆 $\mathbf M_{B1}^{-1}$（9.5 节）。预条件后迭代次数由 $\|\mathbf M_{B1}^{-1}\mathbf E\|$ 控制，解是精确的（到 CG 容差），不是近似。

**$\mathbf M_\Gamma$ 的装配。** 在 $(\ell,m),(\ell',m')$ 块中，

$$
(\mathbf M_\Gamma)_{\alpha\beta}=\int_{S^2}\!\int_0^{R(\hat{\boldsymbol\omega})}\rho\,\mathbf B_\alpha\cdot\mathbf B_\beta\,r^2dr\,d\Omega ,
$$

径向积分用 Gauss–Jacobi 求积（权 $\zeta^{2\ell+2}$ 对应的 Jacobi 权），角向用与步骤 1 相同的求积。$R$ 与 $\rho$ 的球谐展开阶数决定带宽（命题 6.1 的选择规则给出非零块的位置）。


（`tests/verify_closure.py` 仅为抽象线性代数恒等式独立测试：它复核鞍点系统与受约束分解的代数恒等式，不构成对 Galerkin 装配、物理模型或数值结果的验证。）

---

## 9.5 Track B：块对角代理及其截断误差

**定义 9.3（Track B = B1 块对角代理）。** 取

$$
\mathbf M_B:=\mathbf M_{B1}=\operatorname{blockdiag}_{(\ell,m)}\mathbf M_{\ell m,\ell m},\qquad \mathbf u_B:=\mathbf M_B^{-1}\mathbf m,
$$

其中对每个 $(\ell,m)$，$\mathbf M_{\ell m,\ell m}\in\mathbb R^{8\times8}$ 是 $\mathbf M_\Gamma$ 的对角主子块（极向 4 个、环向 4 个径向模式；一般几何下包含 $\mathbf M_{PT}$ 的块内交叉元）。主子块继承 $\mathbf M_\Gamma$ 的对称正定性，故 $\mathbf M_B$ 对称正定、逐块可逆；残差算子 $\mathbf E:=\mathbf M_\Gamma-\mathbf M_B$ 仅含跨 $(\ell,m)$ 的 Gaunt 耦合块。本文不定义任何节点局部（逐网格点）近似（见 PO-8），实现中亦不得保留此类代码路径。

**性质。**（i）$R\equiv$ 常数且 $\rho=\rho(r)$ 时 $\mathbf E=\mathbf0$，Track B 与 Track A 的无约束解逐位一致（命题 8.8）；（ii）一般几何下 $\mathbf M_B^{-1}\mathbf M_\Gamma$ 与对称正定矩阵 $\mathbf M_B^{-1/2}\mathbf M_\Gamma\mathbf M_B^{-1/2}=\mathbf I+\mathbf M_B^{-1/2}\mathbf E\mathbf M_B^{-1/2}$ 相似，因而谱为**实正数**，不存在符号翻转型发散；其偏离 $1$ 的幅度由可计算量 $\|\mathbf M_B^{-1/2}\mathbf E\mathbf M_B^{-1/2}\|$ 控制，该量对非球形程度的依赖关系未被证明（PO-8），实现须在每个外层步报告其估计值（如幂迭代）；（iii）$\mathbf M_B^{-1}$ 同时是 9.4 节对 $\mathbf M_\Gamma$ 作 PCG 的预条件子，此时预条件后的条件数即 $\kappa(\mathbf I+\mathbf M_B^{-1/2}\mathbf E\mathbf M_B^{-1/2})$。

**定理 9.4（截断误差）**[恒等式与后验界已证；$\varepsilon_B$ 的经验上界与适用域属 PO-8，本定理不依赖它]。 定义模态截断误差算子

$$
\mathbf E_B^{\rm solve}:=\mathbf M_B^{-1}-\mathbf M_\Gamma^{-1},
$$

并记 $\mathbf u_B:=\mathbf M_B^{-1}\mathbf m$（无约束 Track B 恢复），Track A 受约束解 $\mathbf u=\mathcal R_{\rm rb}(\mathbf m)=\mathbf M_\Gamma^{-1}(\mathbf m-\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda)$。则**受约束正交分解**严格成立：

$$
\boxed{\ \mathbf u_B-\mathbf u=\mathbf E_B^{\rm solve}\,\mathbf m+\mathbf M_\Gamma^{-1}\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda
=\big(\mathbf M_B^{-1}-\mathbf M_\Gamma^{-1}\big)\mathbf m+\mathbf M_\Gamma^{-1}\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda.\ }
$$

第一项为质量算子截断误差，第二项为刚体约束诱导的投影差项（由 $E_{\rm rb}$ 度量）。令 $\mathbf u_0:=\mathbf M_\Gamma^{-1}\mathbf m$ 为无约束 Track A 解，则对**无约束**部分有如下恒等式与后验界：

$$
\boxed{\ \mathbf u_B-\mathbf u_0=\mathbf M_B^{-1}\mathbf E\,\mathbf u_0=\mathbf M_\Gamma^{-1}\mathbf E\,\mathbf u_B,\qquad
\|\mathbf u_B-\mathbf u_0\|\le\frac{\|\mathbf m-\mathbf M_\Gamma\mathbf u_B\|}{\lambda_{\min}(\mathbf M_\Gamma|_{\ell\ge1})}.\ }
$$

*证明。* $\mathbf M_B^{-1}-\mathbf M_\Gamma^{-1}=\mathbf M_B^{-1}(\mathbf M_\Gamma-\mathbf M_B)\mathbf M_\Gamma^{-1}=\mathbf M_\Gamma^{-1}(\mathbf M_\Gamma-\mathbf M_B)\mathbf M_B^{-1}$，作用于 $\mathbf m$；第二式给出 $\mathbf M_\Gamma(\mathbf u_B-\mathbf u_0)=\mathbf M_\Gamma\mathbf u_B-\mathbf m$，取欧氏范数。∎

（无约束情形下 $\mathbf u_B-\mathbf u_0=\mathbf E_B^{\rm solve}\mathbf m$。）可计算的后验量是 $\mathbf r_B=\mathbf m-\mathbf M_\Gamma\mathbf u_B$（一次稀疏矩阵向量积）。推论：$\mathbf u_B=\mathbf u_0$ 当且仅当 $\mathbf E\mathbf u_0=\mathbf 0$。使用 Track B 时必须报告 $\varepsilon_B:=\|\mathbf r_B\|/\max\{\|\mathbf m\|,\varepsilon_m\}$（零动量情形按附录 A.4 处理）及其对 $\ell$ 的分布，且不得称为“精确”或“等价”（球对称几何除外，见上）。

**使用准则（Track B 红线）。**

1. Track B 的适用域由实测 $\varepsilon_B$ 与 $\|\mathbf M_B^{-1/2}\mathbf E\mathbf M_B^{-1/2}\|$ 的配置阈值裁定；超阈值时该外层步必须回退 Track A；
2. Track B 的任何收敛结论须由 Track A 重新恢复并验收（10.4.2 节）；
3. 生产基准为 Track A。

---

## 9.6 RayBackend 的作用

RayBackend 负责反向特征线的面片定位与求交，不计算输运通量、源项矩阵或质量算子。每条射线得到一个面片后同时读取 8 个场，射线求交次数不变，插值与局部源项计算量按场数线性增长。

---

## 9.7 计算复杂度

设 $N_S$ 为表面网格点数，$B$ 为 BVH 三角形数，$N_{\rm ray}$ 为每点射线数，$N_{\rm hit}$ 为平均命中面片数，$N_{\rm src}$ 为局部 Newton 迭代数。

**局部部分**（求交、插值、源项），每个点 $8$ 个场：

$$
C_{\rm local}=O\!\big[N_SN_{\rm ray}(\log B+N_{\rm hit})+(c_{\rm int}+c_{\rm src}N_{\rm src})N_S\big].
$$

**与单模态的比较**（仅限局部部分）。每步局部成本以 $K=4$ 的四模态源项更新记为 $4+16N_{\rm src}$、单模态为 $1$，则

$$
\frac{C_{K=4}}{C_{K=1}}\approx1+\frac{3+16N_{\rm src}}{N_{\rm ray}(\log B+N_{\rm hit})}.
$$

RT 求交占主导时 $K=4$ 约为 $1.1$–$1.5$ 倍，源项占主导时接近四倍。

**全局部分**（Track A 独有）：球谐分析/综合 $O(N_SL_{\max}^2)$（快速变换可降为 $O(L_{\max}^2\log L_{\max})$ 每场），$\mathbf M_\Gamma$ 稀疏矩阵向量积 $O(N_u\cdot b)$（$b$ 为块带宽），PCG 迭代 $N_{\rm it}$ 次，总计 $O(N_{\rm it}N_ub)$。Track B（B1 块对角）去掉这一项，代价是定理 9.4 的误差。

---

## 9.8 误差预算

各误差分量量纲不同，**严禁**把绝对残差直接相加。总误差预算定义为无量纲相对量之和，$\operatorname{rel}_{\varepsilon}(a;b):=\|a\|/\max\{\|b\|,\varepsilon_b\}$（每个 $\varepsilon_b$ 与对应分母同量纲，同一项内分子分母用同一范数；零载荷、零速度时预算仍有定义）。

**准静态主线（每个地质步）：**

$$
E_{\rm total}^{\rm qs}:=\varepsilon_{\rm mom}^{\rm weak}+\varepsilon_\lambda+\varepsilon_{\rm pic}^{\rm achieved}+\operatorname{rel}_{\varepsilon}(\mathbf r_{\rm rot};\rho_K\mathbf g_{\rm eff})+\operatorname{rel}_{\varepsilon}(E_{\rm frame};R)+\operatorname{rel}_{\varepsilon}(\mathbf u_B-\mathbf u;\mathbf u)+\frac{\delta_{na}}{\max\{D_{\rm ref},\varepsilon_{na}\}},
$$

- $\varepsilon_{\rm mom}^{\rm weak}$、$\varepsilon_\lambda$：2.15 (b) 的 Galerkin 动量残差与刚体不相容度（唯一应力场）；
- $\varepsilon_{\rm pic}^{\rm achieved}$：Picard 末轮实际达到的相对增量（2.15 (a)）；
- $\mathbf r_{\rm rot}$：8.7 节强形式残差（分辨率指标）；
- $E_{\rm frame}$：第 10.4.4 步骤 8 参考系重对齐的半径重采样与 $\rho_s$、$\bar\varepsilon_p$ 插值误差，定义为重对齐后半径与细网格射线重采样参考之差 $\|R'-R'_{\rm ref}\|_\infty$（缺省参考时登记为 PO-13）；
- $\mathbf u_B-\mathbf u$：仅当以 Track B（附录 A.4）近似求解（8.4′）时非零，Track A 中定义为零；
- $\delta_{na}$：牵引一致性偏差（10.2.3 节），$D_{\rm ref}>0$ 为配置的参考尺度。

另须单独报告（不与上式求和）：几何投影偏离 $E_{\rm proj}:=\|d_*-d_{\rm pred}\|_{w,\mathcal A}/\max\{\|d_{\rm pred}\|_{w,\mathcal A},\varepsilon_d\}$（SOCP 校正中没有对应流动的部分，属顺序凸投影的模型误差）、质量松弛 $|s_{M*}|/M$、受拉比例、可动面积分数、$\mathrm{Ek}^{-1}$ 与 $\varepsilon_{\rm Euler}$（2.8 节 B1）。

**动态自由表面模式（每个物理步）：**

$$
E_{\rm total}^{\rm dyn}:=\operatorname{rel}_{\varepsilon}(\mathbf r_{\rm mom};\mathbf f_{\rm ext})+\operatorname{rel}_{\varepsilon}(\mathbf B_{\rm rb}^{\!\top}\boldsymbol\lambda;\mathbf m)+\operatorname{rel}_{\varepsilon}(E_{\ell=0};\mathbf m)+\operatorname{rel}_{\varepsilon}(\mathbf u_B-\mathbf u;\mathbf u)+\operatorname{rel}_{\varepsilon}(E_{\rm split};\mathbf m)+\operatorname{rel}_{\varepsilon}(E_{\rm remap};\mathbf m)+\frac{\delta_{na}}{\max\{D_{\rm ref},\varepsilon_{na}\}}.
$$

刚体项取 $\mathbf M_\Gamma^{-1}$ 范数时等于 $E_{\rm rb}$；Track A 中 $\mathbf u_B-\mathbf u$ 项定义为零。

- $\mathbf r_{\rm mom}$：8.4 节整体系统的动量（平衡）残差，$\mathbf f_{\rm ext}$ 为其右端载荷；
- $E_{\ell=0}$：几何非球形诱导的均值截断误差。$\mathcal R_{\rm rb}$ 仅定义在 $\ell\ge1$ 空间，故**不**把含 $\ell=0$ 的量代入它；$E_{\ell=0}$ 定义在对偶（模态系数）空间的欧氏范数上：
$$
E_{\ell=0}:=\|(\mathbf I-\mathcal P_{\ge1})\mathbf m\|_2=\sqrt{\sum_{X,k}|m_{X,0k0}|^2};
$$
- $E_{\rm split}$：假设 A-M 的算子分裂误差，定义为一步输运+源项更新与细分子步参考结果之差 $\|\mathbf m^{\rm split}-\mathbf m^{\rm ref}\|_2$（Lie $O(\Delta t)$，Strang $O(\Delta t^2)$；PO-4）；
- $E_{\rm remap}$：动态模式中几何改变后模态重映射 $\mathbf m^{\rm new}=\int_{\mathcal V^{\rm new}}\rho\,\mathbf B_\alpha^{\rm new}\cdot\mathbf v^{\rm old}\,dV$ 的误差（$\mathbf v^{\rm old}$ 在 $\mathcal V^{\rm new}\setminus\mathcal V^{\rm old}$ 上按其模态多项式在 $\zeta>1$ 的解析延拓取值），定义为 $\|\mathbf m^{\rm remap}-\mathbf m^{\rm cons,new}\|_2$，参考由细化重映射给出，缺省时登记为 PO-4；
- 离散截断类误差（$E_{\rm radial}$：$K=4$ 径向截断；$E_{\rm angular}$：$L_{\max}$ 截断；$E_{\rm char}=O(\Delta t^p)$；$E_{\rm interp}=O(h_\Gamma^2)$；$E_{\rm alias}$ 混叠；$E_{\rm RTFP}$）不与上式求和，各自以相对差 $\|\mathbf u_{\rm fine}-\mathbf u\|/\|\mathbf u_{\rm fine}\|$ 的网格/阶数加密实验单独报告，如 $E_{4\to8}$。

固定 $K=4$ 与 $L_{\max}$ 时只能说离散解收敛到该降阶模型，不收敛到完整三维解；收敛性须用 $K=8$ 与更高 $L_{\max}$ 的比较 $E_{4\to8}=\|\mathbf u_8-\mathbf u_4\|/\|\mathbf u_8\|$ 检验。

---

## 9.9 时间含义的解耦

| | 动态自由表面 | 准静态主线（模式 B/C） |
|---|---|---|
| 时间变量 | 物理时间 $t$，步长受惯性与 9.3 节显式源项限制 | 地质时间 $t_n$，$t_{n+1}=t_n+\lambda\Delta t_n$；$\Delta t_n$ 只受可动区几何 CFL、侵蚀通量与自转表分辨率限制（10.4.1 节） |
| 速度 | 由输运的模态动量 $\mathbf m$ 恢复（9.4 节） | 每步由（8.4′）+8.7 节隐式求出的瞬时 Stokes 蠕变速度，不跨步携带 |
| 惯性项 | 保留 | 全部略去（$\mathbf M=\mathbf N=\mathbf C=\mathbf 0$） |
| $R$ 的演化 | $\partial_tR+\mathbf c\cdot\nabla_{S^2}R=v_r$（命题 9.1） | $\partial_tR=\chi_Y\,\mathbf v_{\rm sl}\cdot\mathbf n/(\mathbf n\cdot\mathbf e_r)$ 加模式 C 通量，经 $\Delta t_n$ 离散为 $d_{\rm pred}$，再由 SOCP 校正 |
| $R$ 的更新 | 随 $\partial_tR$ 积分 | $R^{(n+1)}=R^{(n)}+\lambda\mathbf d_*$（锁死集 $d_*=0$） |
| 输运 | 8 个模态场 $\mathbf m$ | 仅面密度 $\rho_s$（9.2 节） |
| 自转 | 外生给定 $\omega(t)$ | 外生给定 $\omega(t)$，每步冻结 $\omega_n$ |

两种模式不得在同一个循环里混用。准静态主线的阶段平衡条件是可动集为空且力学验收成立（第 2.15 节 (c)）。

---

# 10. $K=4$ 闭合与 SCF 几何耦合

锁定径向 Galerkin 截断阶数 $K=4$。其含义是：每个球谐指标 $(\ell,m)$、每个速度势（极向 $\Phi$、环向 $\Psi$）各保留 4 个径向模式，所以每个 $(\ell,m)$ 有 $8$ 个速度自由度，速度系数向量为 $\mathbf u=(\mathbf q,\mathbf s)$（第 8.2 节）。径向基统一采用 $\ell$ 依赖的移位 Jacobi 正交基

$$
\psi_{\ell k}(\zeta)=\zeta^{\ell}P_k^{(0,\,2\ell+2)}(2\zeta-1),\qquad \zeta=\frac{r}{R(\hat{\boldsymbol\omega},t)},\quad k=0,1,2,3,
$$

势的展开为

$$
\Phi=R^2\sum_{\ell\ge1,m,k}q_{\ell km}\,\psi_{\ell k}(\zeta)\,Y_{\ell m},\qquad
\Psi=R\sum_{\ell\ge1,m,k}s_{\ell km}\,\psi_{\ell k}(\zeta)\,Y_{\ell m}.
$$

本章与第 8 章使用同一套系数 $(q_{\ell km},s_{\ell km})$，没有额外的重标度。$R$ 随时间或方向变化所产生的导数项由第 9 章的源项 $\boldsymbol{\mathcal S}$ 和第 10.2 节的迹公式显式计入。

本章内容为：10.1 节在固定参考几何下给出质量块的解析结构与条件数；10.2 节给出一般星形表面上的速度迹、曲面不变剪切率与牵引条件；10.3 节由剪切率生成摩擦角；10.4 节给出几何增量子问题与 SCF 循环。

---

## 10.1 固定参考几何下的对角块性质与基底正交性

本节取固定参考球 $R\equiv R_0$ 与常数密度 $\rho\equiv\rho_0$，在此参考几何下明确第 8.3 节的质量块。它们用于单元测试与预条件子构造；变几何、变密度的生产问题必须按第 8.3 节的一般定义（命题 8.7、8.9）重新装配，并在每个外层步复核对称正定性。

**命题 10.1（参考几何下的对角块性质与基底正交性）[PO-6 的固定参考块条件性闭合]。** 设 $R\equiv R_0$，$\rho\equiv\rho_0$，$\ell\ge1$，$\Lambda=\ell(\ell+1)$。

1. **正交性。**
$$
\int_0^1\psi_{\ell j}\,\psi_{\ell k}\,\zeta^2\,d\zeta=\frac{\delta_{jk}}{2k+2\ell+3}.
$$
2. **环向块严格对角。** 单个 $(\ell,m)$ 的环向动能质量块为
$$
\mathbf M^T_\ell=\rho_0R_0^5\,\Lambda\,\operatorname{diag}\!\Big(\frac1{2\ell+3},\ \frac1{2\ell+5},\ \frac1{2\ell+7},\ \frac1{2\ell+9}\Big).
$$
3. **极向块。**
$$
\mathbf M^P_\ell=\rho_0R_0^5\,\Lambda\big(\Lambda\,\tilde{\mathbf N}_\ell+\tilde{\mathbf P}_\ell\big),\qquad
(\tilde{\mathbf N}_\ell)_{ij}=\int_0^1\psi_{\ell i}\psi_{\ell j}\,d\zeta,\quad
(\tilde{\mathbf P}_\ell)_{ij}=\int_0^1(\zeta\psi_{\ell i})'(\zeta\psi_{\ell j})'\,d\zeta,
$$
$\tilde{\mathbf N}_\ell,\tilde{\mathbf P}_\ell$ 对称正定，所以 $\mathbf M^P_\ell$ 对称正定。
4. **块间解耦。** 不同 $(\ell,m)$ 之间、以及极向与环向之间的质量块为零（命题 5.7）。因此 $\mathbf M_\Gamma=\operatorname{blockdiag}(\mathbf M^P_{\ell},\mathbf M^T_{\ell})$，$\mathbf M_\Gamma$ 在 $\ell\ge1$ 上的最小、最大特征值由各块决定。
5. $\ell=0$ 的块恒为零，不参与任何求逆。

*证明。* 1 即命题 8.3(1)。2：环向块元为 $\Lambda\rho_0\int f_if_jr^2dr$（命题 5.7），$f_i=R_0\psi_{\ell i}(r/R_0)$，代入 $r=R_0\zeta$ 得 $\rho_0R_0^5\Lambda\int\psi_{\ell i}\psi_{\ell j}\zeta^2d\zeta$，由 1 得对角。3：命题 5.7 的极向块 $\Lambda^2\!\int\rho g_ig_jdr+\Lambda\!\int\rho(rg_i)'(rg_j)'dr$，$g_i=R_0^2\psi_{\ell i}(r/R_0)$，$rg_i=R_0^3\zeta\psi_{\ell i}$，$(rg_i)'=R_0^2(\zeta\psi_{\ell i})'$，变量代换得上式。$\tilde{\mathbf N}_\ell$ 是线性无关函数 $\psi_{\ell i}$ 的 Gram 矩阵；$\tilde{\mathbf P}_\ell$ 是 $(\zeta\psi_{\ell i})'$ 的 Gram 矩阵，而 $(\zeta\psi)'=0$ 意味着 $\zeta\psi\equiv$ 常数，唯有 $\psi=0$（多项式），故线性无关。4 由命题 5.7。∎

**$\ell=1$ 的解析矩阵。** 由 $\psi_{10}=\zeta$、$\psi_{11}=6\zeta^2-5\zeta$、$\psi_{12}=28\zeta^3-42\zeta^2+15\zeta$、$\psi_{13}=120\zeta^4-252\zeta^3+168\zeta^2-35\zeta$：

$$
\tilde{\mathbf N}_1=\begin{pmatrix}
\frac13&-\frac16&\frac1{10}&-\frac1{15}\\[0.4ex]
-\frac16&\frac8{15}&-\frac25&\frac{61}{210}\\[0.4ex]
\frac1{10}&-\frac25&\frac45&-\frac7{10}\\[0.4ex]
-\frac1{15}&\frac{61}{210}&-\frac7{10}&\frac{17}{15}
\end{pmatrix},\qquad
\tilde{\mathbf P}_1=\begin{pmatrix}
\frac43&\frac73&\frac95&\frac{32}{15}\\[0.4ex]
\frac73&\frac{122}{15}&\frac{42}5&\frac{797}{105}\\[0.4ex]
\frac95&\frac{42}5&\frac{86}5&\frac{81}5\\[0.4ex]
\frac{32}{15}&\frac{797}{105}&\frac{81}5&\frac{428}{15}
\end{pmatrix},
$$

$$
\mathbf M^T_1=2\rho_0R_0^5\operatorname{diag}\!\Big(\frac15,\frac17,\frac19,\frac1{11}\Big),\qquad
\mathbf M^P_1=2\rho_0R_0^5\big(2\tilde{\mathbf N}_1+\tilde{\mathbf P}_1\big).
$$

（$\ell=2$ 的环向块为 $6\rho_0R_0^5\operatorname{diag}(\tfrac17,\tfrac19,\tfrac1{11},\tfrac1{13})$，极向矩阵用同一积分定义计算；所有元素已用精确有理运算复核。）

**条件数（4×4 块）。** 下表是用精确有理 Gram 矩阵再转 float64 计算的 $2$-范数条件数；"单项式对照基"仅用于条件数对照（取 $\zeta^2,\ldots,\zeta^5$ 经同一公式计算），不参与任何装配。

$$
\begin{array}{c|ccccc}
\ell&1&2&3&5&10\\\hline
\operatorname{cond}(\mathbf M^P_\ell),\ \text{Jacobi}&30.1&27.3&26.1&25.1&24.5\\
\operatorname{cond}(\mathbf M^P_\ell),\ \text{对角缩放后}&6.6&8.6&10.2&12.5&15.8\\
\operatorname{cond}(\mathbf M^T_\ell),\ \text{Jacobi}&2.2&1.86&1.67&1.46&1.26\\
\operatorname{cond}(\mathbf M^P_\ell),\ \text{单项式对照基}&4.0\times10^5&3.4\times10^5&3.2\times10^5&3.2\times10^5&3.8\times10^5
\end{array}
$$

此前单项式基下 $L^2$ 质量矩阵 $\big(1/(i+j+5)\big)_{i,j=1}^4$ 的条件数为 $2.35\times10^6$；Jacobi 基下对应的 $L^2$ 质量矩阵 $\int\psi_i\psi_j\zeta^2d\zeta$ 是对角矩阵，条件数 $(2\ell+9)/(2\ell+3)\le 11/5$。动能质量块的条件数由约 $10^5$ 降至约 $25$–$30$（对角缩放后 $\le16$），降低约 4 个数量级。

这一改善是**块内**的；不同 $\ell$ 的块的特征值尺度随 $\Lambda_\ell^2\sim\ell^4$ 增长，所以整体 $\operatorname{cond}(\mathbf M_\Gamma|_{\ell\ge1})$ 仍随 $L_{\max}$ 增大。这部分由第 9.4 节的块对角预条件子（预条件子取 $\mathbf M_{B1}^{-1}$，即各 $(\ell,m)$ 块的逆）处理。

**与第 10.2 节的衔接。** 下一节需要 $\psi_{\ell k}$ 在 $\zeta=1$ 的值与导数，它们来自命题 8.3(2)：$\psi_{\ell k}(1)=1$，$\sigma_{\ell k}=\psi_{\ell k}'(1)=\ell+k(k+2\ell+3)$，二阶导 $\kappa_{\ell k}=\psi_{\ell k}''(1)$ 在 10.2.1 节列表。符号 $\Lambda_\ell$、$Y_{\ell m}$、$R(\hat{\boldsymbol\omega},t)$、$\nabla=\nabla_{S^2}$ 沿用本节与第 8 章的定义。

---

## 10.2 自由表面速度、曲面不变剪切率与牵引条件

本节所用速度系数、径向基与展开均按第 8.1–8.2 节：$\Phi=R^2\sum q_{\ell km}\psi_{\ell k}(r/R)Y_{\ell m}$，$\Psi=R\sum s_{\ell km}\psi_{\ell k}(r/R)Y_{\ell m}$，$R=R(\hat{\boldsymbol\omega},t)$ 为星形边界。记 $\Lambda=\ell(\ell+1)$，$\nabla:=\nabla_{S^2}$，$\Delta:=\Delta_{S^2}$ 为单位球面算子。

### 10.2.1 径向基的表面迹

对 $\psi_{\ell k}(\zeta)=\zeta^\ell P_k^{(0,2\ell+2)}(2\zeta-1)$，记 $\sigma_{\ell k}:=\psi'_{\ell k}(1)$、$\kappa_{\ell k}:=\psi''_{\ell k}(1)$。已用符号计算验证 $\psi_{\ell k}(1)=1$ 与

$$
\boxed{\ \sigma_{\ell k}=\ell+k(k+2\ell+3),\qquad
\begin{array}{c|cccc}
k&0&1&2&3\\\hline
\sigma_{\ell k}&\ell&3\ell+4&5(\ell+2)&7\ell+18\\
\kappa_{\ell k}&\ell(\ell-1)&\ell(5\ell+7)&(\ell+2)(13\ell+15)&(\ell+2)(25\ell+63)
\end{array}\ }
$$

### 10.2.2 一般星形表面上的速度迹

**命题 10.2。** 对 $\Phi,\Psi$ 取上述展开，在 $r=R(\hat{\boldsymbol\omega})$ 处（仅用 $\nabla,\Delta$ 作用于 $Y_{\ell m}$ 与 $R$），对每个模式 $(\ell,k,m)$ 的贡献为

$$
\begin{aligned}
v_r\big|_\Gamma&=\sum q_{\ell km}\,\frac1R\Big[R^2\Lambda Y-2R(2-\sigma)\,\nabla R\cdot\nabla Y-Y\big(R(2-\sigma)\Delta R+(2-2\sigma+\kappa)|\nabla R|^2\big)\Big],\\
\mathbf v_S\big|_\Gamma&=\sum q_{\ell km}\Big[R(1+\sigma)\nabla Y+(2-\kappa)\,Y\nabla R\Big],\\
\mathbf v^T\big|_\Gamma&=\sum s_{\ell km}\Big[R\nabla Y+(1-\sigma)\,Y\nabla R\Big]\times\mathbf e_r,
\end{aligned}
$$

其中 $\sigma=\sigma_{\ell k}$、$\kappa=\kappa_{\ell k}$、$Y=Y_{\ell m}(\hat{\boldsymbol\omega})$。表面速度为 $\mathbf v|_\Gamma=v_r\mathbf e_r+\mathbf v_S+\mathbf v^T$。

*证明。* 由引理 5.5 的恒等式 $v_r=L^2\Phi/r$、$\mathbf v_S=r^{-1}\nabla\,\partial_r(r\Phi)$、$\mathbf v^T=\nabla\Psi\times\mathbf e_r$（$\nabla$ 在固定 $r$ 下作用）。对 $A(\omega;r)=R^2\psi(r/R)$ 有 $A_R=R(2\psi-\zeta\psi')$、$A_{RR}=2\psi-2\zeta\psi'+\zeta^2\psi''$；对 $C=\partial_r(rA)$ 有 $C|_{\zeta=1}=R^2(1+\sigma)$、$C_R|_{\zeta=1}=R(2-\kappa)$；对 $B=R\psi$ 有 $B_R|_{\zeta=1}=1-\sigma$。固定 $r$ 时 $\nabla A=A_R\nabla R$、$\Delta A=A_R\Delta R+A_{RR}|\nabla R|^2$，再用乘积法则 $L^2(AY)=A\Lambda Y-2\nabla A\cdot\nabla Y-Y\Delta A$ 并令 $\zeta=1$ 即得。∎（三式已在非球形 $R=2+0.3\sin\theta\cos\phi+0.2\cos\theta$、$\ell=2$、$k=2$ 上与直接求导的数值比较，在一点上数值一致到约 $10^{-14}$。）

$\nabla R=0$ 时它们退化为 $v_r=R\Lambda q Y$、$\mathbf v_S=R(1+\sigma)q\nabla Y$、$\mathbf v^T=Rs\,\nabla Y\times\mathbf e_r$。

### 10.2.3 曲面不变剪切率

设 $\Gamma$ 的局部坐标 $x^a$（$a=1,2$），切基 $\mathbf e_a=\partial_aX$，度量 $g_{ab}=\mathbf e_a\cdot\mathbf e_b$（9.0 节），单位法向 $\mathbf n$，第二基本形式 $K_{ab}:=\mathbf e_a\cdot\partial_b\mathbf n=-\mathbf n\cdot\partial_a\partial_bX$（外法向球面上 $K_{ab}=g_{ab}/R>0$），平均曲率 $H=g^{ab}K_{ab}$。把 $\mathbf v|_\Gamma$ 分解为

$$
v_a=\mathbf v\cdot\mathbf e_a,\qquad v_n=\mathbf v\cdot\mathbf n=(\mathbf n\cdot\mathbf e_r)\,\partial_tR ,
$$

$\nabla_a$ 为 $g_{ab}$ 的 Levi-Civita 协变导数。则 $\mathbf D=\operatorname{sym}\nabla\mathbf v$ 在表面标架下的分量为

$$
\boxed{\ 
D_{ab}=\nabla_{(a}v_{b)}+v_n\,K_{ab},\qquad
D_{nn}=-g^{ab}D_{ab}=-\big(\nabla_av^a+H\,v_n\big),\qquad
D_{na}=\mathbf n\cdot\mathbf D\cdot\mathbf e_a .\ }
$$

*推导。* $D_{ab}=\mathbf e_a\cdot\mathbf D\cdot\mathbf e_b=\tfrac12(\mathbf e_a\cdot\partial_b\mathbf v+\mathbf e_b\cdot\partial_a\mathbf v)$ 只含切向导数；将 $\mathbf v=v^c\mathbf e_c+v_n\mathbf n$ 代入，用 Gauss–Weingarten 公式 $\partial_a\mathbf e_b=\Gamma^c_{ab}\mathbf e_c-K_{ab}\mathbf n$、$\partial_a\mathbf n=K_a{}^c\mathbf e_c$ 得 $\mathbf e_a\cdot\partial_b\mathbf v=\nabla_bv_a+v_nK_{ab}$。$D_{nn}$ 含法向导数，不能由表面值直接取，但不可压缩性 $\operatorname{tr}\mathbf D=D_{nn}+g^{ab}D_{ab}=0$ 给出上式。∎

$\mathbf D$ 为迹零，$\dot\gamma^2=2\mathbf D:\mathbf D$。在表面标架（逆度量 $g^{ab}$）下

$$
\boxed{\ 
\dot\gamma_\Gamma^2=2\Big[(g^{ab}D_{ab})^2+g^{ac}g^{bd}D_{ab}D_{cd}+2\,g^{ab}D_{na}D_{nb}\Big].\ }
$$

**牵引条件。** 自由表面 $\boldsymbol\sigma\mathbf n=-p_{\rm ext}\mathbf n$，$\boldsymbol\sigma=-p\mathbf I+2\eta\mathbf D+\mathbf T_{\rm tide}$ 时，法向分量给出 $p|_\Gamma=p_{\rm ext}+\mathbf n\cdot\boldsymbol\tau\cdot\mathbf n$（第 4.22 节），切向分量给出

$$
\boxed{\ D_{na}=-\frac{\mathbf n\cdot\mathbf T_{\rm tide}\cdot\mathbf e_a}{2\eta}\quad(\eta>0).\ }
$$

在混合弱式中该条件是自然边界条件，只在 Galerkin 意义下成立；本文把它同时作为曲面剪切率闭合的**一致性关系**使用：冻结 $\eta^{(n)}$、$\mathbf T^{(n)}_{\rm tide}$ 后代入上式得 $D_{na}$，再算 $\dot\gamma_\Gamma$。无潮汐时 $D_{na}=0$，上式第三项消失，

$$
\dot\gamma_\Gamma^2\big|_{\mathbf T_{\rm tide}=0}=2\Big[(g^{ab}D_{ab})^2+g^{ac}g^{bd}D_{ab}D_{cd}\Big].
$$

$\eta$ 本身依赖 $\dot\gamma$（隐式），故 SCF 中取上一迭代的 $\eta^{(n)}$ 冻结；这是迭代近似，不是代数恒等式。

**自洽性报告。** 实际 Galerkin 解的 $D_{na}$ 不严格等于上式；两者之差 $\delta_{na}:=\|D_{na}^{h}-D_{na}^{\rm target}\|_{L^2(\Gamma)}$ 必须随 $K,L_{\max}$ 报告，并作为降阶模型对牵引条件满足程度的度量。

### 10.2.4 特例：固定参考球面

**命题 10.3（降级后的 $D_{rS}=0$）。** 仅在 (i) $R\equiv R_0$ 为常数（固定参考球面，$\mathbf n=\mathbf e_r$，$K_{ab}=g_{ab}/R_0$）；(ii) 速度仅含单个 $(\ell,m)$ 的球谐阶，各 $(\ell,m)$ 解耦；(iii) $\mathbf T_{\rm tide}=\mathbf 0$ 时，命题 10.2 与上述牵引条件化为：

$$
v_r=R_0\Lambda\sum_kq_kY,\quad
D_{rr}=\Lambda\sum_k(\sigma_{\ell k}-1)q_kY,\quad
2D_{rS}=\Big[\sum_k(\Lambda+\kappa_{\ell k}-2)q_k\Big]\nabla Y+\Big[\sum_k(\sigma_{\ell k}-1)s_k\Big]\nabla Y\times\mathbf e_r ,
$$

$$
D_{SS}=\sum_k\Big[(1+\sigma_{\ell k})q_k\operatorname{Hess}_SY+\Lambda q_kY\,\mathbf I_S+s_k\operatorname{sym}\nabla\big(\nabla Y\times\mathbf e_r\big)\Big].
$$

$\operatorname{tr}D_{SS}=-\Lambda\sum_k(\sigma_{\ell k}-1)q_kY=-D_{rr}$，与不可压缩性一致。因 $\nabla Y$ 与 $\nabla Y\times\mathbf e_r$ 点点正交，$D_{rS}=0$ 等价于两个独立的标量约束

$$
\boxed{\ \sum_{k=0}^3c^{P}_{\ell k}q_{\ell km}=0,\ \ c^P_{\ell k}=\Lambda_\ell+\kappa_{\ell k}-2;\qquad
\sum_{k=0}^3c^{T}_{\ell k}s_{\ell km}=0,\ \ c^T_{\ell k}=\sigma_{\ell k}-1.\ }
$$

$$
\begin{array}{c|cccc}
k&0&1&2&3\\\hline
c^P_{\ell k}&2(\ell-1)(\ell+1)&2(3\ell^2+4\ell-1)&14(\ell+1)(\ell+2)&2(\ell+2)(13\ell+31)\\
c^T_{\ell k}&\ell-1&3(\ell+1)&5\ell+9&7\ell+17
\end{array}
$$

一致性检验：$\ell=1,k=0$ 的平移与转动模式 $c^P_{10}=c^T_{10}=0$，即刚体运动不产生表面剪切。

此时 $\dot\gamma_\Gamma^2=2\big[D_{rr}^2+D_{SS}:D_{SS}\big]$。这一整套公式**仅**适用于上述特例；对移动的非球形 $R(\theta,\varphi)$、含环向耦合的跨阶模态、或含潮汐背应力，必须使用 10.2.3 的一般公式，不得再令 $D_{rS}=0$。若进一步取 $s_{\ell km}\equiv0$（无环向流动），只剩极向约束。

---

## 10.3 表面内生非均匀物理摩擦角闭合

说明：$p_{\rm ref}=\rho_{\rm bulk}|\mathbf g\cdot\mathbf n|h_s$ 是浅表流动层的代表性参考压力；所构造的 $\mu_{\rm eff}$ 是基于表层代表性压力 $p_{\rm ref}$ 的名义几何投影角（名义几何投影角：在无内聚或高围压下对压力单调，一般情形不保证保守；真实多轴屈服由外层验收单独检验）。物理有效压力 $p_{*,\mathrm{phys}}=p+p_c$ 仅用于动态流变 $\mu_{d}$，参考有效压力 $p_{*,\mathrm{ref}}=p_{\mathrm{ref}}+p_c$ 仅用于几何投影角 $\mu_{\rm eff}$；本节不使用未带下标的 $p_{*}$。

**材料常量固化。** 密实度 $\nu=\nu_0$（如标准碎石堆 $\nu_0=0.6$）与配位数 $Z=Z_0$（如 $Z_0=6$）是**配置给定的材料常量**，在整个 SCF 与松弛演化中不随迭代更新，也没有演化律；因此 $\rho_{\rm bulk}=\rho_p\nu_0$、$c=c(\nu_0,Z_0)$、$\mu_s=\mu_s(\nu_0,Z_0)$、$\mu_2$、$I_0$、$\mu_y\equiv\mu_s$（2.8 节 B2）、$F_H$ 均为常数（以下沿用原符号但不再带节点或迭代下标）。第 4.7–4.20 节中 $\nu^{(n)},Z^{(n)}$ 的上标 $(n)$ 是退化定理的**参数序列指标**，与 SCF 迭代无关。$\rho_{\rm bulk}$ 是松散表层的堆积密度，$\rho_K$ 是内部 Cauchy 连续体密度，二者是两个独立的配置量，仅通过密度权 $\rho_{\rm bulk}/\rho_K|_\Gamma$ 联系（2.8 节）；应逐步报告比值 $\rho_{\rm bulk}/\rho_K|_\Gamma$ 的范围。

以下量均在第 $n$ 个地质步的冻结几何 $R^{(n)}$、冻结有效重力 $g_i^{(n)}$ 和第 8.7 节的收敛压力 $p_i^{(n)}$（取正则化值 $\tilde p_i^{(n)}=\max([p_i^{(n)}]_+,p_{\min})$，2.8 节）上评价。先定义

$$
 p_{*,{\rm phys},i}^{(n)}:=\tilde p_i^{(n)}+p_{c}\ge p_{\min}>0,
 \qquad
 p_{c}=C_c\frac{Z_0\nu_0F_{H}}{d_p^2}.
$$

若采用特定接触模型给出的 $F_{H}=A_Ha/(6h_0^2)$，再将该表达式代入。数值下界 $p_{\min}>0$（如 $10^{-6}\,\mathrm{Pa}$，f32 实现须在 f64 下复核）消除 $p=0,\ p_c=0$ 时 $\sqrt{p_*/\rho_p}$ 与 $\mu_{\rm eff}$ 分母的除零奇点；$p_{*,{\rm ref}}$ 同样取 $\max(\cdot,p_{\min})$。该下界仅作数值保护，不得只用于分母：按第 4.13 节的强制规则，若在 $\mu_d$ 的分母中使用了下界压力，则应力幅值 $c+\mu_dp$ 中的 $p$ 须一致地取同一正则化压力 $\max(p,p_{\min})$。

表面剪切率采用 $\dot\gamma=\sqrt{2\mathbf D':\mathbf D'}$，节点值取 $\mathcal F_i:=\dot\gamma_i^{(n)}=\dot\gamma_\Gamma(\hat{\boldsymbol\omega}_i)$，由 10.2.3 的一般公式用第 $n$ 步准静态 Stokes 的**收敛解** $\mathbf u^{(n)}$（8.4′、10.4.2 节）计算；它与 2.8 节屈服判据、8.7 节压力使用同一速度场；固定球面的 10.2.4 公式仅作为其特例核验。

**参考压力与 $p_*=0$ 奇点的处理。** 自由表面节点上 $p_i=p_{\rm ext}+\mathbf n\cdot\boldsymbol\tau\cdot\mathbf n$ 可为零（第 4.22 节），$I_c$ 与 $\mu_{\rm eff}$ 的分母可为零。坡度代理取**参考深度**有效压力

$$
p_{\mathrm{ref},i}=\rho_{\mathrm{bulk}}|g_i^{(n)}\cdot n_i^{(n)}|\,h_s,\qquad
p_{*,{\rm ref},i}:=\max(p_{\mathrm{ref},i}+p_{c},\,p_{\min})>0,
$$

其中 $h_s>0$ 为配置的活动表层厚度；$p_{*,{\rm ref},i}>0$ 由显式下界 $p_{\min}>0$ 保证，防倒悬约束 $g_i\cdot\widehat n_i\le-\epsilon|g_i|$ 与 $|g_i|>0$ 只保证参考压力具有物理压缩方向。为避免 $\sqrt{p_{*,{\rm phys}}/\rho_p}$ 的除零/下溢，浅表动态摩擦系数 $\mu_{d,i}$ 取已定义并正则化的物理有效压力 $p_{*,{\rm phys},i}$，用有理式

$$
\mu_{d,i}=\mu_s+(\mu_2-\mu_s)\frac{d_p\,\mathcal F_i}{I_0\sqrt{p_{*,{\rm phys},i}/\rho_p}+d_p\,\mathcal F_i},
$$

它关于 $\mathcal F_i\ge0$ 单调不减，值域 $[\mu_s,\mu_2)$；$p_{*,{\rm phys}}\downarrow0$ 且 $\mathcal F_i>0$ 时趋于 $\mu_2$，仅当 $p_{*,{\rm phys}}=\mathcal F_i=0$ 时是真正的不定式，此时规定走静态分支 $\mu_d:=\mu_s$。f32 实现须对 $p_{*,{\rm phys}}$ 设下限 $p_{\min}>0$ 并在 f64 下复核。

**流变链开关（配置项 `mu_proj_mode`，必须显式给定）。**

- `dynamic`：$\mu_{\mathrm{proj},i}:=\mu_{d,i}$。此时剪切率链 $\mathbf u^{(n)}\to\dot\gamma_i\to I_c\to\mu_d\to\mu_{\rm eff}\to\phi_i$ **完全生效**：表面速度迹（命题 10.2）与曲面不变剪切率（10.2.3）直接决定可动节点的几何投影角；
- `static`：$\mu_{\mathrm{proj},i}:=\mu_y=\mu_s$，$\phi_i$ 是与剪切率无关的**静态几何松弛代理**。

两种选项都不改变屈服分类：2.8 节的 $F_\delta$、刚性核与可动集、模式 C 通量一律在唯一应力场上用 $\mu_d$（黏度）与 $\mu_y$（强度）计算，与 `mu_proj_mode` 无关。

两种选项下 $\phi_i$ 都只是 SOCP 的名义几何代理，不是真实多轴屈服的等价改写。把 $\phi=\arctan\mu$ 识别为安息角是 Mohr–Coulomb 与 Drucker–Prager 之间的建模假设，仅对已声明适用的松散表层成立。

**潮汐背应力的标量化与名义投影角（基于表层代表性压力 $p_{\rm ref}$；非绝对保守）。** 记 $\|A\|_{\mathrm{eq}}=\|A\|_F/\sqrt2$。屈服条件 $\|s-T\|_{\mathrm{eq}}\le c+\mu_yp$ 只能推出 $\|s\|_{\mathrm{eq}}\le c+\mu_yp+\|T\|_{\mathrm{eq}}$，这是**必要**条件，并不保守：例如 $c=0,\ \mu_yp=1,\ T=\mathrm{diag}(0.6,-0.6,0),\ s=\mathrm{diag}(-0.5,0.5,0)$ 时 $\|s\|_{\mathrm{eq}}=0.5\le1.6$，但 $\|s-T\|_{\mathrm{eq}}=1.1>1$ 已屈服。对给定压力 $p$ 的**充分**条件由 $\|s-T\|_{\mathrm{eq}}\le\|s\|_{\mathrm{eq}}+\|T\|_{\mathrm{eq}}$ 得到：若 $\|s\|_{\mathrm{eq}}\le c+\mu_yp-\|T\|_{\mathrm{eq}}$，则 $\|s-T\|_{\mathrm{eq}}\le c+\mu_yp$。故取

$$
\mu_{\mathrm{eff},i}^{(n)}
:=\frac{\Big[c+\mu_{\mathrm{proj},i}\,p_{\mathrm{ref},i}-\|T_{\mathrm{tide},i}\|_F/\sqrt2\Big]_+}{p_{*,\mathrm{ref},i}}\ge0,
\qquad
\boxed{\phi_i^{(n)}=\arctan\mu_{\mathrm{eff},i}^{(n)}}.
$$

**单调性与反例（$\phi_i$ 不是绝对保守角）。** 记 $T:=\|T_{\rm tide}\|_{\rm eq}$、$\mu:=\mu_{\rm proj}$，把代理角视为压力的函数 $\mu_{\rm eff}(p)=[c+\mu p-T]_+/(p+p_c)$。当 $c+\mu p-T>0$ 时
$$
\frac{d}{dp}\,\frac{c+\mu p-T}{p+p_c}=\frac{\mu(p+p_c)-(c+\mu p-T)}{(p+p_c)^2}=\frac{\mu p_c-(c-T)}{(p+p_c)^2}.
$$
故 $\mu_{\rm eff}$ 关于 $p$ 非减当且仅当 $\mu p_c\ge c-T$。此时 $p_{\rm ref}\le p$ 意味着 $\mu_{\rm eff}(p_{\rm ref})\le\mu_{\rm eff}(p)$，代理角不高于真实压力下的值（偏紧）。反之若 $c-T>\mu p_c$（内聚较大），$\mu_{\rm eff}$ 关于 $p$ 严格递减，$p_{\rm ref}<p$ 时代理角**反而偏大**（偏松），“$p_{\rm ref}\le p$ 即保守”不成立。因此对一般颗粒体不作“绝对保守充分”的声明：$\phi_i$ 仅是基于表层代表性压力 $p_{\rm ref}$ 的名义几何投影角（在无内聚，$c=0$、$T\ge0$ 时 $\mu p_c\ge-T$ 恒成立，或高围压下单调），作为 SOCP 的几何代理，不是代数等价改写；一般情形下真实多轴屈服由外层验收中的张量屈服条件 $\|s-T_{\rm tide}\|_{\rm eq}\le c+\mu_yp$ 单独检验。

由 $\mu_{\mathrm{eff}}=\tan\phi$，投影中使用

$$
\sin^2\phi_i^{(n)}=\frac{(\mu_{\mathrm{eff},i}^{(n)})^2}{1+(\mu_{\mathrm{eff},i}^{(n)})^2}.
$$

**顶点 Hessian 的规范伪效应。** $T_{\mathrm{tide}}\propto\operatorname{dev}H_W$ 在网格顶点和棱上取 $H^{\mathrm{FP}}_{\ell_0}$，而那里 $C_{ij}\ne0$，结果依赖任意参考长度 $\ell_0$（第 1.3 节：$\Delta H=G\rho C\log(\ell_0/\ell_1)$），是多面体离散的伪量。为用于应力，须在面片内部（面心）求值，或取面片两侧单侧极限的平均（对平面片内点，$H^{\pm}=H^{\mathrm{FP}}\pm2\pi G\rho_K\,\mathbf n\mathbf n^{\!\top}$，“$+$”为外侧极限、“$-$”为内侧极限；$H^{\rm FP}$ 含全部非局部有限部分，面片内点 $C_{ij}=0$，故与 $\ell_0$ 无关，两侧平均恰为 $H^{\mathrm{FP}}$，跳跃 $H^+-H^-=4\pi G\rho_K\mathbf n\mathbf n^{\!\top}$ 与 $\operatorname{tr}H^{-}-\operatorname{tr}H^{+}=-4\pi G\rho_K$ 一致；不得把非局部部分丢弃而只保留局部接触项 $-\tfrac{2\pi}{3}G\rho_K\mathbf I\pm2\pi G\rho_K\mathbf n\mathbf n^{\!\top}$，否则大尺度潮汐张量被漏掉）；顶点值不得直接进入 $T_{\mathrm{tide}}$。

---

## 10.4 与前述方程组的几何自洽场（SCF）闭环

### 10.4.1 时间层级、外生自转与地质步长

准静态主线有三个嵌套层级，各层的“时间”含义不同，不得混用：

| 层级 | 变量 | 含义与求解方式 |
|---|---|---|
| 外生自转表 | $t\mapsto\omega(t)$ | 配置给定，**不求解**；不与内部流动、惯量变化或角动量做任何反馈 |
| 地质步 $n$ | $t_n$，$\Delta t_n$，$\lambda$ | 几何推进 $R^{(n)}\to R^{(n+1)}$，$t_{n+1}=t_n+\lambda\Delta t_n$ |
| Picard 迭代 $k$ | $(\mathbf u^k,p^k)$ | 固定几何与 $\omega_n$ 下瞬时 Stokes 平衡的非线性求解，**不含时间** |

**外生自转表（配置项 `spin_schedule`，必须显式给定其一）。**

- `stages`：$\{(\omega_j,T_j)\}_{j=1}^{J}$，$T_j>0$；阶段 $j$ 占据 $[s_{j-1},s_j)$，$s_0=0$，$s_j=s_{j-1}+T_j$，其上 $\omega(t)\equiv\omega_j$；$T_{\rm end}=s_J$。
- `linear`：$\omega(t)=\omega_0+\dot\omega_{\rm ext}\,t$，$t\in[0,T_{\rm end}]$，$\dot\omega_{\rm ext}$ 为给定常数（如 YORP 平均加速率），要求 $\omega(t)\ge0$。
- `table`：节点 $(t_k,\omega_k)_{k=0}^{K_\omega}$，$t_k$ 严格递增，$\omega$ 取分段线性插值，$T_{\rm end}=t_{K_\omega}$。

每个地质步冻结 $\omega_n:=\omega(t_n)$；$\omega$ 只经 $\mathbf g_{\rm eff}=\nabla U+\omega_n^2\mathbf x_\perp$ 与 $H_W=\nabla^2U+\omega_n^2\operatorname{diag}(1,1,0)$（从而 $\mathbf T_{\rm tide}$）进入模型。**假设 A-S（主轴自转）：** 自转轴始终是体坐标 $z$ 轴，即当前最大惯量主轴（由步骤 8 的重对齐保证）；章动阻尼视为相对 $\Delta t_n$ 瞬时完成。$\omega$ 的取值从不由惯量张量或角动量更新——形变改变 $I_1$ 不会反馈到 $\omega$，这是外生给定的定义，不是遗漏。

**地质步长。**

$$
\Delta t_n=\min\{\Delta t_{\rm CFL},\ \Delta t_C,\ \Delta t_{\rm char},\ \Delta t_\omega,\ \Delta t_{\max},\ t_{\rm next}-t_n\},
$$

- 可动区几何 CFL：$\Delta t_{\rm CFL}=\theta_{\rm CFL}\,h_{\Gamma,\min}\big/\max_{i\in\mathcal A^{(n)}}\big(|\mathbf v_{{\rm sl},i}\cdot\mathbf n_i|/(\mathbf n_i\cdot\mathbf e_r)\big)$，$\theta_{\rm CFL}\le0.5$；$\mathcal A^{(n)}=\varnothing$ 时取 $+\infty$；
- 侵蚀通量：$\Delta t_C=\theta_C\min\Big\{\dfrac{\sigma_0}{k_e\max_i[F_{\delta,i}]_+},\ \dfrac{\rho_{\rm bulk}h_s}{\max_i|J_{\rm dep}-J_{\rm ero}|_i}\Big\}$，$\theta_C<1$，保证 $\rho_s\ge0$ 且单步层厚变化不超过 $\theta_Ch_s$；无通量时取 $+\infty$；
- $\rho_s$ 特征线单值：$\Delta t_{\rm char}=\theta_{\rm ch}/\|\nabla_{S^2}\mathbf c\|_\infty$，$\theta_{\rm ch}<1$，$\mathbf c=\chi_A\mathbf v_{{\rm sl},S}/R$（第 9.2 节）；
- 自转表分辨率：`stages` 下不设（由 $t_{\rm next}$ 处理）；`linear`/`table` 下 $\Delta t_\omega=\varepsilon_\omega\max\{\omega(t_n),\omega_{\rm ref}\}/\max\{|\dot\omega(t_n)|,\epsilon\}$，使单步相对自转变化不超过 $\varepsilon_\omega$；
- $\Delta t_{\max}$ 为配置上限；$t_{\rm next}$ 为下一个阶段边界、表节点或 $T_{\rm end}$。

**无刚性灾难的理由。** 速度由（8.4′）隐式求出，$\Delta t_n$ 不出现在速度方程中，与 $\eta_{\rm hi}=(c+\mu_2\tilde p)/\dot\gamma_\varepsilon$ 无关；刚性核中的正则化蠕变被 $\chi_Y=0$ 剔除，也不进入 $\Delta t_{\rm CFL}$。因此所需步数约为 $T\cdot\max_{\mathcal A}|\mathbf v_{\rm sl}\cdot\mathbf n|/(\theta_{\rm CFL}h_{\Gamma,\min})$，只由真实滑坡速度决定，没有黏性扩散型 CFL；可动集为空的平衡阶段只需一步即跳到阶段终点。$\lambda\in(0,1]$ 缺省为 1，只在外层几何验收困难时减小；累计时间按 $\lambda\Delta t_n$ 计量；因倒悬/自交而缩小信赖域重做的步不推进时间。阶段平衡与终止条件见第 2.15 节 (c)；其稳定性（步长准则的充分性）与有限步达到阶段平衡的条件属 PO-12。

### 10.4.2 准静态 Stokes–压力求解（固定几何与 $\omega_n$）

第 $n$ 个地质步在冻结的 $R^{(n)}$、$\rho_K$、$\omega_n$、$\mathbf g_{\rm eff}$、$\mathbf T_{\rm tide}$ 上执行：

```text
输入：初值 u^0（上一地质步收敛系数原样复用；首步或刚执行过 8b 重对齐时取 u^0 = 0），
      p~^0（上一地质步收敛压力系数在新几何上合成并取 max([.]_+, p_min)；首步取 p_min）。
for k = 0, 1, ..., k_max-1:
  (P1) 压力：以冻结 (u^k, p~^k) 计算 f = rho_K g_eff + div T_tide + div tau_flow 与 Dirichlet
       数据 p_Gamma = p_ext + n.T_tide.n + n.tau_flow.n，解 8.7 节 Poisson 得 p^{k+1}；
       p~^{k+1} = max([p^{k+1}]_+, p_min)。
  (P2) 黏度：在体内求积点 X_vol 上 gd^k = sqrt(2 D(u^k):D(u^k))，
       I_c = d_p gd^k / sqrt((p~^{k+1} + p_c)/rho_p)，mu_d = mu_d(I_c)（4.5 节），
       eta^{k+1/2} = (c + mu_d p~^{k+1}) / sqrt((gd^k)^2 + gd_eps^2)；
       可选松弛 eta <- theta_eta eta^{k+1/2} + (1 - theta_eta) eta^{k-1/2}，theta_eta in (0,1]。
  (P3) 速度：装配 K(eta^{k+1/2}) 与 F，按 8.4 节值域空间法解 (8.4') 得 (u^{k+1}, lambda^{k+1})。
  (P4) 若第 2.15 节判据 (a) 成立则停止。
未在 k_max 内收敛：按 gd_eps 连续化（先以 10 gd_eps 求解，再逐级降到目标值，每级以上一级解为初值）
与减小 theta_eta 重试；仍失败则终止并报告"流变未收敛"，不得输出通过状态。
输出：u^(n), p^(n), lambda^(n), eta^(n)；随即执行第 2.15 节判据 (b)。
```

（P1）的 $\nabla\cdot\boldsymbol\tau_{\rm flow}=\eta\Delta\mathbf v+2\mathbf D\nabla\eta$ 与 $\nabla\cdot\mathbf T_{\rm tide}=-\tfrac{8\pi}{3}GC_T\rho_p\nu_0d_p^2\nabla\rho_K$ 见第 8.7 节。收敛的 $(\mathbf u^{(n)},p^{(n)})$ 同时满足无散 Galerkin 动量方程（8.4′）与压力 Poisson 方程，即唯一应力场 $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D$ 的准静态平衡；2.8 节的屈服分类、10.3 节的摩擦角链与模式 C 通量全部在这同一个解上计算。

**Track B（附录 A.4）。** 若在（P3）中以块对角逆 $\mathbf K_{\gamma,B1}^{-1}$ 代替 $\mathbf K_\gamma^{-1}$（截断模型），所得 $\mathbf u_B$ 仍须用**完整** $\mathbf K$ 计算 2.15 (b) 的 $\varepsilon_{\rm mom}^{\rm weak}$；该残差即 Track B 截断误差的后验量，超阈值时该步必须改用 Track A 重解。

### 10.4.3 几何增量子问题

几何校正是**力学预测后的顺序凸增量子问题**，基准为当前几何 $R^{(n)}$（第 2.13 节）。在当前迭代点作一阶线性化，记

$$
\widehat n_i(d)=n_i^{(n)}+D n_i^{(n)}d,
\qquad
\widehat{\mathcal M}(d)=\mathcal M(R^{(n)},\rho^{(n)})+
D\mathcal M^{(n)}d.
$$

力学预测 $d_{\mathrm{pred}}^{(n)}$ 与节点三分类 $\mathcal L^{(n)}$（锁死）、$\mathcal D^{(n)}$（纯沉积）、$\mathcal A^{(n)}$（可动）按第 2.8 节（B3–B4）唯一给出：

$$
d_{\mathrm{pred},i}^{(n)}=\frac{\chi_{Y,i}\,\Delta t_n\,\mathbf v_{{\rm sl},i}^{(n)}\!\cdot n_i^{(n)}+\dfrac{\rho_{\rm bulk}}{\rho_K(\mathbf x_i)}\,\Delta h_{C,i}}{n_i^{(n)}\cdot\mathbf e_r}\qquad(n\cdot e_r\ge c_0>0),
$$

其中 $\mathbf v_{\rm sl}$ 是第 $n$ 步准静态 Stokes 收敛解相对刚性核的滑移速度（命题 10.2 合成表面迹），$\chi_{Y,i}$ 是基于唯一应力场屈服函数 $F_\delta$ 与刚性核连通性的可动指示，$\Delta h_C=\Delta t_n(J_{\rm dep}-J_{\rm ero})/\rho_{\rm bulk}$，密度权 $\rho_{\rm bulk}/\rho_K$ 使质量行中的通量增量为 $\Delta h_C^*$。Clarabel 求解

$$
\boxed{
\begin{aligned}
(d_*^{(n)},t_*,s_{M*})=\arg\min_{d,\,t\ge0,\,s_M}\;&
\frac12\sum_{i\in\mathcal A^{(n)}} w_i\left(d_i-d_{{\mathrm{pred}},i}^{(n)}\right)^2
+\varrho\Big(\sum_{i}t_i+|s_M|\Big)\\
\text{s.t. }&d_i=0\quad(i\in\mathcal L^{(n)}),\qquad d_i=d_{\mathrm{pred},i}^{(n)}\quad(i\in\mathcal D^{(n)}),\\
&\frac{\|g_i^{(n)}\times\widehat n_i(d)\|}{|g_i^{(n)}|}
\le\sin\phi_i^{(n)}+t_i\quad(i\in\mathcal S_{\mathrm{slope}}^{(n)}),\\
&g_i^{(n)}\cdot\widehat n_i(d)\le-\epsilon_n|g_i^{(n)}|\quad(i\in\mathcal S_{\mathrm{slope}}^{(n)}),\\
&D\mathcal M^{(n)}d+s_M=M-\mathcal M(R^{(n)},\rho^{(n)}),\\
&d\in\mathcal A_{\mathrm{lin}}^{(n)},
\qquad \|d_{\mathcal A}\|\le\Delta^{(n)},
\end{aligned}}
$$

其中 $\mathcal S_{\rm slope}^{(n)}\subseteq\mathcal A^{(n)}$ 仅包含可动集中适用安息角代理的松散表层节点，$d_{\mathcal A}$ 是 $d$ 在可动集上的限制（信赖域只约束自由变量）。目标函数**不含** $R_{\rm eq}$、$W_0$ 或任何等势面项；$\mathcal A^{(n)}=\varnothing$ 时子问题无自由变量，$d_*=d_{\rm pred}$（锁死集为零、沉积集为沉积量），不调用 Clarabel。

质量行等价于 $\widehat{\mathcal M}(d)=M-s_M$。可行性构造：取 $d_{\mathcal L}=0$、$d_{\mathcal D}=d_{\rm pred,\mathcal D}$、$d_{\mathcal A}=0$，$s_M=M-\widehat{\mathcal M}(d)$，$t_i=[\,\|g_i\times\widehat n_i(d)\|/|g_i|-\sin\phi_i]_+$；若防倒悬行在该点满足，则它是可行点（防倒悬行无松弛：若违反，须在外层通过几何预处理沿 $-g$ 方向修复该点，或在 $\mathcal S_{\rm slope}$ 的定义阶段剔除该点）。这只是可行性构造，不是 Slater 严格内点证书；若需要严格内点，必须另给 SOC 正裕度并同时严格满足全部未松弛线性约束，如第 2.13 节所述。目标在 $d_{\mathcal A}$ 上严格凸（$w_i>0$）加 $\ell_1$ 项，解存在且 $d_{\mathcal A*}$ 唯一；SOC 约束、仿射约束与信赖球使其为凸 SOCP。是否真正满足由 $t_*,s_{M*}$ 是否为零与外层非线性几何验收判定。

$\mathcal A_{\mathrm{lin}}^{(n)}$ 只包含节点顺序、正体积和信赖域的仿射/锥内近似。真实正体积和拓扑条件不是 Clarabel 的凸约束，须在外层验收；不满足时缩小信赖域并重求子问题（**信赖域策略（全文统一）：** 内层 Clarabel SOCP 因带弹性松弛变量 $t_i,s_M$ 而保证凸可行，内层绝对不缩小信赖域；仅当外层非线性几何验收判定出现网格倒悬或自交（含正体积与拓扑条件失败）时，外层才缩小信赖域半径 $\Delta^{(n)}$ 并重新生成子问题，见 2.13 节）。保留法向范数约束时是带二次目标的 SOCP；约束全部仿射化后才可称为纯 QP。

**潮汐背应力与屈服在子问题中的地位。** 子问题中的 $\sin\phi_i^{(n)}$ 来自 10.3 的基于 $p_{\rm ref}$ 的名义几何投影代理，只约束可动松散表层的几何；$t_*=0$ 只表明代理约束成立，$t_*\ne0$ 也不能据此断言张量屈服。真实张量屈服不是几何验收的约束：它是驱动流动的物理状态，在下一地质步的新几何、新 $\mathbf T_{\rm tide}$ 与新 Stokes 解上重新求 $F_\delta$ 并重新分类（2.8 节）。外层几何验收只检查正体积、无自交、星形性 $n\cdot e_r\ge c_0$ 与质量残差；无可行候选时缩小信赖域，不以流体等势面作为回退目标。

### 10.4.4 准静态演化主循环

```text
0. 初始化：读入星形多面体 R^(0)、Cauchy 密度参数（含核中心 c_k）、材料常量
   (nu_0, Z_0, c, mu_s, mu_2, I_0, d_p, rho_p, C_c, F_H, C_T, gd_eps, delta_Y, p_min)、
   模式 C 参数 (k_e, sigma_0, 核 K, h_s)、spin_schedule、全部容差；
   缺任何一项 -> ModelNotClosed，拒绝运行。rho_s^0 = rho_bulk h_s，eps_p_bar = 0，t_0 = 0；
   执行一次步骤 8b 的参考系对齐（强制，不看触发条件）。
对 n = 0, 1, 2, ... 直到 t_n >= T_end：
1. 正演：omega_n = omega(t_n)；在 V^(n)、rho_K 上用统一 RTFP/Feynman 计算
   表面面心 Hessian（10.3）与节点 g_i、H_W,i，以及体内求积点 X_vol（(8.4') 装配点）
   与分类网格 X_Y（2.8 B3）上的 grad U、Hess U（体内点经典收敛）；
   g_eff = grad U + omega_n^2 x_perp，H_W = Hess U + omega_n^2 diag(1,1,0)，
   T_tide = C_T rho_p nu_0 d_p^2 dev H_W。
2. 准静态 Stokes-压力 Picard（10.4.2）得 (u^(n), p^(n), lambda^(n), eta^(n))；
   2.15 (a)(b) 任一失败 -> 按 10.4.2 的连续化重试，仍失败则终止并报告。
3. 屈服分类（2.8 B2-B3）：在 X_Y 上合成 v、D、p~，计算 tau_flow、tau_y、F_delta；
   在图 G_Y 上求未屈服连通分量，取质量最大者为刚性核 U_core；得 chi_Y,i；
   最小二乘拟合核刚体运动 (a_c, theta_c)，v_sl = v - a_c - theta_c x x；
   更新诊断 eps_p_bar、Phi_p；报告可动面积分数、受拉比例、Ek^{-1}、eps_Euler。
4. 摩擦角链（10.3）：由 u^(n) 按命题 10.2 合成表面速度，10.2.3 计算 gd_i，
   得 mu_d,i、mu_proj,i、phi_i（只对 S_slope 中的可动节点需要）。
5. 模式 C（2.8 节）：chi_A = chi_Y * 1[rho_s > 0]，J_ero、J_dep（核 K 归一化，严格守恒）。
6. 步长（10.4.1）：Delta t_n。
   若 spin_schedule = stages 且 A^(n) 为空且 J_ero = 0（于是 J_dep = 0）：
     阶段平衡（2.15 (c)）-> 输出阶段终态，t_{n+1} = 当前阶段终点 s_j，R^(n+1) = R^(n)，转步骤 1；
   若 A^(n) 为空且非 stages：Delta t_n = min(Delta t_omega, Delta t_max, t_next - t_n)，
     d_pred = 0，直接转步骤 8（几何不变，只推进 omega）。
7. d_pred（2.8 B4）与节点分类 L / D / A；解 10.4.3 的 Clarabel SOCP 得 d_*^(n)。
8. 几何更新与参考系重对齐：
   8a. R^(n+1) = R^(n) + lambda d_*^(n)（锁死集 d_* = 0），重建多面体；外层几何验收：
       正体积、无自交、n.e_r >= c_0、|M(V^(n+1)) - M| <= eps_M；失败 -> 缩小 Delta^(n)
       重解步骤 7，不推进时间。rho_s：沿 c = chi_A v_sl,S / R 以步长 lambda Delta t_n
       作单场半拉格朗日输运（9.2 节，含 Jacobian 修正），再加 lambda Delta t_n (J_dep - J_ero)。
   8b. 标准参考系重对齐。触发条件：|x_cm| > eps_frame R_ref / 2 或
       max(|I_xz|, |I_yz|) > eps_frame I_1 / 2（未触发则跳过 8b，避免反复重采样的数值扩散）：
       (i)   质心 x_cm = (1/M_V) int_{V(R^(n+1))} rho_K x dV，M_V = int rho_K dV；
       (ii)  平移：全部网格点 x <- x - x_cm；核中心 c_k <- c_k - x_cm；
       (iii) 惯量张量 I = int rho_K (|x|^2 Id - x x^T) dV，特征分解 I_1 >= I_2 >= I_3，
             单位特征向量 a_1, a_2, a_3；
       (iv)  新标架：e_z' = a_1（最大主矩 I_1 的轴，即自转轴），e_x' = a_3（最小主矩 I_3
             的轴），e_y' = e_z' x e_x'；符号取 a_1.e_z >= 0、a_3.e_x >= 0（使 Q 最接近恒等）；
             近简并：若 I_1 - I_2 <= eps_deg I_1，e_z' 取旧 e_z 在 span{a_1, a_2} 上的投影
             归一化；若 I_2 - I_3 <= eps_deg I_1，e_x' 取旧 e_x 在 span{a_2, a_3} 与 e_z'^perp
             交集上的投影归一化；三者全简并时 Q = Id；投影为零时取该子空间中下标最小的
             特征向量。旋转矩阵 Q 的行为 (e_x', e_y', e_z')，det Q = +1；
             x <- Q x，c_k <- Q c_k（核中心必须与网格同步，否则物理密度场改变）；
       (v)   以新原点沿固定节点方向 w_i 对变换后多面体作射线求交，重采样 R'(w_i)
             （RayBackend）；须 n.e_r >= c_0，否则拒绝本步，缩小 Delta^(n) 重做；
       (vi)  标量重心插值：表面 rho_s、分类网格上的诊断 eps_p_bar；速度不跨步携带，
             下一步 Picard 取 u^0 = 0；omega 不因重对齐改变（外生给定），新标架中
             Omega = omega e_z'；
       (vii) 复核 |x_cm'| <= eps_frame R_ref 与 max(|I_xz'|, |I_yz'|) <= eps_frame I_1；
             通过后附录 A.3 的刚体相容条件在构造上成立，重采样误差计入 E_frame。
9. t_{n+1} = t_n + lambda Delta t_n；记录 R^(n+1)、rho_s、诊断与 9.8 节误差预算；回到步骤 1。
```

**阶段平衡与微蠕变。** 第 2.15 节 (c) 的阶段平衡是严格不动点：可动集为空时 $d_*=0$ 恰好成立，与 $\Delta t_n$、$\lambda$ 无关。正则化 $\dot\gamma_\varepsilon>0$ 使刚性核内仍有量级 $|\mathbf v|\sim\dot\gamma_Y R$ 的数值蠕变，它只作为诊断报告（$\max_{\mathcal U_{\rm core}}|\mathbf v_{\rm sl}|$），不进入几何、步长、通量或任何状态量，因此不存在“流速底限”式的收敛判据，也不存在蠕变引起的几何或历史漂移。$\dot\gamma_\varepsilon\to0$ 时核内蠕变趋于零，可动集收敛到未正则化模型的屈服区与滑块（PO-12）。

**动态自由表面模式**的时间推进不使用本循环：每个物理步以 $\partial_tR+\mathbf v_S\cdot\nabla_SR=v_r$ 积分 $R$，速度由第 9 章的模态动量输运与质量恢复给出，几何子问题作为约束校正；其误差预算为第 9.8 节的 $E_{\rm total}^{\rm dyn}$。

准静态主线的闭环为

$$
\boxed{
\omega(t_n)
\xrightarrow{\text{RTFP 正演}}
(\mathbf g_{\rm eff},\mathbf T_{\rm tide})
\xrightarrow{\text{Stokes–Poisson Picard}}
(\mathbf u,p;\ \boldsymbol\sigma)
\xrightarrow{F_\delta,\ \mathcal U_{\rm core}}
(\chi_Y,\mathbf v_{\rm sl},J)
\xrightarrow{\Delta t_n}
d_{\mathrm{pred}}
\xrightarrow{\mathrm{Clarabel}\ \text{弹性 SOCP}}
d_*^{(n)}
\xrightarrow{\text{更新}+\text{重对齐}}
R^{(n+1)} .
}
$$

该闭环中只有一个应力场 $\boldsymbol\sigma$，速度每步隐式重解，自转外生给定，几何只在可动集上改变。模式 A 的 $W_0/R_{\mathrm{eq}}$ 接口不进入此闭环；须保留其接口注释，供未来显式流体模拟调用，不能在代码实现时误删。


# 11. 结论

本文给出了以模式 B 准静态黏塑性 Stokes 平衡为核心、以模式 C 局部屈服侵蚀/沉积为辅助的 Cauchy 密度引力、变密度极向—环向双势流、曲面守恒输运和增量 SOCP 闭环。外部场采用 Feynman 参数化与 Newton 表示相互校验；内部和边界 Hessian 明确采用有限部分与接触项规范。力学上只有一个应力场 $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+2\eta\mathbf D$：每个地质步在外生给定的自转 $\omega(t_n)$ 下，以精确无散的移位 Jacobi–球谐双势基隐式求解准静态 Stokes 系统（8.4′），压力由独立标量 Poisson 问题给出，二者经 Picard 迭代闭合；在同一应力上用带容差的屈服函数 $F_\delta$ 与未屈服区连通性划分刚性核与可动集，核上地形严格锁死，只有屈服区与滑块按相对核的滑移速度产生几何预测。Clarabel 以当前形状为基准只校正该预测增量，从不以等势面为目标；地质步长只受真实滑坡速度的几何 CFL 与侵蚀通量限制，不存在黏性扩散刚性。Track A（全局求解）为基准，Track B（球谐块对角代理）只作预条件子或带后验残差的近似。模式 A 仅保留显式标记的未来流体接口。

模型的严格性依赖于正压力、可积性、有限部分规范、径向模态截断和本构正则化等条件；$K=4$ 的数值结果应通过更高阶径向基和三维参考解进行收敛验证。该框架因此提供的是可以逐项建立误差预算的可实现降阶模型，而不是无条件的三维闭式等价变换。

# 附录 A. 规范性约定与证明义务登记（单一真理源）

本附录是第 5–10 章的规范性约定，取代曾单独存在的契约文件；其中“必须/不得”为强制条款。第 1–4 章为本附录的输入，仅被引用。凡标“定理”的结论须注明所依赖的证明义务（A.5）；未证者只能称“假设”或“数值观测”。

## A.1 符号消歧表

准静态主线的基本未知量为 $(\mathbf v,p,R)$（另有模式 C 的表层面密度 $\rho_s$）；密度 $\rho_K$ 在体坐标系中空间固定、不作连续性推进（假设 A-ρ，PO-14），自转 $\omega(t)$ 外生给定（10.4.1 节）。

| 符号 | 含义 | 量纲 / 约定 |
|---|---|---|
| $\mathbf x,\ r,\ \mathbf e_r$ | 旋转系位置（原点取质心）、径向坐标与单位向量 | L；$\boldsymbol\Omega=\omega\mathbf e_z$ |
| $\hat{\boldsymbol\omega}=(\theta,\varphi)\in S^2$ | 方向（**一律写 $\hat{\boldsymbol\omega}$**，避免与角速度 $\omega$ 混淆） | 1 |
| $R(\hat{\boldsymbol\omega},t)$，$\zeta=r/R$ | 星形边界半径、归一化半径 | L，1；$0<R_{\min}\le R\le R_{\max}$ |
| $\mathcal V(R)$，$\Gamma=\partial\mathcal V$，$\mathbf n$ | 物质域、自由表面、外法向 | $\mathbf n\cdot\mathbf e_r\ge c_0>0$ |
| $\ell,m,k$ | 球谐阶/序（速度仅 $\ell\ge1$）；径向模式 $k=0..3$（$K=4$） | 1 |
| $\Lambda_\ell=\ell(\ell+1)$，$L^2=-\Delta_{S^2}$ | 球面 Laplace 本征值与角动量平方 | 1 |
| $\mathbf v,\ p$ | 速度、压力 $p=-\operatorname{tr}\boldsymbol\sigma/3$ | L T⁻¹；M L⁻¹ T⁻² |
| $\boldsymbol\sigma=-p\mathbf I+\mathbf T_{\rm tide}+\boldsymbol\tau_{\rm flow}$ | Cauchy 应力（**粗体**，区别于核宽度 $\sigma_k$） | M L⁻¹ T⁻² |
| $\Phi,\Psi$；$q_{\ell km},s_{\ell km}$ | 极向/环向势及其模态系数 | L² T⁻¹，L T⁻¹；系数 T⁻¹ |
| $\mathbf u=(\mathbf q,\mathbf s)$ | 速度自由度向量，$N_u=8N_{\ell m}$，$N_{\ell m}=\#\{(\ell,m):1\le\ell\le L_{\max}\}$ | T⁻¹ |
| $\mathbf M_\Gamma$，$\mathbf m=\mathbf M_\Gamma\mathbf u$ | 全局动能质量算子、守恒模态变量（仅动态模式） | $\sim\rho R^5$ |
| $\mathbf K(\eta)$，$\mathbf F$ | 准静态黏性刚度与载荷（8.4′） | $\sim\eta R^3$；$\sim\rho gR^4$ |
| $\eta,\ \tilde p$ | 正则化黏度、唯一正则化压力 $\max([p]_+,p_{\min})$ | M L⁻¹ T⁻¹；M L⁻¹ T⁻² |
| $\tau_y,\ F,\ F_\delta,\ \delta_Y$ | 静态强度 $c+\mu_y\tilde p$、屈服函数、带容差屈服函数、容差（2.8 节 B2） | M L⁻¹ T⁻²；$\delta_Y$ 无量纲 |
| $\mathcal U_{\rm core}$，$\chi_{Y,i}$，$\mathcal L,\mathcal D,\mathcal A$ | 刚性核、可动指示、锁死/纯沉积/可动节点集（2.8 节 B3–B4） | — |
| $\mathbf v_{\rm sl}$ | 相对刚性核的滑移速度 $\mathbf v-\mathbf a_c-\boldsymbol\theta_c\times\mathbf x$ | L T⁻¹ |
| $\omega(t)$，$\omega_n$ | 外生给定自转表、第 $n$ 步冻结值（10.4.1 节） | T⁻¹ |

时间记号：$t_n$ 为地质时间，$\Delta t_n$ 为地质步长（第 10.4.1 节），$t_{n+1}=t_n+\lambda\Delta t_n$；Picard 迭代指标 $k$ 不代表时间；本文不使用伪时间、$\Delta\tau$ 或物理松弛步 $\Delta t_{\rm phys}$。欠松弛因子 $\lambda$（标量）与刚体乘子 $\boldsymbol\lambda\in\mathbb R^6$（粗体）不同。$\mathbf x_{\rm cm}$、$\mathbf Q$ 为参考系重对齐的质心与旋转（步骤 8b）。本文不定义弹性模量、硬化变量或返回映射应力。

强制消歧：Feynman 参数仍为 $u$（仅第 1–2 章），自由度 $\mathbf u$ 仅用粗体，位移增量用 $\mathbf d$（径向 $d_i$）；应力 $\boldsymbol\sigma$ / 核宽度 $\sigma_k$ / 增长率 $\lambda_{\rm gr}$；总质量 $\mathsf M_{\rm tot}$ / 质量算子 $\mathbf M_\Gamma$；区域 $\mathcal V$ / 角速度向量 $\boldsymbol\Omega$；核尺度 $a_k=1/\sigma_k$，节点序号 $a$ 仅限 Track B，颗粒直径 $d_p$。指标约定：$\sum_{\ell,m,k}$ 取 $1\le\ell\le L_{\max}$、$|m|\le\ell$、$0\le k\le3$；实现默认采用实正交归一球谐基。第 6.1、7.3–7.4、8.9 节中的 $m$ 选择规则按复球谐记号书写；在实基中应将对应的 $m$ 与 $-m$ 合并为等价的二维实块，不能把 $m'=m+M$ 当作逐列的实基索引恒等式。笛卡尔指标用 $i,j,l$。

## A.2 函数空间与迹空间规范

**几何假设（强制）。** $R\in W^{1,\infty}(S^2)$，$R\ge R_{\min}>0$，$\mathbf n\cdot\mathbf e_r\ge c_0>0$ 几乎处处，故 $\mathcal V$ 为 Lipschitz 星形域，迹算子、Korn 与 Poincaré 型不等式可用。

$$
\mathbb V=H^1(\mathcal V;\mathbb R^3),\qquad \mathbb Q=L^2(\mathcal V),\qquad \mathbb V_\sigma=\{\mathbf w\in\mathbb V:\nabla\cdot\mathbf w=0\}.
$$

迹 $\gamma_0:\mathbb V\to H^{1/2}(\Gamma;\mathbb R^3)$ 连续满射；牵引 $\mathbf t=\boldsymbol\sigma\mathbf n\in H^{-1/2}(\Gamma;\mathbb R^3)$，配对为 $H^{-1/2}$–$H^{1/2}$ 对偶。自由表面为自然边界（默认 $\mathbf t_{\rm ext}=0$ 进入右端表面项）。星形域上 $\mathbf w\in\mathbb V_\sigma$ 的 $\ell\ge1$ 部分唯一写为 $\mathbf v=\nabla\times\nabla\times(\Phi\mathbf r)+\nabla\times(\Psi\mathbf r)$，$\mathbf r=r\mathbf e_r$（第 5.2 节）。连续层 $p\in\mathbb Q$ 在全表面自然牵引条件下无自由常数；离散层见第 8.4 节。

**非线性本构的函数论前提。** $\mu(I)$ 关于 $\mathbf D$ 非单调，本文**不声称**连续问题的存在唯一性；Track A 的“解”定义为准静态离散系统（8.4′）与压力 Poisson（8.7 节）在给定 $(L_{\max},K=4,\text{求积},\dot\gamma_\varepsilon,p_{\min})$ 下的 Picard 不动点（10.4.2 节）；固定黏度时的线性可解性由引理 8.12 保证，非线性不动点的存在性与 Picard 收敛属 PO-1、PO-11（动态全隐式基准另涉 PO-2）。本构分母要求 $p_*\ge p_{c,\min}>0$ 或显式压力正则化。

**备选偶次基。** 若要求 $R$ 为常数时的中心 $C^\infty$ 光滑度，改用 $\psi^{\rm ev}_{\ell k}=\zeta^\ell P_k^{(0,\ell+1/2)}(2\zeta^2-1)$（对权 $\zeta^{2\ell+2}$ 正交）；$R$ 非常数时势函数各项含方向依赖因子 $R(\hat{\boldsymbol\omega})^{2-n}$，中心光滑度由该因子决定而与径向基奇偶性无关（命题 8.4 的 $\ell=1$ 例外不变）。默认基见定义 8.2，其径向参数固定为 $(0,2\ell+2)$、自变量 $2\zeta-1$，不得使用 $P_k^{(2\ell+2,0)}(2\zeta-1)$。$K=4$ 指每个 $(\ell,m)$、每个势各 4 个径向模式，即每个 $(\ell,m)$ 共 8 个速度自由度，不是“全局 4 个标量场”。

## A.3 刚体零空间与约束矩阵

**$\ell=0$ 奇异性（强制）。** 对仅依赖 $r$ 的 $f$，$\nabla\times(f\mathbf r)=\mathbf 0$，故 $\ell=0$ 的 $\Phi,\Psi$ 对 $\mathbf v$ 无贡献：速度自由度只含 $\ell\ge1$；$\mathbf K$、$\mathbf K_\gamma$ 与 $\mathbf M_\Gamma$ 仅在 $\ell\ge1$ 子空间上装配与求逆，任何实现不得对含 $\ell=0$ 的矩阵求逆；$R$ 的 $\ell=0$ 分量由总质量约束 $\int_{\mathcal V}\rho\,dV=\mathsf M_{\rm tot}$ 决定，不由运动学方程推进。

**刚体空间。** $\mathbf D(\mathbf w)=\mathbf0$ 当且仅当 $\mathbf w$ 为刚体运动。平移 $\mathbf w=\mathbf e$ 对应 $\Phi=\tfrac12\mathbf e\cdot\mathbf x$，转动 $\mathbf w=\boldsymbol\theta\times\mathbf x$ 对应 $\Psi=\boldsymbol\theta\cdot\mathbf x$，径向剖面 $\zeta^1$。$R$ 为常数时它们恰为 $\ell=1,k=0$ 的 6 个系数；$R$ 非常数时不一定落在 $\mathbb V_h^\sigma$ 内，须逐步核验。

**约束矩阵（强制，推荐形式）。** 质量加权零动量/零角动量约束 $\int_{\mathcal V}\rho\mathbf v\,dV=\mathbf0$、$\int_{\mathcal V}\rho\,\mathbf x\times\mathbf v\,dV=\mathbf0$ 的代数化为 $\mathbf B_{\rm rb}\mathbf u=\mathbf0$，$\mathbf B_{\rm rb}\in\mathbb R^{6\times N_u}$，行为 $\int_{\mathcal V}\rho\,\mathbf B_\alpha dV$ 与 $\int_{\mathcal V}\rho\,\mathbf x\times\mathbf B_\alpha dV$，要求 $\operatorname{rank}(\mathbf B_{\rm rb})=6$；乘子 $\boldsymbol\lambda$ 的语义与监控量 $E_{\rm rb}$ 见第 9.4 节。备选形式是直接固定 $q_{1m0}=s_{1m0}=0$，仅当两者等价时允许并须报告差异。

**可解性相容条件（强制检查）。** 准静态问题有解的必要条件是外力对刚体模式不做功：$\int\rho\mathbf g_{\rm eff}dV+\oint_\Gamma\mathbf t_{\rm ext}dS=\mathbf0$，$\int\rho\,\mathbf x\times\mathbf g_{\rm eff}dV+\oint_\Gamma\mathbf x\times\mathbf t_{\rm ext}dS=\mathbf0$；自引力合力恒为零，离心力要求质心在转轴上且转轴为惯量主轴（$I_{xz}=I_{yz}=0$）。不满足时必须报告“不相容”，不得靠乘子吸收。这两个条件在准静态主线中由第 10.4.4 节步骤 8b 的**标准参考系重对齐**在构造上保证：几何更新后若失配超过触发阈值，将原点平移回质心、并把最大主矩 $I_1$ 的惯量主轴旋转到自转 $z$ 轴（于是质心在转轴上且 $I_{xz}=I_{yz}=0$，自转绕最大主矩轴也是稳定转动态，假设 A-S），同步平移并旋转全部 Cauchy 核中心 $\mathbf c_k$；重对齐后仍须复核残差 $|\mathbf x_{\rm cm}|\le\varepsilon_{\rm frame}R_{\rm ref}$、$|I_{xz}|,|I_{yz}|\le\varepsilon_{\rm frame}I_1$，否则才报告“不相容”。外生 $\dot\omega$ 的 Euler 力与外力矩成对略去（2.8 节 B1），不破坏力矩相容。对每个 $\ell\ge1$ 须报告整体刚度矩阵的最小奇异值。

## A.4 Track A / Track B 定义与定理 A.1

两轨针对同一个对称正定算子 $\mathbf A$ 定义：准静态主线 $\mathbf A=\mathbf K_\gamma$（8.4′ 的增广黏性刚度，右端 $\mathbf b=\mathbf F$），动态模式 $\mathbf A=\mathbf M_\Gamma$（9.4 节，右端 $\mathbf b=\mathbf m$）。**Track A（基准）：** 用完整 $\mathbf A$ 精确求解（到迭代容差）。**Track B（快速对角代理，截断模型）：** 取 $\mathbf A_B:=\operatorname{blockdiag}_{(\ell,m)}\mathbf A_{\ell m,\ell m}$（每个 $(\ell,m)$ 一个 $8\times8$ 主子块；对 $\mathbf M_\Gamma$ 即 9.5 节的 $\mathbf M_{B1}$），$\mathbf u_B=\mathbf A_B^{-1}\mathbf b$；残差算子 $\mathbf E:=\mathbf A-\mathbf A_B$，球对称几何下 $\mathbf E=\mathbf0$，与 A 一致。Track B 仅指 B1 块对角代理（PO-8）；$\mathbf A_B^{-1}$ 同时是 Track A 中 PCG 的预条件子。

**定理 A.1（精确误差恒等式）。** 若 $\mathbf A_B,\mathbf A$ 均可逆，则对同一 $\mathbf b$，$\mathbf u_B-\mathbf u_0=\mathbf A_B^{-1}\mathbf E\,\mathbf u_0$，$\mathbf u_0=\mathbf A^{-1}\mathbf b$。*证明。* $\mathbf A_B^{-1}-\mathbf A^{-1}=\mathbf A_B^{-1}(\mathbf A-\mathbf A_B)\mathbf A^{-1}$ 作用于 $\mathbf b$。∎ 推论：$\mathbf u_B=\mathbf u_0$ 当且仅当 $\mathbf E\mathbf u_0=\mathbf0$，故 Track B 不是精确等价。动态模式含刚体约束的完整形式为定理 9.4；准静态主线中 Track B 的后验量就是以完整 $\mathbf K$ 计算的 $\varepsilon_{\rm mom}^{\rm weak}$（10.4.2 节）。

**强制报告：** Track B 每次运行须给出 $\varepsilon_B:=\|\mathbf r_B\|/\max\{\|\mathbf b\|,\varepsilon_b\}$（$\mathbf r_B=\mathbf b-\mathbf A\mathbf u_B$；当 $\|\mathbf b\|=0$ 时，若 $\mathbf r_B=0$ 取 $\varepsilon_B=0$，否则该步判为未定义并失败）及其按 $\ell$ 的分布，并附 $\mathbf A_B^{-1}-\mathbf A^{-1}$ 的估计；不得称 Track B “精确”“等价”或“严格”。**红线：** Track B 仅在实测 $\varepsilon_B$ 低于配置阈值时可用，且其收敛结论须由 Track A 复核（9.5 节，PO-8）；生产基准为 Track A。两轨共用第 2.13/10.4.3 节的弹性 SOCP 几何校正，仅在线性求解方式上不同。

## A.5 证明义务（PO）登记表

| 编号 | 内容 | 状态 |
|---|---|---|
| PO-1 | 离散非线性系统（8.4′）+8.7 节在 $\dot\gamma_\varepsilon$ 连续化下的局部可解性（含 $p_*\ge p_{c,\min}$）；固定 $\eta$ 的线性子问题已由引理 8.12 闭合 | 线性子问题已闭合；非线性未证 |
| PO-2 | inf–sup 条件、$\mathbb V_h^g$/$\mathbb Q_h$ 的具体匹配（只涉及动态全隐式基准；准静态主线速度空间精确无散，不需要） | 未证 |
| PO-3 | 基的中心协调性 $\mathbf v\in H^1$（含 $\ell=1$） | 已闭合：命题 8.4 |
| PO-4 | 假设 A-M（沿表面输运 $\mathbf m$）及几何重映射的模型误差界与源项完整展开（仅动态模式；准静态主线不输运 $\mathbf m$） | 未证 |
| PO-5 | 含 $\Psi$ 的自由表面剪切率与两个切向牵引约束的显式模态公式 | 条件性闭合：命题 10.2、10.3（允许一致性残差 $\delta_{na}$） |
| PO-6 | $\mathbf M_\Gamma$ 在 Jacobi 基下的块元素、带宽与条件数 | 条件性闭合：命题 10.1（仅固定参考球块） |
| PO-7 | 速度方程各项在基下的装配公式 | 条件性闭合：命题 6.2、6.4、7.3、7.4（仅球对称径向密度的 Coriolis 块） |
| PO-8 | 已废弃 B2（节点局部冻结 $\Lambda_\star$ 模型：数值观测其迭代矩阵 $\mathbf I-\widetilde{\mathbf M}_{B2}^{-1}\mathbf M_\Gamma$ 在高阶模态谱半径远大于 1，会发散或伪收敛）；Track B 仅指 B1 块对角代理。剩余未证部分：$\|\mathbf M_B^{-1/2}\mathbf E\mathbf M_B^{-1/2}\|$ 对非球形程度的先验依赖与 $\varepsilon_B$ 的先验上界 | B2 已废除；B1 先验界未证 |
| PO-9 | 地质步几何更新（显式于几何的 RTFP–Stokes–SOCP 序列）对 $\Delta t_n\to0$ 的相容性与稳定性；SOCP 投影偏离 $E_{\rm proj}$ 的上界 | 未证（依赖 CFL、欠松弛与信赖域） |
| PO-10 | $\mu(I)$ 黏塑性在高惯性数处的短波不适定性（命题 5.8）及其在 $\dot\gamma_\varepsilon$、$p_{\min}$ 正则化下的应变局部化与网格依赖性 | 未证 |
| PO-11 | 压力 Poisson（8.7 节）与准静态 Stokes（8.4′）Picard 迭代的收敛性；$\nabla\cdot\boldsymbol\tau_{\rm flow}$ 对冻结 $(\mathbf u^k,\tilde p^k)$ 的稳定性 | 未证（未收敛时按 2.15 (a) 判失败） |
| PO-12 | 正则化屈服面（$F_\delta=0$，固定 $\delta_Y$）与刚性核/可动集在 $\dot\gamma_\varepsilon\to0$ 时向未正则化 Bingham/$\mu(I)$ 屈服区与滑块的收敛；10.4.1 节步长准则的充分性；恒定 $\omega_j$ 下可动集在有限步内变空（阶段平衡可达）的条件 | 未证（不可达时以“阶段未平衡”如实报告，2.15 (c)） |
| PO-13 | 参考系重对齐的重采样误差界 $E_{\rm frame}$；星形性在新原点下的保持。自转演化不属于证明义务：$\omega(t)$ 外生给定，主轴自转为显式模型假设 A-S（10.4.1 节） | 部分闭合（误差界未证） |
| PO-14 | 假设 A-ρ：$\rho_K$ 在体坐标系中空间固定、不作连续性输运；滑坡物质跨越密度梯度时的质量分布误差界 | 模型假设；以 $s_{M*}$、$\rho_{\rm bulk}/\rho_K\rvert_\Gamma$ 范围与质量残差逐步报告 |

---

# 参考文献

[1] Forterre, Y., & Pouliquen, O. (2008). An experimental study of quasi-static granular flows. *Journal of Fluid Mechanics*, 616, 163–188. https://doi.org/10.1017/S0022112008000491

[2] Gidaspow, D. (1994). *Multiphase Flow and Fluidization: Continuum and Kinetic Theory Descriptions*. Academic Press.

[3] Greengard, L., & Rokhlin, V. (1987). A fast algorithm for particle simulations. *Journal of Computational Physics*, 73(2), 325–348. https://doi.org/10.1016/0021-9991(87)90140-9

[4] Staniforth, A., & Côté, J. (1991). Semi-Lagrangian integration schemes for atmospheric models—A review. *Monthly Weather Review*, 119(9), 2206–2223. https://doi.org/10.1175/1520-0493(1991)119<2206:SLISFA>2.0.CO;2
