# 多面体广义 Cauchy 密度引力梯度：RTFP 与 Feynman 参数化两条主线

作者：Ruilai Chen

## 阅读约定与结论等级

**预期解决的目标。** 给出一个共同的数学问题，再分别建立两条独立的表示和计算路线。凡称“等式”均说明其定义域；凡称“算法”均与数学恒等式区分。

本稿在现有 mathRTFP(3).md 的理论主线上修订：保留 Newton 分布导数、Cauchy 密度、射线有限部分、球面二阶矩、径向原函数、条件性椭圆积分与数值实现的核心内容；重新组织了不能严格闭合的全局分解。第二条 Feynman 路线是一种独立的变换，不是第一条路线的残差或 Werner 算子的组成部分。既不假设所有外层积分存在初等闭式，也不把尚未经误差证明的浮点结果称为严格数值保证。

每节开始列出目标，最后说明实际得到的结论。所有距离具有长度量纲，所有对数的自变量均为无量纲量。

**本节实际解决的总结。** 后文的两条路线共享同一个势函数；证明、条件性约化和数值估计分别陈述。

# 第一部分　共同问题和定理

## 第一节　多面体、密度与势函数

**预期解决的目标。** 固定几何、密度、符号和函数的适用范围。

设 $V\subset\mathbb R^3$ 是有限个闭合、无自交、定向一致的三角面片围成的有界多面体实体，可以非凸，可以有有限个体积连通分量。相邻面片沿完整公共边配对；各面片相对内部两两不交。假设 $V=\overline{\operatorname{int}V}$，且边界满足分片 Lipschitz 条件。面片或棱边上的零测交集在体积分中只计一次。

令 $K\in\mathbb N$，$\alpha_k>0$，$\sigma_k>0$，$w_k\in\mathbb R$，$\mathbf c_k\in\mathbb R^3$。定义

$$
\phi_k(\mathbf y)=\left[1+\sigma_k^2\|\mathbf y-\mathbf c_k\|^2\right]^{-\alpha_k},
\qquad
\rho_K(\mathbf y)=\sum_{k=1}^{K}w_k\phi_k(\mathbf y).
$$

物理密度要求 $\rho_K\geq0$ 几乎处处；单个权重可以为负，只要整体满足该约束。有限和是一个指定的平滑模型，不是任意间断密度的恒等表示。全空间延拓 $\rho_K$ 光滑，但实际源密度为 $\rho_K\mathbf1_V$，在边界可发生跳跃。

取正引力势的约定：

$$
U_K(\mathbf x)=G\int_V\frac{\rho_K(\mathbf y)}{\|\mathbf x-\mathbf y\|}\,d^3\mathbf y,
\qquad
\mathbf g_K(\mathbf x)=\nabla_{\mathbf x}U_K(\mathbf x).
$$

当 $\mathbf x\notin V$ 时，经典 Hessian 定义为

$$
H_K(\mathbf x)=\nabla_{\mathbf x}\nabla_{\mathbf x}U_K(\mathbf x).
$$

当场点在 $V$ 的内部或边界时，必须分别给出导数解释。任何“表面 Hessian”均应说明是外侧极限、内侧极限，还是采用指定球形排异和尺度的有限部分。

**本节实际解决的总结。** 后文仅研究这个有限多面体和有限 Cauchy 核模型；实测或真实密度误差须另作验证。

## 第二节　Newton 核的二阶导数

**预期解决的目标。** 明确二阶导数在源点重合时的分布项，固定符号和迹。

令 $\mathbf r=\mathbf x-\mathbf y$，$R=\|\mathbf r\|>0$。逐次求导：

$$
\frac{\partial R}{\partial x_i}=\frac{r_i}{R},\qquad
\frac{\partial}{\partial x_i}\frac1R=-\frac{r_i}{R^3},
$$

$$
\frac{\partial^2}{\partial x_i\partial x_j}\frac1R
=-\frac{\delta_{ij}}{R^3}+\frac{3r_ir_j}{R^5}
=\frac{3r_ir_j-R^2\delta_{ij}}{R^5}.
$$

球形排异定义的分布恒等式为

$$
\boxed{
\partial_i\partial_j\frac1R
=\operatorname{p.v.}_{\mathrm{ball}}
\frac{3r_ir_j-R^2\delta_{ij}}{R^5}
-\frac{4\pi}{3}\delta_{ij}\delta_0(\mathbf r).
}
$$

核的非局部张量 $3r_ir_j/R^5-\delta_{ij}/R^3$ 在 $R>0$ 无迹，接触项的迹为 $-4\pi\delta_0$，所以 $\Delta(1/R)=-4\pi\delta_0$。对于光滑密度在内部点作卷积，这给出 $\operatorname{Tr}H_K=-4\pi G\rho_K$。在边界点，$\delta_0$ 与不连续指示函数的乘积不存在规范的逐点解释；后文明确选取球对称约定。

**本节实际解决的总结。** 分布恒等式适用；边界上的接触值是额外声明的代表，不是分布理论强制给出的唯一经典点值。

## 第三节　固定半径的局部切锥与球形有限部分

**预期解决的目标。** 为两条路线提供统一的边界规范，并避免随方向趋于零的局部区间端点。

令 $\mathbf x$ 为外部、内部、面片相对内部、棱边相对内部或顶点上的一个固定场点。对内部或边界点，因网格有限，存在 $r_*(\mathbf x)>0$，使得开球 $B_{r_*}(\mathbf x)$ 内的实体严格等于 $\mathbf x$ 处的多面体局部切锥与该球的交。取任意 $0<r_0<r_*(\mathbf x)$。源射线取 $\mathbf y=\mathbf x-R\mathbf u$，$R>0$，$\mathbf u\in\mathbb S^2$。定义

$$
\Gamma_{\mathbf x}
=\{\mathbf u\in\mathbb S^2:\mathbf x-R\mathbf u\in V
\text{ 对所有足够小的 }R>0\},
$$

在球面几何零测的例外方向之外，所有 $0<R<r_0$ 的归属同时由 $\Gamma_{\mathbf x}$ 决定。若 $\mathbf x\notin V$，取 $0<r_0<\operatorname{dist}(\mathbf x,V)$，并设 $\Gamma_{\mathbf x}=\varnothing$。定义

$$
T(\mathbf u)=3\mathbf u\mathbf u^T-I,
\qquad
C_{\mathbf x}=\int_{\Gamma_{\mathbf x}}T(\mathbf u)\,d\Omega,
\qquad
\vartheta_{\mathbf x}=\frac{|\Gamma_{\mathbf x}|}{4\pi}.
$$

固定正长度 $\ell_0>0$。定义球排异的截断核张量

$$
J_\varepsilon(\mathbf x)
=G\int_{V\setminus B_\varepsilon(\mathbf x)}
\rho_K(\mathbf y)
\frac{3(\mathbf x-\mathbf y)(\mathbf x-\mathbf y)^T
-\|\mathbf x-\mathbf y\|^2I}
{\|\mathbf x-\mathbf y\|^5}\,d^3\mathbf y.
$$

在内部和边界点规定接触代表

$$
H_{\rm contact}(\mathbf x)
=-\frac{4\pi G}{3}\vartheta_{\mathbf x}\rho_K(\mathbf x)I.
$$

定义球排异有限部分

$$
\boxed{
H_{K,\ell_0}^{\mathrm{FP}}(\mathbf x)
=\lim_{\varepsilon\downarrow0}
\left[J_\varepsilon(\mathbf x)
-G\rho_K(\mathbf x)C_{\mathbf x}
\log\frac{\ell_0}{\varepsilon}\right]
+H_{\rm contact}(\mathbf x).
}
$$

此处的极限在以下第五节由显式绝对可积的差分式证明；在边界，它是本稿选定的点值约定。当 $\rho_K(\mathbf x)C_{\mathbf x}\ne0$ 时，未扣除对数项的 $J_\varepsilon$ 无有限极限。由于 $\operatorname{Tr}C_{\mathbf x}=0$，上述定义给出

$$
\operatorname{Tr}H_{K,\ell_0}^{\mathrm{FP}}(\mathbf x)
=-4\pi G\vartheta_{\mathbf x}\rho_K(\mathbf x).
$$

换尺度得到

$$
\boxed{
H_{K,\ell_1}^{\mathrm{FP}}
-H_{K,\ell_0}^{\mathrm{FP}}
=G\rho_K(\mathbf x)C_{\mathbf x}\log\frac{\ell_0}{\ell_1}.
}
$$

所以与尺度无关的充要条件是 $\rho_K(\mathbf x)C_{\mathbf x}=0$，仅当 $\rho_K(\mathbf x)\ne0$ 才能简化为 $C_{\mathbf x}=0$。内部全空间锥与光滑面半球均满足 $C_{\mathbf x}=0$。棱和顶点须具体计算。

**本节实际解决的总结。** 球排异、参考长度与接触约定均已固定；没有把边界有限部分写成无条件的经典 Hessian。

# 第二部分　主线 A：RTFP 射线积分与严格远近场划分

## 第四节　球坐标变换与距离二次配方

**预期解决的目标。** 不跳过体积雅可比与 Cauchy 核径向配方的计算。

写 $\mathbf u=(\sin\theta\cos\varphi,\sin\theta\sin\varphi,\cos\theta)$，$\mathbf y=\mathbf x-R\mathbf u$。球面测度和体积测度为

$$
d\Omega=\sin\theta\,d\theta\,d\varphi,
\qquad
d^3\mathbf y=R^2\,dR\,d\Omega.
$$

对非零 $R$，

$$
\frac{3(\mathbf x-\mathbf y)(\mathbf x-\mathbf y)^T-R^2I}{R^5}
\,d^3\mathbf y
=\frac{R^2(3\mathbf u\mathbf u^T-I)}{R^5}R^2\,dR\,d\Omega
=T(\mathbf u)\frac{dR}{R}\,d\Omega.
$$

对第 $k$ 个核定义 $\mathbf p_k=\mathbf x-\mathbf c_k$，$b_k(\mathbf u)=\mathbf p_k\cdot\mathbf u$，$a_k=1/\sigma_k$，以及

$$
d_k^2(\mathbf u)
=a_k^2+\|\mathbf p_k\|^2-b_k^2(\mathbf u)
\ge a_k^2>0.
$$

由 $\|\mathbf u\|=1$，依次得到

$$
\begin{aligned}
\|\mathbf p_k-R\mathbf u\|^2
&=\|\mathbf p_k\|^2-2Rb_k+R^2\\
&=(R-b_k)^2+\|\mathbf p_k\|^2-b_k^2,
\end{aligned}
$$

$$
1+\sigma_k^2\|\mathbf p_k-R\mathbf u\|^2
=\sigma_k^2[(R-b_k)^2+d_k^2].
$$

本式是标量二次配方；四元数写法不会提供额外的收敛或闭式结论。

**本节实际解决的总结。** RTFP 的核成为 $T(\mathbf u)\phi_k(\mathbf x-R\mathbf u)/R$，且 Cauchy 径向二次项在实轴严格为正。

## 第五节　局部项、远场项与无遗漏定理

**预期解决的目标。** 对所有射线交段提供严格互补的分区，包括外部场点仅有第一段的情形。

对任意方向定义 $E_{\mathbf x}(\mathbf u)=\{R>0:\mathbf x-R\mathbf u\in V\}$。令

$$
E^{\rm near}_{\mathbf x}(\mathbf u)
=E_{\mathbf x}(\mathbf u)\cap(0,r_0),
\qquad
E^{\rm far}_{\mathbf x}(\mathbf u)
=E_{\mathbf x}(\mathbf u)\cap[r_0,\infty).
$$

球面零测例外方向之外，

$$
E^{\rm near}_{\mathbf x}(\mathbf u)
=\begin{cases}(0,r_0),&\mathbf u\in\Gamma_{\mathbf x},\\
\varnothing,&\mathbf u\notin\Gamma_{\mathbf x}.
\end{cases}
$$

对球排异截断，近场直接分为

$$
\begin{aligned}
G\int_{\Gamma_{\mathbf x}}T(\mathbf u)
\int_\varepsilon^{r_0}
\frac{\rho_K(\mathbf x-R\mathbf u)}R\,dR\,d\Omega
={}&G\rho_K(\mathbf x)C_{\mathbf x}
\log\frac{r_0}{\varepsilon}\\
&+G\int_{\Gamma_{\mathbf x}}T(\mathbf u)
\int_\varepsilon^{r_0}
\frac{\rho_K(\mathbf x-R\mathbf u)-\rho_K(\mathbf x)}R
\,dR\,d\Omega.
\end{aligned}
$$

因为 $\rho_K$ 光滑，在包含 $\overline V$ 的紧集中可取 $M<\infty$ 使 $\|\nabla\rho_K\|\le M$。由一元积分形式的中值定理，

$$
|\rho_K(\mathbf x-R\mathbf u)-\rho_K(\mathbf x)|
\le MR.
$$

所以差分的绝对值除以 $R$ 不超过 $M$，可用支配收敛定理令 $\varepsilon\downarrow0$。定义三个确定的张量

$$
H^{\rm cone}_{\ell_0}
=G\rho_K(\mathbf x)C_{\mathbf x}\log\frac{r_0}{\ell_0},
$$

$$
H^{\rm reg}
=G\int_{\Gamma_{\mathbf x}}T(\mathbf u)
\int_0^{r_0}
\frac{\rho_K(\mathbf x-R\mathbf u)-\rho_K(\mathbf x)}R
\,dR\,d\Omega,
$$

$$
H^{\rm far}
=G\int_{\mathbb S^2}T(\mathbf u)
\int_{E^{\rm far}_{\mathbf x}(\mathbf u)}
\frac{\rho_K(\mathbf x-R\mathbf u)}R\,dR\,d\Omega.
$$

代入第三节的定义，得到严格的全局等式

$$
\boxed{
H_{K,\ell_0}^{\mathrm{FP}}
=H^{\rm cone}_{\ell_0}+H^{\rm reg}
+H^{\rm far}+H_{\rm contact}.
}
$$

每个方向的所有射线段先与 $[r_0,\infty)$ 相交，再把非空交集写成有限个区间；它们全部进入 $H^{\rm far}$，不论编号为 $\ell=1$ 还是 $\ell\ge2$。如果改变允许的 $r_0$，新的局部项与远端项相互抵消，使总有限部分不变。证明是把 $r_0< R<r'_0$ 这一整段分别在两种划分中挪动，且被积函数逐点相同。

**本节实际解决的总结。** 固定半径替代了依赖方向的 $L(\mathbf u)$：近场绝对可积，远场与近场严格互补，外部第一交段不会遗漏。

## 第六节　远场再分流的唯一索引

**预期解决的目标。** 把解析通道和数值通道的职责写成互补集合上的积分，不出现未定义的提取项。

按有限的球面可见性单元 $\Omega_m$ 分片，在各开单元列出 $E^{\rm far}_{\mathbf x}(\mathbf u)$ 的所有非空区间 $[a_{m\ell}(\mathbf u),b_{m\ell}(\mathbf u)]$，其中 $a_{m\ell}\ge r_0>0$。必要时把由 $r_0$ 截断的区间单独标号。

对 $q=(m,\ell,k)$ 定义完整贡献

$$
I_q=Gw_k\int_{\Omega_m}T(\mathbf u)
\left[\int_{a_{m\ell}(\mathbf u)}^{b_{m\ell}(\mathbf u)}
\frac{\phi_k(\mathbf x-R\mathbf u)}R\,dR\right]d\Omega.
$$

令 $\mathcal Q$ 为全部三元索引的集合。选择一个精确定义的布尔标签 $\chi_q\in\{0,1\}$，并令

$$
\mathcal Q_A=\{q\in\mathcal Q:\chi_q=1\},
\qquad
\mathcal Q_B=\{q\in\mathcal Q:\chi_q=0\}.
$$

于是 $\mathcal Q_A\cap\mathcal Q_B=\varnothing$，$\mathcal Q_A\cup\mathcal Q_B=\mathcal Q$，并有

$$
\boxed{
H^{\rm far}
=\sum_{q\in\mathcal Q_A}I_q
+\sum_{q\in\mathcal Q_B}I_q.
}
$$

第一和可称“可解析评价部分”，第二和可称“直接求积部分”，但 $I_q$ 始终包含完整角向积分。若对同一个 $I_q$ 既运行 Carlson 又运行二维求积，这只是交叉核验，不能把两个结果相加。若算法另外采用逐点减去解析函数的方案，必须显式给出 $F=F^{\rm ext}+F^{\rm rem}$ 并分别求积；本稿的默认调度不使用这种未经定义的减法。

**本节实际解决的总结。** 所有远场贡献一次且仅一次相加；通道是同一 $I_q$ 的评价器，而非新物理项。

## 第七节　指数一的径向原函数和有限部分

**预期解决的目标。** 完整推导 $\alpha_k=1$ 的径向公式并反查其接触极限中的符号。

固定 $k$ 与 $\mathbf u$，令

$$
A_k=1+\sigma_k^2\|\mathbf p_k\|^2,
\quad b_k=\mathbf p_k\cdot\mathbf u,
\quad d_k^2=\sigma_k^{-2}+\|\mathbf p_k\|^2-b_k^2,
$$

$$
D_k(R)=1+\sigma_k^2\|\mathbf p_k-R\mathbf u\|^2
=\sigma_k^2R^2-2\sigma_k^2b_kR+A_k
=\sigma_k^2[(R-b_k)^2+d_k^2].
$$

对待积式先作部分分式：

$$
\begin{aligned}
\frac1{RD_k(R)}
&=\frac1{A_k}\left[\frac1R-
\frac{\sigma_k^2R-2\sigma_k^2b_k}{D_k(R)}\right],\\
\sigma_k^2R-2\sigma_k^2b_k
&=\frac12D_k'(R)-\sigma_k^2b_k.
\end{aligned}
$$

逐项积分并选择 $R>0$ 的实反正切连续分支，得到

$$
P_k(R,\mathbf u)
=\frac1{A_k}\left[
\log R-\frac12\log D_k(R)
+\frac{b_k}{d_k}\arctan\frac{R-b_k}{d_k}
\right].
$$

反向求导：

$$
\begin{aligned}
P_k'(R)
&=\frac1{A_k}\left[
\frac1R-\frac{\sigma_k^2(R-b_k)}{D_k(R)}
+\frac{\sigma_k^2b_k}{D_k(R)}
\right]\\
&=\frac1{A_k}\left[
\frac1R-\frac{\sigma_k^2R-2\sigma_k^2b_k}{D_k(R)}
\right]\\
&=\frac{D_k(R)-\sigma_k^2R^2+2\sigma_k^2b_kR}
{A_kRD_k(R)}
=\frac1{RD_k(R)}.
\end{aligned}
$$

因此当 $0<a<b$ 时，径向定积分是 $P_k(b)-P_k(a)$。作为独立的局部检验，对 $L>0$ 定义径向有限部分

$$
J_{k,\ell_0}^{\rm FP}(0,L)
=\lim_{\varepsilon\downarrow0}
\left[P_k(L)-P_k(\varepsilon)
-\frac1{A_k}\log\frac{\ell_0}{\varepsilon}\right].
$$

由 $D_k(0)=A_k$ 及 $\arctan(-b_k/d_k)=-\arctan(b_k/d_k)$，

$$
P_k(\varepsilon)
=\frac1{A_k}\left[
\log\varepsilon-\frac12\log A_k
-\frac{b_k}{d_k}\arctan\frac{b_k}{d_k}
\right]+o(1).
$$

所以

$$
\boxed{
J_{k,\ell_0}^{\rm FP}(0,L)
=\frac1{A_k}\left[
\log\frac{L}{\ell_0}
-\frac12\log\frac{D_k(L)}{A_k}
+\frac{b_k}{d_k}\left(
\arctan\frac{L-b_k}{d_k}
+\arctan\frac{b_k}{d_k}\right)
\right].
}
$$

不能把 $P_k(\varepsilon)-A_k^{-1}\log(\ell_0/\varepsilon)$ 当作有限极限；其中含 $2A_k^{-1}\log\varepsilon$。本节的单条射线有限部分与第五节固定半径的张量有限部分相容，但两者不是可以相加的额外项。

**本节实际解决的总结。** 径向初等原函数经反向求导核实；接触下限的极限顺序和符号均被明确。

## 第八节　Appell、Carlson 和 Theta 的合法位置

**预期解决的目标。** 分清固定方向径向求值与完整二维单元求值，并陈述椭圆/超椭圆通道的真实条件。

对固定方向令 $R=b_k+d_kt$，$\beta=b_k/d_k$。由

$$
\frac{dR}{R}=\frac{dt}{t+\beta},
\quad
1+\sigma_k^2\|\mathbf p_k-R\mathbf u\|^2
=\sigma_kd_k^2(1+t^2),
$$

得到一般指数的径向式

$$
\int\frac{\phi_k(\mathbf x-R\mathbf u)}R\,dR
=\sigma_kd_k^{-2\alpha_k}
\int\frac{dt}{t+\beta(1+t^2)^{\alpha_k}}.
$$

在选定复路径及连续分支且参数允许的情况下，这个一维原函数可用 Appell $F_1$ 或其退化函数表示。即使得到固定 $\mathbf u$ 的端点差 $P_q(\mathbf u)$，完整单元依然是

$$
I_q=Gw_k\int_{\Omega_m}T(\mathbf u)P_q(\mathbf u)\,d\Omega.
$$

非整数 $\alpha_k$ 不保证该球面二维积分变为 Appell 闭式；例如特殊参数会使 Appell 参数消去而退化为单变量函数。

沿某条大圆弧取 $\mathbf u(t)=\mathbf e_1\cos t+\mathbf e_2\sin t$。令 $\gamma=(\mathbf p_k\cdot\mathbf e_1-i\mathbf p_k\cdot\mathbf e_2)/2$、$z=e^{it}$。则

$$
b_k(z)=\gamma z+\frac{\overline\gamma}{z},
$$

$$
d_k^2(z)
=\frac{P_4(z)}{z^2},
\quad
P_4(z)=-\gamma^2z^4+\left(\sigma_k^{-2}+\|\mathbf p_k\|^2-2|\gamma|^2\right)z^2-\overline{\gamma}^{2}.
$$

若 $\gamma\ne0$，$P_4$ 确为四次且平方自由，则 $y^2=P_4(z)$ 的光滑射影紧化是亏格一曲线。但这只说明某个已证明化为有理函数乘 $dz/y$ 的一维子积分可按椭圆积分处理，不证明原二维面积分整体是椭圆闭式。若所谓一维化仍含同伦参数，就必须继续计算那个参数积分。

对 $\Phi(t)=\arctan(N(t)/\sqrt{P_4(t)})$，只有在路径上已选连续根号和反正切分支、且 $P_4(t)\ne0$ 及 $P_4(t)+N(t)^2\ne0$ 时，才有

$$
\Phi'(t)=\frac{N'(t)P_4(t)-\tfrac12N(t)P'_4(t)}
{[P_4(t)+N(t)^2]\sqrt{P_4(t)}}.
$$

若路径穿过极点、分支点或切触，则需要分别规定主值、绕行和跳跃修正。Carlson $(R_F,R_D,R_J,R_C)$ 是已识别的椭圆子积分的评价方法，其浮点停止式若没有余项、舍入与分支界，不能宣称严格误差证书。Theta 方法只用于确实得到平方自由的 $y^2=P_{2g+1}$ 或 $y^2=P_{2g+2}$ 且 $g\ge2$ 的特定 Abel 子积分；非凸性本身不推出这类曲线。

**本节实际解决的总结。** 特殊函数都被放在经证明适用的子积分层级，绝不当作整个 Hessian 的自动闭式。

# 第三部分　主线 B：Feynman 参数化与移动中心径向核

## 第九节　Feynman 恒等式与量纲正确的配方

**预期解决的目标。** 对任意 $\alpha_k>0$ 严格熔合两个中心的二次型，核对先前参考说法中的量纲错误。

固定一个核并暂省下标 $k$。取 $a=\sigma^{-1}>0$，

$$
A=\|\mathbf y-\mathbf x\|^2,
\qquad
B=\|\mathbf y-\mathbf c\|^2+a^2.
$$

两个量都是长度平方。对 $A>0$、$B>0$ 与 $\alpha>0$，Gamma 积分给出

$$
A^{-1/2}=\frac1{\Gamma(1/2)}
\int_0^\infty s^{-1/2}e^{-sA}\,ds,
$$

$$
B^{-\alpha}=\frac1{\Gamma(\alpha)}
\int_0^\infty t^{\alpha-1}e^{-tB}\,dt.
$$

令 $s=u\lambda$，$t=(1-u)\lambda$，其中 $0<u<1$、$\lambda>0$。雅可比为 $ds\,dt=\lambda\,du\,d\lambda$。于是

$$
\begin{aligned}
A^{-1/2}B^{-\alpha}
={}&\frac1{\Gamma(1/2)\Gamma(\alpha)}
\int_0^1u^{-1/2}(1-u)^{\alpha-1}\,du
\int_0^\infty \lambda^{\alpha-1/2}
e^{-\lambda[uA+(1-u)B]}\,d\lambda\\
={}&\frac{\Gamma(\alpha+1/2)}{\Gamma(1/2)\Gamma(\alpha)}
\int_0^1u^{-1/2}(1-u)^{\alpha-1}
[uA+(1-u)B]^{-\alpha-1/2}\,du.
\end{aligned}
$$

令 $\mathbf d=\mathbf x-\mathbf c$ 并定义

$$
\mathbf z(u)=u\mathbf x+(1-u)\mathbf c,
\qquad
m^2(u)=(1-u)a^2+u(1-u)\|\mathbf d\|^2.
$$

直接展开二次型：

$$
\begin{aligned}
u\|\mathbf y-\mathbf x\|^2
+(1-u)\|\mathbf y-\mathbf c\|^2
={}&\|\mathbf y\|^2
-2\mathbf y\cdot[u\mathbf x+(1-u)\mathbf c]\\
&+u\|\mathbf x\|^2+(1-u)\|\mathbf c\|^2\\
={}&\|\mathbf y-\mathbf z(u)\|^2
+u(1-u)\|\mathbf x-\mathbf c\|^2.
\end{aligned}
$$

所以

$$
\boxed{uA+(1-u)B=\|\mathbf y-\mathbf z(u)\|^2+m^2(u).}
$$

当 $u<1$ 时 $m^2(u)>0$，当 $u\uparrow1$ 时 $m^2(u)\downarrow0$。参考式中 $u+(1-u)\sigma^2$ 相加量纲不一致；这里通过先把 Cauchy 核写成 $\sigma^{-2\alpha}B^{-\alpha}$ 消除了此错误。对于 $\alpha=1$，Gamma 系数为 $\Gamma(3/2)/[\Gamma(1/2)\Gamma(1)]=1/2$。

**本节实际解决的总结。** Feynman 核熔合适用于所有 $\alpha>0$，移动中心沿 $\mathbf c$ 至 $\mathbf x$ 的直线段，端点 $u=1$ 仍有退化。

## 第十节　严格的势函数参数式和积分交换

**预期解决的目标。** 把空间积分与参数积分的交换条件和剩余积分维数说清楚。

记 $p_k=\alpha_k+1/2$，

$$
C_{\alpha_k}=
\frac{\Gamma(\alpha_k+1/2)}{\Gamma(1/2)\Gamma(\alpha_k)},
$$

$$
S_p(\mathbf z,m;V)
=\int_V[\|\mathbf y-\mathbf z\|^2+m^2]^{-p}\,d^3\mathbf y.
$$

由第九节逐点恒等式，

$$
\boxed{
U_K(\mathbf x)
=G\sum_{k=1}^{K}w_k\sigma_k^{-2\alpha_k}C_{\alpha_k}
\int_0^1u^{-1/2}(1-u)^{\alpha_k-1}
S_{p_k}(\mathbf z_k(u),m_k(u);V)\,du.
}
$$

单个无符号核在 $V\times(0,1)$ 非负。Tonelli 定理允许先后交换积分；原 Newton 势的局部 $1/R$ 在三维可积且 $V$ 有界，故总积分有限。对于有正负权重的有限和，可以先分别交换每个核，最后相加。参考文中“空间积分已经初等闭合”不是上述恒等式的推论；此时 $S_p$ 仍是三维体积分。

特例 $\alpha_k=1$ 给出

$$
U_k(\mathbf x)
=\frac{Gw_k}{2\sigma_k^2}
\int_0^1u^{-1/2}
S_{3/2}(\mathbf z_k(u),m_k(u);V)\,du.
$$

**本节实际解决的总结。** 势函数的单参数表示严格成立；空间积分并未因核呈径向而自行消失。

## 第十一节　径向核如何降至面以及为何不能自动降至棱

**预期解决的目标。** 给出可验证的散度场，并准确标明降维到面和棱的界限。

固定 $p>0$、$m>0$，令 $\mathbf r=\mathbf y-\mathbf z$，$r=\|\mathbf r\|$。构造

$$
f_p(r;m)=r^{-3}\int_0^r
s^2(s^2+m^2)^{-p}\,ds,
\qquad
\mathbf F_p(\mathbf r;m)=f_p(r;m)\mathbf r.
$$

在 $r=0$ 定义连续值 $f_p(0;m)=m^{-2p}/3$。对 $r>0$ 求导：

$$
\begin{aligned}
\nabla_{\mathbf r}\cdot\mathbf F_p
&=3f_p(r;m)+rf'_p(r;m)\\
&=\frac1{r^2}\frac{d}{dr}[r^3f_p(r;m)]\\
&=\frac1{r^2}\frac{d}{dr}
\int_0^r s^2(s^2+m^2)^{-p}\,ds\\
&=(r^2+m^2)^{-p}.
\end{aligned}
$$

因为 $m>0$，该向量场在固定 $u<1$ 的 $\overline V$ 邻域光滑。由散度定理严格得到

$$
\boxed{
S_p(\mathbf z,m;V)
=\sum_{f=1}^{F}
\int_{\Delta_f}\mathbf F_p(\mathbf y-\mathbf z;m)
\cdot\mathbf n_f\,dS.
}
$$

对 $p=3/2$ 可直接求出径向反导数：

$$
\int_0^r\frac{s^2}{(s^2+m^2)^{3/2}}\,ds
=\operatorname{arsinh}\frac rm
-\frac{r}{\sqrt{r^2+m^2}}.
$$

它的导数是

$$
\frac1{\sqrt{r^2+m^2}}
-\frac{m^2}{(r^2+m^2)^{3/2}}
=\frac{r^2}{(r^2+m^2)^{3/2}}.
$$

故 $\mathbf F_{3/2}$ 是显式初等函数，但每个面片还剩二维积分：

$$
\int_{\Delta_f}
f_{3/2}(\|\mathbf y-\mathbf z\|;m)(\mathbf y-\mathbf z)\cdot\mathbf n_f\,dS.
$$

在固定面片的平面上，法向因子 $(\mathbf y-\mathbf z)\cdot\mathbf n_f$ 是常数；径向因子仍随面内坐标变化。可以再构造面内向量场并转化为棱边一维积分，但其原函数不保证对任意 $p$、$m$ 和三角形都只有有限个对数、反双曲正弦与反正切；棱边积分和 $u$ 积分也未自动解析。特别不能把此软化径向核的散度式等同于常密度 Newton 核的 Werner–Scheeres 闭式。

**本节实际解决的总结。** 在 $m>0$ 时，三维到二维面片的降维得到严格证明；二维到一维棱边及一维外参数积分是否初等，需要分别推导，不可由球对称直接断言。

## 第十二节　Feynman 势函数的外部 Hessian

**预期解决的目标。** 展开对移动中心与平滑尺度的求导，给出外部场点可交换求导的精确公式。

固定 $k$，记

$$
q_k(u,\mathbf y;\mathbf x)
=u\|\mathbf y-\mathbf x\|^2
+(1-u)[\|\mathbf y-\mathbf c_k\|^2+a_k^2],
\quad p_k=\alpha_k+\frac12.
$$

由于未配方的这个表达式对 $\mathbf x$ 求导更简单，

$$
\partial_{x_i}q_k=2u(x_i-y_i),
\qquad
\partial_{x_j}\partial_{x_i}q_k=2u\delta_{ij}.
$$

先求第一导数：

$$
\partial_{x_i}q_k^{-p_k}
=-p_kq_k^{-p_k-1}\partial_{x_i}q_k
=-2p_ku(x_i-y_i)q_k^{-p_k-1}.
$$

再求第二导数：

$$
\begin{aligned}
\partial_{x_j}\partial_{x_i}q_k^{-p_k}
={}&-2p_ku\delta_{ij}q_k^{-p_k-1}\\
&-2p_ku(x_i-y_i)(-p_k-1)q_k^{-p_k-2}
\partial_{x_j}q_k\\
={}&-2p_ku\delta_{ij}q_k^{-p_k-1}
+4p_k(p_k+1)u^2(x_i-y_i)(x_j-y_j)q_k^{-p_k-2}.
\end{aligned}
$$

若 $\operatorname{dist}(\mathbf x,V)>0$，则原势在 $\mathbf x$ 的一个邻域内可逐点微分并交换积分。对 Feynman 参数式，可先在 $u\le1-\delta$ 微分，随后利用 $\|\mathbf x-\mathbf y\|\ge d>0$ 与非负核的标准主导估计将 $\delta\downarrow0$。得到

$$
\boxed{
\begin{aligned}
(H_K)_{ij}(\mathbf x)
={}&G\sum_{k=1}^{K}w_k\sigma_k^{-2\alpha_k}C_{\alpha_k}
\int_0^1u^{-1/2}(1-u)^{\alpha_k-1}\,du\\
&\times\int_V
\left[-2p_ku\delta_{ij}q_k^{-p_k-1}
+4p_k(p_k+1)u^2(x_i-y_i)(x_j-y_j)q_k^{-p_k-2}
\right]d^3\mathbf y.
\end{aligned}}
$$

式中同一行的 $du\times\int_V$ 解释为嵌套二重积分。它是外部 Hessian 的另一个严格表示，不能因仅有一个显式参数 $u$ 而忽略内层空间积分。

**本节实际解决的总结。** 外部 Hessian 的参数式逐项展开；内部和边界的极限不能未经处理直接套用这个求导交换步骤。

## 第十三节　参数端点与有限部分之间的边界

**预期解决的目标。** 说明 Feynman 变换并未消灭近场奇异性，也没有独立提供边棱处的点值。

令 $v=1-u\downarrow0$。直接代入第九节的定义：

$$
\mathbf z(1-v)=\mathbf x-v(\mathbf x-\mathbf c),
$$

$$
m^2(1-v)
=v a^2+v(1-v)\|\mathbf x-\mathbf c\|^2
=v(a^2+\|\mathbf x-\mathbf c\|^2)-v^2\|\mathbf x-\mathbf c\|^2.
$$

因此 $\mathbf z\to\mathbf x$，$m\to0$。当 $\mathbf x\in V$ 或 $\mathbf x\in\partial V$，空间核在 $u=1$ 恢复源点重合的奇异性。势积分仍有限，但对参数化表达式求两次导数、交换极限以及把边界点称为 Hessian 都需要第三节的有限部分或其他清楚声明的界面意义。

定义 Feynman 路线的严谨使用规则：外部点直接使用第十二节；内部、光滑边界、棱边或顶点先求具有正距离的场点并分析一侧极限，或独立证明与第三节同一球形排异有限部分一致，随后才可声称在该位置计算 $H_{K,\ell_0}^{\rm FP}$。简单地在 $u=1$ 对浮点求积不构成此证明。

**本节实际解决的总结。** Feynman 是独立的积分表示，近表面正则化问题仍存在并在参数端点集中体现。

# 第四部分　数值协议、交叉验证与结论

## 第十四节　两条路线的实现分工和误差声明

**预期解决的目标。** 使实现互不重复，明确绝对误差与经验容差的区别。

| 位置或任务 | RTFP 路线 | Feynman 路线 |
|---|---|---|
| 外部点的势和 Hessian | 全射线远场、普通径向区间 | 移动中心径向核加参数积分；外部 Hessian 可直接用第十二节 |
| 内部点 | 固定切锥局部差分加完整远场 | 需证明导数与端点交换或取合适极限 |
| 光滑边界 | 明确球排异代表及一侧极限区别 | 参数 $u=1$ 需单独端点分析 |
| 棱、顶点 | 按 $\rho_KC_{\mathbf x}$ 判定对数发散和尺度 | 不能靠移动中心自动消去局部发散 |
| 几何处理 | 射线多交段的完备枚举 | 每个参数的径向核可严格降为各面片积分 |
| 尚存积分 | 角向面积分及必要的径向积分 | 面片二维积分与一维参数积分；如经证明，可进一步降为棱边一维积分 |

用 $H_{\rm raw}$ 保存未经强制对称化的张量，定义

$$
r_{\rm sym}=
\frac{\|H_{\rm raw}-H_{\rm raw}^{T}\|_F}
{\max(\|H_{\rm raw}\|_F,\tau)},\qquad\tau>0.
$$

在检查以后可报告 $(H_{\rm raw}+H_{\rm raw}^T)/2$，但对称化后的差值不能作为独立验证。外部迹为零，内部球排异代表迹为 $-4\pi G\rho_K(\mathbf x)$，球排异边界代表迹为 $-4\pi G\vartheta_{\mathbf x}\rho_K(\mathbf x)$；迹检验不能排除无迹的矩阵误差。

若数值例程只执行步长折半 $|I_{h/2}-I_h|<\mathrm{tol}$，仅能返回估计误差。严格误差界还需径向、角向、面片、参数端点截断、特殊函数余项及浮点舍入的有效上界，其和不大于声明容差。对 Carlson 和 Theta 也适用同一原则。节点数如 15–30 或预期相对误差 $10^{-15}$ 不得写为本理论的普适保证。

**本节实际解决的总结。** 两条主线的输入、输出和开放问题已经逐项分开；误差证书只在完整界已建立时授予。

## 第十五节　最小闭环验证与不能推出的主张

**预期解决的目标。** 给出可证伪的交叉验证场景，不用单一迹或单点基准代替全面核验。

先选凸多面体的外部场点，在同一 $V$、同一 $\rho_K$、同一 $G$ 下计算三组对象：原体积积分的高精度独立参考、RTFP 全部交段之和、Feynman 参数式中仍包含正确求值的空间积分。对势、三个加速度分量及九个未经对称化的 Hessian 分量分别比较。

然后选择非凸多面体的外部点，以确保 RTFP 的多段全部计入。再选择光滑面相对内部，分别记录内侧与外侧的 Hessian 极限和第三节定义的球排异有限部分。最后在棱与顶点验证 $J_\varepsilon-G\rho_KC_{\mathbf x}\log(\ell_0/\varepsilon)$ 的稳定趋势；若 $\rho_KC_{\mathbf x}\ne0$，原 $J_\varepsilon$ 必须呈现指定的对数增长。

上述检验能发现实现错误，但任何有限次数的测试不能推出任意多面体、任意精度和任意浮点平台的绝对正确性。分析意义上的等式已经在相应假设下证明；实际程序正确性和误差控制仍要单独建立。

**本节实际解决的总结。** 给出了外部、非凸、光滑界面和棱顶点四类独立验证任务，同时限定了它们的证明力。

## 最终结论

**预期解决的目标。** 用同一套定义收束两条主线，不做数学或性能上的过度承诺。

1. RTFP 给出指定球形排异规范下的有限部分恒等式，局部固定切锥与全部远端交段严格互补。它与 Werner 面棱闭式在定义和算法上相互独立。
2. Feynman 参数化严格把两个中心的二次型合成沿 $\mathbf c_k$ 到 $\mathbf x$ 移动的一个中心；它把每个固定参数的径向核体积分严格降到多面体各面片积分，但不会自动消掉面片积分、参数积分或边界奇异性。
3. 对外部场点，两条路线计算同一个经典 $H_K$，可以相互交叉验证。对内部及边界，只有先固定同一导数和球排异约定，再证明极限交换，才能比较数值对象。
4. Appell、Carlson、Theta、同伦 DE 和直接求积仅在其各自有证明的子问题上使用；不存在由算法名称自动推导出的全局解析闭式或绝对误差保证。

**本节实际解决的总结。** 得到两条独立、可比较、条件清楚的推导主线；剩余积分和数值证明责任均有明确位置。
