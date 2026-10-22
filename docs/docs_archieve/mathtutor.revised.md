# 第一小节：我们在解决什么物理问题？（背景、几何与密度模型）

## 1. 物理背景：为什么航天器降落小行星这么难？

想象你正在指挥一艘探测器（比如日本的“隼鸟2号”探测龙宫小行星，或者 NASA 的“奥西里斯-雷克斯”探测贝努小行星）在小行星表面降落采样。

在地球轨道上，地球很大很圆，引力几乎就是一个理想的质点引力（中心引力场：$F = GM/r^2$）。
但在小行星表面：
1. **形状极其怪异**：它不像行星那样是球体，而像一个到处是棱角、大石块、深坑的“大土豆”或“陀螺”。
2. **引力极度微弱且敏感**：探测器受到的潮汐力、轨道摄动，很大程度上取决于引力随着空间位置变化得有多快——这在数学上叫作**引力梯度（Gravity Gradient）**，也就是引力势的二阶空间导数矩阵（**Hessian 矩阵**）。
3. **传统假设失效**：过去 30 年经典的算法（如 1996 年著名的 Werner-Scheeres 多面体模型）都假设小行星**内部密度处处相同（均匀常数 $\rho_0$）**。但实际的小行星是碎石堆积体，内部有空隙、有重金属石块，密度是**变密度**。

我们要做的，就是为**任意形状、且内部密度不均匀的小天体**，推导出一套高精度计算其**表面与空间引力梯度（Hessian）**的严格数学体系。

---

## 2. 几何建模：空间区域 $V$

我们把这个小天体在三维空间中占有的实体积记为：
$$V \subset \mathbb{R}^3$$
- 它是一个**有界区域**（体积有限）。
- 它的外表面记作 $\partial V$。在工程实际中，这个外表面通常是用三维计算机图形学里的**三角网格多面体**（由许多个三角形平面拼成的封闭立体）来逼近的。

---

## 3. 核心创新：广义 Cauchy 密度核模型

为了描述小行星内部“东一块重、西一块轻”的连续不均匀质量分布，我们用有限个（共 $K$ 个）**广义 Cauchy 径向基函数**的线性叠加来表示密度分布 $\rho_K(\mathbf{y})$：

$$\rho_K(\mathbf{y}) = \sum_{k=1}^K w_k \phi_k(\mathbf{y})$$

其中每一个分量核 $\phi_k(\mathbf{y})$ 定义为：
$$\phi_k(\mathbf{y}) = \left[ 1 + \sigma_k^2 \|\mathbf{y} - \mathbf{c}_k\|^2 \right]^{-\alpha_k}$$

这个公式看似复杂，其实每一项都有非常直观的物理意义：
- **$\mathbf{y} \in V$**：小行星实体内部任意一点的三维坐标。
- **$\mathbf{c}_k \in \mathbb{R}^3$**：第 $k$ 个“质量团/密度异常区”的中心位置。
- **$\|\mathbf{y} - \mathbf{c}_k\|$**：从点 $\mathbf{y}$ 到这个中心 $\mathbf{c}_k$ 的欧氏几何距离。
- **$\sigma_k > 0$**：**空间逆尺度参数**。$\sigma_k$ 越大，说明这个质量团越小、越集中；$\sigma_k$ 越小，说明这个质量团弥散得越开。
- **$\alpha_k > 0$**：**衰减幂次**。
  - 当 $\alpha_k = 1$ 时，它就是最经典的柯西-洛伦兹分布（Cauchy / Lorentzian 分布），分母是典型的二次多项式 $1 + \sigma_k^2 r^2$；
  - 当 $\alpha_k$ 很大时，它的边缘衰减会非常快，行为接近高斯分布。
- **$w_k \in \mathbb{R}$**：第 $k$ 个核的**权重系数**（振幅），可以理解为这个区域贡献的密度基数。

> **通俗理解**：这就好比在小行星里埋了 $K$ 个“软石块/气泡”，每个都有自己的中心 $\mathbf{c}_k$、大小影响范围 $1/\sigma_k$ 和密度高度 $w_k$。把它们叠加起来，就能拟合小行星内部任意平滑变化的真实密度场。

---

## 4. 我们的计算目标：引力势 $U$ 与 Hessian 矩阵 $H$

设空间中有一个观测点 $\mathbf{x} \in \mathbb{R}^3$（比如探测器当前所在的位置）：

### (1) 引力势（标量）
根据经典牛顿万有引力定律，整个小天体在点 $\mathbf{x}$ 处激发的引力势是全天体的体积分：
$$U_K(\mathbf{x}) = G \int_V \frac{\rho_K(\mathbf{y})}{\|\mathbf{x} - \mathbf{y}\|} d^3\mathbf{y}$$
（其中 $G$ 为万有引力常数，$d^3\mathbf{y}$ 是微元体积 $dx\,dy\,dz$）。

### (2) 物理引力加速度（矢量）
此处 $U_K=+G\int_V\rho_K(\mathbf y)/\|\mathbf x-\mathbf y\|\,d^3\mathbf y$ 定义的是**正值势函数**，不是负值的单位质量势能。因此物理加速度为：
$$\mathbf{g}(\mathbf{x}) = \nabla_\mathbf{x}U_K(\mathbf{x})=-G\int_V\rho_K(\mathbf y)\frac{\mathbf x-\mathbf y}{\|\mathbf x-\mathbf y\|^3}\,d^3\mathbf y.$$
点质量检验：$U=GM/r$，所以 $\mathbf g=-GM\mathbf x/r^3$，方向指向质量中心。若改用负势 $\Phi=-U$，则等价地写作 $\mathbf g=-\nabla\Phi$；全文固定使用正值 $U$。

### (3) Hessian 矩阵（$3 \times 3$ 对称张量）
把引力势对场点坐标 $\mathbf{x}$ 求**二阶导数**，就得到了 Hessian 矩阵：
$$H_K(\mathbf{x}) = \nabla_\mathbf{x} \nabla_\mathbf{x} U_K(\mathbf{x})$$
写成分量形式，它是一个 $3 \times 3$ 的矩阵，第 $(i, j)$ 个元素就是：
$$(H_K)_{ij} = \frac{\partial^2 U_K}{\partial x_i \partial x_j}$$
按照本文的正值势约定，引力梯度张量是 $\Gamma=\nabla\mathbf g=H_K$。若改用负势 $\Phi=-U$，则 $\Gamma=-\nabla\nabla\Phi$，数值相同。

---

## 5. 为什么这个问题在数学上极其致命？

请注意引力势积分里的分母：
$$\frac{1}{\|\mathbf{x} - \mathbf{y}\|}$$
设距离为 $R = \|\mathbf{x} - \mathbf{y}\|$。
- 求一阶导数（引力加速度）：核变成 $\sim \frac{1}{R^2}$；
- 求二阶导数（Hessian）：核直接变成了 $\sim \frac{1}{R^3}$！

如果探测器在很远的太空中（$\mathbf{x}$ 在天体外远处），$R$ 永远不会为零，这个积分用简单的数值方法就能算。
**但是，一旦探测器贴近表面、降落甚至在表面移动时**，观测点 $\mathbf{x}$ 就在积分域 $V$ 的边界上，积分源点 $\mathbf{y}$ 可以无限逼近 $\mathbf{x}$，导致：
$$R = \|\mathbf{x} - \mathbf{y}\| \to 0$$
在三维空间中，$\frac{1}{R^3}$ 的积分是**强奇异甚至发散的（超奇异）**！普通的微积分在这里会直接崩溃。

这就是为什么整篇论文需要用**狄拉克分布（Dirac Delta）、Hadamard 有限部分、四元数、代数几何与椭圆积分**这整套数学重武器去解构它。

---

### 第一小节小结与确认

到现在为止，我们交代了：
1. **输入**：多面体区域 $V$、广义 Cauchy 密度核 $\rho_K(\mathbf{y})$；
2. **目标**：求引力势的二阶空间导数矩阵 $H_K(\mathbf{x})$；
3. **核心难点**：当探测器靠近表面时，二阶导数带来的 $\frac{1}{R^3}$ 奇点会导致发散。

---

# 第二小节：数学工具底座与 Newton 核的二阶导数（狄拉克分布的诞生）

---

## 1. 基础热身：多维几何距离的偏导数怎么求？

设场点为 $\mathbf{x} = (x_1, x_2, x_3)$，源点为 $\mathbf{y} = (y_1, y_2, y_3)$。
定义相对位移矢量：
$$\mathbf{r} = \mathbf{x} - \mathbf{y} = (x_1 - y_1, \; x_2 - y_2, \; x_3 - y_3)$$
两个点之间的欧氏距离记为 $R$：
$$R = \|\mathbf{r}\| = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + (x_3 - y_3)^2}$$
我们要对场点坐标 $x_i$ 求偏导数（记作 $\partial_i = \frac{\partial}{\partial x_i}$）。

### (1) 距离 $R$ 的一阶导数
根据复合函数链式法则：
$$\partial_i R = \frac{\partial}{\partial x_i} \left( \sum_{k=1}^3 (x_k - y_k)^2 \right)^{1/2} = \frac{1}{2\sqrt{\dots}} \cdot 2(x_i - y_i) = \frac{x_i - y_i}{R} = \frac{r_i}{R}$$
> **直观理解**：距离 $R$ 随着 $x_i$ 的变化率，就是相对位移在该坐标轴上的投影比例（即方向余弦）。

### (2) 牛顿核 $1/R$ 的一阶导数
求引力势对坐标的导数（也就是引力加速度分量）：
$$\partial_i \left( \frac{1}{R} \right) = -\frac{1}{R^2} \cdot \partial_i R = -\frac{1}{R^2} \left( \frac{r_i}{R} \right) = -\frac{r_i}{R^3}$$
写成三维矢量形式，就是大家很熟悉的牛顿引力场形式：
$$\nabla \left( \frac{1}{R} \right) = -\frac{\mathbf{r}}{R^3}$$

### (3) 牛顿核 $1/R$ 的二阶导数（当 $R \neq 0$ 时）
现在我们要计算引力梯度（Hessian 矩阵的每个分量），需要对 $x_j$ 再次求偏导：
$$\partial_i \partial_j \left( \frac{1}{R} \right) = \partial_j \left( -\frac{r_i}{R^3} \right)$$
利用微积分的商法则（或乘积法则）：
$$\partial_j \left( -\frac{r_i}{R^3} \right) = -\frac{(\partial_j r_i) R^3 - r_i (\partial_j R^3)}{(R^3)^2}$$
注意：
- $\partial_j r_i = \frac{\partial (x_i - y_i)}{\partial x_j} = \delta_{ij}$（当 $i=j$ 时为 $1$，当 $i \neq j$ 时为 $0$，这是 Kronecker 符号）；
- $\partial_j (R^3) = 3R^2 \partial_j R = 3R^2 \left( \frac{r_j}{R} \right) = 3R r_j$。

代入分式中：
$$\partial_i \partial_j \left( \frac{1}{R} \right) = -\frac{\delta_{ij} R^3 - r_i (3R r_j)}{R^6} = \frac{3 r_i r_j - R^2 \delta_{ij}}{R^5} \quad (R \neq 0)$$

我们把它写成无坐标的 $3 \times 3$ 矩阵形式：
$$\nabla \nabla \left( \frac{1}{R} \right) = \frac{3\mathbf{r}\mathbf{r}^T - R^2 I}{R^5}$$
（其中 $\mathbf{r}\mathbf{r}^T$ 是两个列向量的外积，生成一个 $3 \times 3$ 矩阵；$I$ 是单位矩阵）。

---

## 2. 惊人的数学矛盾：经典微积分在 $R=0$ 处暴毙

让我们对上面这个 $3 \times 3$ 矩阵求它的**矩阵迹（Trace，即主对角线元素之和）**。
矩阵的迹就是拉普拉斯算子（Laplacian）：
$$\mathrm{Tr}\left[ \nabla \nabla \left( \frac{1}{R} \right) \right] = \sum_{i=1}^3 \partial_i \partial_i \left( \frac{1}{R} \right) = \nabla^2 \left( \frac{1}{R} \right)$$
把上面的公式直接代入求和：
$$\sum_{i=1}^3 \frac{3 r_i^2 - R^2 \delta_{ii}}{R^5} = \frac{3 \sum r_i^2 - R^2 \sum \delta_{ii}}{R^5}$$
因为 $\sum_{i=1}^3 r_i^2 = R^2$，而三维空间中单位矩阵的对角线和 $\sum_{i=1}^3 \delta_{ii} = 1 + 1 + 1 = 3$，所以分子变成了：
$$3R^2 - R^2(3) = 0$$
也就是说，在任何 $R \neq 0$ 的地方：
$$\nabla^2 \left( \frac{1}{R} \right) = 0$$
（这说明在没有质量的真空中，引力势满足调和方程/拉普拉斯方程）。

### 矛盾来了！
不能把仅在 $R\neq0$ 成立的逐点恒等式跨奇点直接积分。若在挖掉原点的穿孔球上运用散度定理，必须同时计入内球面通量；若把原点包括进来，应改用分布意义的导数。
但我们用大二学过的**高斯散度定理（Gauss Divergence Theorem）**来算一次：
$$\int_{B_\varepsilon} \nabla^2 \left( \frac{1}{R} \right) d^3\mathbf{r} = \int_{B_\varepsilon} \nabla \cdot \left[ \nabla \left( \frac{1}{R} \right) \right] d^3\mathbf{r} = \oint_{\partial B_\varepsilon} \nabla \left( \frac{1}{R} \right) \cdot \mathbf{n} \, dS$$
- 球面外法向单位向量是 $\mathbf{n} = \frac{\mathbf{r}}{\varepsilon}$；
- 引力梯度是 $\nabla(1/R) = -\frac{\mathbf{r}}{\varepsilon^3}$；
- 二者内积：$\nabla(1/R) \cdot \mathbf{n} = -\frac{\mathbf{r} \cdot \mathbf{r}}{\varepsilon^4} = -\frac{\varepsilon^2}{\varepsilon^4} = -\frac{1}{\varepsilon^2}$；
- 球面的表面积是 $4\pi \varepsilon^2$。

把它们乘起来：
$$\oint_{\partial B_\varepsilon} \left( -\frac{1}{\varepsilon^2} \right) dS = -\frac{1}{\varepsilon^2} \cdot (4\pi \varepsilon^2) = -4\pi \neq 0 \text{ ！！}$$

**为什么散度定理算出来是 $-4\pi$，而逐点求导算出来全是 $0$？**
因为在原点 $R=0$ 处普通函数求导失效；分布意义的拉普拉斯算子在原点集中一个总质量为 $-4\pi$ 的 Dirac 项。

---

## 3. 数学武器 1：狄拉克 $\delta$ 分布（Dirac Delta）

为了拯救这个矛盾，物理学家狄拉克和数学家施瓦茨（Laurent Schwartz）建立了**广义函数（分布，Distribution）理论**。

在分布意义下，泊松方程的严格表述是：
$$\nabla^2 \left( \frac{1}{R} \right) = -4\pi \delta(\mathbf{r})$$
其中 $\delta(\mathbf{r})$ 就是三维狄拉克函数。它的核心性质是“采样子性质”：只要某个连续函数 $f(\mathbf{r})$ 和它一起积分，就会把原点的值抠出来：
$$\int_{\mathbb{R}^3} f(\mathbf{r}) \delta(\mathbf{r}) d^3\mathbf{r} = f(\mathbf{0})$$

---

## 4. 数学武器 2：柯西主值（Cauchy Principal Value, $\mathrm{p.v.}$）

因为 $\frac{3r_i r_j - R^2 \delta_{ij}}{R^5}$ 在 $R \to 0$ 时是 $O(1/R^3)$ 的强发散，如果你直接拿它去积分，积分值是不存在的（发散到无穷大）。
但数学家发现，这个函数有一个极其优美的性质——**奇偶对称性**！
这个张量核关于 $\mathbf r$ 是偶函数，不能用 $\mathbf r$ 与 $-\mathbf r$ 的奇偶性抵消。真正的抵消来自完整球面的角平均：由 $\int_{S^2}u_i u_j\,d\Omega=(4\pi/3)\delta_{ij}$，有 $\int_{S^2}(3u_i u_j-\delta_{ij})\,d\Omega=0$。在全空间的光滑测试函数上做球形排异后，才得到主值：
$$\mathrm{p.v.} \int_V K(\mathbf{r}) d^3\mathbf{r} \equiv \lim_{\varepsilon \to 0^+} \int_{V \setminus B_\varepsilon(\mathbf{0})} K(\mathbf{r}) d^3\mathbf{r}$$
这里抵消的是完整球面上的角平均；若区域只包含部分方向（例如棱锥切锥），主值未必存在，必须明确采用第六小节的 Hadamard 有限部分。

---

## 5. 严格推导：二阶导数张量的完整分布恒等式

我们把迹的性质（$-4\pi \delta(\mathbf{r})$）分配给各个坐标分量。
由于三维空间具有完全的旋转对称性（各向同性），三个对角线方向 $x, y, z$ 的地位必须绝对均等。
因为对角线三项之和为 $-4\pi \delta(\mathbf{r})$，所以均摊到每一个对角线方向，正好是：
$$-\frac{4\pi}{3} \delta(\mathbf{r})$$

数学上经过严格证明（通过在测试函数空间做外微分和边界分部积分），得到了在全空间都成立的**严格分布二阶导数恒等式**：
$$\partial_i \partial_j \left( \frac{1}{R} \right) = \mathrm{p.v.}\left[ \frac{3r_i r_j - R^2 \delta_{ij}}{R^5} \right] - \frac{4\pi}{3} \delta_{ij} \delta(\mathbf{r})$$

写成我们论文里的 $3 \times 3$ 矩阵张量形式，就是：
$$\nabla \nabla \left( \frac{1}{R} \right) = \mathrm{p.v.}\left[ \frac{3\mathbf{r}\mathbf{r}^T - R^2 I}{R^5} \right] - \frac{4\pi}{3} \delta(\mathbf{r}) I$$

> **敲黑板（通俗总结）**：
> 以后见到二阶导数 $\nabla\nabla(1/R)$，在球形排异的分布约定下，它**不能只写前面的分式**，而包含两件东西：
> 1. 第一项：**主值分式项**，描述场点远离源点时的长程潮汐力（通过对称挖洞定义）；
> 2. 第二项：**狄拉克脉冲项** $-\frac{4\pi}{3}\delta(\mathbf{r})I$，这是分布恒等式中的局部项；它不是边界处经典 Hessian 的唯一“点值”。

---

## 6. 物理落地：代入小天体引力势 Hessian

现在，我们把这个金光闪闪的分布恒等式代回小天体的 Hessian 积分定义中：
$$H_K(\mathbf{x}) = G \int_V \rho_K(\mathbf{y}) \nabla_\mathbf{x} \nabla_\mathbf{x} \left( \frac{1}{\|\mathbf{x} - \mathbf{y}\|} \right) d^3\mathbf{y}$$
直接代入分解公式：
$$H_K(\mathbf{x}) = G \,\mathrm{p.v.} \int_V \rho_K(\mathbf{y}) \frac{3\mathbf{r}\mathbf{r}^T - R^2 I}{R^5} d^3\mathbf{y} - \frac{4\pi G}{3} I \int_V \rho_K(\mathbf{y}) \delta(\mathbf{x} - \mathbf{y}) d^3\mathbf{y}$$

请紧紧盯住第二项的积分：
$$\int_V \rho_K(\mathbf{y}) \delta(\mathbf{x} - \mathbf{y}) d^3\mathbf{y}$$
这里的 $\mathbf{x}$ 是我们的观测点位置，$\mathbf{y}$ 是天体区域 $V$ 里的点。狄拉克函数只有在 $\mathbf{y} = \mathbf{x}$ 处才激发出值。
那么，点 $\mathbf{x}$ 究竟是不是在区域 $V$ 里面？
这就引出了**局部内立体角比例 $\theta_{\mathrm{ball}}(\mathbf{x})$**：

$$\theta_{\mathrm{ball}}(\mathbf{x}) = \frac{\Omega_{\mathrm{in}}(\mathbf{x})}{4\pi}$$

我们来进行分类讨论：

```
                      观测点 x 的四种空间几何位置
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
    【天体内部】             【光滑表面】              【天体外部】
  x 深度处于 V 内         x 落在光滑的外表面          x 在太空真空中
  周围 4π 全是介质        半个空间是介质 (2π)       周围没有任何介质 (0)
   Ω_in = 4π                Ω_in = 2π               Ω_in = 0
   θ_ball(x) = 1              θ_ball(x) = 1/2            θ_ball(x) = 0
                                  │
                                  ▼
                         【多面体棱边与顶点】
                       x 落在三角形交界的折线上
                       介质只占局部切向开锥体
                       Ω_in = 局部切锥立体角
                       θ_ball(x) = Ω_in / (4π)
```

于是，第二项被彻底算出来了，变成了一个纯代数项：
$$- \frac{4\pi G}{3}\,\theta_{\mathrm{ball}}(\mathbf{x})\rho_K(\mathbf{x})I.$$
这里 $\theta_{\mathrm{ball}}(\mathbf{x})=\Omega_{\mathrm{in}}(\mathbf{x})/(4\pi)$ **只定义一种以球对称平滑核逼近指示函数得到的边界取值约定**；内部为 $1$，外部为 $0$，平直光滑边界为 $1/2$。在棱顶为相应固角占比，但不能把它称作边界处唯一的点值。不同正则化约定可能不同，后面的有限部分若采用球形截断，必须沿用同一个约定。

**一侧极限检验（不可省略）**：半径 $a$、均匀密度 $\rho_0$ 的球体，总质量 $M=4\pi a^3\rho_0/3$。内部势 $U_{\rm in}(r)=GM(3a^2-r^2)/(2a^3)$，故 $H_{\rm in}=-GM I/a^3=-(4\pi G\rho_0/3)I$；外部势 $U_{\rm out}(r)=GM/r$，故在 $r=a$ 处 $H_{\rm out}=GM(3\mathbf n\mathbf n^T-I)/a^3$。两侧相减：
$$H_{\rm out}-H_{\rm in}=\frac{GM}{a^3}(3\mathbf n\mathbf n^T-I+I)=4\pi G\rho_0\mathbf n\mathbf n^T\ne0.$$
所以即便光滑表面 $C_{\mathbf x}=0$、球形主值有限，也**不能**推断经典 Hessian 在边界处存在唯一两侧极限。下文在边界书写 $H^{\rm FP}$ 时，一律指事先声明的球形排异有限部分，不能与 $H_{\rm in}$ 或 $H_{\rm out}$ 混同。

整套理论的第一座大山，至此被我们完全翻过！Hessian 严格分解为了：
$$H_K^{\mathrm{ball}}(\mathbf{x}) = G \,\mathrm{F.p.}_{\ell_0}^{\mathrm{ball}} \int_V \rho_K(\mathbf{y}) \frac{3\mathbf{r}\mathbf{r}^T - R^2 I}{R^5} d^3\mathbf{y} - \frac{4\pi G}{3} \theta_{\mathrm{ball}}(\mathbf{x}) \rho_K(\mathbf{x}) I.$$
其中内点的 $\mathrm{F.p.}$ 在角抵消时退化为球形 $\mathrm{p.v.}$；外点为普通积分；边界点是**约定的有限部分**，并非未经说明的经典 Hessian。

---

### 第二小节小结与确认

在本节中，我们恢复了这些关键数学工具的记忆：
1. **多维导数与链式法则**：为什么普通求导分式分母是 $R^5$，分子是 $3\mathbf{r}\mathbf{r}^T - R^2 I$；
2. **高斯散度定理与矛盾**：揭示了 $R=0$ 处不为零，而是具有 $-4\pi$ 的拓扑通量；
3. **施瓦茨分布与狄拉克函数**：平摊到 3 个维度产生 $-\frac{4\pi}{3}\delta(\mathbf{r})I$；
4. **柯西主值 $\mathrm{p.v.}$**：通过对称抠洞，保证发散分式能够合法积分；
5. **几何局部角 $\theta_{\mathrm{ball}}(\mathbf{x})$**：根据探测器在内部（1）、表面（1/2）、外部（0）还是棱角处，确定接触突变项的大小。

---

# 第三小节：球坐标系转换与射线空间参数化（球坐标射线化）

---

## 1. 为什么必须把三维体积分拆成“球面 + 射线”？

在第二小节结束时，我们已经把引力梯度张量（Hessian 矩阵）严格拆成了两部分：
在内点的球形对称排异下记为 $\mathrm{p.v.}$；在一般边界切锥处记为第六小节定义的球形 $\mathrm{F.p.}$：
$$H_K(\mathbf{x}) = G \,\mathrm{F.p.} \int_V \rho_K(\mathbf{y}) \frac{3\mathbf{r}\mathbf{r}^T - R^2 I}{R^5} d^3\mathbf{y} - \frac{4\pi G}{3} \theta_{\mathrm{ball}}(\mathbf{x}) \rho_K(\mathbf{x}) I$$

后面那项带单位矩阵 $I$ 的狄拉克脉冲项，我们上一小节已经完完整整算出来了：
$$- \frac{4\pi G}{3} \theta_{\mathrm{ball}}(\mathbf{x}) \rho_K(\mathbf{x}) I = \begin{pmatrix} -\frac{4\pi G}{3}\theta_{\mathrm{ball}}(\mathbf{x})\rho_K(\mathbf{x}) & 0 & 0 \\ 0 & -\frac{4\pi G}{3}\theta_{\mathrm{ball}}(\mathbf{x})\rho_K(\mathbf{x}) & 0 \\ 0 & 0 & -\frac{4\pi G}{3}\theta_{\mathrm{ball}}(\mathbf{x})\rho_K(\mathbf{x}) \end{pmatrix}$$
它是一个纯粹的常数对角矩阵，不需要再积任何分。

现在所有的拦路虎全部集中在前面这一个**三维体积分**上：
$$H_K^{\mathrm{nl}}(\mathbf{x}) = G \,\mathrm{F.p.} \int_V \rho_K(\mathbf{y}) \frac{3\mathbf{r}\mathbf{r}^T - R^2 I}{R^5} d^3\mathbf{y}$$

为什么这个积分在笛卡尔坐标系（$dx\,dy\,dz$）下直接算是自杀行为？
1. **积分域 $V$ 是个不规则多面体**：它的边界有成千上万个三角形平面，在笛卡尔坐标系下，积分的上限和下限根本写不出简单的函数关系；
2. **被积函数极度暴躁**：分母是 $R^5$，分子是二次多项式，当源点 $\mathbf{y}$ 靠近场点 $\mathbf{x}$ 时，$R \to 0$，被积函数按 $\frac{1}{R^3}$ 的速度急剧发散。

**破局思路**：
不要在长方体网格里死磕 $dx\,dy\,dz$。我们站在观测点 $\mathbf{x}$ 的视角，向四周环顾（看遍整整一个球面的所有方向），把整个三维天体看成**无数条从 $\mathbf{x}$ 打出去的射线**！
这样，一个无比艰难的**三维体积分**，就被精准切分成：
- 外层：在**二维单位球面**上扫过所有方向；
- 内层：在**一维射线**上沿着距离 $R$ 做直线积分。

---

## 2. 坐标转换的每一步展开（不省略、不跳步）

### (1) 坐标与相对位移向量
设观测点 $\mathbf{x}$ 的笛卡尔直角坐标为：
$$\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}$$
设天体内部积分源点 $\mathbf{y}$ 的笛卡尔直角坐标为：
$$\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ y_3 \end{pmatrix}$$
从源点 $\mathbf{y}$ 指向场点 $\mathbf{x}$ 的相对位移向量为：
$$\mathbf{r} = \mathbf{x} - \mathbf{y} = \begin{pmatrix} x_1 - y_1 \\ x_2 - y_2 \\ x_3 - y_3 \end{pmatrix} = \begin{pmatrix} r_1 \\ r_2 \\ r_3 \end{pmatrix}$$
相对位移的大小（欧氏几何距离）为非负实数 $R$：
$$R = \|\mathbf{r}\| = \sqrt{r_1^2 + r_2^2 + r_3^2}$$

### (2) 引入球面单位方向向量 $\mathbf{u}$
我们定义一个模长严格为 1 的三维单位方向向量 $\mathbf{u}$：
$$\mathbf{u} = \begin{pmatrix} u_1 \\ u_2 \\ u_3 \end{pmatrix} \in \mathbb{S}^2$$
所谓属于单位二维球面 $\mathbb{S}^2$，意味着它的三个分量必须严格满足平方和为 1：
$$\|\mathbf{u}\|^2 = u_1^2 + u_2^2 + u_3^2 = 1$$

既然 $\mathbf{u}$ 是纯方向（模长为 1），而 $R$ 是纯距离，那么相对位移向量 $\mathbf{r}$ 就可以完美分解为：
$$\mathbf{r} = R \mathbf{u} = R \begin{pmatrix} u_1 \\ u_2 \\ u_3 \end{pmatrix} = \begin{pmatrix} R u_1 \\ R u_2 \\ R u_3 \end{pmatrix}$$
将这三个分量逐一对应写出来：
$$r_1 = R u_1$$
$$r_2 = R u_2$$
$$r_3 = R u_3$$

### (3) 反解天体内部源点 $\mathbf{y}$
因为 $\mathbf{r} = \mathbf{x} - \mathbf{y}$，所以把 $\mathbf{y}$ 移到左边：
$$\mathbf{y} = \mathbf{x} - \mathbf{r} = \mathbf{x} - R \mathbf{u}$$
写成分量形式：
$$\begin{pmatrix} y_1 \\ y_2 \\ y_3 \end{pmatrix} = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} - R \begin{pmatrix} u_1 \\ u_2 \\ u_3 \end{pmatrix} = \begin{pmatrix} x_1 - R u_1 \\ x_2 - R u_2 \\ x_3 - R u_3 \end{pmatrix}$$
也就是：
$$y_1 = x_1 - R u_1$$
$$y_2 = x_2 - R u_2$$
$$y_3 = x_3 - R u_3$$

---

## 3. 三维体积元 $d^3\mathbf{y}$ 的雅可比行列式严格计算

在直角坐标系下，体积分微元就是长方体微元：
$$d^3\mathbf{y} = dy_1 \, dy_2 \, dy_3$$
转换到以 $\mathbf{x}$ 为中心的球坐标系下，方向 $\mathbf{u}$ 可以用天顶角（极角）$\theta \in [0, \pi]$ 和方位角 $\phi \in [0, 2\pi)$ 明确写出：
$$u_1 = \sin\theta \cos\phi$$
$$u_2 = \sin\theta \sin\phi$$
$$u_3 = \cos\theta$$

代入到 $\mathbf{y}$ 的表达式中：
$$y_1 = x_1 - R \sin\theta \cos\phi$$
$$y_2 = x_2 - R \sin\theta \sin\phi$$
$$y_3 = x_3 - R \cos\theta$$

根据多元微积分的换元定理，体积元变换由**雅可比矩阵（Jacobian Matrix）的行列式绝对值**决定：
$$dy_1 \, dy_2 \, dy_3 = |\det(J)| \, dR \, d\theta \, d\phi$$
我们把雅可比矩阵 $J$ 的 9 个偏导数项一个不落全部算出来：
$$J = \begin{pmatrix} \frac{\partial y_1}{\partial R} & \frac{\partial y_1}{\partial \theta} & \frac{\partial y_1}{\partial \phi} \\ \frac{\partial y_2}{\partial R} & \frac{\partial y_2}{\partial \theta} & \frac{\partial y_2}{\partial \phi} \\ \frac{\partial y_3}{\partial R} & \frac{\partial y_3}{\partial \theta} & \frac{\partial y_3}{\partial \phi} \end{pmatrix} = \begin{pmatrix} -\sin\theta \cos\phi & -R \cos\theta \cos\phi & R \sin\theta \sin\phi \\ -\sin\theta \sin\phi & -R \cos\theta \sin\phi & -R \sin\theta \cos\phi \\ -\cos\theta & R \sin\theta & 0 \end{pmatrix}$$

我们沿着第三行展开求行列式：
$$\det(J) = (-\cos\theta) \cdot \begin{vmatrix} -R \cos\theta \cos\phi & R \sin\theta \sin\phi \\ -R \cos\theta \sin\phi & -R \sin\theta \cos\phi \end{vmatrix} - (R \sin\theta) \cdot \begin{vmatrix} -\sin\theta \cos\phi & R \sin\theta \sin\phi \\ -\sin\theta \sin\phi & -R \sin\theta \cos\phi \end{vmatrix} + 0$$

计算第一个 $2 \times 2$ 行列式：
$$\begin{vmatrix} -R \cos\theta \cos\phi & R \sin\theta \sin\phi \\ -R \cos\theta \sin\phi & -R \sin\theta \cos\phi \end{vmatrix} = (-R \cos\theta \cos\phi)(-R \sin\theta \cos\phi) - (R \sin\theta \sin\phi)(-R \cos\theta \sin\phi)$$
$$= R^2 \sin\theta \cos\theta \cos^2\phi + R^2 \sin\theta \cos\theta \sin^2\phi = R^2 \sin\theta \cos\theta (\cos^2\phi + \sin^2\phi) = R^2 \sin\theta \cos\theta$$

计算第二个 $2 \times 2$ 行列式：
$$\begin{vmatrix} -\sin\theta \cos\phi & R \sin\theta \sin\phi \\ -\sin\theta \sin\phi & -R \sin\theta \cos\phi \end{vmatrix} = (-\sin\theta \cos\phi)(-R \sin\theta \cos\phi) - (R \sin\theta \sin\phi)(-\sin\theta \sin\phi)$$
$$= R \sin^2\theta \cos^2\phi + R \sin^2\theta \sin^2\phi = R \sin^2\theta (\cos^2\phi + \sin^2\phi) = R \sin^2\theta$$

把两个结果拼回总行列式：
$$\det(J) = (-\cos\theta)(R^2 \sin\theta \cos\theta) - (R \sin\theta)(R \sin^2\theta) = -R^2 \sin\theta \cos^2\theta - R^2 \sin^3\theta$$
提取公因式 $-R^2 \sin\theta$：
$$\det(J) = -R^2 \sin\theta (\cos^2\theta + \sin^2\theta) = -R^2 \sin\theta$$
取绝对值：
$$|\det(J)| = R^2 \sin\theta$$

因为单位球面的微分立体角定义就是：
$$d\Omega = \sin\theta \, d\theta \, d\phi$$
所以三维体积元完完整整地推导为：
$$d^3\mathbf{y} = R^2 \, dR \, d\Omega$$

---

## 4. 无迹方向张量 $T(\mathbf{u})$ 的完整九个分量展开

现在看被积函数里的分子矩阵：
$$3\mathbf{r}\mathbf{r}^T - R^2 I$$
我们把 $\mathbf{r} = R\mathbf{u}$ 代进去：
$$3\mathbf{r}\mathbf{r}^T - R^2 I = 3(R\mathbf{u})(R\mathbf{u})^T - R^2 I = 3 R^2 \mathbf{u}\mathbf{u}^T - R^2 I$$
提取标量公因子 $R^2$：
$$= R^2 (3\mathbf{u}\mathbf{u}^T - I)$$

这里出现了一个只与方向 $\mathbf{u}$ 有关、与距离 $R$ 完全无关的矩阵，我们把它正式定义为**无迹方向张量 $T(\mathbf{u})$**：
$$T(\mathbf{u}) = 3\mathbf{u}\mathbf{u}^T - I$$

我们把 $T(\mathbf{u})$ 的 9 个分量每一个都清清楚楚写出来：
1. $\mathbf{u}\mathbf{u}^T$ 矩阵展开：
   $$\mathbf{u}\mathbf{u}^T = \begin{pmatrix} u_1 \\ u_2 \\ u_3 \end{pmatrix} \begin{pmatrix} u_1 & u_2 & u_3 \end{pmatrix} = \begin{pmatrix} u_1^2 & u_1 u_2 & u_1 u_3 \\ u_2 u_1 & u_2^2 & u_2 u_3 \\ u_3 u_1 & u_3 u_2 & u_3^2 \end{pmatrix}$$
2. 乘以 3：
   $$3\mathbf{u}\mathbf{u}^T = \begin{pmatrix} 3u_1^2 & 3u_1 u_2 & 3u_1 u_3 \\ 3u_2 u_1 & 3u_2^2 & 3u_2 u_3 \\ 3u_3 u_1 & 3u_3 u_2 & 3u_3^2 \end{pmatrix}$$
3. 减去单位矩阵 $I$：
   $$T(\mathbf{u}) = \begin{pmatrix} 3u_1^2 & 3u_1 u_2 & 3u_1 u_3 \\ 3u_2 u_1 & 3u_2^2 & 3u_2 u_3 \\ 3u_3 u_1 & 3u_3 u_2 & 3u_3^2 \end{pmatrix} - \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} = \begin{pmatrix} 3u_1^2 - 1 & 3u_1 u_2 & 3u_1 u_3 \\ 3u_2 u_1 & 3u_2^2 - 1 & 3u_2 u_3 \\ 3u_3 u_1 & 3u_3 u_2 & 3u_3^2 - 1 \end{pmatrix}$$

我们再来算一遍它的迹（Trace，对角线相加）：
$$\mathrm{Tr}(T(\mathbf{u})) = (3u_1^2 - 1) + (3u_2^2 - 1) + (3u_3^2 - 1) = 3(u_1^2 + u_2^2 + u_3^2) - (1 + 1 + 1)$$
因为 $\mathbf{u}$ 是单位矢量，$u_1^2 + u_2^2 + u_3^2 = 1$，所以：
$$\mathrm{Tr}(T(\mathbf{u})) = 3(1) - 3 = 0$$
**迹永远为零！** 这就是为什么它被称为“无迹张量”。

---

## 5. 奇迹时刻：被积核乘以体积元后的极端降幂

现在我们把几何矩阵除以分母 $R^5$：
$$\frac{3\mathbf{r}\mathbf{r}^T - R^2 I}{R^5} = \frac{R^2 T(\mathbf{u})}{R^5} = \frac{T(\mathbf{u})}{R^3}$$
看，分母原本是 $R^5$，由于分子的 $R^2$ 提出来，瞬间降到了 $R^3$！

再乘上三维体积微元 $d^3\mathbf{y} = R^2 \, dR \, d\Omega$：
$$\frac{3\mathbf{r}\mathbf{r}^T - R^2 I}{R^5} d^3\mathbf{y} = \frac{T(\mathbf{u})}{R^3} \cdot \left( R^2 \, dR \, d\Omega \right) = \frac{T(\mathbf{u})}{R} \, dR \, d\Omega$$

请停下来体会这一步的震撼之处：
原本是令所有人头疼的超强奇异分式 $\frac{1}{R^5}$，在球坐标系下乘上微元面积的几何膨胀雅可比 $R^2$ 之后，在径向距离 $R$ 上，竟然**只剩下了最最普通的 $\frac{1}{R}$（一次反比）！**
空间方向矩阵 $T(\mathbf{u})$ 和径向变量 $R$ 彻底实现了解耦！

---

## 6. 射线与天体的几何交集集合 $E_\mathbf{x}(\mathbf{u})$

我们固定一个单位方向 $\mathbf{u}$。
从场点 $\mathbf{x}$ 出发，沿着 $-\mathbf{u}$ 的方向射出一条虚拟射线（因为相对位移是 $\mathbf{y} = \mathbf{x} - R\mathbf{u}$）。
定义该方向上的射线与天体实体 $V$ 的交集线段集合为 $E_\mathbf{x}(\mathbf{u})$：
$$E_\mathbf{x}(\mathbf{u}) = \{ R > 0 : \mathbf{x} - R\mathbf{u} \in V \}$$

这里产生至关重要的**拓扑分类**：

### 情况 1：天体是凸多面体（比如椭球、标准多面体）
若 $V$ 是凸集，则任意直线与 $V$ 的交集是空集、一个点或一个闭区间；因此，在避开相切方向的几乎处处意义下，该方向上的实体距离区间至多只有一段：
$$E_\mathbf{x}(\mathbf{u}) = [R^-(\mathbf{u}), \; R^+(\mathbf{u})]$$
- $R^-(\mathbf{u})$：光线打进天体表面的距离（入口距离）；
- $R^+(\mathbf{u})$：光线穿出天体表面的距离（出口距离）。

### 情况 2：天体是非凸的复杂多面体（比如真实的龙宫小行星、贝努小行星）
小行星表面有巨石突出、有大坑洼、甚至是哑铃状的双叶天体。一束光线打过去，可能会“穿进 $\to$ 穿出到坑洞虚空 $\to$ 再次穿进岩石 $\to$ 再次穿出”。
因此，在方向 $\mathbf{u}$ 上，射线穿过的实体线段是由**有限个互不相交的闭区间并集**构成的：
$$E_\mathbf{x}(\mathbf{u}) = [R_1^-, R_1^+] \cup [R_2^-, R_2^+] \cup \dots \cup [R_{L(\mathbf{u})}^-, R_{L(\mathbf{u})}^+] = \bigcup_{\ell=1}^{L(\mathbf{u})} \left[ R_\ell^-(\mathbf{u}), \; R_\ell^+(\mathbf{u}) \right]$$
式中：
- $L(\mathbf{u})$ 是该射线上穿透实体的总段数；
- 第 $\ell$ 段的入口距离记作 $R_\ell^-(\mathbf{u})$；
- 第 $\ell$ 段的出口距离记作 $R_\ell^+(\mathbf{u})$。

---

## 7. 代入广义 Cauchy 密度核：完整的双重积分表达

现在我们把第一小节的广义 Cauchy 密度分布也带进来。
密度函数在源点 $\mathbf{y} = \mathbf{x} - R\mathbf{u}$ 处的值为：
$$\rho_K(\mathbf{x} - R\mathbf{u}) = \sum_{k=1}^K w_k \phi_k(\mathbf{x} - R\mathbf{u}) = \sum_{k=1}^K w_k \left[ 1 + \sigma_k^2 \|\mathbf{x} - R\mathbf{u} - \mathbf{c}_k\|^2 \right]^{-\alpha_k}$$

把它代入我们刚才降幂化简好的积分公式中，主值积分项完整重写为：
$$H_K^{\mathrm{nl}}(\mathbf{x}) = G \sum_{k=1}^K w_k \,\mathrm{F.p.} \int_{\mathbb{S}^2} T(\mathbf{u}) \left[ \int_{E_\mathbf{x}(\mathbf{u})} \frac{\left[ 1 + \sigma_k^2 \|\mathbf{x} - R\mathbf{u} - \mathbf{c}_k\|^2 \right]^{-\alpha_k}}{R} \, dR \right] d\Omega$$

最后，按第二小节声明的球形边界取值约定加入局部项，得到小天体引力梯度在全宇宙在该有限部分约定下的**球坐标射线化表达公式**：

$$H_K^{\mathrm{ball}}(\mathbf{x}) = G \sum_{k=1}^K w_k \,\mathrm{F.p.} \int_{\mathbb{S}^2} \begin{pmatrix} 3u_1^2 - 1 & 3u_1 u_2 & 3u_1 u_3 \\ 3u_2 u_1 & 3u_2^2 - 1 & 3u_2 u_3 \\ 3u_3 u_1 & 3u_3 u_2 & 3u_3^2 - 1 \end{pmatrix} \left[ \sum_{\ell=1}^{L(\mathbf{u})} \int_{R_\ell^-(\mathbf{u})}^{R_\ell^+(\mathbf{u})} \frac{\left[ 1 + \sigma_k^2 \|\mathbf{x} - R\mathbf{u} - \mathbf{c}_k\|^2 \right]^{-\alpha_k}}{R} \, dR \right] d\Omega - \begin{pmatrix} \frac{4\pi G}{3}\theta_{\mathrm{ball}}(\mathbf{x})\rho_K(\mathbf{x}) & 0 & 0 \\ 0 & \frac{4\pi G}{3}\theta_{\mathrm{ball}}(\mathbf{x})\rho_K(\mathbf{x}) & 0 \\ 0 & 0 & \frac{4\pi G}{3}\theta_{\mathrm{ball}}(\mathbf{x})\rho_K(\mathbf{x}) \end{pmatrix}$$

---

### 第三小节小结与确认

看，在本节中我们：
1. 完整用雅可比行列式算出了 $d^3\mathbf{y} = R^2 dR d\Omega$，中间没有任何省略；
2. 把方向张量 $T(\mathbf{u})$ 的 9 个分量每一个都写出来了，并再次验证了它的迹为 0；
3. 展示了被积核分母为什么从 $R^5$ 奇迹般降到了 $\frac{1}{R}$；
4. 详尽分类了射线穿过凸体和非凸天体时的区间并集形式；
5. 把整体公式一字不漏、完全展开拼成了一个大合体。

---

# 第四小节：四元数空间重写与径向一元二次配方

---

## 1. 为什么必须在这里引入四元数？

在第三小节的最后，我们把引力梯度化简成了球面积分套着内层径向积分：
$$H_K^{\mathrm{nl}}(\mathbf{x}) = G \sum_{k=1}^K w_k \,\mathrm{F.p.} \int_{\mathbb{S}^2} T(\mathbf{u}) \left[ \sum_{\ell=1}^{L(\mathbf{u})} \int_{R_\ell^-(\mathbf{u})}^{R_\ell^+(\mathbf{u})} \frac{\left[ 1 + \sigma_k^2 \|\mathbf{x} - R\mathbf{u} - \mathbf{c}_k\|^2 \right]^{-\alpha_k}}{R} \, dR \right] d\Omega$$

请大家把目光死死锁定在内层积分分母中的这个几何项：
$$\|\mathbf{x} - R\mathbf{u} - \mathbf{c}_k\|^2$$

这里面混杂了：
- 场点坐标 $\mathbf{x} = (x_1, x_2, x_3)^T$；
- 第 $k$ 个 Cauchy 密度核中心 $\mathbf{c}_k = (c_{k,1}, c_{k,2}, c_{k,3})^T$；
- 球面单位方向向量 $\mathbf{u} = (u_1, u_2, u_3)^T$；
- 径向积分标量距离 $R$。

如果我们直接在三维直角坐标系下把这个三维范数平方硬拆开：
$$\|\mathbf{x} - R\mathbf{u} - \mathbf{c}_k\|^2 = (x_1 - R u_1 - c_{k,1})^2 + (x_2 - R u_2 - c_{k,2})^2 + (x_3 - R u_3 - c_{k,3})^2$$
把它平方展开：
$$= \sum_{i=1}^3 \left[ (x_i - c_{k,i})^2 - 2R(x_i - c_{k,i})u_i + R^2 u_i^2 \right]$$
项数非常多、极其冗长，而且后续我们还要做复变量代换、找代数曲线、做分部积分。如果带着这一大坨三维坐标去推导，整个数学结构会混乱到根本无法看清本质。

**数学家的妙招**：
借用**哈密顿四元数（Hamilton Quaternions）**这一强大的代数工具，把三维空间的平移、旋转、尺度和距离，全部统一包装进一个四元数模长中！

---

## 2. 四元数数学基础全面扫盲（从零复习，完全展开）

### (1) 四元数的定义
四元数集合记作 $\mathbb{H}$。任意一个四元数 $q \in \mathbb{H}$ 由一个实数标量和三个互相垂直的虚数基元组成：
$$q = q_0 + q_1 \mathbf{i} + q_2 \mathbf{j} + q_3 \mathbf{k}$$
式中：
- $q_0 \in \mathbb{R}$ 称为**实部（标量部分）**，记作 $\mathrm{Sc}(q)$；
- $q_1, q_2, q_3 \in \mathbb{R}$ 构成**虚部（矢量部分）**，记作 $\mathrm{Vec}(q) = q_1 \mathbf{i} + q_2 \mathbf{j} + q_3 \mathbf{k}$。

### (2) 四元数的三条核心乘法公理
1843 年哈密顿刻在爱尔兰布鲁姆桥上的划时代乘法公理：
$$\mathbf{i}^2 = -1$$
$$\mathbf{j}^2 = -1$$
$$\mathbf{k}^2 = -1$$
$$\mathbf{i} \mathbf{j} \mathbf{k} = -1$$

由上面四条公理，可以立刻严格推导出基元之间的交叉乘法规则：
- **正向循环乘积（右手螺旋）**：
  $$\mathbf{i} \mathbf{j} = \mathbf{k}$$
  $$\mathbf{j} \mathbf{k} = \mathbf{i}$$
  $$\mathbf{k} \mathbf{i} = \mathbf{j}$$
- **反向循环乘积（反对称反对易性）**：
  $$\mathbf{j} \mathbf{i} = -\mathbf{k}$$
  $$\mathbf{k} \mathbf{j} = -\mathbf{i}$$
  $$\mathbf{i} \mathbf{k} = -\mathbf{j}$$

### (3) 四元数的共轭与模长平方
定义四元数 $q$ 的**四元数共轭** $\bar{q}$ 为虚部全部取反：
$$\bar{q} = q_0 - q_1 \mathbf{i} - q_2 \mathbf{j} - q_3 \mathbf{k}$$
现在我们把 $q$ 和它的共轭 $\bar{q}$ 严格乘一次，看看它的模长平方是怎么来的：
$$q \bar{q} = (q_0 + q_1 \mathbf{i} + q_2 \mathbf{j} + q_3 \mathbf{k})(q_0 - q_1 \mathbf{i} - q_2 \mathbf{j} - q_3 \mathbf{k})$$
把每一项用乘法分配律展开：
$$= q_0^2 - q_0 q_1 \mathbf{i} - q_0 q_2 \mathbf{j} - q_0 q_3 \mathbf{k} + q_1 q_0 \mathbf{i} - q_1^2 \mathbf{i}^2 - q_1 q_2 \mathbf{i}\mathbf{j} - q_1 q_3 \mathbf{i}\mathbf{k} + q_2 q_0 \mathbf{j} - q_2 q_1 \mathbf{j}\mathbf{i} - q_2^2 \mathbf{j}^2 - q_2 q_3 \mathbf{j}\mathbf{k} + q_3 q_0 \mathbf{k} - q_3 q_1 \mathbf{k}\mathbf{i} - q_3 q_2 \mathbf{k}\mathbf{j} - q_3^2 \mathbf{k}^2$$
我们把虚数基元代换：
- $-\mathbf{i}^2 = -(-1) = 1$，$-\mathbf{j}^2 = 1$，$-\mathbf{k}^2 = 1$；
- 交叉项互相抵消：比如 $-q_1 q_2 \mathbf{i}\mathbf{j} - q_2 q_1 \mathbf{j}\mathbf{i} = -q_1 q_2 \mathbf{k} - q_2 q_1(-\mathbf{k}) = -q_1 q_2 \mathbf{k} + q_1 q_2 \mathbf{k} = 0$。

所有的交叉虚数项全部奇迹般归零，只剩下实数的平方和：
$$|q|^2 = q \bar{q} = q_0^2 + q_1^2 + q_2^2 + q_3^2$$
这是一个绝对非负的实数，其平方根 $|q| = \sqrt{q_0^2 + q_1^2 + q_2^2 + q_3^2}$ 就是四元数的范数（模长）。

---

## 3. 纯虚四元数与三维向量的无缝融合

所谓**纯虚四元数（Pure Imaginary Quaternion）**，就是实部为零的四元数（$q_0 = 0$）：
$$q = q_1 \mathbf{i} + q_2 \mathbf{j} + q_3 \mathbf{k}$$
三维欧氏空间中的任何一个几何向量 $\mathbf{v} = (v_1, v_2, v_3)^T$，都可以严格一对一嵌入纯虚四元数代数空间 $\mathbb{H}_0$：
$$\mathbf{v} = v_1 \mathbf{i} + v_2 \mathbf{j} + v_3 \mathbf{k}$$

### 两个三维向量作为四元数相乘，会发生什么？
设有两个三维向量 $\mathbf{p} = p_1 \mathbf{i} + p_2 \mathbf{j} + p_3 \mathbf{k}$ 和 $\mathbf{u} = u_1 \mathbf{i} + u_2 \mathbf{j} + u_3 \mathbf{k}$。
我们把它们按照四元数乘法直接相乘，一项一项展开：
$$\mathbf{p} \mathbf{u} = (p_1 \mathbf{i} + p_2 \mathbf{j} + p_3 \mathbf{k})(u_1 \mathbf{i} + u_2 \mathbf{j} + u_3 \mathbf{k})$$
$$= p_1 u_1 \mathbf{i}^2 + p_1 u_2 \mathbf{i}\mathbf{j} + p_1 u_3 \mathbf{i}\mathbf{k} + p_2 u_1 \mathbf{j}\mathbf{i} + p_2 u_2 \mathbf{j}^2 + p_2 u_3 \mathbf{j}\mathbf{k} + p_3 u_1 \mathbf{k}\mathbf{i} + p_3 u_2 \mathbf{k}\mathbf{j} + p_3 u_3 \mathbf{k}^2$$

把基元乘法代入进去：
- $\mathbf{i}^2 = \mathbf{j}^2 = \mathbf{k}^2 = -1$；
- $\mathbf{i}\mathbf{j} = \mathbf{k}$，$\mathbf{j}\mathbf{i} = -\mathbf{k}$；
- $\mathbf{j}\mathbf{k} = \mathbf{i}$，$\mathbf{k}\mathbf{j} = -\mathbf{i}$；
- $\mathbf{k}\mathbf{i} = \mathbf{j}$，$\mathbf{i}\mathbf{k} = -\mathbf{j}$。

把实部和虚部拆开归类：
$$\mathbf{p} \mathbf{u} = -(p_1 u_1 + p_2 u_2 + p_3 u_3) + (p_2 u_3 - p_3 u_2) \mathbf{i} + (p_3 u_1 - p_1 u_3) \mathbf{j} + (p_1 u_2 - p_2 u_1) \mathbf{k}$$

仔细看这两部分：
1. 前面的实部：$-(p_1 u_1 + p_2 u_2 + p_3 u_3)$ 正好就是三维向量的**点积（内积）**取负号：$-\mathbf{p} \cdot \mathbf{u}$！
2. 后面的虚部：三个分量正好是三维向量的**叉积（外积）**：$\mathbf{p} \times \mathbf{u}$！

所以，我们得到了四元数代数中最核心、最美妙的**基本乘法恒等式**：
$$\mathbf{p} \mathbf{u} = - \mathbf{p} \cdot \mathbf{u} + \mathbf{p} \times \mathbf{u}$$

同理，调换相乘顺序（因为点积对称，叉积反对称）：
$$\mathbf{u} \mathbf{p} = - \mathbf{u} \cdot \mathbf{p} + \mathbf{u} \times \mathbf{p} = - \mathbf{p} \cdot \mathbf{u} - \mathbf{p} \times \mathbf{u}$$

两式相加与相减，我们就能够把经典的点积和叉积，用纯粹的四元数代数表示：
- **点积（投影标量）**：
  $$\mathbf{p} \cdot \mathbf{u} = - \frac{1}{2}(\mathbf{p}\mathbf{u} + \mathbf{u}\mathbf{p})$$
- **叉积（力臂矢量）**：
  $$\mathbf{p} \times \mathbf{u} = \frac{1}{2}(\mathbf{p}\mathbf{u} - \mathbf{u}\mathbf{p})$$

---

## 4. 构造带尺度的特征四元数 $Q_k(R, \mathbf{u})$

现在正式把这套工具装配到我们的引力物理问题上。

### (1) 定义物理几何量
对于观测场点 $\mathbf{x}$ 和第 $k$ 个 Cauchy 密度核中心 $\mathbf{c}_k$，定义相对位移纯虚四元数：
$$\mathbf{p}_k = \mathbf{x} - \mathbf{c}_k = (x_1 - c_{k,1}) \mathbf{i} + (x_2 - c_{k,2}) \mathbf{j} + (x_3 - c_{k,3}) \mathbf{k}$$
定义第 $k$ 个核的**特征空间尺度**为逆尺度 $\sigma_k$ 的倒数（它是一个具有长度量纲的正常正实数）：
$$a_k = \frac{1}{\sigma_k} > 0 \iff \sigma_k = \frac{1}{a_k}$$

### (2) 构造四元数 $Q_k(R, \mathbf{u})$
我们构造一个实部为 $a_k$、虚部为空间位移的完整非纯四元数：
$$Q_k(R, \mathbf{u}) = a_k + \mathbf{p}_k - R\mathbf{u}$$
把它的实部和虚部完完整整写出来：
- 实部：$a_k$
- 虚部：$\mathbf{p}_k - R\mathbf{u} = \left[ (x_1 - c_{k,1}) - R u_1 \right] \mathbf{i} + \left[ (x_2 - c_{k,2}) - R u_2 \right] \mathbf{j} + \left[ (x_3 - c_{k,3}) - R u_3 \right] \mathbf{k}$

### (3) 计算它的模长平方
根据四元数模长平方的定义（实部平方 + 虚部各分量平方）：
$$|Q_k(R, \mathbf{u})|^2 = a_k^2 + \|\mathbf{p}_k - R\mathbf{u}\|^2$$

现在我们对比一下 Cauchy 密度核分母里的几何因子：
$$1 + \sigma_k^2 \|\mathbf{p}_k - R\mathbf{u}\|^2$$
提取公因子 $\sigma_k^2$：
$$= \sigma_k^2 \left( \frac{1}{\sigma_k^2} + \|\mathbf{p}_k - R\mathbf{u}\|^2 \right)$$
因为 $a_k = \frac{1}{\sigma_k}$，所以 $\frac{1}{\sigma_k^2} = a_k^2$，代进去：
$$= \sigma_k^2 \left( a_k^2 + \|\mathbf{p}_k - R\mathbf{u}\|^2 \right) = \sigma_k^2 |Q_k(R, \mathbf{u})|^2$$

两边取 $-\alpha_k$ 次方：
$$\left[ 1 + \sigma_k^2 \|\mathbf{p}_k - R\mathbf{u}\|^2 \right]^{-\alpha_k} = \left( \sigma_k^2 |Q_k(R, \mathbf{u})|^2 \right)^{-\alpha_k} = \sigma_k^{-2\alpha_k} |Q_k(R, \mathbf{u})|^{-2\alpha_k}$$
**大功告成！** 极其复杂的柯西密度核，在四元数框架下，完全变成了**四元数模长的单一纯代数幂次**！

---

## 5. 径向距离 $R$ 的一元二次配方（一步一步算）

虽然化成了四元数模长，但在做积分时，积分变量是实数标量距离 $R$。我们必须把模长平方 $|Q_k(R, \mathbf{u})|^2$ 还原成关于 $R$ 的标量函数。

我们来看虚部范数平方：
$$\|\mathbf{p}_k - R\mathbf{u}\|^2$$
利用向量内积的分配律展开：
$$\|\mathbf{p}_k - R\mathbf{u}\|^2 = (\mathbf{p}_k - R\mathbf{u}) \cdot (\mathbf{p}_k - R\mathbf{u})$$
$$= \mathbf{p}_k \cdot \mathbf{p}_k - 2R(\mathbf{p}_k \cdot \mathbf{u}) + R^2 (\mathbf{u} \cdot \mathbf{u})$$
因为 $\mathbf{u}$ 是单位矢量，$\mathbf{u} \cdot \mathbf{u} = \|\mathbf{u}\|^2 = 1$，$\mathbf{p}_k \cdot \mathbf{p}_k = \|\mathbf{p}_k\|^2$，所以：
$$\|\mathbf{p}_k - R\mathbf{u}\|^2 = R^2 - 2(\mathbf{p}_k \cdot \mathbf{u}) R + \|\mathbf{p}_k\|^2$$

为了简化这个式子，我们定义两个只与角度方向 $\mathbf{u}$ 有关的几何物理量：

### (1) 纵向投影标量 $b_k(\mathbf{u})$
定义相对位移在射线方向 $\mathbf{u}$ 上的正交投影长度为：
$$b_k(\mathbf{u}) = \mathbf{p}_k \cdot \mathbf{u} = (x_1 - c_{k,1})u_1 + (x_2 - c_{k,2})u_2 + (x_3 - c_{k,3})u_3$$
用四元数表示就是：
$$b_k(\mathbf{u}) = -\frac{1}{2}(\mathbf{p}_k \mathbf{u} + \mathbf{u} \mathbf{p}_k)$$
> **物理意义**：当射线打出去时，$R = b_k(\mathbf{u})$ 代表射线上距离第 $k$ 个质量核中心**几何距离最近的那一个点**。

### (2) 横向力臂向量 $\mathbf{c}_k(\mathbf{u})$
定义相对位移与射线方向的叉积矢量为：
$$\mathbf{c}_k(\mathbf{u}) = \mathbf{p}_k \times \mathbf{u} = \begin{pmatrix} (x_2 - c_{k,2})u_3 - (x_3 - c_{k,3})u_2 \\ (x_3 - c_{k,3})u_1 - (x_1 - c_{k,1})u_3 \\ (x_1 - c_{k,1})u_2 - (x_2 - c_{k,2})u_1 \end{pmatrix}$$
用四元数表示就是：
$$\mathbf{c}_k(\mathbf{u}) = \frac{1}{2}(\mathbf{p}_k \mathbf{u} - \mathbf{u} \mathbf{p}_k)$$
根据勾股定理（Lagrange 恒等式）：
$$\|\mathbf{c}_k(\mathbf{u})\|^2 = \|\mathbf{p}_k \times \mathbf{u}\|^2 = \|\mathbf{p}_k\|^2 \|\mathbf{u}\|^2 \sin^2\theta = \|\mathbf{p}_k\|^2 (1 - \cos^2\theta) = \|\mathbf{p}_k\|^2 - (\mathbf{p}_k \cdot \mathbf{u})^2$$
因为 $\mathbf{p}_k \cdot \mathbf{u} = b_k(\mathbf{u})$，所以得到极简的勾股关系：
$$\|\mathbf{c}_k(\mathbf{u})\|^2 = \|\mathbf{p}_k\|^2 - b_k^2(\mathbf{u}) \iff \|\mathbf{p}_k\|^2 = b_k^2(\mathbf{u}) + \|\mathbf{c}_k(\mathbf{u})\|^2$$
> **物理意义**：$\|\mathbf{c}_k(\mathbf{u})\|$ 就是射线到核中心 $\mathbf{c}_k$ 的**最近垂直几何距离（碰撞参数/离距）**。

---

## 6. 一元二次多项式的终极完全配方

把上面勾股定理反解出的 $\|\mathbf{p}_k\|^2 = b_k^2(\mathbf{u}) + \|\mathbf{c}_k(\mathbf{u})\|^2$ 代回刚才的展开式：
$$\|\mathbf{p}_k - R\mathbf{u}\|^2 = R^2 - 2 b_k(\mathbf{u}) R + \|\mathbf{p}_k\|^2$$
$$= R^2 - 2 b_k(\mathbf{u}) R + b_k^2(\mathbf{u}) + \|\mathbf{c}_k(\mathbf{u})\|^2$$
前三项构成一个完全平方公式：
$$R^2 - 2 b_k(\mathbf{u}) R + b_k^2(\mathbf{u}) = (R - b_k(\mathbf{u}))^2$$
代进去：
$$\|\mathbf{p}_k - R\mathbf{u}\|^2 = (R - b_k(\mathbf{u}))^2 + \|\mathbf{c}_k(\mathbf{u})\|^2$$

现在算总的四元数模长平方 $|Q_k(R, \mathbf{u})|^2$：
$$|Q_k(R, \mathbf{u})|^2 = a_k^2 + \|\mathbf{p}_k - R\mathbf{u}\|^2 = a_k^2 + (R - b_k(\mathbf{u}))^2 + \|\mathbf{c}_k(\mathbf{u})\|^2$$
我们把不带积分变量 $R$ 的两个常数项加在一起，定义为**合成横向有效尺度平方 $d_k^2(\mathbf{u})$**：
$$d_k^2(\mathbf{u}) = a_k^2 + \|\mathbf{c}_k(\mathbf{u})\|^2 = a_k^2 + \|\mathbf{p}_k\|^2 - b_k^2(\mathbf{u})$$

把 $d_k^2(\mathbf{u})$ 代入模长平方，最终得到震惊全场的极简形式：
$$|Q_k(R, \mathbf{u})|^2 = (R - b_k(\mathbf{u}))^2 + d_k^2(\mathbf{u})$$

---

## 7. 径向核心标量积分 $J_k(E; \mathbf{u})$

我们把上述全部化简成果代入原本令人望而生畏的径向积分中。
定义单条射线上关于第 $k$ 个核的**径向响应函数**为：
$$J_k(E; \mathbf{u}) = \sigma_k^{-2\alpha_k} \,\mathrm{F.p.} \int_E \frac{dR}{R \left[ (R - b_k(\mathbf{u}))^2 + d_k^2(\mathbf{u}) \right]^{\alpha_k}}$$

我们来审视这个最终得到的积分：
1. 空间三维向量被彻底粉碎了，被积函数分母只剩下一个最经典的高等数学一元二次函数：
   $$(R - b_k)^2 + d_k^2$$
2. $b_k(\mathbf{u})$ 决定了二次抛物线对称轴的位置（沿射线的投影距离）；
3. $d_k(\mathbf{u}) > 0$ 决定了二次多项式永远不会等于零（因为即使垂直距离为零，还有内在尺度 $a_k = 1/\sigma_k > 0$ 在兜底），保证分母绝对没有二次虚根落在实轴上！

---

### 第四小节小结与确认

在本节中，我们按照最高标准：
1. 详细推导了四元数代数乘法公理，一步一步算出了为什么 $\mathbf{p}\mathbf{u} = -\mathbf{p}\cdot\mathbf{u} + \mathbf{p}\times\mathbf{u}$；
2. 构造了实部为 $a_k$、虚部为三维位移的特征四元数 $Q_k(R, \mathbf{u})$；
3. 用初中数学配方法和高中向量勾股定理，把三维范数平方完全配方为 $(R - b_k)^2 + \|\mathbf{c}_k\|^2$；
4. 最终证明了柯西密度核的分母在射线上就是一个标准的一元二次多项式 $(R - b_k)^2 + d_k^2$。

---

# 第五小节：多面体几何分片、平面求交与球面可见性分解

---

## 1. 为什么必须进行多面体几何分片？

在上一小节中，我们通过四元数配方法，把第 $k$ 个核在单条射线上沿距离 $R$ 的标量积分完全搞定了：
$$J_k(E; \mathbf{u}) = \sigma_k^{-2\alpha_k} \,\mathrm{F.p.} \int_E \frac{dR}{R \left[ (R - b_k(\mathbf{u}))^2 + d_k^2(\mathbf{u}) \right]^{\alpha_k}}$$

但是，请注意积分区间 $E = E_\mathbf{x}(\mathbf{u})$！
在三维空间中，小行星的边界 $\partial V$ 是由成百上千个微小的三角形平面拼成的网格。
当观测点 $\mathbf{x}$ 固定时，如果我们向四周任意一个球面方向 $\mathbf{u}$ 打出一条射线：
- 这条射线到底打穿了哪些三角形？
- 它从哪个三角形钻进天体内部（入口距离 $R^-$ 是多少）？
- 它又从哪个三角形钻出天体进入虚空（出口距离 $R^+$ 是多少）？
- 随着方向 $\mathbf{u}$ 在球面上连续扫动，穿透的三角形何时发生更换？

本小节的任务，是推导三角形平面的射线交点距离公式，并在避开临界方向的意义下把单位球面剖分成有限个组合类型恒定的几何单元。

---

## 2. 三角多面体表面网格的严格数学定义

设小行星的几何表面由 $F$ 个平面三角形面片拼接封闭而成：
$$\partial V=\bigcup_{f=1}^F\Delta_f.$$
三角面片构成的是二维边界，不是三维体积。本节假设网格闭合、一致定向、无自相交；否则必须重新定义其内部和穿入/穿出。
对于第 $f$ 个三角形平面面片 $\Delta_f$（其中索引 $f \in \{1, 2, \dots, F\}$），它在三维直角坐标系中由三个按右手定则逆时针排列的空间顶点坐标确定：
- 顶点 0：$\mathbf{v}_{f,0} = \begin{pmatrix} v_{f,0,1} \\ v_{f,0,2} \\ v_{f,0,3} \end{pmatrix}$
- 顶点 1：$\mathbf{v}_{f,1} = \begin{pmatrix} v_{f,1,1} \\ v_{f,1,2} \\ v_{f,1,3} \end{pmatrix}$
- 顶点 2：$\mathbf{v}_{f,2} = \begin{pmatrix} v_{f,2,1} \\ v_{f,2,2} \\ v_{f,2,3} \end{pmatrix}$

### (1) 从三顶点求三角形平面外法向单位向量 $\mathbf{n}_f$
从顶点 0 出发，构建两条边向量：
$$\mathbf{e}_{f,1} = \mathbf{v}_{f,1} - \mathbf{v}_{f,0} = \begin{pmatrix} v_{f,1,1} - v_{f,0,1} \\ v_{f,1,2} - v_{f,0,2} \\ v_{f,1,3} - v_{f,0,3} \end{pmatrix}$$
$$\mathbf{e}_{f,2} = \mathbf{v}_{f,2} - \mathbf{v}_{f,0} = \begin{pmatrix} v_{f,2,1} - v_{f,0,1} \\ v_{f,2,2} - v_{f,0,2} \\ v_{f,2,3} - v_{f,0,3} \end{pmatrix}$$

计算两条边向量的三维叉积（外积），得到一个垂直于该三角面的法向量 $\mathbf{N}_f$。我们把叉积的三个分量完完整整写出来：
$$\mathbf{N}_f = \mathbf{e}_{f,1} \times \mathbf{e}_{f,2} = \begin{pmatrix} (v_{f,1,2} - v_{f,0,2})(v_{f,2,3} - v_{f,0,3}) - (v_{f,1,3} - v_{f,0,3})(v_{f,2,2} - v_{f,0,2}) \\ (v_{f,1,3} - v_{f,0,3})(v_{f,2,1} - v_{f,0,1}) - (v_{f,1,1} - v_{f,0,1})(v_{f,2,3} - v_{f,0,3}) \\ (v_{f,1,1} - v_{f,0,1})(v_{f,2,2} - v_{f,0,2}) - (v_{f,1,2} - v_{f,0,2})(v_{f,2,1} - v_{f,0,1}) \end{pmatrix}$$

向量 $\mathbf{N}_f$ 的欧氏模长平方为三个分量的平方和：
$$\|\mathbf{N}_f\| = \sqrt{N_{f,1}^2 + N_{f,2}^2 + N_{f,3}^2}$$
将其除以模长进行归一化；若顶点顺序按右手规则给出的方向指向体外，则得到单位外法向，若指向体内必须取负号：
$$\mathbf{n}_f = \frac{\mathbf{N}_f}{\|\mathbf{N}_f\|} = \begin{pmatrix} n_{f,1} \\ n_{f,2} \\ n_{f,3} \end{pmatrix}, \quad \text{满足 } n_{f,1}^2 + n_{f,2}^2 + n_{f,3}^2 = 1$$

---

## 3. 三角形所在平面的几何方程与截距

在三维空间中，第 $f$ 个三角形所在的无限延展大平面上，任意一点 $\mathbf{y} = (y_1, y_2, y_3)^T$ 到平面上已知顶点 $\mathbf{v}_{f,0}$ 的连线向量，必须严格与法向量 $\mathbf{n}_f$ 垂直。
因此，平面的点法式方程为：
$$\mathbf{n}_f \cdot (\mathbf{y} - \mathbf{v}_{f,0}) = 0$$
把点积拆开：
$$\mathbf{n}_f \cdot \mathbf{y} - \mathbf{n}_f \cdot \mathbf{v}_{f,0} = 0 \iff \mathbf{n}_f \cdot \mathbf{y} = \mathbf{n}_f \cdot \mathbf{v}_{f,0}$$

定义坐标原点到该平面的**有符号垂直投影截距常数**为 $h_f$：
$$h_f = \mathbf{n}_f \cdot \mathbf{v}_{f,0} = n_{f,1} v_{f,0,1} + n_{f,2} v_{f,0,2} + n_{f,3} v_{f,0,3}$$
于是，该三角面所在平面的通用代数方程为：
$$\mathbf{n}_f \cdot \mathbf{y} = h_f$$
写成分量形式就是：
$$n_{f,1} y_1 + n_{f,2} y_2 + n_{f,3} y_3 = h_f$$

---

## 4. 空间虚拟射线与平面的相交距离推导（一步一步解代数方程）

现在，我们从空间场点 $\mathbf{x} = (x_1, x_2, x_3)^T$ 出发，沿着给定的球面单位方向 $\mathbf{u} = (u_1, u_2, u_3)^T$ 的反方向射出一条光线。
根据第二小节和第三小节的定义，相对位移是 $\mathbf{r} = \mathbf{x} - \mathbf{y} = R\mathbf{u}$，因此射线上动点 $\mathbf{y}$ 随着标量距离 $R$ 的轨迹参数方程为：
$$\mathbf{y}(R) = \mathbf{x} - R\mathbf{u}$$
写成分量形式：
$$\begin{pmatrix} y_1(R) \\ y_2(R) \\ y_3(R) \end{pmatrix} = \begin{pmatrix} x_1 - R u_1 \\ x_2 - R u_2 \\ x_3 - R u_3 \end{pmatrix}$$

我们要求解这条射线与第 $f$ 个三角面所在平面的交点。
把射线方程 $\mathbf{y}(R) = \mathbf{x} - R\mathbf{u}$ 直接代入平面的方程 $\mathbf{n}_f \cdot \mathbf{y} = h_f$ 中：
$$\mathbf{n}_f \cdot (\mathbf{x} - R\mathbf{u}) = h_f$$
根据向量点积的线性分配律展开：
$$\mathbf{n}_f \cdot \mathbf{x} - R (\mathbf{n}_f \cdot \mathbf{u}) = h_f$$
这是一个关于未知标量距离 $R$ 的一元一次方程！我们把带 $R$ 的项移到等号右边：
$$R (\mathbf{n}_f \cdot \mathbf{u}) = \mathbf{n}_f \cdot \mathbf{x} - h_f$$
两边同时除以标量 $(\mathbf{n}_f \cdot \mathbf{u})$，解出相交距离 $R_f(\mathbf{u})$：
$$R_f(\mathbf{u}) = \frac{\mathbf{n}_f \cdot \mathbf{x} - h_f}{\mathbf{n}_f \cdot \mathbf{u}}$$

我们把分子和分母的分量全展开写出来：
- **分子**：
  $$\mathbf{n}_f \cdot \mathbf{x} - h_f = (n_{f,1} x_1 + n_{f,2} x_2 + n_{f,3} x_3) - h_f$$
  > **几何物理意义**：分子表示观测点 $\mathbf{x}$ 到第 $f$ 个平面的**垂直有符号几何距离**。如果分子为 0，说明观测点 $\mathbf{x}$ 本身就在这个平面上。
- **分母**：
  $$\mathbf{n}_f \cdot \mathbf{u} = n_{f,1} u_1 + n_{f,2} u_2 + n_{f,3} u_3$$
  > **几何物理意义**：分母是平面的法向量与射线方向之间的**方向余弦**。

### 相交几何分类讨论：
1. **平行情形**：若分母 $\mathbf{n}_f \cdot \mathbf{u} = 0$，直线与平面平行；若分子也为零则整条直线共面，否则没有交点，二者都没有唯一的有限交点；
2. **反向交点**：若计算出来的 $R_f(\mathbf{u}) < 0$，说明射线与平面的数学交点位于射线的后方，物理光线根本碰不到该面；
3. **真实正向交点**：当且仅当 $R_f(\mathbf{u}) > 0$ 时，射线才在前方真正穿过该平面。

---

## 5. 球面可见性分片（Spherical Visibility Partition）的生成机制

天体是由成百上千个有边界的**三角形**拼成的，而不是无限大平面。一条射线只有打在三角形三条边围成的内部时，才算真正击中天体表面。
从观测点 $\mathbf{x}$ 看过去，空间中每个三角形面片的三条物理线段边，投影到天球（二维单位球面 $\mathbb{S}^2$）上，会变成**三条球面大圆弧（Great Circle Arcs）**。

### 大圆弧如何在球面上切割出单元 $\Omega_m$？
当所有 $F$ 个三角形的所有边（共 $3F/2$ 条唯一的物理边）全部投影到单位球面 $\mathbb{S}^2$ 上时，它们就像一把把精密的刻刀，在球面上交织成一个庞大的网状拓扑。
这个大圆弧网络将整球面 $\mathbb{S}^2$ 划分成有限个球面多边形单元 $\Omega_m$；单元内部互不相交，边界弧可能由相邻单元共同拥有：
$$\mathbb{S}^2 = \bigcup_m \overline{\Omega_m}.$$

```
                          球面可见性单元划分 S² = ⋃ Ω_m
                                
                  ┌─────────────────┬─────────────────┐
                 /   单元 Ω₁        │    单元 Ω₂       \
                │  穿透面序列固定:   │   穿透面序列固定:    │
                │  进: Face 3       │   进: Face 3        │
                │  出: Face 12      │   出: Face 15       │
                 \                 / \                 /
                  ├───────────────┼───┴───────────────┤  <--- 投影大圆弧边界
                 /   单元 Ω₃       \     单元 Ω₄       \      (跨过此弧，进出面发生跳变)
                │  穿透面序列固定:   │   穿透面序列固定:    │
                │  进: Face 7       │   进: Face 7        │
                │  出: Face 12      │   出: Face 15       │
                 \                 / \                 /
                  └─────────────────┴─────────────────┘
```

### 为什么这个划分在数学上价值连城？
因为在**任意一个固定的球面单元 $\Omega_m$ 内部**：
在避开投影弧线、相切和顶点的单元内部，方向 $\mathbf{u}$ 的小幅变化不会改变射线穿透面片的组合；单元边界处需按极限处理。
- 射线从哪几个面钻进天体，面索引始终是固定的 $(\mathrm{entry}_1, \mathrm{entry}_2, \dots)$；
- 射线从哪几个面钻出天体，面索引始终是固定的 $(\mathrm{exit}_1, \mathrm{exit}_2, \dots)$；
- 绝不会发生“走着走着突然冒出一个新面”的非连续跳变。

---

## 6. 射线区间在单元 $\Omega_m$ 内的显式代数配对

对于落在同一个球面单元 $\Omega_m$ 内的任意方向 $\mathbf{u} \in \Omega_m$：
射线穿透实体的交集集合 $E_\mathbf{x}(\mathbf{u})$ 具有完全确定的段数 $L_m$，并且可以被严格展开为 $L_m$ 个连续区间的并集：
$$E_\mathbf{x}(\mathbf{u}) = \bigcup_{\ell=1}^{L_m} \left[ R^-_{m,\ell}(\mathbf{u}), \; R^+_{m,\ell}(\mathbf{u}) \right]$$

我们把每一段的入口距离函数 $R^-_{m,\ell}(\mathbf{u})$ 和出口距离函数 $R^+_{m,\ell}(\mathbf{u})$ 用平面求交公式完完整整写出来：
设在第 $\ell$ 段区间中，负责“进”的三角面全局索引为 $\mathrm{in}(m, \ell)$，负责“出”的三角面全局索引为 $\mathrm{out}(m, \ell)$。
代入第 4 小节推导的距离公式：
$$R^-_{m,\ell}(\mathbf{u}) = \frac{\mathbf{n}_{\mathrm{in}(m,\ell)} \cdot \mathbf{x} - h_{\mathrm{in}(m,\ell)}}{\mathbf{n}_{\mathrm{in}(m,\ell)} \cdot \mathbf{u}}$$
$$R^+_{m,\ell}(\mathbf{u}) = \frac{\mathbf{n}_{\mathrm{out}(m,\ell)} \cdot \mathbf{x} - h_{\mathrm{out}(m,\ell)}}{\mathbf{n}_{\mathrm{out}(m,\ell)} \cdot \mathbf{u}}$$

我们把分量全部展开，没有任何省略：
$$R^-_{m,\ell}(\mathbf{u}) = \frac{(n_{\mathrm{in},1} x_1 + n_{\mathrm{in},2} x_2 + n_{\mathrm{in},3} x_3) - h_{\mathrm{in}}}{n_{\mathrm{in},1} u_1 + n_{\mathrm{in},2} u_2 + n_{\mathrm{in},3} u_3}$$
$$R^+_{m,\ell}(\mathbf{u}) = \frac{(n_{\mathrm{out},1} x_1 + n_{\mathrm{out},2} x_2 + n_{\mathrm{out},3} x_3) - h_{\mathrm{out}}}{n_{\mathrm{out},1} u_1 + n_{\mathrm{out},2} u_2 + n_{\mathrm{out},3} u_3}$$

### 特殊边界情况处理（观测点在天体内部或表面）：
如果观测点 $\mathbf{x}$ 位于小行星实体内部，对于从点 $\mathbf{x}$ 发出的所有射线，第 1 段区间（$\ell = 1$）是从自身开始的，它不需要从外部穿入。因此：
$$R^-_{m,1}(\mathbf{u}) \equiv 0$$
只有出口距离 $R^+_{m,1}(\mathbf{u})$ 由穿出的第一个表面三角面决定。

---

## 7. 全局 Hessian 非局部积分项的三重求和终极合体

现在，我们把：
1. 第三小节的方向张量 $T(\mathbf{u}) = 3\mathbf{u}\mathbf{u}^T - I$；
2. 第四小节四元数化后的径向积分 $J_k$；
3. 本小节的球面单元划分 $\mathbb{S}^2 = \bigcup_m \Omega_m$ 及区间上下限 $R^-, R^+$；

完完整整、一笔一画、不带任何缩写地拼进全局非局部 Hessian 张量中：

$$H_K^{\mathrm{nl}}(\mathbf{x}) = G \sum_{m} \sum_{\ell=1}^{L_m} \sum_{k=1}^K w_k \int_{\Omega_m} \begin{pmatrix} 3u_1^2 - 1 & 3u_1 u_2 & 3u_1 u_3 \\ 3u_2 u_1 & 3u_2^2 - 1 & 3u_2 u_3 \\ 3u_3 u_1 & 3u_3 u_2 & 3u_3^2 - 1 \end{pmatrix} \left[ \sigma_k^{-2\alpha_k} \,\mathrm{F.p.} \int_{R^-_{m,\ell}(\mathbf{u})}^{R^+_{m,\ell}(\mathbf{u})} \frac{dR}{R \left[ (R - b_k(\mathbf{u}))^2 + d_k^2(\mathbf{u}) \right]^{\alpha_k}} \right] d\Omega$$

式中所有的内层几何参数全部有显式坐标公式：
- $b_k(\mathbf{u}) = (x_1 - c_{k,1})u_1 + (x_2 - c_{k,2})u_2 + (x_3 - c_{k,3})u_3$
- $d_k^2(\mathbf{u}) = \sigma_k^{-2} + \sum_{i=1}^3 (x_i - c_{k,i})^2 - b_k^2(\mathbf{u})$
- 上下限 $R^-_{m,\ell}(\mathbf{u}), R^+_{m,\ell}(\mathbf{u})$ 由本小节的平面求交方程严格给出。

---

### 第五小节小结与确认

在本节中，我们完全兑现承诺：
1. 从三角网格的三个顶点坐标出发，一步一步用叉乘算出了外法向单位向量 $\mathbf{n}_f$ 和平面截距 $h_f$；
2. 把射线参数方程代入平面方程，完整解出了一元一次方程，推导出了相交距离 $R_f(\mathbf{u})$ 的全展开分式；
3. 阐明了三角面边界投影形成的大圆弧网格，以及球面可见性单元 $\Omega_m$ 的拓扑不变性；
4. 明确写出了每一段穿透区间的上下限代数表达式；
5. 把所有模块全部组装回了 $H_K^{\mathrm{nl}}(\mathbf{x})$ 的三重求和大矩阵中。

---

# 第六小节：近邻奇异性分析、局部切锥张量 $C_\mathbf{x}$ 与 Hadamard 有限部分

---

## 1. 为什么会在 $R \to 0$ 处产生暴烈的对数发散？

请大家看第五小节最后推导出的径向积分核心：
$$\int_{R^-}^{R^+} \frac{\rho_K(\mathbf{x} - R\mathbf{u})}{R} \, dR$$

在太空中（探测器距离小行星还有一段距离），所有实际穿透区间的 $R^- > 0$。分母上的距离 $R$ 永远不会碰到 0，这个积分平滑、温和、人畜无害。
**但是，一旦探测器降落到小行星表面，或者处于天体内部时**：
对于那些朝向小行星内部打进去的射线，射线的起点就是探测器自身！
这意味着：
$$R^- \equiv 0$$
积分下限直接撞上了 0：
$$\int_0^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x} - R\mathbf{u})}{R} \, dR$$
式中 $L(\mathbf{u})$ 是该射线打穿实体后、从另一侧破体而出的第一段出口距离。

在 $R \to 0$ 附近，根据物理连续性，小行星内部的密度不可能为零（$\rho_K(\mathbf{x}) > 0$）。
因此，在距离 $R$ 极小的微元区间 $[0, \varepsilon]$ 内，被积函数的主导行为就是：
$$\frac{\rho_K(\mathbf{x})}{R}$$
学过微积分都知道，$\frac{1}{R}$ 的原函数是对数 $\log R$！
如果我们直接代入上下限：
$$\int_0^\varepsilon \frac{\rho_K(\mathbf{x})}{R} \, dR = \left. \rho_K(\mathbf{x}) \log R \right|_0^\varepsilon = \rho_K(\mathbf{x}) \left[ \log \varepsilon - \log(0) \right]$$
因为 $\log(0) \to -\infty$，**这个积分在下限处彻底炸裂，发散到负无穷大！**

本小节的核心任务，就是用数学手术刀，把这个无限大的发散项从积分里精准解剖出来，搞清楚它到底是由什么几何因素决定的。

---

## 2. 局部方向切锥 $\Gamma_\mathbf{x}$ 的严格几何定义

并非所有方向的射线都会扎进天体内部。我们站在观测点 $\mathbf{x}$ 上环顾四周，把所有能直接扎进小行星内部的方向集合挑出来。
定义观测点 $\mathbf{x}$ 处的**局部方向切锥（Local Tangent Cone）**为：
$$\Gamma_\mathbf{x} = \left\{ \mathbf{u} \in \mathbb{S}^2 \;:\; \exists \, L(\mathbf{u}) > 0 \text{ 使得 } [0, L(\mathbf{u})] \subset E_\mathbf{x}(\mathbf{u}) \right\}$$
直观翻译：只要一个单位方向 $\mathbf{u}$ 能让射线在离开点 $\mathbf{x}$ 的一瞬间立刻进入小行星实体，这个方向 $\mathbf{u}$ 就属于切锥 $\Gamma_\mathbf{x}$。

```
                     空间四种位置的局部方向切锥 Γ_x
                     
      【天体内部】            【光滑表面】              【棱边与顶点】
      全实心包围              半实心、半虚空           像切开的西瓜块
      
          ↑                       ↑                       ↑
       ←  ●  →                 ←  ●  →                    ● 
          ↓                                              / \
                                                        /   \
  Γ_x = 整个单位球面 S²   Γ_x = 正半球面 S²₊         Γ_x = 局部多面棱锥开集
  立体角 = 4π             立体角 = 2π               立体角 = Ω_in
```

---

## 3. 广义 Cauchy 密度的泰勒展开与奇异性分离（一步一步推导）

对于任意一个扎进体内的方向 $\mathbf{u} \in \Gamma_\mathbf{x}$，因为广义 Cauchy 密度函数 $\rho_K(\mathbf{y})$ 在空间中是无穷次可微的平滑函数，我们可以在点 $\mathbf{x}$ 处，沿着射线方向 $-\mathbf{u}$ 对它进行一阶**多元泰勒展开（Taylor Expansion）**：

$$\rho_K(\mathbf{x} - R\mathbf{u}) = \rho_K(\mathbf{x}) - R \left( \nabla\rho_K(\mathbf{x}) \cdot \mathbf{u} \right) + O(R^2)$$

我们把一阶导数点积项的每一个分量写出来：
$$\nabla\rho_K(\mathbf{x}) \cdot \mathbf{u} = \frac{\partial \rho_K}{\partial x_1} u_1 + \frac{\partial \rho_K}{\partial x_2} u_2 + \frac{\partial \rho_K}{\partial x_3} u_3$$
式中 $O(R^2)$ 是距离的二阶及更高阶微小残余项。

为了避开 $R=0$ 的无穷大炸弹，我们引入一个极小的正数截断半径 $\varepsilon > 0$（先不让积分直接碰到 0，而是从 $\varepsilon$ 开始积，事后再看 $\varepsilon \to 0$ 时的极限）：
$$\int_\varepsilon^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x} - R\mathbf{u})}{R} \, dR$$

把泰勒展开式代入被积函数的分子：
$$\int_\varepsilon^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x}) - R (\nabla\rho_K \cdot \mathbf{u}) + O(R^2)}{R} \, dR$$
拆成三项逐个积分：
$$= \int_\varepsilon^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x})}{R} \, dR - \int_\varepsilon^{L(\mathbf{u})} (\nabla\rho_K \cdot \mathbf{u}) \, dR + \int_\varepsilon^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x} - R\mathbf{u}) - \rho_K(\mathbf{x}) + R (\nabla\rho_K \cdot \mathbf{u})}{R} \, dR$$

### 逐项攻克：
1. **第一项（奇异核心）**：
   $$\int_\varepsilon^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x})}{R} \, dR = \rho_K(\mathbf{x}) \left[ \log L(\mathbf{u}) - \log\varepsilon \right]$$
   引入一个任意选定的宏观物理无量纲参考长度尺度 $\ell_0 > 0$（比如小行星的平均半径，如 500 米），我们在对数里配凑一个 $\ell_0$：
   $$\log L(\mathbf{u}) - \log\varepsilon = \log\left( \frac{L(\mathbf{u})}{\ell_0} \cdot \frac{\ell_0}{\varepsilon} \right) = \log\frac{\ell_0}{\varepsilon} + \log\frac{L(\mathbf{u})}{\ell_0}$$
   所以第一项严格写为：
   $$\rho_K(\mathbf{x}) \log\frac{\ell_0}{\varepsilon} + \rho_K(\mathbf{x}) \log\frac{L(\mathbf{u})}{\ell_0}$$
2. **第二项与第三项（有限正则部分）**：
   第二项直接积出来是 $-(\nabla\rho_K \cdot \mathbf{u})(L(\mathbf{u}) - \varepsilon)$；
   第三项分子的级数是 $O(R^2)$，除以 $R$ 之后变成了 $O(R)$，在 $R \to 0$ 时极限为 0，完全没有奇点！
   因此，当 $\varepsilon \to 0$ 时，后面这些非奇异项可以直接合并为良态的正则积分：
   $$\int_0^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x} - R\mathbf{u}) - \rho_K(\mathbf{x})}{R} \, dR + o(1)$$

### 径向积分的最终渐近分离式：
$$\int_\varepsilon^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x} - R\mathbf{u})}{R} \, dR = \rho_K(\mathbf{x}) \log\frac{\ell_0}{\varepsilon} + \rho_K(\mathbf{x}) \log\frac{L(\mathbf{u})}{\ell_0} + \int_0^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x} - R\mathbf{u}) - \rho_K(\mathbf{x})}{R} \, dR + o(1)$$

---

## 4. 局部切锥张量 $C_\mathbf{x}$ 的完全矩阵展开

现在，把上面这个径向积分式子乘上方向张量 $T(\mathbf{u}) = 3\mathbf{u}\mathbf{u}^T - I$，并对所有的扎入方向（切锥 $\Gamma_\mathbf{x}$）做外层球面度规积分 $\int_{\Gamma_\mathbf{x}} d\Omega$。

请看那个带有发散因子 $\log\frac{\ell_0}{\varepsilon}$ 的项：
$$G \int_{\Gamma_\mathbf{x}} T(\mathbf{u}) \left[ \rho_K(\mathbf{x}) \log\frac{\ell_0}{\varepsilon} \right] d\Omega = G \rho_K(\mathbf{x}) \left[ \int_{\Gamma_\mathbf{x}} T(\mathbf{u}) \, d\Omega \right] \log\frac{\ell_0}{\varepsilon}$$

括号里的这个 $3 \times 3$ 几何张量，决定了发散项在三维各个方向上的发散强度。
我们正式将它定义为**局部切锥张量（Local Cone Tensor）$C_\mathbf{x}$**：
$$C_\mathbf{x} = \int_{\Gamma_\mathbf{x}} T(\mathbf{u}) \, d\Omega = \int_{\Gamma_\mathbf{x}} (3\mathbf{u}\mathbf{u}^T - I) \, d\Omega$$

我们把 $C_\mathbf{x}$ 矩阵的 9 个元素完完整整写出来，绝不使用任何简写：
$$C_\mathbf{x} = \begin{pmatrix} \int_{\Gamma_\mathbf{x}} (3u_1^2 - 1) \, d\Omega & \int_{\Gamma_\mathbf{x}} 3u_1 u_2 \, d\Omega & \int_{\Gamma_\mathbf{x}} 3u_1 u_3 \, d\Omega \\ \int_{\Gamma_\mathbf{x}} 3u_2 u_1 \, d\Omega & \int_{\Gamma_\mathbf{x}} (3u_2^2 - 1) \, d\Omega & \int_{\Gamma_\mathbf{x}} 3u_2 u_3 \, d\Omega \\ \int_{\Gamma_\mathbf{x}} 3u_3 u_1 \, d\Omega & \int_{\Gamma_\mathbf{x}} 3u_3 u_2 \, d\Omega & \int_{\Gamma_\mathbf{x}} (3u_3^2 - 1) \, d\Omega \end{pmatrix}$$

---

## 5. 决定命运的时刻：$C_\mathbf{x}$ 到底等不等于零？

这个 $C_\mathbf{x}$ 矩阵到底是真是假？它会不会自动等于零？
我们对空间中的不同位置进行严密的逐个积分计算！

---

### 分类 A：观测点在天体内部（$\mathbf{x} \in \mathrm{int}(V)$）
此时，探测器四面八方全被岩石包围，切锥就是**整整一个单位球面**：
$$\Gamma_\mathbf{x} = \mathbb{S}^2$$
整个球面的立体角是 $4\pi$。我们在球坐标系下计算 9 个分量的积分：
- 单位向量分量为：$u_1 = \sin\theta\cos\phi, \; u_2 = \sin\theta\sin\phi, \; u_3 = \cos\theta$；
- 面积微元为：$d\Omega = \sin\theta \, d\theta \, d\phi$，其中 $\theta \in [0, \pi], \; \phi \in [0, 2\pi)$。

#### 1. 算对角线元素（以 $u_3$ 为例）：
$$\int_{\mathbb{S}^2} u_3^2 \, d\Omega = \int_0^{2\pi} d\phi \int_0^\pi \cos^2\theta \sin\theta \, d\theta$$
令 $t = \cos\theta, \, dt = -\sin\theta d\theta$：
$$= 2\pi \int_{-1}^1 t^2 \, dt = 2\pi \left[ \frac{t^3}{3} \right]_{-1}^1 = 2\pi \left( \frac{1}{3} - \left(-\frac{1}{3}\right) \right) = 2\pi \cdot \frac{2}{3} = \frac{4\pi}{3}$$
因为球面的三维对称性完全均等：
$$\int_{\mathbb{S}^2} u_1^2 \, d\Omega = \int_{\mathbb{S}^2} u_2^2 \, d\Omega = \int_{\mathbb{S}^2} u_3^2 \, d\Omega = \frac{4\pi}{3}$$
而整球面积分 $\int_{\mathbb{S}^2} 1 \, d\Omega = 4\pi$。
现在算主对角线元素 $(3u_i^2 - 1)$ 的积分：
$$\int_{\mathbb{S}^2} (3u_i^2 - 1) \, d\Omega = 3 \int_{\mathbb{S}^2} u_i^2 \, d\Omega - \int_{\mathbb{S}^2} 1 \, d\Omega = 3\left(\frac{4\pi}{3}\right) - 4\pi = 4\pi - 4\pi = 0$$

#### 2. 算非对角线元素（以 $u_1 u_2$ 为例）：
$$\int_{\mathbb{S}^2} 3u_1 u_2 \, d\Omega = 3 \int_0^\pi \sin^3\theta \, d\theta \int_0^{2\pi} \sin\phi \cos\phi \, d\phi$$
后半段关于方位角 $\phi$ 的积分：
$$\int_0^{2\pi} \sin\phi \cos\phi \, d\phi = \frac{1}{2} \int_0^{2\pi} \sin(2\phi) \, d\phi = \left. -\frac{1}{4}\cos(2\phi) \right|_0^{2\pi} = -\frac{1}{4}(1 - 1) = 0$$
所有非对角线元素由于正负半周对称性全部严格为 0！

#### 结论 A：
在天体内部：
$$C_\mathbf{x} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} \equiv 0$$
**发散项系数为零，内部完全没有对数发散！**

---

### 分类 B：观测点在光滑表面上（$\mathbf{x} \in \partial V_{\mathrm{smooth}}$）
此时探测器正下方的半个空间是岩石，上半个空间是虚空。局部切锥是一个**标准的上半单位球面 $\mathbb{S}^2_+$**：
建立局部坐标系，让外法向量沿着 $z$ 轴正方向，介质在下半球（或令朝内的法向为正 $z$ 轴）：
此时极角范围是半球：$\theta \in [0, \pi/2], \; \phi \in [0, 2\pi)$。

#### 1. 算对角线元素：
$$\int_{\mathbb{S}^2_+} u_3^2 \, d\Omega = \int_0^{2\pi} d\phi \int_0^{\pi/2} \cos^2\theta \sin\theta \, d\theta = 2\pi \int_0^1 t^2 \, dt = 2\pi \left[ \frac{t^3}{3} \right]_0^1 = \frac{2\pi}{3}$$
而半球面的总立体角是：
$$\int_{\mathbb{S}^2_+} 1 \, d\Omega = 2\pi$$
因此第 3 行第 3 列的对角元素为：
$$\int_{\mathbb{S}^2_+} (3u_3^2 - 1) \, d\Omega = 3\left(\frac{2\pi}{3}\right) - 2\pi = 2\pi - 2\pi = 0$$
同理，由圆周对称性，$u_1$ 和 $u_2$ 的对角元素也严格相减为零！

#### 2. 算非对角线元素：
方位角 $\phi$ 的积分依然遍历整整一圈 $[0, 2\pi)$，所有交叉项积出来依然全部是 0！

#### 结论 B：
在光滑平坦的表面上：
$$C_\mathbf{x} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} \equiv 0$$
**平面切锥的对数系数为零；这不推出跨越边界的经典二阶导数极限相等。**

---

### 分类 C：观测点在多面体的棱边或顶点（Edge or Vertex）
在棱边或顶点处，局部切锥一般既不是整球也不是半球。此时旋转对称性通常不足以直接抵消二阶矩，但“非光滑”本身不能推出非零结果。必须按定义计算
$$C_{ij}=3\int_{\Gamma_{\mathbf x}}u_i u_j\,d\Omega-\delta_{ij}|\Gamma_{\mathbf x}|.$$
若该矩阵非零且 $\rho_K(\mathbf x)\ne0$，则截断局部项确实含有
$$G\rho_K(\mathbf x)C_{\mathbf x}\log\frac{\ell_0}{\varepsilon};$$
若实际计算得到 $C_{\mathbf x}=0$，则该对数项消失。因而棱边和顶点只能称为“需要检验 $C_{\mathbf x}$ 的位置”，不能无条件称为必然发散。

## 6. 四大项完备分解式与 Hadamard 有限部分提取

以下四项分解先针对 $\mathbf x\in V$ 且存在从 $R=0$ 开始的局部射线段来写；若 $\mathbf x\notin V$，则 $\Gamma_{\mathbf x}=\varnothing$，应把所有实体段直接归入远段项，且没有局部对数项。

基于上述推导，我们在截断参数 $\varepsilon > 0$ 下写为四项之和：

这里 $H_K^\varepsilon$ 定义为球形排异后的非局部核积分再加上固定的分布接触项。于是
$$H_K^\varepsilon(\mathbf{x}) = H_K^{\mathrm{div},\varepsilon}(\mathbf{x}) + H_K^{\mathrm{cone}}(\mathbf{x}) + H_K^{\mathrm{reg}}(\mathbf{x}) + H_K^{\mathrm{multi}}(\mathbf{x})$$

我们把每一项的完整数学表达式全部列出：

### 1. 对数系数项（在实际 $\rho_K(\mathbf x)C_{\mathbf x}\ne0$ 时非零）：
$$H_K^{\mathrm{div},\varepsilon}(\mathbf{x}) = G \rho_K(\mathbf{x}) C_\mathbf{x} \log\frac{\ell_0}{\varepsilon}$$

### 2. 有限方向锥项（包含尺度比对数与第二小节推导的狄拉克项）：
$$H_K^{\mathrm{cone}}(\mathbf{x}) = G \rho_K(\mathbf{x}) \int_{\Gamma_\mathbf{x}} T(\mathbf{u}) \log\frac{L(\mathbf{u})}{\ell_0} \, d\Omega - \frac{4\pi G}{3} \theta_{\mathrm{ball}}(\mathbf{x}) \rho_K(\mathbf{x}) I$$

### 3. 局部正则积分项（平滑差值，处处收敛）：
$$H_K^{\mathrm{reg}}(\mathbf{x}) = G \int_{\Gamma_\mathbf{x}} T(\mathbf{u}) \left[ \int_0^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x} - R\mathbf{u}) - \rho_K(\mathbf{x})}{R} \, dR \right] d\Omega$$

### 4. 非局部多区间远离项（光线穿透天体深坑、空隙的远端实体段）：
$$H_K^{\mathrm{multi}}(\mathbf{x}) = G \sum_m \sum_{\ell \ge 2} \int_{\Omega_m} T(\mathbf{u}) \left[ \int_{R^-_{m,\ell}(\mathbf{u})}^{R^+_{m,\ell}(\mathbf{u})} \frac{\rho_K(\mathbf{x} - R\mathbf{u})}{R} \, dR \right] d\Omega$$

---

## 7. 什么是 Hadamard 有限部分（Finite Part）？

如果在物理或工程计算中，我们选择报告一个明示参考长度 $\ell_0$ 的有限部分，我们就可以采用法国数学家雅克·阿达马（Jacques Hadamard）在偏微分方程中创立的工具——**Hadamard 有限部分提取法（$\mathrm{F.p.}$）**：
把发散对数项直接作为奇异基底丢弃，只保留当 $\varepsilon \to 0$ 后的极限有限部分：

$$H_K^{\mathrm{FP}}(\mathbf{x}) \equiv \lim_{\varepsilon \to 0^+} \left[ H_K^\varepsilon(\mathbf{x}) - G \rho_K(\mathbf{x}) C_\mathbf{x} \log\frac{\ell_0}{\varepsilon} \right] = H_K^{\mathrm{cone}}(\mathbf{x}) + H_K^{\mathrm{reg}}(\mathbf{x}) + H_K^{\mathrm{multi}}(\mathbf{x})$$

这样，在已经验证发散系数的情况下，可以按声明的参考尺度提取有限部分；其数值依赖 $\ell_0$，见第十六节第 5 段。

---

### 第六小节小结与确认

在本节中，我们：
1. 完整展示了泰勒展开如何把 $\rho_K(\mathbf{x})/R$ 诱导的 $\log\varepsilon$ 奇异性剥离出来；
2. 一笔一画写出了局部切锥张量 $C_\mathbf{x}$ 的全部 9 个分量；
3. 证明了内部点和光滑平面边界的 $C_\mathbf{x}=0$；对于棱边和顶点，给出了必须实际计算 $C_\mathbf{x}$ 的判据，未把非光滑性误当成充分条件；
4. 明确给出了总 Hessian 的四大分项以及 Hadamard 有限部分的无歧义定义。

太到位了！这个质疑极其敏锐，直接切中了**奇异积分与有限部分理论最核心的命脉**！

上一小节我说了一句“代入极限得到 $o(1)$”，这确实太粗略了。
现在我们停下来，**彻底掰开、揉碎、一步不落地把 $\varepsilon$ 到底是什么、它是怎么趋向于零的、每一个微元在 $\varepsilon \to 0^+$ 时到底是爆炸、消失还是变成常数，完完全全写清楚！**

---

# 深度专讲：$\varepsilon \to 0^+$ 极限过程的严格数学解剖

---

## 1. 什么是 $\varepsilon$？它的几何图景到底是什么？

我们回到最根本的物理场景：
观测点是 $\mathbf{x}$。当 $\mathbf{x}$ 落在天体内部或表面时，牛顿核的距离分母是 $R = \|\mathbf{x} - \mathbf{y}\|$。
在 $R = 0$ 处，被积函数分母为零，这在数学上叫作**奇异点（Singularity）**。

为了在数学上能合法地做积分，数学家采用的办法叫作**邻域排除法（截断挖洞法）**：
我们在三维空间中，以观测点 $\mathbf{x}$ 为球心，画一个半径为 $\varepsilon > 0$ 的极小微型三维实体球，记作：
$$B_\varepsilon(\mathbf{x}) = \{ \mathbf{y} \in \mathbb{R}^3 \;:\; \|\mathbf{x} - \mathbf{y}\| \le \varepsilon \}$$
我们把这个小球内部的所有物质暂时“挖掉不积”，只对小球外部的区域积：
$$V_\varepsilon = V \setminus B_\varepsilon(\mathbf{x})$$

转换到我们的**射线坐标系**：
从点 $\mathbf{x}$ 出发的任意一条射线，原本距离 $R$ 的积分区间是从 $0$ 到出口距离 $L(\mathbf{u})$。
现在因为挖掉了半径为 $\varepsilon$ 的小球，光线从球心出发，**必须先走过距离 $\varepsilon$ 穿出小球，才能开始碰到被积物质**！
因此，积分下限不再是 $0$，而是严格变成了正数 $\varepsilon$：
$$I_\varepsilon(\mathbf{u}) = \int_\varepsilon^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x} - R\mathbf{u})}{R} \, dR$$

现在，所谓“$\varepsilon$ 趋近于零（$\varepsilon \to 0^+$）”，意思就是：
**我们让这个挖掉的小球越来越小、越来越小，最终无限缩回成原点本身。在这个收缩过程中，我们要盯死积分里的每一项到底发生了什么！**

---

## 2. 把密度函数展开到二阶，进行完全逐项积分（毫无保留）

广义 Cauchy 密度函数 $\rho_K(\mathbf{y})$ 是光滑的。我们在点 $\mathbf{x}$ 处，沿着方向 $-\mathbf{u}$，把密度函数写出**包含二阶导数的完整泰勒展开式**：

$$\rho_K(\mathbf{x} - R\mathbf{u}) = \rho_K(\mathbf{x}) - R (\nabla\rho_K(\mathbf{x}) \cdot \mathbf{u}) + \frac{1}{2} R^2 \left( \mathbf{u}^T \nabla^2\rho_K(\mathbf{x}) \mathbf{u} \right) + \mathcal{R}_3(R)$$
式中：
- $\rho_K(\mathbf{x})$ 是一个常数（点 $\mathbf{x}$ 处的局部密度）；
- $\nabla\rho_K(\mathbf{x}) \cdot \mathbf{u} = \sum_{i=1}^3 \frac{\partial \rho_K}{\partial x_i} u_i$ 是密度的方向导数（也是一个与 $R$ 无关的常数）；
- $\mathbf{u}^T \nabla^2\rho_K(\mathbf{x}) \mathbf{u} = \sum_{i=1}^3 \sum_{j=1}^3 \frac{\partial^2 \rho_K}{\partial x_i \partial x_j} u_i u_j$ 是二阶方向导数常数；
- $\mathcal{R}_3(R)$ 是三阶及更高阶的高阶无穷小残余，满足 $|\mathcal{R}_3(R)| \le M R^3$（$M$ 为某个有界正实数）。

现在把这个展开式代入被积函数中，每一项都除以分母 $R$：
$$\frac{\rho_K(\mathbf{x} - R\mathbf{u})}{R} = \frac{\rho_K(\mathbf{x})}{R} - (\nabla\rho_K(\mathbf{x}) \cdot \mathbf{u}) + \frac{1}{2} R \left( \mathbf{u}^T \nabla^2\rho_K(\mathbf{x}) \mathbf{u} \right) + \frac{\mathcal{R}_3(R)}{R}$$

我们对这四项，从 $\varepsilon$ 到 $L(\mathbf{u})$ 逐项计算定积分！

---

### 第一项（$1/R$ 奇异项）的定积分与对数爆炸
$$\int_\varepsilon^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x})}{R} \, dR = \rho_K(\mathbf{x}) \left[ \log R \right]_\varepsilon^{L(\mathbf{u})} = \rho_K(\mathbf{x}) \left[ \log(L(\mathbf{u})) - \log\varepsilon \right]$$
我们引入宏观参考物理长度 $\ell_0 > 0$（比如小行星参考半径 500 米），利用对数的性质：
$$\log(L(\mathbf{u})) - \log\varepsilon = \log\left( \frac{L(\mathbf{u})}{\ell_0} \cdot \frac{\ell_0}{\varepsilon} \right) = \log\frac{L(\mathbf{u})}{\ell_0} + \log\frac{\ell_0}{\varepsilon}$$
代入后得到第一项的严格结果：
$$= \rho_K(\mathbf{x}) \log\frac{\ell_0}{\varepsilon} + \rho_K(\mathbf{x}) \log\frac{L(\mathbf{u})}{\ell_0}$$

---

### 第二项（常数项）的定积分与 $\varepsilon$ 的线性消失
$$- \int_\varepsilon^{L(\mathbf{u})} (\nabla\rho_K(\mathbf{x}) \cdot \mathbf{u}) \, dR = - (\nabla\rho_K(\mathbf{x}) \cdot \mathbf{u}) [R]_\varepsilon^{L(\mathbf{u})} = - (\nabla\rho_K(\mathbf{x}) \cdot \mathbf{u}) \left[ L(\mathbf{u}) - \varepsilon \right]$$
乘开来：
$$= - (\nabla\rho_K(\mathbf{x}) \cdot \mathbf{u}) L(\mathbf{u}) + \varepsilon (\nabla\rho_K(\mathbf{x}) \cdot \mathbf{u})$$

---

### 第三项（一次项 $R$）的定积分与 $\varepsilon^2$ 的高阶消失
$$\int_\varepsilon^{L(\mathbf{u})} \frac{1}{2} R \left( \mathbf{u}^T \nabla^2\rho_K(\mathbf{x}) \mathbf{u} \right) \, dR = \frac{1}{2} \left( \mathbf{u}^T \nabla^2\rho_K(\mathbf{x}) \mathbf{u} \right) \left[ \frac{R^2}{2} \right]_\varepsilon^{L(\mathbf{u})} = \frac{1}{4} \left( \mathbf{u}^T \nabla^2\rho_K(\mathbf{x}) \mathbf{u} \right) \left[ L^2(\mathbf{u}) - \varepsilon^2 \right]$$
乘开来：
$$= \frac{1}{4} \left( \mathbf{u}^T \nabla^2\rho_K(\mathbf{x}) \mathbf{u} \right) L^2(\mathbf{u}) - \frac{\varepsilon^2}{4} \left( \mathbf{u}^T \nabla^2\rho_K(\mathbf{x}) \mathbf{u} \right)$$

---

### 第四项（高阶残余项）的定积分
$$\left| \int_\varepsilon^{L(\mathbf{u})} \frac{\mathcal{R}_3(R)}{R} \, dR \right| \le \int_\varepsilon^{L(\mathbf{u})} M R^2 \, dR = \frac{M}{3} \left[ L^3(\mathbf{u}) - \varepsilon^3 \right]$$

---

## 3. 把这四项加起来，严密分类关于 $\varepsilon$ 的依赖关系

我们把上面四项全部合并，按照它们在 $\varepsilon \to 0^+$ 时扮演的角色，严格分成三大家族：

$$I_\varepsilon(\mathbf{u}) = \text{【发散族】} + \text{【有限常数族】} + \text{【随 } \varepsilon \text{ 消失的微小族】}$$

我们把这三大家族里的每一项清清楚楚写在下面：

### 1. 【发散族】（包含 $\log\varepsilon$，当 $\varepsilon \to 0^+$ 时趋于无穷大）：
$$\rho_K(\mathbf{x}) \log\frac{\ell_0}{\varepsilon}$$

### 2. 【有限常数族】（完全由天体宏观边界 $L(\mathbf{u})$ 和局部导数决定，与 $\varepsilon$ 毫无关系）：
$$\rho_K(\mathbf{x}) \log\frac{L(\mathbf{u})}{\ell_0} - (\nabla\rho_K(\mathbf{x}) \cdot \mathbf{u}) L(\mathbf{u}) + \frac{1}{4} \left( \mathbf{u}^T \nabla^2\rho_K(\mathbf{x}) \mathbf{u} \right) L^2(\mathbf{u}) + \int_0^{L(\mathbf{u})} \frac{\mathcal{R}_3(R)}{R} \, dR$$
这个族就是我们上一节定义的平滑有限积分：
$$\rho_K(\mathbf{x}) \log\frac{L(\mathbf{u})}{\ell_0} + \int_0^{L(\mathbf{u})} \frac{\rho_K(\mathbf{x} - R\mathbf{u}) - \rho_K(\mathbf{x})}{R} \, dR$$

### 3. 【随 $\varepsilon$ 消失的微小族】（这就是之前简写的 $o(1)$，现在把真实结构全部扒开！）：
$$\Delta(\varepsilon) = \varepsilon (\nabla\rho_K(\mathbf{x}) \cdot \mathbf{u}) - \frac{\varepsilon^2}{4} \left( \mathbf{u}^T \nabla^2\rho_K(\mathbf{x}) \mathbf{u} \right) - \int_0^\varepsilon \frac{\mathcal{R}_3(R)}{R} \, dR$$

---

## 4. 显微镜下看极限：当 $\varepsilon \to 0^+$ 时，各项到底怎么变？

我们让截断尺寸 $\varepsilon$ 真正走向零：$10^{-2} \to 10^{-4} \to 10^{-8} \to 10^{-16} \to 0$。

### (1) 看【微小族】$\Delta(\varepsilon)$：
- 第一项正比于 $\varepsilon$：当 $\varepsilon \to 0$ 时，$\varepsilon \cdot (\dots) \to 0$；
- 第二项正比于 $\varepsilon^2$：当 $\varepsilon \to 0$ 时，$\varepsilon^2 \cdot (\dots) \to 0$；
- 第三项的上限是 $\frac{M}{3}\varepsilon^3$：当 $\varepsilon \to 0$ 时，$\varepsilon^3 \to 0$。
所以：
$$\lim_{\varepsilon \to 0^+} \Delta(\varepsilon) = 0 - 0 - 0 \equiv 0$$
**数学结论 1**：所有正幂次项（$\varepsilon, \varepsilon^2, \varepsilon^3$）在极限中**彻底死亡、灰飞烟灭，对最终物理量没有任何贡献！**

---

### (2) 看【发散族】$\log\frac{\ell_0}{\varepsilon}$：
设参考尺度 $\ell_0 = 1$ 米。我们来看具体数值随着 $\varepsilon$ 变小的变化：
- 当 $\varepsilon = 10^{-2}$ 米（1厘米）：$\log(10^2) = 2 \times 2.3026 \approx 4.61$
- 当 $\varepsilon = 10^{-4}$ 米（0.1毫米）：$\log(10^4) = 4 \times 2.3026 \approx 9.21$
- 当 $\varepsilon = 10^{-8}$ 米（0.1纳米）：$\log(10^8) = 8 \times 2.3026 \approx 18.42$
- 当 $\varepsilon = 10^{-16}$ 米（原子核尺度）：$\log(10^{16}) = 16 \times 2.3026 \approx 36.84$
- 当 $\varepsilon \to 0^+$ 时：
$$\lim_{\varepsilon \to 0^+} \log\frac{\ell_0}{\varepsilon} = +\infty$$
**数学结论 2**：对数项是绝对发散的！它以对数速度不可逆转地奔向正无穷大。

---

## 5. 外层球面积分后：对数发散项在不同位置的生死抉择

现在，我们把整个径向积分乘以张量 $T(\mathbf{u}) = 3\mathbf{u}\mathbf{u}^T - I$，并在切锥 $\Gamma_\mathbf{x}$ 上做球面度规积分 $\int_{\Gamma_\mathbf{x}} d\Omega$。

整个带 $\varepsilon$ 的总 Hessian 矩阵为：
$$H_K^\varepsilon(\mathbf{x}) = G \rho_K(\mathbf{x}) \left[ \int_{\Gamma_\mathbf{x}} T(\mathbf{u}) \, d\Omega \right] \log\frac{\ell_0}{\varepsilon} + H_{\mathrm{finite}}(\mathbf{x}) + G \int_{\Gamma_\mathbf{x}} T(\mathbf{u}) \Delta(\varepsilon) \, d\Omega$$

回顾局部切锥张量 $C_\mathbf{x} = \int_{\Gamma_\mathbf{x}} T(\mathbf{u}) \, d\Omega$。
因为当 $\varepsilon \to 0^+$ 时，最后那项带有 $\Delta(\varepsilon)$ 的积分严格趋向于零：
$$\lim_{\varepsilon \to 0^+} G \int_{\Gamma_\mathbf{x}} T(\mathbf{u}) \Delta(\varepsilon) \, d\Omega = 0$$

因此，在已经把有限部分和远段固定后，剩余的非收敛行为只可能来自第一项：
$$G \rho_K(\mathbf{x}) C_\mathbf{x} \log\frac{\ell_0}{\varepsilon}$$

这里产生整个理论中最震撼的**极限行为大分流**：

---

### 情况一：观测点在天体内部或光滑表面（$C_\mathbf{x} \equiv 0$）

我们在上一节已经用微积分原函数严格证明了：
无论是整个球面（内部），还是对称半球面（光滑表面），$T(\mathbf{u})$ 在上面的积分全部相消为零矩阵！
$$C_\mathbf{x} = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

我们把这个绝对的零矩阵代入发散项：
$$G \rho_K(\mathbf{x}) \cdot C_\mathbf{x} \cdot \log\frac{\ell_0}{\varepsilon} = G \rho_K(\mathbf{x}) \cdot \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} \cdot \log\frac{\ell_0}{\varepsilon} \equiv \begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

请注意这里的数学逻辑：
**这不是 $0 \times \infty$ 的未定式！**
因为对于**任意一个固定的大于零的数 $\varepsilon$**（不管是 $0.1$ 还是 $10^{-5}$），前面的张量积分 $C_\mathbf{x}$ 早就已经是精确的绝对零矩阵了！
一个货真价实的零矩阵，乘以一个有限的数 $\log\frac{\ell_0}{\varepsilon}$，其结果在极限过程之前就已经**恒等于零矩阵**！

所以，对 $\varepsilon \to 0^+$ 取极限：
$$\lim_{\varepsilon \to 0^+} H_K^\varepsilon(\mathbf{x}) = \mathbf{0} + H_{\mathrm{finite}}(\mathbf{x}) + \mathbf{0} = H_{\mathrm{finite}}(\mathbf{x})$$
**在内部和光滑表面上，当 $\varepsilon \to 0^+$ 时，极限不仅存在，而且完全收敛、绝对有限！**

---

### 情况二：观测点在多面体棱边或顶点（需先计算 $C_\mathbf{x}$）

当探测器站在三角形的折线上或顶点上时，切锥 $\Gamma_\mathbf{x}$ 的对称性通常被破坏，但必须由
$$C_{ij}=3\int_{\Gamma_\mathbf{x}}u_i u_jd\Omega-\delta_{ij}|\Gamma_\mathbf{x}|$$
逐项计算是否为非零矩阵。若且仅若
$$\rho_K(\mathbf{x})C_\mathbf{x}\ne0,$$
才有
$$G\rho_K(\mathbf{x})C_\mathbf{x}\log\frac{\ell_0}{\varepsilon}$$
这一非零对数项，因而经典截断极限不收敛；若 $C_\mathbf{x}=0$ 或 $\rho_K(\mathbf{x})=0$，该项则消失。数值程序也不能据此断言必然出现 `NaN`，而应先显式减去第 7 节定义的发散项。

## 6. 这正是为什么必须定义 Hadamard 有限部分！

阿达马（Hadamard）的洞察力就在这里：
一旦计算确认 $G \rho_K(\mathbf{x}) C_\mathbf{x}\ne0$，发散部分正是已知的 $G \rho_K(\mathbf{x}) C_\mathbf{x} \log\frac{\ell_0}{\varepsilon}$；因此可以在取极限之前将它从原式中减掉。

这就是 Hadamard 有限部分的严格数学极限定义：
$$H_K^{\mathrm{FP}}(\mathbf{x}) \equiv \lim_{\varepsilon \to 0^+} \left[ H_K^\varepsilon(\mathbf{x}) - G \rho_K(\mathbf{x}) C_\mathbf{x} \log\frac{\ell_0}{\varepsilon} \right]$$

我们把前面展开好的式子代入这个定义中：
$$= \lim_{\varepsilon \to 0^+} \left[ \left( G \rho_K(\mathbf{x}) C_\mathbf{x} \log\frac{\ell_0}{\varepsilon} + H_{\mathrm{finite}}(\mathbf{x}) + G \int_{\Gamma_\mathbf{x}} T(\mathbf{u}) \Delta(\varepsilon) d\Omega \right) - G \rho_K(\mathbf{x}) C_\mathbf{x} \log\frac{\ell_0}{\varepsilon} \right]$$

请看：
1. 括号里的发散项 $G \rho_K C_\mathbf{x} \log\frac{\ell_0}{\varepsilon}$ 与后面减去的项**完全相消、精确归零**！
2. 剩下的项中，$H_{\mathrm{finite}}(\mathbf{x})$ 与 $\varepsilon$ 完全无关；
3. 残余项 $\Delta(\varepsilon)$ 在 $\varepsilon \to 0^+$ 时，如我们在第 4 节证明的，严格趋向于 0！

因此极限瞬间变得完全合法：
$$H_K^{\mathrm{FP}}(\mathbf{x}) = H_{\mathrm{finite}}(\mathbf{x}) + \mathbf{0} = H_{\mathrm{finite}}(\mathbf{x})$$

---

### 本次深化总结

经过这次毫无保留的极限展开，我们彻底搞清楚了关于 $\varepsilon \to 0^+$ 的全套数学细节：
1. $\varepsilon$ 的物理来源是以 $\mathbf{x}$ 为中心的排异小球半径；
2. 泰勒展开后，由于分母除以 $R$，使得零阶项积分出对数爆炸 $\log\varepsilon$，而一阶及以上正幂次项乘上 $\varepsilon, \varepsilon^2$ 后在 $\varepsilon \to 0^+$ 时全部严格死绝（趋向于 0）；
3. 在内部和光滑表面，因为对称性使得 $C_\mathbf{x} \equiv \mathbf{0}$，对数项被零矩阵直接掐死在摇篮里，不需要做任何减法极限就天然收敛；
4. 在任何实际得到 $C_\mathbf{x}\ne0$ 的位置，直接令 $\varepsilon\to0^+$ 会出现对数项；减去明确的 Hadamard 发散项后，有限部分才按所选参考尺度定义。

---

# 第七小节：球面几何矩与 Gauss-Green 散度定理边界线积分化

## 1. 原向量场的反例和正确目标

要计算的矩是 $M_\Gamma=\int_\Gamma\mathbf u\mathbf u^{T}\,d\Omega$，并非已在前几节算出的完整变密度方向积分。原版向量场
$$\mathbf A_{\rm old}=\tfrac12[(\mathbf a\cdot\mathbf u)(\mathbf b\times\mathbf u)+(\mathbf b\cdot\mathbf u)(\mathbf a\times\mathbf u)]$$
的散度实际恒为零：因为 $\nabla_S(\mathbf a\cdot\mathbf u)=\mathbf a-(\mathbf a\cdot\mathbf u)\mathbf u$ 且 $\operatorname{div}_S(\mathbf b\times\mathbf u)=0$，第一项散度为
$$[\mathbf a-(\mathbf a\cdot\mathbf u)\mathbf u]\cdot(\mathbf b\times\mathbf u)=(\mathbf a\times\mathbf b)\cdot\mathbf u;$$
第二项交换 $\mathbf a,\mathbf b$ 得到其负值。故
$$\operatorname{div}_S\mathbf A_{\rm old}=\tfrac12[(\mathbf a\times\mathbf b)+(\mathbf b\times\mathbf a)]\cdot\mathbf u=0.$$
例如令 $\mathbf a=\mathbf b=(1,0,0)$、$\mathbf u=(1,0,0)$，原版所声称的右边是 $1-1/3=2/3$。原版球面散度恒等式因此不能使用。

## 2. 正确的零阶矩、二阶矩与球面场

令 $f_{\mathbf a,\mathbf b}(\mathbf u)=(\mathbf a\cdot\mathbf u)(\mathbf b\cdot\mathbf u)$，并从中减去球面平均值：
$$h_{\mathbf a,\mathbf b}(\mathbf u)=f_{\mathbf a,\mathbf b}(\mathbf u)-\frac{\mathbf a\cdot\mathbf b}{3}.$$
记 $P_{\mathbf u}=I-\mathbf u\mathbf u^T$，则 $\nabla_S(\mathbf a\cdot\mathbf u)=P_{\mathbf u}\mathbf a$。逐项应用乘积法则：
$$\nabla_S h=(\mathbf b\cdot\mathbf u)P_{\mathbf u}\mathbf a+(\mathbf a\cdot\mathbf u)P_{\mathbf u}\mathbf b$$
$$=(\mathbf b\cdot\mathbf u)\mathbf a+(\mathbf a\cdot\mathbf u)\mathbf b
-2(\mathbf a\cdot\mathbf u)(\mathbf b\cdot\mathbf u)\mathbf u.$$
球面上一阶球谐量满足 $\Delta_S(\mathbf a\cdot\mathbf u)=-2(\mathbf a\cdot\mathbf u)$。也可直接在 $\mathbf a=\mathbf b=\mathbf e_3$ 的局部坐标算出 $\Delta_S(u_3^2-1/3)=-6(u_3^2-1/3)$，再由双线性极化与旋转不变性推广，故
$$\Delta_S h=-6h.$$
现在定义正确的切向向量场
$$\boxed{\mathbf A_{\mathbf a,\mathbf b}=-\frac16\nabla_S h_{\mathbf a,\mathbf b}}.$$
它确实切向：$\mathbf u\cdot P_{\mathbf u}\mathbf a=\mathbf u\cdot\mathbf a-(\mathbf u\cdot\mathbf a)\|\mathbf u\|^2=0$。其散度逐步得到
$$\operatorname{div}_S\mathbf A=-\frac16\operatorname{div}_S\nabla_Sh
=-\frac16\Delta_Sh=h=f_{\mathbf a,\mathbf b}-\frac{\mathbf a\cdot\mathbf b}{3}.$$

## 3. Gauss–Green 的严格适用范围及显式线积分

设 $\Gamma\subset S^2$ 的边界为有限条分段 $C^1$ 曲线；每条曲线配有球面内指向区域外部的单位余法向 $\boldsymbol\nu$。球面散度定理给出
$$\int_\Gamma h\,d\Omega=\oint_{\partial\Gamma}\mathbf A\cdot\boldsymbol\nu\,ds.$$
把 $h$ 的定义代回，且记 $\Omega_\Gamma=\int_\Gamma d\Omega$，得到
$$\boxed{\int_\Gamma(\mathbf a\cdot\mathbf u)(\mathbf b\cdot\mathbf u)\,d\Omega
=\frac{\mathbf a\cdot\mathbf b}{3}\Omega_\Gamma-\frac16\oint_{\partial\Gamma}(\nabla_Sh)\cdot\boldsymbol\nu\,ds.}$$
取 $\mathbf a=\mathbf e_i,\mathbf b=\mathbf e_j$，矩阵各元分别为
$$M_{ij}=\frac{\delta_{ij}}3\Omega_\Gamma-\frac16\oint_{\partial\Gamma}
[(u_j\mathbf e_i+u_i\mathbf e_j-2u_i u_j\mathbf u)\cdot\boldsymbol\nu]\,ds.$$
因此局部切锥张量满足**可直接核验的边界表达**
$$\boxed{(C_\Gamma)_{ij}=3M_{ij}-\Omega_\Gamma\delta_{ij}
=-\frac12\oint_{\partial\Gamma}(u_j\nu_i+u_i\nu_j)\,ds,}$$
因为 $\mathbf u\cdot\boldsymbol\nu=0$。矩阵形式：
$$\boxed{C_\Gamma=-\frac12\oint_{\partial\Gamma}
(\mathbf u\boldsymbol\nu^T+\boldsymbol\nu\mathbf u^T)\,ds.}$$
迹检验：$\operatorname{tr}C_\Gamma=-\oint_{\partial\Gamma}\mathbf u\cdot\boldsymbol\nu\,ds=0$。
全球无边界，故 $C_{S^2}=0$；半球的赤道积分也为零；通常的非平直棱锥则须按几何实际计算，不得无条件声称每条网格边必发散。

## 4. 大圆弧展开：参数、余法向和端点式

给一条定向大圆弧选择正交单位基 $\mathbf e_1,\mathbf e_2$，令
$$\mathbf u(t)=\mathbf e_1\cos t+\mathbf e_2\sin t,\quad
\boldsymbol\tau(t)=\mathbf u'(t)=-\mathbf e_1\sin t+\mathbf e_2\cos t,\quad ds=dt.$$
由于 $\mathbf u\cdot\boldsymbol\tau=0$ 且 $\|\boldsymbol\tau\|=1$，两种球面内单位法向为 $\pm(\boldsymbol\tau\times\mathbf u)$；由区域在弧的哪一侧决定符号，不能仅从旋转轴武断指定“外部”。
令该边的实际外余法向为常矢量 $\boldsymbol\nu_e$。直接积分
$$\int_{t_0}^{t_1}\mathbf u(t)\,dt
=\mathbf e_1(\sin t_1-\sin t_0)+\mathbf e_2(\cos t_0-\cos t_1).$$
将其代入上面的 $C_\Gamma$ 矩阵式，对每条有向边求和：
$$C_\Gamma=-\frac12\sum_e
\left[\mathbf v_e\boldsymbol\nu_e^T+\boldsymbol\nu_e\mathbf v_e^T\right],\qquad
\mathbf v_e=\mathbf e_{1e}\Delta\sin t-\mathbf e_{2e}\Delta\cos t.$$
这里 $\Delta\sin t=\sin t_1-\sin t_0$、$\Delta\cos t=\cos t_1-\cos t_0$，已经将所有边界矩积分化为端点量。若采用四元数，请取
$q(t)=\exp(t\boldsymbol\omega/2)$、$\mathbf u(t)=q(t)\mathbf u_0q(t)^{-1}$；如需初始姿态 $q_0$，应写 $q(t)=\exp(t\boldsymbol\omega/2)q_0$ 并对初始基向量作用，不能混用世界轴与体轴。

**适用界限：**这里降维的是权重为常数的球面二阶矩及 $C_\Gamma$。若积分为 $\int_\Gamma T(\mathbf u)S(\mathbf u)\,d\Omega$，其中 $S$ 随方向变化，须重新求解带权的球面方程或使用第八小节的完整同伦形式；本节的 $\mathbf A_{\mathbf a,\mathbf b}$ 不能直接替换进去。

# 第八小节：球面微分形式与 Poincaré 同伦逆算子

---

## 1. 为什么在处理广义 Cauchy 变密度时，必须升级到“微分形式”？

在第七小节中，我们成功利用经典微积分的 Gauss-Green 散度定理，把简单的方向张量矩 $\int \mathbf{u}\mathbf{u}^T d\Omega$ 降维成了边界线积分。
但是，请大家回想一下第五小节中那个**真正统领全局的物理被积函数**：

$$F_{ij, m}(\mathbf{u}) = T_{ij}(\mathbf{u}) \sum_{k=1}^K w_k \sum_{\ell=1}^{L_m} J_{k, m\ell}(\mathbf{u})$$

这里的 $J_k$ 是带有 Cauchy 变密度的径向积分：
$$J_k = \sigma_k^{-2\alpha_k} \int_{R^-}^{R^+} \frac{dR}{R \left[ (R - b_k(\mathbf{u}))^2 + d_k^2(\mathbf{u}) \right]^{\alpha_k}}$$
它不是一个简单的常数，而是一个随着方向 $\mathbf{u}$ 剧烈变化的**高度复杂的超越/代数复合函数**！

如果我们试图用大二普通矢量微积分去给这一大坨函数生造一个矢量势 $\mathbf{A}$（去硬凑 $\nabla \times \mathbf{A}$ 或 $\nabla \cdot \mathbf{A}$），变量之间的偏导数链式法则会庞杂到让人绝望，根本凑不出来。
**数学家的终极解法**：
放弃笨拙的三维矢量记号，切换到现代微分几何的“母语”——**微分形式（Differential Forms）与外微分流形代数**！

---

## 2. 微分形式入门扫盲（从零复习，完全展开）

设我们在球面单元 $\Omega_m$ 上建立了一个局部的二维平直参数坐标系，记两个坐标分量为：
$$(\xi, \eta) \in \mathbb{R}^2$$

### (1) 什么是 0-形式、1-形式与 2-形式？
- **0-形式（0-form）**：就是最普通的标量函数，比如 $f(\xi, \eta)$；
- **1-形式（1-form）**：类似于普通微积分里的线积分被积微元，形式为：
  $$\alpha = P(\xi, \eta) \, d\xi + Q(\xi, \eta) \, d\eta$$
- **2-形式（2-form）**：类似于二维面积分的被积面元，形式为：
  $$\omega = F(\xi, \eta) \, d\xi \wedge d\eta$$

### (2) 楔积（Wedge Product $\wedge$）的两大核心铁律
微分形式之间的乘法叫作**楔积**（记作 $\wedge$）。它具有颠扑不破的代数公理：
1. **反对易律（交换位置立刻添负号）**：
   $$d\xi \wedge d\eta = - d\eta \wedge d\xi$$
2. **零幂律（同类微分相乘严格等于零）**：
   $$d\xi \wedge d\xi = 0$$
   $$d\eta \wedge d\eta = 0$$

> **直观理解**：$d\xi \wedge d\eta$ 代表一个**有方向的微元面积**。如果你颠倒了坐标轴的顺序（比如把 $x$ 轴和 $y$ 轴对调），右手法则的朝向就彻底翻转了，所以必须变号；而两个平行线段张成的面积自然是 0，所以同项相乘为 0。

### (3) 外微分算子 $d$ 是怎么求导的？
外微分算子 $d$ 是一种把低阶形式变成高阶形式的统一微分运算：
- 对 0-形式求外微分（就是全微分）：
  $$d f(\xi, \eta) = \frac{\partial f}{\partial \xi} \, d\xi + \frac{\partial f}{\partial \eta} \, d\eta$$
- 对 1-形式 $\alpha = P \, d\xi + Q \, d\eta$ 求外微分：
  $$d\alpha = d(P \, d\xi + Q \, d\eta) = (dP) \wedge d\xi + (dQ) \wedge d\eta$$
  我们把 $dP$ 和 $dQ$ 的全微分代进去：
  $$= \left( \frac{\partial P}{\partial \xi} d\xi + \frac{\partial P}{\partial \eta} d\eta \right) \wedge d\xi + \left( \frac{\partial Q}{\partial \xi} d\xi + \frac{\partial Q}{\partial \eta} d\eta \right) \wedge d\eta$$
  利用乘法分配律展开：
  $$= \frac{\partial P}{\partial \xi} (d\xi \wedge d\xi) + \frac{\partial P}{\partial \eta} (d\eta \wedge d\xi) + \frac{\partial Q}{\partial \xi} (d\xi \wedge d\eta) + \frac{\partial Q}{\partial \eta} (d\eta \wedge d\eta)$$
  代入楔积公理：$d\xi \wedge d\xi = 0$，$d\eta \wedge d\eta = 0$，以及 $d\eta \wedge d\xi = - d\xi \wedge d\eta$：
  $$= 0 - \frac{\partial P}{\partial \eta} (d\xi \wedge d\eta) + \frac{\partial Q}{\partial \xi} (d\xi \wedge d\eta) + 0$$
  合并同类项，得到经典格林公式的核心旋度式：
  $$d\alpha = \left( \frac{\partial Q}{\partial \xi} - \frac{\partial P}{\partial \eta} \right) d\xi \wedge d\eta$$

---

## 3. 把球面 Hessian 积分包装成微分二形式

在第 $m$ 个球面多边形单元 $\Omega_m$ 上，球面面积微元在局部坐标下表示为：
$$d\Omega = J_m(\xi, \eta) \, d\xi \wedge d\eta$$
（式中 $J_m(\xi, \eta)$ 为球面投影的雅可比行列式标量）。

我们把引力梯度张量第 $(i, j)$ 个分量在这个单元上的二维被积函数定义为：
$$F_{ij, m}(\xi, \eta) = T_{ij}(\mathbf{u}(\xi, \eta)) \, J_m(\xi, \eta) \left[ \sum_{k=1}^K w_k \sum_{\ell=1}^{L_m} J_{k, m\ell}(\mathbf{u}(\xi, \eta)) \right]$$
由此，在该单元上待积的物理量构成一个严格的微分 2-形式：
$$\omega_{ij, m} = F_{ij, m}(\xi, \eta) \, d\xi \wedge d\eta$$

我们需要求解的二维面积分就是：
$$H_{ij, m} = G \iint_{\Omega_m} \omega_{ij, m} = G \iint_{\Omega_m} F_{ij, m}(\xi, \eta) \, d\xi d\eta$$

---

## 4. Poincaré 引理：寻找降维的“原一形式”

法国数学巨擘庞加莱（Henri Poincaré）提出了著名的**庞加莱引理（Poincaré Lemma）**：
> **引理内容**：在任何拓扑可收缩的区域（比如星形区域）上，任何闭微分形式都必然是恰当形式（Exact Form）。

翻译成通俗语言：
若该单元在所选坐标中是星形、且 $F_{ij,m}$ 在收缩区域内具有所需可积正则性，则二维 2-形式自动闭合（因为三维外微分在二维域上为零），并由 Poincaré 引理存在一个 1-形式 $\eta_{ij,m}$，满足：
$$d\eta_{ij, m} = \omega_{ij, m}$$

如果找到了这个 $\eta_{ij, m}$，根据流形上的**广义 Stokes 定理**：
$$\iint_{\Omega_m} \omega_{ij, m} = \iint_{\Omega_m} d\eta_{ij, m} = \oint_{\partial\Omega_m} \eta_{ij, m}$$
**看！二维曲面积分 $\iint_{\Omega_m}$ 瞬间被彻底转化为一维边界曲线积分 $\oint_{\partial\Omega_m}$！**

---

## 5. 显式同伦逆算子（Homotopy Operator）的构造与逐行严格验证

引理只证明了“存在”这样一个 1-形式，但它长什么样？怎么把它算出来？
Poincaré 给出了一个具有构造性的显式代数公式，叫作**同伦逆算子（Homotopy Inversion Operator）**。

假设局部坐标系使得单元区域相对于原点 $(0, 0)$ 是星形的。
引入一个无量纲的收缩参数 $s \in [0, 1]$：
- 当 $s = 1$ 时，点 $(s\xi, s\eta) = (\xi, \eta)$ 就是原始点；
- 当 $s = 0$ 时，点 $(s\xi, s\eta) = (0, 0)$ 收缩成了原点。
随着 $s$ 从 0 变到 1，点沿着一条直线从原点辐射到边界，这就是“同伦收缩路径”。

Poincaré 同伦算子给出的 1-形式解析构造为：
$$\eta_{ij, m}(\xi, \eta) = \left[ \int_0^1 s \, F_{ij, m}(s\xi, s\eta) \, ds \right] (\xi \, d\eta - \eta \, d\xi)$$

### 质疑：这个式子的外微分 $d\eta$，真的严格等于 $F d\xi \wedge d\eta$ 吗？
我们来做一次**没有任何跳步的求导验证**！

为了书写清晰，先令内层积分为一个辅助标量函数：
$$H(\xi, \eta) = \int_0^1 s \, F(s\xi, s\eta) \, ds$$
那么 1-形式简写为：
$$\eta = H(\xi, \eta) (\xi \, d\eta - \eta \, d\xi) = (- \eta H) \, d\xi + (\xi H) \, d\eta$$
对比标准 1-形式 $\alpha = P d\xi + Q d\eta$，这里：
$$P(\xi, \eta) = -\eta H(\xi, \eta)$$
$$Q(\xi, \eta) = \xi H(\xi, \eta)$$

根据第 2 小节证明的外微分公式，计算 $d\eta$：
$$d\eta = \left( \frac{\partial Q}{\partial \xi} - \frac{\partial P}{\partial \eta} \right) d\xi \wedge d\eta$$

我们把两个偏导数分别一步一步算出来：
1. 计算 $\frac{\partial Q}{\partial \xi}$（运用乘积求导法则）：
   $$\frac{\partial Q}{\partial \xi} = \frac{\partial}{\partial \xi} (\xi H) = \frac{\partial \xi}{\partial \xi} H + \xi \frac{\partial H}{\partial \xi} = H + \xi \frac{\partial H}{\partial \xi}$$
2. 计算 $\frac{\partial P}{\partial \eta}$：
   $$\frac{\partial P}{\partial \eta} = \frac{\partial}{\partial \eta} (-\eta H) = - \left( \frac{\partial \eta}{\partial \eta} H + \eta \frac{\partial H}{\partial \eta} \right) = - H - \eta \frac{\partial H}{\partial \eta}$$

把两个式子相减：
$$\frac{\partial Q}{\partial \xi} - \frac{\partial P}{\partial \eta} = \left( H + \xi \frac{\partial H}{\partial \xi} \right) - \left( - H - \eta \frac{\partial H}{\partial \eta} \right) = 2H + \xi \frac{\partial H}{\partial \xi} + \eta \frac{\partial H}{\partial \eta}$$

现在把 $H(\xi, \eta) = \int_0^1 s F(s\xi, s\eta) ds$ 代入后两项中：
$$\xi \frac{\partial H}{\partial \xi} + \eta \frac{\partial H}{\partial \eta} = \xi \int_0^1 s \frac{\partial}{\partial \xi}[F(s\xi, s\eta)] ds + \eta \int_0^1 s \frac{\partial}{\partial \eta}[F(s\xi, s\eta)] ds$$
根据复合函数链式法则：
$$\frac{\partial}{\partial \xi}[F(s\xi, s\eta)] = s \frac{\partial F}{\partial \xi_1}, \quad \frac{\partial}{\partial \eta}[F(s\xi, s\eta)] = s \frac{\partial F}{\partial \eta_1}$$
因此：
$$\xi \frac{\partial H}{\partial \xi} + \eta \frac{\partial H}{\partial \eta} = \int_0^1 s^2 \left( \xi \frac{\partial F}{\partial \xi_1} + \eta \frac{\partial F}{\partial \eta_1} \right) ds$$
注意！括号里的项，根据全微分定义，正好是函数关于收缩参数 $s$ 的导数：
$$\frac{d}{ds} [F(s\xi, s\eta)] = \xi \frac{\partial F}{\partial \xi_1} + \eta \frac{\partial F}{\partial \eta_1}$$
把它代回去：
$$\xi \frac{\partial H}{\partial \xi} + \eta \frac{\partial H}{\partial \eta} = \int_0^1 s^2 \frac{d}{ds}[F(s\xi, s\eta)] \, ds$$

现在对这个积分使用大一微积分的**分部积分法**（$u = s^2, dv = \frac{dF}{ds} ds$）：
$$\int_0^1 s^2 \frac{dF}{ds} \, ds = \left[ s^2 F(s\xi, s\eta) \right]_0^1 - \int_0^1 \frac{d(s^2)}{ds} F(s\xi, s\eta) \, ds$$
- 上下限代入：$1^2 F(\xi, \eta) - 0^2 F(0, 0) = F(\xi, \eta)$；
- 剩下的积分项：$\frac{d(s^2)}{ds} = 2s$，所以是 $- 2 \int_0^1 s F(s\xi, s\eta) ds$。
因此：
$$\xi \frac{\partial H}{\partial \xi} + \eta \frac{\partial H}{\partial \eta} = F(\xi, \eta) - 2 \int_0^1 s F(s\xi, s\eta) \, ds = F(\xi, \eta) - 2H$$

把这个结果代回刚才相减的总式子：
$$\frac{\partial Q}{\partial \xi} - \frac{\partial P}{\partial \eta} = 2H + \left( \xi \frac{\partial H}{\partial \xi} + \eta \frac{\partial H}{\partial \eta} \right) = 2H + [F(\xi, \eta) - 2H] \equiv F(\xi, \eta) \text{ ！！！}$$

两边的 $2H$ **分毫不差、精确抵消**，只剩下了唯一的 $F(\xi, \eta)$！
我们严格证明了：
$$d\eta_{ij, m} = F_{ij, m}(\xi, \eta) \, d\xi \wedge d\eta = \omega_{ij, m}$$
**庞加莱同伦算子成立得无懈可击！**

---

## 6. 冷静客观看待同伦算子：揭开“虚假一维”的真相

很多粗糙的文献读到这里，会兴奋地宣称：“看！我们用同伦算子把二维球面积分彻底变成了一维闭合边界线积分！”
**但作为严谨的科学工作者，我们必须揭露真相**：
我们来看把 $\eta$ 代入 Stokes 定理后得到的边界积分到底是什么：

$$\oint_{\partial\Omega_m} \eta_{ij, m} = \oint_{\partial\Omega_m} \left[ \int_0^1 s \, F_{ij, m}(s\xi, s\eta) \, ds \right] (\xi \, d\eta - \eta \, d\xi)$$

设边界曲线用参数 $t \in [0, 1]$ 参数化，坐标为 $(\xi(t), \eta(t))$。
代入进去，整个式子写成：
$$\int_0^1 \left[ \int_0^1 s \, F_{ij, m}(s\xi(t), s\eta(t)) \, ds \right] \left( \xi(t) \eta'(t) - \eta(t) \xi'(t) \right) dt$$

请看这个式子里的积分号：
**它外面有一个关于边界参数 $t$ 的积分号，里面还套着一个关于同伦收缩参数 $s$ 的积分号！**
它本质上**依然是一个关于 $(t, s)$ 的二维二重积分**！它只是把原先在多边形内部的面积分，换了一个积分变量坐标网格而已，并没有在数学上凭空变出初等解析闭式！

---

## 7. 解析解耦与残差项的诞生

若要得到无需保留二维数值求积的解析子式，可以尝试将物理被积形式 $F$ 进行**可验证的结构拆解**：
$$F_{ij, m} = F_{ij, m}^{\mathrm{ell}} + F_{ij, m}^{\mathrm{rem}}$$
1. **$F_{ij, m}^{\mathrm{ell}}$（椭圆可解析化部分）**：
   只有在给出具体代换、Jacobian、分支和反向求导验证后，才可把具有相应代数结构的子项化为 Hermite/Carlson 表达式；一般 $F_{ij,m}$ 不能直接套用该结论；
2. **$F_{ij, m}^{\mathrm{rem}}$（不可解析的剩余残差部分）**：
   无法用有限个椭圆函数封闭表达的剩余核，留在同伦双重积分里，由后续的非零残差模块（$H^{\mathrm{res}}$）通过高精度双指数算法保底计算。

---

### 第八小节小结与确认

在本节中，我们完成了一次从经典向量微积分到现代微分形式的跨越：
1. 彻底扫盲了楔积 $\wedge$ 的反对易律和外微分算子 $d$；
2. 将复杂的多面体 Hessian 球面积分规范化为微分二形式 $\omega = F d\xi \wedge d\eta$；
3. 一笔一画推导并**用分部积分法严格验证**了 Poincaré 同伦逆算子的成立条件；
4. 诚实指出了同伦算子得到的式子本质上是 $(t, s)$ 二维嵌套积分，确立了将物理量拆解为“椭圆解析部 + 同伦残差部”的必然逻辑。

---

# 第九小节：$\alpha_k = 1$ 的径向解析原函数构造

---

## 1. 为什么 $\alpha_k = 1$ 是整个理论中最关键的一役？

在上一小节中，我们确立了必须将物理量拆解为“椭圆解析部 + 同伦残差部”。
而椭圆解析部的基石，就是对广义 Cauchy 密度核中最核心的物理情形——**标准柯西衰减指数 $\alpha_k = 1$**，求出它沿射线的**严格不定积分（原函数）**。

回顾第四小节推导出的径向积分定义（当 $\alpha_k = 1$ 时）：
$$\mathcal{I}(R) = \int \frac{dR}{R \left[ 1 + \sigma_k^2 \|\mathbf{x} - R\mathbf{u} - \mathbf{c}_k\|^2 \right]}$$

这是一个看似非常简单的有理分式积分。但在过去很多引力文献中，作者往往直接甩出一句“查积分表易得”，跳过了极其关键的代数配方与对数项分离。
本小节我们将**从零开始，用纯初等微积分，把这个原函数一笔一画、毫无保留地完整推导出来！**

---

## 2. 几何参数设定与二次多项式 $D_k(R, \mathbf{u})$ 的严格展开

设相对位移纯虚四元数（三维相对位移向量）为：
$$\mathbf{p}_k = \mathbf{x} - \mathbf{c}_k = \begin{pmatrix} x_1 - c_{k,1} \\ x_2 - c_{k,2} \\ x_3 - c_{k,3} \end{pmatrix}$$
我们严格定义以下三个与积分变量 $R$ 无关的常数：

### (1) 观测点处的本底无量纲常数 $A_k$
$$A_k = 1 + \sigma_k^2 \|\mathbf{p}_k\|^2 = 1 + \sigma_k^2 \left[ (x_1 - c_{k,1})^2 + (x_2 - c_{k,2})^2 + (x_3 - c_{k,3})^2 \right]$$
注意：因为 $\sigma_k > 0$ 且范数平方非负，所以恒有 $A_k \ge 1 > 0$。

### (2) 射线方向投影标量 $b_k(\mathbf{u})$
$$b_k(\mathbf{u}) = \mathbf{p}_k \cdot \mathbf{u} = (x_1 - c_{k,1})u_1 + (x_2 - c_{k,2})u_2 + (x_3 - c_{k,3})u_3$$

### (3) 横向有效尺度平方 $d_k^2(\mathbf{u})$
$$d_k^2(\mathbf{u}) = \frac{1}{\sigma_k^2} + \|\mathbf{p}_k\|^2 - b_k^2(\mathbf{u})$$
根据我们在第四小节证明的勾股定理 $\|\mathbf{p}_k\|^2 - b_k^2 = \|\mathbf{p}_k \times \mathbf{u}\|^2 \ge 0$，所以恒有 $d_k^2(\mathbf{u}) \ge \frac{1}{\sigma_k^2} > 0$。

---

### (4) 展开分母中的二次多项式
现在看被积函数分母中除以 $R$ 之外的那个核心代数多项式，记作 $D_k(R, \mathbf{u})$：
$$D_k(R, \mathbf{u}) = 1 + \sigma_k^2 \|\mathbf{p}_k - R\mathbf{u}\|^2$$
利用向量内积展开范数平方：
$$\|\mathbf{p}_k - R\mathbf{u}\|^2 = \mathbf{p}_k \cdot \mathbf{p}_k - 2R(\mathbf{p}_k \cdot \mathbf{u}) + R^2 (\mathbf{u} \cdot \mathbf{u}) = \|\mathbf{p}_k\|^2 - 2 b_k(\mathbf{u}) R + R^2$$
代入 $D_k$：
$$D_k(R, \mathbf{u}) = 1 + \sigma_k^2 \left( \|\mathbf{p}_k\|^2 - 2 b_k(\mathbf{u}) R + R^2 \right) = \left( 1 + \sigma_k^2 \|\mathbf{p}_k\|^2 \right) - 2\sigma_k^2 b_k(\mathbf{u}) R + \sigma_k^2 R^2$$
把刚才定义的常数 $A_k$ 代入：
$$D_k(R, \mathbf{u}) = A_k - 2\sigma_k^2 b_k(\mathbf{u}) R + \sigma_k^2 R^2$$

这是一个按 $R$ 的降幂排列的完美一元二次多项式：
$$D_k(R, \mathbf{u}) = \sigma_k^2 R^2 - 2\sigma_k^2 b_k R + A_k$$
当 $R = 0$ 时，显然有：
$$D_k(0, \mathbf{u}) = A_k$$

---

## 3. 待定系数法做部分分式分解（一步一步解代数方程）

我们现在要算的不定积分是被积有理分式：
$$f(R) = \frac{1}{R \, D_k(R, \mathbf{u})} = \frac{1}{R (\sigma_k^2 R^2 - 2\sigma_k^2 b_k R + A_k)}$$

分母由一个一次因式 $R$ 和一个二次不可约因式 $D_k(R)$ 乘积构成。
根据大一有理函数积分法则，设定待定系数 $C_1, C_2, C_3 \in \mathbb{R}$：
$$\frac{1}{R \, D_k(R)} = \frac{C_1}{R} + \frac{C_2 R + C_3}{D_k(R)}$$

为了求出这三个系数，等式两边同时乘以公分母 $R \cdot D_k(R)$：
$$1 = C_1 D_k(R) + (C_2 R + C_3) R$$
把 $D_k(R) = \sigma_k^2 R^2 - 2\sigma_k^2 b_k R + A_k$ 代入：
$$1 = C_1 (\sigma_k^2 R^2 - 2\sigma_k^2 b_k R + A_k) + C_2 R^2 + C_3 R$$
把右边的所有乘法完全乘开：
$$1 = C_1 \sigma_k^2 R^2 - 2 C_1 \sigma_k^2 b_k R + C_1 A_k + C_2 R^2 + C_3 R$$
按照自变量 $R$ 的幂次（$R^2$ 项、$R^1$ 项、$R^0$ 常数项）完全合并同类项：
$$1 = (C_1 \sigma_k^2 + C_2) R^2 + (- 2 C_1 \sigma_k^2 b_k + C_3) R + (C_1 A_k)$$

因为这个等式对于任意实数 $R$ 必须恒成立，两边对应幂次的系数必须严格相等！
对比等号左边的 $1 = 0 \cdot R^2 + 0 \cdot R + 1$，建立三元一次线性方程组：

$$\begin{cases} R^2 \text{ 的系数：} & C_1 \sigma_k^2 + C_2 = 0 \quad \text{—— (方程 1)} \\ R^1 \text{ 的系数：} & - 2 C_1 \sigma_k^2 b_k + C_3 = 0 \quad \text{—— (方程 2)} \\ R^0 \text{ 的系数：} & C_1 A_k = 1 \quad \text{—— (方程 3)} \end{cases}$$

### 解方程组（绝不跳步）：
1. 由方程 3，直接解出 $C_1$：
   $$C_1 = \frac{1}{A_k}$$
2. 把 $C_1$ 代入方程 1，解出 $C_2$：
   $$C_2 = - C_1 \sigma_k^2 = - \frac{\sigma_k^2}{A_k}$$
3. 把 $C_1$ 代入方程 2，解出 $C_3$：
   $$C_3 = 2 C_1 \sigma_k^2 b_k = \frac{2\sigma_k^2 b_k}{A_k}$$

把解出来的三个系数带回部分分式中：
$$\frac{1}{R \, D_k(R)} = \frac{1}{A_k} \frac{1}{R} + \frac{-\frac{\sigma_k^2}{A_k} R + \frac{2\sigma_k^2 b_k}{A_k}}{D_k(R)}$$
提取全局公因式 $\frac{1}{A_k}$：
$$\frac{1}{R \, D_k(R)} = \frac{1}{A_k} \left[ \frac{1}{R} - \frac{\sigma_k^2 R - 2\sigma_k^2 b_k}{D_k(R)} \right]$$

---

## 4. 凑微分法与逐项求不定积分

现在被积函数被漂亮地拆成了两大部分，我们两边求不定积分：
$$\int \frac{dR}{R \, D_k(R)} = \frac{1}{A_k} \left[ \int \frac{1}{R} \, dR - \int \frac{\sigma_k^2 R - 2\sigma_k^2 b_k}{D_k(R)} \, dR \right]$$

### (1) 第一项积分：
$$\int \frac{1}{R} \, dR = \log R$$

---

### (2) 第二项积分的分子配凑
看第二项积分：
$$\int \frac{\sigma_k^2 R - 2\sigma_k^2 b_k}{D_k(R)} \, dR$$
我们先对分母 $D_k(R) = \sigma_k^2 R^2 - 2\sigma_k^2 b_k R + A_k$ 求导：
$$D_k'(R) = \frac{d}{dR}[D_k(R)] = 2\sigma_k^2 R - 2\sigma_k^2 b_k$$
观察我们积分式子里的分子是：$\sigma_k^2 R - 2\sigma_k^2 b_k$。
我们把它强行凑出分母的导数 $D_k'(R)$：
$$\sigma_k^2 R - 2\sigma_k^2 b_k = \frac{1}{2}(2\sigma_k^2 R - 2\sigma_k^2 b_k) - \sigma_k^2 b_k = \frac{1}{2} D_k'(R) - \sigma_k^2 b_k$$
代回积分式中拆成两项：
$$\int \frac{\sigma_k^2 R - 2\sigma_k^2 b_k}{D_k(R)} \, dR = \frac{1}{2} \int \frac{D_k'(R)}{D_k(R)} \, dR - \sigma_k^2 b_k \int \frac{dR}{D_k(R)}$$

#### 攻克子项 A（对数导数）：
$$\frac{1}{2} \int \frac{D_k'(R)}{D_k(R)} \, dR = \frac{1}{2} \int \frac{d(D_k(R))}{D_k(R)} = \frac{1}{2} \log D_k(R)$$

#### 攻克子项 B（反三角函数）：
看后半截积分：
$$\sigma_k^2 b_k \int \frac{dR}{D_k(R)}$$
利用第四小节推导的配方恒等式：$D_k(R) = \sigma_k^2 \left[ (R - b_k)^2 + d_k^2 \right]$，代入分母：
$$\sigma_k^2 b_k \int \frac{dR}{\sigma_k^2 \left[ (R - b_k)^2 + d_k^2 \right]} = b_k \int \frac{dR}{(R - b_k)^2 + d_k^2}$$
分子分母同时除以常数 $d_k^2$：
$$= \frac{b_k}{d_k^2} \int \frac{dR}{\left( \frac{R - b_k}{d_k} \right)^2 + 1} = \frac{b_k}{d_k} \int \frac{d\left( \frac{R - b_k}{d_k} \right)}{\left( \frac{R - b_k}{d_k} \right)^2 + 1}$$
根据经典高数积分公式 $\int \frac{du}{u^2 + 1} = \arctan u$：
$$= \frac{b_k}{d_k} \arctan\left( \frac{R - b_k}{d_k} \right)$$

---

## 5. 径向解析原函数 $P_k(R, \mathbf{u})$ 的终极封闭式

把上面所有的分块积分完完整整拼回去，注意各项的正负号（第二项前面自带一个负号）：
$$\int \frac{dR}{R \, D_k(R)} = \frac{1}{A_k} \left[ \log R - \left( \frac{1}{2}\log D_k(R) - \frac{b_k}{d_k}\arctan\left(\frac{R - b_k}{d_k}\right) \right) \right]$$
把负号分配进括号：
$$\int \frac{dR}{R \, D_k(R)} = \frac{1}{A_k} \left[ \log R - \frac{1}{2}\log D_k(R) + \frac{b_k}{d_k}\arctan\left(\frac{R - b_k}{d_k}\right) \right]$$

我们正式将这个举世瞩目的解析式定义为**单核径向原函数 $P_k(R, \mathbf{u})$**：

$$P_k(R, \mathbf{u}) = \frac{1}{A_k} \left[ \log R - \frac{1}{2}\log D_k(R, \mathbf{u}) + \frac{b_k(\mathbf{u})}{d_k(\mathbf{u})} \arctan\left( \frac{R - b_k(\mathbf{u})}{d_k(\mathbf{u})} \right) \right]$$

---

## 6. 严密的反向求导验算（绝不存疑）

为了彻底验证这个原函数没有哪怕一个正负号的差错，我们对 $P_k(R)$ 沿着自变量 $R$ 求一次全导数 $\frac{d}{dR} P_k(R)$：

1. 对 $\log R$ 求导：
   $$\frac{d}{dR}(\log R) = \frac{1}{R}$$
2. 对 $-\frac{1}{2}\log D_k(R)$ 求导：
   $$-\frac{1}{2} \frac{D_k'(R)}{D_k(R)} = -\frac{1}{2} \frac{2\sigma_k^2 R - 2\sigma_k^2 b_k}{D_k(R)} = -\frac{\sigma_k^2 R - \sigma_k^2 b_k}{D_k(R)}$$
3. 对 $\frac{b_k}{d_k}\arctan\left(\frac{R - b_k}{d_k}\right)$ 求导：
   $$\frac{b_k}{d_k} \cdot \frac{1}{1 + \left(\frac{R - b_k}{d_k}\right)^2} \cdot \frac{1}{d_k} = \frac{b_k}{d_k^2 + (R - b_k)^2}$$
   因为 $D_k(R) = \sigma_k^2 [(R - b_k)^2 + d_k^2]$，所以 $(R - b_k)^2 + d_k^2 = \frac{D_k(R)}{\sigma_k^2}$，代进去：
   $$= \frac{b_k}{\frac{D_k(R)}{\sigma_k^2}} = \frac{\sigma_k^2 b_k}{D_k(R)}$$

把这三项加起来，并乘以前面的 $\frac{1}{A_k}$：
$$\frac{d P_k}{dR} = \frac{1}{A_k} \left[ \frac{1}{R} - \frac{\sigma_k^2 R - \sigma_k^2 b_k}{D_k(R)} + \frac{\sigma_k^2 b_k}{D_k(R)} \right] = \frac{1}{A_k} \left[ \frac{1}{R} - \frac{\sigma_k^2 R - 2\sigma_k^2 b_k}{D_k(R)} \right]$$
通分：
$$= \frac{1}{A_k} \left[ \frac{D_k(R) - R(\sigma_k^2 R - 2\sigma_k^2 b_k)}{R \, D_k(R)} \right]$$
把分子乘开并代入 $D_k(R) = \sigma_k^2 R^2 - 2\sigma_k^2 b_k R + A_k$：
$$D_k(R) - \sigma_k^2 R^2 + 2\sigma_k^2 b_k R = (\sigma_k^2 R^2 - 2\sigma_k^2 b_k R + A_k) - \sigma_k^2 R^2 + 2\sigma_k^2 b_k R \equiv A_k$$
所有的 $R^2$ 项和 $R$ 项**分毫不差、精确抵消**！只剩下一个纯常数 $A_k$！
代回通分式：
$$\frac{d P_k}{dR} = \frac{1}{A_k} \left[ \frac{A_k}{R \, D_k(R)} \right] \equiv \frac{1}{R \, D_k(R)}$$
**求导验算绝对吻合！原函数完美无缺！**

---

## 7. 两个分支区间的物理定积分求值

有了这个无懈可击的原函数，我们对光线穿透实体的任意线段区间进行定积分求值，严格划分为两大分支：

### 分支 1：普通远离区间（$R^- > 0$）
此时上下限都是良态正数，直接应用牛顿-莱布尼茨公式作差：
$$J_{k, m\ell}(\mathbf{u}) = \int_{R^-}^{R^+} \frac{dR}{R \, D_k(R)} = P_k(R^+, \mathbf{u}) - P_k(R^-, \mathbf{u})$$
完全是纯代数计算，没有任何数值误差！

---

### 分支 2：接触零点的奇异区间（$R^- = 0, R^+ = L$）
当场点在表面或内部时，下限为 0，$\log R$ 发散。
根据第六小节建立的 **Hadamard 有限部分（$\mathrm{F.p.}$）法则**：
$$J_k^{\mathrm{FP}}(0, L) \equiv \lim_{\varepsilon \to 0^+} \left[ \int_\varepsilon^L \frac{dR}{R \, D_k(R)} - \frac{1}{A_k} \log\frac{\ell_0}{\varepsilon} \right]$$
我们把原函数在上下限展开：
$$\int_\varepsilon^L \frac{dR}{R \, D_k(R)} = P_k(L, \mathbf{u}) - P_k(\varepsilon, \mathbf{u})$$

在下限 $\varepsilon \to 0$ 处观察 $P_k(\varepsilon)$：
$$P_k(\varepsilon) = \frac{1}{A_k} \left[ \log\varepsilon - \frac{1}{2}\log D_k(\varepsilon) + \frac{b_k}{d_k}\arctan\left(\frac{\varepsilon - b_k}{d_k}\right) \right]$$
因为当 $\varepsilon \to 0$ 时，$D_k(\varepsilon) \to D_k(0) = A_k$，且 $\arctan\left(\frac{0 - b_k}{d_k}\right) = \arctan\left(-\frac{b_k}{d_k}\right) = -\arctan\left(\frac{b_k}{d_k}\right)$。
代入减法极限中，$\log\varepsilon$ 与 $-\log\frac{\ell_0}{\varepsilon} = \log\varepsilon - \log\ell_0$ **精确对消**！

最终得到**接触零点时的 Hadamard 有限部分解析闭式**：

$$J_k^{\mathrm{FP}}(0, L) = \frac{1}{A_k} \left[ \log\frac{L}{\ell_0} - \frac{1}{2}\log\frac{D_k(L)}{A_k} + \frac{b_k}{d_k} \left( \arctan\frac{L - b_k}{d_k} + \arctan\frac{b_k}{d_k} \right) \right]$$

> **特别提醒（工程避坑）**：
> 注意最后那两项 $\arctan$ 的加号！
> 因为在下限处，原本是 $-\arctan(-\frac{b_k}{d_k})$，负负得正，变成了 $+\arctan(\frac{b_k}{d_k})$！
> 在 GPU 或 C++ 代码中，必须使用 `atan2` 函数来保证全象限幅值的严格连续。

---

### 第九小节小结与确认

在本节中，我们完成了一次纯数学上的完美攻坚：
1. 一步不漏地用待定系数法解出三元一次方程组，得到了严格的部分分式分解；
2. 逐项凑微分，推导出了包含对数和反正切的单核径向原函数 $P_k(R, \mathbf{u})$；
3. **通过完全求导反算**，100% 验证了原函数的绝对正确性；
4. 明确给出了远离区间作差公式，以及利用 Hadamard 规则在 $R=0$ 处消去对数发散后的有限部分解析闭式。

---

# 第十小节：球面边界大圆弧复参数化与四次代数曲线

---

## 1. 承上启下的核心矛盾：这个反正切项怎么积？

在第九小节中，我们通过部分分式分解，推导出了标准柯西核（$\alpha_k = 1$）在单条射线上的严格原函数：
$$P_k(R, \mathbf{u}) = \frac{1}{A_k} \left[ \log R - \frac{1}{2}\log D_k(R, \mathbf{u}) + \frac{b_k(\mathbf{u})}{d_k(\mathbf{u})} \arctan\left( \frac{R - b_k(\mathbf{u})}{d_k(\mathbf{u})} \right) \right]$$

而在第七和第八小节中，我们证明了：整个天体的引力梯度，最终会被压到各个可见性单元的多边形边界 $\partial\Omega_m$ 上。
多边形的每一条边，都是单位球面 $\mathbb{S}^2$ 上的一段**大圆弧（Great Circle Arc）**。
这就意味着，我们要沿着这一段大圆弧，对上面的原函数做一维线积分：
$$\int_{\text{大圆弧}} P_k(R(\mathbf{u}), \mathbf{u}) \, ds$$

请大家死死盯住原函数里最硬的那块骨头：
$$\frac{b_k(\mathbf{u})}{d_k(\mathbf{u})} \arctan\left( \frac{R(\mathbf{u}) - b_k(\mathbf{u})}{d_k(\mathbf{u})} \right)$$
- 分子有投影量 $b_k(\mathbf{u})$；
- 分母有横向有效尺度 $d_k(\mathbf{u}) = \sqrt{a_k^2 + \|\mathbf{p}_k\|^2 - b_k^2(\mathbf{u})}$；
- 括号里还套着一个反正切函数 $\arctan$！

当方向 $\mathbf{u}$ 沿着大圆弧连续滑动时，这些几何量会怎么变？这到底是一个什么级别的数学积分？
本小节我们将通过**复平面欧拉代换**，彻底揭开它的真面目——**它在复数空间中精确诱导了一条亏格为 1 的代数椭圆曲线！**

---

## 2. 大圆弧的正交基底与实三角参数化

在三维空间中，任何一条大圆弧必然位于一个穿过坐标原点 $\mathbf{0}$ 的二维平面内。
在这个大圆弧所在的平面上，我们选取两个互相垂直的单位基底向量 $\mathbf{e}_1, \mathbf{e}_2 \in \mathbb{R}^3$，它们满足：
$$\|\mathbf{e}_1\| = \sqrt{e_{1,1}^2 + e_{1,2}^2 + e_{1,3}^2} = 1$$
$$\|\mathbf{e}_2\| = \sqrt{e_{2,1}^2 + e_{2,2}^2 + e_{2,3}^2} = 1$$
$$\mathbf{e}_1 \cdot \mathbf{e}_2 = e_{1,1}e_{2,1} + e_{1,2}e_{2,2} + e_{1,3}e_{2,3} = 0$$

设大圆弧的弧长角度参数为 $t \in [t_0, t_1]$。
圆弧上任意一点的三维单位方向矢量 $\mathbf{u}(t)$ 可以严格用正弦和余弦参数化：
$$\mathbf{u}(t) = \cos t \, \mathbf{e}_1 + \sin t \, \mathbf{e}_2$$
我们验证一下它是不是单位矢量：
$$\|\mathbf{u}(t)\|^2 = (\cos t \, \mathbf{e}_1 + \sin t \, \mathbf{e}_2) \cdot (\cos t \, \mathbf{e}_1 + \sin t \, \mathbf{e}_2)$$
$$= \cos^2 t (\mathbf{e}_1 \cdot \mathbf{e}_1) + 2 \cos t \sin t (\mathbf{e}_1 \cdot \mathbf{e}_2) + \sin^2 t (\mathbf{e}_2 \cdot \mathbf{e}_2)$$
代入基底的正交归一性（$\mathbf{e}_1 \cdot \mathbf{e}_1 = 1, \mathbf{e}_1 \cdot \mathbf{e}_2 = 0, \mathbf{e}_2 \cdot \mathbf{e}_2 = 1$）：
$$= \cos^2 t (1) + 0 + \sin^2 t (1) = \cos^2 t + \sin^2 t \equiv 1$$
**模长恒为 1，完全合法！**

---

## 3. 投影标量 $b_k(t)$ 的实三角展开

回顾第四小节中投影标量 $b_k(\mathbf{u})$ 的定义（相对位移 $\mathbf{p}_k = \mathbf{x} - \mathbf{c}_k$ 在方向 $\mathbf{u}$ 上的投影）：
$$b_k(\mathbf{u}) = \mathbf{p}_k \cdot \mathbf{u}$$
把 $\mathbf{u}(t) = \cos t \, \mathbf{e}_1 + \sin t \, \mathbf{e}_2$ 代进去：
$$b_k(t) = \mathbf{p}_k \cdot (\cos t \, \mathbf{e}_1 + \sin t \, \mathbf{e}_2) = (\mathbf{p}_k \cdot \mathbf{e}_1) \cos t + (\mathbf{p}_k \cdot \mathbf{e}_2) \sin t$$

定义两个与参数 $t$ 无关的几何常数：
$$\beta_{1k} = \mathbf{p}_k \cdot \mathbf{e}_1 = (x_1 - c_{k,1})e_{1,1} + (x_2 - c_{k,2})e_{1,2} + (x_3 - c_{k,3})e_{1,3}$$
$$\beta_{2k} = \mathbf{p}_k \cdot \mathbf{e}_2 = (x_1 - c_{k,1})e_{2,1} + (x_2 - c_{k,2})e_{2,2} + (x_3 - c_{k,3})e_{2,3}$$
对于本文的大圆弧参数化，$\mathbf u(t)=\mathbf e_1\cos t+\mathbf e_2\sin t$ 且 $\mathbf e_1,\mathbf e_2$ 张成过原点的平面，所以严格有 $\beta_{0k}=0$；因此本问题中
$$b_k(t)=\beta_{1k}\cos t+\beta_{2k}\sin t.$$
若另行研究不经过原点的小圆或仿射参数曲线，才可引入独立的常数项 $\beta_{0k}$；那是不同几何问题，不能混入当前大圆弧推导。

---

## 4. 欧拉复代数化（Euler Mapping $z = e^{it}$）的完全展开

实数域下的三角函数 $\cos t$ 和 $\sin t$ 是超越函数，极难处理它们与根号、分式的复合积分。
我们要用数学史上最伟大的公式——**欧拉公式（Euler's Formula）**，把它映射到复平面：
$$z = e^{it} = \cos t + i \sin t$$
则它的倒数是共轭位置：
$$z^{-1} = \frac{1}{z} = e^{-it} = \cos t - i \sin t$$

把两式相加：
$$z + z^{-1} = 2 \cos t \implies \cos t = \frac{1}{2} (z + z^{-1}) = \frac{z^2 + 1}{2z}$$
把两式相减：
$$z - z^{-1} = 2i \sin t \implies \sin t = \frac{1}{2i} (z - z^{-1}) = \frac{z^2 - 1}{2iz} = -i \frac{z^2 - 1}{2z}$$
微分微元 $dt$ 的变换：
$$dz = d(e^{it}) = i e^{it} dt = i z \, dt \implies dt = \frac{dz}{iz}$$

所有的实数正弦、余弦以及微分微元 $dt$，**全部变成了关于复变量 $z$ 的纯粹有理代数分式！**

---

## 5. 投影量 $b_k(t)$ 化为复 Laurent 多项式

现在把 $\cos t$ 和 $\sin t$ 的复数展开代入 $b_k(t)$：
$$b_k(z) = \beta_{0k} + \beta_{1k} \left[ \frac{1}{2}(z + z^{-1}) \right] + \beta_{2k} \left[ \frac{1}{2i}(z - z^{-1}) \right]$$
注意 $\frac{1}{i} = -i$：
$$= \beta_{0k} + \frac{1}{2}\beta_{1k} z + \frac{1}{2}\beta_{1k} z^{-1} - \frac{i}{2}\beta_{2k} z + \frac{i}{2}\beta_{2k} z^{-1}$$
按照 $z$ 的幂次（$z^1, z^0, z^{-1}$）合并：
$$= \left( \frac{\beta_{1k} - i\beta_{2k}}{2} \right) z + \beta_{0k} + \left( \frac{\beta_{1k} + i\beta_{2k}}{2} \right) z^{-1}$$

为了书写极度规整，定义一个复常数 $\gamma_k$ 及其复共轭 $\bar{\gamma}_k$：
$$\gamma_k = \frac{\beta_{1k} - i\beta_{2k}}{2}, \quad \bar{\gamma}_k = \frac{\beta_{1k} + i\beta_{2k}}{2}$$
代进去：
$$b_k(z) = \gamma_k z + \beta_{0k} + \frac{\bar{\gamma}_k}{z}$$
通分，提取分母 $z$：
$$b_k(z) = \frac{\gamma_k z^2 + \beta_{0k} z + \bar{\gamma}_k}{z} \equiv \frac{B_{2,k}(z)}{z}$$

式中分子 $B_{2,k}(z)$ 是一个标准的一元二次多项式：
$$B_{2,k}(z) = \gamma_k z^2 + \beta_{0k} z + \bar{\gamma}_k$$

---

## 6. 横向尺度 $d_k^2(t)$ 与四次多项式 $P_{4,k}(z)$ 的完全逐项展开

回顾横向有效尺度平方的定义：
$$d_k^2(\mathbf{u}) = a_k^2 + \|\mathbf{p}_k\|^2 - b_k^2(\mathbf{u})$$
定义不随角度变化的常数部分为 $M_k$：
$$M_k = a_k^2 + \|\mathbf{p}_k\|^2 = \frac{1}{\sigma_k^2} + \|\mathbf{p}_k\|^2 > 0$$
则横向尺度平方写为：
$$d_k^2(t) = M_k - b_k^2(t)$$

现在把刚刚算出的 $b_k(z) = \frac{B_{2,k}(z)}{z}$ 代入进去：
$$d_k^2(z) = M_k - \left( \frac{B_{2,k}(z)}{z} \right)^2 = M_k - \frac{[B_{2,k}(z)]^2}{z^2}$$
通分：
$$d_k^2(z) = \frac{M_k z^2 - [B_{2,k}(z)]^2}{z^2}$$

现在我们把分子中的二次多项式平方 $[B_{2,k}(z)]^2$ 一项一项完全平方乘开：
$$[B_{2,k}(z)]^2 = (\gamma_k z^2 + \beta_{0k} z + \bar{\gamma}_k)^2$$
利用三项和的完全平方公式 $(a+b+c)^2 = a^2 + b^2 + c^2 + 2ab + 2bc + 2ca$：
$$= (\gamma_k z^2)^2 + (\beta_{0k} z)^2 + (\bar{\gamma}_k)^2 + 2(\gamma_k z^2)(\beta_{0k} z) + 2(\beta_{0k} z)(\bar{\gamma}_k) + 2(\gamma_k z^2)(\bar{\gamma}_k)$$
按 $z$ 的降幂（$z^4, z^3, z^2, z^1, z^0$）整理：
$$= \gamma_k^2 z^4 + 2\gamma_k \beta_{0k} z^3 + (\beta_{0k}^2 + 2\gamma_k \bar{\gamma}_k) z^2 + 2\beta_{0k} \bar{\gamma}_k z + \bar{\gamma}_k^2$$
注意：$\gamma_k \bar{\gamma}_k = \left(\frac{\beta_{1k}-i\beta_{2k}}{2}\right)\left(\frac{\beta_{1k}+i\beta_{2k}}{2}\right) = \frac{\beta_{1k}^2 + \beta_{2k}^2}{4} = |\gamma_k|^2$。

把这个完全展开式代回分子的 $M_k z^2 - [B_{2,k}(z)]^2$：
$$M_k z^2 - [B_{2,k}(z)]^2 = M_k z^2 - \left[ \gamma_k^2 z^4 + 2\gamma_k \beta_{0k} z^3 + (\beta_{0k}^2 + 2|\gamma_k|^2) z^2 + 2\beta_{0k} \bar{\gamma}_k z + \bar{\gamma}_k^2 \right]$$
把所有负号分配进去，并按 $z$ 的降幂彻底合并同类项：
$$= - \gamma_k^2 z^4 - 2\gamma_k \beta_{0k} z^3 + \left[ M_k - (\beta_{0k}^2 + 2|\gamma_k|^2) \right] z^2 - 2\beta_{0k} \bar{\gamma}_k z - \bar{\gamma}_k^2$$

我们正式将这个分子定义为**核心四次多项式 $P_{4,k}(z)$**：

$$P_{4,k}(z) \equiv - \gamma_k^2 z^4 - 2\gamma_k \beta_{0k} z^3 + \left[ M_k - \beta_{0k}^2 - 2|\gamma_k|^2 \right] z^2 - 2\beta_{0k} \bar{\gamma}_k z - \bar{\gamma}_k^2$$

由此，$d_k^2(z)$ 具有完美的代数紧凑分式形式：
$$d_k^2(z) = \frac{P_{4,k}(z)}{z^2}$$
两边开平方根，就解出了横向有效尺度的复数解析式：
沿选定的连续平方根分支，$d_k(z)=y(z)/z$，其中 $y(z)^2=P_{4,k}(z)$；这个等式不是对复平方根主值的无条件恒等式，换支会同时改变 $y$ 的符号。

---

## 7. 代数曲线 $\mathcal{E}_k : y^2 = P_{4,k}(z)$ 与亏格（Genus）判定

请看上面的分母是 $z$，分子是一个带有**四次多项式开平方根**的代数无理式：
$$\sqrt{P_{4,k}(z)}$$

在代数几何与复分析中，引入一个复变量 $y$，令
$$y^2=P_{4,k}(z).$$
该方程先定义复仿射代数曲线；只有在无穷远处作光滑射影紧化后，才得到用于亏格计算的紧黎曼面。若 $P_{4,k}$ 有重根，仿射曲线还可能有奇点，必须先正规化。

### 这条曲线的拓扑亏格（Genus）是多少？
根据代数拓扑与黎曼面理论的 **Riemann-Hurwitz 亏格公式**：
对于一条由 $y^2 = P_n(z)$ 定义的二叶分歧覆盖黎曼面，若多项式的次数为 $n$，其紧化后的拓扑亏格 $g$（曲面上的“洞/环柄”个数）严格由下式给出：
$$g = \left\lfloor \frac{n - 1}{2} \right\rfloor$$
在这里，多项式的最高次数是 $n = 4$！代入公式：
$$g = \left\lfloor \frac{4 - 1}{2} \right\rfloor = \left\lfloor \frac{3}{2} \right\rfloor = 1$$

---

## 8. 代数几何严格分类讨论

四次多项式 $P_{4,k}(z) = 0$ 在复平面上有 4 个复数根，记作 $e_1, e_2, e_3, e_4$。
四次多项式的判别式定义为：
$$\mathrm{Disc}(P_{4,k}) = \prod_{1 \le i < j \le 4} (e_i - e_j)^2$$

这里产生严格的**代数曲线几何分类**：

### 分类 A：非退化情形（$\mathrm{Disc}(P_{4,k}) \neq 0$）
若 4 个根互不相同（无重根，平方自由 Square-Free）：
- 曲线没有任何奇点，光滑无瑕；
- 其拓扑结构是严格的**亏格 1 黎曼面（复环面 Torus，即甜甜圈形状）**；
- **在数学上，这就是严格的“椭圆曲线（Elliptic Curve）”！**
这意味着：沿大圆弧的所有后续积分，本质上全部是定义在这个亏格 1 椭圆曲线上的**第一、第二、第三类 Abel 椭圆积分**！

### 分类 B：退化情形（$\mathrm{Disc}(P_{4,k}) = 0$）
若存在重根（例如有两个根重合，多项式包含 $(z - z_0)^2$ 因式）：
- 平方根可以部分开出来：$\sqrt{(z - z_0)^2 Q_2(z)} = (z - z_0)\sqrt{Q_2(z)}$；
- 四次多项式退化为二次多项式开方；
- 亏格从 $g = 1$ 骤降为 $g = 0$（曲线退化为复球面）；
- 此时不需要高阶椭圆函数，积分可完全退化为初等反三角函数与普通对数函数！

---

### 第十小节小结与确认

在本节中，我们完成了一次震撼人心的代数升级：
1. 建立了大圆弧的正交归一基底 $\mathbf{e}_1, \mathbf{e}_2$ 与实三角参数化；
2. 运用 Euler 变换 $z = e^{it}$，将超越三角函数彻底转化为纯代数 Laurent 多项式；
3. **一步一步展开了二次多项式的平方**，完全导出了四次多项式 $P_{4,k}(z)$ 的 5 个系数；
4. 在固定连续平方根分支后写出 $d_k(z)=y(z)/z$，并说明主值平方根不能无条件替代路径上的分支；
5. 说明了：当 $P_{4,k}$ 平方自由并完成射影紧化时，曲线为亏格 $1$；重根时必须先正规化，不能直接套用亏格 $1$ 结论。

---

# 第十一小节：分部积分法（IBP）消去反正切超越项

---

## 1. 为什么反正切项是解析计算的“头号拦路虎”？

在第九小节和第十小节中，我们在大圆弧边界上推导出了核心积分形式。在同伦参数化或边界角向积分中，典型的数学形式表现为一个关于参数 $t \in [0, 1]$ 的积分：

$$I = \int_0^1 t \, \Phi(t) \, dt$$

式中被积核心角函数为：
$$\Phi(t) = \arctan\left( \frac{N(t)}{\sqrt{P_4(t)}} \right)$$
这里面包含了两个完全不同维度的数学对象：
1. **代数无理式 $\sqrt{P_4(t)}$**：我们在第十小节证明了它定义了一个亏格 1 的代数椭圆曲线；
2. **超越函数 $\arctan$**：反三角函数根本不在代数几何的范畴内！

在积分表中，经典的**椭圆积分（Elliptic Integrals）**只能处理形如 $\int \frac{R(t)}{\sqrt{P_4(t)}} dt$ 的**代数有理分式**，世界上没有任何一本椭圆积分手册能够直接查出“外面套着一个反正切函数”的解析闭式！
如果不能干掉 $\arctan$，我们前面建立的整个代数曲线大厦就完全派不上用场。

**数学家的惊天妙招**：
利用微积分中的**分部积分法（Integration By Parts, IBP）**，把积分算子从外面移走，直接对反正切求导！
求导之后，我们将见证数学史上最不可思议的代数对消！

---

## 2. 分部积分法（IBP）的骨架搭建

大一微积分的分部积分公式：
$$\int u \, dv = u v - \int v \, du$$

对于我们的积分 $I = \int_0^1 t \, \Phi(t) \, dt$，我们精准设定两部分：
- 令 $u(t) = \Phi(t) = \arctan\left( \frac{N(t)}{\sqrt{P_4(t)}} \right)$；
- 令 $dv = t \, dt \implies v(t) = \int t \, dt = \frac{1}{2} t^2$。

直接套用分部积分定理，把积分写成边界项与新积分之差：
$$I = \int_0^1 \Phi(t) \, d\left( \frac{1}{2} t^2 \right) = \left[ \frac{1}{2} t^2 \Phi(t) \right]_0^1 - \int_0^1 \frac{1}{2} t^2 \, d(\Phi(t))$$

### 计算第一项（代数边界项）：
代入上限 $t = 1$ 与下限 $t = 0$：
$$\left[ \frac{1}{2} t^2 \Phi(t) \right]_0^1 = \frac{1}{2} (1)^2 \Phi(1) - \frac{1}{2} (0)^2 \Phi(0) = \frac{1}{2} \Phi(1) - 0 = \frac{1}{2} \Phi(1)$$
式中 $\Phi(1) = \arctan\left( \frac{N(1)}{\sqrt{P_4(1)}} \right)$ 是一个完全确定的常数端点值，不需要积任何分！

将边界项代回去，原积分变成：
$$I = \frac{1}{2} \Phi(1) - \frac{1}{2} \int_0^1 t^2 \Phi'(t) \, dt$$
现在的唯一焦点，全部落在导数 $\Phi'(t) = \frac{d\Phi}{dt}$ 上！

---

## 3. 奇迹时刻：求导 $\Phi'(t)$ 的完全展开与震撼对消

我们来求 $\Phi(t) = \arctan\left( \frac{N(t)}{\sqrt{P_4(t)}} \right)$ 关于自变量 $t$ 的一阶导数。

引入内层复合变量：
$$Y(t) = \frac{N(t)}{\sqrt{P_4(t)}}$$
则 $\Phi(t) = \arctan(Y(t))$。
根据高等数学复合函数求导链式法则：
$$\Phi'(t) = \frac{d}{dt}[\arctan(Y(t))] = \frac{1}{1 + Y(t)^2} \cdot Y'(t)$$

我们把这个式子拆成两个零件，一步一步算：

---

### 零件 1：计算内层导数 $Y'(t)$（商法则求导）
$$Y(t) = \frac{N(t)}{[P_4(t)]^{1/2}}$$
根据商法则：
$$Y'(t) = \frac{N'(t) [P_4(t)]^{1/2} - N(t) \frac{d}{dt}\left([P_4(t)]^{1/2}\right)}{\left( [P_4(t)]^{1/2} \right)^2}$$

求分母根式的导数（复合链式法则）：
$$\frac{d}{dt}\left([P_4(t)]^{1/2}\right) = \frac{1}{2} [P_4(t)]^{-1/2} \cdot P_4'(t) = \frac{P_4'(t)}{2\sqrt{P_4(t)}}$$

代入分子中：
$$Y'(t) = \frac{N'(t)\sqrt{P_4(t)} - N(t) \left( \frac{P_4'(t)}{2\sqrt{P_4(t)}} \right)}{P_4(t)}$$
我们对分子的两项通分，提取公分母 $2\sqrt{P_4(t)}$：
$$N'(t)\sqrt{P_4(t)} - \frac{N(t) P_4'(t)}{2\sqrt{P_4(t)}} = \frac{2 N'(t) [\sqrt{P_4(t)}]^2 - N(t) P_4'(t)}{2\sqrt{P_4(t)}} = \frac{2 N'(t) P_4(t) - N(t) P_4'(t)}{2\sqrt{P_4(t)}}$$

把这个通分后的分子除以分母 $P_4(t)$：
$$Y'(t) = \frac{2 N'(t) P_4(t) - N(t) P_4'(t)}{2 P_4(t) \sqrt{P_4(t)}} = \frac{N'(t) P_4(t) - \frac{1}{2} N(t) P_4'(t)}{P_4(t) \sqrt{P_4(t)}}$$

---

### 零件 2：计算分母因式 $1 + Y(t)^2$
$$1 + Y(t)^2 = 1 + \left( \frac{N(t)}{\sqrt{P_4(t)}} \right)^2 = 1 + \frac{N(t)^2}{P_4(t)}$$
通分，提取公分母 $P_4(t)$：
$$1 + Y(t)^2 = \frac{P_4(t) + N(t)^2}{P_4(t)}$$

---

### 零件 1 与零件 2 乘合体：
现在把两个零件代回 $\Phi'(t) = \frac{1}{1 + Y(t)^2} \cdot Y'(t)$：
$$\Phi'(t) = \frac{1}{\frac{P_4(t) + N(t)^2}{P_4(t)}} \cdot \frac{N'(t) P_4(t) - \frac{1}{2} N(t) P_4'(t)}{P_4(t) \sqrt{P_4(t)}}$$

第一项繁分式倒置相乘：
$$\Phi'(t) = \left[ \frac{P_4(t)}{P_4(t) + N(t)^2} \right] \cdot \left[ \frac{N'(t) P_4(t) - \frac{1}{2} N(t) P_4'(t)}{P_4(t) \sqrt{P_4(t)}} \right]$$

**看！奇迹发生了！**
第一个分式的分子上有 $P_4(t)$；
第二个分式的分母上也有 $P_4(t)$；
**这两个庞大复杂的四次多项式 $P_4(t)$，在上下位置完全相消、精确抵消为 1！**

把对消后的结果写出来：
$$\Phi'(t) = \frac{N'(t) P_4(t) - \frac{1}{2} N(t) P_4'(t)}{\left[ P_4(t) + N(t)^2 \right] \sqrt{P_4(t)}}$$

> **停下来深呼吸，体会这一步的数学震撼**：
> 那个折磨了我们无数个日夜的超越反正切函数 $\arctan$，在经过求导和代数通分后，**彻彻底底从宇宙中蒸发了！**
> 剩下的式子里，没有任何反三角函数，没有任何超越对数，分母只剩下了一个干干净净的代数多项式开方：$\sqrt{P_4(t)}$！

---

## 4. 有理分式通分与完全代数化

在实际物理几何中，分子函数 $N(t)$ 通常也是一个有理分式：
$$N(t) = \frac{n(t)}{q(t)}$$
式中 $n(t)$ 和 $q(t)$ 是关于 $t$ 的纯普通多项式。

我们求导 $N'(t)$（商法则）：
$$N'(t) = \frac{n'(t) q(t) - n(t) q'(t)}{q(t)^2}$$

把 $N(t) = \frac{n(t)}{q(t)}$ 和 $N'(t)$ 代入刚才导数公式的分子中：
$$N'(t) P_4(t) - \frac{1}{2} N(t) P_4'(t) = \left[ \frac{n' q - n q'}{q^2} \right] P_4 - \frac{1}{2} \left[ \frac{n}{q} \right] P_4'$$
两项通分，公分母为 $q(t)^2$：
$$= \frac{[n' q - n q'] P_4 - \frac{1}{2} n q P_4'}{q^2}$$

再看分母中的多项式项 $P_4(t) + N(t)^2$：
$$P_4(t) + \left( \frac{n(t)}{q(t)} \right)^2 = P_4(t) + \frac{n^2}{q^2} = \frac{P_4 q^2 + n^2}{q^2}$$

把它们代回 $\Phi'(t)$ 的表达式：
$$\Phi'(t) = \frac{\frac{[n' q - n q'] P_4 - \frac{1}{2} n q P_4'}{q^2}}{\left[ \frac{P_4 q^2 + n^2}{q^2} \right] \sqrt{P_4(t)}} = \frac{[n' q - n q'] P_4 - \frac{1}{2} n q P_4'}{\left( P_4 q^2 + n^2 \right) \sqrt{P_4(t)}}$$
两边的 $q(t)^2$ 再次完美抵消！

现在把 $\Phi'(t)$ 代回我们分部积分剩下的积分项 $-\frac{1}{2}\int_0^1 t^2 \Phi'(t) dt$ 中：
$$-\frac{1}{2} \int_0^1 t^2 \left[ \frac{[n' q - n q'] P_4 - \frac{1}{2} n q P_4'}{\left( P_4 q^2 + n^2 \right) \sqrt{P_4(t)}} \right] dt = -\frac{1}{2} \int_0^1 \frac{R(t)}{\sqrt{P_4(t)}} \, dt$$

式中定义了全局**纯代数有理函数 $R(t)$**：
$$R(t) \equiv \frac{t^2 \left( [n'(t) q(t) - n(t) q'(t)] P_4(t) - \frac{1}{2} n(t) q(t) P_4'(t) \right)}{P_4(t) q(t)^2 + n(t)^2}$$

---

## 5. 分支跳跃修正项 $I_{\mathrm{branch}}$ 从何而来？

上面的分部积分是在假设函数 $\Phi(t)$ 在整个区间 $t \in [0, 1]$ 上是绝对平滑、没有断裂的前提下成立的。
但是，在复分析中，反正切函数与复对数紧密相连：
$$\arctan(w) = \frac{1}{2i} \log\left( \frac{1 + iw}{1 - iw} \right)$$
在实数轴上，当内层比值 $Y(t) = \frac{N(t)}{\sqrt{P_4(t)}}$ 从负无穷大穿越到正无穷大（即分母穿过零点）时，或者在多值复数主值割线（Branch Cut）上跨界时，主值反正切函数会发生一个离散的**常数相位突变（Jump Discontinuity）**，大小通常为 $\pi$。

如果在区间 $(0, 1)$ 内存在若干个跳跃点 $t_j$（满足 $j = 1, 2, \dots$），在每一个跳跃点处，函数值的跳跃幅度为：
$$\Delta\Phi(t_j) = \Phi(t_j^+) - \Phi(t_j^-)$$
根据广义微积分的断点分部积分公式，边界项在每个断点都会掉下一个残值。将这些离散断点的贡献累加起来，就构成了严格的**分支跳跃修正项 $I_{\mathrm{branch}}$**：

$$I_{\mathrm{branch}} = - \frac{1}{2} \sum_j t_j^2 \Delta\Phi(t_j)$$

---

## 6. 最终大一统公式：向代数几何进发！

综合以上所有严密推导，原本包含超越反正切函数的复杂同伦积分，被我们严格重写为：

$$I = \frac{1}{2} \Phi(1) - \frac{1}{2} \int_0^1 \frac{R(t)}{\sqrt{P_4(t)}} \, dt + I_{\mathrm{branch}}$$

将代数曲线方程 $y = \sqrt{P_4(t)} \iff y^2 = P_4(t)$ 代入被积项：
$$\frac{R(t)}{\sqrt{P_4(t)}} \, dt = \frac{R(t)}{y} \, dt$$

**全剧终的核心里程碑**：
这正是代数几何中大名鼎鼎的**定义在椭圆代数曲线 $\mathcal{E}: y^2 = P_4(t)$ 上的阿贝尔微分形式（Abelian Differential）**！
我们彻底完成了从“超越反三角微积分”到“代数曲线纯代数 Abel 微分”的世纪飞跃！

---

### 第十一小节小结与确认

在本节中，我们完成了一次极为震撼的微积分推导：
1. 搭建了分部积分框架，把导数转移到 $\arctan$ 上，提取出常数端点项 $\frac{1}{2}\Phi(1)$；
2. **一步一步展开商法则求导**，推导出了 $Y'(t)$ 的精确结构；
3. 展现了分子分母中四次多项式 $P_4(t)$ 的完全对消，**彻底消灭了反正切超越项**；
4. 构造了纯代数有理函数 $R(t)$，将剩余积分严密表示为椭圆曲线上的 Abel 微分 $\frac{R(t)}{y} dt$；
5. 详细剖析了幅角跨越割线时的跳跃修正项 $I_{\mathrm{branch}}$。

---

# 第十二小节：椭圆曲线上的 Hermite 约化与三大基本 Abel 微分

## 1. 定理的输入、输出与适用限制

本节研究一个**已单独证明**能化成
$$I=\int_{t_0}^{t_1}\frac{R(t)}{y}\,dt,\qquad y^2=P_4(t),$$
的标量子积分，其中 $R$ 是有理函数，$P_4(t)=c_4t^4+c_3t^3+c_2t^2+c_1t+c_0$、$c_4\neq0$ 且 $P_4$ 无重根。积分路径、$y$ 的连续分支必须预先固定。这里的代数积分一般与第八小节的二维同伦积分**不是同一个对象**；只有给出使二者相等的原一形式，才能用于完整 Hessian。

## 2. 有理函数的部分分式和三种极点

在复数域中可写
$$R(t)=Q(t)+\sum_a\sum_{m=1}^{M_a}\frac{\mu_{a,m}}{(t-a)^m},$$
其中 $Q$ 是多项式，$a$ 为 $R$ 的有限极点。把它代入输入式，逐项得到
$$\frac{R(t)}{y}dt=\sum_{n\ge0}q_n\frac{t^n}{y}dt+\sum_{a,m}\mu_{a,m}\frac{dt}{(t-a)^m y}.$$
必须分别处理：高次多项式、位于非分支点的有限高阶极点、落在分支点的有限极点。最后一种情况下 $P_4(a)=0$，下面用 $1/P_4(a)$ 的操作不适用。

## 3. 多项式的准确降阶：从 $n\ge3$ 开始

设 $n\ge3$，取代数函数 $t^{n-3}y$。因为 $2yy'=P_4'$，乘积法则给出
$$d(t^{n-3}y)=\left[(n-3)t^{n-4}y+t^{n-3}\frac{P_4'}{2y}\right]dt
=\frac{2(n-3)t^{n-4}P_4+t^{n-3}P_4'}{2y}dt.$$
逐项展开分子最高幂的系数：第一项贡献 $2(n-3)c_4t^n$，第二项贡献 $4c_4t^n$，和为 $2(n-1)c_4t^n$。其余记为次数不超过 $n-1$ 的 $W_{n-1}(t)$，故
$$d(t^{n-3}y)=\left[(n-1)c_4\frac{t^n}{y}+\frac{W_{n-1}(t)}{2y}\right]dt,$$
$$\boxed{\frac{t^n}{y}dt=\frac{d(t^{n-3}y)}{(n-1)c_4}-\frac{W_{n-1}(t)}{2(n-1)c_4y}dt,\qquad n\ge3.}$$
因 $n-3\ge0$，新端点函数不会人为引入 $t=0$ 的极点。递归使最高次数 $n$ 逐次下降至 $2$；**不能**把这个式子在 $n=2$ 时套成 $t^{-1}y$，然后仍宣称仅剩 $1,t$ 两个多项式基底。因此一般保留
$$\frac{dt}{y},\qquad\frac{t\,dt}{y},\qquad\frac{t^2\,dt}{y}.$$

## 4. 非分支点的高阶极点降阶

设 $m\ge2$ 且 $P_4(a)\ne0$。再次使用 $y'=P_4'/(2y)$：
$$d\left(\frac{y}{(t-a)^{m-1}}\right)=\left[\frac{P_4'}{2(t-a)^{m-1}y}-\frac{(m-1)y}{(t-a)^m}\right]dt$$
$$=\frac{P_4'(t)(t-a)-2(m-1)P_4(t)}{2(t-a)^m y}dt.$$
将 $P_4(t)=P_4(a)+P_4'(a)(t-a)+O((t-a)^2)$ 代入，最高极点系数为 $-(m-1)P_4(a)$。记其余低阶极点及多项式项为 $L_{m-1}(t)dt/y$，则
$$d\left(\frac{y}{(t-a)^{m-1}}\right)=-(m-1)P_4(a)\frac{dt}{(t-a)^m y}+L_{m-1}(t)\frac{dt}{y},$$
$$\boxed{\frac{dt}{(t-a)^m y}=-\frac{1}{(m-1)P_4(a)}d\left(\frac{y}{(t-a)^{m-1}}\right)+\frac{L_{m-1}(t)}{(m-1)P_4(a)y}dt.}$$
这里 $L_{m-1}$ 的有限极点阶数严格小于 $m$；其多项式部分再按第 3 段处理。若 $P_4(a)=0$，应使用分支点的局部坐标 $v=\sqrt{t-a}$ 重新作 Laurent 分析，不能代入上式。

## 5. 终止形式与微分的正确分类

在所有极点避开分支点的前提下，以上递归得到
$$\boxed{\frac{R(t)}y\,dt=dF_{\rm alg}(t,y)+\lambda_0\frac{dt}{y}+\lambda_1\frac{t\,dt}{y}+\lambda_2\frac{t^2\,dt}{y}+\sum_a\mu_a\frac{dt}{(t-a)y}.}$$
这些系数由多项式除法及逐次消去求得，**没有**在本文中给出具体几何实例的全部系数，故不是现成 GPU 代码。分类也要谨慎：$dt/y$ 在光滑四次曲线上是第一类；$t\,dt/y$ 在两个无穷远点通常有非零留数，是第三类而不是第二类。令 $s=1/t$，选 $y\sim\sqrt{c_4}t^2[1+c_3/(2c_4t)+O(t^{-2})]$，则
$$\frac{t\,dt}{y}\sim-\frac{ds}{\sqrt{c_4}s}+O(ds),$$
可直接看见 $ds/s$ 的留数。为了构造无穷远处零留数的第二类候选项，可结合
$$\frac{(t^2+\frac{c_3}{2c_4}t)dt}{y}\sim-\frac{ds}{\sqrt{c_4}s^2}+O(ds);$$
这一步还要对另一无穷远点作同样核验。$dt/[(t-a)y]$ 在 $a$ 非分支点时为第三类。$F_{\rm alg}$ 是代数原函数，其端点差不能称为“处处全纯零成本”，分支与消去相抵都要核对。

## 6. 定积分端点式及检查

沿选定的同一提升路径，把上一恒等式积分得
$$I=F_{\rm alg}(t_1,y_1)-F_{\rm alg}(t_0,y_0)+\lambda_0 I_0+\lambda_1 I_1+\lambda_2 I_2+\sum_a\mu_a I_a,$$
$$I_j=\int_{t_0}^{t_1}\frac{t^jdt}{y}\quad(j=0,1,2),\qquad I_a=\int_{t_0}^{t_1}\frac{dt}{(t-a)y}.$$
必须把右边所有项**重新求导**并恢复 $R(t)/y$，才完成某个实际积分的 Hermite 约化。若积分路径经过极点，需另外说明主值或绕行方式；本节公式没有自动解决该问题。

# 第十三小节：Weierstrass 椭圆函数理论表述

## 1. 四次到三次：写清坐标和微分变换

本节继续以第十二小节的**条件性标量椭圆子积分**为对象。设平方自由四次多项式
$$P_4(t)=c_4(t-e_1)(t-e_2)(t-e_3)(t-e_4),\qquad e_i\ne e_j.$$ 
取 $e_4$ 为一根并令 $t=e_4+1/x$。逐项变换：
$$t-e_4=\frac1x,\quad t-e_i=\frac{1+(e_4-e_i)x}{x}\ (i=1,2,3),\quad dt=-\frac{dx}{x^2}.$$
因此
$$P_4(t)=\frac{c_4\prod_{i=1}^3[1+(e_4-e_i)x]}{x^4}=\frac{Q_3(x)}{x^4}.$$
定义新曲线变量 $Y=x^2y$，则 $Y^2=Q_3(x)$；**不**把复数 $\sqrt{Q_3/x^4}$ 的主值无条件写成 $\sqrt{Q_3}/x^2$，而是沿既定路径使用 $Y=x^2y$。于是
$$\frac{dt}{y}=\frac{-dx/x^2}{Y/x^2}=-\frac{dx}{Y},\qquad
\frac{t\,dt}{y}=-\left(e_4+\frac1x\right)\frac{dx}{Y},$$
$$\frac{t^2dt}{y}=-\left(e_4+\frac1x\right)^2\frac{dx}{Y},\qquad
\frac{dt}{(t-a)y}=-\frac{dx}{(e_4-a+1/x)Y}.$$
这些显式式子说明：原来 $t\,dt/y$ 变成含 $1/x$ 的微分，**不能直接声称**其原函数只是 $\zeta$ 的端点差。

## 2. 三次曲线的 Weierstrass 规范化

将 $Q_3(x)=A x^3+B x^2+C x+D$（$A\ne0$）平移 $x=X-B/(3A)$，消去二次项：
$$Q_3=A X^3+\left(C-\frac{B^2}{3A}\right)X+\left(D-\frac{BC}{3A}+\frac{2B^3}{27A^2}\right).$$
设 $\widetilde Y=2Y/\sqrt A$ 并在整个积分路径固定 $\sqrt A$ 的同一分支，则
$$\widetilde Y^2=4X^3-g_2X-g_3,$$
$$g_2=\frac4A\left(\frac{B^2}{3A}-C\right),\qquad
g_3=-\frac4A\left(D-\frac{BC}{3A}+\frac{2B^3}{27A^2}\right).$$
由 $Y=\sqrt A\widetilde Y/2$，准确的微分系数为
$$\frac{dt}{y}=-\frac{2}{\sqrt A}\frac{dX}{\widetilde Y}.$$
因此不能省略 $-2/\sqrt A$ 这个尺度因子。

## 3. 均一化与三类原函数的边界

设 $X=\wp(u)$、$\widetilde Y=\wp'(u)$，并沿提升路径连续选定 $u$，则 $dX/\widetilde Y=du$，故
$$I_0=\int\frac{dt}{y}=-\frac{2}{\sqrt A}(u_1-u_0).$$
对其余项，须先把第 1 节列出的关于 $x=X-B/(3A)$ 的有理式做部分分式，再逐项积分。已知 $\zeta'(u)=-\wp(u)$、$(\log\sigma)'=\zeta$。例如在 $v$ 不是半周期且 $\wp'(v)\ne0$ 时，标准恒等式是
$$\frac{\wp'(v)}{\wp(u)-\wp(v)}=\zeta(u-v)-\zeta(u+v)+2\zeta(v).$$
因此
$$\int\frac{du}{\wp(u)-\wp(v)}=\frac1{\wp'(v)}\left[\log\frac{\sigma(u-v)}{\sigma(u+v)}+2u\zeta(v)\right]+C.$$
注意这里的 $2u\zeta(v)$ 是**加号**；直接微分是检查符号的可靠办法。若 $\wp'(v)=0$，必须作极限或另用第二类表示，不能除以零。各端点仍应连续跟踪周期、对数分支，不能只凭 $\wp(u_j)=X_j$ 选两个任意反像相减。

## 4. 可证明的结论

本节证明了四次/三次曲线的双有理变换和第一类积分的明确尺度，给出了处理其他项的标准函数工具。对实际 $R(t)$，必须提供部分分式系数、端点提升和逐项反向求导，才能写一个具体 Weierstrass/Carlson 表达式。这一条件性标量计算不能自动替代完整二维球面积分。

# 第十四小节：模参数 $\tau$ 与 $q$-级数展开理论

---

## 1. 为什么我们需要 $q$-级数展开？

在第十三小节中，我们得到了包含 Weierstrass 函数 $\wp(u)$、$\zeta(u)$、$\sigma(u)$ 的绝美理论闭式：
$$I = \text{代数端点项} + c_0 \Delta u + c_1 \Delta\zeta(u) + \sum_a c_a \Delta \log\left( \frac{\sigma(u - u_a)}{\sigma(u + u_a)} \right) + I_{\mathrm{branch}}$$

但是，工程计算机和显卡（GPU）的硬件算术逻辑单元（ALU）非常“质朴”：
- 它们硬件原生支持加减乘除、开平方 `sqrt`；
- 通过微指令多项式近似支持基本初等函数 `sin`、`cos`、`exp`、`log`；
- **但世界上没有任何一块芯片的硬件指令集里，直接内置了 $\wp(u)$ 函数或 $\sigma(u)$ 函数的直接求值指令！**

要让计算机算出这几个函数的值，经典解析数论给出的终极武器就是**模形式（Modular Forms）**与**雅可比 $\theta$ 函数（Jacobi Theta Functions）的 $q$-级数展开**。
在很多理论文献中，作者往往会非常乐观地写道：
> *“Jacobi $\theta$ 函数具有指数平方级的超快收敛速度，级数只需要截断 3 到 5 项，就能在计算机上达到 IEEE 754 双精度（$10^{-16}$）极限！”*

这个说法听起来极具诱惑力，但在小行星引力场计算中，**它真的能无脑落地吗？它的物理陷阱在哪里？**
本小节我们将彻底搞清楚 $q$-级数的来龙去脉，并严密剖析它在工程实现中的边界与死穴！

---

## 2. 复环面上的两个基本全周期 $\omega_1, \omega_2$

回顾第十小节和第十三小节：椭圆代数曲线 $\mathcal{E}: y^2 = P_4(t)$ 在复数域上是一个亏格为 1 的紧黎曼面，在拓扑几何上就是一个**甜甜圈（复二维环面 $\mathbb{T}^2$）**。

一个甜甜圈表面上有两个最本质的、无法缩成一个点的独立闭合环路（同调群 $H_1(\mathcal{E}, \mathbb{Z})$ 的一组正交基底）：
- **$\gamma_1$（经线圈）**：绕着甜甜圈管子横截面转一圈的闭合回路；
- **$\gamma_2$（纬线圈）**：沿着甜甜圈中央大孔转一圈的闭合回路。

我们将全纯第一类 Abel 微分 $\frac{dt}{y} = \frac{dt}{\sqrt{P_4(t)}}$ 分别沿着这两个不可约闭回路积一整圈，得到曲线的**两个基本完全周期（Fundamental Complete Periods）**：

$$\omega_1 = \oint_{\gamma_1} \frac{dt}{y} = \oint_{\gamma_1} \frac{dt}{\sqrt{P_4(t)}}$$
$$\omega_2 = \oint_{\gamma_2} \frac{dt}{y} = \oint_{\gamma_2} \frac{dt}{\sqrt{P_4(t)}}$$

由于 $\gamma_1$ 和 $\gamma_2$ 在几何拓扑上是严格独立的，$\omega_1$ 和 $\omega_2$ 作为复数，在实数域上是**严格线性无关的**。
它们在复平面上张成了一个倾斜的平行四边形网格，这就是定义双周期函数的**周期格点 $\Lambda = \{ 2m\omega_1 + 2n\omega_2 \;:\; m, n \in \mathbb{Z} \}$**。

---

## 3. 模参数 $\tau$ 与诺姆（Nome）$q$ 的严格数学定义

### (1) 模参数 $\tau$（半周期比）
我们把两个基本周期作除法，定义**模参数（Modular Parameter）$\tau$**：
$$\tau = \frac{\omega_2}{\omega_1}$$

根据复分析中的黎曼双线性关系（Riemann Bilinear Relations），通过对回路基底定向的合理选取（保证逆时针方向），模参数 $\tau$ 的虚部必须严格恒大于零：
$$\mathrm{Im}\,\tau > 0$$
这意味着 $\tau$ 永远稳稳落在复平面的**上半平面（Upper Half-Plane $\mathbb{H}$）**中！
设 $\tau$ 的实部为 $\tau_1$，虚部为 $\tau_2$（满足 $\tau_2 > 0$）：
$$\tau = \tau_1 + i \tau_2$$

---

### (2) 诺姆（Nome）$q$
数学家利用复指数函数，将上半平面映射到复平面原点附近的开单位圆盘内部。
定义核心代数展开模数——**诺姆（Nome，记作 $q$）**：
$$q \equiv e^{i \pi \tau} = \exp(i \pi \tau)$$

我们来算诺姆 $q$ 的绝对值（模长）：
$$|q| = \left| e^{i \pi (\tau_1 + i \tau_2)} \right| = \left| e^{i \pi \tau_1} \cdot e^{-\pi \tau_2} \right| = \left| e^{i \pi \tau_1} \right| \cdot e^{-\pi \tau_2}$$
因为 $\left| e^{i \pi \tau_1} \right| = \sqrt{\cos^2(\pi\tau_1) + \sin^2(\pi\tau_1)} \equiv 1$，所以：
$$|q| = e^{-\pi \tau_2} = e^{-\pi \mathrm{Im}\,\tau}$$

因为虚部 $\mathrm{Im}\,\tau > 0$，所以指数上的幂次 $-\pi \mathrm{Im}\,\tau < 0$ 恒为负实数！
因为 $\operatorname{Im}\tau>0$ 且指数函数从不取零，
$$0<|q|<1.$$

---

## 4. $q$-级数的条件收敛速度与截断误差

为什么数学家如此偏爱诺姆 $q$？
因为 Weierstrass 椭圆函数族（$\wp, \zeta, \sigma$），可以通过纯代数转换，全部用**四大 Jacobi $\theta$ 函数**的商式精准表达！
我们来看最经典的 Jacobi 第三大 $\theta$ 函数 $\theta_3(z | \tau)$ 的傅里叶展开式：

$$\theta_3(z | \tau) = 1 + 2 \sum_{n=1}^\infty q^{n^2} \cos(2n z)$$
我们把求和的前几项一笔一画写出来：
$$\theta_3(z | \tau) = 1 + 2 q^{1^2} \cos(2z) + 2 q^{2^2} \cos(4z) + 2 q^{3^2} \cos(6z) + 2 q^{4^2} \cos(8z) + 2 q^{5^2} \cos(10z) + \dots$$
把指数上的平方算出来：
$$\theta_3(z | \tau) = 1 + 2 q^1 \cos(2z) + 2 q^4 \cos(4z) + 2 q^9 \cos(6z) + 2 q^{16} \cos(8z) + 2 q^{25} \cos(10z) + \dots$$

**请看指数上的数字序列：$1,4,9,16,25,36,\dots$。**
普通几何级数按 $q^n$ 衰减，而这里的项按 $q^{n^2}$ 衰减；但实际误差还取决于 $z$、前因子以及 $|q|$ 是否远离 $1$。

### 我们来带入实际数字算一次：
如果几何形状非常标准，使得 $\mathrm{Im}\,\tau \approx 1$，那么：
$$|q| = e^{-\pi \times 1} = e^{-3.14159} \approx 0.0432$$
此时各项的衰减速度如下：
- 第 1 项（$n=1$）：$q^1 \approx 0.0432 = 4.3 \times 10^{-2}$
- 第 2 项（$n=2$）：$q^4 \approx (0.0432)^4 \approx 3.5 \times 10^{-6}$
- 第 3 项（$n=3$）：$q^9 \approx (0.0432)^9 \approx 1.2 \times 10^{-13}$
- 第 4 项（$n=4$）：$q^{16} \approx (0.0432)^{16} \approx 1.8 \times 10^{-23} < 10^{-16} \text{ ！！}$

这个例子只说明在 $|q|\\approx e^{-\\pi}$ 且 $z$ 有界时收敛很快；它不能推出对所有模参数、端点和函数商都只需三到五项。实际截断阶数应由尾项上界或加倍计算差值确定。

---

## 5. 致命死穴：为什么在小行星引力计算中，这个假设会彻底暴毙？

理论虽美，但在真实的小行星多面体引力计算工程中，**如果我们直接把这套 $q$-级数写进 GPU Shader，程序会在很多区域发生灾难性的性能骤降和精度塌陷！**

这背后隐藏着五个根本性的数学物理死穴：

---

### 死穴 1：几何临界退化使得 $|q| \to 1$（级数罢工）
当探测器极其靠近表面三角形、或者三角形本身形状极其狭长扁平时：
四次多项式 $P_4(t)$ 的其中两个根会无限逼近彼此：$|e_i - e_j| \to 0$。
曲线无限趋向于退化奇异曲线，判别式 $\mathrm{Disc}(P_4) \to 0$。
此时：
- 周期 $\omega_1$ 发生对数爆炸：$\omega_1 \to \infty$；
- 模参数虚部趋向于零：$\mathrm{Im}\,\tau = \frac{\omega_2}{\omega_1} \to 0^+$！

我们来看看此时的诺姆 $|q|$ 变成了什么：
$$|q| = e^{-\pi \mathrm{Im}\,\tau} \to e^0 = 1^-$$
比如当 $|q| = 0.999$ 时：
$q^{16} = (0.999)^{16} \approx 0.984$！
**不仅 4 项到不了 $10^{-16}$，你就算算到 500 项，截断误差依然高达百分之几！超指数收敛彻底退化成了比蜗牛还慢的极慢收敛！**

---

### 死穴 2：全周期 vs 不完全积分（Incomplete Integrals）
上述的超快收敛主要发生在零值 $\theta_i(0 | \tau)$（对应完全椭圆周期）上。
但我们多面体边弧的积分是**不完全 Abel 积分**（积分上限是任意一个边界点 $t_1$）：
$$u(t_1) = \int_{t_0}^{t_1} \frac{dt}{\sqrt{P_4(t)}}$$
要对任意不完全积分求逆反解，涉及复杂的复数幅角追踪，展开式中的项数需求呈倍数级激增。

---

### 死穴 3：端点靠近分支点（Branch Point Collision）
当光线穿出多面体表面的交点，恰好接近大圆弧与判别式平面的交界时，积分端点 $t_1$ 无限逼近四次多项式的零点根 $e_k$。
在零点根附近，$y = \sqrt{P_4(t)} \to 0$，被积函数产生一阶奇异，复映射导数爆炸，$q$-展开的解析半径被急剧压缩。

---

### 死穴 4：第三类极点的除以零风险
回顾第十三小节推导的第三类积分：
$$I_a = \int \frac{dt}{(t - a)y}$$
它包含一个单极点 $t = a$。在 $\theta$ 函数表达体系中，这个极点体现为分母上的雅可比第一 $\theta$ 函数趋向于零：
$$\theta_1(z - z_a | \tau) \to 0$$
在计算机有限精度浮点数运算中，极小的数值除以一个极小的数值，会引发灾难性的**有效数字全部丢失（Catastrophic Cancellation）**。

---

### 死穴 5：GPU 硬件的“线程发散（Warp Divergence）”绝杀
在现代 GPU 硬件架构（如 NVIDIA CUDA、AMD RDNA、Apple Silicon Metal）中，32 个或 64 个并行线程被捆绑在一个**线程束（Warp / Wavefront）**中同步执行同一行代码。
如果某个小行星表面的射线网格中：
- 线程 A 对应的方向算出来 $|q| = 0.04$，展开 4 项就收敛了；
- 紧挨着它的线程 B 对应的方向正好碰到了狭长面，$|q| = 0.98$，需要循环展开 300 项！
**GPU 的机制是：整个 Warp 的 32 个线程必须齐步走！展开快的线程必须全部停下来，干等那个需要算 300 次循环的倒霉线程！**
这会导致 GPU 计算单元发生严重的空转等待，原本极速的硬件吞吐量暴跌数倍。

---

## 6. 最终结论与工程技术路线大分流

基于以上五个铁一样的数学与硬件分析，我们得到极其权威的学术与工程定论：

> **条件性结论**：
> 1. **模参数 $\tau$ 与 $q$-级数展开路线**：
>    在数学上高度深刻、形式优美，非常适合作为**CPU 端的高精度理论基准（Ground Truth Benchmark）**，用于检验其他数值算法的极限精度；
> 2. **但在工业级 GPU 大规模并行引力计算中**：
>    它绝不是最稳妥、最健壮的主路径！
>    工程上通常希望采用分支较少、可自适应控制误差的对称椭圆积分实现；这仍需针对输入域证明数值稳定性。

这个能完美替代经典体系、拯救 GPU 算力的救世主，正是我们在第十五小节要学习的——**Carlson 对称标准椭圆积分体系**！

---

### 第十四小节小结与确认

在本节中，我们完成了一次从纯数学理论到计算机底层硬件的全面审视：
1. 完整推导了复环面的基本双周期 $\omega_1, \omega_2$、模参数 $\tau$ 和诺姆 $q$ 的定义；
2. 一笔一画展开了 Jacobi $\theta$ 函数，展示了指数平方 $q^{n^2}$ 在标准区域下惊人的超指数收敛；
3. 说明了模退化、不完全积分、分支点和极点会破坏固定少项的统一误差保证；
4. 明确了该路线作为“理论参考线”而非“GPU 主路径”的技术定位。

---

> **严格性修正**：Carlson $R_F,R_D,R_J,R_C$ 只能表示已经明确化为椭圆积分的标量子积分。本文没有给出一般 $C_0,C_1,C_a,D_b$ 的交比系数，因此后文的组合式是接口模板而非已完成的通用闭式。复根、平方根与对数必须沿同一积分路径连续选支，并用原积分核做数值核对。

# 第十五小节：Carlson 对称标准积分体系（GPU 稳健闭式）

---

## 1. 为什么 20 世纪末的数学家要彻底重构椭圆积分？

在第十四小节中，我们深刻剖析了为什么经典分析学中的 $q$-级数展开无法胜任 GPU 并行工程。
事实上，早在 20 世纪 70 年代以前，计算机科学家使用的一直是 19 世纪欧拉、勒让德建立的经典三大勒让德椭圆积分体系：
- 第一类：$F(\phi, k) = \int_0^\phi \frac{d\theta}{\sqrt{1 - k^2 \sin^2\theta}}$
- 第二类：$E(\phi, k) = \int_0^\phi \sqrt{1 - k^2 \sin^2\theta} \, d\theta$
- 第三类：$\Pi(\phi, n, k) = \int_0^\phi \frac{d\theta}{(1 - n\sin^2\theta)\sqrt{1 - k^2 \sin^2\theta}}$

但在计算机芯片上，这套经典体系存在**极其恶劣的数值陷阱**：
1. **参数严重不对称**：变量 $\phi$ 是角度，模数 $k$ 是离心率，性质完全不同，一旦积分上下限互换或跨越象限，需要写几十个复杂的 `if-else` 分支去打补丁；
2. **极点伪奇异**：在 $\phi = \pi/2$ 或 $n \to 1$ 附近，分母会出现伪无穷大；
3. **GPU 硬件极不友好**：大量的条件分支会导致 GPU 线程束严重发散（Warp Divergence）。

为了彻底终结这个混乱局面，美国艾奥瓦州立大学的著名应用数学家 **B.C. Carlson** 提出了震撼学术界的**完全对称标准椭圆积分体系（Carlson Symmetric Forms）**！
它是现代数值椭圆积分中常用的对称表示，也便于在 GPU 中实现；具体稳定性仍取决于参数域和误差控制。

---

## 2. 四大 Carlson 对称标准积分的严格定义

Carlson 体系通过将积分变量映射到半实轴 $s \in [0, \infty)$，定义了四个代数对称积分。

---

### (1) 第一类对称椭圆积分 $R_F(x, y, z)$
$$R_F(x, y, z) \equiv \frac{1}{2} \int_0^\infty \frac{ds}{\sqrt{(s + x)(s + y)(s + z)}}$$
- **参数要求**：$x, y, z$ 至多只能有一个为零，且不能有负实数；
- **全对称性（Total Symmetry）**：无论你把 $x, y, z$ 这三个变量怎么任意交换顺序（共 $3! = 6$ 种排列），函数值**绝对完全相等**：
  $$R_F(x, y, z) = R_F(y, x, z) = R_F(z, x, y) = R_F(y, z, x) = R_F(z, y, x) = R_F(x, z, y)$$
- **齐次性（Homogeneity）**：它是 $-1/2$ 次齐次函数，尺度缩放完全自洽：
  $$R_F(\lambda x, \lambda y, \lambda z) = \lambda^{-1/2} R_F(x, y, z)$$

---

### (2) 第二类对称椭圆积分 $R_D(x, y, z)$
$$R_D(x, y, z) \equiv \frac{3}{2} \int_0^\infty \frac{ds}{(s + z) \sqrt{(s + x)(s + y)(s + z)}}$$
- **特点**：对于前两个变量 $x$ 和 $y$ 是严格对称的（$R_D(x, y, z) = R_D(y, x, z)$），第三个变量 $z$ 代表具有二阶极点的轴；
- **齐次性**：它是 $-3/2$ 次齐次函数：
  $$R_D(\lambda x, \lambda y, \lambda z) = \lambda^{-3/2} R_D(x, y, z)$$

---

### (3) 第三类对称椭圆积分 $R_J(x, y, z, p)$
$$R_J(x, y, z, p) \equiv \frac{3}{2} \int_0^\infty \frac{ds}{(s + p) \sqrt{(s + x)(s + y)(s + z)}}$$
- **特点**：对前三个变量 $x, y, z$ 完全对称，第四个参数 $p$ 专门用于吸收有理分式的**一阶单极点（留数奇点）**！
- **齐次性**：同样是 $-3/2$ 次齐次函数：
  $$R_J(\lambda x, \lambda y, \lambda z, \lambda p) = \lambda^{-3/2} R_J(x, y, z, p)$$

---

### (4) 退化初等对称积分 $R_C(x, y)$
$$R_C(x, y) \equiv R_F(x, y, y) = \frac{1}{2} \int_0^\infty \frac{ds}{(s + y) \sqrt{s + x}}$$
当后两个变量相等时，$R_F$ 自动退化为初等函数。
这个看似平淡无奇的积分，以无分支的方式**包揽了所有的初等反三角函数与复对数**：
- 当 $x < y$ 时：
  $$R_C(x, y) = \frac{\arccos\sqrt{x/y}}{\sqrt{y - x}} = \frac{\arctan\sqrt{(y - x)/x}}{\sqrt{y - x}}$$
- 当 $x > y$ 时：
  $$R_C(x, y) = \frac{\mathrm{arccosh}\sqrt{x/y}}{\sqrt{x - y}} = \frac{\log\left( \sqrt{x/y} + \sqrt{(x - y)/y} \right)}{\sqrt{x - y}}$$
- 当 $x = y$ 时：
  $$R_C(x, x) = \frac{1}{\sqrt{x}}$$

> **工程价值**：一个 $R_C$ 函数，彻底消除了计算机判断“该用 `arctan` 还是 `log`”的代码分支！

---

## 3. 代数根变换：三大 Abel 积分映射到 Carlson 体系

在第十二小节中，我们通过 Hermite 约化得到了三个基本积分：
$$I_0 = \int_{t_0}^{t_1} \frac{dt}{\sqrt{P_4(t)}}, \quad I_1 = \int_{t_0}^{t_1} \frac{t \, dt}{\sqrt{P_4(t)}}, \quad I_a = \int_{t_0}^{t_1} \frac{dt}{(t - a)\sqrt{P_4(t)}}$$

设四次多项式 $P_4(t) = 0$ 的四个互异根按大小排列（或在复数域配对）。
通过标准的分式代换（交比映射，Cross-Ratio Mapping）：
$$s(t) = \frac{(t - e_1)(e_4 - e_2)}{(t - e_2)(e_4 - e_1)}$$
这个有理变换把四次根式化为标准三次根式；具体端点通常对应有限的 $s_0,s_1$，只有再作端点平移或补段后才可写成 $[0,\infty)$。因此下面的 Carlson 线性组合只能作为在每个具体端点配置下完成代换并核对系数后的模板：

$$I_0 = C_0 R_F(x_0, y_0, z_0)$$
$$I_1 = C_1 R_F(x_1, y_1, z_1) + C_2 R_D(x_1, y_1, z_1)$$
$$I_a = C_{3a} R_F(x_a, y_a, z_a) + C_{4a} R_J(x_a, y_a, z_a, p_a) + C_{5a} R_C(x_c, y_c)$$

式中：
- 所有的系数 $C_0, C_1, C_2, C_{3a}, C_{4a}, C_{5a}$；
- 以及所有的输入自变量 $x, y, z, p$；
**全部是由端点坐标 $t_0, t_1$、极点 $a$ 以及四次多项式根 $e_i$ 构成的纯代数有理交比式！**
不需要进行任何符号微积分，输入参数即可瞬间代数求出！

---

## 4. 显卡计算的核心引擎：Carlson 倍角定理与无分支极速迭代算法

为什么这套公式在 GPU 和现代 CPU 上被称为“性能怪兽”？
因为 Carlson 证明了一个惊人的**倍角定理（Duplication Theorem）**：
以第一类积分 $R_F(x, y, z)$ 为例：
$$R_F(x,y,z)=\frac12R_F(x+\lambda,y+\lambda,z+\lambda),\qquad \lambda=\sqrt{x}\sqrt{y}+\sqrt{y}\sqrt{z}+\sqrt{z}\sqrt{x}.$$ 
若 $x'=(x+\lambda)/4$，$y'=(y+\lambda)/4$，$z'=(z+\lambda)/4$，则同一倍增定理也可写成 $R_F(x,y,z)=R_F(x',y',z')$；不能把两个等式写成前式的“$2R_F$”。
其中参数 $\lambda$ 为优美的几何对称平均值：
$$\lambda = \sqrt{x}\sqrt{y} + \sqrt{y}\sqrt{z} + \sqrt{z}\sqrt{x}$$

### 迭代算法的完全展开（无分支循环）
设初始输入为：
$$x_0 = x, \quad y_0 = y, \quad z_0 = z$$
在每一步循环 $n = 0, 1, 2, \dots$ 中，执行以下三条纯加减乘除与开平方操作：

1. **计算对称均值 $\lambda_n$**：
   $$\lambda_n = \sqrt{x_n} \sqrt{y_n} + \sqrt{y_n} \sqrt{z_n} + \sqrt{z_n} \sqrt{x_n}$$
2. **更新三个自变量**：
   $$x_{n+1} = \frac{x_n + \lambda_n}{4}$$
   $$y_{n+1} = \frac{y_n + \lambda_n}{4}$$
   $$z_{n+1} = \frac{z_n + \lambda_n}{4}$$
3. **计算三者的算术平均中心 $\mu_n$**：
   $$\mu_n = \frac{x_n + y_n + z_n}{3}$$

---

### 为什么这个循环收敛得不可思议？
我们来考察三个变量偏离中心值 $\mu_n$ 的相对偏差：
$$X_n = 1 - \frac{x_n}{\mu_n}, \quad Y_n = 1 - \frac{y_n}{\mu_n}, \quad Z_n = 1 - \frac{z_n}{\mu_n}$$
由于倍角定理中的除以 4，每一次循环迭代，相对偏差的量级不是减半，而是**按 16 倍的超高速度暴跌**：
$$|X_{n+1}| \sim \frac{1}{16} |X_n|^2$$
误差差异在合适条件下呈二次型快速缩小；具体阶数和常数取决于输入域及终止多项式，不能仅凭“除以 4”宣称统一四阶。
这些数量级只是某些良态输入的示意，不能作为任意输入的误差定理。迭代次数必须由归一化偏差和终止误差估计决定；不能固定为 5 次，也不能保证任意退化输入达到双精度。

---

### 终结公式（泰勒展开收尾）
当归一化偏差满足预设阈值时，才使用与该阈值匹配阶数的终止多项式；下式只展示低阶近似，不是固定五次循环的普适保证：

$$R_F(x, y, z) \approx \frac{1}{\sqrt{\mu}} \left( 1 - \frac{1}{10} E_2 + \frac{1}{14} E_3 + \frac{1}{24} E_2^2 - \frac{3}{44} E_2 E_3 \right)$$
式中：
$$E_2=XY+YZ+ZX,\qquad E_3=XYZ,$$

> **请看这个算法在 GPU 硬件上的绝美表现**：
> 1. **零超越函数**：里面没有任何一个 `sin`、`cos`、`exp`、`log`，全是最基本的加减乘除和硬件单周期指令 `sqrt`！
> 2. GPU 中可以采用统一迭代上限，但必须处理非正参数、复分支、退化和溢出；统一循环不等于零误差。
> 3. Carlson 迭代对合法输入稳定，但“任意接近退化都无条件稳健”需要具体误差分析。

---

## 5. 单条大圆弧在 GPU 上的终极解析求值公式

现在，我们把第十一小节的分部积分、第十二小节的 Hermite 约化，以及本小节的 Carlson 算子彻底闭合连接！
在多面体可见性单元的第 $m$ 个区域中，对于第 $k$ 个 Cauchy 核，其沿第 $e$ 条大圆弧的积分贡献 $\mathcal{E}_{kme}$，在 GPU 上可以直接用如下公式**一步执行求值**：

$$\mathcal{E}_{kme} = \frac{1}{2}\Phi_{kme}(1) - \frac{1}{2}\Delta F_{kme}^{\mathrm{alg}} - \frac{1}{2}\left[ C_{0,kme} R_F + C_{1,kme} R_D + \sum_a C_{a,kme} R_J + \sum_b D_{b,kme} R_C \right] + I_{kme}^{\mathrm{branch}}$$

我们对这个最终公式中的每一个分项做一次终极点名：
1. **$\Phi_{kme}(1)$**：分部积分掉下来的反正切端点值；
2. **$\Delta F_{kme}^{\mathrm{alg}}$**：Hermite 约化中消除所有 $t^n (n \ge 2)$ 和 $(t-a)^m (m \ge 2)$ 产生的高阶代数全微分在两端点的纯代数差值；
3. **$R_F$**：第一类全纯椭圆微积分贡献；
4. **$R_D$**：第二类无穷远极点椭圆微积分贡献；
5. **$\sum_a C_a R_J$**：第三类有限留数单极点椭圆微积分贡献；
6. **$\sum_b D_b R_C$**：退化初等对数/反三角函数贡献；
7. **$I_{kme}^{\mathrm{branch}}$**：幅角穿越割线时的离散相位跳跃补偿项。

---

### 第十五小节小结与确认

在本节中，我们见证了整个解析闭式主线的最高潮：
1. 明确了为什么经典勒让德积分必须被 Carlson 对称积分全面取代；
2. 一字不差地定义了四大对称标准积分 $R_F, R_D, R_J, R_C$；
3. 展现了代数四次多项式根代换如何将三大 Abel 积分投影到 Carlson 算子基上；
4. **给出了 Carlson 倍增迭代和条件性终止多项式**；迭代次数与误差必须自适应，不能固定为 5 次。
5. 组装出了能在 GPU Compute Shader 中直接执行的单弧解析求值终极公式 $\mathcal{E}_{kme}$。

---

> **严格性修正**：第七小节只降维了常权球面矩。一般的 $\int_{\Omega_m}T(\mathbf u)S_m(\mathbf u)d\Omega$ 含方向依赖的 $S_m$，不能直接套用该边界公式；$H^{\mathrm{ell}}$ 只能代表另行证明可边界化的子项，其余必须保留为明确残差。

# 第十六小节：全局主线解析大叠加与残差项的提出

## 1. 先固定对象、截断方式和方向分区

本节只对第六小节给出的球形排异有限部分进行整理。令
$$T(\mathbf u)=3\mathbf u\mathbf u^{T}-I,\qquad \mathbf u\in\mathbb S^2,$$
并令
$$E_{\mathbf x}(\mathbf u)=\bigcup_{\ell=1}^{L(\mathbf u)}[R^-_\ell(\mathbf u),R^+_\ell(\mathbf u)]$$
是射线 $\mathbf y=\mathbf x-R\mathbf u$ 与 $V$ 的交集。所有端点都只在避开相切、顶点和共面方向的球面满测集上使用；这些例外方向的面积为零，边界值由邻域极限定义。

若 $\mathbf x\in V$，定义局部方向集合
$$\Gamma_{\mathbf x}=\{\mathbf u\in\mathbb S^2:R^-_1(\mathbf u)=0\},$$
并把从 $R=0$ 开始的第一段写成 $[0,L(\mathbf u)]$。若 $\mathbf x\notin V$，则 $\Gamma_{\mathbf x}=\varnothing$。对 $\varepsilon>0$ 定义球形排异核
$$I_\varepsilon(\mathbf x)=G\int_{\mathbb S^2}T(\mathbf u)\sum_\ell\int_{[R^-_\ell,R^+_\ell]\cap[\varepsilon,\infty)}\frac{\rho(\mathbf x-R\mathbf u)}R\,dR\,d\Omega.$$
这里 $I_\varepsilon$ 只表示非局部核积分；完整 Hessian 还要加上分布恒等式产生的接触项。

## 2. 局部段逐行分解

在 $\Gamma_{\mathbf x}$ 上，取 $0<\varepsilon<L(\mathbf u)$。逐项加减 $\rho(\mathbf x)$：
$$\begin{aligned}
\int_\varepsilon^{L(\mathbf u)}\frac{\rho(\mathbf x-R\mathbf u)}R\,dR
&=\int_\varepsilon^{L(\mathbf u)}\frac{\rho(\mathbf x)}R\,dR
+\int_\varepsilon^{L(\mathbf u)}\frac{\rho(\mathbf x-R\mathbf u)-\rho(\mathbf x)}R\,dR\\
&=\rho(\mathbf x)\left(\log L(\mathbf u)-\log\varepsilon\right)
+\int_\varepsilon^{L(\mathbf u)}\frac{\rho(\mathbf x-R\mathbf u)-\rho(\mathbf x)}R\,dR.
\end{aligned}$$
令 $\ell _0>0$ 为任意固定参考长度，并拆分
$$\log\frac{L(\mathbf u)}\varepsilon=\log\frac{\ell _0}\varepsilon+\log\frac{L(\mathbf u)}{\ell _0}.$$
因此
$$\begin{aligned}
I_\varepsilon(\mathbf x)= {}&G\rho(\mathbf x)C_{\mathbf x}\log\frac{\ell _0}{\varepsilon}
+G\rho(\mathbf x)\int_{\Gamma_{\mathbf x}}T(\mathbf u)\log\frac{L(\mathbf u)}{\ell _0}\,d\Omega\\
&+G\int_{\Gamma_{\mathbf x}}T(\mathbf u)\int_\varepsilon^{L(\mathbf u)}\frac{\rho(\mathbf x-R\mathbf u)-\rho(\mathbf x)}R\,dR\,d\Omega
+I_{\rm far}(\mathbf x),
\end{aligned}$$
其中
$$C_{\mathbf x}=\int_{\Gamma_{\mathbf x}}T(\mathbf u)\,d\Omega,$$
以及远段的定义是
$$I_{\rm far}(\mathbf x)=G\int_{\mathbb S^2}T(\mathbf u)\sum_{\ell:\,R^-_\ell>0}\int_{R^-_\ell}^{R^+_\ell}\frac{\rho(\mathbf x-R\mathbf u)}R\,dR\,d\Omega.$$
若同一方向有多个远段，上式中的求和逐段进行；没有把局部段重复计算。

## 3. 截断极限和可积性证明

假设 $\rho\in C^1$ 于 $V$ 的邻域，且 $V$ 有界。由一阶均值定理，对 $0<R<L(\mathbf u)$，存在 $\xi$ 使
$$|\rho(\mathbf x-R\mathbf u)-\rho(\mathbf x)|\le R\sup_{\mathbf z\in V'}\|\nabla\rho(\mathbf z)\|.$$
故
$$\left|\frac{\rho(\mathbf x-R\mathbf u)-\rho(\mathbf x)}R\right|\le M,$$
其中 $M<\infty$ 与 $R$、$\mathbf u$ 无关。由于 $L(\mathbf u)$ 有界，支配收敛定理给出
$$\lim_{\varepsilon\downarrow0}\int_\varepsilon^{L(\mathbf u)}\frac{\rho(\mathbf x-R\mathbf u)-\rho(\mathbf x)}R\,dR
=\int_0^{L(\mathbf u)}\frac{\rho(\mathbf x-R\mathbf u)-\rho(\mathbf x)}R\,dR.$$
于是严格得到
$$\boxed{I_\varepsilon=G\rho(\mathbf x)C_{\mathbf x}\log\frac{\ell _0}{\varepsilon}+I_{\rm fin}+o(1)},$$
$$\begin{aligned}
I_{\rm fin}={}&G\rho(\mathbf x)\int_{\Gamma_{\mathbf x}}T(\mathbf u)\log\frac{L(\mathbf u)}{\ell _0}\,d\Omega\\
&+G\int_{\Gamma_{\mathbf x}}T(\mathbf u)\int_0^{L(\mathbf u)}\frac{\rho(\mathbf x-R\mathbf u)-\rho(\mathbf x)}R\,dR\,d\Omega+I_{\rm far}(\mathbf x).
\end{aligned}$$
这里的 $o(1)$ 来自被删去的区间 $(0,\varepsilon)$：其绝对值不超过 $4\pi\|T\|_\infty M\varepsilon$ 的常数倍。

完整的球形有限部分为
$$H_K^{\rm FP}(\mathbf x)=I_{\rm fin}(\mathbf x)-\frac{4\pi G}{3}\theta_{\rm ball}(\mathbf x)\rho(\mathbf x)I,$$
而不是把接触项再次放进 $I_{\rm fin}$。

## 4. $C_{\mathbf x}$ 的可证明性质

方向张量的迹为
$$\operatorname{tr}T(\mathbf u)=3\|\mathbf u\|^2-3=0,$$
所以 $\operatorname{tr}C_{\mathbf x}=0$。若 $\Gamma_{\mathbf x}=\mathbb S^2$，则
$$\int_{\mathbb S^2}u_i u_j\,d\\Omega=\frac{4\pi}{3}\delta_{ij},qquad C_{\mathbf x}=0.$$
若 $\Gamma_{\mathbf x}$ 是任意半球，选法向为 $\mathbf n$，利用对称性
$$\int_{\Gamma}u_i u_j\,d\\Omega=\frac{2\pi}{3}\delta_{ij},qquad \int_\Gamma d\Omega=2\pi,$$
仍有 $C_{\mathbf x}=0$。对棱边或顶点的切锥，不能仅凭“非光滑”推出 $C_{\mathbf x}\ne0$；必须实际计算
$$C_{ij}=3\int_{\Gamma_{\mathbf x}}u_i u_j\,d\\Omega-\delta_{ij}\,|\Gamma_{\mathbf x}|.$$
如果该矩阵恰好为零，则对数项也恰好消失。只有在计算得到 $C_{\mathbf x}\ne0$ 且 $\rho(\mathbf x)\ne0$ 时，截断 Hessian 才有对数发散。

## 5. 有限部分的尺度依赖

若把参考长度由 $\ell _0$ 换成 $\ell _1$，则
$$\log\frac{\ell _0}{\varepsilon}=\log\frac{\ell _1}{\varepsilon}+\log\frac{\ell _0}{\ell _1}.$$
因此有限部分变为
$$H^{\rm FP}_{\ell _1}-H^{\rm FP}_{\ell _0}=G\rho(\mathbf x)C_{\mathbf x}\log\frac{\ell _0}{\ell _1}.$$
所以当 $C_{\mathbf x}\ne0$ 时，Hadamard 有限部分依赖所声明的尺度约定；“唯一数值”只有在固定 $\ell _0$ 后才有意义。

---

# 第十七小节：严格的数学边界与理论链条终审总结

## 1. 在假设下成立的主定理

设 $V\subset\mathbb R^3$ 有界，边界为有限个 Lipschitz 面片；设 $\rho\in C^1$ 于 $V$ 的邻域。对避开切触方向的射线分解，以上各积分均按 Lebesgue 意义定义，并按球形排异取极限。则：

1. 球坐标变换 $\mathbf y=\mathbf x-R\mathbf u$ 的 Jacobian 为 $R^2$；
2. 分布恒等式为
$$\partial_i\partial_j\frac1R=\operatorname{p.v.}\frac{3r_ir_j-R^2\delta_{ij}}{R^5}-\frac{4\pi}{3}\delta_{ij}\delta(\mathbf r);$$
3. 第十六节的局部加减恒等式给出精确的 $\log\varepsilon$ 系数 $G\rho(\mathbf x)C_{\mathbf x}$；
4. 在 $C^1$ 假设下，其余局部差值积分绝对收敛，远段积分也绝对收敛；
5. 完整球形有限部分由上一节的 $I_{\rm fin}$ 减去接触项得到。

证明就是第 2、3 节的逐项代数恒等式、均值定理和支配收敛定理；没有使用拓扑猜测替代积分估计。

## 2. 必须保留的条件性结论

沿大圆弧的代数曲线 $y^2=P_4(t)$ 只有在 $P_4$ 平方自由、路径不穿越分支点且确实得到形如 $R(t)dt/y$ 的标量子积分时，才能使用第十二、十三节的 Hermite 或 Weierstrass 约化。非整数幂次的 Appell 表达式也只适用于满足其分支、参数和端点条件的单条径向积分。它们都不是对完整二维同伦积分的自动等价替换。

## 3. 已删除的过强断言

不能无条件声称：每个棱边都产生发散、所有边界积分都能化为椭圆函数、固定少量 Carlson 或 DE 节点就有统一误差、非凸天体必然产生高亏格、或任何残差都能被某个“解析主线”消灭。每项都必须先验证输入假设和等式的反向求导。

---

# 第十八小节：残差张量 $H^{\mathrm{res}}$ 的严格可计算定义与已提取解析核

## 1. 残差不是标签，而是逐点相减后的函数

在每个球面单元 $\Omega_m$ 和径向段 $[a_{m\ell}(\mathbf u),b_{m\ell}(\mathbf u)]$ 上，定义原始标量核
$$F_{m\ell,k}(\mathbf u)=\int_{a_{m\ell}(\mathbf u)}^{b_{m\ell}(\mathbf u)}\frac{\phi_k(\mathbf x-R\mathbf u)}R\,dR,$$
其中
$$\phi_k(\mathbf y)=\left[1+\sigma_k^2\|\mathbf y-\mathbf c_k\|^2\right]^{-\alpha_k}.$$
这里先不假设它有闭式原函数。

## 2. 一个无歧义的解析提取方案

对接触段 $a=0$，并且 $\rho$ 可微，定义可证明的局部提取项
$$F^{\rm ext}_{m,k}(\mathbf u)=\phi_k(\mathbf x)\log\frac{b_m(\mathbf u)}{\ell _0}.$$
对应的逐点残差为
$$F^{\rm rem}_{m,k}(\mathbf u)=\int_0^{b_m(\mathbf u)}\frac{\phi_k(\mathbf x-R\mathbf u)-\phi_k(\mathbf x)}R\,dR.$$
由均值定理，$F^{\rm rem}_{m,k}$ 有界且可积。对非接触段，取
$$F^{\rm ext}_{m\ell,k}=0,qquad F^{\rm rem}_{m\ell,k}=F_{m\ell,k}.$$
这是一种不依赖猜测的分解；若另有经过反向求导验证的闭式 $P_{m\ell,k}$，可以改取 $F^{\rm ext}=P$，但必须把定义和验证式写出。

## 3. 残差张量的完整公式

令 $J=G\sum_k w_k$，则
$$\begin{aligned}
H^{\rm ext}_{ij}(\mathbf x)&=G\sum_{m,k}w_k\int_{\Omega_m}T_{ij}(\mathbf u)F^{\rm ext}_{m,k}(\mathbf u)\,d\Omega,\\
H^{\rm res}_{ij}(\mathbf x)&=G\sum_{m,k}w_k\int_{\Omega_m}T_{ij}(\mathbf u)\sum_\ell F^{\rm rem}_{m\ell,k}(\mathbf u)\,d\Omega.
\end{aligned}$$
逐项相加立即得到
$$H^{\rm nl}_{ij}=H^{\rm ext}_{ij}+H^{\rm res}_{ij}.$$
若接触段存在 $\log(\ell _0/\varepsilon)$，应把该显式发散项另列为第十六节的 $G\rho C\log(\ell _0/\varepsilon)$，不能藏进数值残差。

## 4. 对称性和迹的校验

因为 $T_{ij}=T_{ji}$，有
$$H^{\rm res}_{ij}=H^{\rm res}_{ji}.$$
又因为 $\sum_iT_{ii}=0$，非接触的核积分满足
$$\sum_iH^{\rm res}_{ii}=0.$$
完整 Hessian 的迹还必须加入分布接触项，因此在 $\mathbf x$ 为密度连续的内点时
$$\operatorname{tr}H_K=-4\pi G\rho(\mathbf x),$$
在外点为 $0$；这是独立于数值求积的强制检查。

---

# 第十九小节：高亏格超椭圆 Abel 积分与多维 Riemann Theta 函数（条件性模块）

## 1. 何时才会出现高亏格

单条大圆弧的四次平方根最多给出亏格 $1$（非退化时）。只有在另外构造出的代数子积分具有
$$Y^2=P_{2g+1}(z)\quad\text{或}\quad Y^2=P_{2g+2}(z),$$
且多项式平方自由、$g\ge2$ 时，才得到亏格 $g$ 的超椭圆曲线。这个条件必须由具体代换实际产生；不能因为原问题是二维就断言存在 $H^{\rm hyp}$。

## 2. 周期与 Theta 函数定义

在已构造的紧化曲线上选取辛同调基 $(a_j,b_j)_{j=1}^g$，取第一类微分
$$\nu_j=\frac{z^{j-1}dz}{Y},\qquad j=1,\ldots,g,$$
并定义周期矩阵
$$A_{ij}=\oint_{a_i}\nu_j,\qquad B_{ij}=\oint_{b_i}\nu_j,\qquad \tau=A^{-1}B.$$
在标准取向下 $\tau=\tau^T$ 且 $\operatorname{Im}\tau$ 正定。对 $z\in\mathbb C^g$，Riemann Theta 函数为
$$\Theta(z\mid\tau)=\sum_{n\in\mathbb Z^g}\exp\!\left(\pi i n^T\tau n+2\pi i n^Tz\right).$$
这些定义只有在周期确实来自同一条曲线和同一组基底时才可使用。

## 3. 截断误差的可核验上界

设 $\lambda=\lambda_{\min}(\operatorname{Im}\tau)>0$，$v=\|\operatorname{Im}z\|_2$。对满足 $\|n\|_\infty=r$ 的整数向量，有
$$|e^{\pi i n^T\tau n+2\pi i n^Tz}|
\le e^{-\pi\lambda r^2+2\pi v\sqrt g\,r}.$$
该壳层的点数为
$$N_r=(2r+1)^g-(2r-1)^g.$$
因此截断到 $\|n\|_\infty\le N$ 的尾项满足显式估计
$$|R_N(z)|\le\sum_{r=N+1}^{\infty}N_r e^{-\pi\lambda r^2+2\pi v\sqrt g\,r}.$$
实际选择 $N$ 时必须计算或上界这个级数；不能只凭“保留三到五项”给出统一精度。

## 4. 与原 Hessian 的关系

只有在给出从原始 $F_{m\ell,k}$ 到上述超椭圆微分的逐步双有理代换，并证明端点、分支和 Jacobian 都匹配后，才能定义
$$H^{\rm hyp}_{ij}=G\sum_{m,k}w_k\int_{\Omega_m}T_{ij}(\mathbf u)F^{\rm hyp}_{m,k}(\mathbf u)d\Omega.$$
否则本节只是一个条件性工具箱，原问题仍由第十八节的残差定义或第二十二节直接积分给出。

---

# 第二十小节：同伦原一形式的双指数（DE / Tanh-Sinh）数值求积（条件性模块）

## 1. 适用对象

设某个球面单元上的二维积分已明确写成
$$I=\int_{s_0}^{s_1}\int_{t_0(s)}^{t_1(s)}f(s,t)\,dt\,ds,$$
并且经过端点参数化后，内层函数在开区间解析、端点至多具有可积代数或对数奇异性。只有在这些假设下，才可使用 tanh-sinh 变换。

## 2. 变换和 Jacobian 的逐步计算

对标准区间 $[-1,1]$，取
$$t=\tanh\!\left(\frac\pi2\sinh u\right).$$
令 $q(u)=\frac\pi2\sinh u$，则
$$\frac{dq}{du}=\frac\pi2\cosh u,$$
$$\frac{dt}{du}=\operatorname{sech}^2(q(u))\frac\pi2\cosh u.$$
因此
$$\int_{-1}^{1}g(t)dt=\int_{-\infty}^{\infty}g(t(u))\frac\pi2\frac{\cosh u}{\cosh^2(q(u))}\,du.$$
步长 $h$ 的对称梯形近似为
$$Q_{h,M}=h\sum_{j=-M}^{M}g(t(jh))\frac\pi2\frac{\cosh(jh)}{\cosh^2(q(jh))}.$$
一般区间 $[a,b]$ 再用 $t=(a+b)/2+(b-a)\xi/2$，并乘 Jacobian $(b-a)/2$。

## 3. 误差声明

DE 变换通常使尾项快速衰减，但“节点数 $15$ 或 $30$ 必然达到某精度”不是定理。可靠做法是计算
$$|Q_{h,M}-Q_{h/2,2M}|$$
并同时改变 $M$、$h$；若两次结果在目标范数下稳定，再与一个独立方法或已知特例比较。若被积函数有未处理的内部极点、分支跳跃或不可积奇异性，必须先分段、取主值或改用第十八、二十二节的定义。

---

# 第二十一小节：非整数幂次 $\alpha_k$ 的 Appell $F_1$ 解析闭环（条件性模块）

## 1. 径向积分和无量纲化

固定 $\mathbf u$，令 $\mathbf p=\mathbf x-\mathbf c_k$，
$$b=\mathbf p\cdot\mathbf u,\qquad d^2=\sigma_k^{-2}+\|\mathbf p\|^2-b^2>0.$$
则
$$1+\sigma_k^2\|\mathbf p-R\mathbf u\|^2
=\sigma_k^2\bigl((R-b)^2+d^2\bigr).$$
对 $0<A<B$，有
$$\mathcal J_\alpha(A,B)=\int_A^B\frac{[1+\sigma_k^2\|\mathbf p-R\mathbf u\|^2]^{-\alpha}}R\,dR
=\sigma_k^{-2\alpha}\int_A^B\frac{dR}{R[(R-b)^2+d^2]^\alpha}.$$
取 $R=dt$、$t=d\,q$ 只在 $d>0$ 时使用。更直接地取 $R=d\,r$、$\beta=b/d$，则 $dR=d\,dr$ 且
$$R-b=d(r-\beta),$$
$$\boxed{\mathcal J_\alpha(A,B)=\sigma_k^{-2\alpha}d^{-2\alpha}\int_{A/d}^{B/d}\frac{dr}{r[1+(r-\beta)^2]^\alpha}.}$$
这里没有额外的 $d^{-1}$：分子 $dR$ 与分母 $R=dr$ 的 $d$ 已相消。

## 2. 适用于 $F_1$ 的代换

令 $q=r-\beta$，再令
$$w=\frac{q-i}{q+i},\qquad q=i\frac{1+w}{1-w},$$
则
$$dq=\frac{2i}{(1-w)^2}dw,$$
$$1+q^2=-\frac{4w}{(1-w)^2}.$$
同时
$$r=q+\beta=\frac{(\beta+i)+(i-\beta)w}{1-w}
=(\beta+i)\frac{1-\lambda w}{1-w},
$$
其中
$$\lambda=\frac{\beta-i}{\beta+i},$$
要求 $\beta\ne-i$ 且路径不穿过变换的极点。选定同一连续分支后，逐项代入：
$$\frac{dq}{r(1+q^2)^\alpha}
=\frac{2i}{(\beta+i)(-4)^\alpha}
 w^{-\alpha}(1-w)^{2\alpha-1}(1-\lambda w)^{-1}dw.$$

## 3. Appell $F_1$ 原函数

Euler 型恒等式为
$$\int_0^z s^{a-1}(1-s)^{c-a-1}(1-x s)^{-b_1}(1-y s)^{-b_2}ds
=\frac{z^a}{a}F_1(a;b_1,b_2;c;xz,yz),$$
其中路径、参数和幂函数分支须连续选定。取
$$a=1-\alpha,\quad c=a+1=2-\alpha,\quad b_1=1,\quad b_2=1-2\alpha,$$
可得一条从固定基点 $w_0$ 到 $w$ 的原函数
$$\mathcal A_\alpha(w)=\frac{w^{1-\alpha}}{1-\alpha}
F_1(1-\alpha;1,1-2\alpha;2-\alpha;\lambda w,w),$$
这里使用了等价的参数重排；直接微分右端是最终检查。更稳妥的定积分写法是保留 Euler 积分：
$$\mathcal J_\alpha(A,B)=\sigma_k^{-2\alpha}d^{-2\alpha}\frac{2i}{(\beta+i)(-4)^\alpha}
\int_{w(A)}^{w(B)}w^{-\alpha}(1-w)^{2\alpha-1}(1-\lambda w)^{-1}dw.$$
当 $\alpha=1$ 时上式的 $1-\alpha$ 分母退化，应回到第九节的初等原函数，不能直接代入。

## 4. 分支和接触端点

若 $A=0$，原积分含 $dR/R$；其发散部分为
$$\phi_k(\mathbf x)\log\frac{\ell _0}{\varepsilon},$$
应先按第十六节减去，再对差值积分使用 Appell 表达式。若路径穿越 $w=0,1,1/\lambda$ 或幂函数割线，必须分段并在分段端点连续拼接；不能只使用主值 $F_1$ 数值而忽略跳跃。

---

# 第二十二小节：非代数剩余维度的二维数值保底（$H^{\mathrm{direct}}$）与总体算法流

## 1. 直接项的严格定义

对每个方向和每个实体段，直接项就是原始积分
$$H^{\rm direct}_{ij}(\mathbf x;\varepsilon)=G\sum_k w_k\int_{\mathbb S^2}T_{ij}(\mathbf u)
\sum_\ell\int_{[R^-_\ell,R^+_\ell]\cap[\varepsilon,\infty)}
\frac{\phi_k(\mathbf x-R\mathbf u)}R\,dR\,d\Omega,$$
再按第十六节减去显式 $\log\varepsilon$ 项并取极限。该定义与是否存在特殊函数闭式无关。

## 2. 分片、正则化和数值顺序

严格的数值流程是：
1. 由平面求交计算每个 $R_f(\mathbf u)$；
2. 按临界大圆弧把 $\mathbb S^2$ 分成单元 $\Omega_m$；
3. 对接触段先做
$$\frac{\phi_k(\mathbf x-R\mathbf u)}R
=\frac{\phi_k(\mathbf x)}R+\frac{\phi_k(\mathbf x-R\mathbf u)-\phi_k(\mathbf x)}R;$$
4. 第一项按解析对数积分，第二项按有界核积分；
5. 非接触段直接求积；
6. 最后加入 $-(4\pi G/3)\theta_{\rm ball}\rho I$。
每一步都保留矩阵分量，因此可分别检查对称性和迹。

## 3. Smolyak 只能提供一种求积组织方式

若把单元参数化为 $[0,1]^2$ 并得到一维嵌套规则 $Q_n^{(1)}$，Smolyak 算子可定义为
$$\mathcal A(q,d)=\sum_{|\boldsymbol\ell|_1\le q+d-1}
(-1)^{q+d-1-|\boldsymbol\ell|_1}\binom{d-1}{q+d-1-|\boldsymbol\ell|_1}
\bigotimes_{j=1}^d Q_{\ell_j}^{(1)}.$$
这是线性组合的定义，不自动给出误差。误差必须由函数的混合光滑性、规则阶数和实际细化差值估计；在临界弧、分支或对数端点附近应先用前两节的解析减法。

## 4. 总体组装和独立校验

最终按第十八节的分区先组装
$$H_K^{\rm FP}=H^{\rm ext}+H^{\rm res}-\frac{4\pi G}{3}\theta_{\rm ball}\rho(\mathbf x)I.$$
随后，$H^{\rm res}$ **至多选择一种**经验证的表示：直接积分 $H^{\rm direct}$，或用 $H^{\rm hyp}$、$H^{\rm hom}$、$H^{\rm hypgeom}$ 中的一种替换相同的残差；这些通道不能相加，否则会重复计数。

每次计算至少检查：
$$H_{ij}=H_{ji},\qquad \sum_iH_{ii}=-4\pi G\rho(\mathbf x)\quad(\mathbf x\in\operatorname{int}V),$$
$$\sum_iH_{ii}=0\quad(\mathbf x\notin\overline V),$$
并对均匀球体比较
$$H_{\rm in}=-\frac{4\pi G\rho_0}{3}I,\qquad
H_{\rm out}=\frac{GM}{r^3}(3\mathbf n\mathbf n^T-I).$$
这些是对符号、接触项、方向张量和数值残差的独立回归检验。
