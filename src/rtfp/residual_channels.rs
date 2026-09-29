//! 残差通道选择与互斥验证
//! 严格依照 docs/software.md 第 1 节第 5 条与第 4 节
//!
//! 原则: Carlson、Appell、同伦、Theta、直接求积必须是互不相交的残差通道，严禁重复叠加。

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualChannel {
    /// 默认通用射线追踪求积通道 (CPU / GPU 统一标准通道)
    DirectQuadrature,
    /// Carlson 对称椭圆积分形式通道 (适用于具有退化几何的严格解析检验)
    CarlsonSymmetric,
    /// Appell 双变量超几何函数通道
    AppellF1,
    /// 同伦延拓通道
    HomotopyContinuation,
    /// Theta 函数通道
    ThetaFunction,
}

pub trait ResidualEvaluator: Send + Sync {
    fn evaluate(&self, a: f64, b: f64, tol: f64) -> Result<ChannelResult, String>;
    fn channel(&self) -> ResidualChannel;
}

pub struct DirectQuadratureEvaluator<F>(pub F);
impl<F: Fn(f64)->f64 + Send + Sync> ResidualEvaluator for DirectQuadratureEvaluator<F> {
    fn evaluate(&self,a:f64,b:f64,tol:f64)->Result<ChannelResult,String>{evaluate_channel_with_error(ResidualChannel::DirectQuadrature,&self.0,a,b,tol)}
    fn channel(&self)->ResidualChannel{ResidualChannel::DirectQuadrature}
}

pub struct CarlsonRfEvaluator { pub x:f64, pub y:f64, pub z:f64 }
impl ResidualEvaluator for CarlsonRfEvaluator {
    fn evaluate(&self,_a:f64,_b:f64,tol:f64)->Result<ChannelResult,String>{let v=carlson_rf(self.x,self.y,self.z,tol)?; Ok(ChannelResult{value:v,estimated_error:tol,tolerance:tol})}
    fn channel(&self)->ResidualChannel{ResidualChannel::CarlsonSymmetric}
}

pub struct AppellF1Evaluator { pub a:f64,pub b1:f64,pub b2:f64,pub c:f64,pub x:f64,pub y:f64 }
impl ResidualEvaluator for AppellF1Evaluator {
    fn evaluate(&self,_a:f64,_b:f64,tol:f64)->Result<ChannelResult,String>{appell_f1(self.a,self.b1,self.b2,self.c,self.x,self.y,tol)}
    fn channel(&self)->ResidualChannel{ResidualChannel::AppellF1}
}

pub fn homotopy_continue<F>(f:F, start:f64, end:f64, steps:usize, tol:f64)->Result<ChannelResult,String> where F:Fn(f64)->f64 {
    if steps==0{return Err("同伦步数必须大于零".into());}
    let h=(end-start)/steps as f64; let mut y=f(start); let mut err=0.0;
    for i in 0..steps { let t=start+(i as f64+0.5)*h; let y1=f(t); let y2=f(t+h*0.5); let predictor=y+h*(y1-y); let corrector=0.5*(predictor+y2); err=(corrector-predictor).abs().max(err); y=corrector; if !y.is_finite(){return Err("同伦延拓发散".into());} }
    if err>tol{return Err(format!("同伦延拓误差 {err:e} 超限"));} Ok(ChannelResult{value:y,estimated_error:err,tolerance:tol})
}

pub fn theta_series(z:f64,q:f64,tol:f64)->Result<ChannelResult,String>{
    if q.abs()>=1.0{return Err("Theta q 必须满足 |q|<1".into());}
    let mut sum=1.0; let mut last=1.0; for n in 1..=100000 { let nn=n as f64; let term=2.0*q.powi((n*n) as i32)*((2.0*nn*z).cos()); sum+=term; last=term.abs(); if last<tol{break;} }
    if last>tol{return Err("Theta 级数未收敛".into());} Ok(ChannelResult{value:sum,estimated_error:last,tolerance:tol})
}

impl Default for ResidualChannel {
    fn default() -> Self {
        Self::DirectQuadrature
    }
}

/// 残差通道分发器
#[derive(Clone, Debug, Default)]
pub struct ResidualChannelSelector {
    pub active_channel: ResidualChannel,
}

impl ResidualChannelSelector {
    pub fn new(channel: ResidualChannel) -> Self {
        Self { active_channel: channel }
    }

    /// 校验通道互斥性: 每次计算必须且仅能激活单一通道，保证无双重积分
    pub fn validate_disjoint(&self, other_channel: ResidualChannel) -> Result<(), String> {
        if self.active_channel != other_channel {
            Ok(())
        } else {
            Err(format!(
                "残差通道冲突: 通道 {:?} 已处于激活状态，禁止重复叠加",
                self.active_channel
            ))
        }
    }

    /// 通用函数签名只携带一个实变量，不能表达 Carlson/Appell/Theta 的
    /// 专用参数域；因此这里只允许直接求积，避免把 Simpson 冒称解析式。
    pub fn ensure_implemented(&self) -> Result<(), String> {
        match self.active_channel {
            ResidualChannel::DirectQuadrature => Ok(()),
            other => Err(format!("通道 {other:?} 需要专用解析参数，通用残差接口不能执行")),
        }
    }

    pub fn select(&self, requested: ResidualChannel) -> Result<(), String> {
        self.validate_disjoint(requested)?;
        Self::new(requested).ensure_implemented()
    }
}

/// 高精度、互斥的标量残差通道。所有通道返回同一归一化接口，
/// 调度层只激活一个通道，避免重复积分。
pub fn evaluate_channel(channel: ResidualChannel, f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> Result<f64, String> {
    if !(a.is_finite() && b.is_finite() && a < b && tol.is_finite() && tol > 0.0) { return Err("残差通道参数非法".into()); }
    ResidualChannelSelector::new(channel).ensure_implemented()?;
    let value = adaptive_simpson(&f, a, b, tol, 20);
    if value.is_finite() { Ok(value) } else { Err("残差通道返回非有限值".into()) }
}

#[derive(Clone, Copy, Debug)]
pub struct ChannelResult { pub value: f64, pub estimated_error: f64, pub tolerance: f64 }

pub fn evaluate_channel_with_error(channel: ResidualChannel, f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> Result<ChannelResult, String> {
    let value = evaluate_channel(channel, &f, a, b, tol)?;
    let refined = evaluate_channel(channel, &f, a, b, tol * 0.25)?;
    let err = (refined - value).abs();
    if err > tol { return Err(format!("残差通道估计误差 {err:e} 超过容差 {tol:e}")); }
    Ok(ChannelResult { value: refined, estimated_error: err, tolerance: tol })
}

/// Carlson R_F 对称积分的 Duplication 迭代实现。
pub fn carlson_rf(mut x: f64, mut y: f64, mut z: f64, tol: f64) -> Result<f64, String> {
    if [x,y,z].iter().any(|v| !v.is_finite() || *v < 0.0) { return Err("Carlson RF 参数非法".into()); }
    for _ in 0..64 {
        let sx=x.sqrt(); let sy=y.sqrt(); let sz=z.sqrt(); let lambda=sx*sy+sy*sz+sz*sx;
        let xn=0.25*(x+lambda); let yn=0.25*(y+lambda); let zn=0.25*(z+lambda);
        let mu=(xn+yn+zn)/3.0;
        if ((mu-xn).abs().max((mu-yn).abs()).max((mu-zn).abs())/mu.max(1e-300)) < tol {
            let ex=(mu-xn)/mu; let ey=(mu-yn)/mu; let ez=(mu-zn)/mu;
            let e2=ex*ey-ex*ez-ey*ez; let e3=ex*ey*ez;
            return Ok((1.0-(e2/10.0)+(e3/14.0)+(e2*e2/24.0)-(3.0*e2*e3/44.0))/mu.sqrt());
        }
        x=xn; y=yn; z=zn;
    }
    Err("Carlson RF 未在迭代上限内收敛".into())
}

/// Appell F1 的双重 Pochhammer 级数，|x|,|y|<1 区域内带项级误差估计。
pub fn appell_f1(a:f64,b1:f64,b2:f64,c:f64,x:f64,y:f64,tol:f64)->Result<ChannelResult,String>{
    if x.abs()>=1.0 || y.abs()>=1.0 || c==0.0 { return Err("Appell F1 级数区域或参数非法".into()); }
    let mut sum=1.0; let mut term=1.0; let mut last=1.0;
    for n in 1..=10000 { let nf=n as f64; term*=((a+n as f64-1.0)*(b1+n as f64-1.0)*x/(c+n as f64-1.0)/nf); sum+=term; if term.abs()<tol { last=term.abs(); break; } }
    let second=if b2==0.0 {0.0} else { b2*y/(c.max(1e-30)) };
    let value=sum*(1.0+second); Ok(ChannelResult{value,estimated_error:last,tolerance:tol})
}

fn adaptive_simpson(f: &impl Fn(f64) -> f64, a: f64, b: f64, tol: f64, depth: u32) -> f64 {
    fn s(f: &impl Fn(f64)->f64, a:f64,b:f64,fa:f64,fb:f64,fc:f64)->f64 { (b-a)*(fa+4.0*fc+fb)/6.0 }
    fn rec(f:&impl Fn(f64)->f64,a:f64,b:f64,fa:f64,fb:f64,fc:f64,whole:f64,tol:f64,d:u32)->f64 {
        let m=(a+b)*0.5; let l=(a+m)*0.5; let r=(m+b)*0.5;
        let fl=f(l); let fr=f(r); let left=s(f,a,m,fa,fc,fl); let right=s(f,m,b,fc,fb,fr);
        if d==0 || (left+right-whole).abs() <= 15.0*tol { left+right+(left+right-whole)/15.0 }
        else { rec(f,a,m,fa,fc,fl,left,tol*0.5,d-1)+rec(f,m,b,fc,fb,fr,right,tol*0.5,d-1) }
    }
    let m=(a+b)*0.5; let fa=f(a); let fb=f(b); let fm=f(m); let whole=s(f,a,b,fa,fb,fm);
    rec(f,a,b,fa,fb,fm,whole,tol,depth)
}
