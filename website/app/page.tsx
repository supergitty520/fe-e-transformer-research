"use client";

import { useEffect, useMemo, useState } from "react";

type EnvironmentId = "none" | "high_frequency" | "energy" | "concentration";
type VariantId = "gs_sham" | "gsf_q01" | "gsf_q03" | "gsf_q05";
type DataMetric = "confirmed" | "slope" | "time";

type Run = {
  environment: EnvironmentId;
  variant: VariantId;
  label: string;
  dose: number;
  confirmed: number;
  onset10: number;
  first50: number;
  first90: number;
  first99: number;
  transitionWidth: number;
  slope: number;
  saved: number;
  timeRatio: number;
  stepRatio: number;
  cosine: number | null;
  interventions: number;
};

const ENVIRONMENTS: Array<{
  id: EnvironmentId;
  short: string;
  title: string;
  description: string;
}> = [
  {
    id: "none",
    short: "正常传播",
    title: "无传播噪声",
    description: "检验FE-E作为默认训练组件是否存在普遍收益。",
  },
  {
    id: "high_frequency",
    short: "高频扰动",
    title: "交替层深高频扰动",
    description: "残差传播在相邻深度方向交替放大与缩小。",
  },
  {
    id: "energy",
    short: "能量放大",
    title: "全局残差能量放大",
    description: "所有残差增量同时放大，检验质量约束的响应。",
  },
  {
    id: "concentration",
    short: "中层集中",
    title: "中层梯度能量集中",
    description: "传播能量持续集中在中间层，检验熵约束的条件性作用。",
  },
];

const RUNS: Run[] = [
  { environment: "none", variant: "gs_sham", label: "GS-SHAM", dose: 0, confirmed: 1536, onset10: 1408, first50: 1472, first90: 1472, first99: 1472, transitionWidth: 64, slope: -0.019718, saved: 0, timeRatio: 1, stepRatio: 1, cosine: null, interventions: 0 },
  { environment: "none", variant: "gsf_q01", label: "GSF 1%", dose: 1, confirmed: 1600, onset10: 1440, first50: 1536, first90: 1536, first99: 1536, transitionWidth: 96, slope: -0.008603, saved: -64, timeRatio: 1.109, stepRatio: 1.065, cosine: -0.250, interventions: 16 },
  { environment: "none", variant: "gsf_q03", label: "GSF 3%", dose: 3, confirmed: 1632, onset10: 1440, first50: 1536, first90: 1568, first99: 1568, transitionWidth: 128, slope: 0.005047, saved: -96, timeRatio: 1.211, stepRatio: 1.140, cosine: -0.253, interventions: 48 },
  { environment: "none", variant: "gsf_q05", label: "GSF 5%", dose: 5, confirmed: 1888, onset10: 1376, first50: 1792, first90: 1824, first99: 1824, transitionWidth: 448, slope: -0.011926, saved: -352, timeRatio: 1.486, stepRatio: 1.209, cosine: -0.237, interventions: 95 },
  { environment: "high_frequency", variant: "gs_sham", label: "GS-SHAM", dose: 0, confirmed: 1344, onset10: 1184, first50: 1248, first90: 1280, first99: 1280, transitionWidth: 96, slope: -0.026536, saved: 0, timeRatio: 1, stepRatio: 1, cosine: null, interventions: 0 },
  { environment: "high_frequency", variant: "gsf_q01", label: "GSF 1%", dose: 1, confirmed: 1312, onset10: 1120, first50: 1184, first90: 1216, first99: 1248, transitionWidth: 96, slope: -0.010282, saved: 32, timeRatio: 0.990, stepRatio: 1.014, cosine: -0.236, interventions: 13 },
  { environment: "high_frequency", variant: "gsf_q03", label: "GSF 3%", dose: 3, confirmed: 1536, onset10: 1408, first50: 1440, first90: 1472, first99: 1472, transitionWidth: 64, slope: -0.019720, saved: -192, timeRatio: 1.195, stepRatio: 1.046, cosine: -0.266, interventions: 46 },
  { environment: "high_frequency", variant: "gsf_q05", label: "GSF 5%", dose: 5, confirmed: 1472, onset10: 1312, first50: 1376, first90: 1376, first99: 1408, transitionWidth: 64, slope: -0.007891, saved: -128, timeRatio: 1.178, stepRatio: 1.075, cosine: -0.213, interventions: 74 },
  { environment: "energy", variant: "gs_sham", label: "GS-SHAM", dose: 0, confirmed: 1440, onset10: 1280, first50: 1344, first90: 1376, first99: 1376, transitionWidth: 96, slope: -0.000617, saved: 0, timeRatio: 1, stepRatio: 1, cosine: null, interventions: 0 },
  { environment: "energy", variant: "gsf_q01", label: "GSF 1%", dose: 1, confirmed: 1504, onset10: 1344, first50: 1408, first90: 1408, first99: 1440, transitionWidth: 64, slope: -0.008611, saved: -64, timeRatio: 1.046, stepRatio: 1.001, cosine: -0.215, interventions: 15 },
  { environment: "energy", variant: "gsf_q03", label: "GSF 3%", dose: 3, confirmed: 1696, onset10: 1536, first50: 1600, first90: 1632, first99: 1632, transitionWidth: 96, slope: -0.005587, saved: -256, timeRatio: 1.215, stepRatio: 1.032, cosine: -0.271, interventions: 51 },
  { environment: "energy", variant: "gsf_q05", label: "GSF 5%", dose: 5, confirmed: 1600, onset10: 1440, first50: 1504, first90: 1504, first99: 1536, transitionWidth: 64, slope: -0.013864, saved: -160, timeRatio: 1.170, stepRatio: 1.053, cosine: -0.223, interventions: 80 },
  { environment: "concentration", variant: "gs_sham", label: "GS-SHAM", dose: 0, confirmed: 1952, onset10: 1792, first50: 1856, first90: 1856, first99: 1888, transitionWidth: 64, slope: -0.002335, saved: 0, timeRatio: 1, stepRatio: 1, cosine: null, interventions: 0 },
  { environment: "concentration", variant: "gsf_q01", label: "GSF 1%", dose: 1, confirmed: 1984, onset10: 1760, first50: 1888, first90: 1888, first99: 1920, transitionWidth: 128, slope: -0.007335, saved: -32, timeRatio: 1.028, stepRatio: 1.011, cosine: -0.185, interventions: 20 },
  { environment: "concentration", variant: "gsf_q03", label: "GSF 3%", dose: 3, confirmed: 1824, onset10: 1664, first50: 1728, first90: 1728, first99: 1760, transitionWidth: 64, slope: -0.009560, saved: 128, timeRatio: 0.954, stepRatio: 1.020, cosine: -0.208, interventions: 55 },
  { environment: "concentration", variant: "gsf_q05", label: "GSF 5%", dose: 5, confirmed: 1888, onset10: 1728, first50: 1792, first90: 1824, first99: 1824, transitionWidth: 96, slope: -0.013759, saved: 64, timeRatio: 1.007, stepRatio: 1.041, cosine: -0.195, interventions: 95 },
];

const VARIANT_COLORS: Record<VariantId, string> = {
  gs_sham: "#171a16",
  gsf_q01: "#267a73",
  gsf_q03: "#e14b2f",
  gsf_q05: "#7555b7",
};

const POLICIES = {
  none: {
    label: "完全不提示",
    note: "保留探索空间，但持续卡住时没有纠偏。",
    short: "依学习者而定",
    transfer: "可能较高",
    dependence: "低",
  },
  fixed: {
    label: "固定频率提示",
    note: "短期练习更流畅，也最容易形成提示依赖。",
    short: "通常较快",
    transfer: "未必改善",
    dependence: "高",
  },
  observer: {
    label: "观测器触发",
    note: "先区分有益探索与持续失稳，再决定是否介入。",
    short: "依状态调整",
    transfer: "待实验验证",
    dependence: "目标是降低",
  },
} as const;

type PolicyId = keyof typeof POLICIES;

function signed(value: number) {
  return value > 0 ? "+" + value : String(value);
}

function metricValue(run: Run, metric: DataMetric) {
  if (metric === "confirmed") return run.confirmed + " 步";
  if (metric === "slope") return run.slope.toFixed(5);
  return run.timeRatio.toFixed(3) + "×";
}

function metricNote(run: Run, metric: DataMetric) {
  if (metric === "confirmed") return signed(run.saved) + " 步 vs SHAM";
  if (metric === "slope") return "每100步 task loss";
  return run.timeRatio <= 1 ? "墙钟时间减少" : "墙钟时间增加";
}

export default function Home() {
  const [environment, setEnvironment] = useState<EnvironmentId>("energy");
  const [metric, setMetric] = useState<DataMetric>("confirmed");
  const [policy, setPolicy] = useState<PolicyId>("observer");
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const update = () => {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0);
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, []);

  const activeRuns = useMemo(
    () => RUNS.filter((run) => run.environment === environment),
    [environment],
  );
  const activeEnvironment = ENVIRONMENTS.find((item) => item.id === environment)!;
  const chartMax = Math.ceil(Math.max(...activeRuns.map((run) => run.confirmed)) / 250) * 250;
  const ticks = Array.from({ length: chartMax / 250 + 1 }, (_, index) => index * 250);
  const phasePosition = (step: number) => (step / chartMax) * 100;
  const policyData = POLICIES[policy];

  return (
    <main>
      <div className="reading-progress" style={{ width: String(progress) + "%" }} />

      <header className="topbar">
        <a className="brand" href="#top" aria-label="返回顶部">
          <span className="brand-mark" aria-hidden="true">F∴E</span>
          <span>传播约束实验室</span>
        </a>
        <nav aria-label="页面导航">
          <a href="#experiment">实验</a>
          <a href="#counteraction">反作用</a>
          <a href="#observer">观测器</a>
          <a href="#education">教育引申</a>
        </nav>
        <span className="topbar-status">seed 47 · MLX</span>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">FE–E 实验、反证与认知引申</p>
          <h1>刚度约束和信息熵<br />对深度学习的<span>反作用</span>及引申</h1>
          <p className="hero-lead">
            一种约束可以让局部传播更规整、损失下降更陡，
            却未必让模型更早学会。稳定不是学习的同义词。
          </p>
          <div className="hero-actions">
            <a className="primary-action" href="#experiment">进入实验台 <span>↓</span></a>
            <a className="text-action" href="#article">阅读全文</a>
          </div>
        </div>
        <aside className="hero-panel" aria-label="实验关键结论">
          <p className="panel-kicker">核心反例</p>
          <div className="formula-card">
            <span>更陡的局部斜率</span>
            <strong>≠</strong>
            <span>更早的学习相变</span>
          </div>
          <div className="hero-stat-grid">
            <div><strong>3</strong><span>/ 12 配对提前</span></div>
            <div><strong>9</strong><span>/ 12 配对延后</span></div>
            <div><strong>2.24×</strong><span>单介入步成本</span></div>
            <div><strong>1</strong><span>个模型种子</span></div>
          </div>
          <p className="boundary-note">机制证据，不是通用优化器结论。</p>
        </aside>
      </section>

      <section className="protocol-strip" aria-label="实验协议">
        <span>128 层</span>
        <span>宽度 32</span>
        <span>4 注意力头</span>
        <span>26,208 训练步</span>
        <span>8-batch 固定验证</span>
        <span>99% × 连续 3 次</span>
      </section>

      <section className="section experiment-section" id="experiment">
        <div className="section-heading">
          <div>
            <p className="eyebrow">01 / 实验台</p>
            <h2>把“学得快”拆成<br />平台期与相变期</h2>
          </div>
          <p>
            四种传播环境、四种介入剂量。时间轴只展示日志中真实记录的
            10%、90%与连续确认里程碑，不用平滑曲线制造确定性。
          </p>
        </div>

        <div className="environment-tabs" role="tablist" aria-label="选择传播环境">
          {ENVIRONMENTS.map((item) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={environment === item.id}
              className={environment === item.id ? "active" : ""}
              onClick={() => setEnvironment(item.id)}
            >
              <span>{item.short}</span>
              <small>{item.id === "none" ? "CONTROL" : "STRESS"}</small>
            </button>
          ))}
        </div>

        <div className="experiment-intro">
          <div>
            <span className="status-dot" />
            <strong>{activeEnvironment.title}</strong>
          </div>
          <p>{activeEnvironment.description}</p>
        </div>

        <div className="phase-card">
          <div className="phase-head">
            <div>
              <span className="mini-label">PHASE RACE</span>
              <h3>学习相变里程碑</h3>
            </div>
            <div className="phase-legend" aria-label="图例">
              <span><i className="mark-10" />持续10%</span>
              <span><i className="mark-90" />首次90%</span>
              <span><i className="mark-confirm" />确认99%</span>
            </div>
          </div>

          <div className="timeline">
            <div className="timeline-axis" aria-hidden="true">
              {ticks.map((tick) => (
                <span key={tick} style={{ left: String(phasePosition(tick)) + "%" }}>
                  {tick}
                </span>
              ))}
            </div>
            {activeRuns.map((run) => (
              <div className="timeline-row" key={run.variant}>
                <div className="timeline-label">
                  <i style={{ backgroundColor: VARIANT_COLORS[run.variant] }} />
                  <span>{run.label}</span>
                </div>
                <div className="timeline-track">
                  {ticks.map((tick) => (
                    <span
                      className="gridline"
                      aria-hidden="true"
                      key={tick}
                      style={{ left: String(phasePosition(tick)) + "%" }}
                    />
                  ))}
                  <span
                    className="phase-span"
                    style={{
                      left: String(phasePosition(run.onset10)) + "%",
                      width: String(phasePosition(run.first90 - run.onset10)) + "%",
                      backgroundColor: VARIANT_COLORS[run.variant],
                    }}
                  />
                  <button
                    className="phase-marker marker-10"
                    style={{ left: String(phasePosition(run.onset10)) + "%" }}
                    aria-label={run.label + " 持续10%：" + run.onset10 + "步"}
                  >
                    <span>10% · {run.onset10}</span>
                  </button>
                  <button
                    className="phase-marker marker-90"
                    style={{ left: String(phasePosition(run.first90)) + "%" }}
                    aria-label={run.label + " 首次90%：" + run.first90 + "步"}
                  >
                    <span>90% · {run.first90}</span>
                  </button>
                  <button
                    className="phase-marker marker-confirm"
                    style={{
                      left: String(phasePosition(run.confirmed)) + "%",
                      borderColor: VARIANT_COLORS[run.variant],
                    }}
                    aria-label={run.label + " 连续确认99%：" + run.confirmed + "步"}
                  >
                    <span>
                      50% {run.first50} · 99% {run.first99} · 确认 {run.confirmed}
                    </span>
                  </button>
                </div>
                <div className={"saved-chip " + (run.saved > 0 ? "positive" : run.saved < 0 ? "negative" : "")}>
                  {run.variant === "gs_sham" ? "基线" : signed(run.saved) + "步"}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="metric-toolbar" aria-label="切换数据指标">
          <span>查看指标</span>
          {([
            ["confirmed", "确认步数"],
            ["slope", "短期斜率"],
            ["time", "总墙钟比"],
          ] as Array<[DataMetric, string]>).map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={metric === id ? "active" : ""}
              onClick={() => setMetric(id)}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="metric-grid">
          {activeRuns.map((run) => (
            <article key={run.variant} className="metric-card">
              <div className="metric-card-head">
                <i style={{ backgroundColor: VARIANT_COLORS[run.variant] }} />
                <span>{run.label}</span>
                <small>{run.interventions}次介入</small>
              </div>
              <strong>{metricValue(run, metric)}</strong>
              <p>{metricNote(run, metric)}</p>
              <div className="mini-meter">
                <span
                  style={{
                    width:
                      metric === "confirmed"
                        ? String((run.confirmed / chartMax) * 100) + "%"
                        : metric === "time"
                          ? String(Math.min(run.timeRatio / 1.5, 1) * 100) + "%"
                          : String(Math.min(Math.abs(run.slope) / 0.03, 1) * 100) + "%",
                    backgroundColor: VARIANT_COLORS[run.variant],
                  }}
                />
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="section paradox-section" id="counteraction">
        <div className="section-heading inverse-heading">
          <div>
            <p className="eyebrow">02 / 反作用</p>
            <h2>斜率是一段速度，<br />不是抵达时间</h2>
          </div>
          <p>
            点越靠左，loss下降越陡；越靠上，越早于同环境SHAM达标。
            如果斜率可以预测收敛，所有点都应沿左上方向排列——实际并没有。
          </p>
        </div>

        <div className="paradox-grid">
          <div className="scatter-card">
            <div className="scatter-label y-label">提前步数 ↑</div>
            <div className="scatter-label x-label">斜率更陡 ←　相变前loss斜率　→ 上升</div>
            <div className="scatter-plot">
              <span className="zero-line horizontal" aria-hidden="true" />
              <span className="zero-line vertical" aria-hidden="true" />
              {activeRuns.map((run) => {
                const x = ((run.slope + 0.03) / 0.04) * 100;
                const y = (1 - (run.saved + 360) / 500) * 100;
                return (
                  <button
                    type="button"
                    key={run.variant}
                    className="scatter-point"
                    style={{
                      left: String(Math.max(3, Math.min(97, x))) + "%",
                      top: String(Math.max(3, Math.min(97, y))) + "%",
                      backgroundColor: VARIANT_COLORS[run.variant],
                    }}
                    aria-label={run.label + "，斜率" + run.slope.toFixed(5) + "，相对SHAM" + signed(run.saved) + "步"}
                  >
                    <span>
                      <b>{run.label}</b>
                      斜率 {run.slope.toFixed(5)}<br />
                      终点 {signed(run.saved)} 步
                    </span>
                  </button>
                );
              })}
            </div>
            <div className="scatter-corners" aria-hidden="true">
              <span>陡，但更慢</span>
              <span>陡，而且更快</span>
            </div>
          </div>

          <article className="counterexample-card">
            <p className="mini-label">GLOBAL ENERGY / 5% FE-E</p>
            <h3>最清楚的反例</h3>
            <div className="counter-lines">
              <div>
                <span>短期loss斜率</span>
                <strong>−0.00062 → −0.01386</strong>
                <small>看起来下降明显加快</small>
              </div>
              <div>
                <span>持续10%相变起点</span>
                <strong>1280 → 1440</strong>
                <small>真正开始学习晚了160步</small>
              </div>
              <div>
                <span>99%连续确认</span>
                <strong>1440 → 1600</strong>
                <small>最终仍然晚了160步</small>
              </div>
            </div>
            <blockquote>
              FE-E压缩了相变内部，却延长了相变之前的平台。
            </blockquote>
          </article>
        </div>

        <div className="time-equation">
          <span>T<sub>99</sub></span>
          <b>=</b>
          <div><strong>平台等待</strong><small>何时找到方向</small></div>
          <b>+</b>
          <div><strong>相变宽度</strong><small>找到后走多快</small></div>
          <b>+</b>
          <div><strong>连续确认</strong><small>终点是否稳定</small></div>
        </div>
      </section>

      <article className="article-shell" id="article">
        <header className="article-header">
          <p className="eyebrow">03 / 完整文章</p>
          <h2>约束为什么会反过来妨碍学习</h2>
          <p>
            FE-E最初试图解决传播失稳；实验迫使问题发生改变：
            “规整的传播”与“有效的学习”之间，到底隔着什么？
          </p>
        </header>

        <div className="article-body">
          <section>
            <span className="section-number">Ⅰ</span>
            <h3>约束的初衷</h3>
            <p>
              深度学习中的许多困难都可以被描述为传播问题。网络越深，隐藏状态和梯度经过的层级越多，
              局部误差也越可能在反复复合中被放大。梯度爆炸、梯度消失、层间振荡、特征集中和表示坍缩，
              背后共同追问的是：信息能否以稳定的形态穿过深层网络。
            </p>
            <p>
              FE-E把网络深度视为一维网格，将各层梯度视为网格节点上的向量场。刚度项抑制相邻层突变，
              质量项约束总梯度能量，信息熵防止能量集中在少数层或特征方向。
            </p>
            <div className="formula-row">
              <code>E<sub>stiff</sub> = Σ ‖g<sub>l+1</sub> − g<sub>l</sub>‖² / Δz</code>
              <code>H<sub>g</sub> = −Σ p<sub>l</sub> log(p<sub>l</sub> + ε)</code>
            </div>
            <p>
              这里隐含着一个容易被忽略的前提：传播越平滑、能量越稳定、分布越均匀，学习就越好。
              后续实验表明，这个前提并不总是成立。
            </p>
          </section>

          <section>
            <span className="section-number">Ⅱ</span>
            <h3>实验出现的反向结果</h3>
            <p>
              在128层MLX实验中，基线采用AdamW与Gradient Smoothing。FE-E单次施加量被限制为任务梯度范数的5%，
              再以1%、3%和5%的冻结频率介入。12个FE-E配对中只有3个更早到达终点，其余9个更慢。
            </p>
            <p>
              正常环境下纯GS在1536步确认达标，三个FE-E剂量分别需要1600、1632和1888步。
              中层能量集中环境的3% FE-E相对同环境SHAM提前128步，但仍慢于正常GS，而且只是单种子、
              从12个比较中出现的条件性局部正点，不能称为已找到最佳介入率。
            </p>
            <p>
              单个FE-E介入步的中位耗时约为普通步骤的2.24倍。FE-E梯度与任务梯度的平均余弦在所有测试单元中均为负，
              说明它通常是在牺牲一部分任务方向，换取传播形态的规整。
            </p>
          </section>

          <section>
            <span className="section-number">Ⅲ</span>
            <h3>三种反作用</h3>
            <div className="mechanism-list">
              <div>
                <b>刚度的反作用</b>
                <p>高频变化不必然是噪声。少数层率先发生较大更新，可能正是建立新表示和重新分工所必需的过程。</p>
              </div>
              <div>
                <b>信息熵的反作用</b>
                <p>梯度集中既可能是坍缩，也可能是有效聚焦。熵只能判断是否集中，不能判断集中是否有害。</p>
              </div>
              <div>
                <b>质量能量的反作用</b>
                <p>学习未必需要恒定能量。某些相变依赖短暂脉冲，错误的参考能量会同时压制爆炸与有益跃迁。</p>
              </div>
            </div>
            <p>
              FE-E不是中性的数学修复器。它包含明确的结构偏好：连续、稳定、不过度集中。
              只有当任务中的异常恰好违反这些偏好，而且这种违反确实有害时，约束才可能产生净收益。
            </p>
          </section>
        </div>
      </article>

      <section className="section observer-section" id="observer">
        <div className="section-heading">
          <div>
            <p className="eyebrow">04 / 角色转换</p>
            <h2>不再命令梯度，<br />先观察传播</h2>
          </div>
          <p>
            把刚度、质量与熵从损失函数中抽离，变成停止梯度的状态指标。
            它们首先要证明自己能预测未来失稳，而不是直接获得干预权。
          </p>
        </div>

        <div className="observer-flow">
          <article>
            <span>01</span>
            <b>测量</b>
            <p>归一化刚度、总能量、层深熵</p>
          </article>
          <i aria-hidden="true">→</i>
          <article>
            <span>02</span>
            <b>持续性判断</b>
            <p>区分瞬时探索与连续有害异常</p>
          </article>
          <i aria-hidden="true">→</i>
          <article>
            <span>03</span>
            <b>异常分类</b>
            <p>突变、爆炸、衰减或层间集中</p>
          </article>
          <i aria-hidden="true">→</i>
          <article>
            <span>04</span>
            <b>最小干预</b>
            <p>裁剪、降学习率、短暂平滑或回滚</p>
          </article>
        </div>

        <div className="observer-principle">
          <p>新的研究终点</p>
          <strong>
            当前FE-E指标
            <span>能否预测</span>
            未来64～128步的训练恶化？
          </strong>
          <small>如果不能预测，观测器路线同样应当终止。</small>
        </div>
      </section>

      <section className="education-section" id="education">
        <div className="education-copy">
          <p className="eyebrow">05 / 教育引申</p>
          <h2>帮助得更多，<br />不等于学会得更早</h2>
          <p>
            如果把Transformer理解为多阶段认知加工的计算隐喻，持续FE-E介入就像过度脚手架：
            它让眼前步骤更整齐，却可能压缩试错、策略切换和概念重组的空间。
          </p>
          <p>
            这不是“网络层等于脑区”的神经科学主张，而是一条可检验的教育假说：
            干预应优化无提示掌握、迁移与延迟保持，而不只是练习期斜率。
          </p>
        </div>

        <div className="policy-lab">
          <div className="policy-tabs" role="tablist" aria-label="选择教育干预策略">
            {(Object.keys(POLICIES) as PolicyId[]).map((id) => (
              <button
                type="button"
                role="tab"
                aria-selected={policy === id}
                className={policy === id ? "active" : ""}
                key={id}
                onClick={() => setPolicy(id)}
              >
                {POLICIES[id].label}
              </button>
            ))}
          </div>
          <div className="policy-result">
            <p className="mini-label">概念模型 · 不是实验结果</p>
            <h3>{policyData.label}</h3>
            <p>{policyData.note}</p>
            <dl>
              <div><dt>短期表现</dt><dd>{policyData.short}</dd></div>
              <div><dt>迁移能力</dt><dd>{policyData.transfer}</dd></div>
              <div><dt>提示依赖</dt><dd>{policyData.dependence}</dd></div>
            </dl>
          </div>
        </div>

        <div className="education-thesis">
          <span>教育命题</span>
          <blockquote>
            教育干预的价值不在于让短期错误下降得最快，
            而在于让学习者更早形成稳定、可迁移的认知结构。
          </blockquote>
        </div>
      </section>

      <section className="closing-section">
        <p className="eyebrow">结语</p>
        <h2>约束从命令变成观测，<br />问题才真正开始。</h2>
        <p>
          真正重要的不是让局部轨迹始终规整，而是判断何种波动正在破坏学习，
          何种波动正在孕育新的结构。什么时候应该介入，什么时候应该允许探索——
          这也许是FE-E实验留下的最重要启示。
        </p>
        <div className="closing-meta">
          <span>实验口径：128层 · seed 47 · 单种子机制研究</span>
          <span>结论边界：不外推到人脑或大模型生产训练</span>
        </div>
      </section>

      <footer>
        <span>FE–E / STIFFNESS × ENTROPY</span>
        <a href="#top">回到顶部 ↑</a>
      </footer>
    </main>
  );
}
