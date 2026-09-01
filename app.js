(() => {
  "use strict";

  const DIM = 32;
  const ARMS = ["P", "B"];
  const RIDGE = 1.0;
  const FORGETTING = 0.90;
  const UPDATE_WEIGHT = 1.0;
  const ALPHA = 0.5;
  const ALPHA_MAX_SCALE = 1.60;
  const SCORE_TIE_EPS = 0.000001;
  const SCORE_TEMP = 2.0;
  const PROB_MIN = 0.42;
  const PROB_MAX = 0.58;
  const AVG_CARDS_PER_HAND = 4.9;
  const DECKS = 8;
  const TOTAL_CARDS = 52 * DECKS;
  const STORAGE_KEY = "bgs32d_frozen_direct_tech_panel_v1";

  const SHOE_NAMES = [
    "remaining_cards_ratio","penetration_ratio","estimated_hands_remaining_norm","shoe_maturity_ratio",
    "rank_A_relative_ratio","rank_2_relative_ratio","rank_3_relative_ratio","rank_4_relative_ratio",
    "rank_5_relative_ratio","rank_6_relative_ratio","rank_7_relative_ratio","rank_8_relative_ratio",
    "rank_9_relative_ratio","rank_10JQK_relative_ratio","physical_edge_proxy","shoe_information_reliability"
  ];
  const ROAD_NAMES = [
    "current_side_banker_binary","current_run_length_norm","previous_run_length_norm","previous2_run_length_norm",
    "recent5_banker_ratio","recent8_banker_ratio","recent12_banker_ratio","recent5_turn_rate",
    "recent8_turn_rate","recent12_turn_rate","run_length_hazard_rate","hsmm_stable_probability",
    "big_eye_regularity","small_road_regularity","cockroach_road_regularity","derived_road_consensus"
  ];

  const el = id => document.getElementById(id);
  const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));

  let state = {
    history: [],
    active: false,
    brain: null,
    lastPrediction: null
  };

  function freshArm() {
    return { A: eye(DIM), b: Array(DIM).fill(0), n: 0, effective_n: 0 };
  }

  function freshBrain() {
    return {
      arms: { P: freshArm(), B: freshArm() },
      updates: 0,
      last_selected: "",
      selection_streak: 0
    };
  }

  function eye(n) {
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => i === j ? RIDGE : 0)
    );
  }

  function save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (_) {}
  }

  function load() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (raw && Array.isArray(raw.history)) {
        state = {
          history: raw.history.filter(x => ["B","P","T"].includes(x)).slice(-500),
          active: !!raw.active,
          brain: raw.brain && raw.brain.arms ? raw.brain : freshBrain(),
          lastPrediction: raw.lastPrediction || null
        };
      }
    } catch (_) {}
  }

  function bp(seq) {
    return seq.filter(x => x === "B" || x === "P");
  }

  function runs(seq) {
    const s = bp(seq);
    if (!s.length) return [];
    const out = [];
    let side = s[0], n = 1;
    for (const v of s.slice(1)) {
      if (v === side) n++;
      else { out.push([side, n]); side = v; n = 1; }
    }
    out.push([side, n]);
    return out;
  }

  function bankerRatio(seq, w) {
    const a = bp(seq).slice(-Math.max(1, w));
    return a.length ? a.filter(x => x === "B").length / a.length : 0.5;
  }

  function turnRate(seq, w) {
    const a = bp(seq).slice(-Math.max(2, w));
    if (a.length < 2) return 0.5;
    let t = 0;
    for (let i = 1; i < a.length; i++) if (a[i] !== a[i - 1]) t++;
    return t / (a.length - 1);
  }

  function derivedMark(heights, c, row, newCol, offset) {
    if (newCol) {
      if (c < offset + 1) return "";
      return heights[c - 1] === heights[c - 1 - offset] ? "R" : "U";
    }
    if (c < offset) return "";
    const refHeight = heights[c - offset];
    const sameRow = refHeight >= row;
    const aboveRow = refHeight >= row - 1;
    return sameRow === aboveRow ? "R" : "U";
  }

  function buildDerivedRoads(seq) {
    const s = bp(seq);
    const sides = [];
    const heights = [];
    const out = { big_eye: [], small_road: [], cockroach_road: [] };
    const offsets = { big_eye: 1, small_road: 2, cockroach_road: 3 };

    for (const side of s) {
      const newCol = !sides.length || side !== sides[sides.length - 1];
      if (newCol) {
        sides.push(side);
        heights.push(1);
      } else {
        heights[heights.length - 1]++;
      }
      const c = heights.length - 1;
      const row = heights[heights.length - 1];
      for (const [name, offset] of Object.entries(offsets)) {
        const mark = derivedMark(heights, c, row, newCol, offset);
        if (mark) out[name].push(mark);
      }
    }
    return out;
  }

  function regularity(values, w = 8) {
    const a = values.slice(-w).filter(x => x === "R" || x === "U");
    if (!a.length) return [0.5, 0];
    return [a.filter(x => x === "R").length / a.length, a.length];
  }

  function lengthBucket(n) {
    n = Math.max(1, Math.floor(n));
    return n <= 5 ? String(n) : "6+";
  }

  function hazardContexts(side, cur, prev) {
    const ph = prev.length ? prev[prev.length - 1] : 0;
    const deltas = [];
    for (let i = 1; i < prev.length; i++) {
      deltas.push(prev[i] > prev[i - 1] ? "UP" : prev[i] < prev[i - 1] ? "DOWN" : "EQUAL");
    }
    const d1 = deltas.length ? deltas[deltas.length - 1] : "NA";
    const d2 = deltas.length >= 2 ? deltas[deltas.length - 2] : "NA";
    const c = lengthBucket(cur), p = ph ? lengthBucket(ph) : "0";
    return [
      ["full", `HZF|side=${side || "NA"}|cur=${c}|prev=${p}|d1=${d1}|d2=${d2}`],
      ["structure", `HZS|cur=${c}|prev=${p}|d1=${d1}|d2=${d2}`],
      ["shape", `HZP|cur=${c}|prev=${p}|d1=${d1}`],
      ["length", `HZL|cur=${c}`],
      ["global", "HZG|GLOBAL"]
    ];
  }

  function hazardTable(rs) {
    const done = rs.slice(0, -1);
    const heights = done.map(x => x[1]);
    const table = {};
    const bucket = key => table[key] || (table[key] = { CONTINUE: 0, TURN: 0 });
    done.forEach(([side, finalLen], idx) => {
      const prev = heights.slice(0, idx);
      for (let at = 1; at <= Math.max(1, finalLen); at++) {
        const ev = at < finalLen ? "CONTINUE" : "TURN";
        for (const [, key] of hazardContexts(side, at, prev)) bucket(key)[ev] += 1;
      }
    });
    return table;
  }

  function hazardPosterior(c) {
    const co = +c.CONTINUE || 0, tu = +c.TURN || 0, den = co + tu + 6;
    return den > 1e-12
      ? { CONTINUE: (co + 3) / den, TURN: (tu + 3) / den }
      : { CONTINUE: 0.5, TURN: 0.5 };
  }

  function hazard(seq) {
    const rs = runs(seq);
    if (!rs.length) return 0.5;
    const [side, cur] = rs[rs.length - 1];
    const heights = rs.slice(0, -1).map(x => x[1]);
    const table = hazardTable(rs);
    const contexts = hazardContexts(side, cur, heights);
    let tier = "prior";
    let prob = { CONTINUE: 0.5, TURN: 0.5 };
    let penalty = 1;

    for (let i = 0; i < contexts.length; i++) {
      const [name, key] = contexts[i];
      const c = table[key] || { CONTINUE: 0, TURN: 0 };
      const support = c.CONTINUE + c.TURN;
      const p = hazardPosterior(c);
      if (support >= 4) { tier = name; prob = p; break; }
      if (i < contexts.length - 1) penalty *= 0.75;
    }

    if (tier === "prior") {
      const g = table["HZG|GLOBAL"] || { CONTINUE: 0, TURN: 0 };
      if (g.CONTINUE + g.TURN > 0) prob = hazardPosterior(g);
      else penalty = 0;
    }
    const cont = (1 - penalty) * 0.5 + penalty * prob.CONTINUE;
    return clip(1 - cont);
  }

  function entropyNorm(seq, w = 12) {
    const a = seq.slice(-w);
    if (!a.length) return 1;
    let e = 0;
    for (const o of ["B","P","T"]) {
      const p = a.filter(x => x === o).length / a.length;
      if (p > 0) e -= p * Math.log2(p);
    }
    return clip(e / Math.log2(3));
  }

  function runVolatility(seq) {
    const h = runs(seq).slice(-6).map(x => x[1]);
    if (h.length < 2) return 0.25;
    let d = 0;
    for (let i = 1; i < h.length; i++) d += Math.abs(h[i] - h[i - 1]);
    return clip((d / (h.length - 1)) / 3);
  }

  function hsmmStable(seq) {
    const a = turnRate(seq, 10);
    const rs = runs(seq);
    const cur = rs.length ? rs[rs.length - 1][1] : 0;
    const r = clip(cur / 6);
    const e = entropyNorm(seq, 12);
    const v = runVolatility(seq);

    const p = Math.exp(-(((a-.25)/.24)**2) - (((r-.70)/.28)**2) - (((e-.62)/.24)**2) - (((v-.26)/.24)**2));
    const q = Math.exp(-(((a-.84)/.18)**2) - (((r-.18)/.20)**2) - (((e-.70)/.23)**2) - (((v-.30)/.24)**2));
    const t = Math.exp(-(((a-.52)/.28)**2) - (((r-.34)/.26)**2) - (((e-.82)/.18)**2) - (((v-.72)/.23)**2));
    const n = Math.exp(-(((a-.55)/.30)**2) - (((r-.27)/.24)**2) - (((e-.94)/.11)**2) - (((v-.55)/.28)**2));
    const weights = [.25*p, .25*q, .20*t, .30*n];
    const total = weights.reduce((x, y) => x + y, 0) || 1;
    return clip((weights[0] + weights[1]) / total);
  }

  function context32(seq) {
    const used = Math.min(TOTAL_CARDS, seq.length * AVG_CARDS_PER_HAND);
    const remaining = Math.max(0, TOTAL_CARDS - used);
    const rr = clip(remaining / TOTAL_CARDS);
    const penetration = clip(1 - rr);
    const maturity = clip(seq.length / 70);

    // Button-only panel has no exact rank/card input.
    // Therefore point-rank ratios stay neutral, exactly as the no-exact-composition fallback.
    const shoe = [
      rr, penetration, rr, maturity,
      1,1,1,1,1,1,1,1,1,1,
      0,0
    ];

    const rs = runs(seq);
    const current = rs.length ? rs[rs.length - 1] : ["", 0];
    const prev = rs.length >= 2 ? rs[rs.length - 2] : ["", 0];
    const prev2 = rs.length >= 3 ? rs[rs.length - 3] : ["", 0];
    const sideB = current[0] === "B" ? 1 : current[0] === "P" ? 0 : 0.5;

    const dr = buildDerivedRoads(seq);
    const [be] = regularity(dr.big_eye);
    const [sm] = regularity(dr.small_road);
    const [cr] = regularity(dr.cockroach_road);
    const mean = (be + sm + cr) / 3;
    const consensus = clip(1 - (Math.abs(be - mean) + Math.abs(sm - mean) + Math.abs(cr - mean)) / 1.5);

    const road = [
      sideB,
      clip(current[1] / 8),
      clip(prev[1] / 8),
      clip(prev2[1] / 8),
      bankerRatio(seq, 5),
      bankerRatio(seq, 8),
      bankerRatio(seq, 12),
      turnRate(seq, 5),
      turnRate(seq, 8),
      turnRate(seq, 12),
      hazard(seq),
      hsmmStable(seq),
      be, sm, cr, consensus
    ];

    return [...shoe, ...road];
  }

  function solve(A, b) {
    const n = A.length;
    const m = A.map((row, i) => [...row, +b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) {
        if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
      }
      if (Math.abs(m[p][c]) < 1e-10) {
        const bumped = A.map((row, i) => row.map((v, j) => v + (i === j ? RIDGE : 0)));
        return solve(bumped, b);
      }
      [m[c], m[p]] = [m[p], m[c]];
      const d = m[c][c];
      for (let j = c; j <= n; j++) m[c][j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = m[r][c];
        for (let j = c; j <= n; j++) m[r][j] -= f * m[c][j];
      }
    }
    return m.map(row => row[n]);
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function scoreArm(arm, x, alphaScale) {
    const theta = solve(arm.A, arm.b);
    const solvedX = solve(arm.A, x);
    const mean = dot(x, theta);
    const uncertainty = Math.sqrt(Math.max(0, dot(x, solvedX)));
    const effectiveAlpha = ALPHA * Math.max(0.5, Math.min(2.5, alphaScale));
    return {
      mean,
      uncertainty,
      effectiveAlpha,
      score: mean + effectiveAlpha * uncertainty
    };
  }

  function choose(brain, seq) {
    const x = context32(seq);
    const nBP = bp(seq).length;
    const baseScale = nBP < 8 ? 1.35 : nBP < 15 ? 1.15 : 1.0;
    const eff = {
      P: Math.max(0, +brain.arms.P.effective_n || 0),
      B: Math.max(0, +brain.arms.B.effective_n || 0)
    };
    const total = eff.P + eff.B;
    const scores = {};

    for (const arm of ARMS) {
      const imbalance = Math.sqrt(Math.max(1, total + 2) / Math.max(1, eff[arm] + 1));
      const scale = baseScale * clip(imbalance, 0.85, ALPHA_MAX_SCALE);
      scores[arm] = scoreArm(brain.arms[arm], x, scale);
    }

    const gap = scores.B.score - scores.P.score;
    let direction, reason;

    if (Math.abs(gap) <= SCORE_TIE_EPS) {
      if (Math.abs(eff.B - eff.P) > 1e-9) {
        direction = eff.B < eff.P ? "B" : "P";
        reason = "tie_less_sampled_arm";
      } else if (["B","P"].includes(brain.last_selected)) {
        direction = brain.last_selected === "B" ? "P" : "B";
        reason = "tie_opposite_previous_arm";
      } else {
        let h = 0;
        const token = "LOCAL_32D|" + seq.join("");
        for (let i = 0; i < token.length; i++) h = (h * 31 + token.charCodeAt(i)) >>> 0;
        direction = h % 2 ? "B" : "P";
        reason = "tie_deterministic_history_hash";
      }
    } else {
      direction = gap > 0 ? "B" : "P";
      reason = "linucb_ucb_score_argmax";
    }

    const rawPB = 1 / (1 + Math.exp(-Math.max(-8, Math.min(8, gap / SCORE_TEMP))));
    const pB = clip(rawPB, PROB_MIN, PROB_MAX);
    const pP = 1 - pB;
    const confidence = direction === "B" ? pB : pP;

    const prev = brain.last_selected;
    brain.selection_streak = prev === direction ? (brain.selection_streak || 0) + 1 : 1;
    brain.last_selected = direction;

    return { direction, reason, x, scores, gap, probabilities: { B: pB, P: pP }, confidence };
  }

  function decayBrain(brain) {
    for (const armName of ARMS) {
      const arm = brain.arms[armName];
      for (let i = 0; i < DIM; i++) {
        for (let j = 0; j < DIM; j++) {
          const identity = i === j ? RIDGE : 0;
          arm.A[i][j] = identity + FORGETTING * (arm.A[i][j] - identity);
        }
        arm.b[i] *= FORGETTING;
      }
      arm.effective_n *= FORGETTING;
    }
  }

  function trainingUpdate(brain, action, x, actual) {
    decayBrain(brain);
    brain.updates++;

    if (actual === "T") return;

    const reward = action === actual ? (action === "B" ? 0.95 : 1.0) : -1.0;
    const arm = brain.arms[action];

    for (let i = 0; i < DIM; i++) {
      arm.b[i] += UPDATE_WEIGHT * reward * x[i];
      for (let j = 0; j < DIM; j++) {
        arm.A[i][j] += UPDATE_WEIGHT * x[i] * x[j];
      }
    }
    arm.n++;
    arm.effective_n++;
  }

  function bootstrap(history) {
    const brain = freshBrain();
    for (let target = 1; target < history.length; target++) {
      const prefix = history.slice(0, target);
      const pred = choose(brain, prefix);
      trainingUpdate(brain, pred.direction, pred.x, history[target]);
    }
    return brain;
  }

  function addOutcome(outcome) {
    if (!["B","P","T"].includes(outcome)) return;
    state.history.push(outcome);
    // 跟原測試面板一樣：輸入歷史只改 history，不自動預測、不更新 A/b。
    state.lastPrediction = null;
    setMessage(`已加入第 ${state.history.length} 局：${outcome === "B" ? "莊" : outcome === "P" ? "閒" : "和"}。按「開始分析」＝沿用32D本地腦直接預測。`);
    save();
    render();
  }

  function startAnalysis() {
    if (!state.history.length) {
      setMessage("請先用莊／閒／和輸入歷史紀錄。", true);
      return;
    }
    if (!state.brain || !state.brain.arms) state.brain = freshBrain();

    // 完全對應原測試面板「沿用32D本地腦直接預測」：
    // load local brain -> 用目前完整 history 算32D Context -> predict -> save local brain
    // 不 bootstrap、不回放、不結算上一筆、不更新 A/b、不 decay。
    state.active = true;
    state.lastPrediction = choose(state.brain, state.history);
    setMessage(`已沿用32D本地腦直接預測第 ${state.history.length + 1} 局。A/b 未更新。`);
    save();
    render();
  }

  function endAnalysis() {
    if (!state.active && !state.lastPrediction) {
      setMessage("目前沒有進行中的分析結果。", true);
      return;
    }
    state.active = false;
    state.lastPrediction = null;
    // 本地 brain 保留，等同測試面板 localStorage 腦，不清空 A/b。
    setMessage("分析已結束。歷史與32D本地腦都保留，可繼續追加紀錄後再次開始分析。");
    save();
    render();
  }

  function backOne() {
    if (!state.history.length) {
      setMessage("目前沒有可以返回的牌局。", true);
      return;
    }
    const removed = state.history.pop();
    state.active = false;
    state.lastPrediction = null;
    setMessage(`已返回上一局（移除 ${removed}）。本地32D腦沒有更新；要預測請再按「開始分析」。`);
    save();
    render();
  }

  function setMessage(text, warn = false) {
    const box = el("message");
    box.textContent = text;
    box.style.borderLeftColor = warn ? "#f7d46a" : "#55d6ff";
    box.style.color = warn ? "#dfc986" : "#9ab4c7";
  }

  function renderHistory() {
    const box = el("historyTrack");
    if (!state.history.length) {
      box.innerHTML = '<div class="empty-history">尚未輸入牌局</div>';
      return;
    }
    box.innerHTML = state.history.map((v, i) =>
      `<div class="history-chip ${v}" title="第${i+1}局">${v}</div>`
    ).join("");
  }

  function renderFeatures(vector) {
    const v = Array.isArray(vector) && vector.length === 32 ? vector : Array(32).fill(0);
    el("shoeGrid").innerHTML = SHOE_NAMES.map((name, i) =>
      `<div class="feature-row"><span>${String(i + 1).padStart(2, "0")} · ${name}</span><b>${(+v[i]).toFixed(5)}</b></div>`
    ).join("");
    el("roadGrid").innerHTML = ROAD_NAMES.map((name, i) => {
      const idx = i + 16;
      return `<div class="feature-row"><span>${String(idx + 1).padStart(2, "0")} · ${name}</span><b>${(+v[idx]).toFixed(5)}</b></div>`;
    }).join("");
  }

  function renderPrediction() {
    const p = state.lastPrediction;
    const orb = el("directionOrb");

    if (!p || !state.active) {
      el("directionText").textContent = "—";
      el("directionCode").textContent = "WAIT";
      el("confidence").textContent = "—";
      el("ucbB").textContent = "—";
      el("ucbP").textContent = "—";
      el("scoreGap").textContent = "—";
      orb.className = "direction-orb idle";
      renderFeatures(context32(state.history));
      return;
    }

    el("directionText").textContent = p.direction === "B" ? "莊" : "閒";
    el("directionCode").textContent = p.direction === "B" ? "BANKER · B" : "PLAYER · P";
    el("confidence").textContent = (p.confidence * 100).toFixed(2) + "%";
    el("ucbB").textContent = p.scores.B.score.toFixed(6);
    el("ucbP").textContent = p.scores.P.score.toFixed(6);
    el("scoreGap").textContent = p.gap.toFixed(6);
    orb.className = "direction-orb " + (p.direction === "B" ? "banker" : "player");
    renderFeatures(p.x);
  }

  function renderDebug() {
    const brain = state.brain;
    const debug = {
      model: "32D Frozen Direct Local Brain",
      version: "FROZEN-DIRECT-V9-STATIC-PANEL",
      active: state.active,
      totalHistory: state.history.length,
      history: state.history.join(""),
      brain: brain ? {
        stored_updates: brain.updates,
        effective_n: {
          B: brain.arms.B.effective_n,
          P: brain.arms.P.effective_n
        },
        raw_n: {
          B: brain.arms.B.n,
          P: brain.arms.P.n
        },
        last_selected: brain.last_selected,
        selection_streak: brain.selection_streak,
        direct_predict_only: true,
        no_bootstrap_on_start: true,
        no_feedback_update: true
      } : null,
      prediction: state.lastPrediction ? {
        direction: state.lastPrediction.direction,
        confidence: state.lastPrediction.confidence,
        gap: state.lastPrediction.gap,
        reason: state.lastPrediction.reason,
        context32: state.lastPrediction.x
      } : null
    };
    el("debug").textContent = JSON.stringify(debug, null, 2);
  }

  function render() {
    el("modePill").textContent = state.active ? "已完成本次預測" : "準備歷史";
    el("roundPill").textContent = `${state.history.length} 局`;
    el("brainState").textContent = "沿用32D本地腦";
    el("seedCount").textContent = state.history.length;
    el("liveCount").textContent = state.brain ? Math.round((state.brain.updates || 0)) : 0;

    el("btnStart").disabled = false;
    el("btnEnd").disabled = !state.active && !state.lastPrediction;

    renderHistory();
    renderPrediction();
    renderDebug();
  }

  function initCanvas() {
    const canvas = el("techCanvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let w = 0, h = 0, nodes = [];

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth; h = window.innerHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.max(18, Math.min(44, Math.floor(w / 32)));
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * w, y: Math.random() * h,
        vx: (Math.random() - .5) * .18, vy: (Math.random() - .5) * .18,
        r: 1 + Math.random() * 1.5
      }));
    }

    function frame() {
      ctx.clearRect(0, 0, w, h);
      for (const n of nodes) {
        n.x += n.vx; n.y += n.vy;
        if (n.x < 0 || n.x > w) n.vx *= -1;
        if (n.y < 0 || n.y > h) n.vy *= -1;
      }
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy);
          if (d < 135) {
            ctx.strokeStyle = `rgba(74,194,255,${(1 - d / 135) * .10})`;
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
      }
      for (const n of nodes) {
        ctx.fillStyle = "rgba(98,220,255,.32)";
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fill();
      }
      requestAnimationFrame(frame);
    }

    resize();
    window.addEventListener("resize", resize, { passive: true });
    frame();
  }

  el("btnB").addEventListener("click", () => addOutcome("B"));
  el("btnP").addEventListener("click", () => addOutcome("P"));
  el("btnT").addEventListener("click", () => addOutcome("T"));
  el("btnStart").addEventListener("click", startAnalysis);
  el("btnEnd").addEventListener("click", endAnalysis);
  el("btnBack").addEventListener("click", backOne);

  load();

  // 沒有既有 local brain 時建立空白32D腦；不做 Walk-forward。
  if (!state.brain || !state.brain.arms) state.brain = freshBrain();
  if (state.active && !state.lastPrediction) state.active = false;

  render();
  initCanvas();
})();
