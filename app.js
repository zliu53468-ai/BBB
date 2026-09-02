(() => {
  "use strict";

  const SHOE_DIM = 32;
  const ROAD_DIM = 32;
  const DIM = SHOE_DIM + ROAD_DIM;
  const PRIMARY_WEIGHTS = Object.freeze({ shoe: 0.60, road: 0.40, key: "60/40" });
  const BASELINE_WEIGHTS = Object.freeze({ shoe: 0.50, road: 0.50, key: "50/50" });
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
  const STORAGE_KEY = "bgs64d_shoe60_road40_ab_online_panel_v4";
  const PREVIOUS_WEIGHTED_STORAGE_KEY = "bgs64d_weighted_ab_online_panel_v3";
  const PREVIOUS_64D_STORAGE_KEY = "bgs64d_32plus32_frozen_direct_tech_panel_v2";
  const LEGACY_32D_STORAGE_KEY = "bgs32d_frozen_direct_tech_panel_v1";

  const SHOE_NAMES = [
    "remaining_cards_ratio","penetration_ratio","estimated_hands_remaining_norm","shoe_maturity_ratio",
    "rank_A_relative_deviation","rank_2_relative_deviation","rank_3_relative_deviation","rank_4_relative_deviation",
    "rank_5_relative_deviation","rank_6_relative_deviation","rank_7_relative_deviation","rank_8_relative_deviation",
    "rank_9_relative_deviation","rank_10JQK_relative_deviation","physical_edge_proxy","shoe_information_reliability",
    "shoe_phase_early","shoe_phase_middle","shoe_phase_late","estimated_hands_played_norm",
    "remaining_decks_ratio","hands_elapsed_log_norm","tie_ratio_all","tie_ratio_recent8",
    "tie_ratio_recent16","bp_balance_strength","bp_entropy_recent12","outcome_entropy_recent12",
    "outcome_entropy_recent24","sample_support_norm","composition_missing_indicator","shoe_progression_confidence"
  ];
  const ROAD_NAMES = [
    "current_side_banker_binary","current_run_length_norm","previous_run_length_norm","previous2_run_length_norm",
    "recent5_banker_ratio","recent8_banker_ratio","recent12_banker_ratio","recent5_turn_rate",
    "recent8_turn_rate","recent12_turn_rate","run_length_hazard_rate","hsmm_stable_probability",
    "big_eye_regularity","small_road_regularity","cockroach_road_regularity","derived_road_consensus",
    "current_side_player_binary","previous3_run_length_norm","recent3_banker_ratio","recent20_banker_ratio",
    "recent3_turn_rate","recent20_turn_rate","run_continue_probability","recent8_outcome_entropy",
    "recent20_outcome_entropy","recent6_run_volatility","recent5_run_height_trend","big_eye_support_norm",
    "small_road_support_norm","cockroach_road_support_norm","last2_same_side","last3_same_side"
  ];

  const el = id => document.getElementById(id);
  const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));

  let state = {
    history: [],
    active: false,
    brain: null,
    baselineBrain: null,
    lastPrediction: null,
    lastBaselinePrediction: null,
    stats: null,
    settlements: []
  };
  let migratedFromPreviousVersion = false;

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

  function freshStats() {
    return {
      settled: 0,
      correct: 0,
      pushes: 0,
      recent: [],
      current_losing_streak: 0,
      max_losing_streak: 0,
      brier_sum: 0
    };
  }

  function freshExperimentStats() {
    return { weighted: freshStats(), baseline: freshStats() };
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

  function validArm(arm) {
    return !!arm &&
      Array.isArray(arm.A) && arm.A.length === DIM &&
      arm.A.every(row => Array.isArray(row) && row.length === DIM) &&
      Array.isArray(arm.b) && arm.b.length === DIM;
  }

  function validBrain(brain) {
    return !!brain && !!brain.arms && validArm(brain.arms.P) && validArm(brain.arms.B);
  }

  function validPrediction(prediction) {
    return !!prediction && ["B","P"].includes(prediction.direction) &&
      Array.isArray(prediction.x) && prediction.x.length === DIM &&
      prediction.probabilities && Number.isFinite(+prediction.probabilities.B);
  }

  function normalizeStats(raw) {
    const base = freshStats();
    if (!raw || typeof raw !== "object") return base;
    base.settled = Math.max(0, Math.floor(+raw.settled || 0));
    base.correct = Math.max(0, Math.min(base.settled, Math.floor(+raw.correct || 0)));
    base.pushes = Math.max(0, Math.floor(+raw.pushes || 0));
    base.recent = Array.isArray(raw.recent) ? raw.recent.filter(x => x === 0 || x === 1).slice(-20) : [];
    base.current_losing_streak = Math.max(0, Math.floor(+raw.current_losing_streak || 0));
    base.max_losing_streak = Math.max(base.current_losing_streak, Math.floor(+raw.max_losing_streak || 0));
    base.brier_sum = Math.max(0, +raw.brier_sum || 0);
    return base;
  }

  function load() {
    try {
      const current = localStorage.getItem(STORAGE_KEY);
      const previousWeighted = current ? null : localStorage.getItem(PREVIOUS_WEIGHTED_STORAGE_KEY);
      const previous64 = current || previousWeighted ? null : localStorage.getItem(PREVIOUS_64D_STORAGE_KEY);
      const previous32 = current || previousWeighted || previous64 ? null : localStorage.getItem(LEGACY_32D_STORAGE_KEY);
      const previous = previousWeighted || previous64 || previous32;
      const raw = JSON.parse(current || previous || "null");
      if (raw && Array.isArray(raw.history)) {
        const compatibleState = !!current && validBrain(raw.brain) && validBrain(raw.baselineBrain);
        const compatiblePredictions = compatibleState && validPrediction(raw.lastPrediction) &&
          validPrediction(raw.lastBaselinePrediction) &&
          raw.lastPrediction.historyLength === raw.history.length &&
          raw.lastBaselinePrediction.historyLength === raw.history.length;
        migratedFromPreviousVersion = !!previous;
        state = {
          history: raw.history.filter(x => ["B","P","T"].includes(x)).slice(-500),
          active: compatiblePredictions ? !!raw.active : false,
          brain: compatibleState ? raw.brain : freshBrain(),
          baselineBrain: compatibleState ? raw.baselineBrain : freshBrain(),
          lastPrediction: compatiblePredictions ? raw.lastPrediction : null,
          lastBaselinePrediction: compatiblePredictions ? raw.lastBaselinePrediction : null,
          stats: compatibleState ? {
            weighted: normalizeStats(raw.stats && raw.stats.weighted),
            baseline: normalizeStats(raw.stats && raw.stats.baseline)
          } : freshExperimentStats(),
          settlements: compatibleState && Array.isArray(raw.settlements) ? raw.settlements : []
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

  function binaryEntropyNorm(seq, w = 12) {
    const a = bp(seq).slice(-w);
    if (!a.length) return 1;
    const pB = a.filter(x => x === "B").length / a.length;
    const pP = 1 - pB;
    let e = 0;
    if (pB > 0) e -= pB * Math.log2(pB);
    if (pP > 0) e -= pP * Math.log2(pP);
    return clip(e);
  }

  function tieRatio(seq, w = 0) {
    const a = w > 0 ? seq.slice(-w) : seq;
    return a.length ? a.filter(x => x === "T").length / a.length : 0;
  }

  function runVolatility(seq) {
    const h = runs(seq).slice(-6).map(x => x[1]);
    if (h.length < 2) return 0.25;
    let d = 0;
    for (let i = 1; i < h.length; i++) d += Math.abs(h[i] - h[i - 1]);
    return clip((d / (h.length - 1)) / 3);
  }

  function runHeightTrend(seq, w = 5) {
    const h = runs(seq).slice(-w).map(x => x[1]);
    if (h.length < 2) return 0.5;
    const slope = (h[h.length - 1] - h[0]) / (h.length - 1);
    return clip(0.5 + slope / 6);
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

  function normalizeBlock(values) {
    const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
    return norm > 1e-12 ? values.map(value => value / norm) : values.map(() => 0);
  }

  function context64(seq, weights = PRIMARY_WEIGHTS) {
    const used = Math.min(TOTAL_CARDS, seq.length * AVG_CARDS_PER_HAND);
    const remaining = Math.max(0, TOTAL_CARDS - used);
    const rr = clip(remaining / TOTAL_CARDS);
    const penetration = clip(1 - rr);
    const maturity = clip(seq.length / 70);
    const estimatedHandCapacity = TOTAL_CARDS / AVG_CARDS_PER_HAND;
    const handsPlayedNorm = clip(seq.length / estimatedHandCapacity);
    const phaseEarly = clip(1 - penetration / 0.35);
    const phaseMiddle = clip(1 - Math.abs(penetration - 0.50) / 0.35);
    const phaseLate = clip((penetration - 0.55) / 0.35);
    const allBankerRatio = bankerRatio(seq, Math.max(1, bp(seq).length));

    // Button-only panel has no exact rank/card input.
    // Relative ratios are centered as (ratio - 1), so missing/neutral composition is 0
    // and cannot consume the shoe block's normalized energy as a fake constant signal.
    const shoe = [
      rr, penetration, rr, maturity,
      0,0,0,0,0,0,0,0,0,0,
      0,0,
      phaseEarly,
      phaseMiddle,
      phaseLate,
      handsPlayedNorm,
      rr,
      clip(Math.log1p(seq.length) / Math.log1p(estimatedHandCapacity)),
      tieRatio(seq),
      tieRatio(seq, 8),
      tieRatio(seq, 16),
      clip(1 - Math.abs(allBankerRatio - 0.5) * 2),
      binaryEntropyNorm(seq, 12),
      entropyNorm(seq, 12),
      entropyNorm(seq, 24),
      clip(seq.length / 32),
      1,
      clip(Math.sqrt(seq.length) / Math.sqrt(estimatedHandCapacity))
    ];

    const rs = runs(seq);
    const current = rs.length ? rs[rs.length - 1] : ["", 0];
    const prev = rs.length >= 2 ? rs[rs.length - 2] : ["", 0];
    const prev2 = rs.length >= 3 ? rs[rs.length - 3] : ["", 0];
    const prev3 = rs.length >= 4 ? rs[rs.length - 4] : ["", 0];
    const sideB = current[0] === "B" ? 1 : current[0] === "P" ? 0 : 0.5;
    const sideP = current[0] === "P" ? 1 : current[0] === "B" ? 0 : 0.5;

    const dr = buildDerivedRoads(seq);
    const [be, beSupport] = regularity(dr.big_eye);
    const [sm, smSupport] = regularity(dr.small_road);
    const [cr, crSupport] = regularity(dr.cockroach_road);
    const mean = (be + sm + cr) / 3;
    const consensus = clip(1 - (Math.abs(be - mean) + Math.abs(sm - mean) + Math.abs(cr - mean)) / 1.5);
    const turnHazard = hazard(seq);
    const roadBP = bp(seq);
    const last2Same = roadBP.length >= 2 ? +(roadBP.at(-1) === roadBP.at(-2)) : 0.5;
    const last3Same = roadBP.length >= 3 ? +(roadBP.at(-1) === roadBP.at(-2) && roadBP.at(-2) === roadBP.at(-3)) : 0.5;

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
      turnHazard,
      hsmmStable(seq),
      be, sm, cr, consensus,
      sideP,
      clip(prev3[1] / 8),
      bankerRatio(seq, 3),
      bankerRatio(seq, 20),
      turnRate(seq, 3),
      turnRate(seq, 20),
      clip(1 - turnHazard),
      entropyNorm(seq, 8),
      entropyNorm(seq, 20),
      runVolatility(seq),
      runHeightTrend(seq, 5),
      clip(beSupport / 8),
      clip(smSupport / 8),
      clip(crSupport / 8),
      last2Same,
      last3Same
    ];

    if (shoe.length !== SHOE_DIM || road.length !== ROAD_DIM) {
      throw new Error(`64D context shape mismatch: shoe=${shoe.length}, road=${road.length}`);
    }

    // LinUCB 的能量項是 xᵀA⁻¹x，因此以 sqrt(weight) 縮放，
    // 讓兩個已各自 L2 正規化的區塊真正呈現指定的 60/40 或 50/50 佔比。
    const shoeScale = Math.sqrt(weights.shoe);
    const roadScale = Math.sqrt(weights.road);
    const vector = [
      ...normalizeBlock(shoe).map(value => value * shoeScale),
      ...normalizeBlock(road).map(value => value * roadScale)
    ];
    if (vector.length !== DIM) throw new Error(`64D context total mismatch: total=${vector.length}`);
    return vector;
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

  function choose(brain, seq, weights = PRIMARY_WEIGHTS) {
    const x = context64(seq, weights);
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
        const token = "LOCAL_64D_32PLUS32|" + seq.join("");
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

    return {
      direction,
      reason,
      x,
      scores,
      gap,
      probabilities: { B: pB, P: pP },
      confidence,
      weights: { shoe: weights.shoe, road: weights.road, key: weights.key },
      historyLength: seq.length,
      targetRound: seq.length + 1
    };
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
    if (actual === "T") return;

    decayBrain(brain);
    brain.updates++;

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

  function recordStats(stats, prediction, actual) {
    if (actual === "T") {
      stats.pushes++;
      return;
    }

    const hit = prediction.direction === actual ? 1 : 0;
    stats.settled++;
    stats.correct += hit;
    stats.recent.push(hit);
    stats.recent = stats.recent.slice(-20);
    stats.current_losing_streak = hit ? 0 : stats.current_losing_streak + 1;
    stats.max_losing_streak = Math.max(stats.max_losing_streak, stats.current_losing_streak);

    const actualB = actual === "B" ? 1 : 0;
    const pB = clip(+prediction.probabilities.B);
    stats.brier_sum += (pB - actualB) ** 2;
  }

  function predictionSnapshot(prediction) {
    return {
      direction: prediction.direction,
      x: prediction.x,
      pB: prediction.probabilities.B
    };
  }

  function applySettlement(entry) {
    const weightedPrediction = {
      direction: entry.weighted.direction,
      x: entry.weighted.x,
      probabilities: { B: entry.weighted.pB, P: 1 - entry.weighted.pB }
    };
    const baselinePrediction = {
      direction: entry.baseline.direction,
      x: entry.baseline.x,
      probabilities: { B: entry.baseline.pB, P: 1 - entry.baseline.pB }
    };

    recordStats(state.stats.weighted, weightedPrediction, entry.actual);
    recordStats(state.stats.baseline, baselinePrediction, entry.actual);
    trainingUpdate(state.brain, weightedPrediction.direction, weightedPrediction.x, entry.actual);
    trainingUpdate(state.baselineBrain, baselinePrediction.direction, baselinePrediction.x, entry.actual);
  }

  function settlePending(actual) {
    if (!validPrediction(state.lastPrediction) || !validPrediction(state.lastBaselinePrediction)) return null;
    if (state.lastPrediction.historyLength !== state.history.length ||
        state.lastBaselinePrediction.historyLength !== state.history.length) return null;

    const entry = {
      round: state.history.length + 1,
      actual,
      weighted: predictionSnapshot(state.lastPrediction),
      baseline: predictionSnapshot(state.lastBaselinePrediction)
    };
    state.settlements.push(entry);
    applySettlement(entry);
    return entry;
  }

  function rebuildLearningState() {
    const retained = state.settlements.filter(entry =>
      entry && entry.round <= state.history.length && ["B","P","T"].includes(entry.actual) &&
      entry.weighted && entry.baseline &&
      Array.isArray(entry.weighted.x) && entry.weighted.x.length === DIM &&
      Array.isArray(entry.baseline.x) && entry.baseline.x.length === DIM
    );
    state.brain = freshBrain();
    state.baselineBrain = freshBrain();
    state.stats = freshExperimentStats();
    state.settlements = retained;
    for (const entry of retained) applySettlement(entry);
  }

  function addOutcome(outcome) {
    if (!["B","P","T"].includes(outcome)) return;
    const settlement = settlePending(outcome);
    state.history.push(outcome);
    state.active = false;
    state.lastPrediction = null;
    state.lastBaselinePrediction = null;

    const label = outcome === "B" ? "莊" : outcome === "P" ? "閒" : "和";
    if (!settlement) {
      setMessage(`已加入第 ${state.history.length} 局：${label}。尚無待結算預測，按「開始分析」預測下一局。`);
    } else if (outcome === "T") {
      setMessage(`第 ${state.history.length} 局開和：兩組測試皆記為 Push，不計勝負且不更新 A/b。`);
    } else {
      const weightedHit = settlement.weighted.direction === outcome ? "命中" : "未中";
      const baselineHit = settlement.baseline.direction === outcome ? "命中" : "未中";
      setMessage(`第 ${state.history.length} 局已延遲結算：60/40 ${weightedHit}；50/50 ${baselineHit}。兩組 A/b 已各自更新。`);
    }
    save();
    render();
  }

  function startAnalysis() {
    if (!state.history.length) {
      setMessage("請先用莊／閒／和輸入歷史紀錄。", true);
      return;
    }
    if (!validBrain(state.brain)) state.brain = freshBrain();
    if (!validBrain(state.baselineBrain)) state.baselineBrain = freshBrain();
    if (!state.stats) state.stats = freshExperimentStats();

    if (validPrediction(state.lastPrediction) && validPrediction(state.lastBaselinePrediction) &&
        state.lastPrediction.historyLength === state.history.length &&
        state.lastBaselinePrediction.historyLength === state.history.length) {
      state.active = true;
      setMessage(`第 ${state.history.length + 1} 局已存在待結算預測，請輸入實際莊／閒／和後再繼續。`, true);
      render();
      return;
    }

    state.active = true;
    state.lastPrediction = choose(state.brain, state.history, PRIMARY_WEIGHTS);
    state.lastBaselinePrediction = choose(state.baselineBrain, state.history, BASELINE_WEIGHTS);
    const weightedDirection = state.lastPrediction.direction === "B" ? "莊" : "閒";
    const baselineDirection = state.lastBaselinePrediction.direction === "B" ? "莊" : "閒";
    setMessage(`第 ${state.history.length + 1} 局：60/40 主模型預測${weightedDirection}；50/50對照組預測${baselineDirection}。輸入開獎結果後才延遲更新。`);
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
    state.lastBaselinePrediction = null;
    setMessage("分析已結束。歷史、兩組64D本地腦與A/B統計都會保留。");
    save();
    render();
  }

  function backOne() {
    if (!state.history.length) {
      setMessage("目前沒有可以返回的牌局。", true);
      return;
    }
    const removed = state.history.pop();
    const before = state.settlements.length;
    state.settlements = state.settlements.filter(entry => entry.round <= state.history.length);
    if (state.settlements.length !== before) rebuildLearningState();
    state.active = false;
    state.lastPrediction = null;
    state.lastBaselinePrediction = null;
    const rollback = state.settlements.length !== before ? "，對應學習與統計已回滾" : "";
    setMessage(`已返回上一局（移除 ${removed}）${rollback}；要預測請再按「開始分析」。`);
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
    const v = Array.isArray(vector) && vector.length === DIM ? vector : Array(DIM).fill(0);
    el("shoeGrid").innerHTML = SHOE_NAMES.map((name, i) =>
      `<div class="feature-row"><span>${String(i + 1).padStart(2, "0")} · ${name}</span><b>${(+v[i]).toFixed(5)}</b></div>`
    ).join("");
    el("roadGrid").innerHTML = ROAD_NAMES.map((name, i) => {
      const idx = i + SHOE_DIM;
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
      renderFeatures(context64(state.history, PRIMARY_WEIGHTS));
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

  function accuracyText(stats, recent = false) {
    const values = recent ? stats.recent : null;
    const total = recent ? values.length : stats.settled;
    const correct = recent ? values.reduce((sum, value) => sum + value, 0) : stats.correct;
    return total ? `${(correct / total * 100).toFixed(2)}%` : "—";
  }

  function brierText(stats) {
    return stats.settled ? (stats.brier_sum / stats.settled).toFixed(4) : "—";
  }

  function directionText(prediction) {
    if (!state.active || !validPrediction(prediction)) return "—";
    return prediction.direction === "B" ? "莊 · B" : "閒 · P";
  }

  function renderExperimentStats() {
    const weighted = state.stats.weighted;
    const baseline = state.stats.baseline;

    el("weightedNext").textContent = directionText(state.lastPrediction);
    el("weightedSettled").textContent = weighted.settled;
    el("weightedAccuracy").textContent = accuracyText(weighted);
    el("weightedRecent").textContent = accuracyText(weighted, true);
    el("weightedMaxLoss").textContent = weighted.max_losing_streak;
    el("weightedBrier").textContent = brierText(weighted);

    el("baselineNext").textContent = directionText(state.lastBaselinePrediction);
    el("baselineSettled").textContent = baseline.settled;
    el("baselineAccuracy").textContent = accuracyText(baseline);
    el("baselineRecent").textContent = accuracyText(baseline, true);
    el("baselineMaxLoss").textContent = baseline.max_losing_streak;
    el("baselineBrier").textContent = brierText(baseline);

    const summary = el("abSummary");
    if (!weighted.settled) {
      summary.textContent = "至少完成一筆『預測 → 輸入下一局結果』後，才會開始比較。";
      return;
    }
    const weightedRate = weighted.correct / weighted.settled;
    const baselineRate = baseline.correct / Math.max(1, baseline.settled);
    const rateGap = (weightedRate - baselineRate) * 100;
    const weightedBrier = weighted.brier_sum / weighted.settled;
    const baselineBrier = baseline.brier_sum / Math.max(1, baseline.settled);
    const leader = rateGap > 0 ? "60/40主模型領先" : rateGap < 0 ? "50/50對照組領先" : "兩組命中率相同";
    summary.textContent = `${leader} ${Math.abs(rateGap).toFixed(2)} 個百分點；Brier差 ${(weightedBrier - baselineBrier).toFixed(4)}（負值代表60/40較佳）。`;
  }

  function renderDebug() {
    const brain = state.brain;
    const baselineBrain = state.baselineBrain;
    const debug = {
      model: "64D weighted online LinUCB A/B lab",
      version: "WEIGHTED-ONLINE-V12-64D-SHOE60-AB",
      active: state.active,
      totalHistory: state.history.length,
      history: state.history.join(""),
      primary_40_60: brain ? {
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
        weights: PRIMARY_WEIGHTS,
        no_bootstrap_on_start: true,
        delayed_feedback_update: true
      } : null,
      baseline_50_50: baselineBrain ? {
        stored_updates: baselineBrain.updates,
        effective_n: {
          B: baselineBrain.arms.B.effective_n,
          P: baselineBrain.arms.P.effective_n
        },
        raw_n: {
          B: baselineBrain.arms.B.n,
          P: baselineBrain.arms.P.n
        },
        last_selected: baselineBrain.last_selected,
        selection_streak: baselineBrain.selection_streak,
        weights: BASELINE_WEIGHTS,
        no_bootstrap_on_start: true,
        delayed_feedback_update: true
      } : null,
      stats: state.stats,
      settlements: state.settlements.length,
      primary_prediction: state.lastPrediction ? {
        direction: state.lastPrediction.direction,
        confidence: state.lastPrediction.confidence,
        gap: state.lastPrediction.gap,
        reason: state.lastPrediction.reason,
        context64: state.lastPrediction.x,
        dimensions: { shoe: SHOE_DIM, road: ROAD_DIM, total: DIM }
      } : null,
      baseline_prediction: state.lastBaselinePrediction ? {
        direction: state.lastBaselinePrediction.direction,
        confidence: state.lastBaselinePrediction.confidence,
        gap: state.lastBaselinePrediction.gap,
        reason: state.lastBaselinePrediction.reason,
        context64: state.lastBaselinePrediction.x,
        dimensions: { shoe: SHOE_DIM, road: ROAD_DIM, total: DIM }
      } : null
    };
    el("debug").textContent = JSON.stringify(debug, null, 2);
  }

  function render() {
    el("modePill").textContent = state.active ? "等待下一局結算" : "準備下一次預測";
    el("roundPill").textContent = `${state.history.length} 局`;
    el("brainState").textContent = "64D · 牌靴60%／牌路40%";
    el("seedCount").textContent = state.history.length;
    el("liveCount").textContent = state.brain ? Math.round((state.brain.updates || 0)) : 0;
    el("baselineLiveCount").textContent = state.baselineBrain ? Math.round((state.baselineBrain.updates || 0)) : 0;

    el("btnStart").disabled = false;
    el("btnEnd").disabled = !state.active && !state.lastPrediction;

    renderHistory();
    renderPrediction();
    renderExperimentStats();
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

  // 兩組64D腦從同一時間點開始，確保60/40與50/50比較公平。
  if (!validBrain(state.brain)) state.brain = freshBrain();
  if (!validBrain(state.baselineBrain)) state.baselineBrain = freshBrain();
  if (!state.stats) state.stats = freshExperimentStats();
  if (!Array.isArray(state.settlements)) state.settlements = [];
  if (state.active && (!validPrediction(state.lastPrediction) || !validPrediction(state.lastBaselinePrediction))) {
    state.active = false;
    state.lastPrediction = null;
    state.lastBaselinePrediction = null;
  }

  render();
  if (migratedFromPreviousVersion) {
    setMessage("已保留舊版歷史，並為60/40主模型與50/50對照組建立兩套乾淨64D腦；A/B統計從零開始。", true);
    save();
  }
  initCanvas();
})();
