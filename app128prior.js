(() => {
  "use strict";

  const SHOE_DIM = 64;
  const ROAD_DIM = 64;
  const DIM = SHOE_DIM + ROAD_DIM;
  const ARMS = ["P", "B"];
  const RIDGE = 4.0;
  const ALPHA = 0.12;
  const SCORE_TIE_EPS = 1e-9;
  const SCORE_TEMP = 0.32;
  const PROB_MIN = 0.43;
  const PROB_MAX = 0.59;
  const AVG_CARDS_PER_HAND = 4.9;
  const DECKS = 8;
  const TOTAL_CARDS = 52 * DECKS;
  const PRIOR_VERSION = "FROZEN-PRIOR-V1-128D-ENGINEERED";
  const PRIOR_SOURCE = "engineered_static_prior_v1_no_training_dataset";
  const STORAGE_KEY = "bgs128d_frozen_prior_static_v1";
  const LEGACY_STORAGE_KEYS = [
    "bgs128d_64plus64_frozen_direct_tech_panel_v1",
    "bgs64d_32plus32_frozen_direct_tech_panel_v2",
    "bgs32d_frozen_direct_tech_panel_v1"
  ];

  const SHOE_NAMES = [
    "remaining_cards_ratio","penetration_ratio","estimated_hands_remaining_norm","shoe_maturity_ratio",
    "rank_A_relative_ratio","rank_2_relative_ratio","rank_3_relative_ratio","rank_4_relative_ratio",
    "rank_5_relative_ratio","rank_6_relative_ratio","rank_7_relative_ratio","rank_8_relative_ratio",
    "rank_9_relative_ratio","rank_10JQK_relative_ratio","physical_edge_proxy","shoe_information_reliability",
    "shoe_phase_early","shoe_phase_middle","shoe_phase_late","estimated_hands_played_norm",
    "remaining_decks_ratio","hands_elapsed_log_norm","tie_ratio_all","tie_ratio_recent8",
    "tie_ratio_recent16","bp_balance_strength","bp_entropy_recent12","outcome_entropy_recent12",
    "outcome_entropy_recent24","sample_support_norm","composition_missing_indicator","shoe_progression_confidence",
    "tie_ratio_recent4","tie_ratio_recent6","tie_ratio_recent12","tie_ratio_recent24",
    "tie_ratio_recent32","bp_entropy_recent4","bp_entropy_recent6","bp_entropy_recent8",
    "bp_entropy_recent16","bp_entropy_recent24","bp_entropy_recent32","outcome_entropy_recent6",
    "outcome_entropy_recent8","outcome_entropy_recent16","outcome_entropy_recent32","bp_balance_recent6",
    "bp_balance_recent8","bp_balance_recent16","bp_balance_recent24","bp_balance_recent32",
    "penetration_squared","penetration_sqrt","remaining_squared","remaining_sqrt",
    "shoe_phase_very_early","shoe_phase_early_mid","shoe_phase_mid_late","shoe_phase_very_late",
    "sample_support_8","sample_support_16","sample_support_24","sample_support_48"
  ];

  const ROAD_NAMES = [
    "current_side_banker_binary","current_run_length_norm","previous_run_length_norm","previous2_run_length_norm",
    "recent5_banker_ratio","recent8_banker_ratio","recent12_banker_ratio","recent5_turn_rate",
    "recent8_turn_rate","recent12_turn_rate","run_length_hazard_rate","hsmm_stable_probability",
    "big_eye_regularity","small_road_regularity","cockroach_road_regularity","derived_road_consensus",
    "current_side_player_binary","previous3_run_length_norm","recent3_banker_ratio","recent20_banker_ratio",
    "recent3_turn_rate","recent20_turn_rate","run_continue_probability","recent8_outcome_entropy",
    "recent20_outcome_entropy","recent6_run_volatility","recent5_run_height_trend","big_eye_support_norm",
    "small_road_support_norm","cockroach_road_support_norm","last2_same_side","last3_same_side",
    "recent2_banker_ratio","recent4_banker_ratio","recent6_banker_ratio","recent10_banker_ratio",
    "recent16_banker_ratio","recent24_banker_ratio","recent32_banker_ratio","recent48_banker_ratio",
    "recent2_turn_rate","recent4_turn_rate","recent6_turn_rate","recent10_turn_rate",
    "recent16_turn_rate","recent24_turn_rate","recent32_turn_rate","recent48_turn_rate",
    "big_eye_regularity_w4","small_road_regularity_w4","cockroach_road_regularity_w4",
    "big_eye_regularity_w16","small_road_regularity_w16","cockroach_road_regularity_w16",
    "previous4_run_length_norm","avg_run_last4_norm","avg_run_last8_norm","max_run_last8_norm",
    "run_std_last8_norm","run_delta_last_norm","alternating_last4","alternating_last6",
    "last4_same_side","last5_same_side"
  ];
  const FEATURE_NAMES = [...SHOE_NAMES, ...ROAD_NAMES];

  if (SHOE_NAMES.length !== SHOE_DIM || ROAD_NAMES.length !== ROAD_DIM || FEATURE_NAMES.length !== DIM) {
    throw new Error(`128D feature-name mismatch: shoe=${SHOE_NAMES.length}, road=${ROAD_NAMES.length}`);
  }

  const el = id => document.getElementById(id);
  const clip = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, Number.isFinite(+v) ? +v : lo));
  const mean = xs => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;

  let state = { history: [], active: false, lastPrediction: null };
  let migratedFromLegacy = "";

  function bp(seq) { return seq.filter(x => x === "B" || x === "P"); }

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

  function balanceStrength(seq, w) {
    return clip(1 - Math.abs(bankerRatio(seq, w) - 0.5) * 2);
  }

  function turnRate(seq, w) {
    const a = bp(seq).slice(-Math.max(2, w));
    if (a.length < 2) return 0.5;
    let turns = 0;
    for (let i = 1; i < a.length; i++) if (a[i] !== a[i - 1]) turns++;
    return turns / (a.length - 1);
  }

  function tieRatio(seq, w = 0) {
    const a = w > 0 ? seq.slice(-w) : seq;
    return a.length ? a.filter(x => x === "T").length / a.length : 0;
  }

  function entropyNorm(seq, w = 12) {
    const a = seq.slice(-w);
    if (!a.length) return 1;
    let e = 0;
    for (const o of ["B", "P", "T"]) {
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
    const s = bp(seq), sides = [], heights = [];
    const out = { big_eye: [], small_road: [], cockroach_road: [] };
    const offsets = { big_eye: 1, small_road: 2, cockroach_road: 3 };
    for (const side of s) {
      const newCol = !sides.length || side !== sides[sides.length - 1];
      if (newCol) { sides.push(side); heights.push(1); }
      else heights[heights.length - 1]++;
      const c = heights.length - 1, row = heights[c];
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
    const d1 = deltas.length ? deltas.at(-1) : "NA";
    const d2 = deltas.length >= 2 ? deltas.at(-2) : "NA";
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
    const done = rs.slice(0, -1), heights = done.map(x => x[1]), table = {};
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
    return den > 1e-12 ? { CONTINUE: (co + 3) / den, TURN: (tu + 3) / den } : { CONTINUE: 0.5, TURN: 0.5 };
  }

  function hazard(seq) {
    const rs = runs(seq);
    if (!rs.length) return 0.5;
    const [side, cur] = rs.at(-1);
    const heights = rs.slice(0, -1).map(x => x[1]);
    const table = hazardTable(rs), contexts = hazardContexts(side, cur, heights);
    let prob = { CONTINUE: 0.5, TURN: 0.5 }, penalty = 1, found = false;
    for (let i = 0; i < contexts.length; i++) {
      const [, key] = contexts[i], c = table[key] || { CONTINUE: 0, TURN: 0 };
      if (c.CONTINUE + c.TURN >= 4) { prob = hazardPosterior(c); found = true; break; }
      if (i < contexts.length - 1) penalty *= 0.75;
    }
    if (!found) {
      const g = table["HZG|GLOBAL"] || { CONTINUE: 0, TURN: 0 };
      if (g.CONTINUE + g.TURN > 0) prob = hazardPosterior(g); else penalty = 0;
    }
    const cont = (1 - penalty) * 0.5 + penalty * prob.CONTINUE;
    return clip(1 - cont);
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
    const slope = (h.at(-1) - h[0]) / (h.length - 1);
    return clip(0.5 + slope / 6);
  }

  function hsmmStable(seq) {
    const a = turnRate(seq, 10);
    const rs = runs(seq), cur = rs.length ? rs.at(-1)[1] : 0;
    const r = clip(cur / 6), e = entropyNorm(seq, 12), v = runVolatility(seq);
    const p = Math.exp(-(((a-.25)/.24)**2) - (((r-.70)/.28)**2) - (((e-.62)/.24)**2) - (((v-.26)/.24)**2));
    const q = Math.exp(-(((a-.84)/.18)**2) - (((r-.18)/.20)**2) - (((e-.70)/.23)**2) - (((v-.30)/.24)**2));
    const t = Math.exp(-(((a-.52)/.28)**2) - (((r-.34)/.26)**2) - (((e-.82)/.18)**2) - (((v-.72)/.23)**2));
    const n = Math.exp(-(((a-.55)/.30)**2) - (((r-.27)/.24)**2) - (((e-.94)/.11)**2) - (((v-.55)/.28)**2));
    const weights = [.25*p, .25*q, .20*t, .30*n], total = weights.reduce((x, y) => x + y, 0) || 1;
    return clip((weights[0] + weights[1]) / total);
  }

  function alternatingTail(seq, w) {
    const a = bp(seq).slice(-w);
    if (a.length < 2) return 0.5;
    let ok = 0;
    for (let i = 1; i < a.length; i++) if (a[i] !== a[i - 1]) ok++;
    return ok / (a.length - 1);
  }

  function sameTail(seq, w) {
    const a = bp(seq).slice(-w);
    if (a.length < w) return 0.5;
    return +(a.every(v => v === a[0]));
  }

  function runStats(seq) {
    const h = runs(seq).map(x => x[1]);
    const h4 = h.slice(-4), h8 = h.slice(-8);
    const avg4 = mean(h4), avg8 = mean(h8), max8 = h8.length ? Math.max(...h8) : 0;
    const var8 = h8.length ? mean(h8.map(v => (v - avg8) ** 2)) : 0;
    return {
      avg4: clip(avg4 / 8), avg8: clip(avg8 / 8), max8: clip(max8 / 8), std8: clip(Math.sqrt(var8) / 4)
    };
  }

  function context128(seq) {
    const used = Math.min(TOTAL_CARDS, seq.length * AVG_CARDS_PER_HAND);
    const remaining = Math.max(0, TOTAL_CARDS - used);
    const rr = clip(remaining / TOTAL_CARDS), penetration = clip(1 - rr), maturity = clip(seq.length / 70);
    const estimatedHandCapacity = TOTAL_CARDS / AVG_CARDS_PER_HAND;
    const handsPlayedNorm = clip(seq.length / estimatedHandCapacity);
    const phaseEarly = clip(1 - penetration / 0.35);
    const phaseMiddle = clip(1 - Math.abs(penetration - 0.50) / 0.35);
    const phaseLate = clip((penetration - 0.55) / 0.35);
    const allBankerRatio = bankerRatio(seq, Math.max(1, bp(seq).length));

    const shoe = [
      rr, penetration, rr, maturity,
      1,1,1,1,1,1,1,1,1,1,
      0,0,
      phaseEarly, phaseMiddle, phaseLate, handsPlayedNorm,
      rr, clip(Math.log1p(seq.length) / Math.log1p(estimatedHandCapacity)),
      tieRatio(seq), tieRatio(seq, 8), tieRatio(seq, 16),
      clip(1 - Math.abs(allBankerRatio - 0.5) * 2),
      binaryEntropyNorm(seq, 12), entropyNorm(seq, 12), entropyNorm(seq, 24),
      clip(seq.length / 32), 1, clip(Math.sqrt(seq.length) / Math.sqrt(estimatedHandCapacity)),
      tieRatio(seq, 4), tieRatio(seq, 6), tieRatio(seq, 12), tieRatio(seq, 24), tieRatio(seq, 32),
      binaryEntropyNorm(seq, 4), binaryEntropyNorm(seq, 6), binaryEntropyNorm(seq, 8),
      binaryEntropyNorm(seq, 16), binaryEntropyNorm(seq, 24), binaryEntropyNorm(seq, 32),
      entropyNorm(seq, 6), entropyNorm(seq, 8), entropyNorm(seq, 16), entropyNorm(seq, 32),
      balanceStrength(seq, 6), balanceStrength(seq, 8), balanceStrength(seq, 16), balanceStrength(seq, 24), balanceStrength(seq, 32),
      penetration * penetration, Math.sqrt(penetration), rr * rr, Math.sqrt(rr),
      clip(1 - penetration / 0.18), clip(1 - Math.abs(penetration - 0.30) / 0.22),
      clip(1 - Math.abs(penetration - 0.62) / 0.24), clip((penetration - 0.72) / 0.22),
      clip(seq.length / 8), clip(seq.length / 16), clip(seq.length / 24), clip(seq.length / 48)
    ];

    const rs = runs(seq);
    const current = rs.length ? rs.at(-1) : ["", 0];
    const prev = rs.length >= 2 ? rs.at(-2) : ["", 0];
    const prev2 = rs.length >= 3 ? rs.at(-3) : ["", 0];
    const prev3 = rs.length >= 4 ? rs.at(-4) : ["", 0];
    const prev4 = rs.length >= 5 ? rs.at(-5) : ["", 0];
    const sideB = current[0] === "B" ? 1 : current[0] === "P" ? 0 : 0.5;
    const sideP = current[0] === "P" ? 1 : current[0] === "B" ? 0 : 0.5;

    const dr = buildDerivedRoads(seq);
    const [be, beSupport] = regularity(dr.big_eye, 8), [sm, smSupport] = regularity(dr.small_road, 8), [cr, crSupport] = regularity(dr.cockroach_road, 8);
    const [be4] = regularity(dr.big_eye, 4), [sm4] = regularity(dr.small_road, 4), [cr4] = regularity(dr.cockroach_road, 4);
    const [be16] = regularity(dr.big_eye, 16), [sm16] = regularity(dr.small_road, 16), [cr16] = regularity(dr.cockroach_road, 16);
    const regMean = (be + sm + cr) / 3;
    const consensus = clip(1 - (Math.abs(be - regMean) + Math.abs(sm - regMean) + Math.abs(cr - regMean)) / 1.5);
    const turnHazard = hazard(seq), roadBP = bp(seq), stats = runStats(seq);
    const last2Same = roadBP.length >= 2 ? +(roadBP.at(-1) === roadBP.at(-2)) : 0.5;
    const last3Same = roadBP.length >= 3 ? +(roadBP.at(-1) === roadBP.at(-2) && roadBP.at(-2) === roadBP.at(-3)) : 0.5;
    const runDelta = rs.length >= 2 ? clip(0.5 + (current[1] - prev[1]) / 12) : 0.5;

    const road = [
      sideB, clip(current[1] / 8), clip(prev[1] / 8), clip(prev2[1] / 8),
      bankerRatio(seq, 5), bankerRatio(seq, 8), bankerRatio(seq, 12),
      turnRate(seq, 5), turnRate(seq, 8), turnRate(seq, 12),
      turnHazard, hsmmStable(seq), be, sm, cr, consensus,
      sideP, clip(prev3[1] / 8), bankerRatio(seq, 3), bankerRatio(seq, 20),
      turnRate(seq, 3), turnRate(seq, 20), clip(1 - turnHazard),
      entropyNorm(seq, 8), entropyNorm(seq, 20), runVolatility(seq), runHeightTrend(seq, 5),
      clip(beSupport / 8), clip(smSupport / 8), clip(crSupport / 8), last2Same, last3Same,
      bankerRatio(seq, 2), bankerRatio(seq, 4), bankerRatio(seq, 6), bankerRatio(seq, 10),
      bankerRatio(seq, 16), bankerRatio(seq, 24), bankerRatio(seq, 32), bankerRatio(seq, 48),
      turnRate(seq, 2), turnRate(seq, 4), turnRate(seq, 6), turnRate(seq, 10),
      turnRate(seq, 16), turnRate(seq, 24), turnRate(seq, 32), turnRate(seq, 48),
      be4, sm4, cr4, be16, sm16, cr16,
      clip(prev4[1] / 8), stats.avg4, stats.avg8, stats.max8, stats.std8, runDelta,
      alternatingTail(seq, 4), alternatingTail(seq, 6), sameTail(seq, 4), sameTail(seq, 5)
    ];

    const vector = [...shoe, ...road];
    if (shoe.length !== SHOE_DIM || road.length !== ROAD_DIM || vector.length !== DIM) {
      throw new Error(`128D context shape mismatch: shoe=${shoe.length}, road=${road.length}, total=${vector.length}`);
    }
    return vector;
  }

  function featureIndex(name) {
    const i = FEATURE_NAMES.indexOf(name);
    if (i < 0) throw new Error(`Unknown prior feature: ${name}`);
    return i;
  }

  function buildFrozenPrior() {
    const thetaB = Array(DIM).fill(0), thetaP = Array(DIM).fill(0), aDiag = Array(DIM).fill(RIDGE);
    const setPair = (name, bCoef, pCoef, precision = RIDGE) => {
      const i = featureIndex(name);
      thetaB[i] = bCoef;
      thetaP[i] = pCoef;
      aDiag[i] = precision;
    };

    // Symmetric, fixed directional prior. Shoe-only progression features do not directly choose B/P;
    // they participate in deterministic regime gating below. This avoids inventing card-composition signal.
    setPair("current_side_banker_binary", 0.18, -0.18, 6.0);
    setPair("current_side_player_binary", -0.18, 0.18, 6.0);

    const directionalRatios = [
      ["recent2_banker_ratio", 0.018], ["recent3_banker_ratio", 0.024], ["recent4_banker_ratio", 0.030],
      ["recent5_banker_ratio", 0.035], ["recent6_banker_ratio", 0.040], ["recent8_banker_ratio", 0.050],
      ["recent10_banker_ratio", 0.046], ["recent12_banker_ratio", 0.042], ["recent16_banker_ratio", 0.036],
      ["recent20_banker_ratio", 0.032], ["recent24_banker_ratio", 0.028], ["recent32_banker_ratio", 0.022],
      ["recent48_banker_ratio", 0.016]
    ];
    for (const [name, c] of directionalRatios) setPair(name, c, -c, 5.0);

    const bB = thetaB.map((v, i) => v * aDiag[i]);
    const bP = thetaP.map((v, i) => v * aDiag[i]);
    return {
      version: PRIOR_VERSION,
      source: PRIOR_SOURCE,
      arms: {
        B: { A_diag: [...aDiag], b: bB },
        P: { A_diag: [...aDiag], b: bP }
      },
      nonzero_theta: thetaB.filter((v, i) => Math.abs(v) + Math.abs(thetaP[i]) > 0).length
    };
  }

  const FROZEN_PRIOR = buildFrozenPrior();

  function neutralCenter(name) {
    if (name.startsWith("rank_") && name.endsWith("_relative_ratio")) return 1;
    if (name === "composition_missing_indicator") return 1;
    if (name === "physical_edge_proxy" || name === "shoe_information_reliability") return 0;
    return 0.5;
  }

  function normalizeContext(raw) {
    return raw.map((v, i) => {
      const center = neutralCenter(FEATURE_NAMES[i]);
      if (center === 1) return clip(v, 0, 2) - 1;
      if (center === 0) return clip(v, -1, 1);
      return clip((v - center) / 0.5, -1, 1);
    });
  }

  function regimeGate(seq) {
    const nBP = bp(seq).length;
    const rs = runs(seq), currentRun = rs.length ? rs.at(-1)[1] : 0;
    const tr8 = turnRate(seq, 8), tr16 = turnRate(seq, 16), hz = hazard(seq), stable = hsmmStable(seq);
    const alt6 = alternatingTail(seq, 6), same4 = sameTail(seq, 4);
    const entropy12 = binaryEntropyNorm(seq, 12);
    const support = clip(nBP / 18);
    const derived = buildDerivedRoads(seq);
    const [be, beN] = regularity(derived.big_eye, 8), [sm, smN] = regularity(derived.small_road, 8), [cr, crN] = regularity(derived.cockroach_road, 8);
    const derivedSupport = clip((beN + smN + crN) / 24);
    const derivedAgreement = clip(1 - (Math.abs(be-sm) + Math.abs(sm-cr) + Math.abs(be-cr)) / 3);

    const trendEvidence = 0.28 * (1 - tr8) + 0.18 * (1 - tr16) + 0.20 * (1 - hz) + 0.16 * same4 + 0.10 * stable + 0.08 * clip(currentRun / 5);
    const reverseEvidence = 0.30 * tr8 + 0.18 * tr16 + 0.22 * hz + 0.18 * alt6 + 0.12 * (1 - stable);
    let directionalMode = clip((trendEvidence - reverseEvidence) * 2.2, -1, 1);
    if (Math.abs(directionalMode) < 0.08) directionalMode *= 0.5;

    const noisePenalty = clip(0.55 * entropy12 + 0.45 * runVolatility(seq));
    const structureConfidence = clip(0.30 + 0.35 * support + 0.20 * derivedSupport * derivedAgreement + 0.15 * (1 - noisePenalty));

    let shoeWeight = nBP < 8 ? 0.65 : nBP < 20 ? 0.60 : 0.56;
    if (noisePenalty > 0.72) shoeWeight = Math.min(0.68, shoeWeight + 0.05);
    if (currentRun >= 4 && tr8 < 0.45) shoeWeight = Math.max(0.52, shoeWeight - 0.04);
    const roadWeight = 1 - shoeWeight;

    return {
      nBP, currentRun, turnRate8: tr8, turnRate16: tr16, hazard: hz, stable,
      directionalMode, structureConfidence, shoeWeight, roadWeight,
      regime: directionalMode > 0.22 ? "trend" : directionalMode < -0.22 ? "reversal" : "neutral"
    };
  }

  function modelContext(raw, seq) {
    const x = normalizeContext(raw), gate = regimeGate(seq);
    const roadScale = gate.roadWeight * (0.30 + 0.70 * gate.structureConfidence);
    const shoeScale = gate.shoeWeight;
    const directionalNames = new Set([
      "current_side_banker_binary", "current_side_player_binary",
      "recent2_banker_ratio", "recent3_banker_ratio", "recent4_banker_ratio", "recent5_banker_ratio",
      "recent6_banker_ratio", "recent8_banker_ratio", "recent10_banker_ratio", "recent12_banker_ratio",
      "recent16_banker_ratio", "recent20_banker_ratio", "recent24_banker_ratio", "recent32_banker_ratio", "recent48_banker_ratio"
    ]);

    for (let i = 0; i < DIM; i++) {
      if (i < SHOE_DIM) x[i] *= shoeScale;
      else x[i] *= roadScale;
      if (directionalNames.has(FEATURE_NAMES[i])) x[i] *= gate.directionalMode;
    }
    return { x, gate };
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function scoreArm(arm, x) {
    const theta = arm.b.map((v, i) => v / arm.A_diag[i]);
    const meanScore = dot(x, theta);
    let quad = 0;
    for (let i = 0; i < x.length; i++) quad += (x[i] * x[i]) / arm.A_diag[i];
    const uncertainty = Math.sqrt(Math.max(0, quad));
    return { mean: meanScore, uncertainty, effectiveAlpha: ALPHA, score: meanScore + ALPHA * uncertainty };
  }

  function choose(seq) {
    const rawX = context128(seq);
    const transformed = modelContext(rawX, seq);
    const scores = {
      B: scoreArm(FROZEN_PRIOR.arms.B, transformed.x),
      P: scoreArm(FROZEN_PRIOR.arms.P, transformed.x)
    };
    const gap = scores.B.score - scores.P.score;
    let direction, reason;
    if (Math.abs(gap) <= SCORE_TIE_EPS) {
      let h = 2166136261 >>> 0;
      const token = `${PRIOR_VERSION}|${seq.join("")}`;
      for (let i = 0; i < token.length; i++) { h ^= token.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
      direction = h & 1 ? "B" : "P";
      reason = "fixed_hash_tie_only";
    } else {
      direction = gap > 0 ? "B" : "P";
      reason = "frozen_prior_linucb_argmax";
    }

    const rawPB = 1 / (1 + Math.exp(-Math.max(-8, Math.min(8, gap / SCORE_TEMP))));
    const pB = clip(rawPB, PROB_MIN, PROB_MAX), pP = 1 - pB;
    const confidence = direction === "B" ? pB : pP;
    return {
      direction, reason, rawX, x: transformed.x, gate: transformed.gate, scores, gap,
      probabilities: { B: pB, P: pP }, confidence
    };
  }

  function save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        version: PRIOR_VERSION,
        history: state.history,
        active: state.active,
        lastPrediction: state.lastPrediction
      }));
    } catch (_) {}
  }

  function load() {
    try {
      let source = STORAGE_KEY, text = localStorage.getItem(STORAGE_KEY);
      if (!text) {
        for (const key of LEGACY_STORAGE_KEYS) {
          const candidate = localStorage.getItem(key);
          if (candidate) { text = candidate; source = key; break; }
        }
      }
      const raw = JSON.parse(text || "null");
      if (!raw || !Array.isArray(raw.history)) return;
      state.history = raw.history.filter(x => ["B", "P", "T"].includes(x)).slice(-500);
      state.active = false;
      state.lastPrediction = null;
      migratedFromLegacy = source !== STORAGE_KEY ? source : "";
    } catch (_) {}
  }

  function addOutcome(outcome) {
    if (!["B", "P", "T"].includes(outcome)) return;
    state.history.push(outcome);
    state.active = false;
    state.lastPrediction = null;
    setMessage(`已加入第 ${state.history.length} 局：${outcome === "B" ? "莊" : outcome === "P" ? "閒" : "和"}。Frozen prior 完全不更新；按「開始分析」重新計算下一局。`);
    save(); render();
  }

  function startAnalysis() {
    if (!state.history.length) { setMessage("請先用莊／閒／和輸入歷史紀錄。", true); return; }
    // Frozen inference only: no bootstrap, no walk-forward, no replay, no settlement,
    // no A/b update, no decay. Static A/b are regenerated from source code every load.
    state.active = true;
    state.lastPrediction = choose(state.history);
    setMessage(`Frozen Prior V1 已預測第 ${state.history.length + 1} 局。A/b 固定、updates=0。`);
    save(); render();
  }

  function endAnalysis() {
    if (!state.active && !state.lastPrediction) { setMessage("目前沒有進行中的分析結果。", true); return; }
    state.active = false;
    state.lastPrediction = null;
    setMessage("分析已結束。歷史保留；Frozen prior 本身從未修改。" );
    save(); render();
  }

  function backOne() {
    if (!state.history.length) { setMessage("目前沒有可以返回的牌局。", true); return; }
    const removed = state.history.pop();
    state.active = false;
    state.lastPrediction = null;
    setMessage(`已返回上一局（移除 ${removed}）。Frozen A/b 沒有任何變動。`);
    save(); render();
  }

  function setMessage(text, warn = false) {
    const box = el("message");
    box.textContent = text;
    box.style.borderLeftColor = warn ? "#f7d46a" : "#55d6ff";
    box.style.color = warn ? "#dfc986" : "#9ab4c7";
  }

  function renderHistory() {
    const box = el("historyTrack");
    if (!state.history.length) { box.innerHTML = '<div class="empty-history">尚未輸入牌局</div>'; return; }
    box.innerHTML = state.history.map((v, i) => `<div class="history-chip ${v}" title="第${i+1}局">${v}</div>`).join("");
  }

  function renderFeatures(vector) {
    const v = Array.isArray(vector) && vector.length === DIM ? vector : Array(DIM).fill(0);
    el("shoeGrid").innerHTML = SHOE_NAMES.map((name, i) =>
      `<div class="feature-row"><span>${String(i + 1).padStart(3, "0")} · ${name}</span><b>${(+v[i]).toFixed(5)}</b></div>`
    ).join("");
    el("roadGrid").innerHTML = ROAD_NAMES.map((name, i) => {
      const idx = i + SHOE_DIM;
      return `<div class="feature-row"><span>${String(idx + 1).padStart(3, "0")} · ${name}</span><b>${(+v[idx]).toFixed(5)}</b></div>`;
    }).join("");
  }

  function renderPrediction() {
    const p = state.lastPrediction, orb = el("directionOrb");
    if (!p || !state.active) {
      el("directionText").textContent = "—";
      el("directionCode").textContent = "WAIT";
      el("confidence").textContent = "—";
      el("ucbB").textContent = "—";
      el("ucbP").textContent = "—";
      el("scoreGap").textContent = "—";
      orb.className = "direction-orb idle";
      renderFeatures(context128(state.history));
      return;
    }
    el("directionText").textContent = p.direction === "B" ? "莊" : "閒";
    el("directionCode").textContent = p.direction === "B" ? "BANKER · B" : "PLAYER · P";
    el("confidence").textContent = (p.confidence * 100).toFixed(2) + "%";
    el("ucbB").textContent = p.scores.B.score.toFixed(6);
    el("ucbP").textContent = p.scores.P.score.toFixed(6);
    el("scoreGap").textContent = p.gap.toFixed(6);
    orb.className = "direction-orb " + (p.direction === "B" ? "banker" : "player");
    renderFeatures(p.rawX);
  }

  function renderDebug() {
    const p = state.lastPrediction;
    const debug = {
      model: "128D Frozen Prior Contextual LinUCB",
      version: PRIOR_VERSION,
      prior_source: PRIOR_SOURCE,
      pretrained_from_dataset: false,
      note: "Repository contains no historical shoe dataset; this test uses a transparent engineered static prior, not claimed pretrained weights.",
      dimensions: { shoe: SHOE_DIM, road: ROAD_DIM, total: DIM },
      A_representation: "fixed diagonal precision vector",
      b_representation: "fixed vector derived once from A_diag * theta",
      nonzero_prior_features: FROZEN_PRIOR.nonzero_theta,
      immutable_runtime: {
        bootstrap: false,
        walk_forward: false,
        replay: false,
        settle_previous: false,
        update_A_b: false,
        decay: false,
        updates: 0
      },
      active: state.active,
      totalHistory: state.history.length,
      history: state.history.join(""),
      prediction: p ? {
        direction: p.direction,
        confidence: p.confidence,
        reason: p.reason,
        gap: p.gap,
        regime_gate: p.gate,
        scores: p.scores,
        raw_context_128: p.rawX,
        model_context_128: p.x
      } : null
    };
    el("debug").textContent = JSON.stringify(debug, null, 2);
  }

  function render() {
    el("modePill").textContent = state.active ? "Frozen Prior 已預測" : "準備歷史";
    el("roundPill").textContent = `${state.history.length} 局`;
    el("brainState").textContent = "Frozen Prior V1";
    el("seedCount").textContent = state.history.length;
    el("liveCount").textContent = "0";
    el("btnStart").disabled = false;
    el("btnEnd").disabled = !state.active && !state.lastPrediction;
    renderHistory(); renderPrediction(); renderDebug();
  }

  function initCanvas() {
    const canvas = el("techCanvas"), ctx = canvas && canvas.getContext("2d");
    if (!ctx) return;
    let w = 0, h = 0, nodes = [];
    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth; h = window.innerHeight;
      canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr);
      canvas.style.width = w + "px"; canvas.style.height = h + "px";
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
      for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j], dx = a.x - b.x, dy = a.y - b.y, d = Math.hypot(dx, dy);
        if (d < 135) {
          ctx.strokeStyle = `rgba(74,194,255,${(1 - d / 135) * .10})`;
          ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
      }
      for (const n of nodes) {
        ctx.fillStyle = "rgba(98,220,255,.32)";
        ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fill();
      }
      requestAnimationFrame(frame);
    }
    resize(); window.addEventListener("resize", resize, { passive: true }); frame();
  }

  el("btnB").addEventListener("click", () => addOutcome("B"));
  el("btnP").addEventListener("click", () => addOutcome("P"));
  el("btnT").addEventListener("click", () => addOutcome("T"));
  el("btnStart").addEventListener("click", startAnalysis);
  el("btnEnd").addEventListener("click", endAnalysis);
  el("btnBack").addEventListener("click", backOne);

  load();
  render();
  if (migratedFromLegacy) {
    setMessage("已保留舊版本歷史；舊 32D/64D/128D A/b 全部未沿用。現在使用程式內固定 Frozen Prior V1。", true);
    save();
  }
  initCanvas();
})();