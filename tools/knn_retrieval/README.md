# 相似局面检索：阶段 1 前置审计

日期：2026-10-29（Asia/Taipei）。基线为 PR #72 最终 head
`42106a9adf082630357b3ae7ccd1c422c1add489`，不是 PR 合并前的 main。
独立分支：`calib/pr72-20261029-knn-retrieval`。

## 本次回放统计状态

**INSUFFICIENT_SAMPLE；未执行回放、建库、拟合或显著性检验。**
该基线的完整递归文件树没有完整回放数据。全部 26 个 JSON 文件均已检查顶层结构，
没有逐局回放 rows 数组；模型、汇总报告、固定测试夹具不能作为建库样本。
检查清单见 `../../validation/knn_retrieval/source_audit.json`。
`research/README.md` 与 `docs/MIDDLE_LATE_RETRAIN_EXECUTION.md` 也明确记录没有提供真实完整逐靴资料。
仅检查了指定基线可访问的仓库内容；没有声称已查遍用户设备、其他分支或所有 Actions 产物。

| 数据切分 | 可用完整靴数 | 可用检索样本数 | 时间范围 |
|---|---:|---:|---|
| 前 70% 建库 | 0 | 0 | 未提供 |
| 后 30% 内前半：validation_fit | 0 | 0 | 未提供 |
| 后 30% 内后半：validation_test | 0 | 0 | 未提供 |

| 回放统计指标 | 基线 | +检索无闸门 | +检索有闸门 |
|---|---|---|---|
| standard_hit_rate | N/A | N/A | N/A |
| pct_shoes_max_losses_le_2 | N/A | N/A | N/A |
| pct_shoes_with_four_streak | N/A | N/A | N/A |
| mean_shoe_five_window_coverage | N/A | N/A | N/A |
| mean_shoe_six_window_coverage | N/A | N/A | N/A |
| action_rate | N/A | N/A | N/A |

逐靴表为空；增量与置信区间为 null。不能用无数据把熔断指标填成 1.0。
`NO_INCREMENT=null` 表示没有评估，不是已经证明无增量。
停止于资料前置检查，不建立 500 万组库，也不建立或复制假样本凑 5 万组。
本分支只实现前置检查和规范；**没有实现 KNN、融合拟合或三组正式 runtime 回放**。

## 固定的后续实验规范（尚未执行）

- 按可追溯靴开始时间由旧到新排序，完整靴不得交错，含和局与未下注局。
  先排除长度不在 50–70 局的靴并记录 OUT_OF_RANGE_SHOE，再按完整合格靴切前 70% 建库、后 30% 验证。
  验证部分内部按时间分半：前半仅拟合闸门与融合，后半一次性评估；二者均不能进库。
  小数向下取整，余数留给最终测试。随机种子 20261029。
- 每条库记录对应一局之前的状态和该局事后 B/P/T 标签；首局不作为正式出手机会。
  仅用 50,000–500,000 条独立已观测记录；超过上限时仅在建库区用固定种子无放回抽样。
  数据不足不得扩增、重复采样或生成模拟结果冒充回放。
- 检索键共 11 维：事前剩余牌总张数，以及点值 0、1、…、9 的各自剩余张数比例。
  0 合并 10/J/Q/K，不冒称 A–K 牌阶。计数必须来自可审计初始牌组、烧牌及已公开消耗，
  并在预测前实际可取得。纯 B/P/T、最终点数或总张数无法还原真实组成。
  不接受模拟器隐藏的 oracle 组成冒充线上可得特征；若仅可获得 PF 后验，需要另行定义和验证数据契约。
- 排除：大路、珠盘路、连庄/连闲、跳跃、B/P/T 频率、最近结果序列、256D Context、
  LinUCB 状态、Physics/PF 输出、未来牌值、当前局结果、时间和 shoe_id。
  原因：键只表示可观察的组成，避免路单代理、身份记忆和未来泄漏。
- 选择标准化欧氏距离：均值和标准差只从建库样本拟合；零方差维标准差设为 1。
  保留总量消耗差异，且让比例维与张数维可比较。k 默认 30、可配置 20–50；
  本次不搜索 k。相同距离用库记录稳定顺序打破平局。
- 邻居输出 B/P/T 计数与分布、相对当局基线候选方向的 W/L 计数及条件分布、和局数、
  欧氏距离均值与总体方差。和局为 push，不挤入 W/L 分母。熵用 B/P/T 分布的归一化 Shannon 熵。
- 独立投票使用邻居非和条件 P(B)。保留基线方向、概率、standard/Skip 和全部风控诊断。
  不替换原预测，不反转原方向，不把 Skip 改成出手；只允许额外拒绝原本通过的 standard 机会。
  这会改变 action_rate，必须与命中率一起报告。
- 无闸门组：在 validation_fit 上以二元 log loss 拟合非负 logistic stacking
  （基线 logit、邻居 logit、截距），以及附加拒绝阈值；不得手写融合权重。
  有闸门组：同样仅在 validation_fit 拟合，距离/熵不通过时回退原基线。
  闸门候选仅用 validation_fit 的距离与熵分位数 0.25/0.5/0.75/1.0，
  结合其内部按靴的时间前推验证选择，权重仅用各折过去部分拟合。
  两组冻结后才解封 validation_test，不以测试结果重选参数。
- 每组须独立重放自身出手/结算历史，并使用相同未修改的正式熔断路径。
  如果无法在不改核心的情况下取得正式 standard 分类和熔断状态，停止为实现阻挡，
  不用事后截掉第三次亏损伪装熔断通过。

## 指标与停止规则（预注册，尚未计算）

正式 standard 分类必须沿用 runtime，不把任意 B/P 动作自动视为 standard。
standard_hit_rate 为 standard 胜数/非和 standard 出手数；action_rate 为 standard 出手数/
第 2 局起所有机会（含和）。逐靴最大连败在 standard 已结算序列计算，和局/Skip 不重置；
`pct_shoes_max_losses_le_2` 必须实测为 1.0，否则硬失败且禁止扩库。
四连胜在非和 standard 出手序列计算，和局/Skip 不重置胜段。
五/六窗口覆盖率按原始连续局号滚动窗口计算：窗口内至少含四笔连续获胜的
standard 非和结算，中间不得有 standard 亏损；和局/Skip 占窗口位置，窗口不能跨靴。
每靴比例再等权平均，完整靴无出手时四连胜/窗口覆盖为 0，命中率为 null。
若仓库已有不同定义，必须先核对并固定兼容口径，不能混报。

最终测试至少 30 靴且每组至少 500 笔非和 standard 出手，否则 INSUFFICIENT_SAMPLE。
这是预设最低操作门槛，不代表统计功效充分。
用 10,000 次配对整靴 bootstrap，每次合并重抽靴的胜/负计数重新计算总体命中率；
覆盖率为重抽靴等权均值。报告全部六项指标的逐靴表、跨靴汇总、增量与区间。
两候选组 × 三个共同主要终点（命中率、五窗口、六窗口覆盖）用 Bonferroni
同时区间：每项双侧 99.1667% CI，并同时给描述性 95% CI。
三项同时区间下界均 >0 且熔断硬门槛通过，才视为该组本次回放统计有显著增量。
有足够数据但任一目标未显著提升则 NO_INCREMENT，停止；不扩到 500 万，不改熔断。
所有结果限于回放统计，不提供未来胜率保证。

## 前置检查与回滚

```bash
python -m unittest discover -s tools/knn_retrieval -p 'test_*.py' -v
python tools/knn_retrieval/preflight.py --output /tmp/knn-preflight.json
# 数据到位后：仅检查资料，不启动未实现的实验
python tools/knn_retrieval/preflight.py --input complete_hands.json --output /tmp/knn-data-audit.json
```

无资料/资料不合格退出码 2；通过仅代表 DATA_PREFLIGHT_PASSED_NOT_EVALUATED。
输入 JSON 包含 data_kind=real_observations、source、chronological_order=oldest_to_newest、rows。
每行包含 shoe_id、round_index、outcome、带时区 shoe_start、remaining_counts_before（10 个非负整数）、
composition_source=observed_prefix、composition_through_round=round_index-1、composition_evidence。
字段声明不能替代人工来源及牌数守恒审计；当前检查器不验证实际牌局补牌规则或逐手消耗。

本分支仅新增 tools/knn_retrieval 与 validation/knn_retrieval；未改运行时、模型、页面或 workflow。
main 与 PR #72 没有修改。撤销实验提交时在本实验分支执行 `git revert <实验提交SHA>`，
或离开实验分支：`git switch --detach 42106a9adf082630357b3ae7ccd1c422c1add489`。
无需回滚线上代码，因为没有部署或改动线上路径。
