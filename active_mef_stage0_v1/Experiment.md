# Experiment
Active MEF Stage 0 —实验操作记录
项目路径：C:\Users\Lenovo\Desktop\code\HDR\active_mef_stage0_v1
数据集：SICE Part1（训练）+ Part2（测试），共 210 个测试场景
实验目标：在进入完整方法开发前，通过三个不需要复杂模型的 Kill Test 验证核心假设是否成立

一、研究问题与核心假设
现有多曝光融合（MEF）研究解决了"如何融合给定的曝光集合"，但没有解决"应该拍哪张曝光"。本工作验证以下假设：

给定当前已采集的曝光集合 C_t，候选曝光 a的边际价值 ΔQ(a|C_t)具有明显的 scene-dependent 和 state-dependent 结构，并且可以在实际采集前从 (H_t, a) 预测。

Stage 0 只做可行性验证，不训练复杂模型。三个 Kill Test：

Kill Test 1：Oracle headroom 是否足够大？（adaptive 比fixed bracket 好多少）
Kill Test 2：Candidate value 是否可预测？（H_t 是否比 histogram 更好的state表征）
Kill Test 3：One-step greedy 是否足够？（是否需要 World Model）
Kill Test 1 和 3 已完成；Kill Test 2 待完成。

二、代码结构

active_mef_stage0_v1/
├── src/active_mef/
│   ├── oracle.py          # SceneEvaluator:枚举 oracle gain，greedy/sequence 搜索
│   ├── metrics.py         # psnr, psnr_mu, ssim
│   ├── policies.py        # HistogramCoverageHeuristic, rollout_random
│   ├── analysis.py        # summarize_results, paired_bootstrap_delta
│   ├── io.py              # read_image (支持 max_size resize)
│   ├── config.py          # load_yaml
│   ├── fusion/
│   │   ├── base.py        # FusionBackend抽象类
│   │   ├── linear_radiance.py  # 纯 NumPy weighted radiance fusion（Stage 0 使用）
│   │   ├── mertens.py     # cv2 Mertens fusion
│   │   └── freemef_backend.py  # FreeMEF（需要外部 checkpoint）
│   ├── sim/camera.py      # CameraSimulator（HDR→LDR 模拟，Stage 0 未使用）
│   └── data/
│       ├── manifest.py    # ManifestExposurePoolDataset（含 EXIF 方向修正）
│       └── builders.py    # build_sequence_pool_manifest
├── scripts/
│   ├── run_stage0.py      # 主入口：运行所有 oracle 分析
│   ├── analyze_stage0.py  # 分析 value tensor 结构
│   ├── build_manifest.py  # 构建 JSONL manifest
│   └── make_toy_dataset.py
├── configs/
│   ├── stage0_sice_mertens.yaml   # SICE 正式实验配置
│   └── stage0_toy.yaml            # Toy 数据快速验证配置
├── data/manifests/
│   ├── sice_train.jsonl   # 230 场景（Part1，7/9帧场景）
│   └── sice_test.jsonl    # 210 场景（Part2，7/9帧场景）
├── SICE/                # 原始数据（17MP，未修改）
├── SICE_512/              # 预处理后的数据（512px，EXIF 修正）
└── results/kill_test_sice/        # Kill Test 1+3 结果
三、环境配置

# Python 3.14，在项目根目录执行
pip install -r requirements.txt   # numpy, pandas, PyYAML, imageio, opencv-python-headless, scikit-image, tqdm
pip install piexif                # EXIF 方向修正（resize脚本需要）
pip install pytest                # 运行 smoke test
pip install -e .                  # 安装本包

# 验证环境
python -m pytest tests/ -v        # 应看到 1 passed
四、数据准备
4.1 SICE 目录结构

SICE/
├── Dataset_Part1/Dataset_Part1/# 360 场景（LDR 序列）
│   ├──1/1.JPG 2.JPG ... 7.JPG
│   ├── 2/  ...
│   └── Label/1.JPG 2.JPG ...# GT 参考图
└── Dataset_Part2/Dataset_Part2/   # 229 场景
    ├── 1/  ...
    └── Label/  ...
每个场景有 3–20 帧不等；本实验只使用 7帧或9帧 的场景（数量最多，EV分配一致）。Part1 EV 分配（7帧，ordinal step=1）：-3, -2, -1, 0, +1, +2, +3；9帧：-4, -3, -2, -1, 0, +1, +2, +3, +4。

4.2 离线resize（必须先做）
原始图像为 3456×5184（17MP），直接加载极慢。需要先 resize 到 512px，同时修正 JPEG EXIF 旋转（否则 GT 和曝光帧方向不一致，PSNR 计算错误）：


# 在项目根目录执行，约 25 分钟，结果写入 SICE_512/
python - << 'EOF'
import cv2, re, piexif
from pathlib import Path
from tqdm import tqdm

MAX_SIZE = 512
SICE_ROOT = Path("SICE")
OUT_ROOT  = Path("SICE_512")

EXIF_ROT = {
    2: lambda img: cv2.flip(img, 1),
    3: lambda img: cv2.rotate(img, cv2.ROTATE_180),
    4: lambda img: cv2.flip(img, 0),
    5: lambda img: cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_CLOCKWISE),
    6: lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    7: lambda img: cv2.rotate(cv2.flip(img, 0), cv2.ROTATE_90_CLOCKWISE),
    8: lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
}

def fix_and_resize(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None: return
    try:
        exif = piexif.load(str(src))
        ori= exif.get("0th", {}).get(piexif.ImageIFD.Orientation, 1)
        if ori in EXIF_ROT: img = EXIF_ROT[ori](img)
    except Exception: pass
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        s = MAX_SIZE / max(h, w)
        img = cv2.resize(img, (max(1,int(round(w*s))), max(1,int(round(h*s)))),
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img)

def nkey(p):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r'(\d+)', p.name)]

for part_name in ["Dataset_Part1/Dataset_Part1", "Dataset_Part2/Dataset_Part2"]:
    part_in= SICE_ROOT / part_name
    label_in = part_in / "Label"
    part_out = OUT_ROOT / part_name
    for scene in tqdm(sorted([s for s in part_in.iterdir()if s.is_dir() and s.name != "Label"],key=lambda p: int(p.name) if p.name.isdigit() else 99999),
                      desc=part_name.split('/')[0]):
        frames = sorted([f for f in scene.iterdir()
                         if f.suffix.lower() in {".jpg",".jpeg",".png"}], key=nkey)
        if len(frames) not in (7, 9): continue
        for src in frames:
            fix_and_resize(src, part_out / scene.name / src.name)
        for ext in (".JPG",".jpg",".jpeg",".png"):
            src_gt = label_in / (scene.name + ext)
            if src_gt.exists():
                fix_and_resize(src_gt, part_out / "Label" / src_gt.name); break
print("Done")
EOF
4.3 构建 Manifest

python scripts/build_manifest.py sequence_pool \--input-root "SICE_512/Dataset_Part1/Dataset_Part1" \
  --gt-root    "SICE_512/Dataset_Part1/Dataset_Part1/Label" \
  --output     "data/manifests/sice_train.jsonl"
# 输出：Wrote 230 records

python scripts/build_manifest.py sequence_pool \
  --input-root "SICE_512/Dataset_Part2/Dataset_Part2" \
  --gt-root    "SICE_512/Dataset_Part2/Dataset_Part2/Label" \
  --output     "data/manifests/sice_test.jsonl"
# 输出：Wrote 210 records
五、Kill Test 1 — Oracle Headroom
5.1 问题
对每个场景，adaptive 选择的最大可能收益（oracle greedy）相比固定 bracket（±2EV）有多大？

5.2 方法
对测试集中每个场景构造 SceneEvaluator，以PSNR（对比GT Label）为质量指标：

oracle_greedy(budget)：每步选 marginal gain 最大的候选 EV，贪心序列
oracle_sequence(budget)：穷举所有 budget 大小的EV 子集，返回全局最优
best_fixed：config 中预先设定的固定 bracket（不从训练集搜索，保证 baseline 真实）
strong_heuristic：基于 histogram coverage 的启发式策略
random：随机选择，5次重复取均值
5.3 配置文件
configs/stage0_sice_mertens.yaml 关键参数：


dataset:
  test_manifest: data/manifests/sice_test.jsonl

fusion:
  type: linear_radiance      # 纯 NumPy，oracle对比有效

experiment:
  candidate_evs: [-3, -2, -1, 0, 1, 2, 3]
  base_ev: 0.0
  budgets: [1, 2, 3, 4]
  metric: psnr
  fixed_sets:                # 真正固定的 bracket，不经过训练集搜索
    2: [-1.0, 0.0]
    3: [-2.0, 0.0, 2.0]      # 标准±2EV，最重要的对比
    4: [-3.0, -1.0, 1.0, 3.0]
  enumerate_value_tensor: true
  value_tensor_max_subset_size: 2# 产生 Kill Test 2 所需监督数据
5.4 执行命令

python scripts/run_stage0.py \
  --config configs/stage0_sice_mertens.yaml \
  --output results/kill_test_sice
# 约 45 分钟（210场景 × ~13s/scene）
5.5 结果（210场景，Bootstrap 95% CI）
Oracle Greedy vs固定 ±2EV bracket：

budget	mean gain	95% CI	>0.5dB	>1.0dB
2	+0.73 dB	[0.65, 0.81]	54%	32%
3	+1.99 dB	[1.82, 2.16]	99%	78%
4	+1.54 dB	[1.44, 1.64]	91%	76%
结论：GO。 adaptive 在真实 SICE 数据上相比固定 bracket 有充足的可挖掘空间，几乎所有场景都能显著受益。

5.6 分析命令（utility structure）

python scripts/analyze_stage0.py --result-dir results/kill_test_sice
关键输出（utility structure，Kill Test 2 的前置验证）：


mean_best_action_change_rate:0.355# 最优动作随state 变化频率 35.5%
mean_unique_best_actions_per_scene: 3.3# 每场景平均 3.3 个不同的最优动作
diminishing_return_violation_ratio: 0.33# 违反递减收益的比例（接近 submodular）
best_action_change_rate=35.5% 是 Kill Test 2 可行性的重要前置信号——如果最优动作从不变化，预测 value 就没有意义。

六、Kill Test 3 — Greedy vs Sequence
6.1 问题
One-step greedy 是否等价于全局最优序列？是否需要 multi-step 规划？

6.2 方法
同Kill Test 1，在 run_stage0.py 中同时计算 oracle_greedy 和 oracle_sequence，输出 greedy_sequence_gap.csv。

6.3 结果
budget	mean gap	max gap	场景中 gap > 0.01dB
2	0.000 dB	0.000 dB	0%
3	0.0002 dB	0.046 dB	0.5%
4	0.0002 dB	0.033 dB	0.5%
结论：KEEP SIMPLE。 Greedy 在 99.5% 的场景与全局最优完全等价，剩余 0.5% 差距不超过 0.05dB，感知不可分辨。方法中不需要 World Model 或多步规划，one-step candidate valuation 即可。

七、Kill Test 2 — Value 可预测性（待完成）
7.1 问题
候选曝光 a 相对于当前 fusion state 的边际价值 ΔQ(a|C_t)，能否在candidate 被实际采集之前，从 (state, a) 预测？fusion state H_t 是否比 histogram 更适合作为 acquisition state？

7.2 已有的监督数据
Kill Test 1 运行时自动生成了：


results/kill_test_sice/oracle_value_tensor.jsonl
每行为一条样本，格式：


{
  "scene_id": "Dataset_Part2_1",
  "current": [-3.0, 0.0],// 当前已选EV 集合
  "action": 2.0,                  // 候选动作（EV 值）
  "score_before": 13.2,           // 融合前 PSNR
  "score_after": 14.8,            // 加入 action 后 PSNR
  "gain": 1.6     // ΔQ(a|C_t) = score_after - score_before
}
总行数约 210 × 90~230 = 15万–30万条（取决于场景帧数）。

7.3 实验设计
目标：比较三种 state representation 预测 ΔQ(a|C_t) 的能力

State Representation	特征构造	模型
Histogram	对current 中每帧计算 64-bin亮度直方图，concat；加action EV scalar	MLP 3层
Image Feature	用 frozen ResNet-18 提取 current 各帧特征，average pool；加 action EV scalar	MLP 3层
Fusion State H_t	用 LinearRadiance/FreeMEF encoder 获取 current 的 fusion state；加 action EV encoding	MLP 3层
划分：用 sice_train.jsonl 对应场景的 value tensor 训练，sice_test.jsonl 对应场景评估。

评估指标：


from scipy.stats import spearmanr
import numpy as np

# 对每个(scene_id, current) 分组，计算跨 action 的排名预测
# 指标 1：Spearman rank correlation（跨所有样本）
rho, _ = spearmanr(true_gains, predicted_gains)

# 指标 2：Top-1 candidate accuracy（预测的最佳 action 是否与oracle 一致）
top1_acc = mean(predicted_best_action == oracle_best_action for each state)

# 指标 3：Action regret（选predicted_best 而非 oracle_best 损失多少 dB）
regret = mean(oracle_gain - gain_of_predicted_best for each state)
7.4 GO/STOP 判断标准
结果	判断
H_t Spearman ρ > 0.60 且显著优于 Histogram（Δρ > 0.08）	GO — fusion state 有效
H_t与 Histogram 相近（Δρ < 0.03）	REFRAME — representation 需重新设计
所有方法 ρ < 0.30	STOP — value 不可预测
H_t > Histogram 但 top-1 accuracy < 50%	SIMPLIFY — scalar value 已不准，ECF 无必要
7.5 执行步骤（待完成）
Step 1：构建训练数据集


import json, numpy as np
from pathlib import Path

rows = []
for line in Path("results/kill_test_sice/oracle_value_tensor.jsonl").open():
    rows.append(json.loads(line))

# rows 中每条有: scene_id, current (list of EVs), action (float), gain (float)
# 按 scene_id 划分 train/test（对应 sice_train/sice_test manifest）
Step 2：提取各state representation 的特征


# Histogram baseline
def histogram_state(exposures, current_evs, n_bins=64):
    hists = []
    for ev in sorted(current_evs):
        lum = exposures[ev].mean(axis=2).flatten()
        h, _ = np.histogram(lum, bins=n_bins, range=(0,1))
        hists.append(h / h.sum())
    return np.concatenate(hists)   # shape: (n_current * n_bins,)

# Action encoding
def action_encoding(ev, all_evs):
    # One-hot or scalar
    return np.array([ev])
Step 3：训练 MLP predictor，评估三个指标

Step 4：对照 GO/STOP 标准给出结论

7.6 注意事项
训练时注意 按场景划分，不能按样本随机划分（同一场景的样本高度相关）
ΔQ 分布有长尾（多数gain 接近0，少数较大），建议同时评估 MSE 和排名指标，以排名指标为主
Histogram 基线可以不用图像，只用已有的 EV list和全局统计，速度更快；Image feature 需要加载图像
八、已知问题与注意事项
问题	说明	状态
SICE JPEG EXIF 旋转	原始大图EXIF 旋转标记与 cv2 不兼容，需用 piexif 修正后resize	✅ 已在SICE_512 修正
manifest.py 中 GT 方向修正	若 GT 与曝光帧维度转置，自动 rot90 修正	✅ 已修复
oracle_sequence 初始化	原代码 best_score 用1帧分数初始化，会在多帧比单帧差时返回错误结果	✅ 已修复
psnr_mu 静默回退	原代码 mu_psnr 别名实际调用线性 PSNR	✅ 已修复
Mertens uint8 量化	OpenCV Mertens 输入应为 float32，uint8 量化损失质量	✅ 已修复
make_pool重复normalize	normalize_hdr 在每个EV 重复调用，现改为每pool 只调用一次	✅ 已修复
fixed set fallback	EV 不匹配时不补齐，导致 budget 不足	✅ 已修复
random policy 只存均值	丢失方差信息，现每个repeat 单独存行	✅ 已修复
SICE 原图分辨率 17MP	直接加载极慢，必须先离线 resize 到 512px	✅ 已离线 resize
best_fixed_sets 过慢	在 train上穷举搜索最优 EV 组合，对大数据集耗时几小时	✅ 已改为 config 中直接设定 fixed_sets
九、结果文件说明

results/kill_test_sice/
├── per_scene.csv# 每场景每方法每budget的 PSNR 分数
├── summary.csv                # 按方法和budget 汇总的均值/标准差
├── greedy_sequence_gap.csv    # Kill Test 3：每场景的 greedy-sequence gap
├── oracle_action_distribution.csv   # oracle 第一步选择的 EV 分布
├── paired_bootstrap.csv       # 方法对比的 bootstrap 95% CI
├── oracle_value_tensor.jsonl  # Kill Test 2的训练数据（每条为一个state-action对）
├── quality_cost_auc.csv       # 各方法的 quality-cost AUC（analyze_stage0.py 生成）
├── diminishing_returns.csv    # submodularity 违反分析
├── utility_structure.json# value tensor 统计摘要
└── resolved_config.json       # 本次运行使用的完整配置
当前状态：Kill Test 1 ✅ Kill Test 3 ✅ Kill Test 2 ⏳（监督数据已就绪，等待训练 predictor）
