# PanoCrawler

Google Street View 全景图采集与拼接工具。根据 GPS 坐标搜索并下载全景图，支持瓦片拼接、元数据获取、配额追踪、训练数据生成与浏览器端可视化。

## 功能

- **全景图搜索** — 根据经纬度搜索附近可用的 Street View 全景图
- **全景图下载** — 自动拼接 Google 瓦片图，自动处理黑边并按朝向旋转，输出完整 360° 全景图
- **配额追踪** — 持久化记录每日 Tile/Search/Session/Metadata 调用，软限触发时优雅停止
- **随机增量抓取** — 在多个全球区域内随机采样，按需补充新全景到 `info.csv`
- **Sequence-Aware 抓取** — 在每个搜索点按 `date` 分组，仅保留同次拍摄的 session（跨年份重复全景被丢弃），可选沿 anchor heading 步进扩链组成连续帧序列
- **元数据获取** — 通过 Google API 获取全景图拍摄日期、位置、姿态等信息，并落盘缓存避免重复消耗配额
- **训练数据生成** — 将多个全景图配对，生成 GAN/CycleGAN 风格的训练数据集
- **WebGL 可视化** — 在 3D 地球上展示已下载的搜索点

## 安装

```bash
pip install -r requirements.txt
```

## 配置

在项目根目录创建 `.apikey` 文件，用于存放私密配置：

```dotenv
GOOGLE_API_KEY=你的Google_API_Key
```

> 不要把 `GOOGLE_API_KEY` 写入 `config.yaml`，它应当只保存在 `.apikey` 文件中。

主要配置位于 `config.yaml`：

- `crawl_mode` — `random_incremental` / `random_search_sequence` / `fixed`（`fixed` 使用 `main.py` 中 `locList`）
- `random_crawl_target_new` / `random_crawl_max_searches` — 随机模式新增目标与最大搜索次数
- `sequence_crawl_target_new` / `sequence_crawl_max_searches` — Sequence 模式上限
- `sequence_walk_enabled` / `sequence_walk_bidirectional` / `sequence_max_length` / `sequence_step_meters` — 是否沿 heading 步进扩链、是否同时朝前后两侧走、最大序列长度、每步距离（米）；每步使用候选 pano 自身 heading，可跟随弯道
- `skip_low_pano_search` / `min_panos_per_search` — 搜索结果过少时跳过该点
- `images_root` / `images_dir` / `metadata_path` — 输出路径
- `google_api_daily_quota` / `google_api_daily_soft_limit` — 日配额上限
- `metadata_cache_path` / `metadata_cache_ttl_seconds` — Metadata API 缓存路径与 TTL
- `pair_selector_mode` — 训练对配对策略：`all_within_distance`（默认） / `same_sequence`（仅同 session 全两两） / `same_sequence_and_distance`（同 session 且通过距离阈值）

### CSV schema

`info.csv` 字段：`pano_id, lat, lon, heading, pitch, roll, date, search_point, search_point_id`。
旧文件缺少 `search_point_id` 时，加载会自动补 `"unknown"`，下次写入即升级到新 schema。Sequence 模式记录会写入共享的 `search_point_id`（anchor pano_id，与该搜索点一一对应）。

## 使用

```bash
python3 main.py
```

`main.py` 默认按 `crawl_mode: random_incremental` 运行；切换到 `fixed` 时使用文件内的 `locList`，切换到 `random_search_sequence` 时启用同 session 分组 + heading 步进的连续帧抓取。
所有输出路径通过 `config.yaml` 配置，配额状态写入 `runtime/google_api_usage.json`，Metadata 缓存写入 `runtime/pano_meta_cache.json`。

### Sequence 诊断 CLI

`integration/sequence_audit.py` 按 `search_point_id` 离线核对 `info.csv`，输出每个序列的长度、平均步距、最大步距、断裂数（默认阈值 30 m）：

```bash
python -m integration.sequence_audit                     # 默认读 config.metadata_path
python -m integration.sequence_audit --gap-threshold-meters 50
python -m integration.sequence_audit --json              # 机读 JSON 报告
```

### 可视化（Visualization）

`Visualization/app.js` 通过 `fetch()` 读取 `../runtime/info.csv` 与 `../config.yaml`，因此必须从**项目根目录**启动一个静态 HTTP 服务器（直接 `file://` 打开 `index.html` 浏览器会拦截跨源 fetch）：

```bash
# 在项目根目录运行
python3 -m http.server 8765
# 或使用 venv 中的 python
.venv/bin/python -m http.server 8765
```

然后浏览器打开 `http://127.0.0.1:8765/Visualization/`。

地球上只渲染搜索点。hover 时显示该搜索点的 `search_point_id`（即 anchor pano_id）作为 ID；CSV 中同一搜索点的所有下载点共享该 ID，点击后右侧面板列出完整下载点。

## 项目结构

```text
PanoCrawler/
├── main.py                       # 全景搜索、下载和元数据写入入口
├── build_training_pairs.py       # 训练配对数据生成入口
├── config.yaml                   # 路径、下载、处理、配额参数
├── panorama/
│   ├── api.py                    # Google Street View Metadata / Static API
│   ├── meta_cache.py             # Metadata 调用结果落盘缓存（计入每日配额）
│   ├── search.py                 # 全景图搜索与解析
│   ├── download.py               # 瓦片下载、拼接、Maps Tile API 会话
│   ├── process_images.py         # 黑边检测与裁剪
│   ├── quality.py                # 朝向旋转与质量度量
│   ├── quota.py                  # 每日 API 调用计数与软限拦截
│   ├── geometry.py               # 欧拉角→旋转矩阵等几何工具
│   └── config.py                 # YAML / .apikey 加载
├── integration/
│   ├── build_quality_dataset.py  # 质量评估数据集生成
│   ├── sequence_audit.py         # Sequence 完整性诊断 CLI
│   └── panoid_download.py        # 手动 pano_id 下载脚本
├── Visualization/                # 本地 WebGL 可视化（ECharts-GL）
├── tests/test_panorama.py        # 单元测试
├── runtime/                      # 配额状态与本地软链
├── images/                       # 备用元数据与图片目录
└── .apikey                       # API Key（不上传）
```

## 测试

```bash
.venv/bin/python -m pytest -q
```

## 依赖

- `numpy`
- `Pillow`
- `requests`
- `pydantic`
- `python-dotenv`
- `pyyaml`
- `urllib3`
