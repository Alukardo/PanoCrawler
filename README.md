# PanoramaData

Google Street View 全景图采集与拼接工具。根据 GPS 坐标搜索并下载全景图，支持瓦片拼接、元数据获取和训练数据生成。

## 功能

- **全景图搜索** — 根据经纬度搜索附近可用的 Street View 全景图
- **全景图下载** — 自动拼接 Google 瓦片图，输出完整 360° 全景图
- **元数据获取** — 通过 Google API 获取全景图拍摄日期、位置、姿态等信息
- **训练数据生成** — 将多个全景图配对，生成 GAN/CycleGAN 风格的训练数据集

## 安装

```bash
pip install -r requirements.txt
```

## 配置

在项目根目录创建 `.apikey` 文件，用于存放私密配置：

```
GOOGLE_API_KEY=你的Google_API_Key
```

> 不要把 `GOOGLE_API_KEY` 写入 `config.yaml`，它应当只保存在 `.apikey` 文件中。

## 使用

```bash
python3 main.py
```

`main.py` 中可配置搜索坐标列表 `locList`。输出路径统一在 `config.yaml` 中配置。

## 项目结构

```
Panoramas/
├── main.py              # 全景搜索、下载和元数据写入入口
├── process.py           # 训练配对数据生成入口
├── config.yaml          # 路径、下载、处理参数
├── unities.py           # 兼容旧导入的几何工具包装
├── panorama/
│   ├── api.py           # Google Street View API
│   ├── search.py        # 全景图搜索与解析
│   ├── download.py      # 瓦片下载与拼接
│   ├── process_images.py # 图片后处理
│   └── geometry.py      # 欧拉角→旋转矩阵等几何工具
├── tests/
│   ├── test_panorama.py # 单元测试
│   └── integration/     # 手动联网验证脚本
├── images/
│   ├── info.csv         # 元数据 CSV
│   └── pano/            # 全景图片
├── temp/                # 训练数据生成输出
└── .apikey              # API Key（不上传）
```

## 依赖

- `numpy`
- `Pillow`
- `requests`
- `pydantic`
- `python-dotenv`
