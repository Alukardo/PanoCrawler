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

在项目根目录创建 `.env` 文件：

```
GOOGLE_API_KEY=你的Google_API_Key
```

## 使用

```bash
cd Panoramas
python main.py
```

`main.py` 中可配置搜索坐标列表 `locList` 和输出路径。

## 项目结构

```
Panoramas/
├── main.py              # 主入口
├── process.py           # 配对数据生成
├── unities.py           # 欧拉角→旋转矩阵工具
├── panorama/
│   ├── api.py           # Google Street View API
│   ├── search.py        # 全景图搜索与解析
│   └── download.py      # 瓦片下载与拼接
└── .env                 # API Key（不上传）
```

## 依赖

- `numpy`
- `Pillow`
- `requests`
- `pydantic`
- `python-dotenv`
