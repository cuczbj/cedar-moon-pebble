# 同学复现指南

本仓库包含第一问的Step1–16代码、配置、测试和文本遮挡感受野实现。此前已在Windows、Python 3.12.14、RTX 5060 Ti 16GB上运行100条样本。新电脑的安装与全量运行仍需按以下步骤验证；不同硬件不保证逐位一致，以质量门禁及模型哈希为准。

## 1. 克隆与原始数据

这是私有仓库，复现者需要仓库读取权限（仓库所有者在GitHub的Settings → Collaborators中邀请）。

```powershell
git clone https://github.com/cuczbj/q1-multimodal-alignment.git
cd q1-multimodal-alignment
```

将比赛附件1原样解压。默认布局如下，label表与video_id文件夹必须同级：

```text
工作目录/
├─ q1-multimodal-alignment/     # 仓库根目录
│  ├─ run.py
│  ├─ config.yaml
│  └─ tools/
└─ E题数据/
   └─ ...附件1目录.../
      ├─ label-100.xlsx
      ├─ video_id_1/
      │  ├─ 0.mp4
      │  └─ 1.mp4
      └─ video_id_2/
```

也可复制`config.yaml`为`config.local.yaml`，修改`data_root`为自己的数据路径，运行时传入`--config config.local.yaml`。YAML里的Windows路径建议用正斜杠。路径相对于配置文件；data_root下应只找到一份`label-100.xlsx`。附件2不是第一问运行输入。

原始数据、预训练模型、OpenFace二进制、特征结果均未上传，需要分别准备或运行生成。预留数GB以上空间用于模型及中间文件。

## 2. Python环境（Windows）

使用独立Python 3.12环境，避免混用已有深度学习项目。以下命令在仓库根目录运行。

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
# NVIDIA GPU：与原机器相同的PyTorch CUDA 12.8构建，需要兼容驱动。
.venv/Scripts/python -m pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements-reproduce.txt
# 仅安装WhisperX包本身，所需alignment模块的依赖已经在上面安装。
.venv/Scripts/python -m pip install --no-deps whisperx==3.8.6
.venv/Scripts/python -m nltk.downloader -d models/nltk punkt_tab
```

没有NVIDIA GPU时，将torch安装命令的索引改为`https://download.pytorch.org/whl/cpu`，配置`device: cpu`；计算会较慢，该平台尚未进行100条端到端验证。不要同时执行两套torch安装命令。

`--no-deps`是原运行使用的**对齐模块专用安装方式**：只导入`whisperx.alignment`，不使用WhisperX的ASR、说话人分离或其命令行。WhisperX包级依赖检查可能提示其他功能所需依赖缺失；不要为消除提示自动升级torch或transformers。本指南不是完整WhisperX安装教程。

`requirements-reproduce.txt`记录已运行环境的直接依赖版本，不是包含所有传递依赖的完整锁文件。若包镜像找不到固定版本，先核对镜像和Python版本；替换版本后需要重新检查结果并记录差异。

检查对齐模块与设备：

```powershell
.venv/Scripts/python -c "import torch, torchaudio, opensmile, transformers; from whisperx.alignment import load_align_model, align; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

BERT与强制对齐模型会在首次运行时下载至`models/`，需要能访问Hugging Face和PyTorch模型下载地址。BERT revision已固定为原机器缓存使用的提交`86b5e0934494bd15c9632b12f734a8a67f723594`。模型参数哈希基准见`reproduction_baseline.json`。

## 3. OpenFace

从[OpenFace 2.2官方发布页](https://github.com/TadasBaltrusaitis/OpenFace/releases/tag/OpenFace_2.2.0)下载Windows x64发布包，完整解压为：

```text
tools/OpenFace_2.2.0_win_x64/FeatureExtraction.exe
tools/OpenFace_2.2.0_win_x64/model/main_clnf_general.txt
```

保留发布包中的DLL及其余模型文件，不要只复制exe。若目录不同，修改`vision.executable`。本配置使用发布包附带的CLNF关键点模型和静态AU；不需要改为CEN。出现DLL错误时按官方说明安装所需Microsoft VC++运行库。

Linux需自行安装或构建OpenFace并配置可执行文件，之后使用`bash run_all.sh`；该路径尚未在本项目进行完整验证。

## 4. 先检查，再跑全量

```powershell
# 16项测试，不需要真实视频、预训练权重或OpenFace。
.venv/Scripts/python -m pytest tests -q
# 只检查数据清单和真实媒体时间轴，不下载模型。
.venv/Scripts/python run.py --limit 2 --to-step s02_probe_media
# 2条样本完整冒烟；首次需要下载模型。
.venv/Scripts/python run.py --limit 2
# 冒烟通过后运行全部100条。
.venv/Scripts/python run.py
```

自定义配置时给每条`run.py`命令加`--config config.local.yaml`。也可用`./run_all.ps1`启动；脚本优先使用`Q1_PYTHON`，其次仓库内`.venv`，然后兼容原工作区的`../.venv-q1`，最后系统python。

冒烟输出在`outputs_smoke_2/`，正式输出在`outputs/`。失败会以非零退出码结束；检查`logs/last_run.json`、阶段日志及`meta/quality_report.json`，不要把失败产生的中间文件当成正式结果。修改上游配置后，按依赖顺序重跑后续步骤。

## 5. 核对结果

原机器基准：

| 项目 | 结果 |
| --- | --- |
| 样本数 | 100，任何失败样本仍保留 |
| full text/audio/vision | 100×88×768 / 100×88×25 / 100×88×50 |
| compat50 text/audio/vision | 100×50×768 / 100×50×25 / 100×50×50 |
| 超过50位置样本 | 6 |
| 有语音词/低置信度词 | 1926 / 280 |
| 无有效池化视觉位置的样本 | 5，保留零值与无效mask |
| 单元及集成测试 | 16通过 |

全文结果与人工复核边界见`IMPLEMENTATION_STATUS.md`。语音/视觉维度与附件2的74/35不同；compat50只统一序列长度，不代表特征定义相同。

```python
from src.export import load_features

data = load_features('outputs/features/q1_features_full.pkl')
print(data['text'].shape, data['audio'].shape, data['vision'].shape)
print(data['sample_ids'][0], data['valid_length'][0])
```

PKL为gzip压缩，必须使用上述加载函数或gzip.open。还应查看`meta/summary_100.csv`、`meta/quality_report.json`、`meta/low_confidence_alignment.csv`、`logs/environment.json`以及三张可视化图。自动检查不替代人工回听和人脸核验，遮挡热力图也不是情感贡献的真值。

复现交流时请一并提供Python版本、torch/CUDA版本、代码提交号和失败阶段日志；分享日志前检查其中的本机路径。
