# 问题1离散词级对齐与文本感受野追溯

**首次复现请先阅读 [REPRODUCE.md](REPRODUCE.md)**：包含克隆、固定依赖安装、附件1目录结构、OpenFace配置、冒烟运行及结果核验。仓库根目录就是原工作区的`q1/`；原始数据、模型、外部工具及本机输出不包含在Git仓库中。[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)记录的是原机器的100条运行结果，不代表新电脑已运行通过。

本项目从附件1的官方标签表和100条原始视频提取特征。序列位置是整句BERT的WordPiece位置，音视频依据真实时间区间池化到这些位置。不训练情感预测模型，不生成ASR替代文本，不使用附件2数值特征冒充自主提取结果。

## 方法与解释范围

官方文本先按空白建立单词单元，保留原文及字符区间。数字和连字符等只在送入对齐器的副本上规范化，同时保存对齐单元与原单词的对应。Fast tokenizer通过word_ids建立WordPiece映射。BERT整句编码，不在编码前截断到50；超过模型位置上限直接报错。

`R_raw[k,j] = max(0, 1 - cosine(h[k], h_without_word_j[k]))`。同一单词的所有WordPiece同时替换成MASK，一条样本的全部单词遮挡变体组成一个batch。`R_occ`是R_raw的行归一化；`R_word`是同一单词输出行的均值。它衡量遮挡引起的表示敏感度，**不等同于真实语义占比、情感重要性或因果贡献**。R_rollout只是注意力对照。零敏感度行不伪造成均匀贡献，质量检查会报错。

音频采用openSMILE eGeMAPSv02 LLD，通常25维，保留实际列名、窗口与帧移。其配置含多种内部窗长和内置平滑，不能宣称每个声学值严格只依赖一个25ms窗口。程序不额外做时序平滑。视觉采用OpenFace 2.2的AU、姿态、视线、置信度，默认静态AU；动态模式的逐视频归一化属于全局依赖。两者的帧级追溯是对帧描述量及池化操作的追溯，不是对原波形或图像像素的精确归因。

## 环境与外部工具

推荐Python 3.12。先按设备从PyTorch官方安装兼容的torch和torchaudio，再安装本目录requirements.txt。WhisperX版本需与torch/torchaudio兼容；只调用其alignment模块，不加载ASR、说话人分离模型。此实现用到的API为`load_align_model`及`align(..., interpolate_method='ignore')`。已安装环境的具体版本及模型参数哈希写入outputs/logs/environment.json。

固定版本和本项目仅使用WhisperX对齐模块的安装方式见[REPRODUCE.md](REPRODUCE.md)。`requirements.txt`保留通用版本范围；复现原运行优先使用`requirements-reproduce.txt`，并按指南分别安装torch/torchaudio和WhisperX。

OpenFace官方发布页：https://github.com/TadasBaltrusaitis/OpenFace/releases/tag/OpenFace_2.2.0 。Windows下载x64包，解压至tools目录，在config.yaml配置`vision.executable`。安装其模型资源与Microsoft VC++运行库（若本机缺少）。模型资源必须真实存在，不允许用全零假数据替代安装失败。

官方AU模式说明：https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units 。`static`会传入`-au_static`，`dynamic`不传。默认指定发布包自带的CLNF关键点模型`model/main_clnf_general.txt`，不依赖额外下载CEN模型；可用`vision.landmark_model`切换。程序将按解码顺序导出的图像目录交给FeatureExtraction，检查输出frame编号与图像数量，再绑定原视频真实PTS，**不使用OpenFace自行按帧率生成的timestamp**。中间分析图像默认JPEG质量95，以控制有限磁盘占用；可将`media.image_format`改为png使用无损帧。格式、质量均记录到encoder_info，原始视频不修改。

PyAV用于解码、读取真实PTS和16-kHz单声道重采样，依赖wheel自带的FFmpeg库。imageio-ffmpeg提供独立FFmpeg可执行文件，用于记录工具版本；无需修改系统PATH。音频按源PTS放入以0秒为起点的缓冲区，间隙保存coverage掩码，负时间预滚样本裁除。所有对齐时间相对于源视频片段。默认不长期保存PNG：视觉阶段一次只解码一条样本到临时ASCII路径，完成后释放，避免100条视频的无损帧耗尽磁盘；可视化按源视频PTS重新解码所需关键帧。openSMILE配置也复制到临时ASCII目录以适配Windows中文工作路径。

如果使用MFA，在独立conda环境安装montreal-forced-aligner，下载英文声学模型和词典，将`alignment.backend`改为`mfa`并填写可执行文件位置。MFA TextGrid不提供与WhisperX等价的单词置信分数，因此保存null，列入人工审核清单，不捏造置信度。

## 运行

所有路径和算法参数在config.yaml中，路径相对于配置文件所在目录。默认data_root指向本项目父目录的E题数据。

```powershell
# 使用本工作区已经建立的独立环境
./run_all.ps1
# 单步骤重跑
../.venv-q1/Scripts/python run.py --step s06b_text_receptive_field
# 从某一步开始，到某一步结束
../.venv-q1/Scripts/python run.py --from-step s09_11_align_pool_mask --to-step s15_export
# 只用前2个样本冒烟测试，写入outputs_smoke_2，绝不覆盖正式100条结果
../.venv-q1/Scripts/python run.py --limit 2
# 单步骤脚本同样可运行
../.venv-q1/Scripts/python src/s01_manifest.py
../.venv-q1/Scripts/python -m pytest tests -q
```

Linux/macOS使用`Q1_PYTHON=/path/to/python bash run_all.sh`。各阶段独立读取中间文件，修改上游配置后应按依赖顺序重跑下游阶段，不能把旧输出当作新配置的结果。每步失败会保留样本清单、记录原因并返回非零退出码；整流水线继续收集独立阶段的诊断，正式导出由质量门禁阻止。

## 主要参数

`text.layer=-1`选BERT最后一层；`text.revision`可固定模型提交。为满足严格的1e-5批次一致性检查，默认`stable_linear=true`将Linear实现为每词元相同形状的批量矩阵乘法（权重和数学表达式不变），`inference_pad_multiple=128`使单句与batch使用一致的padding宽度；仅返回有效位置，不把推理padding保存成真实token。这一数值实现也用于遮挡版本，避免batch相关的GEMM舍入差异污染微小敏感度。计算仍全部float32。

`alignment.min_score`仅用于人工复核提示，不删除低分词。对齐缺失的词保留文本表示，音视频mask为无效。`pooling.subword_time`支持shared/equal。`pooling.method`支持mean/mean_std；后者可重算特征，但标准差不是固定线性加权，所以通用trace_sample会拒绝假装用均值权重精确追溯标准差分支。

帧区间与词元区间有正长度交集才纳入；空时间窗可按配置扩展，并记录事件。窗口中已有帧但全部检测失败时不向外搜索、不复制上一帧。池化索引保存候选帧的闭区间，结合frame_valid排除中间无效帧；每位置精确的有效帧索引和均值权重还保存到alignment_records CSV中。

invalid_reason：0有效，1特殊token结构性无效，2padding，3没有有效对齐时间，4窗口内没有有效帧。CLS/SEP文本有效、音视频无效。零值本身不定义缺失；以掩码为准。

full补齐到全量最大WordPiece数量向上取整至8的倍数。存在超过50的位置时派生compat50，只裁剪输出行，保留全部单词列和整句上下文。没有超过50的样本时只输出full；其存储长度仍遵循8倍数规则，并不强行变成50。

## 输出与读取

- `meta/manifest.json, manifest.csv`：全量官方样本及媒体属性。
- `intermediate/`：各步骤float32中间文件、原始对齐输出、PNG帧、音频及有效覆盖掩码。
- `features/q1_features_full.pkl`：按字段集中存储，`data['text'][j]`访问第j条。
- `features/q1_features_compat50.pkl`：如有必要生成；R_occ列保留全部单词。
- `features/q1_frames.npz`：压缩帧特征、时间与有效掩码，按s000、s001等前缀组织；映射见export_report.json。
- `meta/summary_100.csv`、`alignment_records/`：全量质量和逐位置对应。
- `meta/text_mapping_issues.csv`、`low_confidence_alignment.csv`：需要人工核验的词。
- `meta/quality_report.json`：float16池化重算、文本重算、贡献矩阵和时间检查。
- `meta/receptive_field_agreement.csv`：R_occ与R_rollout逐行Spearman；梯度方法本次未启用。
- `figures/`：300-dpi单词敏感度热力图、时间投影、完整音视频对齐图。

默认PKL使用gzip压缩但保留需求指定的文件名；请使用加载函数，不要直接pickle.load压缩文件：

```python
from src.export import load_features
data = load_features('outputs/features/q1_features_full.pkl')
text = data['text'][0].astype('float32')
R = data['R_occ'][0].astype('float32')
```

数值计算用float32，特征与贡献矩阵导出为float16，整数索引int32。帧时间单独保留float32，避免较长片段的float16秒数误差。词语JSON和逐词CSV也保留高精度时间，可核验token_time的量化。

trace_utils.py四个函数把非负位置权重映射到单词或帧；无语音/无时间单词的重要性单独返回，不偷偷丢弃。trace_sample输入是单样本特征字段，并从帧文件添加`audio_frame_valid/audio_frame_times/vision_frame_valid/vision_frame_times`。相同词的多个子词映射到同一批帧时，重要性按位置相加，这是对给定位置质量的分配，不意味着独立观测次数增加。

## 质量和体积门禁

所有样本都写入summary_100.csv，失败原因不为空。未完成任何必需阶段时不生成伪造的正式特征。BERT单独与含padding的批次编码最大误差必须小于1e-5；重复遮挡、R有效行和、padding、有限性都检查。可选R_grad没有实现：文档中的向量输出梯度需要进一步规定完整Jacobian或标量化目标，不能用任意标量梯度冒充同一定义。

特征文件目标25MiB：先gzip与float16，超限依次删除R_raw、R_rollout，最后只保留compat50需要的帧前缀并明确标记frame_scope。若触发最后一项，full的全部帧重算需保留intermediate，不再声称压缩帧文件覆盖full所有时间。仍超限则报错，不假报达标。50MB竞赛总限制还需为后续模型与代码预留空间，不要把本地依赖、模型缓存、PNG中间帧打入提交包。

自动检查只验证实现和可追溯性，人工回听、表情核验及强制对齐真实误差仍需人工完成。代码不会声称这些人工验收已完成。
