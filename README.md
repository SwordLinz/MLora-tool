# Dataset Toolbox

从 [Kohya-MauveLinz](https://github.com/bmaltais/kohya_ss) 独立提取的数据集工具集，包含 5 个 Gradio Tab，不依赖任何训练主流程。

## 功能

| Tab | 说明 |
|---|---|
| **Dataset Tag Manager** | 三栏式标签管理：图片缩略图 + 单图 tag 编辑（增删/去重/Google 翻译）+ 全局 tag 统计与过滤；支持批量添加/移除 tag（`.txt` / `.caption`） |
| **Batch Crop** | BIRME 风格批量裁剪缩放：目标尺寸/比例、基于边缘能量的智能焦点裁剪（可选 OpenCV 加速）、手动焦点微调、自定义百分比区域、PNG 重命名 |
| **Single Crop** | 单图交互式裁剪：编辑器内拖拽裁剪/缩放，可选比例居中裁剪、裁后缩放（拉伸/裁剪补齐/留白填充）、透明背景拍平，输出 png/jpg |
| **Video to Images** | 视频批量抽帧：按帧数/秒数/每秒张数三种模式，输出 png/jpg，每视频一个短名+哈希子文件夹（规避 Windows MAX_PATH） |
| **RunningHub batch** | 对文件夹内每张图调用 [RunningHub](https://www.runninghub.cn) 云端 ComfyUI 工作流，批量生图并下载到本地 |

## 安装

需要 Python 3.10+（推荐 3.11）。

```bat
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## 启动

```bat
.venv\Scripts\python main.py
```

或直接双击 `start.bat`。浏览器打开 http://127.0.0.1:7860。

可选参数：

```bat
python main.py --listen --port 7861   # 局域网访问 + 自定义端口
python main.py --config path\to\config.toml
python main.py --headless             # 隐藏 📂/运行按钮（自动化环境）
```

## 配置（可选）

复制 `config.example.toml` 为 `config.toml` 可预置各 Tab 的默认目录；没有 config.toml 也能正常运行（默认使用项目下 `data\` 与 `outputs\`）。

```toml
[utilities]
dataset_tag_manager_dir = "D:/datasets/my_lora"
batch_crop_input = "D:/datasets/raw"
batch_crop_output = "D:/datasets/cropped"
single_crop_output = "D:/datasets/cropped"
video_extract_input = "D:/videos"
video_extract_output = "D:/datasets/frames"
runninghub_input = "D:/datasets/raw"
runninghub_output_dir = "D:/datasets/generated"
```

RunningHub Tab 的 API Key、节点 ID 等设置保存在 `~/.runninghub_batch_gui.json`（页面上的"保存设置"按钮写入）。

## RunningHub batch 使用说明

1. 在 RunningHub 开通 API，并在网页上至少成功运行过该工作流一次；
2. 填入 API Key、Workflow ID 及 Load Image / 保存 / KSampler 种子三个节点 ID（与网页节点右上角 ID 一致）；
3. 点击"开始批量处理"，日志区实时显示上传、排队、轮询与下载进度。

命令行等价入口（不需要 GUI）：

```bat
.venv\Scripts\python -m app.runninghub_batch_workflow --api-key XXX --input-dir D:\imgs --output-dir D:\out
```

## 目录结构

```
main.py                        # 入口
app/
  common_gui.py                # 精简共享工具（scriptdir / 文件夹选择对话框）
  custom_logging.py            # rich 日志（原样提取）
  class_gui_config.py          # TOML 配置读取（原样提取）
  dataset_tag_manager_gui.py   # Tab 1（原样提取）
  batch_crop_gui.py            # Tab 2（原样提取）
  single_crop_gui.py           # Tab 3（单图交互式裁剪）
  video_extract_gui.py         # Tab 4（原样提取）
  runninghub_batch_gui.py      # Tab 4 GUI（仅 import 方式调整）
  runninghub_batch_workflow.py # Tab 4 后端 + CLI（原样提取）
tests/
```

## 依赖说明

- `opencv-python`：视频解码 + 智能裁剪的 Sobel 边缘检测（缺失时智能裁剪退化为居中裁剪）
- `deep-translator`：标签翻译（缺失时翻译按钮提示安装）
- 其余为 gradio / PIL / numpy / requests / rich / toml
