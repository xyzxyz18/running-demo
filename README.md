# 跑步姿态分析 Demo

上传跑步视频，在浏览器中查看二维/三维骨架、脚踝周期轨迹、关节角度和 PDF 报告。支持实时摄像头与历史重新分析，使用 RTMPose + VideoPose3D。

## 启动

建议 Python 3.10 / 3.11，系统安装 FFmpeg（macOS：`brew install ffmpeg`）。在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

启动后自动打开 <http://127.0.0.1:8000>。服务器启动可加 `--no-browser`。首次分析会下载模型。

## 目录

```text
跑步demo/
├── app.py                 # 唯一的 Web 启动入口
├── pace/                  # 应用代码与网页
│   ├── web.py             # Web 接口、任务与历史
│   ├── pipeline.py        # 视频分析流程和命令行入口
│   ├── config.py          # 分析参数
│   ├── paths.py           # 数据路径（含 PACE_DATA_DIR）
│   ├── pose/              # 二维检测、三维估计、平面约束
│   ├── biomechanics/      # 关节角度、步态事件
│   ├── analysis/          # 指标、稳定性、视角校正
│   ├── visualization/     # 图表、视频叠加、PDF
│   ├── static/            # JavaScript 与 CSS
│   └── templates/         # HTML
├── tools/                 # 可选的标定与对照实验工具
├── tests/                 # 回归测试
├── docs/                  # 详细说明、历史方案与参考图片
├── output/                # 本地数据与模型，不上传 GitHub
├── requirements.txt       # Python 依赖
├── Dockerfile             # 服务镜像
├── compose.yaml           # 服务与持久化数据卷
└── deploy.sh              # Linux 部署入口
```

整理文件时：业务功能放 `pace/`，独立实验放 `tools/`，说明和参考资料放 `docs/`，运行生成的内容放 `output/`。保持根目录只有入口、说明及项目配置。

## 常用命令

视频分析：

```bash
python -m pace.pipeline input.mp4 --output output/my-run
```

可选实验工具（均在项目根目录运行）：

```bash
python -m tools.calibrate --input input.mp4
python -m tools.run_pipeline --input input.mp4 --output output/experiments/my-run
python -m tools.plane_case --input output/my-run/skeleton3d.npz --output output/leg-plane-case
```

原根目录的 `main.py`、`calibrate.py`、`run_pipeline.py`、`plane_case.py` 已迁入对应模块，使用以上命令。Web 启动仍为 `python app.py`，Gunicorn 入口仍为 `app:app`。

回归测试（使用安装了项目依赖的环境）：

```bash
python -m unittest discover -s tests -t . -v
```

## 数据与 GitHub

`output/jobs/` 保存上传视频及报告，`output/models/` 保存三维模型，`output/experiments/` 保存实验结果。已有数据保留原路径。设置 `PACE_DATA_DIR` 可更改数据根目录。

`.gitignore` 已排除 `output/`、`outputs/`、虚拟环境、缓存和本地 `.env`。提交源码时包含 `pace/`、`tools/`、`tests/`、`docs/` 和根目录配置。请将后续生成结果也写入 `output/`。

## 部署与详细说明

Linux 服务器中执行 `./deploy.sh`；需要修改端口时先 `cp .env.example .env`。Docker 使用数据卷持久化历史视频和报告。

模型、拍摄建议、标定、输出文件和部署细节见 [详细使用说明](docs/usage.md)。旧方案仅供追溯，放在 [docs/archive/](docs/archive/README.md)。

三维模型的第三方代码与许可证保留在 `pace/pose/vendor/`。分析结果仅供运动观察，单目三维估计不代表真实米制测量。
