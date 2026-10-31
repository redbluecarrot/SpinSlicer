<p align="center">
  <img src="assets/spinslicer.svg" alt="SpinSlicer" width="160">
</p>

<h1 align="center">SpinSlicer</h1>

<p align="center">
  用于断层扫描式体积增材制造（VAM）的本地软件。
</p>

<p align="center">
  <a href="README.md"><img src="assets/lang-en.svg" alt="English" height="36"></a>
  <a href="README.ru.md"><img src="assets/lang-ru.svg" alt="Русский" height="36"></a>
  <a href="README.es.md"><img src="assets/lang-es.svg" alt="Español" height="36"></a>
  <a href="README.zh-CN.md"><img src="assets/lang-zh.svg" alt="中文" height="36"></a>
</p>

SpinSlicer 为断层扫描式体积增材制造（VAM / CAL）准备打印：装有树脂的
样品瓶在投影仪前旋转，数百个角度的光叠加起来，使整个零件同时固化。SpinSlicer
根据 STL 计算这些投影帧，显示它们将固化出什么，并按计划的曝光时间精确地投射。

## 功能

版本 0.5.0 —— 详见[发布说明](docs/releases/v0.5.0.md)（英文和俄文）。

- **三个步骤：** *模型*（加载、摆放并检查 STL）、*光*（计算投影帧）、*打印*
  （投射到投影仪、保存 MP4 或 PNG）。
- **FastCAL 引擎：** 圆形样品瓶中所有角度在光学上相同，只需追踪一个角度的光线。
  模型包含折射、衰减、菲涅耳损耗、投影仪像素轮廓与光锥，以及每帧期间的旋转。
- **虚拟打印：** 将投影仪实际显示的帧重新换算为剂量，在 3D 中与模型并排显示固化结果，
  并给出吻合度（IoU）和曝光余量。
- **打印计划**（时间与圈数）和**剂量图**，其滑块即打印曝光。
- **可直接用于打印的投影输出：** 帧以 1:1 放置在投影仪原生屏幕上，每层一行，
  按时钟选帧（无漂移），实时亮度调节，剂量达标后自动停止。
- **在普通硬件上也流畅：** 计算在独立进程中运行，并使用所有 CPU 核心。
- **英文和俄文界面**，日间与夜间主题。

## 安装与软件包

### 预编译二进制包（GitHub Releases）

可直接从 [Releases](https://github.com/redbluecarrot/SpinSlicer/releases) 下载预编译程序：

- **Windows 10 / 11：**
  - **安装程序：** 下载并运行 `SpinSlicer-Setup-0.5.0.exe`
  - **便携版：** 下载 `SpinSlicer-windows-x64.zip`，解压后运行 `SpinSlicer.exe`
- **Arch Linux / Manjaro：**
  ```bash
  sudo pacman -U SpinSlicer-0.5.0-1-x86_64.pkg.tar.zst
  ```
- **Ubuntu / Debian / Mint：**
  ```bash
  sudo apt install ./SpinSlicer-0.5.0-amd64.deb
  ```
- **Fedora / RHEL / openSUSE：**
  ```bash
  sudo dnf install ./SpinSlicer-0.5.0-1.x86_64.rpm
  ```
- **通用 Linux 便携版：**
  ```bash
  tar -xzf SpinSlicer-linux-x64.tar.gz
  ./SpinSlicer
  ```

### 从源码搭建开发环境

SpinSlicer 支持 Python 3.11 和 3.12。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install -r requirements.txt -r requirements-dev.txt
python build_sirt.py
```

运行应用：

```bash
python SpinSlicer.py
```

## 带螺纹的 STL 模型

通过 **Load STL** 将螺母作为普通 STL 加载。在截面光栅化过程中，内孔和螺纹退刀槽
默认会被保留，即使启用了网格修复选项也是如此。请使螺距和螺纹深度在所选网格分辨率下
占据数个体素的宽度；任何基于体素的投影器都无法可靠表示过小的特征。

## 投影后端

投影引擎遵循与 Tomo 和 VAMToolbox 相同的概念阶段：

1. 对目标几何体进行体素化；
2. 在等间隔角度上生成或优化投影序列；
3. 应用统一的投影缩放，并导出用于播放的帧序列。

界面默认后端会追踪经小瓶或比色皿并按斯涅尔定律折射的光线，对体素穿越和
比尔–朗伯衰减进行积分，然后用 SIRT 拟合非负的相对剂量投影。可选的原生库使用
C++/OpenMP 加速同一计算。Python API 还保留了内部 Radon 路径和可选的 VAMToolbox 兼容路径。

预览会根据导出的 PNG 灰度值重建，并在拟合统一缩放后报告相对剂量形状 RMSE。
这不是经过校准的固化预测。菲涅耳反射、散射、材料固化响应和硬件绝对剂量都需要额外校准。
参见[模型与可复现检查](docs/validation/physical-optics-v0.3.1.md)。

方形比色皿在每个角度都使用等于外部对角线的固定投影视场。该跨度和物理像素间距会记录在
清单中；投影仪校准必须使用该跨度。参见[方形视场校正与角部覆盖检查](docs/validation/square-detector-field.md)。

## 输出格式

每次成功生成都会创建一个唯一的运行目录：

```text
output_frames/
└── run-20260903T120000Z-a1b2c3d4/
    ├── frame_0000.png
    ├── frame_0001.png
    ├── slice_meta.json
    ├── dose_stats.txt
    ├── reconstruction_dose.npy
    ├── reconstruction_target.npy
    ├── reconstruction_preview.png
    └── manifest.json
```

`manifest.json` 记录来源标识（当来源为 STL 时）、模型类型与参数、变换矩阵、投影后端、
树脂设置、帧尺寸以及 `complete: true`。在播放器使用某次运行之前，帧仓库会校验图像数量、
尺寸、元数据、字节预算和完成状态。包含 `frame_*.png` 的旧式扁平文件夹仍可读取。

## 项目结构

| 文件 | 用途 |
| --- | --- |
| `SpinSlicer.py` | 主窗口、语言选择器、共享状态栏和日志。 |
| `assets/spinslicer.svg` | 桌面界面和安装包使用的应用图标。 |
| `slicer_tab.py` | STL 加载、变换和投影生成界面。 |
| `nut_dialog.py` | 为保持 API 兼容而保留的旧版螺母对话框。 |
| `threaded_nut.py` | 为保持 API 兼容而保留的旧版参数化螺母生成器。 |
| `ui_panels.py` | 打印机/工艺设置和紧凑的模型变换控件。 |
| `slicing_engine.py` | 网格截面光栅化、投影规划和经验证的 PNG 导出。 |
| `optics/` | 斯涅尔光线、精确体素路径积分和非负相对剂量 SIRT。 |
| `vam_backend.py` | 可选的 VAMToolbox CAL 适配器和正弦图布局归一化。 |
| `video_tab.py` | 帧播放和 MP4 导出。 |
| `frame_io.py` | 经验证的帧集存储、元数据和清单。 |
| `profiles.py` | 随每次投影运行记录的经验证的设备与树脂设置。 |
| `validation.py` | 网格/资源预检和工作负载预算。 |
| `model_node.py` | 原始网格及对 GPU 友好的变换状态。 |
| `viewport.py` | PyVista/VTK 3D 视口。 |
| `workers.py` | 用于模型加载、生成和视频的后台 Qt 工作线程。 |
| `synthetic_shapes.py` | 标准数值验证体模。 |
| `tests/` | 几何、投影、存储、安全和验证测试。 |

## 验证与安全

生成之前，SpinSlicer 会检查网格有限性、三角形数量、边界、是否适配树脂槽、层数、
估算内存、帧尺寸和输出预算。耗时任务在 `QThread` 工作线程中执行并支持取消。
只有在所有 PNG 文件和元数据通过验证后，输出目录才会提交。本地路径通过 Qt 的桌面 API
打开，而不是通过 shell 命令。

## 开发检查

```bash
python -m pytest
python -m ruff check .
python -m mypy
```

GitHub Actions 会在 push 和 pull request 时运行相同的检查。

## 发布

要发布可下载的桌面版本，请更新 `pyproject.toml` 中的版本号，然后创建并推送对应的版本标签：

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

`Build release artifacts` 工作流会将产物发布到 [GitHub Releases 页面](https://github.com/redbluecarrot/SpinSlicer/releases)：

- `SpinSlicer-windows-x64.zip` — 包含 64 位 Windows 的 `SpinSlicer.exe`；
- `SpinSlicer-Setup-<version>.exe` — 按用户安装的 Windows 安装程序，带开始菜单和可选桌面快捷方式；
- `SpinSlicer.exe` — 独立的 Windows 可执行文件；
- `SpinSlicer-linux-x64.tar.gz` — 便携式 64 位 Linux 可执行文件；
- `SpinSlicer-<version>-amd64.deb` — Debian/Ubuntu 软件包；
- `SpinSlicer-<version>-1.x86_64.rpm` — 兼容 Fedora/RHEL 的软件包。

在 Linux 上，解压归档后运行 `chmod +x SpinSlicer && ./SpinSlicer`。
也可以使用 `sudo apt install ./SpinSlicer-<version>-amd64.deb`
或 `sudo dnf install ./SpinSlicer-<version>-1.x86_64.rpm` 安装原生软件包。
两种软件包都会将 `SpinSlicer` 添加到桌面应用菜单。Linux 构建面向 x86_64。

## 科学参考

- [Tomo — Print Preparation Software](https://opencal-org.readthedocs.io/en/stable/software/tomo/)
- [VAMToolbox 文档](https://vamtoolbox.readthedocs.io/en/latest/)
- [VAMToolbox 源码仓库](https://github.com/computed-axial-lithography/VAMToolbox)
- [Automatic Exposure Volumetric Additive Manufacturing](https://doi.org/10.1002/admt.202402168)

该 DOI 论文介绍了断层扫描式 VAM，以及通过实时散射光反馈实现的自动曝光控制。
SpinSlicer 目前仅实现投影生成和帧预览；自动曝光的硬件反馈超出了该桌面原型的范围。
