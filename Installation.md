# 官方安装指南

* 在 requirements-detail.txt 里，我们提供了详细的各个库的版本，这个对应的环境是 `python3.10 + cuda12.2`。
* 本项目依赖于几个重要的pypi包，这几个包安装起来会有一些困难。

### nvdiffrast 安装

* nvdiffrast 会在第一次运行时，编译对应的torch插件，这一步需要 ninja 及 cudatoolkit的支持。
* 因此需要先确保正确安装了 ninja 以及 cudatoolkit 并正确配置了 CUDA_HOME 环境变量。
* cudatoolkit 安装可以参考 [linux-cuda-installation-guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), [windows-cuda-installation-guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
* ninja 则可用直接 `pip install ninja`
* 然后设置 CUDA_HOME 变量为 cudatoolkit 的安装目录，如 `/usr/local/cuda`。
* 最后 `pip install nvdiffrast` 即可。
* 如果无法在目标服务器上安装 cudatoolkit （如权限不够），可用使用我修改的[预编译版本 nvdiffrast](https://github.com/wukailu/nvdiffrast-torch) 在另一台拥有 cudatoolkit 且环境相似（python, torch, cuda版本相同）的服务器上预编译后安装。

### onnxruntime-gpu 安装

* 注意，同时安装 `onnxruntime` 与 `onnxruntime-gpu` 可能导致最终程序无法运行在GPU，而运行在CPU，导致极慢的推理速度。
* [onnxruntime 官方安装指南](https://onnxruntime.ai/docs/install/#python-installs)
* TLDR: For cuda11.x, `pip install onnxruntime-gpu`. For cuda12.x, `pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
`.
* 进一步的，可用安装基于 tensorrt 的 onnxruntime，进一步加快推理速度。
* 注意：如果没有安装基于 tensorrt 的 onnxruntime，建议将 `https://github.com/AiuniAI/Unique3D/blob/4e1174c3896fee992ffc780d0ea813500401fae9/scripts/load_onnx.py#L4` 中 `TensorrtExecutionProvider` 删除。
* 对于 cuda12.x 可用使用如下命令快速安装带有tensorrt的onnxruntime (注意将 `/root/miniconda3/lib/python3.10/site-packages` 修改为你的python 对应路径，将 `/root/.bashrc` 改为你的用户下路径 `.bashrc` 路劲)
```
pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/
pip install onnxruntime-gpu==1.17.0 --index-url=https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install tensorrt==8.6.0
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/:/root/miniconda3/lib/python3.10/site-packages/tensorrt:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> /root/.bashrc
```

### pytorch3d 安装

* 根据 [pytorch3d 官方的安装建议](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-wheels-for-linux)，建议使用预编译版本
```
import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
!pip install fvcore iopath
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html
```

### torch_scatter 安装

* 在[torch_scatter 官方安装指南](https://github.com/rusty1s/pytorch_scatter?tab=readme-ov-file#installation) 使用预编译的安装包快速安装。
* 或者直接编译安装 `pip install git+https://github.com/rusty1s/pytorch_scatter.git`

### 其他安装

* 其他文件 `pip install -r requirements.txt` 即可。

-----

# Detailed Installation Guide

* In `requirements-detail.txt`, we provide detailed versions of all packages, which correspond to the environment of `python3.10 + cuda12.2`.
* This project relies on several important PyPI packages, which may be difficult to install.

### Installation of nvdiffrast

* nvdiffrast will compile the corresponding torch plugin the first time it runs, which requires support from ninja and cudatoolkit.
* Therefore, it is necessary to ensure that ninja and cudatoolkit are correctly installed and that the CUDA_HOME environment variable is properly configured.
* For the installation of cudatoolkit, you can refer to the [Linux CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [Windows CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).
* Ninja can be directly installed with `pip install ninja`.
* Then set the CUDA_HOME variable to the installation directory of cudatoolkit, such as `/usr/local/cuda`.
* Finally, `pip install nvdiffrast`.
* If you cannot install cudatoolkit on the computer (e.g., insufficient permissions), you can use my modified [pre-compiled version of nvdiffrast](https://github.com/wukailu/nvdiffrast-torch) to pre-compile on another computer that has cudatoolkit and a similar environment (same versions of python, torch, cuda) and then install the `.whl`.

### Installation of onnxruntime-gpu

* Note that installing both `onnxruntime` and `onnxruntime-gpu` may result in not running on the GPU but on the CPU, leading to extremely slow inference speed.
* [Official ONNX Runtime Installation Guide](https://onnxruntime.ai/docs/install/#python-installs)
* TLDR: For cuda11.x, `pip install onnxruntime-gpu`. For cuda12.x, `pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`.
* Furthermore, you can install onnxruntime based on tensorrt to further increase the inference speed.
* Note: If you do not correctly installed onnxruntime based on tensorrt, it is recommended to remove `TensorrtExecutionProvider` from `https://github.com/AiuniAI/Unique3D/blob/4e1174c3896fee992ffc780d0ea813500401fae9/scripts/load_onnx.py#L4`.
* For cuda12.x, you can quickly install onnxruntime with tensorrt using the following commands (note to change the path `/root/miniconda3/lib/python3.10/site-packages` to the corresponding path of your python, and change `/root/.bashrc` to the path of `.bashrc` under your user directory):
```
pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/ 
pip install onnxruntime-gpu==1.17.0 --index-url=https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple/ 
pip install tensorrt==8.6.0
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/:/root/miniconda3/lib/python3.10/site-packages/tensorrt:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> /root/.bashrc
```

### Installation of pytorch3d

* According to the [official installation recommendations of pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-wheels-for-linux), it is recommended to use the pre-compiled version:
```
import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
!pip install fvcore iopath
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html 
```

### Installation of torch_scatter

* Use the pre-compiled installation package according to the [official installation guide of torch_scatter](https://github.com/rusty1s/pytorch_scatter?tab=readme-ov-file#installation) for a quick installation.
* Alternatively, you can directly compile and install with `pip install git+https://github.com/rusty1s/pytorch_scatter.git`.

### Other Installations

* For other packages, simply `pip install -r requirements.txt`.

-----

# 官方インストールガイド

* `requirements-detail.txt` には、各ライブラリのバージョンが詳細に提供されており、これは Python 3.10 + CUDA 12.2 に対応する環境です。
* このプロジェクトは、いくつかの重要な PyPI パッケージに依存しており、これらのパッケージのインストールにはいくつかの困難が伴います。

### nvdiffrast のインストール

* nvdiffrast は、最初に実行するときに、torch プラグインの対応バージョンをコンパイルします。このステップには、ninja および cudatoolkit のサポートが必要です。
* したがって、ninja および cudatoolkit の正確なインストールと、CUDA_HOME 環境変数の正確な設定を確保する必要があります。
* cudatoolkit のインストールについては、[Linux CUDA インストールガイド](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)、[Windows CUDA インストールガイド](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) を参照してください。
* ninja は、直接 `pip install ninja` でインストールできます。
* 次に、CUDA_HOME 変数を cudatoolkit のインストールディレクトリに設定します。例えば、`/usr/local/cuda` のように。
* 最後に、`pip install nvdiffrast` を実行します。
* 目標サーバーで cudatoolkit をインストールできない場合（例えば、権限が不足している場合）、私の修正した[事前コンパイル済みバージョンの nvdiffrast](https://github.com/wukailu/nvdiffrast-torch)を使用できます。これは、cudatoolkit があり、環境が似ている（Python、torch、cudaのバージョンが同じ）別のサーバーで事前コンパイルしてからインストールすることができます。

### onnxruntime-gpu のインストール

* 注意：`onnxruntime` と `onnxruntime-gpu` を同時にインストールすると、最終的なプログラムが GPU 上で実行されず、CPU 上で実行される可能性があり、推論速度が非常に遅くなることがあります。
* [onnxruntime 公式インストールガイド](https://onnxruntime.ai/docs/install/#python-installs)
* TLDR: cuda11.x 用には、`pip install onnxruntime-gpu` を使用します。cuda12.x 用には、`pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/` を使用します。
* さらに、TensorRT ベースの onnxruntime をインストールして、推論速度をさらに向上させることができます。
* 注意：TensorRT ベースの onnxruntime がインストールされていない場合は、`https://github.com/AiuniAI/Unique3D/blob/4e1174c3896fee992ffc780d0ea813500401fae9/scripts/load_onnx.py#L4` の `TensorrtExecutionProvider` を削除することをお勧めします。
* cuda12.x の場合、次のコマンドを使用して迅速に TensorRT を備えた onnxruntime をインストールできます（`/root/miniconda3/lib/python3.10/site-packages` をあなたの Python に対応するパスに、`/root/.bashrc` をあなたのユーザーのパスの下の `.bashrc` に変更してください）。
```bash
pip install ort-nightly-gpu --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-12-nightly/pypi/simple/   
pip install onnxruntime-gpu==1.17.0 --index-url=https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-cuda-12/pypi/simple/   
pip install tensorrt==8.6.0
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/:/root/miniconda3/lib/python3.10/site-packages/tensorrt:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> /root/.bashrc
```

### pytorch3d のインストール

* [pytorch3d 公式のインストール提案](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#2-install-wheels-for-linux)に従い、事前コンパイル済みバージョンを使用することをお勧めします。
```python
import sys
import torch
pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
!pip install fvcore iopath
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html   
```

### torch_scatter のインストール

* [torch_scatter 公式インストールガイド](https://github.com/rusty1s/pytorch_scatter?tab=readme-ov-file#installation)に従い、事前コンパイル済みのインストールパッケージを使用して迅速インストールします。
* または、直接コンパイルしてインストールする `pip install git+https://github.com/rusty1s/pytorch_scatter.git` も可能です。

### その他のインストール

* その他のファイルについては、`pip install -r requirements.txt` を実行するだけです。

