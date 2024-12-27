**他の言語のバージョン [英語](README.md) [中国語](README_zh.md)**

# Unique3D
Unique3D: 単一画像からの高品質かつ効率的な3Dメッシュ生成の公式実装。

[Kailu Wu](https://scholar.google.com/citations?user=VTU0gysAAAAJ&hl=zh-CN&oi=ao), [Fangfu Liu](https://liuff19.github.io/), Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, [Yueqi Duan](https://duanyueqi.github.io/), [Kaisheng Ma](https://group.iiis.tsinghua.edu.cn/~maks/)

## [論文](https://arxiv.org/abs/2405.20343) | [プロジェクトページ](https://wukailu.github.io/Unique3D/) | [Huggingfaceデモ](https://huggingface.co/spaces/Wuvin/Unique3D) | [オンラインデモ](https://www.aiuni.ai/)

* デモ推論速度: Gradioデモ > Huggingfaceデモ > Huggingfaceデモ2 > オンラインデモ

**Gradioデモが残念ながらハングアップしたり、非常に混雑している場合は、[aiuni.ai](https://www.aiuni.ai/)のオンラインデモを使用できます。これは無料で試すことができます（登録招待コードを取得するには、Discordに参加してください: https://discord.gg/aiuni）。ただし、オンラインデモはGradioデモとは少し異なり、推論速度が遅く、生成結果が安定していない可能性がありますが、素材の品質は良いです。**

<p align="center">
    <img src="assets/teaser_safe.jpg">
</p>

Unique3Dは、野生の単一画像から高忠実度および多様なテクスチャメッシュを30秒で生成します。

## より多くの機能

リポジトリはまだ構築中です。ご理解いただきありがとうございます。
- [x] 重みのアップロード。
- [x] ローカルGradioデモ。
- [ ] 詳細なチュートリアル。
- [x] Huggingfaceデモ。
- [ ] 詳細なローカルデモ。
- [x] Comfyuiサポート。
- [x] Windowsサポート。
- [ ] Dockerサポート。
- [ ] ノーマルでより安定した再構築。
- [ ] トレーニングコードのリリース。

## 推論の準備

### Linuxシステムセットアップ

Ubuntu 22.04.4 LTSおよびCUDA 12.1に適応。
```angular2html
conda create -n unique3d python=3.11
conda activate unique3d

pip install ninja
pip install diffusers==0.27.2

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.1/index.html

pip install -r requirements.txt
```

[oak-barry](https://github.com/oak-barry)は、[こちら](https://github.com/oak-barry/Unique3D)でtorch210+cu121の別のセットアップスクリプトを提供しています。

### Windowsセットアップ

* `jtydhr88`によるWindowsインストール方法に非常に感謝します！[issues/15](https://github.com/AiuniAI/Unique3D/issues/15)を参照してください。

[issues/15](https://github.com/AiuniAI/Unique3D/issues/15)によると、コマンドを実行するバッチスクリプトを実装したので、以下の手順に従ってください。
1. [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/?q=build+tools)からVisual Studio Build Toolsが必要になる場合があります。
2. conda envを作成し、アクティブにします。
   1. `conda create -n unique3d-py311 python=3.11`
   2. `conda activate unique3d-py311`
3. [triton whl](https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-2.1.0-cp311-cp311-win_amd64.whl)をダウンロードし、このプロジェクトに配置します。
4. **install_windows_win_py311_cu121.bat**を実行します。
5. onnxruntimeおよびonnxruntime-gpuのアンインストールを求められた場合は、yと回答します。
6. ドライバールートの下に**tmp\gradio**フォルダを作成します（例：F:\tmp\gradio）。
7. python app/gradio_local.py --port 7860

詳細は[issues/15](https://github.com/AiuniAI/Unique3D/issues/15)を参照してください。

### インタラクティブ推論：ローカルGradioデモを実行する

1. [huggingface spaces](https://huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt)または[Tsinghua Cloud Drive](https://cloud.tsinghua.edu.cn/d/319762ec478d46c8bdf7/)から重みをダウンロードし、`ckpt/*`に抽出します。
```
Unique3D
    ├──ckpt
        ├── controlnet-tile/
        ├── image2normal/
        ├── img2mvimg/
        ├── realesrgan-x4.onnx
        └── v1-inference.yaml
```

2. インタラクティブ推論をローカルで実行します。
```bash
python app/gradio_local.py --port 7860
```

## ComfyUIサポート

[jtydhr88](https://github.com/jtydhr88)からの[ComfyUI-Unique3D](https://github.com/jtydhr88/ComfyUI-Unique3D)の実装に感謝します！

## より良い結果を得るためのヒント

1. Unique3Dは入力画像の向きに敏感です。トレーニングデータの分布により、正面を向いた直交画像は常に良い再構築につながります。
2. 遮蔽のある画像は、4つのビューがオブジェクトを完全にカバーできないため、再構築が悪化します。遮蔽の少ない画像は、より良い結果につながります。
3. 可能な限り高解像度の画像を入力として使用してください。

## 謝辞

以下のリポジトリからコードを大量に借用しました。コードを共有してくれた著者に感謝します。
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Wonder3d](https://github.com/xxlong0/Wonder3D)
- [Zero123Plus](https://github.com/SUDO-AI-3D/zero123plus)
- [Continues Remeshing](https://github.com/Profactor/continuous-remeshing)
- [Depth from Normals](https://github.com/YertleTurtleGit/depth-from-normals)

## コラボレーション
私たちの使命は、3Dの概念を持つ4D生成モデルを作成することです。これは私たちの最初のステップであり、前途はまだ長いですが、私たちは自信を持っています。あらゆる形態の潜在的なコラボレーションを探求し、議論に参加することを心から歓迎します。<span style="color:red">**私たちと連絡を取りたい、またはパートナーシップを結びたい方は、メールでお気軽にお問い合わせください (wkl22@mails.tsinghua.edu.cn)**</span>。

- 最新情報を入手するには、Twitterをフォローしてください: https://x.com/aiuni_ai
- DiscordでAIGC 3D/4D生成コミュニティに参加してください: https://discord.gg/aiuni
- 研究協力については、ai@aiuni.aiまでご連絡ください。

## 引用

Unique3Dが役立つと思われる場合は、私たちのレポートを引用してください：
```bibtex
@misc{wu2024unique3d,
      title={Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image}, 
      author={Kailu Wu and Fangfu Liu and Zhihan Cai and Runjie Yan and Hanyang Wang and Yating Hu and Yueqi Duan and Kaisheng Ma},
      year={2024},
      eprint={2405.20343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
