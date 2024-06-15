@echo off

set "triton_whl=%~dp0\triton-2.1.0-cp311-cp311-win_amd64.whl"

echo Starting to install Unique3D...

echo Installing torch, xformers, etc

pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121 

echo Installing triton

pip install "%triton_whl%"

pip install Ninja

pip install diffusers==0.27.2

pip install grpcio werkzeug tensorboard-data-server

pip install -r requirements-win-py311-cu121.txt

echo Removing default onnxruntime and onnxruntime-gpu

pip uninstall onnxruntime
pip uninstall onnxruntime-gpu

echo Installing correct version onnxruntime-gpu for cuda 12.1

pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

echo Install Finished. Press any key to continue...

pause