# Docker setup

This docker setup is tested on Windows 10.

make sure you are under this directory yourworkspace/Unique3D/docker

Build docker image:

```
docker build -t unique3d -f Dockerfile .
```

Run docker image at the first time:

```
docker run -it --name unique3d -p 7860:7860 --gpus all unique3d python app.py
```

After first time:
```
docker start unique3d
docker exec unique3d python app.py
```

Stop the container:
```
docker stop unique3d
```

You can find the demo link showing in terminal, such as `https://94fc1ba77a08526e17.gradio.live/` or something similar else (it will be changed after each time to restart the container) to use the demo.

Some notes:
1. this docker build is using https://huggingface.co/spaces/Wuvin/Unique3D rather than this repo to clone the source.
2. the total built time might take more than one hour.
3. the total size of the built image will be more than 70GB.