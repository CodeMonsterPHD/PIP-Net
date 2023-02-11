# PIP-Net

### The pytorch code of "Prototype-based Intent Perception"

![image](https://user-images.githubusercontent.com/54825614/210133264-7ee6b3d7-b67b-4cdc-a35b-4d7189e49688.png)

### 1. Environment

The project needs 1 NVIDIA 1080TI

- python=3.8
- pytorch=1.10.1
- opencv=4.5.5
- tensorboard=2.4.1
- torchvision=0.11.2
- matplotlib=3.6.0
- numpy=1.23.1
- pillow=9.2.0
- pyyaml=5.4.1
- scikit-learn=1.1.1

or you can install the environment by `conda env create -f environment.yml`

### 2. Data Preparation

Download dataset from Googledrive or [Baiduyun](https://pan.baidu.com/s/1bMc6tWke9sznzBlmYCYZ8Q?pwd=1234) to `$ROOT/data`

### 3. Train

```
python tools/main.py
```

### 4. Contact US

If you have some questions about this project, please contact me, my email is [wbl921129@gmail.com](mailto:wbl921129@gmail.com)
