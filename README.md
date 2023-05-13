# clipFramework

## this is python framework which predicts the images, this uses python 3.8

### complete info is in https://github.com/openai/CLIP
### Steps to install this framework
1. install pthon
2. install anaconda
3. now create a env
```bash
conda create --name my_env python=3.8.16
conda activate my_env
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

! now clone my repository check for file named reqirements.txt
```bash
cd path_to\clipFramework
pip install -r requirements.txt
```
this will install all required packages for using clip and flask frmaeworks
