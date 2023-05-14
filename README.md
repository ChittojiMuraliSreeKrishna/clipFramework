# clipFramework

## this is python framework which predicts the images, this uses python 3.8

### complete info is in https://github.com/openai/CLIP
#### Steps to install clip with flask and other required frameworks
1. install any version of python, i recomend going 1 version lower than the latest version 
2. link: https://www.python.org/downloads/ 
3. now install and set the environment variables
4. install anaconda and add all required config for you pc 
5. link: https://www.anaconda.com/download
6. now open the command prompt or other terminal based on your os
7. now create a conda env by following the below steps
```bash
conda create --name myenv python=3.8.16
conda activate myenv
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

! now clone my repository & check for a file named reqirements.txt
```bash
cd path_to\clipFramework
pip install -r requirements.txt
```
this will install all required packages for using clip and flask frameworks

#### to run the project
```bash
python app.py
```
this will run image search program using flask app, local server will started and you can upload images and search the images
