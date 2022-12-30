# **Face Recognition App**

# How to run
Following this instructions you can run successfully in CPU, but GPU :((( so I will update soon  
- Install [Mini anaconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) (or [Anaconda](https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh))
- Create environment 
```
$ conda create -n env_name python=3.6
$ conda activate env_name
$ conda install -c fastai opencv-python-headless
$ conda install -c pytorch faiss-cpu
$ pip install -r requirements/cpu_only.txt
```
- Dowload model [here](https://drive.google.com/drive/folders/1RydA2wyWzVk2S53LYvvUeYxT5eo0ZE0u?usp=sharing) and put it into ./weights
- Init Mongo database
    -   You can install by apt
    - Or create a docker container (highly recommend)  
    EX:
    ```
    docker run -it --mongo_db -p 27017:27017 -e MONGO_INITDB_ROOT_USERNAME=face_admin -e MONGO_INITDB_ROOT_PASSWORD=123456 mongo:latest
    ```
- Last step:
```
python main.py
```