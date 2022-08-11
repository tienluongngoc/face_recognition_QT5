# **Face Recognition - Registration (FRR) API**



## **Getting started**
### Prerequisites
#### 1. MongoDB server
 - Example create mongodb from docker 
```
$ docker run --name frr_database -p 27017:27017 -d mongo:latest
```
#### 2. Model server (optional)
 - Refer to [this repo](https://github.com/khai9xht/model_server)
### Clone repo and create enviroment
```
$ git clone https://gitlab.com/edsolabsrnd/aip/ai-core/face_recognition_api
$ cd face_recognition_api
$ pip install -r requirements.txt
```
### Set up **Mongodb** and **Model Server** (optional)
- fill IP, Port, Host, User, Pass, ... of Mongodb or Model Server to [configs/config.yaml](./configs/config.yaml). If you don't want to use model server, change running = false in config and put all models (ArcFace and SCRFD) in [weights](weights) folder.
### Run APi
- Set up IP, Port of API in [configs/config.yaml](./configs/config.yaml)
- Run this command line to start API with uvicorn
```
$ python main.py
```  