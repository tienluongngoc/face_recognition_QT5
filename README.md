# **Face Recognition App**
# Update soon


# Setup for GPU (Windows)
- Install Mini anaconda (or Anaconda)
- Create environment 
```
conda create -n {env_name} python=3.9
```
- install prerequisites
```
conda activate {env_name}
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements\gpu.txt
conda install -c conda-forge faiss-gpu
```

# How to run
- [Download model](https://drive.google.com/drive/folders/1vwxtT9rPRu3XvZSsA5C9bBa1fenbjVPJ?usp=sharing) and put into weights
- Use default config, mofify the host, post of MongoDB
```
conda activate {env_name}
python main.py
```

# notes
- Cài đặt Faiss: conda install faiss-cpu -c pytorch --> CPU
- Opencv cài đặt bản opencv-python==4.5.3.56 trên windows để tương thích với pyinstaller, Ubuntu thì cài bản opencv-python-headless để
không bị conlict với PyQt5
- 