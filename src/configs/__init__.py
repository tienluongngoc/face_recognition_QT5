from .face_recog_api_config import FaceRecogAPIConfig
from .scrfd_config import SCRFDConfig
from .arcface_config import ArcFaceConfig
from .fasnet_config import FASNetConfig
from .faiss_config import FaissConfig

all_config_path = "configs/face_recog_api.yaml"
all_config = FaceRecogAPIConfig(config_path=all_config_path)

mongodb_config = all_config.mongodb
api_config = all_config.api
model_server_config = all_config.model_server
faces_config = all_config.faces