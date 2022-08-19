from inferences import ArcFace
from configs.face_recognition_config import FaceRecogAPIConfig


class FaceEncodeFactory:
    def __init__(self, config:FaceRecogAPIConfig) -> None:
        self.engine_name = config.encode_engine
        self.encode_config = config.encode

    def get_engine(self):
        if self.engine_name == "arcface":
            engine = ArcFace(self.encode_config)
        elif self.engine_name == "arcface_trt":
            from inferences.face_encode.arcface_trt import ArceFaceTRT
            engine = ArceFaceTRT(self.encode_config)
        return engine