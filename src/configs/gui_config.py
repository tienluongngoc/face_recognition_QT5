from .model_config import ModelConfig



class GUIConfig(ModelConfig):
    def __init__(self, config_path) -> None:
           super(GUIConfig).__init__(config_path)
    
    @property
    def preview_face_with(self):
        width = self.config[""]

    
