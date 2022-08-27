from .model_config import ModelConfig



class UIConfig(ModelConfig):
    def __init__(self, config_path) -> None:
           super(UIConfig, self).__init__(config_path)
    
    @property
    def preview_face_with(self):
        value = self.config["management"]["preview_face"]["img_with"]
        return value


    @property
    def preview_face_height(self):
        value = self.config["management"]["preview_face"]["img_height"]
        return value

    
