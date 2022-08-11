import sys
sys.path.append("./src")

import uvicorn
from src.configs import api_config

if __name__ == "__main__":
    uvicorn.run(
        'src.app:app',
        host=api_config["server_host"],
        port=api_config["server_port"],
        # access_log=True,
        # log_level=api_config["log_level"],
        # log_config=api_config["log_config"],
        # reload=True,
    )