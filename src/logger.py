from torch.utils.tensorboard import SummaryWriter
import os
import json

from utils.debug import *

class Logger:
    def __init__(self, args) -> None:
        
        self.log_path = os.path.join(os.path.dirname(os.path.abspath('.')), 'results', 'tb_logs', args.name, args.unique_token)
        os.makedirs(self.log_path, exist_ok=True)
        self.writer = SummaryWriter(self.log_path)

        config_dict = vars(args).copy()
        for key in list(config_dict.keys()):
            if not (isinstance(config_dict[key], str) or isinstance(config_dict[key], int) or isinstance(config_dict[key], float)):
                config_dict.pop(key)
        config_str = json.dumps(config_dict, indent=4)
        with open(os.path.join(self.log_path, 'config.json'), 'w') as config_file:
                config_file.write(config_str)

    
    def add(self, name:str, value, t):
        self.writer.add_scalar(name, value, t)