import os

import DREAMPlace.dreamplace.Placer as Placer
import DREAMPlace.dreamplace.Params as Params

from utils.debug import *

class DMPcaller:
    def __init__(self, args, benchmark, macro_placement_path) -> None:
        self.args = args

        json_file = os.path.join(
            '../config/dmp_json',
            f'{benchmark}.json'
        )
        self.params = Params.Params()
        self.params.load(json_file)

        self.def_input_path = os.path.join(macro_placement_path, f"{benchmark}.def")
        self.params.def_input = self.def_input_path
            
        self.params.gpu = 0 if args.device == 'cpu' else 1
        self.params.global_place_flag = 1
        self.params.legalize_flag = 1
        self.params.detaield_place_flag = 1
        self.params.detailed_place_engine = ""
        self.params.name = args.name
        self.params.unique_token = args.unique_token
        self.params.random_seed = args.seed
        self.params.benchmark = benchmark
        
        # control numpy multithreading
        os.environ["OMP_NUM_THREADS"] = "%d" % (self.params.num_threads)

        

    def call(self, placement_save_path, call_id, test_mode=False, optimization=True):
        self.params.t_env = self.args.t_env
        self.params.i_episode = self.args.i_episode
        self.params.test_mode = test_mode
        self.params.call_id = call_id
        self.params.placement_save_path = placement_save_path
        self.params.optimization = optimization
        self.params.def_input = self.def_input_path if optimization else self.args.dataset_path
        self.params.random_center_init_flag = 1 if optimization else 0
        metrics, placement_info = Placer.place(self.params)
        return metrics, placement_info
    
        