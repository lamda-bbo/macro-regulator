import numpy as np
import os
import math

from utils.read_superblue import read_benchmark as read_superblue, read_def, get_scaling_ratio, get_inv_scaling_ratio, write_def
from utils.debug import *
from place_env.dmp_caller import DMPcaller
from itertools import combinations


class ProblemInstance():
    def __init__(self, args, benchmark):
        self.benchmark = benchmark
        benchmark_path = "../benchmark"
        self.args = args

        self.database = {}
        placedb_info = read_superblue(database=self.database, benchmark=os.path.join(benchmark_path, benchmark), args=args)
        self.node_info = placedb_info["node_info"]
        self.node_info_raw_id_name = placedb_info["node_info_raw_id_name"]
        self.node_cnt = placedb_info["node_cnt"]
        self.port_info = placedb_info["port_info"]
        self.net_info = placedb_info["net_info"]
        self.net_cnt = placedb_info["net_cnt"]
        self.max_height = placedb_info["max_height"]
        self.max_width = placedb_info["max_width"]
        self.standard_cell_name = placedb_info["standard_cell_name"]
        self.port_to_net_dict = placedb_info["port_to_net_dict"]
        self.cell_total_area = placedb_info["cell_total_area"]

        self.node_to_net_dict = get_node_to_net_dict(self.node_info, self.net_info)
        self.node_id_to_name = get_node_id_to_name_topology(self.node_info, self.node_to_net_dict, self.net_info, self.benchmark)
        
        self.max_net_per_node = 0
        for node in self.node_to_net_dict:
            self.max_net_per_node = max(self.max_net_per_node, len(self.node_to_net_dict[node]))

        self._compress_canvas()
        self.dmp_caller = None

        self.macro_pos = {}
        for node in self.node_info:
            raw_x = self.node_info[node]["raw_x"]
            raw_y = self.node_info[node]["raw_y"]
            pos_x = math.floor(max(0, (raw_x - self.ratio_x)/self.ratio_x))
            pos_y = math.floor(max(0, (raw_y - self.ratio_y)/self.ratio_y))
            size_x = math.ceil(max(1, self.node_info[node]['x']/self.ratio_x))
            size_y = math.ceil(max(1, self.node_info[node]['y']/self.ratio_y))
            self.macro_pos[node] = (pos_x, pos_y, size_x, size_y)

        
    def _compress_canvas(self):
        self.grid = self.args.grid
        self.ratio_x = self.max_width / self.grid
        self.ratio_y = self.max_height / self.grid

        self.place_grid = 0
        for self.place_grid in range(1, self.grid+1):
            if (self.place_grid * self.ratio_x) * (self.place_grid * self.ratio_y) > self.cell_total_area:
                break
        self.place_grid += int((self.grid - self.place_grid) // self.args.grid_soft_coeff)

        self.ratio_sum = self.ratio_x + self.ratio_y

    def init_dmp(self, macro_placement_path):
        if self.dmp_caller is None:
            self.dmp_caller = DMPcaller(args=self.args, 
                                        benchmark=self.benchmark, 
                                        macro_placement_path=macro_placement_path)
            self.macro_placement_path = macro_placement_path

    def call_dmp(self, call_id, placement_save_path, test_mode=False, optimization=True):
        return self.dmp_caller.call(placement_save_path=placement_save_path, 
                                    call_id=call_id, 
                                    test_mode=test_mode,
                                    optimization=optimization)
        
    def save_def(self, macro_pos=None, is_dataset=False):
        if macro_pos is None:
            macro_pos = self.macro_pos
        file_name = os.path.join(self.macro_placement_path, f'{self.benchmark}.def')
        inv_scale_ratio_x, inv_scale_ratio_y = get_inv_scaling_ratio(self.database)
        limited_x = round(round(self.place_grid * self.ratio_x + self.ratio_x) * inv_scale_ratio_x)
        limited_y = round(round(self.place_grid * self.ratio_y + self.ratio_y) * inv_scale_ratio_y)
        write_def(macro_pos=macro_pos, 
                  database=self.database, 
                  def_file=file_name, 
                  ratio_x=self.ratio_x,
                  ratio_y=self.ratio_y,
                  limited_x=limited_x,
                  limited_y=limited_y,
                  is_dataset=is_dataset)
        
    def set_mp_hpwl(self, mp_hpwl):
        self.mp_hpwl = mp_hpwl
    
    def set_gp_hpwl(self, gp_hpwl):
        self.gp_hpwl = gp_hpwl
    
    def set_regularity(self, regularity):
        self.regularity = regularity

    def set_macro_pos(self, dmp_placement_info):
        for i in range(len(dmp_placement_info['node_name'])):
            if "DREAMPlace" in dmp_placement_info['node_name'][i]:
                dmp_placement_info['node_name'][i] = dmp_placement_info['node_name'][i].split('.')[0]
        scaling_ratio_x, scaling_ratio_y = get_scaling_ratio(self.database)
        self.macro_pos.clear()
        for macro in self.node_info:
            idx = np.where(dmp_placement_info['node_name'] == macro)[0]
            pos_x = dmp_placement_info['x'][idx] * scaling_ratio_x
            pos_y = dmp_placement_info['y'][idx] * scaling_ratio_y
            pos_x = math.floor(max(0, (pos_x - self.ratio_x)/self.ratio_x))
            pos_y = math.floor(max(0, (pos_y - self.ratio_y)/self.ratio_y))
            size_x = math.ceil(max(1, self.node_info[macro]['x']/self.ratio_x))
            size_y = math.ceil(max(1, self.node_info[macro]['y']/self.ratio_y))
            self.macro_pos[macro] = (pos_x, pos_y, size_x, size_y)

    def read_macro_pos_from_def(self, def_path):
        database = {}
        database["nodes"] = {}
        database["macros"] = []
        read_def(database=database, def_file=def_path)
        
        scaling_ratio_x, scaling_ratio_y = get_scaling_ratio(self.database)
        for macro in self.macro_pos:
            raw_x = eval(database["nodes"][macro]['x']) * scaling_ratio_x
            raw_y = eval(database["nodes"][macro]['y']) * scaling_ratio_y

            pos_x = math.floor(max(0, (raw_x - self.ratio_x)/self.ratio_x))
            pos_y = math.floor(max(0, (raw_y - self.ratio_y)/self.ratio_y))
            size_x = math.ceil(max(1, self.node_info[macro]['x']/self.ratio_x))
            size_y = math.ceil(max(1, self.node_info[macro]['y']/self.ratio_y))
            self.macro_pos[macro] = (pos_x, pos_y, size_x, size_y)
        
    

def get_node_to_net_dict(node_info, net_info):
    node_to_net_dict = {}
    for node_name in node_info:
        node_to_net_dict[node_name] = set()
    for net_name in net_info:
        for node_name in net_info[net_name]["nodes"]:
            node_to_net_dict[node_name].add(net_name)
    return node_to_net_dict

def get_node_id_to_name_topology(node_info, node_to_net_dict, net_info, benchmark):
    node_id_to_name = []
    adjacency = {}

    for net_name in net_info:
        for node_name_1, node_name_2 in list(combinations(net_info[net_name]['nodes'],2)):
            if node_name_1 not in adjacency:
                adjacency[node_name_1] = set()
            if node_name_2 not in adjacency:
                adjacency[node_name_2] = set()
            adjacency[node_name_1].add(node_name_2)
            adjacency[node_name_2].add(node_name_1)

    visited_node = set()

    node_net_num = {}
    for node_name in node_info:
        node_net_num[node_name] = len(node_to_net_dict[node_name])

    node_net_num_fea= {}
    node_net_num_max = max(node_net_num.values())
    print("node_net_num_max", node_net_num_max)
    for node_name in node_info:
        node_net_num_fea[node_name] = node_net_num[node_name]/node_net_num_max
    
    node_area_fea = {}
    node_area_max_node = max(node_info, key = lambda x : node_info[x]['x'] * node_info[x]['y'])
    node_area_max = node_info[node_area_max_node]['x'] * node_info[node_area_max_node]['y']
    print("node_area_max = {}".format(node_area_max))
    for node_name in node_info:
        node_area_fea[node_name] = node_info[node_name]['x'] * node_info[node_name]['y'] / node_area_max
    
    if "V" in node_info:
        add_node = "V"
        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node]))
        node_net_num.pop(add_node)
    
    add_node = max(node_net_num, key = lambda v: node_net_num[v])
    visited_node.add(add_node)
    node_id_to_name.append((add_node, node_net_num[add_node]))
    node_net_num.pop(add_node)

    while len(node_id_to_name) < len(node_info):
        candidates = {}
        for node_name in visited_node:
            if node_name not in adjacency:
                continue
            for node_name_2 in adjacency[node_name]:
                if node_name_2 in visited_node:
                    continue
                if node_name_2 not in candidates:
                    candidates[node_name_2] = 0
                candidates[node_name_2] += 1
        for node_name in node_info:
            if node_name not in candidates and node_name not in visited_node:
                candidates[node_name] = 0
        if len(candidates) > 0:
            if benchmark != 'ariane':
                if benchmark == "bigblue3":
                    add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*100000 +\
                        node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)
                else:
                    add_node = max(candidates, key = lambda v: candidates[v]*1 + node_net_num[v]*1000 +\
                        node_info[v]['x']*node_info[v]['y'] * 1 +int(hash(v)%10000)*1e-6)
            else:
                add_node = max(candidates, key = lambda v: candidates[v]*30000 + node_net_num[v]*1000 +\
                    node_info[v]['x']*node_info[v]['y']*1 +int(hash(v)%10000)*1e-6)
        else:
            if benchmark != 'ariane':
                if benchmark == "bigblue3":
                    add_node = max(node_net_num, key = lambda v: node_net_num[v]*100000 + node_info[v]['x']*node_info[v]['y']*1)
                else:
                    add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1)
            else:
                add_node = max(node_net_num, key = lambda v: node_net_num[v]*1000 + node_info[v]['x']*node_info[v]['y']*1)

        visited_node.add(add_node)
        node_id_to_name.append((add_node, node_net_num[add_node])) 
        node_net_num.pop(add_node)
    for i, (node_name, _) in enumerate(node_id_to_name):
        node_info[node_name]["id"] = i
        
    node_id_to_name_res = [x for x, _ in node_id_to_name]
    return node_id_to_name_res