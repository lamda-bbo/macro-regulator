import gym
from gym import spaces
import numpy as np
import os
import math
import sys
import copy
import matplotlib.pyplot as plt
from PIL import Image

from matplotlib import patches
from sklearn.neighbors import NearestNeighbors
sys.path.append('../..')

from utils.debug import *
from utils.state_parsing import StateParsing
from utils.comp_res import comp_res
from utils.constant import INF

class PlaceEnv(gym.Env):
    def __init__(self, args) -> None:
        self.args = args
        self.grid = args.grid
        self.action_space = spaces.Discrete(self.grid * self.grid)
        self.observation_space = spaces.Discrete(3 + 8 * self.grid * self.grid)
        self.placedb = args.placedb
        self.problem_train = self.placedb.problem_train
        self.problem_eval = self.placedb.problem_eval

        # for reward scaling
        self.wire_reward_max = -INF
        self.wire_reward_min = INF
        self.regular_reward_max = -INF
        self.regular_reward_min = INF
        self.reward_scaling_flag = False

        self.n_macro = args.n_macro
        self.place_idx = 0
        self.old_macro_pos = {}
        self.new_macro_pos = {}

        self.macro_to_place = []
        self.macro_placed = []

        self.state_parsing = StateParsing(args)
        self.test_mode = False

  
        self.macro_placement_path = os.path.join(args.ROOT_DIR , "macro_placement", f'{self.args.name}/{self.args.unique_token}') 
        self.dmp_temp_placement_path = os.path.join(args.ROOT_DIR, "dmp_temp_placement", f'{self.args.name}/{self.args.unique_token}')

        self.result_path = args.RESULT_DIR
        self.reference_placement_path = os.path.join(self.result_path, "reference_placement", f'{self.args.name}/{self.args.unique_token}')
        self.dmp_result_path = os.path.join(self.result_path, "dmp_results", f'{self.args.name}/{self.args.unique_token}')
        self.full_figure_path = os.path.join(self.result_path, "full_figures", f'{self.args.name}/{self.args.unique_token}')
        self.n_dmp_eval_path = os.path.join(self.result_path, "n_dmp_eval", f'{self.args.name}/{self.args.unique_token}')

        os.makedirs(self.macro_placement_path, exist_ok=True)
        os.makedirs(self.dmp_temp_placement_path, exist_ok=True)
        os.makedirs(self.reference_placement_path, exist_ok=True)
        os.makedirs(self.dmp_result_path, exist_ok=True)
        os.makedirs(self.full_figure_path, exist_ok=True)
        os.makedirs(self.n_dmp_eval_path, exist_ok=True)

        for problem in self.problem_train:
            problem.init_dmp(macro_placement_path = self.macro_placement_path)
        
        for problem in self.problem_eval:
            problem.init_dmp(macro_placement_path = self.macro_placement_path)

        for problem in set(self.problem_train + self.problem_eval):
            if self.args.dataset_path is not None and os.path.exists(self.args.dataset_path):
                print(f"reading dataset from {self.args.dataset_path}")
                problem.read_macro_pos_from_def(def_path=self.args.dataset_path)
                metric, placement_info = problem.call_dmp(
                    placement_save_path=self.reference_placement_path,
                    call_id=0,
                    test_mode=True,
                    optimization=False
                )
                problem.set_gp_hpwl(metric.hpwl)
                mp_hpwl, regularity = comp_res(problem=problem, 
                                        node_pos=problem.macro_pos,
                                        ratio_x=problem.ratio_x,
                                        ratio_y=problem.ratio_y,
                                        grid=problem.place_grid)
                problem.set_mp_hpwl(mp_hpwl)
                problem.set_regularity(regularity)
            else:
                # init dataset from dmp
                print(f"not such a dataset in benchmark {problem.benchmark}")
                print(f"init dataset from dmp")
                problem.save_def(is_dataset=True)

                hpwl_lst = []
                placement_info_lst = []
                for call_id in range(0, self.args.n_dmp_eval):
                    metric, placement_info = problem.call_dmp(call_id=call_id, 
                                            placement_save_path=self.dmp_temp_placement_path,
                                            test_mode=True)
                    hpwl_lst.append(metric.hpwl.item())
                    placement_info_lst.append(placement_info)

                min_idx = np.argmin(hpwl_lst)
                min_hpwl = hpwl_lst[min_idx]
                placement_info = placement_info_lst[min_idx]

                os.makedirs(os.path.join(self.dmp_temp_placement_path, problem.benchmark), exist_ok=True)
                os.makedirs(os.path.join(self.reference_placement_path, problem.benchmark), exist_ok=True)
                os.system(f"cp {os.path.join(self.dmp_temp_placement_path, problem.benchmark, f'{min_idx}.def')}"+\
                        f" {os.path.join(self.reference_placement_path, problem.benchmark, f'{problem.benchmark}_{min_hpwl/1e7:.4f}.def')}")
                os.system(f"cp {os.path.join(self.dmp_temp_placement_path, problem.benchmark, f'{min_idx}.png')}"+\
                        f" {os.path.join(self.reference_placement_path, problem.benchmark, f'{problem.benchmark}_{min_hpwl/1e7:.4f}.png')}")
                self.flip_dmp_figure(figure_path=os.path.join(self.reference_placement_path, problem.benchmark, f'{problem.benchmark}_{min_hpwl/1e7:.4f}.png'))

                problem.set_macro_pos(placement_info)
                problem.set_gp_hpwl(min_hpwl)
                mp_hpwl, regularity = comp_res(problem=problem, 
                                        node_pos=problem.macro_pos,
                                        ratio_x=problem.ratio_x,
                                        ratio_y=problem.ratio_y,
                                        grid=problem.place_grid)
                problem.set_mp_hpwl(mp_hpwl)
                problem.set_regularity(regularity)
                    
    
    def reset(self, test_mode, benchmark=None, reward_scaling_flag=False):
        self.test_mode = test_mode
        self.place_idx = 0
        self.macro_to_place.clear()
        self.macro_placed.clear()
        self.old_macro_pos = {}
        self.new_macro_pos = {}
        self.reward_scaling_flag = reward_scaling_flag

        if benchmark is None:
            self.problem = np.random.choice(self.problem_train, size=1)[0]
        else:
            for problem in set(self.problem_train + self.problem_eval):
                if benchmark == problem.benchmark:
                    self.problem = problem
                    break

        self.ratio_x = self.problem.ratio_x
        self.ratio_y = self.problem.ratio_y
        self.ratio_sum = self.ratio_x + self.ratio_y
        self.place_grid = self.problem.place_grid

        old_canvas = np.zeros((self.grid, self.grid))
        new_canvas = np.zeros((self.grid, self.grid))
        for macro in self.problem.macro_pos:
            pos_x, pos_y, size_x, size_y = self.problem.macro_pos[macro]
            assert pos_x + size_x <= self.place_grid, (pos_x, size_x, self.place_grid)
            assert pos_y + size_y <= self.place_grid, (pos_y, size_y, self.place_grid)
            self.old_macro_pos[macro] = (pos_x, pos_y, size_x, size_y)
            self.new_macro_pos[macro] = (pos_x, pos_y, size_x, size_y)

            old_canvas = self.__draw_canvas(old_canvas, pos_x, pos_y, size_x, size_y)


        self.macro_placed = []
        self.macro_to_place = list(self.old_macro_pos.keys())

        for macro in self.macro_placed:
            pos_x, pos_y, size_x, size_y = self.old_macro_pos[macro]
            new_canvas = self.__draw_canvas(new_canvas, pos_x, pos_y, size_x, size_y)
        
        self.set_place_order()
        self.reset_net_to_macro()

        # get mask
        _, _, size_x, size_y = self.old_macro_pos[self.macro_to_place[0]]
        regular_mask  = self.get_regular_mask(self.macro_to_place[0])
        position_mask = self.get_position_mask(size_x=size_x, size_y=size_y)
        wire_mask     = self.get_wire_mask(self.macro_to_place[0])

        if len(self.macro_to_place) > 1:
            _, _, next_size_x, next_size_y = self.old_macro_pos[self.macro_to_place[1]]
            next_regular_mask  = self.get_regular_mask(self.macro_to_place[1])
            next_position_mask = self.get_position_mask(size_x=next_size_x, size_y=next_size_y)
            next_wire_mask     = self.get_wire_mask(self.macro_to_place[1])
        else:
            next_regular_mask  = np.array((self.grid, self.grid))
            next_position_mask = np.array((self.grid, self.grid))
            next_wire_mask     = np.array((self.grid, self.grid))
        
        # mask normalization
        regular_mask, next_regular_mask = self.__mask_normalization(regular_mask, next_regular_mask)
        wire_mask   , next_wire_mask    = self.__mask_normalization(wire_mask, next_wire_mask)
        
        self.state = self.state_parsing.get_state(
            place_idx=self.place_idx,
            old_canvas=old_canvas,
            new_canvas=new_canvas,
            regular_mask=regular_mask,
            position_mask=position_mask,
            wire_mask=wire_mask,
            next_regular_mask=next_regular_mask,
            next_position_mask=next_position_mask,
            next_wire_mask=next_wire_mask,
            size_x=size_x,
            size_y=size_y
        )
        
        return self.state.copy()

    def reset_net_to_macro(self):
        # reset net_to_macro
        self.net_to_macro = {}
        
        for port_name in self.problem.port_to_net_dict:
            for net_name in self.problem.port_to_net_dict[port_name]:
                pin_x = round(self.problem.port_info[port_name]['x'] / self.ratio_x)
                pin_y = round(self.problem.port_info[port_name]['y'] / self.ratio_y)

                if net_name in self.net_to_macro:
                    self.net_to_macro[net_name][port_name] = (pin_x, pin_y)
                else:
                    self.net_to_macro[net_name] = {}
                    self.net_to_macro[net_name][port_name] = (pin_x, pin_y)
        
        for macro in self.old_macro_pos:
            for net_name in self.problem.node_to_net_dict[macro]:
                x, y, _, _ = self.old_macro_pos[macro]
                pin_x = round((x * self.problem.ratio_x + self.problem.node_info[macro]['x']/2 + \
                        self.problem.net_info[net_name]["nodes"][macro]["x_offset"])/self.ratio_x)
                pin_y = round((y * self.problem.ratio_y + self.problem.node_info[macro]['y']/2 + \
                        self.problem.net_info[net_name]["nodes"][macro]["y_offset"])/self.ratio_y)
        
                if net_name in self.net_to_macro:
                    self.net_to_macro[net_name][macro] = (pin_x, pin_y)
                else:
                    self.net_to_macro[net_name] = {}
                    self.net_to_macro[net_name][macro] = (pin_x, pin_y)


    def step(self, action):
        
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        
        old_canvas = self.state_parsing.state2canvas(self.state, new=False)
        new_canvas = self.state_parsing.state2canvas(self.state, new=True)
        position_mask = self.state_parsing.state2position_mask(self.state, next_next_macro=False)
        reward = 0
        x = round(action // self.grid)
        y = round(action % self.grid)
        
        legal_reward = 0
        if position_mask[x][y] == 1:
            legal_reward += -200000
        
        macro = self.macro_to_place[self.place_idx]
        _, _, size_x, size_y = self.old_macro_pos[macro]

        assert abs(size_x - self.state[-2]*self.grid) < 1e-5
        assert abs(size_y - self.state[-1]*self.grid) < 1e-5

        # update new_canvas, macro position and macro_placed
        new_canvas = self.__draw_canvas(new_canvas, x, y, size_x, size_y)
        self.new_macro_pos[macro] = (x, y, size_x, size_y)
        self.macro_placed.append(macro)

        # compute HPWL reward
        wire_reward_mask = self.get_wire_mask(macro)
        wire_reward = -wire_reward_mask[x, y]
        for net_name in self.problem.node_to_net_dict[macro]:
            pin_x = round((x * self.ratio_x + self.problem.node_info[macro]['x']/2 + \
                    self.problem.net_info[net_name]["nodes"][macro]["x_offset"])/self.ratio_x)
            pin_y = round((y * self.ratio_y + self.problem.node_info[macro]['y']/2 + \
                    self.problem.net_info[net_name]["nodes"][macro]["y_offset"])/self.ratio_y)
            self.net_to_macro[net_name][macro] = (pin_x, pin_y)

        # compute regular reward
        regular_reward  = -self.get_regular_mask(macro)[x, y] 

        # reward scaling
        if self.args.use_reward_scaling and self.reward_scaling_flag:
            self.wire_reward_max = max(self.wire_reward_max, wire_reward)
            self.wire_reward_min = min(self.wire_reward_min, wire_reward)
            self.regular_reward_max = max(self.regular_reward_max, regular_reward)
            self.regular_reward_min = min(self.regular_reward_min, regular_reward)

        if self.args.use_reward_scaling:
            wire_reward = (wire_reward - self.wire_reward_min + 1e-10) / (self.wire_reward_max - self.wire_reward_min + 1e-10)
            regular_reward = (regular_reward - self.regular_reward_min + 1e-10) / (self.regular_reward_max - self.regular_reward_min + 1e-10)
        
        # reward
        reward += legal_reward + self.args.wire_coeff * wire_reward + (1 - self.args.wire_coeff) * regular_reward
        
        # get next macro
        self.place_idx += 1

        # get mask
        size_x, size_y = 0, 0
        regular_mask  = np.zeros((self.grid, self.grid))
        position_mask = np.zeros((self.grid, self.grid))
        wire_mask     = np.zeros((self.grid, self.grid))
        next_regular_mask  = np.zeros((self.grid, self.grid))
        next_position_mask = np.zeros((self.grid, self.grid))
        next_wire_mask     = np.zeros((self.grid, self.grid))
        
        if self.place_idx < len(self.macro_to_place):
            _, _, size_x, size_y = self.old_macro_pos[self.macro_to_place[self.place_idx]]
            regular_mask  = self.get_regular_mask(self.macro_to_place[self.place_idx])
            position_mask = self.get_position_mask(size_x=size_x, size_y=size_y) 
            wire_mask     = self.get_wire_mask(self.macro_to_place[self.place_idx])
        if self.place_idx < len(self.macro_to_place) - 1:
            _, _, next_size_x, next_size_y = self.old_macro_pos[self.macro_to_place[self.place_idx + 1]]
            next_regular_mask  = self.get_regular_mask(self.macro_to_place[self.place_idx + 1])
            next_position_mask = self.get_position_mask(size_x=next_size_x, size_y=next_size_y) 
            next_wire_mask     = self.get_wire_mask(self.macro_to_place[self.place_idx + 1])
        
        done = False
        if self.place_idx >= len(self.macro_to_place):
            self.place_idx = 0
            self.macro_to_place.clear()
            self.macro_placed.clear()
            self.old_macro_pos = copy.deepcopy(self.new_macro_pos)
            done = True
            if self.test_mode and not self.reward_scaling_flag:
                self.problem.save_def(macro_pos=self.new_macro_pos,
                                        is_dataset=False)
                
                # eval for n_dmp_eval times and choose the one with the lowest GP HPWL
                min_global_hpwl = INF
                hpwl_lst = []
                min_call_id = 0
                for call_id in range(self.args.n_dmp_eval):
                    metric, _ = self.problem.call_dmp(call_id=call_id, 
                                                        placement_save_path=self.dmp_temp_placement_path,
                                                        test_mode=self.test_mode,
                                                        optimization=True)
                    hpwl_lst.append(metric.hpwl)
                    if metric.hpwl < min_global_hpwl:
                        min_global_hpwl = metric.hpwl
                        min_call_id = call_id
                os.makedirs(os.path.join(self.dmp_result_path, f"{self.args.name}/{self.args.unique_token}/{self.problem.benchmark}"),
                            exist_ok=True)
                os.makedirs(os.path.join(self.full_figure_path, f"{self.args.name}/{self.args.unique_token}/{self.problem.benchmark}"), 
                            exist_ok=True)
                os.system(f'cp {os.path.join(self.dmp_temp_placement_path, self.problem.benchmark, f"{min_call_id}.def")}'+\
                            f' {os.path.join(self.dmp_result_path, f"{self.args.name}/{self.args.unique_token}/{self.problem.benchmark}/{self.args.i_episode}_{self.args.t_env}_{min_global_hpwl/1e7:.4f}.def")}')
                os.system(f'cp {os.path.join(self.dmp_temp_placement_path, self.problem.benchmark, f"{min_call_id}.png")}'+\
                            f' {os.path.join(self.full_figure_path, f"{self.args.name}/{self.args.unique_token}/{self.problem.benchmark}/{self.args.i_episode}_{self.args.t_env}_{min_global_hpwl/1e7:.4f}.png")}')
                
                self.flip_dmp_figure(figure_path=os.path.join(self.full_figure_path, f"{self.args.name}/{self.args.unique_token}/{self.problem.benchmark}/{self.args.i_episode}_{self.args.t_env}_{min_global_hpwl/1e7:.4f}.png"))

                os.makedirs(os.path.join(self.n_dmp_eval_path, f"{self.args.name}/{self.args.unique_token}"), exist_ok=True)
                n_dmp_eval_file_path = os.path.join(self.n_dmp_eval_path, f"{self.args.name}/{self.args.unique_token}/{self.problem.benchmark}.txt")
                if not os.path.exists(n_dmp_eval_file_path):
                    with open(n_dmp_eval_file_path, 'a') as f:
                        header = ""
                        for i in range(1, len(hpwl_lst)+1):
                            header += f"dmp_{i}_hpwl\t"
                        header += "\n"
                        f.write(header)
                with open(n_dmp_eval_file_path, 'a') as f:
                    content = ""
                    for hpwl in hpwl_lst:
                        content += f"{hpwl}\t"
                    content += "\n"
                    f.write(content)
                
                global_hpwl = min_global_hpwl
            else:
                global_hpwl = 0
            
        
        regular_mask, next_regular_mask = self.__mask_normalization(regular_mask, next_regular_mask)
        wire_mask   , next_wire_mask = self.__mask_normalization(wire_mask, next_wire_mask)

        self.state = self.state_parsing.get_state(
            place_idx=self.place_idx,
            old_canvas=old_canvas,
            new_canvas=new_canvas,
            regular_mask=regular_mask,
            position_mask=position_mask,
            wire_mask=wire_mask,
            next_regular_mask=next_regular_mask,
            next_position_mask=next_position_mask,
            next_wire_mask=next_wire_mask,
            size_x=size_x,
            size_y=size_y
        )

        info = {
            "benchmark" : self.problem.benchmark,
            "legal_reward" : legal_reward, 
            "wire_reward" : wire_reward, 
            "regular_reward" : regular_reward,
            "wire_reward/regular_reward" : np.abs(wire_reward) / (np.abs(regular_reward)+1e-10),
            "global_hpwl" : 0 if (not done) else global_hpwl,
        }

        return self.state.copy(), reward, done, info
                
    def set_place_order(self):
        self.macro_to_place = sorted(self.macro_to_place, key=lambda x:self.problem.node_id_to_name.index(x))

    def get_regular_mask(self, macro):
        x, y, size_x, size_y = self.old_macro_pos[macro]
        mask = np.zeros((self.grid, self.grid))
        start_x = 1
        start_y = 1
        end_x = self.place_grid - size_x - 1
        end_y = self.place_grid - size_y - 1

        # mask
        x_mask_1 = np.zeros((self.grid, self.grid))
        x_mask_2 = np.zeros((self.grid, self.grid))
        for i in range(start_x, end_x+1):
            x_mask_1[i:end_x+1, :] += 1 * (self.ratio_x / self.ratio_sum)
        for i in range(end_x, start_x-1, -1):
            x_mask_2[start_x:i+1, :] += 1 * (self.ratio_x / self.ratio_sum)
        x_mask = np.minimum(x_mask_1, x_mask_2)

        y_mask_1 = np.zeros((self.grid, self.grid))
        y_mask_2 = np.zeros((self.grid, self.grid))
        for j in range(start_y, end_y+1):
            y_mask_1[:, j:end_y+1] += 1 * (self.ratio_y / self.ratio_sum)
        for j in range(end_y, start_y-1, -1):
            y_mask_2[:, start_y:j+1] += 1 * (self.ratio_y / self.ratio_sum)
        y_mask = np.minimum(y_mask_1, y_mask_2)

        mask = x_mask + y_mask
        mask -= mask[x, y]

        return mask

    def get_position_mask(self, size_x, size_y):
        mask = np.zeros((self.grid, self.grid))
        for macro in self.macro_placed:
            start_x = max(0, self.new_macro_pos[macro][0] - size_x + 1)
            start_y = max(0, self.new_macro_pos[macro][1] - size_y + 1)
            end_x = min(self.new_macro_pos[macro][0] + self.new_macro_pos[macro][2] - 1, self.grid)
            end_y = min(self.new_macro_pos[macro][1] + self.new_macro_pos[macro][3] - 1, self.grid)
            mask[start_x: end_x + 1, start_y: end_y + 1] = 1

        mask[self.place_grid - size_x + 1:, :] = 1
        mask[:, self.place_grid - size_y + 1:] = 1
        return mask

    def get_wire_mask(self, macro):
        mask = np.zeros(shape=(self.grid, self.grid))  
        
        for net_name in self.problem.node_to_net_dict[macro]:
            if net_name in self.net_to_macro:
                delta_pin_x = round((self.problem.node_info[macro]['x']/2 + \
                    self.problem.net_info[net_name]["nodes"][macro]["x_offset"])/self.ratio_x)
                delta_pin_y = round((self.problem.node_info[macro]['y']/2 + \
                    self.problem.net_info[net_name]["nodes"][macro]["y_offset"])/self.ratio_y)
                
                pin_x, pin_y = self.net_to_macro[net_name][macro]
                del self.net_to_macro[net_name][macro]

                pin_array = np.array(list(self.net_to_macro[net_name].values()))
                max_x = max(pin_array[:, 0])
                min_x = min(pin_array[:, 0])
                max_y = max(pin_array[:, 1])
                min_y = min(pin_array[:, 1])
        
                start_x = min_x - delta_pin_x
                end_x = max_x - delta_pin_x
                start_y = min_y - delta_pin_y
                end_y = max_y - delta_pin_y

                start_x = max(start_x, 0)
                start_y = max(start_y, 0)
                end_x = max(end_x, 0)
                end_y = max(end_y, 0)

                start_x = min(start_x, self.grid)
                start_y = min(start_y, self.grid)
                end_x  = min(end_x, self.grid)
                end_y  = min(end_y, self.grid)

                if not 'weight' in self.problem.net_info[net_name]:
                    weight = 1.0
                else:
                    weight = self.problem.net_info[net_name]['weight']

                for i in range(0, start_x):
                    mask[i, :] += (start_x - i) * weight * (self.ratio_x/self.ratio_sum)
                for i in range(end_x+1, self.grid):
                    mask[i, :] +=  (i- end_x) * weight * (self.ratio_x/self.ratio_sum)
                for j in range(0, start_y):
                    mask[:, j] += (start_y - j) * weight * (self.ratio_y/self.ratio_sum)
                for j in range(end_y+1, self.grid):
                    mask[:, j] += (j - end_y) * weight * (self.ratio_y/self.ratio_sum)
                
                mask -= mask[self.old_macro_pos[macro][0], self.old_macro_pos[macro][1]]
                self.net_to_macro[net_name][macro] = (pin_x, pin_y)

        return mask
    
    def __draw_canvas(self, canvas, x, y, size_x, size_y):
        canvas[x : x+size_x, y : y+size_y] = 1.0
        canvas[x : x + size_x, y] = 0.5
        if y + size_y -1 < self.grid:
            canvas[x : x + size_x, max(0, y + size_y -1)] = 0.5
        canvas[x, y: y + size_y] = 0.5
        if x + size_x - 1 < self.grid:
            canvas[max(0, x+size_x-1), y: y + size_y] = 0.5
        
        return canvas

    def __mask_normalization(self, mask1, mask2):
        if np.abs(mask1).max() > 0 or np.abs(mask2).max() > 0:
            mask1 /= max(np.abs(mask1).max(), np.abs(mask2).max())
            mask2 /= max(np.abs(mask1).max(), np.abs(mask2).max())

        return mask1, mask2
    
    def flip_dmp_figure(self, figure_path):
        img = Image.open(figure_path)
        out = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.close()
        out.save(figure_path)
    


