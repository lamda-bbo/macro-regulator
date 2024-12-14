import numpy as np
import torch as th
import random
import os
import time
import datetime
import yaml
import gym
import place_env

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(ROOT_DIR, "benchmark")
SRC_DIR = os.path.join(ROOT_DIR, "src")
RESULT_DIR = os.path.join(ROOT_DIR, "results")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
UTILS_DIR = os.path.join(ROOT_DIR, "utils")
DREAMPLACE_PARENT_DIR = os.path.join(ROOT_DIR, "DREAMPlace")
DREAMPLACE_DIR = os.path.join(DREAMPLACE_PARENT_DIR, "dreamplace")
sys.path.extend(
    [ROOT_DIR, BENCHMARK_DIR, SRC_DIR, RESULT_DIR, CONFIG_DIR, UTILS_DIR, DREAMPLACE_PARENT_DIR, DREAMPLACE_DIR]
)

from collections import namedtuple
from types import SimpleNamespace
from place_db import PlaceDB
from agent import PPOAgent
from logger import Logger
from utils.comp_res import comp_res
from utils.constant import INF


from utils.debug import *

Transition = namedtuple('Transition',['state', 'action', 'reward', 'action_log_prob', 'next_state', 'done'])


def process_args():
    # cmd config
    params = [arg.lstrip("--") for arg in sys.argv if arg.startswith("--")]

    cmd_config_dict = {}
    for arg in params:
        key, value = arg.split('=')
        try:
            cmd_config_dict[key] = eval(value)
        except:
            cmd_config_dict[key] = value

        if key in ["benchmark_train", "benchmark_eval"]:
            value = value.split("[")[1].split("]")[0]
            benchmark_lst = value.replace(' ', '').split(',')
            benchmark_lst = [benchmark for benchmark in benchmark_lst if "superblue" in benchmark]
            cmd_config_dict[key] = benchmark_lst

    # default config
    config_path = os.path.join(CONFIG_DIR, "default.yaml")
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, value in cmd_config_dict.items():
        config_dict[key] = value


    args = SimpleNamespace(**config_dict)
    print(f"train on benchmark:\t{args.benchmark_train}")
    print(f"eval on benchmark:\t{args.benchmark_eval}")

    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if th.cuda.is_available() and args.use_cuda:
        args.device = 'cuda'
    else:
        args.use_cuda = False
        args.device = 'cpu'
    print(f'using device:{args.device}')

    # set unique token
    unique_token = "seed_{}_{}".format(args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    assert args.grid % 32 == 0, 'grid should be a multiple of 32'

    setattr(args, "ROOT_DIR", ROOT_DIR)
    setattr(args, "BENCHMARK_DIR", BENCHMARK_DIR)
    setattr(args, "SRC_DIR", SRC_DIR)
    setattr(args, "RESULT_DIR", RESULT_DIR)
    setattr(args, "CONFIG_DIR", CONFIG_DIR)
    setattr(args, "UTILS_DIR", UTILS_DIR)
    setattr(args, "DREAMPLACE_DIR", DREAMPLACE_DIR)
    return args

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

def main():
    args = process_args()

    placedb =PlaceDB(args)
    args.placedb = placedb
    args.t_env = 0
    args.i_episode = 0
    
    logger = Logger(args=args)
    args.logger = logger
    

    env = gym.make('place_env-v0', args=args).unwrapped
    agent = PPOAgent(args=args)

    seed_torch(args.seed)
    if hasattr(args, 'check_point_path') and len(args.check_point_path) > 0:
        agent.load_model(args.check_point_path)
        print('successfully load model')
        if args.eval_policy:
            # eval policy on benchmark_eval
            content = "\n"
            for benchmark in args.benchmark_eval:
                _, reward_info = run(
                    env=env,
                    agent=agent,
                    t_env=0,
                    test_mode=True,
                    benchmark=benchmark,
                    reward_scaling_flag=False
                )
                our_mp_hpwl, our_regularity = comp_res(problem=env.problem, 
                                                   node_pos=env.new_macro_pos, 
                                                   ratio_x=env.ratio_x, 
                                                   ratio_y=env.ratio_y, 
                                                   grid=env.place_grid)
                content += f'benchmark: {benchmark}\tour_gp_hpwl: {reward_info["global_hpwl"]}\tour_regularity: {our_regularity}\n'

                hpwl_path = os.path.join(args.RESULT_DIR, "hpwl", f'{args.name}/{args.unique_token}')
                os.makedirs(hpwl_path, exist_ok=True)
                hpwl_file_name = f'{benchmark}.txt'
                save_eval_metrics(os.path.join(hpwl_path, hpwl_file_name),
                                    episode=0,
                                    t_env=0,
                                    our_mp_hpwl=our_mp_hpwl,
                                    our_gp_hpwl=reward_info['global_hpwl'],
                                    dataset_mp_hpwl=env.problem.mp_hpwl,
                                    dataset_gp_hpwl=env.problem.gp_hpwl,
                                    our_regularity=our_regularity,
                                    dataset_regularity=env.problem.regularity)
            print(content)
            print("eval finish")
            return 

    # scaling reward
    if args.use_reward_scaling:
        for benchmark in args.benchmark_train:
            _, _ = run(env=env, 
                    agent=agent, 
                    t_env=0, 
                    test_mode=True,
                    benchmark=benchmark,
                    reward_scaling_flag=True)
            
            
    last_test_episode = -args.test_interval - 1

    gp_hpwl_min = {"average" : INF}
    for benchmark in args.benchmark_eval:
        gp_hpwl_min[benchmark] = INF

    run_time = []
    run_time_with_dmp = []
    update_time = []
    for i_episode in range(1, args.episode+1):
        args.i_episode = i_episode
        t_start = time.time()
        args.t_env, reward_info = run(env=env, 
                                      agent=agent, 
                                      t_env=args.t_env,
                                      test_mode=False,
                                      benchmark=None,
                                      reward_scaling_flag=False)
        
        run_time.append(time.time() - t_start)
        t_update = agent.update(args.t_env)
        if t_update != 0:
            update_time.append(t_update)

        if i_episode - last_test_episode >= args.test_interval:

            # eval on all eval benchmark
            content = "\n"
            gp_hpwl_each_benchmark = []
            for benchmark in args.benchmark_eval:
                t_start = time.time()
                _, reward_info = run(env=env, 
                                     agent=agent, 
                                     t_env=args.t_env, 
                                     test_mode=True,
                                     benchmark=benchmark,
                                     reward_scaling_flag=False)
                assert benchmark == reward_info['benchmark']
                
                run_time_with_dmp.append(time.time() - t_start)
                our_mp_hpwl, our_regularity = comp_res(problem=env.problem, 
                                                          node_pos=env.new_macro_pos, 
                                                          ratio_x=env.ratio_x, 
                                                          ratio_y=env.ratio_y, 
                                                          grid=env.place_grid)
                
                gp_hpwl_each_benchmark.append(reward_info["global_hpwl"].item())
                if benchmark not in args.benchmark_train:
                    eval_type = "eval"
                else:
                    eval_type = "train"
                content += f'benchmark: {benchmark} ({eval_type})\tepisode: {i_episode}\tt_env: {args.t_env}\tour_gp_hpwl: {reward_info["global_hpwl"]}\tour_regularity: {our_regularity}\n'
                
                logger.add(f'mp_hpwl/{benchmark}/hpwl', our_mp_hpwl, args.t_env)
                logger.add(f'mp_hpwl/{benchmark}/less_than_init', env.problem.mp_hpwl - our_mp_hpwl, args.t_env)
                logger.add(f'regularity/{benchmark}/regularity', our_regularity, args.t_env)
                logger.add(f'regularity/{benchmark}/less_than_init', env.problem.regularity - our_regularity, args.t_env)

                logger.add(f'gp_hpwl/{benchmark}/hpwl', reward_info['global_hpwl'], args.t_env)
                logger.add(f'gp_hpwl/{benchmark}/less_than_init', env.problem.gp_hpwl - reward_info['global_hpwl'], args.t_env)
                
                logger.add(f'reward/{benchmark}/tot_reward', reward_info['tot_reward'], args.t_env)
                logger.add(f'reward/{benchmark}/wire_reward', reward_info['wire_reward'], args.t_env)
                logger.add(f'reward/{benchmark}/regular_reward', reward_info['regular_reward'], args.t_env)
                logger.add(f'reward/{benchmark}/legal_reward', reward_info['legal_reward'], args.t_env)
                logger.add(f'reward/{benchmark}/wire_regular_reward_ratio', 
                        np.mean(reward_info['wire_regular_reward_ratio']), args.t_env)
                
                if gp_hpwl_min[benchmark] > reward_info['global_hpwl']:
                    gp_hpwl_min[benchmark] = reward_info['global_hpwl']
                    if args.save_model:
                        agent.save_model(gp_hpwl_min[benchmark], benchmark=benchmark)
                
                hpwl_path = os.path.join(args.RESULT_DIR, "hpwl", f'{args.name}/{args.unique_token}')
                os.makedirs(hpwl_path, exist_ok=True)
                hpwl_file_name = f'{benchmark}.txt'
                save_eval_metrics(os.path.join(hpwl_path, hpwl_file_name),
                                    episode=i_episode,
                                    t_env=args.t_env,
                                    our_mp_hpwl=our_mp_hpwl,
                                    our_gp_hpwl=reward_info['global_hpwl'],
                                    dataset_mp_hpwl=env.problem.mp_hpwl,
                                    dataset_gp_hpwl=env.problem.gp_hpwl,
                                    our_regularity=our_regularity,
                                    dataset_regularity=env.problem.regularity)
                
            logger.add('time/run(training)', np.mean(run_time), args.t_env)
            logger.add('time/run(test)', np.mean(run_time_with_dmp), args.t_env)
            if len(update_time) > 0:
                logger.add('time/update', np.mean(update_time), args.t_env)
            
            content += "\n"
            print(content)

            if np.mean(gp_hpwl_each_benchmark) < gp_hpwl_min['average']:
                gp_hpwl_min['average'] = np.mean(gp_hpwl_each_benchmark)
                if args.save_model:
                    agent.save_model(gp_hpwl=gp_hpwl_min['average'],
                                    benchmark=None)

            last_test_episode = i_episode

            
def run(env, agent, t_env, test_mode=False, benchmark=None, reward_scaling_flag=False):
    state = env.reset(test_mode=test_mode,
                      benchmark=benchmark,
                      reward_scaling_flag=reward_scaling_flag)
    agent.train() if not test_mode else agent.eval()
    reward_info = {
        'benchmark' : benchmark,
        'tot_reward' : 0,
        'wire_reward' : 0,
        'regular_reward' : 0,
        'legal_reward' : 0,
        'wire_regular_reward_ratio' : []
    }
    done = False
    while not done:
        action, action_log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        reward_info['tot_reward']     += reward
        reward_info['benchmark']       = info['benchmark']
        reward_info['wire_reward']    += info['wire_reward']
        reward_info['regular_reward'] += info['regular_reward']
        reward_info['legal_reward']   += info['legal_reward']
        reward_info['wire_regular_reward_ratio'].append(info['wire_reward/regular_reward'])
        reward_info['global_hpwl'] = 0 if not done else info['global_hpwl']
        if not test_mode:
            trans = Transition(state=state,
                                action=action,
                                reward=reward / 200.0,
                                action_log_prob=action_log_prob,
                                next_state=next_state, 
                                done=done)

            agent.store_transition(trans)
            t_env += 1
        state = next_state
    
    return t_env, reward_info

def save_eval_metrics(path, episode, t_env, our_mp_hpwl, our_gp_hpwl, dataset_mp_hpwl, dataset_gp_hpwl, our_regularity, dataset_regularity):
    if not os.path.exists(path):
        with open(path, 'a') as f:
            f.write(f"episode\tt_env\tour_mp_hpwl\tour_gp_hpwl\tdataset_mp_hpwl\tdataset_gp_hpwl\tour_regularity\tdataset_regularity\n")
    with open(path, 'a') as f:
        f.write(f'{episode}\t{t_env}\t{our_mp_hpwl}\t{our_gp_hpwl}\t{dataset_mp_hpwl}\t{dataset_gp_hpwl}\t{our_regularity}\t{dataset_regularity}\n')
    
        

if __name__ == '__main__':
    main()
    
