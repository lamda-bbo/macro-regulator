import pickle
from .debug import *
from .read_superblue import get_inv_scaling_ratio

def comp_res(problem, node_pos, ratio_x, ratio_y, grid):

    hpwl = 0.0
    for net_name in problem.net_info:
        max_x = 0.0
        min_x = problem.max_height * 1.1
        max_y = 0.0
        min_y = problem.max_height * 1.1
        for node_name in problem.net_info[net_name]["nodes"]:
            if node_name not in node_pos:
                continue
            h = problem.node_info[node_name]['x']
            w = problem.node_info[node_name]['y']
            pin_x = node_pos[node_name][0] * ratio_x + h / 2.0 + problem.net_info[net_name]["nodes"][node_name]["x_offset"]
            pin_y = node_pos[node_name][1] * ratio_y + w / 2.0 + problem.net_info[net_name]["nodes"][node_name]["y_offset"]
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)
        for port_name in problem.net_info[net_name]["ports"]:
            h = problem.port_info[port_name]['x']
            w = problem.port_info[port_name]['y']
            pin_x = h
            pin_y = w
            max_x = max(pin_x, max_x)
            min_x = min(pin_x, min_x)
            max_y = max(pin_y, max_y)
            min_y = min(pin_y, min_y)
        if min_x <= problem.max_height:
            hpwl_tmp = (max_x - min_x) + (max_y - min_y)
        else:
            hpwl_tmp = 0
        if "weight" in problem.net_info[net_name]:
            hpwl_tmp *= problem.net_info[net_name]["weight"]
        hpwl += hpwl_tmp


    regularity = compute_regularity(node_pos=node_pos, grid=grid, ratio_x=ratio_x, ratio_y=ratio_y, problem=problem)
    return hpwl, regularity


def compute_regularity(node_pos, grid, ratio_x, ratio_y, problem):
    inv_r_x, inv_r_y = get_inv_scaling_ratio(problem.database)

    bound_x = round(round(grid * ratio_x + ratio_x) * inv_r_x)
    bound_y = round(round(grid * ratio_y + ratio_y) * inv_r_y)

    total_area = 0
    x_dis_from_edge = 0
    y_dis_from_edge = 0
    for macro in node_pos.keys():
        x, y, size_x, size_y = node_pos[macro]
        lower_x = round(round(x * ratio_x + ratio_x) * inv_r_x)
        lower_y = round(round(y * ratio_y + ratio_y) * inv_r_y)
        upper_x = round(round((x+size_x) * ratio_x + ratio_x) * inv_r_x)
        upper_y = round(round((y+size_y) * ratio_y + ratio_y) * inv_r_y)

        area = round(round(size_x * ratio_x + ratio_x) * inv_r_x) * round(round(size_y * ratio_y + ratio_y) * inv_r_y)
        total_area += area

        x_dis_from_edge += min(lower_x, max(bound_x - upper_x, 0)) * area
        y_dis_from_edge += min(lower_y, max(bound_y - upper_y, 0)) * area

    return (x_dis_from_edge + y_dis_from_edge) / total_area

    