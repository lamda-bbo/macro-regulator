import numpy as np
import torch as th

class StateParsing:
    def __init__(self, args) -> None:
        self.args = args
        self.grid = args.grid
    
    def get_state(self,
        place_idx, 
        old_canvas,
        new_canvas,
        regular_mask,
        position_mask,
        wire_mask,
        next_regular_mask,
        next_position_mask,
        next_wire_mask,
        size_x,
        size_y
        ):
        return np.concatenate((
                    np.array([place_idx]),
                    old_canvas.flatten(),
                    new_canvas.flatten(),
                    regular_mask.flatten(),
                    position_mask.flatten(),
                    wire_mask.flatten(),
                    next_regular_mask.flatten(),
                    next_position_mask.flatten(),
                    next_wire_mask.flatten(),
                    np.array([size_x/self.grid, size_y/self.grid])
                ), axis=0
                )


    def state2canvas(self, state, new=True):
        if len(state.shape) == 1:
            if new:
                return state[1+self.grid*self.grid : 1+self.grid*self.grid * 2].reshape(self.grid, self.grid)
            else:
                return state[1 : 1+self.grid*self.grid].reshape(self.grid, self.grid)
        elif len(state.shape) == 2:
            if new:
                return state[:, 1+self.grid*self.grid : 1+self.grid*self.grid * 2].reshape(-1, self.grid, self.grid)
            else:
                return state[:, 1 : 1+self.grid*self.grid].reshape(-1, self.grid, self.grid)
        else:
            raise NotImplementedError

    def state2regular_mask(self, state, next_next_macro=False):
        if len(state.shape) == 1:
            if next_next_macro:
                return state[1+self.grid*self.grid * 5 : 1+self.grid*self.grid * 6].reshape(self.grid, self.grid)
            else:
                return state[1+self.grid*self.grid * 2 : 1+self.grid*self.grid * 3].reshape(self.grid, self.grid)
        elif len(state.shape) == 2:
            if next_next_macro:
                return state[:, 1+self.grid*self.grid * 5 : 1+self.grid*self.grid * 6].reshape(-1, self.grid, self.grid)
            else:
                return state[:, 1+self.grid*self.grid * 2 : 1+self.grid*self.grid * 3].reshape(-1, self.grid, self.grid)
        else:
            raise NotImplementedError

    def state2position_mask(self, state, next_next_macro=False):
        if len(state.shape) == 1:
            if next_next_macro:
                return state[1+self.grid*self.grid * 6 : 1+self.grid*self.grid * 7].reshape(self.grid, self.grid)
            else:
                return state[1+self.grid*self.grid * 3 : 1+self.grid*self.grid * 4].reshape(self.grid, self.grid)
        elif len(state.shape) == 2:
            if next_next_macro:
                return state[:, 1+self.grid*self.grid * 6 : 1+self.grid*self.grid * 7].reshape(-1, self.grid, self.grid)
            else:
                return state[:, 1+self.grid*self.grid * 3 : 1+self.grid*self.grid * 4].reshape(-1, self.grid, self.grid)
        else:
            raise NotImplementedError

    def state2wire_mask(self, state, next_next_macro=False):
        if len(state.shape) == 1:
            if next_next_macro:
                return state[1+self.grid*self.grid * 7 : 1+self.grid*self.grid * 8].reshape(self.grid, self.grid)
            else:
                return state[1+self.grid*self.grid * 4 : 1+self.grid*self.grid * 5].reshape(self.grid, self.grid)
        elif len(state.shape) == 2:
            if next_next_macro:
                return state[:, 1+self.grid*self.grid * 7 : 1+self.grid*self.grid * 8].reshape(-1, self.grid, self.grid)
            else:
                return state[:, 1+self.grid*self.grid * 4 : 1+self.grid*self.grid * 5].reshape(-1, self.grid, self.grid)
        else:
            raise NotImplementedError
