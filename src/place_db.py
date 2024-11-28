
import sys
sys.path.append('..')

from problem_instance import ProblemInstance

class PlaceDB():
    def __init__(self, args):
        self.args=args
        self.problem_train = []
        self.problem_eval = []
        benchmark_all = set(args.benchmark_train + args.benchmark_eval)
        for benchmark in benchmark_all:
            problem = ProblemInstance(args=args, benchmark=benchmark)
            if benchmark in args.benchmark_train:
                self.problem_train.append(problem)
            if benchmark in args.benchmark_eval:
                self.problem_eval.append(problem) 
        


                







