# training on superblue1 and evaluating on superblue1
python main.py --name=test --gpu=0 \
               --benchmark_train=[] \
               --benchmark_eval=[superblue1] \
               --check_point_path=../policy/pretrained_model.pkl \
               --eval_policy=True