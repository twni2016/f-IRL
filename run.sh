export PYTHONPATH=${PWD}:$PYTHONPATH
# UNIT TEST

# REACHER GOAL REACHING 
# python firl/irl_density.py configs/density/reacher_trace_gauss.yml
# python firl/irl_density.py configs/density/reacher_trace_mix.yml
# python baselines/main_density.py configs/density/reacher_trace_gauss.yml
# python baselines/main_density.py configs/density/reacher_trace_mix.yml

# POINTMASS DOWNSTREAM
# python firl/irl_density.py configs/density/grid_uniform.yml
# python continuous/adv_irl/main.py continuous/configs/RSS/reacher_trace_gauss.yml

# MUJOCO IRL BENCHMARK
# python common/train_expert.py configs/samples/experts/hopper.yml
# python firl/irl_samples.py configs/samples/agents/hopper.yml
# python common/train_optimal.py configs/samples/experts/halfcheetah.yml
# python baselines/bc.py configs/samples/agents/hopper.yml
# python baselines/main_samples.py configs/samples/agents/hopper.yml
