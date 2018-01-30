name=bc_reward_iter20_random50_mpc10
nohup python mpc_with_learned_reward.py --exp_name=${name} --onpol_iters=20 --random_paths=50 --mpc_horizon=10 &

name=bc_reward_iter20_random100_mpc10
nohup python mpc_with_learned_reward.py --exp_name=${name} --onpol_iters=20 --random_paths=100 --mpc_horizon=10 &

name=bc_reward_iter20_random200_mpc10
nohup python mpc_with_learned_reward.py --exp_name=${name} --onpol_iters=20 --random_paths=200 --mpc_horizon=10 &



