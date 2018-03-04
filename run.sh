for horizon in 17 20 25

do
    name=mpc_ppo_r_${horizon}_nosexp
    nohup python -u train_mpc_ppo_r.py --exp_name=${name} --mpc --ppo --mpc_horizon=${horizon} & > mpc_ppo_r_h_${horizon}.out &

done

# name=mpc_ppo_r_${simpaths}
# nohup python -u train_mpc_ppo_r.py --exp_name=${name} --onpol_iters=20 --random_paths=100 --mpc_horizon=10 &

# name=mpc_ppo_r_${simpaths}
# nohup python -u train_mpc_ppo_r.py --exp_name=${name} --onpol_iters=20 --random_paths=200 --mpc_horizon=10 &



