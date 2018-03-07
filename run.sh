

for learning_rate in 1e-2 1e-4 3e-4
do
    mpc_horizon=7
    dyiter=200
    name=mpc_ppo_h_${mpc_horizon}_diter_${dyiter}_mlr_${learning_rate}

    nohup python -u train_mpc_ppo.py --dyn_iters=${dyiter} --exp_name=${name} --LEARN_REWARD=False --mpc=True --ppo=True --mpc_horizon=${mpc_horizon} --learning_rate=${learning_rate} &

done

# nohup tensorboard --logdir=./ &
# name=mpc_ppo_r_${simpaths}
# nohup python -u train_mpc_ppo_r.py --exp_name=${name} --onpol_iters=20 --random_paths=100 --mpc_horizon=10 &

# name=mpc_ppo_r_${simpaths}
# nohup python -u train_mpc_ppo_r.py --exp_name=${name} --onpol_iters=20 --random_paths=200 --mpc_horizon=10 &

