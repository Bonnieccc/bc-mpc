

for ppo_lr in 1e-4 1e-3 5e-4 5e-5
do
    mpc_horizon=7
    dyiter=200
    name=mpc_ppo_h_${mpc_horizon}_diter_${dyiter}_ppo_lr_${ppo_lr}

    nohup python -u train_mpc_ppo.py --dyn_iters=${dyiter} --exp_name=${name} --LEARN_REWARD=False --mpc=True --ppo=True --mpc_horizon=${mpc_horizon} --ppo_lr=${ppo_lr} &

done

# nohup tensorboard --logdir=./ &
# name=mpc_ppo_r_${simpaths}
# nohup python -u train_mpc_ppo_r.py --exp_name=${name} --onpol_iters=20 --random_paths=100 --mpc_horizon=10 &

# name=mpc_ppo_r_${simpaths}
# nohup python -u train_mpc_ppo_r.py --exp_name=${name} --onpol_iters=20 --random_paths=200 --mpc_horizon=10 &

