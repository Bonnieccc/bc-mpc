

for MPC_EXP in  0 0.2, 0.4 0.6 0.8 1
do
    seed=1
    ppo_lr=1e-4
    name=exp_test_${seed}_mpc_exp_${MPC_EXP}

    nohup python -u train_mpc_ppo.py --MPC_EXP=${MPC_EXP} --SELFEXP=False --exp_name=${name} --mpc_rand=False --seed=${seed} --LEARN_REWARD=False --mpc=True --ppo=True --ppo_lr=${ppo_lr} --LAYER_NORM=True&

done


# for ppo_lr in 1e-3 1e-5 3e-4 5e-5
# do
#     MPC_EXP=0.1
#     seed=1
#     name=mpc_ppo_r_MPC_EXP_${MPC_EXP}_seed_${seed}_${ppo_lr}

#     nohup python -u train_mpc_ppo.py  --exp_name=${name} --seed=${seed} --LEARN_REWARD=True --mpc=True --ppo=True --MPC_EXP=${MPC_EXP} --ppo_lr=${ppo_lr}&

# done
# nohup tensorboard --logdir=./ &
# name=mpc_ppo_r_${simpaths}
# nohup python -u train_mpc_ppo_r.py --exp_name=${name} --onpol_iters=20 --random_paths=100 --mpc_horizon=10 &

# name=mpc_ppo_r_${simpaths}
# nohup python -u train_mpc_ppo_r.py --exp_name=${name} --onpol_iters=20 --random_paths=200 --mpc_horizon=10 &

