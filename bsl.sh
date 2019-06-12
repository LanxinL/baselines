#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
export PATH=$PATH:/usr/local/cuda-9.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-9.0

eName=ganRl
seeds_num=10
alg=ddpg
#alg=ppo2
#alg=a2c
#nsteps=2048

#nvs="Swimmer-v2 HalfCheetah-v2" 
#envs="HalfCheetah-v2 Walker2d-v2 Ant-v2 Humanoid-v2 Hopper-v2" 
#envs="Walker2d-v2 Hopper-v2" 
#envs="HalfCheetah-v2 Ant-v2 Humanoid-v2 Hopper-v2 Walker2d-v2" 
envs="InvertedDoublePendulum-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2 Humanoid-v2 Swimmer-v2 Hopper-v2" 
#envs="InvertedPendulum-v2 InvertedDoublePendulum-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2 Humanoid-v2 Swimmer-v2 Hopper-v2" 
#envs="InvertedDoublePendulum-v2 Swimmer-v2" 
#envs="Humanoid-v2 Ant-v2" 

#seeds='3'
seeds='0 1 2 3 4' # 5 6 7 8 9
#seeds='20 21 22 23 24 25 26 27 28 29'
#seeds='10 11 12 13 14 15 16 17 18 19'
useReset0s='False' # True

#export MUJOCO_PY_FORCE_CPU=True # for server 110

num_envs='1 3 6 12'
num_timesteps_list='1e6' 

Nproc=25 #最大并发进程数 8 in 211, 32 in 110 18,16,16
resetPipeProc=true
pipe_file=proc.pip

save_interval=200
SHELL_LOG="./$(date +%Y-%m-%d).log"
LOCK_FILE="/tmp/lockForGitGanCheckoutInSh.lock"

rm $LOCK_FILE

# for 195
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export PATH=$PATH:/usr/local/cuda-10.0/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.0

source activate $eName
export PYTHONPATH=$PYTHONPATH:./baselines/
export MUJOCO_PY_FORCE_CPU=True # for server 110
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/llx/.mujoco/mjpro150/bin #for 211\
export LOCK_FILE=$LOCK_FILE
export TERRAINRL_PATH=~/dev/TerrainRLSim/ # for terrain
export eval=$eval
export TfNorResetFlag=$TfNorResetFlag
export warmUpByBuffer=$warmUpByBuffer

# 用来控制最大进程数
if [ $resetPipeProc == true -o ! -p "$pipe_file" ] ; then
    if [ -p "$pipe_file" ] ; then
        rm $pipe_file
    fi
    mkfifo $pipe_file && exec   777<>  $pipe_file     #通过文件描述符777访问fifo文件
    for ((i=0; i<$Nproc; i++)) #向fifo文件先填充等于Nproc值的行数
    do  
        echo  "init time add $i" >&777
    done
else
    exec   777<>  $pipe_file
fi

#Write Log 
shell_log(){
    LOG_INFO=$1
    echo "$(date "+%Y-%m-%d") $(date "+%H-%M-%S") : ${SHELL_NAME} : ${LOG_INFO}" >> ${SHELL_LOG}
}

# lock for git checkout
shell_lock(){
    echo '============locked==========='
    touch ${LOCK_FILE}
}

check_loce_file(){
    while [ -f "$LOCK_FILE" ]
    do
        sleep 1
    done
    shell_lock
    shell_log "lock"
}

checkout_git_branch(){
    cd baseline_gan_rl
    while [ ! "`git status | grep "On branch $gitBranch"`" ]; do #保证checkout后文件完全加载完成
        # sleep 19 #保证在另一分支上的实验能被成功run起来
        git checkout $gitBranch  
        shell_log "checkout to "$gitBranch
        # sleep 5 #保证checkout后文件完全加载完成
    done
    cd ..
}

echo_running_start(){
    echo
    echo "running with useFilter=$useFilter eName=$env, seed=$seed,  GAN_STD=$GAN_STD, num_env=$num_env, filterNetLayerNum=$filterNetLayerNum, env=$env, alg=$alg, epsilon_fl=$epsilon_fl, max_times=$max_time, separableFlRlGrad=$separableFlRlGrad, filterInfluencePolicy=$filterInfluencePolicy, flMinBatchRatio=$flMinBatchRatio, shorterInver=$shorterInver is running"
    echo '-----------------'
}

echo_running_over(){
    echo "running with useFilter=$useFilter eName=$env, seed=$seed,  GAN_STD=$GAN_STD, num_env=$num_env, filterNetLayerNum=$filterNetLayerNum, env=$env, alg=$alg, epsilon_fl=$epsilon_fl, max_times=$max_time, separableFlRlGrad=$separableFlRlGrad, filterInfluencePolicy=$filterInfluencePolicy, flMinBatchRatio=$flMinBatchRatio, shorterInver=$shorterInver is over"
    echo '============================'
    echo
}

Cproc=0

for seed in $seeds; do
for env in $envs;   do
for num_timesteps in $num_timesteps_list;   do
for useReset0 in $useReset0s;               do
for num_env in $num_envs;do
    read  -u  777  
    Cproc=$[$Cproc+1]

    check_loce_file

    export useReset0=$useReset0

    {
        PIDS[$Cproc]=$!; 
        echo_running_start

        save_path="ppoParMujoco/0/alg_"$alg"_e_"$num_timesteps"_useReset0_"$useReset0"_num_env_"$num_env"_nsteps_"$nsteps
            
        #save_path='test/'
        
        CUDA_VISIBLE_DEVICES=$(( $Cproc % 8)) python -m baselines.run --alg=$alg --env=$env --network=mlp --num_timesteps=$num_timesteps --num_env=$num_env --seed=$seed  --log_path ./result/$alg/$save_path/$env/$seed/  #--save_interval $save_interval #--nsteps $nsteps #--ent_coef 0.002
        
        echo_running_over
        echo 1>&777 
    }&                  
done;done;done;done;done

trap "rm -f ${LOCK_FILE};kill ${PIDS[*]}" SIGINT
wait
