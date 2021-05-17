#export CUDA_VISIBLE_DEVICES=''
export CUDA_VISIBLE_DEVICES=0

wandb_project="0514_IBERT_8bit_reproduce"


for task in  "MRPC" #"RTE" "CoLA"
do
for bs in 4  #4 16
do
for max_epochs in 12
do
for lr in 2e-5 
do
for weight_decay in 0.1
do


    case $task in 
    MRPC) size=3668;;				
    RTE) size=2490;;
    CoLA) size=8551;;
    SST-2) size=67349;;
    esac 
    total_num_updates=$(( ${size} * $(($max_epochs - 2)) / ${bs}))

    for i in 1 
    do
    wandb_run="${task}_bs${bs}_epoch${max_epochs}_lr${lr}_${i}"
    python run.py \
    --arch roberta_base \
    --task $task \
    --restore-file ../ssd/models/roberta.base/model.pt \
    --no-save \
    --wandb_project_name $wandb_project \
    --wandb_run_name $wandb_run \
    --seed $RANDOM \
    --iteration $i \
    --total_num_updates $total_num_updates \
    --lr $lr \
    --max-epochs $max_epochs \
    --bs $bs \
    --weight-decay $weight_decay
    done

done
done
done
done
done

