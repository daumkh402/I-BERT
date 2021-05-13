#export CUDA_VISIBLE_DEVICES=''
export CUDA_VISIBLE_DEVICES=1

wandb_project="0514_IBERT_8bit_reproduce"


for task in  "RTE" 
do

for bs in 16
do

for max_epochs in 12
do

for lr in  1.5e-5 #2e-5
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

    for i in 1 2 3
    do
    wandb_run="${task}_bs${bs}_epoch${max_epochs}_lr${lr}_${i}"
    python run.py \
    --arch roberta_base \
    --task $task \
    --restore-file  ./models/roberta.base/model.pt \
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
# for lr in 5e-7 1e-6 1.5e-6 2e-6
# do
# done
# for dropout in 0.1 0.2
# do
# done
# for attention_dropout in 0.0 0.1
# do
# done
# --lr $lr \
# --attn_dropout ${attention_dropout} \
# # --dropout ${dropout}
# --max_epoch 1 \
# --output-dir outputs/test
