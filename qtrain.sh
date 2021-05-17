#export CUDA_VISIBLE_DEVICES=''
export CUDA_VISIBLE_DEVICES=0

wandb_project="0518_test"


for task in  "MRPC" #"RTE" "CoLA"
do
for bs in 16
do
for max_epochs in 15
do
for lr in 1e-5 
do
for weight_decay in 0.1
do


    for i in 101
    do
    wandb_run="${task}_bs${bs}_epoch${max_epochs}_lr${lr}_${i}"
    python run.py \
    --arch roberta_base \
    --task $task \
    --restore-file ../ssd/models/roberta.base/model.pt \
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

