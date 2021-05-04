export CUDA_VISIBLE_DEVICES=0 
ROBERTA_PATH=./robera.base/model.pt

for task in  "QQP" #"CoLA" #"QQP" #"CoLA" #"SST-2" #"QQP" #"RTE" 
do
for i in 2 3 4
do
python run.py \
--arch roberta_base \
--task $task \
--iteration $i \
--seed $RANDOM \
--no-save 
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
