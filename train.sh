export CUDA_VISIBLE_DEVICES=3 
ROBERTA_PATH=./robera.base/model.pt
TASK=MRPC

for i in 1
do
python run.py \
--arch roberta_base \
--task $TASK \
--max-epoch 12 \
--no-save \
--iteration $i
done
