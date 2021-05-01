export CUDA_VISIBLE_DEVICES=3 
ROBERTA_PATH=./robera.base/model.pt
TASK=MRPC

for task in "CoLA" "RTE" "STS-B" "SST-2" "QQP"
do


for i in 1 2 3
do
python run.py \
--arch roberta_base \
--task $TASK \
--no-save \
--iteration $i
done

done
