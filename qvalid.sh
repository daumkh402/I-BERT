export CUDA_VISIBLE_DEVICES=0 

model_path="../QMRPC.pt"
output_dir="../valid_outputs"
task="MRPC"  #"RTE" #"QQP" #"CoLA" #"QQP" #"CoLA" #"SST-2" #"QQP" #"RTE"

python run_qeval.py \
--arch roberta_base \
--restore-file $model_path \
--task $task \
--bs 32 \
--output-dir $output_dir

