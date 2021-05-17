#export CUDA_VISIBLE_DEVICES=''
export CUDA_VISIBLE_DEVICES=2



neuron=128
wandb_project="0514_IBERT_8bit_reproduce"
wandb_on='False'

exp_filename='./NN_exp_neuron'${neuron}'.pth'
div_filename='./NN_divide_neuron'${neuron}'.pth'
gelu_filename='./NN_gelu_neuron'${neuron}'.pth'
softmax_type="nn"



for task in "RTE"
do
for i in 1
do
python run.py \
--arch roberta_base \
--task $task \
--no-save \
--output-dir "../ssd/IBERT/nprc_freezed" \
--seed $RANDOM \
--iteration $i \
--exp_filename $exp_filename \
--div_filename $div_filename \
--softmax_type $softmax_type \
--gelu_filename $gelu_filename \
--gelu_type $gelu_type

done;done

