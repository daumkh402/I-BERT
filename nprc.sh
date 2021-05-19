#export CUDA_VISIBLE_DEVICES=''
export CUDA_VISIBLE_DEVICES=2



neuron=15

# exp_filename='/home/NPRC/NN_exp_neuron'${neuron}'.pth'
# div_filename='./NN_divide_neuron'${neuron}'.pth'
gelu_filename='/home/hs402/NPRC/nn_approx/lut_file/0519/NN_gelu_neuron15.pth'
# softmax_type='nn'
gelu_type='nn'

for task in "MRPC"
do
for i in 1
do
python run_eval.py \
--arch roberta_base \
--task $task \
--no-save \
--output-dir "../valid_outputs" \
--seed $RANDOM \
--iteration $i \
--exp_filename $exp_filename \
--div_filename $div_filename \
--gelu_filename $gelu_filename \
--softmax_type $softmax_type \
--gelu_type $gelu_type \
--layernorm_type $layernorm_type 
done;done

