export CUDA_VISIBLE_DEVICES=3 
ROBERTA_PATH=./robera.base/model.pt

# exp_filename='/home/NPRC/NN_exp_neuron'${neuron}'.pth'
# div_filename='./NN_divide_neuron'${neuron}'.pth'
gelu_filename='/home/hs402/NPRC/nn_approx/lut_file/0519/NN_gelu_neuron15.pth'
exp_filename='/home/hs402/NPRC/nn_approx/lut_file/NN_exp_neuron32.pth'
div_filename='/home/hs402/NPRC/nn_approx/lut_file/NN_divide_neuron32.pth'
layernorm_filename='/home/hs402/NN_sqrt_div_neuron15.pth'
# softmax_type='nn'
# gelu_type='nn'
layernorm_type='nn'
# softmax_type='nn'
model_path="/home/hs402/fp_mrpc.pt"
output_dir="valid_outputs"
task="MRPC"  #"RTE" #"QQP" #"CoLA" #"QQP" #"CoLA" #"SST-2" #"QQP" #"RTE"


python run_eval.py \
--arch roberta_base \
--path $model_path \
--task $task \
--bs 32 \
--output-dir '../valid_outputs' \
--layernorm_type $layernorm_type \
--layernorm_filename $layernorm_filename
# --gelu_filename $gelu_filename \
# --gelu_type $gelu_type 
# --exp_filename $exp_filename \
# --div_filename $div_filename \
# --softmax_type $softmax_type 




# --gelu_filename $gelu_filename \
# --gelu_type $gelu_type 
# --layernorm_type $layernorm_type 



