export CUDA_VISIBLE_DEVICES=1

for lr in 1e-6 #1.5e-6 2e-6 1e-5 2e-5 
do
    for i in 1 #2 3
    do
        python run.py \
        --arch roberta_base \
        --task MRPC \
        --restore-file  ../ssd/roberta.base/model.pt \
        --lr $lr \
        --no-save \
        --iteration $i \
        --output-dir ../ssd/outputs
    done
done
