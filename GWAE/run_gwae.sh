mkdir -p logs_final
#python train.py settings/gwae.yaml | tee logs/log_celeba_rasgw.txt
python train.py settings/gwae.yaml 2>&1 | tee logs_final/log_mnist_ebsgw_till_conv_1.txt

