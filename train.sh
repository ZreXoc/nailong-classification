stamp=$(date "+%m%d%H%M")
nohup bash -c "python train.py" | tee $stamp.log
