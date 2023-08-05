#!/bin/bash/

python train.py dataset=synth trainer.gpus=2 ckpt_name=clipocr_synth model=clipocr model.batch_size=160 trainer.val_check_interval=1.0 model.lr=0.0014 trainer.max_epochs=20 +trainer.limit_train_batches=0.20496

python test.py /path/to/ckpt/clipocr_synth/run/checkpoints --data_root path/to/data

