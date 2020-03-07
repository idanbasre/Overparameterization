SET TRAIN_DATA_DIR=UCIData
SET LOGS_DIR=Logs
SET LAYER_WIDTH=128,10,1
SET P=2
SET NUM_EPOCHS=10
SET LR=0.001


python -m trainer.task ^
        --train_data_dir %TRAIN_DATA_DIR% ^
        --logs_dir %LOGS_DIR% ^
        --layer_width %LAYER_WIDTH% ^
        --p %P% ^
        --num_epochs %NUM_EPOCHS% ^
        --lr %LR% ^
