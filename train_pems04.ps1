# PEMS04 Training Script for HybridRAFT

# 数据集路径配置
$root_path = "../data/PEMS04/"
$data_path = "PEMS04.csv"

# 模型与任务配置
$model_id = "PEMS04_96_96"
$model = "HybridRAFT"
$data = "custom"
$features = "M" # M: Multivariate predict multivariate

# 序列长度配置
$seq_len = 96
$label_len = 12
$pred_len = 12

# 模型超参数
$e_layers = 2
$d_layers = 1
$factor = 3
$enc_in = 307  # PEMS04 has 307 sensors
$dec_in = 307
$c_out = 307
$d_model = 512
$d_ff = 2048
$n_heads = 8

# 训练参数
$des = "Exp"
$itr = 1
$train_epochs = 30
$num_workers = 0 # Windows下建议设为0，避免多进程报错
$batch_size = 64
$learning_rate = 0.0001

# 执行训练命令
python run.py `
    --is_training 1 `
    --root_path $root_path `
    --data_path $data_path `
    --model_id $model_id `
    --model $model `
    --data $data `
    --features $features `
    --seq_len $seq_len `
    --label_len $label_len `
    --pred_len $pred_len `
    --e_layers $e_layers `
    --d_layers $d_layers `
    --factor $factor `
    --enc_in $enc_in `
    --dec_in $dec_in `
    --c_out $c_out `
    --d_model $d_model `
    --d_ff $d_ff `
    --n_heads $n_heads `
    --des $des `
    --itr $itr `
    --train_epochs $train_epochs `
    --num_workers $num_workers `
    --batch_size $batch_size `
    --learning_rate $learning_rate `
    --patience 5 `
    --loss MAE `
    --lradj cosine
