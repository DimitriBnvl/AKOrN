# Training Command for Mac
python train_sudoku.py --exp_name=sudoku_akorn --eval_freq=10 --epochs=100 --model=akorn --lr=0.001 --T=16 --use_omega=True --global_omg=True --init_omg=0.5 --learn_omg=True --num_workers=0 --checkpoint_every=10

# Training Command for Linux / CUDA (with checkpoints)
python train_sudoku.py --exp_name=sudoku_akorn  --eval_freq=10 --epochs=100 --model=akorn --lr=0.001 --T=16 --use_omega=True --global_omg=True --init_omg=0.5 --learn_omg=True --checkpoint_every=10

# Evaluation
export data=ood # id or ood

# Inference with test-time extension of the Kuramoto updates. (Accuracy: 51.7%)
python eval_sudoku.py --data=${data} --model=akorn --model_path=runs/sudoku_akorn/ema_99.pth --T=128
# Test-time extension and energy-based voting (Accuracy: 81.6%)
python eval_sudoku.py --data=${data} --model=akorn --model_path=runs/sudoku_akorn/ema_99.pth --T=128 --K=100 --evote_type=sum
# Number of random samples increased from 100 to 4096 (best results) (Accuracy: 89.5%)
python eval_sudoku.py --data=${data} --model=akorn --model_path=runs/sudoku_akorn/ema_99.pth --T=128 --K=4096 --evote_type=sum
