"""Main operations

Created by Kunhong Yu
Date: 2021/07/15
"""

from train import train
from eval import test
import fire


def main(**kwargs):
    if 'only_test' in kwargs and kwargs['only_test']:
        test(**kwargs)
    else:
        train(**kwargs)
        test(**kwargs)


if __name__ == '__main__':
    fire.Fire()
    print('\nDone!\n')

    """
    Usage:
    1. Run 
        python main.py main \
            --data_root='./data/' \
            --dataset='cifar10' \
            --input_size=160 \
            --model_name='cmt_ti' \
            --use_gpu=True \
            --n_gpu=1 \
            --batch_size=32 \
            --epochs=160 \
            --optimizer='adamw' \
            --init_lr=1e-5 \
            --gamma=0.2 \
            --milestones=[30,60,90,120,150] \
            --weight_decay=1e-5 \
            --only_test=False

    2. Simply run
        python main.py main --dataset='cifar10' --input_size=160 --model_name='cmt_ti' --batch_size=32 --epochs=160  --only_test=False
    3. Run scripts
        ./run.sh
    """