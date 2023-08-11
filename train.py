import os
import sys
import argparse
from core.engine import Trainer

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def get_args_parser():
    parser = argparse.ArgumentParser()
    # data, dataset, dataloader, ...
    parser.add_argument("--data-path", type=str, default="../data/DF2K")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--file-check", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hr-size", type=int, default=256)
    parser.add_argument("--lr-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    
    # super-resolution model
    parser.add_argument("--model", type=str, choices=['rlfn', 'rlfn_s', 'abpn'], default='rlfn')
    parser.add_argument("--feature", type=int, default=None)    # "None" means use default value
    parser.add_argument("--load", type=str, default=None)       # 모델의 weight를 로드할 경우 True로 지정
    
    # training cfg
    parser.add_argument("--loss", type=str, default='l1', choices=['l1', 'l2'])
    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--save_interval", type=int, default=10)
    
    # hardware dependency
    parser.add_argument("--device", type=str, default='cuda', choices=['cpu', 'cuda', 'mps'])
    
    return parser

def main(args):
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    # print(args)
    main(args)
    
# python train.py --data-path ../DIV2K --preload --file-check --batch-size 16 --model abpn --device cpu