"""Parsing the parameters."""
import argparse

import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GETNext.")
    parser.add_argument('--max-seq-len',type=int,default=100,help='max seq len')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed')

    parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='')
    parser.add_argument('--dataset',
                        type=str,
                        default='dataset/NYC/NYC.csv',
                        help='dataset path')

    parser.add_argument('--embed-mode',
                        type=str,default='poi-sage',)
    parser.add_argument('--pure-transformer',
                        type=bool, default=True)
    parser.add_argument('--cpus',type=int,default=4)
    parser.add_argument('--geo-k',
                        type=int,
                        default=10,
                        help='geo distance less than geo_dis regarded as context poi')
    parser.add_argument('--restart-prob',
                        type=float,
                        default=0.5,
                        help='random walk with restart prob')
    parser.add_argument('--num-walks',
                        type=int,
                        default=5,
                        help='random walk with restart step')

    # Model hyper-parameters
    parser.add_argument('--poi-id-dim',
                        type=int,
                        default=100,
                        help='POI embedding dimensions')
    parser.add_argument('--poi-sage-dim',
                        type=int,
                        default=120,
                        help='POI embedding dimensions')
    parser.add_argument('--user-embed-dim',
                        type=int,
                        default=128,
                        help='User embedding dimensions')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for gru')


    parser.add_argument('--time-embed-dim',
                        type=int,
                        default=32,
                        help='Time embedding dimensions')
    parser.add_argument('--cat-embed-dim',
                        type=int,
                        default=32,
                        help='Category embedding dimensions')
    parser.add_argument('--time-loss-weight',
                        type=int,
                        default=10,
                        help='Scale factor for the time loss term')


    # Training hyper-parameters
    parser.add_argument('--batch',
                        type=int,
                        default=20,
                        help='Batch size.')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor',
                        type=float,
                        default=0.1,
                        help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    # Experiment config
    parser.add_argument('--save-weights',
                        action='store_true',
                        default=True,
                        help='whether save the model')
    parser.add_argument('--save-embeds',
                        action='store_true',
                        default=False,
                        help='whether save the embeddings')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Num of workers for dataloader.')
    parser.add_argument('--project',
                        default='runs/train',
                        help='save to project/name')
    parser.add_argument('--name',
                        default='exp',
                        help='save to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--mode',
                        type=str,
                        default='client',
                        help='python console use only')
    parser.add_argument('--port',
                        type=int,
                        default=64973,
                        help='python console use only')

    return parser.parse_args()
