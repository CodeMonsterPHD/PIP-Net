import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--seed',
        type=int,
        default=0)
    parser.add_argument(
        '--Intent_class',
        type=int,
        default=28)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64)
    parser.add_argument(
        '--found_lr',
        type=float,
        default=5e-3)
    # optimizer of Network's learning rate.
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9)
    # optimizer of prototype's learning rate.
    parser.add_argument(
        '--lr_prototype',
        type=float,
        default=1e-3)
    # epochs
    parser.add_argument(
        '--epoch',
        type=int,
        default=40)
    # milestones to reduce lr
    parser.add_argument(
        '--milestones',
        type=list,
        default=[20, 30])
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.15)

    # Overall Dataset settings
    parser.add_argument(
        '--train_json_file',
        type=str,
        default='./data/dataset/annotation/intentonomy_train2020.json')
    parser.add_argument(
        '--val_json_file',
        type=str,
        default='./data/dataset/annotation/intentonomy_val2020.json')
    parser.add_argument(
        '--test_json_file',
        type=str,
        default='./data/dataset/annotation/intentonomy_test2020.json')
    parser.add_argument(
        '--img_dir',
        type=str,
        default='./data/dataset/total/')
    parser.add_argument(
        '--rotation',
        type=int,
        default=5)
    parser.add_argument(
        '--HorizontalFlip',
        type=float,
        default=0.5)
    parser.add_argument(
        '--Crop_padding',
        type=int,
        default=10)

    # K-means setting
    parser.add_argument(
        '--component',
        type=int,
        default=4)
    parser.add_argument(
        '--disgard_rate',
        type=float,
        default=0.2)
    parser.add_argument(
        '--old_rate',
        type=float,
        default=0.999)

    parser.add_argument(
        '--CUDA_DEVICE',
        type=str,
        default='0')

    args = parser.parse_args()

    return args