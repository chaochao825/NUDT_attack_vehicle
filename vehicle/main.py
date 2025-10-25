import argparse
import yaml
from easydict import EasyDict
import os

from utils.sse import sse_input_validated #, sse_working_path_created, sse_source_unzip_completed
from utils.yaml_rw import load_yaml, save_yaml
from nudt_ultralytics.main import main as yolo
from nudt_pytorch_lightning.main import main as badnet

model_list = [
    "yolov5n",
    "yolov8n",
    "yolov10n",
]

dataset_list = [
    "coco8",
    "coco128",
    "coco",
    "imagenet",
]

attach_method_list = [
    "fgsm",
    "bim",
    "pgd",
    "cw",
    "deepfool",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_PATH', type=str, default='./input', help='input path')
    parser.add_argument('--OUTPUT_PATH', type=str, default='./output', help='output path')
    
    parser.add_argument('--process', type=str, default='adv', help='[adv, attack, defend, train]')
    parser.add_argument('--model', type=str, default='yolov8', help='model name')
    parser.add_argument('--data', type=str, default='imagenet10', help='data name')
    # parser.add_argument('--data', type=str, default='coco8', help='data name')
    
    parser.add_argument('--attack_method', type=str, default='cw', help='attack method [cw, deepfool, bim, fgsm, pdg]')
    parser.add_argument('--defend_method', type=str, default=None, help='defend method')
    
    parser.add_argument('--task', type=str, default='classify', help='task name')
    # parser.add_argument('--task', type=str, default='detect', help='task name')
    parser.add_argument('--cfg_path', type=str, default='./cfgs', help='cfg path')
    
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--device', type=int, default=0, help='which gpu for cuda')
    
    parser.add_argument('--epsilon', type=float, default=8/255, help='epsilon for attack method')
    parser.add_argument('--step_size', type=int, default=2/255, help='epsilon for attack method')
    parser.add_argument('--max_iterations', type=int, default=50, help='epsilon for attack method')
    parser.add_argument('--random_start', type=bool, default=False, help='initial random start for attack method')
    parser.add_argument('--loss_function', type=str, default='CrossEntropy', help='loss function for attack method')
    parser.add_argument('--optimization_method', type=str, default='Adam', help='optimization for attack method')
    
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict_lower = {}
    for key, value in args_dict.items():
        args_dict_lower[f'{key.lower()}'] = value
    args_easydict = EasyDict(args_dict_lower)
    return args_easydict



def yolo_cfg(args):
    
    model = f'./nudt_ultralytics/cfgs/models/{args.task}/{args.model}.yaml'
    os.system(f'cp {model} {args.cfg_path}')
    
    data = f'./nudt_ultralytics/cfgs/datasets/{args.data}.yaml'
    dataset = load_yaml(data)
    dataset['path'] = f'{args.input_path}/datasets/{args.data}'
    data = f'{args.cfg_path}/{args.data}.yaml'
    save_yaml(dataset, data)
    
    cfg_file = f'./nudt_ultralytics/cfgs/models/{args.task}/cfg.yaml'
    cfg = load_yaml(cfg_file)
    cfg = EasyDict(cfg)
    
    cfg.task = args.task
    cfg.model = f'{args.cfg_path}/{args.model}.yaml'
    if args.task == 'classify':
        cfg.data = f'{args.input_path}/datasets/{args.data}'
    else:
        cfg.data = f'{args.cfg_path}/{args.data}.yaml'
    cfg.save_dir = args.output_path
    
    if args.process == 'adv':
        cfg.mode = 'predict'
        cfg.batch = 1
        cfg.pretrained = f'{args.input_path}/weights/{args.model}.pt'
    elif args.process == 'attack':
        cfg.mode = 'validate'
        cfg.batch = 1
        cfg.pretrained = f'{args.input_path}/weights/{args.model}.pt'
    elif args.process == 'defend':
        cfg.mode = 'train'
        cfg.epochs = args.epochs
        cfg.batch = args.batch
        cfg.device = args.device
    elif args.process == 'train':
        cfg.mode = 'train'
        cfg.epochs = args.epochs
        cfg.batch = args.batch
        cfg.device = args.device
    
    args.cfg = f'{args.cfg_path}/cfg.yaml'
    cfg = dict(cfg)
    save_yaml(cfg, args.cfg)
    
    return args

def main(args):
    if 'yolo' in args.model:
        args = yolo_cfg(args)
        yolo(args)
    else:
        args = yolo_cfg(args)
        badnet(args)
        
if __name__ == '__main__':
    args = parse_args()
    
    sse_input_validated(args.input_path)
    # sse_working_path_created(args.working_path)
    # sse_source_unzip_completed(args.dataset_path, args.working_path)
    main(args)
    
    
    
    
    
    
