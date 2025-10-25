import os
os.environ['YOLO_VERBOSE'] = 'false'

import yaml
from easydict import EasyDict
from .ultralytics import YOLO

from nudt_ultralytics.callbacks.callbacks import callbacks_dict


def load_yaml(load_path):
    with open(load_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
    
def main(args):
    cfg = load_yaml(args.cfg)
    cfg = EasyDict(cfg)
    
    
    # if cfg.mode == 'train':
    #     if cfg.pretrained is None:
    #         model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     else:
    #         model = YOLO(model=cfg.pretrained, task=cfg.task, verbose=cfg.verbose)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     results = model.train(cfg=args.cfg)
    # if cfg.mode == 'train':
    #     if cfg.pretrained is None:
    #         model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     else:
    #         model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose).load(cfg.pretrained)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     results = model.train(cfg=args.cfg)
    # if cfg.mode == 'train':
    #     model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     if cfg.pretrained is None:
    #         results = model.train(cfg=args.cfg)
    #     else:
    #         results = model.train(cfg=args.cfg, pretrained=cfg.pretrained)
    # elif cfg.mode == 'validate':
    #     model = YOLO(model=cfg.pretrained, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     results = model.val(data=cfg.data, imgsz=cfg.imgsz, batch=cfg.batch, conf=cfg.conf, iou=cfg.iou, device=cfg.device, project=cfg.project, name=cfg.name)
    # elif cfg.mode == 'predict':
    #     model = YOLO(model=cfg.pretrained, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     results = model.predict(source="path/to/image.jpg", conf=cfg.conf)
    # elif cfg.mode == 'tune':
    #     model = YOLO(model=cfg.pretrained, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
    #     for (event, func) in callbacks_dict.items():
    #         model.add_callback(event, func)
    #     results = model.tune(data=cfg.data, iterations=5)
    
    if args.process == 'train':
        model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        for (event, func) in callbacks_dict.items():
            model.add_callback(event, func)
        results = model.train(cfg=args.cfg)
    if args.process == 'defend':
        if cfg.pretrained is None:
            model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        else:
            model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose).load(cfg.pretrained)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        for (event, func) in callbacks_dict.items():
            model.add_callback(event, func)
        results = model.train(cfg=args.cfg)
    elif args.process == 'attack':
        model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose).load(cfg.pretrained)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        for (event, func) in callbacks_dict.items():
            model.add_callback(event, func)
        results = model.val(data=cfg.data, imgsz=cfg.imgsz, batch=cfg.batch, conf=cfg.conf, iou=cfg.iou, device=cfg.device, project=cfg.project, name=cfg.name)
    elif args.process == 'adv':
        from attacks.attacks import attacks
        att = attacks(cfg)
        att.run_adv(args)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfgs/yolo.yaml', help='cfg yaml file')

    args = parser.parse_args()
    return EasyDict(vars(args))

if __name__ == '__main__':
    args = parse_args()
    main(args)