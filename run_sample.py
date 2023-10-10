import argparse
import os
import numpy as np
import torch
import random
from misc import pyutils

if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    seed = 8000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--voc12_root", default='../dataset/dataset1/', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="voc12/train.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str,
                        help="voc12/train.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Mapc
    parser.add_argument("--lb_network", default="net.resnet50_lb", type=str)
    parser.add_argument("--arb_network", default="net.resnet50_arb", type=str)
    parser.add_argument("--lb_crop_size", default=512, type=int)
    parser.add_argument("--lb_batch_size", default=16, type=int) # original: 16
    parser.add_argument("--lb_num_epoches", default=1, type=int)
    parser.add_argument("--lb_learning_rate", default=0.0002, type=float)

    parser.add_argument("--arb_batch_size", default=16, type=int)
    parser.add_argument("--arb_num_epoches", default=1, type=int)
    parser.add_argument("--arb_learning_rate", default=0.02, type=float)

    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.20, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0),
                        help="Multi-scale inferences")

    parser.add_argument("--weight", default=0, type=float)

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=16, type=int)
    parser.add_argument("--irn_num_epoches", default=1, type=int)
    parser.add_argument("--irn_learning_rate", default=0.0001, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.4)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--lb_weights_name", default="sess/lb.pth", type=str)
    parser.add_argument("--arb_weights_name", default="sess/arb.pth", type=str)
    parser.add_argument("--irn_weights_name", default="sess/irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="result/cam", type=str)
    parser.add_argument("--ir_label_out_dir", default="result/ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="result/sem_seg", type=str)
    parser.add_argument("--ins_seg_out_dir", default="result/ins_seg", type=str)

    # Step
    parser.add_argument("--train_lb_pass", type=str2bool, default=False)
    parser.add_argument("--train_arb_pass", type=str2bool, default=False)
    parser.add_argument("--make_cam_pass", type=str2bool, default=True)
    parser.add_argument("--eval_cam_pass", type=str2bool, default=True)
    parser.add_argument("--cam_to_ir_label_pass", type=str2bool, default=True)
    parser.add_argument("--train_irn_pass", type=str2bool, default=True)
    parser.add_argument("--make_ins_seg_pass", type=str2bool, default=False)
    parser.add_argument("--eval_ins_seg_pass", type=str2bool, default=False)
    parser.add_argument("--make_sem_seg_pass", type=str2bool, default=True)
    parser.add_argument("--eval_sem_seg_pass", type=str2bool, default=True)

    args = parser.parse_args()

    os.makedirs("sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)
    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_lb_pass is True:
        import step.train_lb

        timer = pyutils.Timer('step.train_lb:')
        step.train_lb.run(args)

    if args.train_arb_pass is True:
        import step.train_arb

        timer = pyutils.Timer('step.train_arb:')
        step.train_arb.run(args)

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)


    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)


    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    if args.make_ins_seg_pass is True:
        import step.make_ins_seg_labels

        timer = pyutils.Timer('step.make_ins_seg_labels:')
        step.make_ins_seg_labels.run(args)

    if args.eval_ins_seg_pass is True:
        import step.eval_ins_seg

        timer = pyutils.Timer('step.eval_ins_seg:')
        step.eval_ins_seg.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels

        args.sem_seg_bg_thres = float(args.sem_seg_bg_thres)
        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)

