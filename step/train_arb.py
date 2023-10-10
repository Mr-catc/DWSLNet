import os
import cv2
from apex import amp
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import importlib

import voc12.dataloader
from misc import pyutils, torchutils

def validate(model, data_loader):
    print('validating ... ', flush=True, end='')

    val_loss_meter = pyutils.AverageMeter('loss', 'loss1', 'loss2', 'loss3')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            logits1, cam1, logits2, cam2 = model(img)
            losscls1 = F.multilabel_soft_margin_loss(logits1, label)
            losscls2 = F.multilabel_soft_margin_loss(logits2, label)

            loss_cps = torch.mean(torch.abs(cam1[1:,:,:]-cam2[1:,:,:]))

            loss1 = losscls2 + 0.1 * loss_cps

            val_loss_meter.add({'loss': loss1.item()})
            val_loss_meter.add({'loss1': losscls1.item()})
            val_loss_meter.add({'loss2': losscls2.item()})
            val_loss_meter.add({'loss3': loss_cps.item()})

    model.train()

    print('loss: %.4f' % (val_loss_meter.pop('loss')))
    print('loss1: %.4f' % (val_loss_meter.pop('loss1')))
    print('loss2: %.4f' % (val_loss_meter.pop('loss2')))
    print('loss3: %.4f' % (val_loss_meter.pop('loss3')))

    return


def run(args):

    model = getattr(importlib.import_module(args.arb_network), 'Net')()

    model.load_state_dict(torch.load(args.lb_weights_name + '.pth'), strict=False)

    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    train_data_loader = DataLoader(train_dataset, batch_size=args.arb_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    max_step = (len(train_dataset) // args.arb_batch_size) * args.arb_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset, batch_size=args.arb_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
 
    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.arb_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * args.arb_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.arb_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    model.to("cuda")
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(args.arb_num_epoches):

        print('Epoch %d/%d' % (ep+1, args.arb_num_epoches))

        for step, pack in enumerate(train_data_loader):

            img = pack['img']
            img = img.cuda()
            label = pack['label'].cuda(non_blocking=True)
            model.zero_grad()
            logits1, cam1, logits2, cam2 = model(img)

            optimizer.zero_grad()

            losscls2 = F.multilabel_soft_margin_loss(logits2, label)

            loss_cps = torch.mean(torch.abs(cam1 - cam2))

            loss = losscls2 + 0.1 * loss_cps

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            avg_meter.add({'loss1': loss.item()})


            optimizer.step()
            if (optimizer.global_step-1)%100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % ((step + 1) * args.arb_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.arb_weights_name + '.pth')
    torch.cuda.empty_cache()