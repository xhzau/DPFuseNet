"""
Train for part segmentation with evaluation aligned to test-time metrics (no voting).
- Per-class Precision/Recall/F1/IoU computed ONLY over valid labels from seg_classes
- Best checkpoint selected by dataset_avg_miou (mean IoU over valid labels)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import datetime
import logging
import sys
import importlib
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

# ====== your project deps ======
sys.path.append("./")
sys.path.append("./models")
from data_utils.data import PartPlants
from modules import transform


# ----------------------------
# Dataset categories & labels
# ----------------------------
# 只评 seg_classes 中的“有效标签集合”；下面示例与你给的一致
# seg_classes = {'ptomato': [0, 1, 2]}
# seg_classes = {'Cabbage': [0,1,2,3,4]}
seg_classes = {'Soybean': [0,1]}

seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    # optional: in-place ReLUs
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """one-hot encode a tensor of class indices"""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pvdst_part_seg_v4', help='model name')
    parser.add_argument('--batch_size', type=int, default=3
                        , help='batch Size during training')
    parser.add_argument('--epoch', default=400, type=int, help='epoch to run')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--log_dir', type=str, default='/mnt/data1/new_work/PVDST-main/log/part_seg/loop=10/1028_Abliation/Soybean_all_11-11',
                        help='log path')
    parser.add_argument('--npoint', type=int, default=10000, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--ckpts', type=str, default="", help='ckpts')
    parser.add_argument('--root', type=str, default='/mnt/data1/new_work/data',
                        help='data root')
    parser.add_argument('--class_choice', type=str, default="Soybean", help='choose a class to train or test')
    parser.add_argument('--sample_num', type=int, default=1000, help='choose a sample number')
    parser.add_argument('--aug', type=bool, default=True, help='data augmentation')
    parser.add_argument('--num_classes', type=int, default=10, help='choose a class to train or test')
    parser.add_argument('--num_shapes', type=int, default=10, help='choose a class to train or test')

    return parser.parse_args()


# =========================
#  EVALUATION (NO VOTING)
# =========================
@torch.no_grad()
def evaluate(classifier,
                               data_loader,
                               device,
                               args,
                               seg_classes,
                               seg_label_to_cat,
                               num_logits,
                               log_fn=print):

    classifier.eval()

    valid_labels = sorted({l for labels in seg_classes.values() for l in labels})
    if len(valid_labels) == 0:
        raise ValueError("seg_classes is empty; cannot evaluate.")
    L = max(valid_labels) + 1  # allocate arrays up to max label id; only index valid_labels

    total_correct = 0
    total_seen = 0

    total_seen_class = np.zeros(L, dtype=np.int64)
    total_correct_class = np.zeros(L, dtype=np.int64)

    class_tp = np.zeros(L, dtype=np.int64)
    class_fp = np.zeros(L, dtype=np.int64)
    class_fn = np.zeros(L, dtype=np.int64)
    class_tn = np.zeros(L, dtype=np.int64)

    # per-category (coarse type) shape-wise IoU (as in your test code)
    shape_ious = {cat: [] for cat in set(seg_label_to_cat.values())}

    for batch_id, (points, label, target, cloud_names) in enumerate(data_loader):
        B, N, _ = points.size()
        points = points.float().to(device)
        label = label.long().to(device)
        target = target.long().to(device)
        points = points.transpose(2, 1)  # (B, 3, N)

        # forward once (NO VOTING)
        seg_pred = classifier(points, to_categorical(label, args.num_classes))  # (B, C, N) or (B, N, C)

        # unify to (B, N, C)
        if seg_pred.dim() == 3 and seg_pred.size(1) == num_logits and seg_pred.size(2) == N:
            seg_pred = seg_pred.transpose(1, 2).contiguous()  # (B, N, C)

        logits_np = seg_pred.detach().cpu().numpy()  # (B, N, C)
        pred_np = np.zeros((B, N), dtype=np.int32)
        target_np = target.detach().cpu().numpy()    # (B, N)

        # argmax within valid part set of its category, then offset back to global labels
        for i in range(B):
            cat = seg_label_to_cat[target_np[i, 0]]
            cands = seg_classes[cat]                 # e.g., [0,1,2]
            logits_i = logits_np[i, :, :]            # (N, C)
            pred_np[i, :] = np.argmax(logits_i[:, cands], axis=1) + cands[0]

        # overall accuracy (point-wise)
        total_correct += np.sum(pred_np == target_np)
        total_seen += (B * N)

        # per-class accuracy counting (only valid labels)
        for l in valid_labels:
            total_seen_class[l] += np.sum(target_np == l)
            total_correct_class[l] += np.sum((pred_np == l) & (target_np == l))

        # shape-wise IoU & class TP/FP/FN/TN (only valid labels inside the sample's category set)
        for i in range(B):
            segp = pred_np[i, :]
            segl = target_np[i, :]
            cat = seg_label_to_cat[segl[0]]
            cands = seg_classes[cat]
            part_ious = [0.0 for _ in range(len(cands))]
            for idx, l in enumerate(cands):
                tp = np.sum((segp == l) & (segl == l))
                fp = np.sum((segp == l) & (segl != l))
                fn = np.sum((segp != l) & (segl == l))
                tn = np.sum((segp != l) & (segl != l))
                class_tp[l] += tp; class_fp[l] += fp; class_fn[l] += fn; class_tn[l] += tn

                # test-time rule: if both gt & pred absent for this part in this sample, IoU=1
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                    iou_l = 1.0
                else:
                    denom = tp + fp + fn
                    iou_l = (tp / denom) if denom > 0 else 0.0
                part_ious[idx] = iou_l
            shape_ious[cat].append(float(np.mean(part_ious)))

    # aggregate per-category (shape-wise)
    all_shape_ious = []
    for cat in shape_ious.keys():
        all_shape_ious.extend(shape_ious[cat])
        shape_ious[cat] = float(np.mean(shape_ious[cat])) if len(shape_ious[cat]) else 0.0

    class_avg_iou = float(np.mean(list(shape_ious.values()))) if len(shape_ious) else 0.0
    instance_avg_iou = float(np.mean(all_shape_ious)) if len(all_shape_ious) else 0.0
    overall_accuracy = total_correct / float(total_seen) if total_seen > 0 else 0.0

    # per-class metrics (ONLY over valid_labels)
    per_class = {}
    P_list, R_list, F1_list, IoU_list = [], [], [], []
    for l in valid_labels:
        tp, fp, fn, tn = class_tp[l], class_fp[l], class_fn[l], class_tn[l]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        per_class[l] = dict(Precision=precision, Recall=recall, F1=f1, IoU=iou,
                            TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))
        P_list.append(precision); R_list.append(recall); F1_list.append(f1); IoU_list.append(iou)

    dataset_avg_precision = float(np.mean(P_list)) if P_list else 0.0
    dataset_avg_recall    = float(np.mean(R_list)) if R_list else 0.0
    dataset_avg_f1        = float(np.mean(F1_list)) if F1_list else 0.0
    dataset_avg_miou      = float(np.mean(IoU_list)) if IoU_list else 0.0  # ★ used for best model

    return dict(
        overall_accuracy=overall_accuracy,
        class_avg_iou=class_avg_iou,
        instance_avg_iou=instance_avg_iou,
        dataset_avg_precision=dataset_avg_precision,
        dataset_avg_recall=dataset_avg_recall,
        dataset_avg_f1=dataset_avg_f1,
        dataset_avg_miou=dataset_avg_miou,     # ★ selection metric
        per_class=per_class,
        per_cat_shape_iou=shape_ious,
        valid_labels=valid_labels
    )


def main(args):
    # --------- logging & dirs ----------
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/').resolve()
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir / f'{args.model}.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(s):  # unified print + logger
        logger.info(s)
        print(s)

    log_string('PARAMETERS ...')
    log_string(str(args))
    writer = SummaryWriter(str(log_dir))

    # --------- data ---------
    if args.aug:
        train_transform = transform.Compose([
            transform.RandomRotate(along_z=True),
            transform.RandomScale(scale_low=0.8, scale_high=1.2),
            transform.RandomJitter(sigma=0.01, clip=0.05),
            transform.RandomShift()
        ])
        test_transform = transform.Compose([
            transform.RandomRotate(along_z=True)
        ])
    else:
        train_transform = None
        test_transform = None

    TRAIN_DATASET = PartPlants(root=args.root, npoints=args.npoint, split='trainval',
                               class_choice=args.class_choice, normal_channel=False,
                               sample_num=args.sample_num, loop=10, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,
                                               shuffle=True, num_workers=28, drop_last=True)

    TEST_DATASET = PartPlants(root=args.root, npoints=args.npoint, split='test',
                              class_choice=args.class_choice, normal_channel=False,
                              sample_num=1000, loop=1, transform=None)
    test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,
                                              shuffle=False, num_workers=28)

    log_string(f"The number of training data is: {len(TRAIN_DATASET)}")
    log_string(f"The number of test data is: {len(TEST_DATASET)}")

    # --------- model ---------
    MODEL = importlib.import_module(args.model)
    # backup
    os.makedirs(str(exp_dir), exist_ok=True)
    shutil.copy(f'./models/{args.model}.py', str(exp_dir))
    shutil.copy('models/pvdst_utils_3.py', str(exp_dir))

    classifier = MODEL.PVDST_partseg(args).cuda()
    classifier = torch.nn.DataParallel(classifier)
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    log_string(f'# params: {sum(p.numel() for p in classifier.parameters())}')

    # --------- optimizer & sched ---------
    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [{'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

    param_groups = add_weight_decay(classifier, weight_decay=0.05)
    optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=0.05)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epoch,
        t_mul=1,
        lr_min=1e-6,
        decay_rate=0.1,
        warmup_lr_init=1e-6,
        warmup_t=args.warmup_epoch,
        cycle_limit=1,
        t_in_epochs=True
    )

    # --------- training state ---------
    best_dataset_avg_miou = 0.0  # ★ selection metric
    best_overall_acc = 0.0
    best_class_avg_iou = 0.0
    best_instance_avg_iou = 0.0

    # --------- train loop ---------
    device = torch.device('cuda')
    global_epoch = 0

    for epoch in range(args.epoch):
        classifier.train()
        losses = []
        correct_points = []
        log_string(f'Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch})')

        for batch_id, (points, label, target, _) in tqdm(enumerate(train_loader),
                                                         total=len(train_loader), smoothing=0.9):
            points = points.float().cuda(non_blocking=True)           # (B,N,3[+C])
            label = label.long().cuda(non_blocking=True)
            target = target.long().cuda(non_blocking=True)
            points = points.transpose(2, 1)                            # (B,3(+C),N)

            optimizer.zero_grad()
            seg_pred = classifier(points, to_categorical(label, args.num_classes))  # (B,C,N)
            seg_pred = seg_pred.contiguous().view(-1, args.num_shapes)              # (B*N, C)
            target_flat = target.view(-1, 1)[:, 0]                                  # (B*N,)

            loss = criterion(seg_pred, target_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=10.0)
            optimizer.step()

            # train acc (point-wise, rough)
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target_flat.data).cpu().sum().item()
            acc = correct / float(points.size(0) * args.npoint)
            correct_points.append(acc)
            losses.append(loss.detach().cpu().item())

        # step scheduler
        if isinstance(scheduler, list):
            for sch in scheduler:
                sch.step(epoch)
        else:
            scheduler.step(epoch)

        train_acc = float(np.mean(correct_points)) if len(correct_points) else 0.0
        train_loss = float(np.mean(losses)) if len(losses) else 0.0
        log_string(f'Train accuracy is: {train_acc:.5f}')
        log_string(f'Train loss: {train_loss:.5f}')
        log_string(f'lr: {optimizer.param_groups[0]["lr"]:.6f}')
        writer.add_scalar('train/acc', train_acc, epoch + 1)
        writer.add_scalar('train/loss', train_loss, epoch + 1)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch + 1)

        # -------------- EVAL (no voting; aligned with test) --------------
        metrics = evaluate(
            classifier=classifier,
            data_loader=test_loader,
            device=device,
            args=args,
            seg_classes=seg_classes,
            seg_label_to_cat=seg_label_to_cat,
            num_logits=args.num_shapes,
            log_fn=log_string
        )

        # per-class
        log_string('---- Per-class (Precision / Recall / F1 / IoU) ----')
        for l in metrics['valid_labels']:
            m = metrics['per_class'][l]
            log_string(f'Class {l:>3d}: {m["Precision"]:.4f} / {m["Recall"]:.4f} / {m["F1"]:.4f} / {m["IoU"]:.4f}')

        # per-category (shape-wise)
        log_string('---- Class-wise mean IoU (per-category) ----')
        for cat in sorted(metrics['per_cat_shape_iou'].keys()):
            log_string(f'{cat:<14s}: {metrics["per_cat_shape_iou"][cat]:.4f}')

        # dataset averages (from per-class over valid labels)
        log_string('---- Dataset averages (from per-class) ----')
        log_string('Avg Precision: %.4f  Avg Recall: %.4f  Avg F1: %.4f  class-avg mIoU: %.4f'
                   % (metrics['dataset_avg_precision'], metrics['dataset_avg_recall'],
                      metrics['dataset_avg_f1'], metrics['dataset_avg_miou']))

        # overview (keep for reference)
        log_string('Epoch %d  Overall Acc: %.4f  class-avg mIoU: %.4f  instance-avg mIoU: %.4f'
                   % (epoch + 1, metrics['overall_accuracy'], metrics['class_avg_iou'], metrics['instance_avg_iou']))

        # tensorboard
        writer.add_scalar('eval/overall_acc', metrics['overall_accuracy'], epoch + 1)
        writer.add_scalar('eval/class_avg_mIoU', metrics['class_avg_iou'], epoch + 1)
        writer.add_scalar('eval/instance_avg_mIoU', metrics['instance_avg_iou'], epoch + 1)
        writer.add_scalar('eval/avg_precision', metrics['dataset_avg_precision'], epoch + 1)
        writer.add_scalar('eval/avg_recall', metrics['dataset_avg_recall'], epoch + 1)
        writer.add_scalar('eval/avg_f1', metrics['dataset_avg_f1'], epoch + 1)
        writer.add_scalar('eval/dataset_avg_mIoU', metrics['dataset_avg_miou'], epoch + 1)

        # -------------- Selection by dataset_avg_miou --------------
        save_is_best = False
        if metrics['dataset_avg_miou'] >= best_dataset_avg_miou:
            best_dataset_avg_miou = metrics['dataset_avg_miou']
            save_is_best = True

        if save_is_best:
            savepath = str(checkpoints_dir / 'best_model.pth')
            log_string(f'Saving best (by dataset_avg_mIoU) to {savepath}')
            state = {
                'epoch': epoch,
                'overall_acc': metrics['overall_accuracy'],
                'class_avg_iou': metrics['class_avg_iou'],
                'instance_avg_iou': metrics['instance_avg_iou'],
                'dataset_avg_precision': metrics['dataset_avg_precision'],
                'dataset_avg_recall': metrics['dataset_avg_recall'],
                'dataset_avg_f1': metrics['dataset_avg_f1'],
                'dataset_avg_miou': metrics['dataset_avg_miou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

        # track best (for display only)
        best_overall_acc = max(best_overall_acc, metrics['overall_accuracy'])
        best_class_avg_iou = max(best_class_avg_iou, metrics['class_avg_iou'])
        best_instance_avg_iou = max(best_instance_avg_iou, metrics['instance_avg_iou'])

        log_string('Best Overall Acc: %.5f' % best_overall_acc)
        log_string('Best class-avg mIoU: %.5f' % best_class_avg_iou)
        log_string('Best instance-avg mIoU: %.5f' % best_instance_avg_iou)
        log_string('Best dataset-avg mIoU (selection metric): %.5f' % best_dataset_avg_miou)

        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)