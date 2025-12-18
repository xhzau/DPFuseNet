"""
Author: Benny
Date: Nov 2019
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
# import os
from data_utils.data import PartPlants,Cabbage_2_cls
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import shutil
from torch.nn.parallel import DataParallel
from modules import transform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

sys.path.append("./")
# seg_classes = {'Tomato_seed': [0,1,2]}
seg_classes = {"Corn":[0,1]}
# seg_classes = {"Maize":[0,1,2]}
# seg_classes = {"Soybean":[0,1]}
# seg_classes = {'ptomato': [0,1,2]}
# seg_classes = {'Potato': [0,1]}
# seg_classes = {'Tomato': [0,1,2]}
# seg_classes = {'Cabbage': [0,1,2,3,4]}
# seg_classes = {'Cabbage': [1,2]}
# seg_classes = {'Sugarcane': [0,1]}   #Sugarcane 032501

# seg_classes = {'Rapeseed': [0,1,2,3]}
# seg_classes = {'Rice': [0,1]}
# seg_classes = {'Cotton': [0,1,2]}
# seg_classes = {'Cotton': [0,1,2]}
# seg_classes = {"Maize":[0,1]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pvdst_part_seg_v4', help='model name')
    parser.add_argument('--batch_size', type=int, default=5, help='batch Size during training')
    parser.add_argument('--warmup_epoch', default=100, type=int, help='warmup epoch')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--log_dir', type=str, default='/mnt/data1/new_work/PVDST-main/log/part_seg/loop=10/1028_Abliation/Corn_1202/Corn_1202/test1', help='log path')
    parser.add_argument('--npoint', type=int, default=10000, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--ckpts', type=str,
                        default="/mnt/data1/new_work/PVDST-main/log/part_seg/loop=10/1028_Abliation/Corn_1202/Corn_1202/checkpoints/best_model.pth",
                        help='ckpts')
    parser.add_argument('--root', type=str, default='/mnt/data1/new_work/data', help='data root')#/public/xiek/data/straw_10W_labeled sugarcane  #/public/xiek/data/Corn
    parser.add_argument('--class_choice', type=str, default="Corn", help='choose a class to train or test')#/mnt/data1/new_work/pvdst_vis/Cabbage_vis
    parser.add_argument('--sample_num', type=int, default=10000, help='choose a sample number')
    parser.add_argument('--aug', type=bool, default=False, help='data augmentation')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--save_folder', type=str, default='/mnt/data1/new_work/PVDST-main/log/part_seg/loop=10/1028_Abliation/Corn_1202/Corn_1202/test1/output',#./log/part_seg/plant_mae_finetune/point_m2ae_tomato_base_loop30_128/tomato_paper/output
                        help='save pred_data_path')
    parser.add_argument('--save_pred_label', type=bool, default=True, help=' ')
    parser.add_argument('--num_classes', type=int, default=10, help=' ')
    parser.add_argument('--num_shapes', type=int, default=10, help=' ')


    return parser.parse_args()


def intersectionAndUnion(output, target, K):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def st_caclucate_ious(pred, label, num_parts):
    intersection, union, target = intersectionAndUnion(np.concatenate(pred), np.concatenate(label),
                                                       num_parts)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    # str = 'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc)
    # print("st caculate method!")
    # for i in range(num_parts):
    #     out_str = 'Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i])
    #     print(out_str)
    #
    # print(str)
    return mIoU,mAcc,allAcc,iou_class,accuracy_class


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    #experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % args.log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    traget_root = args.save_folder
    if not os.path.exists(traget_root):
        os.mkdir(traget_root)
    # if args.aug:
    #     test_transform = transform.Compose([
    #         transform.RandomRotate(along_z=True),
    #     ])
    # else:
    #     test_transform=None
    TEST_DATASET = PartPlants(root=args.root, npoints=args.npoint, split='test', class_choice=args.class_choice,
                              normal_channel=False, sample_num=1000, loop=1, transform=None)
    # TEST_DATASET = Cabbage_2_cls(split='test', loop=1, transform=None)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = args.num_classes
    num_part = args.num_shapes #len(seg_classes[args.class_choice])
    # data, label, seg = testDataLoader

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(args.log_dir))
    classifier = MODEL.PVDST_partseg(args).cuda()
    classifier = torch.nn.DataParallel(classifier)
    # classifier = DataParallel(classifier)

    if args.ckpts is not None:
        checkpoint = torch.load(args.ckpts)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        # new_state_dict = {}
        # for k, v in checkpoint['model_state_dict'].items():
        #     new_key = k.replace('module.', '')
        #     new_state_dict[new_key] = v
        # classifier.load_state_dict(new_state_dict)
    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}


        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        seg_pred_all = []
        label_all = []
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat
        iou_per_class = []
        acc_per_class = []

        class_tp = {l: 0 for l in seg_classes[cat]}
        class_fp = {l: 0 for l in seg_classes[cat]}
        class_fn = {l: 0 for l in seg_classes[cat]}
        class_tn = {l: 0 for l in seg_classes[cat]}

        classifier = classifier.eval()
        for batch_id, (points, label, target, cloud_names) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                                   smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()

            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred= classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            # seg = seg_pred
            cur_pred_val = seg_pred.cpu().data.numpy()

            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            correct = np.sum(cur_pred_val == target)
            seg_pred_all.append(cur_pred_val.reshape(-1))
            label_all.append(target.reshape(-1))
            if args.save_pred_label:
                points = points.permute(0, 2, 1).cpu()
                for i in range(len(cloud_names)):
                    # if os.path.exists(os.path.join(traget_root,)f"outputs/partseg_tomato_10k/pred_label/{cloud_names[i]}.txt"):
                    os.makedirs(os.path.join(traget_root, 'pred_label'), exist_ok=True)
                    # raw = read_ply2np(f"/home/zero/sdb4/zhujianzhong/pointnet/zhu_data/sl-sample-scale/{file_names[i]}.ply")
                    pred_cloud = np.concatenate([points[i], cur_pred_val[i].reshape(-1, 1), target[i].reshape(-1, 1)],
                                                axis=1)
                    np.savetxt(os.path.join(traget_root, f"pred_label/{cloud_names[i]}.txt"), pred_cloud)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                part_acc = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    true_positives = np.sum((segp == l) & (segl == l))
                    false_positives = np.sum((segp == l) & (segl != l))
                    false_negatives = np.sum((segp != l) & (segl == l))
                    true_negatives = np.sum((segp != l) & (segl != l))

                    class_tp[l] += true_positives
                    class_fp[l] += false_positives
                    class_fn[l] += false_negatives
                    class_tn[l] += true_negatives
        class_metrics = {}
        for l in seg_classes[cat]:
            tp = class_tp[l]
            fp = class_fp[l]
            fn = class_fn[l]
            accuracy = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * accuracy * recall) / (accuracy + recall) if (accuracy + recall) > 0 else 0
            if tp + fp + fn == 0:
                iou = 1
            else:
                iou = tp / (tp + fp + fn)
            class_metrics[l] = {
                'TP': tp,
                'FP': fp,
                'TN': class_tn[l],
                'FN': fn,
                'Accuracy': accuracy,
                'Recall': recall,
                'F1 Score': f1,
                'IoU': iou
            }
        for l, metrics in class_metrics.items():
            log_string('New Cal!!!!')
            log_string('Class {}'.format(l))
            log_string('Val result: Accuracy/Recall/F1 Score/Iou {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(metrics['Accuracy'], metrics['Recall'], metrics['F1 Score'], metrics['IoU']))

            print(f"Class {l}:")
            print(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, TN: {metrics['TN']}, FN: {metrics['FN']}")
            print(
                f"  Accuracy: {metrics['Accuracy']:.4f}, Recall: {metrics['Recall']:.4f}, F1 Score: {metrics['F1 Score']:.4f}")

        # 计算总的召回率、IoU、精度和 F1 分数
        total_tp = sum(class_tp.values())
        total_fp = sum(class_fp.values())
        total_fn = sum(class_fn.values())
        total_tn = sum(class_tn.values())

        overall_accuracy = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_F1 = (2 * overall_accuracy * overall_recall) / (overall_accuracy + overall_recall) if (overall_accuracy + overall_recall) > 0 else 0

        log_string(
            'Val all result: overall_accuracy/overall_recall/overall_F1 {:.4f}/{:.4f}/{:.4f}.'.format(overall_accuracy, overall_recall,
                                                                                overall_F1))

        # mIoU,mAcc,allAcc,iou_class,accuracy_class = st_caclucate_ious(seg_pred_all, label_all, len(seg_classes[args.class_choice]))
        # log_string('st caculae!!!!')
        # log_string('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        #
        # for i in range(len(iou_class)):
        #     log_string('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))


if __name__ == '__main__':
    args = parse_args()
    main(args)
