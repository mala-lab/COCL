import os, argparse
import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.ImbalanceImageNet import LT_Dataset
from models.resnet_imagenet import ResNet50
from datasets.imagenet_ood import ImageNet_ood

from utils.utils import *
from utils.ltr_metrics import *

from test import get_measures
import random

# to prevent PIL error from reading large images:
# See https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162#issuecomment-491115265
# or https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1
    auroc = roc_auc_score(labels, examples)
    aupr_in = average_precision_score(labels, examples)
    labels_rev = np.zeros(len(examples), dtype=np.int32)
    labels_rev[len(pos):] += 1
    aupr_out = average_precision_score(labels_rev, -examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr_in, aupr_out, fpr, pos.mean(), neg.mean()

def val_imagenet():
    '''
    Evaluate ID acc and OOD detection on ImageNet
    '''
    model.eval()

    test_acc_meter = AverageMeter()
    score_list = []
    labels_list = []
    pred_list = []

    with torch.no_grad():
      for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)

            # outlier-class-aware logit calibration
            if args.OLC:
                p = torch.cat((prior, torch.ones(1).cuda()), dim = 0)
                logits = logits - args.tau * p.log()

            probs = F.softmax(logits, dim=1)
            scores = probs[:, -1]
            pred = logits.data[:, :-1].max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            score_list.append(scores.detach().cpu().numpy())
            labels_list.append(targets.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())
            test_acc_meter.append(acc.item())
    

    test_acc = test_acc_meter.avg
    in_scores = np.concatenate(score_list, axis=0)
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)
    
    many_acc, median_acc, low_acc, _ = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=True)

    clean_str = 'ACC: %.4f (%.4f, %.4f, %.4f)' % (test_acc, many_acc, median_acc, low_acc)
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()

    # confidence distribution of correct samples:
    ood_score_list, sc_labels_list = [], [], []

    with torch.no_grad():
        for images, sc_labels in ood_loader:
            images, sc_labels = images.cuda(), sc_labels.cuda()
            logits = model(images)

            # outlier-class-aware logit calibration
            if args.OLC:
                p = torch.cat((prior, torch.ones(1).cuda()), dim = 0)
                logits = logits - args.tau * p.log()

            probs = F.softmax(logits, dim=1)
            scores = probs[:, -1]
            # append loss:
            ood_score_list.append(scores.detach().cpu().numpy())
            sc_labels_list.append(sc_labels.detach().cpu().numpy())
    
    ood_scores = np.concatenate(ood_score_list, axis=0)
    sc_labels = np.concatenate(sc_labels_list, axis=0)
    print('in_scores:', in_scores.shape)
    print('ood_scores:', ood_scores.shape)

    # only tail samples as ID data
    in_scores = in_scores[in_labels>=100]

    # only head samples as ID data
    in_scores = in_scores[in_labels<20]

    # print:
    auroc, aupr_in, aupr_out, fpr95, id_meansocre, ood_meanscore = get_measures(-in_scores, -ood_scores)
    ood_detectoin_str = 'auroc: %.4f, aupr_in: %.4f, aupr_out: %.4f, fpr95: %.4f, ood_meanscore: %.4f, id_meansocre: %.4f' % (auroc, aupr_in, aupr_out, fpr95, ood_meanscore, id_meansocre)
    print(ood_detectoin_str)
    fp.write(ood_detectoin_str + '\n')
    fp.flush()
    fp.close()

    classwise_results_dir = os.path.join(save_dir, 'classwise_results')
    create_dir(classwise_results_dir)
    # classwise acc:
    acc_each_class = np.full(num_classes, np.nan)
    for i in range(num_classes):
        _pred = in_preds[in_labels==i]
        _label = in_labels[in_labels==i]
        _N = np.sum(in_labels==i)
        acc_each_class[i] = np.sum(_pred==_label) / _N
    np.save(os.path.join(classwise_results_dir, 'ACC_each_class.npy'), acc_each_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test an ImageNet Classifier')
    parser.add_argument('--seed', default=25, type=int, help='fix the random seed for reproduction. Default is 25.')
    parser.add_argument('--gpu', default='0,1,2,3', help='which GPU to use.')
    parser.add_argument('--num_workers', type=int, default=32, help='number of threads for data loader')
    # dataset:
    parser.add_argument('--dataset', '--ds', default='imagenet', choices=['imagenet'], help='which dataset to use')
    parser.add_argument('--data_root_path', '--drp', default='./datasets', help='Where you save all your datasets.')
    parser.add_argument('--dout', default='imagenet-10k', choices=['imagenet-10k'], help='which dout to use')
    parser.add_argument('--model', '--md', default='ResNet50', choices=['ResNet50'], help='which model to use')
    # 
    parser.add_argument('--test_batch_size', '--tb', type=int, default=100)
    parser.add_argument('--ckpt_path', default='')
    parser.add_argument('--tnorm', action='store_true', help='If true, use t-norm for LT inference')
    parser.add_argument('--OLC', action='store_true', help='If true, use outlier-class-aware logit calibration for LT inference')
    parser.add_argument('--tau', default='1', type=int, help='hyperparameter to balance prior in OLC')
    args = parser.parse_args()
    print(args)

    # ============================================================================
    # fix random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.OLC:
        save_dir = os.path.join(args.ckpt_path, 'OCL', args.dout)
    elif args.tnorm:
        save_dir = os.path.join(args.ckpt_path, 'tnorm', args.dout)
    else:
        save_dir = os.path.join(args.ckpt_path, 'normal', args.dout)
    create_dir(save_dir)

    # data:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    num_classes = 1000
    train_set = LT_Dataset(
        os.path.join(args.data_root_path, 'ImageNet_LT/train'), './datasets/ImageNet_LT_train.txt', transform=train_transform, 
        class_idx=np.arange(0,1000))
    test_set = LT_Dataset(
            os.path.join(args.data_root_path, 'ImageNet_LT/val'), './datasets/ImageNet_LT_val.txt', transform=test_transform,
            class_idx=np.arange(0,1000))
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, 
                                drop_last=False, pin_memory=True)
    din_str = 'Din is %s with %d images' % (args.dataset, len(test_set))
    print(din_str)
    
    ood_set = ImageNet_ood(os.path.join(args.data_root_path, 'ImageNet10k_eccv2010/imagenet10k'), transform=train_transform, txt="./datasets/imagenet_ood_test_1k_wnid_list_picture.txt")
    ood_loader = DataLoader(ood_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                                drop_last=False, pin_memory=True)
    dout_str = 'Dout is %s with %d images' % (args.dout, len(ood_set))
    print(dout_str)

    img_num_per_cls = np.array(train_set.img_num_per_cls)
    prior = img_num_per_cls / np.sum(img_num_per_cls)
    prior = torch.from_numpy(prior).float().cuda()

    # model:
    model = ResNet50(num_classes=num_classes + 1).cuda()
    model = torch.nn.DataParallel(model)

    # load model:
    ckpt = torch.load(os.path.join(args.ckpt_path, 'latest.pth'))['model']
    model.load_state_dict(ckpt)   
    model.requires_grad_(False)

    # log file:
    if args.tnorm:
        '''
        Decoupling representation and classifier for long-tailed recognition. ICLR, 2020.
        '''
        w = model.linear.weight.data
        w_row_norm = torch.norm(w, p='fro', dim=1)
        print(w_row_norm)
        model.linear.weight.data = w / w_row_norm[:,None]
        model.linear.bias.zero_()
    test_result_file_name = 'test_results.txt'
    fp = open(os.path.join(save_dir, test_result_file_name), 'a+')
    fp.write('\n===%s===\n' % (args.dout))
    fp.write(din_str + '\n')
    fp.write(dout_str + '\n')


    val_imagenet()