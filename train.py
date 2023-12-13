import argparse, os, datetime, time
from sklearn.metrics import f1_score
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from datasets.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.ImbalanceImageNet import LT_Dataset
from datasets.tinyimages_300k import TinyImages
from datasets.imagenet_ood import ImageNet_ood
from models.resnet import ResNet18, ResNet34
from models.resnet_imagenet import ResNet50

from utils.utils import *
from utils.ltr_metrics import shot_acc
from utils.distance import compute_dist
from skimage.filters import gaussian as gblur

import matplotlib.pyplot as plt

# to prevent PIL error from reading large images:
# See https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162#issuecomment-491115265
# or https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

def get_args_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
    parser.add_argument('--seed', default=25, type=int, help='fix the random seed for reproduction. Default is 25.')
    parser.add_argument('--replay', default='replay3', help='repetitions for reproduction.')
    parser.add_argument('--gpu', default='4', help='which GPU to use.')
    parser.add_argument('--num_workers', '--cpus', default=4, help='number of threads for data loader')
    parser.add_argument('--data_root_path', '--drp', default='./datasets', help='data root path')
    parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50'], help='which model to use')
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    # training params:
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='input batch size for training')
    parser.add_argument('--ood_batch_size', '--ob', type=int, default=256, help='OOD batch size for training')
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--opt', default='adam', choices=['sgd', 'adam'], help='which optimizer to use')
    parser.add_argument('--Lambda1', default=0.05, type=float, help='OCL loss term tradeoff hyper-parameter: 0.05 for CIFAR and 0.02 for ImageNet')
    parser.add_argument('--Lambda2', default=0.05, type=float, help='OOD-Aware Tail Class Prototype Learning loss term tradeoff hyper-parameter:  0.05 for CIFAR and 0.01 for ImageNet')
    parser.add_argument('--Lambda3', default=0.1, type=float, help='Debiased Head Class Learning loss term tradeoff hyper-parameter:  0.1 for CIFAR and 0.01 for ImageNet')
    parser.add_argument('--num_ood_samples', default=300000, type=int, help='Number of OOD samples to use.')
    parser.add_argument('--tau', type=float, default=1, help='hyperparameter to balance prior in OLC')
    parser.add_argument('--margin', type=float, default=1, help='hyperparameter to DHCL')
    parser.add_argument('--temperature', type=float, default=0.07, help='temperature in OOD-Aware Tail Class Prototype Learning loss')
    parser.add_argument('--headrate', default=0.4, type=float, help='percentage of head to use')
    parser.add_argument('--tailrate', default=0.4, type=float, help='percentage of head to use')
    parser.add_argument('--noise_type', default='None', choices=['None', 'Gaussian', 'Rademacher', 'Blob'], help='whether use synthesis auxiliary data')
    parser.add_argument('--save_root_path', '--srp', default='./result', help='data root path')
    args = parser.parse_args()

    return args


def create_save_path():
    # mkdirs:
    opt_str = 'e%d-b%d-%d-%s-lr%s-wd%s' % (args.epochs, args.batch_size, args.ood_batch_size, args.opt, args.lr, args.wd)
    loss_str = 'Lambda1%s-Lambda2%s-Lambda3%s' % (args.Lambda1, args.Lambda2, args.Lambda3)
    exp_str = '%s_%s' % (opt_str, loss_str)
    dataset_str = '%s-%s-OOD%d' % (args.dataset, args.imbalance_ratio, args.num_ood_samples) if 'imagenet' not in args.dataset else '%s-lt' % (args.dataset)
    save_dir = os.path.join(args.save_root_path, dataset_str, args.model, exp_str, args.replay)
    create_dir(save_dir)
    print('Saving to %s' % save_dir)

    return save_dir, dataset_str

def create_ood_noise(noise_type, ood_num_examples, num_to_avg):
    if noise_type == "Gaussian":
        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(np.float32(np.clip(
            np.random.normal(size=(ood_num_examples * num_to_avg, 3, 32, 32), scale=0.5), -1, 1)))
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    elif noise_type == "Rademacher":
        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(np.random.binomial(
            n=1, p=0.5, size=(ood_num_examples * num_to_avg, 3, 32, 32)).astype(np.float32)) * 2 - 1
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    elif noise_type == "Blob":
        ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * num_to_avg, 32, 32, 3)))
        for i in range(ood_num_examples * num_to_avg):
            ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
            ood_data[i][ood_data[i] < 0.75] = 0.0

        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    return ood_data

def train(args): 

    # get batch size:
    train_batch_size = args.batch_size 
    ood_batch_size = args.ood_batch_size 
    num_workers = args.num_workers

    save_dir = args.save_dir 
    device = 'cuda'

    # data:
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.dataset == 'imagenet':
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
    if args.dataset == 'cifar10':
        num_classes = 10
        train_set = IMBALANCECIFAR10(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = IMBALANCECIFAR100(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'imagenet':
        num_classes = 1000
        train_set = LT_Dataset(
            os.path.join(args.data_root_path, 'ImageNet_LT/train'), './datasets/ImageNet_LT_train.txt', transform=train_transform, 
            class_idx=np.arange(0,num_classes))
        test_set = LT_Dataset(
            os.path.join(args.data_root_path, 'ImageNet_LT/val'), './datasets/ImageNet_LT_val.txt', transform=test_transform,
            class_idx=np.arange(0,num_classes))
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers, 
                                drop_last=False, pin_memory=True)
    if args.dataset in ['cifar10', 'cifar100']:
        if args.noise_type == 'None':
            ood_set = Subset(TinyImages(args.data_root_path, transform=train_transform, dataset = args.dataset), list(range(args.num_ood_samples)))
        else:
            ood_set = create_ood_noise(args.noise_type, args.num_ood_samples, 1)
    elif args.dataset == 'imagenet':
        ood_set = ImageNet_ood(os.path.join(args.data_root_path, 'ImageNet10k_eccv2010/imagenet10k'), transform=train_transform, txt="./datasets/imagenet_extra_1k_wnid_list_picture.txt")
    ood_loader = DataLoader(ood_set, batch_size=ood_batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True, pin_memory=True)
    print('Training on %s with %d images and %d validation images | %d OOD training images.' % (args.dataset, len(train_set), len(test_set), len(ood_set)))
    
    # get prior distributions:
    img_num_per_cls = np.array(train_set.img_num_per_cls)
    prior = img_num_per_cls / np.sum(img_num_per_cls)
    prior = torch.from_numpy(prior).float().to(device)
    assert np.sum(img_num_per_cls) == len(train_set), 'Sum of image numbers per class %d neq total image number %d' % (np.sum(img_num_per_cls), len(train_set))
    plt.plot(np.sort(img_num_per_cls)[::-1])
    plt.savefig(os.path.join(save_dir, 'img_num_per_cls.png'))
    plt.close()

    # model:
    if args.model == 'ResNet18':
        model = ResNet18(num_classes=num_classes + 1, return_features=True).to(device)
    elif args.model == 'ResNet34':
        model = ResNet34(num_classes=num_classes + 1, return_features=True).to(device)
    elif args.model == 'ResNet50':
        model = ResNet50(num_classes=num_classes + 1, return_features=True).to(device)
    else:
        raise ValueError("illegal training model")

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    

    # optimizer:
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # train:
    training_losses, test_clean_losses = [], []
    f1s, overall_accs, many_accs, median_accs, low_accs = [], [], [], [], []
    start_epoch = 0
    
    fp = open(os.path.join(save_dir, 'train_log.txt'), 'a+')
    fp_val = open(os.path.join(save_dir, 'val_log.txt'), 'a+')


    for epoch in range(start_epoch, args.epochs):
        model.train()
        training_loss_meter = AverageMeter()

        start_time = time.time()

        for batch_idx, ((in_data, labels), (ood_data, ood_labels)) in enumerate(zip(train_loader, ood_loader)):
            
            in_data, labels = in_data.to(device), labels.to(device)
            ood_data, ood_labels = ood_data.to(device), ood_labels.to(device)
            

            # forward:
            all_data = torch.cat([in_data, ood_data], dim=0)
            all_logits, p4 = model(all_data)
            
            N_in = in_data.shape[0]
            in_logits = all_logits[0:N_in]
            ood_logits = all_logits[N_in:]

            # outlier class learning
            in_loss = F.cross_entropy(in_logits, labels)
            OCL_loss = F.cross_entropy(ood_logits, ood_labels)

            f_id_view = p4[0:N_in]
            f_ood = p4[N_in:]
            head_idx = labels<= round(args.headrate*num_classes) # dont use int! since 1-0.9=0.0999!=0.1
            tail_idx = labels>= round((1-args.tailrate)*num_classes) # dont use int! since 1-0.9=0.0999!=0.1
            f_id_head_view = f_id_view[head_idx] # i.e., 6,7,8,9 in cifar10
            f_id_tail_view = f_id_view[tail_idx] # i.e., 6,7,8,9 in cifar10
            labels_tail = labels[tail_idx]

            # OOD-aware tail class prototype learning
            if len(f_id_tail_view)>0 and args.Lambda2 > 0:
                if torch.cuda.device_count() > 1:
                    logits = model.module.forward_weight(f_id_tail_view, f_ood, temperature=args.temperature)
                else:
                    logits = model.forward_weight(f_id_tail_view, f_ood, temperature=args.temperature)
                tail_loss = F.cross_entropy(logits, labels_tail-round((1-args.tailrate)*num_classes))
            else:
                tail_loss = 0 * OCL_loss

            # debiased head class learning
            if args.Lambda3 > 0:
                dist1 = compute_dist(f_ood, f_ood)
                _, dist_max1 = torch.max(dist1, 1)
                positive = f_ood[dist_max1]

                dist2 = torch.randint(low = 0, high= len(f_id_head_view), size = (1, len(f_ood))).to(device).squeeze()
                negative = f_id_head_view[dist2]
                
                triplet_loss = torch.nn.TripletMarginLoss(margin=args.margin)
                head_loss = triplet_loss(f_ood, positive, negative)
            else:
                head_loss = 0 * OCL_loss

            loss = in_loss + args.Lambda1 * OCL_loss + args.Lambda2 * tail_loss + args.Lambda3 * head_loss

            # backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append:
            training_loss_meter.append(loss.item())
            if batch_idx % 100 == 0:
                train_str = 'epoch %d batch %d (train): loss %.4f (%.4f, %.4f, %.4f, %.4f)' % (epoch, batch_idx, loss.item(), in_loss.item(), OCL_loss.item(), tail_loss.item(), head_loss.item()) 
                train_str = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '  |  ' + train_str
                print(train_str)
                fp.write(train_str + '\n')
                fp.flush()
          

        # lr update:
        scheduler.step()
        model.eval()

        test_acc_meter, test_loss_meter = AverageMeter(), AverageMeter()
        preds_list, labels_list = [], []
        pred_calibration_list = []

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                logits, _ = model(data)
                pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                loss = F.cross_entropy(logits, labels)
                test_acc_meter.append((logits.data[:, :-1].max(1)[1] == labels).float().mean().item())
                test_loss_meter.append(loss.item())
                preds_list.append(pred)
                labels_list.append(labels)

                # outlier-class-aware logit calibration
                p = torch.cat((prior, torch.ones(1).to(device)), dim = 0)
                pred_calibration = logits - args.tau * p.log()[None,:] 
                pred_calibration = pred_calibration.data[:, :-1].max(1)[1]
                pred_calibration_list.append(pred_calibration)

        labels = torch.cat(labels_list, dim=0).detach().cpu().numpy()
        preds = torch.cat(preds_list, dim=0).detach().cpu().numpy().squeeze()
        overall_acc= (preds == labels).sum().item() / len(labels)
        f1 = f1_score(labels, preds, average='macro')
        many_acc, median_acc, low_acc, _ = shot_acc(preds, labels, img_num_per_cls, acc_per_cls=True)
        val_str = 'epoch %d (test): ACC %.4f (%.4f, %.4f, %.4f) | F1 %.4f | time %.2f' % (epoch, overall_acc, many_acc, median_acc, low_acc, f1, time.time()-start_time) 
        val_str = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '  |  ' + val_str
        print(val_str)
        fp_val.write(val_str + '\n')
        fp_val.flush()

        test_clean_losses.append(test_loss_meter.avg)
        f1s.append(f1)
        overall_accs.append(overall_acc)
        many_accs.append(many_acc)
        median_accs.append(median_acc)
        low_accs.append(low_acc)

        # save curves:
        training_losses.append(training_loss_meter.avg)
        plt.plot(training_losses, 'b', label='training_losses')
        plt.plot(test_clean_losses, 'g', label='test_clean_losses')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'losses.png'))
        plt.close()

        plt.plot(overall_accs, 'm', label='overall_accs')
        if args.imbalance_ratio < 1:
            plt.plot(many_accs, 'r', label='many_accs')
            plt.plot(median_accs, 'g', label='median_accs')
            plt.plot(low_accs, 'b', label='low_accs')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'test_accs.png'))
        plt.close()

        plt.plot(f1s, 'm', label='f1s')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'test_f1s.png'))
        plt.close()

        # save pth:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch, 
            'training_losses': training_losses, 
            'test_clean_losses': test_clean_losses, 
            'f1s': f1s, 
            'overall_accs': overall_accs, 
            'many_accs': many_accs, 
            'median_accs': median_accs, 
            'low_accs': low_accs, 
            }, 
            os.path.join(save_dir, 'latest.pth'))
        

        pred_calibration = torch.cat(pred_calibration_list, dim=0).detach().cpu().numpy().squeeze()
        overall_acc_calibration= (pred_calibration == labels).sum().item() / len(labels)
        f1_calibration = f1_score(labels, pred_calibration, average='macro')
        many_acc_calibration, median_acc_calibration, low_acc_calibration, _ = shot_acc(pred_calibration, labels, img_num_per_cls, acc_per_cls=True)
        val_str = 'epoch %d (test): ACC %.4f (%.4f, %.4f, %.4f) | F1 %.4f | time %.2f' % (epoch, overall_acc_calibration, many_acc_calibration, median_acc_calibration, low_acc_calibration, f1_calibration, time.time()-start_time) 
        val_str = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + '  |  ' + val_str
        print(val_str)
        fp_val.write(val_str + '\n')
        fp_val.flush()

if __name__ == '__main__':
    # get args:
    args = get_args_parser()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)  # Numpy module.

    # mkdirs:
    save_dir, dataset_str = create_save_path()
    args.save_dir = save_dir
    args.dataset_str = dataset_str
    
    # intialize device:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    torch.backends.cudnn.benchmark = True
    
    train(args)
