import argparse
import time
import torch
import numpy as np
import torch.optim as optim
from skimage.io import imread, imshow
# custom modules

from loss import MonodepthLoss
from utils import get_model, to_device, prepare_dataloader, readlines
from skimage.metrics import structural_similarity as ssim
from loss import MonodepthLoss, ICPLoss
import os
import torch.nn.functional as F
import PIL.Image as pil
from torchvision import transforms
import cv2
import matplotlib.cm as cm

# plot params

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)

file_dir = os.path.dirname(__file__)  # the directory that main.py resides in


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('--data_dir',
                        type=str,
                        help='path to the dataset folder',
                        default='/disk_three/Dataset/Endovis_depth')
    parser.add_argument('--val_data_dir',
                        help='path to the validation dataset folder',
                        default='/disk_three/Dataset/Endovis_depth/Test')
    parser.add_argument('--split',
                        type=str,
                        help='splits to load data',
                        default='Endovis_origin')
    parser.add_argument('--model_path',
                        default=os.path.join(file_dir, "weights"),
                        help='path to the trained model')
    parser.add_argument('--output_directory',
                        help='where save dispairities\
                        for tested images')
    parser.add_argument('--input_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=320)
    parser.add_argument('--full_height', type=int, help='input height',
                        default=1024)
    parser.add_argument('--full_width', type=int, help='input width',
                        default=1280)
    parser.add_argument('--model', default='resnet18_md',
                        help='encoder architecture: ' +
                        'resnet18_md or resnet50_md ' + '(default: resnet18)'
                        + 'or torchvision version of any resnet model')
    parser.add_argument('--resume', default=None,
                        help='load weights to continue train from where it last stopped')
    parser.add_argument('--load_weights_folder', default=os.path.join(file_dir, "weights"),
                        help='folder to load weights to continue train from where it last stopped')
    parser.add_argument('--pretrained', default=False,
                        help='Use weights of pretrained model')
    parser.add_argument('--mode', default='train',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of total epochs to run')
    parser.add_argument('--startEpoch', type=int, default=0,
                        help='number of total epochs to run')
    parser.add_argument('--testepoch', type=str, default='border_cpt',
                        help='number of total epochs to test')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=22,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose cpu or cuda:0 device"')
    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')
    parser.add_argument('--augment_parameters', default=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                        help='lowest and highest values for gamma,\
                        brightness and color respectively')
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
    parser.add_argument('--input_channels', default=3,
                        help='Number of channels in input tensor')
    parser.add_argument('--num_workers', default=4,
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=True)
    parser.add_argument('--focal_length', type=float, default=1135, help='mean focal length')   # 7918.42273452993
    parser.add_argument('--baseline', type=float, help='baseline', default=4.2)                 # 5.045158438885819
    parser.add_argument('--endovis_test_key', default=True, help="if true, then error EndovisOriginSplit")
    parser.add_argument('--applyICP', default=True,
                        help="if true, then calculate ICP loss with or without applying masks")
    parser.add_argument('--ICPMask', default=True, help="if true, then calculate ICP with MASK")
    parser.add_argument('--ICPweight', type=float, default=1/1000, help='weights for ICP in the final loss')
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


class Model:
    def __init__(self, args):
        self.args = args
        # create weight folder
        if os.path.isdir(args.model_path):
            print('Weights folder exists')
        else:
            print('Weights folder create')
            os.makedirs(args.model_path)

        # Set up model
        self.device = args.device
        self.model = get_model(args.model, input_channels=args.input_channels,
                               pretrained=args.pretrained)
        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)
        if self.args.applyICP:
            # intrinsic matrix
            self.K = np.array([[1.18849248e+03, 0.00000000e+00, 6.41449814e+02, 0.00000000e+00],
                               [0.00000000e+00, 1.18849248e+03, 5.20022934e+02, 0.00000000e+00],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)
            self.inv_K = transforms.ToTensor()(np.linalg.pinv(self.K)).to(self.device)
            self.inv_K = self.inv_K.repeat(self.args.batch_size, 1, 1)
            # Extrinsic parameters
            self.T = np.eye(4, dtype=np.float32)
            self.T[0, 3] = -4.2  # average baseline
            self.T = torch.from_numpy(self.T).to(self.device)

        if args.mode == 'train':
            args.data_dir = os.path.join(args.data_dir, 'Train')
            self.loss_function = MonodepthLoss(
                n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1).to(self.device)
            #### ICP ####
            if self.args.applyICP:
                self.ICPLossMask = ICPLoss(self.args.focal_length, self.args.baseline,
                                             self.args.full_width, self.args.full_height,
                                             self.inv_K, self.T, self.args.ICPMask)

            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.learning_rate)

            if args.resume is not None:
                self.load_model_continue_train(os.path.join(self.args.model_path, 'weights_last.pt'))
                self.args.startEpoch = self.startEpoch

            fpath = os.path.join(os.path.dirname(__file__), "splits", args.split, "{}_files.txt")
            train_filenames = readlines(fpath.format("train"))
            val_filenames = readlines(fpath.format("val"))

            self.val_n_img, self.val_loader = prepare_dataloader(args.val_data_dir, args.mode, val_filenames,
                                                                 args.augment_parameters,
                                                                 False, args.batch_size,
                                                                 (args.input_height, args.input_width),
                                                                 args.num_workers)
            # Load data
            self.output_directory = args.output_directory
            self.input_height = args.input_height
            self.input_width = args.input_width
            self.n_img, self.loader = prepare_dataloader(args.data_dir, args.mode,
                                                         train_filenames,
                                                         args.augment_parameters,
                                                         args.do_augmentation, args.batch_size,
                                                         (args.input_height, args.input_width),
                                                         args.num_workers)
        else:
            args.test_model_path = os.path.join(self.args.model_path, args.testepoch + '.pth')
            self.model.load_state_dict(torch.load(args.test_model_path))
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1
            if args.mode == 'test':
                args.data_dir = os.path.join(args.data_dir, 'test')

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def train(self):
        losses = []
        val_losses = []
        ICPLosses = 0.0
        best_val_loss = float('Inf')
        running_val_loss = 0.0
        self.model.eval()
        for data in self.val_loader:
            data = to_device(data, self.device)  # dict
            left = data['left_image']
            right = data['right_image']
            disps = self.model(left)
            loss = self.loss_function(disps, [left, right])
            #### ICP ####
            if self.args.applyICP:
                ICPLoss = self.args.ICPweight * self.ICPLossMask(disps, [left, right])
                loss = loss + ICPLoss
            val_losses.append(loss.item())
            running_val_loss += loss.item()

        running_val_loss /= self.val_n_img / self.args.batch_size
        print('Val_loss:', running_val_loss)

        for epoch in range(self.args.startEpoch, self.args.epochs):
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer, epoch,
                                     self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            self.model.train()
            for data in self.loader:
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']

                # One optimization iteration
                self.optimizer.zero_grad()
                disps = self.model(left)
                loss = self.loss_function(disps, [left, right])
                #### ICP####
                if self.args.applyICP:
                    ICPLoss = self.args.ICPweight * self.ICPLossMask(disps, [left, right])
                    loss = loss + ICPLoss
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                running_loss += loss.item()
                if self.args.applyICP:
                    ICPLosses += ICPLoss

            running_val_loss = 0.0
            self.model.eval()
            for data in self.val_loader:
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                disps = self.model(left)
                loss = self.loss_function(disps, [left, right])
                val_losses.append(loss.item())
                running_val_loss += loss.item()

            # Estimate loss per image
            running_loss /= self.n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            print(
                'Epoch:',
                epoch + 1,
                'train_loss:',
                running_loss,
                'val_loss:',
                running_val_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
            )
            self.save(os.path.join(self.args.model_path, 'border_last.pth'))
            self.save_continue_train(epoch, running_loss, 'weights_last.pt')
            # save weights for every epoch
            self.save(os.path.join(self.args.model_path, 'epoch{}.pth'.format(str(epoch))))
            if running_val_loss < best_val_loss:
                self.save(os.path.join(self.args.model_path, 'border_cpt.pth'))
                self.save_continue_train(epoch, running_val_loss, 'weights_cpt.pt')
                best_val_loss = running_val_loss
                print('Model_saved')

        print('Finished Training.')
        # self.save(os.path.join(self.args.model_path, 'train_end.pth'))
        self.save_continue_train(self.args.epochs, running_val_loss, 'train_end.pt')

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def save_continue_train(self, epoch, loss, path):
        save_path = os.path.join(self.args.model_path, path)
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    }, save_path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def load_model_continue_train(self, path):
        assert os.path.isfile(path), \
            "Cannot find folder {}".format(path)
        print("loading model from folder {}".format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.startEpoch = checkpoint['epoch']

    def test(self):
        self.model.eval()

        ''' train on full endovis and test on Endovis test dataset keyframe'''
        if self.args.endovis_test_key:
            errors = []
            baseline = 4.2  # 5.045158438885819
            focal = 1135  # # 7866.0520212773545
            transform_resize = transforms.Resize((256, 320))
            with torch.no_grad():
                ground_truth_dir = '/disk_three/Dataset/Endovis_depth/TestKeyFrameOnly/depth'
                test_data_dir = '/disk_three/Dataset/Endovis_depth/TestKeyFrameOnly/image'
                image02_file = os.path.join(ground_truth_dir, 'image_02')
                image03_file = os.path.join(ground_truth_dir, 'image_03')
                for image in sorted(os.listdir(image02_file)):
                    ground_truth_image_file_left = os.path.join(image02_file, image)
                    ground_truth_image_file_right = os.path.join(image03_file, image)
                    test_RGB_image_file_left = os.path.join(test_data_dir, 'image_02', image)
                    test_RGB_image_file_right = os.path.join(test_data_dir, 'image_03', image)

                    if not os.path.exists(ground_truth_image_file_left) and os.path.exists(
                            test_RGB_image_file_left):
                        print('Error: point could not found - {}'.format(test_RGB_image_file_left))
                    ''' Load in Input image '''
                    left_input_image = pil.open(test_RGB_image_file_left).convert('RGB')
                    left_input_image = transform_resize(left_input_image)
                    right_input_image = pil.open(test_RGB_image_file_right).convert('RGB')
                    left_input_image = transforms.ToTensor()(left_input_image).unsqueeze(0)
                    right_input_image = transforms.ToTensor()(right_input_image).unsqueeze(0)
                    left_input_image = left_input_image.to(self.device)
                    right_input_image = right_input_image.to(self.device)
                    ''' Load in grond truth'''
                    depth_gt_left = pil.open(ground_truth_image_file_left).convert('L')
                    depth_gt_left = np.asarray(depth_gt_left, dtype="float32")

                    if np.sum(depth_gt_left > 0.1) < (0.1 * np.size(depth_gt_left)):
                        print('abe == -1')
                    else:
                        disps = self.model(left_input_image)
                        disps_upsample = F.interpolate(disps[0][:, 0, :, :].unsqueeze(1),
                                                       [self.args.full_height, self.args.full_width], mode="bilinear",
                                                       align_corners=False).squeeze().cpu().detach().numpy()
                        depth_pred = (baseline * focal) / (disps_upsample * 1280)

                    errors.append(compute_errors(depth_gt_left, depth_pred))
                    mean_errors = np.array(errors).mean(0)

                #### 7 criteria ####
                print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
                print("\n-> Done!")


def compute_errors(gt, pred, MIN_DEPTH=25, MAX_DEPTH=300):
    """Computation of error metrics between predicted and ground truth depths
    """
    mask = np.logical_and(gt >= MIN_DEPTH,  gt <= MAX_DEPTH)
    gt = gt[mask]
    pred = pred[mask]
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def main():
    args = return_arguments()
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test()


if __name__ == '__main__':
    main()

