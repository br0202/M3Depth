import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import iterative_closest_point
import matplotlib.pyplot as plt
import matplotlib as mpl
import einops


class MonodepthLoss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.85, disp_gradient_w=1.0, lr_w=1.0):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()
        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)
        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='border')

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(self.n)]

    def forward(self, input, target):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        left, right = target
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)

        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est
        # Generate images
        left_est = [self.generate_image_left(right_pyramid[i],
                    disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(left_pyramid[i],
                     disp_right_est[i]) for i in range(self.n)]
        self.left_est = left_est
        self.right_est = right_est

        # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                           disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                           disp_right_est[i]) for i in range(self.n)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_pyramid)

        # L1
        l1_left = [torch.mean(torch.abs(left_est[i] - left_pyramid[i]))
                   for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i]
                    - right_pyramid[i])) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(left_est[i],
                     left_pyramid[i])) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i],
                      right_pyramid[i])) for i in range(self.n)]

        image_loss_left = [self.SSIM_w * ssim_left[i]
                           + (1 - self.SSIM_w) * l1_left[i]
                           for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i]
                            + (1 - self.SSIM_w) * l1_right[i]
                            for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]
                        - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]
                         - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [torch.mean(torch.abs(
                          disp_left_smoothness[i])) / 2 ** i
                          for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(
                           disp_right_smoothness[i])) / 2 ** i
                           for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        loss = image_loss + self.disp_gradient_w * disp_gradient_loss\
               + self.lr_w * lr_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss


##### ICP #####
class ICPLoss(nn.modules.Module):
    def __init__(self, focal_length, baseline, imgWidth, imgHeight, inv_K, T, applyMask=False):
        super(ICPLoss, self).__init__()
        self.focal_length = focal_length
        self.baseline = baseline
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        self.applyMask = applyMask
        self.inv_K = inv_K
        self.T = T
        self.MonodepthLoss = MonodepthLoss()

    def disp_to_depth(self, disp):
        depth = self.focal_length * self.baseline / (disp * self.imgWidth) # here disps should be splited
        return depth

    def apply_disparity_for_ICP(self, img, disp):    # this is from left to right ### use when apply mask
        batch_size_ICP, _, height, width = img.size()
        # Original coordinates of pixels
        x_base_ICP = torch.linspace(0, 1, width).repeat(batch_size_ICP,
                                                    height, 1).type_as(img)
        y_base_ICP = torch.linspace(0, 1, height).repeat(batch_size_ICP,
                                                     width, 1).transpose(1, 2).type_as(img)
        # Apply shift in X direction
        x_shifts_ICP = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field_ICP = torch.stack((x_base_ICP + x_shifts_ICP, y_base_ICP), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        reconstruct_image = F.grid_sample(img, 2 * flow_field_ICP - 1, mode='bilinear', padding_mode='zeros')

        return reconstruct_image

    def depth_to_pcl(self, depth, inv_K, applyMask, mask=None):
        if applyMask:
            depth = depth * (mask)
            backproject_depth = {}
            backproject_depth[0] = (BackprojectDepth(depth.shape[0], self.imgHeight, self.imgWidth)).to(device=depth.device)
            cam_points = backproject_depth[0](depth, inv_K)

        else:
            backproject_depth = {}
            backproject_depth[0] = (BackprojectDepth(depth.shape[0], self.imgHeight, self.imgWidth)).to(device=depth.device)
            cam_points = backproject_depth[0](depth, inv_K)

        return cam_points

    def compute_ICP_loss(self, pclLeft, pclRight):
        indexL = torch.randint(0, 1310720, (20000,)).cuda()
        indexR = torch.randint(0, 1310720, (20000,)).cuda()
        icploss , _ = chamfer_distance(torch.index_select(pclLeft.permute(0, 2, 1)[:, :, :3], 1, indexL),
                                       torch.index_select(pclRight.permute(0, 2, 1)[:, :, :3], 1, indexR))
        return icploss

    def compute_ICP_loss_no_MASK(self, pclLeft, pclRight):
        batchsize = pclLeft.shape[0]
        PCL_L = torch.zeros(batchsize, 3, 1000)
        PCL_R = torch.zeros(batchsize, 3, 1000)

        for item in range(batchsize):
            single_Left = pclLeft[item]
            filtered_pclLeft = single_Left[:3, :]
            single_Right = pclRight[item]
            filtered_pclRight = single_Right[:3, :]
            index = torch.randint(0, min(filtered_pclLeft.shape[1], filtered_pclRight.shape[1]), (1000,)).cuda()

            pcl_l = torch.index_select(filtered_pclLeft, 1, index)
            pcl_r = torch.index_select(filtered_pclRight, 1, index)
            # pcl_r_normal = pcl_r / max_r.unsqueeze(1)
            # PCL_L[item, :, :] = torch.matmul(self.T, pcl_l)[:3, :]
            PCL_L[item, :, :] = pcl_l
            PCL_R[item, :, :] = pcl_r

        _, icploss, _, _, _ = iterative_closest_point(PCL_L.permute(0, 2, 1),
                                                      PCL_R.permute(0, 2, 1))  # the second from last is RTs

        icploss = icploss.mean().to(pclLeft.device)
        return icploss

    def compute_ICP_loss_with_MASK(self, pclLeft, pclRight):
        batchsize = pclLeft.shape[0]
        PCL_L = torch.zeros(batchsize, 3, 1000)
        PCL_R = torch.zeros(batchsize, 3, 1000)

        for item in range(batchsize):
            single_Left = pclLeft[item]
            filtered_pclLeft = single_Left[:3, single_Left[2, :] > 0]
            single_Right = pclRight[item]
            filtered_pclRight = single_Right[:3, single_Right[2, :] > 0]
            index = torch.randint(0, min(filtered_pclLeft.shape[1], filtered_pclRight.shape[1]), (1000,)).cuda()

            pcl_l = torch.index_select(filtered_pclLeft, 1, index)
            pcl_r = torch.index_select(filtered_pclRight, 1, index)
            PCL_L[item, :, :] = pcl_l
            PCL_R[item, : ,:] = pcl_r

        _, icploss, _, _, _ = iterative_closest_point(PCL_L.permute(0, 2, 1), PCL_R.permute(0, 2, 1))  # the second from last is RTs

        icploss = icploss.mean().to(pclLeft.device)
        return icploss

    def generate_mask(self, input, target):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        left, right = target
        left_pyramid = self.MonodepthLoss.scale_pyramid(left, self.MonodepthLoss.n)
        right_pyramid = self.MonodepthLoss.scale_pyramid(right, self.MonodepthLoss.n)

        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        # Generate images
        reconstruct_left = [self.apply_disparity_for_ICP(right_pyramid[i],
                                             -disp_left_est[i]) for i in range(self.MonodepthLoss.n)]
        reconstruct_right = [self.apply_disparity_for_ICP(left_pyramid[i],
                                               disp_right_est[i]) for i in range(self.MonodepthLoss.n)]
        #### MASK ####
        left_mask = [(torch.where(reconstruct_left[0][:, 0, :, :] == 0, torch.tensor(0).cuda(), torch.tensor(1).cuda())).unsqueeze(1)]
        right_mask = [(torch.where(reconstruct_right[0][:, 0, :, :] == 0, torch.tensor(0).cuda(), torch.tensor(1).cuda())).unsqueeze(1)]
        left_mask = F.interpolate(left_mask[0].float(), [self.imgHeight, self.imgWidth], mode="bilinear", align_corners=False)
        right_mask = F.interpolate(right_mask[0].float(), [self.imgHeight, self.imgWidth], mode="bilinear", align_corners=False)

        return left_mask, right_mask

    def forward(self, disps, img=None):
        # ICPLossCompute
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            img [left, right]

        Return:
            (float): The loss
        """
        ##### please note here only calculate for scale 0, the original input size

        disp_left = disps[0][:, 0, :, :].unsqueeze(1)
        disp_right = disps[0][:, 1, :, :].unsqueeze(1)

        if not self.applyMask:
            depth_left = F.interpolate(self.disp_to_depth(disp_left), [self.imgHeight, self.imgWidth], mode="bilinear",
                                       align_corners=False)
            depth_right = F.interpolate(self.disp_to_depth(disp_right), [self.imgHeight, self.imgWidth],
                                        mode="bilinear", align_corners=False)
            pcl_left = self.depth_to_pcl(depth_left, self.inv_K, self.applyMask)   # depth_to_pcl should be changed with more parameters
            pcl_right = self.depth_to_pcl(depth_right, self.inv_K, self.applyMask)
            ICPLoss = self.compute_ICP_loss_no_MASK(pcl_left, pcl_right)

        else:
            depth_left = F.interpolate(self.disp_to_depth(disp_left), [self.imgHeight, self.imgWidth], mode="bilinear",
                                       align_corners=False)
            depth_right = F.interpolate(self.disp_to_depth(disp_right), [self.imgHeight, self.imgWidth],
                                        mode="bilinear", align_corners=False)
            # depth_left = self.disp_to_depth(disp_left)
            # depth_right = self.disp_to_depth(disp_right)
            left_mask, right_mask = self.generate_mask(disps, img)
            pcl_left = self.depth_to_pcl(depth_left, self.inv_K, self.applyMask, left_mask)  # depth_to_pcl should be changed with more parameters
            pcl_right = self.depth_to_pcl(depth_right, self.inv_K, self.applyMask, right_mask)
            ICPLoss = self.compute_ICP_loss_with_MASK(pcl_left, pcl_right)

        return ICPLoss


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

