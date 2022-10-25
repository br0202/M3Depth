## [Self-Supervised Depth Estimation in Laparoscopic Image using 3D Geometric Consistency](https://arxiv.org/abs/2208.08407) (MICCAI 2022)
By [Baoru Huang](https://baoru.netlify.app/), Jian-Qing Zheng, [Anh Nguyen](https://www.csc.liv.ac.uk/~anguyen), Chi Xu, Ioannis Gkouzionis, Kunal Vyas, David Tuch, Stamatia Giannarou, Daniel S. Elson

![image](https://github.com/br0202/M3Depth/blob/master/vis/frameworks.png "m3depth")

### Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Training&Testing](#training)
4. [Notes](#notes)


### Requirements

1. Python version 3.8.5
2. Pytorch version: pytorch==1.6.0 torchvision==0.8.2 torchaudio==0.7.0 cudatoolkit=10.2.89
3. pytorch3d version 0.3.0
2. Cuda version 10.2

### Installation

1. Clone the repository into your `$M3Depth` folder.
	
	
2. Build ....

	
### Training & Testing

1. We train M3Depth on [UPDATE_HERE dataset](https://sites.google.com/site/iitaffdataset/)
	- We need to format IIT-AFF dataset as in Pascal-VOC dataset for training.
	- For your convinience, we did it for you. Just download this file ([Google Drive](https://drive.google.com/file/d/0Bx3H_TbKFPCjV09MbkxGX0k1ZEU/view?usp=sharing), [One Drive](https://studenthcmusedu-my.sharepoint.com/:u:/g/personal/nqanh_mso_hcmus_edu_vn/EXQok71Y2kFAmhaabY2TQO8BFIO1AqqH5GcMOfPqgn_q2g?e=7rH3Kd)) and extract it into your `$AffordanceNet_ROOT` folder.
	- The extracted folder should contain three sub-folders: `$AffordanceNet_ROOT/data/cache`, `$AffordanceNet_ROOT/data/imagenet_models`, and `$AffordanceNet_ROOT/data/VOCdevkit2012` .

2. Train M3Depth:
	- `cd $M3Depth`
	- `python main.py --mode train`
	

3. Test M3Depth:
    - `cd $M3Depth`
    - `python main.py --mode test`

4. Results
![image](https://github.com/br0202/M3Depth/blob/master/vis/quan_results.png "results")


### Citing 

If you find our paper useful in your research, please consider citing:

        @inproceedings{huang2022self,
          title={Self-supervised Depth Estimation in Laparoscopic Image Using 3D Geometric Consistency},
          author={Huang, Baoru and Zheng, Jian-Qing and Nguyen, Anh and Xu, Chi and Gkouzionis, Ioannis and Vyas, Kunal and Tuch, David and Giannarou, Stamatia and Elson, Daniel S},
          booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
          pages={13--22},
          year={2022},
          organization={Springer}
        }


### License
MIT License

### Acknowledgement
This repo used a lot of source code from [UPDATE_HERE](https://github.com/rbgirshick/UPDATE_LINK)
