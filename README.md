# Im2pcl 
Official PyTorch implementation of [**Height and Uprightness Invariance for 3D Prediction from a Single View (CVPR 2020)**](http://openaccess.thecvf.com/content_CVPR_2020/html/Baradad_Height_and_Uprightness_Invariance_for_3D_Prediction_From_a_Single_CVPR_2020_paper.html)

<p align="center">
  <img width="460" src="https://github.com/mbaradad/im2pcl/blob/master/assets/teaser.gif?raw=true">
</p>

The proposed model allows obtaining pointclouds at scale in a canonical pose for arbitrary images in the wild. Compared to other depth models, ours jointly estimates 
camera intrinsics and extrinsics. The provided models were trained using the Scannet dataset, and generalize well to images in-the-wild that have similar contents. 

For images with other content types, such as outdoors or indoors with people covering a large portion of the image, we do not recommend using our pretrained models.

### Requirements

- Python 3.7.4
- PyTorch 1.5 (other versions are likely to work, but have not been tested)

To install the required Python dependencies, we recommend creating a new conda environment and running:

```
conda create --name im2pcl python=3.7.4
source activate im2pcl
pip install -r requirements.txt
conda install -c conda-forge opencv
```


# Testing
To test the provided checkpoints download and unzip them from [here](https://drive.google.com/drive/folders/1mRPd6KkCiiv1whwoP5o47d7vk63Ofxxl?usp=sharing):

Then, to test a single image, run a visdom server (for pointcloud visualizatioons) and use the command:
```
python -m visdom.server -port 10000
python test.py --visdom-port 10000 --imgs assets/test.jpg --test-path checkpoints/full_fov
```
With visdom, you'll be able to get interactive visualization of the results such as:

<p align="center">
  <img width="460" src="https://github.com/mbaradad/im2pcl/blob/master/assets/visdom_visuals.jpg">
</p>

You can similarly test all images in a folder using --img-folder command. To test images from the datasets following the paper, 
you will need to download SUN360 or SUNRGBD (which contains NYUv2 as a subset), and edit the paths.py to point to the appropriate locations. 
Then, you can use the --dataset argument to test images on these datasets.


# Training
To train the model from scratch, you will need to download [Scannet](http://www.scan-net.org/) and expand the dataset (produce images and depth maps) using Scannet scripts or our expanding script in datasets/scannet_utils/expand_dataset.py. 
Finally, edit paths.py to reflect your paths configuration.

If you want to use knn-normals (to fully reproduce our results) you will need to precompute them, or if you don't want to you can train with nearest neighbors normals (by setting --use-knn-normals option to False).
To precomupte the knn-normals you can run:
```
python datasets/scannet_world_dataset.py
```
Then, you can simply train as the model with.
```
python train.py
```
By default, the model will create a cache with the cannonical pointclouds obtained from Scannet, to reduce CPU usage. If you want to disable this option, 
simply setting the --use-cache argument to False. To replicate the experiments in the paper, you will need to train with 4 gpus of at least 12GB. To select what gpus to use, you can use the --gpus argument.


# Citation
```
@InProceedings{Baradad_2020_CVPR,
  author = {Baradad, Manel and Torralba, Antonio},
  title = {Height and Uprightness Invariance for 3D Prediction From a Single View},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```
