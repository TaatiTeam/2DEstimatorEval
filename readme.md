# Dataset
We have used Human3.6M dataset for this project. For downloading it, please send a request to the [official website](http://vision.imar.ro/human3.6m/description.php) or download the preprocessed versions from [our google drive](https://drive.google.com/drive/folders/1K8LxjHjxyztfnbF5rckw_3gHce0eLlC8?usp=sharing). Short description of each files:
- `data_2d_h36m_gt.npz`: 2D groundtruth data
- `data_2d_h36m_vitpose.npz`: 2D data estimated by ViTPose-H
- `data_2d_h36m_cpn_ft_h36m_dbb.npz`: 2D data estimated by CPN (finetuned on Human3.6M Dataset)
- `data_2d_h36m_cpnnpz`: 2D data estimated by CPN without finetuning
- `data_2d_h36m_detectron_ft_h36m.npz`: 2D data estimated by Detectron (finetuned on Human3.6M Dataset)
- `data_2d_h36m_moganet.npz`: 2D data estimated by MogaNet
- `data_2d_h36m_pct.npz`: 2D data estimated by PCT
- `data_2d_h36m_transpose.npz`: 2D data estimated by TransPose
- `data_2d_h36m_merge_average.npz`: Average merging dataset
- `data_2d_h36m_merge_weighted_average.npz`: Weighted average merging dataset
- `data_2d_h36m_merge_manual.npz`: WTA merging dataset
- `data_3d_h36m.npz`: 3D data

After downloading them, place them in `data` directory. Note that for concatenate merging we don't provide a separe npz file and we build it into the code by merging ViTPose, PCT, and MogaNet.
 
## Preprocessing
This is needed only in case you're using the official website CDF files provided by Human3.6M. In case of using our preprocessed data, you can ignore these steps.
### Setup from original source (CDF files)
Among the given files from official webiste of Human3.6M, download `Poses_D3_Positions_<SUBJECT>.tgz` files where `<SUBJECT>` is S1, S5, S6, S7, S8, S9, S11. Then extract them all and put them in a folder with an arbitrary name (let's call `cdf_files` here). Next, put the folder into this project under `data/preprocess` directory. It's expected to have following structure of files:
```
data/preprocess/cdf_files/S1/MyPoseFeatures/D3_Positions/Directions 1.cdf
data/preprocess/cdf_files/S1/MyPoseFeatures/D3_Positions/Directions.cdf
...
```
Finally run the preprocessing script:
```
cd data
python prepare_data_h36m.py --cdf-dir cdf_files
```
After executing the script above, these files should be generated under `data` folder:
- `data_2d_h36m_gt.npz`: This file contains all the 2D ground truths that can be achieved by projecting the 3D coordinates into 2D pixels by using the 4 camera intrinsic parameters used for recording the dataset. The way to read the data and its structure is as follows:
```python
data_2d = np.load(args.dir_2d, allow_pickle=True)['positions_2d'].item()

"""
Structure of data_2d:
{
    <SUBJECT>: {
        <ACTION>: [<np.ndarray with shape(n_frames, 17, 2)>] x4
    }
}
Where:
      <SUBJECT> is one of the followings: ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

      <ACTION> is one of the followings: ['Directions 1', 'Directions', 'Discussion 1', 'Discussion', 'Eating 2', 'Eating', 'Greeting 1', 'Greeting', 'Phoning 1', 'Phoning', 'Posing 1', 'Posing', 'Purchases 1', 'Purchases', 'Sitting 1', 'Sitting 2', 'SittingDown 2', 'SittingDown', 'Smoking 1', 'Smoking', 'Photo 1', 'Photo', 'Waiting 1', 'Waiting', 'Walking 1', 'Walking', 'WalkDog 1', 'WalkDog', 'WalkTogether 1', 'WalkTogether']

Note that each sequence of action is recorded using 4 different cameras positioned in 4 different corners of room. So we have sequences of 4 different views for the same performed action.
"""
```
- `data_3d_h36m.npz`: This file contains the 3D coordinates. It's structured as follows:
```python
data_3d = np.load(args.dir_3d, allow_pickle=True)['positions_3d'].item()

"""
Structure of data_3d:
{
    <SUBJECT>: {
        <ACTION>: [<np.ndarray with shape(n_frames, 32, 3)>] x4
    }
}

Note that 3D coordinates are captured using 32 keypoints but 17 of them is used as a ground truth
throughout the training. So later in run_poseformer.py the main 17 keypoints will be selected.
"""
```

### Preparing the output of 2D pose estimations.
It's expected that outputs of 2D estimator to be located in a directory with the following structure:
```
.
└── h36m_<DETECTOR NAME>/
    ├── S1/
    │   ├── <ACTION NAME>_<CAMERA ID>_pose_sequence.npy
    │   └── ...
    ├── S5/
    │   └── ...
    ├── S6/
    │   └── ...
    ├── S7/
    │   └── ...
    ├── S8/
    │   └── ...
    ├── S9/
    │   └── ...
    └── S11/
        └── ...
```
Where `<DETECTOR NAME>` is one of the 2D estimators like vitpose, pct, etc. By placing this folder inside `data/preprocess` folder, we can run the following script to preprocess the data:
```
cd data
python prepare_2d_estimation.py --detector <DETECTOR NAME>
```
The script aboves converts them from COCO format to Human3.6M format and stores them in the following structure:
```python
data_2d = np.load(args.dir_2d, allow_pickle=True)['positions_2d'].item()

"""
Structure of data_2d:
{
    <SUBJECT>: {
        <ACTION>: [<np.ndarray with shape(n_frames, 17, 2)>] x4
    }
}
"""
```

### Preparing merged dataset
**Note:** The merged estimations are also located in [our google drive](https://drive.google.com/drive/u/2/folders/1K8LxjHjxyztfnbF5rckw_3gHce0eLlC8). Instructions below is just for the case of reproducing the results.

We have proposed 3 different merging strategies. The code to create the merged dataset can be executed as follows:
```
cd data
python merge.py --strategy <MERGING-STRATEGY>
```
Where `<MERGING-STRATEGY>` is one of these options:
- **manual**: This option mainly uses ViTPose but replaces joints (2, 8, 10, 14) with PCT (Refer to the paper report to see the reason). In paper, it's referred as WTA merging.
- **average**: This strategy takes the average of PCT, MogaNet, and ViTPose for each frame.
- **weighted_average**: This strategy takes the weighted average such that weights depend on the confidence scores.

**Note**: For the manual one, `data_2d_h36m_vitpose.npz` and `data_2d_h36m_pct.npz` and for other two options, `data_2d_h36m_vitpose.npz`, `data_2d_h36m_pct.npz`, and `data_2d_h36m_moganet.npz` should be located in data directory.

**Note**: For the weighted_average option, the keypoints should have confidence scores. So for generating keypoints using `prepare_2d_estimation.py`, you should also pass `--keep-conf` as argument (the output npz will have `_w_conf` at the end).

## Visualization
For dataset visualization, you need to have the following npz files in data directory:
- **data_3d_h36m.npz**
- **data_2d_h36m_gt.npz**
- And one of the npz files from 2D estimations

Then you can run the following code:
```
cd data

python visualize.py --dir-2d <PATH TO THE 2D ESTIMATIONS> --subject <SUBJECT> --action <ACTION> --camera <CAMERA ID>
```
Where:
 - `<PATH TO THE 2D ESTIMATIONS>`: By default it's set to ViTPose npz file.
 - `<SUBJECt>`: By default it is set to `S1`. The available options are:
 ```
 ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
 ```
 - `<ACTION>`: By default it is set to `Walking`. The available options are:
 ```
 ['Directions 1', 'Directions', 'Discussion 1', 'Discussion', 'Eating 2', 'Eating', 'Greeting 1', 'Greeting', 'Phoning 1', 'Phoning', 'Posing 1', 'Posing', 'Purchases 1', 'Purchases', 'Sitting 1', 'Sitting 2', 'SittingDown 2', 'SittingDown', 'Smoking 1', 'Smoking', 'Photo 1', 'Photo', 'Waiting 1', 'Waiting', 'Walking 1', 'Walking', 'WalkDog 1', 'WalkDog', 'WalkTogether 1', 'WalkTogether']
 ```
 - `<CAMERA>`: By default it is set to `55011271`. The available options are:
 ```
 ['54138969', '55011271', '58860488', '60457274']
 ```

 It usually takes some time to render the output file as they are usually some long videos (~1500 frames). Sample of one of the visualizations (trimmed):
 <p><img src="figs/sample_data.gif" alt="" /></p>
 In the visualization above, 2D estimation belongs to ViTPose.

# Training and evaluation
## Training PoseFormerV2
You can train the model by running the `run_poseformer.py`. Sample for ViTPose is as follows:
```
python run_poseformer.py --keypoints vitpose \
                         --batch-size 1024 \
                         -frame 27 \
                         -frame-kept 3 \
                         -coeff-kept 3  \
                         --checkpoint checkpoint/vitpose \
                         --epochs 200 \
                         --wandb-name poseformerv2-vitpose
```
The script above stores the checkpoints in `checkpoint/vitpose` directory. It trains for 200 epochs and because of `--keypoints vitpose`, it reads the npz file that belongs to vitpose (basically in `data_2d_h36m_vitpose.npz` file name it cares about whatever comes after last underscore).

Also in case that you want to resume training if it stops during training for some reason, you can do the following:
```
python run_poseformer.py --keypoints vitpose \
                         --batch-size 1024 \
                         -frame 27 \
                         -frame-kept 3 \
                         -coeff-kept 3  \
                         --checkpoint checkpoint/vitpose \
                         --epochs 200 \
                         --wandb-name poseformerv2-vitpose \
                         --resume last_epoch.bin \
                         --wandb-id fnwki6ko 
```
Where you have to find the `wandb-id` from the wandb run id. After the end of training, the script above uplodas the model weights into wandb server for future usage.

## Training MotionAGFormer
You can train as follows:
```
python run_motionagformer.py --wandb-name MotionAGFormer-vitpose \
                              --checkpoint checkpoint/vitpose \
                              --number-of-frames 243 \
                              --epochs 60 \
                              --keypoints vitpose
```

## Pretrained weights
- [PoseFormerV2 weights](https://drive.google.com/drive/u/2/folders/1ET2y9MpanDvsrOceuox2toWwIZNTdlKw)
- [MotionAGFormer weights](https://drive.google.com/drive/folders/19eDxxTMbsJKTTVJW6K_icyknHKSegMSF?usp=drive_link)


## Acknowledgement
Our code refers to the following repositories:
- [PoseformerV2](https://github.com/QitaoZhao/PoseFormerV2)
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/)