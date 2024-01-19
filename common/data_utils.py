import numpy as np
import os
import copy
from common.camera import world_to_camera, normalize_screen_coordinates
from common.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq
from common.utils import deterministic_random


def fetch(subjects, stride, keypoints, dataset, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]


    return out_camera_params, out_poses_3d, out_poses_2d


def create_checkpoint_dir_if_not_exists(checkpoint_dir):
    """
    Creates a new directory for checkpoints if it doesn't exist already.

    Args:
        checkpoint_dir (str): Path to the checkpoint directory
    """
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
    except OSError as e:
        raise RuntimeError('Unable to create checkpoint directory:', checkpoint_dir)
    

def preprocess_3d_data(dataset):
    """
    Preprocesses the data by doing the following things:
    - Changes the coordinate from world coordinate to camera coordinate using camera extrinsic parameters.
    - Centeralizes the 3D Positions.

    Args:
        dataset (Human36mDataset): dataset containing the 3D sequences.
    """
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d -= pos_3d[:, :1] # Remove global offset
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d


def concatenate_2d_data(keypoints_2d, keypoints_2d_new):
    """
    Concatenates the new keypoints 2d to the keypoints_2d dictionarry.
    
    Args:
        keypoints_2d (dict): A dictionary containing the 2D keypoints for each action of different subjects in different viewpoints.
        keypoints_2d_new (dict): A new dictionary to be concatenated (on the numpy tensors)
    """
    for subject in keypoints_2d:
        for action in keypoints_2d[subject]:
            sequence_list = keypoints_2d[subject][action]
            sequence_list_new = keypoints_2d_new[subject][action]
            for i in range(len(sequence_list)):
                sequence, new_sequence = sequence_list[i], sequence_list_new[i]
                concatenated_sequence = np.concatenate((sequence, new_sequence), axis=-1)
                sequence_list[i] = concatenated_sequence

def load_2d_data(path):
    """
    Loads the data containing 2D sequences.

    Args:
        path (str): Path to where 2D data is located
    
    Returns:
        keypoints (dict): A dictionary containing 2D pose sequences.
        kps_left (list): List of indices corresponding to left joints of body
        kps_right (list): List of indices corresponding to right joints of body
        num_joints (int): Number of joints
    """
    keypoints = np.load(path, allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    num_joints  = keypoints_metadata['num_joints']
    keypoints = keypoints['positions_2d'].item()

    return keypoints, kps_left, kps_right, num_joints


def verify_2d_3d_matching(keypoints_2d, dataset_3d):
    """
    Verifies for each 3D data we have 2D data with exact same number of frames.
    In case if we have more frames for 2D compared to 3D, it throws away the last few frames from 2D.

    Args:
        keypoints_2d (dict): Dictionary containing 2D data.
        dataset_3d (Human36mDataset): dataset containing the 3D sequences.
    """
    for subject in dataset_3d.subjects():
        assert subject in keypoints_2d, f'Subject {subject} is missing from the 2D detections dataset'
        for action in dataset_3d[subject].keys():
            assert action in keypoints_2d[subject], f'Action {action} of subject {subject} is missing from the 2D detections dataset'
            if 'positions_3d' not in dataset_3d[subject][action]:
                continue

            for cam_idx in range(len(keypoints_2d[subject][action])):

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset_3d[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints_2d[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints_2d[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints_2d[subject][action][cam_idx] = keypoints_2d[subject][action][cam_idx][:mocap_length]

            assert len(keypoints_2d[subject][action]) == len(dataset_3d[subject][action]['positions_3d'])


def normalize_2d_data(keypoints_2d, dataset_3d):
    """
    Normalizes 2D sequence to be in range [-1, 1]

    Args:
        keypoints_2d (dict): Dictionary containing 2D data.
        dataset_3d (Human36mDataset): dataset containing the 3D sequences.
    """
    for subject in keypoints_2d.keys():
        for action in keypoints_2d[subject]:
            for cam_idx, kps in enumerate(keypoints_2d[subject][action]):
                # Normalize camera frame
                cam = dataset_3d.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints_2d[subject][action][cam_idx] = kps


def init_train_generator(subjects_train, keypoints_2d, dataset_3d, pad, kps_left, kps_right,
                         joints_left, joints_right, stride, action_filter, args):
    """
    Initializes train generator.

    Args:
        subjects_train (list): List of train subjects
        keypoints_2d (dict): Dictionary containing 2D data.
        dataset_3d (Human36mDataset): dataset containing the 3D sequences.
        pad (int): padding
        kps_left (list): List of indices corresponding to left joints of body (2D sequence)
        kps_right (list): List of indices corresponding to right joints of body (2D sequence)
        joints_left (list): List of indices corresponding to left joints of body (3D sequence)
        joints_right (list): List of indices corresponding to right joints of body (3D sequence)
        stride (int): if > 1, it downsamples the sequences
        action_filter (list): List of action to filter out. None if empty
        args (object): Command-line arguments.
    """
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, stride,
                                                       keypoints_2d, dataset_3d,
                                                       action_filter, subset=args.subset)
    train_generator = ChunkedGenerator_Seq(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.number_of_frames,
                                       pad=pad, causal_shift=0, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    return train_generator


def init_test_generator(subjects_test, keypoints_2d, dataset_3d, pad, kps_left,
                        kps_right, joints_left, joints_right, stride, action_filter):
    """
    Initializes test generator.

    Args:
        subjects_test (list): List of test subjects
        keypoints_2d (dict): Dictionary containing 2D data.
        dataset_3d (Human36mDataset): dataset containing the 3D sequences.
        pad (int): padding
        kps_left (list): List of indices corresponding to left joints of body (2D sequence)
        kps_right (list): List of indices corresponding to right joints of body (2D sequence)
        joints_left (list): List of indices corresponding to left joints of body (3D sequence)
        joints_right (list): List of indices corresponding to right joints of body (3D sequence)
        stride (int): if > 1, it downsamples the sequences
        action_filter (list): List of action to filter out. None if empty
    """
    cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, stride,
                                                       keypoints_2d, dataset_3d,
                                                       action_filter)
    test_generator = UnchunkedGenerator_Seq(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=0, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    return test_generator


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data