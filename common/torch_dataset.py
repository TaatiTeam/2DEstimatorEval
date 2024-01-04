import torch
from torch.utils.data import Dataset
import numpy as np
from common.data_utils import flip_data


class H36MTorchDataset(Dataset):
    def __init__(self, split, keypoints_2d, dataset_3d, n_frames, stride_ratio, flip=True):
        """
        Torch dataset for Human3.6M dataset.
        
        Args:
            split (str): Either 'train' or 'test'
            keypoints_2d (dict): A dictionary containing subjects -> actions -> camera_indices -> sequence
            dataset_3d (Human36mDataset): it's containing subjects -> actions -> positions_3d -> camera_indices -> sequence
            n_frames (int): Number of frames after splitting into clips:
            stride_ratio (int): stride = n_frames / stride_ratio. Useful for having overlap between training sequences
            flip (bool): If True, flips the sequence with probability of 50% in __getitem__ of training data.
        """
        assert split in ['train', 'test']
        self.split = split
        if self.split == 'train':
            self.subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        else:
            self.subjects = ['S9', 'S11']
            stride_ratio = 1
        self.sequences_2d, self.sequences_3d = self.partition_videos(keypoints_2d, dataset_3d,
                                                                     n_frames, stride_ratio)
        self.flip = flip

    def __len__(self):
        return len(self.sequences_3d)

    def __getitem__(self, idx):
        seq_2d, seq_3d = self.sequences_2d[idx], self.sequences_3d[idx]
        if self.split == 'train':
            if self.flip and np.random.random() > 0.5:
                seq_2d = flip_data(seq_2d)
                seq_3d = flip_data(seq_3d)
            
        return torch.FloatTensor(seq_2d), torch.FloatTensor(seq_3d)
                
    def partition_videos(self, keypoints_2d, dataset_3d, num_frames, stride_ratio):
        sequences_3d, sequences_2d = [], []
        for subject in dataset_3d.subjects():
            if subject not in self.subjects:
                continue
            for action in dataset_3d[subject].keys():
                if 'positions_3d' not in dataset_3d[subject][action]:
                    print(f"[WARN] dataset 3d doesn't have positions for subject '{subject}' action '{action}'")
                    continue
                for cam_idx in range(len(keypoints_2d[subject][action])):
                    sequence_2d = keypoints_2d[subject][action][cam_idx]
                    sequence_3d = dataset_3d[subject][action]['positions_3d'][cam_idx]
                    sequence_2d_clips, sequence_3d_clips = self.split_into_clips(sequence_2d,
                                                                                 sequence_3d,
                                                                                 num_frames,
                                                                                 num_frames // stride_ratio)
                    sequences_2d.extend(sequence_2d_clips)
                    sequences_3d.extend(sequence_3d_clips)
        return sequences_2d, sequences_3d
    
    @staticmethod
    def resample(original_length, target_length):
        """
        Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L68

        Returns an array that has indices of frames. elements of array are in range (0, original_length -1) and
        we have target_len numbers (So it interpolates the frames)
        """
        even = np.linspace(0, original_length, num=target_length, endpoint=False)
        result = np.floor(even)
        result = np.clip(result, a_min=0, a_max=original_length - 1).astype(np.uint32)
        return result
    
    def split_into_clips(self, sequence_2d, sequence_3d, clip_length, data_stride=81):
        assert sequence_2d.shape[0] == sequence_3d.shape[0]
        video_length = sequence_2d.shape[0]
        clips_2d, clips_3d = [], []
        if video_length <= clip_length:
            new_indices = self.resample(video_length, clip_length)
            clips_2d.append(sequence_2d[new_indices])
            clips_3d.append(sequence_3d[new_indices])
        else:
            start_frame = 0
            while (video_length - start_frame) >= clip_length:
                clips_2d.append(sequence_2d[start_frame:start_frame + clip_length])
                clips_3d.append(sequence_3d[start_frame:start_frame + clip_length])
                start_frame += data_stride
            new_indices = self.resample(video_length - start_frame, clip_length) + start_frame
            clips_2d.append(sequence_2d[new_indices])
            clips_3d.append(sequence_3d[new_indices])
        return clips_2d, clips_3d