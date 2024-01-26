from glob import glob
import os
import argparse
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', default='vitpose', type=str, choices=['vitpose', 'pct', 'moganet', 'transpose'])
    parser.add_argument('--keep-conf', action='store_true')
    args = parser.parse_args()
    return args

def find_smallest_value_bigger_than(query, sorted_list):
    for key in sorted_list:
        if key > query:
            return key


def coco2h36m(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 12, :]
    new_keypoints[..., 2, :] = keypoints[..., 14, :]
    new_keypoints[..., 3, :] = keypoints[..., 16, :]
    new_keypoints[..., 4, :] = keypoints[..., 11, :]
    new_keypoints[..., 5, :] = keypoints[..., 13, :]
    new_keypoints[..., 6, :] = keypoints[..., 15, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = (keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 5, :]
    new_keypoints[..., 12, :] = keypoints[..., 7, :]
    new_keypoints[..., 13, :] = keypoints[..., 9, :]
    new_keypoints[..., 14, :] = keypoints[..., 6, :]
    new_keypoints[..., 15, :] = keypoints[..., 8, :]
    new_keypoints[..., 16, :] = keypoints[..., 10, :]

    return new_keypoints


def main():
    cam_to_idx = {
         '54138969': 0,
         '55011271': 1,
         '58860488': 2,
         '60457274': 3
    }
    args = parse_args()
    files = glob(f"h36m_{args.detector}/*/*.npy")
    output = {}
    for file in tqdm(files):
        _, subject, file_name = file.split(os.sep)
        action, camera_id = file_name.split("_")[:2]

        # Use consistent naming convention
        action = action.replace('TakingPhoto', 'Photo') \
                                .replace('WalkingDog', 'WalkDog')

        if (subject == 'S11' and action == 'Directions') or action == "":
                continue # Discard corrupted video
        
        if subject not in output:
             output[subject] = {action: [None, None, None, None]}
        elif action not in output[subject]:
             output[subject][action] = [None, None, None, None]
        
        sequence = np.load(file, allow_pickle=True)
        if args.detector == "transpose":
            keypoints_2d = {}
            sequence = sequence.item()
            for frame in sequence:
                if sequence[frame] is not None:
                    keypoints_2d[frame] = sequence[frame]
        else:
            keypoints_2d = {}
            box_conf = {}
            for frame in sequence:
                if frame['frame'] in keypoints_2d and frame['bbox'][-1] < box_conf[frame['frame']]:
                    continue
                keypoints_2d[frame['frame']] = frame['keypoints'][None, ...]
                box_conf[frame['frame']] = frame['bbox'][-1]
        

        # Fill missing frames in middle if they're less than 20 frames
        recorded_frames = sorted(list(keypoints_2d.keys()))
        for i in range(recorded_frames[-1] + 1):
            if i not in keypoints_2d:
                next_frame = find_smallest_value_bigger_than(i, recorded_frames)
                for j in range(i, next_frame):
                    keypoints_2d[j] = keypoints_2d[i-1]

        try:
            keypoints_2d = np.concatenate([keypoints_2d[frame] for frame in range(len(keypoints_2d))])
        except KeyError as e:
             last_valid_frame = e.args[0] - 1
             keypoints_2d = np.concatenate([keypoints_2d[frame] for frame in range(last_valid_frame + 1)])
        if args.keep_conf:
            keypoints_2d_converted = coco2h36m(keypoints_2d)
        else:
            keypoints_2d_converted = coco2h36m(keypoints_2d[..., :2])

        try:
            cam_index = cam_to_idx[camera_id]
        except KeyError:
             continue
        
        output[subject][action][cam_index] = keypoints_2d_converted
    
    # Check there is no None value
    for subject in output:
        for action in output[subject]:
            for sequence in output[subject][action]:
                if sequence is None:
                    raise Exception(f'There is a None in subject "{subject}" action "{action}"')

    print('Saving...')
    metadata = {
        'num_joints': 17,
        'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
    }
    conf_name = "_w_conf" if args.keep_conf else ""
    np.savez_compressed(f"data_2d_h36m_{args.detector}{conf_name}", positions_2d=output, metadata=metadata)
    print('Done.')

if __name__ == "__main__":
    main()