import numpy as np
import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default="data_2d_h36m_detectron_pt_coco.npz", type=str, metavar='PATH')
    parser.add_argument('--detector', default="detectron_pt_h36m", type=str)
    args = parser.parse_args()
    return args


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
    args = parse_args()
    data_2d = np.load(args.dir, allow_pickle=True)['positions_2d'].item()
    metadata = {
        'layout_name': 'h36m',
        'num_joints': 17,
        'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
    }
    data_2d_converted = {}
    for subject in data_2d:
        for action in data_2d[subject]:
            if subject not in data_2d_converted:
                data_2d_converted[subject] = {action: [None, None, None, None]}
            else:
                data_2d_converted[subject][action] = [None, None, None, None]
            for idx, sequence in enumerate(data_2d[subject][action]):
                data_2d_converted[subject][action][idx] = coco2h36m(sequence[..., :2])
    np.savez_compressed(f"data_2d_h36m_{args.detector}", positions_2d=data_2d_converted, metadata=metadata)
    print('Done.')
    




if __name__ == "__main__":
    main()