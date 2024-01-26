import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
sys.path.append('../')
from common.skeleton import Skeleton

h36m_skeleton = Skeleton(parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
       joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
       joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

connections = [
    (10, 9),
    (9, 8),
    (8, 11),
    (8, 14),
    (14, 15),
    (15, 16),
    (11, 12),
    (12, 13),
    (8, 7),
    (7, 0),
    (0, 4),
    (0, 1),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6)
]

cam_to_idx = {
         '54138969': 0,
         '55011271': 1,
         '58860488': 2,
         '60457274': 3
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-3d', default="data_3d_h36m.npz", type=str, metavar='PATH')
    parser.add_argument('--dir-2d-gt', default="data_2d_h36m_gt.npz", type=str, metavar='PATH')
    parser.add_argument('--dir-2d', default="data_2d_h36m_vitpose.npz", type=str, metavar='PATH')
    parser.add_argument('--subject', default="S1", type=str, choices=['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'])
    parser.add_argument('--action', default="Walking", type=str, choices=['Directions 1', 'Directions', 'Discussion 1', 'Discussion', 'Eating 2', 'Eating 1', 'Eating', 'Greeting 1', 'Greeting', 'Phoning 1', 'Phoning', 'Posing 1', 'Posing', 'Purchases 1', 'Purchases', 'Sitting 1', 'Sitting', 'Sitting 2', 'SittingDown 2', 'SittingDown', 'Smoking 1', 'Smoking', 'Photo 1', 'Photo', 'Waiting 1', 'Waiting', 'Walking 1', 'Walking', 'WalkDog 1', 'WalkDog', 'WalkTogether 1', 'WalkTogether'])
    parser.add_argument('--camera', default="55011271", type=str, choices=['54138969', '55011271', '58860488', '60457274'])
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    return args


def visualize_sequences(sequence_2d_gt, sequence_2d, sequence_3d, interval=50):
    assert sequence_3d.shape[0] == sequence_2d_gt.shape[0], "Number of frames should be equal"

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear() 

        ax1.set_title('2D estimation')
        ax2.set_title("2D projection")
        ax3.set_title('3D')

        ax1.set_xlim([min_x_2d, max_x_2d])
        ax1.set_ylim([min_y_2d, max_y_2d])
        ax2.set_xlim([min_x_2d, max_x_2d])
        ax2.set_ylim([min_y_2d, max_y_2d])
        ax3.set_xlim3d([min_x_3d, max_x_3d])
        ax3.set_ylim3d([min_y_3d, max_y_3d])
        ax3.set_zlim3d([min_z_3d, max_z_3d])
        ax3.set_box_aspect(aspect_ratio_3d)

        # estimated 2D
        x_2d = sequence_2d[frame, :, 0]
        y_2d = sequence_2d[frame, :, 1]
        for connection in connections:
            start = sequence_2d[frame, connection[0], :]
            end = sequence_2d[frame, connection[1], :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            ax1.plot(xs, ys)
        ax1.scatter(x_2d, y_2d)

        # Ground-truth 2D
        x_2d_gt = sequence_2d_gt[frame, :, 0]
        y_2d_gt = sequence_2d_gt[frame, :, 1]
        for connection in connections:
            start = sequence_2d_gt[frame, connection[0], :]
            end = sequence_2d_gt[frame, connection[1], :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            ax2.plot(xs, ys)
        ax2.scatter(x_2d_gt, y_2d_gt)

        # 3D
        x_3d = sequence_3d[frame, :, 0]
        y_3d = sequence_3d[frame, :, 1]
        z_3d = sequence_3d[frame, :, 2]

        for connection in connections:
            start = sequence_3d[frame, connection[0], :]
            end = sequence_3d[frame, connection[1], :]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            zs = [start[2], end[2]]

            ax3.plot(xs, ys, zs)
        ax3.scatter(x_3d, y_3d, z_3d)


    min_x_3d, min_y_3d, min_z_3d = np.min(sequence_3d, axis=(0, 1))
    max_x_3d, max_y_3d, max_z_3d = np.max(sequence_3d, axis=(0, 1))

    min_x_2d, min_y_2d = np.min(sequence_2d_gt, axis=(0, 1))
    max_x_2d, max_y_2d = np.max(sequence_2d_gt, axis=(0, 1))

    x_range_3d = max_x_3d - min_x_3d
    y_range_3d = max_y_3d - min_y_3d
    z_range_3d = max_z_3d - min_z_3d
    aspect_ratio_3d = [x_range_3d, y_range_3d, z_range_3d]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4))
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3 = fig.add_subplot(122, projection='3d')

    ani = FuncAnimation(fig, update, frames=sequence_3d.shape[0], interval=interval)
    ani.save(f'out.gif', writer='pillow')

    plt.close(fig)


def main():
    valid_joints = h36m_skeleton.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
    args =parse_args()
    data_3d = np.load(args.dir_3d, allow_pickle=True)['positions_3d'].item()
    data_2d_gt = np.load(args.dir_2d_gt, allow_pickle=True)['positions_2d'].item()
    data_2d = np.load(args.dir_2d, allow_pickle=True)['positions_2d'].item()

    sequence_3d = data_3d[args.subject][args.action][:, valid_joints]
    sequence_2d_gt = data_2d_gt[args.subject][args.action][cam_to_idx[args.camera]]
    sequence_2d = data_2d[args.subject][args.action][cam_to_idx[args.camera]]

    sequence_2d[..., 1] *= -1
    sequence_2d_gt[..., 1] *= -1
    
    sequence_2d = sequence_2d[:sequence_2d_gt.shape[0]]

    if args.debug:
        sequence_3d = sequence_3d[:200]
        sequence_2d_gt = sequence_2d_gt[:200]
        sequence_2d = sequence_2d[:200]
    

    visualize_sequences(sequence_2d_gt, sequence_2d, sequence_3d)



if __name__ == "__main__":
    main()