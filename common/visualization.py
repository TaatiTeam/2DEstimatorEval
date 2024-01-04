import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


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


def visualize_sequences(sequence_2d, sequence_3d, interval=50):
    assert sequence_3d.shape[0] == sequence_2d.shape[0], "Number of frames should be equal"

    def update(frame):
        ax1.clear()
        ax2.clear()

        ax1.set_title('2D estimation')
        ax2.set_title('3D')

        ax1.set_xlim([min_x_2d, max_x_2d])
        ax1.set_ylim([min_y_2d, max_y_2d])
        ax2.set_xlim3d([min_x_3d, max_x_3d])
        ax2.set_ylim3d([min_y_3d, max_y_3d])
        ax2.set_zlim3d([min_z_3d, max_z_3d])
        ax2.set_box_aspect(aspect_ratio_3d)

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

            ax2.plot(xs, ys, zs)
        ax2.scatter(x_3d, y_3d, z_3d)

    print("Number of frames:", sequence_2d.shape[0])

    min_x_3d, min_y_3d, min_z_3d = np.min(sequence_3d, axis=(0, 1))
    max_x_3d, max_y_3d, max_z_3d = np.max(sequence_3d, axis=(0, 1))

    min_x_2d, min_y_2d = np.min(sequence_2d, axis=(0, 1))
    max_x_2d, max_y_2d = np.max(sequence_2d, axis=(0, 1))

    x_range_3d = max_x_3d - min_x_3d
    y_range_3d = max_y_3d - min_y_3d
    z_range_3d = max_z_3d - min_z_3d
    aspect_ratio_3d = [x_range_3d, y_range_3d, z_range_3d]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 4))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2 = fig.add_subplot(122, projection='3d')

    ani = FuncAnimation(fig, update, frames=sequence_3d.shape[0], interval=interval)
    ani.save(f'out.gif', writer='pillow')

    plt.close(fig)