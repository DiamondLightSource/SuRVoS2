import numpy as np
import pandas as pd


#
# GEOMETRIES
#


def centroid_3d(arr):
    length = arr.shape[0]

    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])

    return sum_x / length, sum_y / length, sum_z / length


def rescale_3d(X, x_scale, y_scale, z_scale):
    X_rescaled = np.zeros_like(X)
    X_rescaled[:, 0] = X[:, 0] * x_scale
    X_rescaled[:, 1] = X[:, 1] * y_scale
    X_rescaled[:, 2] = X[:, 2] * z_scale

    return X_rescaled


def prepare_points3d(img_vol_shape=[100, 100, 100], proj="hunt"):
    """Prepare a set of 3d test points

    Keyword Arguments:
        img_vol_shape {list} -- shape of the image vol to fill with points (default: {[100,100,100]})
        proj {str} -- project code (default: {'hunt'})

    Returns:
        {np.array} -- array of 3d points
    """

    num_bb = 45

    if proj == "hunt":
        try:
            sliceno = 60
            df_zero = df2.loc[df2["subject_metadata_slice"] == sliceno]
            xs = df_zero["T3_x_true_posx"].values
            ys = df_zero["T3_x_true_posy"].values
            # ellipses = [[[x-10, y-10],[x-10,y+10],[x+10,y+10],[x+10,y-10]] for (x, y) in zip(xs,dt.shape[2]-ys)]
            points = [[x, y] for (x, y) in zip(xs, ys)]

        except NameError as e:
            points3d = np.array(
                list(
                    zip(
                        np.random.random(
                            (num_bb, 1),
                        ),
                        np.random.random(
                            (num_bb, 1),
                        ),
                        np.random.random((num_bb, 1)),
                    )
                )
            ).reshape((num_bb, 3))
            points3d[:, 0] = points3d[:, 0] * img_vol_shape[0]
            points3d[:, 1] = points3d[:, 1] * img_vol_shape[1]
            points3d[:, 2] = points3d[:, 2] * img_vol_shape[2]

    elif proj == "vf":
        try:
            points3d = np.array([[z, x, y] for (z, x, y) in sel_clicks2])
        except NameError as e:
            points3d = np.array(
                list(
                    zip(
                        np.random.random(
                            (num_bb, 1),
                        ),
                        np.random.random(
                            (num_bb, 1),
                        ),
                        np.random.random((num_bb, 1)),
                    )
                )
            ).reshape((num_bb, 3))

            points3d[:, 0] = points3d[:, 0] * img_vol_shape[0]
            points3d[:, 1] = points3d[:, 1] * img_vol_shape[1]
            points3d[:, 2] = points3d[:, 2] * img_vol_shape[2]

        logger.info("Size of points array: {}".format(points3d.shape))

    return points3d
