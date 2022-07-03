import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_confusion_matrix_image(conf_mat):
    fig, ax = plt.subplots()
    labels = 'neutral', 'positive', 'negative'
    sns.heatmap(
        conf_mat,
        ax=ax,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
    )
    image = _matplolib_to_array(fig)
    fig.clf()
    plt.close('all')
    return image


def _matplolib_to_array(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=100)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()
    return img_arr
