"""
{'AsapSCIENCE',
 'Bozeman Science',
 'Fun Science',
 'Ideas in Science',
 'India Science',
 'Insider Science',
 'Mathematical Science',
 'Naked Science',
 'Ninja Nerd',
 'Science Is Dope',
 'Science News',
 'Science explained',
 'Science vs Cinema',
 'Simple Science',
 'Smithsonian Science Education Center',
 'VTV Science',
 'science to technology'}
 
480x360

"""

import pandas as pd
import numpy as np
from functions import url_to_numpy_array


df = pd.read_csv("dataset1.csv")
df = df[["thumbnail", "views_wise_outperformed"]]
df1 = df[df["views_wise_outperformed"] == 1]
df2 = df[df["views_wise_outperformed"] == 0][: len(df1)]
df = pd.concat([df1, df2], axis=0)
df = df.sample(frac=1)
df_train = df[: int(len(df) * (0.9))]
df_test = df[int(len(df) * (0.9)) :]


def get_dataset_detail():
    return {"train_len": len(df_train), "test_len": len(df_test)}


def load_training(idx, batch_size):
    temp_df = df_train[idx * batch_size : (idx + 1) * batch_size]
    arrays = []
    for thumbnail in temp_df["thumbnail"]:
        img = url_to_numpy_array(thumbnail)
        img = img[np.newaxis, :, :, :]
        img = img.swapaxes(1, 3)
        arrays.append(img)
    if len(arrays) > 0:
        return (
            np.concatenate(arrays, axis=0),
            np.array(temp_df["views_wise_outperformed"])[:, np.newaxis],
        )
    else:
        return None, None


def load_testing(idx, batch_size):
    temp_df = df_test[idx * batch_size : (idx + 1) * batch_size]
    arrays = []
    for thumbnail in temp_df["thumbnail"]:
        img = url_to_numpy_array(thumbnail)
        img = img[np.newaxis, :, :, :]
        img = img.swapaxes(1, 3)
        arrays.append(img)
    if len(arrays) > 0:
        return (
            np.concatenate(arrays, axis=0),
            np.array(temp_df["views_wise_outperformed"])[:, np.newaxis],
        )
    else:
        return None, None
