# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import random
from datetime import datetime, timezone, timedelta
import numpy as np

import torch

def get_current_time():
    KST = timezone(timedelta(hours=9))
    now = datetime.now(tz=KST)
    
    return (now.day, now.hour, now.minute)
    
    
def seed_everything(random_seed=21):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(hist):
    """Returns accuracy score evaluation result.
      - mean IU
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)

    return mean_iu
