from datetime import datetime
import numpy as np


def now_str():
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


class Logger:
    """
    A class for logging Trainer messages.
    """
    def __init__(self, verbose: bool = True, n_digits: int = 5, zfill: int = 4):
        self.verbose = verbose
        self.n_digits = n_digits
        self.zfill = zfill

    def message(self, msg: str):
        out = " [INFO]: " + msg
        if self.verbose:
            out = now_str() + out
        print(out)

    def warning(self, msg: str):
        out = " [WARN]: " + msg
        if self.verbose:
            out = now_str() + out
        print(out)

    def error(self, msg: str):
        out = "[ERROR]: " + msg
        if self.verbose:
            out = now_str() + out
        print(out)
        raise RuntimeError(msg)

    def train_info(self, i_epoch, train_loss_vae, test_loss_vae, train_loss_disc, test_loss_disc):
        str1 = str(i_epoch).zfill(self.zfill)
        l1 = round(train_loss_vae, self.n_digits)
        l2 = round(test_loss_vae, self.n_digits)
        l3 = round(train_loss_disc, self.n_digits)
        l4 = round(test_loss_disc, self.n_digits)
        msg = "Epoch: {}, VAE loss: ({}, {}), Discriminator loss: ({}, {})".format(str1, l1, l2, l3, l4)
        self.message(msg)

    def assert_shape(self, arr: np.ndarray, arr_name: str, expected: tuple):
        if arr.shape != expected:
            msg = 'Shape of ' + arr_name + ' should be ' + str(expected) + '. Was: ' + str(arr.shape) + '.'
            self.error(msg)



