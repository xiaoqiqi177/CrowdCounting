import hashlib
import numpy as np

def add_rand_seed_entropy(v):
    """add entropy for :func:`sxatable_rand_seed`

    :param v: ``str(v)`` would be used to provide entropy
    """
    add_rand_seed_entropy.seed_entropy.append(str(v))

add_rand_seed_entropy.seed_entropy = []

def stable_rand_seed(self, width=32):
    sha = hashlib.sha256()
    sha.update(str(type(self)).encode('utf-8'))
    sha.update('\n'.join(add_rand_seed_entropy.seed_entropy).encode('utf-8'))
    return int(sha.hexdigest()[23:23+width//4], 16)

def stable_rng(self):
	return np.random.RandomState(stable_rand_seed(self))

def pad_image_to_shape(img, shape, *, return_padding=False):
    """
    Zeros pad the given image to given shape whiling keeping the image
    in the center;
    :param shape: (h, w)
    :param return_padding:
    """
    shape = list(shape[:2])
    if img.ndim > 2:
        shape.extend(img.shape[2:])
    shape = tuple(shape)

    h, w = img.shape[:2]
    assert w <= shape[1] and h <= shape[0]
    pad_width = shape[1] - w
    pad_height = shape[0] - h

    pad_w0 = pad_width // 2
    pad_w1 = shape[1] - (pad_width - pad_w0)
    pad_h0 = pad_height // 2
    pad_h1 = shape[0] - (pad_height - pad_h0)

    ret = np.zeros(shape, dtype=img.dtype)
    ret[pad_h0:pad_h1, pad_w0:pad_w1] = img
    if return_padding:
        return ret, (pad_h0, pad_w0)
    else:
        return ret

def padimg(img, k):
    h, w = img.shape[:2]
    newh = ((h + k - 1) // k) * k
    neww = ((w + k - 1) // k) * k
    newimg = np.zeros((newh, neww, img.shape[2]), dtype='uint8')
    newimg[:h, :w, :] = img
    return newimg
