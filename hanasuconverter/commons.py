import torch
from torch.nn import functional as F

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def sequence_mask(length, max_length=None):
    """
    Creates a boolean mask from sequence length.
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def slice_segments(x, ids_str, segment_size=4):
    """
    Slice segments from a tensor based on indices.
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """
    Randomly slice segments from a tensor.
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str

def generate_path(duration, mask):
    """
    Generate path based on duration.
    """
    device = duration.device
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path

def convert_pad_shape(pad_shape):
    """
    Convert pad shape to match conv2d padding format.
    """
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape

def intersperse(lst, item):
    """
    Insert item between each element of lst.
    """
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

def kl_divergence(m_p, logs_p, m_q, logs_q):
    """
    KL divergence between two Gaussian distributions.
    """
    kl = (logs_q - logs_p) - 0.5 + (torch.exp(2. * logs_p) + (m_p - m_q)**2) / (2. * torch.exp(2. * logs_q))
    return kl

def rand_gumbel(shape):
    """
    Sample from the Gumbel distribution.
    """
    uniform_samples = torch.rand(shape) * 0.99 + 0.005
    return -torch.log(-torch.log(uniform_samples))

def rand_gumbel_like(x):
    """
    Sample from the Gumbel distribution like x.
    """
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g

def slice_pitch_segments(x, ids_str, segment_size=4):
    """
    Slice pitch segments from a tensor based on indices.
    """
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret

def rand_slice_pitch_segments(x, x_lengths=None, segment_size=4):
    """
    Randomly slice pitch segments from a tensor.
    """
    b, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_pitch_segments(x, ids_str, segment_size)
    return ret, ids_str
