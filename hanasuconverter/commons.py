import torch

def sequence_mask(length, max_length=None):
    """
    Create a sequence mask for given lengths

    Args:
        length: Tensor of lengths [B]
        max_length: Maximum length (if None, use max of length)

    Returns:
        Mask tensor [B, T] where T is max_length
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def slice_segments(x, ids_str, segment_size=4096):
    """
    Slice segments from tensor at specified indices

    Args:
        x: Input tensor [B, C, T]
        ids_str: Starting indices [B]
        segment_size: Segment size

    Returns:
        Sliced segments [B, C, segment_size]
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4096):
    """
    Randomly slice segments from tensor

    Args:
        x: Input tensor [B, C, T]
        x_lengths: Lengths of input tensor [B]
        segment_size: Segment size

    Returns:
        Sliced segments [B, C, segment_size], segment start indices [B]
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str

def kl_divergence(m_p, logs_p, m_q, logs_q):
    """
    KL divergence between two Gaussian distributions

    Args:
        m_p: Mean of p distribution [B, C, T]
        logs_p: Log standard deviation of p distribution [B, C, T]
        m_q: Mean of q distribution [B, C, T]
        logs_q: Log standard deviation of q distribution [B, C, T]

    Returns:
        KL divergence [B, C, T]
    """
    kl = (logs_q - logs_p) - 0.5 + \
        0.5 * (torch.exp(2. * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2. * logs_q)
    return kl