import torch

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            # Initialize matched tensors with original tensors
            rl_matched, gl_matched = rl, gl
            
            # Create new tensors instead of modifying in-place
            if rl.dim() > 2 and gl.dim() > 2:  # Both have channel dimension
                if rl.size(2) > gl.size(2):
                    rl_matched = rl[:, :, :gl.size(2)]
                elif gl.size(2) > rl.size(2):
                    gl_matched = gl[:, :, :rl.size(2)]
            # Handle case where tensors have different dimensions
            elif rl.dim() != gl.dim():
                # Create new tensors with compatible shapes
                if rl.dim() < gl.dim():
                    # Clone and expand rl to match gl's dimensions
                    rl_matched = rl.clone()
                    diff = gl.dim() - rl.dim()
                    for _ in range(diff):
                        rl_matched = rl_matched.unsqueeze(-1)
                    rl_matched = rl_matched.expand_as(gl)
                else:
                    # Clone and expand gl to match rl's dimensions
                    gl_matched = gl.clone()
                    diff = rl.dim() - gl.dim()
                    for _ in range(diff):
                        gl_matched = gl_matched.unsqueeze(-1)
                    gl_matched = gl_matched.expand_as(rl)
            
            loss = loss + torch.mean(torch.abs(rl_matched - gl_matched))
    return loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # Initialize matched tensors with original tensors
        dr_matched, dg_matched = dr, dg
        
        # Create new tensors instead of modifying in-place
        if dr.dim() > 2 and dg.dim() > 2:
            # Make sure tensors have the same size along dimension 2
            if dr.size(2) > dg.size(2):
                dr_matched = dr[:, :, :dg.size(2)]
            elif dg.size(2) > dr.size(2):
                dg_matched = dg[:, :, :dr.size(2)]
        # For lower-dimensional tensors, ensure shapes are compatible
        elif dr.dim() != dg.dim():
            # Make dimensions compatible without in-place operations
            if dr.dim() < dg.dim():
                # Create a new tensor with dg's shape
                dr_matched = dr.reshape(*dg.size())
            else:
                # Create a new tensor with dr's shape
                dg_matched = dg.reshape(*dr.size())
            
        r_loss = torch.mean((1-dr_matched)**2)
        g_loss = torch.mean(dg_matched**2)
        loss = loss + (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss = loss + l

    return loss, gen_losses

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l