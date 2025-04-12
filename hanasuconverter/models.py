import torch
from torch import nn
from torch.nn import functional as F
from hanasuconverter.modules import init_weights
from hanasuconverter.modules import ResBlock1, ResBlock2, LRELU_SLOPE
from hanasuconverter.modules import ResidualCouplingLayer, Flip
from hanasuconverter.modules import WN
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

class ContentEncoder(nn.Module):
    """
    Content encoder to extract content information from audio
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_lengths, g=None):
        from hanasuconverter.commons import sequence_mask

        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        x = self.proj(x) * x_mask
        return x, x_mask

class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, tau=1.0):
        from hanasuconverter.commons import sequence_mask

        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # Sample z from posterior q(z|y, g) ~ N(m, exp(logs))
        # Use tau for sampling temperature during inference
        z = (m + torch.randn_like(m) * tau * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 2, 7, 1, padding=3, bias=False)  # Output 2 channels for stereo
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)  # Output is in range [-1, 1]

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()

class SpeakerEncoder(nn.Module):
    """
    Speaker encoder for extracting speaker embeddings from audio
    Optimized for high-pitched anime voices
    """
    def __init__(self, spec_channels, gin_channels=256):
        super().__init__()
        self.spec_channels = spec_channels
        self.gin_channels = gin_channels

        # Improved encoder structure with more capacity
        ref_enc_filters = [32, 64, 128, 256, 256, 512]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)

        # Apply LayerNorm for better training stability
        self.layernorm = nn.LayerNorm(self.spec_channels)

        # Calculate the output size after convolutions for the GRU input
        out_channels_freq = self.calculate_channels(spec_channels, 3, 2, 1, K)
        gru_input_size = ref_enc_filters[-1] * out_channels_freq

        # Bidirectional GRU for better feature extraction
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gin_channels // 2,
            bidirectional=True,
            batch_first=True,
        )

        # Project GRU output to the final global embedding size
        self.proj = nn.Linear(gin_channels, gin_channels)

        # Add attention mechanism for better focus on important features
        self.attention = nn.Sequential(
            nn.Linear(gin_channels, gin_channels // 2),
            nn.Tanh(),
            nn.Linear(gin_channels // 2, 1)
        )

    def forward(self, inputs, mask=None):
        # inputs shape: [N, Ty, n_mels]
        N = inputs.size(0)

        # Apply LayerNorm across frequency dimension
        inputs = self.layernorm(inputs)

        # View for Conv2D: [N, 1, Ty, n_mels]
        out = inputs.unsqueeze(1)

        # Apply convolutions
        for conv in self.convs:
            out = conv(out)
            out = F.relu(out)  # [N, filters[-1], Ty//2^K, n_mels//2^K]

        # Transpose and flatten for GRU
        out = out.transpose(1, 2)  # [N, Ty//2^K, filters[-1], n_mels//2^K]
        T = out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, filters[-1]*n_mels//2^K]

        # Apply GRU
        self.gru.flatten_parameters()
        out, _ = self.gru(out)  # out shape [N, T, gin_channels]

        # Apply attention
        attn_weights = self.attention(out).squeeze(-1)  # [N, T]
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # [N, 1, T]

        # Apply attention weights
        out = torch.bmm(attn_weights, out).squeeze(1)  # [N, gin_channels]

        # Project to final embedding size
        out = self.proj(out)  # [N, gin_channels]

        return out  # Return [N, gin_channels]

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        # Calculates the output size of a dimension after n_convs convolutions
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

class ResidualCouplingBlock(nn.Module):
    def __init__(self,
            channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            n_flows=4,
            gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            # Forward flow
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
            return x
        else:
            # Reverse flow
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
            return x

class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training and Inference
    Implements retrieval-based voice conversion
    """
    def __init__(
        self,
        spec_channels,
        inter_channels,
        hidden_channels,
        filter_channels,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=2,
        gin_channels=256,
        use_external_speaker_embedding=False,
        **kwargs
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_external_speaker_embedding = use_external_speaker_embedding

        # Content encoder to extract content information
        self.content_encoder = ContentEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,  # kernel_size
            1,  # dilation_rate
            16,  # n_layers
            gin_channels=0,  # No speaker conditioning for content encoder
        )

        # Decoder to generate audio from content and speaker information
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

        # Posterior encoder for training
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,  # kernel_size
            1,  # dilation_rate
            16,  # n_layers
            gin_channels=gin_channels,
        )

        # Flow for voice conversion
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )

        # Speaker encoder for extracting speaker embeddings
        self.speaker_encoder = SpeakerEncoder(spec_channels, gin_channels)

        # Speaker embedding layer for training
        if not use_external_speaker_embedding and n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
            # Initialize with normal distribution for better convergence
            nn.init.normal_(self.emb_g.weight, 0.0, 0.1)

    def forward(self, x, x_lengths, y, y_lengths, sid_src=None, sid_tgt=None, g_src=None, g_tgt=None):
        """
        Forward pass for training or inference.

        Args:
            x: Source waveform [B, C, T_wav]
            x_lengths: Source waveform lengths [B]
            y: Target mel spectrogram [B, n_mels, T_mel]
            y_lengths: Target mel spectrogram lengths [B]
            sid_src: Source speaker ID (integer tensor) - Used during training
            sid_tgt: Target speaker ID (integer tensor) - Used during training
            g_src: Source speaker embedding (float tensor) [B, gin_channels, 1] - Used during inference
            g_tgt: Target speaker embedding (float tensor) [B, gin_channels, 1] - Used during inference
        """
        from hanasuconverter.commons import sequence_mask

        # Determine speaker embeddings
        if self.use_external_speaker_embedding:
            # Inference mode: embeddings are provided directly
            if g_src is None or g_tgt is None:
                raise ValueError("g_src and g_tgt must be provided when use_external_speaker_embedding=True")
            # Ensure embeddings have the correct shape [B, C, 1]
            if g_src.dim() == 2:
                g_src = g_src.unsqueeze(-1)
            if g_tgt.dim() == 2:
                g_tgt = g_tgt.unsqueeze(-1)
        else:
            # Training mode: use internal embedding layer
            if sid_src is None or sid_tgt is None:
                raise ValueError("sid_src and sid_tgt must be provided when use_external_speaker_embedding=False")
            if self.n_speakers <= 0:
                raise ValueError("n_speakers must be > 0 for training with internal embeddings")
            if not hasattr(self, 'emb_g'):
                raise RuntimeError("Speaker embedding layer (emb_g) not initialized. Set n_speakers > 0.")

            g_src = self.emb_g(sid_src).unsqueeze(-1)  # [B, gin_channels, 1]
            g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)  # [B, gin_channels, 1]

        # Extract content information from target mel spectrogram
        # This is used as a reference for the posterior encoder
        content, content_mask = self.content_encoder(y, y_lengths)

        # Posterior encoder q(z|y, g_tgt)
        # Condition on target speaker embedding for better disentanglement
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_tgt)

        # Flow: Transform posterior z using source speaker embedding
        z_p = self.flow(z, y_mask, g=g_src, reverse=False)

        # Prior definition (standard Gaussian)
        m_p = torch.zeros_like(z_p)
        logs_p = torch.zeros_like(z_p)

        # Flow: Inverse transform using target speaker embedding
        o = self.flow(z_p, y_mask, g=g_tgt, reverse=True)

        # Decode using target speaker embedding
        y_hat = self.dec(o, g=g_tgt)

        # Return full sequences and parameters for loss calculation
        return y_hat, None, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def voice_conversion(self, spec, spec_lengths, g_src=None, g_tgt=None, tau=0.7):
        """
        Voice conversion inference.

        Args:
            spec: Source mel spectrogram [B, n_mels, T_mel]
            spec_lengths: Source mel spectrogram lengths [B]
            g_src: Source speaker embedding [B, gin_channels, 1]
            g_tgt: Target speaker embedding [B, gin_channels, 1]
            tau: Temperature parameter for conversion

        Returns:
            Converted waveform [B, 2, T_wav]
        """
        # Extract content information from source mel spectrogram
        content, content_mask = self.content_encoder(spec, spec_lengths)

        # Sample from posterior using source speaker embedding
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g_src, tau=tau)

        # Transform to target speaker using flow
        z_p = self.flow(z, spec_mask, g=g_src, reverse=False)
        z_hat = self.flow(z_p, spec_mask, g=g_tgt, reverse=True)

        # Decode to waveform using target speaker embedding
        o_hat = z_hat
        y_hat = self.dec(o_hat, g=g_tgt)

        return y_hat

    def extract_speaker_embedding(self, mel):
        """
        Extract speaker embedding from mel spectrogram

        Args:
            mel: Mel spectrogram [B, n_mels, T_mel]

        Returns:
            Speaker embedding [B, gin_channels]
        """
        # Transpose mel for speaker encoder: [B, T_mel, n_mels]
        mel_transposed = mel.transpose(1, 2)

        # Extract speaker embedding
        g = self.speaker_encoder(mel_transposed)

        return g