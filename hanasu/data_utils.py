import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from loguru import logger
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_filepaths_and_text
from utils import load_wav_to_torch_librosa as load_wav_to_torch
from text import cleaned_text_to_sequence
from hanasu.llama_utils import get_llama_feature
from text.cleaner import clean_text

"""Multi speaker version with stereo audio support"""

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams

        # Add stereo support
        self.channels = getattr(hparams, "audio_channels", 1)  # Default to mono if not specified

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 10000)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (channels * 2)

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        logger.info("Init dataset...")
        for item in tqdm(
            self.audiopaths_sid_text
        ):
            try:
                _id, spk, text = item
            except:
                print(item)
                raise
            audiopath = f"{_id}"
            if True:
                audiopaths_sid_text_new.append(
                    [audiopath, spk, text]
                )
                # Adjust for stereo audio (2 channels)
                lengths.append(os.path.getsize(audiopath) // (2 * self.channels * self.hop_length))
            else:
                skipped += 1
        logger.info(f'min: {min(lengths)}; max: {max(lengths)}' )
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text
        # automatically detect device: cuda, mps, cpu
        device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        phones, llama_emb = clean_text(text, device=device)

        # Convert text characters to phone IDs
        phone_ids = cleaned_text_to_sequence(phones)

        # Convert sequences to tensors
        phones = torch.LongTensor(phone_ids)

        spec, wav = self.get_audio(audiopath)
        sid = int(self.spk_map[sid])
        sid = torch.LongTensor([sid])

        return (phones, spec, wav, sid, llama_emb)

    def get_audio(self, filename):
        audio_norm, sampling_rate = load_wav_to_torch(filename, self.sampling_rate)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.sampling_rate
                )
            )

        # Handle stereo audio
        if audio_norm.dim() == 2 and self.channels == 2:
            # Already stereo, shape: [channels, time]
            audio_norm = audio_norm
        elif audio_norm.dim() == 1 and self.channels == 2:
            # Convert mono to stereo by duplicating
            audio_norm = audio_norm.unsqueeze(0).repeat(2, 1)
        elif audio_norm.dim() == 2 and self.channels == 1:
            # Convert stereo to mono by averaging
            audio_norm = audio_norm.mean(dim=0, keepdim=True)
        else:
            # Mono audio, add channel dimension
            audio_norm = audio_norm.unsqueeze(0)

        spec_filename = filename.replace(".wav", f".{self.channels}ch.spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
        except:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            # For stereo, spec shape will be [channels, freq, time]
            # For mono, spec shape will be [freq, time]
            if self.channels == 1:
                spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text, phone, wav_path):
        phone = cleaned_text_to_sequence(phone)

        # Use Llama embeddings instead of BERT
        llama_path = wav_path.replace(".wav", ".llama.pt")
        try:
            llama_emb = torch.load(llama_path)
            assert llama_emb.shape[-1] == len(phone)
        except Exception as e:
            print(f"Generating new Llama embedding for {wav_path}: {e}")
            llama_emb = get_llama_feature(text)

            # Handle potential size mismatch between text and embeddings
            if llama_emb.shape[-1] != len(phone):
                # Resize embedding to match phone length
                orig_len = llama_emb.shape[-1]
                target_len = len(phone)
                if orig_len > target_len:
                    # Truncate
                    llama_emb = llama_emb[:, :target_len]
                else:
                    # Pad with zeros
                    padding = torch.zeros(llama_emb.shape[0], target_len - orig_len)
                    llama_emb = torch.cat([llama_emb, padding], dim=1)

            torch.save(llama_emb, llama_path)

        assert llama_emb.shape[-1] == len(phone), f"Llama embedding shape {llama_emb.shape} doesn't match phone length {len(phone)}"

        phone = torch.LongTensor(phone)
        return llama_emb, phone

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)

    def shuffle_mapping(self):
        """
        Shuffles the dataset for each epoch to ensure different batching
        """
        random.shuffle(self.audiopaths_sid_text)

class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(-1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(-1) for x in batch])
        max_wav_len = max([x[2].size(-1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)

        # Check if we have stereo or mono spectrograms
        is_stereo_spec = batch[0][1].dim() == 3
        is_stereo_wav = batch[0][2].size(0) > 1

        if is_stereo_spec:
            channels_spec = batch[0][1].size(0)
            spec_padded = torch.FloatTensor(len(batch), channels_spec, batch[0][1].size(1), max_spec_len)
        else:
            spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)

        if is_stereo_wav:
            channels_wav = batch[0][2].size(0)
            wav_padded = torch.FloatTensor(len(batch), channels_wav, max_wav_len)
        else:
            wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        # Get Llama embedding size
        llama_dim = batch[0][4].size(0)
        llama_padded = torch.FloatTensor(len(batch), llama_dim, max_text_len)

        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        llama_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            if is_stereo_spec:
                spec_padded[i, :, :, : spec.size(2)] = spec
                spec_lengths[i] = spec.size(2)
            else:
                spec_padded[i, :, : spec.size(1)] = spec
                spec_lengths[i] = spec.size(1)

            wav = row[2]
            if is_stereo_wav:
                wav_padded[i, :, : wav.size(1)] = wav
                wav_lengths[i] = wav.size(1)
            else:
                wav_padded[i, :, : wav.size(1)] = wav
                wav_lengths[i] = wav.size(1)

            sid[i] = row[3]
            # Get Llama embedding (fix: use correct index 4 instead of 6)
            llama = row[4]
            llama_padded[i, :, : llama.size(1)] = llama

        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
            llama_padded,
        )

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
