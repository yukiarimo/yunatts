import re
import torch
import soundfile
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch
from . import utils
from .models import SynthesizerTrn
from .split_utils import split_sentence

def load_or_download_config(locale, use_hf=True, config_path=None):
    if config_path is None:
        raise ValueError("config_path is required")
    return utils.get_hparams_from_file(config_path)

def load_or_download_model(locale, device, use_hf=True, ckpt_path=None):
    if ckpt_path is None:
        raise ValueError("ckpt_path is required")
    return torch.load(ckpt_path, map_location=device)

class TTS(nn.Module):
    def __init__(self,
                device='auto',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        hps = load_or_download_config(use_hf=use_hf, config_path=config_path)
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device

        # load state_dict
        checkpoint_dict = load_or_download_model(device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, quiet=False):
        texts = split_sentence(text)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        texts = self.split_sentences_into_pieces(text, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            device = self.device
            phones, llama_emb = utils.get_text_for_tts_infer(t, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                llama_emb = llama_emb.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        llama_emb,  # previously ja_bert, now llama_emb
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                del x_tst, llama_emb, x_tst_lengths, speakers
                
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)

    def tts_to_stream(self, text, speaker_id, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8,
                    speed=1.0, quiet=False):
        """
        Streaming version of tts_to_file that yields audio chunks as they're generated.

        Args:
            text (str): The text to convert to speech
            speaker_id (int): ID of the speaker voice to use
            sdp_ratio (float, optional): Stochastic duration predictor ratio. Defaults to 0.2.
            noise_scale (float, optional): Noise scale for phoneme durations. Defaults to 0.6.
            noise_scale_w (float, optional): Noise scale for phoneme variation. Defaults to 0.8.
            speed (float, optional): Speed of speech. Defaults to 1.0.
            quiet (bool, optional): Whether to suppress progress output. Defaults to False.

        Yields:
            tuple: (audio_chunk, sample_rate) where audio_chunk is a numpy array of float32 values
        """
        texts = self.split_sentences_into_pieces(text, quiet)
        device = self.device
        sr = self.hps.data.sampling_rate
        
        if not quiet:
            print("> texts: ", texts)

        for t in texts:
            if not quiet:
                print(f" > Processing: {t}")

            phones, llama_emb = utils.get_text_for_tts_infer(
                t, self.hps, device, self.symbol_to_id
            )

            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                llama_emb = llama_emb.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                speakers = torch.LongTensor([speaker_id]).to(device)

                audio = self.model.infer(
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    llama_emb,  # previously ja_bert, now llama_emb
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=1. / speed,
                )[0][0, 0].data.cpu().float().numpy()

                # Clean up to reduce memory usage
                del x_tst, llama_emb, x_tst_lengths, speakers

            # Add a small silence at the end of each chunk
            silence = np.zeros(int((sr * 0.05) / speed), dtype=np.float32)
            audio_with_silence = np.concatenate([audio, silence])

            yield audio_with_silence, sr

        # Clean up after processing all chunks
        torch.cuda.empty_cache()

    def tts_to_stream_parallel(self, text, speaker_id, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8,
                            speed=1.0, quiet=False, max_workers=2):
        """
        Parallel streaming version of tts_to_file that generates audio chunks in the background
        while current chunks are being processed/played.

        Args:
            text (str): The text to convert to speech
            speaker_id (int): ID of the speaker voice to use
            sdp_ratio (float, optional): Stochastic duration predictor ratio. Defaults to 0.2.
            noise_scale (float, optional): Noise scale for phoneme durations. Defaults to 0.6.
            noise_scale_w (float, optional): Noise scale for phoneme variation. Defaults to 0.8.
            speed (float, optional): Speed of speech. Defaults to 1.0.
            quiet (bool, optional): Whether to suppress progress output. Defaults to False.
            max_workers (int, optional): Maximum number of parallel workers. Defaults to 2.

        Yields:
            tuple: (audio_chunk, sample_rate) where audio_chunk is a numpy array of float32 values
        """
        import threading
        import queue
        from concurrent.futures import ThreadPoolExecutor

        texts = self.split_sentences_into_pieces(text, quiet)
        device = self.device
        sr = self.hps.data.sampling_rate

        if not quiet:
            print(f" > Processing {len(texts)} text segments")

        # Queue to store generated audio chunks
        audio_queue = queue.Queue(maxsize=max_workers)

        # Flag to signal when all processing is complete
        done_event = threading.Event()

        def generate_audio_chunk(t):
            """Generate an audio chunk for a single text segment"""
            try:
                if not quiet:
                    print(f" > Processing in background: {t}")

                phones, llama_emb = utils.get_text_for_tts_infer(
                    t, self.hps, device, self.symbol_to_id
                )

                with torch.no_grad():
                    x_tst = phones.to(device).unsqueeze(0)
                    llama_emb = llama_emb.to(device).unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                    speakers = torch.LongTensor([speaker_id]).to(device)

                    audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        llama_emb,  # previously ja_bert, now llama_emb
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()

                    # Clean up to reduce memory usage
                    del x_tst, llama_emb, x_tst_lengths, speakers

                # Add a small silence at the end of each chunk
                silence = np.zeros(int((sr * 0.05) / speed), dtype=np.float32)
                audio_with_silence = np.concatenate([audio, silence])

                # Put the result in the queue
                audio_queue.put((audio_with_silence, sr))

            except Exception as e:
                if not quiet:
                    print(f"Error processing text segment: {e}")
                # Put None in the queue to indicate an error
                audio_queue.put(None)

        # Worker function that processes all text segments
        def process_all_segments():
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all text segments for processing
                    futures = [executor.submit(generate_audio_chunk, t) for t in texts]

                    # Wait for all processing to complete
                    for future in futures:
                        future.result()  # This will re-raise any exceptions

            finally:
                # Signal that all processing is complete
                done_event.set()

                # Make sure the queue is unblocked by adding None
                try:
                    audio_queue.put(None, block=False)
                except queue.Full:
                    pass

        # Start the background worker thread
        worker_thread = threading.Thread(target=process_all_segments)
        worker_thread.daemon = True
        worker_thread.start()

        # Yield audio chunks as they become available
        while not (done_event.is_set() and audio_queue.empty()):
            try:
                result = audio_queue.get(timeout=0.1)
                if result is not None:
                    yield result
            except queue.Empty:
                # No audio chunk available yet, just continue waiting
                pass

        # Clean up after all chunks are processed
        torch.cuda.empty_cache()