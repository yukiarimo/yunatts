import click
import warnings
import os

@click.command
@click.argument('text')
@click.argument('output_path')
@click.option("--file", '-f', is_flag=True, show_default=True, default=False, help="Text is a file")
@click.option('--speaker', '-spk', default='EN-Default', help='Speaker ID, only for English, leave empty for default, ignored if not English. If English, defaults to "EN-Default"', type=click.Choice(['EN-Default', 'EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU']))
@click.option('--speed', '-s', default=1.0, help='Speed, defaults to 1.0', type=float)
@click.option('--device', '-d', default='auto', help='Device, defaults to auto')
def main(text, file, output_path, speaker, speed, device):
    if file:
        if not os.path.exists(text):
            raise FileNotFoundError(f'Trying to load text from file due to --file/-f flag, but file not found. Remove the --file/-f flag to pass a string.')
        else:
            with open(text) as f:
                text = f.read().strip()
    if text == '':
        raise ValueError('You entered empty text or the file you passed was empty.')
    if speaker == '': speaker = None
    from hanasu.api import TTS
    model = TTS(device=device)
    speaker_ids = model.hps.data.spk2id
    if True:
        if not speaker: speaker = 'EN-Default'
    else:
        spkr = speaker_ids[list(speaker_ids.keys())[0]]
    model.tts_to_file(text, speaker_ids, output_path, speed=speed)
