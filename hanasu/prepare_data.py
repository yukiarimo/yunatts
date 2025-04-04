import json
from collections import defaultdict
from random import shuffle
import math
from typing import Optional
from tqdm import tqdm
import click
from text.cleaner import clean_text
import os
import torch
from text.symbols import symbols

@click.command()
@click.option(
    "--metadata",
    default="data/example/metadata.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--output_dir", default="data", type=click.Path())
@click.option("--val_ratio", default=0.1, type=float)
@click.option(
    "--config_path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
def main(
    metadata: str,
    output_dir: str,
    val_ratio: float,
    config_path: str,
):
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.list')
    val_path = os.path.join(output_dir, 'val.list')
    out_config_path = os.path.join(os.path.dirname(metadata), 'config.json')
    
    cleaned_path = os.path.join(output_dir, "metadata.cleaned")

    # Process the data with only raw text and language ID
    device = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    out_file = open(cleaned_path, "w", encoding="utf-8")
    
    for line in tqdm(open(metadata, encoding="utf-8").readlines()):
        utt, spk, text = line.strip().split("|")
        
        # Get normalized text and Llama embeddings (ignore phones, tones, word2ph)
        norm_text, llama_emb = clean_text(text, device=device)
        
        # Join the list of characters back into a string if needed
        if isinstance(norm_text, list):
            norm_text = ''.join(norm_text)

        # Check Llama embedding shape for debugging
        print(f"Llama embedding shape for '{utt}': {llama_emb.shape}")
        
        # Write simplified metadata (no phones, tones, or word2ph)
        out_file.write(
            "{}|{}|{}\n".format(
                utt,
                spk,
                norm_text
            )
        )
        
        # Save Llama embeddings
        llama_path = utt.replace(".wav", ".llama.pt")
        os.makedirs(os.path.dirname(llama_path), exist_ok=True)
        torch.save(llama_emb.cpu(), llama_path)
        
        # Check file size
        file_size_bytes = os.path.getsize(llama_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"Saved embedding to {llama_path} (Size: {file_size_mb:.2f} MB)")
    
    out_file.close()
    
    # Split into train/val using val_ratio
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(cleaned_path, encoding="utf-8") as f:
        for line in f.readlines():
            # Parse simplified format (no phones, tones, word2ph)
            parts = line.strip().split("|")
            utt, spk = parts[0], parts[1]
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_count = max(1, math.ceil(len(utts) * val_ratio))  # At least 1 validation sample
        val_list += utts[:val_count]
        train_list += utts[val_count:]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    # Update config
    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map

    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["data"]["n_speakers"] = len(spk_id_map)
    config["symbols"] = symbols
    
    # Add summary output
    print(f"Data preparation complete.")
    print(f"Training files: {len(train_list)}")
    print(f"Validation files: {len(val_list)}")
    print(f"Number of speakers: {len(spk_id_map)}")
    print(f"Config saved to: {out_config_path}")

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()