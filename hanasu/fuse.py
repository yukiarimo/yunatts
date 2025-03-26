import torch
import os
import argparse
from hanasu.api import TTS

def load_model(ckpt_path):
    """Load a model checkpoint and return its state dict"""
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found: {ckpt_path}")
        return None
    print(f"Loading checkpoint: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")

def fix_optimizer_zeros(optimizer_state, ref_optimizer_state):
    """Fix zero values in optimizer state by borrowing from reference state"""
    if optimizer_state is None or ref_optimizer_state is None:
        return optimizer_state

    fixed_count = 0

    # Fix zeros in optimizer state
    if 'state' in optimizer_state:
        for param_id, param_state in optimizer_state['state'].items():
            for key, value in param_state.items():
                if isinstance(value, torch.Tensor):
                    if torch.count_nonzero(value) == 0 and param_id in ref_optimizer_state['state']:
                        ref_value = ref_optimizer_state['state'][param_id].get(key)
                        if ref_value is not None and isinstance(ref_value, torch.Tensor):
                            if torch.count_nonzero(ref_value) > 0:
                                param_state[key] = ref_value
                                fixed_count += 1
                                print(f"Fixed zero tensor in optimizer state param {param_id}, key {key}")

    print(f"Fixed {fixed_count} zero tensors in optimizer state")
    return optimizer_state

def get_component_type(key):
    """Determine the component type based on parameter key"""
    if key.startswith('enc_p'):
        return 'encoder'
    elif key.startswith('dec'):
        return 'decoder'
    elif 'flows' in key and 'attn' in key:
        return 'flow_attention'
    elif 'flows' in key:
        return 'flow_other'
    elif key.startswith('dp') or key.startswith('sdp'):
        return 'duration'
    else:
        return 'other'

def combine_models_with_ratios(checkpoints, mix_ratios=None, fallback_to_non_zero=True):
    """
    Combine multiple model checkpoints with specified component-specific ratios.

    Args:
        checkpoints: List of model checkpoints
        mix_ratios: Dict mapping component types to lists of mixing ratios for each model
        fallback_to_non_zero: Whether to fallback to non-zero values when mixed result is zero

    Returns:
        A combined checkpoint
    """
    if not checkpoints or all(ckpt is None for ckpt in checkpoints):
        print("No valid checkpoints provided, cannot combine")
        return None

    # Filter out None checkpoints
    valid_checkpoints = [ckpt for ckpt in checkpoints if ckpt is not None]

    if len(valid_checkpoints) == 1:
        print("Only one valid checkpoint provided, returning it directly")
        return valid_checkpoints[0]

    # Default mixing ratios (equal distribution)
    default_ratio = [1.0/len(valid_checkpoints)] * len(valid_checkpoints)

    # Use provided mix_ratios or default to equal weights
    if mix_ratios is None:
        mix_ratios = {
            'encoder': default_ratio,
            'decoder': default_ratio,
            'flow_attention': default_ratio,
            'flow_other': default_ratio,
            'duration': default_ratio,
            'other': default_ratio
        }

    # Ensure all component types have properly sized ratio lists
    for component, ratios in mix_ratios.items():
        if len(ratios) != len(valid_checkpoints):
            print(f"Warning: {component} has {len(ratios)} ratios but {len(valid_checkpoints)} models. Using equal weights.")
            mix_ratios[component] = default_ratio

    combined = {}

    # Extract state dicts
    state_dicts = [ckpt["model"] for ckpt in valid_checkpoints]

    combined_state_dict = {}
    zero_params_fixed = 0
    total_params = 0

    # Get all unique keys across all state dicts
    all_keys = set()
    for sd in state_dicts:
        all_keys.update(sd.keys())

    for key in all_keys:
        total_params += 1

        # Check if key exists in all state dicts and shapes match
        key_in_all = all(key in sd for sd in state_dicts)

        if key_in_all:
            shapes_match = all(sd[key].shape == state_dicts[0][key].shape for sd in state_dicts)

            if shapes_match:
                # Determine component type and get its ratios
                component_type = get_component_type(key)
                ratios = mix_ratios.get(component_type, default_ratio)

                # Normalize ratios to sum to 1
                ratios = [r / sum(ratios) for r in ratios]

                # Calculate mixed tensor as weighted sum
                mixed_tensor = sum(r * sd[key] for r, sd in zip(ratios, state_dicts))

                # Handle zero values if needed
                if fallback_to_non_zero and torch.count_nonzero(mixed_tensor) == 0:
                    # Find the first non-zero tensor among the inputs
                    for i, sd in enumerate(state_dicts):
                        if torch.count_nonzero(sd[key]) > 0:
                            mixed_tensor = sd[key]
                            zero_params_fixed += 1
                            print(f"Warning: Mixed tensor for {key} was zero, using model{i+1}'s non-zero tensor")
                            break

                combined_state_dict[key] = mixed_tensor
            else:
                # Use the first model's weights when shapes don't match
                print(f"Shape mismatch for key {key}. Using first model's weights.")
                combined_state_dict[key] = state_dicts[0][key]
        else:
            # Use the first available model's weights
            for i, sd in enumerate(state_dicts):
                if key in sd:
                    print(f"Key {key} not found in all models, using model{i+1}'s weights")
                    combined_state_dict[key] = sd[key]
                    break

    print(f"Total parameters: {total_params}, Fixed zero parameters: {zero_params_fixed}")

    # Preserve other checkpoint data from the model with highest iteration
    combined["model"] = combined_state_dict
    combined["iteration"] = max(ckpt.get("iteration", 0) for ckpt in valid_checkpoints)
    combined["learning_rate"] = valid_checkpoints[0].get("learning_rate", 0.0)

    # Get optimizer state from the model with highest iteration count
    max_iter_idx = max(range(len(valid_checkpoints)),
                      key=lambda i: valid_checkpoints[i].get("iteration", 0))
    combined["optimizer"] = valid_checkpoints[max_iter_idx].get("optimizer", None)

    # Fix zeros in optimizer using other optimizers
    for i, ckpt in enumerate(valid_checkpoints):
        if i != max_iter_idx and "optimizer" in ckpt:
            combined["optimizer"] = fix_optimizer_zeros(combined["optimizer"],
                                                      ckpt.get("optimizer", None))

    return combined

def save_model(model_dict, output_path):
    """Save the mixed model checkpoint"""
    if model_dict is None:
        print(f"Cannot save to {output_path} - model is None")
        return

    print(f"Saving mixed model to: {output_path}")
    torch.save(model_dict, output_path)

def generate_audio(config_path, model_path, output_path, text, speaker_id=0,
                   device='cpu', sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8,
                   speed=1.0):
    """Generate audio using a model checkpoint"""
    print(f"Generating audio with model: {model_path}")
    model = TTS(
        language='EN',
        device=device,
        use_hf=False,
        config_path=config_path,
        ckpt_path=model_path
    )

    audio = model.tts_to_file(
        text=text,
        speaker_id=speaker_id,
        output_path=output_path,
        sdp_ratio=sdp_ratio,
        noise_scale=noise_scale,
        noise_scale_w=noise_scale_w,
        speed=speed,
        quiet=True
    )

    print(f"Audio generated and saved to: {output_path}")
    return audio

def main():
    parser = argparse.ArgumentParser(description="Fuse multiple Hanasu TTS models with component-specific ratios")

    # Model loading arguments
    parser.add_argument("--model_dirs", nargs="+", required=True, help="Directories containing model checkpoints")
    parser.add_argument("--model_steps", nargs="+", required=True, help="Step numbers for each model")
    parser.add_argument("--output_dir", required=True, help="Directory to save mixed checkpoints")
    parser.add_argument("--config_path", help="Path to the model config file (required for audio generation)")

    # Mixing arguments
    parser.add_argument("--encoder_ratios", nargs="+", type=float, help="Mixing ratios for encoder components")
    parser.add_argument("--decoder_ratios", nargs="+", type=float, help="Mixing ratios for decoder components")
    parser.add_argument("--flow_attention_ratios", nargs="+", type=float, help="Mixing ratios for flow attention components")
    parser.add_argument("--flow_other_ratios", nargs="+", type=float, help="Mixing ratios for other flow components")
    parser.add_argument("--duration_ratios", nargs="+", type=float, help="Mixing ratios for duration predictor components")
    parser.add_argument("--other_ratios", nargs="+", type=float, help="Mixing ratios for other components")

    # Audio generation arguments
    parser.add_argument("--generate_audio", action="store_true", help="Generate audio with mixed model")
    parser.add_argument("--text_file", help="File containing text to synthesize")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker ID for synthesis")
    parser.add_argument("--device", default="cpu", help="Device for audio generation (cpu/cuda)")
    parser.add_argument("--sdp_ratio", type=float, default=0.2, help="SDP ratio for audio synthesis")
    parser.add_argument("--noise_scale", type=float, default=0.6, help="Noise scale for audio synthesis")
    parser.add_argument("--noise_scale_w", type=float, default=0.8, help="Noise scale w for audio synthesis")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor for audio synthesis")

    # Other arguments
    parser.add_argument("--fallback_to_non_zero", action="store_true", help="Fallback to non-zero values if mixed result is zero")

    args = parser.parse_args()

    # Validate arguments
    if len(args.model_dirs) != len(args.model_steps):
        parser.error("Number of model directories must match number of model steps")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect model paths
    g_paths = []
    d_paths = []
    dur_paths = []

    for model_dir, model_step in zip(args.model_dirs, args.model_steps):
        g_paths.append(os.path.join(model_dir, f"G_{model_step}.pth"))
        d_paths.append(os.path.join(model_dir, f"D_{model_step}.pth"))
        dur_paths.append(os.path.join(model_dir, f"DUR_{model_step}.pth"))

    # Output path
    output_step = f"mix_{'_'.join(args.model_steps)}"
    g_output = os.path.join(args.output_dir, f"G_{output_step}.pth")
    d_output = os.path.join(args.output_dir, f"D_{output_step}.pth")
    dur_output = os.path.join(args.output_dir, f"DUR_{output_step}.pth")

    # Load model checkpoints
    g_checkpoints = [load_model(path) for path in g_paths]
    d_checkpoints = [load_model(path) for path in d_paths]
    dur_checkpoints = [load_model(path) for path in dur_paths]

    # Create mix ratios dictionary
    mix_ratios = {}

    if args.encoder_ratios:
        mix_ratios['encoder'] = args.encoder_ratios
    if args.decoder_ratios:
        mix_ratios['decoder'] = args.decoder_ratios
    if args.flow_attention_ratios:
        mix_ratios['flow_attention'] = args.flow_attention_ratios
    if args.flow_other_ratios:
        mix_ratios['flow_other'] = args.flow_other_ratios
    if args.duration_ratios:
        mix_ratios['duration'] = args.duration_ratios
    if args.other_ratios:
        mix_ratios['other'] = args.other_ratios

    # Use default equal ratios if none provided
    if not mix_ratios:
        mix_ratios = None
        print("No specific mixing ratios provided, using equal weights for all components")

    # Mix models
    mixed_g = combine_models_with_ratios(g_checkpoints, mix_ratios, args.fallback_to_non_zero)
    mixed_d = combine_models_with_ratios(d_checkpoints, mix_ratios, args.fallback_to_non_zero)
    mixed_dur = combine_models_with_ratios(dur_checkpoints, mix_ratios, args.fallback_to_non_zero)

    # Save mixed models
    save_model(mixed_g, g_output)
    save_model(mixed_d, d_output)
    save_model(mixed_dur, dur_output)

    print(f"Model fusion complete! Mixed models saved to {args.output_dir}")

    # Generate audio if requested
    if args.generate_audio:
        if not args.config_path:
            print("Config path is required for audio generation. Skipping audio generation.")
        elif not args.text and not args.text_file:
            print("No text provided for synthesis. Skipping audio generation.")
        else:
            # Get text from file or argument
            if args.text_file:
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            else:
                text = args.text

            # Generate audio
            audio_output = os.path.join(args.output_dir, f"audio_{output_step}.wav")
            generate_audio(
                config_path=args.config_path,
                model_path=g_output,
                output_path=audio_output,
                text=text,
                speaker_id=args.speaker_id,
                device=args.device,
                sdp_ratio=args.sdp_ratio,
                noise_scale=args.noise_scale,
                noise_scale_w=args.noise_scale_w,
                speed=args.speed
            )

if __name__ == "__main__":
    main()