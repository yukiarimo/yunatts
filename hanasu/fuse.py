import torch
import os
import argparse
import numpy as np

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

def combine_models(checkpoint1, checkpoint2, ratio=0.5, fallback_to_non_zero=True):
    """Combine two model checkpoints with specified ratio"""
    if checkpoint1 is None or checkpoint2 is None:
        if checkpoint1 is None and checkpoint2 is not None:
            print("Using only model2 as model1 is missing")
            return checkpoint2
        elif checkpoint2 is None and checkpoint1 is not None:
            print("Using only model1 as model2 is missing")
            return checkpoint1
        else:
            print("Both models are missing, cannot combine")
            return None
        
    combined = {}
    
    # Mix model weights
    state_dict1 = checkpoint1["model"]
    state_dict2 = checkpoint2["model"]
    
    combined_state_dict = {}
    zero_params_fixed = 0
    total_params = 0
    
    for key in state_dict1:
        total_params += 1
        if key in state_dict2 and state_dict1[key].shape == state_dict2[key].shape:
            # Calculate mixed tensor
            mixed_tensor = ratio * state_dict1[key] + (1 - ratio) * state_dict2[key]
            
            # Check if result has zeros when it shouldn't
            if fallback_to_non_zero and torch.count_nonzero(mixed_tensor) == 0:
                # If mixed result is zero but one of the sources isn't, use the non-zero one
                if torch.count_nonzero(state_dict1[key]) > 0:
                    mixed_tensor = state_dict1[key]
                    zero_params_fixed += 1
                    print(f"Warning: Mixed tensor for {key} was zero, using model1's non-zero tensor")
                elif torch.count_nonzero(state_dict2[key]) > 0:
                    mixed_tensor = state_dict2[key]
                    zero_params_fixed += 1
                    print(f"Warning: Mixed tensor for {key} was zero, using model2's non-zero tensor")
            
            combined_state_dict[key] = mixed_tensor
        else:
            # Handle the case where only one model has this parameter or shapes don't match
            if key not in state_dict2:
                print(f"Key {key} not found in model2, using model1's weights")
                combined_state_dict[key] = state_dict1[key]
            else:
                print(f"Shape mismatch for key {key}: {state_dict1[key].shape} vs {state_dict2[key].shape}. Using model1's weights.")
                combined_state_dict[key] = state_dict1[key]
    
    # Add any keys that are in model2 but not in model1
    for key in state_dict2:
        if key not in state_dict1:
            print(f"Key {key} not found in model1, using model2's weights")
            combined_state_dict[key] = state_dict2[key]
            total_params += 1
    
    print(f"Total parameters: {total_params}, Fixed zero parameters: {zero_params_fixed}")
    
    # Preserve other checkpoint data
    combined["model"] = combined_state_dict
    combined["iteration"] = max(checkpoint1.get("iteration", 0), checkpoint2.get("iteration", 0))
    combined["learning_rate"] = checkpoint1.get("learning_rate", 0.0)
    
    # Get optimizer state from the model with higher iteration count and fix zeros
    if checkpoint1.get("iteration", 0) > checkpoint2.get("iteration", 0):
        combined["optimizer"] = checkpoint1.get("optimizer", None)
        # Use model2's optimizer to fix zero values in model1's optimizer
        combined["optimizer"] = fix_optimizer_zeros(combined["optimizer"], checkpoint2.get("optimizer", None))
    else:
        combined["optimizer"] = checkpoint2.get("optimizer", None)
        # Use model1's optimizer to fix zero values in model2's optimizer
        combined["optimizer"] = fix_optimizer_zeros(combined["optimizer"], checkpoint1.get("optimizer", None))
    
    return combined

def save_model(model_dict, output_path):
    """Save the mixed model checkpoint"""
    if model_dict is None:
        print(f"Cannot save to {output_path} - model is None")
        return
        
    print(f"Saving mixed model to: {output_path}")
    torch.save(model_dict, output_path)

def main():
    parser = argparse.ArgumentParser(description="Mix two Hanasu TTS models with a specified ratio")
    parser.add_argument("--model1_dir", required=True, help="Directory containing first model checkpoints")
    parser.add_argument("--model2_dir", required=True, help="Directory containing second model checkpoints")
    parser.add_argument("--output_dir", required=True, help="Directory to save mixed checkpoints")
    parser.add_argument("--model1_step", required=True, help="Step number for first model")
    parser.add_argument("--model2_step", required=True, help="Step number for second model")
    parser.add_argument("--ratio", type=float, default=0.5, help="Mixing ratio (0.0 to 1.0) for model1")
    parser.add_argument("--fallback_to_model2", action="store_true", help="Fallback to model2 if zeros are present")
    parser.add_argument("--prefer_model2_optimizer", action="store_true", help="Always use model2's optimizer")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths for model1
    g1_path = os.path.join(args.model1_dir, f"G_{args.model1_step}.pth")
    d1_path = os.path.join(args.model1_dir, f"D_{args.model1_step}.pth")
    dur1_path = os.path.join(args.model1_dir, f"DUR_{args.model1_step}.pth")
    
    # Paths for model2
    g2_path = os.path.join(args.model2_dir, f"G_{args.model2_step}.pth")
    d2_path = os.path.join(args.model2_dir, f"D_{args.model2_step}.pth")
    dur2_path = os.path.join(args.model2_dir, f"DUR_{args.model2_step}.pth")
    
    # Output paths
    output_step = f"mix_{args.model1_step}_{args.model2_step}"
    g_output = os.path.join(args.output_dir, f"G_{output_step}.pth")
    d_output = os.path.join(args.output_dir, f"D_{output_step}.pth")
    dur_output = os.path.join(args.output_dir, f"DUR_{output_step}.pth")
    
    # Load models
    g1_checkpoint = load_model(g1_path)
    g2_checkpoint = load_model(g2_path)
    d1_checkpoint = load_model(d1_path)
    d2_checkpoint = load_model(d2_path)
    dur1_checkpoint = load_model(dur1_path)
    dur2_checkpoint = load_model(dur2_path)
    
    # Force using model2's optimizer if requested
    if args.prefer_model2_optimizer and g1_checkpoint is not None and g2_checkpoint is not None:
        print("Forcing use of model2's optimizer state")
        g1_checkpoint["iteration"] = 0  # This ensures model2's optimizer will be chosen
        d1_checkpoint["iteration"] = 0
        if dur1_checkpoint is not None:
            dur1_checkpoint["iteration"] = 0
    
    # If we should fallback to model2 for zeros, use a special ratio
    if args.fallback_to_model2:
        print("Using model2 (108000) as fallback for zeros")
        # Mix models
        mixed_g = combine_models(g1_checkpoint, g2_checkpoint, args.ratio, fallback_to_non_zero=True)
        mixed_d = combine_models(d1_checkpoint, d2_checkpoint, args.ratio, fallback_to_non_zero=True)
        mixed_dur = combine_models(dur1_checkpoint, dur2_checkpoint, args.ratio, fallback_to_non_zero=True)
    else:
        # Normal mixing
        mixed_g = combine_models(g1_checkpoint, g2_checkpoint, args.ratio)
        mixed_d = combine_models(d1_checkpoint, d2_checkpoint, args.ratio)
        mixed_dur = combine_models(dur1_checkpoint, dur2_checkpoint, args.ratio)
    
    # Save mixed models
    save_model(mixed_g, g_output)
    save_model(mixed_d, d_output)
    save_model(mixed_dur, dur_output)
    
    print(f"Model mixing complete! Mixed models saved to {args.output_dir}")

if __name__ == "__main__":
    main()