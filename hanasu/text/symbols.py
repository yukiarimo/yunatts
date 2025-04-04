"""
Defines the set of symbols used in text input to the model.
"""

# Basic punctuation and special symbols
punctuation = ["!", "?", ",", ".", " ", "'"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# Raw characters for all languages (lowercase letters, numbers, and basic punctuation)
raw_symbols = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

# combine all symbols
symbols = [pad] + raw_symbols + pu_symbols

# Create a mapping from symbols to IDs
_symbol_to_id = {s: i for i, s in enumerate(symbols)}