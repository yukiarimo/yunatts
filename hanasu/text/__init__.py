from .symbols import *

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def cleaned_text_to_sequence(cleaned_text, symbol_to_id=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      cleaned_text: processed text characters
      tones: not used, kept for compatibility 
      language: language code (kept for compatibility)
      symbol_to_id: custom symbol to ID mapping
    Returns:
      tuple: (phones, dummy_tones, lang_ids)
    """
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]

    return phones