from uroman import Uroman
import re
import string
from hanasu.llama_utils import get_llama_feature
uroman = Uroman()
from hanasu.text.symbols import _symbol_to_id

def process_raw_text(text):
    """
    Process raw text for the model:
    1. Convert to lowercase
    2. Convert numbers to words (not implemented yet, placeholder)
    3. Keep only allowed characters (a-z, 0-9, !?.,')
    4. Replace other punctuation with commas
    """
    # Convert to lowercase
    text = text.lower()

    # Replace punctuation except for !?.,' with commas
    allowed_punctuation = "!?.,''"  # Added apostrophes and single quotes
    for char in string.punctuation:
        if char not in allowed_punctuation:
            text = text.replace(char, ",")

    # Keep only allowed characters
    allowed_chars = set(string.ascii_lowercase + string.digits + allowed_punctuation + " ")
    text = ''.join(c for c in text if c in allowed_chars)

    # Remove extra spaces and commas
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r',+', ',', text)

    return text

def transliterate_to_english(text):
    """
    Transliterate text to English characters using uroman

    Args:
        text (str): Input text in any language
        language (str, optional): Language code (not used by uroman, kept for backward compatibility)

    Returns:
        str: Transliterated text in Latin characters
    """
    # Use uroman for transliteration (language is automatically detected)
    language=None
    #return uroman.romanize_string(text, language=language)
    return text

def clean_text(text, device=None):
    """
    Clean and process text for the model:
    1. Transliterate to English using uroman
    2. Process raw text
    3. Return character sequence and Llama embeddings

    Args:
        text (str): Input text in any language
        language (str, optional): Language code (kept for backward compatibility)
        device (str, optional): Device for Llama model

    Returns:
        tuple: (processed_text, phones, tones, word2ph, llama_emb)
    """
    # Transliterate to English using uroman
    text = transliterate_to_english(text)

    # Process raw text
    processed_text = process_raw_text(text)

    # Convert to character sequence
    phones = list(processed_text)

    # Transform phones to IDs
    phones = [phone if phone in _symbol_to_id else _symbol_to_id.get(phone, _symbol_to_id[' ']) for phone in phones]

    # Get Llama embeddings
    llama_emb = get_llama_feature(processed_text, device=device)

    return phones, llama_emb

def text_to_sequence(text):
    """
    Convert text to sequence of character IDs

    Args:
        text (str): Input text in any language
        language (str, optional): Language code (kept for backward compatibility)

    Returns:
        list: Sequence of character IDs
    """
    from . import cleaned_text_to_sequence
    norm_text, phones = clean_text(text)[:4]
    return cleaned_text_to_sequence(phones, symbol_to_id=_symbol_to_id)