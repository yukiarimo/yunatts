import re
import re

def split_sentence(text, min_len=10):
    sentences = split_sentences_latin(text, min_len=min_len)
    return sentences

def split_sentences_latin(text, min_len=10):
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    text = re.sub('[“”]', '"', text)
    text = re.sub('[‘’]', "'", text)
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»]+", "", text)
    return [item.strip() for item in txtsplit(text, 256, 512) if item.strip()]

def merge_short_sentences_en(sens):
    """Avoid short sentences by merging them with the following sentence.

    Args:
        List[str]: list of input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1].split(" ")) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1].split(" ")) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out

def merge_short_sentences_zh(sens):
    # return sens
    """Avoid short sentences by merging them with the following sentence.

    Args:
        List[str]: list of input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1]) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1]) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out

def txtsplit(text, desired_length=100, max_length=200):
    """
    Split text into chunks based on simple punctuation rules.
    Only splits on periods, exclamation marks, question marks, and ellipses.

    Args:
        text (str): The text to split
        min_length (int): Minimum length for a sentence to be considered standalone

    Returns:
        list: List of text chunks
    """
    min_length = 10
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Handle ellipses first (to avoid confusion with periods)
    text = re.sub(r'\.\.\.', ' ELLIPSIS_MARKER ', text)

    # Add spaces after punctuation for easier splitting
    text = re.sub(r'([.!?])', r'\1 SENTENCE_BREAK ', text)

    # Split by the markers we inserted
    raw_sentences = [s.strip() for s in text.split('SENTENCE_BREAK')]

    # Restore ellipses
    raw_sentences = [s.replace('ELLIPSIS_MARKER', '...') for s in raw_sentences]

    # Replace " - " with a comma
    raw_sentences = [s.replace(' - ', ', ') for s in raw_sentences]

    # Replace "em dash" with a comma
    raw_sentences = [s.replace('—', ', ') for s in raw_sentences]

    # Filter out empty sentences and merge short ones
    sentences = []
    current = ""

    for sentence in raw_sentences:
        if not sentence:  # Skip empty sentences
            continue

        if len(current) == 0:
            current = sentence
        elif len(sentence) < min_length:
            # If this sentence is too short, append it to the current one
            current += " " + sentence
        else:
            # If we have accumulated content and the new sentence is long enough
            if current:
                sentences.append(current)
            current = sentence

    # Don't forget to add the last sentence
    if current:
        sentences.append(current)

    return [s for s in sentences if s]  # Final filter for any empty strings