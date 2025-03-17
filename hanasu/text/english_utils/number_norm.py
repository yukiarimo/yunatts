import re
import inflect
_inflect = inflect.engine()

def normalize_numbers(text):
    # Remove commas in numbers
    text = re.sub(r"([0-9][0-9\,]+[0-9])", lambda m: m.group(1).replace(",", ""), text)
    
    # Convert currency expressions
    currencies = {
        "$": {0.01: "cent", 0.02: "cents", 1: "dollar", 2: "dollars"},
        "€": {0.01: "cent", 0.02: "cents", 1: "euro", 2: "euros"},
        "£": {0.01: "penny", 0.02: "pence", 1: "pound sterling", 2: "pounds sterling"},
        "¥": {0.02: "sen", 2: "yen"},
    }
    
    def expand_currency(m):
        unit, value = m.group(1), m.group(2)
        currency = currencies[unit]
        parts = value.replace(",", "").split(".")
        if len(parts) > 2: return f"{value} {currency[2]}"
        
        text = []
        integer = int(parts[0]) if parts[0] else 0
        if integer > 0:
            text.append(f"{integer} {currency.get(integer, currency[2])}")
        
        fraction = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if fraction > 0:
            text.append(f"{fraction} {currency.get(fraction / 100, currency[0.02])}")
        
        return " ".join(text) if text else f"zero {currency[2]}"
    
    text = re.sub(r"(£|\$|¥)([0-9\,\.]*[0-9]+)", expand_currency, text)
    
    # Convert decimal points
    text = re.sub(r"([0-9]+\.[0-9]+)", lambda m: m.group(1).replace(".", " point "), text)
    
    # Convert ordinals
    text = re.sub(r"[0-9]+(st|nd|rd|th)", lambda m: _inflect.number_to_words(m.group(0)), text)
    
    # Convert numbers
    def expand_number(m):
        num = int(m.group(0))
        if 1000 < num < 3000:
            if num == 2000: return "two thousand"
            if 2000 < num < 2010: return f"two thousand {_inflect.number_to_words(num % 100)}"
            if num % 100 == 0: return f"{_inflect.number_to_words(num // 100)} hundred"
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
        return _inflect.number_to_words(num, andword="")
    
    return re.sub(r"-?[0-9]+", expand_number, text)