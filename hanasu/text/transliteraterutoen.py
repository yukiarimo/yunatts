from hanasu.text.cyrillic import transliterate

def process_list_file(input_file, output_file):
    """
    Process a .list file and transliterate Russian text to English
    
    Args:
        input_file (str): Path to input .list file
        output_file (str): Path to output .list file
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    transliterated_lines = []
    
    for line in lines:
        if not line.strip():
            continue
            
        # Split the line by pipe delimiter
        parts = line.strip().split('|')
        
        if len(parts) >= 4:
            # Transliterate only the Russian text part (index 3)
            parts[3] = transliterate(parts[3], source='ru', target='en')
            
        # Rejoin the parts with pipe delimiter
        transliterated_lines.append('|'.join(parts))

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in transliterated_lines:
            f.write(line + '\n')

def main():
    input_file = 'input.list'
    output_file = 'output.list'
    
    try:
        process_list_file(input_file, output_file)
        print(f"Successfully transliterated text from {input_file}")
        print(f"Output written to {output_file}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == '__main__':
    main()