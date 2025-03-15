from typing import List
import re

def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits the input text into chunks of specified size, ensuring meaningful boundaries and optional overlap.
    
    Args:
        text (str): The input text to split.
        chunk_size (int): The target size for each chunk.
        overlap (int): The number of overlapping words between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    # Preprocess text and split into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    current_chunk = []

    for sentence in sentences:
        # Add sentence while checking the length
        if len(' '.join(current_chunk + [sentence])) <= chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]

    # Append the remaining chunk if not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Add overlapping sections if required
    if overlap > 0:
        final_chunks = []
        for i in range(len(chunks)):
            if i > 0:
                # Add overlap
                overlap_part = ' '.join(chunks[i - 1].split()[-overlap:])
                final_chunks.append(overlap_part + ' ' + chunks[i])
            else:
                final_chunks.append(chunks[i])
        return final_chunks
    
    return chunks
