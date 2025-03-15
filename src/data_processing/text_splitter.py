from typing import List, Dict
import re

def split_text(
    text: str, 
    chunk_size: int = 1000, 
    overlap: int = 200, 
    title: str = "Leave Policy"
) -> List[Dict[str, str]]:
    """
    Splits the input text into paragraph-based chunks of specified size, ensuring meaningful boundaries 
    and optional overlap, and attaches metadata like title and headers.

    Args:
        text (str): The input text to split.
        chunk_size (int): The target size for each chunk.
        overlap (int): The number of overlapping words between chunks.
        title (str): The title of the document.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing text chunks and metadata.
    """
    # Split text into paragraphs
    paragraphs = text.split("\n\n")  # Assumes paragraphs are separated by double newlines
    
    chunks = []
    current_chunk = []

    for paragraph in paragraphs:
        # Add paragraph while checking the length
        if len(' '.join(current_chunk + [paragraph])) <= chunk_size:
            current_chunk.append(paragraph)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [paragraph]

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
                final_chunks.append({
                    "title": title,
                    "header": f"Section {i + 1}",
                    "content": overlap_part + ' ' + chunks[i]
                })
            else:
                final_chunks.append({
                    "title": title,
                    "header": f"Section {i + 1}",
                    "content": chunks[i]
                })
        return final_chunks
    
    # If no overlap, add metadata directly
    return [
        {"title": title, "header": f"Section {i + 1}", "content": chunk}
        for i, chunk in enumerate(chunks)
    ]
