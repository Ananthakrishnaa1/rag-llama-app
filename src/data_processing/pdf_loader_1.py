from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdf(file_path):
    """
    Load and extract text from a PDF file
    Args:
        file_path (str): Path to the PDF file
    Returns:
        str: Extracted text from PDF
    Raises:
        ValueError: If file is not accessible or not a valid PDF
    """
    try:
        logger.info(f"Loading PDF from: {file_path}")
        text = extract_text(file_path)
        
        if not text.strip():
            logger.error("Empty PDF content")
            raise ValueError("No text content found in PDF")
            
        logger.info(f"Successfully extracted {len(text)} characters")
        return text
        
    except PDFSyntaxError as e:
        logger.error(f"PDF syntax error: {str(e)}")
        raise ValueError("Invalid or corrupted PDF file")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise ValueError(f"PDF file not found: {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise ValueError(f"Failed to process PDF: {str(e)}")