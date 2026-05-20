import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning to remove multiple consecutive spaces, 
    normalize newlines, and strip leading/trailing whitespace.
    """
    if not text:
        return ""
    # Normalize whitespaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize consecutive newlines to at most double newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
