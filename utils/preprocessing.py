def clean_text(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())  # remove extra spaces
    return text
