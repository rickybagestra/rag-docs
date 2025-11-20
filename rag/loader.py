def load_file(file):
    import pypdf

    if file.type == "application/pdf":
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    elif "text" in file.type:
        return file.read().decode("utf-8")

    return ""
