import gradio as gr


def read_md_file_to_string(file_path: str = "./markdown.md"):
    """
    Convert the markdown file to string

    Parameters:
        - file_path (str): The first number.
    
    Returns:
        - str: file content
    """
    file_content = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except Exception as e:
        print(f"An error occurred: {e}")
    return file_content


app_theme = gr.Theme.from_hub("ParityError/Interstellar")

tonai_creative_html = read_md_file_to_string(
    "stuffs/html/tonai_creative_info.html")
with open("stuffs/tips.md") as txtfile:
    tips_content = txtfile.read()


custom_css = """
/* General file uploader styling */
.file-uploader {
    height: 125px;
    overflow: hidden; /* Hide overflow if needed */
}

.file-uploader input[type="file"] {
    height: 100%;
    width: 100%;
    opacity: 0; /* Hide the default file input */
}

.file-uploader .custom-file-uploader {
    height: 100%;
    line-height: 125px; /* Adjust line height for centering text */
    display: flex;
    align-items: center;
    justify-content: center;
}

#stop-button {
    background-color: #f44336; /* blue color */
}
"""
