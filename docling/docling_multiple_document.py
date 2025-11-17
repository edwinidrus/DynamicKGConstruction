#ingest the data from pdf document into ontolgy with docling


## import the depencies
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
import os

## pdf explorer
def explore_pdfs_in_folder(folder_path):
    paths = []
    target_path_folder = "C:/Users/HP/Downloads"

    for m,n,v in os.walk(target_path_folder):
        if m == target_path_folder:
            for file in v:
                if file.endswith('.pdf'):
                    file_path = os.path.join(m, file)
                    paths.append(file_path)

    return paths

def process_pdfs_to_text(folder_path, output_dir="build"):
    """
    Process all PDF files in a folder and convert them to text using Docling.
    
    Args:
        folder_path (str): Path to the folder containing PDF files
        output_dir (str): Directory to save the output text files (default: "build")
    
    Returns:
        list: List of paths to the generated text files
    """
    # Get all PDF paths from the folder
    pdf_paths = explore_pdfs_in_folder(folder_path)
    
    if not pdf_paths:
        print(f"No PDF files found in {folder_path}")
        return []
    
    OUT_DIR = Path(output_dir)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Initialize converter once
    converter = DocumentConverter()
    
    output_files = []
    
    # Process each PDF file
    for pdf_path in pdf_paths:
        try:
            print(f"Processing: {pdf_path}")
            
            # 1) Parse with Docling
            doc = converter.convert(pdf_path).document
            
            # 2) Export to Markdown to retain structure (headings, lists, tables)
            md_text = MarkdownDocSerializer(doc=doc).serialize().text
            
            # 3) Save as plain text with unique filename based on original PDF
            pdf_filename = Path(pdf_path).stem  # Get filename without extension
            TXT_PATH = OUT_DIR / f"{pdf_filename}_docling.txt"
            TXT_PATH.write_text(md_text, encoding="utf-8")
            
            output_files.append(str(TXT_PATH))
            print(f"✓ Saved parsed text to: {TXT_PATH}")
        except Exception as e:
            print(f"✗ Error processing {pdf_path}: {e}")
    
    print(f"\nCompleted processing {len(pdf_paths)} PDF files.")
    return output_files




