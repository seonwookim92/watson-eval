from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    YoutubeLoader
)
from langchain_core.documents import Document

class ParserFactory:
    @staticmethod
    def get_loader(input_path: str):
        if input_path.startswith("http"):
            if "youtube.com" in input_path or "youtu.be" in input_path:
                return YoutubeLoader.from_youtube_url(input_path, add_video_info=False, language=['en', 'ko'])


            else:
                return WebBaseLoader(input_path)
        
        ext = input_path.split(".")[-1].lower()
        if ext == "pdf":
            return PyPDFLoader(input_path)
        elif ext in ["txt", "md", "json"]:
            return TextLoader(input_path)
        else:
            # Default to text loader for unknown extensions if they are files
            return TextLoader(input_path)

    @classmethod
    def load_content(cls, input_path: str) -> List[Document]:
        loader = cls.get_loader(input_path)
        try:
            docs = loader.load()
            
            # If it's a single text document and looks like a URL, try to load the URL instead
            if len(docs) == 1 and not input_path.startswith("http"):
                content = docs[0].page_content.strip()
                if content.startswith("http") and "\n" not in content and " " not in content:
                    print(f"[*] Detected URL in file: {content}. Re-loading from URL...")
                    return cls.load_content(content)
            
            return docs
        except Exception as e:
            print(f"Error loading source {input_path}: {e}")
            return []

