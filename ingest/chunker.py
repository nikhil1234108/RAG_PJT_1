import os
import sys
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env.txt"))

ARTICLES_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data","articles")

def doc_loader(articles_dir:str = ARTICLES_DIRECTORY) -> List[Document]:
    loader = DirectoryLoader(
        articles_dir,
        glob = "*.txt",
        loader_cls = TextLoader,
        loader_kwargs = {"encoding": "utf-8", "autodetect_encoding": True}
    )

    docs = loader.load()
    total_characters = sum(len(doc.page_content) for doc in docs)

    for doc in docs:
        fname = os.path.basename(doc.metadata["source"])
        url_id = fname.replace(".txt","")
        doc.metadata["url_id"] = url_id

    print(f"loaded {len(docs)} full documents ({total_characters} characters)")
    return docs

def chunk_docs(docs:List[Document], verbose: bool = False) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n" ,"\n", ".", " "]
    )

    chunks = splitter.split_documents(docs)
    print(f"loaded {len(chunks)} chunks")
    if verbose:
        for i, chunk in enumerate(chunks):
            print(f"processing chunk {len(chunk.page_content)} characters")
            print(f"processing chunk No.{i+1} {chunk}")
    return chunks

if __name__ == "__main__":
    documents = doc_loader()
    chunk_docs(documents, verbose=True)
