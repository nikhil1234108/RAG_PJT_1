import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

ARICLES_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data","articles")

def doc_loader(articles_dir:str = ARICLES_DIRECTORY) -> List[Document]:
    loader = DirectoryLoader(
        articles_dir,
        glob = ".txt",
        loader_cls = TextLoader
    )

    docs = loader.load()

    for doc in docs:
        fname = os.path.basename(doc.metadata["source"])
        url_id = fname.replace(".txt","")
        doc.metadata["url_id"] = url_id

    print(f"loaded {len(docs)} documents")
    return docs

def chunk_docs(docs:List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n" ,"\n", ".", " "]
    )

    chunks = splitter.split_documents(docs)
    print(f"loaded {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"processing chunk {len(chunk)}")
        print(f"processing chunk No.{i+1} {chunk}")
    return chunks