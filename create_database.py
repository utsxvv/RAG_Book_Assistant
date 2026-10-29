from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()
loader = PyPDFLoader("document loaders/book.pdf")
documents = loader.load()
documents = [d for d in documents if d.page_content and d.page_content.strip()]
if not documents:
    raise ValueError("No readable text extracted. Scanned PDFs may need OCR.")

print(f"Loaded {len(documents)} pages containing text.")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
if not chunks:
    raise ValueError("No chunks created. Check the extracted PDF text.")
print(f"Created {len(chunks)} chunks.")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db",
    collection_name="book_collection",
)
print("Persistent database created successfully!")
