import os
import tempfile
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


def build_vectorstore(pdf_bytes, embedding_model):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        pages = PyPDFLoader(temp_path).load()
        if not pages:
            raise ValueError("No pages were found in this PDF.")

        pages = [p for p in pages if p.page_content and p.page_content.strip()]
        if not pages:
            raise ValueError(
                "No readable text was extracted. This may be a scanned PDF; run OCR first."
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        if not chunks:
            raise ValueError("No text chunks were created from this PDF.")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name="book_" + uuid.uuid4().hex[:20],
        )
        return vectorstore, len(chunks), len(pages)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def answer_question(question, vectorstore, llm):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)
    if not docs:
        return "I don't know based on the provided book content.", []

    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful book assistant.
Answer using only the provided context from the uploaded book.
If the answer is not present in the context, say:
"I don't know based on the provided book content."
Do not invent information."""),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    formatted_prompt = prompt.invoke({"context": context, "question": question})
    answer = llm.invoke(formatted_prompt)
    source_pages = sorted({
        int(doc.metadata["page"]) + 1
        for doc in docs if "page" in doc.metadata
    })
    return answer.content, source_pages
