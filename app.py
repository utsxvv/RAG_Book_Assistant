import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from rag_engine import build_vectorstore, answer_question

load_dotenv()

st.set_page_config(
    page_title="BookChat | RAG Assistant",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("📚 BookChat")
st.caption("Upload a PDF book and ask questions about its content.")

st.divider()


st.subheader("📖 Upload Your Book")

uploaded_file = st.file_uploader(
    "Choose a PDF book",
    type=["pdf"],
    help="Upload a text-based PDF to start asking questions."
)

col1, col2 = st.columns([1, 1])

with col1:
    process_clicked = st.button(
        "⚡ Process Book",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None
    )

with col2:
    clear_clicked = st.button(
        "🗑️ Clear Book & Chat",
        use_container_width=True,
        disabled=st.session_state.get("vectorstore") is None
    )


if clear_clicked:
    for key in (
        "vectorstore",
        "book_name",
        "chunk_count",
        "page_count",
        "messages"
    ):
        st.session_state.pop(key, None)

    st.rerun()


if process_clicked and uploaded_file is not None:
    try:
        with st.spinner("Reading and indexing your book... Please wait."):
            vectorstore, chunk_count, page_count = build_vectorstore(
                uploaded_file.getvalue(),
                get_embedding_model()
            )

        st.session_state.vectorstore = vectorstore
        st.session_state.book_name = uploaded_file.name
        st.session_state.chunk_count = chunk_count
        st.session_state.page_count = page_count
        st.session_state.messages = []

        st.success("Book processed successfully!")

    except Exception as exc:
        st.error(f"Could not process this PDF: {exc}")


if st.session_state.get("vectorstore") is not None:
    st.divider()
    st.subheader("✅ Currently Loaded Book")

    info1, info2 = st.columns(2)

    with info1:
        st.metric("Book", st.session_state.book_name)

    with info2:
        st.metric("Pages with Text", st.session_state.page_count)



st.divider()
st.subheader("💬 Chat with Your Book")

if st.session_state.get("vectorstore") is None:
    st.info(
        "Upload a PDF book above and click **Process Book** "
        "before asking questions."
    )

else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📄 Source Pages"):
                    st.write(
                        ", ".join(
                            str(page) for page in message["sources"]
                        )
                    )

    question = st.chat_input(
        f"Ask a question about {st.session_state.book_name}..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the book and generating an answer..."):
                try:
                    answer, source_pages = answer_question(
                        question,
                        st.session_state.vectorstore,
                        get_llm()
                    )

                    st.markdown(answer)

                    if source_pages:
                        with st.expander("📄 Source Pages"):
                            st.write(
                                ", ".join(
                                    str(page) for page in source_pages
                                )
                            )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": source_pages
                    })

                except Exception as exc:
                    error_message = (
                        f"Sorry, I couldn't answer that question. Error: {exc}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
