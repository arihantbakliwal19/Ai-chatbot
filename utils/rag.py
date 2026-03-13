from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models.embeddings import get_embedding_model


def load_documents(path):

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(docs)


def create_vector_db(documents):

    embeddings = get_embedding_model()

    vector_db = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_db


def retrieve_context(query, vector_db):

    docs = vector_db.similarity_search(query, k=3)

    context = "\n".join([d.page_content for d in docs])

    return context