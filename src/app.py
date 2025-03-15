import streamlit as st
import tempfile
import os
from config.config import settings 
from data_processing.pdf_loader_1 import load_pdf
from data_processing.text_splitter import split_text
from embedding.llama_embedder import LLamaEmbedder
# from vector_store.milvus_store import MilvusStore
from vector_store.pinecone_store import PineconeStore

def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Process the PDF
        pdf_text = load_pdf(tmp_path)
        chunks = list(split_text(pdf_text))

        embedder = LLamaEmbedder()
        embeddings = [embedder.embed(chunk) for chunk in chunks]
        
        # Get dimension from first embedding
        # embedding_dim = len(embeddings[0])

        # store = MilvusStore(settings.MILVUS_HOST, settings.MILVUS_PORT)
        # store.create_collection(dim=embedding_dim)
        
        embeddings = [embedder.embed(chunk) for chunk in chunks]

        store = PineconeStore(
            api_key=settings.PINECONE_API_KEY,
            index_name=settings.PINECONE_INDEX_NAME
        )
        store.insert(embeddings, chunks)

        return True, "PDF processed and embedded successfully!"
    except Exception as e:
        return False, f"Error processing PDF: {str(e)}"
    finally:
        os.unlink(tmp_path)


def main():
    st.title("PDF Document Processor")

    # Add a tab for viewing stored data
    tab1, tab2 = st.tabs(["Upload PDF", "View Stored Data"])

    with tab1:
        st.write("Upload a PDF file to process and store embeddings")

        # File uploader for PDF
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Process the uploaded PDF file
            success, message = process_pdf(uploaded_file)
            if success:
                st.success(message)
            else:
                st.error(message)

    with tab2:
        if st.button("Show Collection Stats"):
            try:
                # store = MilvusStore(settings.MILVUS_HOST, settings.MILVUS_PORT)
                store = PineconeStore(
                    api_key=settings.PINECONE_API_KEY,
                    index_name=settings.PINECONE_INDEX_NAME
                )
                stats = store.get_collection_stats()
                
                st.write(f"Collection Name: {stats['name']}")
                st.write(f"Total Documents: {stats['count']}")
                
                if stats.get('samples'):
                    st.write("Sample Documents:")
                    for doc in stats['samples']:
                        st.text_area(f"Document ID: {doc['id']}", doc['text'], height=100)
            except Exception as e:
                st.error(f"Error fetching stats: {str(e)}")

if __name__ == "__main__":
    main()
