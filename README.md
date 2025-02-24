# File: /rag-llama-app/rag-llama-app/README.md

# RAG Llama Application

This project implements a Retrieval-Augmented Generation (RAG) application using LLAMA and Milvus Vector Store. The application takes a PDF as input, splits it into chunks, and uses LLAMA's embedding to convert the chunks into vectors stored in Milvus.

## Project Structure

```
rag-llama-app
├── src
│   ├── data_processing
│   │   ├── __init__.py
│   │   ├── pdf_loader.py
│   │   └── text_splitter.py
│   ├── embedding
│   │   ├── __init__.py
│   │   └── llama_embedder.py
│   ├── vector_store
│   │   ├── __init__.py
│   │   └── milvus_store.py
│   ├── config
│   │   ├── __init__.py
│   │   └── settings.py
│   └── main.py
├── tests
│   ├── __init__.py
│   ├── test_pdf_loader.py
│   ├── test_text_splitter.py
│   └── test_llama_embedder.py
├── requirements.txt
├── .env
├── docker-compose.yml
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd rag-llama-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in the `.env` file.

## Usage

1. Place your PDF file in the appropriate directory.
2. Run the application:
   ```
   python src/main.py
   ```

## Testing

To run the tests, use:
```
pytest tests/
```

## License

This project is licensed under the MIT License.