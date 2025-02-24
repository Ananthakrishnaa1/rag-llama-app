import os

MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
MILVUS_PORT = os.getenv('MILVUS_PORT', '19530')
PDF_PATH = os.getenv('PDF_PATH', 'path/to/pdf')
