from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

class MilvusStore:
    def __init__(self, host, port):
        try:
            connections.connect("default", host=host, port=port)
            self.collection_name = 'embeddings'
            self.collection = None
            self.dim = 3072  # Will be set dynamically
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")

    def create_collection(self, dim=None):
        """
        Creates a collection with specified dimensions
        Args:
            dim (int): Dimension of the embedding vectors
        """
        try:
            # If collection exists, load it and verify dimensions
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                schema = self.collection.schema
                existing_dim = next(f.params['dim'] for f in schema.fields if f.name == 'embedding')
                if dim and dim != existing_dim:
                    utility.drop_collection(self.collection_name)
                else:
                    return

            if not dim:
                raise ValueError("Dimension size must be specified")

            self.dim = 3072
            print(f"Creating collection with dimension: {self.dim}")

            # Define fields
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            # Create collection schema
            schema = CollectionSchema(fields=fields, description="Document embeddings")
            
            # Create collection
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            
        except Exception as e:
            raise Exception(f"Failed to create collection: {str(e)}")

    def insert(self, embeddings, texts):
        """
        Insert embeddings and their corresponding texts into the collection
        """
        try:
            data = [
                {"embedding": embedding, "text": text}
                for embedding, text in zip(embeddings, texts)
            ]
            self.collection.insert(data)
            return True
        except Exception as e:
            raise Exception(f"Failed to insert data: {str(e)}")
    def __init__(self, host, port):
        try:
            connections.connect("default", host=host, port=port)
            self.collection_name = 'embeddings'
            self.collection = None
            self.dim = None  # Will be set dynamically
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")

    def create_collection(self, dim=None):
        """
        Creates a collection with specified dimensions
        Args:
            dim (int): Dimension of the embedding vectors
        """
        try:
            # If collection exists, load it and verify dimensions
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                schema = self.collection.schema
                existing_dim = next(f.params['dim'] for f in schema.fields if f.name == 'embedding')
                if dim and dim != existing_dim:
                    utility.drop_collection(self.collection_name)
                else:
                    return

            if not dim:
                raise ValueError("Dimension size must be specified")

            self.dim = dim
            print(f"Creating collection with dimension: {self.dim}")

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            schema = CollectionSchema(fields=fields, description="Document embeddings")
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            
        except Exception as e:
            raise Exception(f"Failed to create collection: {str(e)}")
    def __init__(self, host, port):
        # Establishes connection to a Milvus server
        # Sets up configuration for embedding storage with 768 dimensions
        try:
            connections.connect("default", host=host, port=port)
            self.collection_name = 'embeddings'
            self.dim = 3072  # dimension of embeddings
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")

    def create_collection(self):
        # Creates a collection with three fields:
    # - id: auto-generated primary key
    # - embedding: vector field for storing embeddings
    # - text: varchar field for storing associated text
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                return

            # Define fields
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            # Create collection schema
            schema = CollectionSchema(fields=fields, description="Document embeddings")
            
            # Create collection
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create index (optional but recommended)
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            
        except Exception as e:
            raise Exception(f"Failed to create collection: {str(e)}")

    def insert(self, embeddings, texts):
        # Inserts pairs of embeddings and their corresponding texts
        try:
            if not isinstance(embeddings, list) or not isinstance(texts, list):
                raise ValueError("Embeddings and texts must be lists")
            
            if len(embeddings) != len(texts):
                raise ValueError("Number of embeddings and texts must match")
            
            entities = [
                {"embedding": embedding, "text": text}
                for embedding, text in zip(embeddings, texts)
            ]
            
            self.collection.insert(entities)
            self.collection.flush()  # Ensure data is persisted
            
        except Exception as e:
            raise Exception(f"Failed to insert data: {str(e)}")
    
    def get_collection_stats(self):
        """Returns statistics about the collection"""
        try:
            # Load collection
            self.collection.load()
            
            # Get basic stats
            stats = {
                "name": self.collection_name,
                "count": self.collection.num_entities,
                "schema": self.collection.schema
            }
            
            # Get some sample entries
            if self.collection.num_entities > 0:
                results = self.collection.query(
                    expr="id < 5",  # Get first few entries
                    output_fields=["id", "text"],
                    limit=5
                )
                stats["samples"] = results
            
            return stats
            
        except Exception as e:
            raise Exception(f"Failed to get collection stats: {str(e)}")