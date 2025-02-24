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
        
        
    
    def get_collection_stats(self):
        """
        Returns statistics about the collection including name, count and sample documents
        
        Returns:
            dict: Collection statistics including name, count and samples
        """
        try:
            if not utility.has_collection(self.collection_name):
                raise Exception("Collection does not exist")
                
            # Load the collection if not already loaded
            if self.collection is None:
                self.collection = Collection(self.collection_name)
                self.collection.load()

            # Get collection stats
            stats = {
                'name': self.collection_name,
                'count': self.collection.num_entities
            }

            # Get sample documents if collection is not empty
            if stats['count'] > 0:
                # Query some sample documents
                results = self.collection.query(
                    expr="id < 5",  # Get first 5 documents
                    output_fields=["text"],
                    consistency_level="Strong"
                )
                stats['samples'] = [{'id': i, 'text': doc['text']} 
                                  for i, doc in enumerate(results)]

            return stats
        except Exception as e:
            raise Exception(f"Failed to get collection stats: {str(e)}")
        finally:
            # Release collection after use
            if self.collection:
                self.collection.release()