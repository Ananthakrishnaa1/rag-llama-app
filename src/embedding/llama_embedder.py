from typing import List
import ollama

class LLamaEmbedder:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text using Ollama's API
        
        Args:
            text (str): Input text to generate embeddings for
            
        Returns:
            List[float]: Vector embeddings
        """
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            # Convert the response to string and extract just the numbers
            response_str = str(response)
            # Remove 'embedding=' prefix and extract the list of floats
            embedding_str = response_str.replace('embedding=', '').strip('[]')
            # Convert string of numbers to actual float list
            embedding_list = [float(x.strip()) for x in embedding_str.split(',') if x.strip()]
            return embedding_list
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[List[float]]: List of vector embeddings
        """
        return [self.embed(text) for text in texts]