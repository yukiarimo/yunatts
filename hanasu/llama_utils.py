import torch
from transformers import AutoTokenizer, AutoModel

class LlamaEmbedding:
    # Class-level singleton instance
    _instance = None

    @classmethod
    def get_instance(cls, model_name="yukiarimo/himitsu-v1-full", device=None):
        """
        Get or create the singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(model_name, device)
        return cls._instance

    def __init__(self, model_name="yukiarimo/himitsu-v1-full", device=None):
        """
        Initialize the Llama embedding model

        Args:
            model_name (str): Name of the Llama model to use
            device (str): Device to run the model on (cuda, cpu, mps)
        """
        self.model_name = model_name

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # For Apple Silicon
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading Llama model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        print(f"Llama model loaded successfully.")

        # Verify embedding size
        self.embedding_size = self.model.config.hidden_size
        print(f"Llama embedding size: {self.embedding_size}")

    def get_embeddings(self, text, max_length=512):
        """
        Get embeddings for a text

        Args:
            text (str): Input text
            max_length (int): Maximum sequence length

        Returns:
            torch.Tensor: Embeddings tensor of shape [embedding_size, sequence_length]
        """
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                               truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Use the last hidden state as embeddings
        embeddings = outputs.last_hidden_state[0].transpose(0, 1)  # [embedding_size, sequence_length]

        return embeddings

    def get_embeddings_batch(self, texts, max_length=512):
        """
        Get embeddings for a batch of texts

        Args:
            texts (list): List of input texts
            max_length (int): Maximum sequence length

        Returns:
            list: List of embeddings tensors
        """
        embeddings_list = []

        for text in texts:
            embeddings = self.get_embeddings(text, max_length)
            embeddings_list.append(embeddings)

        return embeddings_list

# Global module-level functions that use the singleton

def get_llama_feature(text, device=None):
    """
    Get Llama embeddings for a text

    Args:
        text (str): Input text
        device (str): Device to run the model on

    Returns:
        torch.Tensor: Embeddings tensor
    """
    # Get the singleton instance
    model = LlamaEmbedding.get_instance(device=device)

    # Get embeddings
    embeddings = model.get_embeddings(text)

    return embeddings

def get_llama_features_batch(texts, device=None):
    """
    Get Llama embeddings for a batch of texts

    Args:
        texts (list): List of input texts
        device (str): Device to run the model on

    Returns:
        list: List of embeddings tensors
    """
    # Get the singleton instance
    model = LlamaEmbedding.get_instance(device=device)

    # Get embeddings
    embeddings_list = model.get_embeddings_batch(texts)

    return embeddings_list