"""
Refactored by Tom McManus - example provided by Onkar Bhardwaj and Elena Lowery

This code sample shows how to invoke watsonx.ai Generative AI API that's hosted in IBM Cloud.
You will need to provide your own API key for accessing watsonx.ai
This example shows a simple Q&A use case without comprehensive prompt tuning.

IMPORTANT: Be aware of the disk space that will be taken up by documents when they're loaded into
ChromaDB on your laptop. The size in chroma will likely be the same as .txt file size
 
 pip install chromadb
 pip install sentence_transformers
 pip install "ibm-generative-ai[langchain]"
"""
import os
import chromadb
try: 
  from langchain.text_splitter import CharacterTextSplitter
  from langchain.document_loaders import TextLoader
  from sentence_transformers import SentenceTransformer
  from chromadb.api.types import EmbeddingFunction
  from genai import Model
  from genai.credentials import Credentials
  from genai.schemas import GenerateParams
except ImportError:
    raise ImportError("Could not import langchain: Please 'pip install "ibm-generative-ai[langchain]" sentence_transformers chromadb' ")  


# Embedding function
class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')

    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()

def authenticate():
    # We hardcoded the API key to reuse code in Watson Studio/notebooks
    # A better approach for Python applications is to use the .env file
    # The API key is available in the watsonx.a workbench UI
    api_key = os.getenv("GENAI_KEY", None)
    api_url = os.getenv("GENAI_API", None)

    # Create the credentials opbejct
    my_credentials = Credentials(api_key, api_endpoint=api_url)

    return my_credentials

def get_model():

    my_credentials=authenticate()

    # Instantiate parameters for text generation
    # See documentation for descriptions of parameters: https://workbench.res.ibm.com/docs/api-reference#generate
    model_parameters = GenerateParams(decoding_method="sample",max_new_tokens=300,min_new_tokens=50,stream=False,temperature=0.2,top_k=100,top_p=1,)

    # Create a model object
    # We're using one of the most popular model types, but you can try other models
    # See documentation for model types: https://workbench.res.ibm.com/docs/models
    model_response = Model("google/flan-ul2", params=model_parameters, credentials=my_credentials)

    return model_response

def create_embedding(filepath):

    loader = TextLoader(filepath)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)

    print(type(texts))

    # Load chunks into chromadb
    client = chromadb.Client()
    collection = client.create_collection(
        "docs_and_qs_test_minilm6v2",
        embedding_function=MiniLML6V2EmbeddingFunction()
    )
    collection.upsert(
        documents=[doc.page_content for doc in texts],
        ids=[str(i) for i in range(len(texts))],  # unique for each doc
    )

    return collection

def create_prompt(filepath, question):

    # Create embeddings for the text file
    collection = create_embedding(filepath)

    # query relevant information
    relevant_chunks = collection.query(
        query_texts=[question],
        n_results=5,
    )

    context = "\n\n\n".join(relevant_chunks["documents"][0])
    # Prompt for passages without unanswerable pv3
    prompt = (f"{context}\n\nPlease answer a question using this "
              + f"text. "
              + f"If the question is unanswerable, say \"unanswerable\"."
              + f"{question}")

    return prompt

def answer_questions_from_doc():

    # Ask a question relevant to the info in the document
    question="What did the president say about COVID-19?"
    # Provide the path relative to the dir in which the script is running
    # In this example the .txt file is in the same directory
    filepath='state_of_the_union.txt'

    # Get the watsonx model
    model=get_model()

    # Get the prompt
    prompt = create_prompt(filepath, question)

    # Let's review the prompt
    print("----------------------------------------------------------------------------------------------------")
  #  print("*** Prompt:" + prompt + "***")
    # Let's review the prompt
    print("----------------------------------------------------------------------------------------------------")

    responses = model.generate([prompt])
    print(question)
    print(responses[0].generated_text)

# Invoke the main function
answer_questions_from_doc()