#import inspect
import os

from genai.extensions.langchain import LangChainInterface
#from langchain.callbacks.base import BaseCallbackHandler

from genai.schemas import GenerateParams
from genai.credentials import Credentials
#from genai.model import Model
#from typing import Any, Optional
#from uuid import UUID



if __name__ == '__main__':
    print("Hello langchain")
    print(os.environ['GENAI_KEY'])
    

api_key = os.getenv("GENAI_KEY", None) 
api_url = os.getenv("GENAI_API", None)
# credentials object to access GENAI
creds = Credentials(api_key, api_endpoint=api_url)

print("\n------------- Example (LangChain)-------------\n")

params = GenerateParams(decoding_method="greedy")

print("Using GenAI Model expressed as LangChain with callbacks")

langchain_model = LangChainInterface(model="google/flan-t5-xxl", params=params, credentials=creds)
print(langchain_model("Answer this question: What is life?"))





    

    




