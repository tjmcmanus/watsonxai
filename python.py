import os

from genai.schemas import GenerateParams
from genai.credentials import Credentials
from genai.model import Model

api_key = os.getenv("GENAI_KEY", None) 
api_url = os.getenv("GENAI_API", None)
# credentials object to access GENAI
creds = Credentials(api_key, api_endpoint=api_url)

params1 = GenerateParams( decoding_method="greedy", max_new_tokens=15, min_new_tokens=1, stream=False)

model = Model("google/flan-t5-xxl", params=params1, credentials=creds)
             
modelresponse = model.generate(["Answer this question: What is life?"])

textout= modelresponse[0].generated_text 
print(f"Generated reponse is --> {textout}")


