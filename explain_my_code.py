import inspect
import os


from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams

# make sure you have a .env file under genai root with
api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", None)

print("\n------------- Example (Explain my code)-------------\n")

params = GenerateParams(
    decoding_method="sample",
    max_new_tokens=50,
    min_new_tokens=1,
    stream=False,
    temperature=0.7,
    top_k=50,
    top_p=1,
)

creds = Credentials(api_key, api_endpoint)
code_explainer = Model("google/flan-ul2", params=params, credentials=creds)


# pass in an actual python function to explain
def add_numbers(number_one, number_two):
    return number_one + number_two


prompt = inspect.getsource(add_numbers) + "# Explanation of what the code does"
print(prompt + "\n")
responses = code_explainer.generate([prompt])
for response in responses:
    print(f"Generated text:\n {response.generated_text}")