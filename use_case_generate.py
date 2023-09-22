"""
author: Elena Lowery

This code sample shows how to invoke watsonx.ai Generative AI API that's hosted in IBM Cloud.
At this time the API endpoint is hardcoded in the API itself (can't be edited).
You will need to provide your own API key for accessing watsonx.ai
This example shows a simple GENERATE/TRANSFORM use case without comprehensive prompt tuning.
"""

# Install genai in your Python env prior to running this example:
# pip install genai

import os
from genai.model import  Model
from genai.credentials import Credentials
from genai.schemas import GenerateParams

TASK_DEFAULT = "default"
TASK_GENERATE_EMAIL = "generate email"

def authenticate():
    # We hardcoded the API key to reuse code in Watson Studio/notebooks
    # A better approach for Python applications is to use the .env file
    # The API key is available in the watsonx.a workbench UI
  #  api_key = "***"
    api_key = os.getenv("GENAI_KEY", None)
    api_url = os.getenv("GENAI_API", None)

    # Create the credentials opbejct
    my_credentials = Credentials(api_key, api_endpoint=api_url)

    return my_credentials

def get_model():

    my_credentials=authenticate()

    # Instantiate parameters for text generation
    # See documentation for descriptions of parameters: https://workbench.res.ibm.com/docs/api-reference#generate
    model_parameters = GenerateParams(decoding_method="sample", temperature = 0.5, min_new_tokens=30, max_new_tokens=150)

    # Create a model object
    # We're using one of the most popular model types, but you can try other models
    # See documentation for model types: https://workbench.res.ibm.com/docs/models
    flan_ul2 = Model("google/flan-t5-xxl", params=model_parameters, credentials=my_credentials)

    return flan_ul2

def get_review():

    # This code can be replaced with getting a review from a file or another sources
    # Source of this review: DeepLearningAI Prompt Class example https://www.consumeraffairs.com

    # Try different types of reviews - one at a time or modify the code to read from file.

    # review for a blender
    service_review = f"""
    So, they still had the 17 piece system on seasonal sale for around $49 in the month of
    November, about  half off, but for some reason (call it price gouging) 
    around the second week of December the prices all went up to about anywhere from 
    between $70-$89 for the same system. And the 11 piece system went up around $10 or 
    so in price also from the earlier sale price of $29. So it looks okay, but if you look at the base, the part 
    where the blade locks into place doesn’t look as good as in previous editions from a few years ago, but I 
    plan to be very gentle with it (example, I crush very hard items like beans, ice, rice, etc. in the 
    blender first then pulverize them in the serving size I want in the blender then switch to the whipping 
    blade for a finer flour, and use the cross cutting blade first when making smoothies, then use the flat blade 
    if I need them finer/less pulpy). Special tip when making smoothies, finely cut and freeze the fruits and 
    vegetables (if using spinach-lightly stew soften the spinach then freeze until ready for use-and if making 
    sorbet, use a small to medium sized food processor) that you plan to use that way you can avoid adding so 
    much ice if at all-when making your smoothie. After about a year, the motor was making a funny noise. 
    I called customer service but the warranty expired already, so I had to buy another one. FYI: The overall 
    quality has gone done in these types of products, so they are kind of counting on brand recognition and 
    consumer loyalty to maintain sales. Got it in about two days.
    """

    return service_review

def get_prompt(service_review, task_type, sentiment):

    # Get the complete prompt by replacing variables

    if task_type == TASK_GENERATE_EMAIL:
        complete_prompt = f"""
        You are a customer service AI assistant. Your task is to generate an email reply to the customer.
        Using text delimited by ``` Generate a reply to thank the customer for their review.
        
        If the sentiment is positive or neutral, thank them for  their review.
        If the sentiment is negative, apologize and suggest that they can reach out to customer service. 
        
        Use specific details from the review.
        
        Write in a concise and professional tone.
        
        Sign the email as `AI customer agent`.
        Customer review: ```{service_review}```
        Review sentiment: {sentiment}
        """
    else:
        # Provide a summary of the text
        complete_prompt = f"""
        Summarize the review below, delimited by ''' , in at most 100 words.
        Review: ```{service_review}```
        """

    return complete_prompt

def generate():

    model=get_model()
    review = get_review()

    complete_prompt1 = get_prompt(review, TASK_GENERATE_EMAIL,"positive")

    # Sentiment
    prompts = [complete_prompt1]
    # print model response
    for response in model.generate(prompts):
        print("--------------------------------- Genereated email -----------------------------------")
        print("Prompt: " + complete_prompt1.strip())
        print("---------------------------------------------------------------------------------------------")
        print("Generated email: " + response.generated_text)
        print("*********************************************************************************************")

# Invoke the main function
generate()
