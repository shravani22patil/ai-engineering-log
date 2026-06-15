import os   #built-in Python - accessing environment variables
from dotenv import load_dotenv  #reads .env file into os.environ
import anthropic  #Anthropic's official Python SDK for making API calls

#This will load my ANTHROPIC_API_KEY from the .env file into my environment variables
#without this, my API dosen't exist in python's scope and I won't be able to use it
load_dotenv() #loads the .env file in the current directory and adds the variables to os.environ

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise SystemExit("Missing ANTHROPIC_API_KEY environment variable. Check your .env file and remove any extra quotes around the key.")

#Creating a client - this will be my connection to the Anthropic API, and I'll use it to send requests and receive responses from the API
client = anthropic.Anthropic(api_key=api_key)

#Making a call - breaking down every parameter of the API call:
try:
    message = client.messages.create(
        model = "claude-haiku-4-5-20251001", #the model I want to use for this API call. This is the latest version of Claude Haiku, which has a 200k token context window and is optimized for reasoning and creating long-form content
        max_tokens = 256, #maximum number of tokens I want in the response.
        #not the request - caps my cost per call

        messages =[   # list of conversation turns
        {
            "role": "user",  #"user" = human's message to the model
            "content": "What is a token in the context of LLMs? Answer in 3 sentences."
        }
     ]
    )
except Exception as e:
    print("===API ERROR===")
    print(type(e).__name__ + ":", e)
    print("\nThis means the request did not complete. If you are using Anthropic, check your account credit or billing settings.")
    import sys
    sys.exit(1)

#The response object contains a lot of information - exploring them all 
print("===RESPONSE TEXT===")
print(message.content[0].text) #the actual answer from the model

print("\n===TOKEN USAGE===")
print(f"Input tokens: {message.usage.input_tokens}") #number of tokens in the prompt I sent to the model (including system instructions and conversation history)
print(f"Output tokens: {message.usage.output_tokens}") #number of tokens in the model's response
print(f"Total tokens: {message.usage.total_tokens}") #total number of tokens used in this API call (input + output tokens)

#Calculating the cost of this API call based on the number of tokens used and the model's pricing
# Claude Haiku: $0.00025/1k input, $0.00125/1k output
input_cost = message.usage.input_tokens / 1000 * 0.00025
output_cost = message.usage.output_tokens / 1000 * 0.00125
print(f"\nThis call cost : ${input_cost + output_cost:.6f}")

print("\n===FULL RESPONSE OBJECT===")
print(f"Model: {message.model}") #the model that generated this response 
print(f"Stop reason: {message.stop_reason}") #why the model stopped generating text , 
                                             #'end_turn' = model finished normally
                                             # 'max_tokens' = you cut it off