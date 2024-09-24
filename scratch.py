from litellm import completion
import os

os.environ['GROQ_API_KEY'] = "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"
response = completion(
    model="groq/llama3-8b-8192", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)