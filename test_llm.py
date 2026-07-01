from groq import Groq

# Put your API key here
client = Groq(api_key="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")

# Send a message to the LLM
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "What is artificial intelligence? Explain in 3 lines."
        }
    ]
)

# Print the response
print(response.choices[0].message.content)