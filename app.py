from openai import OpenAI
import config
client = OpenAI(api_key=config.api_key)
response = client.chat.completions.create(
   model="gpt-3.5-turbo",
   messages=[
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Explain how a transformer model works."}
   ],
   temperature=0.7
)


print(response.choices[0].message.content)
