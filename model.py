#Just for test
from openai import AzureOpenAI

client = AzureOpenAI(api_key = "",
                     api_version = '2024-06-01',
                     azure_endpoint = 'https://hkust.azure-api.net')



response = client.chat.completions.create(
    model="gpt-35-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who are you?"},
    ],
)
    
print(response.choices[0].message.content)
