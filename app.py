from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="meta/llama-3.2-3b-instruct",
  messages=[{"role":"user","content":"Provide me an article of Deep learning."}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
