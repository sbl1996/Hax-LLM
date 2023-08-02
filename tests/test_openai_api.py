import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "none"

# Test completion

req = openai.ChatCompletion.create(
    model="chatglm2-6b",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=False
)
print(req.choices[0].message.content)


# Test stream 

for chunk in openai.ChatCompletion.create(
    model="chatglm2-6b",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True
):
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)

print()

for chunk in openai.ChatCompletion.create(
    model="chatglm2-6b",
    messages=[
        {"role": "user", "content": "晚上睡不着应该怎么办"},
    ],
    stream=True
):
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)