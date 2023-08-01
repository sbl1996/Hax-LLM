import time
import os
import re
import importlib

from typing import List, Literal, Optional, Union

from transformers import AutoTokenizer

import jax.numpy as jnp

import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import ServerSentEvent, EventSourceResponse

from haxllm.chat.conversation import get_conv_template
from haxllm.pipeline.text_generation import ChatPipeline
from haxllm.chat.inference import generate_stream

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        system_message = prev_messages.pop(0).content
    else:
        system_message = ""

    conv = get_conv_template(conv_template)
    conv.system = system_message

    for msg in prev_messages:
        if msg.role == "user":
            conv.append_message(conv.roles[0], msg.content)
        elif msg.role == "assistant":
            conv.append_message(conv.roles[1], msg.content)
        elif msg.role == "system":
            raise HTTPException(status_code=400, detail="Invalid request")
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    gen_params = {
        "temperature": request.temperature or model.temperature,
        "top_k": request.top_k or model.top_k,
        "top_p": request.top_p or model.top_p,
        "max_len": request.max_length or model.max_len,
        "stop_token_ids": conv.stop_token_ids,
    }

    if request.stream:
        generate = predict(prompt, gen_params, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"][0]
    input_echo_len = len(input_ids)

    output_ids = model.random_sample(input_ids, **gen_params)
    output = tokenizer.decode(output_ids[input_echo_len:], skip_special_tokens=True)
    response = output.strip()

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def predict(prompt, gen_params, model_id: str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    model.reset_chat_state()

    gen_params["prompt"] = prompt
    output_stream = generate_stream(model, gen_params)

    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        # More fluent, but more requests, and more likely to be cut off
        # seps = r'([！，。？；：、 ?,.:“”‘’【】《》（）\n])'
        # Less fluent, and less requests, suitable for online chat
        seps = r'([！，。？；： \n])'
        output_text = re.split(seps, output_text)
        output_text = [x for x in output_text if x]
        now = len(output_text) - 1
        if now > pre:
            new_text = "".join(output_text[pre:now])
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=new_text),
                finish_reason=None
            )
            chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))
            pre = now
    new_text = "".join(output_text[pre:])
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=new_text),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    # current_length = 0
    # for outputs in output_stream:
    #     new_response = outputs["text"]
    #     print("Full: " + new_response)
    #     if len(new_response) == current_length:
    #         continue

    #     new_text = new_response[current_length:]
    #     print(new_text)
    #     current_length = len(new_response)
    #     choice_data = ChatCompletionResponseStreamChoice(
    #         index=0,
    #         delta=DeltaMessage(content=new_text),
    #         finish_reason=None
    #     )
    #     chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    #     yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


if __name__ == "__main__":
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.initialize_cache(os.path.expanduser("~/jax_cache"))
    import logging
    logging.getLogger("jax").setLevel(logging.WARNING)

    start = time.time()

    model_name = "chatglm2-6b"

    random_seed = 0
    tokenizer_name = "THUDM/chatglm2-6b"
    conv_template = "chatglm2-6b"

    checkpoint = "/home/hastur/Code/weights/chatglm2-6b_np.safetensors"

    temperature = 0.8
    top_p = 0.8
    top_k = -1

    mesh = [1, 8]
    dtype = jnp.bfloat16
    param_dtype = jnp.bfloat16

    print(f"Loading tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.decode(tokenizer("init")['input_ids'])
    print("Load tokenizer {}".format(time.time() - start))

    parallel = mesh is not None

    module = "haxllm.model"
    mod = importlib.import_module(module + "." + "chatglm2")

    model_config = {"name": model_name, "scan": True}
    config = getattr(mod, "load_config")(
        dtype=jnp.dtype(dtype),
        param_dtype=jnp.dtype(param_dtype),
        **model_config,
        decode=True,
        shard=parallel,
        shard_cache=parallel,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.pad_token_id

    model = getattr(mod, "TransformerLMHeadModel")(config)

    max_len = 1024
    model = ChatPipeline(
        tokenizer, model, max_len=max_len, seed=random_seed,
        pad_multiple=128, temperature=temperature, top_p=top_p, top_k=top_k)
    model.init(transformer_weight=checkpoint, mesh=mesh)

    print("Conversation setting:")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  top_k: {top_k}")
    print(f"  max_len: {max_len}")

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)