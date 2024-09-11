import time
import os
import re

from typing import List, Literal, Optional
import shortuuid
import hydra
from omegaconf import DictConfig

import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from asyncio import sleep

from haxllm.chat.conversation import get_conv_template
from haxllm.chat.inference import generate_stream
from haxllm.chat.config_utils import load_config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

global model, tokenizer, conv_template

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


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    model: str
    temperature: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    model: str
    object: str = "chat.completion"
    choices: List[ChatCompletionResponseChoice]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    prev_messages = request.messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0].role == "system":
        system_message = prev_messages.pop(0).content.strip()
    else:
        system_message = None

    conv = get_conv_template(conv_template)
    if system_message is not None:
        # if it's "", system prompt is disabled
        conv.config.system = system_message

    # Magic to attack
    magic_cmd = "[@]"
    if magic_cmd in query:
        index = query.index(magic_cmd)
        adv_prompt = query[index+len(magic_cmd):].strip()
        query = query[:index].strip()
    else:
        adv_prompt = ""
    
    for msg in prev_messages:
        if msg.role == "user":
            content = msg.content
            index = content.find(magic_cmd)
            if index != -1:
                content = content[:index].strip()
            conv.append_message(conv.config.roles[0], content)
        elif msg.role == "assistant":
            conv.append_message(conv.config.roles[1], msg.content)
        elif msg.role == "system":
            raise HTTPException(status_code=400, detail="Invalid request")
    
    conv.append_message(conv.config.roles[0], query)
    conv.append_message(conv.config.roles[1], None)
    prompt = conv.get_prompt() + adv_prompt
    print("adv prompt:", adv_prompt)
    print(prompt)

    repetition_penalty = model.repetition_penalty
    if request.frequency_penalty is not None:
        repetition_penalty = request.frequency_penalty + 1.0
    gen_params = {
        "temperature": request.temperature or model.temperature,
        # "top_k": request.top_k or model.top_k,
        "top_p": request.top_p or model.top_p,
        "repetition_penalty": repetition_penalty,
        "max_len": request.max_tokens or model.max_len,
        "max_new_tokens": model.max_new_tokens,
        "min_p": model.min_p,
    }

    if request.stream:
        generator = chat_completion_stream_generator(prompt, gen_params, request.model, adv_prompt)
        return StreamingResponse(generator, media_type="text/event-stream")

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

    id = f"chatcmpl-{shortuuid.random()}"
    return ChatCompletionResponse(id=id, model=request.model, choices=[choice_data])


def wrap_sse(chunk):
    return f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"


async def chat_completion_stream_generator(prompt, gen_params, model_id: str, prefill: str):
    id = f"chatcmpl-{shortuuid.random()}"
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionStreamResponse(
        id=id, model=model_id, choices=[choice_data])
    yield wrap_sse(chunk)

    model.reset_chat_state()

    gen_params["prompt"] = prompt
    output_stream = generate_stream(model, gen_params)
    
    start = time.time()
    times = []
    pre = 0
    for outputs in output_stream:
        end = time.time()
        times.append(end - start)
        start = end
        output_text = prefill + outputs["text"]
        usage = outputs["usage"]
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
            chunk = ChatCompletionStreamResponse(id=id, model=model_id, choices=[choice_data])
            yield wrap_sse(chunk)
            pre = now

            # HACK: without this, SSE won't work for NextChat
            await sleep(0.001)
    times = times[:-1]
    prefill_time = times[0]
    decode_time = sum(times[1:])
    prefill_tps = usage['prompt_tokens'] / prefill_time
    decode_tps = usage['completion_tokens'] / decode_time
    print(
        f"prompt: {usage['prompt_tokens']}/{prefill_time:.3f}/{prefill_tps:.2f}, "
        f"completion: {usage['completion_tokens']}/{decode_time:.3f}/{decode_tps:.2f}, "
        f"total: {usage['total_tokens']}")

    new_text = "".join(output_text[pre:])
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(content=new_text),
        finish_reason=None
    )
    chunk = ChatCompletionStreamResponse(
        id=id, model=model_id, choices=[choice_data])
    yield wrap_sse(chunk)

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionStreamResponse(
        id=id, model=model_id, choices=[choice_data])
    yield wrap_sse(chunk)
    yield "data: [DONE]\n\n"


def compile():
    start = time.time()
    print("Compiling model stream mode...")
    model.reset_chat_state()
    gen_params = {
        "temperature": model.temperature,
        "top_p": model.top_p,
        "min_p": model.min_p,
        "repetition_penalty": model.repetition_penalty,
        "max_len": model.max_len,
        "max_new_tokens": 3,
        "prompt": "hello"
    }
    output = generate_stream(model, gen_params)
    s = ""
    for x in output:
        s += x["text"]
    print(f"Prompt: {gen_params['prompt']}, output: {repr(s)}")
    print(f"Compilation finished in {time.time() - start:.2f}s")


@hydra.main(version_base=None, config_path="../../configs/chat", config_name="base")
def chat_api_server(cfg: DictConfig) -> None:
    from jax_smi import initialise_tracking
    initialise_tracking()
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.set_cache_dir(os.path.expanduser("~/jax_cache"))
    import logging
    logging.getLogger("jax").setLevel(logging.WARNING)

    global model, tokenizer, conv_template

    pipeline, conv_template = load_config(cfg)
    model = pipeline
    tokenizer = pipeline.tokenizer

    print("Default conversation setting:")
    print(f"  temperature: {pipeline.temperature}")
    print(f"  top_p: {pipeline.top_p}")
    print(f"  min_p: {pipeline.min_p}")
    print(f"  top_k: {pipeline.top_k}")
    print(f"  repetition_penalty: {pipeline.repetition_penalty}")
    print(f"  max_len: {pipeline.max_len}")
    print(f"  chunk_size: {pipeline.pad_multiple}")
    compile()

    host = getattr(cfg, "host", "127.0.0.1")
    port = getattr(cfg, "port", 8000)
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    chat_api_server()