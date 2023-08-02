import time
import os
import re

from typing import List, Literal, Optional, Union

import hydra
from omegaconf import DictConfig

import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

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

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


@hydra.main(version_base=None, config_path="../../configs/chat", config_name="base")
def chat_api_server(cfg: DictConfig) -> None:
    from jax.experimental.compilation_cache import compilation_cache as cc
    cc.initialize_cache(os.path.expanduser("~/jax_cache"))
    import logging
    logging.getLogger("jax").setLevel(logging.WARNING)

    global model, tokenizer, conv_template

    pipeline, conv_template = load_config(cfg)
    model = pipeline
    tokenizer = pipeline.tokenizer

    print("Default conversation setting:")
    print(f"  temperature: {pipeline.temperature}")
    print(f"  top_p: {pipeline.top_p}")
    print(f"  top_k: {pipeline.top_k}")
    print(f"  max_len: {pipeline.max_len}")

    host = getattr(cfg, "host", "127.0.0.1")
    port = getattr(cfg, "port", 8000)
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    chat_api_server()