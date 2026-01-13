'''
This is a fastapi dinfer serving of llada
'''

# pylint: disable=import-error, no-name-in-module
# pylint: disable=global-statement, global-variable-not-assigned
# pylint: disable=too-few-public-methods
# pylint: disable=broad-exception-caught

import json
import logging
import os
from queue import Queue
import threading
import time
from typing import Dict, List, Optional
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer
import uvicorn


def init_logger(log_path: str = None):
    ''' init logger with exception handling '''
    log_format = '[%(asctime)s](%(created).17g) - %(levelname)s - ' \
                 '|%(pathname)s|%(funcName)s|%(lineno)d| - %(message)s'
    date_format = '%Y-%m-%d_%H:%M:%S'

    class CustomFormatter(logging.Formatter):
        ''' CustomFormatter '''

        def format(self, record):
            ''' format '''
            try:
                return super().format(record)
            except TypeError:
                if hasattr(record, 'args') and record.args:
                    record.msg = f"{record.msg} - Unformattable args: {record.args}"
                    record.args = ()
                return super().format(record)

    task_logging_level = os.environ.get("TASK_LOGGING_LEVEL", "INFO")
    log_level = logging.INFO if task_logging_level == "INFO" else logging.DEBUG
    logging.basicConfig(filename=log_path,
                        level=log_level,
                        format=log_format,
                        datefmt=date_format)

    for handler in logging.root.handlers:
        handler.setFormatter(CustomFormatter(log_format, date_format))

    return logging


logging = init_logger()

from dinfer import DiffusionLLMServing, SamplingParams, ThresholdParallelDecoder  # pylint: disable=wrong-import-position


class CompletionsRequest(BaseModel):
    ''' Completions Request
    '''
    messages: List[Dict] = Field(title='messages')
    stream: bool = Field(title='stream')


class StreamRequest(BaseModel):
    '''Stream Request'''
    chat_uuid: str = Field(title='chat_uuid')
    curr_x: Optional[List] = Field(title='curr_x')


app = FastAPI(
    title='xyz dllm serving',
    redoc_url=None,
    docs=None,
)
STATUS_OK = 1
STATUS_ERR = 0

SPECIAL_MODEL_DIR = os.environ.get('SPECIAL_MODEL_DIR')
TASK_DLLM_NUM_GPUS = int(os.environ.get('TASK_DLLM_NUM_GPUS', 1))

TASK_DLLM_GEN_LENGTH = int(os.environ.get('TASK_DLLM_GEN_LENGTH', 512))
TASK_DLLM_BLOCK_LENGTH = int(os.environ.get('TASK_DLLM_BLOCK_LENGTH', 32))

TASK_DLLM_MAX_LENGTH = int(os.environ.get('TASK_DLLM_MAX_LENGTH', 4096))
TASK_DLLM_BATCH_SIZE = int(os.environ.get('TASK_DLLM_BATCH_SIZE', 2))
TASK_DLLM_TEMPERATURE = float(os.environ.get('TASK_DLLM_TEMPERATURE', 0.0))
TASK_DLLM_THRESHOLD = float(os.environ.get('TASK_DLLM_THRESHOLD', 0.9))

TASK_DLLM_MASK_ID = int(os.environ.get('TASK_DLLM_MASK_ID', 156895))
TASK_DLLM_EOS_ID = int(os.environ.get('TASK_DLLM_EOS_ID', 156892))

TASK_STREAM_SLEEP_SECONDS = float(
    os.environ.get('TASK_STREAM_SLEEP_SECONDS', 0.005))


def get_dllm():
    ''' get dllm '''
    dllm_tokenizer = AutoTokenizer.from_pretrained(SPECIAL_MODEL_DIR,
                                                   trust_remote_code=True)
    decoder = ThresholdParallelDecoder(temperature=TASK_DLLM_TEMPERATURE,
                                       threshold=TASK_DLLM_THRESHOLD,
                                       mask_id=TASK_DLLM_MASK_ID,
                                       eos_id=TASK_DLLM_EOS_ID)
    sample_params = SamplingParams(threshold=TASK_DLLM_THRESHOLD,
                                   cache='prefix',
                                   temperature=0.,
                                   early_stop=True,
                                   cont_weight=0,
                                   prefix_look=0,
                                   after_look=0,
                                   warmup_steps=0,
                                   enable_torch_compile=True,
                                   mask_id=TASK_DLLM_MASK_ID,
                                   eos_id=TASK_DLLM_EOS_ID,
                                   parallel_decoding='threshold',
                                   use_credit=False,
                                   use_bd=True,
                                   max_length=TASK_DLLM_MAX_LENGTH,
                                   ep_size=1,
                                   batch_size=TASK_DLLM_BATCH_SIZE,
                                   mini_batch_size=TASK_DLLM_BATCH_SIZE,
                                   use_naive_batching=True)
    dllm_server = DiffusionLLMServing(SPECIAL_MODEL_DIR,
                                      model_type='llada2-mini',
                                      sample_params=sample_params,
                                      server_port=40570,
                                      num_gpus=TASK_DLLM_NUM_GPUS,
                                      dp_size=1,
                                      tpep_size=TASK_DLLM_NUM_GPUS,
                                      backend='sglang')
    return dllm_tokenizer, dllm_server, decoder


tokenizer, MODEL_DLLM = None, None
global_stream_dict = {}
stream_lock = threading.Lock()


def stream_put_api(chat_uuid: str, curr_x: Optional[List]):
    ''' stream_put_api '''
    with stream_lock:
        if chat_uuid not in global_stream_dict:
            global_stream_dict[chat_uuid] = Queue()

        if curr_x is None:
            global_stream_dict[chat_uuid].put([])
        else:
            global_stream_dict[chat_uuid].put(curr_x)


def stream_get_api(chat_uuid: str) -> Optional[List]:
    ''' stream_get_api '''
    try:
        with stream_lock:
            if chat_uuid not in global_stream_dict:
                return None

            queue = global_stream_dict[chat_uuid]
            result = queue.get(timeout=0)

            if result == []:
                del global_stream_dict[chat_uuid]

            return result

    except Exception:
        return None


def generate_in_background(chat_uuid: str, data: Dict):
    ''' generate_in_background '''

    def _generate():
        try:
            global tokenizer, MODEL_DLLM
            input_ids = tokenizer.apply_chat_template(
                data['messages'],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors='pt',
            )
            batch_input_ids = torch.zeros(
                (input_ids.shape[0], TASK_DLLM_MAX_LENGTH),
                dtype=torch.long).fill_(TASK_DLLM_MASK_ID)
            for s_k in range(input_ids.shape[0]):
                batch_input_ids[s_k, :input_ids.shape[-1]] = input_ids[s_k]

            _ = MODEL_DLLM.generate(batch_input_ids,
                                    chat_uuid=chat_uuid,
                                    gen_length=TASK_DLLM_GEN_LENGTH,
                                    block_length=TASK_DLLM_BLOCK_LENGTH)
            stream_put_api(chat_uuid, None)

        except Exception as err:
            logging.error("Generate error: %s", err)
            stream_put_api(chat_uuid, None)

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()


def get_answer_openai(chat_uuid: str, data: Dict) -> str:
    '''get answer openai'''

    generate_in_background(chat_uuid, data)
    resp = {}

    while True:
        curr_x = stream_get_api(chat_uuid)

        if curr_x is None:
            time.sleep(TASK_STREAM_SLEEP_SECONDS)
            continue

        if curr_x == []:
            break

        x_str = tokenizer.decode(curr_x[0])
        text = x_str.split('<role>ASSISTANT</role>')[-1]
        text = text.replace('<|endoftext|>', '').replace('<|role_end|>', '')
        text = text.replace('<|mask|>', '　').rstrip('　')
        resp = {
            'id':
            chat_uuid,
            'object':
            'chat.completion.chunk',
            'created':
            time.time(),
            'model':
            'xyz-dllm',
            'choices': [{
                'index': 0,
                'delta': {
                    'role': 'assistant',
                    'content': text,
                    'resoning_content': None
                },
                'logprobs': None,
                'finish_reason': None,
            }],
            'prompt_token_ids':
            None
        }
        yield 'data: ' + json.dumps(resp, ensure_ascii=False) + '\n'

    logging.info('[%s] resp: %s', chat_uuid, resp)

    with stream_lock:
        if chat_uuid in global_stream_dict:
            del global_stream_dict[chat_uuid]


def get_answer_openai_no_stream(chat_uuid: str, data: Dict) -> str:
    ''' get answer openai (no stream) '''
    global tokenizer, MODEL_DLLM
    input_ids = tokenizer.apply_chat_template(
        data['messages'],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt',
    )
    batch_input_ids = torch.zeros((input_ids.shape[0], TASK_DLLM_MAX_LENGTH),
                                  dtype=torch.long).fill_(TASK_DLLM_MASK_ID)
    for s_k in range(input_ids.shape[0]):
        batch_input_ids[s_k, :input_ids.shape[-1]] = input_ids[s_k]

    x_tokens_yield = MODEL_DLLM.generate(batch_input_ids,
                                         gen_length=TASK_DLLM_GEN_LENGTH,
                                         block_length=TASK_DLLM_BLOCK_LENGTH)

    resp = {}
    x_tokens_final = x_tokens_yield

    x_str = tokenizer.decode(x_tokens_final[0])
    text = x_str.split('<role>ASSISTANT</role>')[-1]
    text = text.replace('<|endoftext|>', '').replace('<|role_end|>', '')
    text = text.replace('<|mask|>', '　')
    resp = {
        'id':
        chat_uuid,
        'object':
        'chat.completion',
        'created':
        time.time(),
        'model':
        'xyz-dllm',
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': text,
                'resoning_content': None
            },
            'logprobs': None,
            'finish_reason': 'stop',
        }],
        'prompt_token_ids':
        None,
        "usage": {
            "prompt_tokens": input_ids.shape[-1],
            "total_tokens": len(x_tokens_final[0]),
            "completion_tokens": len(x_tokens_final[0]) - input_ids.shape[-1],
            "prompt_tokens_details": None
        }
    }
    logging.info('[%s] resp: %s', chat_uuid, resp)
    return json.dumps(resp, ensure_ascii=False) + '\n'


@app.post('/v1/chat/completions')
def chat_openai(request: CompletionsRequest):
    ''' chat '''
    chat_uuid = f'chat-{str(uuid.uuid4())}'
    data = request.dict()
    logging.info('[%s] req: %s', chat_uuid, data)
    if request.stream:
        return StreamingResponse(get_answer_openai(chat_uuid, data))
    return Response(
        content=get_answer_openai_no_stream(chat_uuid, data),
        media_type='text/plain',
    )


@app.post('/v1/stream_put')
def stream_put_endpoint(request: StreamRequest):
    ''' stream_put_endpoint '''
    stream_put_api(request.chat_uuid, request.curr_x)
    return {"status": "success", "message": "Data added to stream"}


@app.post('/v1/stream_get')
def stream_get_endpoint(request: StreamRequest):
    ''' stream_get_endpoint '''
    curr_x = stream_get_api(request.chat_uuid)
    return curr_x


def mission():
    ''' mission '''
    global tokenizer, MODEL_DLLM
    tokenizer, MODEL_DLLM, _ = get_dllm()
    port = int(os.environ.get('TASK_SERVER_PORT', '40081'))
    uvicorn.run(app, host='0.0.0.0', port=port)


if __name__ == '__main__':
    mission()
