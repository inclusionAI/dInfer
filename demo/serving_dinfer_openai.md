# Demo of dinfer openai serving

## Serving

```bash
export SPECIAL_MODEL_DIR=/models/LLaDA2.0-mini--572899f-C8 # Download from https://www.modelscope.cn/models/inclusionAI/LLaDA2.0-mini
export TASK_DLLM_BATCH_SIZE=2
python3 serving_dinfer_openai.py
```

## Client

```bash
date && curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_API_KEY" -N -d '{"messages": [{"role": "user", "content": "你好， 我是小明"}], "stream": false}' http://0.0.0.0:48000/v1/chat/completions && date
```

## Web demo

```bash
date && docker pull ghcr.io/open-webui/open-webui:main && date

mkdir data-open-webui
cd data-open-webui

date && docker run -d -p 60111:8080 -v $PWD:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main && date
```

Config your open-webui with [http://0.0.0.0:48000/v1](http://0.0.0.0:48000/v1) like [this](https://developer.volcengine.com/articles/7533551308616237092#heading10)
