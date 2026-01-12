# Demo of dinfer openai serving

## Serving

```bash
python3 serving_dinfer_openai.py
```

## Client

```bash
date && curl -X POST -H "Content-Type: application/json" -H "Authorization: Bearer 12345678" -N -d '{"messages": [{"role": "user", "content": "你好， 我是小明"}], "stream": false}' http://0.0.0.0:48000/v1/chat/completions && date
```
