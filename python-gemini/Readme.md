# Google AI with Python


## SETUP
- Get an API key
- `docker build -t shreyasgune/gman-py-ai:0.0.1 . `
- `docker run -it --env GEMINI_API_KEY=$GEMINI_API_KEY  -v $(pwd):/app shreyasgune/gman-py-ai:0.0.1 bash`
- `python test.py "what is the capital of India?"`

```
response:
GenerateContentResponse(
    done=True,
    iterator=None,
    result=protos.GenerateContentResponse({
      "candidates": [
        {
          "content": {
            "parts": [
              {
                "text": "The capital of India is **New Delhi**.\n"
              }
            ],
            "role": "model"
          },
          "finish_reason": "STOP",
          "avg_logprobs": -0.005283908173441887
        }
      ],
      "usage_metadata": {
        "prompt_token_count": 9,
        "candidates_token_count": 10,
        "total_token_count": 19
      }
    }),
)
```