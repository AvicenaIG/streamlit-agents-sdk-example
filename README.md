# Steamlit AI Agent

A flexible **Streamlit** app that integrates with **OpenAI's Agents SDK** to power a conversational AI agent with tools, multi-agent, and multi-LLM support.

Supports Anthropic, Hugging Face, OpenAI, and xAI (Grok) — experiment with different providers on the fly.

## Project Structure

```
streamlit-agent-ai/
│── src/                           
│   │── ui/                             
│   │   │── streamlit_app.py            # Streamlit UI for interaction
│   │   │── utils.py                    # helper functions
│   │── agent/
│   │   │── agent.py                    # multi-agent example
│── requirements.txt                
│── README.md                        
│── .gitignore                        

```

## Steps to set-up and run locally:

1. Create Virtual Environments:
```
python -m venv .venv  # Create a virtual environment named .venv
source .venv/bin/activate  # Activate on Mac/Linux
.venv\Scripts\activate     # Activate on Windows
pip install -r requirements.txt  # Install dependencies
```

2. Copy local secrets file

Run this in your terminal:
```cp .streamlit/secrets.example.toml .streamlit/secrets.toml```

3. Start Streamlit 

```
python -m streamlit run src/ui/steamlit_app.py
```

4. Define LLM API Key(s)

a. Get API keys for each provider you would like to use:

- Anthropic: https://console.anthropic.com/settings/keys

- Hugging Face: https://huggingface.co/settings/tokens  
    Note: You must also set up an Inference Endpoint to use with your token ([guide](https://huggingface.co/docs/inference-endpoints/en/guides/create_endpoint))

- OpenAI: https://platform.openai.com/account/api-keys

- xAI (Grok via api.x.ai): https://x.ai (Log in and access your developer console)

b. [Optional] Update .streamlit/secrets.toml with your API keys.  
For Hugging Face you must also update the base_url with your endpoint URL. Alternatively, you can enter keys in the streamlit UI.

## License

Do whatever you want with it!