"""
This module provides utility functions for the Streamlit UI of the multi-agent system.

It implements various helper functions for session state management, LLM client configuration,
response streaming, and message formatting.
"""

import asyncio

import streamlit as st
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    Runner,
    set_default_openai_client,
    set_tracing_disabled,
    set_default_openai_api,
    HandoffOutputItem
)

from typing import Dict, List, AsyncGenerator

def initialize_session_state():
    """Initialize all session state variables used in the app."""
    defaults = {
        "messages": [],
        "llm_provider": None,
        "model": None,
    }
    
    # Set default values for any uninitialized state
    for key, default_value in defaults.items():
        st.session_state.setdefault(key, default_value)

def configure_llm_client(selected_provider: str, selected_provider_label: str):
    """Configures the LLM client based on the selected provider."""

    # Load provider config
    provider_config = st.secrets.get(selected_provider)
    
    # Get API key: instead of using an .env file, we get it from the user/.secrets
    api_key = provider_config.get("api_key") or st.sidebar.text_input(
        f"Enter {selected_provider_label} API Key", type="password"
    )
    if not api_key:
        st.sidebar.error(f"Please enter your {selected_provider_label} API key.")
        st.stop()

    # Instantiate LLM client that is compatible with OpenAI API
    custom_client = AsyncOpenAI(
        api_key=api_key, base_url=provider_config.get("base_url")
    )
    set_default_openai_client(custom_client)

    # If provider is not OpenAI, turn off tracing which is not supported
    if selected_provider != "openai":
        set_default_openai_api(
            "chat_completions"
        )  # most providers dont support responses API yet
        set_tracing_disabled(disabled=True)

    # Store selected provider and model in session state
    st.session_state["llm_provider"] = selected_provider
    st.session_state["model"] = provider_config.get("model")
    st.sidebar.success(f"Configured {selected_provider_label} client!")

def get_emojis(provider: str):
    """Set the emojis for the user and agent based on the provider."""
    emojis = {
        "openai": ("🐱", "🦖"),
        "xai": ("🐱", "👽"),
        "huggingface": ("🐱", "🤗"),
        "anthropic": ("🐱", "👼"),
    }
    # Default to cat and robot emojis
    return emojis.get(provider, ("🐱", "🤖"))


def get_conversation_history_for_agent(messages: list[dict]) -> list[dict]:
    """
    Prepares a clean conversation history for LLM input by stripping out
    non-relevant metadata (like emojis).
    """
    return [
        {"role": msg.get("role", "user"), "content": msg.get("content", "")}
        for msg in messages
        if "content" in msg
    ]

async def generate_response_stream(agent: Agent, prompt: str) -> AsyncGenerator[Dict, None]:
    """Yields token deltas and agent handover updates."""
    result = Runner.run_streamed(agent, input=prompt)
    current_agent = agent.name

    async for event in result.stream_events():
        
        if event.type == "agent_updated_stream_event":
            new_agent = event.new_agent.name
            if new_agent != current_agent:
                print(f"[agent handover]: {current_agent} -> {new_agent}")
                yield {"step": f"🔄 Handover **{current_agent}** -> **{new_agent}**"}
                current_agent = new_agent
        elif event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            yield {"delta": event.data.delta}

def process_handoffs(result) -> List[str]:
    """Processes handoff events from HandoffOutputItem and returns steps list."""
    steps = []
    for item in result.new_items:
        if isinstance(item, HandoffOutputItem):
            steps.append(f"🔄 Handover **{item.source_agent.name}** -> **{item.target_agent.name}**")
    return steps

async def get_agent_response(agent: Agent, prompt: str, stream: bool = False) -> Dict:
    """Unified function to get agent response with optional streaming."""
    if stream:
        generator = generate_response_stream(agent, prompt)
        full_response = ""
        steps = []
        
        async for chunk in generator:
            if "delta" in chunk:
                full_response += chunk["delta"]
            elif "step" in chunk:
                steps.append(chunk["step"])
        
        return {"response": full_response, "steps": steps}
    else:
        result = await Runner.run(agent, input=prompt)
        steps = process_handoffs(result)
        return {"response": result.final_output, "steps": steps}

def render_streaming_response(generator, agent_emoji: str) -> Dict:
    """Renders streaming content to Streamlit."""
    with st.chat_message("assistant", avatar=agent_emoji):
        steps_expander = st.expander("Steps")
        message_container = st.empty()
        
        full_response = ""
        steps = []
        
        async def stream_response():
            nonlocal full_response, steps
            async for chunk in generator:
                if "delta" in chunk:
                    full_response += chunk["delta"]
                    message_container.markdown(full_response)
                elif "step" in chunk:
                    steps.append(chunk["step"])
                    steps_expander.markdown("\n".join(steps))
            return {"response": full_response, "steps": steps}
        
        # Streamlit requires running the async function in a thread
        final_result = asyncio.run(stream_response())
        return final_result

def render_static_response(response: Dict, agent_emoji: str) -> Dict:
    """Renders static response to Streamlit."""
    with st.chat_message("assistant", avatar=agent_emoji):
        st.markdown(response["response"])
        if response["steps"]:
            with st.expander("Steps"):
                st.markdown("\n".join(response["steps"]))
    return response