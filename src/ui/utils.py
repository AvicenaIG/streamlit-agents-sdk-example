"""
This module provides utility functions for the Streamlit UI of the multi-agent system.

It implements various helper functions for session state management, LLM client configuration,
response streaming, and message formatting.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, AsyncGenerator, Optional, Tuple

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

#------------------------------------------------------------------------------
# UI RENDERING FUNCTIONS
#------------------------------------------------------------------------------

def load_css() -> None:
    """Load external CSS file."""
    css_file = Path(__file__).parent / "static" / "styles.css"
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_header() -> None:
    """Render the application header with logo and title."""
    st.markdown(
        '<div class="main-header">'
        '<div class="main-header-content">'
        '<img src="https://emoji.aranja.com/static/emoji-data/img-apple-160/1f916.png" '
        'class="main-header-image">'
        '<div>'
        '<h1>✨ Talk to the Bots</h1>'
        '<p>An agentic, multi-provider chatbot with character personalities</p>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

def message_with_feedback(message: Dict, index: int) -> None:
    """Display a single message with feedback if it's from the assistant."""
    with st.chat_message(message["role"], avatar=message.get("emoji", "🤖")):
        # If the message has steps, display them in an expander
        if "steps" in message and message["steps"] and st.session_state.get("show_thinking", False):
            with st.expander("Steps", expanded=False):
                st.write("\n".join(message["steps"]))
        
        # Display the message content
        st.markdown(message["content"])

        # Show feedback for assistant messages
        if message["role"] == "assistant":
            feedback = st.feedback(
                options="thumbs",
                key=f"feedback_{index}"
            )
            if feedback:
                st.session_state["messages"][index]["feedback"] = feedback

def display_chat_history() -> None:
    """Displays the chat in Streamlit."""
    for i, message in enumerate(st.session_state["messages"]):
        message_with_feedback(message, i)

#------------------------------------------------------------------------------
# CONFIGURATION AND SESSION STATE MANAGEMENT
#------------------------------------------------------------------------------

def setup_sidebar() -> None:
    """Sets up the sidebar configuration."""
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('<hr class="sidebar-hr">', unsafe_allow_html=True)

        # Configure provider selection
        _configure_provider_selection()
        
        st.markdown('<hr class="sidebar-hr">', unsafe_allow_html=True)
        
        # Configure feature toggles
        _configure_feature_toggles()
        
        st.markdown('<hr class="sidebar-hr">', unsafe_allow_html=True)
        
        # About section
        _display_about_section()
        
        st.markdown("</div>", unsafe_allow_html=True)

def _configure_provider_selection() -> None:
    """Configure provider selection dropdown and setup LLM client."""
    # Get provider configuration
    active_providers = st.secrets["providers"]["active"]
    labels = st.secrets["provider_labels"]
    
    # Add heading
    st.markdown('<h3 class="provider-heading">Select LLM Provider</h3>', unsafe_allow_html=True)
    
    # Create provider selection dropdown
    selected_provider = st.selectbox(
        label="provider",
        label_visibility="hidden",
        options=active_providers,
        index=0,
        format_func=lambda x: labels[x],
        key="provider_select",
    )

    # Configure the selected LLM client
    configure_llm_client(
        selected_provider, 
        selected_provider_label=labels[selected_provider]
    )

def _configure_feature_toggles() -> None:
    """Configure feature toggles in the sidebar."""
    st.markdown('<h3 class="sidebar-heading">Features</h3>', unsafe_allow_html=True)
    
    # Toggle for showing thinking process
    st.toggle(
        "Show Thinking Process", 
        value=st.session_state.get("show_thinking", True), 
        key="show_thinking"
    )
    
    # Configure streaming toggle based on provider support
    selected_provider = st.session_state.get("provider_select")
    provider_supports_streaming = get_streaming_status(selected_provider)
    
    # Toggle for response streaming
    st.toggle(
        "Use Response Streaming", 
        value=provider_supports_streaming,
        disabled=not provider_supports_streaming,
        key="use_streaming"
    )
    
    # Add note for disabled streaming
    if not provider_supports_streaming:
        provider_label = st.secrets["provider_labels"][selected_provider]
        st.caption(f"Note: disabled streaming for {provider_label}")

def _display_about_section() -> None:
    """Display the about section in the sidebar."""
    st.markdown('<h3 class="sidebar-heading">About</h3>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card">'
        'This is a demo of a multi-agent chat system powered by different LLM providers.'
        '</div>',
        unsafe_allow_html=True
    )

def get_streaming_status(provider: str) -> bool:
    """
    Determine if streaming is supported for a given provider.
    
    Args:
        provider: The LLM provider name
        
    Returns:
        Boolean indicating if streaming is supported
    """
    non_streaming_providers = {
        "anthropic": False,
    }
    return non_streaming_providers.get(provider, True)

def initialize_session_state() -> None:
    """Initialize all session state variables used in the app."""
    defaults = {
        "messages": [],
        "llm_provider": None,
        "model": None,
    }
    
    # Set default values for any uninitialized state
    for key, default_value in defaults.items():
        st.session_state.setdefault(key, default_value)

def configure_llm_client(selected_provider: str, selected_provider_label: str) -> None:
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

#------------------------------------------------------------------------------
# AGENT INTERACTION UTILS
#------------------------------------------------------------------------------

def get_emojis(provider: str) -> Tuple[str, str]:
    """Set the emojis for the user and agent based on the provider."""
    emojis = {
        "openai": ("🐱", "🦖"),
        "xai": ("🐱", "👽"),
        "huggingface": ("🐱", "🤗"),
        "anthropic": ("🐱", "👼"),
    }
    # Default to cat and robot emojis
    return emojis.get(provider, ("🐱", "🤖"))


def get_conversation_history_for_agent(messages: List[Dict]) -> List[Dict]:
    """
    Prepares a clean conversation history for LLM input by stripping out
    non-relevant metadata (like emojis).
    """
    return [
        {"role": msg.get("role", "user"), "content": msg.get("content", "")}
        for msg in messages
        if "content" in msg
    ]

#------------------------------------------------------------------------------
# RESPONSE STREAMING AND PROCESSING
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
# UI RESPONSE RENDERING
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
# CHAT INTERACTION HANDLER
#------------------------------------------------------------------------------

def handle_chat_interaction(agent: Agent) -> None:
    """Handles chat interaction with streaming/non-streaming options."""
    provider = st.session_state.get("llm_provider")
    user_emoji, agent_emoji = get_emojis(provider)
    
    # Get streaming preference from session state
    use_streaming = st.session_state.get("use_streaming", True)

    if prompt := st.chat_input("Ask me anything...", key="chat_input"):
        # Save user message
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt,
            "emoji": user_emoji
        })
        
        with st.chat_message("user", avatar=user_emoji):
            st.markdown(prompt)

        conversation_history = get_conversation_history_for_agent(st.session_state["messages"])
        print(f"[debug - conversation_history]: {conversation_history}")
        print(f"[debug - model]: {st.session_state['model']}")
        # Get and render response
        if use_streaming:
            generator = generate_response_stream(agent, conversation_history)
            response = render_streaming_response(generator, agent_emoji)
        else:
            with st.spinner("Thinking..."):
                response = asyncio.run(get_agent_response(agent, conversation_history, stream=False))
            response = render_static_response(response, agent_emoji)

        # Save assistant response
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response["response"],
            "emoji": agent_emoji,
            "steps": response["steps"]
        })
        # Force rerun to update the chat display with feedback
        st.rerun()