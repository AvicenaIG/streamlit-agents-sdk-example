"""
This module implements a Streamlit UI for a multi-agent chat system.
It handles the web interface, chat display, and agent responses.
"""
import asyncio

import streamlit as st
from agents import Agent

from src.ui.utils import (
    initialize_session_state,
    configure_llm_client,
    get_emojis,
    get_conversation_history_for_agent,
    generate_response_stream,
    render_streaming_response,
    render_static_response,
    get_agent_response
)

from src.agent.agent import create_agent


def setup_sidebar():
    """Sets up the sidebar configuration."""
    st.sidebar.title("Configuration")

    # Dropdown to select provider
    active_providers = st.secrets["providers"]["active"]
    labels = st.secrets["provider_labels"]

    selected_provider = st.sidebar.selectbox(
        "Select LLM Provider",
        options=active_providers,
        index=0,
        format_func=lambda x: labels[x],
        key="provider_select",
    )

    # LLM Provider Configuration
    configure_llm_client(
        selected_provider, selected_provider_label=labels[selected_provider]
    )


def message_with_feedback(message: dict, index: int) -> None:
    """Display a single message with feedback if it's from the assistant."""
    with st.chat_message(message["role"], avatar=message.get("emoji", "🤖")):
        # If the message has steps, display them in an expander
        if "steps" in message and message["steps"]:
            with st.expander("Steps", expanded=False):
                st.markdown("\n".join(message["steps"]))
        
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


def display_chat_history():
    """Displays the chat in Streamlit."""
    for i, message in enumerate(st.session_state["messages"]):
        message_with_feedback(message, i)

def handle_chat(agent: Agent, use_streaming: bool = True):
    """Handles chat with streaming/non-streaming option."""
    provider = st.session_state.get("llm_provider")
    user_emoji, agent_emoji = get_emojis(provider)
    
    # temp: providers that do not currently work with opena-agents-sdk stream_events()
    non_streaming_providers = {
        "anthropic": False,
    }
    
    # Override use_streaming based on provider compatibility
    use_streaming = use_streaming and non_streaming_providers.get(provider, True)

    if prompt := st.chat_input("What's up?"):
        # Save user message
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt,
            "emoji": user_emoji
        })
        
        with st.chat_message("user", avatar=user_emoji):
            st.markdown(prompt)

        conversation_history = get_conversation_history_for_agent(st.session_state["messages"])
        
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


def main():
    """Main function to run the Streamlit app."""
    st.title("🤖✨ Talk to the Bots")
    st.write(
        "<p style='font-size: 18px; color: gray;'>An agentic, multi-provider chatbot with animated characters personalities</p>",
        unsafe_allow_html=True
    )
    
    setup_sidebar()
    initialize_session_state()
    display_chat_history()
    
    # Create the agent and manage chat with model passed as an argument
    model = st.session_state.get("model")
    agent = create_agent(model=model)
    handle_chat(agent)


if __name__ == "__main__":
    main()