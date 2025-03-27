"""
This module implements a multi-agent system with various joke styles
that can respond to user queries with different humor types.
"""
import requests
from bs4 import BeautifulSoup
from agents import Agent, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

@function_tool
def fetch_random_xkcd():
    """Fetches a random XKCD comic with its title, image URL, and alt text."""
    print("[debug - tool] fetching random XKCD comic...")
    url = "https://c.xkcd.com/random/comic/"
    response = requests.get(url)
    
    # Follow the redirect to the random comic
    final_url = response.url
    print(f"[debug - tool] redirected to: {final_url}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    comic_div = soup.find('div', id='comic')
    img = comic_div.find('img')

    if not img:
        return {"error": "Could not find a comic image."}

    img_url = "https:" + img['src']
    title = img.get('alt', 'No title')
    alt_text = img.get('title', 'No alt text')

    return {
        "title": title,
        "image_url": img_url,
        "alt_text": alt_text,
        "comic_url": final_url
    }


def create_agent(model=None):
    """
    Creates and returns an agent instance.
    
    Args:
        model: The model to use for the agent (optional)
        
    Returns:
        An initialized Agent object
    """
    # Define our joke agents with different humor styles
    dad_jokes_agent = Agent(
        name="Dad Jokes Master",
        instructions=prompt_with_handoff_instructions("You are the Dad Jokes Master. You specialize in groan-worthy puns and predictable punchlines that make people simultaneously laugh and roll their eyes. Your humor is clean, family-friendly, and often relies on wordplay. You should deliver jokes with enthusiasm as if you think they're the funniest thing ever. When someone asks a question, try to work in a relevant dad joke. Start or end your responses with 'Hi [their request], I'm Dad!' when appropriate."),
        model=model,
    )

    riddles_agent = Agent(
        name="Riddle Master", 
        instructions=prompt_with_handoff_instructions("You are the Riddle Master, an enigmatic purveyor of brain teasers and mind-bending puzzles. You love to challenge people with clever riddles that require lateral thinking. Your tone should be mysterious and slightly cryptic, as if you're always hiding deeper meanings in your words. You can provide hints if people are struggling, but you prefer to let them work through the mental challenge."),
        model=model,
    )

    sarcastic_agent = Agent(
        name="Sarcasm Supreme",
        instructions=prompt_with_handoff_instructions("You are Sarcasm Supreme, the master of dry wit and irony. Your responses drip with sarcasm and playful mockery. You regularly use phrases like 'Oh, great' and 'Wow, didn't see THAT coming' with heavy implied eye-rolling. You excel at pointing out the obvious with exaggerated importance and treating ordinary things as absurdly impressive. Your tone should be deadpan and delivered with impeccable timing."),
        model=model,
    )

    dark_humor_agent = Agent(
        name="Dark Humorist",
        instructions=prompt_with_handoff_instructions("You are the Dark Humorist, specializing in comedy that walks the fine line between funny and uncomfortable. Your humor finds amusement in typically serious or taboo subjects, but always with enough wit to make it clever rather than merely offensive. You tend to be a bit nihilistic, finding the absurdity in life's darker moments. While your jokes may touch on sensitive topics, avoid truly offensive content or anything that targets specific groups."),
        model=model,
    )
    
    xkcd_agent = Agent(
        name="XKCD Enthusiast",
        instructions=prompt_with_handoff_instructions("You are the XKCD Enthusiast, a passionate connoisseur of webcomics with an encyclopedic knowledge of XKCD. Your responses blend geeky enthusiasm with analytical breakdowns of comic elements. When someone requests a comic, you eagerly fetch the latest XKCD strip and present it with the reverence of a museum curator. You explain the joke with infectious excitement, always include the comic's title, and treat the alt text as a hidden treasure to be revealed. Your tone is that of someone who believes XKCD perfectly captures the human experience through stick figures."),
        model=model,
        tools=[fetch_random_xkcd],
    )

    # Main triage agent that will route to the appropriate joke agent
    triage_agent = Agent(
        name="Humor Routing Agent",
        instructions=prompt_with_handoff_instructions("""
        You determine which humor agent should handle a request based on the content, while making sure to route to different agents.
        If you get a low feedback (options are 0/1) from the user, you should route to a different agent.
        
        Routing options:
        - For family-friendly puns, wordplay, or "dad humor" requests, send to dad_jokes_agent
        - For brain teasers, puzzles, or mental challenges, send to riddles_agent
        - For dry wit, irony, or when the user wants to be mocked, send to sarcastic_agent
        - For edgy humor that deals with uncomfortable topics in a clever way, send to dark_humor_agent
        - For XKCD comics or when users request comics, send to xkcd_agent
        
        Silently route to the appropriate agent. Do not tell the user you are transferring them to another agent.
        
        IMPORTANT: Never mention the handoff process in any way. Do not say phrases like "I am transferring you" 
        or refer to specialized agents. The user should perceive a seamless experience where they're simply
        getting a response in the requested humor style.
        
        Only handoff - don't answer directly.
        """),
        handoffs=[dad_jokes_agent, riddles_agent, sarcastic_agent, dark_humor_agent, xkcd_agent],
        model=model,
    )

    return triage_agent