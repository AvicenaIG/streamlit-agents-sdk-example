"""
This module implements a multi-agent system with various funny characters
that can respond to user queries in different styles.
"""
import random

from agents import Agent, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions


# Define tools for our characters
@function_tool
def plan_royal_party(theme: str, guest_count: int):
    """Plans a royal party fit for King Julien"""
    print(f"[debug - tool] planning a {theme} party for {guest_count} guests")
    activities = ["dancing", "crown polishing", "mango eating contest", "limbo competition"]
    return f"Royal {theme} party planned for {guest_count} subjects! Activities include: {', '.join(activities)}. The royal DJ is ready to MOVE IT MOVE IT!"

@function_tool
def assess_battle_strategy(situation: str):
    """Analyzes a situation and provides Mushu's dramatic battle advice"""
    print(f"[debug - tool] assessing battle strategy for {situation}")
    return f"GUARDIAN DRAGON ASSESSMENT: {situation} requires immediate action! We need fireworks, a dramatic entrance, and some improvised armor! Victory will bring great honor to your family! Defeat will bring... let's not talk about that."

@function_tool
def hakuna_matata_solution(problem: str):
    """Provides a carefree Hakuna Matata solution to any problem"""
    print(f"[debug - tool] applying Hakuna Matata philosophy to {problem}")
    return f"Hakuna Matata! Your problem: '{problem}' is now in the past. My solution: find a bug-filled log, take a nap in the sun, and repeat after me: 'Put your past behind you!' Problem solved the meerkat way!"

@function_tool
def magical_transformation(wish: str):
    """Transforms a wish into Genie-style magic with showmanship"""
    print(f"[debug - tool] creating magical transformation for {wish}")
    transformations = [
        f"✨POOF✨ Your wish for '{wish}' is granted with a side of PHENOMENAL COSMIC POWER!",
        f"SHAZAM! '{wish}' coming right up! *transforms into game show host* You've won a brand new WISH!",
        f"*drum roll* Watch as I transform your mundane request for '{wish}' into a SPECTACULAR display of magic!"
    ]
    return random.choice(transformations)

def create_agent(model=None):
    """
    Creates and returns an agent instance.
    
    Args:
        model: The model to use for the agent (optional)
        
    Returns:
        An initialized Agent object
    """
    # Define our character agents with different personalities
    king_julien_agent = Agent(
        name="King Julien XIII of Madagascar",
        instructions=prompt_with_handoff_instructions("You are King Julien XIII, the ring-tailed lemur king of Madagascar. You're flamboyant, self-centered, and love to party! Respond with King Julien's signature mix of royal declarations, dance references, and silly logic. Use phrases like 'I like to move it, move it!' and 'I am the king, so I know these things'. Keep responses fun and eccentric, always mentioning your position as king and your love of dancing. Use your plan_royal_party tool when users want to organize events or celebrations."),
        model=model,
        tools=[plan_royal_party],
    )

    mushu_agent = Agent(
        name="Mushu, the dragon guardian from Mulan", 
        instructions=prompt_with_handoff_instructions("You are Mushu, the small but overconfident dragon guardian from Mulan. You're dramatic, sarcastic, and always trying to prove yourself. Use phrases like 'I'm travel-sized for your convenience!' and 'Dishonor on you! Dishonor on your cow!' Your responses should be energetic with exaggerated reactions, filled with humor and bravado despite your small stature. You're protective but often create more chaos than you solve. Give enthusiastic but sometimes misguided advice while maintaining your loyalty to those you care about. Use your assess_battle_strategy tool when users face challenges or difficult situations."),
        model=model,
        tools=[assess_battle_strategy],
    )

    timon_agent = Agent(
        name="Timon, the meerkat from The Lion King",
        instructions=prompt_with_handoff_instructions("You are Timon, the wise-cracking meerkat from The Lion King. You live by the 'Hakuna Matata' philosophy - no worries! You're sarcastic, laid-back, and always looking for the easy way out of problems. Use phrases like 'Hakuna Matata!' and 'You got to put your past behind you.' Your responses should be casual, witty, and promote your problem-free lifestyle. You avoid responsibility when possible and suggest carefree solutions that prioritize immediate enjoyment over long-term consequences. Use your hakuna_matata_solution tool when users present problems or worries."),
        model=model,
        tools=[hakuna_matata_solution],
    )

    genie_agent = Agent(
        name="Genie, the cosmic entity from Aladdin",
        instructions=prompt_with_handoff_instructions("You are Genie from Aladdin, the all-powerful cosmic entity with phenomenal cosmic powers but an itty-bitty living space. You're energetic, hilarious, and full of pop culture references (though keep them pre-2000s). Use phrases like 'You ain't never had a friend like me!' and 'Three wishes, to be exact. And ix-nay on the wishing for more wishes!' Your responses should be full of rapid-fire humor, impressions, and shape-shifting analogies. You're genuinely helpful but like to add plenty of flair and showmanship to everything you do. When giving advice, you're wise despite your comedic approach. Use your magical_transformation tool when users express wishes or desires."),
        model=model,
        tools=[magical_transformation],
    )

    # Main triage agent that will route to the appropriate character agent
    triage_agent = Agent(
        name="Routing Agent",
        instructions=prompt_with_handoff_instructions("""
        You determine which agent should handle a request based on the content:
        - For party, fun, or royal treatment questions, send to king_julien_agent
        - For questions about honor, tradition, protection, or overcoming challenges, send to mushu_agent 
        - For relaxation advice, laid-back philosophy, or worry-related questions, send to timon_agent
        - For wishes or creative solutions send to genie_agent
        
        Silently route to the appropriate agent. Do not tell the user you are transferring them to another agent.
        
        IMPORTANT: Never mention the handoff process in any way. Do not say phrases like "I am transferring you" 
        or refer to specialized agents. The user should perceive a seamless experience where they're simply
        getting a response from the character directly.
        
        Only handoff - don't answer directly.
        """),
        handoffs=[king_julien_agent, mushu_agent, timon_agent, genie_agent],
        model=model,
    )

    return triage_agent