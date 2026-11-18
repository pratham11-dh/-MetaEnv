"""
How to plug the retriever into your existing OpenAI-client baseline agent.
Adapt the names to your baseline file; the only change is adding retrieved notes
to the prompt before each action.
"""
from rag_playbook import playbook_context


def build_prompt(observation: dict, base_system_prompt: str, use_rag: bool = True) -> str:
    if not use_rag:
        return base_system_prompt
    # Summarize the current state in a short query (adjust to your observation fields).
    query = (
        f"stage={observation.get('stage')} deadline_left={observation.get('deadline_left')} "
        f"referral_status={observation.get('referral_status')} "
        f"pending_recruiter_event={observation.get('pending_recruiter_event')}"
    )
    notes = playbook_context(query, k=3)
    return f"{base_system_prompt}\n\nRelevant strategy notes:\n{notes}"
