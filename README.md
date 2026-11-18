# MetaEnv RAG playbook (LangChain + Chroma)

1. Copy this folder into the MetaEnv repo (e.g. `agent/rag/`).
2. `pip install -r requirements.txt`, set `OPENAI_API_KEY`.
3. Smoke test offline: `python rag_playbook.py --build --fake --query "recruiter event"`
4. Real index: `python rag_playbook.py --build`
5. In your baseline agent, call `build_prompt(...)` from `baseline_integration.py`
   (adjust the observation field names to match your 18-field observation space).
6. Run your baseline on easy / medium / hard WITH and WITHOUT `use_rag` and record both
   score sets. Only report numbers you actually measured.
7. Commit and push, then update the resume.
