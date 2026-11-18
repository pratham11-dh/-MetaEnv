"""
Retrieval-augmented playbook for the MetaEnv baseline agent.

Builds a Chroma vector store from the markdown files in ./playbook using LangChain,
then retrieves the most relevant strategy notes for the agent's current observation.

Usage:
  python rag_playbook.py --build            # index the playbook (needs OPENAI_API_KEY)
  python rag_playbook.py --query "referral ghosted, deadline close"
  python rag_playbook.py --build --fake --query "recruiter event"   # offline smoke test
"""
import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PLAYBOOK_DIR = Path(__file__).parent / "playbook"
PERSIST_DIR = str(Path(__file__).parent / ".chroma")
COLLECTION = "metaenv_playbook"


def get_embeddings(fake: bool = False):
    if fake:
        from langchain_core.embeddings import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=256)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def load_documents():
    docs = []
    for path in sorted(PLAYBOOK_DIR.glob("*.md")):
        docs.append(Document(page_content=path.read_text(), metadata={"source": path.name}))
    return docs


def build_index(fake: bool = False) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(load_documents())
    return Chroma.from_documents(
        chunks,
        embedding=get_embeddings(fake),
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
    )


def get_retriever(k: int = 3, fake: bool = False):
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(fake),
        persist_directory=PERSIST_DIR,
    )
    return store.as_retriever(search_kwargs={"k": k})


def playbook_context(observation_summary: str, k: int = 3, fake: bool = False) -> str:
    """Return retrieved notes formatted for insertion into the agent prompt."""
    notes = get_retriever(k, fake).invoke(observation_summary)
    return "\n".join(f"- ({d.metadata.get('source')}) {d.page_content.strip()}" for d in notes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", type=str)
    ap.add_argument("--fake", action="store_true", help="offline fake embeddings (smoke test only)")
    args = ap.parse_args()
    if args.build:
        build_index(args.fake)
        print("Index built.")
    if args.query:
        print(playbook_context(args.query, fake=args.fake))
