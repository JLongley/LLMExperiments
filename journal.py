import argparse
import logging
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import openai

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

LOGSEQ_DIR = os.getenv("LOGSEQ_DIR")
openai.api_key = os.getenv("OPENAI_API_KEY")

if Path("./storage").exists():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
else:
    journal_documents = SimpleDirectoryReader(LOGSEQ_DIR + "/journals").load_data()
    page_documents = SimpleDirectoryReader(LOGSEQ_DIR + "/pages").load_data()
    documents = journal_documents + page_documents
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir="./storage")


if __name__ == "__main__":
    """
    Usage: python journal.py -q "What did I do in May 2023?"
    """
    query_engine = index.as_query_engine()
    # cli argument parser
    parser = argparse.ArgumentParser(
        prog="QueryJournal",
        description="Query my LogSeq journal using Llama Index.",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Ask a question answerable in my journal",
        required=True,
    )
    args = parser.parse_args()
    query = args.query

    if query:
        res = query_engine.query(query)
        print(f"Query: {query}")
        print(f"Results: \n {res}")
    else:
        print("No query provided. Exiting...")
        exit(0)
