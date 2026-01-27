#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    SemanticSearch,
    verify_model,
    verify_embeddings,
    embed_text,
    embed_query_text,
    semantic_search,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model is loaded")

    subparsers.add_parser(
        "verify_embeddings", help="Verify that embeddings can be generated"
    )

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for given text"
    )
    single_embed_parser.add_argument("text", help="Text to embed")

    query_embed_parser = subparsers.add_parser(
        "embedquery", help="Generate embedding for given search query"
    )
    query_embed_parser.add_argument("query", help="Query to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search documents using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.text)
        case "search":
            semantic_search(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
