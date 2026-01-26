#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model,
    verify_embeddings,
    embed_text,
    embed_query_text,
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
