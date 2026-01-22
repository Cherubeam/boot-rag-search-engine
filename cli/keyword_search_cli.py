#!/usr/bin/env python3

import argparse

from lib.keyword_search import InvertedIndex, search_command, build_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")


    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            search_results = search_command(args.query)
            for i, result in enumerate(search_results, 1):
                print(f"{i}. ({result['id']}) {result['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            doc_id = args.doc_id
            term = args.term
            idx = InvertedIndex()
            idx.load()
            tf = idx.get_tf(doc_id, term)
            print(f"Term Frequency of '{term}' in document {doc_id}: {tf}")
        case _:
            parser.print_help()
    

if __name__ == "__main__":
    main()