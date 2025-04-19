import os
import argparse
import xml.etree.ElementTree as ET
from elasticsearch import Elasticsearch
from time import time
from tqdm import tqdm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import sys

es = Elasticsearch("http://localhost:9200")
index_name = "webir_index"

def check_index_exists():
    if not es.indices.exists(index=index_name):
        print(f"Error: Index '{index_name}' does not exist. Please create it in Kibana first.")
        sys.exit(1)

def load_queries(query_file_path):
    tree = ET.parse(query_file_path)
    root = tree.getroot()
    queries = []
    for query in root.findall("QUERY"):
        qid = query.find("ID").text.strip()
        title = query.find("TITLE").text.strip()
        queries.append((qid, title))
    return queries

def search_queries(queries):
    import itertools
    weights = list(itertools.product([1.0, 1.5, 2.0, 2.5, 3.0], repeat=2))
    best_weights = None
    best_avg_precision = 0
    best_results = None
    all_results = []

    for wt_title, wt_body in weights:
        print(f"\nEvaluating with weights - title: {wt_title}, body: {wt_body}")
        first_run_times = []
        second_run_times = []
        results = {}

        for qid, title in tqdm(queries):
            query = {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"title": {"query": title, "boost": wt_title}}},
                            {"match": {"body": {"query": title, "boost": wt_body}}}
                        ]
                    }
                },
                "size": 100
            }

            start1 = time()
            es.search(index=index_name, body=query)
            elapsed1 = time() - start1

            start2 = time()
            response2 = es.search(index=index_name, body=query)
            elapsed2 = time() - start2

            first_run_times.append(elapsed1)
            second_run_times.append(elapsed2)
            results[qid] = [hit["_source"]["docid"] for hit in response2["hits"]["hits"]]

        p20 = precision_at_n(results, 20)
        avg_precision = sum(p20) / len(p20)
        avg_first = sum(first_run_times)/len(first_run_times)
        avg_second = sum(second_run_times)/len(second_run_times)

        print(f"Average Precision@20: {avg_precision:.4f}")
        print(f"Avg First Search Time: {avg_first:.4f}s")
        print(f"Avg Second Search Time: {avg_second:.4f}s")

        all_results.append({
            "title_weight": wt_title,
            "body_weight": wt_body,
            "precision_at_20": avg_precision,
            "avg_first_time": avg_first,
            "avg_second_time": avg_second
        })

        if avg_precision > best_avg_precision:
            best_avg_precision = avg_precision
            best_weights = (wt_title, wt_body)
            best_results = results

    pd.DataFrame(all_results).to_csv("weight_evaluation_results.csv", index=False)

    print(f"\nBest weights based on Precision@20: title={best_weights[0]}, body={best_weights[1]} (Avg P@20={best_avg_precision:.4f})")
    return best_results

def load_relevance(query_id):
    path = f"queries/query-{query_id}.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    relevant_docs = set()
    for doc in root.findall(".//doc"):
        if doc.find("label").text.strip() == "1":
            relevant_docs.add(doc.find("docid").text.strip())
    return relevant_docs

def precision_at_n(results, n=20):
    scores = []
    for qid, docs in results.items():
        relevant = load_relevance(qid)
        retrieved = docs[:n]
        correct = len([doc for doc in retrieved if doc in relevant])
        scores.append(correct / n)
    return scores

def plot_precisions(all_pns):
    df = pd.DataFrame(all_pns)
    df.index = [f"Query {i+1}" for i in range(len(df))]
    df.plot(figsize=(12, 6), title="Precision at N per Query", marker='o')
    plt.xlabel("Query")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.show()
    df.mean().plot(kind='bar', title="Average Precision@N")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate search engine performance")
    parser.add_argument("--queries", type=str, default="queries/query.xml", help="Path to query file")
    args = parser.parse_args()

    check_index_exists()
    queries = load_queries(args.queries)
    best_results = search_queries(queries)
    p20 = precision_at_n(best_results, 20)
    p50 = precision_at_n(best_results, 50)
    p100 = precision_at_n(best_results, 100)
    plot_precisions({"p@20": p20, "p@50": p50, "p@100": p100})
