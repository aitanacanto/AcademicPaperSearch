# Reproducibility and evaluation scope

## Historical experiment and new local corpora

The notebook preserves outputs from the academic run. The original report documents 630 collected papers, 615 indexed documents, 9,732 terms and 12 evaluation queries. These are historical observations, not a fresh benchmark.

The original corpus and relevance-annotation snapshot have not been recovered. Newly generated files may now exist locally under `notebooks/data/` or `data/`; they are excluded from Git. Collection against OpenAlex can return different papers over time. Preserve a fixed corpus and relevance judgments before comparing future experiments.

## Local interface

`app.py` is a later interface addition. It rebuilds an in-memory index from the notebook's raw or processed JSONL corpus. The Boolean parser and TF-IDF retrieval functions are preserved from the notebook. Tests with the locally generated corpus confirm matching postings, document frequencies, ranked document IDs and TF-IDF scores.

The new screenshot illustrates that local corpus. Its counts should not be substituted for the original report's dataset or used to claim reproduction of the historical metrics.

## Original metrics

Precision@10 describes manually judged top-10 results. Only those results are annotated. The original code sets the recall denominator to the number of relevant documents in the same retrieved list; recall becomes 1 whenever at least one result is relevant. It does not measure coverage of relevant documents in the corpus. F1 inherits this limitation.

Average precision is normalised by the relevant documents found within the judged top 10. Treat AP/MAP as a restricted, within-list ranking analysis, not a complete corpus-level retrieval benchmark.

No new relevance labels or evaluation metrics have been invented. BM25, embeddings and neural reranking remain possible extensions, not implemented features.
