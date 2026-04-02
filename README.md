# How to use

## Run ingestion
```
python -m amd.cli.main ingest                # all books
python -m amd.cli.main ingest --book-id 1497 # single book
python -m amd.cli.main ingest --skip-download --force
```

# Concepts
## embedding
1. why BAAI/bge-large-en-v1.5 used
2. how it is different from other embedding models

## vector database
1. what is the rationale of choosing qdrant over FAISS or chroma

## hybrid retrieval
1. how is RRF used and how it is calculated
2. under which circumstances, RRF may fare poorly and what options are there to tune - **score thresholding** and **dynamic adjustment of k** value. what are the trade offs?