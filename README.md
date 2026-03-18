# How to use

## Run ingestion
```
python -m amd.cli.main ingest                # all books
python -m amd.cli.main ingest --book-id 1497 # single book
python -m amd.cli.main ingest --skip-download --force
```