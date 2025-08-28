"""
Build BM25 index for wiki_dump2018_nq_open dataset on stlinux environment.
"""

import os
import pyterrier as pt
from datasets import load_dataset

# 2) Start PyTerrier / JVM
if not pt.started():
    pt.init(logging="WARN")  # use "INFO" for more logs

# 3) Writable temp dir (stlinux)
os.makedirs("/scratch/2980356s/data/SST2/tmp", exist_ok=True)
pt.set_property("terrier.tmp.dir", "/scratch/2980356s/data/SST2/tmp")

# Avoid Zstd meta compression (using gzip to dodge JNI issues)
pt.set_property("terrier.index.compression.meta.compression", "gzip")
# Optional: also force gzip for postings/direct if you hit zstd errors
pt.set_property("terrier.index.compression.inverted.compression", "gzip")
pt.set_property("terrier.index.compression.direct.compression", "gzip")

# Optional token pipeline (uncomment to enable stemming/stopwords)
# pt.set_property("termpipelines", "Stopwords,PorterStemmer")

# 4) Load corpus from Hugging Face
print("Loading corpus from Hugging Face...")
corpus = load_dataset("florin-hf/wiki_dump2018_nq_open", split="train")

# 5) Helper to stream HF rows to Terrier
def hf_corpus_iter(ds, id_field="id", text_field="text", title_field="title"):
    has_title = title_field in ds.features
    for r in ds:
        docno = str(r[id_field]) if id_field in r else str(r.get("id", ""))
        text  = r[text_field] if text_field in r else ""
        d = {"docno": docno, "text": text}
        if has_title:
            d["title"] = r[title_field]
        yield d

# 6) Build BM25 index in a writable directory
index_dir = "/scratch/2980356s/data/SST2/wiki2018_nqopen"
os.makedirs(index_dir, exist_ok=True)

# Store docno + text (+ title if present); adjust lengths as needed
meta = {"docno": 64, "text": 32768}
if "title" in corpus.features:
    meta["title"] = 512

indexer = pt.IterDictIndexer(
    index_dir,
    meta=meta,
    overwrite=True,
    blocks=False,
    threads=max(1, (os.cpu_count() or 2) - 1)  # leave a core free
)

print("Indexing (this can take a while)...")
indexref = indexer.index(hf_corpus_iter(corpus, id_field="id", text_field="text", title_field="title"))
print("Index built at:", index_dir)
print("IndexRef:", indexref.toString())
