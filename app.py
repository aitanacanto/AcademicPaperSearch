#!/usr/bin/env python3
"""Academic Paper Search: a local browser UI for the original retrieval algorithms.

Run: python app.py
Student: Aitana Cantó Rico. Information Retrieval, University of West Attica, 2025-2026.
The retrieval functions below are preserved from the submitted notebook.
"""
import argparse
import json
import math
import re
import time
import webbrowser
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import parse_qs, urlparse

TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}
RIGHT_ASSOC = {"NOT"}

STOPWORDS = {'to', 'when', 'but', 'our', 'had', 'they', 'than', 'those', 'between', 'its', 'not', 'that', 'these', 'might', 'the', 'for', 'must', 'such', 'no', 'were', 'under', 'before', 'their', 'with', 'done', 'from', 'be', 'is', 'we', 'without', 'on', 'as', 'while', 'me', 'nor', 'should', 'how', 'of', 'them', 'my', 'about', 'during', 'where', 'will', 'if', 'you', 'your', 'can', 'being', 'over', 'or', 'it', 'does', 'why', 'in', 'having', 'then', 'i', 'have', 'been', 'by', 'do', 'may', 'was', 'there', 'after', 'shall', 'among', 'has', 'this', 'at', 'also', 'doing', 'and', 'did', 'an', 'a', 'into', 'are', 'here', 'would', 'could'}

def normalize_query_tokens(q: str) -> List[str]:
    # converting to tokens and to uppercase operators
    # keep AND/OR/NOT and brackets
    q = q.strip()

    # separating brackets
    q = q.replace("(", " ( ").replace(")", " ) ")

    parts = q.split()
    out = []
    for p in parts:
        up = p.upper()
        if up in {"AND", "OR", "NOT", "(", ")"}:
            out.append(up)
        else:
            # normal word -> lowercase token
            toks = TOKEN_RE.findall(p.lower())
            # ignoring weird toks
            out.extend(toks)
    return out

def get_docs_for_term(term: str, postings: Dict[str, List[Tuple[str, int]]]) -> Set[str]:
    plist = postings.get(term, [])
    return set(doc_id for doc_id, _tf in plist)

def to_rpn(tokens: List[str]) -> List[str]:
    output = []
    ops = []

    for t in tokens:
        if t in {"AND", "OR", "NOT"}:
            while ops:
                top = ops[-1]
                if top == "(":
                    break
                if (PRECEDENCE[top] > PRECEDENCE[t]) or (
                    PRECEDENCE[top] == PRECEDENCE[t] and t not in RIGHT_ASSOC
                ):
                    output.append(ops.pop())
                else:
                    break
            ops.append(t)
        elif t == "(":
            ops.append(t)
        elif t == ")":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if not ops:
                raise ValueError("Mismatched parentheses in query.")
            ops.pop()  # remove "("
        else:
            output.append(t)

    while ops:
        top = ops.pop()
        if top in {"(", ")"}:
            raise ValueError("Mismatched parentheses in query.")
        output.append(top)

    return output

def eval_rpn(rpn: List[str], postings: Dict[str, List[Tuple[str, int]]], universe: Set[str]) -> Set[str]:
    stack: List[Set[str]] = []

    for t in rpn:
        if t == "NOT":
            if not stack:
                raise ValueError("NOT operator missing operand.")
            a = stack.pop()
            stack.append(universe - a)
        elif t in {"AND", "OR"}:
            if len(stack) < 2:
                raise ValueError(f"{t} operator missing operand(s).")
            b = stack.pop()
            a = stack.pop()
            stack.append(a & b if t == "AND" else a | b)
        else:
            stack.append(get_docs_for_term(t, postings))

    if len(stack) != 1:
        raise ValueError("Invalid query (could not reduce to single result set).")
    return stack[0]

def tokenize_query(q: str) -> List[str]:
    q = q.lower()
    toks = TOKEN_RE.findall(q)
    clean = []
    for t in toks:
        if t in STOPWORDS:
            continue
        if len(t) < 2:
            continue
        clean.append(t)
    return clean

def compute_idf(df: int, N: int) -> float:
    # idf classic
    # avoid division by zero and idf weird negatives
    return math.log((N + 1) / (df + 1)) + 1.0

def tf_weight(tf: int) -> float:
    # log normalization
    return 1.0 + math.log(tf) if tf > 0 else 0.0

def rank_tfidf(
    query: str,
    postings: Dict[str, List[Tuple[str, int]]],
    df_map: Dict[str, int],
    N: int
) -> Dict[str, float]:
    """
    Score(d) = sum_{t in q} (tf_weight(tf_{t,d}) * idf(t))
    """
    q_terms = tokenize_query(query)
    if not q_terms:
        return {}

    scores = defaultdict(float)

    for t in q_terms:
        df = df_map.get(t, 0)
        if df == 0:
            continue
        idf = compute_idf(df, N)
        for doc_id, tf in postings.get(t, []):
            scores[doc_id] += tf_weight(tf) * idf

    return dict(scores)


class SearchEngine:
    """Build an in-memory index from the notebook's JSONL files, without pickle."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.postings = defaultdict(list)
        self.df = defaultdict(int)
        self.meta = {}
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {number}: {error.msg}") from error
            if not isinstance(doc, dict):
                raise ValueError(f"Line {number} must contain a JSON object")
            did = doc.get("doc_id") or doc.get("paperId") or doc.get("id")
            title = (doc.get("title") or "").replace("\u00ad", "").replace("\n", " ").strip()
            abstract = (doc.get("abstract") or "").replace("\u00ad", "").replace("\n", " ").strip()
            tokens = doc.get("tokens")
            if tokens is None:
                tokens = tokenize_query((title + "\n" + abstract).strip())
            if not isinstance(tokens, list) or any(not isinstance(t, str) for t in tokens):
                raise ValueError(f"Invalid tokens on line {number}")
            if not did or len(tokens) < 5 or did in self.meta:
                continue
            self.meta[did] = {**doc, "title": title, "abstract": abstract}
            for term, tf in Counter(tokens).items():
                self.postings[term].append((did, tf))
                self.df[term] += 1
        if not self.meta:
            raise ValueError("The JSONL file contains no indexable papers")
        for entries in self.postings.values():
            entries.sort(key=lambda item: item[0])

    def search(self, query, mode="tfidf", limit=10):
        if not query.strip():
            return {"total": 0, "results": []}
        if mode == "boolean":
            ids = eval_rpn(to_rpn(normalize_query_tokens(query)), self.postings, set(self.meta))
            ranked = sorted(ids, key=lambda did: (
                self.meta[did].get("year") or 0,
                self.meta[did].get("citationCount") or 0,
                did,
            ), reverse=True)
            rows = [(did, None) for did in ranked]
        elif mode == "tfidf":
            scores = rank_tfidf(query, self.postings, self.df, len(self.meta))
            rows = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        else:
            raise ValueError("Choose TF-IDF or Boolean retrieval")
        results = []
        for did, score in rows[:limit]:
            doc = self.meta[did]
            url = doc.get("url") or ""
            if urlparse(url).scheme not in {"http", "https"}:
                url = ""
            results.append({"id": did, "title": doc.get("title"),
                            "abstract": doc.get("abstract"), "year": doc.get("year"),
                            "venue": doc.get("venue"), "url": url, "score": score})
        return {"total": len(rows), "results": results}


HTML_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Academic Paper Search</title>
<style>
:root{color-scheme:light;--ink:#162e31;--muted:#5c6f70;--green:#176657;--paper:#f7f6f0;--line:#d9dfd7}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
.top{border-bottom:1px solid var(--line);padding:18px 5%;display:flex;justify-content:space-between;gap:15px;align-items:center}.brand{font-weight:750;font-size:14px;letter-spacing:.08em}.badge{font-size:12px;background:#e8eee4;padding:4px 12px;border-radius:20px;color:#325645}
main{max-width:1150px;margin:auto;padding:55px 28px 30px}.eyebrow{font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:var(--green);font-weight:700}h1{font-family:Georgia,serif;font-weight:400;font-size:clamp(38px,6vw,65px);line-height:1.08;letter-spacing:-.04em;margin:15px 0 20px;max-width:800px}.intro{max-width:680px;color:var(--muted);font-size:17px;margin:0 0 30px}
.stats{display:flex;gap:32px;flex-wrap:wrap;margin:24px 0 34px}.stat{font-size:12px;color:var(--muted)}.stat strong{display:block;color:var(--ink);font-size:24px;font-weight:650;line-height:1.2}
.search-panel{background:white;border:1px solid var(--line);border-radius:16px;padding:24px;box-shadow:0 10px 35px #193c3210}.search-row{display:flex;gap:12px}.search-row input{flex:1;min-width:0;padding:14px 17px;font:inherit;border:1px solid #b6c3ba;border-radius:9px;background:#fcfcf8;color:var(--ink)}input:focus,select:focus{outline:2px solid #76a695;outline-offset:2px}button{font:inherit;cursor:pointer}.submit{background:var(--green);color:white;border:0;border-radius:9px;padding:13px 28px;font-weight:650}.submit:hover{background:#104e42}.submit:disabled{opacity:.5;cursor:wait}.options{display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap;margin-top:18px;font-size:13px}.modes{display:flex;gap:17px}.modes label{cursor:pointer}select{padding:6px;border:1px solid var(--line);border-radius:5px;background:white;color:var(--ink)}.hint{margin:16px 0 0;font-size:12px;color:var(--muted)}.examples{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 0}.chip{font-size:12px;color:#33594f;background:#edf3ec;border:1px solid #dce6d9;border-radius:20px;padding:6px 12px}.chip:hover{background:#dfeade}
.results-bar{display:flex;justify-content:space-between;align-items:center;margin:34px 0 16px;gap:12px}.results-bar h2{font-size:19px;margin:0;font-weight:600}.timing{font-size:12px;color:var(--muted)}.card{background:white;border:1px solid var(--line);border-radius:12px;margin-bottom:14px;padding:24px 26px}.card-top{display:flex;justify-content:space-between;gap:18px;align-items:baseline}.card h3{font-size:19px;line-height:1.35;margin:0;letter-spacing:-.02em;max-width:860px}.score{font-size:11px;font-weight:600;white-space:nowrap;color:var(--green);background:#edf3ec;border-radius:5px;padding:4px 8px}.meta{font-size:12px;color:var(--muted);margin-top:9px}.abstract{font-size:14px;color:#465b5d;margin:16px 0}.link{font-size:13px;font-weight:650;color:var(--green)}.empty{border:1px dashed #bac8bb;border-radius:12px;padding:30px;color:var(--muted);text-align:center;font-size:14px}.error{color:#8a332f;background:#fbeee9;border:1px solid #e6c6bc;border-radius:8px;padding:12px 16px;margin-top:18px;font-size:14px}.status{font-size:12px;color:var(--muted);margin-top:15px}footer{max-width:1094px;margin:25px auto 0;border-top:1px solid var(--line);padding:20px 0 35px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:16px}a:focus-visible,button:focus-visible{outline:2px solid var(--green);outline-offset:3px}[hidden]{display:none!important}
@media(max-width:650px){main{padding:32px 18px}.top{padding:15px 18px}.top .badge{display:none}.search-panel{padding:18px}.search-row{flex-direction:column}.card{padding:20px}.card-top{flex-direction:column;gap:8px}.stats{gap:24px}footer{margin:25px 18px 0;flex-direction:column}}
</style></head><body>
<header class="top"><div class="brand">APS / ACADEMIC PAPER SEARCH</div><span class="badge">Information Retrieval · Academic project</span></header>
<main><div class="eyebrow">Research, connected.</div><h1>Find the literature.<br>Follow the evidence.</h1><p class="intro">Explore academic papers on document verification, misinformation and the reliability of language models.</p>
<div class="stats"><div class="stat"><strong id="papers">—</strong>indexed papers</div><div class="stat"><strong id="terms">—</strong>unique terms</div><div class="stat"><strong>2</strong>retrieval modes</div></div>
<section class="search-panel" aria-label="Search papers"><form id="search-form"><div class="search-row"><input id="query" name="query" aria-label="Search query" placeholder="Try document verification or LLM hallucination detection" value="document verification" maxlength="500" required><button class="submit" id="submit" type="submit">Search papers ↗</button></div><div class="options"><div class="modes"><label><input type="radio" name="mode" value="tfidf" checked> TF-IDF ranking</label><label><input type="radio" name="mode" value="boolean"> Boolean search</label></div><label>Show <select id="limit" aria-label="Number of results"><option>5</option><option selected>10</option><option>20</option><option>30</option></select> results</label></div></form><p class="hint" id="hint">Results ranked by term frequency and inverse document frequency.</p><div class="examples"><button class="chip" type="button" data-query="document verification">Document verification</button><button class="chip" type="button" data-query="llm hallucination detection">LLM hallucinations</button><button class="chip" type="button" data-query="document authenticity">Document authenticity</button><button class="chip" type="button" data-query="(llm OR ai) AND NOT misinformation" data-mode="boolean">Try Boolean search</button></div><p class="status" id="data-status">Loading the local index…</p><div class="error" id="error" role="alert" hidden></div></section>
<div class="results-bar"><h2 id="result-title">Explore the collection</h2><span class="timing" id="timing"></span></div><section id="results" aria-live="polite"><div class="empty">Enter a query or select one of the examples to begin.</div></section>
</main><footer><span>Aitana Cantó Rico · University of West Attica · 2025–2026</span><span>Local interface · OpenAlex metadata · Boolean & TF-IDF</span></footer>
<script>
const $=id=>document.getElementById(id);let requestId=0;
function error(message){$('error').textContent=message;$('error').hidden=!message;}
function mode(){return document.querySelector('input[name="mode"]:checked').value;}
function updateHint(){$('hint').textContent=mode()==='boolean'?'Use explicit AND, OR, NOT and parentheses. Example: document AND verification. Quoted phrases are not supported.':'Results ranked by term frequency and inverse document frequency.';}
document.querySelectorAll('input[name="mode"]').forEach(x=>x.addEventListener('change',updateHint));
function el(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;}
async function search(){const query=$('query').value.trim();if(!query)return;const current=++requestId;error('');$('submit').disabled=true;
try{const params=new URLSearchParams({q:query,mode:mode(),limit:$('limit').value});const res=await fetch('/api/search?'+params);const data=await res.json();if(current!==requestId)return;if(!res.ok)throw new Error(data.error||'Search failed.');$('results').replaceChildren();$('result-title').textContent=data.total+' matching paper'+(data.total===1?'':'s');$('timing').textContent='Showing '+data.results.length+' · '+data.elapsed_ms+' ms';
if(!data.results.length){$('results').append(el('div','empty','No papers match this query. Try broader terms or a different Boolean expression.'));}
data.results.forEach((paper,i)=>{const card=el('article','card');const top=el('div','card-top');top.append(el('h3','',(i+1)+'. '+(paper.title||'Untitled paper')));if(paper.score!==null)top.append(el('span','score','TF-IDF '+paper.score.toFixed(3)));card.append(top);card.append(el('div','meta',[paper.year,paper.venue].filter(Boolean).join(' · ')||'Publication details unavailable'));if(paper.abstract){const s=paper.abstract;card.append(el('p','abstract',s.length>650?s.slice(0,650)+'…':s));}if(paper.url){const a=el('a','link','Read paper ↗');a.href=paper.url;a.target='_blank';a.rel='noopener noreferrer';card.append(a);}$('results').append(card);});
}catch(e){if(current===requestId)error(e.message);}finally{if(current===requestId)$('submit').disabled=false;}}
$('search-form').addEventListener('submit',e=>{e.preventDefault();search();});document.querySelectorAll('.chip').forEach(b=>b.addEventListener('click',()=>{$('query').value=b.dataset.query;document.querySelector('input[name="mode"][value="'+(b.dataset.mode||'tfidf')+'"]').checked=true;updateHint();search();}));
fetch('/api/stats').then(r=>r.json()).then(d=>{$('papers').textContent=d.documents.toLocaleString();$('terms').textContent=d.terms.toLocaleString();$('data-status').textContent=d.documents?'Local corpus loaded · no external requests are sent when searching.':'No corpus loaded. Stop the server and run python app.py --data PATH_TO_JSONL, or run the notebook collection and preprocessing cells first.';}).catch(()=>error('Cannot connect to the local server. Keep python app.py running and reload this page.'));
</script></body></html>'''


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload, content_type="application/json; charset=utf-8"):
            body = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            expected_host = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") not in {expected_host, f"localhost:{self.server.server_port}"}:
                return self.reply(403, {"error": "Use the local server address"})
            parsed = urlparse(self.path)
            if parsed.path == "/":
                return self.reply(200, HTML_PAGE, "text/html; charset=utf-8")
            if parsed.path == "/api/stats":
                return self.reply(200, {"documents": len(engine.meta) if engine else 0,
                                        "terms": len(engine.df) if engine else 0})
            if parsed.path != "/api/search":
                return self.reply(404, {"error": "Not found"})
            if engine is None:
                return self.reply(409, {"error": "No corpus loaded. Start with python app.py --data PATH_TO_JSONL, or prepare data using the notebook first."})
            params = parse_qs(parsed.query)
            try:
                query = params.get("q", [""])[0]
                mode = params.get("mode", ["tfidf"])[0]
                limit = int(params.get("limit", ["10"])[0])
                if len(query) > 500 or not 1 <= limit <= 30:
                    raise ValueError("Use at most 500 query characters and between 1 and 30 results")
                start = time.perf_counter()
                result = engine.search(query, mode, limit)
                result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
                return self.reply(200, result)
            except ValueError as error:
                return self.reply(400, {"error": str(error)})

        def log_message(self, format, *args):
            # Avoid logging the research queries typed in the interface.
            pass

    return Handler


def find_corpus(root):
    for relative in (
        "data/processed/processed_papers.jsonl",
        "notebooks/data/processed/processed_papers.jsonl",
        "data/raw/raw_papers.jsonl",
        "notebooks/data/raw/raw_papers.jsonl",
    ):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Run Academic Paper Search locally in your browser")
    parser.add_argument("--data", type=Path, help="Notebook raw/processed JSONL corpus")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535")
    path = args.data or find_corpus(Path(__file__).resolve().parent)
    try:
        engine = SearchEngine(path) if path else None
    except (OSError, ValueError, TypeError) as error:
        parser.error(f"Cannot load the corpus: {error}")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    except OSError as error:
        parser.error(f"Cannot start the local server: {error}. Try --port 8766")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Academic Paper Search: {url}", flush=True)
    print(f"Indexed papers: {len(engine.meta) if engine else 0}", flush=True)
    print("Keep this terminal open. Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
