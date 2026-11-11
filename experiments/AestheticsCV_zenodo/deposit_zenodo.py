"""
deposit_zenodo.py -- staged Zenodo deposit driver.
Reads the token ONLY from env var ZENODO_TOKEN (never written to disk).
Non-secret state (deposition id, bucket url, reserved DOI) is saved to deposit_state.json.

Subcommands:
  verify                 GET depositions (confirms token + host)
  create                 POST empty draft -> save id/bucket/reserved DOI
  setmeta <meta.json>    PUT metadata
  upload <file> [..]     PUT file(s) to the bucket
  status                 GET deposition (state, files, doi)
  publish                POST actions/publish  (IRREVERSIBLE)
"""
import os, sys, json, urllib.request, urllib.error

BASE = os.environ.get("ZENODO_BASE", "https://zenodo.org/api")
TOKEN = os.environ.get("ZENODO_TOKEN")
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "deposit_state.json")
if not TOKEN:
    sys.exit("ZENODO_TOKEN not set")


def _req(method, url, data=None, raw=None, ctype=None):
    h = {"Authorization": f"Bearer {TOKEN}"}
    body = None
    if data is not None:
        body = json.dumps(data).encode(); h["Content-Type"] = "application/json"
    elif raw is not None:
        body = raw; h["Content-Type"] = ctype or "application/octet-stream"
    req = urllib.request.Request(url, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=590) as r:
            t = r.read()
            try:
                return r.status, json.loads(t)
            except Exception:
                return r.status, t
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save(s):
    json.dump(s, open(STATE, "w"), indent=2)


cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"

if cmd == "verify":
    code, body = _req("GET", f"{BASE}/deposit/depositions?size=1")
    print("HOST", BASE, "STATUS", code)
    print("OK token valid" if code == 200 else f"FAIL {str(body)[:400]}")

elif cmd == "create":
    code, body = _req("POST", f"{BASE}/deposit/depositions", data={})
    if code not in (200, 201):
        sys.exit(f"create failed {code}: {str(body)[:600]}")
    dep = body["id"]
    bucket = body["links"].get("bucket")
    pre = (body.get("metadata") or {}).get("prereserve_doi") or {}
    doi = pre.get("doi")
    s = {"id": dep, "bucket": bucket, "doi": doi,
         "html": body["links"].get("html"), "self": body["links"].get("self")}
    save(s)
    print(json.dumps(s, indent=2))

elif cmd == "setmeta":
    s = state(); meta = json.load(open(sys.argv[2]))
    code, body = _req("PUT", f"{BASE}/deposit/depositions/{s['id']}", data=meta)
    print("setmeta STATUS", code)
    if code != 200:
        sys.exit(str(body)[:800])
    print("title:", body["metadata"].get("title"))
    pre = body["metadata"].get("prereserve_doi") or {}
    if pre.get("doi"):
        s["doi"] = pre["doi"]; save(s)
    print("reserved DOI:", s.get("doi"))

elif cmd == "upload":
    import requests
    s = state(); bucket = s["bucket"]
    for path in sys.argv[2:]:
        name = os.path.basename(path)
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            r = requests.put(f"{bucket}/{name}", data=f,
                             headers={"Authorization": f"Bearer {TOKEN}"},
                             timeout=(30, 3600))
        ok = r.status_code in (200, 201)
        print(f"upload {name} ({round(size/1e6,1)} MB) STATUS {r.status_code} "
              f"{'OK' if ok else r.text[:400]}")
        if not ok:
            sys.exit(1)
    print("ALL UPLOADS OK")

elif cmd == "status":
    s = state()
    code, body = _req("GET", f"{BASE}/deposit/depositions/{s['id']}")
    print("STATUS", code)
    if code == 200:
        print("state:", body.get("state"), "| submitted:", body.get("submitted"),
              "| doi:", (body.get("metadata") or {}).get("prereserve_doi", {}).get("doi") or body.get("doi"))
        print("title:", (body.get("metadata") or {}).get("title"))
        print("files:")
        for fobj in body.get("files", []):
            print("  ", fobj.get("filename"), round(fobj.get("filesize", 0) / 1e6, 1), "MB")
    else:
        print(str(body)[:600])

elif cmd == "publish":
    s = state()
    code, body = _req("POST", f"{BASE}/deposit/depositions/{s['id']}/actions/publish")
    print("publish STATUS", code)
    if code in (200, 202):
        print("DOI:", body.get("doi"))
        print("record:", (body.get("links") or {}).get("record_html") or (body.get("links") or {}).get("html"))
        s["published_doi"] = body.get("doi"); save(s)
    else:
        sys.exit(str(body)[:800])

else:
    sys.exit("unknown command")
