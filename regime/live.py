"""Shared live runtime: durable cache, conservative HTTP and one worker per role."""
import io
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pandas as pd

from .data import ROOT, config

LOG = logging.getLogger(__name__)
DIRECTORY = ROOT / "data/live"
HOSTS = {"api.binance.com", "fapi.binance.com", "data.binance.vision", "fred.stlouisfed.org",
         "query1.finance.yahoo.com", "community-api.coinmetrics.io", "stablecoins.llama.fi", "api.llama.fi"}


class Store:
    def __init__(self, path=None):
        self.path = Path(path or DIRECTORY / "runtime.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS cache (name TEXT PRIMARY KEY, value BLOB NOT NULL)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def blob(self, name, value=None):
        with self.connection() as db:
            if value is not None:
                db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (name, value))
                return value
            row = db.execute("SELECT value FROM cache WHERE name=?", (name,)).fetchone()
            return row[0] if row else None

    def json(self, name, value=None):
        if value is not None:
            self.blob(name, json.dumps(value, default=str, allow_nan=False).encode())
            return value
        body = self.blob(name)
        return json.loads(body) if body else {}

    def frame(self, name, value=None):
        if value is not None:
            body = io.BytesIO()
            value.to_parquet(body)
            self.blob(name, body.getvalue())
            return value
        body = self.blob(name)
        return pd.read_parquet(io.BytesIO(body)) if body is not None else None


@contextmanager
def exclusive(path):
    """OS releases this lock even after an unclean shutdown; shared across processes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError:
            pass
        yield locked
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


class Deferred(RuntimeError):
    def __init__(self, until, message):
        self.until = until
        super().__init__(message)


class Client:
    def __init__(self, store, opener=urlopen, clock=time.time):
        self.store, self.opener, self.clock = store, opener, clock

    def get(self, url):
        host = urlsplit(url).hostname
        if host not in HOSTS:
            raise ValueError("Unapproved live data host")
        with exclusive(self.store.path.parent / ("http-"+host+".lock")) as owner:
            if not owner:
                raise Deferred(self.clock()+2, "Another request is using this provider")
            return self._get(url)

    def _get(self, url):
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in HOSTS:
            raise ValueError("Unapproved live data host")
        key = "http:" + parsed.hostname
        while True:
            now = self.clock()
            with self.store.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT value FROM cache WHERE name=?", (key,)).fetchone()
                state = json.loads(row[0]) if row else {}
                if state.get("until", 0) > now:
                    raise Deferred(state["until"], state.get("error", "API cooldown"))
                wait = state.get("next", 0) - now
                if wait <= 0:
                    state["next"] = now + 2
                    db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(state).encode()))
                    break
            time.sleep(min(wait, 2))
        try:
            request = Request(url, headers={"User-Agent": "CryptoRegimeDashboard/0.2", "Accept": "application/json,text/csv,*/*"})
            with self.opener(request, timeout=45 if parsed.hostname == "fred.stlouisfed.org" else 20) as response:
                if urlsplit(response.geturl()).hostname not in HOSTS:
                    raise ValueError("Unexpected provider redirect")
                body = response.read(50_000_001)
                if len(body) > 50_000_000:
                    raise ValueError("Live response exceeds 50 MB")
                weight = int(response.headers.get("X-MBX-USED-WEIGHT-1M", 0))
            state.update(failures=0, error="", used_weight=weight)
            if parsed.hostname == "api.binance.com":
                limit = self.store.json("exchange").get("minute_weight", 6000)
                if weight >= limit - 20:
                    state["until"] = (int(self.clock() // 60) + 1) * 60 + 1
                    state["error"] = "Waiting for Binance weight reset"
            self.store.json(key, state)
            return body
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            now = self.clock()
            failures = state.get("failures", 0) + 1
            delay = min(3600, 60 * 2 ** min(failures-1, 6))
            code = getattr(error, "code", None)
            if code in (403, 451):
                delay = 86400
            if code == 418:
                delay = 3 * 86400
            retry = error.headers.get("Retry-After") if isinstance(error, HTTPError) and error.headers else None
            if retry:
                try:
                    delay = max(delay if code not in (418, 429) else 0, float(retry))
                except ValueError:
                    delay = max(delay, parsedate_to_datetime(retry).timestamp() - now)
            state.update(failures=failures, until=now+max(delay, 1), error=f"HTTP {code}" if code else type(error).__name__)
            # A not-yet-published archive is not a host-wide rate-limit incident.
            if code != 404:
                self.store.json(key, state)
            if code == 404:
                raise
            raise Deferred(state["until"], state["error"]) from error

    def json(self, url):
        return json.loads(self.get(url))


def update_quotes(store, client):
    now = time.time()
    previous = store.json("quotes")
    if now - previous.get("fetched_at", 0) < 60:
        return previous
    from urllib.parse import urlencode
    assets = config()["assets"]
    values = client.json("https://api.binance.com/api/v3/ticker/price?" + urlencode({"symbols": json.dumps(assets,separators=(",", ":"))}))
    prices = {row["symbol"]: float(row["price"]) for row in values}
    if set(prices) != set(assets) or any(not 0 < p < float("inf") for p in prices.values()):
        raise ValueError("Invalid live ticker response")
    return store.json("quotes", {"fetched_at": time.time(), "prices": prices})


def daily_update(store, client):
    from .live_data import update_controls, update_spot
    from .live_models import update_models

    clock = client.json("https://api.binance.com/api/v3/time")
    now = pd.Timestamp(clock["serverTime"], unit="ms", tz="UTC")
    # Allow five minutes after UTC rollover; the current bar is always excluded separately.
    cutoff = now.floor("D") - pd.Timedelta(days=1 if now.minute >= 5 or now.hour else 2)
    target = str(cutoff.date())
    prior = store.json("daily")
    if prior.get("target") == target and prior.get("complete"):
        return
    store.json("daily", {"target": target, "started_at": str(now), "complete": False, "stage": "Spot data"})
    frames = update_spot(store, client, cutoff)
    store.json("daily", {"target":target,"complete":False,"stage":"Daily model inference"})
    update_models(store, frames, cutoff, core_only=True)
    store.json("daily", {"target": target, "complete": False, "stage": "Funding, OI and external controls"})
    frames = update_controls(store, client, frames, cutoff)
    update_models(store, frames, cutoff, core_only=False)
    rows = [store.json("model:"+a+":"+m) for a in config()["assets"] for m in store.json("models")["assets"][a]]
    complete = all(row.get("through") == target and not row.get("error") for row in rows)
    store.json("daily", {"target": target, "complete": complete, "stage": "Ready" if complete else "Waiting for source data / retry",
                         "finished_at": str(pd.Timestamp.now(tz="UTC"))})


class Service:
    def __init__(self, directory=None):
        self.directory = Path(directory or DIRECTORY)
        self.store = Store(self.directory / "runtime.sqlite")
        self.client = Client(self.store)
        self.stop = threading.Event()
        self.threads = []

    def start(self):
        for role, interval, action in (("quotes", 60, update_quotes), ("daily", 600, daily_update)):
            thread = threading.Thread(target=self._run, args=(role, interval, action), daemon=True, name="crypto-"+role)
            thread.start()
            self.threads.append(thread)
        return self

    def _run(self, role, interval, action):
        while not self.stop.is_set():
            with exclusive(self.directory / (role+".lock")) as owner:
                if owner:
                    while not self.stop.is_set():
                        try:
                            action(self.store, self.client)
                            self.store.json("worker:"+role, {"updated_at": time.time(), "error": ""})
                        except Deferred as error:
                            self.store.json("worker:"+role, {"updated_at": time.time(), "error": str(error)[:400],
                                                            "retry_at": error.until})
                        except Exception as error:
                            LOG.exception("Live %s update failed", role)
                            self.store.json("worker:"+role, {"updated_at": time.time(), "error": str(error)[:400]})
                        self.stop.wait(interval)
            self.stop.wait(10)

    def close(self):
        self.stop.set()


def run(once=False):
    service = Service()
    if once:
        for role, action in (("quotes", update_quotes), ("daily", daily_update)):
            with exclusive(service.directory / (role+".lock")) as owner:
                if not owner:
                    print(f"{role}: another worker already runs")
                    continue
                action(service.store, service.client)
        return
    service.start()
    try:
        while not service.stop.wait(1):
            pass
    except KeyboardInterrupt:
        service.close()
