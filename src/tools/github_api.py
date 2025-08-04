from __future__ import annotations

import os
import time
import logging
from typing import Optional

import requests

# Local imports
from ..config import settings

GITHUB_API = "https://api.github.com"
RAW_ACCEPT = "application/vnd.github.raw+json"
JSON_ACCEPT = "application/vnd.github+json"
API_VERSION = "2022-11-28"

# --- Logging configuration -------------------------------------------------
# Ensure a dedicated log directory exists at the project root so that
# GithubClient messages are persisted there. We keep console output too.
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "github_client.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode="a"),  # Persist to /log/github_client.log
        logging.StreamHandler()  # Still echo to stdout for immediate feedback
    ],
)
# ---------------------------------------------------------------------------


class GithubClient:
    def __init__(self, token: Optional[str] = None, max_retries: int = 5, timeout: int = 60):
        # Logger is configured at the module level; no need to call basicConfig here.

        self.s = requests.Session()
        self.s.headers.update({
            "Accept": JSON_ACCEPT,
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "gh-acr-fetcher/1.0",
        })
        token = token or os.getenv("GITHUB_TOKEN")
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"
        self.max_retries = max_retries
        self.timeout = timeout
        # Track last request time for simple global rate-limiting
        self._last_request_time: float = 0.0
        # Interval between requests sourced from global settings
        self.request_interval: float = getattr(settings, "REQUEST_INTERVAL", 1.0)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _respect_rate_limit(self, resp: requests.Response, attempt: int):
        if resp.status_code in (403, 429):
            remaining = resp.headers.get("x-ratelimit-remaining")
            reset = resp.headers.get("x-ratelimit-reset")
            retry_after = resp.headers.get("retry-after")
            if remaining == "0" and reset:
                wait = max(0, int(reset) - int(time.time())) + 1
            elif retry_after:
                wait = int(retry_after)
            else:
                wait = min(60 * (attempt + 1), 300)
            print(f"Rate limit exceeded. Waiting for {wait} seconds.")
            time.sleep(wait)

    def _request(self, url: str, accept: Optional[str] = None) -> requests.Response:
        last_exc = None
        for attempt in range(self.max_retries):
            # Enforce a minimum 1-second interval between outbound requests
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self.request_interval:
                time.sleep(self.request_interval - elapsed)

            headers = {}
            if accept:
                headers["Accept"] = accept
            try:
                # Log the outgoing request
                self.logger.info(f"GET {url} (attempt {attempt + 1})")

                resp = self.s.get(url, headers=headers, timeout=self.timeout, allow_redirects=True)

                # Update last request timestamp after a successful network call
                self._last_request_time = time.time()
            except requests.RequestException as e:
                last_exc = e
                time.sleep(min(2 ** attempt, 10))
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code in (403, 429):
                self._respect_rate_limit(resp, attempt)
                continue
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 10))
                continue
            
            resp.raise_for_status()
        
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Failed to GET {url} after {self.max_retries} retries.")

    def get_commit(self, owner: str, repo: str, sha: str) -> dict:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}"
        resp = self._request(url)
        return resp.json()

    def get_merge_base(self, owner: str, repo: str, base: str, head: str) -> dict:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/compare/{base}...{head}"
        resp = self._request(url)
        return resp.json()["merge_base_commit"]
    
    def get_file_content(self, owner: str, repo: str, path: str, ref: str) -> str:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}?ref={ref}"
        resp = self._request(url, accept=RAW_ACCEPT)
        return resp.text 