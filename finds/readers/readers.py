"""Miscellaneous reader tools 

- external requests

Copyright 2022, Terence Lim

MIT License
"""
import requests
import time
import random
import pandas as pd
from pandas import DataFrame
from typing import Dict

_VERBOSE = 0
_headers = {'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36'
            'OPR/38.0.2220.41'}
def requests_get(url: str,
                 params: Dict | None = None,
                 retry: int = 7,
                 sleep: float = 2.,
                 timeout: float = 3.,
                 delay: float = 0.25,
                 trap: bool = False,
                 headers: Dict | None = _headers,
                 verbose: int = _VERBOSE) -> requests.Response | None:
    """Wrapper over requests.get, with retry loops and delays

    Args:
      url: URL address to request
      params: Payload of &key=value to append to url
      headers: User-Agent, Connection and other headers dict
      timeout: Number of seconds before timing out one request try
      retry: Number of times to retry request
      sleep: Number of seconds to wait between retries
      trap: On timed-out: if True raise exception, else return False
      delay: Number of seconds to wait initially
      verbose: Whether to display verbose debugging messages

    Returns:
      requests.Response or None if timed-out or status_code != 200
    """
    def _print(*args, **kwargs):
        """helper to print verbose messages"""
        if verbose > 0:
            print(*args, **kwargs)
            
    _print(url)
    if delay:
        time.sleep(random.uniform(delay, 2*delay))
    for i in range(retry):
        try:
            r = requests.get(url,
                             headers=headers,
                             timeout=timeout,
                             params=params)
            assert(r.status_code >= 200 and r.status_code <= 404)
            break
        except Exception as e:
            _print(f"(requests_url {i}/{retry})", e)
            time.sleep(sleep * (2 ** i) + sleep*random.uniform(0, 1))
            r = None
    if r is None:  # likely timed-out after retries:
        if trap:     # raise exception if trap, else silently return None
            raise Exception(f"requests_get: {url} {time.time()}")
        return None
    if r.status_code != 200:
        _print(r.status_code, r.content)
        return None
    return r

if __name__ == "__main__":
    response = requests_get('https://www.soa.org/', verbose=1)
    print(response.text)
    
