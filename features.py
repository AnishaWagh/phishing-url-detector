import math
import re
from urllib.parse import urlparse

import tldextract

# Use the bundled suffix list so it works offline and on Streamlit Cloud
_extract = tldextract.TLDExtract(suffix_list_urls=())

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "bank",
    "confirm", "password", "paypal", "wallet", "free", "bonus", "support",
    "billing", "claim", "reward", "crypto",
]

IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$|^0x[0-9a-fA-F]+$")
SPECIAL_CHARS = "@?=&%#_~$!*"


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = {c: text.count(c) for c in set(text)}
    n = len(text)
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


def extract_features(url: str) -> dict:
    url = str(url).strip()
    is_https = int(url.lower().startswith("https://"))

    to_parse = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url) else "http://" + url
    try:
        parsed = urlparse(to_parse)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        has_port = int(parsed.port is not None)
    except ValueError:
        host, path, query, has_port = "", "", "", 0

    ext = _extract(host)
    domain = ext.domain
    suffix = ext.suffix
    subdomain = ext.subdomain
    subdomain_parts = [p for p in subdomain.split(".") if p and p != "www"]

    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    url_len = max(len(url), 1)

    return {
        "url_length": len(url),
        "hostname_length": len(host),
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_hyphens_domain": domain.count("-"),
        "num_digits": digits,
        "digit_ratio": digits / url_len,
        "letter_ratio": letters / url_len,
        "num_special_chars": sum(url.count(c) for c in SPECIAL_CHARS),
        "num_slashes": url.count("/"),
        "has_at": int("@" in url),
        "has_ip": int(bool(IP_PATTERN.match(host))),
        "has_port": has_port,
        "is_https": is_https,
        "num_subdomains": len(subdomain_parts),
        "tld_length": len(suffix),
        "domain_entropy": _entropy(domain),
        "domain_has_digits": int(any(c.isdigit() for c in domain)),
        "suspicious_word_count": sum(w in url.lower() for w in SUSPICIOUS_WORDS),
        "double_slash_in_path": int("//" in path),
    }


FEATURE_NAMES = list(extract_features("http://example.com").keys())
def extract_domain_features(url: str) -> dict:
    """Features from the HOSTNAME only. Ignores scheme, path and query."""
    url = str(url).strip()
    to_parse = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url) else "http://" + url
    try:
        parsed = urlparse(to_parse)
        host = (parsed.hostname or "").lower()
        has_userinfo = int(parsed.username is not None)
        has_port = int(parsed.port is not None)
    except ValueError:
        host, has_userinfo, has_port = "", 0, 0

    if host.startswith("www."):  # avoid the "www" shortcut
        host = host[4:]

    ext = _extract(host)
    domain, suffix = ext.domain, ext.suffix
    sub_parts = [p for p in ext.subdomain.split(".") if p]
    letters = sum(c.isalpha() for c in host)
    digits = sum(c.isdigit() for c in host)
    n = max(len(host), 1)

    return {
        "hostname_length": len(host),
        "domain_length": len(domain),
        "num_dots": host.count("."),
        "num_hyphens": host.count("-"),
        "num_hyphens_domain": domain.count("-"),
        "num_digits": digits,
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "has_ip": int(bool(IP_PATTERN.match(host))),
        "has_userinfo": has_userinfo,
        "has_port": has_port,
        "num_subdomains": len(sub_parts),
        "tld_length": len(suffix),
        "domain_entropy": _entropy(domain),
        "domain_has_digits": int(any(c.isdigit() for c in domain)),
        "suspicious_word_count": sum(w in host for w in SUSPICIOUS_WORDS),
    }


DOMAIN_FEATURE_NAMES = list(extract_domain_features("http://example.com").keys())