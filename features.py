"""
features.py
Turns a URL into a row of numbers (features) that the model can understand.
This file is used by prepare.py (to build training data), by evaluate.py and
test_urls.py (to test), and by app.py (to score a URL typed by the user).
"""
import math                      # math.log2 is needed for the entropy calculation
import re                        # regular expressions: pattern matching on text
from urllib.parse import urlparse  # splits a URL into scheme, host, path, etc.

import tldextract                # splits a hostname into subdomain / domain / suffix

# Build the extractor with NO online lookups. It uses the suffix list bundled
# inside the package, so it works offline and on Streamlit Cloud.
_extract = tldextract.TLDExtract(suffix_list_urls=())

# Words that phishing domains often contain to look trustworthy.
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "bank",
    "confirm", "password", "paypal", "wallet", "free", "bonus", "support",
    "billing", "claim", "reward", "crypto",
]

# Matches a host that is an IPv4 address (like 192.168.1.5) or a hex IP (0x7f000001).
IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$|^0x[0-9a-fA-F]+$")


def _entropy(text: str) -> float:
    """Measure how random a string looks (0 = very regular, higher = more random)."""
    if not text:                                   # empty text has no randomness
        return 0.0
    counts = {c: text.count(c) for c in set(text)}  # how many times each character appears
    n = len(text)                                  # total number of characters
    # Shannon entropy formula: -sum(p * log2(p)) where p is a character's share
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


def extract_domain_features(url: str) -> dict:
    """Features from the HOSTNAME only. Ignores scheme, path and query."""
    url = str(url).strip()  # make sure it is text and remove spaces at the ends

    # urlparse needs a scheme (http://) to find the host. Add one if it is missing.
    to_parse = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url) else "http://" + url
    try:
        parsed = urlparse(to_parse)                    # split the URL into parts
        host = (parsed.hostname or "").lower()         # domain part, lowercase
        has_userinfo = int(parsed.username is not None)  # 1 if the URL has user@ (the '@' trick)
        has_port = int(parsed.port is not None)        # 1 if it names a port like :8080
    except ValueError:                                 # badly broken URLs raise this error
        host, has_userinfo, has_port = "", 0, 0

    if host.startswith("www."):  # remove 'www.' so the model cannot use it as a shortcut
        host = host[4:]

    ext = _extract(host)                               # split host into subdomain, domain, suffix
    domain, suffix = ext.domain, ext.suffix            # e.g. 'google' and 'com'
    sub_parts = [p for p in ext.subdomain.split(".") if p]  # list of subdomain pieces
    letters = sum(c.isalpha() for c in host)           # count letters in the host
    digits = sum(c.isdigit() for c in host)            # count digits in the host
    n = max(len(host), 1)                              # host length (min 1 to avoid divide by zero)

    return {
        "hostname_length": len(host),                  # long hostnames are often suspicious
        "domain_length": len(domain),                  # length of the main domain name
        "num_dots": host.count("."),                   # many dots = many subdomains
        "num_hyphens": host.count("-"),                # hyphens in the whole host
        "num_hyphens_domain": domain.count("-"),       # hyphens in the main domain only
        "num_digits": digits,                          # how many digits
        "digit_ratio": digits / n,                     # share of characters that are digits
        "letter_ratio": letters / n,                   # share of characters that are letters
        "has_ip": int(bool(IP_PATTERN.match(host))),   # 1 if the host is an IP address
        "has_userinfo": has_userinfo,                  # 1 if '@' hides the real destination
        "has_port": has_port,                          # 1 if a port is given
        "num_subdomains": len(sub_parts),              # how many subdomain pieces
        "tld_length": len(suffix),                     # length of the ending (com, co.in, xyz)
        "domain_entropy": _entropy(domain),            # randomness of the domain name
        "domain_has_digits": int(any(c.isdigit() for c in domain)),  # 1 if domain has a digit
        "suspicious_word_count": sum(w in host for w in SUSPICIOUS_WORDS),  # keywords found in host
    }


# The list of feature names in order. The model is trained and used with this order.
DOMAIN_FEATURE_NAMES = list(extract_domain_features("http://example.com").keys())