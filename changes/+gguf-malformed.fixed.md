GGUF reading now raises `ValueError` for any malformed file, including duplicate keys, and rejects absurd header counts and array lengths up front instead of hanging.
