import requests
schema = requests.get("https://data.ademe.fr/data-fair/api/v1/datasets/meg-83tjwtg8dyz4vv7h1dqe").json()["schema"]
print([c["key"] for c in schema if any(m in c["key"] for m in ("ban", "geo", "coord"))])