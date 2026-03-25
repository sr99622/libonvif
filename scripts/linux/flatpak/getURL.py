#************************************************************************
#
#   Python program for collecting urls of packages required by onvif-gui. 
#   Not perfect, but a good starting point. Use pip freeze to generate
#   a requirements.txt file which is then used to collect the wheel urls.
#   Some are missed for unknown reasons, torch-cpu-only done manually
#
#************************************************************************

import requests
from packaging.requirements import Requirement

def get_package_info(package_spec):
    req = Requirement(package_spec)

    name = req.name
    version = None
    for spec in req.specifier:
        if spec.operator == "==":
            version = spec.version
            break

    if not version:
        raise ValueError(f"Requirement '{package_spec}' must be pinned with '==version'.")

    url = f"https://pypi.org/pypi/{name}/{version}/json"
    resp = requests.get(url)
    if resp.status_code == 404:
        raise ValueError(f"Package '{name}=={version}' not found on PyPI.")
    resp.raise_for_status()
    data = resp.json()

    # Prefer data["releases"][version], fallback to data["urls"]
    files = data.get("releases", {}).get(version, [])
    if not files:
        files = data.get("urls", [])

    if not files:
        raise ValueError(f"No files found for '{name}=={version}' on PyPI.")

    results = []
    for file in files:
        file_url = file["url"]
        sha256_digest = file["digests"]["sha256"]
        results.append((package_spec, file_url, sha256_digest))
    return results

def main(requirements_file):
    with open(requirements_file) as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    for package in packages:
        try:
            entries = get_package_info(package)
            for _, url, digest in entries:
                if ("cp312-manylinux" in url and "x86_64" in url) or "py3-none-any" in url:
                    print("- type: file")
                    print(f"  url: {url}")
                    print(f"  sha256: {digest}")
                    print("  x-checker-data:")
                    print("    type: pypi")
                    print(f"    name: {package.split("==")[0]}")
                
        except Exception as e:
            print(f"Error with {package}: {e}")

if __name__ == "__main__":
    main("requirements.txt")
