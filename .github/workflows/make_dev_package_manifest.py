#!/usr/bin/env python3

# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate a set of Catalyst development packages and describe it for S3 publication.

Consumes the directory that upload-dev-packages.yaml downloads the CPack archives into, and:

  1. parses the version and platform back out of each file name,
  2. checks the set is complete and self-consistent before anything is published,
  3. writes a ``<archive>.sha256`` sidecar next to each archive,
  4. writes a ``manifest.json`` describing the whole set, and
  5. optionally writes a markdown table for the job summary.

Only the archives, their sidecars and ``manifest.json`` end up in ``--dist-dir``; everything that
is not meant to be published is written elsewhere, because the publish step uploads that directory
wholesale.

The file names are the source of truth for the version. They are produced by
CPACK_PACKAGE_FILE_NAME in mlir/CMakeLists.txt, runtime/CMakeLists.txt and
mlir/cmake/LLVMDevCPackConfig.cmake as

    <package>-<system>-<processor>-<version>.tar.gz

for example

    catalyst-dev-Linux-x86_64-0.14.0-dev56.tar.gz
    catalyst-runtime-dev-Darwin-arm64-0.14.0-dev56.tar.gz
    catalyst-llvm-dev-Linux-x86_64-22.0.0git-8f264586.tar.gz

Re-deriving the version here rather than passing it in keeps the publisher honest: the version in
the S3 prefix is by construction the version CPack stamped into the archives it is publishing.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The processor group is hyphen-free (x86_64, arm64, aarch64) so that the greedy version group can
# safely contain hyphens, which it does for both version flavours: "0.14.0-dev56" and
# "22.0.0git-8f264586".
PACKAGE_RE = re.compile(
    r"^(?P<name>catalyst(?:-runtime|-llvm)?-dev)"
    r"-(?P<system>Linux|Darwin)"
    r"-(?P<processor>[^-]+)"
    r"-(?P<version>.+)\.tar\.gz$"
)

# catalyst-dev and catalyst-runtime-dev share CATALYST_PACKAGE_VERSION and are only valid when
# consumed as a matched pair, so they are published together. catalyst-llvm-dev is versioned on the
# pinned LLVM commit instead, moves independently of Catalyst releases, and therefore gets its own
# channel.
CHANNEL_PACKAGES = {
    "nightly": {"catalyst-dev", "catalyst-runtime-dev"},
    "release": {"catalyst-dev", "catalyst-runtime-dev"},
    "llvm": {"catalyst-llvm-dev"},
}

SCHEMA_VERSION = 1


def sha256_of(path, chunk_size=1 << 20):
    """Return the hex SHA-256 digest of a file, read incrementally."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def collect(dist_dir, channel, required_platforms):
    """Parse every archive in dist_dir and verify the set is publishable.

    Returns (version, entries). Exits non-zero with every problem reported at once, rather than
    failing on the first, so a broken build only costs one workflow run to diagnose.
    """
    archives = sorted(dist_dir.glob("*.tar.gz"))
    if not archives:
        sys.exit(f"error: no .tar.gz archives found in {dist_dir}")

    expected_packages = CHANNEL_PACKAGES[channel]
    entries = []
    problems = []

    for archive in archives:
        match = PACKAGE_RE.match(archive.name)
        if not match:
            problems.append(f"{archive.name}: does not match the CPack naming scheme")
            continue

        fields = match.groupdict()
        if fields["name"] not in expected_packages:
            problems.append(
                f"{archive.name}: package '{fields['name']}' does not belong in the "
                f"'{channel}' channel (expected one of {sorted(expected_packages)})"
            )
            continue

        entries.append(
            {
                "name": fields["name"],
                "platform": f"{fields['system']}-{fields['processor']}",
                "system": fields["system"],
                "processor": fields["processor"],
                "version": fields["version"],
                "file": archive.name,
                "size_bytes": archive.stat().st_size,
                "sha256": sha256_of(archive),
            }
        )

    if problems:
        sys.exit("error: " + "\nerror: ".join(problems))

    # A mixed-version set means two different builds got merged into one artifact download; the
    # pair would be published under a single prefix and silently mismatch at consumption time.
    versions = {entry["version"] for entry in entries}
    if len(versions) > 1:
        sys.exit(
            "error: archives disagree on the version: "
            + ", ".join(f"{entry['file']} -> {entry['version']}" for entry in entries)
        )

    # Publish all platforms or none. A partial set under a version prefix looks complete to a
    # consumer that only checks whether the prefix exists.
    for package in sorted(expected_packages):
        found = {entry["platform"] for entry in entries if entry["name"] == package}
        missing = required_platforms - found
        if missing:
            problems.append(f"{package}: missing platform(s) {', '.join(sorted(missing))}")

    if problems:
        sys.exit("error: " + "\nerror: ".join(problems))

    return versions.pop(), entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir", required=True, type=Path, help="directory holding the downloaded archives"
    )
    parser.add_argument("--channel", required=True, choices=sorted(CHANNEL_PACKAGES))
    parser.add_argument(
        "--platforms",
        required=True,
        help="comma separated platforms that must be present, e.g. 'Linux-x86_64,Darwin-arm64'",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="write a markdown table describing the set here; must be outside --dist-dir",
    )
    args = parser.parse_args()

    required_platforms = {item.strip() for item in args.platforms.split(",") if item.strip()}
    if not required_platforms:
        sys.exit("error: --platforms must name at least one platform")

    # Checked before any work: --dist-dir is published wholesale, so a summary written into it
    # would leak into the bucket.
    if (
        args.summary_file is not None
        and args.summary_file.resolve().parent == args.dist_dir.resolve()
    ):
        sys.exit("error: --summary-file must not live in --dist-dir; it would be published")

    version, entries = collect(args.dist_dir, args.channel, required_platforms)
    prefix = f"{args.channel}/{version}/"

    # sha256sum-compatible sidecars (two spaces, binary mode), so a consumer can verify with
    # `sha256sum -c <file>.sha256` and CMake can pin via URL_HASH.
    for entry in entries:
        sidecar = args.dist_dir / f"{entry['file']}.sha256"
        sidecar.write_text(f"{entry['sha256']}  {entry['file']}\n", encoding="UTF-8")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "channel": args.channel,
        "version": version,
        "prefix": prefix,
        "commit": os.environ.get("GITHUB_SHA", ""),
        "ref": os.environ.get("GITHUB_REF_NAME", ""),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_url": (
            f"{os.environ.get('GITHUB_SERVER_URL', '')}/{os.environ.get('GITHUB_REPOSITORY', '')}"
            f"/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
        ),
        "packages": sorted(entries, key=lambda entry: (entry["name"], entry["platform"])),
    }

    manifest_path = args.dist_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="UTF-8")

    print(f"Validated {len(entries)} archive(s) for {args.channel} {version}")
    for entry in manifest["packages"]:
        print(f"  {entry['file']}  {entry['size_bytes']} bytes  sha256:{entry['sha256'][:12]}...")

    if args.summary_file is not None:
        rows = [
            "| Package | Platform | Size | SHA-256 |",
            "|---|---|---|---|",
        ]
        for entry in manifest["packages"]:
            size = entry["size_bytes"] / (1024 * 1024)
            rows.append(
                f"| `{entry['name']}` | {entry['platform']} | {size:.1f} MiB "
                f"| `{entry['sha256'][:16]}...` |"
            )
        args.summary_file.write_text("\n".join(rows) + "\n", encoding="UTF-8")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="UTF-8") as handle:
            handle.write(f"version={version}\n")
            handle.write(f"prefix={prefix}\n")
            handle.write(f"latest_key={args.channel}/latest.json\n")


if __name__ == "__main__":
    main()
