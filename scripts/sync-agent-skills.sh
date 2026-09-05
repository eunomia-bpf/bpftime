#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
submodule_path='.agents/sources/agent-skills'

git -C "$repo_root" submodule update --init -- "$submodule_path"
"$repo_root/$submodule_path/scripts/link-skills.sh" "$repo_root/.agents/skills"
