# shellcheck shell=bash
# Shared helpers for the setup scripts. Sourced, not run.
# read_secret and retry are the same helpers as in anjijava16/glcoud-run-agents.

# Profile file holding your exported keys. Override with PROFILE_FILE=... if yours lives elsewhere.
PROFILE_FILE="${PROFILE_FILE:-$HOME/.zprofile.sh}"

# read_secret NAME -> prints the value of NAME.
# Order: already exported in this shell -> evaluated from $PROFILE_FILE with zsh ->
# plain "[export] NAME=value" line in $PROFILE_FILE. Prints nothing if not found.
read_secret() {
  local name="$1" val=""
  val="${!name:-}"
  if [ -z "$val" ] && [ -f "$PROFILE_FILE" ]; then
    if command -v zsh >/dev/null 2>&1; then
      val=$(zsh -c 'source "$1" >/dev/null 2>&1; eval "printf %s \"\${$2:-}\""' _ "$PROFILE_FILE" "$name" 2>/dev/null || true)
    fi
    if [ -z "$val" ]; then
      val=$(grep -E "^[[:space:]]*(export[[:space:]]+)?${name}=" "$PROFILE_FILE" | tail -n 1 \
        | sed -E "s/^[[:space:]]*(export[[:space:]]+)?${name}=//; s/[[:space:]]+#.*\$//; s/^[\"']//; s/[\"'][[:space:]]*\$//")
    fi
  fi
  printf '%s' "$val"
}

# pick NAME1 NAME2 ... -> sets PICK_VALUE to the first of these that is set (shell
# or $PROFILE_FILE) and PICK_NAME to the variable it came from. Call it directly,
# NOT inside $(...): a subshell would swallow both variables.
# shellcheck disable=SC2034  # PICK_VALUE / PICK_NAME are read by the caller
pick() {
  local n v
  PICK_VALUE=""; PICK_NAME=""
  for n in "$@"; do
    v=$(read_secret "$n")
    if [ -n "$v" ]; then PICK_VALUE="$v"; PICK_NAME="$n"; return 0; fi
  done
  return 0
}

# ask_hidden PROMPT -> reads a value without echoing it.
ask_hidden() {
  local v=""
  while [ -z "$v" ]; do
    printf "   %s (input hidden), then Enter: " "$1" >&2
    stty -echo 2>/dev/null; IFS= read -r v; stty echo 2>/dev/null; echo >&2
  done
  printf '%s' "$v"
}

# put_secret SECRET_ID VALUE -> adds a new version only if VALUE differs from the
# latest one, so re-running after you rotate a key in ~/.zprofile.sh just works.
# The value goes in through stdin, never on a command line.
put_secret() {
  local id="$1" val="$2" current
  current=$(gcloud secrets versions access latest --secret="$id" 2>/dev/null || true)
  if [ "$current" = "$val" ]; then
    echo "   $id: unchanged"
  else
    printf '%s' "$val" | gcloud secrets versions add "$id" --data-file=- >/dev/null
    echo "   $id: new version added"
  fi
}

# has_secret_value SECRET_ID -> true if the secret has an enabled version.
has_secret_value() {
  [ -n "$(gcloud secrets versions list "$1" --filter='state:ENABLED' --format='value(name)' --limit=1 2>/dev/null)" ]
}

# retry CMD... -> runs CMD up to 6 times, 15s apart. New service accounts can take a
# minute or two to become visible to IAM, so IAM calls go through this.
retry() {
  local attempt=1 max=6 wait=15 out
  while true; do
    if out=$("$@" 2>&1); then return 0; fi
    if [ "$attempt" -ge "$max" ]; then
      echo "$out" >&2
      return 1
    fi
    echo "   (not ready yet, retrying in ${wait}s: attempt $attempt/$max)" >&2
    sleep "$wait"; attempt=$((attempt + 1))
  done
}

step() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m   ! %s\033[0m\n' "$*"; }
exists() { "$@" >/dev/null 2>&1; }
