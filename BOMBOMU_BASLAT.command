#!/bin/zsh
set -e

cd "${0:A:h}"

if command -v node >/dev/null 2>&1; then
  BOMBOM_NODE="$(command -v node)"
elif [[ -x "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node" ]]; then
  BOMBOM_NODE="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
else
  print "Node.js bulunamadı. Codex çalışma ortamını veya Node.js 24+ kurulumunu açıp yeniden deneyin."
  read "?Kapatmak için Enter'a basın."
  exit 1
fi

export BOMBOM_OPEN_BROWSER=1
exec "$BOMBOM_NODE" runtime/dev-all.mjs
