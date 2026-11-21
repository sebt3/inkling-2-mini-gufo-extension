#!/usr/bin/env bash
# Dev loop for the extension. Sources are synced to worker04 and built inside the inkling-dev pod
# (gufo-runtime image: nix toolchain, no PATH entries). Never build in the code-server pod.
#   tools/dev.sh sync            rsync-like (tar) of the tree to /var/tmp/inkling-ext/src
#   tools/dev.sh run <script>    run a bash script (stdin) in the dev pod with the toolchain env
#   tools/dev.sh build-ref       build tools/inkling_ref
set -euo pipefail
cd "$(dirname "$0")/.."
POD=${POD:-inkling-dev}; NS=kydah-llms
case "${1:-}" in
  sync) tar --exclude=.git --exclude=build -cf - . | ssh infra-worker04 'rm -rf /var/tmp/inkling-ext/src && mkdir -p /var/tmp/inkling-ext/src && tar -xf - -C /var/tmp/inkling-ext/src' ;;
  run) shift; { cat tools/env.sh; cat; } | ssh infra-m1 "sudo -i kubectl -n $NS exec -i $POD -c dev -- bash -s" ;;
  build-ref) "$0" run <<'X'
cd /src/src && mkdir -p /tmp/build && g++ -O3 -march=native -fopenmp -std=c++20 -o /tmp/build/inkling_ref tools/inkling_ref.cpp 2>&1 | head -40
ls -la /tmp/build/inkling_ref
X
  ;;
  build-oracle-dump) "$0" run <<'X'
cd /src/src && g++ -O2 -std=c++17 -o /tmp/build/oracle_dump tools/oracle_dump.cpp -I/src/llama.cpp/include -I/src/llama.cpp/ggml/include -L/tmp/llbuild/bin -lllama -lggml -lggml-base -Wl,-rpath,/tmp/llbuild/bin 2>&1 | head -20
ls -la /tmp/build/oracle_dump
X
  ;;
  *) echo "usage: $0 sync|run|build-ref|build-oracle-dump"; exit 2 ;;
esac
