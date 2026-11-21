#!/usr/bin/env bash
# Runs the M1 gates: G1 (tokenizer, gates/tokenizer_gate.py) and G2 (chat template, gates/template_gate.py).
# The C++ probes are built and run in the dev pod (tools/dev.sh, TREE=m1); the oracle is queried from infra-m1,
# where the gate scripts run. WORK is a scratch directory on this machine (default: a fresh temp dir).
#   tools/m1_gates.sh            both gates
#   tools/m1_gates.sh tokenizer  G1 only
#   tools/m1_gates.sh template   G2 only
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-$(mktemp -d)}
TREE=m1
export TREE
ICU_INC=/nix/store/i8jf1bw06qbvb9w7apkz1zy796fnyizw-icu4c-78.3-dev/include
ICU_LIB=/nix/store/9wwqcw8hr8z47avkdcigdyhclkq62f0a-icu4c-78.3/lib
MODEL=/models/Inkling-2-Mini-UD-Q4_K_M.gguf
which=${1:-all}

oracle_check() {  # $1 gate script, $2 cases, $3 probe results
  ssh infra-m1 'cat > /tmp/m1gate_cases.jsonl' < "$2"
  ssh infra-m1 'cat > /tmp/m1gate_probe.jsonl' < "$3"
  ssh infra-m1 'cat > /tmp/m1gate.py' < "$1"
  ssh infra-m1 'sudo -i bash -s' <<'EOF'
IP=$(kubectl -n kydah-llms get pod inkling-oracle -o jsonpath='{.status.podIP}')
python3 /tmp/m1gate.py check --oracle http://$IP:8001 --cases /tmp/m1gate_cases.jsonl --probe /tmp/m1gate_probe.jsonl
EOF
}

if [[ $which == all || $which == tokenizer ]]; then
  python3 gates/tokenizer_gate.py gen > "$WORK/tok_cases.jsonl"
  {
    echo "cat > /tmp/tok_cases.jsonl <<'CASES_EOF'"; cat "$WORK/tok_cases.jsonl"; echo "CASES_EOF"
    echo "cd \$SRC && g++ -O2 -std=c++20 -Wall -Wextra -I$ICU_INC -o \$BUILD/tokenizer_probe tools/tokenizer_probe.cpp host/tokenizer.cpp -L$ICU_LIB -licuuc -licudata -Wl,-rpath,$ICU_LIB"
    echo "\$BUILD/tokenizer_probe $MODEL < /tmp/tok_cases.jsonl"
  } > "$WORK/run_tok.sh"
  tools/dev.sh sync
  tools/dev.sh run < "$WORK/run_tok.sh" > "$WORK/tok_probe.jsonl"
  oracle_check gates/tokenizer_gate.py "$WORK/tok_cases.jsonl" "$WORK/tok_probe.jsonl"
fi

if [[ $which == all || $which == template ]]; then
  python3 gates/template_gate.py gen > "$WORK/tpl_cases.jsonl"
  {
    echo "cat > /tmp/tpl_cases.jsonl <<'CASES_EOF'"; cat "$WORK/tpl_cases.jsonl"; echo "CASES_EOF"
    echo "cd \$SRC && g++ -O2 -std=c++20 -Wall -Wextra -o \$BUILD/template_probe tools/template_probe.cpp host/chat_template.cpp"
    echo "\$BUILD/template_probe < /tmp/tpl_cases.jsonl"
  } > "$WORK/run_tpl.sh"
  tools/dev.sh sync
  tools/dev.sh run < "$WORK/run_tpl.sh" > "$WORK/tpl_probe.jsonl"
  oracle_check gates/template_gate.py "$WORK/tpl_cases.jsonl" "$WORK/tpl_probe.jsonl"
fi
