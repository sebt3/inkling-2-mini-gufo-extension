#!/usr/bin/env bash
# Runs the M9 gates (tool calls):
#   G9a  prompt rendering with tools (gates/tools_gate.py) versus the oracle /apply-template, 100 % identical;
#   scan the tool-scan unit tests (tests/tool_scan_test.cpp, no oracle).
# The C++ binaries are built and run in the dev pod (tools/dev.sh, TREE=m9); the oracle is queried from infra-m1,
# where the gate script runs. WORK is a scratch directory on this machine (default: a fresh temp dir).
#   G2   the M1 template gate (gates/template_gate.py, 113 cases), rebuilt from this tree: must stay 113/113.
#   tools/m9_gates.sh            all
#   tools/m9_gates.sh tools      G9a only
#   tools/m9_gates.sh template   G2 only
#   tools/m9_gates.sh scan       tool scanner tests only
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-$(mktemp -d)}
TREE=m9
export TREE
which=${1:-all}
status=0

oracle_check() {  # $1 gate script, $2 cases, $3 probe results
  ssh infra-m1 'cat > /tmp/m9gate_cases.jsonl' < "$2"
  ssh infra-m1 'cat > /tmp/m9gate_probe.jsonl' < "$3"
  ssh infra-m1 'cat > /tmp/m9gate.py' < "$1"
  ssh infra-m1 'sudo -i bash -s' <<'EOF'
IP=$(kubectl -n kydah-llms get pod inkling-oracle -o jsonpath='{.status.podIP}')
python3 /tmp/m9gate.py check --oracle http://$IP:8001 --cases /tmp/m9gate_cases.jsonl --probe /tmp/m9gate_probe.jsonl
EOF
}

if [[ $which == all || $which == tools ]]; then
  python3 gates/tools_gate.py gen > "$WORK/tools_cases.jsonl"
  {
    echo "cat > /tmp/tools_cases.jsonl <<'CASES_EOF'"; cat "$WORK/tools_cases.jsonl"; echo "CASES_EOF"
    echo "cd \$SRC && mkdir -p \$BUILD && g++ -O2 -std=c++20 -Wall -Wextra -o \$BUILD/tools_probe tools/tools_probe.cpp host/chat_template.cpp host/json.cpp"
    echo "\$BUILD/tools_probe < /tmp/tools_cases.jsonl"
  } > "$WORK/run_tools.sh"
  tools/dev.sh sync
  tools/dev.sh run < "$WORK/run_tools.sh" > "$WORK/tools_probe.jsonl"
  oracle_check gates/tools_gate.py "$WORK/tools_cases.jsonl" "$WORK/tools_probe.jsonl" || status=1
fi

if [[ $which == all || $which == template ]]; then
  python3 gates/template_gate.py gen > "$WORK/tpl_cases.jsonl"
  {
    echo "cat > /tmp/tpl_cases.jsonl <<'CASES_EOF'"; cat "$WORK/tpl_cases.jsonl"; echo "CASES_EOF"
    echo "cd \$SRC && mkdir -p \$BUILD && g++ -O2 -std=c++20 -Wall -Wextra -o \$BUILD/template_probe tools/template_probe.cpp host/chat_template.cpp host/json.cpp"
    echo "\$BUILD/template_probe < /tmp/tpl_cases.jsonl"
  } > "$WORK/run_tpl.sh"
  tools/dev.sh sync
  tools/dev.sh run < "$WORK/run_tpl.sh" > "$WORK/tpl_probe.jsonl"
  oracle_check gates/template_gate.py "$WORK/tpl_cases.jsonl" "$WORK/tpl_probe.jsonl" || status=1
fi

if [[ $which == all || $which == scan ]]; then
  tools/dev.sh sync
  tools/dev.sh run <<'X' || status=1
cd $SRC && mkdir -p $BUILD && g++ -O2 -std=c++20 -Wall -Wextra -o $BUILD/tool_scan_test tests/tool_scan_test.cpp host/tool_scan.cpp host/json.cpp && $BUILD/tool_scan_test
X
fi

exit $status
