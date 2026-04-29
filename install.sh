#!/usr/bin/env bash
# Orze — one-line setup
#
# Usage:
#   curl -sL https://orze.ai/install | bash
#   curl -sL https://orze.ai/install | bash -s /path/to/project
#
# With pro:
#   ORZE_PRO_KEY=ORZE-PRO-xxx curl -sL https://orze.ai/install | bash
#
# With API key:
#   GEMINI_API_KEY=AIza... curl -sL https://orze.ai/install | bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${GREEN}[+]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
step()  { echo -e "\n${BOLD}${CYAN}==> $*${RESET}"; }

# When piped via curl|bash, stdin is the script itself.
# All subprocesses must use </dev/null and interactive prompts use /dev/tty.
HAS_TTY=false
if [ -e /dev/tty ]; then
    HAS_TTY=true
fi

echo ""
echo -e "${BOLD}orze — GPU Experiment Orchestrator${RESET}"
echo "==================================="

# --- 1. Python ---------------------------------------------------------------
step "Checking Python..."
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=${ver%%.*}
        minor=${ver#*.}
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON="$cmd"
            info "Python $ver ($cmd)"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    warn "Python >= 3.9 required. Install it first."
    exit 1
fi

# --- 2. Install orze ---------------------------------------------------------
step "Installing orze..."
INSTALL_CMD="$PYTHON -m pip install --quiet"

if command -v orze &>/dev/null; then
    ORZE_VER=$(orze --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "0.0.0")
    info "orze $ORZE_VER already installed — upgrading"
    $INSTALL_CMD --upgrade orze </dev/null 2>/dev/null || true
else
    $INSTALL_CMD orze </dev/null 2>&1 || {
        warn "pip install failed. Try: $PYTHON -m pip install orze"
        exit 1
    }
fi
ORZE_VER=$(orze --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
info "orze $ORZE_VER"

# --- 3. orze-pro (optional) --------------------------------------------------
step "Checking orze-pro..."

INSTALL_PRO=false
PRO_KEY="${ORZE_PRO_KEY:-}"

if orze pro status </dev/null 2>/dev/null | grep -qi "licensed"; then
    info "orze-pro already activated"
    INSTALL_PRO=true
elif [ -n "$PRO_KEY" ]; then
    INSTALL_PRO=true
    info "License key found in ORZE_PRO_KEY"
elif [ "$HAS_TTY" = true ]; then
    echo ""
    echo -e "  ${DIM}orze-pro adds autopilot: autonomous research agents,"
    echo -e "  auto-fix, code evolution, The Professor, 7 FSM procedures.${RESET}"
    echo ""
    echo -n "  Enter orze-pro license key (or Enter to skip): "
    read -r PRO_KEY </dev/tty
    [ -n "$PRO_KEY" ] && INSTALL_PRO=true
else
    info "No ORZE_PRO_KEY set — skipping pro"
fi

if [ "$INSTALL_PRO" = true ]; then
    if [ -n "$PRO_KEY" ]; then
        PRO_INDEX="--extra-index-url https://admin:${PRO_KEY}@pypi.orze.ai/simple/"
    else
        PRO_INDEX=""
    fi
    if ! $PYTHON -c "import orze_pro" </dev/null &>/dev/null; then
        info "Installing orze-pro..."
        $INSTALL_CMD orze-pro $PRO_INDEX </dev/null 2>&1 || {
            warn "orze-pro install failed. Check your license key or visit orze.ai/pro"
        }
    else
        info "Upgrading orze-pro..."
        $INSTALL_CMD --upgrade orze-pro $PRO_INDEX </dev/null 2>/dev/null || true
    fi
    if [ -n "$PRO_KEY" ]; then
        info "Activating license..."
        orze pro activate "$PRO_KEY" </dev/null 2>&1 | head -1 || true
    fi
fi

# --- 4. API key ---------------------------------------------------------------
step "Checking API keys..."

API_KEY=""
API_VAR=""

for var in ANTHROPIC_API_KEY GEMINI_API_KEY OPENAI_API_KEY; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        API_KEY="$val"
        API_VAR="$var"
        info "Found $var"
        break
    fi
done

if [ -z "$API_KEY" ] && [ "$HAS_TTY" = true ]; then
    echo ""
    echo -e "  ${DIM}An API key enables LLM-powered codebase analysis and idea generation.${RESET}"
    echo -n "  Enter API key (Anthropic/Gemini/OpenAI, or Enter to skip): "
    read -r API_KEY </dev/tty
    if [ -n "$API_KEY" ]; then
        case "$API_KEY" in
            sk-ant*)  API_VAR="ANTHROPIC_API_KEY" ;;
            AIza*)    API_VAR="GEMINI_API_KEY" ;;
            sk-*)     API_VAR="OPENAI_API_KEY" ;;
            *)        API_VAR="GEMINI_API_KEY" ;;
        esac
        info "Detected: $API_VAR"
    fi
fi

# --- 5. Project path ---------------------------------------------------------
PROJECT_PATH="${1:-.}"

# Export API key so orze init can detect it
if [ -n "$API_KEY" ] && [ -n "$API_VAR" ]; then
    export "$API_VAR=$API_KEY"
fi

# --- 6. Auto-detect leader/follower -------------------------------------------
RESULTS_DIR="$PROJECT_PATH/orze_results"
HEARTBEAT_FILE="$RESULTS_DIR/.orze_leader.heartbeat"

if [ -f "$HEARTBEAT_FILE" ]; then
    # Another host wrote a heartbeat — check if it's fresh (< 5 min)
    LEADER_HOST=$($PYTHON -c "
import json, time, socket
try:
    hb = json.load(open('$HEARTBEAT_FILE'))
    age = time.time() - hb.get('ts', 0)
    leader = hb.get('host', '')
    me = socket.gethostname()
    if age < 300 and leader != me:
        print(leader)
except: pass
" 2>/dev/null)
    if [ -n "$LEADER_HOST" ]; then
        info "Active leader detected on $LEADER_HOST — this node will be a follower"
        export ORZE_FORCE_FOLLOWER=1
    fi
fi

# --- 7. Initialize ------------------------------------------------------------
step "Initializing project..."

orze init "$PROJECT_PATH" </dev/null || {
    warn "Initialization failed. Check the output above."
    exit 1
}

# Write API key to .env if provided manually
if [ -n "$API_KEY" ] && [ -n "$API_VAR" ]; then
    ENV_FILE="$PROJECT_PATH/.env"
    if [ -f "$ENV_FILE" ]; then
        if ! grep -q "^${API_VAR}=" "$ENV_FILE" 2>/dev/null; then
            echo "${API_VAR}=${API_KEY}" >> "$ENV_FILE"
            info "API key written to .env"
        fi
    fi
fi

# --- Done ---------------------------------------------------------------------
echo ""
echo -e "${BOLD}Done.${RESET}"
if [ -n "${ORZE_FORCE_FOLLOWER:-}" ]; then
    echo -e "  Running as ${CYAN}follower${RESET} (leader on $LEADER_HOST)."
    echo -e "  This node runs training only — roles run on the leader."
elif [ "$INSTALL_PRO" = true ]; then
    echo -e "  orze + orze-pro installed and running."
else
    echo -e "  orze installed and running."
    echo -e "  ${DIM}Upgrade to autopilot: pip install orze-pro${RESET}"
fi
echo ""
