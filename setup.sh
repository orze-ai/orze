#!/usr/bin/env bash
# Orze — zero-dependency setup
# Requirements: bash + curl (present on all Linux/macOS systems)
#
# Usage:
#   curl -sL https://orze.ai/setup.sh | bash
#   curl -sL https://orze.ai/setup.sh | bash -s /path/to/project
#
# With pro:
#   ORZE_PRO_KEY=ORZE-PRO-xxx curl -sL https://orze.ai/setup.sh | bash
#
# With API key:
#   GEMINI_API_KEY=AIza... curl -sL https://orze.ai/setup.sh | bash
set -euo pipefail

# --- Colors ---------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${GREEN}[+]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
step()  { echo -e "${BOLD}${CYAN}==> $*${RESET}"; }

echo ""
echo -e "${BOLD}Orze — GPU Experiment Orchestrator${RESET}"
echo "==================================="
echo ""

# --- Install uv if missing ------------------------------------------------
step "Checking for uv..."
if command -v uv &>/dev/null; then
    info "uv already installed ($(uv --version))"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        warn "uv installed but not on PATH. Add ~/.local/bin to your PATH."
        exit 1
    fi
    info "uv installed ($(uv --version))"
fi

# --- Install orze ----------------------------------------------------------
step "Installing orze..."
uv tool install --quiet --force "orze>=3.2.4"
info "orze installed ($(orze --version 2>/dev/null || echo 'unknown'))"

# --- orze-pro (optional) ---------------------------------------------------
step "Checking orze-pro..."

INSTALL_PRO=false
PRO_KEY="${ORZE_PRO_KEY:-}"

# Already installed?
if orze pro status 2>/dev/null | grep -q "Licensed"; then
    info "orze-pro already activated"
    INSTALL_PRO=true
elif [ -n "$PRO_KEY" ]; then
    # Key provided via env var
    INSTALL_PRO=true
    info "License key found in ORZE_PRO_KEY"
elif [ -t 0 ]; then
    # Interactive: ask
    echo ""
    echo -e "  ${DIM}orze-pro adds: autonomous research agents, The Professor,"
    echo -e "  code evolution, bug fixer, 7 FSM procedures.${RESET}"
    echo ""
    echo -n "  Enter orze-pro license key (or Enter to skip): "
    read -r PRO_KEY
    if [ -n "$PRO_KEY" ]; then
        INSTALL_PRO=true
    fi
else
    info "Skipping orze-pro (non-interactive, no ORZE_PRO_KEY)"
    echo -e "  ${DIM}To install later: pip install orze-pro${RESET}"
fi

if [ "$INSTALL_PRO" = true ]; then
    # Install package
    if ! python3 -c "import orze_pro" &>/dev/null 2>&1; then
        info "Installing orze-pro..."
        uv tool install --quiet --force "orze-pro>=0.2.4" 2>/dev/null || \
            uv pip install --quiet "orze-pro>=0.2.4" 2>/dev/null || \
            pip install --quiet "orze-pro>=0.2.4" 2>/dev/null || true
    fi

    # Activate license
    if [ -n "$PRO_KEY" ]; then
        info "Activating license..."
        orze pro activate "$PRO_KEY" 2>&1 | head -1 || true
    fi
fi

# --- API key ---------------------------------------------------------------
step "Checking API keys..."

API_KEY=""
API_VAR=""

# Check env
for var in GEMINI_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY; do
    val="${!var:-}"
    if [ -n "$val" ]; then
        API_KEY="$val"
        API_VAR="$var"
        info "Found $var in environment"
        break
    fi
done

# Ask if not found (interactive only)
if [ -z "$API_KEY" ] && [ -t 0 ]; then
    echo ""
    echo -e "  ${DIM}An API key enables autonomous idea generation.${RESET}"
    echo -n "  Enter API key (Gemini/Anthropic/OpenAI, or Enter to skip): "
    read -r API_KEY
    if [ -n "$API_KEY" ]; then
        case "$API_KEY" in
            AIza*)    API_VAR="GEMINI_API_KEY" ;;
            sk-ant*)  API_VAR="ANTHROPIC_API_KEY" ;;
            sk-*)     API_VAR="OPENAI_API_KEY" ;;
            *)        API_VAR="GEMINI_API_KEY" ;;
        esac
        info "Detected: $API_VAR"
    fi
fi

if [ -z "$API_KEY" ] && [ ! -t 0 ]; then
    info "No API key (set GEMINI_API_KEY or ANTHROPIC_API_KEY before running)"
fi

# --- Auto-detect shared storage -------------------------------------------
PROJECT_PATH="${1:-}"

if [ -z "$PROJECT_PATH" ]; then
    step "Detecting shared storage..."
    SHARED_MOUNT=""
    if [ -f /proc/mounts ]; then
        SHARED_MOUNT=$(awk '$3 ~ /^(lustre|nfs|nfs4|efs|gpfs|beegfs|glusterfs|pvfs2|fuse\.s3fs|fuse\.goofys)$/ || $1 ~ /@tcp:/ { print $2 }' /proc/mounts | head -1)
    fi
    if [ -z "$SHARED_MOUNT" ] && command -v mount &>/dev/null; then
        SHARED_MOUNT=$(mount | awk '$5 == "nfs" || $5 == "nfs4" { print $3 }' | head -1)
    fi
    if [ -n "$SHARED_MOUNT" ]; then
        info "Detected shared storage: ${SHARED_MOUNT}"
        PROJECT_PATH="$SHARED_MOUNT"
    else
        info "No shared storage detected, using current directory"
        PROJECT_PATH="."
    fi
fi

# --- Initialize project ---------------------------------------------------
step "Initializing project at ${PROJECT_PATH}..."

# Export API key so orze --init can detect it
if [ -n "$API_KEY" ] && [ -n "$API_VAR" ]; then
    export "$API_VAR=$API_KEY"
fi

if ! orze --init "$PROJECT_PATH"; then
    warn "orze --init failed. Check the output above for details."
    exit 1
fi

# Write API key to .env
if [ -n "$API_KEY" ] && [ -n "$API_VAR" ]; then
    cd "$PROJECT_PATH"
    # Remove placeholder line if present, add real key
    if [ -f .env ]; then
        grep -v "^# .*${API_VAR}" .env > .env.tmp 2>/dev/null || true
        mv .env.tmp .env
    fi
    echo "${API_VAR}=${API_KEY}" >> .env
    info "API key written to .env"
    cd - >/dev/null
fi

# --- Done ------------------------------------------------------------------
echo ""
echo "==========================================="
if [ "$INSTALL_PRO" = true ]; then
    info "Orze + Pro installed and initialized."
else
    info "Orze installed and initialized."
fi
echo -e "  ${BOLD}Project:${RESET} ${PROJECT_PATH}"
echo ""
echo -e "  ${BOLD}Next steps:${RESET}"
echo -e "    cd ${PROJECT_PATH}"
echo -e "    ${CYAN}# In Claude Code / Cursor / Codex CLI:${RESET}"
echo -e "    do @ORZE-AGENT.md"
echo -e "    ${CYAN}# Or run directly:${RESET}"
echo -e "    orze --check"
if [ "$INSTALL_PRO" != true ]; then
    echo ""
    echo -e "  ${DIM}Upgrade to autopilot: pip install orze-pro${RESET}"
fi
echo ""
