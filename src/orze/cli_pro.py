"""Pro license management commands.

Calling spec:
    from orze.cli_pro import pro_activate, pro_status, pro_deactivate

    pro_activate(key=None)       # activate with key or prompt interactively
    pro_status()                 # print license status
    pro_deactivate(force=False)  # remove saved key (prompts unless force)
"""

from pathlib import Path


def pro_activate(key=None):
    """Activate license. Accepts key as argument or prompts interactively."""
    # Check if orze-pro is installed
    try:
        import orze_pro
    except ImportError:
        print("orze-pro is not installed.")
        print("Install it with: pip install orze-pro")
        return

    key_path = Path.home() / ".orze-pro.key"

    # If no key provided, check existing or prompt
    if not key:
        if key_path.exists():
            from orze_pro.license import check_license, license_info
            existing = check_license()
            if existing:
                print(f"Already activated: {license_info()}")
                try:
                    resp = input("Replace with a new key? [y/N] ").strip().lower()
                except EOFError:
                    print("To replace: orze pro activate NEW-KEY")
                    return
                if resp != "y":
                    return

        print("Enter your license key (get one at orze.ai/pro):")
        try:
            key = input("> ").strip()
        except EOFError:
            print("Usage: orze pro activate ORZE-PRO-xxx...")
            return
        if not key:
            print("No key entered.")
            return

    # Verify before saving
    from orze_pro.license import verify_key, verify_key_reason
    payload = verify_key(key)
    if payload is None:
        reason = verify_key_reason(key)
        print(f"\033[31m{reason}\033[0m")
        print()
        print("Troubleshooting:")
        print("  - Make sure you copied the entire key from your email")
        print("  - The key should start with ORZE-PRO- and contain a dot (.)")
        print("  - Run 'orze pro activate' and paste the key when prompted")
        print("  - Contact support@orze.ai if the problem persists")
        return

    # Save
    key_path.write_text(key)
    key_path.chmod(0o600)

    customer = payload.get("customer", "?")
    tier = payload.get("tier", "pro")
    expires = payload.get("expires", "never")

    print(f"\033[32m\u2713 Licensed to {customer} ({tier}), expires {expires}\033[0m")
    print(f"  Key saved to {key_path}")
    print(f"  Pro features activate automatically — no config changes needed.")


def pro_status():
    """Show license status."""
    try:
        import orze_pro
    except ImportError:
        print("orze-pro is not installed.")
        print("Install it with: pip install orze-pro")
        return

    from orze.extensions import has_pro, pro_version, pro_features
    from orze_pro.license import license_info, is_licensed

    print(f"orze-pro {pro_version()}")
    print(f"Status: {license_info()}")
    if is_licensed():
        _descriptions = {
            "role_runner": "Multi-agent orchestration",
            "agents.research": "Autonomous research agents",
            "agents.research_context": "Context builder for research LLM",
            "agents.research_llm": "LLM backends (Gemini/OpenAI/Anthropic)",
            "agents.code_evolution": "Auto-evolve code on plateau",
            "agents.meta_research": "Meta-level strategy adjustment",
            "agents.bug_fixer": "Auto-fix failed experiments",
            "agents.bot": "Interactive Telegram/Slack bot",
        }
        features = pro_features()
        print(f"Features: {len(features)} available")
        for f in features:
            desc = _descriptions.get(f, "")
            print(f"  \033[32m\u2713\033[0m {f:30s} {desc}")
    else:
        print()
        print("Activate with: orze pro activate")


def pro_deactivate(force=False):
    """Remove saved license key."""
    key_path = Path.home() / ".orze-pro.key"
    if key_path.exists():
        if not force:
            try:
                resp = input("Remove license key? Pro features will be disabled. [y/N] ").strip().lower()
            except EOFError:
                resp = "n"
            if resp != "y":
                print("Cancelled. (Use -y to skip confirmation)")
                return
        key_path.unlink()
        print(f"License key removed from {key_path}")
        print("Pro features deactivated.")
    else:
        print("No saved license key found.")
