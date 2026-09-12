"""Invoke real Core help; fail closed if it tries Pro, credentials, network, or children."""
import json
from pathlib import Path
import sys

events = []
def guard(event, args):
    if event == "import" and (args[0] == "orze_pro" or args[0].startswith("orze_pro.")):
        events.append("Pro import")
        raise RuntimeError("help must not import Pro or its license chain")
    if event == "open" and isinstance(args[0], (str, bytes)):
        name = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).name
        if name in (".env", ".orze-pro.key", "credentials", "credentials.json", "id_rsa", "id_ed25519"):
            events.append("credential path")
            raise RuntimeError("help must not read credential paths")
    if event in ("subprocess.Popen", "os.system", "socket.connect"):
        events.append(event)
        raise RuntimeError("help must not start a child or contact a service")

sys.addaudithook(guard)
from orze.cli import main
sys.argv = ["orze", "--help"]
try:
    result = main()
except SystemExit as exc:
    result = exc.code
assert result in (0, None)
assert events == []
assert not any(n == "orze_pro" or n.startswith("orze_pro.") for n in sys.modules)
print("HELP_ISOLATION=" + json.dumps({"core_help_exit": 0, "forbidden_events": [],
      "pro_imported": False, "guard_replaces_no_product_function": True}, sort_keys=True))

