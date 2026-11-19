"""Run a Python file inside the Blender that has the MCP for Blender add-on's server running (localhost:9876).

    python blender/run_in_blender.py blender/head.py [KEY=VALUE ...]

The file runs in Blender's own interpreter (`bpy` is available) with the KEY=VALUE pairs as the dict `ARGS`.
Anything the script prints comes back on stdout. Blender must be open with the server started
(sidebar N > "MCP for Blender" > connected).
"""
import json, os, socket, sys

HOST, PORT = "127.0.0.1", 9876


def call(command_type, params=None, timeout=600):
    with socket.create_connection((HOST, PORT), timeout=10) as s:
        s.settimeout(timeout)
        s.sendall(json.dumps({"type": command_type, "params": params or {}}).encode("utf-8"))
        buf = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            try:
                return json.loads(buf.decode("utf-8"))
            except ValueError:
                continue
    raise RuntimeError("no complete response from Blender")


def main():
    path = sys.argv[1]
    args = dict(a.split("=", 1) for a in sys.argv[2:])
    # the repo root, so a script can exec another one from blender/
    args.setdefault("root", os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace(chr(92), "/"))
    code = "ARGS = %r\n" % (args,) + open(path, encoding="utf-8").read()
    r = call("execute_code", {"code": code})
    if r.get("status") != "success":
        print("ERROR:", r.get("message"))
        sys.exit(1)
    print(r["result"].get("result", ""), end="")


if __name__ == "__main__":
    main()
