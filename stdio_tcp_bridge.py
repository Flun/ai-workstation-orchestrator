"""Expose a JSON-lines stdio agent protocol over a local TCP port.

Each client gets a fresh child process.  The bridge itself stays resident so
CLI agents such as Pi RPC and OMP ACP can be managed like the other services.
"""

from __future__ import annotations

import argparse
import os
import socketserver
import subprocess
import threading


class AgentHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        command = self.server.command  # type: ignore[attr-defined]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.expanduser("~"),
            start_new_session=True,
        )

        def pump(source, destination) -> None:
            try:
                while True:
                    chunk = os.read(source.fileno(), 65536)
                    if not chunk:
                        break
                    destination.sendall(chunk)
            except (OSError, BrokenPipeError):
                pass

        stdout_thread = threading.Thread(target=pump, args=(process.stdout, self.request), daemon=True)
        stdout_thread.start()

        def drain_stderr() -> None:
            for line in process.stderr:
                print(line.decode("utf-8", errors="replace").rstrip(), flush=True)

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()
        try:
            while True:
                chunk = self.request.recv(65536)
                if not chunk:
                    break
                process.stdin.write(chunk)
                process.stdin.flush()
        except (OSError, BrokenPipeError):
            pass
        finally:
            try:
                process.stdin.close()
            except (OSError, BrokenPipeError):
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
            stdout_thread.join(timeout=1)


class ThreadingAgentServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required")
    with ThreadingAgentServer((args.host, args.port), AgentHandler) as server:
        server.command = command
        print(f"stdio bridge listening on {args.host}:{args.port} -> {' '.join(command)}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
