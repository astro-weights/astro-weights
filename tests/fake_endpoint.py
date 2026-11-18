"""A fake OpenAI-compatible chat server for tests: no network beyond 127.0.0.1."""

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Reply = tuple[int, str]
Responder = Callable[[dict[str, Any]], Reply]


def ok(
    content: Any = "2", *, finish: str = "stop", usage: Any = None, **extra: Any
) -> Reply:
    obj: dict[str, Any] = {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"completion_tokens": 3} if usage is None else usage,
    }
    obj.update(extra)
    return 200, json.dumps(obj)


class Server:
    def __init__(self, respond: Responder) -> None:
        self.respond = respond
        self.requests: list[dict[str, Any]] = []
        self.models: Reply = (200, json.dumps({"data": []}))
        self.location: str | None = None


@contextmanager
def serve(respond: Responder) -> Iterator[tuple[str, Server]]:
    state = Server(respond)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, reply: Reply) -> None:
            data = reply[1].encode()
            self.send_response(reply[0])
            self.send_header("Content-Length", str(len(data)))
            if state.location is not None:
                self.send_header("Location", state.location)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            state.requests.append(
                {"path": self.path, "auth": self.headers["Authorization"]}
            )
            self._send(state.models)

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(raw)
            state.requests.append(
                {"path": self.path, "auth": self.headers["Authorization"], "body": body}
            )
            try:
                self._send(state.respond(body))
            except (BrokenPipeError, ConnectionError):
                pass

        def log_message(self, format: str, *args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def tool_reply(
    *calls: tuple[str, dict[str, Any] | str],
    tokens: tuple[int, int] = (10, 5),
    content: str | None = None,
) -> Reply:
    """A chat reply that asks for tool calls ``(name, arguments)``, in order."""
    made = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args if isinstance(args, str) else json.dumps(args),
            },
        }
        for i, (name, args) in enumerate(calls)
    ]
    usage = {
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
        "total_tokens": sum(tokens),
    }
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if made:
        message["tool_calls"] = made
    obj = {
        "choices": [
            {
                "message": message,
                "finish_reason": "tool_calls" if made else "stop",
            }
        ],
        "usage": usage,
    }
    return 200, json.dumps(obj)
