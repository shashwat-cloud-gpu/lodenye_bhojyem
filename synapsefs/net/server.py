"""
HTTP Sync Server Daemon for SynapseFS Peer-to-Peer Synchronization.
Built using standard library ThreadingHTTPServer for zero-dependency high portability.
"""

import json
import socketserver
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from synapsefs.core.cas import ContentAddressableStore
from synapsefs.core.dag import RepositoryDAG


class SynapseHTTPRequestHandler(BaseHTTPRequestHandler):
    dag: RepositoryDAG
    cas: ContentAddressableStore

    def _send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_json(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status=status)

    def do_GET(self) -> None:
        path = self.path

        if path == "/api/v1/ping":
            self._send_json({"status": "ok", "service": "SynapseFS"})
            return

        if path == "/api/v1/refs":
            branches = {}
            for b in self.dag.list_branches():
                branches[b] = self.dag.get_branch_commit(b)
            curr_branch, head_cid = self.dag.get_head_ref()
            self._send_json({
                "branches": branches,
                "current_branch": curr_branch,
                "head_commit": head_cid,
            })
            return

        if path == "/api/v1/objects/list":
            objs = list(self.cas.list_all_objects())
            self._send_json({"objects": objs, "count": len(objs)})
            return

        if path.startswith("/api/v1/objects/"):
            obj_hash = path.split("/api/v1/objects/")[1]
            if not self.cas.exists(obj_hash):
                self.send_error(HTTPStatus.NOT_FOUND, f"Object {obj_hash} not found")
                return

            try:
                data = self.cas.get(obj_hash)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def do_POST(self) -> None:
        path = self.path

        if path.startswith("/api/v1/objects/"):
            obj_hash = path.split("/api/v1/objects/")[1]
            content_len = int(self.headers.get("Content-Length", 0))
            if content_len <= 0:
                self._send_error_json("Missing or zero Content-Length")
                return

            data = self.rfile.read(content_len)
            try:
                saved_hash = self.cas.put(data)
                if saved_hash != obj_hash:
                    self._send_error_json(
                        f"Checksum mismatch: received {saved_hash} != {obj_hash}",
                        status=400
                    )
                    return
                self._send_json({"status": "stored", "hash": saved_hash})
            except Exception as e:
                self._send_error_json(str(e), status=500)
            return

        if path.startswith("/api/v1/refs/"):
            branch_name = path.split("/api/v1/refs/")[1]
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8"))
            commit_id = data.get("commit_id")
            if not commit_id or not self.cas.exists(commit_id):
                self._send_error_json(f"Invalid or missing commit {commit_id}", status=400)
                return

            self.dag.update_branch(branch_name, commit_id)
            self._send_json({"status": "updated", "branch": branch_name, "commit_id": commit_id})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def log_message(self, format, *args):
        # Suppress noisy standard HTTP access logging during CLI use
        pass


class SyncServer:
    """
    Spawns and manages background HTTP server for peer synchronization.
    """

    def __init__(self, dag: RepositoryDAG, host: str = "0.0.0.0", port: int = 8000):
        self.dag = dag
        self.cas = dag.cas
        self.host = host
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self, block: bool = True) -> None:
        handler_class = SynapseHTTPRequestHandler
        handler_class.dag = self.dag
        handler_class.cas = self.cas

        self.server = ThreadingHTTPServer((self.host, self.port), handler_class)
        print(f"[*] SynapseFS Peer Sync Server listening on http://{self.host}:{self.port}")

        if block:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                print("\n[*] Stopping sync server...")
                self.server.shutdown()
        else:
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
