"""
Differential Push/Pull Client with Resumable Transfer Protocol.
Synchronizes Merkle DAG histories transferring only missing content-addressed blocks.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from synapsefs.core.cas import ContentAddressableStore
from synapsefs.core.dag import Commit, RepositoryDAG
from synapsefs.net.protocol import BlockDiffingNegotiator


class SyncClient:
    """
    Sync client executing differential push and pull operations over HTTP.
    """

    def __init__(self, dag: RepositoryDAG, timeout: float = 30.0):
        self.dag = dag
        self.cas = dag.cas
        self.timeout = timeout

    def _http_get_json(self, url: str) -> Dict[str, Any]:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8"))

    def _http_get_bytes(self, url: str) -> bytes:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def _http_post_bytes(self, url: str, data: bytes) -> Dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(data))},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_post_json(self, url: str, data: dict) -> Dict[str, Any]:
        payload = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def pull(self, remote_url: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """
        Pulls branch history and missing content blocks from remote peer.
        """
        remote_url = remote_url.rstrip("/")
        # 1. Fetch remote refs
        remote_refs = self._http_get_json(f"{remote_url}/api/v1/refs")
        target_branch = branch or remote_refs.get("current_branch") or "main"
        remote_commit_id = remote_refs.get("branches", {}).get(target_branch)

        if not remote_commit_id:
            raise ValueError(f"Branch '{target_branch}' does not exist on remote {remote_url}")

        local_branch, local_commit_id = self.dag.get_head_ref()
        if local_commit_id == remote_commit_id:
            return {"status": "Already up to date", "transferred_blocks": 0, "branch": target_branch}

        # 2. Fetch remote commit and manifest objects
        queue = [remote_commit_id]
        visited_commits = set()
        transferred_bytes = 0
        structural_blocks = 0

        while queue:
            cid = queue.pop(0)
            if not cid or cid in visited_commits:
                continue
            visited_commits.add(cid)

            if not self.cas.exists(cid):
                c_bytes = self._http_get_bytes(f"{remote_url}/api/v1/objects/{cid}")
                self.cas.put(c_bytes)
                transferred_bytes += len(c_bytes)
                structural_blocks += 1

            c_obj = Commit.from_bytes(self.cas.get(cid))
            m_hash = c_obj.manifest_hash
            if m_hash and not self.cas.exists(m_hash):
                m_bytes = self._http_get_bytes(f"{remote_url}/api/v1/objects/{m_hash}")
                self.cas.put(m_bytes)
                transferred_bytes += len(m_bytes)
                structural_blocks += 1

            for pid in c_obj.parent_ids:
                if pid and not self.cas.exists(pid):
                    queue.append(pid)

        # 3. Discover required data blobs
        required_objects = BlockDiffingNegotiator.collect_required_objects(self.dag, remote_commit_id)
        local_existing_objects = self.cas.list_all_objects()
        missing_objects = BlockDiffingNegotiator.compute_missing_objects(
            required_objects, local_existing_objects
        )

        total_transferred_blocks = structural_blocks + len(missing_objects)

        # 4. Stream missing data blocks
        for obj_hash in missing_objects:
            blob_bytes = self._http_get_bytes(f"{remote_url}/api/v1/objects/{obj_hash}")
            self.cas.put(blob_bytes)
            transferred_bytes += len(blob_bytes)

        # 5. Update local branch ref
        self.dag.update_branch(target_branch, remote_commit_id)
        if local_branch == target_branch:
            self.dag.set_head_branch(target_branch)

        return {
            "status": "Success",
            "branch": target_branch,
            "commit_id": remote_commit_id,
            "transferred_blocks": total_transferred_blocks,
            "transferred_bytes": transferred_bytes,
        }

    def push(self, remote_url: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """
        Pushes branch history and missing content blocks to remote peer.
        """
        remote_url = remote_url.rstrip("/")
        curr_branch, head_commit_id = self.dag.get_head_ref()
        target_branch = branch or curr_branch or "main"

        local_commit_id = self.dag.get_branch_commit(target_branch) or head_commit_id
        if not local_commit_id:
            raise ValueError(f"Cannot push: local branch '{target_branch}' has no commits")

        # 1. Fetch remote object inventory
        remote_inventory = self._http_get_json(f"{remote_url}/api/v1/objects/list")
        remote_objects = set(remote_inventory.get("objects", []))

        # 2. Collect local required objects for this branch
        local_required = BlockDiffingNegotiator.collect_required_objects(self.dag, local_commit_id)
        missing_on_remote = BlockDiffingNegotiator.compute_missing_objects(
            local_required, remote_objects
        )

        transferred_bytes = 0

        # 3. Stream missing objects to remote
        for obj_hash in missing_on_remote:
            obj_bytes = self.cas.get(obj_hash)
            self._http_post_bytes(f"{remote_url}/api/v1/objects/{obj_hash}", obj_bytes)
            transferred_bytes += len(obj_bytes)

        # 4. Update remote branch ref
        self._http_post_json(
            f"{remote_url}/api/v1/refs/{target_branch}",
            {"commit_id": local_commit_id}
        )

        return {
            "status": "Success",
            "branch": target_branch,
            "commit_id": local_commit_id,
            "transferred_blocks": len(missing_on_remote),
            "transferred_bytes": transferred_bytes,
        }
