"""
Merkle DAG, Commit Graph, Branching, and 3-Way Merge Engine.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from synapsefs.core.cas import ContentAddressableStore, compute_hash


class Manifest:
    """
    Represents the complete structure of a checkpoint (header metadata + tensor blobs).
    Serialized to canonical JSON and stored in CAS.
    """

    def __init__(
        self,
        tensors: Dict[str, Dict[str, Any]],
        metadata: Optional[Dict[str, str]] = None,
        config_hash: Optional[str] = None,
        base_commit_id: Optional[str] = None,
    ):
        self.tensors = tensors  # tensor_name -> {dtype, shape, offsets, blob_hash, is_delta, residual_hash, perm_hash, base_tensor_name}
        self.metadata = metadata or {}
        self.config_hash = config_hash
        self.base_commit_id = base_commit_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "manifest",
            "metadata": self.metadata,
            "config_hash": self.config_hash,
            "base_commit_id": self.base_commit_id,
            "tensors": self.tensors,
        }

    def serialize(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Manifest":
        d = json.loads(data.decode("utf-8"))
        return cls(
            tensors=d.get("tensors", {}),
            metadata=d.get("metadata", {}),
            config_hash=d.get("config_hash"),
            base_commit_id=d.get("base_commit_id"),
        )


class Commit:
    """
    Represents an immutable commit in the Merkle DAG.
    """

    def __init__(
        self,
        manifest_hash: str,
        parent_ids: List[str],
        message: str,
        author: str = "user",
        timestamp: Optional[float] = None,
    ):
        self.manifest_hash = manifest_hash
        self.parent_ids = parent_ids
        self.message = message
        self.author = author
        self.timestamp = timestamp if timestamp is not None else time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "commit",
            "manifest_hash": self.manifest_hash,
            "parent_ids": self.parent_ids,
            "message": self.message,
            "author": self.author,
            "timestamp": self.timestamp,
        }

    def serialize(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Commit":
        d = json.loads(data.decode("utf-8"))
        return cls(
            manifest_hash=d["manifest_hash"],
            parent_ids=d.get("parent_ids", []),
            message=d.get("message", ""),
            author=d.get("author", "user"),
            timestamp=d.get("timestamp", 0.0),
        )


class RepositoryDAG:
    """
    Manages repository references, commit lineage, branching, and Merkle DAG traversal.
    """

    def __init__(self, repo_root: Path, cas: ContentAddressableStore):
        self.repo_root = Path(repo_root)
        self.synapse_dir = self.repo_root / ".synapsefs"
        self.refs_heads_dir = self.synapse_dir / "refs" / "heads"
        self.head_file = self.synapse_dir / "HEAD"
        self.cas = cas

        self.refs_heads_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def init(cls, repo_root: Path) -> "RepositoryDAG":
        """Initializes a new .synapsefs repository."""
        synapse_dir = repo_root / ".synapsefs"
        cas = ContentAddressableStore(synapse_dir)
        dag = cls(repo_root, cas)

        # Set default HEAD to point to refs/heads/main
        if not dag.head_file.exists():
            dag.head_file.write_text("ref: refs/heads/main\n", encoding="utf-8")
        return dag

    def get_head_ref(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Reads HEAD. Returns (branch_name, commit_id).
        If detached, branch_name is None.
        """
        if not self.head_file.exists():
            return None, None
        content = self.head_file.read_text(encoding="utf-8").strip()
        if content.startswith("ref: "):
            ref_path = content[5:].strip()
            branch_name = ref_path.replace("refs/heads/", "")
            branch_file = self.synapse_dir / ref_path
            commit_id = branch_file.read_text(encoding="utf-8").strip() if branch_file.exists() else None
            return branch_name, commit_id
        else:
            return None, content if content else None

    def set_head_branch(self, branch_name: str) -> None:
        """Sets HEAD to point to a named branch."""
        self.head_file.write_text(f"ref: refs/heads/{branch_name}\n", encoding="utf-8")

    def set_head_detached(self, commit_id: str) -> None:
        """Sets HEAD to a detached commit hash."""
        self.head_file.write_text(f"{commit_id}\n", encoding="utf-8")

    def get_branch_commit(self, branch_name: str) -> Optional[str]:
        """Gets current commit hash of a branch."""
        branch_file = self.refs_heads_dir / branch_name
        if branch_file.is_file():
            return branch_file.read_text(encoding="utf-8").strip()
        return None

    def update_branch(self, branch_name: str, commit_id: str) -> None:
        """Updates or creates a branch ref pointing to commit_id."""
        branch_file = self.refs_heads_dir / branch_name
        branch_file.parent.mkdir(parents=True, exist_ok=True)
        branch_file.write_text(f"{commit_id}\n", encoding="utf-8")

    def list_branches(self) -> List[str]:
        """Lists all existing branches."""
        if not self.refs_heads_dir.exists():
            return []
        return sorted([f.name for f in self.refs_heads_dir.iterdir() if f.is_file()])

    def create_commit(
        self,
        manifest_hash: str,
        message: str,
        parent_ids: Optional[List[str]] = None,
        author: str = "user",
    ) -> str:
        """
        Creates a new commit and updates the current branch / HEAD.
        """
        current_branch, current_commit = self.get_head_ref()
        if parent_ids is None:
            parent_ids = [current_commit] if current_commit else []

        commit = Commit(
            manifest_hash=manifest_hash,
            parent_ids=parent_ids,
            message=message,
            author=author,
        )
        commit_bytes = commit.serialize()
        commit_id = self.cas.put(commit_bytes)

        if current_branch:
            self.update_branch(current_branch, commit_id)
        else:
            self.set_head_detached(commit_id)

        return commit_id

    def get_commit(self, commit_id: str) -> Commit:
        """Retrieves and deserializes a commit object."""
        commit_bytes = self.cas.get(commit_id)
        return Commit.from_bytes(commit_bytes)

    def get_manifest(self, manifest_hash: str) -> Manifest:
        """Retrieves and deserializes a manifest object."""
        manifest_bytes = self.cas.get(manifest_hash)
        return Manifest.from_bytes(manifest_bytes)

    def walk_history(self, start_commit_id: Optional[str] = None) -> List[Tuple[str, Commit]]:
        """Walks commit lineage starting from start_commit_id (or HEAD) in reverse topological order."""
        if start_commit_id is None:
            _, start_commit_id = self.get_head_ref()
            if start_commit_id is None:
                return []

        visited: Set[str] = set()
        queue: List[str] = [start_commit_id]
        history: List[Tuple[str, Commit]] = []

        while queue:
            cid = queue.pop(0)
            if cid in visited or not cid:
                continue
            visited.add(cid)
            commit = self.get_commit(cid)
            history.append((cid, commit))
            for pid in commit.parent_ids:
                if pid not in visited:
                    queue.append(pid)

        return history

    def find_lowest_common_ancestor(self, commit_a: str, commit_b: str) -> Optional[str]:
        """Finds Lowest Common Ancestor (LCA) between two commits in the DAG."""
        if commit_a == commit_b:
            return commit_a

        # Get all ancestors of commit_a
        ancestors_a: Set[str] = set()
        queue = [commit_a]
        while queue:
            curr = queue.pop(0)
            if curr in ancestors_a or not curr:
                continue
            ancestors_a.add(curr)
            try:
                commit = self.get_commit(curr)
                queue.extend(commit.parent_ids)
            except Exception:
                pass

        # Traverse commit_b ancestors via BFS to find first common ancestor
        queue = [commit_b]
        visited_b: Set[str] = set()
        while queue:
            curr = queue.pop(0)
            if curr in visited_b or not curr:
                continue
            visited_b.add(curr)
            if curr in ancestors_a:
                return curr
            try:
                commit = self.get_commit(curr)
                queue.extend(commit.parent_ids)
            except Exception:
                pass

        return None

    def merge_branches(
        self,
        source_branch: str,
        message: Optional[str] = None,
        author: str = "user"
    ) -> Tuple[str, str]:
        """
        Merges source_branch into the current active branch.
        Returns (status, merge_commit_id). Status can be 'Fast-forward', 'Already-up-to-date', or 'Merged'.
        """
        current_branch, current_commit = self.get_head_ref()
        if not current_branch:
            raise ValueError("Cannot merge into detached HEAD")
        if not current_commit:
            raise ValueError(f"Current branch '{current_branch}' has no commits")

        source_commit = self.get_branch_commit(source_branch)
        if not source_commit:
            raise ValueError(f"Source branch '{source_branch}' not found")

        lca = self.find_lowest_common_ancestor(current_commit, source_commit)

        if lca == source_commit:
            return "Already-up-to-date", current_commit

        if lca == current_commit:
            # Fast-forward merge
            self.update_branch(current_branch, source_commit)
            return "Fast-forward", source_commit

        # True 3-Way Merge
        lca_commit = self.get_commit(lca) if lca else None
        lca_manifest = self.get_manifest(lca_commit.manifest_hash) if lca_commit else None

        head_commit = self.get_commit(current_commit)
        head_manifest = self.get_manifest(head_commit.manifest_hash)

        source_commit_obj = self.get_commit(source_commit)
        source_manifest = self.get_manifest(source_commit_obj.manifest_hash)

        # 3-Way Manifest Reconciliation
        merged_tensors: Dict[str, Any] = {}
        all_tensor_names = set(head_manifest.tensors.keys()) | set(source_manifest.tensors.keys())

        for name in all_tensor_names:
            in_head = head_manifest.tensors.get(name)
            in_source = source_manifest.tensors.get(name)
            in_lca = lca_manifest.tensors.get(name) if lca_manifest else None

            if in_head == in_source:
                merged_tensors[name] = in_head
            elif in_head == in_lca and in_source is not None:
                # Changed only in source -> take source
                merged_tensors[name] = in_source
            elif in_source == in_lca and in_head is not None:
                # Changed only in head -> take head
                merged_tensors[name] = in_head
            else:
                # Modified in both: prefer head or source if head is base
                merged_tensors[name] = in_head if in_head else in_source

        merged_manifest = Manifest(
            tensors=merged_tensors,
            metadata=head_manifest.metadata or source_manifest.metadata,
            config_hash=head_manifest.config_hash or source_manifest.config_hash,
            base_commit_id=current_commit,
        )
        merged_manifest_bytes = merged_manifest.serialize()
        merged_manifest_hash = self.cas.put(merged_manifest_bytes)

        merge_msg = message or f"Merge branch '{source_branch}' into {current_branch}"
        merge_commit_id = self.create_commit(
            manifest_hash=merged_manifest_hash,
            message=merge_msg,
            parent_ids=[current_commit, source_commit],
            author=author,
        )

        return "Merged", merge_commit_id
