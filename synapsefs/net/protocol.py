"""
Differential Block-Diffing Protocol Specification.
Determines exact set of missing content-addressed objects between peers.
"""

from typing import Any, Dict, List, Set
from synapsefs.core.dag import Commit, Manifest, RepositoryDAG


class BlockDiffingNegotiator:
    """
    Computes minimal missing block sets between local and remote Merkle DAGs.
    """

    @staticmethod
    def collect_required_objects(dag: RepositoryDAG, head_commit_id: str) -> Set[str]:
        """
        Traverses the DAG from head_commit_id and collects all required object hashes:
        - Commits
        - Manifests
        - Raw tensor blobs
        - Compressed residual deltas
        - Permutation maps
        """
        required_hashes: Set[str] = set()
        queue = [head_commit_id]
        visited_commits: Set[str] = set()

        while queue:
            cid = queue.pop(0)
            if not cid or cid in visited_commits:
                continue
            visited_commits.add(cid)
            required_hashes.add(cid)

            try:
                commit = dag.get_commit(cid)
                for pid in commit.parent_ids:
                    if pid and pid not in visited_commits:
                        queue.append(pid)

                m_hash = commit.manifest_hash
                if m_hash:
                    required_hashes.add(m_hash)
                    try:
                        manifest = dag.get_manifest(m_hash)
                        for tname, tinfo in manifest.tensors.items():
                            if tinfo.get("blob_hash"):
                                required_hashes.add(tinfo["blob_hash"])
                            if tinfo.get("residual_hash"):
                                required_hashes.add(tinfo["residual_hash"])
                            if tinfo.get("perm_hash"):
                                required_hashes.add(tinfo["perm_hash"])
                    except Exception:
                        pass
            except Exception:
                pass

        return required_hashes

    @staticmethod
    def compute_missing_objects(
        required_objects: Set[str],
        peer_existing_objects: Set[str]
    ) -> List[str]:
        """
        Returns sorted list of missing object hashes: required - existing.
        """
        missing = required_objects - peer_existing_objects
        return sorted(list(missing))
