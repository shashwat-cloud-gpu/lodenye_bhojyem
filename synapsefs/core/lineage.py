"""
Cryptographic Lineage Verification and Tamper Detection for SynapseFS.
"""

from typing import Any, Dict, List, Optional, Set
from synapsefs.core.cas import ContentAddressableStore, compute_hash
from synapsefs.core.dag import Commit, Manifest, RepositoryDAG


class VerificationResult:
    def __init__(self):
        self.is_valid: bool = True
        self.verified_commits: int = 0
        self.verified_manifests: int = 0
        self.verified_blobs: int = 0
        self.total_bytes_verified: int = 0
        self.tampered_objects: List[Dict[str, str]] = []
        self.missing_objects: List[Dict[str, str]] = []
        self.errors: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "verified_commits": self.verified_commits,
            "verified_manifests": self.verified_manifests,
            "verified_blobs": self.verified_blobs,
            "total_bytes_verified": self.total_bytes_verified,
            "tampered_objects": self.tampered_objects,
            "missing_objects": self.missing_objects,
            "errors": self.errors,
        }


class LineageVerifier:
    """
    Cryptographically verifies Merkle DAG history, manifests, and CAS binary blobs.
    """

    def __init__(self, dag: RepositoryDAG):
        self.dag = dag
        self.cas = dag.cas

    def verify_blob(self, blob_hash: str, object_type: str, result: VerificationResult) -> bool:
        """Verifies integrity of an individual content-addressed blob."""
        if not self.cas.exists(blob_hash):
            result.is_valid = False
            result.missing_objects.append({"hash": blob_hash, "type": object_type})
            return False

        try:
            data = self.cas.get(blob_hash)
            actual_hash = compute_hash(data)
            if actual_hash != blob_hash:
                result.is_valid = False
                result.tampered_objects.append({
                    "expected_hash": blob_hash,
                    "actual_hash": actual_hash,
                    "type": object_type,
                })
                return False
            result.verified_blobs += 1
            result.total_bytes_verified += len(data)
            return True
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Error reading blob {blob_hash}: {str(e)}")
            return False

    def verify_commit_lineage(self, start_commit_id: Optional[str] = None) -> VerificationResult:
        """
        Walks the entire DAG from start_commit_id (or HEAD) to root and validates:
        1. Commit hash matches serialized content
        2. Manifest hash matches serialized manifest
        3. All tensor blobs, permutation maps, and residuals exist and match their hashes
        """
        result = VerificationResult()
        
        if start_commit_id is None:
            _, start_commit_id = self.dag.get_head_ref()
            if start_commit_id is None:
                result.errors.append("Repository has no commits to verify")
                return result

        visited_commits: Set[str] = set()
        visited_manifests: Set[str] = set()
        visited_blobs: Set[str] = set()
        queue: List[str] = [start_commit_id]

        while queue:
            cid = queue.pop(0)
            if not cid or cid in visited_commits:
                continue
            visited_commits.add(cid)

            # 1. Verify Commit Node
            if not self.cas.exists(cid):
                result.is_valid = False
                result.missing_objects.append({"hash": cid, "type": "commit"})
                continue

            try:
                commit_bytes = self.cas.get(cid)
                actual_cid = compute_hash(commit_bytes)
                if actual_cid != cid:
                    result.is_valid = False
                    result.tampered_objects.append({
                        "expected_hash": cid,
                        "actual_hash": actual_cid,
                        "type": "commit",
                    })
                    continue

                commit = Commit.from_bytes(commit_bytes)
                result.verified_commits += 1
                result.total_bytes_verified += len(commit_bytes)

                # Queue parents
                for pid in commit.parent_ids:
                    if pid and pid not in visited_commits:
                        queue.append(pid)

            except Exception as e:
                result.is_valid = False
                result.errors.append(f"Corrupted commit {cid}: {str(e)}")
                continue

            # 2. Verify Manifest
            m_hash = commit.manifest_hash
            if m_hash not in visited_manifests:
                visited_manifests.add(m_hash)
                if not self.cas.exists(m_hash):
                    result.is_valid = False
                    result.missing_objects.append({"hash": m_hash, "type": "manifest"})
                    continue

                try:
                    manifest_bytes = self.cas.get(m_hash)
                    actual_m_hash = compute_hash(manifest_bytes)
                    if actual_m_hash != m_hash:
                        result.is_valid = False
                        result.tampered_objects.append({
                            "expected_hash": m_hash,
                            "actual_hash": actual_m_hash,
                            "type": "manifest",
                        })
                        continue

                    manifest = Manifest.from_bytes(manifest_bytes)
                    result.verified_manifests += 1
                    result.total_bytes_verified += len(manifest_bytes)

                    # 3. Verify All Blobs referenced in Manifest
                    for tname, tinfo in manifest.tensors.items():
                        # Base or raw blob
                        b_hash = tinfo.get("blob_hash")
                        if b_hash and b_hash not in visited_blobs:
                            visited_blobs.add(b_hash)
                            self.verify_blob(b_hash, f"tensor_blob({tname})", result)

                        # Permutation map blob
                        p_hash = tinfo.get("perm_hash")
                        if p_hash and p_hash not in visited_blobs:
                            visited_blobs.add(p_hash)
                            self.verify_blob(p_hash, f"perm_map({tname})", result)

                        # Residual delta blob
                        r_hash = tinfo.get("residual_hash")
                        if r_hash and r_hash not in visited_blobs:
                            visited_blobs.add(r_hash)
                            self.verify_blob(r_hash, f"residual_delta({tname})", result)

                except Exception as e:
                    result.is_valid = False
                    result.errors.append(f"Corrupted manifest {m_hash}: {str(e)}")

        return result
