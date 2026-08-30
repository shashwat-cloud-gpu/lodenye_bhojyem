"""
SynapseFS Command-Line Interface (CLI).
Implements all required commands:
init, commit, checkout, branch, log, verify, merge, mount, unmount, serve, push, pull.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


from synapsefs.alignment.out_of_core import OutOfCoreAlignmentEngine
from synapsefs.alignment.reconstructor import CheckpointReconstructor
from synapsefs.core.cas import ContentAddressableStore
from synapsefs.core.crash_resilience import TransactionJournal
from synapsefs.core.dag import Manifest, RepositoryDAG
from synapsefs.core.lineage import LineageVerifier
from synapsefs.net.client import SyncClient
from synapsefs.net.server import SyncServer
from synapsefs.utils.safetensors_helper import read_safetensors_header, save_safetensors_file


def find_repo_root() -> Optional[Path]:
    """Finds the root directory containing .synapsefs."""
    curr = Path.cwd().resolve()
    while curr != curr.parent:
        if (curr / ".synapsefs").is_dir():
            return curr
        curr = curr.parent
    return None


def get_repo_or_die() -> Tuple[Path, RepositoryDAG]:
    root = find_repo_root()
    if not root:
        print("[ERROR] Not a SynapseFS repository (or any parent up to mount point). Run 'synapsefs init' first.", file=sys.stderr)
        sys.exit(1)
    
    # Run crash recovery check
    journal = TransactionJournal(root / ".synapsefs")
    cas = ContentAddressableStore(root / ".synapsefs")
    journal.recover(cas)
    
    dag = RepositoryDAG(root, cas)
    return root, dag


def cmd_init(args) -> None:
    target_dir = Path(args.directory).resolve()
    synapse_dir = target_dir / ".synapsefs"
    if synapse_dir.exists():
        print(f"[!] Reinitialized existing SynapseFS repository in {synapse_dir}")
        return

    RepositoryDAG.init(target_dir)
    print(f"[+] Initialized empty SynapseFS repository in {synapse_dir}")


def cmd_commit(args) -> None:
    root, dag = get_repo_or_die()
    cas = dag.cas
    journal = TransactionJournal(root / ".synapsefs")

    target_file = Path(args.checkpoint).resolve()
    if not target_file.is_file():
        print(f"[ERROR] Target checkpoint '{target_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config).resolve() if args.config else None
    if config_path and not config_path.is_file():
        config_path = None

    current_branch, current_commit_id = dag.get_head_ref()
    base_commit_id = args.base or current_commit_id

    tx_id = journal.begin_transaction("commit", {"target": str(target_file), "base": base_commit_id})

    try:
        if base_commit_id is None:
            # Initial base commit: store raw tensor blobs directly in CAS
            print(f"[*] Creating initial base commit for '{target_file.name}'...")
            h_size, header, d_start = read_safetensors_header(target_file)
            manifest_tensors = {}
            total_bytes = 0

            with open(target_file, "rb") as f:
                for tname, tmeta in header.items():
                    if tname == "__metadata__":
                        continue
                    rel_start, rel_end = tmeta["data_offsets"]
                    f.seek(d_start + rel_start)
                    raw_data = f.read(rel_end - rel_start)
                    blob_hash = cas.put(raw_data)
                    journal.record_blob(tx_id, blob_hash)
                    total_bytes += len(raw_data)

                    manifest_tensors[tname] = {
                        "dtype": tmeta["dtype"],
                        "shape": tmeta["shape"],
                        "data_offsets": tmeta["data_offsets"],
                        "is_delta": False,
                        "blob_hash": blob_hash,
                        "uncompressed_bytes": len(raw_data),
                        "compressed_bytes": len(raw_data),
                    }

            manifest = Manifest(
                tensors=manifest_tensors,
                metadata=header.get("__metadata__", {}),
                base_commit_id=None,
            )
            manifest_bytes = manifest.serialize()
            manifest_hash = cas.put(manifest_bytes)
            journal.record_blob(tx_id, manifest_hash)

            commit_id = dag.create_commit(
                manifest_hash=manifest_hash,
                message=args.message or f"Initial commit of {target_file.name}",
                parent_ids=[],
            )
            journal.mark_committed(tx_id, commit_id)
            journal.close_transaction(tx_id)

            print(f"[+] Committed base checkpoint {commit_id[:10]} on branch '{current_branch or 'main'}' ({total_bytes / (1024*1024):.2f} MB stored)")

        else:
            # Delta commit: out-of-core isomorphic alignment against base commit
            print(f"[*] Aligning target checkpoint against base commit {base_commit_id[:10]}...")
            
            # Temporarily reconstruct base to staging for alignment streaming
            base_commit = dag.get_commit(base_commit_id)
            base_manifest = dag.get_manifest(base_commit.manifest_hash)
            
            # We reconstruct base model to temp file for lazy layer streaming
            temp_base_file = root / ".synapsefs" / "tmp" / f"base_{base_commit_id[:8]}.safetensors"
            reconstructor = CheckpointReconstructor(dag)
            reconstructor.reconstruct_to_file(base_manifest, temp_base_file)

            try:
                engine = OutOfCoreAlignmentEngine(cas)
                manifest, stats = engine.align_and_diff(
                    base_checkpoint_path=temp_base_file,
                    target_checkpoint_path=target_file,
                    config_path=config_path,
                    base_commit_id=base_commit_id,
                )

                manifest_bytes = manifest.serialize()
                manifest_hash = cas.put(manifest_bytes)
                journal.record_blob(tx_id, manifest_hash)

                commit_id = dag.create_commit(
                    manifest_hash=manifest_hash,
                    message=args.message or f"Delta commit of {target_file.name}",
                    parent_ids=[base_commit_id],
                )
                journal.mark_committed(tx_id, commit_id)
                journal.close_transaction(tx_id)

                print(f"[+] Successfully aligned and committed {commit_id[:10]} on branch '{current_branch or 'main'}'")
                print(f"    - Original Target Size: {stats['total_target_bytes'] / (1024*1024):.2f} MB")
                print(f"    - Stored Residual Size: {stats['total_residual_bytes'] / (1024*1024):.2f} MB")
                print(f"    - Compression Ratio:    {stats['residual_ratio'] * 100:.2f}% (Savings: {(1.0 - stats['residual_ratio'])*100:.2f}%)")
                print(f"    - Alignment Confidence: {stats['mean_confidence'] * 100:.1f}%")
                if not stats["is_alignable"]:
                    print("    [!] Warning: Models exhibited low structural correlation; stored safe fallback delta.")

            finally:
                if temp_base_file.exists():
                    temp_base_file.unlink()

    except Exception as e:
        journal.close_transaction(tx_id)
        raise e


def cmd_checkout(args) -> None:
    root, dag = get_repo_or_die()
    target = args.target

    # 1. Check if target is a branch name
    branches = dag.list_branches()
    if target in branches:
        dag.set_head_branch(target)
        commit_id = dag.get_branch_commit(target)
        print(f"[*] Switched to branch '{target}' (HEAD at {commit_id[:10] if commit_id else 'empty'})")
        if args.output and commit_id:
            commit = dag.get_commit(commit_id)
            manifest = dag.get_manifest(commit.manifest_hash)
            reconstructor = CheckpointReconstructor(dag)
            reconstructor.reconstruct_to_file(manifest, Path(args.output))
            print(f"[+] Reconstructed {args.output} from branch '{target}'")
        return

    # 2. Check if target is a commit ID
    if dag.cas.exists(target):
        dag.set_head_detached(target)
        commit = dag.get_commit(target)
        manifest = dag.get_manifest(commit.manifest_hash)
        print(f"[*] Checked out commit {target[:10]} in detached HEAD state")
        if args.output:
            reconstructor = CheckpointReconstructor(dag)
            reconstructor.reconstruct_to_file(manifest, Path(args.output))
            print(f"[+] Reconstructed {args.output} from commit {target[:10]}")
        return

    print(f"[ERROR] '{target}' did not match any branch or valid commit in repository.", file=sys.stderr)
    sys.exit(1)


def cmd_branch(args) -> None:
    root, dag = get_repo_or_die()
    
    if args.name:
        # Create branch
        curr_branch, curr_commit = dag.get_head_ref()
        if not curr_commit:
            print("[ERROR] Cannot create branch from an empty repository.", file=sys.stderr)
            sys.exit(1)
        dag.update_branch(args.name, curr_commit)
        print(f"[+] Created branch '{args.name}' pointing to {curr_commit[:10]}")
    else:
        # List branches
        branches = dag.list_branches()
        curr_branch, _ = dag.get_head_ref()
        for b in branches:
            prefix = "* " if b == curr_branch else "  "
            cid = dag.get_branch_commit(b)
            print(f"{prefix}{b} ({cid[:10] if cid else 'empty'})")


def cmd_log(args) -> None:
    root, dag = get_repo_or_die()
    history = dag.walk_history(args.commit)

    if not history:
        print("No commits found.")
        return

    for cid, commit in history:
        t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(commit.timestamp))
        print(f"commit {cid}")
        if len(commit.parent_ids) > 1:
            print(f"Merge: {' '.join([p[:8] for p in commit.parent_ids])}")
        print(f"Author: {commit.author}")
        print(f"Date:   {t_str}")
        print(f"\n    {commit.message}\n")


def cmd_verify(args) -> None:
    root, dag = get_repo_or_die()
    verifier = LineageVerifier(dag)

    print("[*] Walking Merkle DAG and verifying cryptographic integrity...")
    t0 = time.time()
    result = verifier.verify_commit_lineage(args.commit)
    elapsed = time.time() - t0

    print(f"\n--- Verification Summary ({elapsed:.3f}s) ---")
    print(f"Status:             {'[PASSED] Integrity 100% Verified' if result.is_valid else '[FAILED] Tampering or Corruption Detected'}")
    print(f"Commits Verified:   {result.verified_commits}")
    print(f"Manifests Verified: {result.verified_manifests}")
    print(f"Blobs Verified:     {result.verified_blobs}")
    print(f"Data Bytes Checked: {result.total_bytes_verified / (1024*1024):.2f} MB")

    if result.tampered_objects:
        print("\n[!] Tampered Objects Detected:")
        for obj in result.tampered_objects:
            print(f"    - {obj['type']} Expected: {obj['expected_hash']} Actual: {obj['actual_hash']}")

    if result.missing_objects:
        print("\n[!] Missing Objects:")
        for obj in result.missing_objects:
            print(f"    - {obj['type']} Hash: {obj['hash']}")

    if not result.is_valid:
        sys.exit(1)


def cmd_merge(args) -> None:
    root, dag = get_repo_or_die()
    source_branch = args.branch
    try:
        status, commit_id = dag.merge_branches(source_branch, message=args.message)
        print(f"[+] Merge status: {status} (Commit: {commit_id[:10]})")
    except Exception as e:
        print(f"[ERROR] Merge failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


def cmd_mount(args) -> None:
    root, dag = get_repo_or_die()
    mount_point = Path(args.mountpoint).resolve()
    
    commit_id = args.commit
    if not commit_id:
        _, commit_id = dag.get_head_ref()
    if not commit_id:
        print("[ERROR] No commit to mount.", file=sys.stderr)
        sys.exit(1)

    from synapsefs.vfs.fuse_daemon import run_fuse_mount
    run_fuse_mount(
        dag=dag,
        commit_id=commit_id,
        mount_point=mount_point,
        foreground=not args.daemon,
        max_cache_mb=args.cache_mb,
    )


def cmd_unmount(args) -> None:
    mount_point = Path(args.mountpoint).resolve()
    print(f"[*] Unmounting {mount_point}...")
    if sys.platform.startswith("linux"):
        res = subprocess.run(["fusermount", "-u", str(mount_point)], capture_output=True)
        if res.returncode != 0:
            res = subprocess.run(["umount", str(mount_point)], capture_output=True)
        if res.returncode == 0:
            print(f"[+] Successfully unmounted {mount_point}")
        else:
            print(f"[ERROR] Unmount failed: {res.stderr.decode('utf-8')}", file=sys.stderr)
    else:
        print(f"[!] Unmount command on non-Linux platforms is a no-op.")


def cmd_serve(args) -> None:
    root, dag = get_repo_or_die()
    server = SyncServer(dag, host=args.host, port=args.port)
    server.start(block=True)


def cmd_push(args) -> None:
    root, dag = get_repo_or_die()
    client = SyncClient(dag)
    try:
        res = client.push(remote_url=args.remote, branch=args.branch)
        print(f"[+] Push completed to {args.remote}: transferred {res['transferred_blocks']} blocks ({res['transferred_bytes'] / 1024:.2f} KB)")
    except Exception as e:
        print(f"[ERROR] Push failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


def cmd_pull(args) -> None:
    root, dag = get_repo_or_die()
    client = SyncClient(dag)
    try:
        res = client.pull(remote_url=args.remote, branch=args.branch)
        print(f"[+] Pull completed from {args.remote}: transferred {res['transferred_blocks']} blocks ({res['transferred_bytes'] / 1024:.2f} KB)")
    except Exception as e:
        print(f"[ERROR] Pull failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="synapsefs",
        description="SynapseFS: Permutation-Aware Cryptographic VCS for Neural Network Checkpoints"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Initialize a new SynapseFS repository")
    p_init.add_argument("directory", nargs="?", default=".", help="Directory to initialize")
    p_init.set_defaults(func=cmd_init)

    # commit
    p_commit = subparsers.add_parser("commit", help="Commit a checkpoint to the repository")
    p_commit.add_argument("-c", "--checkpoint", required=True, help="Path to .safetensors checkpoint file")
    p_commit.add_argument("-b", "--base", default=None, help="Base commit ID to align against")
    p_commit.add_argument("--config", default=None, help="Path to config.json topology definition")
    p_commit.add_argument("-m", "--message", default="", help="Commit message")
    p_commit.set_defaults(func=cmd_commit)

    # checkout
    p_checkout = subparsers.add_parser("checkout", help="Restore a checkpoint version or switch branches")
    p_checkout.add_argument("target", help="Branch name or Commit ID")
    p_checkout.add_argument("-o", "--output", default=None, help="Output .safetensors path to restore")
    p_checkout.set_defaults(func=cmd_checkout)

    # branch
    p_branch = subparsers.add_parser("branch", help="List or create branches")
    p_branch.add_argument("name", nargs="?", default=None, help="Name of new branch")
    p_branch.set_defaults(func=cmd_branch)

    # log
    p_log = subparsers.add_parser("log", help="Display commit lineage history")
    p_log.add_argument("commit", nargs="?", default=None, help="Starting commit ID (default HEAD)")
    p_log.set_defaults(func=cmd_log)

    # verify
    p_verify = subparsers.add_parser("verify", help="Cryptographically verify commit DAG integrity")
    p_verify.add_argument("commit", nargs="?", default=None, help="Commit ID to verify (default HEAD)")
    p_verify.set_defaults(func=cmd_verify)

    # merge
    p_merge = subparsers.add_parser("merge", help="Merge another branch into current branch")
    p_merge.add_argument("branch", help="Source branch to merge")
    p_merge.add_argument("-m", "--message", default=None, help="Merge commit message")
    p_merge.set_defaults(func=cmd_merge)

    # mount
    p_mount = subparsers.add_parser("mount", help="Mount virtual filesystem for on-demand checkpoint access")
    p_mount.add_argument("mountpoint", help="Target mount directory")
    p_mount.add_argument("commit", nargs="?", default=None, help="Commit ID to mount (default HEAD)")
    p_mount.add_argument("--daemon", action="store_true", help="Run mount in background daemon mode")
    p_mount.add_argument("--cache-mb", type=int, default=512, help="Max LRU tensor cache in MB")
    p_mount.set_defaults(func=cmd_mount)

    # unmount
    p_unmount = subparsers.add_parser("unmount", help="Unmount virtual filesystem")
    p_unmount.add_argument("mountpoint", help="Mounted directory")
    p_unmount.set_defaults(func=cmd_unmount)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start peer sync server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to listen on")
    p_serve.set_defaults(func=cmd_serve)

    # push
    p_push = subparsers.add_parser("push", help="Push commits and missing blocks to peer")
    p_push.add_argument("remote", help="Remote peer URL (e.g. http://127.0.0.1:8000)")
    p_push.add_argument("branch", nargs="?", default=None, help="Branch name to push")
    p_push.set_defaults(func=cmd_push)

    # pull
    p_pull = subparsers.add_parser("pull", help="Pull commits and missing blocks from peer")
    p_pull.add_argument("remote", help="Remote peer URL (e.g. http://127.0.0.1:8000)")
    p_pull.add_argument("branch", nargs="?", default=None, help="Branch name to pull")
    p_pull.set_defaults(func=cmd_pull)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
