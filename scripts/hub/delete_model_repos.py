"""
Delete incorrectly uploaded model repositories.
These were uploaded to models instead of datasets.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, list_repo_files

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from source_dataset_mapping import DATASET_IDS


def delete_model_repos():
    """Delete all model repositories that were created by mistake."""
    all_datasets = DATASET_IDS + ["math-verl-unified"]

    print(f"Deleting {len(all_datasets)} incorrectly uploaded MODEL repositories...")
    print("=" * 70)

    api = HfApi()
    results = []

    for dataset_id in all_datasets:
        repo_id = f"sungyub/{dataset_id}"

        print(f"\n🗑️  Deleting: {repo_id} (model)")
        print("-" * 70)

        try:
            # Try to delete the model repository
            api.delete_repo(repo_id=repo_id, repo_type="model")

            print(f"  ✓ Model repository deleted successfully")
            results.append({"repo": repo_id, "status": "success"})

        except Exception as e:
            error_msg = str(e)

            # Check if repo doesn't exist (which is OK)
            if "does not exist" in error_msg.lower() or "not found" in error_msg.lower() or "404" in error_msg:
                print(f"  ℹ️  Model repository does not exist (already deleted or never created)")
                results.append({"repo": repo_id, "status": "not_found"})
            else:
                print(f"  ✗ Error deleting: {error_msg}")
                results.append({"repo": repo_id, "status": "error", "error": error_msg})

    # Print summary
    print("\n" + "=" * 70)
    print("DELETION SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r["status"] == "success"]
    not_found = [r for r in results if r["status"] == "not_found"]
    failed = [r for r in results if r["status"] == "error"]

    print(f"\n✓ Deleted: {len(successful)}/{len(all_datasets)}")
    print(f"ℹ️  Not found: {len(not_found)}/{len(all_datasets)}")
    print(f"✗ Failed: {len(failed)}/{len(all_datasets)}")

    if failed:
        print("\nFailed deletions:")
        for result in failed:
            print(f"  - {result['repo']}: {result.get('error', 'Unknown error')}")

    print("\n✓ Model repositories cleanup completed!")


if __name__ == "__main__":
    delete_model_repos()
