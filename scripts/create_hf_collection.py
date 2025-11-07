"""Create HuggingFace collection for instruction-following datasets.

This script creates a collection on HuggingFace Hub to organize
instruction-following evaluation datasets in VERL format.

Usage:
    # Use cached token from huggingface-cli login
    python scripts/create_hf_collection.py

    # Or provide token explicitly
    python scripts/create_hf_collection.py --token $HF_TOKEN

    # Dry run (preview without creating)
    python scripts/create_hf_collection.py --dry-run
"""

import argparse
import os

from huggingface_hub import HfApi


def create_if_collection(
    token: str = None,
    title: str = "VERL IF Datasets",
    description: str = None,
    datasets: list = None,
    namespace: str = "sungyub",
    private: bool = False,
    dry_run: bool = False,
):
    """Create instruction-following datasets collection on HuggingFace Hub.

    Args:
        token: HuggingFace authentication token (uses cached if None)
        title: Collection title
        description: Collection description
        datasets: List of dataset IDs to include
        namespace: HuggingFace namespace/username
        private: Whether to create a private collection
        dry_run: If True, only print what would be done without creating

    Returns:
        Collection URL if created, None if dry_run
    """
    # Default description
    if description is None:
        description = (
            "High-quality instruction-following (IF) evaluation datasets in VERL format: "
            "verifiable constraints and constraint-based benchmarks for RL training"
        )

    # Default datasets
    if datasets is None:
        datasets = [
            "sungyub/ifbench-verl",
            "sungyub/ifeval-rlvr-verl",
        ]

    print(f"Creating HuggingFace Collection:")
    print(f"  Title: {title}")
    print(f"  Description: {description}")
    print(f"  Namespace: {namespace}")
    print(f"  Private: {private}")
    print(f"  Datasets ({len(datasets)}):")
    for i, dataset_id in enumerate(datasets, 1):
        print(f"    {i}. {dataset_id}")

    if dry_run:
        print("\n[DRY RUN] Would create collection with above settings")
        print("[DRY RUN] No actual changes made")
        return None

    # Initialize API
    print("\n1. Initializing HuggingFace API...")
    api = HfApi(token=token)
    print("   ✓ API initialized")

    # Create collection
    print("\n2. Creating collection...")
    try:
        collection = api.create_collection(
            title=title,
            description=description,
            namespace=namespace,
            private=private,
            exists_ok=True,  # Don't error if collection already exists
        )
        print(f"   ✓ Collection created: {collection.slug}")
        print(f"   ✓ Collection URL: {collection.url}")
    except Exception as e:
        print(f"   ✗ Failed to create collection: {e}")
        raise

    # Add datasets to collection
    print("\n3. Adding datasets to collection...")
    for i, dataset_id in enumerate(datasets, 1):
        try:
            api.add_collection_item(
                collection_slug=collection.slug,
                item_id=dataset_id,
                item_type="dataset",
                note=None,  # Optional note for the item
            )
            print(f"   ✓ Added {i}/{len(datasets)}: {dataset_id}")
        except Exception as e:
            print(f"   ⚠ Warning: Could not add {dataset_id}: {e}")
            # Continue with other datasets

    # Verify collection
    print("\n4. Verifying collection...")
    try:
        # Get collection info to verify
        collection_info = api.get_collection(collection.slug)
        print(f"   ✓ Collection verified")
        print(f"   ✓ Items in collection: {len(collection_info.items)}")
    except Exception as e:
        print(f"   ⚠ Warning: Could not verify collection: {e}")

    print(f"\n✓ Collection creation complete!")
    print(f"  URL: {collection.url}")

    return collection.url


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create HuggingFace collection for instruction-following datasets"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace authentication token (or use cached token from huggingface-cli login)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="VERL IF Datasets",
        help="Collection title (default: VERL IF Datasets)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Collection description (uses default if not provided)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Dataset IDs to include (default: ifbench-verl, ifeval-rlvr-verl)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="sungyub",
        help="HuggingFace namespace/username (default: sungyub)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private collection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be created without actually creating",
    )

    args = parser.parse_args()

    # Get token from args or environment
    # If None, huggingface_hub will use cached token from huggingface-cli login
    token = args.token or os.environ.get("HF_TOKEN")

    create_if_collection(
        token=token,
        title=args.title,
        description=args.description,
        datasets=args.datasets,
        namespace=args.namespace,
        private=args.private,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
