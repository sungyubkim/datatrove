"""
Download the original working README from HuggingFace Hub.
This will fetch the previous version that was working before our update.
"""

from huggingface_hub import HfApi
import requests

api = HfApi()

# Download README from one of the working datasets
repo_id = "sungyub/deepmath-103k-verl"

print(f"Fetching README history from {repo_id}...")

# Get all commits
commits = api.list_repo_commits(repo_id=repo_id, repo_type="dataset")

print(f"\nFound {len(list(commits))} commits")
print("\nRecent commits:")

for i, commit in enumerate(list(commits)[:10]):
    print(f"\n{i+1}. Commit: {commit.commit_id[:8]}")
    print(f"   Title: {commit.title}")
    print(f"   Date: {commit.created_at}")

# Get the README from the second-to-last commit (before our update)
# The last commit is our update, so we want the one before
commits_list = list(commits)
if len(commits_list) >= 2:
    original_commit = commits_list[1]
    print(f"\n\n{'='*70}")
    print(f"FETCHING ORIGINAL README FROM:")
    print(f"{'='*70}")
    print(f"Commit: {original_commit.commit_id}")
    print(f"Title: {original_commit.title}")
    print(f"Date: {original_commit.created_at}")

    # Download the README from that commit
    url = f"https://huggingface.co/datasets/{repo_id}/raw/{original_commit.commit_id}/README.md"
    print(f"\nURL: {url}")

    response = requests.get(url)
    if response.status_code == 200:
        readme_content = response.text

        # Save to file
        output_path = "/Users/sungyubkim/repos/datatrove/output/original_working_readme.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"\n✓ Original README saved to: {output_path}")

        # Extract just the YAML frontmatter
        if readme_content.startswith('---'):
            parts = readme_content.split('---', 2)
            if len(parts) >= 3:
                yaml_content = parts[1]
                print(f"\n{'='*70}")
                print(f"ORIGINAL YAML FRONTMATTER:")
                print(f"{'='*70}")
                print(yaml_content)
    else:
        print(f"\n✗ Failed to download: {response.status_code}")
else:
    print("\n✗ Not enough commits to find original version")
