"""Upload this project to a Hugging Face Space (Docker) to get a public URL.

1. Create a free account on https://huggingface.co
2. Settings -> Access Tokens -> create a token with WRITE access
3. pip install huggingface_hub
4. python scripts/deploy_to_hf.py <your-hf-username>/<space-name>

The Space is created if needed, the files are uploaded (images and the model are handled
automatically), and Hugging Face builds the Dockerfile. The app is then live at
https://<your-hf-username>-<space-name>.hf.space
"""
import os
import sys

from huggingface_hub import HfApi, login

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python scripts/deploy_to_hf.py <hf-username>/<space-name>")
    repo_id = sys.argv[1]
    login()  # asks for the access token (or uses HF_TOKEN)
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=ROOT, repo_id=repo_id, repo_type="space",
        ignore_patterns=[".git/*", "**/__pycache__/*", "database/*", "models/archive/*", "raw/*",
                         "venv/*", ".venv/*", "*.zip"],
        commit_message="Deploy bean leaf disease app",
    )
    # the retraining demo zip is needed in the Space too (excluded above with the other zips)
    demo_zip = os.path.join(ROOT, "demo_samples", "new_data_for_retraining.zip")
    if os.path.exists(demo_zip):
        api.upload_file(path_or_fileobj=demo_zip, path_in_repo="demo_samples/new_data_for_retraining.zip",
                        repo_id=repo_id, repo_type="space")
    user, name = repo_id.split("/")
    print(f"Uploaded. Build logs: https://huggingface.co/spaces/{repo_id}")
    print(f"App URL (after the build, ~5-10 min): https://{user}-{name}.hf.space".lower())


if __name__ == "__main__":
    main()
