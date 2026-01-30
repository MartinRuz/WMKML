from huggingface_hub import snapshot_download

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
target_folder = "/workspace/models/mistral-7b-v0.3"  # your custom path

snapshot_download(
    repo_id=model_id,
    local_dir=target_folder,
    local_dir_use_symlinks=False
)
