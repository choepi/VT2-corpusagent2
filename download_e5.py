from huggingface_hub import snapshot_download


def main() -> None:
    repo_id = "intfloat/e5-base-v2"
    local_dir = r"D:\Programmieren\GitHub\e5-base-v2"

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )


if __name__ == "__main__":
    main()
