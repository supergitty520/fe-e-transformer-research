# Security

[中文](SECURITY.md) | **English**

This project handles program-generated experiment data and should not contain personal data or production
credentials.

## Never commit

- `.secrets/` or any private key;
- `.env`, API tokens, or cloud credentials;
- Windows SSH, public-key deployment, administrator creation, or revocation scripts;
- blog and Cloudflare publication infrastructure;
- user directories, game data, or unrelated machine information;
- restricted third-party datasets.

## Reporting

If a credential or sensitive machine detail is found in the private repository:

1. Do not copy its value into a normal issue or pull request.
2. Notify the repository owner with the file path and commit hash.
3. Revoke or rotate the credential immediately.
4. If necessary, remove old objects with a Git history-rewrite tool.
5. Scan again from a fresh clone after cleanup.

Deleting only the current file does not remove a pushed secret from Git history.

