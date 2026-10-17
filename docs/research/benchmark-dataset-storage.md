# Free storage for the private benchmark dataset (CI-readable)

Research for [#4](https://github.com/decksray/MTG-Recognition/issues/4) (map: [#2](https://github.com/decksray/MTG-Recognition/issues/2)). Researched 2026-10-17.

## Context

- `data/test/` per `data/test.dvc`: **592,241,908 bytes (~565 MiB), 971 files**. 260 UUID-named ground-truth images (~534 MB) are the owner's own photos and **must stay private**.
- Today: DVC with remote `gdrive://root/decksray/MTG-Recognition/data/test` and a personal OAuth desktop client (`.dvc/config`).
- CI: GitHub Actions on a private repo, free plan. The plan is to `actions/cache` the dataset keyed on the `data/test.dvc` hash, so CI downloads it roughly **once per dataset version** (plus after any cache eviction).
- Option: store images pre-resized to 2048 px longest edge, about 100–150 MB.

### How often CI really downloads

The GitHub Actions cache is 10 GB per repo by default, and "GitHub will remove any cache entries that have not been accessed in over 7 days" ([GitHub: dependency caching](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)). A cold download happens when the dataset changes, after 7 days with no CI run, or when eviction hits the cache. A realistic upper bound is a handful of cold pulls per month, about 0.6–3 GB/month at full size. Egress caps only matter for B2 and Git LFS (below).

## Comparison

| | Google Drive + DVC (service account) | Cloudflare R2 + DVC (S3) | Backblaze B2 + DVC (S3) | Hugging Face private dataset | Git LFS on GitHub |
|---|---|---|---|---|---|
| Free storage | Owner's existing Drive quota; files stay owned by the owner | 10 GB-month | 10 GB | 100 GB private (free user) | 10 GiB |
| Free egress | No published byte cap; "usage limits" apply to service accounts | Free (no egress charges) | 3x average monthly storage (about 1.8 GB/mo for 0.6 GB stored), then $0.01/GB | No byte cap; 5,000 resolver requests per 5 min (free user) | 10 GiB/month, **Actions downloads count** |
| Credit card | No | Unverified: docs only say "complete the checkout flow"; likely yes | No ("No credit card required") | No | No (overage just disables LFS) |
| DVC support | Native (`gdrive`) | Native via S3-compatible `endpointurl` | Native via S3-compatible `endpointurl` | **None as a remote** (read-only `dvc get/import` only) | n/a, replaces DVC |
| CI secret | `GDRIVE_CREDENTIALS_DATA` = SA JSON key | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (R2 "Object Read only" token scoped to the bucket) | Same env vars from a bucket-scoped, read-only application key | `HF_TOKEN` (read) | None (`GITHUB_TOKEN`, `lfs: true` on checkout) |
| Privacy | Folder shared Viewer-only with the SA | Private bucket, scoped token | Private bucket, scoped key | Private repo flag | Private repo |
| Setup effort | Low: create SA, share folder, add secret, change the remote URL to a folder ID | Medium: new account and bucket, migrate with `dvc push -r r2` | Medium: same as R2 | Medium-high: new tooling outside DVC | Medium: move from DVC to LFS, rewrite docs |

## Per option

### 1. DVC + Google Drive with a service account (the current remote)

- **Auth**: set `gdrive_use_service_account true`. In CI, put the SA key JSON in the `GDRIVE_CREDENTIALS_DATA` env var, which DVC docs list for CI/CD and which takes priority over `gdrive_service_account_json_file_path`. A service account "requires your own Google Cloud project". Folders must be shared with the SA, with Viewer access for read-only use ([DVC: Google Drive](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive)).
- **Quota**: Google states "Service accounts don't have storage quota and can't own files" ([Drive API: shared drives](https://developers.google.com/workspace/drive/api/guides/about-shareddrives)). This does not matter for CI reads, because files stay owned by the owner (pushed with the owner's OAuth client) and the SA only reads them. The SA could not `dvc push` into My Drive, and it does not need to.
- **Limits**: DVC warns that "service accounts have their own usage limits, which can be exceeded with frequent `dvc push` or `dvc pull`". No figure is published. One cached pull per dataset version is low volume.
- **Remote URL change (inference, not verified end to end)**: `gdrive://root/...` resolves against the *authenticating account's* My Drive root. For the SA, that is its own empty drive, so the remote should point at the shared folder's ID (`gdrive://<folder-id>`). That form works for the owner's OAuth client too.
- **Why the current setup can't simply run in CI**: a personal OAuth client whose consent screen is in "Testing" status is "issued a refresh token expiring in 7 days" ([Google OAuth 2.0](https://developers.google.com/identity/protocols/oauth2)), so a cached user token in CI would break every week. DVC's default shared OAuth app is blocked by Google ([treeverse/dvc#10516](https://github.com/treeverse/dvc/issues/10516), closed "not planned" Jan 2026). That is why the repo already uses its own client.
- **Risks**: the SA JSON key is a long-lived credential; scope it to Viewer on one folder. Google keeps changing Drive and SA policy (2025 quota removal, OAuth app blocks). Usage limits are undocumented.

### 2. Cloudflare R2 + DVC

- Free tier: "10 GB-month / month", Class A 1M and Class B 10M requests/month. Egress is free via the S3 API ([R2 pricing](https://developers.cloudflare.com/r2/pricing/)). 971 files means about 1k GET requests per pull, far below the limit.
- Enabling R2 means "Complete the checkout flow to add an R2 subscription" ([R2 get started](https://developers.cloudflare.com/r2/get-started/)). Whether a card is mandatory is **not stated on first-party pages**; third-party guides disagree. Assume a payment method is required.
- Auth: an API token with "Object Read only", scoped to specific buckets, yields an S3 Access Key ID and Secret. The endpoint is `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` ([R2 API tokens](https://developers.cloudflare.com/r2/api/tokens/)). DVC takes `endpointurl` plus `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` env vars ([DVC: S3](https://doc.dvc.org/user-guide/data-management/remote-storage/amazon-s3)). DVC docs don't name R2; it's a generic S3-compatible endpoint.
- Risks: a card on file means overages get billed, though 0.6 GB is far under 10 GB. Another vendor account.

### 3. Backblaze B2 + DVC

- "First 10GB storage is always free". "Free 3x monthly egress" (of average stored data), then $0.01/GB. Class A/B/C calls are free; Class D gets 2,500/day free ([B2 pricing](https://www.backblaze.com/cloud-storage/pricing)). "No credit card required" ([B2](https://www.backblaze.com/cloud-storage)).
- Egress math: 0.6 GB stored allows about 1.8 GB/month free egress, about **3 full pulls/month** (about 12–18 if pre-resized). Owner pulls on new machines count too. Overage is cents, but only if billing is set up.
- Auth: application keys can be restricted to one bucket and "Read Only" ([B2 application keys](https://www.backblaze.com/docs/cloud-storage-application-keys)). The S3 endpoint format and the keyID-to-access-key mapping were **not confirmed** on first-party pages fetched here.

### 4. Hugging Face private dataset

- Free user: 100 GB private storage ([HF storage limits](https://huggingface.co/docs/hub/storage-limits)). Free-user rate limit is 5,000 resolver requests per 5 minutes, so pass `HF_TOKEN` ([HF rate limits](https://huggingface.co/docs/hub/rate-limits)). 971 files fit.
- **DVC cannot push to or pull from HF as a remote.** HF is not in DVC's remote list ([DVC remote storage](https://doc.dvc.org/user-guide/data-management/remote-storage)), and the HF integration is read-only `dvc get/import` ([DVC: Hugging Face](https://doc.dvc.org/user-guide/integrations/huggingface)). Using HF means dropping DVC for `hf download` / `huggingface_hub`, plus a separate way to pin the dataset version (a commit SHA).
- HF's guidance for large datasets expects community reuse. That fits a private benchmark of personal photos poorly, although private repos are within the free quota.

### 5. Git LFS on GitHub

- GitHub Free: 10 GiB storage and 10 GiB bandwidth per month. "Downloads of Git LFS files by Actions use the repository owner's bandwidth". Over quota without a payment method, "Git LFS support is disabled on your account until the next month" ([GitHub: Git LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)).
- About 0.6 GB per full pull gives about 16 pulls/month across CI and every clone. Each dataset version adds to storage, which is never freed without history surgery.
- Replaces DVC entirely: no extra account and no secret. Risks: a hard monthly stop that breaks CI and clones, plus migration away from the documented DVC workflow.

## Pre-resizing to 2048 px

Shrinking to about 100–150 MB only changes the verdict for **B2** (about 4x more pulls) and **Git LFS** (about 4x more headroom). Drive, R2 and HF are not egress-bound at this size. The cost: the benchmark would measure re-encoded JPEGs instead of the original photos. Even if the pipeline downsizes to `MAX_IMAGE_DIMENSION` itself, a second JPEG encode changes pixels. It also creates a new dataset version that must be pushed.

## Recommendation

**Keep DVC + Google Drive and add a read-only service account for CI.** It costs nothing, needs no card and no new vendor, and no data moves. It's one GitHub secret (`GDRIVE_CREDENTIALS_DATA`) plus one config change: point the remote at the folder ID. Combined with `actions/cache` keyed on `data/test.dvc`, the SA does about one pull per dataset version. Keep full-resolution images, since pre-resizing buys nothing for this option.

**Fallback: Cloudflare R2** if the SA route proves flaky (undocumented SA limits, Google policy churn). Free egress and a bucket-scoped read-only token make it the cleanest CI story, but it likely needs a card on file. B2 is the no-card alternative, with a tight 3x egress cap. Not recommended: HF (no DVC remote) and Git LFS (hard 10 GiB/month bandwidth stop, leaves DVC).

## What the owner must decide

1. **Drive service account (no card, Google-policy risk) or R2 (likely needs a card, free egress)?** Either way the CI secret goes into GitHub repo secrets.
2. **Keep full-resolution images (recommended) or store pre-resized copies?** This only matters if B2 or Git LFS is chosen.
3. Creating the Google Cloud service account and sharing the Drive folder are owner-only steps (owner's Google account).

## Not verified from primary sources

- Whether Cloudflare requires a payment method to enable R2's free tier.
- B2's S3 endpoint format and key mapping (not on the pages fetched).
- The `gdrive://root/...` to `gdrive://<folder-id>` change needed for the SA: inferred from how Drive resolves `root` per account, not tested.
- Actual Google Drive SA download limits (DVC says limits exist; Google publishes none for this case).
