# Maintaining the benchmark dataset

How the DVC remote for `data/test/` is set up, and the tasks only the maintainer does. Contributors only need [CONTRIBUTING.md](../../CONTRIBUTING.md#dataset-setup); the reasons behind this setup are in [ADR 0001](../adr/0001-benchmark-dataset-on-google-drive-via-dvc.md).

## Who is who

| Thing | What it is | Owned by |
|---|---|---|
| `mtgrecognitiondataset@gmail.com` | The dataset account: a Google account that holds only the dataset folder | you |
| `mtg-recognition-dataset` (Drive folder, ID `10bFfSl2E0RGDAufp8pp4KyeNNLB21fmn`) | The DVC remote. DVC stores the data in its `files` subfolder | dataset account |
| `mtg-recognition-2026` | The Google Cloud project that holds the OAuth app and the service account | your personal account |
| `mtg-recognition` | The OAuth app you push through, signed in as the dataset account | the project |
| `dataset-reader@mtg-recognition-2026.iam.gserviceaccount.com` | The read-only service account. Viewer on the dataset folder; contributors and CI each get their own key | the project |
| `GDRIVE_CREDENTIALS_DATA` | CI's `dataset-reader` key, as a GitHub Actions secret | the repo |

Use the dataset account for Drive, and your personal account for the Cloud console. Don't give the dataset account access to the Cloud project.

## Onboard a contributor

1. As your personal account, open [Service accounts](https://console.cloud.google.com/iam-admin/serviceaccounts?project=mtg-recognition-2026) → `dataset-reader` → **Keys** → **Add key** → **Create new key** → **JSON**.
2. Write down the key ID (shown in the Keys tab) next to the contributor's name, e.g. in a password manager note. The Keys tab has no names, so this note is the only way to tell keys apart later.
3. Send them the JSON file over a private channel (a password manager share, or a direct message), never in an issue, PR or commit. Delete your local copy; if they lose the key, create a new one.
4. Point them at [CONTRIBUTING.md › Dataset Setup](../../CONTRIBUTING.md#dataset-setup).

## Offboard a contributor, or respond to a leaked key

Delete that person's key: [Service accounts](https://console.cloud.google.com/iam-admin/serviceaccounts?project=mtg-recognition-2026) → `dataset-reader` → **Keys** → delete the key ID from your note. Other keys keep working.

A leaked key can only download the dataset; it can't change or delete anything.

## Replace CI's key

Create a key as in onboarding, then `gh secret set GDRIVE_CREDENTIALS_DATA < key.json`, delete the old key in the Keys tab, and delete `key.json`. GitHub never shows a secret's value, so the first CI run that pulls the dataset is what confirms the new key works.

## Push new images

See [CONTRIBUTING.md › Adding New Test Images](../../CONTRIBUTING.md#adding-new-test-images). Contributors can't push, so they send you their images and you add them.

When DVC opens the browser to sign in, pick the **dataset account**. If you pick your personal account, the push fails with "File not found", because your personal account can't see the folder.

## Regular checks

Every few months, and before you go a long time without pushing:

- **The OAuth app is still in production.** [OAuth Audience](https://console.cloud.google.com/auth/audience?project=mtg-recognition-2026) should say "In production". In "Testing", your sign-in expires every 7 days.
- **The OAuth client is still in use.** Google deletes an OAuth client after 6 months without a token exchange; it emails a notice first, and a deleted client can be restored within 30 days ([Google Developers Blog](https://developers.googleblog.com/usability-and-safety-updates-to-google-auth-platform/)). Run `uv run dvc status -c` at least every few months: it signs in through the client.
- **The dataset account is still active.** Google can delete a personal account, Drive included, after 2 years without activity; it emails the account and its recovery email first ([Inactive Google Account Policy](https://support.google.com/accounts/answer/12418290)). Sign in to it at least once a year, e.g. open the [dataset folder](https://drive.google.com/drive/folders/10bFfSl2E0RGDAufp8pp4KyeNNLB21fmn), and keep its recovery email up to date.
- **The folder is still shared with `dataset-reader` as Viewer**, on `mtg-recognition-dataset` itself, not on `files`.

## Don't

- Upload a logo or request verification on the OAuth app's Branding page: either one makes Google review the app.
- Share the folder as Editor with anyone. Pushing goes through the dataset account only.
- Move, rename or delete files inside `files` in the Drive web page. DVC finds them by their hashed names, so `dvc pull` breaks.
- Put `gdrive_use_service_account` or a key path in the committed `.dvc/config`. That would make your own pushes use the read-only key.
- Remove the pyOpenSSL cap by upgrading packages around it. PyDrive2 1.21.3 requires pyOpenSSL below 24.3, because its service-account sign-in calls `OpenSSL.crypto.sign`. A package that needs a newer pyOpenSSL (as `streamlit-webrtc` did) holds PyDrive2 back, and keys stop working.

## If you lose the remote

Your local DVC cache (`.dvc/cache`) holds the whole dataset. To rebuild the remote, create a new folder, point the remote at it (`uv run dvc remote modify origin url gdrive://<folder-id>`), run `uv run dvc push`, share the new folder with `dataset-reader` as Viewer, and commit `.dvc/config`.

## Troubleshooting

| Error | Cause and fix |
|---|---|
| `invalid_grant: Token has been expired or revoked` | Your cached sign-in is no longer valid. Delete `~/.cache/pydrive2fs/<client_id>/default.json` and push again, signing in as the dataset account |
| A contributor gets `File not found: <folder-id>` | The folder isn't shared with `dataset-reader`, or the share is on `files` instead of its parent |
| `module 'OpenSSL.crypto' has no attribute 'sign'` | pyOpenSSL 24.3 or later is installed; see "Don't" above |
