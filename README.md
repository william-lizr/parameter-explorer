# Parameter Explorer

A no-code explorer for parameter sweeps. Load a CSV with one row per simulation run. Mark each column as a parameter, a metric, or a result. Drag parameters to constrain the sweep. The app draws surfaces, heatmaps, slices, sensitivity plots and more.

**How to format your CSV:** see [docs/CSV-format.md](docs/CSV-format.md) or the [wiki](https://github.com/william-lizr/parameter-explorer/wiki/CSV-format). The same guide opens from the app ("How to format your CSV").

## Run it on your computer

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows. On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py                 # or: python app.py my_sweep.csv
```

Open http://127.0.0.1:8050.

To make new sample files: `python generate_sample.py`.

## Deploy to Cloudflare

The app is a Python server (Dash), so it runs in a [Cloudflare Container](https://developers.cloudflare.com/containers/). A small Worker (`src/index.js`) sends every request to the container.

**You need a Workers Paid plan** ($5/month). Containers are not on the free plan.

### Option A: GitHub Actions (this repo already has the workflow)

1. In Cloudflare, go to **My Profile → API Tokens → Create Token**. Use the **Edit Cloudflare Workers** template. Check that it has these **Account** permissions:
   - Workers Scripts: Edit
   - Containers: Edit (add this row, the template does not have it)
   - Account Settings: Read

   Under **Account Resources**, pick the account you deploy to. A token for a different account fails with "No access to the specified resource".
2. Copy your **Account ID** from the Cloudflare dashboard (Workers & Pages → right sidebar).
3. In GitHub, go to **Settings → Secrets and variables → Actions**. Add:
   - `CLOUDFLARE_API_TOKEN`
   - `CLOUDFLARE_ACCOUNT_ID`
4. Go to **Actions → Deploy to Cloudflare → Run workflow**. Each push to `main` deploys again.

The first deploy takes a few minutes. The app URL is `https://parameter-explorer.<your-subdomain>.workers.dev`.

### Option B: Cloudflare dashboard

1. Go to **Workers & Pages → Create → Import a repository**.
2. Pick this repo. Keep the deploy command `npx wrangler deploy`.

### Option C: From your computer

Start Docker Desktop first. Then:

```bash
npm install
npx wrangler login
npx wrangler deploy
```

### Things to know

- **Uploaded data lives in memory.** The container stops after 30 idle minutes (`sleepAfter` in `src/index.js`). Users must load their CSV again after that. The first request after a stop takes some seconds.
- **One instance.** `max_instances` is 1, so all users share one server and its memory. The app keeps the newest 20 datasets (`MAX_DATASETS`).
- **Bigger files.** For large CSVs, change `instance_type` in `wrangler.jsonc` from `basic` (1 GiB) to `standard-1` (4 GiB).
