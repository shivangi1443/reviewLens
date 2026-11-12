# ReviewLens — Customer Sentiment Dashboard

A Streamlit app for exploring customer sentiment labels, review ratings, complaint outcomes, response times, and platform/category segments.

## Run locally

1. Install Python 3.10 or newer.
2. Open PowerShell in this folder.
3. Install packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Start ReviewLens:

   ```powershell
   python -m streamlit run app.py
   ```

The app loads `Customer_Sentiment.csv` from this folder by default. It also supports uploading a compatible CSV from the sidebar.

## What's included

- `app.py` — dashboard and data analysis
- `requirements.txt` — Python packages
- `Customer_Sentiment.csv` — local dataset, excluded from Git by default
- `.gitignore` — keeps customer data, Python cache, environments, and secrets out of commits by default

## Publish the code on GitHub

1. Confirm the dataset is synthetic or that you have permission to publish it. The CSV includes customer IDs and demographic fields. Do not publish real customer data in a public repository or public app.
2. Create an empty GitHub repository. Choose Private for a deployment using private data. Do not initialize it with a README or .gitignore.
3. In PowerShell, run these commands from this folder:

   ```powershell
   git init
   git add app.py requirements.txt README.md .gitignore
   git commit -m "Initial ReviewLens dashboard"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/reviewlens.git
   git push -u origin main
   ```

4. If the CSV is synthetic or approved for the private deployment and should be bundled with the app, explicitly add it:

   ```powershell
   git add -f Customer_Sentiment.csv
   git commit -m "Add approved demo dataset"
   git push
   ```

Replace `YOUR-USERNAME` with your GitHub username. If `origin` already exists, use `git remote set-url origin https://github.com/YOUR-USERNAME/reviewlens.git`.

## Deploy the app

GitHub stores the project; Streamlit Community Cloud runs the Streamlit website.

1. Sign in at <https://share.streamlit.io> with GitHub and authorize access to the repository.
2. Choose **Create app** and select the repository, branch `main`, and entrypoint `app.py`.
3. Keep the app private when using non-public data. Deploying a public app with the CSV committed makes the data available through the app and repository.
4. Deploy. Community Cloud installs the dependencies from `requirements.txt` and provides a shareable app URL.

When the app is deployed without the CSV, users can upload a CSV through the sidebar. For a public portfolio demo, use a synthetic dataset rather than real customer records.

## Model and interpretation notes

The CSV's `sentiment` column is the primary label. ReviewLens also applies VADER to `review_text` and compares its labels with the CSV labels. The match percentage is an agreement measure, not verified model accuracy; the dataset's labeling method should be confirmed. The file has no date field, so time trends and live updates aren't available. Repeated review wording may mean the data is templated or synthetic.
