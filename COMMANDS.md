# PhishGuard: Command Notes

Project folder: `C:\Users\anish\OneDrive\Desktop\phishing-url-detector`
GitHub repo: `https://github.com/AnishaWagh/phishing-url-detector`

All commands are for **PowerShell in the VS Code terminal** (open it with `` Ctrl+` ``).

---

## 1. Start working (every time you open the project)

```powershell
cd C:\Users\anish\OneDrive\Desktop\phishing-url-detector
venv\Scripts\activate
```

You should see `(venv)` at the start of the terminal line. If you don't, the packages won't be found.

To check where you are:

```powershell
pwd
dir
```

To leave the virtual environment later:

```powershell
deactivate
```

---

## 2. Install packages (first time, or on a new PC)

```powershell
pip install pandas numpy scikit-learn xgboost joblib streamlit tldextract
```

Or install everything from the file:

```powershell
pip install -r requirements.txt
```

---

## 3. Run the project

### Retrain the model (only when the data or features change)

Run these **in this order**:

```powershell
python prepare.py     # builds data/features.csv from data/phishing.csv
python train.py       # trains the models, saves models/phishing_model.pkl
python evaluate.py    # prints scores, top features, real-world URL checks
```

### Start the web app

```powershell
streamlit run app.py
```

It opens at `http://localhost:8501`. Stop it with `Ctrl+C` in the terminal.

---

## 4. Create requirements.txt (keep the same versions as your PC)

```powershell
pip freeze | Select-String -Pattern "^(streamlit|pandas|numpy|scikit-learn|xgboost|joblib|tldextract)==" | Out-File -Encoding utf8 requirements.txt
type requirements.txt
```

It should show 7 lines.

---

## 5. Push to GitHub

### Every normal update (use this most of the time)

```powershell
git status
git add .
git commit -m "describe what you changed"
git push
```

### First time only (one-time setup)

1. Create the empty repo on GitHub first: **github.com/new**, name `phishing-url-detector`, Public, no README.
2. Then:

```powershell
git init
git add .
git commit -m "Phishing URL detector"
git branch -M main
git remote add origin https://github.com/AnishaWagh/phishing-url-detector.git
git push -u origin main
```

### One-time Git identity (if git asks who you are)

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### Useful Git commands

| Task | Command |
|---|---|
| See changed files | `git status` |
| See commit history | `git log --oneline` |
| Check the remote URL | `git remote -v` |
| Fix a wrong remote URL | `git remote set-url origin https://github.com/AnishaWagh/phishing-url-detector.git` |
| Add one specific file | `git add models/phishing_model.pkl` |
| Download latest changes | `git pull` |

After you push, Streamlit Community Cloud redeploys the app automatically in a minute or two.

---

## 6. Create new files and folders

Run these in the **project root** (the same folder as `requirements.txt`).

### Create a file and open it in VS Code

```powershell
New-Item filename.py -ItemType File
code filename.py
```

Then paste your code into it and press `Ctrl+S` to save.

### Create a file inside a folder

```powershell
New-Item .streamlit\config.toml -ItemType File
code .streamlit\config.toml
```

### Create folders

```powershell
mkdir data
mkdir models
mkdir .streamlit
```

(PowerShell can also do several at once with commas: `mkdir data, models, notebooks`)

### Create a file with the mouse (no commands)

1. Press `Ctrl+Shift+E` to open the Explorer panel.
2. Click the **New File** icon next to the project name (or right-click empty space, then **New File**).
3. Type the name with its extension, such as `utils.py`, and press Enter.
4. Paste your code and press `Ctrl+S`.

### Check that the file really exists

```powershell
dir
```

If the name shows `.py.txt`, rename it:

```powershell
ren filename.py.txt filename.py
```

---

## 7. Project files at a glance

| File | What it does |
|---|---|
| `features.py` | Turns a URL into numeric features |
| `prepare.py` | Cleans the dataset and writes `data/features.csv` |
| `train.py` | Trains the models and saves `models/phishing_model.pkl` |
| `evaluate.py` | Prints scores and tests real URLs |
| `app.py` | The Streamlit web app |
| `requirements.txt` | Packages for deployment |
| `.gitignore` | Keeps `venv/` and `data/` out of GitHub |
| `.streamlit/config.toml` | App theme |

---

## 8. Common errors and quick fixes

| Error | Fix |
|---|---|
| `can't open file '...py': No such file` | The file isn't in this folder. Run `dir`, then `cd` to the project root or create the file. |
| `ModuleNotFoundError` | Activate the venv (`venv\Scripts\activate`), then `pip install <package>`. |
| `IndentationError: unexpected indent` | A line has extra spaces at the start. Remove them so it touches the left edge. |
| `KeyError` in the app | `models/phishing_model.pkl` is out of date. Re-run `prepare.py`, then `train.py`. |
| `mkdir` says positional parameter | Use commas: `mkdir data, models` (or one folder per command). |
| `fatal: repository ... not found` | Create the repo on GitHub first, then check `git remote -v`. |
| Push is rejected for a big file | Keep the model under 100 MB: use `compress=3` in `joblib.dump` or train a smaller model. |
| `FileNotFoundError: models/phishing_model.pkl` on Streamlit Cloud | Run `git add models/phishing_model.pkl`, commit and push. |

---

## 9. Quick daily routine

```powershell
cd C:\Users\anish\OneDrive\Desktop\phishing-url-detector
venv\Scripts\activate
streamlit run app.py
```

Edit, save, test, then:

```powershell
git add .
git commit -m "my change"
git push
```
