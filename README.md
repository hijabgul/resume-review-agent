# ResumeFit AI — Single-Agent Resume Review App (CrewAI + Streamlit + Groq)

A beginner-friendly app where one AI "agent" reads a candidate's resume and a
job description, and tells you honestly how well they match — without ever
inventing skills the candidate doesn't actually have.

No coding experience needed to run this. Just follow the steps below in order.

---

## 1. What's in this project (file structure)

```
resume-review-agent/
├── app.py                      # The Streamlit web page (what you see/click)
├── resume_crew.py              # The CrewAI "agent" and its instructions
├── pdf_utils.py                # Reads text out of an uploaded PDF resume
├── requirements.txt            # List of Python packages the app needs
├── runtime.txt                 # Tells Streamlit Cloud to use Python 3.11
├── .gitignore                  # Keeps your secret API key out of GitHub
├── .streamlit/
│   └── secrets.toml.example    # Template for your API key (copy this)
└── README.md                   # This guide
```

You never need to touch `resume_crew.py` or `pdf_utils.py` to get the app
running — they're the "engine." `app.py` is the part that draws the page.

**Why only these files?** This is the minimum needed for a clean, working
app — easy to read, easy to upload to GitHub, and nothing extra to confuse
a beginner.

---

## 2. What each piece does, in plain English

- **Agent**: one AI "persona" (a senior resume analyst) with a single job —
  compare the resume to the job description and report back honestly.
- **Task**: the exact instructions given to that agent (what to check, what
  format to answer in, and the rule "never make up a skill").
- **Crew**: CrewAI's term for "the team that runs the task." With one agent
  and one task, it's the simplest possible crew.
- **Streamlit**: turns your Python code into a website with almost no extra
  work — text boxes, buttons, file uploads, all built in.
- **Groq**: the company that runs the AI model *very fast* and *for free at
  low volume*. We use their `openai/gpt-oss-120b` model.

---

## 3. Get a free Groq API key

1. Go to <https://console.groq.com/keys> and sign in (free account).
2. Click **Create API Key**, give it any name, and copy the key
   (it starts with `gsk_...`). You won't be able to see it again, so paste
   it somewhere safe for now.

---

## 4. Run it on your own computer first (recommended)

1. **Install Python 3.11** if you don't have it: <https://www.python.org/downloads/>
   (during install on Windows, tick "Add Python to PATH").
2. **Download this project folder** to your computer.
3. Open a terminal (Command Prompt / Terminal app) **inside the project
   folder** and run:
   ```bash
   pip install -r requirements.txt
   ```
   This installs Streamlit, CrewAI, and the PDF reader.
4. **Add your API key:**
   - Go into the `.streamlit` folder.
   - Copy `secrets.toml.example` and rename the copy to `secrets.toml`.
   - Open `secrets.toml` and replace the placeholder with your real key:
     ```toml
     GROQ_API_KEY = "gsk_your_real_key"
     ```
5. **Start the app:**
   ```bash
   streamlit run app.py
   ```
   A browser tab should open automatically at `http://localhost:8501`.
6. Paste a resume (or upload a PDF), paste a job description, click
   **Analyze Match**, and you'll get a score plus a breakdown.

If something goes wrong, see the **Troubleshooting** section near the bottom.

---

## 5. Put it on GitHub

1. Create a free GitHub account at <https://github.com> if you don't have one.
2. Click **New repository**, name it e.g. `resume-review-agent`, keep it
   **Public** (required for the free tier of Streamlit Community Cloud),
   and click **Create repository**.
3. Upload every file in this folder **except** `.streamlit/secrets.toml`
   (your real key). If you followed step 4 above, `.gitignore` already
   prevents that file from being uploaded when using `git`. If you're
   dragging-and-dropping files on the GitHub website instead, just make
   sure you only upload `secrets.toml.example`, not `secrets.toml`.

   Using the command line instead:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/resume-review-agent.git
   git push -u origin main
   ```

---

## 6. Deploy for free on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with your GitHub account.
2. Click **Create app** → **From existing repo**.
3. Pick your `resume-review-agent` repository, branch `main`, and main file
   path `app.py`.
4. Before/after deploying, open **Advanced settings** (or **App settings →
   Secrets** after it's created) and add your key in the same TOML format:
   ```toml
   GROQ_API_KEY = "gsk_your_real_key"
   ```
5. Click **Deploy**. After a minute or two, your app is live at a public
   `https://your-app-name.streamlit.app` link you can share.

*Python version:* Streamlit Cloud reads `runtime.txt` in this project and
will use Python 3.11 automatically. If it doesn't, set it manually under
**Advanced settings → Python version** when creating the app.

---

## 7. How the "no fabrication" rule works

The agent's instructions (in `resume_crew.py`) explicitly tell it:

> Never state or imply the candidate has a skill, tool, degree,
> certification, or years of experience that is not explicitly stated or
> clearly implied in the resume text. When in doubt, treat it as missing.

Its recommendations are also restricted to *better ways to present what's
already true* — never "add a skill you don't have."

---

## 8. Troubleshooting

| Problem | What it means | Fix |
|---|---|---|
| "No Groq API key found" | `secrets.toml` is missing or empty | Redo step 4.4 (local) or step 6.4 (cloud) |
| "The Groq API key was rejected" | Key is wrong, expired, or mistyped | Generate a new key at console.groq.com/keys |
| "The AI provider is rate-limiting requests" | Too many requests too fast (free tier limit) | Wait ~1 minute and click Analyze again |
| "We couldn't find any selectable text in this PDF" | The PDF is a scanned image, not real text | Switch to "Paste text" and paste the resume manually |
| App won't start locally | A package failed to install | Re-run `pip install -r requirements.txt` and read the error message — it usually names the missing tool (e.g. reinstall Python) |
| Deployed app shows a build error | A typo in `requirements.txt` or missing file | Check the "Manage app" logs on Streamlit Cloud for the exact line that failed |

---

## 9. Ideas for extending this later

- Add a second agent that rewrites specific resume bullet points.
- Let the user download the evaluation as a PDF report.
- Add support for multiple job descriptions at once (batch matching).

Good luck — and remember: you can always re-read `resume_crew.py`, it's
written with beginners in mind and every section is commented.
