# Quick Start - Deploy to Streamlit Cloud

## 🚀 3-Step Deployment

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set **Main file path**: `app.py`
6. Click "Deploy"

### Step 3: Add API Key
1. In your app settings, click "Secrets"
2. Add:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```
3. Save and your app will redeploy automatically

## ✅ Done!

Your app will be live at: `https://your-app-name.streamlit.app`

All 5 apps are accessible from the sidebar navigation.

## 📝 Notes

- **Vercel won't work** - Streamlit apps need a persistent server
- **Streamlit Cloud is free** and designed for Streamlit apps
- Each app shares the same OpenAI API key
- File uploads are limited by Streamlit Cloud (200MB per file)

## 🐛 Troubleshooting

**App won't start?**
- Check that `requirements.txt` has all dependencies
- Verify Python version is 3.9 or 3.10

**API key not working?**
- Make sure it's set in Streamlit Cloud Secrets (not just .env file)
- Check the key starts with `sk-`

**Import errors?**
- All 5 streamlit_*.py files must be in the same directory as app.py

