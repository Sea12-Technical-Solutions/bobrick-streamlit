# Deployment Guide for Streamlit Apps

## ⚠️ Important: Vercel is NOT Suitable

**Vercel cannot deploy Streamlit apps.** Vercel is designed for:
- Static websites
- Serverless functions (Node.js, Python functions)
- Next.js applications

Streamlit apps are **long-running Python applications** that require a persistent server, which Vercel doesn't support.

## ✅ Recommended: Streamlit Cloud (Simplest & Free)

Streamlit Cloud is the easiest and best option for deploying Streamlit apps. It's free and designed specifically for Streamlit.

### Option 1: Deploy All Apps as One Multi-Page App (Recommended)

I've created a main `app.py` that combines all 5 apps into a single deployment. This is the simplest approach.

#### Steps:

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Sign up for Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Sign in with your GitHub account

3. **Deploy the app**
   - Click "New app"
   - Select your repository
   - Set **Main file path** to: `app.py`
   - Set **Python version** to: `3.9` or `3.10`
   - Click "Deploy"

4. **Configure Environment Variables**
   - In your Streamlit Cloud app settings, go to "Secrets"
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```
   - The apps use `python-dotenv` which will read from Streamlit's secrets

5. **Update your apps to use Streamlit secrets** (if needed)
   - Streamlit Cloud uses `st.secrets` instead of `.env` files
   - Your apps already use `load_dotenv()`, but you may need to add fallback:
     ```python
     import os
     from dotenv import load_dotenv
     
     load_dotenv()  # For local development
     # Streamlit Cloud will use st.secrets automatically
     ```

### Option 2: Deploy Each App Separately

If you prefer separate deployments for each app:

1. Create separate GitHub repositories for each app, OR
2. Use different branches, OR
3. Deploy from the same repo but specify different main files:
   - `streamlit_warp_detector.py`
   - `streamlit_chrome_scratch_detector.py`
   - `streamlit_video_rotation_analyzer.py`
   - `streamlit_missing_parts_app.py`
   - `streamlit_missing_parts_app2.py`

## 🔧 Alternative Deployment Options

### Railway (Easy, Free Tier Available)
- Go to https://railway.app/
- Connect GitHub repo
- Add Python buildpack
- Set start command: `streamlit run app.py --server.port $PORT`
- Add environment variables in Railway dashboard

### Render (Free Tier Available)
- Go to https://render.com/
- Create new Web Service
- Connect GitHub repo
- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Add environment variables in Render dashboard

### Heroku (Paid, but reliable)
- Install Heroku CLI
- Create `Procfile`: `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- Deploy: `heroku create` then `git push heroku main`

## 📝 Pre-Deployment Checklist

- [x] All dependencies in `requirements.txt`
- [ ] Test apps locally: `streamlit run app.py`
- [ ] Set up environment variables (OpenAI API key)
- [ ] Push code to GitHub
- [ ] Deploy to Streamlit Cloud
- [ ] Test deployed apps

## 🐛 Troubleshooting

### Apps can't find environment variables
- Make sure you've added secrets in Streamlit Cloud dashboard
- Update apps to use `st.secrets` as fallback:
  ```python
  import os
  import streamlit as st
  
  # Try Streamlit secrets first, then .env
  api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
  ```

### Import errors
- Make sure all dependencies are in `requirements.txt`
- Check Python version compatibility (use 3.9 or 3.10)

### File upload issues
- Streamlit Cloud has file size limits
- Large video files may need to be hosted elsewhere

## 🚀 Quick Start (Streamlit Cloud)

1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Click "New app"
4. Select repo, set main file to `app.py`
5. Add `OPENAI_API_KEY` in Secrets
6. Deploy!

Your app will be live at: `https://your-app-name.streamlit.app`

