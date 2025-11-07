# Quick Deployment Guide

## 🚀 Your API is Ready!

### Current Development URLs (while running on Replit):
- **Random Forest API**: `https://fe43402a-c8c6-4240-9e05-8b49f9aa7f29-00-1wpv1oeo9e9wc.pike.replit.dev:8000/api/predictions/random-forest?hours=12`
- **XGBoost API**: `https://fe43402a-c8c6-4240-9e05-8b49f9aa7f29-00-1wpv1oeo9e9wc.pike.replit.dev:8000/api/predictions/xgboost?hours=12`
- **LSTM API**: `https://fe43402a-c8c6-4240-9e05-8b49f9aa7f29-00-1wpv1oeo9e9wc.pike.replit.dev:8000/api/predictions/lstm?hours=12`

---

## 📝 For Streamlit Cloud Deployment

### Option 1: Deploy ONLY Streamlit App to Streamlit Cloud (Keep API on Replit)

**Step 1 - Publish API on Replit (for permanent URL):**
1. Click **"Publish"** button (top-right in Replit)
2. Select **"Autoscale"** deployment
3. Run command: `uvicorn api:app --host 0.0.0.0 --port 8000`
4. Click **"Publish"**

✅ After publishing, your API URL will be:
```
https://[your-repl-name].[your-username].repl.co/api/predictions/random-forest
https://[your-repl-name].[your-username].repl.co/api/predictions/xgboost
https://[your-repl-name].[your-username].repl.co/api/predictions/lstm
```

**Step 2 - Deploy Streamlit to Streamlit Cloud:**
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Click "New app" → Select your repo
4. Main file: `app.py`
5. **IMPORTANT**: Rename `streamlit_requirements.txt` to `requirements.txt` in your GitHub repo
6. Add secrets in "Advanced settings":
   ```toml
   SUPABASE_URL = "your_url"
   SUPABASE_ANON_KEY = "your_key"
   ```
7. Deploy!

✅ Your Streamlit app will be at:
```
https://[app-name].streamlit.app
```

---

### Option 2: Deploy Everything on Replit (Easiest!)

1. Click **"Publish"** in Replit
2. Choose **"Autoscale"**
3. Run command: `streamlit run app.py --server.port 5000`
4. Click **"Publish"**

✅ Both your Streamlit app AND API will be at:
```
Streamlit UI: https://[your-repl-name].[your-username].repl.co
API Endpoints: https://[your-repl-name].[your-username].repl.co/api/predictions/...
```

---

## 🔍 What Changes After Deployment?

### Before Deployment (Development):
```
API URL: https://fe43402a-c8c6-4240-9e05-8b49f9aa7f29-00-1wpv1oeo9e9wc.pike.replit.dev:8000
└── Temporary development URL (changes when you restart)
```

### After Replit Publishing:
```
API URL: https://[your-repl-name].[your-username].repl.co
└── Permanent public URL ✅
└── Example: https://air-quality-api.yourname.repl.co
```

### After Streamlit Cloud Deployment:
```
Streamlit App: https://[app-name].streamlit.app
└── Permanent public URL ✅
└── Example: https://air-quality-dashboard.streamlit.app
```

---

## 📊 Files Created for Deployment

- ✅ `streamlit_requirements.txt` - Dependencies for Streamlit Cloud (rename to `requirements.txt`)
- ✅ `api_requirements.txt` - Dependencies for API deployment
- ✅ `DEPLOYMENT_GUIDE.md` - Detailed deployment instructions

---

## 🎯 Quick Test After Deployment

```bash
# Replace with your actual deployed URL
curl "https://your-deployed-url/api/predictions/random-forest?hours=6"
```

Expected response:
```json
{
  "prediction_metadata": {
    "model": "Random Forest",
    "generated_at": "2025-11-07T...",
    "total_hours": 6
  },
  "predictions": [...]
}
```

---

## ⚠️ Important Notes

1. **Streamlit Cloud Community cannot run FastAPI** - must deploy separately
2. **Free tier limitations** exist on both platforms
3. **Supabase credentials** must be added as secrets/environment variables
4. The API endpoints are already configured with CORS for cross-origin requests

---

## 📚 More Information

See `DEPLOYMENT_GUIDE.md` for detailed deployment instructions and troubleshooting.
