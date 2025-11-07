# Deployment Guide

## Overview
This app consists of two parts:
1. **FastAPI Backend** (api.py) - Provides prediction endpoints
2. **Streamlit Frontend** (app.py) - User interface

**Important**: Streamlit Cloud Community cannot run FastAPI and Streamlit together. They must be deployed separately.

---

## Option 1: Deploy API on Replit + Streamlit on Streamlit Cloud (Recommended)

### Step 1: Publish FastAPI on Replit

1. **In your Replit workspace**, click the **"Publish"** button in the top-right corner
2. Choose **"Autoscale"** deployment
3. Configure settings:
   - Machine: 1vCPU, 2 GiB RAM (default)
   - Max machines: 3 (default)
   - **Run command**: `uvicorn api:app --host 0.0.0.0 --port 8000`
4. Click **"Publish"** to launch your API

**Your API will be available at:**
```
https://[your-repl-name].[your-username].repl.co
```

**API Endpoints after deployment:**
- Random Forest: `https://[your-repl-name].[your-username].repl.co/api/predictions/random-forest?hours=12`
- XGBoost: `https://[your-repl-name].[your-username].repl.co/api/predictions/xgboost?hours=12`
- LSTM: `https://[your-repl-name].[your-username].repl.co/api/predictions/lstm?hours=12`

### Step 2: Deploy Streamlit App to Streamlit Cloud

1. **Push your code to GitHub**:
   ```bash
   git init
   git add app.py streamlit_requirements.txt .streamlit/
   git commit -m "Streamlit app for deployment"
   git push origin main
   ```

2. **Go to Streamlit Cloud**:
   - Visit: https://share.streamlit.io
   - Click **"New app"**
   - Connect your GitHub repository
   - Set main file path: `app.py`

3. **Configure Secrets** (Click "Advanced settings"):
   ```toml
   SUPABASE_URL = "your_supabase_url"
   SUPABASE_ANON_KEY = "your_supabase_key"
   ```

4. **Rename requirements file**:
   - Streamlit Cloud looks for `requirements.txt`
   - Rename `streamlit_requirements.txt` to `requirements.txt` in your GitHub repo

5. Click **"Deploy"**!

**Your Streamlit app will be available at:**
```
https://[your-app-name].streamlit.app
```

---

## Option 2: Deploy Both on Replit (Simpler)

### Publish as Autoscale Deployment

1. Click **"Publish"** in Replit workspace
2. Choose **"Autoscale"** deployment
3. Configure:
   - **Run command**: `streamlit run app.py --server.port 5000`
4. Click **"Publish"**

**Your Streamlit app will be available at:**
```
https://[your-repl-name].[your-username].repl.co
```

**Note**: The API endpoints will run internally on port 8000 and can be accessed from within the same Replit app.

---

## API URL Structure After Deployment

### If deployed on Replit:
```
Base URL: https://[your-repl-name].[your-username].repl.co

Endpoints:
├── GET  /                                        (API info)
├── GET  /api/predictions/random-forest?hours=12  (Random Forest predictions)
├── GET  /api/predictions/xgboost?hours=12        (XGBoost predictions)
└── GET  /api/predictions/lstm?hours=12           (LSTM predictions)
```

### Sample Response:
```json
{
  "prediction_metadata": {
    "model": "Random Forest",
    "generated_at": "2025-11-07T14:52:15.687792",
    "total_hours": 12
  },
  "predictions": [
    {
      "timestamp": "2025-11-02T11:13:31.463571+00:00",
      "hour_offset": 1,
      "air_quality_metrics": {
        "temperature": 29.48,
        "humidity": 70.85,
        "co2": 468.71,
        "co": 4.51,
        "pm25": 7.95,
        "pm10": 29.95
      }
    }
  ]
}
```

---

## Testing Your APIs

### Using cURL:
```bash
# Test Random Forest
curl "https://your-app-url/api/predictions/random-forest?hours=24"

# Test XGBoost
curl "https://your-app-url/api/predictions/xgboost?hours=24"

# Test LSTM
curl "https://your-app-url/api/predictions/lstm?hours=24"
```

### Using Python:
```python
import requests

api_url = "https://your-app-url/api/predictions/random-forest?hours=24"
response = requests.get(api_url)
data = response.json()
print(data)
```

---

## Important Notes

1. **Supabase Credentials**: Make sure to add your Supabase credentials in Streamlit Cloud secrets or Replit secrets
2. **CORS**: Already configured in `api.py` to accept requests from any origin
3. **Custom Domain**: Both Replit and Streamlit Cloud support custom domains (check their documentation)
4. **Rate Limits**: Free tier has usage limits - check the respective platform documentation

---

## Recommended Approach

**For Production**: Deploy API on Replit (or Render.com) and Streamlit app on Streamlit Cloud Community (free)

**For Quick Testing**: Deploy everything on Replit using Option 2

---

## Need Help?

- Replit Deployment Docs: https://docs.replit.com/hosting/deployments
- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-community-cloud
