# 🚀 Streamlit Cloud Deployment Guide

This guide explains how to deploy the HR Policy Chatbot to Streamlit Cloud.

## ✅ Current Status

**The app is now Streamlit Cloud ready!** All necessary configurations have been added.

## 📋 Prerequisites

1. **GitHub Repository**: Your code must be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **OpenAI API Key**: You'll need a valid OpenAI API key

## 🔧 Files Added for Streamlit Cloud

The following files have been configured for Streamlit Cloud deployment:

- ✅ `requirements.txt` - Dependencies compatible with Streamlit Cloud
- ✅ `packages.txt` - System packages (poppler-utils for PDF processing)
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.streamlit/secrets.toml` - Local secrets template (not committed)
- ✅ `streamlit_app.py` - Updated for cloud deployment with secrets management

## 🚀 Deployment Steps

### 1. Push to GitHub
```bash
git add .
git commit -m "feat: Add Streamlit Cloud deployment support"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Select your repository and branch**
5. **Set main file path**: `streamlit_app.py`
6. **Click "Deploy"**

### 3. Configure Secrets

Once deployed, you need to add your OpenAI API key:

1. **Go to your app's dashboard**
2. **Click the "⚙️" (settings) button**
3. **Go to "Secrets"**
4. **Add the following:**

```toml
OPENAI_API_KEY = "your_actual_openai_api_key_here"
```

5. **Click "Save"**

## 📁 File Upload for Policies

Since Streamlit Cloud doesn't persist files, you have two options for HR policy documents:

### Option A: Include PDFs in Repository (Recommended)
1. Add your PDF files to the `policies/` folder
2. Commit and push them to GitHub
3. The app will automatically process them on startup

### Option B: Upload via Streamlit Interface
The app could be modified to include file upload functionality (not currently implemented).

## ⚙️ Configuration Options

You can customize the app by adding these secrets in Streamlit Cloud:

```toml
# Required
OPENAI_API_KEY = "your_api_key"

# Optional customizations
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DEFAULT_K = 4
MAX_K = 10
CACHE_ENABLED = true
CACHE_TTL = 3600
MAX_CACHE_SIZE = 1000
MAX_QUERY_LENGTH = 1000
```

## 🔍 Features in Streamlit Cloud

The deployed app includes:

- **🤖 Interactive Chat Interface**: Ask questions in natural language
- **📚 Document Search**: Searches through all uploaded HR policies
- **📄 Source Citations**: Shows which documents answers came from
- **⚙️ Configuration Panel**: Adjust search parameters
- **🔄 Index Rebuilding**: Rebuild search index when needed
- **📊 Status Indicators**: Visual feedback on app status

## 🐛 Troubleshooting

### Common Issues:

**1. "OpenAI API key not found"**
- Solution: Add `OPENAI_API_KEY` to Streamlit secrets

**2. "Policies folder not found"**
- Solution: Ensure `policies/` folder with PDF files is in your repository

**3. "Build failed" during deployment**
- Solution: Check that `requirements.txt` and `packages.txt` are present

**4. PDF processing errors**
- Solution: Ensure your PDFs are valid and not corrupted

### Performance Considerations:

- **Cold Starts**: First load may take 1-2 minutes to build index
- **Memory Limits**: Streamlit Cloud has memory limits; large PDF collections may hit limits
- **Processing Time**: Large document sets take longer to process

## 🔒 Security Notes

- **API Keys**: Never commit API keys to your repository
- **PDF Content**: Ensure PDFs don't contain sensitive information if repository is public
- **Access Control**: Streamlit Cloud apps are public unless you have a paid plan

## 🚀 Going Live

Once deployed, your app will be available at:
`https://your-app-name.streamlit.app`

You can share this URL with your team to start using the HR Policy Chatbot!

## 🔄 Updates

To update your deployed app:

1. Make changes locally
2. Push to GitHub
3. Streamlit Cloud automatically redeploys

## 📞 Support

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Create issues in your repository

---

**✨ Your HR Policy Chatbot is ready for Streamlit Cloud deployment!**