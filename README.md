# Quality Control Streamlit Apps

This directory contains 5 Streamlit applications for quality control analysis:

1. **Warp Detector** - Analyzes video feeds to detect if parts are warped (not flush with conveyor belt)
2. **Chrome Scratch Detector** - Detects scratches on chrome plating using vision AI
3. **Video Rotation Analyzer** - Analyzes video rotation and orientation
4. **Missing Parts App** - Detects missing parts in assemblies (subassemblygood.JPG reference)
5. **Missing Parts App 2** - Alternative missing parts detection (autosoapref.JPG reference)

## Quick Start

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. Run the main app:
   ```bash
   streamlit run app.py
   ```

### Deployment

See `QUICK_START.md` for deployment instructions to Streamlit Cloud.

## Files

- `app.py` - Main multi-page application (combines all 5 apps)
- `streamlit_*.py` - Individual Streamlit applications
- `utils.py` - Utility functions for environment variable handling
- `requirements.txt` - Python dependencies
- `autosoapref.JPG` - Reference image for Missing Parts App 2
- `subassemblygood.JPG` - Reference image for Missing Parts App
- `.streamlit/config.toml` - Streamlit configuration

## Requirements

- Python 3.9 or 3.10
- OpenAI API key
- All dependencies listed in `requirements.txt`

