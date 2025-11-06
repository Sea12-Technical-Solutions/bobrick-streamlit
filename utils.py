"""
Utility functions for environment variable handling
Works with both local .env files and Streamlit Cloud secrets
"""
import os
import streamlit as st
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()


def get_env_var(key: str, default: str = None) -> str:
    """
    Get environment variable from Streamlit secrets or environment variables.
    
    Priority:
    1. Streamlit secrets (for Streamlit Cloud)
    2. Environment variables (for local development)
    3. Default value
    
    Args:
        key: Environment variable key
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    # Fall back to environment variables (for local development)
    value = os.getenv(key, default)
    return value


def get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment.
    
    Returns:
        OpenAI API key or None if not found
    """
    return get_env_var("OPENAI_API_KEY")

