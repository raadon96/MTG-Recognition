"""Centralized database configuration for MTG Recognition."""

from pathlib import Path

from sqlmodel import create_engine

# Database file location
LOCAL_DB_FILE = Path("data/reference/MobileCache.sqlite")

# SQLModel engine - shared across the application
engine = create_engine(f"sqlite:///{LOCAL_DB_FILE}?mode=ro&nolock=1")
