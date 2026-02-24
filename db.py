"""
Database connection and session management using SQLAlchemy.
"""
import os
from contextlib import contextmanager
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
except ImportError:
    import sys
    print("\n[!] ERROR: 'sqlalchemy' is not installed.")
    print(f"[*] Current Python: {sys.executable}")
    print("[*] Please ensure you have activated your virtual environment:")
    print("    source venv/bin/activate")
    print("[*] Then run: streamlit run app.py\n")
    sys.exit(1)

DB_URL = os.getenv("DATABASE_URL", "sqlite:///attendance.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False,
                           expire_on_commit=False)   # keeps scalar values after session closes


def init_db():
    """Create all tables if they don't exist yet."""
    from models import Base
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    """Context-manager that yields a DB session and handles commit/rollback."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
