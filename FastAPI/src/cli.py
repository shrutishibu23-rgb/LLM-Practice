"""
Note:
This is done using sync and not async due to greenlet loading issues on AVD.
"""

from src.database import sync_engine
from src.models import Base

def init_db():
    Base.metadata.create_all(bind=sync_engine)

if __name__ == "__main__":
    init_db()

#To run
#python -m src.cli

#To verify tables are created
    """
    sqlite3 test.db
    .tables
    """