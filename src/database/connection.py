"""
Database connection management for AWS RDS PostgreSQL
Handles connection pooling, session management, and configuration
"""
import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import yaml
from dotenv import load_dotenv

from src.database.models import Base

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages database connection and session lifecycle for AWS RDS PostgreSQL
    Implements singleton pattern for connection pooling
    """

    _instance = None
    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize database connection (only once due to singleton)"""
        if self._engine is None:
            self._initialize_connection()

    def _load_config(self) -> dict:
        """Load database configuration from config.yaml and environment variables"""
        # Load config.yaml
        config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Override with environment variables if present
        db_config = config.get('database', {})
        db_config['host'] = os.getenv('DB_HOST', db_config.get('host', 'localhost'))
        db_config['port'] = int(os.getenv('DB_PORT', db_config.get('port', 5432)))
        db_config['database'] = os.getenv('DB_NAME', db_config.get('database', 'indic_annotation'))
        db_config['user'] = os.getenv('DB_USER', db_config.get('user', 'postgres'))
        db_config['password'] = os.getenv('DB_PASSWORD', db_config.get('password', ''))
        db_config['pool_size'] = int(os.getenv('DB_POOL_SIZE', db_config.get('pool_size', 5)))
        db_config['max_overflow'] = int(os.getenv('DB_MAX_OVERFLOW', db_config.get('max_overflow', 10)))
        db_config['echo'] = os.getenv('DB_ECHO', str(db_config.get('echo', False))).lower() == 'true'

        return db_config

    def _build_connection_url(self, config: dict) -> str:
        """
        Build PostgreSQL connection URL for AWS RDS
        Format: postgresql://user:password@host:port/database

        For SSL connection to AWS RDS, append ?sslmode=require
        """
        url = (
            f"postgresql://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['database']}"
        )

        # Add SSL for AWS RDS (recommended for production)
        if os.getenv('DB_SSL', 'true').lower() == 'true':
            url += "?sslmode=require"

        return url

    def _initialize_connection(self):
        """Initialize database engine and session factory with connection pooling"""
        try:
            config = self._load_config()
            connection_url = self._build_connection_url(config)

            # Create engine with connection pooling optimized for AWS RDS
            self._engine = create_engine(
                connection_url,
                poolclass=QueuePool,
                pool_size=config['pool_size'],
                max_overflow=config['max_overflow'],
                pool_pre_ping=True,  # Verify connections before using them
                pool_recycle=3600,   # Recycle connections after 1 hour
                echo=config['echo'],  # Set to True for SQL query debugging
                connect_args={
                    "connect_timeout": 10,  # Connection timeout in seconds
                    "options": "-c timezone=utc"  # Set timezone to UTC
                }
            )

            # Set up connection event listeners for better error handling
            @event.listens_for(self._engine, "connect")
            def receive_connect(dbapi_conn, connection_record):
                """Called when a new DB-API connection is created"""
                logger.debug("New database connection established")

            @event.listens_for(self._engine, "checkout")
            def receive_checkout(dbapi_conn, connection_record, connection_proxy):
                """Called when a connection is retrieved from the pool"""
                logger.debug("Connection checked out from pool")

            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )

            logger.info(f"Database connection initialized successfully: {config['host']}:{config['port']}/{config['database']}")

        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise

    @property
    def engine(self) -> Engine:
        """Get SQLAlchemy engine"""
        if self._engine is None:
            self._initialize_connection()
        return self._engine

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions

        Usage:
            with db_connection.get_session() as session:
                # Use session here
                session.query(Document).all()
        """
        if self._session_factory is None:
            self._initialize_connection()

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def create_all_tables(self):
        """Create all tables defined in models (only use in development/testing)"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("All database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def drop_all_tables(self):
        """Drop all tables (WARNING: use with caution!)"""
        try:
            Base.metadata.drop_all(self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise

    def test_connection(self) -> bool:
        """
        Test database connection
        Returns True if connection is successful, False otherwise
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def close(self):
        """Close all database connections and dispose of the engine"""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")


# Global instance (singleton)
db_connection = DatabaseConnection()


# Convenience functions
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Convenience function to get a database session

    Usage:
        with get_db_session() as session:
            documents = session.query(Document).all()
    """
    with db_connection.get_session() as session:
        yield session


def init_db():
    """Initialize database tables (use for initial setup)"""
    db_connection.create_all_tables()


def test_db_connection() -> bool:
    """Test database connection"""
    return db_connection.test_connection()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test connection
    print("Testing database connection...")
    if test_db_connection():
        print("✅ Connection successful!")

        # Optionally create tables (uncomment for initial setup)
        # print("Creating database tables...")
        # init_db()
        # print("✅ Tables created!")
    else:
        print("❌ Connection failed!")
