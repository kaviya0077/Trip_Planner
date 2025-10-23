import psycopg2
from psycopg2.extras import DictCursor
import json
import uuid
from datetime import datetime
import os
import streamlit as st
import pandas as pd

def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        # This securely reads the DATABASE_URL provided by Neon, Render, or your .env file.
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable not set.")
        # check_same_thread is not a valid parameter for psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except (psycopg2.OperationalError, ValueError) as e:
        st.error(f"FATAL: Could not connect to the database: {e}")
        return None

def initialize_db():
    """
    Connects to the PostgreSQL DB, ensures schema is correct, and then closes.
    """
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cursor:
            # Using PostgreSQL-specific data types for efficiency (UUID, JSONB, SERIAL, TIMESTAMPTZ)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trips (
                    id UUID PRIMARY KEY,
                    trip_data JSONB NOT NULL,
                    prompt_length INTEGER,
                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    trip_id UUID NOT NULL,
                    rating INTEGER NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    event_type VARCHAR(255) NOT NULL,
                    trip_id UUID
                );
            """)
        conn.commit()
    except Exception as e:
        print(f"DATABASE INITIALIZATION ERROR: {e}")
    finally:
        if conn:
            conn.close()

def log_metric_event(event_type, trip_id=None):
    """Opens a connection, writes one metric, and closes immediately."""
    conn = get_db_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as cursor:
            # Using %s as the placeholder for psycopg2
            cursor.execute("INSERT INTO metrics (event_type, trip_id) VALUES (%s, %s)",
                           (event_type, trip_id))
        conn.commit()
    except Exception as e:
        print(f"DATABASE LOGGING ERROR: {e}")
    finally:
        if conn:
            conn.close()

def save_feedback_to_db(trip_id, rating):
    """Opens a connection, saves feedback, and closes immediately."""
    conn = get_db_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO feedback (trip_id, rating) VALUES (%s, %s)",
                           (trip_id, rating))
        conn.commit()
        st.session_state.feedback_given = True
    except Exception as e:
        print(f"DATABASE FEEDBACK ERROR: {e}")
    finally:
        if conn:
            conn.close()

def save_trip_to_db(trip_data, prompt_length):
    """Opens a connection, saves a trip, and closes immediately."""
    trip_id = str(uuid.uuid4())
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        with conn.cursor() as cursor:
            # We must convert the trip_data dict to a JSON string for the JSONB column.
            cursor.execute("INSERT INTO trips (id, trip_data, prompt_length) VALUES (%s, %s, %s)",
                           (trip_id, json.dumps(trip_data), prompt_length))
        conn.commit()
        log_metric_event("trip_saved", trip_id)
        return trip_id
    except Exception as e:
        st.error(f"PostgreSQL error in save_trip_to_db: {e}")
        return None
    finally:
        if conn:
            conn.close()

def load_trip_from_db(trip_id):
    """
    Opens a connection, reads a trip, and closes immediately.
    Returns the trip data dictionary or None if not found.
    """
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        # Use DictCursor to access columns by name
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            # The parameter must be passed as a tuple
            cursor.execute("SELECT trip_data FROM trips WHERE id = %s", (trip_id,))
            result = cursor.fetchone()
            if result:
                # psycopg2 automatically converts a JSONB column to a Python dict
                return result['trip_data']
            else:
                return None
    except Exception as e:
        print(f"PostgreSQL error in load_trip_from_db: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_all_data_for_dashboard():
    """Opens a connection, reads all data for the dashboard, and closes immediately."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        # Pandas read_sql works perfectly with a psycopg2 connection object
        metrics_df = pd.read_sql_query("SELECT * FROM metrics", conn)
        feedback_df = pd.read_sql_query("SELECT * FROM feedback", conn)
        trips_df = pd.read_sql_query("SELECT * FROM trips", conn)
        return metrics_df, feedback_df, trips_df
    except Exception as e:
        print(f"Dashboard Load Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    finally:
        if conn:
            conn.close()