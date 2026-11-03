
import sqlite3
import uuid
import re
import html
from datetime import datetime

DATABASE_NAME = "civicconnect.db"


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_connection():
    connection = sqlite3.connect(DATABASE_NAME)
    connection.row_factory = sqlite3.Row
    return connection


# --------------------------------------------------
# CLEAN COMPLAINT TEXT
# --------------------------------------------------

def clean_complaint_text(value):
    """
    Extracts the original complaint from plain text
    or old HTML content accidentally saved in the database.
    """

    if value is None:
        return "No complaint text"

    text = html.unescape(str(value))

    # Remove Markdown code fences
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "").strip()

    # If an HTML card was saved, extract its first paragraph
    paragraph = re.search(
        r"<p\b[^>]*>(.*?)</p>",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if paragraph:
        text = paragraph.group(1)

    # Remove HTML tags
    text = re.sub(r"<[^>]*>", "", text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else "No complaint text"


# --------------------------------------------------
# INITIALIZE DATABASE
# --------------------------------------------------

def initialize_database():

    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT UNIQUE NOT NULL,
            complaint TEXT NOT NULL,
            language TEXT,
            category TEXT,
            department TEXT,
            priority TEXT,
            status TEXT DEFAULT 'Pending',
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# --------------------------------------------------
# SAVE COMPLAINT
# --------------------------------------------------

def save_complaint(analysis):

    connection = get_connection()

    try:
        # Save only the actual complaint text
        complaint = clean_complaint_text(
            analysis.get("complaint", "")
        )

        complaint_id = "CCA-" + uuid.uuid4().hex[:8].upper()

        created_at = datetime.now().strftime(
            "%d-%m-%Y %I:%M %p"
        )

        connection.execute("""
            INSERT INTO complaints (
                complaint_id,
                complaint,
                language,
                category,
                department,
                priority,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            complaint_id,
            complaint,
            analysis.get("language", "Unknown"),
            analysis.get("category", "Other"),
            analysis.get("department", "Not assigned"),
            analysis.get("priority", "Normal"),
            "Pending",
            created_at
        ))

        connection.commit()

        return complaint_id

    finally:
        connection.close()


# --------------------------------------------------
# GET ONE COMPLAINT
# --------------------------------------------------

def get_complaint(complaint_id):

    connection = get_connection()

    try:
        complaint = connection.execute("""
            SELECT * FROM complaints
            WHERE complaint_id = ?
        """, (complaint_id.strip().upper(),)).fetchone()

        if complaint:
            result = dict(complaint)

            # Clean old complaint text when retrieving it
            result["complaint"] = clean_complaint_text(
                result["complaint"]
            )

            return result

        return None

    finally:
        connection.close()


# --------------------------------------------------
# GET ALL COMPLAINTS
# --------------------------------------------------

def get_all_complaints():

    connection = get_connection()

    try:
        complaints = connection.execute("""
            SELECT * FROM complaints
            ORDER BY id DESC
        """).fetchall()

        results = []

        for row in complaints:
            result = dict(row)

            # Clean old HTML records for display
            result["complaint"] = clean_complaint_text(
                result["complaint"]
            )

            results.append(result)

        return results

    finally:
        connection.close()


# --------------------------------------------------
# UPDATE COMPLAINT STATUS
# --------------------------------------------------

def update_status(complaint_id, status):

    allowed_statuses = [
        "Pending",
        "In Progress",
        "Resolved"
    ]

    if status not in allowed_statuses:
        return False

    connection = get_connection()

    try:
        cursor = connection.execute("""
            UPDATE complaints
            SET status = ?
            WHERE complaint_id = ?
        """, (status, complaint_id.strip().upper()))

        connection.commit()

        return cursor.rowcount > 0

    finally:
        connection.close()


# --------------------------------------------------
# GET STATISTICS
# --------------------------------------------------

def get_statistics():

    connection = get_connection()

    try:
        total = connection.execute(
            "SELECT COUNT(*) FROM complaints"
        ).fetchone()[0]

        pending = connection.execute(
            "SELECT COUNT(*) FROM complaints WHERE status = 'Pending'"
        ).fetchone()[0]

        progress = connection.execute(
            "SELECT COUNT(*) FROM complaints WHERE status = 'In Progress'"
        ).fetchone()[0]

        resolved = connection.execute(
            "SELECT COUNT(*) FROM complaints WHERE status = 'Resolved'"
        ).fetchone()[0]

        return {
            "total": total,
            "pending": pending,
            "in_progress": progress,
            "resolved": resolved
        }

    finally:
        connection.close()


# --------------------------------------------------
# RUN DATABASE
# --------------------------------------------------

if __name__ == "__main__":

    initialize_database()

    print(
        "CivicConnect AI database initialized successfully!"
    )
