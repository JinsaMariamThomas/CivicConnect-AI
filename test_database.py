
from nlp_model import analyze_complaint
from database import (
    initialize_database,
    save_complaint,
    get_complaint,
    get_all_complaints,
    update_status,
    get_statistics
)

# Initialize database
initialize_database()

# Test complaint
complaint = "There is no water supply in my street"

# Analyze using NLP
analysis = analyze_complaint(complaint)

print("\n--- AI ANALYSIS ---")
print(analysis)

# Save complaint
complaint_id = save_complaint(analysis)

print("\n--- COMPLAINT SAVED ---")
print("Tracking ID:", complaint_id)

# Retrieve complaint
saved = get_complaint(complaint_id)

print("\n--- TRACKING DETAILS ---")
print(saved)

# Test status update
update_status(complaint_id, "In Progress")

print("\n--- UPDATED STATUS ---")
print(get_complaint(complaint_id)["status"])

# Display statistics
print("\n--- COMMUNITY STATISTICS ---")
print(get_statistics())

print("\nAll database tests completed!")
