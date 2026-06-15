"""Feature-building modules that sit between ETL and models.

This package contains reusable feature logic, including match-level train,
validation, and test split assignment plus Step 3 smoothed possession sequence
construction. Model files should consume these outputs instead of duplicating
feature-building logic.
"""
