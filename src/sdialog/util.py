"""
util: Utility Functions for sdialog

This module provides helper functions for the sdialog package, including serialization utilities to ensure
objects can be safely converted to JSON for storage or transmission.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import json

def make_serializable(data:dict):
    """
    Converts non-serializable values in a dictionary to strings so the dictionary can be safely serialized to JSON.

    Args:
        data (dict): The dictionary to process.

    Returns:
        dict: The dictionary with all values JSON-serializable.
    """

    for key, value in data.items():
        try:
            json.dumps(value)
        except (TypeError, OverflowError):
            data[key] = str(value)

    return data
