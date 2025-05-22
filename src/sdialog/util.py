# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import json

def make_serializable(data:dict):

    for key, value in data.items():
        try:
            json.dumps(value)
        except (TypeError, OverflowError):
            data[key] = str(value)

    return data
