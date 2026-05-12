"""
remote_prediction.py – Two separate remote services for prediction.

Flow 1 (Horary):     ≤1 month queries → services.futurepointindia.com/horary
                      Returns short interpretation (~20-50 words)
Flow 2 (Timing):     >1 month queries → services.futurepointindia.com/timing
                      Returns detailed prediction (~500 words)

Both use placeholder responses until real endpoints are configured.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Remote Service URLs (to be configured later) ─────────────────────────────
HORARY_REMOTE_URL = "https://services.futurepointindia.com/horary"  # ≤1 month
TIMING_REMOTE_URL = "https://services.futurepointindia.com/timing"  # >1 month


def fetch_horary_prediction(horary_data: Dict[str, Any]) -> str:
    """
    Flow 1: Horary remote service for short-term queries (≤1 month).

    Sends houses, planets, relationships to Horary server.
    Returns short contextual interpretation (~20-50 words).

    Args:
        horary_data: {"h": [7, 10], "p": [6], "rel": 1}

    Returns:
        Horary interpretation text
    """
    try:
        # ── TODO: Wire real endpoint ──────────────────────────────────────
        # import requests as http_requests
        # payload = {
        #     "houses": horary_data.get("h", []),
        #     "planets": horary_data.get("p", []),
        #     "relation": horary_data.get("rel"),
        # }
        # resp = http_requests.post(HORARY_REMOTE_URL, json=payload, timeout=5)
        # resp.raise_for_status()
        # return resp.json().get("interpretation", "")
        # ──────────────────────────────────────────────────────────────────

        logger.info(
            f"[HoraryRemote] Placeholder response for houses={horary_data.get('h', [])}, "
            f"planets={horary_data.get('p', [])}"
        )
        return "You will be good Always"

    except Exception as e:
        logger.error(f"[HoraryRemote] Failed: {e}")
        return "You will be good Always"


def fetch_timing_prediction(
    user_details: Dict[str, Any],
    time_period: Dict[str, Any],
) -> str:
    """
    Flow 2: Timing remote service for long-term queries (>1 month).

    Sends user details + prediction period to Timing server.
    Returns detailed prediction with contextual data (~500 words).

    Args:
        user_details: User chart/profile information
        time_period: {"t": "Yearly", "v": "2026"}

    Returns:
        Detailed prediction text
    """
    try:
        # ── TODO: Wire real endpoint ──────────────────────────────────────
        # import requests as http_requests
        # payload = {
        #     "user": user_details,
        #     "period": time_period,
        # }
        # resp = http_requests.post(TIMING_REMOTE_URL, json=payload, timeout=10)
        # resp.raise_for_status()
        # return resp.json().get("prediction", "")
        # ──────────────────────────────────────────────────────────────────

        logger.info(
            f"[TimingRemote] Placeholder response for period={time_period}"
        )
        return "You will be good Always"

    except Exception as e:
        logger.error(f"[TimingRemote] Failed: {e}")
        return "You will be good Always"
