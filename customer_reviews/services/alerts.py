"""Slack alerts for high-urgency complaints (optional)."""

import json
import urllib.request
from collections.abc import Callable

import pandas as pd


MAX_ALERT_ITEMS = 10
SLACK_WEBHOOK_PREFIX = "https://hooks.slack.com/"


def escape_slack(text) -> str:
    """
    Escape Slack's control characters.

    Summaries come from customer content; without escaping, text such as
    <https://evil.example|Click here> or <!channel> would become a link or
    a mass mention in Slack.
    """

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_urgent_alert(urgent: pd.DataFrame, max_items: int = MAX_ALERT_ITEMS) -> dict:
    """Build the Slack message payload for high-urgency complaints."""

    lines = [f"*:rotating_light: {len(urgent)} high-urgency customer complaint(s) need review*"]

    for record in urgent.head(max_items).to_dict(orient="records"):
        lines.append(
            "• *{id}* ({platform}, {rating}★, {category}): {summary}".format(
                id=escape_slack(record.get("review_id")),
                platform=escape_slack(record.get("platform")),
                rating=escape_slack(record.get("rating")),
                category=escape_slack(record.get("category")),
                summary=escape_slack(record.get("summary")),
            )
        )

    if len(urgent) > max_items:
        lines.append(f"…and {len(urgent) - max_items} more. Open the app for the full list.")

    return {"text": "\n".join(lines)}


def _post_json(url: str, payload: dict, timeout: float) -> int:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.status


def send_slack_alert(
    webhook_url: str,
    payload: dict,
    post: Callable[[str, dict, float], int] = _post_json,
    timeout: float = 10,
) -> None:
    """Post the payload to a Slack incoming webhook."""

    if not webhook_url.startswith(SLACK_WEBHOOK_PREFIX):
        raise ValueError("The Slack webhook URL must start with https://hooks.slack.com/")

    status = post(webhook_url, payload, timeout)

    if status != 200:
        raise RuntimeError(f"Slack returned HTTP {status}")
