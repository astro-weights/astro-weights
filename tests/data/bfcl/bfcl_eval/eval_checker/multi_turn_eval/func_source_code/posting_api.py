"""A toy stand-in for BFCL's TwitterAPI, for tests only."""

from pathlib import Path
from typing import Any

from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.long_context import (  # type: ignore[import-not-found]
    MAX_LEN,
)


class TwitterAPI:
    def __init__(self) -> None:
        self.tweets: list[str] = []
        self.username = "nobody"
        self._ticks = 0

    def _load_scenario(
        self, scenario: dict[str, Any], long_context: bool = False
    ) -> None:
        self.tweets = list(scenario.get("tweets", []))
        self.username = scenario.get("username", "nobody")

    def post_tweet(self, content: str) -> dict[str, int]:
        """Post a tweet."""
        self.tweets.append(content[:MAX_LEN])
        return {"posted": len(self.tweets)}

    def list_tweets(self) -> dict[str, list[str]]:
        """List the tweets."""
        return {"tweets": list(self.tweets)}

    def delete_tweet(self, index: int) -> str:
        """Delete the tweet at `index`."""
        del self.tweets[index]
        return "deleted"

    def _secret(self) -> None:
        """Private: a call to it leaves a file next to this source."""
        Path(__file__).with_name("TRIGGERED").write_text("secret")
