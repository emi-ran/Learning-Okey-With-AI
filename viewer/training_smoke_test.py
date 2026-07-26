"""Playwright smoke test for the live training dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:4173")
    parser.add_argument(
        "--status",
        default="/training_runs/live-smoke-2/status.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("viewer_artifacts/training"),
    )
    parser.add_argument("--expected-episodes", type=int, default=2)
    parser.add_argument("--expected-checkpoints", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_url = (
        f"{args.base_url}/viewer/training.html"
        f"?status={quote(args.status, safe='/%')}"
    )

    console_errors: list[str] = []
    page_errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        desktop = browser.new_page(viewport={"width": 1600, "height": 1000})
        desktop.on(
            "console",
            lambda message: (
                console_errors.append(message.text)
                if message.type == "error"
                else None
            ),
        )
        desktop.on("pageerror", lambda error: page_errors.append(str(error)))
        desktop.goto(dashboard_url, wait_until="networkidle")
        desktop.wait_for_function(
            "() => document.querySelector('#connectionText').textContent "
            "=== 'EĞİTİM TAMAMLANDI'"
        )
        expected_episodes = str(args.expected_episodes)
        assert desktop.locator("#episodeCurrent").inner_text() == expected_episodes
        assert desktop.locator("#episodeTarget").inner_text() == expected_episodes
        assert (
            desktop.locator(".checkpoint-card").count()
            == args.expected_checkpoints
        )
        assert (
            desktop.locator(".checkpoint-card[data-status='ready']").count()
            == args.expected_checkpoints
        )
        assert (
            desktop.locator(".video-shell video").count()
            == args.expected_checkpoints
        )
        assert (
            desktop.locator(".checkpoint-ticks span.ready").count()
            == args.expected_checkpoints
        )
        assert desktop.locator("#episodeLog li").count() == min(
            8,
            args.expected_episodes,
        )
        desktop.screenshot(
            path=str(args.output_dir / "training-desktop.png"),
            full_page=True,
        )

        replay_href = desktop.locator(".replay-link").first.get_attribute("href")
        assert replay_href
        replay_page = browser.new_page(viewport={"width": 1440, "height": 900})
        replay_page.goto(f"{args.base_url}{replay_href}", wait_until="networkidle")
        replay_page.wait_for_function(
            "() => document.querySelector('#replayStatus').textContent === 'REPLAY'"
        )
        assert replay_page.locator("#timeline").get_attribute("max") != "3"
        replay_page.close()

        mobile = browser.new_page(viewport={"width": 360, "height": 800})
        mobile.goto(dashboard_url, wait_until="networkidle")
        mobile.wait_for_function(
            "() => document.querySelector('#connectionText').textContent "
            "=== 'EĞİTİM TAMAMLANDI'"
        )
        assert mobile.evaluate(
            "() => document.documentElement.scrollWidth "
            "<= document.documentElement.clientWidth"
        )
        mobile.screenshot(
            path=str(args.output_dir / "training-mobile.png"),
            full_page=True,
        )
        browser.close()

    assert not console_errors, console_errors
    assert not page_errors, page_errors
    print("training dashboard smoke test passed")


if __name__ == "__main__":
    main()
