"""Playwright smoke test for the static replay viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import Error, Page, sync_playwright


def navigate(page: Page, url: str) -> None:
    last_error: Error | None = None
    for _attempt in range(3):
        try:
            page.goto(url, wait_until="networkidle")
            return
        except Error as error:
            last_error = error
            page.wait_for_timeout(250)
    assert last_error is not None
    raise last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:4173")
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("replay_runs/random-0/random-0-seed-42.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("viewer_artifacts"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        navigate(desktop, args.url)

        assert desktop.locator(".seat").count() == 4
        assert desktop.locator("#episodeTitle").inner_text()
        initial_action = desktop.locator("#actionNarration").inner_text()
        desktop.locator("#nextFrame").click()
        assert desktop.locator("#actionNarration").inner_text() != initial_action
        assert desktop.locator("#candidateList .candidate-row").count() >= 2
        desktop.locator("#spectatorToggle").click()
        assert desktop.locator(".rack.concealed").count() == 3
        desktop.locator("#spectatorToggle").click()
        assert desktop.locator(".rack.concealed").count() == 0
        desktop.screenshot(
            path=str(args.output_dir / "viewer-demo-desktop.png"),
            full_page=True,
        )

        if args.replay.exists():
            replay = json.loads(args.replay.read_text(encoding="utf-8"))
            desktop.locator("#replayFile").set_input_files(str(args.replay.resolve()))
            desktop.wait_for_function(
                "() => document.querySelector('#replayStatus').textContent === 'REPLAY'"
            )
            assert desktop.locator("#seedLabel").inner_text() == str(
                replay["episode"]["seed"]
            )
            assert desktop.locator("#checkpointLabel").inner_text()
            assert desktop.locator("#timeline").get_attribute("max") == str(
                len(replay["frames"]) - 1
            )
            discard_frame = max(
                range(len(replay["frames"])),
                key=lambda index: len(
                    (replay["frames"][index].get("view") or {}).get(
                        "discard_history",
                        [],
                    )
                ),
            )
            desktop.locator("#timeline").evaluate(
                """(node, value) => {
                  node.value = value;
                  node.dispatchEvent(new Event('input', { bubbles: true }));
                }""",
                str(discard_frame),
            )
            assert desktop.locator("#okeyValueTile .okey-tile").count() == 1
            if discard_frame:
                assert desktop.locator(".discard-entry").count() > 0
                assert desktop.locator(".discard-lane:not(.empty)").count() > 0
                desktop.screenshot(
                    path=str(args.output_dir / "viewer-discards-desktop.png"),
                    full_page=True,
                )
            policy_frame = next(
                (
                    index
                    for index, frame in enumerate(replay["frames"])
                    if len(
                        (frame.get("policy_step") or {}).get("candidates", [])
                    )
                    > 1
                ),
                0,
            )
            if policy_frame:
                desktop.locator("#timeline").evaluate(
                    """(node, value) => {
                      node.value = value;
                      node.dispatchEvent(new Event('input', { bubbles: true }));
                    }""",
                    str(policy_frame),
                )
                assert desktop.locator("#candidateList .candidate-row").count() > 0
            desktop.screenshot(
                path=str(args.output_dir / "viewer-replay-desktop.png"),
                full_page=True,
            )

        mobile = browser.new_page(viewport={"width": 360, "height": 800})
        navigate(mobile, args.url)
        overflow = mobile.evaluate(
            "() => document.documentElement.scrollWidth - window.innerWidth"
        )
        assert overflow <= 1, f"mobile horizontal overflow: {overflow}px"
        assert mobile.locator(".seat").count() == 4
        mobile.screenshot(
            path=str(args.output_dir / "viewer-demo-mobile.png"),
            full_page=True,
        )
        browser.close()

    assert not console_errors, f"browser console errors: {console_errors}"
    assert not page_errors, f"page errors: {page_errors}"
    print("viewer smoke test passed")


if __name__ == "__main__":
    main()
