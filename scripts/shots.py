"""Load the UI in headless Chromium at desktop and phone sizes, check for console errors and
horizontal overflow, exercise the replay and the drawer, and save screenshots.

usage: python scripts/shots.py --url http://127.0.0.1:8101/ --out docs/assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

# Only the screenshots the README embeds are written; the other steps still run for the checks below.
KEPT = {"desktop-3-map", "desktop-3b-raw", "desktop-2b-flight", "phone-replay-end"}


def shot(target, out: Path, stem: str) -> None:
    """Write the screenshot only if the README uses it."""
    if any(stem.endswith(k) for k in KEPT):
        target.screenshot(path=str(out / f"{stem}.png"))


def run(url: str, out: Path, tag: str = "") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, kw in {
            "desktop": dict(viewport={"width": 1440, "height": 900}),
            "phone": dict(viewport={"width": 390, "height": 844}, device_scale_factor=2, has_touch=True, is_mobile=True),
        }.items():
            ctx = browser.new_context(**kw)
            page = ctx.new_page()
            errors: list[str] = []
            page.on("console", lambda m, e=errors: m.type == "error" and e.append(m.text))
            page.on("pageerror", lambda x, e=errors: e.append(str(x)))
            page.on("response", lambda r, e=errors: r.status >= 400 and e.append(f"HTTP {r.status} {r.url}"))
            page.on("requestfailed", lambda r, e=errors: e.append(f"request failed {r.url}"))
            page.goto(url, wait_until="networkidle")
            page.wait_for_function("window.__evalbench !== undefined", timeout=15000)
            page.wait_for_timeout(1500)
            # the end of the replay: every prompt scored, every failing sample landed on its island
            page.evaluate("document.fonts.ready")
            page.evaluate("const s = window.__evalbench.state(); window.__evalbench.seek(s.N + s.FLIGHT)")
            if name == "phone":
                # controls just below the sticky bar, so the counter, the lanes and the whole map are in the frame
                page.evaluate(
                    "const h = document.querySelector('header').getBoundingClientRect().height;"
                    "window.scrollTo({top: document.querySelector('.controls').getBoundingClientRect().top + window.scrollY - h - 12, behavior: 'instant'})"
                )
                page.wait_for_timeout(500)
                report["phone_counter_at_end"] = page.inner_text("#count")
                bottom = page.evaluate("document.querySelector('#stage').getBoundingClientRect().bottom")
                if any(f"{tag}{name}-replay-end".endswith(k) for k in KEPT):
                    # cut at the bottom edge of the stage card, not through the toggles below it
                    page.screenshot(path=str(out / f"{tag}{name}-replay-end.png"), clip={"x": 0, "y": 0, "width": 390, "height": bottom + 10})
            page.locator("#stage").scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            shot(page.locator("#stage"), out, f"{tag}{name}-3-map")
            # raw projection view
            page.evaluate("window.__evalbench.view('raw')")
            page.wait_for_timeout(200)
            shot(page.locator("#stage"), out, f"{tag}{name}-3b-raw")
            page.evaluate("window.__evalbench.view('islands')")
            page.wait_for_timeout(200)
            # frame rate while the replay plays (headless, software rendering: a floor, not a phone measurement)
            page.evaluate("window.__evalbench.seek(5); window.__evalbench.play(true)")
            report[f"{name}_fps_while_playing"] = page.evaluate("window.__evalbench.fps()")
            page.evaluate("window.__evalbench.play(false)")
            # a frame with dots in flight
            page.evaluate("window.__evalbench.seek(150)")
            page.locator("#stage").scroll_into_view_if_needed()
            page.wait_for_timeout(200)
            shot(page.locator("#stage"), out, f"{tag}{name}-2b-flight")
            page.evaluate("const s = window.__evalbench.state(); window.__evalbench.seek(s.N + s.FLIGHT)")
            page.wait_for_timeout(200)
            # tap a dot
            pos = page.evaluate("window.__evalbench.dot(0)")
            if pos:
                page.evaluate(f"window.scrollTo({{top: document.getElementById('canvas').getBoundingClientRect().top + window.scrollY + {pos['y']} - 300, behavior: 'instant'}})")
                page.wait_for_timeout(300)
                box = page.locator("#canvas").bounding_box()
                x, y = box["x"] + pos["x"], box["y"] + pos["y"]
                if kw.get("has_touch"):
                    page.touchscreen.tap(x, y)
                else:
                    page.mouse.click(x, y)
                page.wait_for_timeout(900)
                report[f"{name}_drawer_open"] = page.evaluate("!document.getElementById('drawer').hidden")
                page.keyboard.press("Escape")
                page.wait_for_timeout(600)
                report[f"{name}_drawer_closed"] = page.evaluate("document.getElementById('drawer').hidden")
            page.locator("#panels").scroll_into_view_if_needed()
            page.wait_for_timeout(1200)
            page.locator("#diff-sec").scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            report[f"{name}_overflow_x"] = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            small = page.evaluate(
                "[...document.querySelectorAll('button,input[type=range],a.mark,.links a,.checks label')]"
                ".filter(e => e.offsetParent && (e.getBoundingClientRect().height < 43.5 || e.getBoundingClientRect().width < 43.5))"
                ".map(e => (e.id || e.className || e.tagName) + '[' + (e.textContent || '').slice(0, 14) + ']:' + Math.round(e.getBoundingClientRect().width) + 'x' + Math.round(e.getBoundingClientRect().height))"
            )
            report[f"{name}_small_targets"] = small[:12]
            report[f"{name}_console_errors"] = errors
            ctx.close()
        browser.close()
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8101/")
    ap.add_argument("--out", default="docs/assets")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    print(json.dumps(run(a.url, Path(a.out), a.tag), indent=1))
