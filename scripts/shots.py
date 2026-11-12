"""Load the UI in headless Chromium at desktop and phone sizes, check for console errors and
horizontal overflow, exercise the replay and the drawer, and save screenshots.

usage: python scripts/shots.py --url http://127.0.0.1:8101/ --out docs/assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


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
            page.goto(url, wait_until="networkidle")
            page.wait_for_function("window.__evalbench !== undefined", timeout=15000)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(out / f"{tag}{name}-1-hero.png"))
            # mid replay
            page.evaluate("window.__evalbench.seek(60)")
            page.locator("#stage").scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            page.screenshot(path=str(out / f"{tag}{name}-2-replay.png"))
            page.evaluate("const s = window.__evalbench.state(); window.__evalbench.seek(s.N + s.FLIGHT)")
            page.wait_for_timeout(300)
            page.locator("#stage").screenshot(path=str(out / f"{tag}{name}-3-map.png"))
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
                page.screenshot(path=str(out / f"{tag}{name}-4-drawer.png"))
                page.keyboard.press("Escape")
                page.wait_for_timeout(600)
                report[f"{name}_drawer_closed"] = page.evaluate("document.getElementById('drawer').hidden")
            page.locator("#panels").scroll_into_view_if_needed()
            page.wait_for_timeout(1200)
            page.screenshot(path=str(out / f"{tag}{name}-5-panels.png"), full_page=False)
            page.locator("#diff-sec").scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            page.screenshot(path=str(out / f"{tag}{name}-6-diff.png"))
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
