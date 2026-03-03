"""
TruthLens — Fake News Detection Engine
======================================
Python backend with NLP heuristics + matplotlib visualization.
Run: python fake_news_detector.py

Requirements:
    pip install matplotlib numpy
"""

import re
import math
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime

# ─────────────────────────────────────────────
# 1.  LEXICAL DICTIONARIES
# ─────────────────────────────────────────────

FAKE_INDICATORS = {
    "sensational": [
        "shocking", "bombshell", "explosive", "breaking", "unbelievable",
        "jaw-dropping", "nobody expected", "you won't believe", "incredible",
        "stunning", "mind-blowing", "outrageous",
    ],
    "emotional": [
        "terrifying", "horrifying", "disgraceful", "outrage", "scandal",
        "corrupt", "evil", "dangerous", "destroy", "attack", "threat",
        "crisis", "catastrophe", "disaster",
    ],
    "bad_hedging": [
        "allegedly", "reportedly", "sources say", "some say", "many believe",
        "experts claim", "it is said", "according to insiders",
        "anonymous sources", "secret", "hidden",
    ],
    "superlatives": [
        "best ever", "worst ever", "greatest", "most corrupt", "never before",
        "all time", "100%", "absolutely", "completely", "totally",
        "everyone", "nobody",
    ],
    "clickbait": [
        "must see", "you need to know", "what they don't want you to know",
        "the truth about", "exposed", "leaked", "banned", "censored", "silenced",
    ],
    "conspiracy": [
        "deep state", "globalist", "mainstream media lies", "cover-up",
        "false flag", "they don't want you to know", "wake up",
        "sheep", "sheeple", "puppet", "agenda",
    ],
}

REAL_INDICATORS = {
    "attribution": [
        "according to", "said", "reported", "confirmed", "stated", "announced",
        "spokesperson", "official", "told reporters", "press release",
    ],
    "proper_hedging": [
        "could", "may", "might", "suggests", "appears", "indicates",
        "research shows", "study found", "data indicates", "analysis suggests",
    ],
    "specificity": [
        "percent", "million", "billion", "monday", "tuesday", "wednesday",
        "thursday", "friday", "january", "february", "march", "april",
        "may", "june", "july", "august", "september", "october",
        "november", "december",
    ],
    "formal": [
        "however", "furthermore", "nevertheless", "consequently", "therefore",
        "moreover", "in addition", "as a result", "by contrast", "according to data",
    ],
}

# ─────────────────────────────────────────────
# 2.  CORE ANALYSIS ENGINE
# ─────────────────────────────────────────────

def count_matches(text: str, phrases: list[str]) -> int:
    count = 0
    for phrase in phrases:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        count += len(pattern.findall(text))
    return count


def collect_matches(text: str, phrases: list[str]) -> list[str]:
    found = []
    for phrase in phrases:
        if phrase.lower() in text.lower() and phrase not in found:
            found.append(phrase)
    return found


def analyze(text: str) -> dict:
    lower = text.lower()
    words = re.findall(r'\b\w+\b', lower)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]

    # --- dimension scores (0-100) ---
    sens_raw = (
        count_matches(lower, FAKE_INDICATORS["sensational"])
        + count_matches(lower, FAKE_INDICATORS["superlatives"]) * 0.8
        + count_matches(lower, FAKE_INDICATORS["clickbait"]) * 1.2
        + count_matches(lower, FAKE_INDICATORS["conspiracy"]) * 1.5
    )
    emo_raw  = count_matches(lower, FAKE_INDICATORS["emotional"])
    hedge_raw= count_matches(lower, FAKE_INDICATORS["bad_hedging"])
    spec_raw = count_matches(lower, REAL_INDICATORS["specificity"])
    attr_raw = count_matches(lower, REAL_INDICATORS["attribution"])
    form_raw = (
        count_matches(lower, REAL_INDICATORS["formal"])
        + count_matches(lower, REAL_INDICATORS["proper_hedging"]) * 0.8
    )

    # Caps & punctuation penalty
    excl      = len(re.findall(r'!', text))
    all_caps  = len(re.findall(r'\b[A-Z]{3,}\b', text))
    caps_bonus= min(30, all_caps * 8 + excl * 5)

    scores = {
        "Sensationalism":  min(100, sens_raw  * 18 + caps_bonus),
        "Emotional Lang.": min(100, emo_raw   * 15),
        "Anon. Hedging":   min(100, hedge_raw * 20),
        "Specificity":     min(100, spec_raw  * 14 + 10),
        "Attribution":     min(100, attr_raw  * 18 + 5),
        "Formal Tone":     min(100, form_raw  * 16 + 5),
    }

    # Weighted fake probability
    fake_score = (
          scores["Sensationalism"]  * 0.30
        + scores["Emotional Lang."] * 0.20
        + scores["Anon. Hedging"]   * 0.20
        - scores["Specificity"]     * 0.12
        - scores["Attribution"]     * 0.10
        - scores["Formal Tone"]     * 0.08
    )
    fake_pct = max(0, min(100, round(fake_score)))
    real_pct = 100 - fake_pct

    if fake_pct >= 65:
        verdict, confidence = "FAKE", fake_pct
    elif fake_pct <= 35:
        verdict, confidence = "LIKELY REAL", real_pct
    else:
        verdict, confidence = "UNCERTAIN", 50 + abs(fake_pct - 50)

    emotional_words = count_matches(lower, FAKE_INDICATORS["emotional"]) + \
                      count_matches(lower, FAKE_INDICATORS["sensational"])

    flagged_fake  = collect_matches(lower,
        FAKE_INDICATORS["sensational"] + FAKE_INDICATORS["emotional"] +
        FAKE_INDICATORS["clickbait"] + FAKE_INDICATORS["conspiracy"])[:10]
    flagged_real  = collect_matches(lower, REAL_INDICATORS["attribution"] +
                                    REAL_INDICATORS["formal"])[:6]
    flagged_hedge = collect_matches(lower, FAKE_INDICATORS["bad_hedging"])[:5]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "fake_pct": fake_pct,
        "real_pct": real_pct,
        "scores": scores,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "emotional_words": emotional_words,
        "flagged_fake": flagged_fake,
        "flagged_real": flagged_real,
        "flagged_hedge": flagged_hedge,
    }

# ─────────────────────────────────────────────
# 3.  VISUALISATION
# ─────────────────────────────────────────────

DARK_BG   = "#0a0a0f"
SURFACE   = "#12121a"
SURFACE2  = "#1c1c28"
ACCENT    = "#e8ff47"
RED       = "#ff4757"
GREEN     = "#2ed573"
ORANGE    = "#ffa502"
BLUE      = "#1e90ff"
MUTED     = "#6b6b80"
TEXT      = "#f0f0f8"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   SURFACE,
    "axes.edgecolor":   SURFACE2,
    "axes.labelcolor":  MUTED,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       SURFACE2,
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
    "font.size":        9,
})


def _verdict_color(verdict: str) -> str:
    return RED if verdict == "FAKE" else GREEN if verdict == "LIKELY REAL" else ORANGE


def plot_results(result: dict, text: str, save_path: str | None = None):
    scores = result["scores"]
    keys   = list(scores.keys())
    vals   = [round(v) for v in scores.values()]
    vc     = _verdict_color(result["verdict"])

    fig = plt.figure(figsize=(14, 10), facecolor=DARK_BG)
    fig.suptitle(
        "TruthLens — Fake News Detection Report",
        fontsize=16, color=ACCENT, fontweight="bold",
        y=0.98, fontfamily="monospace"
    )

    gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                  left=0.06, right=0.97, top=0.92, bottom=0.06)

    # ── A. Verdict box (top-left, spans 2 cols) ─────────────────────────────
    ax_verdict = fig.add_subplot(gs[0, :2])
    ax_verdict.set_facecolor(DARK_BG)
    ax_verdict.axis("off")

    icon = "🚨" if result["verdict"] == "FAKE" else ("✅" if result["verdict"] == "LIKELY REAL" else "⚠️")
    ax_verdict.text(0.0, 0.75, f"{icon}  {result['verdict']}",
                    color=vc, fontsize=22, fontweight="bold",
                    transform=ax_verdict.transAxes)
    ax_verdict.text(0.0, 0.40,
                    f"Confidence: {result['confidence']}%   |   "
                    f"Fake score: {result['fake_pct']}%   |   "
                    f"Words: {result['word_count']}   |   "
                    f"Sentences: {result['sentence_count']}   |   "
                    f"Emotional words: {result['emotional_words']}",
                    color=MUTED, fontsize=9,
                    transform=ax_verdict.transAxes)

    snippet = textwrap.shorten(text, width=120, placeholder="…")
    ax_verdict.text(0.0, 0.08, f'"{snippet}"',
                    color="#444460", fontsize=8, style="italic",
                    transform=ax_verdict.transAxes)

    # ── B. Confidence gauge (top-right) ──────────────────────────────────────
    ax_gauge = fig.add_subplot(gs[0, 2])
    ax_gauge.set_facecolor(DARK_BG)
    ax_gauge.axis("off")
    ax_gauge.set_title("Fake Probability", color=MUTED, fontsize=8, pad=4)

    theta = np.linspace(np.pi, 0, 200)
    ax_gauge.plot(np.cos(theta), np.sin(theta), color=SURFACE2, lw=12, solid_capstyle="round")
    angle = np.pi - (result["fake_pct"] / 100) * np.pi
    fill_theta = np.linspace(np.pi, angle, 200)
    ax_gauge.plot(np.cos(fill_theta), np.sin(fill_theta), color=vc, lw=12, solid_capstyle="round")

    ax_gauge.text(0, -0.22, f"{result['fake_pct']}%", ha="center", va="center",
                  color=vc, fontsize=20, fontweight="bold")
    ax_gauge.text(0, -0.52, "FAKE SCORE", ha="center", fontsize=7, color=MUTED)
    ax_gauge.set_xlim(-1.3, 1.3); ax_gauge.set_ylim(-0.7, 1.1)

    # ── C. Radar chart (mid-left) ─────────────────────────────────────────────
    ax_radar = fig.add_subplot(gs[1, 0], polar=True)
    ax_radar.set_facecolor(SURFACE)
    ax_radar.set_title("Credibility Radar", color=MUTED, fontsize=8, pad=14)

    # Credibility = invert bad dimensions
    cred_vals = [
        100 - scores["Sensationalism"],
        100 - scores["Emotional Lang."],
        100 - scores["Anon. Hedging"],
        scores["Specificity"],
        scores["Attribution"],
        scores["Formal Tone"],
    ]
    cred_labels = ["Low Sens.", "Low Emotion", "Proper Hedge", "Specificity", "Attribution", "Formal"]
    N = len(cred_labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    cred_vals_plot = cred_vals + cred_vals[:1]

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(cred_labels, size=7, color=MUTED)
    ax_radar.set_yticks([25, 50, 75, 100])
    ax_radar.set_yticklabels(["25", "50", "75", "100"], size=5, color=SURFACE2)
    ax_radar.set_ylim(0, 100)
    ax_radar.plot(angles, cred_vals_plot, color=ACCENT, linewidth=2)
    ax_radar.fill(angles, cred_vals_plot, alpha=0.12, color=ACCENT)
    ax_radar.grid(color=SURFACE2, linewidth=0.5)
    ax_radar.spines["polar"].set_color(SURFACE2)

    # ── D. Bar chart – dimension scores (mid-center) ──────────────────────────
    ax_bar = fig.add_subplot(gs[1, 1])
    bar_colors = [RED, RED, ORANGE, GREEN, BLUE, GREEN]
    bars = ax_bar.barh(keys, vals, color=bar_colors, edgecolor="none", height=0.55)
    ax_bar.set_xlim(0, 115)
    ax_bar.set_title("Signal Scores (0–100)", color=MUTED, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_yticklabels(keys, fontsize=7.5)
    ax_bar.xaxis.set_visible(False)
    ax_bar.spines[:].set_visible(False)
    for bar, val in zip(bars, vals):
        ax_bar.text(val + 2, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=8, color=TEXT)

    # ── E. Doughnut (mid-right) ───────────────────────────────────────────────
    ax_donut = fig.add_subplot(gs[1, 2])
    wedges, _ = ax_donut.pie(
        [result["fake_pct"], result["real_pct"]],
        colors=[RED, GREEN],
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor=DARK_BG, linewidth=2),
        counterclock=False,
    )
    ax_donut.set_title("Fake vs Real Distribution", color=MUTED, fontsize=8)
    ax_donut.text(0, 0, f"{result['fake_pct']}%\nFake", ha="center", va="center",
                  color=RED, fontsize=11, fontweight="bold")
    legend_patches = [
        mpatches.Patch(color=RED,   label=f"Fake  {result['fake_pct']}%"),
        mpatches.Patch(color=GREEN, label=f"Real  {result['real_pct']}%"),
    ]
    ax_donut.legend(handles=legend_patches, loc="lower center",
                    bbox_to_anchor=(0.5, -0.08), ncol=2,
                    fontsize=7, frameon=False, labelcolor=MUTED)

    # ── F. Flagged words (bottom row, spans all) ──────────────────────────────
    ax_words = fig.add_subplot(gs[2, :])
    ax_words.set_facecolor(DARK_BG)
    ax_words.axis("off")
    ax_words.set_title("Flagged Vocabulary", color=MUTED, fontsize=8, loc="left")

    x, y = 0.0, 0.75
    for w in result["flagged_fake"]:
        ax_words.text(x, y, f" {w} ", color=RED, fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", fc="#2a0a0f", ec=RED, lw=0.8),
                      transform=ax_words.transAxes)
        x += 0.10
        if x > 0.88: x = 0.0; y -= 0.38

    x_real = 0.0
    for w in result["flagged_real"]:
        ax_words.text(x_real, y - 0.42, f" {w} ", color=GREEN, fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", fc="#0a2a14", ec=GREEN, lw=0.8),
                      transform=ax_words.transAxes)
        x_real += 0.14

    for w in result["flagged_hedge"]:
        ax_words.text(x_real, y - 0.42, f" {w} ", color=ORANGE, fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", fc="#2a1a00", ec=ORANGE, lw=0.8),
                      transform=ax_words.transAxes)
        x_real += 0.14

    if not (result["flagged_fake"] or result["flagged_real"] or result["flagged_hedge"]):
        ax_words.text(0.0, 0.5, "No significant flagged vocabulary detected.",
                      color=MUTED, fontsize=9, transform=ax_words.transAxes)

    # Legend for word tags
    legend_items = [
        mpatches.Patch(color=RED,    label="🚨 Fake indicators"),
        mpatches.Patch(color=GREEN,  label="✅ Real indicators"),
        mpatches.Patch(color=ORANGE, label="⚠️  Anon. hedging"),
    ]
    ax_words.legend(handles=legend_items, loc="lower right",
                    bbox_to_anchor=(1.0, 0.0), ncol=3, fontsize=7,
                    frameon=False, labelcolor=MUTED)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"  ✅ Chart saved → {save_path}")
    else:
        plt.show()

    plt.close()


def plot_history(history: list[dict], save_path: str | None = None):
    """Line chart showing fake/real score trends across multiple analyses."""
    if len(history) < 2:
        print("  ⚠️  Need at least 2 analyses for history chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    ax.set_facecolor(SURFACE)
    ax.set_title("Session History — Fake / Real Score Trend",
                 color=ACCENT, fontsize=11, fontfamily="monospace")

    indices = list(range(1, len(history) + 1))
    fakes   = [h["fake_pct"] for h in history]
    reals   = [h["real_pct"] for h in history]

    ax.fill_between(indices, fakes, alpha=0.12, color=RED)
    ax.fill_between(indices, reals, alpha=0.08, color=GREEN)
    ax.plot(indices, fakes, color=RED,   lw=2.2, marker="o", ms=6, label="Fake Score")
    ax.plot(indices, reals, color=GREEN, lw=2.2, marker="s", ms=6, label="Real Score")
    ax.axhline(50, color=MUTED, lw=0.8, ls="--", alpha=0.5)

    ax.set_xlim(0.5, len(history) + 0.5)
    ax.set_ylim(0, 110)
    ax.set_xticks(indices)
    ax.set_xticklabels([f"#{i}" for i in indices], color=MUTED)
    ax.set_ylabel("Score (%)", color=MUTED)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[:].set_color(SURFACE2)
    ax.legend(fontsize=9, frameon=False, labelcolor=MUTED)

    for i, (f, r) in enumerate(zip(fakes, reals), 1):
        ax.annotate(f"{f}%", (i, f), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color=RED)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"  ✅ History chart saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────
# 4.  CLI INTERFACE
# ─────────────────────────────────────────────

SAMPLE_ARTICLES = {
    "FAKE — Sensational": (
        "SHOCKING: Scientists REFUSE to admit that this common food DESTROYS your body! "
        "Big Pharma has been HIDING this for decades! You won't believe what they found!! "
        "This BOMBSHELL revelation will change everything you thought you knew about nutrition. "
        "Anonymous sources inside the FDA confirm this cover-up is REAL!"
    ),
    "FAKE — Conspiracy": (
        "The deep state globalist agenda is now fully exposed! "
        "Secret leaked documents reveal mainstream media has been lying to everyone for years. "
        "Nobody in power wants you to know this truth about the coming economic collapse. "
        "Wake up sheeple! They are censoring and silencing all dissent. "
        "Share before it gets banned!"
    ),
    "REAL — Reuters Style": (
        "The Federal Reserve raised interest rates by 0.25 percentage points on Wednesday, "
        "according to an official statement released by the central bank. "
        "The decision was unanimous among voting members of the Federal Open Market Committee. "
        "Fed Chair Jerome Powell stated the move reflects the committee's commitment to "
        "returning inflation to its 2 percent target."
    ),
    "REAL — Scientific": (
        "A peer-reviewed study published in the journal Nature Medicine found that regular "
        "moderate exercise may reduce the risk of cardiovascular disease by approximately 35 percent, "
        "according to researchers at Johns Hopkins University. "
        "The study followed 12,000 participants over 10 years. "
        "However, researchers cautioned that further studies are needed to confirm the findings."
    ),
    "MIXED — Clickbait": (
        "You won't believe what this one simple trick does to your metabolism! Doctors are amazed. "
        "A study reportedly found significant benefits, though exact details were not disclosed. "
        "Many experts claim this could change everything. "
        "Click to find out what mainstream media won't tell you about this incredible discovery!"
    ),
}


def print_result(label: str, result: dict):
    vc = {"FAKE": "\033[91m", "LIKELY REAL": "\033[92m", "UNCERTAIN": "\033[93m"}
    c  = vc.get(result["verdict"], "")
    rs = "\033[0m"
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Verdict    : {c}{result['verdict']}{rs}  ({result['confidence']}% confidence)")
    print(f"  Fake score : {result['fake_pct']}%  |  Real score : {result['real_pct']}%")
    print(f"  Words      : {result['word_count']}  |  Sentences : {result['sentence_count']}")
    print(f"  Dim. scores:")
    for k, v in result["scores"].items():
        bar = "█" * int(v / 5)
        print(f"    {k:<20} {bar:<20} {round(v):>3}")
    if result["flagged_fake"]:
        print(f"  🚨 Fake words  : {', '.join(result['flagged_fake'][:6])}")
    if result["flagged_real"]:
        print(f"  ✅ Real words  : {', '.join(result['flagged_real'][:6])}")


def main():
    print("\n" + "═"*60)
    print("   TruthLens — Fake News Detector (Python Engine)")
    print("═"*60)

    history = []

    for label, text in SAMPLE_ARTICLES.items():
        result = analyze(text)
        print_result(label, result)
        history.append(result)

        # Save individual charts
        safe_label = re.sub(r'[^\w]', '_', label)[:30]
        plot_results(result, text, save_path=f"chart_{safe_label}.png")

    # History trend
    plot_history(history, save_path="chart_history_trend.png")

    print("\n" + "═"*60)
    print("  All charts saved as PNG files.")
    print("  Open fake_news_detector.html for the interactive web version.")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
