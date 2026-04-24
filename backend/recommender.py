"""
recommender.py — Advanced Multi-Factor Song Recommendation Engine
=================================================================
Scoring Formula (as specified):
  Final Score =
      (Mood Match     × 0.30) +
      (Context Match  × 0.20) +
      (Trend Score    × 0.25) +
      (Language Pref  × 0.15) +
      (Aesthetic Match× 0.10)

Additional logic:
  • Mood mismatch penalty  (−0.10 per mismatched secondary)
  • Recency boost          (songs < 30 days old get +5 trend points)
  • Artist diversity       (penalise repeat artists in top-7)
  • Returns ≥ 7 results
"""

from __future__ import annotations
import json
import os
import functools
from typing import List, Dict, Any

from analyzer import AnalysisResult

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset loading  (cached in memory)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_TRENDS_DIR = os.path.join(os.path.dirname(__file__), "trends")

@functools.lru_cache(maxsize=4)
def _load_library(language: str) -> List[Dict[str, Any]]:
    """Load and cache the JSON for a given language. Falls back to English."""
    path = os.path.join(_TRENDS_DIR, f"{language}.json")
    if not os.path.exists(path):
        path = os.path.join(_TRENDS_DIR, "english.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_all() -> List[Dict[str, Any]]:
    """Load all language datasets for cross-language fallback."""
    all_songs = []
    for lang in ("english", "hindi", "punjabi"):
        all_songs.extend(_load_library(lang))
    return all_songs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scoring helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Normalised trend score (raw 0-100 → 0-1)
def _norm_trend(raw: int, days_ago: int) -> float:
    recency_boost = 5 if days_ago < 30 else (2 if days_ago < 90 else 0)
    return min(1.0, (raw + recency_boost) / 100.0)


# Mood match score (0-1)
def _mood_score(song_moods: List[str], primary: str, secondaries: List[str]) -> float:
    song_moods_lower = [m.lower() for m in song_moods]
    score = 0.0

    if primary.lower() in song_moods_lower:
        score += 1.0                            # full primary match
    else:
        # Partial credit for related moods
        RELATED = {
            "happy":      {"fun", "energetic", "romantic"},
            "sad":        {"emotional", "nostalgic", "calm"},
            "energetic":  {"happy", "fun", "aggressive", "motivated"},
            "calm":       {"peaceful", "nostalgic", "romantic", "serene"},
            "romantic":   {"happy", "calm", "emotional", "nostalgic"},
            "dark":       {"intense", "mysterious", "aggressive"},
            "nostalgic":  {"calm", "romantic", "sad", "emotional"},
            "emotional":  {"sad", "romantic", "nostalgic", "calm"},
            "confident":  {"energetic", "happy", "fun", "empowered"},
            "intense":    {"dark", "energetic", "aggressive"},
        }
        related = RELATED.get(primary.lower(), set())
        overlap = set(song_moods_lower) & related
        score += 0.40 * min(1.0, len(overlap) / max(1, len(related)))

    # Secondary mood contributions
    for sm in secondaries:
        if sm.lower() in song_moods_lower:
            score += 0.20
        else:
            score -= 0.05  # mismatch penalty

    return max(0.0, min(1.0, score))


# Context match score (0-1)
def _context_score(song_contexts: List[str], detected: str) -> float:
    song_contexts_lower = [c.lower() for c in song_contexts]
    detected_lower = detected.lower()

    if detected_lower in song_contexts_lower:
        return 1.0

    # Context family matching
    FAMILIES = {
        "outdoor": {"nature", "travel", "beach", "mountains", "sunrise", "sunset"},
        "social":  {"party", "friends", "hangout", "dance", "celebration"},
        "intimate":{"couple", "cozy", "aesthetic", "alone", "night"},
        "active":  {"gym", "workout", "competition", "sports"},
        "urban":   {"city", "night", "street", "urban"},
    }
    for _family, members in FAMILIES.items():
        if detected_lower in members:
            overlap = members & set(song_contexts_lower)
            if overlap:
                return 0.50

    return 0.0


# Aesthetic match score (0-1)
def _aesthetic_match(song_tags: List[str], brightness: float, saturation: float, warm_ratio: float) -> float:
    """
    Map aesthetic tags → expected image attributes and compare.
    """
    TAG_PROFILES = {
        "warm":       {"brightness": 0.60, "saturation": 0.55, "warm_ratio": 0.60},
        "cool":       {"brightness": 0.55, "saturation": 0.45, "warm_ratio": 0.25},
        "dark":       {"brightness": 0.25, "saturation": 0.40, "warm_ratio": 0.30},
        "vibrant":    {"brightness": 0.65, "saturation": 0.70, "warm_ratio": 0.45},
        "soft":       {"brightness": 0.65, "saturation": 0.30, "warm_ratio": 0.50},
        "golden":     {"brightness": 0.70, "saturation": 0.60, "warm_ratio": 0.75},
        "neon":       {"brightness": 0.45, "saturation": 0.85, "warm_ratio": 0.30},
        "moody":      {"brightness": 0.30, "saturation": 0.40, "warm_ratio": 0.25},
        "pastel":     {"brightness": 0.75, "saturation": 0.25, "warm_ratio": 0.55},
        "urban":      {"brightness": 0.40, "saturation": 0.45, "warm_ratio": 0.30},
        "tropical":   {"brightness": 0.70, "saturation": 0.65, "warm_ratio": 0.60},
        "intimate":   {"brightness": 0.50, "saturation": 0.35, "warm_ratio": 0.55},
        "nostalgic":  {"brightness": 0.60, "saturation": 0.40, "warm_ratio": 0.60},
        "bold":       {"brightness": 0.55, "saturation": 0.75, "warm_ratio": 0.40},
        "smooth":     {"brightness": 0.60, "saturation": 0.45, "warm_ratio": 0.50},
    }

    if not song_tags:
        return 0.50

    scores = []
    for tag in song_tags:
        profile = TAG_PROFILES.get(tag.lower())
        if profile is None:
            continue
        diff = (
            abs(brightness - profile["brightness"]) +
            abs(saturation - profile["saturation"]) +
            abs(warm_ratio - profile["warm_ratio"])
        ) / 3.0
        scores.append(max(0.0, 1.0 - diff * 2.5))

    return round(sum(scores) / len(scores), 3) if scores else 0.50


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reason generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MOOD_REASONS = {
    "happy":      "Its bright, feel-good energy perfectly mirrors the joyful mood detected in your image.",
    "energetic":  "The high-energy rhythm matches the dynamic, action-packed feel of your photo.",
    "calm":       "Its gentle, flowing arrangement complements the peaceful atmosphere captured here.",
    "sad":        "The emotional depth resonates with the reflective, introspective tone of your image.",
    "romantic":   "Its warm, intimate sound pairs beautifully with the romantic atmosphere you've captured.",
    "nostalgic":  "The song's nostalgic undertones echo the timeless, memory-evoking quality of your shot.",
    "dark":       "Its moody, atmospheric production matches the intense, dramatic feel of your image.",
    "emotional":  "The song's emotional resonance mirrors the heartfelt sentiment in your image.",
    "confident":  "Its bold, self-assured energy perfectly matches the confidence in your image.",
    "intense":    "The raw intensity of this track complements the powerful feel of your photo.",
}

_CONTEXT_ADDONS = {
    "party":       " A top pick for Stories from social gatherings.",
    "summer":      " Ideal for summer vibes content.",
    "sunset":      " Perfect for that golden-hour aesthetic.",
    "travel":      " Adds a wanderlust feeling to travel posts.",
    "gym":         " An energy boost for fitness content.",
    "nature":      " Enhances the natural, organic vibe.",
    "night":       " Sets the mood for late-night aesthetics.",
    "cozy":        " Creates a warm, intimate atmosphere.",
    "alone":       " Adds introspective depth to personal posts.",
    "aesthetic":   " Elevates the visual aesthetic of your feed.",
    "couple":      " Perfect for couple and relationship content.",
    "friends":     " Great for group and squad moments.",
    "beach":       " A breezy companion for beach-day content.",
    "city":        " Adds an urban edge to city lifestyle posts.",
    "celebration": " Perfect for milestone and celebration posts.",
    "fashion":     " Elevates editorial and fashion content.",
    "road_trip":   " Makes road-trip content come alive.",
}

def _build_reason(song: Dict, mood: str, context: str) -> str:
    base  = _MOOD_REASONS.get(mood, "Its sound perfectly complements the mood of your image.")
    addon = _CONTEXT_ADDONS.get(context, "")
    return base + addon


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trend badge
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _trend_badge(trend_score: int, days_ago: int) -> Dict[str, str]:
    if trend_score >= 92 and days_ago <= 180:
        return {"key": "trending", "label": "🔥 Trending"}
    elif trend_score >= 80 or days_ago <= 90:
        return {"key": "rising",   "label": "📈 Rising"}
    else:
        return {"key": "niche",    "label": "🎧 Niche"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Artist diversity enforcement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _enforce_diversity(ranked: List[tuple], top_n: int) -> List[tuple]:
    """
    Select top_n songs ensuring no artist appears more than twice.
    Falls back to relaxed limit (max 3) if not enough candidates.
    """
    seen_artists: Dict[str, int] = {}
    result = []
    remainder = []

    for item in ranked:
        score, song = item
        artist = song["artist"].lower()
        count  = seen_artists.get(artist, 0)
        if count < 2:
            result.append(item)
            seen_artists[artist] = count + 1
        else:
            remainder.append(item)
        if len(result) >= top_n:
            break

    # If we don't have enough, relax to 3 per artist
    if len(result) < top_n:
        for item in remainder:
            score, song = item
            artist = song["artist"].lower()
            if seen_artists.get(artist, 0) < 3:
                result.append(item)
                seen_artists[artist] = seen_artists.get(artist, 0) + 1
            if len(result) >= top_n:
                break

    return result[:top_n]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def recommend(
    analysis: AnalysisResult,
    language: str = "english",
    top_n: int = 7,
) -> List[Dict[str, Any]]:
    """
    Generate ranked song recommendations using the multi-factor formula:

      Final Score =
          (Mood Match     × 0.30) +
          (Context Match  × 0.20) +
          (Trend Score    × 0.25) +
          (Language Pref  × 0.15) +
          (Aesthetic Match× 0.10)

    Args:
        analysis : AnalysisResult from analyzer.analyse_image()
        language : "english" | "hindi" | "punjabi"
        top_n    : Minimum 7 results

    Returns:
        List of enriched song dicts ready for JSON serialisation.
    """
    top_n = max(top_n, 7)
    library = _load_library(language)

    # If language library has fewer than top_n songs, supplement from others
    if len(library) < top_n:
        library = _load_all()

    mood       = analysis.mood
    secondaries = analysis.secondary_moods
    context    = analysis.context
    cm         = analysis.color
    aesthetic  = analysis.aesthetic

    scored = []
    for song in library:
        # ── Component scores ──
        ms = _mood_score(song.get("mood", []), mood, secondaries)
        cs = _context_score(song.get("context", []), context)
        ts = _norm_trend(song.get("trend_score", 50), song.get("published_days_ago", 365))
        ls = 1.0 if song.get("language", "english") == language else 0.30
        am = _aesthetic_match(
            song.get("aesthetic_tags", []),
            cm.brightness,
            cm.saturation,
            cm.warm_ratio,
        )

        # ── Weighted formula ──
        final = (ms * 0.30) + (cs * 0.20) + (ts * 0.25) + (ls * 0.15) + (am * 0.10)
        final = round(final, 4)

        scored.append((final, {**song, "_ms": ms, "_cs": cs, "_ts": ts, "_ls": ls, "_am": am}))

    # Sort descending
    scored.sort(key=lambda x: (x[0], x[1].get("trend_score", 0)), reverse=True)

    # Enforce artist diversity
    diverse = _enforce_diversity(scored, top_n)

    # Build output
    results = []
    for rank, (final_score, song) in enumerate(diverse, start=1):
        badge = _trend_badge(song.get("trend_score", 50), song.get("published_days_ago", 365))
        mood_pct = int(min(100, song["_ms"] * 100))
        results.append({
            "rank":         rank,
            "song_name":    song["song_name"],
            "artist":       song["artist"],
            "language":     song.get("language", language),
            "category":     song.get("category", ""),
            "mood_match":   f"{mood_pct}%",
            "trend_score":  song.get("trend_score", 50),
            "trend_badge":  badge,
            "reason":       _build_reason(song, mood, context),
            "final_score":  round(final_score * 100, 1),
            "energy":       int(song.get("energy", 0.5) * 100),
            "bpm":          song.get("bpm"),
            "year":         song.get("year"),
            "views":        song.get("views", 0),
            "published_days_ago": song.get("published_days_ago", 0),
            "verified":     song.get("verified", False),
            "score_breakdown": {
                "mood":      round(song["_ms"] * 100, 1),
                "context":   round(song["_cs"] * 100, 1),
                "trend":     round(song["_ts"] * 100, 1),
                "language":  round(song["_ls"] * 100, 1),
                "aesthetic": round(song["_am"] * 100, 1),
            },
        })

    return results