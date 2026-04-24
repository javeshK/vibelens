"""
youtube_fetcher.py — YouTube Data API v3 Song Dataset Builder
==============================================================
Fetches trending / popular music from YouTube's verified channels
and writes the results into the trends/ JSON cache files.

Usage (standalone):
    python youtube_fetcher.py --language english --max 200
    python youtube_fetcher.py --language hindi   --max 200
    python youtube_fetcher.py --language punjabi --max 200

Requires:
    YOUTUBE_API_KEY environment variable  OR  --api-key flag

Rate limits:
    YouTube Data API v3 quota: 10 000 units/day (free tier).
    This script uses search.list (100 units) + videos.list (1 unit each).
    Fetching 200 songs ≈ 300–400 quota units.

To scale to 900+ songs:
    Run with --max 900 across multiple days, or use a paid quota.
    The script merges new entries with existing JSON (no duplicates).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    sys.exit("❌  Install `requests` first:  pip install requests")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YT_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YT_VIDEO_URL  = "https://www.googleapis.com/youtube/v3/videos"
YT_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"

TRENDS_DIR = os.path.join(os.path.dirname(__file__), "trends")
os.makedirs(TRENDS_DIR, exist_ok=True)

# Music mood/context tags mapped to search terms per language
SEARCH_QUERIES: Dict[str, List[str]] = {
    "english": [
        "new pop music 2024 official",
        "trending english songs 2024 VEVO",
        "top pop songs 2024 official video",
        "romantic English songs 2024",
        "upbeat English songs 2024 official",
        "sad English songs 2024 official",
        "EDM 2024 official",
        "hip hop 2024 official music video",
        "indie pop 2024 official",
        "R&B songs 2024 official",
    ],
    "hindi": [
        "new Bollywood songs 2024 official",
        "romantic Hindi songs 2024 T-Series",
        "trending Bollywood songs 2024",
        "Arijit Singh new songs 2024",
        "Hindi sad songs 2024 official",
        "Bollywood dance songs 2024",
        "new Hindi pop songs 2024",
        "Jubin Nautiyal songs 2024",
        "Hindi wedding songs 2024 official",
        "Bollywood party songs 2024",
    ],
    "punjabi": [
        "new Punjabi songs 2024 official",
        "Diljit Dosanjh 2024 official",
        "AP Dhillon new songs 2024",
        "Karan Aujla 2024 official",
        "trending Punjabi songs 2024",
        "Punjabi hip hop 2024 official",
        "Punjabi romantic songs 2024",
        "Punjabi party songs 2024 official",
        "Guru Randhawa 2024 official",
        "new Punjabi pop 2024",
    ],
}

# Mood inference from YouTube video title keywords
MOOD_KEYWORDS = {
    "romantic":  ["love", "dil", "pyaar", "ishq", "romantic", "tere", "tera", "heart", "prem"],
    "sad":       ["sad", "dard", "dukh", "alone", "broken", "miss", "cry", "pain", "lost"],
    "happy":     ["happy", "khushi", "fun", "celebration", "smile", "dance", "party", "joy"],
    "energetic": ["energy", "power", "hustle", "grind", "fire", "beast", "killer", "hype"],
    "calm":      ["chill", "relax", "peaceful", "soft", "gentle", "slow", "rain", "night"],
    "nostalgic": ["memories", "yaad", "purana", "old", "throwback", "classic", "miss"],
    "motivational": ["motivation", "inspire", "hustle", "rise", "winner", "champion"],
}

CONTEXT_KEYWORDS = {
    "party":    ["party", "dance", "club", "wedding", "shaadi", "sangeet"],
    "travel":   ["travel", "road", "trip", "journey", "adventure", "explore"],
    "gym":      ["gym", "workout", "fitness", "training", "exercise"],
    "nature":   ["nature", "rain", "rain", "monsoon", "mountains", "sky"],
    "night":    ["night", "midnight", "dark", "neon", "city"],
    "romantic": ["couple", "love story", "romance", "wedding"],
}

VERIFIED_CHANNEL_KEYWORDS = [
    "VEVO", "Official", "Records", "Music", "T-Series", "Sony Music",
    "YRF", "Zee Music", "Speed Records", "Tips Official",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _is_verified_channel(channel_title: str) -> bool:
    for kw in VERIFIED_CHANNEL_KEYWORDS:
        if kw.lower() in channel_title.lower():
            return True
    return False


def _infer_moods(title: str, description: str = "") -> List[str]:
    text = (title + " " + description).lower()
    matched = []
    for mood, keywords in MOOD_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            matched.append(mood)
    return matched[:3] if matched else ["happy", "calm"]


def _infer_contexts(title: str, description: str = "") -> List[str]:
    text = (title + " " + description).lower()
    matched = []
    for ctx, keywords in CONTEXT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            matched.append(ctx)
    return matched[:3] if matched else ["aesthetic", "hangout"]


def _days_ago(published_at: str) -> int:
    try:
        pub_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        now    = datetime.now(timezone.utc)
        return max(0, (now - pub_dt).days)
    except Exception:
        return 365


def _trend_score(views: int, likes: int, days: int) -> int:
    """
    Composite trend score 0-100.
    Weights: recency × engagement rate × absolute views.
    """
    recency = max(0, 1.0 - days / 730.0)           # 2-year decay
    eng_rate = (likes / max(views, 1)) * 100        # like % capped at reasonable
    view_score = min(1.0, views / 1_000_000_000)    # 1B views = 1.0

    raw = (recency * 40) + (min(eng_rate, 5) * 6) + (view_score * 30) + 20
    return int(min(100, max(30, raw)))


def _extract_artist(title: str, channel: str) -> str:
    """Best-effort artist extraction from title (e.g. 'Song Name - Artist')."""
    if " - " in title:
        parts = title.split(" - ")
        return parts[-1].strip()[:60]
    if "|" in title:
        return title.split("|")[-1].strip()[:60]
    return channel.replace("VEVO", "").replace("Official", "").strip()[:60]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API calls
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _search_videos(query: str, api_key: str, max_results: int = 50) -> List[str]:
    """Return list of video IDs matching query."""
    params = {
        "part":       "id",
        "q":          query,
        "type":       "video",
        "videoCategoryId": "10",   # Music
        "maxResults": min(50, max_results),
        "order":      "viewCount",
        "key":        api_key,
    }
    try:
        resp = requests.get(YT_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [item["id"]["videoId"] for item in data.get("items", [])]
    except Exception as exc:
        print(f"  ⚠ Search failed for '{query}': {exc}")
        return []


def _fetch_video_details(video_ids: List[str], api_key: str) -> List[Dict[str, Any]]:
    """Batch-fetch video statistics and snippet for up to 50 IDs."""
    if not video_ids:
        return []
    params = {
        "part":   "snippet,statistics",
        "id":     ",".join(video_ids[:50]),
        "key":    api_key,
    }
    try:
        resp = requests.get(YT_VIDEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("items", [])
    except Exception as exc:
        print(f"  ⚠ Video details failed: {exc}")
        return []


def _video_to_song(item: Dict, language: str, idx: int) -> Optional[Dict]:
    """Convert a YouTube API video item into our song schema."""
    snippet    = item.get("snippet", {})
    stats      = item.get("statistics", {})
    title      = snippet.get("title", "")
    channel    = snippet.get("channelTitle", "")
    desc       = snippet.get("description", "")[:300]
    pub_at     = snippet.get("publishedAt", "")
    views      = int(stats.get("viewCount",  0))
    likes      = int(stats.get("likeCount",  0))
    video_id   = item.get("id", "")

    if not title or views < 100_000:        # skip very low-view content
        return None

    days   = _days_ago(pub_at)
    ts     = _trend_score(views, likes, days)
    artist = _extract_artist(title, channel)

    # Strip common suffixes from title to get song name
    song_name = title
    for suffix in [" - Official Video", " | Official Music Video", " (Official Video)",
                   " (Lyrical)", " (Full Song)", " | Lyrical Video", " [Official]"]:
        song_name = song_name.replace(suffix, "").replace(suffix.lower(), "")
    song_name = song_name.strip()[:100]

    moods    = _infer_moods(title, desc)
    contexts = _infer_contexts(title, desc)
    verified = _is_verified_channel(channel)

    return {
        "id":                 f"{language[:2]}_{idx:04d}",
        "song_name":          song_name,
        "artist":             artist,
        "mood":               moods,
        "context":            contexts,
        "trend_score":        ts,
        "views":              views,
        "likes":              likes,
        "published_days_ago": days,
        "verified":           verified,
        "language":           language,
        "category":           "Music",
        "bpm":                None,
        "energy":             round(min(1.0, ts / 100.0 * 0.5 + 0.4), 2),
        "aesthetic_tags":     [],
        "channel":            channel,
        "youtube_id":         video_id,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main fetch function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_and_update(language: str, api_key: str, max_songs: int = 200) -> int:
    """
    Fetch YouTube songs for the given language and merge into existing JSON.
    Returns number of new songs added.
    """
    json_path = os.path.join(TRENDS_DIR, f"{language}.json")

    # Load existing data
    existing: List[Dict] = []
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as fh:
            existing = json.load(fh)

    existing_ids = {s.get("youtube_id") for s in existing if s.get("youtube_id")}
    queries = SEARCH_QUERIES.get(language, SEARCH_QUERIES["english"])

    new_songs: List[Dict] = []
    global_idx = len(existing) + 1
    per_query  = max(10, max_songs // len(queries))

    for query in queries:
        if len(new_songs) >= max_songs:
            break
        print(f"  🔍 Searching: {query}")
        video_ids = _search_videos(query, api_key, per_query)
        # Filter already-seen IDs
        video_ids = [v for v in video_ids if v not in existing_ids]
        if not video_ids:
            continue

        items = _fetch_video_details(video_ids, api_key)
        for item in items:
            if len(new_songs) >= max_songs:
                break
            song = _video_to_song(item, language, global_idx)
            if song:
                new_songs.append(song)
                existing_ids.add(song.get("youtube_id"))
                global_idx += 1

        time.sleep(0.3)     # polite rate limiting

    # Merge and save
    merged = existing + new_songs
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)

    print(f"  ✅ {len(new_songs)} new songs added → {json_path} (total: {len(merged)})")
    return len(new_songs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch YouTube songs into trends/*.json")
    parser.add_argument("--language", choices=["english", "hindi", "punjabi", "all"],
                        default="all", help="Language to fetch")
    parser.add_argument("--max",  type=int, default=200, help="Max new songs per language")
    parser.add_argument("--api-key", default=None, help="YouTube Data API key")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        sys.exit("❌  Set YOUTUBE_API_KEY env var or pass --api-key")

    langs = ["english", "hindi", "punjabi"] if args.language == "all" else [args.language]
    for lang in langs:
        print(f"\n🎵 Fetching {lang} songs (max {args.max})…")
        fetch_and_update(lang, api_key, args.max)