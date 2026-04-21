"""
Sentiment Analysis Module
--------------------------
Fetches YouTube comments about an airline, runs VADER sentiment analysis,
and returns aggregated data for charting.

Install dependencies:
    pip install google-api-python-client vaderSentiment

Setup:
    1. Go to https://console.cloud.google.com/
    2. Create a project → Enable "YouTube Data API v3"
    3. Create an API Key credential
    4. Add to .env: YOUTUBE_API_KEY=your_key_here
"""

import os
import re
from collections import Counter
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

sentiment_bp = Blueprint("sentiment", __name__)

analyzer = SentimentIntensityAnalyzer()

STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","need",
    "i","my","me","we","our","you","your","it","its","this","that","these",
    "those","they","them","their","he","she","his","her","what","which","who",
    "how","when","where","why","not","no","so","if","as","from","by","about",
    "just","like","get","got","also","very","really","more","much","some",
    "any","all","one","two","up","out","there","then","than","too","even","back",
    "into","over","after","before","because","went","said","going","go","video",
    "watch","channel","youtube","subscribe","comment","share","click",
    "im","ive","dont","didnt","cant","isnt","wasnt","airline","airlines","flight",
    "flights","plane","airport","ticket","boarding","travel","trip","flying","flew",
    "www","http","https","com","review","experience","passenger"
}


def classify(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def tokenize(text: str) -> list[str]:
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return [w for w in words if w not in STOP_WORDS]


def get_youtube_client():
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise ValueError("YOUTUBE_API_KEY is not set in .env")
    return build("youtube", "v3", developerKey=key)


def search_videos(youtube, airline: str, target: int = 15) -> list[str]:
    """Search for airline review videos and return up to `target` unique video IDs.

    Uses a broad mix of positive, negative, and neutral queries so the comment
    pool is not skewed toward praise videos. Each query fetches up to 10
    candidates; we stop as soon as we have enough unique IDs.
    """
    queries = [
        # Negative / complaint focused — first so they fill the video cap
        f"{airline} airline worst experience",
        f"{airline} flight complaint",
        f"{airline} airline problems",
        f"{airline} bad experience",
        f"{airline} airline delay cancelled",
        f"{airline} airline avoid",
        # General / mixed
        f"{airline} airline review",
        f"{airline} flight experience",
        f"{airline} honest review",
        f"{airline} travel review",
        # Positive / balanced — last priority
        f"{airline} airline best flight",
        f"{airline} customer experience",
        f"{airline} airline worth it",
        f"{airline} business class review",
        f"{airline} economy class review",
    ]
    video_ids = []
    seen = set()

    for query in queries:
        if len(video_ids) >= target:
            break
        res = youtube.search().list(
            q=query,
            part="id,snippet",
            type="video",
            maxResults=10,
            relevanceLanguage="en",
            order="relevance",
            publishedAfter="2023-01-01T00:00:00Z", 
        ).execute()

        for item in res.get("items", []):
            vid = item["id"].get("videoId")
            if vid and vid not in seen:
                seen.add(vid)
                video_ids.append(vid)
            if len(video_ids) >= target:
                break

    return video_ids[:target]


def fetch_comments(youtube, video_id: str, max_comments: int = 60) -> list[dict]:
    """Fetch top-level comments from a video.

    Uses pagination so we reliably collect up to `max_comments` even when a
    single API page returns fewer results.  No upper limit is applied to
    comment length — only the 10-character minimum is kept to discard empty
    or emoji-only entries.
    """
    comments = []
    next_page_token = None

    try:
        while len(comments) < max_comments:
            remaining = max_comments - len(comments)
            # YouTube caps maxResults at 100 per page; request only what's left
            page_size = min(remaining, 100)

            kwargs = dict(
                part="snippet",
                videoId=video_id,
                maxResults=page_size,
                textFormat="plainText",
                order="relevance",
            )
            if next_page_token:
                kwargs["pageToken"] = next_page_token

            res = youtube.commentThreads().list(**kwargs).execute()

            for item in res.get("items", []):
                snippet   = item["snippet"]["topLevelComment"]["snippet"]
                text      = snippet.get("textDisplay", "").strip()
                published = snippet.get("publishedAt", "")
                # Discard blank / emoji-only entries; keep everything else
                if len(text) < 10:
                    continue
                if published and published[:4] < "2023":
                    continue
                comments.append({
                    "text":      text,
                    "published": published,
                    "url":       f"https://www.youtube.com/watch?v={video_id}",
                })
                if len(comments) >= max_comments:
                    break

            next_page_token = res.get("nextPageToken")
            if not next_page_token:
                break   # no more pages available for this video

    except Exception:
        pass   # videos with disabled comments are silently skipped

    return comments


@sentiment_bp.route("/api/sentiment", methods=["GET"])
def analyse():
    airline = request.args.get("airline", "").strip()
    if not airline:
        return jsonify({"error": "airline parameter is required"}), 400

    if not os.getenv("YOUTUBE_API_KEY"):
        return jsonify({"error": "YOUTUBE_API_KEY must be set in .env"}), 500

    try:
        youtube = get_youtube_client()

        video_ids = search_videos(youtube, airline)
        if not video_ids:
            return jsonify({"error": "No videos found for this airline."}), 404

        all_comments = []
        for vid in video_ids:
            all_comments.extend(fetch_comments(youtube, vid))

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    if not all_comments:
        return jsonify({"error": "No comments found. Videos may have comments disabled."}), 404

    # Run VADER on each comment
    posts_data = []
    for c in all_comments:
        score     = analyzer.polarity_scores(c["text"])
        sentiment = classify(score["compound"])

        try:
            month = datetime.strptime(c["published"][:10], "%Y-%m-%d").strftime("%Y-%m")
        except Exception:
            month = "unknown"

        posts_data.append({
            "text":      c["text"],
            "sentiment": sentiment,
            "compound":  score["compound"],
            "month":     month,
            "url":       c["url"],
            "title":     c["text"][:80] + ("…" if len(c["text"]) > 80 else ""),
        })

    # ── Sentiment counts ──────────────────────────────────────────────────────
    counts = Counter(p["sentiment"] for p in posts_data)
    pie = {
        "positive": counts.get("positive", 0),
        "negative": counts.get("negative", 0),
        "neutral":  counts.get("neutral",  0),
    }

    # ── Top 20 words ──────────────────────────────────────────────────────────
    all_words = []
    for p in posts_data:
        all_words.extend(tokenize(p["text"]))
    top_words = [{"word": w, "count": c} for w, c in Counter(all_words).most_common(20)]

    # ── Word clouds per sentiment ─────────────────────────────────────────────
    pos_words, neg_words = [], []
    for p in posts_data:
        tokens = tokenize(p["text"])
        if p["sentiment"] == "positive":
            pos_words.extend(tokens)
        elif p["sentiment"] == "negative":
            neg_words.extend(tokens)

    word_cloud = {
        "positive": [{"word": w, "count": c} for w, c in Counter(pos_words).most_common(40)],
        "negative": [{"word": w, "count": c} for w, c in Counter(neg_words).most_common(40)],
    }

    # ── Monthly volume by sentiment ───────────────────────────────────────────
    monthly: dict[str, dict] = {}
    for p in posts_data:
        m = p["month"]
        if m == "unknown":
            continue
        if m not in monthly:
            monthly[m] = {"month": m, "positive": 0, "negative": 0, "neutral": 0}
        monthly[m][p["sentiment"]] += 1

    monthly_list = sorted(monthly.values(), key=lambda x: x["month"])

    return jsonify({
        "airline":    airline,
        "total":      len(posts_data),
        "pie":        pie,
        "top_words":  top_words,
        "word_cloud": word_cloud,
        "monthly":    monthly_list,
    }), 200
