"""
Demo server.

Serves two things on http://localhost:8000:
  - the simulated "Y" feed page (demo/feed/), filled with tweets from the
    end-to-end TEST set, which the models never saw during training
  - a small API that the Chrome extension calls to classify those tweets

API:
  GET  /api/status    which models are loaded
  GET  /api/labels    category names and colors, in tab order
  GET  /api/events    the disasters in the test set, for the feed's picker
  GET  /api/feed      a page of test tweets (?event=...&mode=all|errors&offset=&limit=)
  POST /api/classify  {"tweets": [{"id": "...", "text": "..."}]} -> labels

Usage:
    python server/app.py \
        --stage1 runs/bertweet-base/stage1/seed1/model \
        --stage2 runs/bertweet-base/stage2/seed1/model \
        --errors runs/end_to_end/stage1-bertweet-base-seed1__stage2-bertweet-base-seed1/predictions.tsv

--errors is optional. It turns on the feed's "Known mistakes" view, which
shows only test tweets the pipeline got wrong, for walking through failure
cases in the demo. Use the end-to-end run that matches the loaded models.

If BERTweet is not trained yet, the TF-IDF models work as a backup:
    python server/app.py --stage1 runs/tfidf_lr/stage1/seed0/model \
                         --stage2 runs/tfidf_lr/stage2/seed0/model
"""

import argparse
import os
import random
import sys
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Let this file import the modules in src/.
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVER_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import config  # noqa: E402
from crisisbench_io import read_tsv  # noqa: E402
from triage_pipeline import TriagePipeline  # noqa: E402


FEED_DIR = os.path.join(PROJECT_ROOT, "demo", "feed")
MAX_TWEETS_PER_REQUEST = 100
MAX_FEED_PAGE = 50


class TweetIn(BaseModel):
    id: str
    text: str


class ClassifyRequest(BaseModel):
    tweets: List[TweetIn]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", required=True, help="Stage 1 model folder")
    parser.add_argument("--stage2", required=True, help="Stage 2 model folder")
    parser.add_argument("--errors", default=None,
                        help="End-to-end predictions.tsv, for the 'Known mistakes' view")
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_STAGE1_THRESHOLD)
    parser.add_argument("--feed-data",
                        default=os.path.join(config.PROCESSED_DATA_DIR, "end_to_end_test.tsv"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def readable_event_name(event):
    """CrisisBench event names look like '2013_pakistan_eq'. Make them nicer
    for the picker without changing their meaning."""
    name = event.replace("_", " ")
    name = name.strip()
    return name


def load_feed_tweets(path):
    rows = read_tsv(path)
    # Shuffle once, with a fixed seed, so the "All disasters" feed mixes
    # events and the order is the same every time the server starts.
    shuffler = random.Random(0)
    shuffler.shuffle(rows)
    return rows


def load_error_ids(path):
    error_ids = set()
    if path is None:
        return error_ids
    rows = read_tsv(path)
    for row in rows:
        if row["gold_end_to_end"] != row["pred_end_to_end"]:
            error_ids.add(row["id"])
    return error_ids


def tweet_for_feed(row):
    """The fields the feed page needs. Gold labels are included so the
    extension's answer key can compare them with the model's labels; the
    page itself never shows them."""
    gold_display = config.get_label_display(row["gold_end_to_end"])
    return {
        "id": row["id"],
        "text": row["text"],
        "event": row["event"],
        "event_display": readable_event_name(row["event"]),
        "disaster_type": row["disaster_type"],
        "source": row["source"],
        "gold_informativeness": row["gold_informativeness"],
        "gold_humanitarian": row["gold_humanitarian"],
        "gold_end_to_end": row["gold_end_to_end"],
        "gold_display": gold_display["display"],
    }


def create_app(args):
    pipeline = TriagePipeline(args.stage1, args.stage2, args.threshold)
    feed_rows = load_feed_tweets(args.feed_data)
    error_ids = load_error_ids(args.errors)
    print("Loaded " + str(len(feed_rows)) + " test tweets for the feed")
    if args.errors is not None:
        print("Loaded " + str(len(error_ids)) + " known mistakes")

    app = FastAPI(title="Disaster tweet triage")

    # The extension runs on the feed page, which this server also serves,
    # so requests are same-origin. CORS is allowed for localhost anyway in
    # case you open the page on 127.0.0.1 and call localhost, or vice versa.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:" + str(args.port),
                       "http://127.0.0.1:" + str(args.port)],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/status")
    def status():
        return {
            "stage1": pipeline.stage1.description,
            "stage2": pipeline.stage2.description,
            "threshold": pipeline.threshold,
            "feed_size": len(feed_rows),
            "known_mistakes_available": len(error_ids) > 0,
        }

    @app.get("/api/labels")
    def labels():
        # Tabs: every Stage 2 category in the order from config.py,
        # then any category the model has that config.py does not list.
        ordered = []
        listed = set()
        for entry in config.LABEL_DISPLAY:
            if entry["label"] in pipeline.stage2.labels:
                ordered.append(entry)
                listed.add(entry["label"])
        for label in pipeline.stage2.labels:
            if label not in listed:
                ordered.append(config.get_label_display(label))

        return {
            "categories": ordered,
            "not_humanitarian_label": config.NOT_HUMANITARIAN_LABEL,
            "not_actionable_label": config.NOT_ACTIONABLE_LABEL,
            "filtered": config.get_label_display(config.NOT_INFORMATIVE_LABEL),
        }

    @app.get("/api/events")
    def events():
        counts = {}
        disaster_types = {}
        for row in feed_rows:
            event = row["event"]
            if event not in counts:
                counts[event] = 0
                disaster_types[event] = row["disaster_type"]
            counts[event] += 1

        def by_count(event_name):
            return counts[event_name]

        event_names = sorted(counts, key=by_count, reverse=True)
        result = []
        for event in event_names:
            result.append({
                "event": event,
                "display": readable_event_name(event),
                "disaster_type": disaster_types[event],
                "count": counts[event],
            })
        return {"events": result}

    @app.get("/api/feed")
    def feed(event: str = "all", mode: str = "all", offset: int = 0, limit: int = 25):
        if mode == "errors" and len(error_ids) == 0:
            raise HTTPException(status_code=400,
                                detail="Start the server with --errors to use this view.")
        if limit > MAX_FEED_PAGE:
            limit = MAX_FEED_PAGE

        matching = []
        for row in feed_rows:
            if event != "all" and row["event"] != event:
                continue
            if mode == "errors" and row["id"] not in error_ids:
                continue
            matching.append(row)

        page = matching[offset:offset + limit]
        tweets = []
        for row in page:
            tweets.append(tweet_for_feed(row))

        return {
            "tweets": tweets,
            "total": len(matching),
            "next_offset": offset + len(page),
        }

    @app.post("/api/classify")
    def classify(request: ClassifyRequest):
        if len(request.tweets) > MAX_TWEETS_PER_REQUEST:
            raise HTTPException(status_code=400,
                                detail="Send at most " + str(MAX_TWEETS_PER_REQUEST)
                                + " tweets per request.")

        texts = []
        for tweet in request.tweets:
            texts.append(tweet.text)
        results = pipeline.classify(texts)

        response = []
        for index in range(len(request.tweets)):
            result = dict(results[index])
            result["id"] = request.tweets[index].id
            response.append(result)
        return {"results": response, "threshold": pipeline.threshold}

    # The feed page. Mounted last so it does not cover the /api routes.
    app.mount("/", StaticFiles(directory=FEED_DIR, html=True), name="feed")

    return app


def main():
    args = parse_args()
    app = create_app(args)
    print("Open http://127.0.0.1:" + str(args.port) + " in Chrome (with the extension loaded)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
