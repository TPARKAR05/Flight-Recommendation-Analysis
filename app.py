import os
import random
import requests
from datetime import date, datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from recommendation import recommend_flights
from sentiment import sentiment_bp
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
app.register_blueprint(sentiment_bp)

AVIATIONSTACK_KEY  = os.getenv("AVIATIONSTACK_KEY")
AVIATIONSTACK_BASE = "http://api.aviationstack.com/v1"


@app.route("/")
def index():
    return jsonify({"message": "Flight Recommendation API (Aviationstack) is running."})


@app.route("/api/flights/search", methods=["GET"])
def search_flights():
    source      = request.args.get("source", "").strip().upper()
    destination = request.args.get("destination", "").strip().upper()
    time_from   = request.args.get("time_from", "").strip()
    time_to     = request.args.get("time_to",   "").strip()
    max_stops   = request.args.get("max_stops", type=int)

    if not source or not destination:
        return jsonify({"error": "source and destination are required."}), 400

    if not AVIATIONSTACK_KEY:
        return jsonify({"error": "AVIATIONSTACK_KEY is not set in .env"}), 500

    params = {
        "access_key": AVIATIONSTACK_KEY,
        "dep_iata":   source,
        "arr_iata":   destination,
        "limit":      50,
    }

    try:
        res = requests.get(f"{AVIATIONSTACK_BASE}/flights", params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
    except requests.HTTPError as e:
        return jsonify({
            "error": "Aviationstack API error",
            "status": e.response.status_code,
            "details": e.response.text
        }), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    if "error" in data:
        return jsonify({
            "error": data["error"].get("message", "Aviationstack error"),
            "code":  data["error"].get("code")
        }), 502

    flights = parse_flights(data.get("data", []))

    if time_from:
        flights = [f for f in flights if f["departure_time"] >= time_from]
    if time_to:
        flights = [f for f in flights if f["departure_time"] <= time_to]

    if max_stops is not None:
        flights = [f for f in flights if f["stops"] <= max_stops]

    if not flights:
        return jsonify({"results": [], "message": "No flights found."}), 200

    ranked = recommend_flights(flights)
    return jsonify({"results": ranked}), 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_price(src: str, dst: str, duration_minutes: int, stops: int) -> int:
    seed = hash(f"{src}{dst}{duration_minutes}") & 0xFFFF
    rng  = random.Random(seed)
    if stops == 0:
        return rng.randint(7000, 8000)
    else:
        return rng.randint(6000, 7000)


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_flights(raw: list) -> list[dict]:
    flights = []
    for i, f in enumerate(raw):
        dep   = f.get("departure", {})
        arr   = f.get("arrival",   {})
        airl  = f.get("airline",   {})
        flt   = f.get("flight",    {})

        dep_time_str = dep.get("scheduled", "")
        arr_time_str = arr.get("scheduled", "")

        duration_minutes = calc_duration(dep_time_str, arr_time_str)
        dep_time = dep_time_str[11:16] if len(dep_time_str) >= 16 else "—"
        arr_time = arr_time_str[11:16] if len(arr_time_str) >= 16 else "—"

        stops = random.randint(0, 2)

        flights.append({
            "id":               flt.get("iata") or i,
            "flight_number":    flt.get("iata", "—"),
            "airline":          airl.get("name", "Unknown"),
            "source":           dep.get("iata", "—"),
            "destination":      arr.get("iata", "—"),
            "departure_date":   f.get("flight_date", "—"),
            "departure_time":   dep_time,
            "arrival_time":     arr_time,
            "duration_minutes": duration_minutes,
            "delay_minutes":    dep.get("delay") or 0,
            "terminal":         dep.get("terminal") or "—",
            "gate":             dep.get("gate") or "—",
            "stops":            stops,
            "price":            _fake_price(dep.get("iata", ""), arr.get("iata", ""), duration_minutes, stops),
            "status":           f.get("flight_status", "—"),
        })

    return flights


def calc_duration(dep_str: str, arr_str: str) -> int:
    try:
        dep_dt = datetime.strptime(dep_str[:19], "%Y-%m-%dT%H:%M:%S")
        arr_dt = datetime.strptime(arr_str[:19], "%Y-%m-%dT%H:%M:%S")
        diff = (arr_dt - dep_dt).total_seconds() / 60
        return int(diff) if diff > 0 else 0
    except Exception:
        return 0


@app.route("/api/debug")
def debug():
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("AVIATIONSTACK_KEY")
    if not key:
        return jsonify({"error": "AVIATIONSTACK_KEY not found in .env"})
    res = requests.get(
        "http://api.aviationstack.com/v1/flights",
        params={"access_key": key, "dep_iata": "BOM", "arr_iata": "DEL", "limit": 1}
    )
    return jsonify({"status": res.status_code, "response": res.json()})


if __name__ == "__main__":
    app.run(debug=True)
