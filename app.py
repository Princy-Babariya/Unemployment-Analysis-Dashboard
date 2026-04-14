from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from analysis import run_analysis


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = BASE_DIR / "unemployment-insights" / "dist"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIST / "assets") if FRONTEND_DIST.exists() else None,
    static_url_path="/assets",
)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.get("/api/health")
def health_check():
    return jsonify({"status": "ok"})


@app.get("/api/data")
def get_data():
    region = request.args.get("region") or None
    year = request.args.get("year") or None

    data = run_analysis(region, year)
    return jsonify(data)


@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def serve_frontend(path: str):
    if FRONTEND_DIST.exists():
        target = FRONTEND_DIST / path
        if path and target.exists() and target.is_file():
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")

    return jsonify(
        {
            "message": "Frontend build not found.",
            "next_step": "Run `npm run build` in `unemployment-insights` or start the Vite dev server.",
        }
    ), 200


if __name__ == "__main__":
    app.run(debug=True)
