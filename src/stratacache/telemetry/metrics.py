from stratacache.telemetry.telemetry import StrataTelemetry
from fastapi import FastAPI
from stratacache.engine.storage_engine import dump_cache_scores

app = FastAPI()

from fastapi.responses import PlainTextResponse

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    telemetry = StrataTelemetry.get_or_create()
    system_stats, per_tier_stats = telemetry.get_stats()
    lines = []
    # System stats
    for k, v in system_stats.__dict__.items():
        try:
            val = float(v)
        except Exception:
            continue
        lines.append(f"stratacache_system_{k} {val}")
    # Per-tier stats
    for tier, stats in per_tier_stats.items():
        for k, v in stats.__dict__.items():
            try:
                val = float(v)
            except Exception:
                continue
            lines.append(f"stratacache_tier_{k}{{tier=\"{tier.name}\"}} {val}")
    # Cache scores
    for entry in dump_cache_scores():
        aid = entry.get("artifact_id")
        score = entry.get("score")
        try:
            val = float(score)
        except Exception:
            continue
        lines.append(f"stratacache_cache_score{{artifact_id=\"{aid}\"}} {val}")
    return "\n".join(lines) + "\n"
