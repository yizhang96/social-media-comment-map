"use client";

import { useEffect, useState } from "react";
import ScatterMap from "../../../components/ScatterMap";
import { withBasePath } from "../../../lib/basePath";

type Meta = { title?: string; description?: string };

export default function DatasetPageClient({ id }: { id: string }) {
  const [mapType, setMapType] = useState<"openai" | "tfidf">("openai");
  const [meta, setMeta] = useState<Meta | null>(null);
  const [effectiveId, setEffectiveId] = useState(id);

  useEffect(() => {
    if (id) {
      setEffectiveId(id);
      return;
    }
    if (typeof window === "undefined") return;
    const base = withBasePath("/");
    const path = window.location.pathname;
    const stripped = base !== "/" && path.startsWith(base) ? path.slice(base.length) : path;
    const m = stripped.match(/^\/d\/([^/]+)\/?/);
    if (m?.[1]) setEffectiveId(decodeURIComponent(m[1]));
  }, [id]);

  useEffect(() => {
    if (!effectiveId) return;
    fetch(withBasePath(`/datasets/${encodeURIComponent(effectiveId)}/dataset.json`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setMeta(d))
      .catch(() => setMeta(null));
  }, [effectiveId]);

  return (
    <main style={{ padding: 16 }}>
      <a href={withBasePath("/")} style={{ fontSize: 12, opacity: 0.75 }}>← Back</a>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginTop: 8 }}>
        <div>
          <h1 style={{ margin: 0 }}>{meta?.title || "Dataset"}</h1>
          {meta?.description ? (
            <p style={{ marginTop: 6, opacity: 0.75 }}>{meta.description}</p>
          ) : null}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => setMapType("openai")}
            style={{
              padding: "6px 10px",
              borderRadius: 10,
              border: "1px solid #ddd",
              background: mapType === "openai" ? "#eee" : "white",
              cursor: "pointer",
            }}
          >
            Semantic
          </button>
          <button
            onClick={() => setMapType("tfidf")}
            style={{
              padding: "6px 10px",
              borderRadius: 10,
              border: "1px solid #ddd",
              background: mapType === "tfidf" ? "#eee" : "white",
              cursor: "pointer",
            }}
          >
            TF-IDF
          </button>
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <ScatterMap datasetId={effectiveId} mapType={mapType} />
      </div>
    </main>
  );
}
