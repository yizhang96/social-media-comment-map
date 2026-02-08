"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import ScatterMap from "../../../components/ScatterMap";

export default function DatasetPage() {
  const params = useParams<{ id: string }>();
  const id = typeof params?.id === "string" ? decodeURIComponent(params.id) : "";
  const [mapType, setMapType] = useState<"openai" | "tfidf">("openai");
  const [meta, setMeta] = useState<{ title?: string; description?: string } | null>(null);

  useEffect(() => {
    if (!id) return;
    fetch(`/datasets/${encodeURIComponent(id)}/dataset.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setMeta(d))
      .catch(() => setMeta(null));
  }, [id]);

  return (
    <main style={{ padding: 16 }}>
      <a href="/" style={{ fontSize: 12, opacity: 0.75 }}>← Back</a>

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
        <ScatterMap datasetId={id} mapType={mapType} />
      </div>
    </main>
  );
}
