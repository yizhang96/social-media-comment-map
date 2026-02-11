"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { withBasePath } from "../lib/basePath";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
}) as unknown as any;

type Point = {
  id: number;
  text: string;
  likes: number | null;
  time: string | null;
  location: string | null;
  user: string | null;
  x: number;
  y: number;
  cluster_id: number;
};

export default function ScatterMap({ datasetId, mapType = "openai" }: { datasetId: string; mapType?: "openai" | "tfidf" }) {
  const [points, setPoints] = useState<Point[]>([]);
  const [selected, setSelected] = useState<Point | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const pointsRef = useRef<Point[]>([]);
  const boundGdRef = useRef<any>(null);
  const handlerRef = useRef<((ev: any) => void) | null>(null);

  useEffect(() => {
    if (!datasetId) return;
    setSelected(null);
    setErr(null);

    const file = mapType === "tfidf" ? "comments_map_tfidf.json" : "comments_map_openai.json";
    fetch(withBasePath(`/datasets/${encodeURIComponent(datasetId)}/${file}`))
          .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => {
        const cleaned = Array.isArray(d)
          ? d.filter((p) => {
              const t = p?.text;
              return typeof t === "string" && t.trim().length > 0;
            })
          : [];
        setPoints(cleaned);
      })
      .catch((e) => setErr(String(e)));
  }, [datasetId, mapType]);

  useEffect(() => {
    pointsRef.current = points;
  }, [points]);

  const handlePointClick = (ev: any) => {
    const idx = ev?.points?.[0]?.customdata?.[0];
    if (typeof idx === "number") {
      const p = pointsRef.current[idx];
      if (p) setSelected({ ...p });
    }
  };

  handlerRef.current = handlePointClick;

  const attachClickHandler = (gd: any) => {
    if (!gd || boundGdRef.current === gd) return;
    boundGdRef.current = gd;
    const handler = (ev: any) => handlerRef.current?.(ev);
    // Defer to ensure Plotly has finished rendering
    setTimeout(() => gd.on?.("plotly_click", handler), 0);
    setTimeout(() => gd.on?.("plotly_click", handler), 100);
  };

  const { traces } = useMemo(() => {
    const likes = points.map((p) => (p.likes ?? 0));
    const maxLikes = Math.max(1, ...likes);
    const sizes = likes.map((l) => 6 + 18 * Math.sqrt(l / maxLikes));
    const selectedIdx = selected ? points.findIndex((p) => p.id === selected.id) : -1;

    const palette = [
      "#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#f97316",
      "#0f766e", "#db2777", "#65a30d", "#ea580c", "#0891b2",
      "#4f46e5", "#ca8a04", "#be123c", "#059669", "#9333ea",
    ];

    const byCluster = new Map<number, number[]>();
    points.forEach((p, i) => {
      const key = p.cluster_id ?? -1;
      const list = byCluster.get(key);
      if (list) list.push(i);
      else byCluster.set(key, [i]);
    });

    const sortedClusters = Array.from(byCluster.keys()).sort((a, b) => a - b);
    const traces = sortedClusters.map((clusterId, idx) => {
      const indices = byCluster.get(clusterId) ?? [];
      const color = clusterId === -1 ? "#9ca3af" : palette[idx % palette.length];
      const name = clusterId === -1 ? "Noise" : `Cluster ${clusterId}`;

      return {
        type: "scattergl",
        mode: "markers",
        name,
        x: indices.map((i) => points[i].x),
        y: indices.map((i) => points[i].y),
        marker: {
          size: indices.map((i) => (i === selectedIdx ? sizes[i] + 6 : sizes[i])),
          color,
          opacity: 0.85,
          line: {
            width: indices.map((i) => (i === selectedIdx ? 2 : 0)),
            color: "#111",
          },
        },
        selected: { marker: { opacity: 0.85 } },
        unselected: { marker: { opacity: 0.85 } },
        customdata: indices.map((i) => {
          const p = points[i];
          const raw = p.text.replaceAll("\n", " ");
          const snippet = raw.length > 60 ? raw.slice(0, 60) + "…" : raw;
          const wrapLine = (s: string, width: number) => {
            const out: string[] = [];
            for (let j = 0; j < s.length; j += width) {
              out.push(s.slice(j, j + width));
            }
            return out.join("<br>");
          };
          const wrapped = wrapLine(snippet, 30);
          return [i, p.id, p.likes ?? "NA", p.cluster_id, wrapped];
        }),
        hovertemplate:
          "id: %{customdata[1]}<br>likes: %{customdata[2]}<br>cluster: %{customdata[3]}<br>%{customdata[4]}<br><span style=\"opacity:.7\">Click for full text</span><extra></extra>",
        hoverlabel: {
          bgcolor: "rgba(255,255,255,0.92)",
          bordercolor: "#111",
          borderwidth: 1,
          font: { color: "#111", size: 12 },
          align: "left",
          borderpad: 6,
        },
      } as any;
    });

    return { traces };
  }, [points, selected]);

  if (!datasetId) return <div style={{ color: "crimson" }}>Missing dataset id.</div>;
  if (err) return <div style={{ color: "crimson" }}>Failed to load map: {err}</div>;
  if (!points.length) return <div style={{ opacity: 0.7 }}>Loading map…</div>;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: 16 }}>
      <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 8 }}>
        <Plot
          key={`${datasetId}:${mapType}:${points.length}`}
          data={traces}
          layout={{
            height: 640,
            margin: { l: 30, r: 10, t: 10, b: 30 },
            xaxis: { zeroline: false },
            yaxis: { zeroline: false },
            hovermode: "closest",
            legend: { title: { text: "Clusters" } },
            clickmode: "event+select",
          }}
          onInitialized={(_fig: any, gd: any) => {
            attachClickHandler(gd);
          }}
          onUpdate={(_fig: any, gd: any) => {
            attachClickHandler(gd);
          }}
          onClick={(ev: any) => {
            handlePointClick(ev);
          }}
          config={{ displayModeBar: true }}
        />
      </div>

      <aside style={{ border: "1px solid #eee", borderRadius: 12, padding: 12, height: 640, overflow: "auto" }}>
        <div style={{ fontWeight: 600 }}>Detail</div>
        {!selected ? (
          <p style={{ marginTop: 12, opacity: 0.7 }}>Click a point to view the full comment.</p>
        ) : (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 12, opacity: 0.75 }}>
              id {selected.id} · cluster {selected.cluster_id} · likes {selected.likes ?? "NA"}
            </div>
            <hr style={{ margin: "12px 0", border: "none", borderTop: "1px solid #eee" }} />
            <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{selected.text}</div>
          </div>
        )}
      </aside>
    </div>
  );
}
