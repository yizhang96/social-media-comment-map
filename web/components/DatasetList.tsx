"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { withBasePath } from "../lib/basePath";

type DatasetMeta = {
  id: string;
  platform?: string;
  title?: string;
  created_at?: string;
  description?: string;
};

export default function DatasetList() {
  const [items, setItems] = useState<DatasetMeta[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch(withBasePath("/datasets/index.json"))
      .then((r) => r.json())
      .then((d) => setItems(d))
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div style={{ color: "crimson" }}>Failed to load datasets: {err}</div>;
  if (!items.length) return <div style={{ opacity: 0.7 }}>Loading datasets…</div>;

  return (
    <div style={{ display: "grid", gap: 12 }}>
      {items.map((d) => (
        <Link
          key={d.id}
          href={`/d/${encodeURIComponent(d.id)}/`}
          style={{
            display: "block",
            border: "1px solid #eee",
            borderRadius: 12,
            padding: 12,
            textDecoration: "none",
            color: "inherit",
          }}
        >
          <div style={{ fontWeight: 600 }}>{d.title ?? d.id}</div>
          <div style={{ fontSize: 12, opacity: 0.75, marginTop: 6 }}>
            {d.platform ?? ""}{d.created_at ? ` · ${d.created_at}` : ""}
          </div>
          {d.description ? <div style={{ marginTop: 8, opacity: 0.85 }}>{d.description}</div> : null}
        </Link>
      ))}
    </div>
  );
}
