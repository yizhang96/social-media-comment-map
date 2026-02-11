import fs from "fs";
import path from "path";
import DatasetPageClient from "./DatasetPageClient";

export function generateStaticParams() {
  try {
    const indexPath = path.join(process.cwd(), "public", "datasets", "index.json");
    const raw = fs.readFileSync(indexPath, "utf-8");
    const items = JSON.parse(raw) as Array<{ id?: string }>;
    return items
      .map((d) => d.id)
      .filter((id): id is string => typeof id === "string" && id.length > 0)
      .map((id) => ({ id }));
  } catch {
    return [];
  }
}

export default function DatasetPage({ params }: { params: { id: string } }) {
  const id = typeof params?.id === "string" ? decodeURIComponent(params.id) : "";
  return <DatasetPageClient id={id} />;
}
