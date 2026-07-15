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

export default async function DatasetPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const resolvedParams = await params;
  const id =
    typeof resolvedParams?.id === "string" ? decodeURIComponent(resolvedParams.id) : "";
  return <DatasetPageClient id={id} />;
}
