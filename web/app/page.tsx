import DatasetList from "../components/DatasetList";

export default function Page() {
  return (
    <main style={{ padding: 24 }}>
      <h1 style={{ fontSize: 20, fontWeight: 600, margin: 0 }}>Social Media Comment Map</h1>
      <p style={{ marginTop: 8, opacity: 0.75 }}>
        Choose a dataset to view its semantic map (default) or TF-IDF map.
      </p>
      <div style={{ marginTop: 16 }}>
        <DatasetList />
      </div>
    </main>
  );
}