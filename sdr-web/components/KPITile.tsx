interface KPITileProps {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
}

export default function KPITile({ label, value, sub, accent }: KPITileProps) {
  return (
    <div
      className={`rounded-lg border p-5 ${
        accent ? "bg-ink text-paper border-ink" : "bg-card border-border"
      }`}
    >
      <div className={`text-[11px] uppercase tracking-wider mb-2 ${accent ? "text-paper/70" : "text-ink-muted"}`}>
        {label}
      </div>
      <div className={`font-display text-3xl num ${accent ? "" : "text-ink"}`}>{value}</div>
      {sub && (
        <div className={`text-xs mt-1 ${accent ? "text-paper/60" : "text-ink-muted"}`}>{sub}</div>
      )}
    </div>
  );
}
