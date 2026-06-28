"use client";

interface PillSelectorProps {
  options: { value: string; label: string; hint?: string }[];
  value: string;
  onChange: (value: string) => void;
}

export default function PillSelector({ options, value, onChange }: PillSelectorProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={`pill px-4 py-2 rounded-full border border-border text-sm ${
            value === opt.value ? "active" : "bg-card hover:bg-paper-deep"
          }`}
        >
          {opt.label}
          {opt.hint && (
            <span className="block text-[10px] opacity-70 mt-0.5">{opt.hint}</span>
          )}
        </button>
      ))}
    </div>
  );
}
