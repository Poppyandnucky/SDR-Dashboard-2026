interface Props {
  children: React.ReactNode;
}

export default function ChartFootnote({ children }: Props) {
  return (
    <p className="mt-3 text-[11px] text-ink-muted leading-relaxed border-t border-border-soft pt-3">
      {children}
    </p>
  );
}
