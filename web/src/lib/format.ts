/** 75.4 -> "1:15", 3725 -> "1:02:05" */
export function clock(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

/** 1122.8 -> "18 min 43 s", 42 -> "42 s" */
export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  const s = Math.round(seconds);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${s % 60} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function num(value: number | null | undefined, digits = 0, unit = ""): string {
  if (value == null || !Number.isFinite(value)) return "–";
  const text = value.toLocaleString("en-GB", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return unit ? `${text} ${unit}` : text;
}

export const pct = (value: number | null | undefined, digits = 0) =>
  value == null || !Number.isFinite(value) ? "–" : `${(value * 100).toFixed(digits)}%`;

export function date(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

/** A team's name. After changeovers a pair plays both ends, so it is named by pair: team A is the
 * pair that started on the near side (players 1 and 2). */
export const teamName = (team: "near" | "far", pairs = false) =>
  pairs ? (team === "near" ? "Team A" : "Team B") : team === "near" ? "Near team" : "Far team";

export const ZONE_LABEL: Record<string, string> = {
  net: "Net",
  transition: "Transition",
  back: "Back court",
  near_half: "Near half",
  far_half: "Far half",
};
