"use client";

import { Flame, PlayCircle, MoreVertical } from "lucide-react";

interface Highlight {
  start_frame: number;
  end_frame: number;
  timestamp_seconds: number;
  label: string;
}

interface HighlightTimelineProps {
  highlights?: Highlight[];
  onSelectHighlight?: (highlight: Highlight) => void;
}

export default function HighlightTimeline({ highlights = [], onSelectHighlight }: HighlightTimelineProps) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <aside className="w-full xl:w-80 bg-[#0E1626] border-l border-[#1E2A40] p-4 shrink-0 flex flex-col">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-[#1E2A40]">
        <h3 className="text-base font-extrabold text-white tracking-wide flex items-center gap-2">
          <Flame className="w-5 h-5 text-orange-500 fill-orange-500/20" />
          <span>Highlights</span>
        </h3>
        <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-[#0250B0]/30 text-[#C6D0DD] border border-[#0250B0]/40">
          {highlights.length} rallies
        </span>
      </div>

      <div className="space-y-3 overflow-y-auto max-h-[600px] pr-1 custom-scrollbar">
        {highlights.length === 0 && (
          <div className="p-4 rounded-xl bg-[#131B2E] border border-[#1E2A40] text-sm text-[#8E9BAE]">
            No highlights were detected for this analysis.
          </div>
        )}

        {highlights.map((highlight, idx) => (
          <div
            key={idx}
            onClick={() => onSelectHighlight?.(highlight)}
            className="group flex items-center justify-between p-3.5 rounded-xl bg-[#131B2E] border border-[#1E2A40] hover:border-[#D0FF41] hover:bg-[#19243C] transition-all cursor-pointer shadow-sm hover:shadow-md"
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-[#0E1626] border border-[#1E2A40] flex items-center justify-center text-orange-400 group-hover:text-[#D0FF41] group-hover:border-[#D0FF41]/40 transition-colors">
                {idx % 2 === 0 ? <Flame className="w-4 h-4" /> : <PlayCircle className="w-4 h-4" />}
              </div>

              <div>
                <h4 className="text-sm font-bold text-[#C6D0DD] group-hover:text-white transition-colors">
                  {highlight.label}
                </h4>
                <div className="flex items-center gap-2 text-xs text-[#8E9BAE] mt-0.5 font-mono">
                  <span>{formatTime(highlight.timestamp_seconds)}</span>
                  <span>/</span>
                  <span>Frames {highlight.start_frame}-{highlight.end_frame}</span>
                </div>
              </div>
            </div>

            <button className="p-1 text-[#8E9BAE] hover:text-white transition-colors">
              <MoreVertical className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
