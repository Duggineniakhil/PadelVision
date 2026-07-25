"use client";

import { API_BASE } from "@/lib/api";
import { Layers, Maximize2 } from "lucide-react";

interface HeatmapViewProps {
  p1Url?: string;
  p2Url?: string;
  p3Url?: string;
  p4Url?: string;
}

const playerConfigs = [
  { id: 1, name: "PLAYER 1", color: "#0250B0", hoverColor: "#0250B0" },
  { id: 2, name: "PLAYER 2", color: "#EC4899", hoverColor: "#D0FF41" },
  { id: 3, name: "PLAYER 3", color: "#0250B0", hoverColor: "#0250B0" },
  { id: 4, name: "PLAYER 4", color: "#EC4899", hoverColor: "#D0FF41" },
] as const;

export default function HeatmapView({ p1Url, p2Url, p3Url, p4Url }: HeatmapViewProps) {
  const urls = [p1Url, p2Url, p3Url, p4Url];
  if (urls.every((u) => !u)) return null;

  const getFullUrl = (url: string) =>
    url.startsWith("http") ? url : `${API_BASE.replace("/api", "")}${url}`;

  return (
    <div className="p-6 rounded-2xl bg-[#131B2E] border border-[#1E2A40] shadow-xl space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-[#1E2A40]">
        <h3 className="text-base font-extrabold text-white flex items-center gap-2">
          <Layers className="w-5 h-5 text-[#0250B0]" />
          <span>Court Coverage Heatmaps</span>
        </h3>
        <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-[#0E1626] text-[#8E9BAE] border border-[#1E2A40]">
          4 Players
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {urls.map((url, idx) => {
          if (!url) return null;
          const config = playerConfigs[idx];
          return (
            <div key={idx} className="flex flex-col items-center gap-2">
              <div className="relative w-full rounded-xl bg-[#1a1a2e] border border-[#1E2A40] overflow-hidden group hover:border-[#0250B0] transition-colors aspect-[4/5]">
                <span className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded-md text-[10px] font-extrabold text-white shadow-sm" style={{ backgroundColor: config.color }}>
                  {config.name}
                </span>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={getFullUrl(url)}
                  alt={`${config.name} Heatmap`}
                  className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-500"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <Maximize2 className="w-6 h-6 text-white" />
                </div>
              </div>
              <span className="text-xs font-semibold text-[#C6D0DD]">{config.name} — Court Coverage</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
