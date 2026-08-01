"use client";

import { API_BASE } from "@/lib/api";
import { Crosshair, Info } from "lucide-react";

interface ShotMapProps {
  url?: string;
}

export default function ShotMap({ url }: ShotMapProps) {
  if (!url) return null;

  const fullUrl = url.startsWith("http")
    ? url
    : `${API_BASE.replace("/api", "")}${url}`;

  return (
    <div className="p-6 md:p-8 rounded-3xl bg-[#0A0F1D]/80 backdrop-blur-2xl border border-white/5 shadow-2xl space-y-6 h-full flex flex-col justify-between hover:shadow-cyan-500/10 transition-all duration-500">
      <div className="flex items-center justify-between pb-5 border-b border-white/5">
        <h3 className="text-lg font-black text-white tracking-tight flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20">
            <Crosshair className="w-5 h-5" />
          </div>
          Ball Trajectories
        </h3>
        <span className="text-xs font-bold px-3 py-1.5 rounded-lg bg-white/5 text-white/50 border border-white/10 uppercase tracking-widest">
          Shot Map
        </span>
      </div>

      <div className="flex flex-col items-center justify-center flex-1">
        <div className="relative w-full max-w-[320px] aspect-[5/8] rounded-2xl bg-white/[0.02] border border-white/10 p-2 overflow-hidden group hover:border-cyan-500/50 hover:shadow-[0_0_20px_rgba(34,211,238,0.15)] transition-all duration-500">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={fullUrl}
            alt="Shot Trajectory Map"
            className="w-full h-full object-contain mix-blend-screen group-hover:scale-[1.03] transition-transform duration-700 ease-out"
          />
        </div>
      </div>

      <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/5 flex items-center gap-3 text-xs text-white/60">
        <div className="p-2 rounded-lg bg-white/5 text-cyan-400">
          <Info className="w-4 h-4 shrink-0" />
        </div>
        <p className="leading-relaxed">
          Color scale indicates shot speed (<span className="text-blue-400 font-black">Blue</span> = Slower, <span className="text-cyan-400 font-black">Cyan</span> = Faster)
        </p>
      </div>
    </div>
  );
}
