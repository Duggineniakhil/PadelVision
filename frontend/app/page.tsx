import Uploader from "@/components/Uploader";
import { Sparkles, Activity, Crosshair, BarChart3 } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-[calc(100vh-4rem)] bg-[#0A0F1D] text-slate-50 relative overflow-hidden flex flex-col justify-between p-6 md:p-12">
      
      {/* Background Orbs & Ambient Glow */}
      <div className="absolute top-[-10%] left-[-10%] w-[55%] h-[55%] rounded-full bg-[#0250B0]/20 blur-[140px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[55%] h-[55%] rounded-full bg-[#D0FF41]/10 blur-[160px] pointer-events-none" />

      {/* Hero Header */}
      <div className="z-10 text-center max-w-4xl mx-auto space-y-6 pt-6">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#131B2E] border border-[#1E2A40] text-xs font-bold text-[#D0FF41] volt-glow">
          <Sparkles className="w-4 h-4 text-[#D0FF41]" />
          <span>NEXT-GEN MATCH ANALYTICS & COMPUTER VISION</span>
        </div>

        <h1 className="text-4xl md:text-6xl font-black tracking-tight text-white leading-tight">
          Analyze Every Rally with <br className="hidden md:block" />
          <span className="bg-gradient-to-r from-[#D0FF41] via-white to-[#0250B0] bg-clip-text text-transparent">
            AI & Ball Trajectory Vision
          </span>
        </h1>

        <p className="text-base md:text-lg text-[#8E9BAE] max-w-2xl mx-auto font-medium leading-relaxed">
          Upload any match video to automatically track players, generate court coverage heatmaps, calculate shot speeds, and clip high-energy rally highlights.
        </p>
      </div>

      {/* Main Upload Card */}
      <div className="z-10 my-8 w-full">
        <Uploader />
      </div>

      {/* SaaS Product Feature Cards */}
      <div className="z-10 max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-6 pt-6">
        
        <div className="p-6 rounded-2xl bg-[#131B2E]/80 border border-[#1E2A40] glass-panel glass-panel-hover space-y-3">
          <div className="w-12 h-12 rounded-xl bg-[#0250B0]/20 border border-[#0250B0]/40 flex items-center justify-center text-[#0250B0]">
            <Activity className="w-6 h-6 text-[#0250B0]" />
          </div>
          <h3 className="text-lg font-bold text-white">Player Tracking</h3>
          <p className="text-xs text-[#8E9BAE] leading-relaxed">
            YOLOv8 deep learning models track player movement across court quadrants to measure coverage distance and stamina.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-[#131B2E]/80 border border-[#1E2A40] glass-panel glass-panel-hover space-y-3">
          <div className="w-12 h-12 rounded-xl bg-[#D0FF41]/20 border border-[#D0FF41]/40 flex items-center justify-center text-[#D0FF41]">
            <Crosshair className="w-6 h-6 text-[#D0FF41]" />
          </div>
          <h3 className="text-lg font-bold text-white">Ball Trajectories</h3>
          <p className="text-xs text-[#8E9BAE] leading-relaxed">
            Frame-by-frame ball detection calculates exact shot speeds, shot depth placement, and bounce coordinates.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-[#131B2E]/80 border border-[#1E2A40] glass-panel glass-panel-hover space-y-3">
          <div className="w-12 h-12 rounded-xl bg-[#C6D0DD]/20 border border-[#C6D0DD]/40 flex items-center justify-center text-[#C6D0DD]">
            <BarChart3 className="w-6 h-6 text-[#C6D0DD]" />
          </div>
          <h3 className="text-lg font-bold text-white">Match Statistics</h3>
          <p className="text-xs text-[#8E9BAE] leading-relaxed">
            Head-to-head comparisons, court coverage heatmaps, and automatic highlight clipping in an executive SaaS dashboard.
          </p>
        </div>

      </div>

    </div>
  );
}
