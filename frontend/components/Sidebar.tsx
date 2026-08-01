"use client";

import { Home, Crosshair, BarChart3, Trophy, Share2, Users } from "lucide-react";

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  player1Name?: string;
  player2Name?: string;
  player3Name?: string;
  player4Name?: string;
}

export default function Sidebar({
  activeTab,
  setActiveTab,
  player1Name = "Player 1",
  player2Name = "Player 2",
  player3Name = "Player 3",
  player4Name = "Player 4",
}: SidebarProps) {
  const navItems = [
    { id: "home", label: "Home", icon: Home },
    { id: "shots", label: "Shot Explorer", icon: Crosshair },
    { id: "stats", label: "Game Stats", icon: BarChart3 },
    { id: "leaderboards", label: "Leaderboards", icon: Trophy },
  ];

  const players = [
    { name: player1Name, role: "Left Side", color: "bg-[#0250B0]", label: "P1" },
    { name: player2Name, role: "Right Side", color: "bg-blue-500", label: "P2" },
    { name: player3Name, role: "Left Side", color: "bg-pink-600", label: "P3" },
    { name: player4Name, role: "Right Side", color: "bg-rose-500", label: "P4" },
  ];

  return (
    <aside className="w-full lg:w-72 bg-[#0A0F1D]/60 backdrop-blur-3xl border-r border-white/5 flex flex-col justify-between p-6 shrink-0 relative overflow-hidden">
      {/* Subtle background glow */}
      <div className="absolute top-0 left-0 w-full h-48 bg-gradient-to-b from-cyan-500/5 to-transparent pointer-events-none" />
      
      <div className="space-y-8 relative z-10">
        {/* Navigation Tabs */}
        <div className="space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3.5 px-4 py-3 rounded-2xl font-bold text-sm transition-all duration-300 text-left group ${
                  isActive
                    ? "bg-white/10 text-cyan-400 shadow-[0_4px_12px_rgba(0,0,0,0.1)] border border-white/10"
                    : "text-white/60 hover:text-white hover:bg-white/[0.04] border border-transparent"
                }`}
              >
                <Icon className={`w-5 h-5 transition-transform duration-300 ${isActive ? "text-cyan-400 scale-110" : "text-white/40 group-hover:scale-110"}`} />
                <span className="tracking-wide">{item.label}</span>
              </button>
            );
          })}
        </div>

        <div className="h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent" />

        {/* Team A Players */}
        <div>
          <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-cyan-500/80 mb-4 px-2 flex items-center gap-2">
            <Users className="w-3.5 h-3.5" />
            <span>Team A</span>
          </h4>

          <div className="space-y-2.5">
            {players.slice(0, 2).map(({ name, role, color, label }) => (
              <div
                key={label}
                className="flex items-center gap-3.5 p-3 rounded-2xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.06] hover:border-cyan-500/30 transition-all duration-300 cursor-pointer group"
              >
                <div className={`w-10 h-10 rounded-xl ${color} flex items-center justify-center font-black text-xs text-white shadow-lg group-hover:scale-105 transition-transform duration-300`}>
                  {label}
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-bold text-white/90 tracking-wide">{name}</span>
                  <span className="text-[11px] font-medium text-white/40 tracking-wider uppercase">{role}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Team B Players */}
        <div className="pt-2">
          <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-pink-500/80 mb-4 px-2 flex items-center gap-2">
            <Users className="w-3.5 h-3.5" />
            <span>Team B</span>
          </h4>

          <div className="space-y-2.5">
            {players.slice(2, 4).map(({ name, role, color, label }) => (
              <div
                key={label}
                className="flex items-center gap-3.5 p-3 rounded-2xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.06] hover:border-pink-500/30 transition-all duration-300 cursor-pointer group"
              >
                <div className={`w-10 h-10 rounded-xl ${color} flex items-center justify-center font-black text-xs text-white shadow-lg group-hover:scale-105 transition-transform duration-300`}>
                  {label}
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-bold text-white/90 tracking-wide">{name}</span>
                  <span className="text-[11px] font-medium text-white/40 tracking-wider uppercase">{role}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Sidebar Footer CTA - Share Button */}
      <div className="pt-8 relative z-10">
        <button
          onClick={() => {
            if (navigator.clipboard) {
              navigator.clipboard.writeText(window.location.href);
              alert("Game link copied to clipboard!");
            }
          }}
          className="w-full flex items-center justify-center gap-2.5 py-4 px-4 rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-black text-sm tracking-wide transition-all duration-300 shadow-[0_0_20px_rgba(34,211,238,0.3)] hover:shadow-[0_0_30px_rgba(34,211,238,0.5)] active:scale-95"
        >
          <span>Share Match</span>
          <Share2 className="w-4 h-4" />
        </button>
      </div>
    </aside>
  );
}
