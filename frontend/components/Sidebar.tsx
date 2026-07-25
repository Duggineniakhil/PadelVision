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
    <aside className="w-full lg:w-64 bg-[#0E1626] border-r border-[#1E2A40] flex flex-col justify-between p-4 shrink-0">
      <div className="space-y-6">
        {/* Navigation Tabs */}
        <div className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl font-medium text-sm transition-all text-left ${
                  isActive
                    ? "bg-[#1E2A40] text-white font-semibold shadow-sm border border-[#2E3E5C]"
                    : "text-[#8E9BAE] hover:text-white hover:bg-[#131B2E]"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-[#D0FF41]" : "text-[#8E9BAE]"}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>

        <div className="h-[1px] bg-[#1E2A40] my-4" />

        {/* Team A Players */}
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wider text-[#8E9BAE] mb-3 px-2 flex items-center gap-2">
            <Users className="w-3 h-3 text-[#0250B0]" />
            <span>Team A</span>
          </h4>

          <div className="space-y-2">
            {players.slice(0, 2).map(({ name, role, color, label }) => (
              <div
                key={label}
                className="flex items-center gap-3 p-2.5 rounded-xl bg-[#131B2E] border border-[#1E2A40] text-[#C6D0DD] hover:border-[#0250B0] transition-colors cursor-pointer"
              >
                <div className={`w-8 h-8 rounded-lg ${color} flex items-center justify-center font-bold text-xs text-white shadow-md`}>
                  {label}
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-semibold text-white">{name}</span>
                  <span className="text-xs text-[#8E9BAE]">{role}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="h-[1px] bg-[#1E2A40] my-4" />

        {/* Team B Players */}
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wider text-[#8E9BAE] mb-3 px-2 flex items-center gap-2">
            <Users className="w-3 h-3 text-pink-500" />
            <span>Team B</span>
          </h4>

          <div className="space-y-2">
            {players.slice(2, 4).map(({ name, role, color, label }) => (
              <div
                key={label}
                className="flex items-center gap-3 p-2.5 rounded-xl bg-[#131B2E] border border-[#1E2A40] text-[#C6D0DD] hover:border-pink-500 transition-colors cursor-pointer"
              >
                <div className={`w-8 h-8 rounded-lg ${color} flex items-center justify-center font-bold text-xs text-white shadow-md`}>
                  {label}
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-semibold text-white">{name}</span>
                  <span className="text-xs text-[#8E9BAE]">{role}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Sidebar Footer CTA - Volt Lime Share Button */}
      <div className="pt-6">
        <button
          onClick={() => {
            if (navigator.clipboard) {
              navigator.clipboard.writeText(window.location.href);
              alert("Game link copied to clipboard!");
            }
          }}
          className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-[#D0FF41] hover:bg-[#B7E62B] text-[#0A0F1D] font-extrabold text-sm transition-all volt-glow active:scale-95 shadow-lg"
        >
          <span>Share this game</span>
          <Share2 className="w-4 h-4 text-[#0A0F1D]" />
        </button>
      </div>
    </aside>
  );
}
