"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import StatusPoller from "@/components/StatusPoller";
import VideoPlayer from "@/components/VideoPlayer";
import StatsPanel from "@/components/StatsPanel";
import HeatmapView from "@/components/HeatmapView";
import ShotMap from "@/components/ShotMap";
import HighlightTimeline from "@/components/HighlightTimeline";
import BreadcrumbHeader from "@/components/BreadcrumbHeader";
import Sidebar from "@/components/Sidebar";
import type { AnalysisData, Highlight } from "@/lib/api";

export default function AnalysisDashboard() {
  const params = useParams();
  const jobId = params.id as string;

  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [activeTab, setActiveTab] = useState<string>("home");
  const [seekTime, setSeekTime] = useState<number | null>(null);

  const handleComplete = (data: AnalysisData) => {
    setAnalysisData(data);
  };

  const handleHighlightSelect = (highlight: Highlight) => {
    setSeekTime(highlight.timestamp_seconds);
  };

  // While processing video
  if (!analysisData) {
    return (
      <div className="min-h-[calc(100vh-4rem)] bg-[#0A0F1D] flex flex-col items-center justify-center p-6 relative overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-[#0250B0]/15 blur-[150px] pointer-events-none" />
        <div className="z-10 w-full max-w-2xl">
          <StatusPoller jobId={jobId} onComplete={handleComplete} />
        </div>
      </div>
    );
  }

  // Processing finished -> PB Vision SaaS Interface
  return (
    <div className="min-h-[calc(100vh-4rem)] bg-[#0A0F1D] text-slate-50 flex flex-col relative">
      {/* Ambient glowing background effects */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] bg-blue-900/20 blur-[150px] rounded-full" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] bg-cyan-900/20 blur-[150px] rounded-full" />
      </div>

      {/* Subheader Breadcrumbs (make it relative to sit above absolute background) */}
      <div className="relative z-10">
        <BreadcrumbHeader
          category="Demos"
          matchType="Padel Match"
          activeTitle={
            activeTab === "home"
              ? "Home"
              : activeTab === "shots"
              ? "Shot Explorer"
              : "Game Stats"
          }
        />
      </div>

      {/* Main 3-Column SaaS Workbench */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden relative z-10">
        {/* Left Sidebar Navigation */}
        <Sidebar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          player1Name="Player 1"
          player2Name="Player 2"
          player3Name="Player 3"
          player4Name="Player 4"
        />

        {/* Center Stage Content */}
        <main className="flex-1 p-4 md:p-6 overflow-y-auto space-y-6 custom-scrollbar">
          {/* Main Video & Scoreboard Area */}
          {analysisData.video_url ? (
            <VideoPlayer
              videoUrl={analysisData.video_url}
              player1Name="Player 1"
              player2Name="Player 2"
              player3Name="Player 3"
              player4Name="Player 4"
              score1={25}
              score2={21}
              seekTime={seekTime}
            />
          ) : (
            <div className="p-6 rounded-2xl bg-[#131B2E] border border-[#1E2A40] text-[#8E9BAE]">
              Processed video is not available for this analysis.
            </div>
          )}

          {/* Visualizations & Match Data */}
          {activeTab === "home" && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 pt-2">
              <div className="lg:col-span-2">
                <StatsPanel
                  player1={analysisData.player_1}
                  player2={analysisData.player_2}
                  player3={analysisData.player_3}
                  player4={analysisData.player_4}
                />
              </div>

              <div className="lg:col-span-1 space-y-6">
                <HeatmapView
                  p1Url={analysisData.heatmap_p1_url}
                  p2Url={analysisData.heatmap_p2_url}
                  p3Url={analysisData.heatmap_p3_url}
                  p4Url={analysisData.heatmap_p4_url}
                />
                <ShotMap url={analysisData.shot_map_url} />
              </div>
            </div>
          )}

          {activeTab === "shots" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-2">
              <ShotMap url={analysisData.shot_map_url} />
              <HeatmapView
                p1Url={analysisData.heatmap_p1_url}
                p2Url={analysisData.heatmap_p2_url}
                p3Url={analysisData.heatmap_p3_url}
                p4Url={analysisData.heatmap_p4_url}
              />
            </div>
          )}

          {activeTab === "stats" && (
            <div className="pt-2">
              <StatsPanel
                player1={analysisData.player_1}
                player2={analysisData.player_2}
                player3={analysisData.player_3}
                player4={analysisData.player_4}
              />
            </div>
          )}


        </main>

        {/* Right Highlights Panel */}
        <HighlightTimeline
          highlights={analysisData.highlights}
          onSelectHighlight={handleHighlightSelect}
        />
      </div>
    </div>
  );
}
