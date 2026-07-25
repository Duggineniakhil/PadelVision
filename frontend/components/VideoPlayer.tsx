"use client";

import { useRef, useState, useEffect } from "react";
import { API_BASE } from "@/lib/api";
import { Play, Pause, Volume2, VolumeX, Maximize, Users } from "lucide-react";

interface VideoPlayerProps {
  videoUrl: string;
  player1Name?: string;
  player2Name?: string;
  player3Name?: string;
  player4Name?: string;
  score1?: number;
  score2?: number;
  seekTime?: number | null;
}

export default function VideoPlayer({
  videoUrl,
  player1Name = "Player 1",
  player2Name = "Player 2",
  player3Name = "Player 3",
  player4Name = "Player 4",
  score1 = 25,
  score2 = 21,
  seekTime = null,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [progress, setProgress] = useState(0);

  const fullUrl = videoUrl.startsWith("http")
    ? videoUrl
    : `${API_BASE.replace("/api", "")}${videoUrl}`;

  // Seek when seekTime prop changes
  useEffect(() => {
    if (seekTime !== null && videoRef.current) {
      videoRef.current.currentTime = seekTime;
      videoRef.current.play();
      setIsPlaying(true);
    }
  }, [seekTime]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const current = videoRef.current.currentTime;
      const duration = videoRef.current.duration || 1;
      setProgress((current / duration) * 100);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (videoRef.current && videoRef.current.duration) {
      const newTime = (parseFloat(e.target.value) / 100) * videoRef.current.duration;
      videoRef.current.currentTime = newTime;
      setProgress(parseFloat(e.target.value));
    }
  };

  const handleFullscreen = () => {
    if (videoRef.current) {
      if (videoRef.current.requestFullscreen) {
        videoRef.current.requestFullscreen();
      }
    }
  };

  return (
    <div className="space-y-4">
      {/* Video Container with Overlays */}
      <div className="relative overflow-hidden rounded-2xl bg-[#080D1A] border border-[#1E2A40] shadow-2xl group">
        {/* Top-Right Status Badge */}
        <div className="absolute top-4 right-4 z-20 flex items-center gap-2 px-3 py-1 rounded-full bg-[#0E1626]/80 backdrop-blur-md border border-[#1E2A40] text-xs font-semibold text-[#D0FF41]">
          <span className="w-2 h-2 rounded-full bg-[#D0FF41] animate-ping" />
          <span>100% Processed</span>
        </div>

        {/* Video Element */}
        <div className="aspect-video w-full flex items-center justify-center bg-black">
          <video
            ref={videoRef}
            src={fullUrl}
            onTimeUpdate={handleTimeUpdate}
            onEnded={() => setIsPlaying(false)}
            className="w-full h-full object-contain cursor-pointer"
            crossOrigin="anonymous"
            onClick={togglePlay}
          />
        </div>

        {/* Custom PB Vision Style Timeline Control Bar */}
        <div className="p-3 bg-[#0E1626] border-t border-[#1E2A40] flex items-center gap-3">
          {/* Frame / Step Buttons */}
          <button
            onClick={togglePlay}
            className="p-2 rounded-lg bg-[#0250B0] hover:bg-[#013B82] text-white transition-colors"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
          </button>

          {/* Seek Bar */}
          <div className="flex-1 relative flex items-center">
            <input
              type="range"
              min="0"
              max="100"
              value={progress}
              onChange={handleSeek}
              className="w-full h-2 bg-[#1E2A40] rounded-lg appearance-none cursor-pointer accent-[#D0FF41]"
            />
          </div>

          {/* Audio & Fullscreen */}
          <button
            onClick={toggleMute}
            className="p-1.5 rounded-lg text-[#8E9BAE] hover:text-white hover:bg-[#131B2E] transition-colors"
          >
            {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>

          <button
            onClick={handleFullscreen}
            className="p-1.5 rounded-lg text-[#8E9BAE] hover:text-white hover:bg-[#131B2E] transition-colors"
          >
            <Maximize className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Padel Match Score Card Bar (Team A vs Team B) */}
      <div className="p-4 rounded-2xl bg-[#131B2E] border border-[#1E2A40] flex items-center justify-between shadow-lg">
        {/* Team A Card */}
        <div className="flex items-center gap-3 w-1/3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-[#0250B0] font-black text-white text-sm shadow-md">
            <Users className="w-5 h-5" />
            <span className="absolute -bottom-1 -right-1 w-3 h-3 rounded-full bg-[#D0FF41] border-2 border-[#131B2E]" />
          </div>
          <div>
            <h4 className="text-base font-bold text-white tracking-tight">Team A</h4>
            <p className="text-xs text-[#8E9BAE]">{player1Name} & {player2Name}</p>
          </div>
        </div>

        {/* Score Display */}
        <div className="flex flex-col items-center justify-center w-1/3">
          <span className="text-xs font-bold tracking-widest text-[#8E9BAE] uppercase mb-1">Score</span>
          <div className="flex items-center gap-3 px-4 py-1.5 rounded-xl bg-[#0E1626] border border-[#1E2A40]">
            <span className="text-2xl font-black text-white">{score1}</span>
            <span className="text-[#8E9BAE] font-bold">—</span>
            <span className="text-2xl font-black text-white">{score2}</span>
          </div>
        </div>

        {/* Team B Card */}
        <div className="flex items-center justify-end gap-3 w-1/3 text-right">
          <div>
            <h4 className="text-base font-bold text-white tracking-tight">Team B</h4>
            <p className="text-xs text-[#8E9BAE]">{player3Name} & {player4Name}</p>
          </div>
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-pink-600 font-black text-white text-sm shadow-md">
            <Users className="w-5 h-5" />
            <span className="absolute -bottom-1 -right-1 w-3 h-3 rounded-full bg-[#D0FF41] border-2 border-[#131B2E]" />
          </div>
        </div>
      </div>
    </div>
  );
}
