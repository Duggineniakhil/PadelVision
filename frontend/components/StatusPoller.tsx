"use client";

import { useEffect, useState } from "react";
import { getStatus, getAnalysis } from "@/lib/api";
import type { AnalysisData } from "@/lib/api";
import { CheckCircle2, AlertTriangle, Loader2 } from "lucide-react";

interface StatusPollerProps {
  jobId: string;
  onComplete: (data: AnalysisData) => void;
}

export default function StatusPoller({ jobId, onComplete }: StatusPollerProps) {
  const [status, setStatus] = useState<string>("queued");
  const [stage, setStage] = useState<string>("Initializing computer vision models...");
  const [progress, setProgress] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const data = await getStatus(jobId);
        setStatus(data.status);
        setStage(data.stage);
        setProgress(data.progress);

        if (data.status === "done") {
          clearInterval(intervalId);
          const analysisData = await getAnalysis(jobId);
          onComplete(analysisData);
        } else if (data.status === "error") {
          clearInterval(intervalId);
          const analysisData = await getAnalysis(jobId);
          setError(analysisData.error || "An error occurred during processing.");
        }
      } catch (err: unknown) {
        console.error("Error polling status:", err);
      }
    };

    checkStatus();
    const intervalId = setInterval(checkStatus, 2500);

    return () => clearInterval(intervalId);
  }, [jobId, onComplete]);

  if (status === "error") {
    return (
      <div className="p-8 text-center bg-red-950/40 border border-red-800 rounded-2xl max-w-lg mx-auto">
        <div className="inline-flex items-center justify-center w-14 h-14 mb-4 bg-red-900/50 rounded-2xl text-red-400 border border-red-700">
          <AlertTriangle className="w-7 h-7" />
        </div>
        <h3 className="text-xl font-bold text-red-400 mb-2">Analysis Failed</h3>
        <p className="text-sm text-red-200/80">{error}</p>
      </div>
    );
  }

  const steps = [
    { label: "Detecting court boundaries & keypoints", minProgress: 10 },
    { label: "Tracking player movement (YOLOv8)", minProgress: 25 },
    { label: "Computing ball trajectories & speed", minProgress: 50 },
    { label: "Calculating match statistics & rallies", minProgress: 75 },
    { label: "Generating heatmaps & highlight clips", minProgress: 90 },
  ];

  return (
    <div className="p-8 text-center bg-[#131B2E] border border-[#1E2A40] rounded-3xl max-w-xl mx-auto w-full shadow-2xl glass-panel">
      
      {/* Circular Progress Gauge */}
      <div className="relative w-28 h-28 mx-auto mb-6">
        <svg className="w-full h-full text-[#1E2A40]" viewBox="0 0 100 100">
          <circle className="stroke-current" strokeWidth="8" cx="50" cy="50" r="42" fill="transparent" />
        </svg>
        <svg className="absolute top-0 left-0 w-full h-full text-[#D0FF41] -rotate-90" viewBox="0 0 100 100">
          <circle
            className="stroke-current transition-all duration-700 ease-out"
            strokeWidth="8"
            strokeLinecap="round"
            cx="50"
            cy="50"
            r="42"
            fill="transparent"
            strokeDasharray="263.8"
            strokeDashoffset={263.8 - (263.8 * progress) / 100}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-black text-white font-mono">{progress}%</span>
          <span className="text-[10px] text-[#8E9BAE] uppercase font-bold tracking-wider">Loaded</span>
        </div>
      </div>

      <h3 className="text-2xl font-extrabold text-white mb-2">Analyzing Match Video</h3>
      <p className="text-sm text-[#D0FF41] font-medium animate-pulse mb-6 flex items-center justify-center gap-2">
        <Loader2 className="w-4 h-4 animate-spin text-[#D0FF41]" />
        <span>{stage}</span>
      </p>

      {/* Checklist Steps */}
      <div className="text-left text-xs bg-[#0E1626] border border-[#1E2A40] p-4 rounded-xl space-y-2.5">
        <p className="font-bold text-[#C6D0DD] mb-3 uppercase tracking-wider text-[11px]">
          AI Computer Vision Pipeline:
        </p>
        {steps.map((step, idx) => {
          const isDone = progress >= step.minProgress;
          return (
            <div key={idx} className="flex items-center gap-2.5">
              <CheckCircle2 className={`w-4 h-4 ${isDone ? "text-[#D0FF41]" : "text-[#1E2A40]"}`} />
              <span className={isDone ? "text-[#C6D0DD] font-medium" : "text-[#8E9BAE]"}>
                {step.label}
              </span>
            </div>
          );
        })}
      </div>

    </div>
  );
}
