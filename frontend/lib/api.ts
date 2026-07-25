export const API_BASE = "/api";

export interface PlayerMetrics {
  total_shots: number;
  avg_shot_speed: number;
  max_shot_speed: number;
  avg_player_speed: number;
  distance_covered: number;
}

export interface Highlight {
  start_frame: number;
  end_frame: number;
  timestamp_seconds: number;
  label: string;
}

export interface StatusData {
  job_id: string;
  status: "queued" | "processing" | "done" | "error";
  stage: string;
  progress: number;
}

export interface AnalysisData extends StatusData {
  player_1?: PlayerMetrics;
  player_2?: PlayerMetrics;
  player_3?: PlayerMetrics;
  player_4?: PlayerMetrics;
  highlights?: Highlight[];
  video_url?: string;
  heatmap_p1_url?: string;
  heatmap_p2_url?: string;
  heatmap_p3_url?: string;
  heatmap_p4_url?: string;
  shot_map_url?: string;
  highlights_video_url?: string;
  total_frames?: number;
  fps?: number;
  error?: string;
}

export async function uploadVideo(file: File): Promise<{ job_id: string; message: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Upload failed");
  }

  return res.json();
}

export async function getStatus(jobId: string): Promise<StatusData> {
  const res = await fetch(`${API_BASE}/status/${jobId}`);
  if (!res.ok) {
    throw new Error("Failed to fetch status");
  }
  return res.json();
}

export async function getAnalysis(jobId: string): Promise<AnalysisData> {
  const res = await fetch(`${API_BASE}/analysis/${jobId}`);
  if (!res.ok) {
    throw new Error("Failed to fetch analysis");
  }
  return res.json();
}
