import useSWR, { mutate } from "swr";
import type { components } from "./api-types";

type Schemas = components["schemas"];
export type RunListItem = Schemas["RunListItem"];
export type RunDetail = Schemas["RunDetail"];
export type PlayerStats = Schemas["PlayerStats"];
export type TeamStats = Schemas["TeamStats"];
export type Rally = Schemas["Rally"];
export type Shot = Schemas["Shot"];
export type MatchEvent = Schemas["Event"];
export type TracksResponse = Schemas["TracksResponse"];
export type OverlayResponse = Schemas["OverlayResponse"];
export type HighlightsResponse = Schemas["HighlightsResponse"];
export type HighlightsVideo = Schemas["HighlightsVideo"];
export type QualityNote = Schemas["QualityNote"];
export type CourtInfo = Schemas["CourtInfo"];
export type Team = PlayerStats["team"];

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(
      `Can't reach the PadelVision API at ${API_URL}. Start it with "padelvision-api".`,
      0,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(detail, res.status);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const fetcher = <T,>(path: string) => request<T>(path);

export const fileUrl = (runId: string, name: string) => `${API_URL}/api/runs/${runId}/files/${name}`;
export const videoUrl = (runId: string) => `${API_URL}/api/runs/${runId}/video`;

// --- reads -----------------------------------------------------------------------------------
export function useRuns() {
  return useSWR<RunListItem[]>("/api/runs", fetcher, {
    refreshInterval: (runs) => (runs?.some((r) => r.status === "processing") ? 2000 : 0),
  });
}

export function useRun(id: string | null) {
  return useSWR<RunDetail>(id ? `/api/runs/${id}` : null, fetcher, {
    refreshInterval: (run) => (run?.status === "processing" ? 2000 : 0),
  });
}

/** Run sub-resources are only fetched once the run is ready (they change while processing). */
function useRunPart<T>(id: string, part: string, ready: boolean) {
  return useSWR<T>(ready ? `/api/runs/${id}/${part}` : null, fetcher, { revalidateOnFocus: false });
}

export const useRallies = (id: string, ready: boolean) =>
  useRunPart<Schemas["RalliesResponse"]>(id, "rallies", ready);
export const useEvents = (id: string, ready: boolean) => useRunPart<MatchEvent[]>(id, "events", ready);
export const useTracks = (id: string, ready: boolean) =>
  useRunPart<TracksResponse>(id, "tracks?sample_fps=5", ready);
export const useHighlights = (id: string, ready: boolean) =>
  useRunPart<HighlightsResponse>(id, "highlights", ready);

export const fetchOverlay = (id: string, start: number, end: number) =>
  request<OverlayResponse>(`/api/runs/${id}/overlay?start=${start}&end=${end}`);

// --- writes ----------------------------------------------------------------------------------
function refreshRun(id: string) {
  void mutate("/api/runs");
  void mutate((key) => typeof key === "string" && key.startsWith(`/api/runs/${id}`));
}

export async function renameRun(id: string, name: string) {
  await request<RunListItem>(`/api/runs/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  refreshRun(id);
}

export async function deleteRun(id: string) {
  await request<void>(`/api/runs/${id}`, { method: "DELETE" });
  void mutate("/api/runs");
}

export async function reprocessRun(id: string) {
  await request<RunListItem>(`/api/runs/${id}/reprocess`, { method: "POST" });
  refreshRun(id);
}

export async function attachVideoPath(id: string, path: string) {
  await request<RunDetail>(`/api/runs/${id}/video`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  refreshRun(id);
}

/** Multipart upload with progress (fetch can't report upload progress). */
function upload<T>(path: string, form: FormData, onProgress?: (fraction: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}${path}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* not JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else {
        const detail = (body as { detail?: unknown } | null)?.detail;
        reject(new ApiError(typeof detail === "string" ? detail : xhr.statusText, xhr.status));
      }
    };
    xhr.onerror = () => reject(new ApiError(`Can't reach the PadelVision API at ${API_URL}.`, 0));
    xhr.send(form);
  });
}

export async function importRun(files: File[], name: string, onProgress?: (f: number) => void) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  if (name.trim()) form.append("name", name.trim());
  const run = await upload<RunListItem>("/api/runs", form, onProgress);
  void mutate("/api/runs");
  return run;
}

export async function uploadVideo(id: string, file: File, onProgress?: (f: number) => void) {
  const form = new FormData();
  form.append("file", file);
  await upload<RunDetail>(`/api/runs/${id}/video`, form, onProgress);
  refreshRun(id);
}
