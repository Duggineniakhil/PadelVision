"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { uploadVideo } from "@/lib/api";
import { UploadCloud, Film } from "lucide-react";

export default function Uploader() {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const handleUpload = useCallback(async (file: File) => {
    try {
      setIsUploading(true);
      setError(null);
      const { job_id } = await uploadVideo(file);
      router.push(`/analysis/${job_id}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to upload video");
      setIsUploading(false);
    }
  }, [router]);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const onDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await handleUpload(e.dataTransfer.files[0]);
    }
  }, [handleUpload]);

  const onChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await handleUpload(e.target.files[0]);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-6">
      
      {/* Upload Box */}
      <div
        className={`relative border-2 border-dashed rounded-3xl p-10 text-center transition-all cursor-pointer ${
          isDragging
            ? "border-[#D0FF41] bg-[#D0FF41]/10 volt-glow"
            : "border-[#1E2A40] bg-[#131B2E]/90 hover:bg-[#19243C] hover:border-[#0250B0] glass-panel"
        }`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        {isUploading ? (
          <div className="flex flex-col items-center justify-center space-y-4 py-8">
            <div className="w-14 h-14 border-4 border-[#D0FF41] border-t-transparent rounded-full animate-spin" />
            <div className="space-y-1">
              <p className="text-xl font-extrabold text-white">Uploading Match Video...</p>
              <p className="text-xs text-[#8E9BAE]">Preparing computer vision pipeline</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center space-y-4 py-4">
            <div className="w-16 h-16 rounded-2xl bg-[#0250B0]/20 border border-[#0250B0]/40 flex items-center justify-center text-[#D0FF41] elite-glow mb-2">
              <UploadCloud className="w-8 h-8 text-[#D0FF41]" />
            </div>

            <div className="space-y-1">
              <h3 className="text-2xl font-black tracking-tight text-white">
                Drag & drop your match video
              </h3>
              <p className="text-sm text-[#8E9BAE]">
                Supports <span className="text-[#C6D0DD] font-semibold">.mp4, .mov, .avi, .mkv</span> formats up to 500MB
              </p>
            </div>

            <label className="inline-flex items-center gap-2 px-6 py-3 mt-4 text-sm font-extrabold text-[#0A0F1D] bg-[#D0FF41] hover:bg-[#B7E62B] rounded-xl cursor-pointer transition-all volt-glow active:scale-95 shadow-lg">
              <Film className="w-4 h-4 text-[#0A0F1D]" />
              <span>Browse File</span>
              <input
                type="file"
                className="hidden"
                accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,.mkv"
                onChange={onChange}
              />
            </label>
          </div>
        )}
      </div>

      {error && (
        <div className="p-4 text-sm text-red-200 bg-red-950/40 rounded-xl border border-red-800 text-center">
          {error}
        </div>
      )}

    </div>
  );
}
