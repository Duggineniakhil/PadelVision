"use client";

import Link from "next/link";
import { Video, ChevronDown } from "lucide-react";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full bg-[#0E1626]/90 backdrop-blur-md border-b border-[#1E2A40]">
      <div className="max-w-[1600px] mx-auto px-4 md:px-6 h-16 flex items-center justify-between">
        
        {/* Brand Logo */}
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2 group">
            <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-[#0250B0] font-black text-white text-lg tracking-tighter elite-glow transition-transform group-hover:scale-105">
              <span>TV</span>
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-[#D0FF41] volt-glow border-2 border-[#0E1626]" />
            </div>
            <div className="flex items-center text-xl font-extrabold tracking-wider text-white">
              <span>PADEL</span>
              <span className="text-[#0250B0] ml-1">VISION</span>
              <span className="inline-block w-2.5 h-2.5 rounded-full bg-[#D0FF41] ml-1 animate-pulse" />
            </div>
          </Link>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-[#C6D0DD]">
            <div className="relative group flex items-center gap-1 hover:text-white cursor-pointer transition-colors py-2">
              <span>Learn</span>
              <ChevronDown className="w-4 h-4 text-[#8E9BAE] group-hover:text-white transition-colors" />
            </div>
            <Link href="/" className="hover:text-white transition-colors py-2">
              Pricing
            </Link>
            <Link href="/" className="hover:text-white transition-colors py-2">
              Demos
            </Link>
          </nav>
        </div>

        {/* Right CTA Actions */}
        <div className="flex items-center gap-4">
          <Link 
            href="/" 
            className="hidden sm:block text-sm font-semibold text-[#C6D0DD] hover:text-white px-3 py-2 transition-colors"
          >
            Sign In
          </Link>

          <Link
            href="/"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[#D0FF41] hover:bg-[#B7E62B] text-[#0A0F1D] font-bold text-sm transition-all volt-glow active:scale-95"
          >
            <span>Get Started</span>
            <Video className="w-4 h-4 text-[#0A0F1D]" />
          </Link>
        </div>

      </div>
    </header>
  );
}
