"use client";

import Link from "next/link";
import { useEffect, useRef, type ComponentProps, type ReactNode } from "react";
import { AlertTriangle, Info, X } from "lucide-react";

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

// --- buttons ---------------------------------------------------------------------------------
type Variant = "primary" | "secondary" | "ghost" | "danger";
const BUTTON: Record<Variant, string> = {
  primary: "bg-accent text-white hover:bg-accent-hover border border-accent",
  secondary: "bg-white text-zinc-800 border border-zinc-300 hover:bg-zinc-50 shadow-xs",
  ghost: "text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 border border-transparent",
  danger: "bg-white text-red-700 border border-zinc-300 hover:bg-red-50 hover:border-red-200",
};
const BUTTON_BASE =
  "inline-flex items-center justify-center gap-1.5 rounded-md px-3 h-8 text-sm font-medium " +
  "whitespace-nowrap transition-colors disabled:opacity-50 disabled:pointer-events-none";

export function Button({
  variant = "secondary",
  className,
  ...props
}: ComponentProps<"button"> & { variant?: Variant }) {
  return <button type="button" className={cx(BUTTON_BASE, BUTTON[variant], className)} {...props} />;
}

export function ButtonLink({
  variant = "secondary",
  className,
  ...props
}: ComponentProps<typeof Link> & { variant?: Variant }) {
  return <Link className={cx(BUTTON_BASE, BUTTON[variant], className)} {...props} />;
}

// --- surfaces --------------------------------------------------------------------------------
export function Card({
  title,
  description,
  action,
  children,
  className,
  bodyClassName,
}: {
  title?: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cx("rounded-lg border border-zinc-200 bg-white", className)}>
      {(title || action) && (
        <header className="flex items-start justify-between gap-4 border-b border-zinc-100 px-4 py-3">
          <div className="min-w-0">
            {title && <h2 className="text-sm font-semibold text-zinc-900">{title}</h2>}
            {description && <p className="mt-0.5 text-xs text-zinc-500">{description}</p>}
          </div>
          {action}
        </header>
      )}
      <div className={cx("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <div className="min-w-0">
      <div className="text-xs font-medium text-zinc-500">{label}</div>
      <div className="tabular mt-1 truncate text-xl font-semibold tracking-tight text-zinc-900">
        {value}
      </div>
      {hint && <div className="tabular mt-0.5 truncate text-xs text-zinc-500">{hint}</div>}
    </div>
  );
}

export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "accent" | "success" | "warning" | "danger";
  children: ReactNode;
}) {
  const tones = {
    neutral: "bg-zinc-100 text-zinc-700",
    accent: "bg-accent-soft text-accent",
    success: "bg-emerald-50 text-emerald-700",
    warning: "bg-amber-50 text-amber-800",
    danger: "bg-red-50 text-red-700",
  };
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium",
        tones[tone],
      )}
    >
      {children}
    </span>
  );
}

export function TeamDot({ team }: { team: "near" | "far" }) {
  return (
    <span
      aria-hidden
      className={cx("inline-block size-2 shrink-0 rounded-full", team === "near" ? "bg-near" : "bg-far")}
    />
  );
}

export function Callout({
  level,
  title,
  children,
}: {
  level: "info" | "warning";
  title: ReactNode;
  children?: ReactNode;
}) {
  const Icon = level === "warning" ? AlertTriangle : Info;
  return (
    <div
      className={cx(
        "flex gap-3 rounded-md border px-3 py-2.5 text-sm",
        level === "warning" ? "border-amber-200 bg-amber-50/60" : "border-zinc-200 bg-zinc-50",
      )}
    >
      <Icon
        className={cx("mt-0.5 size-4 shrink-0", level === "warning" ? "text-amber-600" : "text-zinc-500")}
      />
      <div className="min-w-0">
        <div className="font-medium text-zinc-900">{title}</div>
        {children && <div className="mt-0.5 text-zinc-600">{children}</div>}
      </div>
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  children,
  action,
}: {
  icon?: ReactNode;
  title: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center rounded-lg border border-dashed border-zinc-300 bg-white px-6 py-12 text-center">
      {icon && <div className="mb-3 text-zinc-400">{icon}</div>}
      <h3 className="text-sm font-semibold text-zinc-900">{title}</h3>
      {children && <div className="mt-1 max-w-md text-sm text-zinc-500">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-label="Loading"
      className={cx(
        "inline-block size-4 animate-spin rounded-full border-2 border-zinc-300 border-t-zinc-600",
        className,
      )}
    />
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-zinc-300 bg-white px-1 font-mono text-[11px] text-zinc-600">
      {children}
    </kbd>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("animate-pulse rounded bg-zinc-100", className)} />;
}

// --- dialog ----------------------------------------------------------------------------------
export function Dialog({
  open,
  onClose,
  title,
  children,
  footer,
  width = "max-w-lg",
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  width?: string;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      className={cx(
        "m-auto w-full rounded-lg border border-zinc-200 bg-white p-0 shadow-xl backdrop:bg-zinc-900/30",
        width,
      )}
    >
      <div className="flex items-center justify-between border-b border-zinc-100 px-5 py-3.5">
        <h2 className="text-base font-semibold text-zinc-900">{title}</h2>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900"
          aria-label="Close"
        >
          <X className="size-4" />
        </button>
      </div>
      <div className="px-5 py-4">{children}</div>
      {footer && (
        <div className="flex justify-end gap-2 border-t border-zinc-100 px-5 py-3">{footer}</div>
      )}
    </dialog>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: ReactNode;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-zinc-800">{label}</span>
      <div className="mt-1.5">{children}</div>
      {hint && <span className="mt-1 block text-xs text-zinc-500">{hint}</span>}
    </label>
  );
}

export const inputClass =
  "w-full rounded-md border border-zinc-300 bg-white px-2.5 h-9 text-sm text-zinc-900 " +
  "placeholder:text-zinc-400 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/15";
