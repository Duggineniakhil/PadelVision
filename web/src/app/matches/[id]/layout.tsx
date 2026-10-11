import { Suspense } from "react";
import { MatchLayout } from "@/components/match-layout";
import { Skeleton } from "@/components/ui";

// Match pages render in the browser from API data (nothing to prerender), so navigations into
// them are allowed to wait for that data.
export const instant = false;

export default function Layout({ children }: LayoutProps<"/matches/[id]">) {
  return (
    <Suspense
      fallback={
        <div className="space-y-4">
          <Skeleton className="h-8 w-72" />
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      }
    >
      <MatchLayout>{children}</MatchLayout>
    </Suspense>
  );
}
