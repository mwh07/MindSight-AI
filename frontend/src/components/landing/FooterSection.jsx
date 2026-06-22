import React from "react";
import { Brain } from "lucide-react";

export default function FooterSection({ compact = false }) {
  return (
    <footer className="mt-auto border-t border-border/60 bg-secondary/25">
      <div className="page-container px-4 py-8 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-3">
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
                <Brain className="h-4 w-4 text-primary-foreground" />
              </div>
              <span className="font-display text-lg font-bold">MINDSIGHT</span>
            </div>
            {!compact && (
              <p className="max-w-md text-sm leading-relaxed text-muted-foreground">
                Unified Multi-Domain Psychological Diagnostic Profiling System.
              </p>
            )}
          </div>

          <div className="space-y-2 md:text-right">
            <p className="text-xs font-medium text-muted-foreground">© 2026 KGEC Research Team</p>
            <p className="text-xs leading-relaxed text-muted-foreground/70 max-w-sm md:ml-auto">
              For educational and research purposes only. Not a substitute for professional medical advice.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
