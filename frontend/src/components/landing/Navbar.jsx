import React, { useState, useEffect } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { Button } from "../ui/button";
import { Brain, Menu, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "../../lib/utils";

const NAV_LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/about", label: "About" },
  { to: "/dashboard", label: "Dashboard" },
];

function navLinkClass(isActive, mobile = false) {
  return cn(
    mobile
      ? "block rounded-lg px-3 py-2.5 text-sm font-medium transition-colors"
      : "relative px-1 py-1 text-sm font-medium transition-colors",
    isActive
      ? mobile
        ? "bg-primary/10 text-primary"
        : "text-primary"
      : "text-muted-foreground hover:text-foreground"
  );
}

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border/60 bg-background/85 backdrop-blur-xl shadow-sm">
      <div className="page-container px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <Link
            to="/"
            className="flex items-center gap-2.5 rounded-lg transition-opacity hover:opacity-90"
          >
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary shadow-sm">
              <Brain className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="font-display text-xl font-bold tracking-tight">MINDSIGHT</span>
          </Link>

          <div className="hidden items-center gap-7 md:flex">
            {NAV_LINKS.map(({ to, label, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  cn(
                    "relative py-1 text-sm font-medium transition-colors",
                    isActive
                      ? "text-primary after:absolute after:-bottom-[1.35rem] after:left-0 after:right-0 after:h-0.5 after:rounded-full after:bg-primary after:content-['']"
                      : "text-muted-foreground hover:text-foreground"
                  )
                }
              >
                {label}
              </NavLink>
            ))}
            <Link to="/assessment">
              <Button className="rounded-full px-6 shadow-sm">Start Assessment</Button>
            </Link>
          </div>

          <button
            type="button"
            className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-border/60 text-foreground transition-colors hover:bg-secondary md:hidden"
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            aria-label={open ? "Close menu" : "Open menu"}
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-border/60 bg-background md:hidden"
          >
            <div className="space-y-1 px-4 py-4">
              {NAV_LINKS.map(({ to, label, end }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={end}
                  className={({ isActive }) => navLinkClass(isActive, true)}
                  onClick={() => setOpen(false)}
                >
                  {label}
                </NavLink>
              ))}
              <Link to="/assessment" onClick={() => setOpen(false)} className="block pt-2">
                <Button className="w-full rounded-full">Start Assessment</Button>
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
