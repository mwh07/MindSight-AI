import { Link, useLocation } from "react-router-dom";
import { Home, ArrowLeft, SearchX } from "lucide-react";
import Navbar from "../components/landing/Navbar";
import FooterSection from "../components/landing/FooterSection";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";

export default function PageNotFound() {
  const location = useLocation();
  const pageName = location.pathname.substring(1) || "this page";

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <main className="page-shell flex flex-1 items-center justify-center">
        <Card className="card-surface w-full max-w-md border-border/60">
          <CardContent className="px-6 py-10 text-center sm:px-10">
            <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
              <SearchX className="h-7 w-7 text-primary" />
            </div>

            <p className="font-display text-6xl font-bold tracking-tight text-primary/20">404</p>
            <h1 className="mt-2 font-display text-2xl font-bold">Page not found</h1>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              The page{" "}
              <span className="font-medium text-foreground">&ldquo;/{pageName}&rdquo;</span>{" "}
              doesn&apos;t exist or may have been moved.
            </p>

            <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:justify-center">
              <Link to="/">
                <Button className="w-full rounded-full gap-2 sm:w-auto">
                  <Home className="h-4 w-4" />
                  Go to Home
                </Button>
              </Link>
              <Link to="/assessment">
                <Button variant="outline" className="w-full rounded-full gap-2 sm:w-auto">
                  <ArrowLeft className="h-4 w-4" />
                  Start Assessment
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </main>
      <FooterSection compact />
    </div>
  );
}
