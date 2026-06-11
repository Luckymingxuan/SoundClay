import { ChangeEvent, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  ArrowDownToLine,
  AudioWaveform,
  CheckCircle2,
  FileMusic,
  Loader2,
  Sparkles,
  UploadCloud,
  X,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

type TranscriptionResult = {
  instrument: string;
  confidence: number;
  model: string;
  noteCount: number;
  sampleCount: number;
  durationSeconds: number;
  midiFileName: string;
  midiBytes: number[];
  instrumentFileName: string;
  instrumentBytes: number[];
};

type Notice = {
  title: string;
  message: string;
  path?: string;
};

function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [error, setError] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0] ?? null;
    setSelectedFile(file);
    setResult(null);
    setError("");
    setNotice(null);
  }

  async function transcribe() {
    if (!selectedFile) {
      setError("Choose a WAV file first.");
      return;
    }

    setIsProcessing(true);
    setError("");
    setResult(null);

    try {
      const buffer = await selectedFile.arrayBuffer();
      const audioBytes = Array.from(new Uint8Array(buffer));
      const response = await invoke<TranscriptionResult>("transcribe_audio", {
        fileName: selectedFile.name,
        audioBytes,
      });
      setResult(response);
      setNotice({
        title: "Generation complete",
        message: `Created ${response.noteCount} MIDI notes and ${response.sampleCount} instrument samples. The files have not been saved yet.`,
      });
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : String(caughtError));
    } finally {
      setIsProcessing(false);
    }
  }

  async function saveGeneratedFile(fileName: string, fileBytes: number[], label: string) {
    setError("");
    try {
      const savedPath = await invoke<string>("save_download", {
        fileName,
        fileBytes,
      });
      setNotice({
        title: `${label} saved`,
        message: "The download completed successfully.",
        path: savedPath,
      });
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : String(caughtError));
    }
  }

  async function downloadMidi() {
    if (!result) {
      return;
    }

    await saveGeneratedFile(result.midiFileName, result.midiBytes, "MIDI");
  }

  async function downloadInstrument() {
    if (!result) {
      return;
    }

    await saveGeneratedFile(
      result.instrumentFileName,
      result.instrumentBytes,
      "Instrument",
    );
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[radial-gradient(circle_at_20%_20%,rgba(20,184,166,0.18),transparent_30%),radial-gradient(circle_at_80%_10%,rgba(251,191,36,0.18),transparent_28%),linear-gradient(135deg,#fafaf9_0%,#f5f0e8_100%)] px-5 py-8 text-stone-950">
      <section className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-5xl flex-col justify-center gap-8">
        <div className="max-w-3xl space-y-5">
          <Badge className="bg-white/70 backdrop-blur">SoundClay v1.2</Badge>
          <div className="space-y-4">
            <h1 className="text-5xl font-semibold tracking-tight text-balance md:text-7xl">
              Audio in. MIDI out.
            </h1>
            <p className="max-w-2xl text-lg leading-8 text-stone-600">
              Upload a single-instrument WAV. SoundClay transcribes the notes,
              extracts playable samples, and exports editable MIDI plus SFZ.
            </p>
          </div>
        </div>

        <div className="grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
          <Card className="border-white/70 bg-white/75 shadow-2xl shadow-stone-300/30 backdrop-blur">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UploadCloud className="size-5" />
                Create assets
              </CardTitle>
              <CardDescription>
                v1.2 creates MIDI and an experimental SFZ instrument from WAV input.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <button
                className="group flex min-h-48 w-full flex-col items-center justify-center rounded-2xl border border-dashed border-stone-300 bg-stone-50/70 p-8 text-center transition hover:border-stone-900 hover:bg-white"
                onClick={() => inputRef.current?.click()}
                type="button"
              >
                <AudioWaveform className="mb-4 size-10 text-stone-900 transition group-hover:scale-105" />
                <span className="text-lg font-medium">
                  {selectedFile ? selectedFile.name : "Choose a WAV file"}
                </span>
                <span className="mt-2 text-sm text-stone-500">
                  Piano, guitar, bass, vocal, synth, or strings.
                </span>
              </button>
              <input
                ref={inputRef}
                accept=".wav,audio/wav,audio/wave"
                className="hidden"
                onChange={onFileChange}
                type="file"
              />

              <div className="grid gap-3 sm:grid-cols-3">
                <Button
                  className="flex-1"
                  disabled={!selectedFile || isProcessing}
                  onClick={transcribe}
                  size="lg"
                >
                  {isProcessing ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Sparkles className="size-4" />
                  )}
                  {isProcessing ? "Listening..." : "Generate"}
                </Button>
                <Button
                  disabled={!result}
                  onClick={downloadMidi}
                  size="lg"
                  variant="secondary"
                >
                  <ArrowDownToLine className="size-4" />
                  Download
                </Button>
                <Button
                  disabled={!result}
                  onClick={downloadInstrument}
                  size="lg"
                  variant="secondary"
                >
                  <ArrowDownToLine className="size-4" />
                  Instrument
                </Button>
              </div>

              {error ? (
                <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {error}
                </div>
              ) : null}
            </CardContent>
          </Card>

          <Card className="border-stone-950 bg-stone-950 text-stone-50 shadow-2xl shadow-stone-400/40">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileMusic className="size-5" />
                Result
              </CardTitle>
              <CardDescription className="text-stone-400">
                MIDI and SFZ instrument metadata will appear here.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {result ? (
                <div className="space-y-5">
                  <div>
                    <p className="text-sm text-stone-400">Instrument route</p>
                    <p className="mt-1 text-4xl font-semibold">{result.instrument}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <Metric label="Confidence" value={`${Math.round(result.confidence * 100)}%`} />
                    <Metric label="Notes" value={result.noteCount.toString()} />
                    <Metric label="Samples" value={result.sampleCount.toString()} />
                    <Metric label="Duration" value={`${result.durationSeconds}s`} />
                    <Metric label="MIDI" value={result.midiFileName} />
                    <Metric label="Instrument" value={result.instrumentFileName} />
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/5 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-stone-500">
                      Selected model
                    </p>
                    <p className="mt-2 font-mono text-sm text-stone-200">{result.model}</p>
                  </div>
                </div>
              ) : (
                <div className="flex min-h-72 flex-col justify-center rounded-2xl border border-white/10 bg-white/[0.03] p-6 text-stone-400">
                  <p className="text-sm uppercase tracking-[0.2em]">Waiting</p>
                  <p className="mt-3 text-2xl font-medium text-stone-100">
                    Drop in one sound. We will shape it into notes.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </section>
      {notice ? <SuccessNotice notice={notice} onClose={() => setNotice(null)} /> : null}
    </main>
  );
}

function SuccessNotice({ notice, onClose }: { notice: Notice; onClose: () => void }) {
  return (
    <div
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-stone-950/40 p-5 backdrop-blur-sm"
      role="dialog"
    >
      <Card className="w-full max-w-md border-white/70 bg-white shadow-2xl">
        <CardHeader>
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="flex size-10 items-center justify-center rounded-full bg-emerald-100 text-emerald-700">
                <CheckCircle2 className="size-5" />
              </span>
              <div>
                <CardTitle>{notice.title}</CardTitle>
                <CardDescription className="mt-1">{notice.message}</CardDescription>
              </div>
            </div>
            <Button
              aria-label="Close"
              className="size-8 p-0"
              onClick={onClose}
              size="sm"
              variant="ghost"
            >
              <X className="size-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {notice.path ? (
            <div className="rounded-xl border bg-stone-50 p-3">
              <p className="text-xs font-medium uppercase tracking-wider text-stone-500">
                Saved to
              </p>
              <p className="mt-2 break-all font-mono text-sm text-stone-800">{notice.path}</p>
            </div>
          ) : null}
          <Button className="w-full" onClick={onClose}>
            Done
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-4">
      <p className="text-xs text-stone-500">{label}</p>
      <p className="mt-1 truncate text-lg font-semibold">{value}</p>
    </div>
  );
}

export default App;
