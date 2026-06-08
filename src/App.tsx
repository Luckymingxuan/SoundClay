import { ChangeEvent, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  ArrowDownToLine,
  AudioWaveform,
  FileMusic,
  Loader2,
  Sparkles,
  UploadCloud,
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
  durationSeconds: number;
  midiFileName: string;
  midiBytes: number[];
};

function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [error, setError] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0] ?? null;
    setSelectedFile(file);
    setResult(null);
    setError("");
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
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : String(caughtError));
    } finally {
      setIsProcessing(false);
    }
  }

  function downloadMidi() {
    if (!result) {
      return;
    }

    const blob = new Blob([new Uint8Array(result.midiBytes)], {
      type: "audio/midi",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = result.midiFileName;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[radial-gradient(circle_at_20%_20%,rgba(20,184,166,0.18),transparent_30%),radial-gradient(circle_at_80%_10%,rgba(251,191,36,0.18),transparent_28%),linear-gradient(135deg,#fafaf9_0%,#f5f0e8_100%)] px-5 py-8 text-stone-950">
      <section className="mx-auto flex min-h-[calc(100vh-4rem)] w-full max-w-5xl flex-col justify-center gap-8">
        <div className="max-w-3xl space-y-5">
          <Badge className="bg-white/70 backdrop-blur">SoundClay v1.1</Badge>
          <div className="space-y-4">
            <h1 className="text-5xl font-semibold tracking-tight text-balance md:text-7xl">
              Audio in. MIDI out.
            </h1>
            <p className="max-w-2xl text-lg leading-8 text-stone-600">
              Upload a single-instrument WAV. SoundClay classifies the source,
              selects a v1.1 transcription backend, and exports editable MIDI.
            </p>
          </div>
        </div>

        <div className="grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
          <Card className="border-white/70 bg-white/75 shadow-2xl shadow-stone-300/30 backdrop-blur">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UploadCloud className="size-5" />
                Create MIDI
              </CardTitle>
              <CardDescription>
                v1.1 supports WAV input while the model adapters are being wired in.
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

              <div className="flex flex-col gap-3 sm:flex-row">
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
                  {isProcessing ? "Listening..." : "Generate MIDI"}
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
                Classification and MIDI metadata will appear here.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {result ? (
                <div className="space-y-5">
                  <div>
                    <p className="text-sm text-stone-400">Detected instrument</p>
                    <p className="mt-1 text-4xl font-semibold">{result.instrument}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <Metric label="Confidence" value={`${Math.round(result.confidence * 100)}%`} />
                    <Metric label="Notes" value={result.noteCount.toString()} />
                    <Metric label="Duration" value={`${result.durationSeconds}s`} />
                    <Metric label="Output" value={result.midiFileName} />
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
    </main>
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
