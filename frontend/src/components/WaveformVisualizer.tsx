import { useEffect, useRef } from "react";

/**
 * Live waveform / spectrum visualizer using the Web Audio API's AnalyserNode.
 *
 * Renders frequency-bin bars on a canvas while `active` is true. When `stream`
 * is null or active is false, the canvas shows a flat baseline.
 */
export function WaveformVisualizer({
  stream,
  active,
  height = 64,
}: {
  stream: MediaStream | null;
  active: boolean;
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const cctx = canvas.getContext("2d");
    if (!cctx) return;

    if (!stream || !active) {
      // baseline pulse
      const draw = () => {
        const w = canvas.width;
        const h = canvas.height;
        cctx.fillStyle = "#0b0f15";
        cctx.fillRect(0, 0, w, h);
        cctx.strokeStyle = "#2a2f36";
        cctx.lineWidth = 1;
        cctx.beginPath();
        cctx.moveTo(0, h / 2);
        cctx.lineTo(w, h / 2);
        cctx.stroke();
      };
      draw();
      return;
    }

    const ac = new (window.AudioContext || (window as any).webkitAudioContext)();
    ctxRef.current = ac;
    const source = ac.createMediaStreamSource(stream);
    const analyser = ac.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    const bins = analyser.frequencyBinCount;
    const data = new Uint8Array(bins);

    const draw = () => {
      analyser.getByteFrequencyData(data);
      const w = canvas.width;
      const h = canvas.height;
      cctx.fillStyle = "#0b0f15";
      cctx.fillRect(0, 0, w, h);
      const barW = w / bins;
      for (let i = 0; i < bins; i++) {
        const v = data[i] / 255; // 0..1
        const barH = v * h * 0.95;
        const hue = 210 - v * 40; // blue → cyan as it gets louder
        cctx.fillStyle = `hsl(${hue}, 80%, ${40 + v * 30}%)`;
        cctx.fillRect(i * barW, h - barH, barW * 0.85, barH);
      }
      rafRef.current = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      try {
        source.disconnect();
        analyser.disconnect();
      } catch {}
      if (ctxRef.current && ctxRef.current.state !== "closed") {
        ctxRef.current.close().catch(() => {});
      }
    };
  }, [stream, active]);

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={height}
      className="waveform"
      style={{ width: "100%", height }}
    />
  );
}
