// Throwaway script for the ticket "Test segmentation in ONNX Runtime Web, and
// the Docker image without torch, clip and ultralytics" (#204) of the map #189.
// Never merged.
//
// Loads a YOLOE ONNX export in onnxruntime-web (the WASM backend, as in a
// browser), then for each JSON line on stdin ({input, shape, output0,
// output1}) runs the saved input and writes both outputs as float32, printing
// one line per run for seg_web_check.py to wait on.
//
//     ORT_WEB_DIR=/tmp/ortweb node scripts/seg_web_check.mjs MODEL

import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { createRequire } from "node:module";

const require = createRequire(path.join(process.env.ORT_WEB_DIR, "x.js"));
const ort = require("onnxruntime-web");

ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(process.argv[2], {
  executionProviders: ["wasm"],
});

for await (const line of readline.createInterface({ input: process.stdin })) {
  const job = JSON.parse(line);
  const buffer = fs.readFileSync(job.input);
  const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
  const start = performance.now();
  const results = await session.run({
    [session.inputNames[0]]: new ort.Tensor("float32", data, job.shape),
  });
  const elapsed = performance.now() - start;
  for (const [name, file] of [
    [session.outputNames[0], job.output0],
    [session.outputNames[1], job.output1],
  ]) {
    fs.writeFileSync(file, Buffer.from(results[name].data.buffer));
  }
  console.log(JSON.stringify({ ms: elapsed }));
}
