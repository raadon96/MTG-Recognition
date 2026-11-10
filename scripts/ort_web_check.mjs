// Throwaway script for the ticket "Test PP-OCR in onnxruntime without
// paddlepaddle" (#199) of the map #189. Never merged.
//
// Runs each ONNX model listed in DIR/cases.json through onnxruntime-web (the
// WASM backend, as in a browser) on the input ort_web_check.py saved, and saves
// the output next to it for ort_web_check.py to compare.
//
//     node scripts/ort_web_check.mjs DIR    (with onnxruntime-web on NODE_PATH)

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(path.join(process.env.ORT_WEB_DIR, "x.js"));
const ort = require("onnxruntime-web");

const dir = process.argv[2];
const cases = JSON.parse(fs.readFileSync(path.join(dir, "cases.json"), "utf8"));
ort.env.wasm.numThreads = 1;

for (const c of cases) {
  const session = await ort.InferenceSession.create(c.model, {
    executionProviders: ["wasm"],
  });
  const buffer = fs.readFileSync(path.join(dir, `${c.name}.in.bin`));
  const data = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
  const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", data, c.shape) };
  const start = performance.now();
  const results = await session.run(feeds);
  const elapsed = performance.now() - start;
  const output = results[session.outputNames[0]];
  fs.writeFileSync(path.join(dir, `${c.name}.web.bin`), Buffer.from(output.data.buffer));
  console.log(`${c.name}: output ${JSON.stringify(output.dims)}, ${elapsed.toFixed(0)} ms`);
}
