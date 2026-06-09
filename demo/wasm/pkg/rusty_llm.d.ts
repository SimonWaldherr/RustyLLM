/* tslint:disable */
/* eslint-disable */
export class WasmRunner {
  free(): void;
  /**
   * Cosine similarity between two `Float32Array` embeddings produced by `embed()`.
   * Returns a value in [-1, 1]; higher means more similar.
   */
  static cosine_similarity(a: Float32Array, b: Float32Array): number;
  /**
   * Creates a WASM runner from GGUF bytes supplied by JavaScript.
   */
  constructor(model_bytes: Uint8Array);
  /**
   * Returns an L2-normalised embedding vector as a `Float32Array`.
   * Suitable for cosine similarity / RAG retrieval directly in JS.
   */
  embed(text: string): Float32Array;
  /**
   * Generates a complete response for a plain prompt.
   */
  generate(prompt: string, max_tokens: number, temp: number): string;
  /**
   * Returns the optional model name from GGUF metadata.
   */
  readonly model_name: string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmrunner_free: (a: number, b: number) => void;
  readonly wasmrunner_cosine_similarity: (a: number, b: number, c: number) => void;
  readonly wasmrunner_embed: (a: number, b: number, c: number, d: number) => void;
  readonly wasmrunner_generate: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly wasmrunner_model_name: (a: number, b: number) => void;
  readonly wasmrunner_new: (a: number, b: number, c: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_export_0: (a: number, b: number, c: number) => void;
  readonly __wbindgen_export_1: (a: number, b: number) => number;
  readonly __wbindgen_export_2: (a: number, b: number, c: number, d: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
