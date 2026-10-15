# Decision models listed in Atomic Chat (read 2026-10-14)

Source: the Atomic Chat app, "Model type: Decision" list, read through screenshots with Windows-MCP.
Everything below is what the app's model pages and READMEs say. **None of these models was downloaded or run by me**, so claims such as speed and accuracy are the authors' own and unverified. Text was read from small screenshots, so treat odd digits with care.

| # | Model | Repo | Engine | Languages | Context | Backbone | Params | License | Size on disk |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Kev 4B (**downloaded**) | ggml-org/Kev-4B-GGUF | llama.cpp | EN | 8192 | Qwen3.5-4B-Base | 4.2B | apache-2.0 | 2.83 GB (Q4_K_M) |
| 2 | Laya Multilingual | convaiinnovations/laya-multilingual | TurboQuant | 100+ | 1024 (8192 with `max_len`) | mmBERT-base | 322M | apache-2.0 | 1.24 GB (CPU) |
| 3 | Laya | convaiinnovations/laya | TurboQuant | EN | 512 | ModernBERT-large | 421M | apache-2.0 | 1.57 GB (CPU) |
| 4 | Laya Typed Decisions | convaiinnovations/laya-typed-decisions | TurboQuant | EN | 1024 | ModernBERT-large | 421M | apache-2.0 | 1.57 GB (CPU) |
| 5 | Julia 1 | ggml-org/Julia-1-GGUF | llama.cpp | multilingual | 1024 | mmBERT-small | 144M | apache-2.0 | 0.16 GB (Q8_0) |
| 6 | Laya (GGUF) | ggml-org/Laya-GGUF | llama.cpp | EN | 512 | ModernBERT-large | 421M | apache-2.0 | 0.42 GB (Q8_0) |
| 7 | lev | ggml-org/lev-GGUF | llama.cpp | EN | 8192 | Qwen3.5-4B | 4.2B | apache-2.0 | 2.80 GB (Q4_K_M) |
| 8 | Bespoke Nimble 9B | ggml-org/Bespoke-Nimble-9B-v3-GGUF | llama.cpp | EN | 8192 | Qwen3.5-9B | 9.0B | **CC BY-NC 4.0** | 5.89 GB (Q4_K_M) |
| 9 | Clef Flash | ggml-org/Clef-Flash-GGUF | llama.cpp, reads images | EN | 8192 | Qwen3.5-9B | 9.1B | apache-2.0 | 6.62 GB (Q4_K_M) |
| 10 | Clef | ggml-org/Clef-GGUF | llama.cpp, reads images | EN | 8192 | Qwen3.8-27B (as read; small text) | 27B | apache-2.0 | 18.50 GB (Q4_K_M) |
| 11 | OpenJev | ggml-org/OpenJev-GGUF | llama.cpp, reads images | multilingual | 16384 | Qwen3.5-27B | 27B | **CC BY-NC 4.0** | 18.26 GB (Q4_K_M) |

## Per-model notes

1. **Kev 4B**: English decision model on Qwen3.5 4B with a "pointer head over the options". Validated on documents up to 8,192 tokens. Source: jaredpalmer/kev-4b.
2. **Laya Multilingual**: non-autoregressive "System 1" encoder for 100+ languages. Takes a state plus typed questions (yes/no, choice, score) and returns probabilities in one forward pass, no text generation. README table: 16 to 18 of 20 requests correct up to about 4,000 tokens, 8 to 17 of 20 beyond that. Install with `pip install laya`.
3. **Laya**: English version, up to 512 tokens. README claims about 33 ms and training with reinforcement learning against strictly proper scoring rules, so honest probabilities maximise reward. Optional extras: serve (HTTP), mcp (MCP server), langchain, onnx, fast (GPU).
4. **Laya Typed Decisions**: fine-tuned for agent-trace observability, customer service, invoice processing and security incidents. Its own benchmark (400 cases, 2,000 decisions): accuracy 0.766 against 0.727 for TypeSafe Jev's published figure. The same table shows it **worse** than Jev on soft accuracy (0.471 vs 0.580) and calibration error (ECE 0.213 vs 0.144), better on Brier (0.062 vs 0.148). Jev numbers are third-party, not measured by the authors.
5. **Julia 1**: small multilingual encoder, 2 to 20 options. Source: SupersonicLabs/Julia-1.
6. **Laya (GGUF)**: the English Laya converted for llama.cpp.
7. **lev**: English, reads each answer from the logits, calibrated probabilities over up to several hundred options. Source: interfaze-ai/lev.
8. **Bespoke Nimble 9B**: from Bespoke Labs, for choice and yes/no questions. Non-commercial weights.
9. **Clef Flash**: Cloudflare, decides all questions of a request together, reads images. README says it requires llama.cpp PR 29831.
10. **Clef**: Cloudflare's large version, needs about 20 GB of memory, reads images. Also requires PR 29831.
11. **OpenJev**: large model for text, web pages and screenshots, up to 52 options, needs about 20 GB of memory, non-commercial. Source: openjev/openjev.

## Cross-cutting

- Every llama.cpp model's README says it is used through the **`/v1/systemone` API** (llama.cpp PR 29818), the same endpoint shape as the hosted Codiv/OpenJev API. That would allow a local drop-in. I did not check the PR's status.
- Fits a 6 GB GPU (the RTX 4050 on this machine): numbers 2 to 7. Numbers 8 and 9 are borderline. Numbers 10 and 11 need about 20 GB of memory, so only workable on system RAM if there is enough, which I have not checked.
- Non-commercial licences: OpenJev and Bespoke Nimble 9B. Fine for learning and a portfolio, not for a commercial product.
- Only Kev 4B is downloaded.
