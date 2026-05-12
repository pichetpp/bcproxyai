import { NextRequest } from "next/server";
import { ensureApiKeysLoaded, getNextApiKey } from "@/lib/api-keys";
import { resolveProviderEmbeddingUrl } from "@/lib/provider-resolver";
import { openAIError } from "@/lib/openai-compat";
import { getCostAllowedProviders, isProviderCostAllowed } from "@/lib/cost-policy";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

// Default embedding models per provider — multilingual-capable preferred for Thai support
const DEFAULT_EMBEDDING_MODELS: Record<string, string> = {
  mistral: "mistral-embed",
  cohere: "embed-multilingual-v3.0",
  google: "gemini-embedding-001",
  nvidia: "nvidia/nv-embedqa-e5-v5",
  together: "togethercomputer/m2-bert-80M-32k-retrieval",
  huggingface: "intfloat/multilingual-e5-large",
  openrouter: "openai/text-embedding-3-small",
  ollama: "nomic-embed-text",
};

// Order optimized for Thai/multilingual support + free-tier reliability
const EMBED_PROVIDER_ORDER = [
  "mistral", "cohere", "google", "nvidia", "together", "huggingface", "openrouter", "ollama",
];

/**
 * POST /v1/embeddings — Embedding generation
 * Used by: Continue (codebase indexing), Cody, LangChain, Aider, LibreChat
 */
export async function POST(req: NextRequest) {
  try {
    let body: Record<string, unknown>;
    try {
      body = (await req.json()) as Record<string, unknown>;
    } catch {
      return openAIError(400, { message: "Invalid JSON in request body", code: "invalid_request" });
    }

    if (!body.input) {
      return openAIError(400, { message: "input is required", param: "input" });
    }

    // Load API keys from DB cache — required before getNextApiKey()
    await ensureApiKeysLoaded();

    const requestedModel = (body.model as string) || "auto";
    const isAuto = requestedModel === "auto" || requestedModel === "sml/auto";

    // Try each embedding provider in order
    const providerOrder = EMBED_PROVIDER_ORDER.filter(isProviderCostAllowed);
    const triedReasons: string[] = [];

    for (const provider of providerOrder) {
      const url = resolveProviderEmbeddingUrl(provider);
      if (!url) { triedReasons.push(`${provider}:no_url`); continue; }

      const apiKey = getNextApiKey(provider);
      if (!apiKey && provider !== "ollama") { triedReasons.push(`${provider}:no_key`); continue; }

      const embeddingModel = isAuto
        ? DEFAULT_EMBEDDING_MODELS[provider]
        : requestedModel;
      if (!embeddingModel) { triedReasons.push(`${provider}:no_default_model`); continue; }
      // Note: embedding models are not in FREE_MODEL_CATALOG (chat-only catalog).
      // Provider-level allowlist (isProviderCostAllowed) is sufficient cost gate
      // because embedding pricing is ~100x cheaper than chat and we control the
      // EMBED_PROVIDER_ORDER + DEFAULT_EMBEDDING_MODELS allowlist statically.

      try {
        const headers: Record<string, string> = {
          "Content-Type": "application/json",
        };
        if (apiKey) {
          headers["Authorization"] = `Bearer ${apiKey}`;
        }
        if (provider === "openrouter") {
          headers["HTTP-Referer"] = "https://smlgateway.ai";
          headers["X-Title"] = "SMLGateway Gateway";
        }

        // Google AI Studio uses ?key= query param (also accepts Bearer)
        // — leave Authorization header in place; both work for OpenAI-compat path

        const response = await fetch(url, {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: embeddingModel,
            input: body.input,
            encoding_format: body.encoding_format ?? "float",
          }),
        });

        if (response.ok) {
          const json = await response.json();

          // Ensure standard format
          json.object = "list";
          if (Array.isArray(json.data)) {
            for (let i = 0; i < json.data.length; i++) {
              json.data[i].object = "embedding";
              json.data[i].index = json.data[i].index ?? i;
            }
          }
          json.model = json.model ?? embeddingModel;
          json.usage = json.usage ?? { prompt_tokens: 0, total_tokens: 0 };

          const respHeaders = new Headers();
          respHeaders.set("Content-Type", "application/json");
          respHeaders.set("X-SMLGateway-Provider", provider);
          respHeaders.set("X-SMLGateway-Model", embeddingModel ?? "");
          respHeaders.set("Access-Control-Allow-Origin", "*");

          return new Response(JSON.stringify(json), { status: 200, headers: respHeaders });
        }

        // Capture upstream error preview to make 503 actionable
        const errBody = await response.text().catch(() => "");
        const preview = errBody.slice(0, 120).replace(/\s+/g, " ");
        triedReasons.push(`${provider}:${response.status}(${preview})`);
        console.warn(`[embeddings] ${provider}/${embeddingModel} -> ${response.status}: ${preview}`);
        continue;
      } catch (err) {
        const m = err instanceof Error ? err.message : String(err);
        triedReasons.push(`${provider}:throw(${m.slice(0, 80)})`);
        console.warn(`[embeddings] ${provider} threw:`, m);
        continue;
      }
    }

    return openAIError(503, {
      message: `All embedding providers failed. Tried: ${triedReasons.join(" | ")}. Allowed providers: ${getCostAllowedProviders().join(", ")}.`,
    });
  } catch (err) {
    console.error("[embeddings] error:", err);
    return openAIError(500, { message: String(err) });
  }
}

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    },
  });
}
