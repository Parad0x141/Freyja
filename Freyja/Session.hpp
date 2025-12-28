#pragma once
#include "Common.hpp"
#include "llama.h"
#include "PromptComposer.hpp"

// Full RAII session manager for LLaMA model, context and sampler.

class Session
{
public:
    explicit Session(const std::string& modelPath,
        int nGpuLayers,
        int nCtxSize,
        int nBatchSize,
        const std::string& systemPrompt = "",
        PromptFormat format = PromptFormat::Llama3);

    std::string Ask(const std::string& userPrompt, int maxNew = 512);
    PromptComposer& GetComposer() { return Composer; }

    ~Session();

private:
    llama_model* model = nullptr;
    llama_context* context = nullptr;
    llama_sampler* sampler = nullptr;
    PromptComposer Composer;

    int currentPos_ = 0;
    int nCtxSize_ = 0; // Stored for sliding window management
};