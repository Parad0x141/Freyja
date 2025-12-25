#pragma once

// Welp, LLaMA init does not return anything and does not throw exceptions,
// let's wrap it in a try catch to be safe and make a proper init.

#include "Common.hpp"
#include "llama.h"

/*
inline static bool Init()
{
    try
    {
       // std::cout << "[INIT] Initializing LLaMA backend...\n";
        //llama_backend_init();

        std::cout << "[INIT] Backend initialized!\n";
        const char* version = llama_print_system_info();
        std::cout << version << std::endl;

        std::cout << "[INIT] Init complete!\n";
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[INIT] Error: " << e.what() << std::endl;
        return false;
    }
}*/

inline llama_model* LoadModel(const std::string& modelPath, int nGpuLayers = 35)
{
    try
    {
        std::cout << "[LOAD] Loading model from: " << modelPath << "\n";
        std::cout << "[LOAD] GPU layers: " << nGpuLayers << "\n";

        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = nGpuLayers;

        llama_model* model = llama_model_load_from_file(modelPath.c_str(), params);
        if (!model)
        {
            std::cerr << "[LOAD] Error: cannot load model!" << std::endl;
            return nullptr;
        }

        std::cout << "[LOAD] Model loaded successfully!\n";
        return model;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[LOAD] Error: " << e.what() << std::endl;
        return nullptr;
    }
}

inline llama_context* CreateContext(llama_model* model, int nContext, int nBatch)
{
    try
    {
        std::cout << "[CTX] Creating context (n_ctx=" << nContext << ", n_batch=" << nBatch << ")...\n";

        llama_context_params ctxParams = llama_context_default_params();
        ctxParams.n_ctx = nContext;
        ctxParams.n_batch = nBatch;

        llama_context* context = llama_init_from_model(model, ctxParams);
        if (!context)
        {
            std::cerr << "[CTX] Error: cannot create context!" << std::endl;
            return nullptr;
        }

        std::cout << "[CTX] Context created successfully!\n";
        return context;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[CTX] Error: " << e.what() << std::endl;
        return nullptr;
    }
}

inline llama_sampler* CreateSampler(llama_context* context)
{
    std::cout << "[SAMPLER] Creating sampler...\n";

    llama_sampler* sampler = llama_sampler_init_greedy();
    if (!sampler)
    {
        std::cerr << "[SAMPLER] Error: cannot create sampler!" << std::endl;
        return nullptr;
    }

    std::cout << "[SAMPLER] Sampler created successfully!\n";
    return sampler;
}