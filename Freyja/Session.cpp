#include "Init.hpp"
#include "Session.hpp"
#include "Common.hpp"
#include "PromptComposer.hpp"

Session::Session(const std::string& modelPath,
    int nGpuLayers,
    int nCtxSize,
    int nBatchSize,
    const std::string& systemPrompt,
    PromptFormat format)
    : Composer(systemPrompt, "", format), nCtxSize_(nCtxSize)
{
    


    std::cout << "[SESSION] Loading model...\n";
    model = LoadModel(modelPath, nGpuLayers);
    if (!model)
        throw std::runtime_error("Model is null after initialization.");

    std::cout << "[SESSION] Creating context...\n";
    context = CreateContext(model, nCtxSize, nBatchSize);
    if (!context)
        throw std::runtime_error("Context is null after initialization.");

    std::cout << "[SESSION] Creating sampler...\n";
    sampler = CreateSampler(context);
    if (!sampler)
        throw std::runtime_error("Sampler is null after initialization.");

}

std::string Session::Ask(const std::string& userPrompt, int maxNew)
{
    std::cout << "\n[ASK] ========================================\n";
    std::cout << "[ASK] Starting Ask()\n";

    const llama_vocab* vocab = llama_model_get_vocab(model);
    Composer.SetUser(userPrompt);
    const std::string sanitizedPrompt = Composer.Build();

    std::cout << "[ASK] Prompt length: " << sanitizedPrompt.length() << " chars\n";
    std::cout << "[ASK] Prompt:\n" << sanitizedPrompt << "\n";

    int32_t text_len = static_cast<int32_t>(sanitizedPrompt.length());

    /*   tokenize */
    std::cout << "[ASK] Tokenizing (pass 1)...\n";
    llama_token buf[4096];
    int n = llama_tokenize(vocab,
        sanitizedPrompt.c_str(),
        text_len,
        buf, 4096,
        true,   // add_special
        true);  // parse_special

    if (n < 0) 
    {
        std::cout << "[ASK] First pass failed (" << n << "), retrying without special parsing...\n";
        n = llama_tokenize(vocab,
            sanitizedPrompt.c_str(),
            text_len,
            buf, 4096,
            true,   // add_special
            false); // parse_special
    }
    if (n < 0)
        throw std::runtime_error("[ASK] Tokenization failed with code: " + std::to_string(n));
    if (n == 0 || n > nCtxSize_)
        throw std::runtime_error("[ASK] Invalid token count: " + std::to_string(n));

    std::cout << "[ASK] Tokens: " << n << "\n";
    std::vector<llama_token> tokens(buf, buf + n);

    /*   clear memory for consecutive positions */
    llama_memory_t mem = llama_get_memory(context);
    std::cout << "[ASK] Clearing memory for consecutive positions...\n";
	llama_memory_clear(mem, true); // BAD, stateless that way, but easier for now

    /*  prompt batch */
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n - 1);
    }
    batch.n_tokens = n;

    std::cout << "[ASK] Decoding prompt batch...\n";
    if (llama_decode(context, batch) != 0)
    {
        llama_batch_free(batch);
        throw std::runtime_error("[ASK] Prompt decode failed");
    }
    llama_batch_free(batch);

    /*  generation loop */
    if (!llama_memory_can_shift(mem))
        throw std::runtime_error("[ASK] Memory doesn't support shifting");

    llama_batch single = llama_batch_init(1, 0, 1);
    single.n_seq_id[0] = 1;
    single.seq_id[0][0] = 0;
    single.logits[0] = true;
    single.n_tokens = 1;

    std::string out;
    out.reserve(maxNew * 4);
    int currentPos = n;

    std::cout << "[ASK] Starting token generation...\n";
    for (int i = 0; i < maxNew; ++i)
    {
        if (i && i % 20 == 0)
            std::cout << "[ASK] Generated " << i << " tokens...\n";

        /* sliding window */
        if (currentPos >= nCtxSize_)
        {
            int keep = nCtxSize_ / 2;
            llama_memory_seq_rm(mem, 0, 0, nCtxSize_ - keep);
            llama_memory_seq_add(mem, 0, 0, -1, -(nCtxSize_ - keep));
            currentPos = keep;
        }

        /* sample */
        llama_token id = llama_sampler_sample(sampler, context, -1);
        if (id == llama_vocab_eos(vocab)) break;

        /* token to text */
        char piece[256];
        int len = llama_token_to_piece(vocab, id, piece, sizeof(piece), 0, false);
        out.append(piece, len);

        /* decode single token */
        single.token[0] = id;
        single.pos[0] = currentPos++;
        if (llama_decode(context, single) != 0)
        {
            llama_batch_free(single);
            throw std::runtime_error("[ASK] Token decode failed");
        }
    }

    llama_batch_free(single);
    std::cout << "[ASK] Generation complete! (" << out.size() << " chars)\n";
    return out;
}

Session::~Session()
{
    std::cout << "[SESSION] Destructor called\n";

    if (sampler) 
    {
        llama_sampler_free(sampler);
        sampler = nullptr;
    }
    if (context) 
    {
        llama_free(context);
        context = nullptr;
    }
    if (model)
    {
        llama_model_free(model);
        model = nullptr;
    }

    // Backend free is done in main() for now.

    std::cout << "[SESSION] Cleanup complete\n";
}