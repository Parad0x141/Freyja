#include "PromptComposer.hpp"

void PromptComposer::SanitizePrompt(std::string& p)
{
    // Si la string est vide, rien à faire
    if (p.empty()) return;

    const std::array<std::string, 6> forbid =
    {
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|im_start|>",
        "<|im_end|>"
    };

    for (const auto& s : forbid)
    {
        if (s.empty()) continue;  // Skip if empty or endless loop will occur

        size_t pos = 0;
        while ((pos = p.find(s, pos)) != std::string::npos)
        {
            p.erase(pos, s.length());
        }
    }

    // Trim leading whitespace
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    p.erase(p.begin(), std::find_if(p.begin(), p.end(), not_space));

    // Trim trailing whitespace
    p.erase(std::find_if(p.rbegin(), p.rend(), not_space).base(), p.end());
}