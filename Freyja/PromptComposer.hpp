#pragma once

#include "Common.hpp"
#include <array>
#include <algorithm>
#include <cctype>

enum class PromptFormat
{
    ChatML,      // Qwen, Mistral, etc.
    Llama3       // Llama 3.x
};

class PromptComposer
{
public:
    explicit PromptComposer(std::string system = "", std::string user = "",
        PromptFormat format = PromptFormat::Llama3)
        : systemPrompt(std::move(system)), userPrompt(std::move(user)), format_(format)
    {
        SanitizePrompt(this->systemPrompt);
        SanitizePrompt(this->userPrompt);
    }

    PromptComposer& SetSystem(std::string s)
    {
        systemPrompt = std::move(s);
        SanitizePrompt(systemPrompt);
        return *this;
    }

    PromptComposer& SetUser(std::string u)
    {
        userPrompt = std::move(u);
        SanitizePrompt(userPrompt);
        return *this;
    }

    std::string Build() const
    {
        if (format_ == PromptFormat::Llama3)
            return BuildLlama3();
        else
            return BuildChatML();
    }



private:
    std::string systemPrompt;
    std::string userPrompt;
    PromptFormat format_;

    void SanitizePrompt(std::string& prompt);

    std::string BuildChatML() const
    {
        std::string out;
        if (!systemPrompt.empty()) 
        {
            out += "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n";
        }

        out += "<|im_start|>user\n" + userPrompt + "<|im_end|>\n";
        out += "<|im_start|>assistant\n";
        return out;
    }

    std::string BuildLlama3() const
    {
		std::string out; // No more begin_of_text, LLaMA 3 seems to handle that internally

        if (!systemPrompt.empty())
        {
            out += "<|start_header_id|>system<|end_header_id|>\n\n";
            out += systemPrompt + "<|eot_id|>";
        }

        out += "<|start_header_id|>user<|end_header_id|>\n\n";
        out += userPrompt + "<|eot_id|>";
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";

        return out;
    }
};