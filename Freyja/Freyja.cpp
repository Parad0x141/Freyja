// Code by Cyril "Parad0x141" Bouvier - 2025
#define PROJECT_CODENAME FREYJA

#include <iostream>
#include <string>
#include <windows.h>
#include <io.h>
#include <fcntl.h>

#include "Session.hpp"

int main()
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    std::cout << "==============================================\n";
    std::cout << "                FREYJA AI \n";
    std::cout << "==============================================\n\n";

    std::cout << "[MAIN] Waking up Freyja...\n";

    std::cout << "[MAIN] Initializing LLaMA backend...\n";
    llama_backend_init();
    std::cout << "[MAIN] Backend initialized!\n\n";

    std::cout << "[MAIN] Creating session...\n";

    try
    {
        Session session(
            "C:\\Users\\Parad0x\\source\\repos\\Freyja\\x64\\Debug\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            35,      // GPU layers
            4096,    // Context size
            512,     // Batch size
            "You are Freyja, a helpful and knowledgeable AI assistant."
        );

        std::cout << "\n==============================================\n";
        std::cout << "    FREYJA IS READY!\n";
        std::cout << "==============================================\n\n";

        // Test 1
        std::string response = session.Ask("Insert user prompt here", 256);
        std::cout << "\nFreyja: " << response << "\n\n";


    }
    catch (const std::exception& e) 
    {
        std::cerr << "\n[MAIN] ERROR: " << e.what() << std::endl;
        llama_backend_free();
        return 1;
    }

    std::cout << "[MAIN] Shutting down Freyja...\n";
    llama_backend_free();
    std::cout << "[MAIN] Goodbye!\n";

    return 0;
}