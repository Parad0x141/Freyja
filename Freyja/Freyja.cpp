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
    Sleep(3000);

    std::cout << "[MAIN] Initializing LLaMA backend...\n";
    Sleep(500);
    llama_backend_init();
    std::cout << "[MAIN] Backend initialized!\n\n";

    std::cout << "[MAIN] Creating session...\n";

    try
    {
        Session session(
            "C:\\Users\\Parad0x\\source\\repos\\Freyja\\x64\\Debug\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            35,      // GPU layers
            4096,    // Context size
            2048,    // Batch size
            "You are Freyja, a helpful and knowledgeable AI assistant."
        );

        std::cout << "\n==============================================\n";
        std::cout << "    FREYJA IS READY!\n";
        std::cout << "==============================================\n\n";


        std::string userInput;
        while (true)
        {
            std::cout << ">";
            std::getline(std::cin, userInput);

            if (userInput.empty())
                continue;
            if (userInput == "/exit" || userInput == "/quit")
            {
                std::cout << "Freyja : Goodbye !\n";
                break;
            }


            std::string response = session.Ask(userInput, 512);
            std::cout << "Freyja : " << response << "\n\n";
        }

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