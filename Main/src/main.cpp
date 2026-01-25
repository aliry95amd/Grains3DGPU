/* ========================================================================== */
/*                Discrete Element Method Using NVIDIA CUDA                   */
/*                      Alireza Yazdani, July 2023                            */
/* ========================================================================== */
#include <cuda.h>
#include <cuda_runtime.h>

#include "Grains.hh"
#include "GrainsFactory.hh"
#include "ReaderXML.hh"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <unistd.h>

using namespace std;

/* ============================================================================================== */
/*                                              Main                                              */
/* ============================================================================================== */
/* Main Function */
int main(int argc, char* argv[])
{
    // Temp file handling: store path in a plain char buffer so signal handlers can safely unlink.
    static char              g_tempfile_buf[4096];
    static std::atomic<bool> g_tempfile_set(false);

    auto set_tempfile = [&](const std::string& s) {
        std::strncpy(g_tempfile_buf, s.c_str(), sizeof(g_tempfile_buf) - 1);
        g_tempfile_buf[sizeof(g_tempfile_buf) - 1] = '\0';
        g_tempfile_set.store(true);
    };

    auto cleanup_tempfile = []() {
        if(g_tempfile_set.load())
        {
            ::unlink(g_tempfile_buf);
        }
    };

    auto signal_handler = [](int sig) {
        if(g_tempfile_set.load())
            ::unlink(g_tempfile_buf);
        std::_Exit(128 + sig);
    };

    // Register cleanup and signal handlers
    std::atexit(cleanup_tempfile);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGABRT, signal_handler);
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGFPE, signal_handler);

    // Input file
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input.xml>" << std::endl;
        return 1;
    }
    string filename = argv[1], filename_exe;
    size_t error    = 0;
    size_t pos      = filename.find(".xml");
    if(pos == string::npos)
    {
        cout << "ERROR: input file needs the .xml extension" << endl;
        error = true;
    }

    // Execute the Grains application
    if(!error)
    {
        // Create a temporary input file with the proper XML header
        // We use the double version of GrainsFactory, but it doesn't
        // matter.
        filename_exe = GrainsFactory<double>::init(filename);
        // record tempfile for cleanup handlers
        set_tempfile(filename_exe);

        // Creates the Grains application
        ReaderXML::initialize();
        DOMElement* rootNode = ReaderXML::getRoot(filename_exe);
        string      prc      = ReaderXML::getNodeAttr_String(rootNode, "Precision");
        if(prc == "Single")
        {
            Grains<float>* grains = nullptr;
            grains                = GrainsFactory<float>::create(rootNode);

            // Tasks to perform before time-stepping
            grains->initialize(rootNode);
            rootNode->getOwnerDocument()->release();
            ReaderXML::terminate();

            // Delete the temporary input file
            std::remove(filename_exe.c_str());

            // Run the simulation
            grains->simulate();

            // Tasks to perform after time-stepping
            grains->finalize();

            // Delete the Grains application
            delete grains;
        }
        else
        {
            if(prc != "Double")
            {
                std::cout << "Invalid precision! Creating Grains "
                             "in double "
                          << "precision!" << endl;
            }

            Grains<double>* grains = nullptr;
            grains                 = GrainsFactory<double>::create(rootNode);

            // Tasks to perform before time-stepping
            grains->initialize(rootNode);
            rootNode->getOwnerDocument()->release();
            ReaderXML::terminate();

            // Delete the temporary input file
            std::remove(filename_exe.c_str());

            // Run the simulation
            grains->simulate();

            // Tasks to perform after time-stepping
            grains->finalize();

            // Delete the Grains application
            delete grains;
        }
    }

    return (0);
}