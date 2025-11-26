#pragma once

#include <string>
#include <iostream>
#include <source_location>

namespace gf::basic
{
    class Logger
    {
        private:
            std::string     _name;
            std::ostream&   _os;

            enum struct Level
            {
                INFO,
                DEBUG,
                WARNING,
                ERROR
            };

            void _log(Level lvl, std::string msg, const std::source_location& loc);

        public:
            Logger(std::string name, std::ostream& os = std::cout) : _name(std::move(name)), _os(os) {}

            inline void logInfo(std::string msg, const std::source_location& loc) { _log(Level::INFO, std::move(msg), loc); }

            inline void logDebug(std::string msg, const std::source_location& loc) { _log(Level::DEBUG, std::move(msg), loc); }

            inline void logWarning(std::string msg, const std::source_location& loc) { _log(Level::WARNING, std::move(msg), loc); }

            inline void logError(std::string msg, const std::source_location& loc) { _log(Level::ERROR, std::move(msg), loc); }
    };
}

#define LOG_INFO(logger, msg)       logger.logInfo(msg, std::source_location::current())
#define LOG_DEBUG(logger, msg)      logger.logDebug(msg, std::source_location::current())
#define LOG_WARNING(logger, msg)    logger.logWarning(msg, std::source_location::current())
#define LOG_ERROR(logger, msg)      logger.logError(msg, std::source_location::current())