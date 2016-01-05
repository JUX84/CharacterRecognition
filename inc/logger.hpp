#pragma once

#include <fstream>
#include <string>

/* Logging class */
class Logger {
	private:
		static std::ofstream file;
	public:
		static void init(std::string); /* if we want to log to file */
		static void log(std::string); /* logs to stdout (and file if opened) */
		static void stop(); /* closes file (if opened) */
};
