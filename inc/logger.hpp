#pragma once

#include <fstream>
#include <string>

class Logger {
	private:
		static std::ofstream file;
	public:
		static void init(std::string);
		static void log(std::string);
		static void stop();
};
