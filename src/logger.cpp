#include <iostream>
#include <ctime>

#include "logger.hpp"

std::ofstream Logger::file;

void Logger::init(std::string file_path) {
	if (file_path == "")
		file_path = "cr.log"; // Default value
	file.open(file_path, std::ios::out | std::ios::app);
}

void Logger::log(std::string message) {
	std::time_t time = std::time(nullptr);
	char timestr[9];

	std::string msg;
	if (std::strftime(timestr, sizeof(timestr), "%H:%M:%S", std::localtime(&time))) {
		msg += "[";
		msg += timestr;
		msg += "] ";
	}
	msg += message;

	std::cout << msg << '\n';

	if (file.is_open())
		file << msg << '\n';
}

void Logger::stop() {
	file.close();
}
