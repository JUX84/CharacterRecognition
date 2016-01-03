#include <iostream>
#include <ctime>

#include "logger.hpp"

std::ofstream Logger::file;

void Logger::init(std::string file_path) {
	if (file_path == "")
		file_path = "cr.log"; //default value
	file.open(file_path);
}

void Logger::log(std::string message) {
	std::time_t time = std::time(nullptr);
	char timestr[9];
	if (std::strftime(timestr, sizeof(timestr), "%H:%M:%S", std::localtime(&time)))
		std::cout << "[" << timestr << "] ";
	std::cout << message << '\n';
	if (file.is_open())
		file << message << '\n';
}

void Logger::stop() {
	file.close();
}
