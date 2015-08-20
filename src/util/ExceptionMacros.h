#pragma once

#define THROW(message) throw std::runtime_error(std::string(__FILE__) + " " + toString(__LINE__) + ": " + message);

