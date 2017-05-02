class Log {
public:
    int level;  // 0 error 1 warning 2 info 3 debug
    void debug(string message);
    void info(string message);
    void warning(string message);
    void error(string message);
};

