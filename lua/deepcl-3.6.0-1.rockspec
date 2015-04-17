 package = "DeepCL"
 version = "3.6.0-1"
 source = {
    url = "https://github.com/hughperkins/DeepCL.git"
 }
 description = {
    summary = "DeepCL.",
    detailed = [[
       Convolutional network library using OpenCL
    ]],
    homepage = "https://github.com/hughperkins/DeepCL",
    license = "MPL"
 }
 dependencies = {
    "lua ~> 5.1"
 }
 build = {
    type = "builtin",
    modules = {
        luaDeepCL = {
            sources = {"DeepCL_wrap.cxx"},
            libraries = {"DeepCL", "OpenCLHelper"},
            incdirs = {"../src", "../OpenCLHelper", "../qlearning" },
            libdirs = {"../build" },
            flags = {"-std=c++0x"}
        }
    }
 }
