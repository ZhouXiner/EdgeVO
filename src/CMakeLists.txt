add_library(EdgeVO_LIBS SHARED
        Inputs.cpp
        Frame.cpp System.cpp Camera.cpp EdgePixel.cpp Track.cpp  Viewer.cpp BackEnd.cpp)

target_link_libraries(EdgeVO_LIBS ${THIRD_LIBS})