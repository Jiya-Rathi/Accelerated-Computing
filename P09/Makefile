CXX = g++
CXXFLAGS = -O0 -g3 -Wall -I/usr/local/include -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lfftw3 -lcufft -lcudart -lcublas

OBJS = FFTW_P09.o WavFile.o
TARGET = FFTW_P09

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

FFTW_P09.o: FFTW_P09.cpp WavFile.h
	$(CXX) $(CXXFLAGS) -c FFTW_P09.cpp

WavFile.o: WavFile.cpp WavFile.h
	$(CXX) $(CXXFLAGS) -c WavFile.cpp

clean:
	rm -f $(OBJS) $(TARGET)
