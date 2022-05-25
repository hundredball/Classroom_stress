# Sampling frequency
fs = 1000

# Window size for Fourier Transform
# Our interest frequency is 3 Hz, 2 cycles is 2/3 seconds. 
win = int((2/3)*fs)
# win = 2*fs