Compiling with pkg-config (in Linux)
cc -std=c99 -O2 elt_pupil_status.c eelt_pupil.c $(pkg-config --cflags libxml-2.0) $(pkg-config --libs libxml-2.0) -lcfitsio -lm -o elt_pupil_status

