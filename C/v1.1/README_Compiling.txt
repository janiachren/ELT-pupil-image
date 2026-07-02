## Compiling

gcc -I/usr/local/include \
    -fPIC -shared micado_elt_pupil_status.c \
    -o micado_elt_pupil_status.so \
    $(pkg-config --cflags --libs "cpl libxml-2.0") \
    -lm


#Alt
gcc -std=c99 -fPIC \
    -I$CPL_PREFIX/include \
    -I/usr/include/libxml2 \
    -I/opt/cpl-7.3.2/include \
    -c micado_elt_pupil_status.c


gcc -shared -o micado_elt_pupil_status.so micado_elt_pupil_status.o \
    -L$CPL_PREFIX/lib -lcplcore -lcplui -lcpldrs \
    -lxml2 -lcfitsio -leelt_pupil


## Running via esorex
esorex elt_pupil StatusM1.sof
