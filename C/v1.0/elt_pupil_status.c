#define _GNU_SOURCE
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

#include "eelt_pupil.h"

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <fitsio.h>


/* ======= PARAMETERS (matching Python defaults) ======= */
static const int GRID_SIZE = 822;
static const int NUM_SEGMENTS = 798;
static const double EELT_DIAM = 40.0;          /* Use 40.0 to match Python default (comment notes 38.542) */
static const double PIXSCALE = 40.0 / 822.0 * 1.0275;
static const double SPIDER_WIDTH = 0.54;       /* from ESO-532755 */
static const double GAP = 0.0;                 /* hard gap; softGap is false */

/* Reflectivity model */
static const double MAX_REFLECTIVITY = 0.96;
static const double MIN_REFLECTIVITY = 0.91;
static const double COATING_DEGRADATION_PER_DAY = 0.000125;

/* ======= Utilities ======= */

/* Trim leading/trailing whitespace including UTF-8 BOM remnants */
static char* trim(char* s) {
    if (!s) return s;
    while (*s && isspace((unsigned char)*s)) s++;
    if (s[0] == '\xEF' && s[1] == '\xBB' && s[2] == '\xBF') s += 3; /* UTF-8 BOM */
    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) len--;
    s[len] = '\0';
    return s;
}

/* Parse YYYY-MM-DD into struct tm */
static int parse_date(const char* iso, struct tm* out) {
    if (!iso || !out) return 0;
    memset(out, 0, sizeof(*out));
    int y, m, d;
    if (sscanf(iso, "%d-%d-%d", &y, &m, &d) != 3) return 0;
    out->tm_year = y - 1900;
    out->tm_mon = m - 1;
    out->tm_mday = d;
    out->tm_hour = 0; out->tm_min = 0; out->tm_sec = 0;
    out->tm_isdst = -1;
    return 1;
}

/* Days since given date (local time baseline, close enough for daily degradation) */
static int days_since(const struct tm* then) {
    time_t now_t = time(NULL);
    struct tm now_tm;
#if defined(_WIN32)
    localtime_s(&now_tm, &now_t);
#else
    localtime_r(&now_t, &now_tm);
#endif
    time_t then_t = mktime((struct tm*)then);
    if (then_t == (time_t)-1) return 0;
    double diff = difftime(now_t, then_t);
    return (int)(diff / 86400.0);
}

/* Current UTC timestamp: YYYY-MM-DDTHH:MM:SS */
static void utc_timestamp(char* buf, size_t buflen) {
    time_t now = time(NULL);
    struct tm gmt;
#if defined(_WIN32)
    gmtime_s(&gmt, &now);
#else
    gmtime_r(&now, &gmt);
#endif
    strftime(buf, buflen, "%Y-%m-%dT%H_%M_%S", &gmt);
}

/* Extract the first <segments>...</segments> subtree from an XML document node */
static xmlNode* find_segments_node(xmlNode* root) {
    for (xmlNode* cur = root; cur; cur = cur->next) {
        if (cur->type == XML_ELEMENT_NODE && xmlStrcmp(cur->name, (const xmlChar*)"segments") == 0) {
            return cur;
        }
        xmlNode* child = cur->children;
        xmlNode* found = find_segments_node(child);
        if (found) return found;
    }
    return NULL;
}

/* ======= XML loader ======= */
/*
 * Load segment reflectivities from a file that contains a <segments> block.
 * Each <segment id="N" operational="true|false" last_recoating="YYYY-MM-DD" />
 */
static int load_segments_from_file(const char* filename, double* F1, int n_segments) {
    if (!filename || !F1) return 0;

    /* Read entire file into memory (to tolerate extra text) */
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", filename);
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    if (!buf) {
        fclose(f);
        fprintf(stderr, "ERROR: Out of memory reading file.\n");
        return 0;
    }
    size_t nread = fread(buf, 1, sz, f);
    if (nread != sz) {
        fprintf(stderr, "WARNING: fread read only %zu of %ld bytes\n", nread, sz);
    }
    buf[sz] = '\0';
    fclose(f);

    /* Find the <segments>...</segments> block by naive substring search */
    const char* start = strstr(buf, "<segments");
    if (!start) {
        free(buf);
        fprintf(stderr, "ERROR: No <segments> block found in file.\n");
        return 0;
    }
    const char* end = strstr(start, "</segments>");
    if (!end) {
        free(buf);
        fprintf(stderr, "ERROR: Unterminated <segments> block.\n");
        return 0;
    }
    end += strlen("</segments>");

    size_t block_len = (size_t)(end - start);
    char* xml_block = (char*)malloc(block_len + 1);
    if (!xml_block) {
        free(buf);
        fprintf(stderr, "ERROR: Out of memory extracting XML block.\n");
        return 0;
    }
    memcpy(xml_block, start, block_len);
    xml_block[block_len] = '\0';

    /* Parse the extracted XML block */
    xmlDoc* doc = xmlReadMemory(xml_block, (int)block_len, "segments.xml", NULL, XML_PARSE_NOBLANKS);
    free(xml_block);
    free(buf);

    if (!doc) {
        fprintf(stderr, "ERROR: libxml2 failed to parse segments block.\n");
        return 0;
    }

    xmlNode* root = xmlDocGetRootElement(doc);
    xmlNode* segments = NULL;

    /* The top might already be <segments>, but we also search recursively */
    if (root && root->type == XML_ELEMENT_NODE && xmlStrcmp(root->name, (const xmlChar*)"segments") == 0) {
        segments = root;
    } else {
        segments = find_segments_node(root);
    }

    if (!segments) {
        fprintf(stderr, "ERROR: <segments> element not found in parsed document.\n");
        xmlFreeDoc(doc);
        return 0;
    }

    /* Initialize F1 to zero */
    for (int i = 0; i < n_segments; ++i) F1[i] = 0.0;

    /* Compute reflectivities */
    xmlNode* seg = NULL;
    for (seg = segments->children; seg; seg = seg->next) {
        if (seg->type != XML_ELEMENT_NODE) continue;
        if (xmlStrcmp(seg->name, (const xmlChar*)"segment") != 0) continue;

        xmlChar* id_attr = xmlGetProp(seg, (const xmlChar*)"id");
        xmlChar* operational_attr = xmlGetProp(seg, (const xmlChar*)"operational");
        xmlChar* last_attr = xmlGetProp(seg, (const xmlChar*)"last_recoating");

        if (!id_attr || !operational_attr || !last_attr) {
            if (id_attr) xmlFree(id_attr);
            if (operational_attr) xmlFree(operational_attr);
            if (last_attr) xmlFree(last_attr);
            fprintf(stderr, "WARNING: segment missing attributes; skipping.\n");
            continue;
        }

        char* id_str = trim((char*)id_attr);
        int seg_id = atoi(id_str) - 1; /* XML is 1-based; convert to 0-based */
        if (seg_id < 0 || seg_id >= n_segments) {
            fprintf(stderr, "WARNING: segment id out of range: %d\n", seg_id + 1);
            xmlFree(id_attr); xmlFree(operational_attr); xmlFree(last_attr);
            continue;
        }

        char* operational_str = trim((char*)operational_attr);
        int operational = 0;
        /* Normalize to lower-case */
        for (char* p = operational_str; *p; ++p) *p = (char)tolower((unsigned char)*p);
        if (strcmp(operational_str, "true") == 0) operational = 1;

        char* last_str = trim((char*)last_attr);
        struct tm recoating_tm;
        double reflectivity = 0.0;

        if (operational) {
            if (parse_date(last_str, &recoating_tm)) {
                int days = days_since(&recoating_tm);
                reflectivity = MAX_REFLECTIVITY - COATING_DEGRADATION_PER_DAY * (double)days;
                if (reflectivity < MIN_REFLECTIVITY) reflectivity = MIN_REFLECTIVITY;
            } else {
                /* If parse fails, conservatively use min reflectivity */
                reflectivity = MIN_REFLECTIVITY;
            }
        } else {
            reflectivity = 0.0;
        }

        F1[seg_id] = reflectivity;

        xmlFree(id_attr);
        xmlFree(operational_attr);
        xmlFree(last_attr);
    }

    xmlFreeDoc(doc);
    return 1;
}


/* ======= MJD calculator ======= */
static double compute_mjd_utc() {
    time_t now = time(NULL);
    struct tm utc;
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif

    int Y = utc.tm_year + 1900;
    int M = utc.tm_mon + 1;
    int D = utc.tm_mday;
    int h = utc.tm_hour;
    int m = utc.tm_min;
    int s = utc.tm_sec;

    if (M <= 2) {
        Y -= 1;
        M += 12;
    }

    int A = Y / 100;
    int B = 2 - A + (A / 4);

    double JD = (int)(365.25 * (Y + 4716)) + (int)(30.6001 * (M + 1)) + D + B - 1524.5;
    double frac_day = (h + m / 60.0 + s / 3600.0) / 24.0;

    return JD + frac_day - 2400000.5;  // Convert JD to MJD
}


/* ======= FITS writer via CFITSIO ======= */
static int write_fits(const char* filename, const double* data, int width, int height, const char* date_obs) {
    fitsfile* fptr = NULL;
    int status = 0;

    long naxes[2] = { (long)width, (long)height };
    if (fits_create_file(&fptr, filename, &status)) {
        fits_report_error(stderr, status);
        return 0;
    }

    if (fits_create_img(fptr, DOUBLE_IMG, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return 0;
    }

    /* CFITSIO expects row-major array; write entire image */
    long fpixel[2] = {1, 1};
    long nelements = naxes[0] * naxes[1];
    if (fits_write_pix(fptr, TDOUBLE, fpixel, nelements, (void*)data, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return 0;
    }

    /* ===== ESO HEADER BLOCK ===== */

    /* DATE-OBS */
    if (date_obs && *date_obs) {
        fits_update_key(fptr, TSTRING, "DATE-OBS", (void*)date_obs,
                        "UTC date of observation", &status);
    }

    /* ORIGIN */
    fits_update_key(fptr, TSTRING, "ORIGIN",
                    "ESO ELT PUPIL STATUS GENERATOR",
                    "File origin", &status);

    /* COMMENT block */
    fits_write_comment(fptr,
        "Mirror segment reflectivity and operational status snapshot", &status);
    fits_write_comment(fptr,
        "Generated from StatusM1segments.xml", &status);

    /* MJD-OBS */
    double mjd = compute_mjd_utc();
    fits_update_key(fptr, TDOUBLE, "MJD-OBS", &mjd,
                    "Modified Julian Date of observation", &status);

    /* ESO mandatory keywords */
    fits_update_key(fptr, TSTRING, "INSTRUME", "MICADO",
                    "Instrument name", &status);
    fits_update_key(fptr, TSTRING, "ESO DPR CATG", "CALIB",
                    "Data category", &status);
    fits_update_key(fptr, TSTRING, "ESO DPR TYPE", "PUPIL",
                    "Data type", &status);
    fits_update_key(fptr, TSTRING, "ESO DPR TECH", "IMAGE",
                    "Data technique", &status);

    /* Close FITS */
    if (fits_close_file(fptr, &status)) {
        fits_report_error(stderr, status);
        return 0;
    }
    return 1;
}

/* CFITSIO desires file names like "!filename.fits" to overwrite */
static void build_fits_path(char* out, size_t outlen, const char* timestamp) {
    /* ESO naming convention */
    snprintf(out, outlen,
             "c.ELT.%s.pupil.segmentstatus.fits",
             timestamp);
}





/* ======= Main ======= */
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s StatusM1segments.xml\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* xml_file = argv[1];

    double* F1 = (double*)malloc(NUM_SEGMENTS * sizeof(double));
    if (!F1) {
        fprintf(stderr, "ERROR: Out of memory for F1.\n");
        return EXIT_FAILURE;
    }

    if (!load_segments_from_file(xml_file, F1, NUM_SEGMENTS)) {
        free(F1);
        return EXIT_FAILURE;
    }

    double* pupil_mask = generateEeltPupilReflectivity(
        F1, NUM_SEGMENTS, GRID_SIZE,
        SPIDER_WIDTH,
        GRID_SIZE / 2.0 - 0.5, GRID_SIZE / 2.0 - 0.5,
        PIXSCALE,
        GAP,
        0.0,              /* rotation degrees */
        EELT_DIAM,
        0                 /* softGap false */
        );
    
    free(F1);

    if (!pupil_mask) {
        fprintf(stderr, "ERROR: Failed to generate pupil mask (native).\n");
        return EXIT_FAILURE;
    }
    
    /* Build timestamp and FITS filename */
    char timestamp[32];
    utc_timestamp(timestamp, sizeof(timestamp));

    char fits_path[256];
    build_fits_path(fits_path, sizeof(fits_path), timestamp);

    /* Write FITS */
    int ok = write_fits(fits_path, pupil_mask, GRID_SIZE, GRID_SIZE, timestamp);
    free(pupil_mask);

///    if (Py_IsInitialized()) Py_Finalize();

    if (!ok) {
        fprintf(stderr, "ERROR: Failed to write FITS file.\n");
        return EXIT_FAILURE;
    }

    printf("Saved FITS file: %s\n", fits_path + 1); /* skip leading '!' in printout */
    return EXIT_SUCCESS;
}
