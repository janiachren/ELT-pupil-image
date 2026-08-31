#define _GNU_SOURCE
#define PACKAGE_NAME "micado_elt_pupil_status"

#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <limits.h>

#include <libxml/parser.h>
#include <libxml/tree.h>
#include <fitsio.h>

#include <cpl.h>
#include <cpl_recipedefine.h>

#include "eelt_pupil.h"

/* ======= PARAMETERS ======= */
static const int    GRID_SIZE      = 822;
static const int    NUM_SEGMENTS   = 798;
static const double EELT_DIAM      = 40.0;
static const double PIXSCALE       = 40.0 / 822.0 * 1.0275;
static const double SPIDER_WIDTH   = 0.54;
static const double GAP            = 0.0;

static const double MAX_REFLECTIVITY             = 0.96;
static const double COATING_DEGRADATION_PER_DAY  = 0.000125;

/* ======= Utilities ======= */

static char *trim(char *s)
{
    if (!s) return s;
    while (*s && isspace((unsigned char)*s)) s++;

    if (s[0] == '\xEF' && s[1] == '\xBB' && s[2] == '\xBF')
        s += 3;

    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) len--;
    s[len] = '\0';
    return s;
}

static int parse_date(const char *iso, struct tm *out)
{
    if (!iso || !out) return 0;
    memset(out, 0, sizeof(*out));

    int y, m, d;
    if (sscanf(iso, "%d-%d-%d", &y, &m, &d) != 3) return 0;

    out->tm_year = y - 1900;
    out->tm_mon  = m - 1;
    out->tm_mday = d;
    out->tm_hour = 0;
    out->tm_min  = 0;
    out->tm_sec  = 0;
    out->tm_isdst = -1;

    return 1;
}

static int days_since(const struct tm *then)
{
    time_t now_t = time(NULL);
    struct tm now_tm;

#if defined(_WIN32)
    localtime_s(&now_tm, &now_t);
#else
    localtime_r(&now_t, &now_tm);
#endif

    time_t then_t = mktime((struct tm *)then);
    if (then_t == (time_t)-1) return 0;

    double diff = difftime(now_t, then_t);
    return (int)(diff / 86400.0);
}

static void utc_timestamp(char *buf, size_t buflen)
{
    time_t now = time(NULL);
    struct tm gmt;

#if defined(_WIN32)
    gmtime_s(&gmt, &now);
#else
    gmtime_r(&now, &gmt);
#endif

    strftime(buf, buflen, "%Y-%m-%dT%H_%M_%S", &gmt);
}

static xmlNode *find_segments_node(xmlNode *root)
{
    for (xmlNode *cur = root; cur; cur = cur->next) {
        if (cur->type == XML_ELEMENT_NODE &&
            xmlStrcmp(cur->name, (const xmlChar *)"segments") == 0)
            return cur;

        xmlNode *child = cur->children;
        xmlNode *found = find_segments_node(child);
        if (found) return found;
    }
    return NULL;
}

/* ======= XML loader ======= */

static int load_segments_from_file(const char *filename,
                                   double *F1,
                                   int n_segments)
{
    if (!filename || !F1) return 0;

    FILE *f = fopen(filename, "rb");
    if (!f) {
        cpl_msg_error(__func__, "Cannot open file: %s", filename);
        return 0;
    }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = (char *)malloc(sz + 1);
    if (!buf) {
        fclose(f);
        cpl_msg_error(__func__, "Out of memory reading file.");
        return 0;
    }

    size_t nread = fread(buf, 1, sz, f);
    if (nread != (size_t)sz)
        cpl_msg_warning(__func__, "fread read only %zu of %ld bytes", nread, sz);

    buf[sz] = '\0';
    fclose(f);

    const char *start = strstr(buf, "<segments");
    if (!start) {
        free(buf);
        cpl_msg_error(__func__, "No <segments> block found in file.");
        return 0;
    }

    const char *end = strstr(start, "</segments>");
    if (!end) {
        free(buf);
        cpl_msg_error(__func__, "Unterminated <segments> block.");
        return 0;
    }
    end += strlen("</segments>");

    size_t block_len = (size_t)(end - start);
    char *xml_block = (char *)malloc(block_len + 1);
    if (!xml_block) {
        free(buf);
        cpl_msg_error(__func__, "Out of memory extracting XML block.");
        return 0;
    }

    memcpy(xml_block, start, block_len);
    xml_block[block_len] = '\0';

    xmlDoc *doc = xmlReadMemory(xml_block, (int)block_len,
                                "segments.xml", NULL, XML_PARSE_NOBLANKS);
    free(xml_block);
    free(buf);

    if (!doc) {
        cpl_msg_error(__func__, "libxml2 failed to parse segments block.");
        return 0;
    }

    xmlNode *root = xmlDocGetRootElement(doc);
    xmlNode *segments = NULL;

    if (root && root->type == XML_ELEMENT_NODE &&
        xmlStrcmp(root->name, (const xmlChar *)"segments") == 0)
        segments = root;
    else
        segments = find_segments_node(root);

    if (!segments) {
        cpl_msg_error(__func__, "<segments> element not found in parsed document.");
        xmlFreeDoc(doc);
        return 0;
    }

    for (int i = 0; i < n_segments; ++i)
        F1[i] = 0.0;

    for (xmlNode *seg = segments->children; seg; seg = seg->next) {
        if (seg->type != XML_ELEMENT_NODE) continue;
        if (xmlStrcmp(seg->name, (const xmlChar *)"segment") != 0) continue;

        xmlChar *id_attr          = xmlGetProp(seg, (const xmlChar *)"id");
        xmlChar *operational_attr = xmlGetProp(seg, (const xmlChar *)"operational");
        xmlChar *last_attr        = xmlGetProp(seg, (const xmlChar *)"last_recoating");

        if (!id_attr || !operational_attr || !last_attr) {
            if (id_attr)          xmlFree(id_attr);
            if (operational_attr) xmlFree(operational_attr);
            if (last_attr)        xmlFree(last_attr);
            cpl_msg_warning(__func__, "segment missing attributes; skipping.");
            continue;
        }

        char *id_str = trim((char *)id_attr);
        int seg_id = atoi(id_str) - 1;
        if (seg_id < 0 || seg_id >= n_segments) {
            cpl_msg_warning(__func__, "segment id out of range: %d", seg_id + 1);
            xmlFree(id_attr);
            xmlFree(operational_attr);
            xmlFree(last_attr);
            continue;
        }

        char *operational_str = trim((char *)operational_attr);
        for (char *p = operational_str; *p; ++p)
            *p = (char)tolower((unsigned char)*p);

        int operational = (strcmp(operational_str, "true") == 0);

        char *last_str = trim((char *)last_attr);
        struct tm recoating_tm;
        double reflectivity = 0.0;

        if (operational) {
            if (parse_date(last_str, &recoating_tm)) {
                int days = days_since(&recoating_tm);
                reflectivity = MAX_REFLECTIVITY
                               - COATING_DEGRADATION_PER_DAY * (double)days;
                if (reflectivity < 0.0) reflectivity = 0.0;
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

static double compute_mjd_utc(void)
{
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

    double JD = (int)(365.25 * (Y + 4716))
                + (int)(30.6001 * (M + 1)) + D + B - 1524.5;
    double frac_day = (h + m / 60.0 + s / 3600.0) / 24.0;

    return JD + frac_day - 2400000.5;
}

/* ======= FITS writer ======= */

static int write_fits(const char *filename,
                      const double *data,
                      int width,
                      int height,
                      const char *date_obs)
{
    fitsfile *fptr = NULL;
    int status = 0;

    long naxes[2] = { (long)width, (long)height };
    char overwrite_name[PATH_MAX];

    snprintf(overwrite_name, sizeof(overwrite_name), "!%s", filename);

    if (fits_create_file(&fptr, overwrite_name, &status)) {
        fits_report_error(stderr, status);
        return 0;
    }

    if (fits_create_img(fptr, DOUBLE_IMG, 2, naxes, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return 0;
    }

    long fpixel[2]   = {1, 1};
    long nelements   = naxes[0] * naxes[1];

    if (fits_write_pix(fptr, TDOUBLE, fpixel, nelements,
                       (void *)data, &status)) {
        fits_report_error(stderr, status);
        fits_close_file(fptr, &status);
        return 0;
    }

    if (date_obs && *date_obs) {
        fits_update_key(fptr, TSTRING, "DATE-OBS", (void *)date_obs,
                        "UTC date of observation", &status);
    }

    fits_update_key(fptr, TSTRING, "ORIGIN",
                    "ESO ELT PUPIL STATUS GENERATOR",
                    "File origin", &status);

    fits_write_comment(fptr,
        "Mirror segment reflectivity and operational status snapshot", &status);
    fits_write_comment(fptr,
        "Generated from StatusM1segments.xml", &status);

    double mjd = compute_mjd_utc();
    fits_update_key(fptr, TDOUBLE, "MJD-OBS", &mjd,
                    "Modified Julian Date of observation", &status);

    fits_update_key(fptr, TSTRING, "INSTRUME", "MICADO",
                    "Instrument name", &status);
    fits_update_key(fptr, TSTRING, "ESO DPR CATG", "CALIB",
                    "Data category", &status);
    fits_update_key(fptr, TSTRING, "ESO DPR TYPE", "PUPIL",
                    "Data type", &status);
    fits_update_key(fptr, TSTRING, "ESO DPR TECH", "IMAGE",
                    "Data technique", &status);

    if (fits_close_file(fptr, &status)) {
        fits_report_error(stderr, status);
        return 0;
    }

    return 1;
}

static void build_fits_path(char *out, size_t outlen, const char *timestamp)
{
    snprintf(out, outlen,
             "c.ELT.%s.pupil.segmentstatus.fits",
             timestamp);
}

/* ======= CPL RECIPE CORE ======= */

static int elt_pupil(cpl_frameset *frames,
                     const cpl_parameterlist *parlist)
{
    (void)parlist;

    cpl_frame *frame = cpl_frameset_get_first(frames);
    if (!frame) {
        cpl_msg_error(__func__, "No input frame in frameset.");
        return CPL_ERROR_ILLEGAL_INPUT;
    }

    const char *xml_file = cpl_frame_get_filename(frame);
    if (!xml_file) {
        cpl_msg_error(__func__, "Input frame has no filename.");
        return CPL_ERROR_ILLEGAL_INPUT;
    }

    double *F1 = (double *)malloc(NUM_SEGMENTS * sizeof(double));
    if (!F1) {
        cpl_msg_error(__func__, "Out of memory for F1.");
        return CPL_ERROR_UNSPECIFIED;
    }

    if (!load_segments_from_file(xml_file, F1, NUM_SEGMENTS)) {
        free(F1);
        cpl_msg_error(__func__, "Failed to load segments from XML.");
        return CPL_ERROR_FILE_IO;
    }

    double *pupil_mask = generateEeltPupilReflectivity(
        F1, NUM_SEGMENTS, GRID_SIZE,
        SPIDER_WIDTH,
        GRID_SIZE / 2.0 - 0.5, GRID_SIZE / 2.0 - 0.5,
        PIXSCALE,
        GAP,
        0.0,
        EELT_DIAM,
        0
    );

    free(F1);

    if (!pupil_mask) {
        cpl_msg_error(__func__, "Failed to generate pupil mask.");
        return CPL_ERROR_UNSPECIFIED;
    }

    char timestamp[32];
    utc_timestamp(timestamp, sizeof(timestamp));

    char fits_path[256];
    build_fits_path(fits_path, sizeof(fits_path), timestamp);

    int ok = write_fits(fits_path, pupil_mask, GRID_SIZE, GRID_SIZE, timestamp);
    free(pupil_mask);

    if (!ok) {
        cpl_msg_error(__func__, "Failed to write FITS file.");
        return CPL_ERROR_FILE_IO;
    }

    cpl_msg_info(__func__, "Saved FITS file: %s", fits_path + 1);
    return CPL_ERROR_NONE;
}

static cpl_error_code
elt_pupil_fill_parameterlist(cpl_parameterlist *self)
{
    cpl_parameter *p = cpl_parameter_new_value(
        "dummy", CPL_TYPE_BOOL,
        "Example parameter (unused)", "dummy", 0);

    cpl_parameterlist_append(self, p);
    return CPL_ERROR_NONE;
}

/* ======= CPL RECIPE DEFINITION ======= */

cpl_recipe_define(
    elt_pupil,
    1,
    "Jani Achren",
    "jani.achren@incidentangle.fi",
    "2026",
    "ELT pupil segment status generator",
    "elt_pupil -- generate ELT pupil segment status FITS image\n"
    "This recipe reads an XML file containing <segments> information\n"
    "and produces a MICADO-compatible pupil segment status FITS file.\n"
);
