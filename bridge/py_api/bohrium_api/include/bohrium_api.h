#ifndef BH_API_H
#define BH_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* C API functions */
#define BhAPI_flush_NUM 0
#define BhAPI_flush_RETURN void
#define BhAPI_flush_PROTO (void)

/* Total number of C API pointers */
#define BhAPI_num_of_pointers 1


#ifdef BhAPI_MODULE
/* This section is used when compiling _bh_api.c */

static BhAPI_flush_RETURN BhAPI_flush BhAPI_flush_PROTO;

#else
/* This section is used in modules that use _bh_api.c's API */

#define BhAPI_flush \
 (*(BhAPI_flush_RETURN (*)BhAPI_flush_PROTO) PyBhAPI[BhAPI_flush_NUM])

#ifdef NO_IMPORT_BH_API
    extern void **PyBhAPI;
#else
    void **PyBhAPI;
    /* Return -1 on error, 0 on success.
     * PyCapsule_Import will set an exception if there's an error.
     */
    static int
    import_bh_api(void)
    {
        PyBhAPI = (void **)PyCapsule_Import("bohrium_api._C_API", 0);
        return (PyBhAPI != NULL) ? 0 : -1;
    }
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif /* !defined(BH_API_H) */
