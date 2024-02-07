from cython.parallel cimport prange
cimport openmp
import numpy as np
cimport numpy as np
import cython
cimport cython
cimport openmp
import numexpr

cpdef void fastmax(Py_ssize_t[:,:] a, Py_ssize_t[::1] result):
    cdef:
        openmp.omp_lock_t locker
        Py_ssize_t lenarray = a.shape[0]
        Py_ssize_t widtharray = a.shape[1]
        Py_ssize_t i1, i2
    openmp.omp_init_lock(&locker)
    result[0]=a[0][0]
    for i1 in prange(lenarray,nogil=True):
        for i2 in range(widtharray):
            if a[i1][i2] > result[0]:
                openmp.omp_set_lock(&locker)
                result[0]= a[i1][i2]
                openmp.omp_unset_lock(&locker)
    openmp.omp_destroy_lock(&locker)

# d([a1x[i1], i1, a2x[i2], i2])
def parse_bin_data(
    Py_ssize_t dummyvalue,
    list[list] goodtext,
    Py_ssize_t[:,::1] dstackednormal,
    Py_ssize_t window_shifts,
    Py_ssize_t gootextoldlen,
    Py_ssize_t goodtextnewlen,
    Py_ssize_t[:,::1] windowed_index_array,
    Py_ssize_t[:,::1] normal_index_array,
    Py_ssize_t[:,::1] sequence_counter_normal,
    Py_ssize_t[:,::1] sequence_counter_windowed,
    Py_ssize_t [::1] resultmax,
    np.ndarray a1x, np.ndarray a2x,
    int cpus
):
    cdef:
        list[list] tmpgoodtext = []
        Py_ssize_t arrayshape0 = normal_index_array.shape[0]
        Py_ssize_t arrayshape1 = normal_index_array.shape[1]
        Py_ssize_t dstackednormal_shape0= dstackednormal.shape[0]
        Py_ssize_t shapecleaner0,shapecleaner1,window_shift,normal_index,windowed_index
        bint var_too_big
        Py_ssize_t minlen, index_ax1_loop, index_ax1ax2_loop, tmpgoodtextlen,tmpindex,sequence_start,text_index,i2,i1,offset
        np.ndarray[np.uint8_t, ndim=2] sequence_counter_normal_check=np.zeros((dstackednormal.shape[0], window_shifts), dtype=bool)
        Py_ssize_t a1x_shape = a1x.shape[0]
        Py_ssize_t a2x_shape = a2x.shape[0]
        tuple[nd.array,nd.array] max_sequnce
    openmp.omp_set_num_threads(cpus)
    while goodtextnewlen != gootextoldlen:
        gootextoldlen = goodtextnewlen
        for shapecleaner0 in prange(arrayshape0, nogil=True):
            for shapecleaner1 in range(arrayshape1):
                normal_index_array[shapecleaner0][shapecleaner1] = -1
                windowed_index_array[shapecleaner0][shapecleaner1] = -1
                sequence_counter_normal[shapecleaner0][shapecleaner1] = 0
                sequence_counter_windowed[shapecleaner0][shapecleaner1] = 0
        for window_shift in prange(0, window_shifts, 1,nogil=True):
            sequence_start = 0
            var_too_big = False

            for normal_index in range(dstackednormal_shape0):
                windowed_index = normal_index + window_shift
                if windowed_index >= dstackednormal_shape0:
                    var_too_big = True
                if var_too_big:
                    break
                if dummyvalue == dstackednormal[windowed_index][1]:
                    continue
                if dstackednormal[normal_index][0] == dstackednormal[windowed_index][1]:
                    sequence_start = sequence_start+1
                    sequence_counter_windowed[normal_index][
                        window_shift
                    ] = sequence_start
                    sequence_counter_normal[normal_index][window_shift] = sequence_start
            if sequence_start > 0:
                for tmpindex in range(sequence_start, -1, -1):
                    if sequence_counter_normal[normal_index - tmpindex][window_shift]:
                        sequence_counter_normal[normal_index - tmpindex][
                            window_shift
                        ] = sequence_start
        fastmax(sequence_counter_normal,resultmax)
        numexpr.evaluate('sequence_counter_normal == maxresu',global_dict={},
        local_dict={'sequence_counter_normal':sequence_counter_normal, 'maxresu':resultmax[0] },
        out=sequence_counter_normal_check)

        max_sequnce = np.where(sequence_counter_normal_check)
        index_ax1 = max_sequnce[0]
        index_ax2 = max_sequnce[0] + max_sequnce[1]
        index_ax1ax2_loop = index_ax2.shape[0]
        offset=max_sequnce[1][0]
        tmpgoodtextlen = 0
        for index_ax1_loop in range(index_ax1ax2_loop):
            i1=index_ax1[index_ax1_loop]
            i2=index_ax2[index_ax1_loop]
            if i1 >= a1x.shape[0] or i2 >= a2x_shape:
                continue
            if a1x[i1] and a2x[i2]:
                if a1x[i1] == a2x[i2]:
                    tmpgoodtext.append([a1x[i1], i1, a2x[i2], i2,offset])
                    tmpgoodtextlen = tmpgoodtextlen + 1

        for index_ax1_loop in range(index_ax1ax2_loop):
            i1=index_ax1[index_ax1_loop]
            i2=index_ax2[index_ax1_loop]
            if i1 < a1x_shape:
                a1x[i1] = b""
            if i1 < dstackednormal_shape0:
                dstackednormal[i1][0] = dummyvalue
            if i2 < a2x_shape:
                a2x[i2] = b""
            if i2 < dstackednormal_shape0:
                dstackednormal[i2][1] = dummyvalue
        if tmpgoodtextlen > 0:
            goodtext.append([])
            for text_index in range(tmpgoodtextlen):
                text=tmpgoodtext[text_index]
                text.append(tmpgoodtextlen)
                goodtext[len(goodtext)-1].append(text)
            tmpgoodtext.clear()
        goodtextnewlen = len(goodtext)
