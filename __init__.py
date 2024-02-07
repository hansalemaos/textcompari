import os
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
import numexpr
import sys
from typing import Union
from numpycythonpermutations import hash_2d_nparray

try:
    from .textpari import parse_bin_data
except Exception:
    from cycompi import compile_cython_code
    import os

    numpyincludefolder = np.get_include()
    pyxfile = "textpari.pyx"
    uniqueproductcythonmodule = pyxfile.split(".")[0]
    dirname = os.path.abspath(os.path.dirname(__file__))
    pyxfile_complete_path = os.path.join(dirname, pyxfile)
    optionsdict = {
        "Options.docstrings": False,
        "Options.embed_pos_in_docstring": False,
        "Options.generate_cleanup_code": False,
        "Options.clear_to_none": True,
        "Options.annotate": True,
        "Options.fast_fail": False,
        "Options.warning_errors": False,
        "Options.error_on_unknown_names": True,
        "Options.error_on_uninitialized": True,
        "Options.convert_range": True,
        "Options.cache_builtins": True,
        "Options.gcc_branch_hints": True,
        "Options.lookup_module_cpdef": False,
        "Options.embed": False,
        "Options.cimport_from_pyx": False,
        "Options.buffer_max_dims": 8,
        "Options.closure_freelist_size": 8,
    }
    configdict = {
        "py_limited_api": False,
        "name": uniqueproductcythonmodule,
        "sources": [pyxfile_complete_path],
        "include_dirs": [numpyincludefolder],
        "define_macros": [
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("CYTHON_USE_DICT_VERSIONS", 1),
            ("CYTHON_FAST_GIL", 1),
            ("CYTHON_USE_PYLIST_INTERNALS", 1),
            ("CYTHON_USE_UNICODE_INTERNALS", 1),
            ("CYTHON_ASSUME_SAFE_MACROS", 1),
            ("CYTHON_USE_TYPE_SLOTS", 1),
            ("CYTHON_USE_PYTYPE_LOOKUP", 1),
            ("CYTHON_USE_ASYNC_SLOTS", 1),
            ("CYTHON_USE_PYLONG_INTERNALS", 1),
            ("CYTHON_USE_UNICODE_WRITER", 1),
            ("CYTHON_UNPACK_METHODS", 1),
            ("CYTHON_USE_EXC_INFO_STACK", 1),
            ("CYTHON_ATOMICS", 1),
        ],
        "undef_macros": [],
        "library_dirs": [],
        "libraries": [],
        "runtime_library_dirs": [],
        "extra_objects": [],
        "extra_compile_args": ["/O2", "/Oy", "/openmp"],
        "extra_link_args": [],
        "export_symbols": [],
        "swig_opts": [],
        "depends": [],
        "language": "c",
        "optional": None,
    }
    compiler_directives = {
        "binding": True,
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "nonecheck": False,
        "overflowcheck": False,
        "overflowcheck.fold": False,
        "embedsignature": False,
        "embedsignature.format": "c",  # (c / python / clinic)
        "cdivision": True,
        "cdivision_warnings": False,
        "cpow": True,
        "always_allow_keywords": False,
        "c_api_binop_methods": False,
        "profile": False,
        "linetrace": False,
        "infer_types": True,
        "language_level": "3str",  # (2/3/3str)
        "c_string_type": "bytes",  # (bytes / str / unicode)
        "c_string_encoding": "ascii",  # (ascii, default, utf-8, etc.)
        "type_version_tag": False,
        "unraisable_tracebacks": True,
        "iterable_coroutine": False,
        "annotation_typing": False,
        "emit_code_comments": False,
        "cpp_locals": False,
        "legacy_implicit_noexcept": False,
        "optimize.use_switch": True,
        "optimize.unpack_method_calls": True,
        "warn.undeclared": False,  # (default False)
        "warn.unreachable": True,  # (default True)
        "warn.maybe_uninitialized": False,  # (default False)
        "warn.unused": False,  # (default False)
        "warn.unused_arg": False,  # (default False)
        "warn.unused_result": False,  # (default False)
        "warn.multiple_declarators": True,  # (default True)
        "show_performance_hints": True,  # (default True)
    }

    compile_cython_code(
        name=uniqueproductcythonmodule,
        configdict=configdict,
        optionsdict=optionsdict,
        cmd_line_args=compiler_directives,
        cwd=dirname,
        shell=True,
        env=os.environ.copy(),
    )
    from .textpari import parse_bin_data


def _read_data(afile):
    if isinstance(
        afile,
        str,
    ):
        if os.path.exists(afile):
            with open(afile, mode="rb") as f:
                a = f.read().splitlines()
        else:
            a = afile.encode("utf-8").splitlines()
    elif isinstance(afile, np.ndarray):
        try:
            return afile.astype("S").tolist()
        except Exception as fe:
            return [str(x).encode("utf-8", "backslashreplace") for x in afile]

    elif isinstance(
        afile,
        bytes,
    ):
        a = afile.splitlines()
    elif isinstance(
        afile,
        tuple,
    ):
        a = list(afile)
    elif isinstance(afile, list):
        if isinstance(
            afile[0],
            bytes,
        ):
            return afile
        else:
            return [str(x).encode("utf-8", "backslashreplace") for x in afile]

    else:
        return afile
    return a


def read_data(afile, bfile, dummy=None):
    a = _read_data(afile)
    b = _read_data(bfile)
    a.append(b"")
    b.append(b"")
    original_len_a = len(a)
    original_len_b = len(b)
    if original_len_a > original_len_b:
        b.extend([b""] * (original_len_a - original_len_b))
    elif original_len_a < original_len_b:
        a.extend([b""] * (original_len_b - original_len_a))

    if len(a) > len(b):
        for _ in range(len(b), len(a)):
            b.append(b"")
    elif len(a) < len(b):
        for _ in range(len(a), len(b)):
            a.append(b"")

    if dummy:
        b.append(dummy)
        a.append(dummy)
    b = np.array(b, dtype="S")
    a = np.array(a, dtype="S")

    if a.dtype > b.dtype:
        b = b.astype(a.dtype)
    if a.dtype < b.dtype:
        a = a.astype(b.dtype)

    return (
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
        original_len_a - 1,
        original_len_b - 1,
    )


def get_bin_hash(a1x, a2x):
    a1 = (
        a1x.view(np.uint8)
        .reshape(
            (
                -1,
                a1x.itemsize,
            )
        )
        .copy()
    )
    a2 = (
        a2x.view(np.uint8)
        .reshape(
            (
                -1,
                a2x.itemsize,
            )
        )
        .copy()
    )
    return a1, a2, [hash_2d_nparray(x) for x in [a1, a2]]


def read_binary_data(
    afile: Union[str, bytes, tuple, list, np.ndarray],
    bfile: Union[str, bytes, tuple, list, np.ndarray],
    dummy: Union[None, bytes] = None,
) -> tuple:
    """
    A function to read binary data from the given files and return a tuple containing the processed data and original lengths.

    Args:
        afile: Union[str, bytes, tuple, list, np.ndarray] - The first file containing binary data.
        bfile: Union[str, bytes, tuple, list, np.ndarray] - The second file containing binary data.
        dummy: Union[None, bytes] - Optional dummy data to be used in the process.

    Returns:
        tuple: A tuple containing the processed binary data and original lengths.
    """
    a1xx, a2xx, original_len_a, original_len_b = read_data(afile, bfile, dummy=dummy)
    return (
        a1xx,
        a2xx,
        *get_bin_hash(a1xx, a2xx),
        original_len_a,
        original_len_b,
    )


def generate_df(
    a1xx: np.ndarray,
    a2xx: np.ndarray,
    allnumpyarrayshashxx: np.ndarray,
    window_shifts: int,
    cpus: int,
) -> tuple:
    """
    This function generates a tuple of results based on the input arrays and parameters.
    Parameters:
        a1xx: np.ndarray
        a2xx: np.ndarray
        allnumpyarrayshashxx: np.ndarray
        window_shifts: int
        cpus: int
    Returns:
        tuple: A tuple containing alldfs1 (list of DataFrames), numberofmatches0 (int), numberofmatches1 (int), and dstackednormal (np.ndarray)
    """
    a1x, a2x = a1xx.copy(), a2xx.copy()
    a1, a2, allnumpyarrayshash = a1x.copy(), a2x.copy(), allnumpyarrayshashxx.copy()
    dummyvalue = allnumpyarrayshashxx[0][-1]
    dstackednormal = np.dstack(allnumpyarrayshash).squeeze()
    goodtext = []
    gootextoldlen = -1
    goodtextnewlen = 0
    windowed_index_array = np.zeros(
        (dstackednormal.shape[0], window_shifts), dtype=np.int64
    )
    normal_index_array = np.zeros(
        (dstackednormal.shape[0], window_shifts), dtype=np.int64
    )

    sequence_counter_normal = np.zeros(
        (dstackednormal.shape[0], window_shifts), dtype=np.int64
    )
    sequence_counter_windowed = np.zeros(
        (dstackednormal.shape[0], window_shifts), dtype=np.int64
    )
    resultmax = np.zeros(1, dtype=np.int64)

    parse_bin_data(
        dummyvalue,
        goodtext,
        dstackednormal,
        window_shifts,
        gootextoldlen,
        goodtextnewlen,
        windowed_index_array,
        normal_index_array,
        sequence_counter_normal,
        sequence_counter_windowed,
        resultmax,
        a1x.copy(),
        a2x.copy(),
        cpus,
    )

    numberofmatches0 = 0
    alldfs1 = []
    for t in goodtext:
        numberofmatches0 + len(t)
        for tt in t:
            tt.append(False)
        dfx = pd.DataFrame(
            t,
            columns=[
                "aa_file1",
                "aa_line1",
                "aa_file2",
                "aa_line2",
                "aa_offset",
                "aa_company",
                "aa_up",
            ],
        )
        if not dfx.empty:
            alldfs1.append(dfx)

    goodtext.clear()
    dstackedreversed = np.dstack(allnumpyarrayshash[::-1]).squeeze()
    parse_bin_data(
        dummyvalue,
        goodtext,
        dstackedreversed,
        window_shifts,
        gootextoldlen,
        goodtextnewlen,
        windowed_index_array,
        normal_index_array,
        sequence_counter_normal,
        sequence_counter_windowed,
        resultmax,
        a2x.copy(),
        a1x.copy(),
        cpus,
    )
    numberofmatches1 = 0
    for t in goodtext:
        numberofmatches1 = +len(t)
        for tt in t:
            tt.append(True)
        dfx = pd.DataFrame(
            t,
            columns=[
                "aa_file1",
                "aa_line1",
                "aa_file2",
                "aa_line2",
                "aa_offset",
                "aa_company",
                "aa_up",
            ],
        )
        if not dfx.empty:
            alldfs1.append(dfx)
    return alldfs1, numberofmatches0, numberofmatches1, dstackednormal


def format_dfmatches(
    numberofmatches0: int, numberofmatches1: int, alldfs1: list
) -> list:
    """
    Formats the given DataFrame matches based on the number of matches and the order.

    Args:
        numberofmatches0 (int): The number of matches for category 0.
        numberofmatches1 (int): The number of matches for category 1.
        alldfs1 (list): A list of DataFrames to be concatenated and processed.

    Returns:
        list: A tuple containing the processed DataFrame and the order of concatenation.
    """
    sorder = numberofmatches0 < numberofmatches1
    if sorder:
        droporder = ["aa_line2", "aa_line1"]
        sort_values_order = ["aa_line1", "aa_offset"]
        order_line = "aa_line1"
        sort_values_after = "aa_line2"
        dropduplicate_final = "aa_line1"
        concatorder = ["aa_line1", "aa_line2"]

    else:
        droporder = ["aa_line1", "aa_line2"]
        sort_values_order = ["aa_line2", "aa_offset"]
        order_line = "aa_line2"
        sort_values_after = "aa_line1"
        dropduplicate_final = "aa_line2"
        concatorder = ["aa_line2", "aa_line1"]

    return (
        (
            (
                pd.concat(alldfs1, ignore_index=True)
                .sort_values(
                    by=["aa_company", "aa_offset"],
                    ascending=[False, True],
                )
                .drop_duplicates(subset=droporder)
                .sort_values(by=sort_values_order)
                .reset_index(drop=True)
            )
            .sort_values(by=sort_values_order)
            .drop_duplicates(subset=order_line)
            .sort_values(by=sort_values_after)
            .reset_index(drop=True)
            .drop_duplicates(subset=dropduplicate_final)
            .sort_values(by=order_line)
            .reset_index(drop=True)
        ),
        concatorder,
    )


def get_missing_lines(
    df: pd.DataFrame, a1x: np.ndarray, a2x: np.ndarray, dstackednormal: np.ndarray
) -> pd.DataFrame:
    """
    Get missing lines from the input dataframes and arrays and return a new dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        a1x (np.ndarray): The first input array.
        a2x (np.ndarray): The second input array.
        dstackednormal (np.ndarray): The stacked normal array.

    Returns:
        pd.DataFrame: The concatenated dataframe containing the missing lines.
    """
    fullindex1 = np.arange((len(dstackednormal)))
    diffindex1 = np.setdiff1d(fullindex1, df.aa_line2)
    missing1 = a1x[diffindex1]
    emp1 = [-3 for _ in range(len(missing1))]

    fullindex2 = np.arange((len(dstackednormal)))
    diffindex2 = np.setdiff1d(fullindex2, df.aa_line1)
    missing2 = a2x[diffindex2]
    emp2 = [-2 for _ in range(len(missing2))]

    dfmi2 = pd.concat(
        [
            pd.Series([b"" for _ in range(len(missing1))]),
            pd.Series(diffindex1),
            pd.Series(missing1),
            pd.Series(diffindex1),
            pd.Series(emp1),
            pd.Series(emp1),
            pd.Series([False for _ in range(len(missing1))]),
        ],
        axis=1,
        ignore_index=True,
    )
    dfmi2.columns = df.columns
    dfmi1 = pd.concat(
        [
            pd.Series(missing2),
            pd.Series(diffindex2),
            pd.Series([b"" for _ in range(len(missing2))]),
            pd.Series(diffindex2),
            pd.Series(emp2),
            pd.Series(emp2),
            pd.Series([False for _ in range(len(missing2))]),
        ],
        axis=1,
        ignore_index=True,
    )
    dfmi1.columns = df.columns
    return dfmi1, dfmi2


def match_rest_with_fuzz(
    window_shifts: int,
    dfmi1: pd.DataFrame,
    dfmi2: pd.DataFrame,
    min_match: int = 95,
    usedtype: np.dtype = np.uint8,
    scorer=fuzz.WRatio,
    cpus: int = 5,
) -> pd.DataFrame:
    """
    A function to perform fuzzy matching between two DataFrames using the specified scorer,
    and returns the filtered and sorted matches.

    Parameters:
    - window_shifts: int, the maximum allowed difference in offset between matching values
    - dfmi1: pd.DataFrame, the first DataFrame to be matched
    - dfmi2: pd.DataFrame, the second DataFrame to be matched
    - min_match: int, optional, the minimum match percentage required, default is 95
    - usedtype: np.dtype, optional, the data type to be used, default is np.uint8
    - scorer: function, optional, the scoring function to be used for matching, default is fuzz.WRatio
    - cpus: int, optional, the number of CPUs to be used for parallel processing, default is 5

    Returns:
    - pd.DataFrame, the filtered and sorted matches between the two DataFrames
    """
    a = dfmi1.aa_file1.__array__()
    b = dfmi2.aa_file2.__array__()

    allcom = process.cdist(
        a,
        b,
        scorer=scorer,
        dtype=usedtype,
        workers=cpus,
    )

    max_values = np.amax(allcom, axis=1)
    df1index, df2index = np.where(
        numexpr.evaluate(
            "a==b",
            global_dict={},
            local_dict={
                "a": allcom,
                "b": np.tile(max_values.reshape((-1, 1)), (1, allcom.shape[1])),
            },
        )
    )

    index1_100, index2_100 = np.where(allcom >= min_match)

    goodmatches1 = dfmi1.iloc[index1_100].copy()
    goodmatches2 = dfmi2.iloc[index2_100].copy()
    goodmatches1["old_index"] = goodmatches1.index.__array__().copy()
    goodmatches2["old_index"] = goodmatches2.index.__array__().copy()
    goodmatches1 = goodmatches1.reset_index(drop=True)
    goodmatches2 = goodmatches2.reset_index(drop=True)
    goodmatches1["aa_match"] = allcom[index1_100, index2_100]
    goodmatches1.loc[:, "aa_file2"] = goodmatches2.aa_file2.__array__()
    goodmatches1.loc[:, "aa_line2"] = goodmatches2.aa_line2.__array__()
    goodmatches1["aa_offset"] = goodmatches1.aa_line1 - goodmatches2.aa_line2
    goodmatchesoffsetfiltered = goodmatches1.loc[
        goodmatches1.aa_offset.abs() <= window_shifts
    ]
    return goodmatchesoffsetfiltered.sort_values(
        by=["aa_match", "aa_offset"], ascending=[False, True]
    ).drop_duplicates(subset="old_index")


def get_matching_empty_lines(
    dfmi1: pd.DataFrame, dfmi2: pd.DataFrame, valid_concats: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns a DataFrame containing matching empty lines between the input DataFrames.

    Args:
        dfmi1 (pd.DataFrame): The first input DataFrame.
        dfmi2 (pd.DataFrame): The second input DataFrame.
        valid_concats (pd.DataFrame): DataFrame of valid concatenations.

    Returns:
        pd.DataFrame: DataFrame containing matching empty lines.
    """
    rest2 = dfmi2.loc[~dfmi2.aa_line2.isin(valid_concats.aa_line2)]
    rest1 = dfmi1.loc[~dfmi1.aa_line1.isin(valid_concats.aa_line1)]
    lineintersections = np.intersect1d(rest1.aa_line1, rest2.aa_line2)
    lineinter1 = rest1.loc[rest1.aa_line1.isin(lineintersections)]
    emptylinesmatch = lineinter1.aa_file1.__array__() == lineinter1.aa_file2.__array__()
    emptylinesmatch1 = lineinter1[emptylinesmatch].copy()
    emptylinesmatch1["aa_match"] = 100
    emptylinesmatch1["aa_offset"] = 0
    return emptylinesmatch1


def concat_all_matching_lines(
    df: pd.DataFrame,
    valid_concats: pd.DataFrame,
    emptylinesmatch1: pd.DataFrame,
    concatorder: list,
) -> pd.DataFrame:
    """
    Concatenates all matching lines from the given dataframes and returns the resulting dataframe.

    Args:
        df (pd.DataFrame): The input dataframe to be concatenated.
        valid_concats (pd.DataFrame): The dataframe containing valid concatenations.
        emptylinesmatch1 (pd.DataFrame): The dataframe containing empty lines matching pattern 1.
        concatorder (list): The order of concatenation.

    Returns:
        pd.DataFrame: The resulting concatenated dataframe.
    """
    dffound = (
        pd.concat(
            [
                df.assign(aa_match=100),
                valid_concats.drop(
                    columns=[
                        "aa_up",
                        "old_index",
                    ]
                )
                .assign(aa_company=-7)
                .reset_index(drop=True),
                emptylinesmatch1.assign(aa_company=-8),
            ],
            ignore_index=True,
        )
        .sort_values(by=["aa_line1", "aa_line2"])
        .reset_index(drop=True)
    )
    dffound["aa_offset"] = dffound.aa_line1 - dffound.aa_line2

    dffound["aa_up"] = False
    dffound["aa_up"] = dffound.loc[dffound.aa_offset < 0, "aa_up"] = True
    dffound["aa_same_line"] = dffound.aa_line1 == dffound.aa_line2
    dffound["aa_real_line_1"] = dffound.aa_line1.__array__().copy()
    dffound["aa_real_line_2"] = dffound.aa_line2.__array__().copy()
    dffound["aa_real_line_1"] = dffound["aa_real_line_1"].astype("Int64")
    dffound["aa_real_line_2"] = dffound["aa_real_line_2"].astype("Int64")
    return dffound


def concat_with_not_found(
    dffound: pd.DataFrame, dfmi1: pd.DataFrame, dfmi2: pd.DataFrame, concatorder: int
) -> pd.DataFrame:
    """
    Concatenates dataframes with not found values and returns a new dataframe.

    Args:
        dffound: pd.DataFrame - The dataframe with found values.
        dfmi1: pd.DataFrame - The first dataframe with not found values.
        dfmi2: pd.DataFrame - The second dataframe with not found values.
        concatorder: int - The order by which the concatenated dataframe should be sorted.

    Returns:
        pd.DataFrame - The concatenated and sorted dataframe.
    """
    stillnotfound2 = dfmi2.loc[~dfmi2.aa_line1.isin(dffound.aa_line2)].copy()
    stillnotfound2["aa_company"] = -5
    stillnotfound1 = dfmi1.loc[~dfmi1.aa_line1.isin(dffound.aa_line1)].copy()
    stillnotfound1["aa_company"] = -4
    stillnotfound1["aa_real_line_1"] = stillnotfound1.aa_line1.__array__().copy()
    stillnotfound2["aa_real_line_1"] = stillnotfound2.aa_line1.__array__().copy()
    stillnotfound1["aa_real_line_2"] = stillnotfound1.aa_line2.__array__().copy()
    stillnotfound2["aa_real_line_2"] = stillnotfound2.aa_line2.__array__().copy()
    stillnotfound1.aa_line1 += 0.5
    stillnotfound2.aa_line1 += 0.5
    stillnotfound1.aa_line2 += 0.5
    stillnotfound2.aa_line2 += 0.5
    stillnotfound2["aa_same_line"] = False
    stillnotfound1["aa_same_line"] = False

    return (
        pd.concat(
            [
                dffound,
                stillnotfound1.assign(aa_match=0),
                stillnotfound2.assign(aa_match=0),
            ],
            ignore_index=True,
        )
        .sort_values(by=concatorder)
        .reset_index(drop=True)
    )


def concat_with_not_found_at_all(
    dffound2: pd.DataFrame, original_len_a: int, original_len_b: int, maxfilelen: int
) -> pd.DataFrame:
    """
    Concatenates DataFrames and performs various operations on the data, including filtering and grouping.

    Args:
        dffound2 (pd.DataFrame): The DataFrame to be concatenated and operated on.
        original_len_a (int): The original length of data set A.
        original_len_b (int): The original length of data set B.
        maxfilelen (int): The maximum file length.

    Returns:
        pd.DataFrame: The resulting DataFrame after the operations are performed.
    """
    dffound2 = dffound2.loc[
        ~((dffound2.aa_line1 >= maxfilelen) | (dffound2.aa_line2 >= maxfilelen))
    ].copy()
    dffound2["aa_eof1"] = False
    dffound2["aa_eof2"] = False
    dffound2.loc[dffound2.aa_line1 >= original_len_a, "aa_eof1"] = True
    dffound2.loc[dffound2.aa_line2 >= original_len_b, "aa_eof2"] = True
    dffound2 = dffound2.reset_index(drop=False)

    realindexgroup = (
        dffound2.groupby(["aa_offset", "aa_company"])
        .apply(lambda h: h["index"])
        .explode()
        .reset_index(drop=True)
    )
    grouped = (
        (
            dffound2.groupby(["aa_offset", "aa_company"])
            .apply(lambda h: h.aa_company.count())
            .apply(lambda qq: np.arange(qq))
        )
        .reset_index(drop=True)
        .to_frame()
    )

    grouped["aa_group"] = grouped.apply(lambda h: np.full(len(h[0]), h.name), axis=1)
    grouped.rename(columns={0: "aa_group_element"}, inplace=True)
    groupedexploded = grouped.explode(["aa_group_element", "aa_group"])

    dffound2.loc[
        realindexgroup.__array__(), "aa_group"
    ] = groupedexploded.aa_group.__array__().copy()
    dffound2.loc[
        realindexgroup.__array__(), "aa_group_element"
    ] = groupedexploded.aa_group_element.__array__().copy()
    return dffound2.drop(columns=["aa_company"], inplace=False)


def get_file_diff(
    afile: Union[str, bytes, tuple, list, np.ndarray],
    bfile: Union[str, bytes, tuple, list, np.ndarray],
    window_shifts: int = 5,
    min_fuzz_match: int = 80,
    fuzz_scorer: fuzz.ratio = fuzz.WRatio,
    cpus: int = 5,
) -> pd.DataFrame:
    r"""
    A function to get the difference between two files and return a pandas DataFrame.

    :param afile: A file to compare (str, bytes, tuple, list, np.ndarray)
    :param bfile: Another file to compare (str, bytes, tuple, list, np.ndarray)
    :param window_shifts: The number of shifts for the window (default 5)
    :param min_fuzz_match: The minimum fuzzy match score (default 80)
    :param fuzz_scorer: The fuzzy scorer function (default fuzz.WRatio)
    :param cpus: The number of CPUs to use (default 5)
    :return: A pandas DataFrame containing the difference between the files
    """

    try:
        (
            a1xx,
            a2xx,
            a1xxx,
            a2xxx,
            allnumpyarrayshashxx,
            original_len_a,
            original_len_b,
        ) = read_binary_data(afile, bfile, dummy=None)

        maxfilelen = max(original_len_a, original_len_b)
        minfilelen = min(original_len_a, original_len_b)
        if minfilelen < window_shifts:
            window_shifts = minfilelen - 1
        alldfs1, numberofmatches0, numberofmatches1, dstackednormal = generate_df(
            a1xx, a2xx, allnumpyarrayshashxx, window_shifts, cpus
        )

        df, concatorder = format_dfmatches(numberofmatches0, numberofmatches1, alldfs1)
        dfmi1, dfmi2 = get_missing_lines(df, a1xx, a2xx, dstackednormal)

        valid_concats = match_rest_with_fuzz(
            window_shifts,
            dfmi1,
            dfmi2,
            min_match=min_fuzz_match,
            usedtype=np.uint8,
            scorer=fuzz_scorer,
        )
        emptylinesmatch1 = get_matching_empty_lines(dfmi1, dfmi2, valid_concats)
        dffound = concat_all_matching_lines(
            df, valid_concats, emptylinesmatch1, concatorder
        )
        dffound2 = concat_with_not_found(dffound, dfmi1, dfmi2, concatorder)
        return concat_with_not_found_at_all(
            dffound2, original_len_a, original_len_b, maxfilelen
        )
    except Exception as fe:
        sys.stderr.flush()

        sys.stderr.write(f"No matches found: {fe}")
        sys.stderr.flush()
        return pd.DataFrame(
            columns=[
                "index",
                "aa_file1",
                "aa_line1",
                "aa_file2",
                "aa_line2",
                "aa_offset",
                "aa_up",
                "aa_match",
                "aa_same_line",
                "aa_real_line_1",
                "aa_real_line_2",
                "aa_eof1",
                "aa_eof2",
                "aa_group",
                "aa_group_element",
            ]
        )
