'''Wrapper for mosek.h

Generated with:
ctypesgen.py -a -l C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll -o ../mosek_header.py C:/Program Files/Mosek/7/tools/platform/win64x86/h/mosek.h

Do not modify this file.
'''

__docformat__ =  'restructuredtext'

# Begin preamble

import ctypes, os, sys
from ctypes import *

_int_types = (c_int16, c_int32)
if hasattr(ctypes, 'c_int64'):
    # Some builds of ctypes apparently do not have c_int64
    # defined; it's a pretty good bet that these builds do not
    # have 64-bit pointers.
    _int_types += (c_int64,)
for t in _int_types:
    if sizeof(t) == sizeof(c_size_t):
        c_ptrdiff_t = t
del t
del _int_types

class c_void(Structure):
    # c_void_p is a buggy return type, converting to int, so
    # POINTER(None) == c_void_p is actually written as
    # POINTER(c_void), so it can be treated as a real pointer.
    _fields_ = [('dummy', c_int)]

def POINTER(obj):
    p = ctypes.POINTER(obj)

    # Convert None to a real NULL pointer to work around bugs
    # in how ctypes handles None on 64-bit platforms
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)

    return p

class UserString:
    def __init__(self, seq):
        if isinstance(seq, basestring):
            self.data = seq
        elif isinstance(seq, UserString):
            self.data = seq.data[:]
        else:
            self.data = str(seq)
    def __str__(self): return str(self.data)
    def __repr__(self): return repr(self.data)
    def __int__(self): return int(self.data)
    def __long__(self): return long(self.data)
    def __float__(self): return float(self.data)
    def __complex__(self): return complex(self.data)
    def __hash__(self): return hash(self.data)

    def __cmp__(self, string):
        if isinstance(string, UserString):
            return cmp(self.data, string.data)
        else:
            return cmp(self.data, string)
    def __contains__(self, char):
        return char in self.data

    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.__class__(self.data[index])
    def __getslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
        return self.__class__(self.data[start:end])

    def __add__(self, other):
        if isinstance(other, UserString):
            return self.__class__(self.data + other.data)
        elif isinstance(other, basestring):
            return self.__class__(self.data + other)
        else:
            return self.__class__(self.data + str(other))
    def __radd__(self, other):
        if isinstance(other, basestring):
            return self.__class__(other + self.data)
        else:
            return self.__class__(str(other) + self.data)
    def __mul__(self, n):
        return self.__class__(self.data*n)
    __rmul__ = __mul__
    def __mod__(self, args):
        return self.__class__(self.data % args)

    # the following methods are defined in alphabetical order:
    def capitalize(self): return self.__class__(self.data.capitalize())
    def center(self, width, *args):
        return self.__class__(self.data.center(width, *args))
    def count(self, sub, start=0, end=sys.maxint):
        return self.data.count(sub, start, end)
    def decode(self, encoding=None, errors=None): # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.decode(encoding, errors))
            else:
                return self.__class__(self.data.decode(encoding))
        else:
            return self.__class__(self.data.decode())
    def encode(self, encoding=None, errors=None): # XXX improve this?
        if encoding:
            if errors:
                return self.__class__(self.data.encode(encoding, errors))
            else:
                return self.__class__(self.data.encode(encoding))
        else:
            return self.__class__(self.data.encode())
    def endswith(self, suffix, start=0, end=sys.maxint):
        return self.data.endswith(suffix, start, end)
    def expandtabs(self, tabsize=8):
        return self.__class__(self.data.expandtabs(tabsize))
    def find(self, sub, start=0, end=sys.maxint):
        return self.data.find(sub, start, end)
    def index(self, sub, start=0, end=sys.maxint):
        return self.data.index(sub, start, end)
    def isalpha(self): return self.data.isalpha()
    def isalnum(self): return self.data.isalnum()
    def isdecimal(self): return self.data.isdecimal()
    def isdigit(self): return self.data.isdigit()
    def islower(self): return self.data.islower()
    def isnumeric(self): return self.data.isnumeric()
    def isspace(self): return self.data.isspace()
    def istitle(self): return self.data.istitle()
    def isupper(self): return self.data.isupper()
    def join(self, seq): return self.data.join(seq)
    def ljust(self, width, *args):
        return self.__class__(self.data.ljust(width, *args))
    def lower(self): return self.__class__(self.data.lower())
    def lstrip(self, chars=None): return self.__class__(self.data.lstrip(chars))
    def partition(self, sep):
        return self.data.partition(sep)
    def replace(self, old, new, maxsplit=-1):
        return self.__class__(self.data.replace(old, new, maxsplit))
    def rfind(self, sub, start=0, end=sys.maxint):
        return self.data.rfind(sub, start, end)
    def rindex(self, sub, start=0, end=sys.maxint):
        return self.data.rindex(sub, start, end)
    def rjust(self, width, *args):
        return self.__class__(self.data.rjust(width, *args))
    def rpartition(self, sep):
        return self.data.rpartition(sep)
    def rstrip(self, chars=None): return self.__class__(self.data.rstrip(chars))
    def split(self, sep=None, maxsplit=-1):
        return self.data.split(sep, maxsplit)
    def rsplit(self, sep=None, maxsplit=-1):
        return self.data.rsplit(sep, maxsplit)
    def splitlines(self, keepends=0): return self.data.splitlines(keepends)
    def startswith(self, prefix, start=0, end=sys.maxint):
        return self.data.startswith(prefix, start, end)
    def strip(self, chars=None): return self.__class__(self.data.strip(chars))
    def swapcase(self): return self.__class__(self.data.swapcase())
    def title(self): return self.__class__(self.data.title())
    def translate(self, *args):
        return self.__class__(self.data.translate(*args))
    def upper(self): return self.__class__(self.data.upper())
    def zfill(self, width): return self.__class__(self.data.zfill(width))

class MutableString(UserString):
    """mutable string objects

    Python strings are immutable objects.  This has the advantage, that
    strings may be used as dictionary keys.  If this property isn't needed
    and you insist on changing string values in place instead, you may cheat
    and use MutableString.

    But the purpose of this class is an educational one: to prevent
    people from inventing their own mutable string class derived
    from UserString and than forget thereby to remove (override) the
    __hash__ method inherited from UserString.  This would lead to
    errors that would be very hard to track down.

    A faster and better solution is to rewrite your program using lists."""
    def __init__(self, string=""):
        self.data = string
    def __hash__(self):
        raise TypeError("unhashable type (it is mutable)")
    def __setitem__(self, index, sub):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + sub + self.data[index+1:]
    def __delitem__(self, index):
        if index < 0:
            index += len(self.data)
        if index < 0 or index >= len(self.data): raise IndexError
        self.data = self.data[:index] + self.data[index+1:]
    def __setslice__(self, start, end, sub):
        start = max(start, 0); end = max(end, 0)
        if isinstance(sub, UserString):
            self.data = self.data[:start]+sub.data+self.data[end:]
        elif isinstance(sub, basestring):
            self.data = self.data[:start]+sub+self.data[end:]
        else:
            self.data =  self.data[:start]+str(sub)+self.data[end:]
    def __delslice__(self, start, end):
        start = max(start, 0); end = max(end, 0)
        self.data = self.data[:start] + self.data[end:]
    def immutable(self):
        return UserString(self.data)
    def __iadd__(self, other):
        if isinstance(other, UserString):
            self.data += other.data
        elif isinstance(other, basestring):
            self.data += other
        else:
            self.data += str(other)
        return self
    def __imul__(self, n):
        self.data *= n
        return self

class String(MutableString, Union):

    _fields_ = [('raw', POINTER(c_char)),
                ('data', c_char_p)]

    def __init__(self, obj=""):
        if isinstance(obj, (str, unicode, UserString)):
            self.data = str(obj)
        else:
            self.raw = obj

    def __len__(self):
        return self.data and len(self.data) or 0

    def from_param(cls, obj):
        # Convert None or 0
        if obj is None or obj == 0:
            return cls(POINTER(c_char)())

        # Convert from String
        elif isinstance(obj, String):
            return obj

        # Convert from str
        elif isinstance(obj, str):
            return cls(obj)

        # Convert from c_char_p
        elif isinstance(obj, c_char_p):
            return obj

        # Convert from POINTER(c_char)
        elif isinstance(obj, POINTER(c_char)):
            return obj

        # Convert from raw pointer
        elif isinstance(obj, int):
            return cls(cast(obj, POINTER(c_char)))

        # Convert from object
        else:
            return String.from_param(obj._as_parameter_)
    from_param = classmethod(from_param)

def ReturnString(obj, func=None, arguments=None):
    return String.from_param(obj)

# As of ctypes 1.0, ctypes does not support custom error-checking
# functions on callbacks, nor does it support custom datatypes on
# callbacks, so we must ensure that all callbacks return
# primitive datatypes.
#
# Non-primitive return values wrapped with UNCHECKED won't be
# typechecked, and will be converted to c_void_p.
def UNCHECKED(type):
    if (hasattr(type, "_type_") and isinstance(type._type_, str)
        and type._type_ != "P"):
        return type
    else:
        return c_void_p

# ctypes doesn't have direct support for variadic functions, so we have to write
# our own wrapper class
class _variadic_function(object):
    def __init__(self,func,restype,argtypes):
        self.func=func
        self.func.restype=restype
        self.argtypes=argtypes
    def _as_parameter_(self):
        # So we can pass this variadic function as a function pointer
        return self.func
    def __call__(self,*args):
        fixed_args=[]
        i=0
        for argtype in self.argtypes:
            # Typecheck what we can
            fixed_args.append(argtype.from_param(args[i]))
            i+=1
        return self.func(*fixed_args+list(args[i:]))

# End preamble

_libs = {}
_libdirs = []

# Begin loader

# ----------------------------------------------------------------------------
# Copyright (c) 2008 David James
# Copyright (c) 2006-2008 Alex Holkner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

import os.path, re, sys, glob
import platform
import ctypes
import ctypes.util

def _environ_path(name):
    if name in os.environ:
        return os.environ[name].split(":")
    else:
        return []

class LibraryLoader(object):
    def __init__(self):
        self.other_dirs=[]

    def load_library(self,libname):
        """Given the name of a library, load it."""
        paths = self.getpaths(libname)

        for path in paths:
            if os.path.exists(path):
                return self.load(path)

        raise ImportError("%s not found." % libname)

    def load(self,path):
        """Given a path to a library, load it."""
        try:
            # Darwin requires dlopen to be called with mode RTLD_GLOBAL instead
            # of the default RTLD_LOCAL.  Without this, you end up with
            # libraries not being loadable, resulting in "Symbol not found"
            # errors
            if sys.platform == 'darwin':
                return ctypes.CDLL(path, ctypes.RTLD_GLOBAL)
            else:
                return ctypes.cdll.LoadLibrary(path)
        except OSError,e:
            raise ImportError(e)

    def getpaths(self,libname):
        """Return a list of paths where the library might be found."""
        if os.path.isabs(libname):
            yield libname
        else:
            # FIXME / TODO return '.' and os.path.dirname(__file__)
            for path in self.getplatformpaths(libname):
                yield path

            path = ctypes.util.find_library(libname)
            if path: yield path

    def getplatformpaths(self, libname):
        return []

# Darwin (Mac OS X)

class DarwinLibraryLoader(LibraryLoader):
    name_formats = ["lib%s.dylib", "lib%s.so", "lib%s.bundle", "%s.dylib",
                "%s.so", "%s.bundle", "%s"]

    def getplatformpaths(self,libname):
        if os.path.pathsep in libname:
            names = [libname]
        else:
            names = [format % libname for format in self.name_formats]

        for dir in self.getdirs(libname):
            for name in names:
                yield os.path.join(dir,name)

    def getdirs(self,libname):
        '''Implements the dylib search as specified in Apple documentation:

        http://developer.apple.com/documentation/DeveloperTools/Conceptual/
            DynamicLibraries/Articles/DynamicLibraryUsageGuidelines.html

        Before commencing the standard search, the method first checks
        the bundle's ``Frameworks`` directory if the application is running
        within a bundle (OS X .app).
        '''

        dyld_fallback_library_path = _environ_path("DYLD_FALLBACK_LIBRARY_PATH")
        if not dyld_fallback_library_path:
            dyld_fallback_library_path = [os.path.expanduser('~/lib'),
                                          '/usr/local/lib', '/usr/lib']

        dirs = []

        if '/' in libname:
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))
        else:
            dirs.extend(_environ_path("LD_LIBRARY_PATH"))
            dirs.extend(_environ_path("DYLD_LIBRARY_PATH"))

        dirs.extend(self.other_dirs)
        dirs.append(".")
        dirs.append(os.path.dirname(__file__))

        if hasattr(sys, 'frozen') and sys.frozen == 'macosx_app':
            dirs.append(os.path.join(
                os.environ['RESOURCEPATH'],
                '..',
                'Frameworks'))

        dirs.extend(dyld_fallback_library_path)

        return dirs

# Posix

class PosixLibraryLoader(LibraryLoader):
    _ld_so_cache = None

    def _create_ld_so_cache(self):
        # Recreate search path followed by ld.so.  This is going to be
        # slow to build, and incorrect (ld.so uses ld.so.cache, which may
        # not be up-to-date).  Used only as fallback for distros without
        # /sbin/ldconfig.
        #
        # We assume the DT_RPATH and DT_RUNPATH binary sections are omitted.

        directories = []
        for name in ("LD_LIBRARY_PATH",
                     "SHLIB_PATH", # HPUX
                     "LIBPATH", # OS/2, AIX
                     "LIBRARY_PATH", # BE/OS
                    ):
            if name in os.environ:
                directories.extend(os.environ[name].split(os.pathsep))
        directories.extend(self.other_dirs)
        directories.append(".")
        directories.append(os.path.dirname(__file__))

        try: directories.extend([dir.strip() for dir in open('/etc/ld.so.conf')])
        except IOError: pass

        unix_lib_dirs_list = ['/lib', '/usr/lib', '/lib64', '/usr/lib64']
        if sys.platform.startswith('linux'):
            # Try and support multiarch work in Ubuntu
            # https://wiki.ubuntu.com/MultiarchSpec
            bitage = platform.architecture()[0]
            if bitage.startswith('32'):
                # Assume Intel/AMD x86 compat
                unix_lib_dirs_list += ['/lib/i386-linux-gnu', '/usr/lib/i386-linux-gnu']
            elif bitage.startswith('64'):
                # Assume Intel/AMD x86 compat
                unix_lib_dirs_list += ['/lib/x86_64-linux-gnu', '/usr/lib/x86_64-linux-gnu']
            else:
                # guess...
                unix_lib_dirs_list += glob.glob('/lib/*linux-gnu')
        directories.extend(unix_lib_dirs_list)

        cache = {}
        lib_re = re.compile(r'lib(.*)\.s[ol]')
        ext_re = re.compile(r'\.s[ol]$')
        for dir in directories:
            try:
                for path in glob.glob("%s/*.s[ol]*" % dir):
                    file = os.path.basename(path)

                    # Index by filename
                    if file not in cache:
                        cache[file] = path

                    # Index by library name
                    match = lib_re.match(file)
                    if match:
                        library = match.group(1)
                        if library not in cache:
                            cache[library] = path
            except OSError:
                pass

        self._ld_so_cache = cache

    def getplatformpaths(self, libname):
        if self._ld_so_cache is None:
            self._create_ld_so_cache()

        result = self._ld_so_cache.get(libname)
        if result: yield result

        path = ctypes.util.find_library(libname)
        if path: yield os.path.join("/lib",path)

# Windows

class _WindowsLibrary(object):
    def __init__(self, path):
        self.cdll = ctypes.cdll.LoadLibrary(path)
        self.windll = ctypes.windll.LoadLibrary(path)

    def __getattr__(self, name):
        try: return getattr(self.cdll,name)
        except AttributeError:
            try: return getattr(self.windll,name)
            except AttributeError:
                raise

class WindowsLibraryLoader(LibraryLoader):
    name_formats = ["%s.dll", "lib%s.dll", "%slib.dll"]

    def load_library(self, libname):
        try:
            result = LibraryLoader.load_library(self, libname)
        except ImportError:
            result = None
            if os.path.sep not in libname:
                for name in self.name_formats:
                    try:
                        result = getattr(ctypes.cdll, name % libname)
                        if result:
                            break
                    except WindowsError:
                        result = None
            if result is None:
                try:
                    result = getattr(ctypes.cdll, libname)
                except WindowsError:
                    result = None
            if result is None:
                raise ImportError("%s not found." % libname)
        return result

    def load(self, path):
        return _WindowsLibrary(path)

    def getplatformpaths(self, libname):
        if os.path.sep not in libname:
            for name in self.name_formats:
                dll_in_current_dir = os.path.abspath(name % libname)
                if os.path.exists(dll_in_current_dir):
                    yield dll_in_current_dir
                path = ctypes.util.find_library(name % libname)
                if path:
                    yield path

# Platform switching

# If your value of sys.platform does not appear in this dict, please contact
# the Ctypesgen maintainers.

loaderclass = {
    "darwin":   DarwinLibraryLoader,
    "cygwin":   WindowsLibraryLoader,
    "win32":    WindowsLibraryLoader
}

loader = loaderclass.get(sys.platform, PosixLibraryLoader)()

def add_library_search_dirs(other_dirs):
    loader.other_dirs = other_dirs

load_library = loader.load_library

del loaderclass

# End loader

add_library_search_dirs([])

# Begin libraries

_libs["C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll"] = load_library("C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll")

# 1 libraries
# End libraries

# No modules

wint_t = c_ushort # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 452

wctype_t = c_ushort # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 453

errno_t = c_int # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 472

__time32_t = c_long # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 477

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 674
class struct_threadlocaleinfostruct(Structure):
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 653
class struct_threadmbcinfostruct(Structure):
    pass

pthreadlocinfo = POINTER(struct_threadlocaleinfostruct) # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 654

pthreadmbcinfo = POINTER(struct_threadmbcinfostruct) # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 655

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 656
class struct___lc_time_data(Structure):
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 661
class struct_localeinfo_struct(Structure):
    pass

struct_localeinfo_struct.__slots__ = [
    'locinfo',
    'mbcinfo',
]
struct_localeinfo_struct._fields_ = [
    ('locinfo', pthreadlocinfo),
    ('mbcinfo', pthreadmbcinfo),
]

_locale_tstruct = struct_localeinfo_struct # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 661

_locale_t = POINTER(struct_localeinfo_struct) # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 661

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 669
class struct_tagLC_ID(Structure):
    pass

struct_tagLC_ID.__slots__ = [
    'wLanguage',
    'wCountry',
    'wCodePage',
]
struct_tagLC_ID._fields_ = [
    ('wLanguage', c_ushort),
    ('wCountry', c_ushort),
    ('wCodePage', c_ushort),
]

LC_ID = struct_tagLC_ID # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 669

LPLC_ID = POINTER(struct_tagLC_ID) # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 669

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 680
class struct_anon_1(Structure):
    pass

struct_anon_1.__slots__ = [
    'locale',
    'wlocale',
    'refcount',
    'wrefcount',
]
struct_anon_1._fields_ = [
    ('locale', String),
    ('wlocale', POINTER(c_wchar)),
    ('refcount', POINTER(c_int)),
    ('wrefcount', POINTER(c_int)),
]

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 691
class struct_lconv(Structure):
    pass

struct_threadlocaleinfostruct.__slots__ = [
    'refcount',
    'lc_codepage',
    'lc_collate_cp',
    'lc_handle',
    'lc_id',
    'lc_category',
    'lc_clike',
    'mb_cur_max',
    'lconv_intl_refcount',
    'lconv_num_refcount',
    'lconv_mon_refcount',
    'lconv',
    'ctype1_refcount',
    'ctype1',
    'pctype',
    'pclmap',
    'pcumap',
    'lc_time_curr',
]
struct_threadlocaleinfostruct._fields_ = [
    ('refcount', c_int),
    ('lc_codepage', c_uint),
    ('lc_collate_cp', c_uint),
    ('lc_handle', c_ulong * 6),
    ('lc_id', LC_ID * 6),
    ('lc_category', struct_anon_1 * 6),
    ('lc_clike', c_int),
    ('mb_cur_max', c_int),
    ('lconv_intl_refcount', POINTER(c_int)),
    ('lconv_num_refcount', POINTER(c_int)),
    ('lconv_mon_refcount', POINTER(c_int)),
    ('lconv', POINTER(struct_lconv)),
    ('ctype1_refcount', POINTER(c_int)),
    ('ctype1', POINTER(c_ushort)),
    ('pctype', POINTER(c_ushort)),
    ('pclmap', POINTER(c_ubyte)),
    ('pcumap', POINTER(c_ubyte)),
    ('lc_time_curr', POINTER(struct___lc_time_data)),
]

threadlocinfo = struct_threadlocaleinfostruct # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 698

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 716
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_get_crt_info'):
        continue
    __mingw_get_crt_info = _lib.__mingw_get_crt_info
    __mingw_get_crt_info.argtypes = []
    if sizeof(c_int) == sizeof(c_void_p):
        __mingw_get_crt_info.restype = ReturnString
    else:
        __mingw_get_crt_info.restype = String
        __mingw_get_crt_info.errcheck = ReturnString
    break

_onexit_t = CFUNCTYPE(UNCHECKED(c_int), ) # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 36

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 49
class struct__div_t(Structure):
    pass

struct__div_t.__slots__ = [
    'quot',
    'rem',
]
struct__div_t._fields_ = [
    ('quot', c_int),
    ('rem', c_int),
]

div_t = struct__div_t # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 49

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 54
class struct__ldiv_t(Structure):
    pass

struct__ldiv_t.__slots__ = [
    'quot',
    'rem',
]
struct__ldiv_t._fields_ = [
    ('quot', c_long),
    ('rem', c_long),
]

ldiv_t = struct__ldiv_t # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 54

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 63
class struct_anon_2(Structure):
    pass

struct_anon_2.__slots__ = [
    'ld',
]
struct_anon_2._fields_ = [
    ('ld', c_ubyte * 10),
]

_LDOUBLE = struct_anon_2 # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 63

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 70
class struct_anon_3(Structure):
    pass

struct_anon_3.__slots__ = [
    'x',
]
struct_anon_3._fields_ = [
    ('x', c_double),
]

_CRT_DOUBLE = struct_anon_3 # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 70

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 74
class struct_anon_4(Structure):
    pass

struct_anon_4.__slots__ = [
    'f',
]
struct_anon_4._fields_ = [
    ('f', c_float),
]

_CRT_FLOAT = struct_anon_4 # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 74

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 88
class struct_anon_6(Structure):
    pass

struct_anon_6.__slots__ = [
    'ld12',
]
struct_anon_6._fields_ = [
    ('ld12', c_ubyte * 12),
]

_LDBL12 = struct_anon_6 # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 88

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 101
for _lib in _libs.values():
    try:
        __imp___mb_cur_max = (POINTER(c_int)).in_dll(_lib, '__imp___mb_cur_max')
        break
    except:
        pass

_purecall_handler = CFUNCTYPE(UNCHECKED(None), ) # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 127

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 129
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_set_purecall_handler'):
        continue
    _set_purecall_handler = _lib._set_purecall_handler
    _set_purecall_handler.argtypes = [_purecall_handler]
    _set_purecall_handler.restype = _purecall_handler
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 130
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_purecall_handler'):
        continue
    _get_purecall_handler = _lib._get_purecall_handler
    _get_purecall_handler.argtypes = []
    _get_purecall_handler.restype = _purecall_handler
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 138
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_errno'):
        continue
    _errno = _lib._errno
    _errno.argtypes = []
    _errno.restype = POINTER(c_int)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 140
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_set_errno'):
        continue
    _set_errno = _lib._set_errno
    _set_errno.argtypes = [c_int]
    _set_errno.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 141
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_errno'):
        continue
    _get_errno = _lib._get_errno
    _get_errno.argtypes = [POINTER(c_int)]
    _get_errno.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 143
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__doserrno'):
        continue
    __doserrno = _lib.__doserrno
    __doserrno.argtypes = []
    __doserrno.restype = POINTER(c_ulong)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 145
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_set_doserrno'):
        continue
    _set_doserrno = _lib._set_doserrno
    _set_doserrno.argtypes = [c_ulong]
    _set_doserrno.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 146
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_doserrno'):
        continue
    _get_doserrno = _lib._get_doserrno
    _get_doserrno.argtypes = [POINTER(c_ulong)]
    _get_doserrno.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 151
for _lib in _libs.values():
    try:
        _sys_errlist = (POINTER(c_char) * 1).in_dll(_lib, '_sys_errlist')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 152
for _lib in _libs.values():
    try:
        _sys_nerr = (c_int).in_dll(_lib, '_sys_nerr')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 167
for _lib in _libs.values():
    try:
        __imp___argc = (POINTER(c_int)).in_dll(_lib, '__imp___argc')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 175
for _lib in _libs.values():
    try:
        __imp___argv = (POINTER(POINTER(POINTER(c_char)))).in_dll(_lib, '__imp___argv')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 183
for _lib in _libs.values():
    try:
        __imp___wargv = (POINTER(POINTER(POINTER(c_wchar)))).in_dll(_lib, '__imp___wargv')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 195
for _lib in _libs.values():
    try:
        __imp__environ = (POINTER(POINTER(POINTER(c_char)))).in_dll(_lib, '__imp__environ')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 204
for _lib in _libs.values():
    try:
        __imp__wenviron = (POINTER(POINTER(POINTER(c_wchar)))).in_dll(_lib, '__imp__wenviron')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 213
for _lib in _libs.values():
    try:
        __imp__pgmptr = (POINTER(POINTER(c_char))).in_dll(_lib, '__imp__pgmptr')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 222
for _lib in _libs.values():
    try:
        __imp__wpgmptr = (POINTER(POINTER(c_wchar))).in_dll(_lib, '__imp__wpgmptr')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 226
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_pgmptr'):
        continue
    _get_pgmptr = _lib._get_pgmptr
    _get_pgmptr.argtypes = [POINTER(POINTER(c_char))]
    _get_pgmptr.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 227
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_wpgmptr'):
        continue
    _get_wpgmptr = _lib._get_wpgmptr
    _get_wpgmptr.argtypes = [POINTER(POINTER(c_wchar))]
    _get_wpgmptr.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 232
for _lib in _libs.values():
    try:
        __imp__fmode = (POINTER(c_int)).in_dll(_lib, '__imp__fmode')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 236
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_set_fmode'):
        continue
    _set_fmode = _lib._set_fmode
    _set_fmode.argtypes = [c_int]
    _set_fmode.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 237
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_fmode'):
        continue
    _get_fmode = _lib._get_fmode
    _get_fmode.argtypes = [POINTER(c_int)]
    _get_fmode.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 243
for _lib in _libs.values():
    try:
        __imp__osplatform = (POINTER(c_uint)).in_dll(_lib, '__imp__osplatform')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 252
for _lib in _libs.values():
    try:
        __imp__osver = (POINTER(c_uint)).in_dll(_lib, '__imp__osver')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 261
for _lib in _libs.values():
    try:
        __imp__winver = (POINTER(c_uint)).in_dll(_lib, '__imp__winver')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 270
for _lib in _libs.values():
    try:
        __imp__winmajor = (POINTER(c_uint)).in_dll(_lib, '__imp__winmajor')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 279
for _lib in _libs.values():
    try:
        __imp__winminor = (POINTER(c_uint)).in_dll(_lib, '__imp__winminor')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 284
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_osplatform'):
        continue
    _get_osplatform = _lib._get_osplatform
    _get_osplatform.argtypes = [POINTER(c_uint)]
    _get_osplatform.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 285
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_osver'):
        continue
    _get_osver = _lib._get_osver
    _get_osver.argtypes = [POINTER(c_uint)]
    _get_osver.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 286
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_winver'):
        continue
    _get_winver = _lib._get_winver
    _get_winver.argtypes = [POINTER(c_uint)]
    _get_winver.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 287
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_winmajor'):
        continue
    _get_winmajor = _lib._get_winmajor
    _get_winmajor.argtypes = [POINTER(c_uint)]
    _get_winmajor.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 288
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_winminor'):
        continue
    _get_winminor = _lib._get_winminor
    _get_winminor.argtypes = [POINTER(c_uint)]
    _get_winminor.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 302
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'exit'):
        continue
    exit = _lib.exit
    exit.argtypes = [c_int]
    exit.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 303
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_exit'):
        continue
    _exit = _lib._exit
    _exit.argtypes = [c_int]
    _exit.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 307
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_Exit'):
        continue
    _Exit = _lib._Exit
    _Exit.argtypes = [c_int]
    _Exit.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 316
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'abort'):
        continue
    abort = _lib.abort
    abort.argtypes = []
    abort.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 321
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_set_abort_behavior'):
        continue
    _set_abort_behavior = _lib._set_abort_behavior
    _set_abort_behavior.argtypes = [c_uint, c_uint]
    _set_abort_behavior.restype = c_uint
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 325
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'abs'):
        continue
    abs = _lib.abs
    abs.argtypes = [c_int]
    abs.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 326
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'labs'):
        continue
    labs = _lib.labs
    labs.argtypes = [c_long]
    labs.restype = c_long
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 330
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'atexit'):
        continue
    atexit = _lib.atexit
    atexit.argtypes = [CFUNCTYPE(UNCHECKED(None), )]
    atexit.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 333
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'atof'):
        continue
    atof = _lib.atof
    atof.argtypes = [String]
    atof.restype = c_double
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 334
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_atof_l'):
        continue
    _atof_l = _lib._atof_l
    _atof_l.argtypes = [String, _locale_t]
    _atof_l.restype = c_double
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 336
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'atoi'):
        continue
    atoi = _lib.atoi
    atoi.argtypes = [String]
    atoi.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 337
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_atoi_l'):
        continue
    _atoi_l = _lib._atoi_l
    _atoi_l.argtypes = [String, _locale_t]
    _atoi_l.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 338
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'atol'):
        continue
    atol = _lib.atol
    atol.argtypes = [String]
    atol.restype = c_long
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 339
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_atol_l'):
        continue
    _atol_l = _lib._atol_l
    _atol_l.argtypes = [String, _locale_t]
    _atol_l.restype = c_long
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 342
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'bsearch'):
        continue
    bsearch = _lib.bsearch
    bsearch.argtypes = [POINTER(None), POINTER(None), c_size_t, c_size_t, CFUNCTYPE(UNCHECKED(c_int), POINTER(None), POINTER(None))]
    bsearch.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 343
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'qsort'):
        continue
    qsort = _lib.qsort
    qsort.argtypes = [POINTER(None), c_size_t, c_size_t, CFUNCTYPE(UNCHECKED(c_int), POINTER(None), POINTER(None))]
    qsort.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 345
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_byteswap_ushort'):
        continue
    _byteswap_ushort = _lib._byteswap_ushort
    _byteswap_ushort.argtypes = [c_ushort]
    _byteswap_ushort.restype = c_ushort
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 348
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'div'):
        continue
    div = _lib.div
    div.argtypes = [c_int, c_int]
    div.restype = div_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 349
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'getenv'):
        continue
    getenv = _lib.getenv
    getenv.argtypes = [String]
    if sizeof(c_int) == sizeof(c_void_p):
        getenv.restype = ReturnString
    else:
        getenv.restype = String
        getenv.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 350
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_itoa'):
        continue
    _itoa = _lib._itoa
    _itoa.argtypes = [c_int, String, c_int]
    if sizeof(c_int) == sizeof(c_void_p):
        _itoa.restype = ReturnString
    else:
        _itoa.restype = String
        _itoa.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 359
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ldiv'):
        continue
    ldiv = _lib.ldiv
    ldiv.argtypes = [c_long, c_long]
    ldiv.restype = ldiv_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 360
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_ltoa'):
        continue
    _ltoa = _lib._ltoa
    _ltoa.argtypes = [c_long, String, c_int]
    if sizeof(c_int) == sizeof(c_void_p):
        _ltoa.restype = ReturnString
    else:
        _ltoa.restype = String
        _ltoa.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 361
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'mblen'):
        continue
    mblen = _lib.mblen
    mblen.argtypes = [String, c_size_t]
    mblen.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 362
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_mblen_l'):
        continue
    _mblen_l = _lib._mblen_l
    _mblen_l.argtypes = [String, c_size_t, _locale_t]
    _mblen_l.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 363
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_mbstrlen'):
        continue
    _mbstrlen = _lib._mbstrlen
    _mbstrlen.argtypes = [String]
    _mbstrlen.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 364
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_mbstrlen_l'):
        continue
    _mbstrlen_l = _lib._mbstrlen_l
    _mbstrlen_l.argtypes = [String, _locale_t]
    _mbstrlen_l.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 365
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_mbstrnlen'):
        continue
    _mbstrnlen = _lib._mbstrnlen
    _mbstrnlen.argtypes = [String, c_size_t]
    _mbstrnlen.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 366
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_mbstrnlen_l'):
        continue
    _mbstrnlen_l = _lib._mbstrnlen_l
    _mbstrnlen_l.argtypes = [String, c_size_t, _locale_t]
    _mbstrnlen_l.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 367
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'mbtowc'):
        continue
    mbtowc = _lib.mbtowc
    mbtowc.argtypes = [POINTER(c_wchar), String, c_size_t]
    mbtowc.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 368
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_mbtowc_l'):
        continue
    _mbtowc_l = _lib._mbtowc_l
    _mbtowc_l.argtypes = [POINTER(c_wchar), String, c_size_t, _locale_t]
    _mbtowc_l.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 369
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'mbstowcs'):
        continue
    mbstowcs = _lib.mbstowcs
    mbstowcs.argtypes = [POINTER(c_wchar), String, c_size_t]
    mbstowcs.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 370
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_mbstowcs_l'):
        continue
    _mbstowcs_l = _lib._mbstowcs_l
    _mbstowcs_l.argtypes = [POINTER(c_wchar), String, c_size_t, _locale_t]
    _mbstowcs_l.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 371
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'rand'):
        continue
    rand = _lib.rand
    rand.argtypes = []
    rand.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 372
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_set_error_mode'):
        continue
    _set_error_mode = _lib._set_error_mode
    _set_error_mode.argtypes = [c_int]
    _set_error_mode.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 373
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'srand'):
        continue
    srand = _lib.srand
    srand.argtypes = [c_uint]
    srand.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 377
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'strtod'):
        continue
    strtod = _lib.strtod
    strtod.argtypes = [String, POINTER(POINTER(c_char))]
    strtod.restype = c_double
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 378
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'strtof'):
        continue
    strtof = _lib.strtof
    strtof.argtypes = [String, POINTER(POINTER(c_char))]
    strtof.restype = c_float
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 383
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__strtod'):
        continue
    __strtod = _lib.__strtod
    __strtod.argtypes = [String, POINTER(POINTER(c_char))]
    __strtod.restype = c_double
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 395
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_strtof'):
        continue
    __mingw_strtof = _lib.__mingw_strtof
    __mingw_strtof.argtypes = [String, POINTER(POINTER(c_char))]
    __mingw_strtof.restype = c_float
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 398
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_strtod_l'):
        continue
    _strtod_l = _lib._strtod_l
    _strtod_l.argtypes = [String, POINTER(POINTER(c_char)), _locale_t]
    _strtod_l.restype = c_double
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 399
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'strtol'):
        continue
    strtol = _lib.strtol
    strtol.argtypes = [String, POINTER(POINTER(c_char)), c_int]
    strtol.restype = c_long
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 400
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_strtol_l'):
        continue
    _strtol_l = _lib._strtol_l
    _strtol_l.argtypes = [String, POINTER(POINTER(c_char)), c_int, _locale_t]
    _strtol_l.restype = c_long
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 401
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'strtoul'):
        continue
    strtoul = _lib.strtoul
    strtoul.argtypes = [String, POINTER(POINTER(c_char)), c_int]
    strtoul.restype = c_ulong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 402
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_strtoul_l'):
        continue
    _strtoul_l = _lib._strtoul_l
    _strtoul_l.argtypes = [String, POINTER(POINTER(c_char)), c_int, _locale_t]
    _strtoul_l.restype = c_ulong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 405
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'system'):
        continue
    system = _lib.system
    system.argtypes = [String]
    system.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 407
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_ultoa'):
        continue
    _ultoa = _lib._ultoa
    _ultoa.argtypes = [c_ulong, String, c_int]
    if sizeof(c_int) == sizeof(c_void_p):
        _ultoa.restype = ReturnString
    else:
        _ultoa.restype = String
        _ultoa.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 408
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'wctomb'):
        continue
    wctomb = _lib.wctomb
    wctomb.argtypes = [String, c_wchar]
    wctomb.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 409
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wctomb_l'):
        continue
    _wctomb_l = _lib._wctomb_l
    _wctomb_l.argtypes = [String, c_wchar, _locale_t]
    _wctomb_l.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 410
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'wcstombs'):
        continue
    wcstombs = _lib.wcstombs
    wcstombs.argtypes = [String, POINTER(c_wchar), c_size_t]
    wcstombs.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 411
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wcstombs_l'):
        continue
    _wcstombs_l = _lib._wcstombs_l
    _wcstombs_l.argtypes = [String, POINTER(c_wchar), c_size_t, _locale_t]
    _wcstombs_l.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 415
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'calloc'):
        continue
    calloc = _lib.calloc
    calloc.argtypes = [c_size_t, c_size_t]
    calloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 416
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'free'):
        continue
    free = _lib.free
    free.argtypes = [POINTER(None)]
    free.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 417
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'malloc'):
        continue
    malloc = _lib.malloc
    malloc.argtypes = [c_size_t]
    malloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 418
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'realloc'):
        continue
    realloc = _lib.realloc
    realloc.argtypes = [POINTER(None), c_size_t]
    realloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 419
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_recalloc'):
        continue
    _recalloc = _lib._recalloc
    _recalloc.argtypes = [POINTER(None), c_size_t, c_size_t]
    _recalloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 426
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_aligned_free'):
        continue
    _aligned_free = _lib._aligned_free
    _aligned_free.argtypes = [POINTER(None)]
    _aligned_free.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 427
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_aligned_malloc'):
        continue
    _aligned_malloc = _lib._aligned_malloc
    _aligned_malloc.argtypes = [c_size_t, c_size_t]
    _aligned_malloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 431
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_aligned_offset_malloc'):
        continue
    _aligned_offset_malloc = _lib._aligned_offset_malloc
    _aligned_offset_malloc.argtypes = [c_size_t, c_size_t, c_size_t]
    _aligned_offset_malloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 432
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_aligned_realloc'):
        continue
    _aligned_realloc = _lib._aligned_realloc
    _aligned_realloc.argtypes = [POINTER(None), c_size_t, c_size_t]
    _aligned_realloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 433
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_aligned_recalloc'):
        continue
    _aligned_recalloc = _lib._aligned_recalloc
    _aligned_recalloc.argtypes = [POINTER(None), c_size_t, c_size_t, c_size_t]
    _aligned_recalloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 434
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_aligned_offset_realloc'):
        continue
    _aligned_offset_realloc = _lib._aligned_offset_realloc
    _aligned_offset_realloc.argtypes = [POINTER(None), c_size_t, c_size_t, c_size_t]
    _aligned_offset_realloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 435
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_aligned_offset_recalloc'):
        continue
    _aligned_offset_recalloc = _lib._aligned_offset_recalloc
    _aligned_offset_recalloc.argtypes = [POINTER(None), c_size_t, c_size_t, c_size_t, c_size_t]
    _aligned_offset_recalloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 441
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_itow'):
        continue
    _itow = _lib._itow
    _itow.argtypes = [c_int, POINTER(c_wchar), c_int]
    _itow.restype = POINTER(c_wchar)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 442
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_ltow'):
        continue
    _ltow = _lib._ltow
    _ltow.argtypes = [c_long, POINTER(c_wchar), c_int]
    _ltow.restype = POINTER(c_wchar)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 443
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_ultow'):
        continue
    _ultow = _lib._ultow
    _ultow.argtypes = [c_ulong, POINTER(c_wchar), c_int]
    _ultow.restype = POINTER(c_wchar)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 444
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'wcstod'):
        continue
    wcstod = _lib.wcstod
    wcstod.argtypes = [POINTER(c_wchar), POINTER(POINTER(c_wchar))]
    wcstod.restype = c_double
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 445
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'wcstof'):
        continue
    wcstof = _lib.wcstof
    wcstof.argtypes = [POINTER(c_wchar), POINTER(POINTER(c_wchar))]
    wcstof.restype = c_float
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 447
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_wcstod'):
        continue
    __mingw_wcstod = _lib.__mingw_wcstod
    __mingw_wcstod.argtypes = [POINTER(c_wchar), POINTER(POINTER(c_wchar))]
    __mingw_wcstod.restype = c_double
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 448
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_wcstof'):
        continue
    __mingw_wcstof = _lib.__mingw_wcstof
    __mingw_wcstof.argtypes = [POINTER(c_wchar), POINTER(POINTER(c_wchar))]
    __mingw_wcstof.restype = c_float
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 452
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'wcstof'):
        continue
    wcstof = _lib.wcstof
    wcstof.argtypes = [POINTER(c_wchar), POINTER(POINTER(c_wchar))]
    wcstof.restype = c_float
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 455
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wcstod_l'):
        continue
    _wcstod_l = _lib._wcstod_l
    _wcstod_l.argtypes = [POINTER(c_wchar), POINTER(POINTER(c_wchar)), _locale_t]
    _wcstod_l.restype = c_double
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 456
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'wcstol'):
        continue
    wcstol = _lib.wcstol
    wcstol.argtypes = [POINTER(c_wchar), POINTER(POINTER(c_wchar)), c_int]
    wcstol.restype = c_long
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 457
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wcstol_l'):
        continue
    _wcstol_l = _lib._wcstol_l
    _wcstol_l.argtypes = [POINTER(c_wchar), POINTER(POINTER(c_wchar)), c_int, _locale_t]
    _wcstol_l.restype = c_long
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 458
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'wcstoul'):
        continue
    wcstoul = _lib.wcstoul
    wcstoul.argtypes = [POINTER(c_wchar), POINTER(POINTER(c_wchar)), c_int]
    wcstoul.restype = c_ulong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 459
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wcstoul_l'):
        continue
    _wcstoul_l = _lib._wcstoul_l
    _wcstoul_l.argtypes = [POINTER(c_wchar), POINTER(POINTER(c_wchar)), c_int, _locale_t]
    _wcstoul_l.restype = c_ulong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 460
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wgetenv'):
        continue
    _wgetenv = _lib._wgetenv
    _wgetenv.argtypes = [POINTER(c_wchar)]
    _wgetenv.restype = POINTER(c_wchar)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 463
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wsystem'):
        continue
    _wsystem = _lib._wsystem
    _wsystem.argtypes = [POINTER(c_wchar)]
    _wsystem.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 465
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wtof'):
        continue
    _wtof = _lib._wtof
    _wtof.argtypes = [POINTER(c_wchar)]
    _wtof.restype = c_double
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 466
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wtof_l'):
        continue
    _wtof_l = _lib._wtof_l
    _wtof_l.argtypes = [POINTER(c_wchar), _locale_t]
    _wtof_l.restype = c_double
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 467
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wtoi'):
        continue
    _wtoi = _lib._wtoi
    _wtoi.argtypes = [POINTER(c_wchar)]
    _wtoi.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 468
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wtoi_l'):
        continue
    _wtoi_l = _lib._wtoi_l
    _wtoi_l.argtypes = [POINTER(c_wchar), _locale_t]
    _wtoi_l.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 469
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wtol'):
        continue
    _wtol = _lib._wtol
    _wtol.argtypes = [POINTER(c_wchar)]
    _wtol.restype = c_long
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 470
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wtol_l'):
        continue
    _wtol_l = _lib._wtol_l
    _wtol_l.argtypes = [POINTER(c_wchar), _locale_t]
    _wtol_l.restype = c_long
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 484
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fullpath'):
        continue
    _fullpath = _lib._fullpath
    _fullpath.argtypes = [String, String, c_size_t]
    if sizeof(c_int) == sizeof(c_void_p):
        _fullpath.restype = ReturnString
    else:
        _fullpath.restype = String
        _fullpath.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 485
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_ecvt'):
        continue
    _ecvt = _lib._ecvt
    _ecvt.argtypes = [c_double, c_int, POINTER(c_int), POINTER(c_int)]
    if sizeof(c_int) == sizeof(c_void_p):
        _ecvt.restype = ReturnString
    else:
        _ecvt.restype = String
        _ecvt.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 486
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fcvt'):
        continue
    _fcvt = _lib._fcvt
    _fcvt.argtypes = [c_double, c_int, POINTER(c_int), POINTER(c_int)]
    if sizeof(c_int) == sizeof(c_void_p):
        _fcvt.restype = ReturnString
    else:
        _fcvt.restype = String
        _fcvt.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 487
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_gcvt'):
        continue
    _gcvt = _lib._gcvt
    _gcvt.argtypes = [c_double, c_int, String]
    if sizeof(c_int) == sizeof(c_void_p):
        _gcvt.restype = ReturnString
    else:
        _gcvt.restype = String
        _gcvt.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 488
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_atodbl'):
        continue
    _atodbl = _lib._atodbl
    _atodbl.argtypes = [POINTER(_CRT_DOUBLE), String]
    _atodbl.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 489
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_atoldbl'):
        continue
    _atoldbl = _lib._atoldbl
    _atoldbl.argtypes = [POINTER(_LDOUBLE), String]
    _atoldbl.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 490
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_atoflt'):
        continue
    _atoflt = _lib._atoflt
    _atoflt.argtypes = [POINTER(_CRT_FLOAT), String]
    _atoflt.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 491
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_atodbl_l'):
        continue
    _atodbl_l = _lib._atodbl_l
    _atodbl_l.argtypes = [POINTER(_CRT_DOUBLE), String, _locale_t]
    _atodbl_l.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 492
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_atoldbl_l'):
        continue
    _atoldbl_l = _lib._atoldbl_l
    _atoldbl_l.argtypes = [POINTER(_LDOUBLE), String, _locale_t]
    _atoldbl_l.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 493
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_atoflt_l'):
        continue
    _atoflt_l = _lib._atoflt_l
    _atoflt_l.argtypes = [POINTER(_CRT_FLOAT), String, _locale_t]
    _atoflt_l.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 499
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_lrotl'):
        continue
    _lrotl = _lib._lrotl
    _lrotl.argtypes = [c_ulonglong, c_int]
    _lrotl.restype = c_ulonglong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 500
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_lrotr'):
        continue
    _lrotr = _lib._lrotr
    _lrotr.argtypes = [c_ulonglong, c_int]
    _lrotr.restype = c_ulonglong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 508
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_makepath'):
        continue
    _makepath = _lib._makepath
    _makepath.argtypes = [String, String, String, String, String]
    _makepath.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 509
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_onexit'):
        continue
    _onexit = _lib._onexit
    _onexit.argtypes = [_onexit_t]
    _onexit.restype = _onexit_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 513
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'perror'):
        continue
    perror = _lib.perror
    perror.argtypes = [String]
    perror.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 515
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_putenv'):
        continue
    _putenv = _lib._putenv
    _putenv.argtypes = [String]
    _putenv.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 528
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_rotr'):
        continue
    _rotr = _lib._rotr
    _rotr.argtypes = [c_uint, c_int]
    _rotr.restype = c_uint
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 529
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_rotl'):
        continue
    _rotl = _lib._rotl
    _rotl.argtypes = [c_uint, c_int]
    _rotl.restype = c_uint
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 533
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_searchenv'):
        continue
    _searchenv = _lib._searchenv
    _searchenv.argtypes = [String, String, String]
    _searchenv.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 534
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_splitpath'):
        continue
    _splitpath = _lib._splitpath
    _splitpath.argtypes = [String, String, String, String, String]
    _splitpath.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 535
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_swab'):
        continue
    _swab = _lib._swab
    _swab.argtypes = [String, String, c_int]
    _swab.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 539
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wfullpath'):
        continue
    _wfullpath = _lib._wfullpath
    _wfullpath.argtypes = [POINTER(c_wchar), POINTER(c_wchar), c_size_t]
    _wfullpath.restype = POINTER(c_wchar)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 540
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wmakepath'):
        continue
    _wmakepath = _lib._wmakepath
    _wmakepath.argtypes = [POINTER(c_wchar), POINTER(c_wchar), POINTER(c_wchar), POINTER(c_wchar), POINTER(c_wchar)]
    _wmakepath.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 543
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wperror'):
        continue
    _wperror = _lib._wperror
    _wperror.argtypes = [POINTER(c_wchar)]
    _wperror.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 545
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wputenv'):
        continue
    _wputenv = _lib._wputenv
    _wputenv.argtypes = [POINTER(c_wchar)]
    _wputenv.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 546
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wsearchenv'):
        continue
    _wsearchenv = _lib._wsearchenv
    _wsearchenv.argtypes = [POINTER(c_wchar), POINTER(c_wchar), POINTER(c_wchar)]
    _wsearchenv.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 547
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wsplitpath'):
        continue
    _wsplitpath = _lib._wsplitpath
    _wsplitpath.argtypes = [POINTER(c_wchar), POINTER(c_wchar), POINTER(c_wchar), POINTER(c_wchar), POINTER(c_wchar)]
    _wsplitpath.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 550
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_beep'):
        continue
    _beep = _lib._beep
    _beep.argtypes = [c_uint, c_uint]
    _beep.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 552
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_seterrormode'):
        continue
    _seterrormode = _lib._seterrormode
    _seterrormode.argtypes = [c_int]
    _seterrormode.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 553
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_sleep'):
        continue
    _sleep = _lib._sleep
    _sleep.argtypes = [c_ulong]
    _sleep.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 574
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ecvt'):
        continue
    ecvt = _lib.ecvt
    ecvt.argtypes = [c_double, c_int, POINTER(c_int), POINTER(c_int)]
    if sizeof(c_int) == sizeof(c_void_p):
        ecvt.restype = ReturnString
    else:
        ecvt.restype = String
        ecvt.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 575
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fcvt'):
        continue
    fcvt = _lib.fcvt
    fcvt.argtypes = [c_double, c_int, POINTER(c_int), POINTER(c_int)]
    if sizeof(c_int) == sizeof(c_void_p):
        fcvt.restype = ReturnString
    else:
        fcvt.restype = String
        fcvt.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 576
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gcvt'):
        continue
    gcvt = _lib.gcvt
    gcvt.argtypes = [c_double, c_int, String]
    if sizeof(c_int) == sizeof(c_void_p):
        gcvt.restype = ReturnString
    else:
        gcvt.restype = String
        gcvt.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 577
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'itoa'):
        continue
    itoa = _lib.itoa
    itoa.argtypes = [c_int, String, c_int]
    if sizeof(c_int) == sizeof(c_void_p):
        itoa.restype = ReturnString
    else:
        itoa.restype = String
        itoa.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 578
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ltoa'):
        continue
    ltoa = _lib.ltoa
    ltoa.argtypes = [c_long, String, c_int]
    if sizeof(c_int) == sizeof(c_void_p):
        ltoa.restype = ReturnString
    else:
        ltoa.restype = String
        ltoa.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 579
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'putenv'):
        continue
    putenv = _lib.putenv
    putenv.argtypes = [String]
    putenv.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 583
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'swab'):
        continue
    swab = _lib.swab
    swab.argtypes = [String, String, c_int]
    swab.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 586
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ultoa'):
        continue
    ultoa = _lib.ultoa
    ultoa.argtypes = [c_ulong, String, c_int]
    if sizeof(c_int) == sizeof(c_void_p):
        ultoa.restype = ReturnString
    else:
        ultoa.restype = String
        ultoa.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 587
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'onexit'):
        continue
    onexit = _lib.onexit
    onexit.argtypes = [_onexit_t]
    onexit.restype = _onexit_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 593
class struct_anon_7(Structure):
    pass

struct_anon_7.__slots__ = [
    'quot',
    'rem',
]
struct_anon_7._fields_ = [
    ('quot', c_longlong),
    ('rem', c_longlong),
]

lldiv_t = struct_anon_7 # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 593

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 595
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'lldiv'):
        continue
    lldiv = _lib.lldiv
    lldiv.argtypes = [c_longlong, c_longlong]
    lldiv.restype = lldiv_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 597
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'llabs'):
        continue
    llabs = _lib.llabs
    llabs.argtypes = [c_longlong]
    llabs.restype = c_longlong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 602
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'strtoll'):
        continue
    strtoll = _lib.strtoll
    strtoll.argtypes = [String, POINTER(POINTER(c_char)), c_int]
    strtoll.restype = c_longlong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 603
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'strtoull'):
        continue
    strtoull = _lib.strtoull
    strtoull.argtypes = [String, POINTER(POINTER(c_char)), c_int]
    strtoull.restype = c_ulonglong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 606
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'atoll'):
        continue
    atoll = _lib.atoll
    atoll.argtypes = [String]
    atoll.restype = c_longlong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 609
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'wtoll'):
        continue
    wtoll = _lib.wtoll
    wtoll.argtypes = [POINTER(c_wchar)]
    wtoll.restype = c_longlong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 610
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'lltoa'):
        continue
    lltoa = _lib.lltoa
    lltoa.argtypes = [c_longlong, String, c_int]
    if sizeof(c_int) == sizeof(c_void_p):
        lltoa.restype = ReturnString
    else:
        lltoa.restype = String
        lltoa.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 611
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ulltoa'):
        continue
    ulltoa = _lib.ulltoa
    ulltoa.argtypes = [c_ulonglong, String, c_int]
    if sizeof(c_int) == sizeof(c_void_p):
        ulltoa.restype = ReturnString
    else:
        ulltoa.restype = String
        ulltoa.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 612
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'lltow'):
        continue
    lltow = _lib.lltow
    lltow.argtypes = [c_longlong, POINTER(c_wchar), c_int]
    lltow.restype = POINTER(c_wchar)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 613
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ulltow'):
        continue
    ulltow = _lib.ulltow
    ulltow.argtypes = [c_ulonglong, POINTER(c_wchar), c_int]
    ulltow.restype = POINTER(c_wchar)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 50
class struct__heapinfo(Structure):
    pass

struct__heapinfo.__slots__ = [
    '_pentry',
    '_size',
    '_useflag',
]
struct__heapinfo._fields_ = [
    ('_pentry', POINTER(c_int)),
    ('_size', c_size_t),
    ('_useflag', c_int),
]

_HEAPINFO = struct__heapinfo # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 50

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 53
for _lib in _libs.values():
    try:
        _amblksiz = (c_uint).in_dll(_lib, '_amblksiz')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 103
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_aligned_malloc'):
        continue
    __mingw_aligned_malloc = _lib.__mingw_aligned_malloc
    __mingw_aligned_malloc.argtypes = [c_size_t, c_size_t]
    __mingw_aligned_malloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 104
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_aligned_free'):
        continue
    __mingw_aligned_free = _lib.__mingw_aligned_free
    __mingw_aligned_free.argtypes = [POINTER(None)]
    __mingw_aligned_free.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 105
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_aligned_offset_realloc'):
        continue
    __mingw_aligned_offset_realloc = _lib.__mingw_aligned_offset_realloc
    __mingw_aligned_offset_realloc.argtypes = [POINTER(None), c_size_t, c_size_t, c_size_t]
    __mingw_aligned_offset_realloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 106
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_aligned_realloc'):
        continue
    __mingw_aligned_realloc = _lib.__mingw_aligned_realloc
    __mingw_aligned_realloc.argtypes = [POINTER(None), c_size_t, c_size_t]
    __mingw_aligned_realloc.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 110
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_resetstkoflw'):
        continue
    _resetstkoflw = _lib._resetstkoflw
    _resetstkoflw.argtypes = []
    _resetstkoflw.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 111
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_set_malloc_crt_max_wait'):
        continue
    _set_malloc_crt_max_wait = _lib._set_malloc_crt_max_wait
    _set_malloc_crt_max_wait.argtypes = [c_ulong]
    _set_malloc_crt_max_wait.restype = c_ulong
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 113
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_expand'):
        continue
    _expand = _lib._expand
    _expand.argtypes = [POINTER(None), c_size_t]
    _expand.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 114
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_msize'):
        continue
    _msize = _lib._msize
    _msize.argtypes = [POINTER(None)]
    _msize.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 119
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_alloca'):
        continue
    _alloca = _lib._alloca
    _alloca.argtypes = [c_size_t]
    _alloca.restype = POINTER(None)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 121
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_sbh_threshold'):
        continue
    _get_sbh_threshold = _lib._get_sbh_threshold
    _get_sbh_threshold.argtypes = []
    _get_sbh_threshold.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 122
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_set_sbh_threshold'):
        continue
    _set_sbh_threshold = _lib._set_sbh_threshold
    _set_sbh_threshold.argtypes = [c_size_t]
    _set_sbh_threshold.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 123
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_set_amblksiz'):
        continue
    _set_amblksiz = _lib._set_amblksiz
    _set_amblksiz.argtypes = [c_size_t]
    _set_amblksiz.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 124
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_amblksiz'):
        continue
    _get_amblksiz = _lib._get_amblksiz
    _get_amblksiz.argtypes = [POINTER(c_size_t)]
    _get_amblksiz.restype = errno_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 125
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_heapadd'):
        continue
    _heapadd = _lib._heapadd
    _heapadd.argtypes = [POINTER(None), c_size_t]
    _heapadd.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 126
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_heapchk'):
        continue
    _heapchk = _lib._heapchk
    _heapchk.argtypes = []
    _heapchk.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 127
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_heapmin'):
        continue
    _heapmin = _lib._heapmin
    _heapmin.argtypes = []
    _heapmin.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 128
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_heapset'):
        continue
    _heapset = _lib._heapset
    _heapset.argtypes = [c_uint]
    _heapset.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 129
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_heapwalk'):
        continue
    _heapwalk = _lib._heapwalk
    _heapwalk.argtypes = [POINTER(_HEAPINFO)]
    _heapwalk.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 130
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_heapused'):
        continue
    _heapused = _lib._heapused
    _heapused.argtypes = [POINTER(c_size_t), POINTER(c_size_t)]
    _heapused.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 164
for _lib in _libs.values():
    try:
        _Marker = (c_uint).in_dll(_lib, '_Marker')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 26
class struct__iobuf(Structure):
    pass

struct__iobuf.__slots__ = [
    '_ptr',
    '_cnt',
    '_base',
    '_flag',
    '_file',
    '_charbuf',
    '_bufsiz',
    '_tmpfname',
]
struct__iobuf._fields_ = [
    ('_ptr', String),
    ('_cnt', c_int),
    ('_base', String),
    ('_flag', c_int),
    ('_file', c_int),
    ('_charbuf', c_int),
    ('_bufsiz', c_int),
    ('_tmpfname', String),
]

FILE = struct__iobuf # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 36

_off_t = c_long # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_off_t.h: 5

off32_t = c_long # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_off_t.h: 7

_off64_t = c_longlong # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_off_t.h: 13

off64_t = c_longlong # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_off_t.h: 15

off_t = off32_t # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_off_t.h: 26

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 84
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__iob_func'):
        continue
    __iob_func = _lib.__iob_func
    __iob_func.argtypes = []
    __iob_func.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 141
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_sscanf'):
        _func = _lib.__mingw_sscanf
        _restype = c_int
        _argtypes = [String, String]
        __mingw_sscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 147
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_scanf'):
        _func = _lib.__mingw_scanf
        _restype = c_int
        _argtypes = [String]
        __mingw_scanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 153
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_fscanf'):
        _func = _lib.__mingw_fscanf
        _restype = c_int
        _argtypes = [POINTER(FILE), String]
        __mingw_fscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 164
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_snprintf'):
        _func = _lib.__mingw_snprintf
        _restype = c_int
        _argtypes = [String, c_size_t, String]
        __mingw_snprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 167
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_printf'):
        _func = _lib.__mingw_printf
        _restype = c_int
        _argtypes = [String]
        __mingw_printf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 173
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_fprintf'):
        _func = _lib.__mingw_fprintf
        _restype = c_int
        _argtypes = [POINTER(FILE), String]
        __mingw_fprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 179
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_sprintf'):
        _func = _lib.__mingw_sprintf
        _restype = c_int
        _argtypes = [String, String]
        __mingw_sprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 185
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_asprintf'):
        _func = _lib.__mingw_asprintf
        _restype = c_int
        _argtypes = [POINTER(POINTER(c_char)), String]
        __mingw_asprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 366
for _lib in _libs.values():
    if hasattr(_lib, 'fprintf'):
        _func = _lib.fprintf
        _restype = c_int
        _argtypes = [POINTER(FILE), String]
        fprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 367
for _lib in _libs.values():
    if hasattr(_lib, 'printf'):
        _func = _lib.printf
        _restype = c_int
        _argtypes = [String]
        printf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 368
for _lib in _libs.values():
    if hasattr(_lib, 'sprintf'):
        _func = _lib.sprintf
        _restype = c_int
        _argtypes = [String, String]
        sprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 374
for _lib in _libs.values():
    if hasattr(_lib, 'fscanf'):
        _func = _lib.fscanf
        _restype = c_int
        _argtypes = [POINTER(FILE), String]
        fscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 375
for _lib in _libs.values():
    if hasattr(_lib, 'scanf'):
        _func = _lib.scanf
        _restype = c_int
        _argtypes = [String]
        scanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 376
for _lib in _libs.values():
    if hasattr(_lib, 'sscanf'):
        _func = _lib.sscanf
        _restype = c_int
        _argtypes = [String, String]
        sscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 405
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_filbuf'):
        continue
    _filbuf = _lib._filbuf
    _filbuf.argtypes = [POINTER(FILE)]
    _filbuf.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 406
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_flsbuf'):
        continue
    _flsbuf = _lib._flsbuf
    _flsbuf.argtypes = [c_int, POINTER(FILE)]
    _flsbuf.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 410
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fsopen'):
        continue
    _fsopen = _lib._fsopen
    _fsopen.argtypes = [String, String, c_int]
    _fsopen.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 412
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'clearerr'):
        continue
    clearerr = _lib.clearerr
    clearerr.argtypes = [POINTER(FILE)]
    clearerr.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 413
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fclose'):
        continue
    fclose = _lib.fclose
    fclose.argtypes = [POINTER(FILE)]
    fclose.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 414
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fcloseall'):
        continue
    _fcloseall = _lib._fcloseall
    _fcloseall.argtypes = []
    _fcloseall.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 418
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fdopen'):
        continue
    _fdopen = _lib._fdopen
    _fdopen.argtypes = [c_int, String]
    _fdopen.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 420
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'feof'):
        continue
    feof = _lib.feof
    feof.argtypes = [POINTER(FILE)]
    feof.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 421
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ferror'):
        continue
    ferror = _lib.ferror
    ferror.argtypes = [POINTER(FILE)]
    ferror.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 422
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fflush'):
        continue
    fflush = _lib.fflush
    fflush.argtypes = [POINTER(FILE)]
    fflush.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 423
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fgetc'):
        continue
    fgetc = _lib.fgetc
    fgetc.argtypes = [POINTER(FILE)]
    fgetc.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 424
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fgetchar'):
        continue
    _fgetchar = _lib._fgetchar
    _fgetchar.argtypes = []
    _fgetchar.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 427
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fgets'):
        continue
    fgets = _lib.fgets
    fgets.argtypes = [String, c_int, POINTER(FILE)]
    if sizeof(c_int) == sizeof(c_void_p):
        fgets.restype = ReturnString
    else:
        fgets.restype = String
        fgets.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 428
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fileno'):
        continue
    _fileno = _lib._fileno
    _fileno.argtypes = [POINTER(FILE)]
    _fileno.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 432
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_tempnam'):
        continue
    _tempnam = _lib._tempnam
    _tempnam.argtypes = [String, String]
    if sizeof(c_int) == sizeof(c_void_p):
        _tempnam.restype = ReturnString
    else:
        _tempnam.restype = String
        _tempnam.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 433
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_flushall'):
        continue
    _flushall = _lib._flushall
    _flushall.argtypes = []
    _flushall.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 434
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fopen'):
        continue
    fopen = _lib.fopen
    fopen.argtypes = [String, String]
    fopen.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 435
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fopen64'):
        continue
    fopen64 = _lib.fopen64
    fopen64.argtypes = [String, String]
    fopen64.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 436
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fputc'):
        continue
    fputc = _lib.fputc
    fputc.argtypes = [c_int, POINTER(FILE)]
    fputc.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 437
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fputchar'):
        continue
    _fputchar = _lib._fputchar
    _fputchar.argtypes = [c_int]
    _fputchar.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 438
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fputs'):
        continue
    fputs = _lib.fputs
    fputs.argtypes = [String, POINTER(FILE)]
    fputs.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 439
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fread'):
        continue
    fread = _lib.fread
    fread.argtypes = [POINTER(None), c_size_t, c_size_t, POINTER(FILE)]
    fread.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 440
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'freopen'):
        continue
    freopen = _lib.freopen
    freopen.argtypes = [String, String, POINTER(FILE)]
    freopen.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 441
for _lib in _libs.values():
    if hasattr(_lib, '_fscanf_l'):
        _func = _lib._fscanf_l
        _restype = c_int
        _argtypes = [POINTER(FILE), String, _locale_t]
        _fscanf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 444
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fseek'):
        continue
    fseek = _lib.fseek
    fseek.argtypes = [POINTER(FILE), c_long, c_int]
    fseek.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 448
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fseeko64'):
        continue
    fseeko64 = _lib.fseeko64
    fseeko64.argtypes = [POINTER(FILE), _off64_t, c_int]
    fseeko64.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 449
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fseeko'):
        continue
    fseeko = _lib.fseeko
    fseeko.argtypes = [POINTER(FILE), _off_t, c_int]
    fseeko.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 462
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ftell'):
        continue
    ftell = _lib.ftell
    ftell.argtypes = [POINTER(FILE)]
    ftell.restype = c_long
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 464
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ftello'):
        continue
    ftello = _lib.ftello
    ftello.argtypes = [POINTER(FILE)]
    ftello.restype = _off_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 465
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ftello64'):
        continue
    ftello64 = _lib.ftello64
    ftello64.argtypes = [POINTER(FILE)]
    ftello64.restype = _off64_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 476
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fwrite'):
        continue
    fwrite = _lib.fwrite
    fwrite.argtypes = [POINTER(None), c_size_t, c_size_t, POINTER(FILE)]
    fwrite.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 477
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'getc'):
        continue
    getc = _lib.getc
    getc.argtypes = [POINTER(FILE)]
    getc.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 478
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'getchar'):
        continue
    getchar = _lib.getchar
    getchar.argtypes = []
    getchar.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 479
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_getmaxstdio'):
        continue
    _getmaxstdio = _lib._getmaxstdio
    _getmaxstdio.argtypes = []
    _getmaxstdio.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 480
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'gets'):
        continue
    gets = _lib.gets
    gets.argtypes = [String]
    if sizeof(c_int) == sizeof(c_void_p):
        gets.restype = ReturnString
    else:
        gets.restype = String
        gets.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 481
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_getw'):
        continue
    _getw = _lib._getw
    _getw.argtypes = [POINTER(FILE)]
    _getw.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 486
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_pclose'):
        continue
    _pclose = _lib._pclose
    _pclose.argtypes = [POINTER(FILE)]
    _pclose.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 487
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_popen'):
        continue
    _popen = _lib._popen
    _popen.argtypes = [String, String]
    _popen.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 492
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'putc'):
        continue
    putc = _lib.putc
    putc.argtypes = [c_int, POINTER(FILE)]
    putc.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 493
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'putchar'):
        continue
    putchar = _lib.putchar
    putchar.argtypes = [c_int]
    putchar.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 494
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'puts'):
        continue
    puts = _lib.puts
    puts.argtypes = [String]
    puts.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 495
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_putw'):
        continue
    _putw = _lib._putw
    _putw.argtypes = [c_int, POINTER(FILE)]
    _putw.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 498
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'remove'):
        continue
    remove = _lib.remove
    remove.argtypes = [String]
    remove.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 499
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'rename'):
        continue
    rename = _lib.rename
    rename.argtypes = [String, String]
    rename.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 500
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_unlink'):
        continue
    _unlink = _lib._unlink
    _unlink.argtypes = [String]
    _unlink.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 502
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'unlink'):
        continue
    unlink = _lib.unlink
    unlink.argtypes = [String]
    unlink.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 505
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'rewind'):
        continue
    rewind = _lib.rewind
    rewind.argtypes = [POINTER(FILE)]
    rewind.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 506
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_rmtmp'):
        continue
    _rmtmp = _lib._rmtmp
    _rmtmp.argtypes = []
    _rmtmp.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 507
for _lib in _libs.values():
    if hasattr(_lib, '_scanf_l'):
        _func = _lib._scanf_l
        _restype = c_int
        _argtypes = [String, _locale_t]
        _scanf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 508
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'setbuf'):
        continue
    setbuf = _lib.setbuf
    setbuf.argtypes = [POINTER(FILE), String]
    setbuf.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 509
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_setmaxstdio'):
        continue
    _setmaxstdio = _lib._setmaxstdio
    _setmaxstdio.argtypes = [c_int]
    _setmaxstdio.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 510
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_set_output_format'):
        continue
    _set_output_format = _lib._set_output_format
    _set_output_format.argtypes = [c_uint]
    _set_output_format.restype = c_uint
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 511
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_output_format'):
        continue
    _get_output_format = _lib._get_output_format
    _get_output_format.argtypes = []
    _get_output_format.restype = c_uint
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 512
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_set_output_format'):
        continue
    __mingw_set_output_format = _lib.__mingw_set_output_format
    __mingw_set_output_format.argtypes = [c_uint]
    __mingw_set_output_format.restype = c_uint
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 513
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_get_output_format'):
        continue
    __mingw_get_output_format = _lib.__mingw_get_output_format
    __mingw_get_output_format.argtypes = []
    __mingw_get_output_format.restype = c_uint
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 518
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'setvbuf'):
        continue
    setvbuf = _lib.setvbuf
    setvbuf.argtypes = [POINTER(FILE), String, c_int, c_size_t]
    setvbuf.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 519
for _lib in _libs.values():
    if hasattr(_lib, '_scprintf'):
        _func = _lib._scprintf
        _restype = c_int
        _argtypes = [String]
        _scprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 520
for _lib in _libs.values():
    if hasattr(_lib, '_sscanf_l'):
        _func = _lib._sscanf_l
        _restype = c_int
        _argtypes = [String, String, _locale_t]
        _sscanf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 521
for _lib in _libs.values():
    if hasattr(_lib, '_snscanf'):
        _func = _lib._snscanf
        _restype = c_int
        _argtypes = [String, c_size_t, String]
        _snscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 522
for _lib in _libs.values():
    if hasattr(_lib, '_snscanf_l'):
        _func = _lib._snscanf_l
        _restype = c_int
        _argtypes = [String, c_size_t, String, _locale_t]
        _snscanf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 523
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'tmpfile'):
        continue
    tmpfile = _lib.tmpfile
    tmpfile.argtypes = []
    tmpfile.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 524
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'tmpnam'):
        continue
    tmpnam = _lib.tmpnam
    tmpnam.argtypes = [String]
    if sizeof(c_int) == sizeof(c_void_p):
        tmpnam.restype = ReturnString
    else:
        tmpnam.restype = String
        tmpnam.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 525
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ungetc'):
        continue
    ungetc = _lib.ungetc
    ungetc.argtypes = [c_int, POINTER(FILE)]
    ungetc.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 527
for _lib in _libs.values():
    if hasattr(_lib, '_snprintf'):
        _func = _lib._snprintf
        _restype = c_int
        _argtypes = [String, c_size_t, String]
        _snprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 528
for _lib in _libs.values():
    if hasattr(_lib, '_snprintf_l'):
        _func = _lib._snprintf_l
        _restype = c_int
        _argtypes = [String, c_size_t, String, _locale_t]
        _snprintf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 531
for _lib in _libs.values():
    if hasattr(_lib, '_sprintf_l'):
        _func = _lib._sprintf_l
        _restype = c_int
        _argtypes = [String, String, _locale_t]
        _sprintf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 550
for _lib in _libs.values():
    if hasattr(_lib, '__ms_snprintf'):
        _func = _lib.__ms_snprintf
        _restype = c_int
        _argtypes = [String, c_size_t, String]
        __ms_snprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 557
for _lib in _libs.values():
    try:
        __retval = (c_int).in_dll(_lib, '__retval')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 570
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_set_printf_count_output'):
        continue
    _set_printf_count_output = _lib._set_printf_count_output
    _set_printf_count_output.argtypes = [c_int]
    _set_printf_count_output.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 571
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_get_printf_count_output'):
        continue
    _get_printf_count_output = _lib._get_printf_count_output
    _get_printf_count_output.argtypes = []
    _get_printf_count_output.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 577
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_swscanf'):
        _func = _lib.__mingw_swscanf
        _restype = c_int
        _argtypes = [POINTER(c_wchar), POINTER(c_wchar)]
        __mingw_swscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 581
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_wscanf'):
        _func = _lib.__mingw_wscanf
        _restype = c_int
        _argtypes = [POINTER(c_wchar)]
        __mingw_wscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 585
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_fwscanf'):
        _func = _lib.__mingw_fwscanf
        _restype = c_int
        _argtypes = [POINTER(FILE), POINTER(c_wchar)]
        __mingw_fwscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 590
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_fwprintf'):
        _func = _lib.__mingw_fwprintf
        _restype = c_int
        _argtypes = [POINTER(FILE), POINTER(c_wchar)]
        __mingw_fwprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 592
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_wprintf'):
        _func = _lib.__mingw_wprintf
        _restype = c_int
        _argtypes = [POINTER(c_wchar)]
        __mingw_wprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 598
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_swprintf'):
        _func = _lib.__mingw_swprintf
        _restype = c_int
        _argtypes = [POINTER(c_wchar), POINTER(c_wchar)]
        __mingw_swprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 602
for _lib in _libs.values():
    if hasattr(_lib, '__mingw_snwprintf'):
        _func = _lib.__mingw_snwprintf
        _restype = c_int
        _argtypes = [POINTER(c_wchar), c_size_t, POINTER(c_wchar)]
        __mingw_snwprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 754
for _lib in _libs.values():
    if hasattr(_lib, 'fwscanf'):
        _func = _lib.fwscanf
        _restype = c_int
        _argtypes = [POINTER(FILE), POINTER(c_wchar)]
        fwscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 755
for _lib in _libs.values():
    if hasattr(_lib, 'swscanf'):
        _func = _lib.swscanf
        _restype = c_int
        _argtypes = [POINTER(c_wchar), POINTER(c_wchar)]
        swscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 756
for _lib in _libs.values():
    if hasattr(_lib, 'wscanf'):
        _func = _lib.wscanf
        _restype = c_int
        _argtypes = [POINTER(c_wchar)]
        wscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 784
for _lib in _libs.values():
    if hasattr(_lib, 'fwprintf'):
        _func = _lib.fwprintf
        _restype = c_int
        _argtypes = [POINTER(FILE), POINTER(c_wchar)]
        fwprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 785
for _lib in _libs.values():
    if hasattr(_lib, 'wprintf'):
        _func = _lib.wprintf
        _restype = c_int
        _argtypes = [POINTER(c_wchar)]
        wprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 788
for _lib in _libs.values():
    if hasattr(_lib, 'swprintf'):
        _func = _lib.swprintf
        _restype = c_int
        _argtypes = [POINTER(c_wchar), POINTER(c_wchar)]
        swprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 799
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wfsopen'):
        continue
    _wfsopen = _lib._wfsopen
    _wfsopen.argtypes = [POINTER(c_wchar), POINTER(c_wchar), c_int]
    _wfsopen.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 802
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fgetwc'):
        continue
    fgetwc = _lib.fgetwc
    fgetwc.argtypes = [POINTER(FILE)]
    fgetwc.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 803
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fgetwchar'):
        continue
    _fgetwchar = _lib._fgetwchar
    _fgetwchar.argtypes = []
    _fgetwchar.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 804
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fputwc'):
        continue
    fputwc = _lib.fputwc
    fputwc.argtypes = [c_wchar, POINTER(FILE)]
    fputwc.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 805
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fputwchar'):
        continue
    _fputwchar = _lib._fputwchar
    _fputwchar.argtypes = [c_wchar]
    _fputwchar.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 806
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'getwc'):
        continue
    getwc = _lib.getwc
    getwc.argtypes = [POINTER(FILE)]
    getwc.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 807
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'getwchar'):
        continue
    getwchar = _lib.getwchar
    getwchar.argtypes = []
    getwchar.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 808
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'putwc'):
        continue
    putwc = _lib.putwc
    putwc.argtypes = [c_wchar, POINTER(FILE)]
    putwc.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 809
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'putwchar'):
        continue
    putwchar = _lib.putwchar
    putwchar.argtypes = [c_wchar]
    putwchar.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 810
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'ungetwc'):
        continue
    ungetwc = _lib.ungetwc
    ungetwc.argtypes = [wint_t, POINTER(FILE)]
    ungetwc.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 811
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fgetws'):
        continue
    fgetws = _lib.fgetws
    fgetws.argtypes = [POINTER(c_wchar), c_int, POINTER(FILE)]
    fgetws.restype = POINTER(c_wchar)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 812
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fputws'):
        continue
    fputws = _lib.fputws
    fputws.argtypes = [POINTER(c_wchar), POINTER(FILE)]
    fputws.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 813
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_getws'):
        continue
    _getws = _lib._getws
    _getws.argtypes = [POINTER(c_wchar)]
    _getws.restype = POINTER(c_wchar)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 814
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_putws'):
        continue
    _putws = _lib._putws
    _putws.argtypes = [POINTER(c_wchar)]
    _putws.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 816
for _lib in _libs.values():
    if hasattr(_lib, '_scwprintf'):
        _func = _lib._scwprintf
        _restype = c_int
        _argtypes = [POINTER(c_wchar)]
        _scwprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 817
for _lib in _libs.values():
    if hasattr(_lib, '_swprintf_l'):
        _func = _lib._swprintf_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), c_size_t, POINTER(c_wchar), _locale_t]
        _swprintf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 818
for _lib in _libs.values():
    if hasattr(_lib, '_swprintf_c'):
        _func = _lib._swprintf_c
        _restype = c_int
        _argtypes = [POINTER(c_wchar), c_size_t, POINTER(c_wchar)]
        _swprintf_c = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 820
for _lib in _libs.values():
    if hasattr(_lib, '_snwprintf'):
        _func = _lib._snwprintf
        _restype = c_int
        _argtypes = [POINTER(c_wchar), c_size_t, POINTER(c_wchar)]
        _snwprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 830
for _lib in _libs.values():
    if hasattr(_lib, '__ms_snwprintf'):
        _func = _lib.__ms_snwprintf
        _restype = c_int
        _argtypes = [POINTER(c_wchar), c_size_t, POINTER(c_wchar)]
        __ms_snwprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 835
for _lib in _libs.values():
    try:
        r = (c_int).in_dll(_lib, 'r')
        break
    except:
        pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 852
for _lib in _libs.values():
    if hasattr(_lib, '_fwprintf_p'):
        _func = _lib._fwprintf_p
        _restype = c_int
        _argtypes = [POINTER(FILE), POINTER(c_wchar)]
        _fwprintf_p = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 853
for _lib in _libs.values():
    if hasattr(_lib, '_wprintf_p'):
        _func = _lib._wprintf_p
        _restype = c_int
        _argtypes = [POINTER(c_wchar)]
        _wprintf_p = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 856
for _lib in _libs.values():
    if hasattr(_lib, '_swprintf_p'):
        _func = _lib._swprintf_p
        _restype = c_int
        _argtypes = [POINTER(c_wchar), c_size_t, POINTER(c_wchar)]
        _swprintf_p = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 858
for _lib in _libs.values():
    if hasattr(_lib, '_scwprintf_p'):
        _func = _lib._scwprintf_p
        _restype = c_int
        _argtypes = [POINTER(c_wchar)]
        _scwprintf_p = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 860
for _lib in _libs.values():
    if hasattr(_lib, '_wprintf_l'):
        _func = _lib._wprintf_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), _locale_t]
        _wprintf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 861
for _lib in _libs.values():
    if hasattr(_lib, '_wprintf_p_l'):
        _func = _lib._wprintf_p_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), _locale_t]
        _wprintf_p_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 864
for _lib in _libs.values():
    if hasattr(_lib, '_fwprintf_l'):
        _func = _lib._fwprintf_l
        _restype = c_int
        _argtypes = [POINTER(FILE), POINTER(c_wchar), _locale_t]
        _fwprintf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 865
for _lib in _libs.values():
    if hasattr(_lib, '_fwprintf_p_l'):
        _func = _lib._fwprintf_p_l
        _restype = c_int
        _argtypes = [POINTER(FILE), POINTER(c_wchar), _locale_t]
        _fwprintf_p_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 868
for _lib in _libs.values():
    if hasattr(_lib, '_swprintf_c_l'):
        _func = _lib._swprintf_c_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), c_size_t, POINTER(c_wchar), _locale_t]
        _swprintf_c_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 869
for _lib in _libs.values():
    if hasattr(_lib, '_swprintf_p_l'):
        _func = _lib._swprintf_p_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), c_size_t, POINTER(c_wchar), _locale_t]
        _swprintf_p_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 872
for _lib in _libs.values():
    if hasattr(_lib, '_scwprintf_l'):
        _func = _lib._scwprintf_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), _locale_t]
        _scwprintf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 873
for _lib in _libs.values():
    if hasattr(_lib, '_scwprintf_p_l'):
        _func = _lib._scwprintf_p_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), _locale_t]
        _scwprintf_p_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 875
for _lib in _libs.values():
    if hasattr(_lib, '_snwprintf_l'):
        _func = _lib._snwprintf_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), c_size_t, POINTER(c_wchar), _locale_t]
        _snwprintf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 877
for _lib in _libs.values():
    if hasattr(_lib, '_swprintf'):
        _func = _lib._swprintf
        _restype = c_int
        _argtypes = [POINTER(c_wchar), POINTER(c_wchar)]
        _swprintf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 879
for _lib in _libs.values():
    if hasattr(_lib, '__swprintf_l'):
        _func = _lib.__swprintf_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), POINTER(c_wchar), _locale_t]
        __swprintf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 892
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wtempnam'):
        continue
    _wtempnam = _lib._wtempnam
    _wtempnam.argtypes = [POINTER(c_wchar), POINTER(c_wchar)]
    _wtempnam.restype = POINTER(c_wchar)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 895
for _lib in _libs.values():
    if hasattr(_lib, '_fwscanf_l'):
        _func = _lib._fwscanf_l
        _restype = c_int
        _argtypes = [POINTER(FILE), POINTER(c_wchar), _locale_t]
        _fwscanf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 896
for _lib in _libs.values():
    if hasattr(_lib, '_swscanf_l'):
        _func = _lib._swscanf_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), POINTER(c_wchar), _locale_t]
        _swscanf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 897
for _lib in _libs.values():
    if hasattr(_lib, '_snwscanf'):
        _func = _lib._snwscanf
        _restype = c_int
        _argtypes = [POINTER(c_wchar), c_size_t, POINTER(c_wchar)]
        _snwscanf = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 898
for _lib in _libs.values():
    if hasattr(_lib, '_snwscanf_l'):
        _func = _lib._snwscanf_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), c_size_t, POINTER(c_wchar), _locale_t]
        _snwscanf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 899
for _lib in _libs.values():
    if hasattr(_lib, '_wscanf_l'):
        _func = _lib._wscanf_l
        _restype = c_int
        _argtypes = [POINTER(c_wchar), _locale_t]
        _wscanf_l = _variadic_function(_func,_restype,_argtypes)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 900
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wfdopen'):
        continue
    _wfdopen = _lib._wfdopen
    _wfdopen.argtypes = [c_int, POINTER(c_wchar)]
    _wfdopen.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 901
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wfopen'):
        continue
    _wfopen = _lib._wfopen
    _wfopen.argtypes = [POINTER(c_wchar), POINTER(c_wchar)]
    _wfopen.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 902
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wfreopen'):
        continue
    _wfreopen = _lib._wfreopen
    _wfreopen.argtypes = [POINTER(c_wchar), POINTER(c_wchar), POINTER(FILE)]
    _wfreopen.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 908
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wpopen'):
        continue
    _wpopen = _lib._wpopen
    _wpopen.argtypes = [POINTER(c_wchar), POINTER(c_wchar)]
    _wpopen.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 913
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wremove'):
        continue
    _wremove = _lib._wremove
    _wremove.argtypes = [POINTER(c_wchar)]
    _wremove.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 914
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_wtmpnam'):
        continue
    _wtmpnam = _lib._wtmpnam
    _wtmpnam.argtypes = [POINTER(c_wchar)]
    _wtmpnam.restype = POINTER(c_wchar)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 915
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fgetwc_nolock'):
        continue
    _fgetwc_nolock = _lib._fgetwc_nolock
    _fgetwc_nolock.argtypes = [POINTER(FILE)]
    _fgetwc_nolock.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 916
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fputwc_nolock'):
        continue
    _fputwc_nolock = _lib._fputwc_nolock
    _fputwc_nolock.argtypes = [c_wchar, POINTER(FILE)]
    _fputwc_nolock.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 917
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_ungetwc_nolock'):
        continue
    _ungetwc_nolock = _lib._ungetwc_nolock
    _ungetwc_nolock.argtypes = [wint_t, POINTER(FILE)]
    _ungetwc_nolock.restype = wint_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 947
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_lock_file'):
        continue
    _lock_file = _lib._lock_file
    _lock_file.argtypes = [POINTER(FILE)]
    _lock_file.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 948
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_unlock_file'):
        continue
    _unlock_file = _lib._unlock_file
    _unlock_file.argtypes = [POINTER(FILE)]
    _unlock_file.restype = None
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 949
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fclose_nolock'):
        continue
    _fclose_nolock = _lib._fclose_nolock
    _fclose_nolock.argtypes = [POINTER(FILE)]
    _fclose_nolock.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 950
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fflush_nolock'):
        continue
    _fflush_nolock = _lib._fflush_nolock
    _fflush_nolock.argtypes = [POINTER(FILE)]
    _fflush_nolock.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 951
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fread_nolock'):
        continue
    _fread_nolock = _lib._fread_nolock
    _fread_nolock.argtypes = [POINTER(None), c_size_t, c_size_t, POINTER(FILE)]
    _fread_nolock.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 952
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fseek_nolock'):
        continue
    _fseek_nolock = _lib._fseek_nolock
    _fseek_nolock.argtypes = [POINTER(FILE), c_long, c_int]
    _fseek_nolock.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 953
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_ftell_nolock'):
        continue
    _ftell_nolock = _lib._ftell_nolock
    _ftell_nolock.argtypes = [POINTER(FILE)]
    _ftell_nolock.restype = c_long
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 956
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_fwrite_nolock'):
        continue
    _fwrite_nolock = _lib._fwrite_nolock
    _fwrite_nolock.argtypes = [POINTER(None), c_size_t, c_size_t, POINTER(FILE)]
    _fwrite_nolock.restype = c_size_t
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 957
for _lib in _libs.itervalues():
    if not hasattr(_lib, '_ungetc_nolock'):
        continue
    _ungetc_nolock = _lib._ungetc_nolock
    _ungetc_nolock.argtypes = [c_int, POINTER(FILE)]
    _ungetc_nolock.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 963
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'tempnam'):
        continue
    tempnam = _lib.tempnam
    tempnam.argtypes = [String, String]
    if sizeof(c_int) == sizeof(c_void_p):
        tempnam.restype = ReturnString
    else:
        tempnam.restype = String
        tempnam.errcheck = ReturnString
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 964
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fcloseall'):
        continue
    fcloseall = _lib.fcloseall
    fcloseall.argtypes = []
    fcloseall.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 965
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fdopen'):
        continue
    fdopen = _lib.fdopen
    fdopen.argtypes = [c_int, String]
    fdopen.restype = POINTER(FILE)
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 966
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fgetchar'):
        continue
    fgetchar = _lib.fgetchar
    fgetchar.argtypes = []
    fgetchar.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 967
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fileno'):
        continue
    fileno = _lib.fileno
    fileno.argtypes = [POINTER(FILE)]
    fileno.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 968
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'flushall'):
        continue
    flushall = _lib.flushall
    flushall.argtypes = []
    flushall.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 969
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'fputchar'):
        continue
    fputchar = _lib.fputchar
    fputchar.argtypes = [c_int]
    fputchar.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 970
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'getw'):
        continue
    getw = _lib.getw
    getw.argtypes = [POINTER(FILE)]
    getw.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 971
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'putw'):
        continue
    putw = _lib.putw
    putw.argtypes = [c_int, POINTER(FILE)]
    putw.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 972
for _lib in _libs.itervalues():
    if not hasattr(_lib, 'rmtmp'):
        continue
    rmtmp = _lib.rmtmp
    rmtmp.argtypes = []
    rmtmp.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 989
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_str_wide_utf8'):
        continue
    __mingw_str_wide_utf8 = _lib.__mingw_str_wide_utf8
    __mingw_str_wide_utf8.argtypes = [POINTER(c_wchar), POINTER(POINTER(c_char)), POINTER(c_size_t)]
    __mingw_str_wide_utf8.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 1003
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_str_utf8_wide'):
        continue
    __mingw_str_utf8_wide = _lib.__mingw_str_utf8_wide
    __mingw_str_utf8_wide.argtypes = [String, POINTER(POINTER(c_wchar)), POINTER(c_size_t)]
    __mingw_str_utf8_wide.restype = c_int
    break

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 1012
for _lib in _libs.itervalues():
    if not hasattr(_lib, '__mingw_str_free'):
        continue
    __mingw_str_free = _lib.__mingw_str_free
    __mingw_str_free.argtypes = [POINTER(None)]
    __mingw_str_free.restype = None
    break

enum_MSKsolveform_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 46

MSK_SOLVE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 46

MSK_SOLVE_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 46

MSK_SOLVE_FREE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 46

MSK_SOLVE_PRIMAL = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 46

MSK_SOLVE_DUAL = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 46

enum_MSKproblemitem_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 55

MSK_PI_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 55

MSK_PI_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 55

MSK_PI_VAR = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 55

MSK_PI_CON = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 55

MSK_PI_CONE = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 55

enum_MSKaccmode_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 64

MSK_ACC_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 64

MSK_ACC_END = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 64

MSK_ACC_VAR = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 64

MSK_ACC_CON = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 64

enum_MSKsensitivitytype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 72

MSK_SENSITIVITY_TYPE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 72

MSK_SENSITIVITY_TYPE_END = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 72

MSK_SENSITIVITY_TYPE_BASIS = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 72

MSK_SENSITIVITY_TYPE_OPTIMAL_PARTITION = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 72

enum_MSKintpnthotstart_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 80

MSK_INTPNT_HOTSTART_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 80

MSK_INTPNT_HOTSTART_END = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 80

MSK_INTPNT_HOTSTART_NONE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 80

MSK_INTPNT_HOTSTART_PRIMAL = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 80

MSK_INTPNT_HOTSTART_DUAL = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 80

MSK_INTPNT_HOTSTART_PRIMAL_DUAL = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 80

enum_MSKsparam_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_END = 26 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_BAS_SOL_FILE_NAME = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_DATA_FILE_NAME = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_DEBUG_FILE_NAME = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_FEASREPAIR_NAME_PREFIX = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_FEASREPAIR_NAME_SEPARATOR = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_FEASREPAIR_NAME_WSUMVIOL = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_INT_SOL_FILE_NAME = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_ITR_SOL_FILE_NAME = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_MIO_DEBUG_STRING = 8 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_PARAM_COMMENT_SIGN = 9 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_PARAM_READ_FILE_NAME = 10 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_PARAM_WRITE_FILE_NAME = 11 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_READ_MPS_BOU_NAME = 12 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_READ_MPS_OBJ_NAME = 13 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_READ_MPS_RAN_NAME = 14 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_READ_MPS_RHS_NAME = 15 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_SENSITIVITY_FILE_NAME = 16 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_SENSITIVITY_RES_FILE_NAME = 17 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_SOL_FILTER_XC_LOW = 18 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_SOL_FILTER_XC_UPR = 19 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_SOL_FILTER_XX_LOW = 20 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_SOL_FILTER_XX_UPR = 21 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_STAT_FILE_NAME = 22 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_STAT_KEY = 23 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_STAT_NAME = 24 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

MSK_SPAR_WRITE_LP_GEN_VAR_NAME = 25 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 90

enum_MSKiparam_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_END = 203 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_ALLOC_ADD_QNZ = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_ANA_SOL_BASIS = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_ANA_SOL_PRINT_VIOLATED = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_AUTO_SORT_A_BEFORE_OPT = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_AUTO_UPDATE_SOL_INFO = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_BASIS_SOLVE_USE_PLUS_ONE = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_BI_CLEAN_OPTIMIZER = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_BI_IGNORE_MAX_ITER = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_BI_IGNORE_NUM_ERROR = 8 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_BI_MAX_ITERATIONS = 9 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_CACHE_LICENSE = 10 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_CHECK_CONVEXITY = 11 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_COMPRESS_STATFILE = 12 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_CONCURRENT_NUM_OPTIMIZERS = 13 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_CONCURRENT_PRIORITY_DUAL_SIMPLEX = 14 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_CONCURRENT_PRIORITY_FREE_SIMPLEX = 15 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_CONCURRENT_PRIORITY_INTPNT = 16 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_CONCURRENT_PRIORITY_PRIMAL_SIMPLEX = 17 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_FEASREPAIR_OPTIMIZE = 18 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INFEAS_GENERIC_NAMES = 19 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INFEAS_PREFER_PRIMAL = 20 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INFEAS_REPORT_AUTO = 21 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INFEAS_REPORT_LEVEL = 22 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_BASIS = 23 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_DIFF_STEP = 24 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_FACTOR_DEBUG_LVL = 25 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_FACTOR_METHOD = 26 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_HOTSTART = 27 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_MAX_ITERATIONS = 28 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_MAX_NUM_COR = 29 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_MAX_NUM_REFINEMENT_STEPS = 30 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_OFF_COL_TRH = 31 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_ORDER_METHOD = 32 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_REGULARIZATION_USE = 33 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_SCALING = 34 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_SOLVE_FORM = 35 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_INTPNT_STARTING_POINT = 36 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LIC_TRH_EXPIRY_WRN = 37 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LICENSE_ALLOW_OVERUSE = 38 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LICENSE_DEBUG = 39 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LICENSE_PAUSE_TIME = 40 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LICENSE_SUPPRESS_EXPIRE_WRNS = 41 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LICENSE_WAIT = 42 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG = 43 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_BI = 44 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_BI_FREQ = 45 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_CHECK_CONVEXITY = 46 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_CONCURRENT = 47 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_CUT_SECOND_OPT = 48 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_EXPAND = 49 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_FACTOR = 50 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_FEAS_REPAIR = 51 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_FILE = 52 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_HEAD = 53 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_INFEAS_ANA = 54 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_INTPNT = 55 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_MIO = 56 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_MIO_FREQ = 57 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_NONCONVEX = 58 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_OPTIMIZER = 59 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_ORDER = 60 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_PARAM = 61 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_PRESOLVE = 62 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_RESPONSE = 63 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_SENSITIVITY = 64 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_SENSITIVITY_OPT = 65 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_SIM = 66 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_SIM_FREQ = 67 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_SIM_MINOR = 68 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_SIM_NETWORK_FREQ = 69 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_LOG_STORAGE = 70 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MAX_NUM_WARNINGS = 71 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_BRANCH_DIR = 72 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_BRANCH_PRIORITIES_USE = 73 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_CONSTRUCT_SOL = 74 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_CONT_SOL = 75 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_CUT_LEVEL_ROOT = 76 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_CUT_LEVEL_TREE = 77 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_FEASPUMP_LEVEL = 78 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_HEURISTIC_LEVEL = 79 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_HOTSTART = 80 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_KEEP_BASIS = 81 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_LOCAL_BRANCH_NUMBER = 82 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_MAX_NUM_BRANCHES = 83 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_MAX_NUM_RELAXS = 84 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_MAX_NUM_SOLUTIONS = 85 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_MODE = 86 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_MT_USER_CB = 87 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_NODE_OPTIMIZER = 88 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_NODE_SELECTION = 89 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_OPTIMIZER_MODE = 90 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_PRESOLVE_AGGREGATE = 91 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_PRESOLVE_PROBING = 92 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_PRESOLVE_USE = 93 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_ROOT_OPTIMIZER = 94 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_STRONG_BRANCH = 95 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MIO_USE_MULTITHREADED_OPTIMIZER = 96 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_MT_SPINCOUNT = 97 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_NONCONVEX_MAX_ITERATIONS = 98 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_NUM_THREADS = 99 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_OPF_MAX_TERMS_PER_LINE = 100 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_OPF_WRITE_HEADER = 101 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_OPF_WRITE_HINTS = 102 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_OPF_WRITE_PARAMETERS = 103 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_OPF_WRITE_PROBLEM = 104 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_OPF_WRITE_SOL_BAS = 105 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_OPF_WRITE_SOL_ITG = 106 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_OPF_WRITE_SOL_ITR = 107 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_OPF_WRITE_SOLUTIONS = 108 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_OPTIMIZER = 109 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PARAM_READ_CASE_NAME = 110 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PARAM_READ_IGN_ERROR = 111 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PRESOLVE_ELIM_FILL = 112 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES = 113 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PRESOLVE_ELIMINATOR_USE = 114 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PRESOLVE_LEVEL = 115 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PRESOLVE_LINDEP_ABS_WORK_TRH = 116 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PRESOLVE_LINDEP_REL_WORK_TRH = 117 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PRESOLVE_LINDEP_USE = 118 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PRESOLVE_MAX_NUM_REDUCTIONS = 119 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PRESOLVE_USE = 120 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_PRIMAL_REPAIR_OPTIMIZER = 121 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_QO_SEPARABLE_REFORMULATION = 122 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_ANZ = 123 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_CON = 124 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_CONE = 125 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_DATA_COMPRESSED = 126 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_DATA_FORMAT = 127 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_KEEP_FREE_CON = 128 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_LP_DROP_NEW_VARS_IN_BOU = 129 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_LP_QUOTED_NAMES = 130 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_MPS_FORMAT = 131 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_MPS_KEEP_INT = 132 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_MPS_OBJ_SENSE = 133 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_MPS_RELAX = 134 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_MPS_WIDTH = 135 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_QNZ = 136 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_TASK_IGNORE_PARAM = 137 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_READ_VAR = 138 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SENSITIVITY_ALL = 139 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SENSITIVITY_OPTIMIZER = 140 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SENSITIVITY_TYPE = 141 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_BASIS_FACTOR_USE = 142 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_DEGEN = 143 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_DUAL_CRASH = 144 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_DUAL_PHASEONE_METHOD = 145 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_DUAL_RESTRICT_SELECTION = 146 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_DUAL_SELECTION = 147 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_EXPLOIT_DUPVEC = 148 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_HOTSTART = 149 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_HOTSTART_LU = 150 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_INTEGER = 151 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_MAX_ITERATIONS = 152 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_MAX_NUM_SETBACKS = 153 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_NON_SINGULAR = 154 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_PRIMAL_CRASH = 155 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_PRIMAL_PHASEONE_METHOD = 156 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_PRIMAL_RESTRICT_SELECTION = 157 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_PRIMAL_SELECTION = 158 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_REFACTOR_FREQ = 159 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_REFORMULATION = 160 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_SAVE_LU = 161 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_SCALING = 162 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_SCALING_METHOD = 163 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_SOLVE_FORM = 164 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_STABILITY_PRIORITY = 165 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SIM_SWITCH_OPTIMIZER = 166 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SOL_FILTER_KEEP_BASIC = 167 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SOL_FILTER_KEEP_RANGED = 168 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SOL_READ_NAME_WIDTH = 169 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SOL_READ_WIDTH = 170 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_SOLUTION_CALLBACK = 171 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_TIMING_LEVEL = 172 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WARNING_LEVEL = 173 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_BAS_CONSTRAINTS = 174 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_BAS_HEAD = 175 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_BAS_VARIABLES = 176 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_DATA_COMPRESSED = 177 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_DATA_FORMAT = 178 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_DATA_PARAM = 179 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_FREE_CON = 180 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_GENERIC_NAMES = 181 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_GENERIC_NAMES_IO = 182 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_CONIC_ITEMS = 183 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_ITEMS = 184 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_NL_ITEMS = 185 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_PSD_ITEMS = 186 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_INT_CONSTRAINTS = 187 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_INT_HEAD = 188 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_INT_VARIABLES = 189 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_LP_LINE_WIDTH = 190 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_LP_QUOTED_NAMES = 191 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_LP_STRICT_FORMAT = 192 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_LP_TERMS_PER_LINE = 193 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_MPS_INT = 194 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_PRECISION = 195 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_SOL_BARVARIABLES = 196 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_SOL_CONSTRAINTS = 197 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_SOL_HEAD = 198 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_SOL_IGNORE_INVALID_NAMES = 199 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_SOL_VARIABLES = 200 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_TASK_INC_SOL = 201 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

MSK_IPAR_WRITE_XML_MODE = 202 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 122

enum_MSKsolsta_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_END = 16 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_UNKNOWN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_OPTIMAL = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_PRIM_FEAS = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_DUAL_FEAS = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_PRIM_AND_DUAL_FEAS = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_PRIM_INFEAS_CER = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_DUAL_INFEAS_CER = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_NEAR_OPTIMAL = 8 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_NEAR_PRIM_FEAS = 9 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_NEAR_DUAL_FEAS = 10 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_NEAR_PRIM_AND_DUAL_FEAS = 11 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_NEAR_PRIM_INFEAS_CER = 12 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_NEAR_DUAL_INFEAS_CER = 13 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_INTEGER_OPTIMAL = 14 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

MSK_SOL_STA_NEAR_INTEGER_OPTIMAL = 15 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 331

enum_MSKobjsense_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 352

MSK_OBJECTIVE_SENSE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 352

MSK_OBJECTIVE_SENSE_END = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 352

MSK_OBJECTIVE_SENSE_MINIMIZE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 352

MSK_OBJECTIVE_SENSE_MAXIMIZE = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 352

enum_MSKsolitem_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 360

MSK_SOL_ITEM_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 360

MSK_SOL_ITEM_END = 8 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 360

MSK_SOL_ITEM_XC = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 360

MSK_SOL_ITEM_XX = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 360

MSK_SOL_ITEM_Y = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 360

MSK_SOL_ITEM_SLC = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 360

MSK_SOL_ITEM_SUC = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 360

MSK_SOL_ITEM_SLX = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 360

MSK_SOL_ITEM_SUX = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 360

MSK_SOL_ITEM_SNX = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 360

enum_MSKboundkey_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 374

MSK_BK_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 374

MSK_BK_END = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 374

MSK_BK_LO = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 374

MSK_BK_UP = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 374

MSK_BK_FX = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 374

MSK_BK_FR = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 374

MSK_BK_RA = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 374

enum_MSKbasindtype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 385

MSK_BI_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 385

MSK_BI_END = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 385

MSK_BI_NEVER = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 385

MSK_BI_ALWAYS = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 385

MSK_BI_NO_ERROR = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 385

MSK_BI_IF_FEASIBLE = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 385

MSK_BI_RESERVERED = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 385

enum_MSKbranchdir_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 396

MSK_BRANCH_DIR_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 396

MSK_BRANCH_DIR_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 396

MSK_BRANCH_DIR_FREE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 396

MSK_BRANCH_DIR_UP = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 396

MSK_BRANCH_DIR_DOWN = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 396

enum_MSKliinfitem_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_END = 14 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_BI_CLEAN_DUAL_DEG_ITER = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_BI_CLEAN_DUAL_ITER = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_BI_CLEAN_PRIMAL_DEG_ITER = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_BI_CLEAN_PRIMAL_DUAL_DEG_ITER = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_BI_CLEAN_PRIMAL_DUAL_ITER = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_BI_CLEAN_PRIMAL_DUAL_SUB_ITER = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_BI_CLEAN_PRIMAL_ITER = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_BI_DUAL_ITER = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_BI_PRIMAL_ITER = 8 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_INTPNT_FACTOR_NUM_NZ = 9 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_MIO_INTPNT_ITER = 10 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_MIO_SIMPLEX_ITER = 11 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_RD_NUMANZ = 12 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

MSK_LIINF_RD_NUMQNZ = 13 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 405

enum_MSKstreamtype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 425

MSK_STREAM_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 425

MSK_STREAM_END = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 425

MSK_STREAM_LOG = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 425

MSK_STREAM_MSG = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 425

MSK_STREAM_ERR = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 425

MSK_STREAM_WRN = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 425

enum_MSKsimhotstart_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 435

MSK_SIM_HOTSTART_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 435

MSK_SIM_HOTSTART_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 435

MSK_SIM_HOTSTART_NONE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 435

MSK_SIM_HOTSTART_FREE = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 435

MSK_SIM_HOTSTART_STATUS_KEYS = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 435

enum_MSKcallbackcode_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END = 114 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_BI = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_CONCURRENT = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_CONIC = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_DUAL_BI = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_DUAL_SENSITIVITY = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_DUAL_SETUP_BI = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_DUAL_SIMPLEX = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_DUAL_SIMPLEX_BI = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_FULL_CONVEXITY_CHECK = 8 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_INFEAS_ANA = 9 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_INTPNT = 10 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_LICENSE_WAIT = 11 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_MIO = 12 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_NETWORK_DUAL_SIMPLEX = 13 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_NETWORK_PRIMAL_SIMPLEX = 14 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_NETWORK_SIMPLEX = 15 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_NONCONVEX = 16 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_OPTIMIZER = 17 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_PRESOLVE = 18 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_PRIMAL_BI = 19 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_PRIMAL_DUAL_SIMPLEX = 20 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_PRIMAL_DUAL_SIMPLEX_BI = 21 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_PRIMAL_REPAIR = 22 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_PRIMAL_SENSITIVITY = 23 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_PRIMAL_SETUP_BI = 24 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_PRIMAL_SIMPLEX = 25 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_PRIMAL_SIMPLEX_BI = 26 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_QCQO_REFORMULATE = 27 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_READ = 28 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_SIMPLEX = 29 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_SIMPLEX_BI = 30 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_SIMPLEX_NETWORK_DETECT = 31 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_BEGIN_WRITE = 32 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_CONIC = 33 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_DUAL_SIMPLEX = 34 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_BI = 35 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_CONCURRENT = 36 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_CONIC = 37 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_DUAL_BI = 38 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_DUAL_SENSITIVITY = 39 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_DUAL_SETUP_BI = 40 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_DUAL_SIMPLEX = 41 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_DUAL_SIMPLEX_BI = 42 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_FULL_CONVEXITY_CHECK = 43 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_INFEAS_ANA = 44 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_INTPNT = 45 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_LICENSE_WAIT = 46 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_MIO = 47 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_NETWORK_DUAL_SIMPLEX = 48 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_NETWORK_PRIMAL_SIMPLEX = 49 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_NETWORK_SIMPLEX = 50 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_NONCONVEX = 51 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_OPTIMIZER = 52 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_PRESOLVE = 53 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_PRIMAL_BI = 54 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_PRIMAL_DUAL_SIMPLEX = 55 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_PRIMAL_DUAL_SIMPLEX_BI = 56 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_PRIMAL_REPAIR = 57 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_PRIMAL_SENSITIVITY = 58 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_PRIMAL_SETUP_BI = 59 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_PRIMAL_SIMPLEX = 60 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_PRIMAL_SIMPLEX_BI = 61 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_QCQO_REFORMULATE = 62 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_READ = 63 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_SIMPLEX = 64 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_SIMPLEX_BI = 65 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_SIMPLEX_NETWORK_DETECT = 66 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_END_WRITE = 67 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_BI = 68 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_CONIC = 69 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_DUAL_BI = 70 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_DUAL_SENSIVITY = 71 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_DUAL_SIMPLEX = 72 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_FULL_CONVEXITY_CHECK = 73 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_INTPNT = 74 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_LICENSE_WAIT = 75 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_LU = 76 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_MIO = 77 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_MIO_DUAL_SIMPLEX = 78 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_MIO_INTPNT = 79 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_MIO_PRESOLVE = 80 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_MIO_PRIMAL_SIMPLEX = 81 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_NETWORK_DUAL_SIMPLEX = 82 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_NETWORK_PRIMAL_SIMPLEX = 83 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_NONCONVEX = 84 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_ORDER = 85 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_PRESOLVE = 86 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_PRIMAL_BI = 87 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_PRIMAL_DUAL_SIMPLEX = 88 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_PRIMAL_SENSIVITY = 89 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_PRIMAL_SIMPLEX = 90 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_QO_REFORMULATE = 91 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_READ = 92 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_SIMPLEX = 93 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_IM_SIMPLEX_BI = 94 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_INTPNT = 95 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_NEW_INT_MIO = 96 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_NONCOVEX = 97 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_PRIMAL_SIMPLEX = 98 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_READ_OPF = 99 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_READ_OPF_SECTION = 100 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_DUAL_BI = 101 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_DUAL_SIMPLEX = 102 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_DUAL_SIMPLEX_BI = 103 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_NETWORK_DUAL_SIMPLEX = 104 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_NETWORK_PRIMAL_SIMPLEX = 105 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_NONCONVEX = 106 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_PRESOLVE = 107 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_PRIMAL_BI = 108 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_PRIMAL_DUAL_SIMPLEX = 109 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_PRIMAL_DUAL_SIMPLEX_BI = 110 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_PRIMAL_SIMPLEX = 111 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_UPDATE_PRIMAL_SIMPLEX_BI = 112 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

MSK_CALLBACK_WRITE_OPF = 113 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 444

enum_MSKsymmattype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 564

MSK_SYMMAT_TYPE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 564

MSK_SYMMAT_TYPE_END = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 564

MSK_SYMMAT_TYPE_SPARSE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 564

enum_MSKfeature_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 571

MSK_FEATURE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 571

MSK_FEATURE_END = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 571

MSK_FEATURE_PTS = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 571

MSK_FEATURE_PTON = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 571

MSK_FEATURE_PTOM = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 571

MSK_FEATURE_PTOX = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 571

enum_MSKmark_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 581

MSK_MARK_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 581

MSK_MARK_END = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 581

MSK_MARK_LO = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 581

MSK_MARK_UP = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 581

enum_MSKconetype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 589

MSK_CT_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 589

MSK_CT_END = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 589

MSK_CT_QUAD = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 589

MSK_CT_RQUAD = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 589

enum_MSKfeasrepairtype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 597

MSK_FEASREPAIR_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 597

MSK_FEASREPAIR_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 597

MSK_FEASREPAIR_OPTIMIZE_NONE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 597

MSK_FEASREPAIR_OPTIMIZE_PENALTY = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 597

MSK_FEASREPAIR_OPTIMIZE_COMBINED = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 597

enum_MSKiomode_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 606

MSK_IOMODE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 606

MSK_IOMODE_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 606

MSK_IOMODE_READ = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 606

MSK_IOMODE_WRITE = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 606

MSK_IOMODE_READWRITE = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 606

enum_MSKsimseltype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 615

MSK_SIM_SELECTION_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 615

MSK_SIM_SELECTION_END = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 615

MSK_SIM_SELECTION_FREE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 615

MSK_SIM_SELECTION_FULL = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 615

MSK_SIM_SELECTION_ASE = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 615

MSK_SIM_SELECTION_DEVEX = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 615

MSK_SIM_SELECTION_SE = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 615

MSK_SIM_SELECTION_PARTIAL = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 615

enum_MSKmsgkey_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 627

MSK_MSG_READING_FILE = 1000 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 627

MSK_MSG_WRITING_FILE = 1001 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 627

MSK_MSG_MPS_SELECTED = 1100 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 627

enum_MSKmiomode_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 633

MSK_MIO_MODE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 633

MSK_MIO_MODE_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 633

MSK_MIO_MODE_IGNORED = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 633

MSK_MIO_MODE_SATISFIED = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 633

MSK_MIO_MODE_LAZY = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 633

enum_MSKdinfitem_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_END = 65 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_BI_CLEAN_DUAL_TIME = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_BI_CLEAN_PRIMAL_DUAL_TIME = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_BI_CLEAN_PRIMAL_TIME = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_BI_CLEAN_TIME = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_BI_DUAL_TIME = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_BI_PRIMAL_TIME = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_BI_TIME = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_CONCURRENT_TIME = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_INTPNT_DUAL_FEAS = 8 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_INTPNT_DUAL_OBJ = 9 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_INTPNT_FACTOR_NUM_FLOPS = 10 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_INTPNT_OPT_STATUS = 11 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_INTPNT_ORDER_TIME = 12 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_INTPNT_PRIMAL_FEAS = 13 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_INTPNT_PRIMAL_OBJ = 14 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_INTPNT_TIME = 15 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_MIO_CONSTRUCT_SOLUTION_OBJ = 16 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_MIO_HEURISTIC_TIME = 17 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_MIO_OBJ_ABS_GAP = 18 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_MIO_OBJ_BOUND = 19 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_MIO_OBJ_INT = 20 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_MIO_OBJ_REL_GAP = 21 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_MIO_OPTIMIZER_TIME = 22 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_MIO_ROOT_OPTIMIZER_TIME = 23 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_MIO_ROOT_PRESOLVE_TIME = 24 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_MIO_TIME = 25 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_MIO_USER_OBJ_CUT = 26 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_OPTIMIZER_TIME = 27 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_PRESOLVE_ELI_TIME = 28 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_PRESOLVE_LINDEP_TIME = 29 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_PRESOLVE_TIME = 30 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_PRIMAL_REPAIR_PENALTY_OBJ = 31 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_QCQO_REFORMULATE_TIME = 32 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_RD_TIME = 33 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SIM_DUAL_TIME = 34 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SIM_FEAS = 35 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SIM_NETWORK_DUAL_TIME = 36 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SIM_NETWORK_PRIMAL_TIME = 37 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SIM_NETWORK_TIME = 38 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SIM_OBJ = 39 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SIM_PRIMAL_DUAL_TIME = 40 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SIM_PRIMAL_TIME = 41 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SIM_TIME = 42 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_BAS_DUAL_OBJ = 43 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_BAS_DVIOLCON = 44 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_BAS_DVIOLVAR = 45 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_BAS_PRIMAL_OBJ = 46 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_BAS_PVIOLCON = 47 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_BAS_PVIOLVAR = 48 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITG_PRIMAL_OBJ = 49 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITG_PVIOLBARVAR = 50 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITG_PVIOLCON = 51 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITG_PVIOLCONES = 52 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITG_PVIOLITG = 53 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITG_PVIOLVAR = 54 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITR_DUAL_OBJ = 55 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITR_DVIOLBARVAR = 56 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITR_DVIOLCON = 57 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITR_DVIOLCONES = 58 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITR_DVIOLVAR = 59 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITR_PRIMAL_OBJ = 60 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITR_PVIOLBARVAR = 61 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITR_PVIOLCON = 62 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITR_PVIOLCONES = 63 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

MSK_DINF_SOL_ITR_PVIOLVAR = 64 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 642

enum_MSKparametertype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 713

MSK_PAR_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 713

MSK_PAR_END = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 713

MSK_PAR_INVALID_TYPE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 713

MSK_PAR_DOU_TYPE = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 713

MSK_PAR_INT_TYPE = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 713

MSK_PAR_STR_TYPE = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 713

enum_MSKrescodetype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 723

MSK_RESPONSE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 723

MSK_RESPONSE_END = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 723

MSK_RESPONSE_OK = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 723

MSK_RESPONSE_WRN = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 723

MSK_RESPONSE_TRM = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 723

MSK_RESPONSE_ERR = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 723

MSK_RESPONSE_UNK = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 723

enum_MSKprosta_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_END = 12 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_UNKNOWN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_PRIM_AND_DUAL_FEAS = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_PRIM_FEAS = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_DUAL_FEAS = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_PRIM_INFEAS = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_DUAL_INFEAS = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_PRIM_AND_DUAL_INFEAS = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_ILL_POSED = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_NEAR_PRIM_AND_DUAL_FEAS = 8 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_NEAR_PRIM_FEAS = 9 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_NEAR_DUAL_FEAS = 10 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

MSK_PRO_STA_PRIM_INFEAS_OR_UNBOUNDED = 11 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 734

enum_MSKscalingtype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 752

MSK_SCALING_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 752

MSK_SCALING_END = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 752

MSK_SCALING_FREE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 752

MSK_SCALING_NONE = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 752

MSK_SCALING_MODERATE = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 752

MSK_SCALING_AGGRESSIVE = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 752

enum_MSKrescode_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_OK = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_OPEN_PARAM_FILE = 50 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_LARGE_BOUND = 51 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_LARGE_LO_BOUND = 52 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_LARGE_UP_BOUND = 53 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_LARGE_CON_FX = 54 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_LARGE_CJ = 57 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_LARGE_AIJ = 62 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_ZERO_AIJ = 63 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_NAME_MAX_LEN = 65 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_SPAR_MAX_LEN = 66 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_MPS_SPLIT_RHS_VECTOR = 70 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_MPS_SPLIT_RAN_VECTOR = 71 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_MPS_SPLIT_BOU_VECTOR = 72 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_LP_OLD_QUAD_FORMAT = 80 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_LP_DROP_VARIABLE = 85 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_NZ_IN_UPR_TRI = 200 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_DROPPED_NZ_QOBJ = 201 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_IGNORE_INTEGER = 250 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_NO_GLOBAL_OPTIMIZER = 251 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_MIO_INFEASIBLE_FINAL = 270 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_SOL_FILTER = 300 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_UNDEF_SOL_FILE_NAME = 350 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_SOL_FILE_IGNORED_CON = 351 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_SOL_FILE_IGNORED_VAR = 352 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_TOO_FEW_BASIS_VARS = 400 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_TOO_MANY_BASIS_VARS = 405 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_NO_NONLINEAR_FUNCTION_WRITE = 450 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_LICENSE_EXPIRE = 500 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_LICENSE_SERVER = 501 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_EMPTY_NAME = 502 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_USING_GENERIC_NAMES = 503 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_LICENSE_FEATURE_EXPIRE = 505 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_PARAM_NAME_DOU = 510 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_PARAM_NAME_INT = 511 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_PARAM_NAME_STR = 512 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_PARAM_STR_VALUE = 515 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_PARAM_IGNORED_CMIO = 516 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_ZEROS_IN_SPARSE_ROW = 705 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_ZEROS_IN_SPARSE_COL = 710 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_TOO_MANY_THREADS_CONCURRENT = 750 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_INCOMPLETE_LINEAR_DEPENDENCY_CHECK = 800 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_ELIMINATOR_SPACE = 801 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_PRESOLVE_OUTOFSPACE = 802 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_WRITE_CHANGED_NAMES = 803 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_WRITE_DISCARDED_CFIX = 804 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_CONSTRUCT_SOLUTION_INFEAS = 805 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_CONSTRUCT_INVALID_SOL_ITG = 807 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_CONSTRUCT_NO_SOL_ITG = 810 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_DUPLICATE_CONSTRAINT_NAMES = 850 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_DUPLICATE_VARIABLE_NAMES = 851 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_DUPLICATE_BARVARIABLE_NAMES = 852 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_DUPLICATE_CONE_NAMES = 853 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_ANA_LARGE_BOUNDS = 900 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_ANA_C_ZERO = 901 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_ANA_EMPTY_COLS = 902 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_ANA_CLOSE_BOUNDS = 903 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_ANA_ALMOST_INT_BOUNDS = 904 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_QUAD_CONES_WITH_ROOT_FIXED_AT_ZERO = 930 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_RQUAD_CONES_WITH_ROOT_FIXED_AT_ZERO = 931 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_WRN_NO_DUALIZER = 950 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE = 1000 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_EXPIRED = 1001 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_VERSION = 1002 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SIZE_LICENSE = 1005 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PROB_LICENSE = 1006 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FILE_LICENSE = 1007 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MISSING_LICENSE_FILE = 1008 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SIZE_LICENSE_CON = 1010 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SIZE_LICENSE_VAR = 1011 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SIZE_LICENSE_INTVAR = 1012 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_OPTIMIZER_LICENSE = 1013 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FLEXLM = 1014 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_SERVER = 1015 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_MAX = 1016 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_MOSEKLM_DAEMON = 1017 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_FEATURE = 1018 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PLATFORM_NOT_LICENSED = 1019 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_CANNOT_ALLOCATE = 1020 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_CANNOT_CONNECT = 1021 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_INVALID_HOSTID = 1025 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_SERVER_VERSION = 1026 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_NO_SERVER_SUPPORT = 1027 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LICENSE_NO_SERVER_LINE = 1028 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_OPEN_DL = 1030 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_OLDER_DLL = 1035 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NEWER_DLL = 1036 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LINK_FILE_DLL = 1040 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_THREAD_MUTEX_INIT = 1045 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_THREAD_MUTEX_LOCK = 1046 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_THREAD_MUTEX_UNLOCK = 1047 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_THREAD_CREATE = 1048 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_THREAD_COND_INIT = 1049 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_UNKNOWN = 1050 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SPACE = 1051 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FILE_OPEN = 1052 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FILE_READ = 1053 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FILE_WRITE = 1054 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_DATA_FILE_EXT = 1055 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_FILE_NAME = 1056 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_SOL_FILE_NAME = 1057 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_END_OF_FILE = 1059 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NULL_ENV = 1060 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NULL_TASK = 1061 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_STREAM = 1062 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NO_INIT_ENV = 1063 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_TASK = 1064 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NULL_POINTER = 1065 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LIVING_TASKS = 1066 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_BLANK_NAME = 1070 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_DUP_NAME = 1071 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_OBJ_NAME = 1075 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_CON_NAME = 1076 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_VAR_NAME = 1077 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_CONE_NAME = 1078 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_BARVAR_NAME = 1079 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SPACE_LEAKING = 1080 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SPACE_NO_INFO = 1081 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_READ_FORMAT = 1090 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_FILE = 1100 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_INV_FIELD = 1101 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_INV_MARKER = 1102 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_NULL_CON_NAME = 1103 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_NULL_VAR_NAME = 1104 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_UNDEF_CON_NAME = 1105 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_UNDEF_VAR_NAME = 1106 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_INV_CON_KEY = 1107 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_INV_BOUND_KEY = 1108 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_INV_SEC_NAME = 1109 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_NO_OBJECTIVE = 1110 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_SPLITTED_VAR = 1111 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_MUL_CON_NAME = 1112 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_MUL_QSEC = 1113 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_MUL_QOBJ = 1114 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_INV_SEC_ORDER = 1115 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_MUL_CSEC = 1116 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_CONE_TYPE = 1117 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_CONE_OVERLAP = 1118 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_CONE_REPEAT = 1119 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_INVALID_OBJSENSE = 1122 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_TAB_IN_FIELD2 = 1125 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_TAB_IN_FIELD3 = 1126 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_TAB_IN_FIELD5 = 1127 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MPS_INVALID_OBJ_NAME = 1128 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_ORD_INVALID_BRANCH_DIR = 1130 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_ORD_INVALID = 1131 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LP_INCOMPATIBLE = 1150 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LP_EMPTY = 1151 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LP_DUP_SLACK_NAME = 1152 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_WRITE_MPS_INVALID_NAME = 1153 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LP_INVALID_VAR_NAME = 1154 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LP_FREE_CONSTRAINT = 1155 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_WRITE_OPF_INVALID_VAR_NAME = 1156 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LP_FILE_FORMAT = 1157 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_WRITE_LP_FORMAT = 1158 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_READ_LP_MISSING_END_TAG = 1159 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LP_FORMAT = 1160 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_WRITE_LP_NON_UNIQUE_NAME = 1161 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_READ_LP_NONEXISTING_NAME = 1162 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LP_WRITE_CONIC_PROBLEM = 1163 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LP_WRITE_GECO_PROBLEM = 1164 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_WRITING_FILE = 1166 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_OPF_FORMAT = 1168 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_OPF_NEW_VARIABLE = 1169 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_NAME_IN_SOL_FILE = 1170 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LP_INVALID_CON_NAME = 1171 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_OPF_PREMATURE_EOF = 1172 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_ARGUMENT_LENNEQ = 1197 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_ARGUMENT_TYPE = 1198 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NR_ARGUMENTS = 1199 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_IN_ARGUMENT = 1200 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_ARGUMENT_DIMENSION = 1201 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INDEX_IS_TOO_SMALL = 1203 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INDEX_IS_TOO_LARGE = 1204 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PARAM_NAME = 1205 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PARAM_NAME_DOU = 1206 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PARAM_NAME_INT = 1207 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PARAM_NAME_STR = 1208 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PARAM_INDEX = 1210 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PARAM_IS_TOO_LARGE = 1215 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PARAM_IS_TOO_SMALL = 1216 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PARAM_VALUE_STR = 1217 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PARAM_TYPE = 1218 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INF_DOU_INDEX = 1219 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INF_INT_INDEX = 1220 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INDEX_ARR_IS_TOO_SMALL = 1221 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INDEX_ARR_IS_TOO_LARGE = 1222 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INF_LINT_INDEX = 1225 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_ARG_IS_TOO_SMALL = 1226 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_ARG_IS_TOO_LARGE = 1227 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_WHICHSOL = 1228 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INF_DOU_NAME = 1230 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INF_INT_NAME = 1231 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INF_TYPE = 1232 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INF_LINT_NAME = 1234 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INDEX = 1235 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_WHICHSOL = 1236 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SOLITEM = 1237 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_WHICHITEM_NOT_ALLOWED = 1238 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MAXNUMCON = 1240 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MAXNUMVAR = 1241 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MAXNUMBARVAR = 1242 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MAXNUMQNZ = 1243 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_TOO_SMALL_MAX_NUM_NZ = 1245 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_IDX = 1246 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_MAX_NUM = 1247 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NUMCONLIM = 1250 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NUMVARLIM = 1251 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_TOO_SMALL_MAXNUMANZ = 1252 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_APTRE = 1253 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MUL_A_ELEMENT = 1254 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_BK = 1255 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_BKC = 1256 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_BKX = 1257 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_VAR_TYPE = 1258 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SOLVER_PROBTYPE = 1259 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_OBJECTIVE_RANGE = 1260 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FIRST = 1261 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LAST = 1262 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NEGATIVE_SURPLUS = 1263 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NEGATIVE_APPEND = 1264 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_UNDEF_SOLUTION = 1265 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_BASIS = 1266 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_SKC = 1267 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_SKX = 1268 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_SK_STR = 1269 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_SK = 1270 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_CONE_TYPE_STR = 1271 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_CONE_TYPE = 1272 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_SKN = 1274 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_SURPLUS = 1275 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_NAME_ITEM = 1280 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_PRO_ITEM = 1281 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_FORMAT_TYPE = 1283 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FIRSTI = 1285 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LASTI = 1286 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FIRSTJ = 1287 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LASTJ = 1288 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MAX_LEN_IS_TOO_SMALL = 1289 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NONLINEAR_EQUALITY = 1290 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NONCONVEX = 1291 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NONLINEAR_RANGED = 1292 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CON_Q_NOT_PSD = 1293 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CON_Q_NOT_NSD = 1294 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_OBJ_Q_NOT_PSD = 1295 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_OBJ_Q_NOT_NSD = 1296 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_ARGUMENT_PERM_ARRAY = 1299 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CONE_INDEX = 1300 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CONE_SIZE = 1301 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CONE_OVERLAP = 1302 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CONE_REP_VAR = 1303 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MAXNUMCONE = 1304 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CONE_TYPE = 1305 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CONE_TYPE_STR = 1306 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CONE_OVERLAP_APPEND = 1307 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_REMOVE_CONE_VARIABLE = 1310 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SOL_FILE_INVALID_NUMBER = 1350 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_HUGE_C = 1375 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_HUGE_AIJ = 1380 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LOWER_BOUND_IS_A_NAN = 1390 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_UPPER_BOUND_IS_A_NAN = 1391 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INFINITE_BOUND = 1400 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_QOBJ_SUBI = 1401 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_QOBJ_SUBJ = 1402 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_QOBJ_VAL = 1403 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_QCON_SUBK = 1404 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_QCON_SUBI = 1405 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_QCON_SUBJ = 1406 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_QCON_VAL = 1407 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_QCON_SUBI_TOO_SMALL = 1408 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_QCON_SUBI_TOO_LARGE = 1409 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_QOBJ_UPPER_TRIANGLE = 1415 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_QCON_UPPER_TRIANGLE = 1417 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FIXED_BOUND_VALUES = 1425 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NONLINEAR_FUNCTIONS_NOT_ALLOWED = 1428 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_USER_FUNC_RET = 1430 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_USER_FUNC_RET_DATA = 1431 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_USER_NLO_FUNC = 1432 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_USER_NLO_EVAL = 1433 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_USER_NLO_EVAL_HESSUBI = 1440 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_USER_NLO_EVAL_HESSUBJ = 1441 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_OBJECTIVE_SENSE = 1445 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_UNDEFINED_OBJECTIVE_SENSE = 1446 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_Y_IS_UNDEFINED = 1449 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NAN_IN_DOUBLE_DATA = 1450 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NAN_IN_BLC = 1461 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NAN_IN_BUC = 1462 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NAN_IN_C = 1470 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NAN_IN_BLX = 1471 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NAN_IN_BUX = 1472 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NAN_IN_AIJ = 1473 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_PROBLEM = 1500 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MIXED_PROBLEM = 1501 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_CONIC_PROBLEM = 1502 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_GLOBAL_INV_CONIC_PROBLEM = 1503 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_NETWORK_PROBLEM = 1504 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_OPTIMIZER = 1550 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MIO_NO_OPTIMIZER = 1551 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NO_OPTIMIZER_VAR_TYPE = 1552 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MIO_NOT_LOADED = 1553 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_POSTSOLVE = 1580 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_OVERFLOW = 1590 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NO_BASIS_SOL = 1600 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_BASIS_FACTOR = 1610 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_BASIS_SINGULAR = 1615 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FACTOR = 1650 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FEASREPAIR_CANNOT_RELAX = 1700 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FEASREPAIR_SOLVING_RELAXED = 1701 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_FEASREPAIR_INCONSISTENT_BOUND = 1702 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_REPAIR_INVALID_PROBLEM = 1710 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_REPAIR_OPTIMIZATION_FAILED = 1711 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NAME_MAX_LEN = 1750 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NAME_IS_NULL = 1760 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_COMPRESSION = 1800 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_IOMODE = 1801 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NO_PRIMAL_INFEAS_CER = 2000 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NO_DUAL_INFEAS_CER = 2001 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NO_SOLUTION_IN_CALLBACK = 2500 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_MARKI = 2501 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_MARKJ = 2502 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_NUMI = 2503 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INV_NUMJ = 2504 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CANNOT_CLONE_NL = 2505 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CANNOT_HANDLE_NL = 2506 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_ACCMODE = 2520 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MBT_INCOMPATIBLE = 2550 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MBT_INVALID = 2551 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_TASK_INCOMPATIBLE = 2560 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_TASK_INVALID = 2561 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_LU_MAX_NUM_TRIES = 2800 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_UTF8 = 2900 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_WCHAR = 2901 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NO_DUAL_FOR_ITG_SOL = 2950 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NO_SNX_FOR_BAS_SOL = 2953 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INTERNAL = 3000 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_API_ARRAY_TOO_SMALL = 3001 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_API_CB_CONNECT = 3002 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_API_FATAL_ERROR = 3005 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SEN_FORMAT = 3050 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SEN_UNDEF_NAME = 3051 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SEN_INDEX_RANGE = 3052 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SEN_BOUND_INVALID_UP = 3053 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SEN_BOUND_INVALID_LO = 3054 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SEN_INDEX_INVALID = 3055 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SEN_INVALID_REGEXP = 3056 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SEN_SOLUTION_STATUS = 3057 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SEN_NUMERICAL = 3058 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_CONCURRENT_OPTIMIZER = 3059 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SEN_UNHANDLED_PROBLEM_TYPE = 3080 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_TOO_MANY_CONCURRENT_TASKS = 3090 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_UNB_STEP_SIZE = 3100 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_IDENTICAL_TASKS = 3101 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_AD_INVALID_CODELIST = 3102 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_AD_INVALID_OPERATOR = 3103 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_AD_INVALID_OPERAND = 3104 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_AD_MISSING_OPERAND = 3105 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_AD_MISSING_RETURN = 3106 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_BRANCH_DIRECTION = 3200 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_BRANCH_PRIORITY = 3201 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NO_DUAL_INFO_FOR_ITG_SOL = 3300 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INTERNAL_TEST_FAILED = 3500 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_XML_INVALID_PROBLEM_TYPE = 3600 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_AMPL_STUB = 3700 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INT64_TO_INT32_CAST = 3800 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SIZE_LICENSE_NUMCORES = 3900 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INFEAS_UNDEFINED = 3910 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NO_BARX_FOR_SOLUTION = 3915 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NO_BARS_FOR_SOLUTION = 3916 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_BAR_VAR_DIM = 3920 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SYM_MAT_INVALID_ROW_INDEX = 3940 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SYM_MAT_INVALID_COL_INDEX = 3941 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SYM_MAT_NOT_LOWER_TRINGULAR = 3942 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SYM_MAT_INVALID_VALUE = 3943 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_SYM_MAT_DUPLICATE = 3944 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_SYM_MAT_DIM = 3950 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_API_INTERNAL = 3999 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_FILE_FORMAT_FOR_SYM_MAT = 4000 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_FILE_FORMAT_FOR_CONES = 4005 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_FILE_FORMAT_FOR_GENERAL_NL = 4010 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_DUPLICATE_CONSTRAINT_NAMES = 4500 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_DUPLICATE_VARIABLE_NAMES = 4501 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_DUPLICATE_BARVARIABLE_NAMES = 4502 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_DUPLICATE_CONE_NAMES = 4503 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_NON_UNIQUE_ARRAY = 5000 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_ARGUMENT_IS_TOO_LARGE = 5005 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_MIO_INTERNAL = 5010 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_INVALID_PROBLEM_TYPE = 6000 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_UNHANDLED_SOLUTION_STATUS = 6010 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_ERR_UPPER_TRIANGLE = 6020 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_MAX_ITERATIONS = 10000 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_MAX_TIME = 10001 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_OBJECTIVE_RANGE = 10002 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_MIO_NEAR_REL_GAP = 10003 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_MIO_NEAR_ABS_GAP = 10004 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_STALL = 10006 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_USER_CALLBACK = 10007 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_MIO_NUM_RELAXS = 10008 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_MIO_NUM_BRANCHES = 10009 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_NUM_MAX_NUM_INT_SOLUTIONS = 10015 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_MAX_NUM_SETBACKS = 10020 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_NUMERICAL_PROBLEM = 10025 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_INTERNAL = 10030 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

MSK_RES_TRM_INTERNAL_STOP = 10031 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 762

enum_MSKmionodeseltype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1168

MSK_MIO_NODE_SELECTION_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1168

MSK_MIO_NODE_SELECTION_END = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1168

MSK_MIO_NODE_SELECTION_FREE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1168

MSK_MIO_NODE_SELECTION_FIRST = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1168

MSK_MIO_NODE_SELECTION_BEST = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1168

MSK_MIO_NODE_SELECTION_WORST = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1168

MSK_MIO_NODE_SELECTION_HYBRID = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1168

MSK_MIO_NODE_SELECTION_PSEUDO = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1168

enum_MSKonoffkey_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1180

MSK_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1180

MSK_END = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1180

MSK_OFF = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1180

MSK_ON = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1180

enum_MSKsimdegen_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1188

MSK_SIM_DEGEN_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1188

MSK_SIM_DEGEN_END = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1188

MSK_SIM_DEGEN_NONE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1188

MSK_SIM_DEGEN_FREE = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1188

MSK_SIM_DEGEN_AGGRESSIVE = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1188

MSK_SIM_DEGEN_MODERATE = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1188

MSK_SIM_DEGEN_MINIMUM = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1188

enum_MSKdataformat_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1199

MSK_DATA_FORMAT_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1199

MSK_DATA_FORMAT_END = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1199

MSK_DATA_FORMAT_EXTENSION = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1199

MSK_DATA_FORMAT_MPS = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1199

MSK_DATA_FORMAT_LP = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1199

MSK_DATA_FORMAT_OP = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1199

MSK_DATA_FORMAT_XML = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1199

MSK_DATA_FORMAT_FREE_MPS = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1199

MSK_DATA_FORMAT_TASK = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1199

enum_MSKorderingtype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1212

MSK_ORDER_METHOD_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1212

MSK_ORDER_METHOD_END = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1212

MSK_ORDER_METHOD_FREE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1212

MSK_ORDER_METHOD_APPMINLOC = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1212

MSK_ORDER_METHOD_EXPERIMENTAL = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1212

MSK_ORDER_METHOD_TRY_GRAPHPAR = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1212

MSK_ORDER_METHOD_FORCE_GRAPHPAR = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1212

MSK_ORDER_METHOD_NONE = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1212

enum_MSKproblemtype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1224

MSK_PROBTYPE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1224

MSK_PROBTYPE_END = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1224

MSK_PROBTYPE_LO = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1224

MSK_PROBTYPE_QO = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1224

MSK_PROBTYPE_QCQO = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1224

MSK_PROBTYPE_GECO = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1224

MSK_PROBTYPE_CONIC = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1224

MSK_PROBTYPE_MIXED = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1224

enum_MSKinftype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1236

MSK_INF_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1236

MSK_INF_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1236

MSK_INF_DOU_TYPE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1236

MSK_INF_INT_TYPE = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1236

MSK_INF_LINT_TYPE = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1236

enum_MSKdparam_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_END = 67 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_ANA_SOL_INFEAS_TOL = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_BASIS_REL_TOL_S = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_BASIS_TOL_S = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_BASIS_TOL_X = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_CHECK_CONVEXITY_REL_TOL = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_DATA_TOL_AIJ = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_DATA_TOL_AIJ_HUGE = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_DATA_TOL_AIJ_LARGE = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_DATA_TOL_BOUND_INF = 8 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_DATA_TOL_BOUND_WRN = 9 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_DATA_TOL_C_HUGE = 10 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_DATA_TOL_CJ_LARGE = 11 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_DATA_TOL_QIJ = 12 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_DATA_TOL_X = 13 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_FEASREPAIR_TOL = 14 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_CO_TOL_DFEAS = 15 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_CO_TOL_INFEAS = 16 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_CO_TOL_MU_RED = 17 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_CO_TOL_NEAR_REL = 18 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_CO_TOL_PFEAS = 19 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 20 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_NL_MERIT_BAL = 21 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_NL_TOL_DFEAS = 22 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_NL_TOL_MU_RED = 23 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_NL_TOL_NEAR_REL = 24 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_NL_TOL_PFEAS = 25 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_NL_TOL_REL_GAP = 26 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_NL_TOL_REL_STEP = 27 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_TOL_DFEAS = 28 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_TOL_DSAFE = 29 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_TOL_INFEAS = 30 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_TOL_MU_RED = 31 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_TOL_PATH = 32 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_TOL_PFEAS = 33 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_TOL_PSAFE = 34 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_TOL_REL_GAP = 35 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_TOL_REL_STEP = 36 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_INTPNT_TOL_STEP_SIZE = 37 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_LOWER_OBJ_CUT = 38 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_LOWER_OBJ_CUT_FINITE_TRH = 39 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_DISABLE_TERM_TIME = 40 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_HEURISTIC_TIME = 41 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_MAX_TIME = 42 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_MAX_TIME_APRX_OPT = 43 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_NEAR_TOL_ABS_GAP = 44 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_NEAR_TOL_REL_GAP = 45 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_REL_ADD_CUT_LIMITED = 46 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_REL_GAP_CONST = 47 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_TOL_ABS_GAP = 48 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_TOL_ABS_RELAX_INT = 49 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_TOL_FEAS = 50 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_TOL_REL_GAP = 51 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_TOL_REL_RELAX_INT = 52 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_MIO_TOL_X = 53 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_NONCONVEX_TOL_FEAS = 54 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_NONCONVEX_TOL_OPT = 55 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_OPTIMIZER_MAX_TIME = 56 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_PRESOLVE_TOL_ABS_LINDEP = 57 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_PRESOLVE_TOL_AIJ = 58 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_PRESOLVE_TOL_REL_LINDEP = 59 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_PRESOLVE_TOL_S = 60 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_PRESOLVE_TOL_X = 61 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_QCQO_REFORMULATE_REL_DROP_TOL = 62 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_SIM_LU_TOL_REL_PIV = 63 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_SIMPLEX_ABS_TOL_PIV = 64 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_UPPER_OBJ_CUT = 65 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

MSK_DPAR_UPPER_OBJ_CUT_FINITE_TRH = 66 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1245

enum_MSKsimdupvec_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1318

MSK_SIM_EXPLOIT_DUPVEC_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1318

MSK_SIM_EXPLOIT_DUPVEC_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1318

MSK_SIM_EXPLOIT_DUPVEC_OFF = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1318

MSK_SIM_EXPLOIT_DUPVEC_ON = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1318

MSK_SIM_EXPLOIT_DUPVEC_FREE = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1318

enum_MSKcompresstype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1327

MSK_COMPRESS_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1327

MSK_COMPRESS_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1327

MSK_COMPRESS_NONE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1327

MSK_COMPRESS_FREE = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1327

MSK_COMPRESS_GZIP = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1327

enum_MSKnametype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1336

MSK_NAME_TYPE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1336

MSK_NAME_TYPE_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1336

MSK_NAME_TYPE_GEN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1336

MSK_NAME_TYPE_MPS = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1336

MSK_NAME_TYPE_LP = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1336

enum_MSKmpsformat_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1345

MSK_MPS_FORMAT_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1345

MSK_MPS_FORMAT_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1345

MSK_MPS_FORMAT_STRICT = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1345

MSK_MPS_FORMAT_RELAXED = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1345

MSK_MPS_FORMAT_FREE = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1345

enum_MSKvariabletype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1354

MSK_VAR_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1354

MSK_VAR_END = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1354

MSK_VAR_TYPE_CONT = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1354

MSK_VAR_TYPE_INT = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1354

enum_MSKcheckconvexitytype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1362

MSK_CHECK_CONVEXITY_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1362

MSK_CHECK_CONVEXITY_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1362

MSK_CHECK_CONVEXITY_NONE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1362

MSK_CHECK_CONVEXITY_SIMPLE = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1362

MSK_CHECK_CONVEXITY_FULL = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1362

enum_MSKlanguage_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1371

MSK_LANG_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1371

MSK_LANG_END = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1371

MSK_LANG_ENG = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1371

MSK_LANG_DAN = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1371

enum_MSKstartpointtype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1379

MSK_STARTING_POINT_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1379

MSK_STARTING_POINT_END = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1379

MSK_STARTING_POINT_FREE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1379

MSK_STARTING_POINT_GUESS = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1379

MSK_STARTING_POINT_CONSTANT = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1379

MSK_STARTING_POINT_SATISFY_BOUNDS = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1379

enum_MSKsoltype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1389

MSK_SOL_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1389

MSK_SOL_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1389

MSK_SOL_ITR = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1389

MSK_SOL_BAS = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1389

MSK_SOL_ITG = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1389

enum_MSKscalingmethod_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1398

MSK_SCALING_METHOD_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1398

MSK_SCALING_METHOD_END = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1398

MSK_SCALING_METHOD_POW2 = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1398

MSK_SCALING_METHOD_FREE = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1398

enum_MSKvalue_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1406

MSK_LICENSE_BUFFER_LENGTH = 20 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1406

MSK_MAX_STR_LEN = 1024 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1406

enum_MSKstakey_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1411

MSK_SK_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1411

MSK_SK_END = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1411

MSK_SK_UNK = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1411

MSK_SK_BAS = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1411

MSK_SK_SUPBAS = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1411

MSK_SK_LOW = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1411

MSK_SK_UPR = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1411

MSK_SK_FIX = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1411

MSK_SK_INF = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1411

enum_MSKsimreform_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1424

MSK_SIM_REFORMULATION_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1424

MSK_SIM_REFORMULATION_END = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1424

MSK_SIM_REFORMULATION_OFF = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1424

MSK_SIM_REFORMULATION_ON = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1424

MSK_SIM_REFORMULATION_FREE = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1424

MSK_SIM_REFORMULATION_AGGRESSIVE = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1424

enum_MSKiinfitem_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_END = 97 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_CON = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_CON_EQ = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_CON_FR = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_CON_LO = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_CON_RA = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_CON_UP = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_VAR = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_VAR_BIN = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_VAR_CONT = 8 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_VAR_EQ = 9 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_VAR_FR = 10 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_VAR_INT = 11 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_VAR_LO = 12 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_VAR_RA = 13 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_ANA_PRO_NUM_VAR_UP = 14 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_CONCURRENT_FASTEST_OPTIMIZER = 15 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_INTPNT_FACTOR_DIM_DENSE = 16 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_INTPNT_ITER = 17 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_INTPNT_NUM_THREADS = 18 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_INTPNT_SOLVE_DUAL = 19 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_CONSTRUCT_NUM_ROUNDINGS = 20 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_CONSTRUCT_SOLUTION = 21 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_INITIAL_SOLUTION = 22 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_ACTIVE_NODES = 23 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_BASIS_CUTS = 24 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_BRANCH = 25 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_CARDGUB_CUTS = 26 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_CLIQUE_CUTS = 27 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_COEF_REDC_CUTS = 28 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_CONTRA_CUTS = 29 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_DISAGG_CUTS = 30 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_FLOW_COVER_CUTS = 31 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_GCD_CUTS = 32 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_GOMORY_CUTS = 33 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_GUB_COVER_CUTS = 34 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_INT_SOLUTIONS = 35 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_KNAPSUR_COVER_CUTS = 36 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_LATTICE_CUTS = 37 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_LIFT_CUTS = 38 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_OBJ_CUTS = 39 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_PLAN_LOC_CUTS = 40 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUM_RELAX = 41 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUMCON = 42 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUMINT = 43 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_NUMVAR = 44 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_OBJ_BOUND_DEFINED = 45 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_TOTAL_NUM_CUTS = 46 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_MIO_USER_OBJ_CUT = 47 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_OPT_NUMCON = 48 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_OPT_NUMVAR = 49 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_OPTIMIZE_RESPONSE = 50 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_RD_NUMBARVAR = 51 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_RD_NUMCON = 52 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_RD_NUMCONE = 53 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_RD_NUMINTVAR = 54 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_RD_NUMQ = 55 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_RD_NUMVAR = 56 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_RD_PROTYPE = 57 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_DUAL_DEG_ITER = 58 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_DUAL_HOTSTART = 59 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_DUAL_HOTSTART_LU = 60 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_DUAL_INF_ITER = 61 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_DUAL_ITER = 62 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NETWORK_DUAL_DEG_ITER = 63 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NETWORK_DUAL_HOTSTART = 64 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NETWORK_DUAL_HOTSTART_LU = 65 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NETWORK_DUAL_INF_ITER = 66 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NETWORK_DUAL_ITER = 67 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NETWORK_PRIMAL_DEG_ITER = 68 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NETWORK_PRIMAL_HOTSTART = 69 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NETWORK_PRIMAL_HOTSTART_LU = 70 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NETWORK_PRIMAL_INF_ITER = 71 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NETWORK_PRIMAL_ITER = 72 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NUMCON = 73 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_NUMVAR = 74 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_PRIMAL_DEG_ITER = 75 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_PRIMAL_DUAL_DEG_ITER = 76 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_PRIMAL_DUAL_HOTSTART = 77 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_PRIMAL_DUAL_HOTSTART_LU = 78 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_PRIMAL_DUAL_INF_ITER = 79 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_PRIMAL_DUAL_ITER = 80 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_PRIMAL_HOTSTART = 81 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_PRIMAL_HOTSTART_LU = 82 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_PRIMAL_INF_ITER = 83 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_PRIMAL_ITER = 84 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SIM_SOLVE_DUAL = 85 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SOL_BAS_PROSTA = 86 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SOL_BAS_SOLSTA = 87 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SOL_INT_PROSTA = 88 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SOL_INT_SOLSTA = 89 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SOL_ITG_PROSTA = 90 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SOL_ITG_SOLSTA = 91 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SOL_ITR_PROSTA = 92 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_SOL_ITR_SOLSTA = 93 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_STO_NUM_A_CACHE_FLUSHES = 94 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_STO_NUM_A_REALLOC = 95 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

MSK_IINF_STO_NUM_A_TRANSPOSES = 96 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1434

enum_MSKxmlwriteroutputtype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1537

MSK_WRITE_XML_MODE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1537

MSK_WRITE_XML_MODE_END = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1537

MSK_WRITE_XML_MODE_ROW = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1537

MSK_WRITE_XML_MODE_COL = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1537

enum_MSKoptimizertype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_END = 12 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_FREE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_INTPNT = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_CONIC = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_PRIMAL_SIMPLEX = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_DUAL_SIMPLEX = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_PRIMAL_DUAL_SIMPLEX = 5 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_FREE_SIMPLEX = 6 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_NETWORK_PRIMAL_SIMPLEX = 7 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_MIXED_INT_CONIC = 8 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_MIXED_INT = 9 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_CONCURRENT = 10 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

MSK_OPTIMIZER_NONCONVEX = 11 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1545

enum_MSKpresolvemode_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1563

MSK_PRESOLVE_MODE_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1563

MSK_PRESOLVE_MODE_END = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1563

MSK_PRESOLVE_MODE_OFF = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1563

MSK_PRESOLVE_MODE_ON = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1563

MSK_PRESOLVE_MODE_FREE = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1563

enum_MSKmiocontsoltype_enum = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1572

MSK_MIO_CONT_SOL_BEGIN = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1572

MSK_MIO_CONT_SOL_END = 4 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1572

MSK_MIO_CONT_SOL_NONE = 0 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1572

MSK_MIO_CONT_SOL_ROOT = 1 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1572

MSK_MIO_CONT_SOL_ITG = 2 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1572

MSK_MIO_CONT_SOL_ITG_REL = 3 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1572

MSKchart = c_char # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2072

MSKvoid_t = POINTER(None) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2073

__mskuint64 = c_ulonglong # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2082

__mskint64 = c_longlong # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2087

__mskint32 = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2093

__mskuint32 = c_uint # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2094

MSKsolveforme = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2105

MSKproblemiteme = enum_MSKproblemitem_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2106

MSKaccmodee = enum_MSKaccmode_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2107

MSKsensitivitytypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2108

MSKintpnthotstarte = enum_MSKintpnthotstart_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2109

MSKsparame = enum_MSKsparam_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2110

MSKiparame = enum_MSKiparam_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2111

MSKsolstae = enum_MSKsolsta_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2112

MSKobjsensee = enum_MSKobjsense_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2113

MSKsoliteme = enum_MSKsolitem_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2114

MSKboundkeye = enum_MSKboundkey_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2115

MSKbasindtypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2116

MSKbranchdire = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2117

MSKliinfiteme = enum_MSKliinfitem_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2118

MSKstreamtypee = enum_MSKstreamtype_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2119

MSKsimhotstarte = enum_MSKsimhotstart_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2120

MSKcallbackcodee = enum_MSKcallbackcode_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2121

MSKsymmattypee = enum_MSKsymmattype_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2122

MSKfeaturee = enum_MSKfeature_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2123

MSKmarke = enum_MSKmark_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2124

MSKconetypee = enum_MSKconetype_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2125

MSKfeasrepairtypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2126

MSKiomodee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2127

MSKsimseltypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2128

MSKmsgkeye = enum_MSKmsgkey_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2129

MSKmiomodee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2130

MSKdinfiteme = enum_MSKdinfitem_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2131

MSKparametertypee = enum_MSKparametertype_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2132

MSKrescodetypee = enum_MSKrescodetype_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2133

MSKprostae = enum_MSKprosta_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2134

MSKscalingtypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2135

MSKrescodee = enum_MSKrescode_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2136

MSKmionodeseltypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2137

MSKonoffkeye = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2138

MSKsimdegene = enum_MSKsimdegen_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2139

MSKdataformate = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2140

MSKorderingtypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2141

MSKproblemtypee = enum_MSKproblemtype_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2142

MSKinftypee = enum_MSKinftype_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2143

MSKdparame = enum_MSKdparam_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2144

MSKsimdupvece = enum_MSKsimdupvec_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2145

MSKcompresstypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2146

MSKnametypee = enum_MSKnametype_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2147

MSKmpsformate = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2148

MSKvariabletypee = enum_MSKvariabletype_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2149

MSKcheckconvexitytypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2150

MSKlanguagee = enum_MSKlanguage_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2151

MSKstartpointtypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2152

MSKsoltypee = enum_MSKsoltype_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2153

MSKscalingmethode = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2154

MSKvaluee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2155

MSKstakeye = enum_MSKstakey_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2156

MSKsimreforme = enum_MSKsimreform_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2157

MSKiinfiteme = enum_MSKiinfitem_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2158

MSKxmlwriteroutputtypee = enum_MSKxmlwriteroutputtype_enum # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2159

MSKoptimizertypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2160

MSKpresolvemodee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2161

MSKmiocontsoltypee = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2162

MSKenv_t = POINTER(None) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2225

MSKtask_t = POINTER(None) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2227

MSKuserhandle_t = POINTER(None) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2229

MSKbooleant = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2231

MSKint32t = __mskint32 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2233

MSKint64t = __mskint64 # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2235

MSKwchart = c_wchar # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2237

MSKrealt = c_double # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2239

MSKstring_t = String # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2241

MSKcallbackfunc = CFUNCTYPE(UNCHECKED(MSKint32t), MSKtask_t, MSKuserhandle_t, MSKcallbackcodee, POINTER(MSKrealt), POINTER(MSKint32t), POINTER(MSKint64t)) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2244

MSKexitfunc = CFUNCTYPE(UNCHECKED(None), MSKuserhandle_t, String, MSKint32t, String) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2252

MSKfreefunc = CFUNCTYPE(UNCHECKED(None), MSKuserhandle_t, POINTER(None)) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2258

MSKmallocfunc = CFUNCTYPE(UNCHECKED(POINTER(None)), MSKuserhandle_t, c_size_t) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2262

MSKcallocfunc = CFUNCTYPE(UNCHECKED(POINTER(None)), MSKuserhandle_t, c_size_t, c_size_t) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2266

MSKreallocfunc = CFUNCTYPE(UNCHECKED(POINTER(None)), MSKuserhandle_t, POINTER(None), c_size_t) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2271

MSKnlgetspfunc = CFUNCTYPE(UNCHECKED(MSKint32t), MSKuserhandle_t, POINTER(MSKint32t), POINTER(MSKint32t), MSKint32t, POINTER(MSKbooleant), POINTER(MSKint32t), POINTER(MSKint32t), MSKint32t, MSKint32t, POINTER(MSKint32t), MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t)) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2276

MSKnlgetvafunc = CFUNCTYPE(UNCHECKED(MSKint32t), MSKuserhandle_t, POINTER(MSKrealt), MSKrealt, POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt), MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt), POINTER(MSKrealt), MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2292

MSKstreamfunc = CFUNCTYPE(UNCHECKED(None), MSKuserhandle_t, String) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2315

MSKresponsefunc = CFUNCTYPE(UNCHECKED(MSKrescodee), MSKuserhandle_t, MSKrescodee, String) # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2319

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2335
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_analyzeproblem'):
    MSK_analyzeproblem = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_analyzeproblem
    MSK_analyzeproblem.argtypes = [MSKtask_t, MSKstreamtypee]
    MSK_analyzeproblem.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2340
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_analyzenames'):
    MSK_analyzenames = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_analyzenames
    MSK_analyzenames.argtypes = [MSKtask_t, MSKstreamtypee, MSKnametypee]
    MSK_analyzenames.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2346
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_analyzesolution'):
    MSK_analyzesolution = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_analyzesolution
    MSK_analyzesolution.argtypes = [MSKtask_t, MSKstreamtypee, MSKsoltypee]
    MSK_analyzesolution.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2352
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_initbasissolve'):
    MSK_initbasissolve = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_initbasissolve
    MSK_initbasissolve.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_initbasissolve.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2357
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_solvewithbasis'):
    MSK_solvewithbasis = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_solvewithbasis
    MSK_solvewithbasis.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_solvewithbasis.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2365
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_basiscond'):
    MSK_basiscond = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_basiscond
    MSK_basiscond.argtypes = [MSKtask_t, POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_basiscond.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2371
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_appendcons'):
    MSK_appendcons = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_appendcons
    MSK_appendcons.argtypes = [MSKtask_t, MSKint32t]
    MSK_appendcons.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2376
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_appendvars'):
    MSK_appendvars = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_appendvars
    MSK_appendvars.argtypes = [MSKtask_t, MSKint32t]
    MSK_appendvars.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2381
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_removecons'):
    MSK_removecons = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_removecons
    MSK_removecons.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_removecons.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2387
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_removevars'):
    MSK_removevars = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_removevars
    MSK_removevars.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_removevars.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2393
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_removebarvars'):
    MSK_removebarvars = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_removebarvars
    MSK_removebarvars.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_removebarvars.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2399
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_removecones'):
    MSK_removecones = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_removecones
    MSK_removecones.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_removecones.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2405
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_appendbarvars'):
    MSK_appendbarvars = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_appendbarvars
    MSK_appendbarvars.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_appendbarvars.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2411
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_appendcone'):
    MSK_appendcone = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_appendcone
    MSK_appendcone.argtypes = [MSKtask_t, MSKconetypee, MSKrealt, MSKint32t, POINTER(MSKint32t)]
    MSK_appendcone.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2419
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_appendconeseq'):
    MSK_appendconeseq = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_appendconeseq
    MSK_appendconeseq.argtypes = [MSKtask_t, MSKconetypee, MSKrealt, MSKint32t, MSKint32t]
    MSK_appendconeseq.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2427
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_appendconesseq'):
    MSK_appendconesseq = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_appendconesseq
    MSK_appendconesseq.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKconetypee), POINTER(MSKrealt), POINTER(MSKint32t), MSKint32t]
    MSK_appendconesseq.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2436
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_bktostr'):
    MSK_bktostr = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_bktostr
    MSK_bktostr.argtypes = [MSKtask_t, MSKboundkeye, String]
    MSK_bktostr.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2442
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_callbackcodetostr'):
    MSK_callbackcodetostr = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_callbackcodetostr
    MSK_callbackcodetostr.argtypes = [MSKcallbackcodee, String]
    MSK_callbackcodetostr.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2447
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_calloctask'):
    MSK_calloctask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_calloctask
    MSK_calloctask.argtypes = [MSKtask_t, c_size_t, c_size_t]
    MSK_calloctask.restype = POINTER(None)

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2453
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_callocdbgtask'):
    MSK_callocdbgtask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_callocdbgtask
    MSK_callocdbgtask.argtypes = [MSKtask_t, c_size_t, c_size_t, String, c_uint]
    MSK_callocdbgtask.restype = POINTER(None)

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2461
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_chgbound'):
    MSK_chgbound = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_chgbound
    MSK_chgbound.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, MSKint32t, MSKint32t, MSKrealt]
    MSK_chgbound.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2470
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_conetypetostr'):
    MSK_conetypetostr = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_conetypetostr
    MSK_conetypetostr.argtypes = [MSKtask_t, MSKconetypee, String]
    MSK_conetypetostr.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2476
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_deletetask'):
    MSK_deletetask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_deletetask
    MSK_deletetask.argtypes = [POINTER(MSKtask_t)]
    MSK_deletetask.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2480
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_echotask'):
    _func = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_echotask
    _restype = MSKrescodee
    _argtypes = [MSKtask_t, MSKstreamtypee, String]
    MSK_echotask = _variadic_function(_func,_restype,_argtypes)

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2487
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_freetask'):
    MSK_freetask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_freetask
    MSK_freetask.argtypes = [MSKtask_t, POINTER(None)]
    MSK_freetask.restype = None

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2492
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_freedbgtask'):
    MSK_freedbgtask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_freedbgtask
    MSK_freedbgtask.argtypes = [MSKtask_t, POINTER(None), String, c_uint]
    MSK_freedbgtask.restype = None

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2499
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getaij'):
    MSK_getaij = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getaij
    MSK_getaij.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getaij.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2506
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getapiecenumnz'):
    MSK_getapiecenumnz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getapiecenumnz
    MSK_getapiecenumnz.argtypes = [MSKtask_t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, POINTER(MSKint32t)]
    MSK_getapiecenumnz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2515
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getacolnumnz'):
    MSK_getacolnumnz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getacolnumnz
    MSK_getacolnumnz.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_getacolnumnz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2521
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getacol'):
    MSK_getacol = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getacol
    MSK_getacol.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getacol.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2529
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getarownumnz'):
    MSK_getarownumnz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getarownumnz
    MSK_getarownumnz.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_getarownumnz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2535
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getarow'):
    MSK_getarow = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getarow
    MSK_getarow.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getarow.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2543
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getaslicenumnz'):
    MSK_getaslicenumnz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getaslicenumnz
    MSK_getaslicenumnz.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, MSKint32t, POINTER(MSKint32t)]
    MSK_getaslicenumnz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2551
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getaslicenumnz64'):
    MSK_getaslicenumnz64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getaslicenumnz64
    MSK_getaslicenumnz64.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, MSKint32t, POINTER(MSKint64t)]
    MSK_getaslicenumnz64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2559
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getaslice'):
    MSK_getaslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getaslice
    MSK_getaslice.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, MSKint32t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getaslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2572
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getaslice64'):
    MSK_getaslice64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getaslice64
    MSK_getaslice64.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, MSKint32t, MSKint64t, POINTER(MSKint64t), POINTER(MSKint64t), POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getaslice64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2585
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getarowslicetrip'):
    MSK_getarowslicetrip = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getarowslicetrip
    MSK_getarowslicetrip.argtypes = [MSKtask_t, MSKint32t, MSKint32t, MSKint64t, POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getarowslicetrip.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2596
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getacolslicetrip'):
    MSK_getacolslicetrip = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getacolslicetrip
    MSK_getacolslicetrip.argtypes = [MSKtask_t, MSKint32t, MSKint32t, MSKint64t, POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getacolslicetrip.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2607
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getconbound'):
    MSK_getconbound = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getconbound
    MSK_getconbound.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_getconbound.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2615
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getvarbound'):
    MSK_getvarbound = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getvarbound
    MSK_getvarbound.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_getvarbound.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2623
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbound'):
    MSK_getbound = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbound
    MSK_getbound.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_getbound.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2632
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getconboundslice'):
    MSK_getconboundslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getconboundslice
    MSK_getconboundslice.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_getconboundslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2641
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getvarboundslice'):
    MSK_getvarboundslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getvarboundslice
    MSK_getvarboundslice.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_getvarboundslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2650
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getboundslice'):
    MSK_getboundslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getboundslice
    MSK_getboundslice.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, MSKint32t, POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_getboundslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2660
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putboundslice'):
    MSK_putboundslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putboundslice
    MSK_putboundslice.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, MSKint32t, POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_putboundslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2670
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getcj'):
    MSK_getcj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getcj
    MSK_getcj.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKrealt)]
    MSK_getcj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2676
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getc'):
    MSK_getc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getc
    MSK_getc.argtypes = [MSKtask_t, POINTER(MSKrealt)]
    MSK_getc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2681
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getcallbackfunc'):
    MSK_getcallbackfunc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getcallbackfunc
    MSK_getcallbackfunc.argtypes = [MSKtask_t, POINTER(MSKcallbackfunc), POINTER(MSKuserhandle_t)]
    MSK_getcallbackfunc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2687
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsolutionincallback'):
    MSK_getsolutionincallback = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsolutionincallback
    MSK_getsolutionincallback.argtypes = [MSKtask_t, MSKcallbackcodee, MSKsoltypee, POINTER(MSKprostae), POINTER(MSKsolstae), POINTER(MSKstakeye), POINTER(MSKstakeye), POINTER(MSKstakeye), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_getsolutionincallback.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2706
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getcfix'):
    MSK_getcfix = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getcfix
    MSK_getcfix.argtypes = [MSKtask_t, POINTER(MSKrealt)]
    MSK_getcfix.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2711
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getcone'):
    MSK_getcone = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getcone
    MSK_getcone.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKconetypee), POINTER(MSKrealt), POINTER(MSKint32t), POINTER(MSKint32t)]
    MSK_getcone.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2720
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getconeinfo'):
    MSK_getconeinfo = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getconeinfo
    MSK_getconeinfo.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKconetypee), POINTER(MSKrealt), POINTER(MSKint32t)]
    MSK_getconeinfo.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2728
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getcslice'):
    MSK_getcslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getcslice
    MSK_getcslice.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getcslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2735
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getdouinf'):
    MSK_getdouinf = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getdouinf
    MSK_getdouinf.argtypes = [MSKtask_t, MSKdinfiteme, POINTER(MSKrealt)]
    MSK_getdouinf.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2741
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getdouparam'):
    MSK_getdouparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getdouparam
    MSK_getdouparam.argtypes = [MSKtask_t, MSKdparame, POINTER(MSKrealt)]
    MSK_getdouparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2747
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getdualobj'):
    MSK_getdualobj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getdualobj
    MSK_getdualobj.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_getdualobj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2753
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getenv'):
    MSK_getenv = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getenv
    MSK_getenv.argtypes = [MSKtask_t, POINTER(MSKenv_t)]
    MSK_getenv.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2758
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getinfindex'):
    MSK_getinfindex = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getinfindex
    MSK_getinfindex.argtypes = [MSKtask_t, MSKinftypee, String, POINTER(MSKint32t)]
    MSK_getinfindex.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2765
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getinfmax'):
    MSK_getinfmax = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getinfmax
    MSK_getinfmax.argtypes = [MSKtask_t, MSKinftypee, POINTER(MSKint32t)]
    MSK_getinfmax.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2771
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getinfname'):
    MSK_getinfname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getinfname
    MSK_getinfname.argtypes = [MSKtask_t, MSKinftypee, MSKint32t, String]
    MSK_getinfname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2778
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getintinf'):
    MSK_getintinf = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getintinf
    MSK_getintinf.argtypes = [MSKtask_t, MSKiinfiteme, POINTER(MSKint32t)]
    MSK_getintinf.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2784
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getlintinf'):
    MSK_getlintinf = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getlintinf
    MSK_getlintinf.argtypes = [MSKtask_t, MSKliinfiteme, POINTER(MSKint64t)]
    MSK_getlintinf.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2790
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getintparam'):
    MSK_getintparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getintparam
    MSK_getintparam.argtypes = [MSKtask_t, MSKiparame, POINTER(MSKint32t)]
    MSK_getintparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2796
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getmaxnamelen'):
    MSK_getmaxnamelen = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getmaxnamelen
    MSK_getmaxnamelen.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getmaxnamelen.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2801
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getmaxnumanz'):
    MSK_getmaxnumanz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getmaxnumanz
    MSK_getmaxnumanz.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getmaxnumanz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2806
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getmaxnumanz64'):
    MSK_getmaxnumanz64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getmaxnumanz64
    MSK_getmaxnumanz64.argtypes = [MSKtask_t, POINTER(MSKint64t)]
    MSK_getmaxnumanz64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2811
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getmaxnumcon'):
    MSK_getmaxnumcon = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getmaxnumcon
    MSK_getmaxnumcon.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getmaxnumcon.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2816
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getmaxnumvar'):
    MSK_getmaxnumvar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getmaxnumvar
    MSK_getmaxnumvar.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getmaxnumvar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2821
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnadouinf'):
    MSK_getnadouinf = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnadouinf
    MSK_getnadouinf.argtypes = [MSKtask_t, String, POINTER(MSKrealt)]
    MSK_getnadouinf.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2827
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnadouparam'):
    MSK_getnadouparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnadouparam
    MSK_getnadouparam.argtypes = [MSKtask_t, String, POINTER(MSKrealt)]
    MSK_getnadouparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2833
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnaintinf'):
    MSK_getnaintinf = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnaintinf
    MSK_getnaintinf.argtypes = [MSKtask_t, String, POINTER(MSKint32t)]
    MSK_getnaintinf.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2839
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnaintparam'):
    MSK_getnaintparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnaintparam
    MSK_getnaintparam.argtypes = [MSKtask_t, String, POINTER(MSKint32t)]
    MSK_getnaintparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2845
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarvarnamelen'):
    MSK_getbarvarnamelen = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarvarnamelen
    MSK_getbarvarnamelen.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_getbarvarnamelen.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2851
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarvarname'):
    MSK_getbarvarname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarvarname
    MSK_getbarvarname.argtypes = [MSKtask_t, MSKint32t, MSKint32t, String]
    MSK_getbarvarname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2858
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarvarnameindex'):
    MSK_getbarvarnameindex = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarvarnameindex
    MSK_getbarvarnameindex.argtypes = [MSKtask_t, String, POINTER(MSKint32t), POINTER(MSKint32t)]
    MSK_getbarvarnameindex.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2865
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putconname'):
    MSK_putconname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putconname
    MSK_putconname.argtypes = [MSKtask_t, MSKint32t, String]
    MSK_putconname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2871
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putvarname'):
    MSK_putvarname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putvarname
    MSK_putvarname.argtypes = [MSKtask_t, MSKint32t, String]
    MSK_putvarname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2877
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putconename'):
    MSK_putconename = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putconename
    MSK_putconename.argtypes = [MSKtask_t, MSKint32t, String]
    MSK_putconename.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2883
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putbarvarname'):
    MSK_putbarvarname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putbarvarname
    MSK_putbarvarname.argtypes = [MSKtask_t, MSKint32t, String]
    MSK_putbarvarname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2889
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getvarnamelen'):
    MSK_getvarnamelen = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getvarnamelen
    MSK_getvarnamelen.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_getvarnamelen.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2895
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getvarname'):
    MSK_getvarname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getvarname
    MSK_getvarname.argtypes = [MSKtask_t, MSKint32t, MSKint32t, String]
    MSK_getvarname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2902
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getconnamelen'):
    MSK_getconnamelen = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getconnamelen
    MSK_getconnamelen.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_getconnamelen.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2908
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getconname'):
    MSK_getconname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getconname
    MSK_getconname.argtypes = [MSKtask_t, MSKint32t, MSKint32t, String]
    MSK_getconname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2915
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getconnameindex'):
    MSK_getconnameindex = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getconnameindex
    MSK_getconnameindex.argtypes = [MSKtask_t, String, POINTER(MSKint32t), POINTER(MSKint32t)]
    MSK_getconnameindex.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2922
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getvarnameindex'):
    MSK_getvarnameindex = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getvarnameindex
    MSK_getvarnameindex.argtypes = [MSKtask_t, String, POINTER(MSKint32t), POINTER(MSKint32t)]
    MSK_getvarnameindex.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2929
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getconenamelen'):
    MSK_getconenamelen = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getconenamelen
    MSK_getconenamelen.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_getconenamelen.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2935
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getconename'):
    MSK_getconename = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getconename
    MSK_getconename.argtypes = [MSKtask_t, MSKint32t, MSKint32t, String]
    MSK_getconename.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2942
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getconenameindex'):
    MSK_getconenameindex = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getconenameindex
    MSK_getconenameindex.argtypes = [MSKtask_t, String, POINTER(MSKint32t), POINTER(MSKint32t)]
    MSK_getconenameindex.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2949
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnastrparam'):
    MSK_getnastrparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnastrparam
    MSK_getnastrparam.argtypes = [MSKtask_t, String, MSKint32t, POINTER(MSKint32t), String]
    MSK_getnastrparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2957
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumanz'):
    MSK_getnumanz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumanz
    MSK_getnumanz.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getnumanz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2962
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumanz64'):
    MSK_getnumanz64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumanz64
    MSK_getnumanz64.argtypes = [MSKtask_t, POINTER(MSKint64t)]
    MSK_getnumanz64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2967
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumcon'):
    MSK_getnumcon = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumcon
    MSK_getnumcon.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getnumcon.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2972
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumcone'):
    MSK_getnumcone = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumcone
    MSK_getnumcone.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getnumcone.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2977
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumconemem'):
    MSK_getnumconemem = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumconemem
    MSK_getnumconemem.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_getnumconemem.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2983
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumintvar'):
    MSK_getnumintvar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumintvar
    MSK_getnumintvar.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getnumintvar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2988
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumparam'):
    MSK_getnumparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumparam
    MSK_getnumparam.argtypes = [MSKtask_t, MSKparametertypee, POINTER(MSKint32t)]
    MSK_getnumparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2994
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumqconknz'):
    MSK_getnumqconknz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumqconknz
    MSK_getnumqconknz.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_getnumqconknz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3000
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumqconknz64'):
    MSK_getnumqconknz64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumqconknz64
    MSK_getnumqconknz64.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint64t)]
    MSK_getnumqconknz64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3006
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumqobjnz'):
    MSK_getnumqobjnz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumqobjnz
    MSK_getnumqobjnz.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getnumqobjnz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3011
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumqobjnz64'):
    MSK_getnumqobjnz64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumqobjnz64
    MSK_getnumqobjnz64.argtypes = [MSKtask_t, POINTER(MSKint64t)]
    MSK_getnumqobjnz64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3016
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumvar'):
    MSK_getnumvar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumvar
    MSK_getnumvar.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getnumvar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3021
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumbarvar'):
    MSK_getnumbarvar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumbarvar
    MSK_getnumbarvar.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getnumbarvar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3026
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getmaxnumbarvar'):
    MSK_getmaxnumbarvar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getmaxnumbarvar
    MSK_getmaxnumbarvar.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getmaxnumbarvar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3031
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getdimbarvarj'):
    MSK_getdimbarvarj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getdimbarvarj
    MSK_getdimbarvarj.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_getdimbarvarj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3037
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getlenbarvarj'):
    MSK_getlenbarvarj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getlenbarvarj
    MSK_getlenbarvarj.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint64t)]
    MSK_getlenbarvarj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3043
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getobjname'):
    MSK_getobjname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getobjname
    MSK_getobjname.argtypes = [MSKtask_t, MSKint32t, String]
    MSK_getobjname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3049
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getobjnamelen'):
    MSK_getobjnamelen = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getobjnamelen
    MSK_getobjnamelen.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getobjnamelen.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3054
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getparamname'):
    MSK_getparamname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getparamname
    MSK_getparamname.argtypes = [MSKtask_t, MSKparametertypee, MSKint32t, String]
    MSK_getparamname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3061
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getparammax'):
    MSK_getparammax = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getparammax
    MSK_getparammax.argtypes = [MSKtask_t, MSKparametertypee, POINTER(MSKint32t)]
    MSK_getparammax.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3067
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getprimalobj'):
    MSK_getprimalobj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getprimalobj
    MSK_getprimalobj.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_getprimalobj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3073
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getprobtype'):
    MSK_getprobtype = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getprobtype
    MSK_getprobtype.argtypes = [MSKtask_t, POINTER(MSKproblemtypee)]
    MSK_getprobtype.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3078
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getqconk64'):
    MSK_getqconk64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getqconk64
    MSK_getqconk64.argtypes = [MSKtask_t, MSKint32t, MSKint64t, POINTER(MSKint64t), POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getqconk64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3089
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getqconk'):
    MSK_getqconk = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getqconk
    MSK_getqconk.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getqconk.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3100
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getqobj'):
    MSK_getqobj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getqobj
    MSK_getqobj.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getqobj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3110
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getqobj64'):
    MSK_getqobj64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getqobj64
    MSK_getqobj64.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint64t), POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getqobj64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3120
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getqobjij'):
    MSK_getqobjij = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getqobjij
    MSK_getqobjij.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getqobjij.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3127
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsolution'):
    MSK_getsolution = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsolution
    MSK_getsolution.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKprostae), POINTER(MSKsolstae), POINTER(MSKstakeye), POINTER(MSKstakeye), POINTER(MSKstakeye), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_getsolution.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3145
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getpbi'):
    MSK_getpbi = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getpbi
    MSK_getpbi.argtypes = [MSKtask_t, MSKsoltypee, MSKaccmodee, POINTER(MSKint32t), MSKint32t, POINTER(MSKrealt), MSKint32t]
    MSK_getpbi.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3155
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getdbi'):
    MSK_getdbi = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getdbi
    MSK_getdbi.argtypes = [MSKtask_t, MSKsoltypee, MSKaccmodee, POINTER(MSKint32t), MSKint32t, POINTER(MSKrealt)]
    MSK_getdbi.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3164
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getdeqi'):
    MSK_getdeqi = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getdeqi
    MSK_getdeqi.argtypes = [MSKtask_t, MSKsoltypee, MSKaccmodee, POINTER(MSKint32t), MSKint32t, POINTER(MSKrealt), MSKint32t]
    MSK_getdeqi.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3174
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getpeqi'):
    MSK_getpeqi = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getpeqi
    MSK_getpeqi.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKint32t), MSKint32t, POINTER(MSKrealt), MSKint32t]
    MSK_getpeqi.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3183
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getinti'):
    MSK_getinti = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getinti
    MSK_getinti.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKint32t), MSKint32t, POINTER(MSKrealt)]
    MSK_getinti.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3191
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getpcni'):
    MSK_getpcni = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getpcni
    MSK_getpcni.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKint32t), MSKint32t, POINTER(MSKrealt)]
    MSK_getpcni.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3199
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getdcni'):
    MSK_getdcni = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getdcni
    MSK_getdcni.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKint32t), MSKint32t, POINTER(MSKrealt)]
    MSK_getdcni.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3207
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsolutioni'):
    MSK_getsolutioni = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsolutioni
    MSK_getsolutioni.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, MSKsoltypee, POINTER(MSKstakeye), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_getsolutioni.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3219
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsolsta'):
    MSK_getsolsta = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsolsta
    MSK_getsolsta.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKsolstae)]
    MSK_getsolsta.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3225
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getprosta'):
    MSK_getprosta = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getprosta
    MSK_getprosta.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKprostae)]
    MSK_getprosta.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3231
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getskc'):
    MSK_getskc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getskc
    MSK_getskc.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKstakeye)]
    MSK_getskc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3237
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getskx'):
    MSK_getskx = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getskx
    MSK_getskx.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKstakeye)]
    MSK_getskx.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3243
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getxc'):
    MSK_getxc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getxc
    MSK_getxc.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_getxc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3249
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getxx'):
    MSK_getxx = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getxx
    MSK_getxx.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_getxx.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3255
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_gety'):
    MSK_gety = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_gety
    MSK_gety.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_gety.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3261
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getslc'):
    MSK_getslc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getslc
    MSK_getslc.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_getslc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3267
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsuc'):
    MSK_getsuc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsuc
    MSK_getsuc.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_getsuc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3273
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getslx'):
    MSK_getslx = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getslx
    MSK_getslx.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_getslx.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3279
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsux'):
    MSK_getsux = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsux
    MSK_getsux.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_getsux.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3285
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsnx'):
    MSK_getsnx = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsnx
    MSK_getsnx.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_getsnx.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3291
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getskcslice'):
    MSK_getskcslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getskcslice
    MSK_getskcslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKstakeye)]
    MSK_getskcslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3299
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getskxslice'):
    MSK_getskxslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getskxslice
    MSK_getskxslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKstakeye)]
    MSK_getskxslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3307
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getxcslice'):
    MSK_getxcslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getxcslice
    MSK_getxcslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getxcslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3315
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getxxslice'):
    MSK_getxxslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getxxslice
    MSK_getxxslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getxxslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3323
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getyslice'):
    MSK_getyslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getyslice
    MSK_getyslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getyslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3331
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getslcslice'):
    MSK_getslcslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getslcslice
    MSK_getslcslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getslcslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3339
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsucslice'):
    MSK_getsucslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsucslice
    MSK_getsucslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getsucslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3347
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getslxslice'):
    MSK_getslxslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getslxslice
    MSK_getslxslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getslxslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3355
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsuxslice'):
    MSK_getsuxslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsuxslice
    MSK_getsuxslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getsuxslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3363
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsnxslice'):
    MSK_getsnxslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsnxslice
    MSK_getsnxslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getsnxslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3371
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarxj'):
    MSK_getbarxj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarxj
    MSK_getbarxj.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKrealt)]
    MSK_getbarxj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3378
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarsj'):
    MSK_getbarsj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarsj
    MSK_getbarsj.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKrealt)]
    MSK_getbarsj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3385
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putskc'):
    MSK_putskc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putskc
    MSK_putskc.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKstakeye)]
    MSK_putskc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3391
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putskx'):
    MSK_putskx = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putskx
    MSK_putskx.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKstakeye)]
    MSK_putskx.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3397
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putxc'):
    MSK_putxc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putxc
    MSK_putxc.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_putxc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3403
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putxx'):
    MSK_putxx = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putxx
    MSK_putxx.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_putxx.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3409
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_puty'):
    MSK_puty = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_puty
    MSK_puty.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_puty.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3415
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putslc'):
    MSK_putslc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putslc
    MSK_putslc.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_putslc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3421
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putsuc'):
    MSK_putsuc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putsuc
    MSK_putsuc.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_putsuc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3427
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putslx'):
    MSK_putslx = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putslx
    MSK_putslx.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_putslx.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3433
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putsux'):
    MSK_putsux = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putsux
    MSK_putsux.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_putsux.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3439
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putsnx'):
    MSK_putsnx = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putsnx
    MSK_putsnx.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt)]
    MSK_putsnx.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3445
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putskcslice'):
    MSK_putskcslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putskcslice
    MSK_putskcslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKstakeye)]
    MSK_putskcslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3453
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putskxslice'):
    MSK_putskxslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putskxslice
    MSK_putskxslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKstakeye)]
    MSK_putskxslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3461
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putxcslice'):
    MSK_putxcslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putxcslice
    MSK_putxcslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_putxcslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3469
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putxxslice'):
    MSK_putxxslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putxxslice
    MSK_putxxslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_putxxslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3477
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putyslice'):
    MSK_putyslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putyslice
    MSK_putyslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_putyslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3485
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putslcslice'):
    MSK_putslcslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putslcslice
    MSK_putslcslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_putslcslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3493
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putsucslice'):
    MSK_putsucslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putsucslice
    MSK_putsucslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_putsucslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3501
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putslxslice'):
    MSK_putslxslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putslxslice
    MSK_putslxslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_putslxslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3509
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putsuxslice'):
    MSK_putsuxslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putsuxslice
    MSK_putsuxslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_putsuxslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3517
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putsnxslice'):
    MSK_putsnxslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putsnxslice
    MSK_putsnxslice.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_putsnxslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3525
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putbarxj'):
    MSK_putbarxj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putbarxj
    MSK_putbarxj.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKrealt)]
    MSK_putbarxj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3532
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putbarsj'):
    MSK_putbarsj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putbarsj
    MSK_putbarsj.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKrealt)]
    MSK_putbarsj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3539
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsolutioninf'):
    MSK_getsolutioninf = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsolutioninf
    MSK_getsolutioninf.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKprostae), POINTER(MSKsolstae), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_getsolutioninf.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3555
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getpviolcon'):
    MSK_getpviolcon = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getpviolcon
    MSK_getpviolcon.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getpviolcon.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3563
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getpviolvar'):
    MSK_getpviolvar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getpviolvar
    MSK_getpviolvar.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getpviolvar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3571
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getpviolbarvar'):
    MSK_getpviolbarvar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getpviolbarvar
    MSK_getpviolbarvar.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getpviolbarvar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3579
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getpviolcones'):
    MSK_getpviolcones = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getpviolcones
    MSK_getpviolcones.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getpviolcones.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3587
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getdviolcon'):
    MSK_getdviolcon = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getdviolcon
    MSK_getdviolcon.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getdviolcon.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3595
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getdviolvar'):
    MSK_getdviolvar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getdviolvar
    MSK_getdviolvar.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getdviolvar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3603
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getdviolbarvar'):
    MSK_getdviolbarvar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getdviolbarvar
    MSK_getdviolbarvar.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getdviolbarvar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3611
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getdviolcones'):
    MSK_getdviolcones = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getdviolcones
    MSK_getdviolcones.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getdviolcones.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3619
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsolutioninfo'):
    MSK_getsolutioninfo = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsolutioninfo
    MSK_getsolutioninfo.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_getsolutioninfo.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3635
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsolutionslice'):
    MSK_getsolutionslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsolutionslice
    MSK_getsolutionslice.argtypes = [MSKtask_t, MSKsoltypee, MSKsoliteme, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getsolutionslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3644
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getreducedcosts'):
    MSK_getreducedcosts = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getreducedcosts
    MSK_getreducedcosts.argtypes = [MSKtask_t, MSKsoltypee, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_getreducedcosts.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3652
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getstrparam'):
    MSK_getstrparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getstrparam
    MSK_getstrparam.argtypes = [MSKtask_t, MSKsparame, MSKint32t, POINTER(MSKint32t), String]
    MSK_getstrparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3660
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getstrparamlen'):
    MSK_getstrparamlen = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getstrparamlen
    MSK_getstrparamlen.argtypes = [MSKtask_t, MSKsparame, POINTER(MSKint32t)]
    MSK_getstrparamlen.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3666
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getstrparamal'):
    MSK_getstrparamal = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getstrparamal
    MSK_getstrparamal.argtypes = [MSKtask_t, MSKsparame, MSKint32t, POINTER(MSKstring_t)]
    MSK_getstrparamal.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3673
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnastrparamal'):
    MSK_getnastrparamal = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnastrparamal
    MSK_getnastrparamal.argtypes = [MSKtask_t, String, MSKint32t, POINTER(MSKstring_t)]
    MSK_getnastrparamal.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3680
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsymbcon'):
    MSK_getsymbcon = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsymbcon
    MSK_getsymbcon.argtypes = [MSKtask_t, MSKint32t, MSKint32t, String, POINTER(MSKint32t)]
    MSK_getsymbcon.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3688
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_gettasknamelen'):
    MSK_gettasknamelen = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_gettasknamelen
    MSK_gettasknamelen.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_gettasknamelen.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3693
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_gettaskname'):
    MSK_gettaskname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_gettaskname
    MSK_gettaskname.argtypes = [MSKtask_t, MSKint32t, String]
    MSK_gettaskname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3699
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getvartype'):
    MSK_getvartype = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getvartype
    MSK_getvartype.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKvariabletypee)]
    MSK_getvartype.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3705
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getvartypelist'):
    MSK_getvartypelist = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getvartypelist
    MSK_getvartypelist.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKvariabletypee)]
    MSK_getvartypelist.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3712
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_inputdata'):
    MSK_inputdata = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_inputdata
    MSK_inputdata.argtypes = [MSKtask_t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, POINTER(MSKrealt), MSKrealt, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt), POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_inputdata.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3732
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_inputdata64'):
    MSK_inputdata64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_inputdata64
    MSK_inputdata64.argtypes = [MSKtask_t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, POINTER(MSKrealt), MSKrealt, POINTER(MSKint64t), POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKrealt), POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_inputdata64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3752
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_isdouparname'):
    MSK_isdouparname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_isdouparname
    MSK_isdouparname.argtypes = [MSKtask_t, String, POINTER(MSKdparame)]
    MSK_isdouparname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3758
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_isintparname'):
    MSK_isintparname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_isintparname
    MSK_isintparname.argtypes = [MSKtask_t, String, POINTER(MSKiparame)]
    MSK_isintparname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3764
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_isstrparname'):
    MSK_isstrparname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_isstrparname
    MSK_isstrparname.argtypes = [MSKtask_t, String, POINTER(MSKsparame)]
    MSK_isstrparname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3770
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_linkfiletotaskstream'):
    MSK_linkfiletotaskstream = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_linkfiletotaskstream
    MSK_linkfiletotaskstream.argtypes = [MSKtask_t, MSKstreamtypee, String, MSKint32t]
    MSK_linkfiletotaskstream.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3777
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_linkfunctotaskstream'):
    MSK_linkfunctotaskstream = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_linkfunctotaskstream
    MSK_linkfunctotaskstream.argtypes = [MSKtask_t, MSKstreamtypee, MSKuserhandle_t, MSKstreamfunc]
    MSK_linkfunctotaskstream.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3784
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_unlinkfuncfromtaskstream'):
    MSK_unlinkfuncfromtaskstream = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_unlinkfuncfromtaskstream
    MSK_unlinkfuncfromtaskstream.argtypes = [MSKtask_t, MSKstreamtypee]
    MSK_unlinkfuncfromtaskstream.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3789
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_clonetask'):
    MSK_clonetask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_clonetask
    MSK_clonetask.argtypes = [MSKtask_t, POINTER(MSKtask_t)]
    MSK_clonetask.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3794
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_relaxprimal'):
    MSK_relaxprimal = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_relaxprimal
    MSK_relaxprimal.argtypes = [MSKtask_t, POINTER(MSKtask_t), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_relaxprimal.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3803
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_primalrepair'):
    MSK_primalrepair = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_primalrepair
    MSK_primalrepair.argtypes = [MSKtask_t, POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_primalrepair.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3811
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_reformqcqotosocp'):
    MSK_reformqcqotosocp = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_reformqcqotosocp
    MSK_reformqcqotosocp.argtypes = [MSKtask_t, MSKtask_t]
    MSK_reformqcqotosocp.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3816
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_optimizeconcurrent'):
    MSK_optimizeconcurrent = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_optimizeconcurrent
    MSK_optimizeconcurrent.argtypes = [MSKtask_t, POINTER(MSKtask_t), MSKint32t]
    MSK_optimizeconcurrent.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3822
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_optimize'):
    MSK_optimize = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_optimize
    MSK_optimize.argtypes = [MSKtask_t]
    MSK_optimize.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3826
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_optimizetrm'):
    MSK_optimizetrm = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_optimizetrm
    MSK_optimizetrm.argtypes = [MSKtask_t, POINTER(MSKrescodee)]
    MSK_optimizetrm.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3831
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_printdata'):
    MSK_printdata = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_printdata
    MSK_printdata.argtypes = [MSKtask_t, MSKstreamtypee, MSKint32t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, MSKint32t, MSKint32t]
    MSK_printdata.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3850
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_printparam'):
    MSK_printparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_printparam
    MSK_printparam.argtypes = [MSKtask_t]
    MSK_printparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3854
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_probtypetostr'):
    MSK_probtypetostr = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_probtypetostr
    MSK_probtypetostr.argtypes = [MSKtask_t, MSKproblemtypee, String]
    MSK_probtypetostr.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3860
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_prostatostr'):
    MSK_prostatostr = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_prostatostr
    MSK_prostatostr.argtypes = [MSKtask_t, MSKprostae, String]
    MSK_prostatostr.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3866
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putresponsefunc'):
    MSK_putresponsefunc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putresponsefunc
    MSK_putresponsefunc.argtypes = [MSKtask_t, MSKresponsefunc, MSKuserhandle_t]
    MSK_putresponsefunc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3872
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_commitchanges'):
    MSK_commitchanges = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_commitchanges
    MSK_commitchanges.argtypes = [MSKtask_t]
    MSK_commitchanges.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3876
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putaij'):
    MSK_putaij = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putaij
    MSK_putaij.argtypes = [MSKtask_t, MSKint32t, MSKint32t, MSKrealt]
    MSK_putaij.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3883
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putaijlist'):
    MSK_putaijlist = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putaijlist
    MSK_putaijlist.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putaijlist.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3891
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putacol'):
    MSK_putacol = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putacol
    MSK_putacol.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putacol.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3899
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putarow'):
    MSK_putarow = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putarow
    MSK_putarow.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putarow.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3907
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putarowslice'):
    MSK_putarowslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putarowslice
    MSK_putarowslice.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putarowslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3917
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putarowslice64'):
    MSK_putarowslice64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putarowslice64
    MSK_putarowslice64.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKint64t), POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putarowslice64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3927
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putarowlist'):
    MSK_putarowlist = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putarowlist
    MSK_putarowlist.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putarowlist.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3937
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putarowlist64'):
    MSK_putarowlist64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putarowlist64
    MSK_putarowlist64.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint64t), POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putarowlist64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3947
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putacolslice'):
    MSK_putacolslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putacolslice
    MSK_putacolslice.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putacolslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3957
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putacolslice64'):
    MSK_putacolslice64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putacolslice64
    MSK_putacolslice64.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKint64t), POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putacolslice64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3967
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putacollist'):
    MSK_putacollist = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putacollist
    MSK_putacollist.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putacollist.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3977
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putacollist64'):
    MSK_putacollist64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putacollist64
    MSK_putacollist64.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint64t), POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putacollist64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3987
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putbaraij'):
    MSK_putbaraij = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putbaraij
    MSK_putbaraij.argtypes = [MSKtask_t, MSKint32t, MSKint32t, MSKint64t, POINTER(MSKint64t), POINTER(MSKrealt)]
    MSK_putbaraij.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 3996
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumbarcnz'):
    MSK_getnumbarcnz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumbarcnz
    MSK_getnumbarcnz.argtypes = [MSKtask_t, POINTER(MSKint64t)]
    MSK_getnumbarcnz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4001
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumbaranz'):
    MSK_getnumbaranz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumbaranz
    MSK_getnumbaranz.argtypes = [MSKtask_t, POINTER(MSKint64t)]
    MSK_getnumbaranz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4006
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarcsparsity'):
    MSK_getbarcsparsity = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarcsparsity
    MSK_getbarcsparsity.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint64t), POINTER(MSKint64t)]
    MSK_getbarcsparsity.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4013
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarasparsity'):
    MSK_getbarasparsity = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarasparsity
    MSK_getbarasparsity.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint64t), POINTER(MSKint64t)]
    MSK_getbarasparsity.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4020
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarcidxinfo'):
    MSK_getbarcidxinfo = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarcidxinfo
    MSK_getbarcidxinfo.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint64t)]
    MSK_getbarcidxinfo.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4026
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarcidxj'):
    MSK_getbarcidxj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarcidxj
    MSK_getbarcidxj.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint32t)]
    MSK_getbarcidxj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4032
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarcidx'):
    MSK_getbarcidx = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarcidx
    MSK_getbarcidx.argtypes = [MSKtask_t, MSKint64t, MSKint64t, POINTER(MSKint32t), POINTER(MSKint64t), POINTER(MSKint64t), POINTER(MSKrealt)]
    MSK_getbarcidx.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4042
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbaraidxinfo'):
    MSK_getbaraidxinfo = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbaraidxinfo
    MSK_getbaraidxinfo.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint64t)]
    MSK_getbaraidxinfo.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4048
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbaraidxij'):
    MSK_getbaraidxij = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbaraidxij
    MSK_getbaraidxij.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint32t), POINTER(MSKint32t)]
    MSK_getbaraidxij.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4055
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbaraidx'):
    MSK_getbaraidx = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbaraidx
    MSK_getbaraidx.argtypes = [MSKtask_t, MSKint64t, MSKint64t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint64t), POINTER(MSKint64t), POINTER(MSKrealt)]
    MSK_getbaraidx.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4066
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putbarcblocktriplet'):
    MSK_putbarcblocktriplet = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putbarcblocktriplet
    MSK_putbarcblocktriplet.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putbarcblocktriplet.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4075
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putbarablocktriplet'):
    MSK_putbarablocktriplet = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putbarablocktriplet
    MSK_putbarablocktriplet.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putbarablocktriplet.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4085
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumbarcblocktriplets'):
    MSK_getnumbarcblocktriplets = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumbarcblocktriplets
    MSK_getnumbarcblocktriplets.argtypes = [MSKtask_t, POINTER(MSKint64t)]
    MSK_getnumbarcblocktriplets.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4090
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarcblocktriplet'):
    MSK_getbarcblocktriplet = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarcblocktriplet
    MSK_getbarcblocktriplet.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getbarcblocktriplet.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4100
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumbarablocktriplets'):
    MSK_getnumbarablocktriplets = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumbarablocktriplets
    MSK_getnumbarablocktriplets.argtypes = [MSKtask_t, POINTER(MSKint64t)]
    MSK_getnumbarablocktriplets.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4105
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbarablocktriplet'):
    MSK_getbarablocktriplet = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbarablocktriplet
    MSK_getbarablocktriplet.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint64t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getbarablocktriplet.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4116
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putbound'):
    MSK_putbound = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putbound
    MSK_putbound.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, MSKboundkeye, MSKrealt, MSKrealt]
    MSK_putbound.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4125
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putboundlist'):
    MSK_putboundlist = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putboundlist
    MSK_putboundlist.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, POINTER(MSKint32t), POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_putboundlist.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4135
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putconbound'):
    MSK_putconbound = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putconbound
    MSK_putconbound.argtypes = [MSKtask_t, MSKint32t, MSKboundkeye, MSKrealt, MSKrealt]
    MSK_putconbound.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4143
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putconboundlist'):
    MSK_putconboundlist = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putconboundlist
    MSK_putconboundlist.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_putconboundlist.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4152
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putconboundslice'):
    MSK_putconboundslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putconboundslice
    MSK_putconboundslice.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_putconboundslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4161
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putvarbound'):
    MSK_putvarbound = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putvarbound
    MSK_putvarbound.argtypes = [MSKtask_t, MSKint32t, MSKboundkeye, MSKrealt, MSKrealt]
    MSK_putvarbound.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4169
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putvarboundlist'):
    MSK_putvarboundlist = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putvarboundlist
    MSK_putvarboundlist.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_putvarboundlist.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4178
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putvarboundslice'):
    MSK_putvarboundslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putvarboundslice
    MSK_putvarboundslice.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKboundkeye), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_putvarboundslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4187
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putcallbackfunc'):
    MSK_putcallbackfunc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putcallbackfunc
    MSK_putcallbackfunc.argtypes = [MSKtask_t, MSKcallbackfunc, MSKuserhandle_t]
    MSK_putcallbackfunc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4193
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putcfix'):
    MSK_putcfix = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putcfix
    MSK_putcfix.argtypes = [MSKtask_t, MSKrealt]
    MSK_putcfix.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4198
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putcj'):
    MSK_putcj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putcj
    MSK_putcj.argtypes = [MSKtask_t, MSKint32t, MSKrealt]
    MSK_putcj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4204
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putobjsense'):
    MSK_putobjsense = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putobjsense
    MSK_putobjsense.argtypes = [MSKtask_t, MSKobjsensee]
    MSK_putobjsense.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4209
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getobjsense'):
    MSK_getobjsense = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getobjsense
    MSK_getobjsense.argtypes = [MSKtask_t, POINTER(MSKobjsensee)]
    MSK_getobjsense.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4214
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putclist'):
    MSK_putclist = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putclist
    MSK_putclist.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putclist.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4221
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putcslice'):
    MSK_putcslice = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putcslice
    MSK_putcslice.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKrealt)]
    MSK_putcslice.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4228
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putbarcj'):
    MSK_putbarcj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putbarcj
    MSK_putbarcj.argtypes = [MSKtask_t, MSKint32t, MSKint64t, POINTER(MSKint64t), POINTER(MSKrealt)]
    MSK_putbarcj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4236
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putcone'):
    MSK_putcone = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putcone
    MSK_putcone.argtypes = [MSKtask_t, MSKint32t, MSKconetypee, MSKrealt, MSKint32t, POINTER(MSKint32t)]
    MSK_putcone.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4245
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_appendsparsesymmat'):
    MSK_appendsparsesymmat = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_appendsparsesymmat
    MSK_appendsparsesymmat.argtypes = [MSKtask_t, MSKint32t, MSKint64t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt), POINTER(MSKint64t)]
    MSK_appendsparsesymmat.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4255
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsymmatinfo'):
    MSK_getsymmatinfo = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsymmatinfo
    MSK_getsymmatinfo.argtypes = [MSKtask_t, MSKint64t, POINTER(MSKint32t), POINTER(MSKint64t), POINTER(MSKsymmattypee)]
    MSK_getsymmatinfo.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4263
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnumsymmat'):
    MSK_getnumsymmat = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnumsymmat
    MSK_getnumsymmat.argtypes = [MSKtask_t, POINTER(MSKint64t)]
    MSK_getnumsymmat.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4268
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsparsesymmat'):
    MSK_getsparsesymmat = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsparsesymmat
    MSK_getsparsesymmat.argtypes = [MSKtask_t, MSKint64t, MSKint64t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_getsparsesymmat.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4277
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putdouparam'):
    MSK_putdouparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putdouparam
    MSK_putdouparam.argtypes = [MSKtask_t, MSKdparame, MSKrealt]
    MSK_putdouparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4283
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putintparam'):
    MSK_putintparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putintparam
    MSK_putintparam.argtypes = [MSKtask_t, MSKiparame, MSKint32t]
    MSK_putintparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4289
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putmaxnumcon'):
    MSK_putmaxnumcon = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putmaxnumcon
    MSK_putmaxnumcon.argtypes = [MSKtask_t, MSKint32t]
    MSK_putmaxnumcon.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4294
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putmaxnumcone'):
    MSK_putmaxnumcone = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putmaxnumcone
    MSK_putmaxnumcone.argtypes = [MSKtask_t, MSKint32t]
    MSK_putmaxnumcone.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4299
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getmaxnumcone'):
    MSK_getmaxnumcone = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getmaxnumcone
    MSK_getmaxnumcone.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getmaxnumcone.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4304
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putmaxnumvar'):
    MSK_putmaxnumvar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putmaxnumvar
    MSK_putmaxnumvar.argtypes = [MSKtask_t, MSKint32t]
    MSK_putmaxnumvar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4309
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putmaxnumbarvar'):
    MSK_putmaxnumbarvar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putmaxnumbarvar
    MSK_putmaxnumbarvar.argtypes = [MSKtask_t, MSKint32t]
    MSK_putmaxnumbarvar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4314
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putmaxnumanz'):
    MSK_putmaxnumanz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putmaxnumanz
    MSK_putmaxnumanz.argtypes = [MSKtask_t, MSKint64t]
    MSK_putmaxnumanz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4319
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putmaxnumqnz'):
    MSK_putmaxnumqnz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putmaxnumqnz
    MSK_putmaxnumqnz.argtypes = [MSKtask_t, MSKint64t]
    MSK_putmaxnumqnz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4324
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getmaxnumqnz'):
    MSK_getmaxnumqnz = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getmaxnumqnz
    MSK_getmaxnumqnz.argtypes = [MSKtask_t, POINTER(MSKint32t)]
    MSK_getmaxnumqnz.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4329
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getmaxnumqnz64'):
    MSK_getmaxnumqnz64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getmaxnumqnz64
    MSK_getmaxnumqnz64.argtypes = [MSKtask_t, POINTER(MSKint64t)]
    MSK_getmaxnumqnz64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4334
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putnadouparam'):
    MSK_putnadouparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putnadouparam
    MSK_putnadouparam.argtypes = [MSKtask_t, String, MSKrealt]
    MSK_putnadouparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4340
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putnaintparam'):
    MSK_putnaintparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putnaintparam
    MSK_putnaintparam.argtypes = [MSKtask_t, String, MSKint32t]
    MSK_putnaintparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4346
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putnastrparam'):
    MSK_putnastrparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putnastrparam
    MSK_putnastrparam.argtypes = [MSKtask_t, String, String]
    MSK_putnastrparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4352
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putnlfunc'):
    MSK_putnlfunc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putnlfunc
    MSK_putnlfunc.argtypes = [MSKtask_t, MSKuserhandle_t, MSKnlgetspfunc, MSKnlgetvafunc]
    MSK_putnlfunc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4359
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getnlfunc'):
    MSK_getnlfunc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getnlfunc
    MSK_getnlfunc.argtypes = [MSKtask_t, POINTER(MSKuserhandle_t), POINTER(MSKnlgetspfunc), POINTER(MSKnlgetvafunc)]
    MSK_getnlfunc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4366
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putobjname'):
    MSK_putobjname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putobjname
    MSK_putobjname.argtypes = [MSKtask_t, String]
    MSK_putobjname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4371
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putparam'):
    MSK_putparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putparam
    MSK_putparam.argtypes = [MSKtask_t, String, String]
    MSK_putparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4377
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putqcon'):
    MSK_putqcon = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putqcon
    MSK_putqcon.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putqcon.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4386
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putqconk'):
    MSK_putqconk = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putqconk
    MSK_putqconk.argtypes = [MSKtask_t, MSKint32t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putqconk.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4395
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putqobj'):
    MSK_putqobj = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putqobj
    MSK_putqobj.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKrealt)]
    MSK_putqobj.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4403
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putqobjij'):
    MSK_putqobjij = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putqobjij
    MSK_putqobjij.argtypes = [MSKtask_t, MSKint32t, MSKint32t, MSKrealt]
    MSK_putqobjij.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4410
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putsolution'):
    MSK_putsolution = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putsolution
    MSK_putsolution.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKstakeye), POINTER(MSKstakeye), POINTER(MSKstakeye), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_putsolution.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4426
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putsolutioni'):
    MSK_putsolutioni = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putsolutioni
    MSK_putsolutioni.argtypes = [MSKtask_t, MSKaccmodee, MSKint32t, MSKsoltypee, MSKstakeye, MSKrealt, MSKrealt, MSKrealt, MSKrealt]
    MSK_putsolutioni.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4438
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putsolutionyi'):
    MSK_putsolutionyi = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putsolutionyi
    MSK_putsolutionyi.argtypes = [MSKtask_t, MSKint32t, MSKsoltypee, MSKrealt]
    MSK_putsolutionyi.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4445
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putstrparam'):
    MSK_putstrparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putstrparam
    MSK_putstrparam.argtypes = [MSKtask_t, MSKsparame, String]
    MSK_putstrparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4451
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_puttaskname'):
    MSK_puttaskname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_puttaskname
    MSK_puttaskname.argtypes = [MSKtask_t, String]
    MSK_puttaskname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4456
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putvartype'):
    MSK_putvartype = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putvartype
    MSK_putvartype.argtypes = [MSKtask_t, MSKint32t, MSKvariabletypee]
    MSK_putvartype.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4462
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putvartypelist'):
    MSK_putvartypelist = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putvartypelist
    MSK_putvartypelist.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKvariabletypee)]
    MSK_putvartypelist.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4469
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putvarbranchorder'):
    MSK_putvarbranchorder = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putvarbranchorder
    MSK_putvarbranchorder.argtypes = [MSKtask_t, MSKint32t, MSKint32t, c_int]
    MSK_putvarbranchorder.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4476
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getvarbranchorder'):
    MSK_getvarbranchorder = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getvarbranchorder
    MSK_getvarbranchorder.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKbranchdire)]
    MSK_getvarbranchorder.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4483
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getvarbranchpri'):
    MSK_getvarbranchpri = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getvarbranchpri
    MSK_getvarbranchpri.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t)]
    MSK_getvarbranchpri.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4489
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getvarbranchdir'):
    MSK_getvarbranchdir = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getvarbranchdir
    MSK_getvarbranchdir.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKbranchdire)]
    MSK_getvarbranchdir.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4495
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_readdata'):
    MSK_readdata = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_readdata
    MSK_readdata.argtypes = [MSKtask_t, String]
    MSK_readdata.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4500
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_readdataformat'):
    MSK_readdataformat = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_readdataformat
    MSK_readdataformat.argtypes = [MSKtask_t, String, c_int, c_int]
    MSK_readdataformat.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4507
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_readdataautoformat'):
    MSK_readdataautoformat = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_readdataautoformat
    MSK_readdataautoformat.argtypes = [MSKtask_t, String]
    MSK_readdataautoformat.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4512
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_readparamfile'):
    MSK_readparamfile = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_readparamfile
    MSK_readparamfile.argtypes = [MSKtask_t]
    MSK_readparamfile.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4516
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_readsolution'):
    MSK_readsolution = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_readsolution
    MSK_readsolution.argtypes = [MSKtask_t, MSKsoltypee, String]
    MSK_readsolution.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4522
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_readsummary'):
    MSK_readsummary = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_readsummary
    MSK_readsummary.argtypes = [MSKtask_t, MSKstreamtypee]
    MSK_readsummary.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4527
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_resizetask'):
    MSK_resizetask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_resizetask
    MSK_resizetask.argtypes = [MSKtask_t, MSKint32t, MSKint32t, MSKint32t, MSKint64t, MSKint64t]
    MSK_resizetask.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4536
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_checkmemtask'):
    MSK_checkmemtask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_checkmemtask
    MSK_checkmemtask.argtypes = [MSKtask_t, String, MSKint32t]
    MSK_checkmemtask.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4542
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getmemusagetask'):
    MSK_getmemusagetask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getmemusagetask
    MSK_getmemusagetask.argtypes = [MSKtask_t, POINTER(MSKint64t), POINTER(MSKint64t)]
    MSK_getmemusagetask.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4548
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_setdefaults'):
    MSK_setdefaults = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_setdefaults
    MSK_setdefaults.argtypes = [MSKtask_t]
    MSK_setdefaults.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4552
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_sktostr'):
    MSK_sktostr = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_sktostr
    MSK_sktostr.argtypes = [MSKtask_t, MSKstakeye, String]
    MSK_sktostr.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4558
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_solstatostr'):
    MSK_solstatostr = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_solstatostr
    MSK_solstatostr.argtypes = [MSKtask_t, MSKsolstae, String]
    MSK_solstatostr.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4564
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_solutiondef'):
    MSK_solutiondef = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_solutiondef
    MSK_solutiondef.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKbooleant)]
    MSK_solutiondef.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4570
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_deletesolution'):
    MSK_deletesolution = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_deletesolution
    MSK_deletesolution.argtypes = [MSKtask_t, MSKsoltypee]
    MSK_deletesolution.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4575
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_startstat'):
    MSK_startstat = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_startstat
    MSK_startstat.argtypes = [MSKtask_t]
    MSK_startstat.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4579
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_stopstat'):
    MSK_stopstat = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_stopstat
    MSK_stopstat.argtypes = [MSKtask_t]
    MSK_stopstat.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4583
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_appendstat'):
    MSK_appendstat = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_appendstat
    MSK_appendstat.argtypes = [MSKtask_t]
    MSK_appendstat.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4587
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_onesolutionsummary'):
    MSK_onesolutionsummary = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_onesolutionsummary
    MSK_onesolutionsummary.argtypes = [MSKtask_t, MSKstreamtypee, MSKsoltypee]
    MSK_onesolutionsummary.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4593
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_solutionsummary'):
    MSK_solutionsummary = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_solutionsummary
    MSK_solutionsummary.argtypes = [MSKtask_t, MSKstreamtypee]
    MSK_solutionsummary.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4598
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_updatesolutioninfo'):
    MSK_updatesolutioninfo = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_updatesolutioninfo
    MSK_updatesolutioninfo.argtypes = [MSKtask_t, MSKsoltypee]
    MSK_updatesolutioninfo.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4603
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_optimizersummary'):
    MSK_optimizersummary = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_optimizersummary
    MSK_optimizersummary.argtypes = [MSKtask_t, MSKstreamtypee]
    MSK_optimizersummary.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4608
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_strduptask'):
    MSK_strduptask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_strduptask
    MSK_strduptask.argtypes = [MSKtask_t, String]
    if sizeof(c_int) == sizeof(c_void_p):
        MSK_strduptask.restype = ReturnString
    else:
        MSK_strduptask.restype = String
        MSK_strduptask.errcheck = ReturnString

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4613
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_strdupdbgtask'):
    MSK_strdupdbgtask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_strdupdbgtask
    MSK_strdupdbgtask.argtypes = [MSKtask_t, String, String, c_uint]
    if sizeof(c_int) == sizeof(c_void_p):
        MSK_strdupdbgtask.restype = ReturnString
    else:
        MSK_strdupdbgtask.restype = String
        MSK_strdupdbgtask.errcheck = ReturnString

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4620
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_strtoconetype'):
    MSK_strtoconetype = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_strtoconetype
    MSK_strtoconetype.argtypes = [MSKtask_t, String, POINTER(MSKconetypee)]
    MSK_strtoconetype.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4626
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_strtosk'):
    MSK_strtosk = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_strtosk
    MSK_strtosk.argtypes = [MSKtask_t, String, POINTER(MSKint32t)]
    MSK_strtosk.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4632
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_whichparam'):
    MSK_whichparam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_whichparam
    MSK_whichparam.argtypes = [MSKtask_t, String, POINTER(MSKparametertypee), POINTER(MSKint32t)]
    MSK_whichparam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4639
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_writedata'):
    MSK_writedata = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_writedata
    MSK_writedata.argtypes = [MSKtask_t, String]
    MSK_writedata.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4644
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_writetask'):
    MSK_writetask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_writetask
    MSK_writetask.argtypes = [MSKtask_t, String]
    MSK_writetask.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4649
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_readtask'):
    MSK_readtask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_readtask
    MSK_readtask.argtypes = [MSKtask_t, String]
    MSK_readtask.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4654
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_readbranchpriorities'):
    MSK_readbranchpriorities = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_readbranchpriorities
    MSK_readbranchpriorities.argtypes = [MSKtask_t, String]
    MSK_readbranchpriorities.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4659
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_writebranchpriorities'):
    MSK_writebranchpriorities = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_writebranchpriorities
    MSK_writebranchpriorities.argtypes = [MSKtask_t, String]
    MSK_writebranchpriorities.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4664
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_writeparamfile'):
    MSK_writeparamfile = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_writeparamfile
    MSK_writeparamfile.argtypes = [MSKtask_t, String]
    MSK_writeparamfile.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4669
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getinfeasiblesubproblem'):
    MSK_getinfeasiblesubproblem = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getinfeasiblesubproblem
    MSK_getinfeasiblesubproblem.argtypes = [MSKtask_t, MSKsoltypee, POINTER(MSKtask_t)]
    MSK_getinfeasiblesubproblem.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4675
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_writesolution'):
    MSK_writesolution = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_writesolution
    MSK_writesolution.argtypes = [MSKtask_t, MSKsoltypee, String]
    MSK_writesolution.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4681
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_primalsensitivity'):
    MSK_primalsensitivity = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_primalsensitivity
    MSK_primalsensitivity.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKmarke), MSKint32t, POINTER(MSKint32t), POINTER(MSKmarke), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_primalsensitivity.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4699
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_sensitivityreport'):
    MSK_sensitivityreport = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_sensitivityreport
    MSK_sensitivityreport.argtypes = [MSKtask_t, MSKstreamtypee]
    MSK_sensitivityreport.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4704
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_dualsensitivity'):
    MSK_dualsensitivity = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_dualsensitivity
    MSK_dualsensitivity.argtypes = [MSKtask_t, MSKint32t, POINTER(MSKint32t), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt), POINTER(MSKrealt)]
    MSK_dualsensitivity.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4714
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_checkconvexity'):
    MSK_checkconvexity = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_checkconvexity
    MSK_checkconvexity.argtypes = [MSKtask_t]
    MSK_checkconvexity.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4718
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getlasterror'):
    MSK_getlasterror = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getlasterror
    MSK_getlasterror.argtypes = [MSKtask_t, POINTER(MSKrescodee), MSKint32t, POINTER(MSKint32t), String]
    MSK_getlasterror.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4726
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getlasterror64'):
    MSK_getlasterror64 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getlasterror64
    MSK_getlasterror64.argtypes = [MSKtask_t, POINTER(MSKrescodee), MSKint64t, POINTER(MSKint64t), String]
    MSK_getlasterror64.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4734
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_isinfinity'):
    MSK_isinfinity = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_isinfinity
    MSK_isinfinity.argtypes = [MSKrealt]
    MSK_isinfinity.restype = MSKbooleant

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4738
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_checkoutlicense'):
    MSK_checkoutlicense = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_checkoutlicense
    MSK_checkoutlicense.argtypes = [MSKenv_t, MSKfeaturee]
    MSK_checkoutlicense.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4743
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_checkinlicense'):
    MSK_checkinlicense = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_checkinlicense
    MSK_checkinlicense.argtypes = [MSKenv_t, MSKfeaturee]
    MSK_checkinlicense.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4748
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getbuildinfo'):
    MSK_getbuildinfo = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getbuildinfo
    MSK_getbuildinfo.argtypes = [String, String, String]
    MSK_getbuildinfo.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4754
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getresponseclass'):
    MSK_getresponseclass = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getresponseclass
    MSK_getresponseclass.argtypes = [MSKrescodee, POINTER(MSKrescodetypee)]
    MSK_getresponseclass.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4759
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_callocenv'):
    MSK_callocenv = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_callocenv
    MSK_callocenv.argtypes = [MSKenv_t, c_size_t, c_size_t]
    MSK_callocenv.restype = POINTER(None)

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4765
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_callocdbgenv'):
    MSK_callocdbgenv = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_callocdbgenv
    MSK_callocdbgenv.argtypes = [MSKenv_t, c_size_t, c_size_t, String, c_uint]
    MSK_callocdbgenv.restype = POINTER(None)

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4773
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_deleteenv'):
    MSK_deleteenv = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_deleteenv
    MSK_deleteenv.argtypes = [POINTER(MSKenv_t)]
    MSK_deleteenv.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4777
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_echoenv'):
    _func = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_echoenv
    _restype = MSKrescodee
    _argtypes = [MSKenv_t, MSKstreamtypee, String]
    MSK_echoenv = _variadic_function(_func,_restype,_argtypes)

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4784
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_echointro'):
    MSK_echointro = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_echointro
    MSK_echointro.argtypes = [MSKenv_t, MSKint32t]
    MSK_echointro.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4789
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_freeenv'):
    MSK_freeenv = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_freeenv
    MSK_freeenv.argtypes = [MSKenv_t, POINTER(None)]
    MSK_freeenv.restype = None

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4794
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_freedbgenv'):
    MSK_freedbgenv = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_freedbgenv
    MSK_freedbgenv.argtypes = [MSKenv_t, POINTER(None), String, c_uint]
    MSK_freedbgenv.restype = None

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4801
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getcodedesc'):
    MSK_getcodedesc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getcodedesc
    MSK_getcodedesc.argtypes = [MSKrescodee, String, String]
    MSK_getcodedesc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4807
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getsymbcondim'):
    MSK_getsymbcondim = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getsymbcondim
    MSK_getsymbcondim.argtypes = [MSKenv_t, POINTER(MSKint32t), POINTER(c_size_t)]
    MSK_getsymbcondim.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4813
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getversion'):
    MSK_getversion = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getversion
    MSK_getversion.argtypes = [POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t), POINTER(MSKint32t)]
    MSK_getversion.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4820
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_checkversion'):
    MSK_checkversion = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_checkversion
    MSK_checkversion.argtypes = [MSKenv_t, MSKint32t, MSKint32t, MSKint32t, MSKint32t]
    MSK_checkversion.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4828
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_iparvaltosymnam'):
    MSK_iparvaltosymnam = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_iparvaltosymnam
    MSK_iparvaltosymnam.argtypes = [MSKenv_t, MSKiparame, MSKint32t, String]
    MSK_iparvaltosymnam.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4835
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_linkfiletoenvstream'):
    MSK_linkfiletoenvstream = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_linkfiletoenvstream
    MSK_linkfiletoenvstream.argtypes = [MSKenv_t, MSKstreamtypee, String, MSKint32t]
    MSK_linkfiletoenvstream.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4842
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_linkfunctoenvstream'):
    MSK_linkfunctoenvstream = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_linkfunctoenvstream
    MSK_linkfunctoenvstream.argtypes = [MSKenv_t, MSKstreamtypee, MSKuserhandle_t, MSKstreamfunc]
    MSK_linkfunctoenvstream.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4849
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_unlinkfuncfromenvstream'):
    MSK_unlinkfuncfromenvstream = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_unlinkfuncfromenvstream
    MSK_unlinkfuncfromenvstream.argtypes = [MSKenv_t, MSKstreamtypee]
    MSK_unlinkfuncfromenvstream.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4854
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_makeenv'):
    MSK_makeenv = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_makeenv
    MSK_makeenv.argtypes = [POINTER(MSKenv_t), String]
    MSK_makeenv.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4859
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_makeenvalloc'):
    MSK_makeenvalloc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_makeenvalloc
    MSK_makeenvalloc.argtypes = [POINTER(MSKenv_t), MSKuserhandle_t, MSKmallocfunc, MSKcallocfunc, MSKreallocfunc, MSKfreefunc, String]
    MSK_makeenvalloc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4869
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_initenv'):
    MSK_initenv = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_initenv
    MSK_initenv.argtypes = [MSKenv_t]
    MSK_initenv.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4873
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_getglbdllname'):
    MSK_getglbdllname = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_getglbdllname
    MSK_getglbdllname.argtypes = [MSKenv_t, c_size_t, String]
    MSK_getglbdllname.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4879
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putdllpath'):
    MSK_putdllpath = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putdllpath
    MSK_putdllpath.argtypes = [MSKenv_t, String]
    MSK_putdllpath.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4884
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putlicensedebug'):
    MSK_putlicensedebug = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putlicensedebug
    MSK_putlicensedebug.argtypes = [MSKenv_t, MSKint32t]
    MSK_putlicensedebug.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4889
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putlicensecode'):
    MSK_putlicensecode = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putlicensecode
    MSK_putlicensecode.argtypes = [MSKenv_t, POINTER(MSKint32t)]
    MSK_putlicensecode.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4894
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putlicensewait'):
    MSK_putlicensewait = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putlicensewait
    MSK_putlicensewait.argtypes = [MSKenv_t, MSKint32t]
    MSK_putlicensewait.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4899
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putlicensepath'):
    MSK_putlicensepath = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putlicensepath
    MSK_putlicensepath.argtypes = [MSKenv_t, String]
    MSK_putlicensepath.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4904
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putkeepdlls'):
    MSK_putkeepdlls = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putkeepdlls
    MSK_putkeepdlls.argtypes = [MSKenv_t, MSKint32t]
    MSK_putkeepdlls.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4909
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_maketask'):
    MSK_maketask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_maketask
    MSK_maketask.argtypes = [MSKenv_t, MSKint32t, MSKint32t, POINTER(MSKtask_t)]
    MSK_maketask.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4916
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_makeemptytask'):
    MSK_makeemptytask = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_makeemptytask
    MSK_makeemptytask.argtypes = [MSKenv_t, POINTER(MSKtask_t)]
    MSK_makeemptytask.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4921
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_putexitfunc'):
    MSK_putexitfunc = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_putexitfunc
    MSK_putexitfunc.argtypes = [MSKenv_t, MSKexitfunc, MSKuserhandle_t]
    MSK_putexitfunc.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4927
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_utf8towchar'):
    MSK_utf8towchar = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_utf8towchar
    MSK_utf8towchar.argtypes = [c_size_t, POINTER(c_size_t), POINTER(c_size_t), POINTER(MSKwchart), String]
    MSK_utf8towchar.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4935
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_wchartoutf8'):
    MSK_wchartoutf8 = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_wchartoutf8
    MSK_wchartoutf8.argtypes = [c_size_t, POINTER(c_size_t), POINTER(c_size_t), String, POINTER(MSKwchart)]
    MSK_wchartoutf8.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4943
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_checkmemenv'):
    MSK_checkmemenv = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_checkmemenv
    MSK_checkmemenv.argtypes = [MSKenv_t, String, MSKint32t]
    MSK_checkmemenv.restype = MSKrescodee

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4949
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_strdupenv'):
    MSK_strdupenv = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_strdupenv
    MSK_strdupenv.argtypes = [MSKenv_t, String]
    if sizeof(c_int) == sizeof(c_void_p):
        MSK_strdupenv.restype = ReturnString
    else:
        MSK_strdupenv.restype = String
        MSK_strdupenv.errcheck = ReturnString

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4954
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_strdupdbgenv'):
    MSK_strdupdbgenv = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_strdupdbgenv
    MSK_strdupdbgenv.argtypes = [MSKenv_t, String, String, c_uint]
    if sizeof(c_int) == sizeof(c_void_p):
        MSK_strdupdbgenv.restype = ReturnString
    else:
        MSK_strdupdbgenv.restype = String
        MSK_strdupdbgenv.errcheck = ReturnString

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4961
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_symnamtovalue'):
    MSK_symnamtovalue = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_symnamtovalue
    MSK_symnamtovalue.argtypes = [String, String]
    MSK_symnamtovalue.restype = MSKbooleant

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 4966
if hasattr(_libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'], 'MSK_licensecleanup'):
    MSK_licensecleanup = _libs['C:/Program Files/Mosek/7/tools/platform/win64x86/bin/mosek64_7_0.dll'].MSK_licensecleanup
    MSK_licensecleanup.argtypes = []
    MSK_licensecleanup.restype = MSKrescodee

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __STDC__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __STDC_HOSTED__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GNUC__ = 4
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GNUC_MINOR__ = 7
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GNUC_PATCHLEVEL__ = 0
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __VERSION__ = '4.7.0 20111220 (experimental)'
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __ATOMIC_RELAXED = 0
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __ATOMIC_SEQ_CST = 5
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __ATOMIC_ACQUIRE = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __ATOMIC_RELEASE = 3
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __ATOMIC_ACQ_REL = 4
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __ATOMIC_CONSUME = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __pic__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __PIC__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FINITE_MATH_ONLY__ = 0
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_INT__ = 4
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_LONG__ = 4
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_LONG_LONG__ = 8
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_SHORT__ = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_FLOAT__ = 4
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_DOUBLE__ = 8
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_LONG_DOUBLE__ = 16
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_SIZE_T__ = 8
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __CHAR_BIT__ = 8
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __BIGGEST_ALIGNMENT__ = 16
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __ORDER_LITTLE_ENDIAN__ = 1234
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __ORDER_BIG_ENDIAN__ = 4321
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __ORDER_PDP_ENDIAN__ = 3412
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __BYTE_ORDER__ = __ORDER_LITTLE_ENDIAN__
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLOAT_WORD_ORDER__ = __ORDER_LITTLE_ENDIAN__
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_POINTER__ = 8
except:
    pass

__SIZE_TYPE__ = c_ulonglong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__PTRDIFF_TYPE__ = c_longlong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__WCHAR_TYPE__ = c_uint # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__WINT_TYPE__ = c_uint # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INTMAX_TYPE__ = c_longlong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINTMAX_TYPE__ = c_ulonglong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__CHAR16_TYPE__ = c_uint # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__CHAR32_TYPE__ = c_uint # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__SIG_ATOMIC_TYPE__ = c_int # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT8_TYPE__ = c_char # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT16_TYPE__ = c_int # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT32_TYPE__ = c_int # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT64_TYPE__ = c_longlong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT8_TYPE__ = c_ubyte # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT16_TYPE__ = c_uint # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT32_TYPE__ = c_uint # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT64_TYPE__ = c_ulonglong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT_LEAST8_TYPE__ = c_char # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT_LEAST16_TYPE__ = c_int # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT_LEAST32_TYPE__ = c_int # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT_LEAST64_TYPE__ = c_longlong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT_LEAST8_TYPE__ = c_ubyte # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT_LEAST16_TYPE__ = c_uint # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT_LEAST32_TYPE__ = c_uint # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT_LEAST64_TYPE__ = c_ulonglong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT_FAST8_TYPE__ = c_char # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT_FAST16_TYPE__ = c_int # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT_FAST32_TYPE__ = c_int # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INT_FAST64_TYPE__ = c_longlong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT_FAST8_TYPE__ = c_ubyte # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT_FAST16_TYPE__ = c_uint # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT_FAST32_TYPE__ = c_uint # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINT_FAST64_TYPE__ = c_ulonglong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__INTPTR_TYPE__ = c_longlong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

__UINTPTR_TYPE__ = c_ulonglong # c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GXX_ABI_VERSION = 1002
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __USING_SJLJ_EXCEPTIONS__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SCHAR_MAX__ = 127
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SHRT_MAX__ = 32767
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT_MAX__ = 2147483647
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LONG_MAX__ = 2147483647L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LONG_LONG_MAX__ = 9223372036854775807L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __WCHAR_MAX__ = 65535
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __WCHAR_MIN__ = 0
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __WINT_MAX__ = 65535
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __WINT_MIN__ = 0
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __PTRDIFF_MAX__ = 9223372036854775807L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZE_MAX__ = 18446744073709551615L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INTMAX_MAX__ = 9223372036854775807L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINTMAX_MAX__ = 18446744073709551615L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIG_ATOMIC_MAX__ = 2147483647
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIG_ATOMIC_MIN__ = ((-__SIG_ATOMIC_MAX__) - 1)
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT8_MAX__ = 127
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT16_MAX__ = 32767
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT32_MAX__ = 2147483647
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT64_MAX__ = 9223372036854775807L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT8_MAX__ = 255
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT16_MAX__ = 65535
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT32_MAX__ = 4294967295L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT64_MAX__ = 18446744073709551615L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT_LEAST8_MAX__ = 127
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
def __INT8_C(c):
    return c

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT_LEAST16_MAX__ = 32767
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
def __INT16_C(c):
    return c

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT_LEAST32_MAX__ = 2147483647
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
def __INT32_C(c):
    return c

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT_LEAST64_MAX__ = 9223372036854775807L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT_LEAST8_MAX__ = 255
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
def __UINT8_C(c):
    return c

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT_LEAST16_MAX__ = 65535
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
def __UINT16_C(c):
    return c

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT_LEAST32_MAX__ = 4294967295L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT_LEAST64_MAX__ = 18446744073709551615L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT_FAST8_MAX__ = 127
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT_FAST16_MAX__ = 32767
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT_FAST32_MAX__ = 2147483647
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INT_FAST64_MAX__ = 9223372036854775807L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT_FAST8_MAX__ = 255
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT_FAST16_MAX__ = 65535
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT_FAST32_MAX__ = 4294967295L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINT_FAST64_MAX__ = 18446744073709551615L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __INTPTR_MAX__ = 9223372036854775807L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __UINTPTR_MAX__ = 18446744073709551615L
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_EVAL_METHOD__ = 0
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DEC_EVAL_METHOD__ = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_RADIX__ = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_MANT_DIG__ = 24
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_DIG__ = 6
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_MIN_EXP__ = (-125)
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_MIN_10_EXP__ = (-37)
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_MAX_EXP__ = 128
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_MAX_10_EXP__ = 38
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_DECIMAL_DIG__ = 9
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_MAX__ = 3.4028234663852886e+38
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_MIN__ = 1.1754943508222875e-38
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_EPSILON__ = 1.1920928955078125e-07
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_DENORM_MIN__ = 1.401298464324817e-45
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_HAS_DENORM__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_HAS_INFINITY__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __FLT_HAS_QUIET_NAN__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_MANT_DIG__ = 53
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_DIG__ = 15
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_MIN_EXP__ = (-1021)
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_MIN_10_EXP__ = (-307)
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_MAX_EXP__ = 1024
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_MAX_10_EXP__ = 308
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_DECIMAL_DIG__ = 17
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_MAX__ = 1.7976931348623157e+308
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_MIN__ = 2.2250738585072014e-308
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_EPSILON__ = 2.220446049250313e-16
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_DENORM_MIN__ = 5e-324
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_HAS_DENORM__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_HAS_INFINITY__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DBL_HAS_QUIET_NAN__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_MANT_DIG__ = 64
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_DIG__ = 18
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_MIN_EXP__ = (-16381)
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_MIN_10_EXP__ = (-4931)
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_MAX_EXP__ = 16384
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_MAX_10_EXP__ = 4932
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DECIMAL_DIG__ = 21
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_MAX__ = float('inf')
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_MIN__ = 0.0
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_EPSILON__ = 1.0842021724855044e-19
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_DENORM_MIN__ = 0.0
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_HAS_DENORM__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_HAS_INFINITY__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __LDBL_HAS_QUIET_NAN__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DEC32_MANT_DIG__ = 7
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DEC32_MIN_EXP__ = (-94)
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DEC32_MAX_EXP__ = 97
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DEC64_MANT_DIG__ = 16
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DEC64_MIN_EXP__ = (-382)
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DEC64_MAX_EXP__ = 385
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DEC128_MANT_DIG__ = 34
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DEC128_MIN_EXP__ = (-6142)
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DEC128_MAX_EXP__ = 6145
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GNUC_GNU_INLINE__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __NO_INLINE__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_ATOMIC_BOOL_LOCK_FREE = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_ATOMIC_CHAR_LOCK_FREE = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_ATOMIC_CHAR16_T_LOCK_FREE = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_ATOMIC_CHAR32_T_LOCK_FREE = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_ATOMIC_WCHAR_T_LOCK_FREE = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_ATOMIC_SHORT_LOCK_FREE = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_ATOMIC_INT_LOCK_FREE = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_ATOMIC_LONG_LOCK_FREE = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_ATOMIC_LLONG_LOCK_FREE = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GCC_ATOMIC_POINTER_LOCK_FREE = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __PRAGMA_REDEFINE_EXTNAME = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_INT128__ = 16
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_WCHAR_T__ = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_WINT_T__ = 2
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SIZEOF_PTRDIFF_T__ = 8
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __amd64 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __amd64__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __x86_64 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __x86_64__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __k8 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __k8__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __MMX__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SSE__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SSE2__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SSE_MATH__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SSE2_MATH__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __SEH__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GXX_MERGED_TYPEINFO_NAMES = 0
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __GXX_TYPEINFO_EQUALITY_INLINE = 0
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __MSVCRT__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __MINGW32__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    _WIN32 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __WIN32 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __WIN32__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    WIN32 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __WINNT = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __WINNT__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    WINNT = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    _INTEGRAL_MAX_BITS = 64
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __MINGW64__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __WIN64 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __WIN64__ = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    WIN64 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    _WIN64 = 1
except:
    pass

# c:\\users\\roots\\appdata\\local\\temp\\tmpvur4vo.h: 1
try:
    __DECIMAL_BID_FORMAT__ = 1
except:
    pass

__const = c_int # <command-line>: 4

# <command-line>: 7
try:
    CTYPESGEN = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 10
def __STRINGIFY(x):
    return x

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 11
def __MINGW64_STRINGIFY(x):
    return (__STRINGIFY (x))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 13
try:
    __MINGW64_VERSION_MAJOR = 3
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 14
try:
    __MINGW64_VERSION_MINOR = 0
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 16
try:
    __MINGW64_VERSION_STATE = 'alpha'
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 22
try:
    __MINGW32_MAJOR_VERSION = 3
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 23
try:
    __MINGW32_MINOR_VERSION = 11
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 34
try:
    _ = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 38
try:
    __MINGW_USE_UNDERSCORE_PREFIX = 0
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 55
def __MINGW_USYMBOL(sym):
    return sym

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 66
try:
    __USE_MINGW_OUTPUT_FORMAT_EMU = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 89
try:
    _M_AMD64 = 100
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 90
try:
    _M_X64 = 100
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 142
try:
    __MINGW_HAVE_ANSI_C99_PRINTF = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 143
try:
    __MINGW_HAVE_WIDE_C99_PRINTF = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 144
try:
    __MINGW_HAVE_ANSI_C99_SCANF = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw_mac.h: 145
try:
    __MINGW_HAVE_WIDE_C99_SCANF = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 62
def __MINGW_GNUC_PREREQ(major, minor):
    return 0

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 69
def __MINGW_MSC_PREREQ(major, minor):
    return 0

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 75
try:
    USE___UUIDOF = 0
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 97
try:
    __CRT__NO_INLINE = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 106
def __UNUSED_PARAM(x):
    return x

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 218
try:
    __MINGW_SEC_WARN_STR = 'This function or variable may be unsafe, use _CRT_SECURE_NO_WARNINGS to disable deprecation'
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 219
try:
    __MINGW_MSVC2005_DEPREC_STR = 'This POSIX function is deprecated beginning in Visual C++ 2005, use _CRT_NONSTDC_NO_DEPRECATE to disable deprecation'
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 235
try:
    __MSVCRT_VERSION__ = 1792
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 240
try:
    WINVER = 1282
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 244
try:
    _WIN32_WINNT = 1282
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 291
try:
    _CRT_PACKING = 8
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/vadefs.h: 44
def _ADDRESSOF(v):
    return pointer(v)

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 300
def __CRT_STRINGIZE(_Value):
    return _Value

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 301
def _CRT_STRINGIZE(_Value):
    return (__CRT_STRINGIZE (_Value))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 363
try:
    _SECURECRT_FILL_BUFFER_PATTERN = 253
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 518
try:
    _ARGMAX = 100
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 521
try:
    _TRUNCATE = (-1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 525
def _CRT_UNUSED(x):
    return x

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 20
try:
    PATH_MAX = 260
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 22
try:
    CHAR_BIT = 8
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 23
try:
    SCHAR_MIN = (-128)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 24
try:
    SCHAR_MAX = 127
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 25
try:
    UCHAR_MAX = 255
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 27
try:
    CHAR_MIN = SCHAR_MIN
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 28
try:
    CHAR_MAX = SCHAR_MAX
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 30
try:
    MB_LEN_MAX = 5
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 31
try:
    SHRT_MIN = (-32768)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 32
try:
    SHRT_MAX = 32767
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 33
try:
    USHRT_MAX = 65535
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 34
try:
    INT_MIN = ((-2147483647) - 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 35
try:
    INT_MAX = 2147483647
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 36
try:
    UINT_MAX = 4294967295L
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 37
try:
    LONG_MIN = ((-2147483647L) - 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 38
try:
    LONG_MAX = 2147483647L
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 39
try:
    ULONG_MAX = 4294967295L
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 40
try:
    LLONG_MAX = 9223372036854775807L
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 41
try:
    LLONG_MIN = ((-9223372036854775807L) - 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 42
try:
    ULLONG_MAX = 18446744073709551615L
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 44
try:
    _I8_MIN = ((-127) - 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 45
try:
    _I8_MAX = 127
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 46
try:
    _UI8_MAX = 255
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 48
try:
    _I16_MIN = ((-32767) - 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 49
try:
    _I16_MAX = 32767
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 50
try:
    _UI16_MAX = 65535
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 52
try:
    _I32_MIN = ((-2147483647) - 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 53
try:
    _I32_MAX = 2147483647
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 54
try:
    _UI32_MAX = 4294967295L
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 65
try:
    _I64_MIN = ((-9223372036854775807L) - 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 66
try:
    _I64_MAX = 9223372036854775807L
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 67
try:
    _UI64_MAX = 18446744073709551615L
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 71
try:
    SIZE_MAX = _UI64_MAX
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/limits.h: 79
try:
    SSIZE_MAX = _I64_MAX
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 65
try:
    CHAR_BIT = __CHAR_BIT__
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 74
try:
    SCHAR_MIN = ((-SCHAR_MAX) - 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 76
try:
    SCHAR_MAX = __SCHAR_MAX__
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 83
try:
    UCHAR_MAX = ((SCHAR_MAX * 2) + 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 98
try:
    CHAR_MIN = SCHAR_MIN
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 100
try:
    CHAR_MAX = SCHAR_MAX
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 105
try:
    SHRT_MIN = ((-SHRT_MAX) - 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 107
try:
    SHRT_MAX = __SHRT_MAX__
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 114
try:
    USHRT_MAX = ((SHRT_MAX * 2) + 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 119
try:
    INT_MIN = ((-INT_MAX) - 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 121
try:
    INT_MAX = __INT_MAX__
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 125
try:
    UINT_MAX = ((INT_MAX * 2) + 1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 130
try:
    LONG_MIN = ((-LONG_MAX) - 1L)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 132
try:
    LONG_MAX = __LONG_MAX__
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 136
try:
    ULONG_MAX = ((LONG_MAX * 2L) + 1L)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 153
try:
    LONG_LONG_MIN = ((-LONG_LONG_MAX) - 1L)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 155
try:
    LONG_LONG_MAX = __LONG_LONG_MAX__
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/include-fixed/limits.h: 159
try:
    ULONG_LONG_MAX = ((LONG_LONG_MAX * 2L) + 1L)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 30
try:
    EXIT_SUCCESS = 0
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 31
try:
    EXIT_FAILURE = 1
except:
    pass

onexit_t = _onexit_t # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 39

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 66
def _PTR_LD(x):
    return pointer((x.contents.ld))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 92
try:
    RAND_MAX = 32767
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 108
def __max(a, b):
    return (a > b) and a or b

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 109
def __min(a, b):
    return (a < b) and a or b

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 111
try:
    _MAX_PATH = 260
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 112
try:
    _MAX_DRIVE = 3
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 113
try:
    _MAX_DIR = 256
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 114
try:
    _MAX_FNAME = 256
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 115
try:
    _MAX_EXT = 256
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 117
try:
    _OUT_TO_DEFAULT = 0
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 118
try:
    _OUT_TO_STDERR = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 119
try:
    _OUT_TO_MSGBOX = 2
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 120
try:
    _REPORT_ERRMODE = 3
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 122
try:
    _WRITE_ABORT_MSG = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 123
try:
    _CALL_REPORTFAULT = 2
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 125
try:
    _MAX_ENV = 32767
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 139
try:
    errno = ((_errno ())[0])
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 144
try:
    _doserrno = ((__doserrno ())[0])
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 291
def _countof(_Array):
    return (sizeof(_Array) / sizeof((_Array [0])))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 391
try:
    strtod = __strtod
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 483
try:
    _CVTBUFSIZE = (309 + 40)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 571
try:
    sys_errlist = _sys_errlist
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 572
try:
    sys_nerr = _sys_nerr
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 18
try:
    _HEAP_MAXREQ = 18446744073709551584L
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 32
try:
    _HEAPEMPTY = (-1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 33
try:
    _HEAPOK = (-2)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 34
try:
    _HEAPBADBEGIN = (-3)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 35
try:
    _HEAPBADNODE = (-4)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 36
try:
    _HEAPEND = (-5)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 37
try:
    _HEAPBADPTR = (-6)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 40
try:
    _FREEENTRY = 0
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 41
try:
    _USEDENTRY = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 71
def _mm_free(a):
    return (_aligned_free (a))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 72
def _mm_malloc(a, b):
    return (_aligned_malloc (a, b))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 108
try:
    _MAX_WAIT_MALLOC_CRT = 60000
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 133
try:
    _ALLOCA_S_THRESHOLD = 1024
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 134
try:
    _ALLOCA_S_STACK_MARKER = 52428
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 135
try:
    _ALLOCA_S_HEAP_MARKER = 56797
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 140
try:
    _ALLOCA_S_MARKER_SIZE = 16
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 185
try:
    alloca = _alloca
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 19
try:
    BUFSIZ = 512
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 20
try:
    _NFILE = _NSTREAM_
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 21
try:
    _NSTREAM_ = 512
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 22
try:
    _IOB_ENTRIES = 20
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 23
try:
    EOF = (-1)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 44
try:
    _P_tmpdir = '\\'
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 48
try:
    L_tmpnam = (sizeof(_P_tmpdir) + 12)
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 55
try:
    SEEK_CUR = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 56
try:
    SEEK_END = 2
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 57
try:
    SEEK_SET = 0
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 59
try:
    STDIN_FILENO = 0
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 60
try:
    STDOUT_FILENO = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 61
try:
    STDERR_FILENO = 2
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 63
try:
    FILENAME_MAX = 260
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 64
try:
    FOPEN_MAX = 20
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 65
try:
    _SYS_OPEN = 20
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 66
try:
    TMP_MAX = 32767
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 85
try:
    _iob = (__iob_func ())
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 104
def _FPOSOFF(fp):
    return fp

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 115
try:
    stdin = pointer(((__iob_func ()) [0]))
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 116
try:
    stdout = pointer(((__iob_func ()) [1]))
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 117
try:
    stderr = pointer(((__iob_func ()) [2]))
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 120
try:
    _IOREAD = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 121
try:
    _IOWRT = 2
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 123
try:
    _IOFBF = 0
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 124
try:
    _IOLBF = 64
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 125
try:
    _IONBF = 4
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 127
try:
    _IOMYBUF = 8
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 128
try:
    _IOEOF = 16
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 129
try:
    _IOERR = 32
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 130
try:
    _IOSTRG = 64
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 131
try:
    _IORW = 128
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 136
try:
    _TWO_DIGIT_EXPONENT = 1
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 489
try:
    popen = _popen
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 490
try:
    pclose = _pclose
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 515
try:
    _set_output_format = __mingw_set_output_format
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 516
try:
    _get_output_format = __mingw_get_output_format
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 793
try:
    WEOF = 65535
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 910
try:
    wpopen = _wpopen
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 922
try:
    getwchar = (fgetwc (stdin))
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 923
def putwchar(_c):
    return (fputwc (_c, stdout))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 929
def getwc(_stm):
    return (fgetwc (_stm))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 930
def putwc(_c, _stm):
    return (fputwc (_c, _stm))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 931
def _putwc_nolock(_c, _stm):
    return (_fputwc_nolock (_c, _stm))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 932
def _getwc_nolock(_c):
    return (_fgetwc_nolock (_c))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 938
def _fgetc_nolock(_stream):
    return ((((_stream.contents._cnt).value) - 1) >= 0) and (255 & ((((_stream.contents._ptr).value) + 1)[0])) or (_filbuf (_stream))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 939
def _fputc_nolock(_c, _stream):
    return ((((_stream.contents._cnt).value) - 1) >= 0) and (255 & _c) or (_flsbuf (_c, _stream))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 940
def _getc_nolock(_stream):
    return (_fgetc_nolock (_stream))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 941
def _putc_nolock(_c, _stream):
    return (_fputc_nolock (_c, _stream))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 942
try:
    _getchar_nolock = (_getc_nolock (stdin))
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 943
def _putchar_nolock(_c):
    return (_putc_nolock (_c, stdout))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 944
try:
    _getwchar_nolock = (_getwc_nolock (stdin))
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 945
def _putwchar_nolock(_c):
    return (_putwc_nolock (_c, stdout))

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 960
try:
    P_tmpdir = _P_tmpdir
except:
    pass

# c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 961
try:
    SYS_OPEN = _SYS_OPEN
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 24
try:
    MSK_VERSION_MAJOR = 7
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 25
try:
    MSK_VERSION_MINOR = 0
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 26
try:
    MSK_VERSION_BUILD = 0
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 27
try:
    MSK_VERSION_REVISION = 121
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 28
try:
    MSK_VERSION_STATE = ''
except:
    pass

MSKCONST = c_int # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 32

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 36
try:
    MSK_INFINITY = 1e+30
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1584
try:
    MSK_FIRST_ERR_CODE = 1000
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1585
try:
    MSK_LAST_ERR_CODE = 9999
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1590
try:
    MSK_SPAR_BAS_SOL_FILE_NAME_ = 'MSK_SPAR_BAS_SOL_FILE_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1591
try:
    MSK_SPAR_DATA_FILE_NAME_ = 'MSK_SPAR_DATA_FILE_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1592
try:
    MSK_SPAR_DEBUG_FILE_NAME_ = 'MSK_SPAR_DEBUG_FILE_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1593
try:
    MSK_SPAR_FEASREPAIR_NAME_PREFIX_ = 'MSK_SPAR_FEASREPAIR_NAME_PREFIX'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1594
try:
    MSK_SPAR_FEASREPAIR_NAME_SEPARATOR_ = 'MSK_SPAR_FEASREPAIR_NAME_SEPARATOR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1595
try:
    MSK_SPAR_FEASREPAIR_NAME_WSUMVIOL_ = 'MSK_SPAR_FEASREPAIR_NAME_WSUMVIOL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1596
try:
    MSK_SPAR_INT_SOL_FILE_NAME_ = 'MSK_SPAR_INT_SOL_FILE_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1597
try:
    MSK_SPAR_ITR_SOL_FILE_NAME_ = 'MSK_SPAR_ITR_SOL_FILE_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1598
try:
    MSK_SPAR_MIO_DEBUG_STRING_ = 'MSK_SPAR_MIO_DEBUG_STRING'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1599
try:
    MSK_SPAR_PARAM_COMMENT_SIGN_ = 'MSK_SPAR_PARAM_COMMENT_SIGN'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1600
try:
    MSK_SPAR_PARAM_READ_FILE_NAME_ = 'MSK_SPAR_PARAM_READ_FILE_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1601
try:
    MSK_SPAR_PARAM_WRITE_FILE_NAME_ = 'MSK_SPAR_PARAM_WRITE_FILE_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1602
try:
    MSK_SPAR_READ_MPS_BOU_NAME_ = 'MSK_SPAR_READ_MPS_BOU_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1603
try:
    MSK_SPAR_READ_MPS_OBJ_NAME_ = 'MSK_SPAR_READ_MPS_OBJ_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1604
try:
    MSK_SPAR_READ_MPS_RAN_NAME_ = 'MSK_SPAR_READ_MPS_RAN_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1605
try:
    MSK_SPAR_READ_MPS_RHS_NAME_ = 'MSK_SPAR_READ_MPS_RHS_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1606
try:
    MSK_SPAR_SENSITIVITY_FILE_NAME_ = 'MSK_SPAR_SENSITIVITY_FILE_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1607
try:
    MSK_SPAR_SENSITIVITY_RES_FILE_NAME_ = 'MSK_SPAR_SENSITIVITY_RES_FILE_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1608
try:
    MSK_SPAR_SOL_FILTER_XC_LOW_ = 'MSK_SPAR_SOL_FILTER_XC_LOW'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1609
try:
    MSK_SPAR_SOL_FILTER_XC_UPR_ = 'MSK_SPAR_SOL_FILTER_XC_UPR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1610
try:
    MSK_SPAR_SOL_FILTER_XX_LOW_ = 'MSK_SPAR_SOL_FILTER_XX_LOW'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1611
try:
    MSK_SPAR_SOL_FILTER_XX_UPR_ = 'MSK_SPAR_SOL_FILTER_XX_UPR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1612
try:
    MSK_SPAR_STAT_FILE_NAME_ = 'MSK_SPAR_STAT_FILE_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1613
try:
    MSK_SPAR_STAT_KEY_ = 'MSK_SPAR_STAT_KEY'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1614
try:
    MSK_SPAR_STAT_NAME_ = 'MSK_SPAR_STAT_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1615
try:
    MSK_SPAR_WRITE_LP_GEN_VAR_NAME_ = 'MSK_SPAR_WRITE_LP_GEN_VAR_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1617
try:
    MSK_DPAR_ANA_SOL_INFEAS_TOL_ = 'MSK_DPAR_ANA_SOL_INFEAS_TOL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1618
try:
    MSK_DPAR_BASIS_REL_TOL_S_ = 'MSK_DPAR_BASIS_REL_TOL_S'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1619
try:
    MSK_DPAR_BASIS_TOL_S_ = 'MSK_DPAR_BASIS_TOL_S'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1620
try:
    MSK_DPAR_BASIS_TOL_X_ = 'MSK_DPAR_BASIS_TOL_X'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1621
try:
    MSK_DPAR_CHECK_CONVEXITY_REL_TOL_ = 'MSK_DPAR_CHECK_CONVEXITY_REL_TOL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1622
try:
    MSK_DPAR_DATA_TOL_AIJ_ = 'MSK_DPAR_DATA_TOL_AIJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1623
try:
    MSK_DPAR_DATA_TOL_AIJ_HUGE_ = 'MSK_DPAR_DATA_TOL_AIJ_HUGE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1624
try:
    MSK_DPAR_DATA_TOL_AIJ_LARGE_ = 'MSK_DPAR_DATA_TOL_AIJ_LARGE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1625
try:
    MSK_DPAR_DATA_TOL_BOUND_INF_ = 'MSK_DPAR_DATA_TOL_BOUND_INF'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1626
try:
    MSK_DPAR_DATA_TOL_BOUND_WRN_ = 'MSK_DPAR_DATA_TOL_BOUND_WRN'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1627
try:
    MSK_DPAR_DATA_TOL_C_HUGE_ = 'MSK_DPAR_DATA_TOL_C_HUGE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1628
try:
    MSK_DPAR_DATA_TOL_CJ_LARGE_ = 'MSK_DPAR_DATA_TOL_CJ_LARGE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1629
try:
    MSK_DPAR_DATA_TOL_QIJ_ = 'MSK_DPAR_DATA_TOL_QIJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1630
try:
    MSK_DPAR_DATA_TOL_X_ = 'MSK_DPAR_DATA_TOL_X'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1631
try:
    MSK_DPAR_FEASREPAIR_TOL_ = 'MSK_DPAR_FEASREPAIR_TOL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1632
try:
    MSK_DPAR_INTPNT_CO_TOL_DFEAS_ = 'MSK_DPAR_INTPNT_CO_TOL_DFEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1633
try:
    MSK_DPAR_INTPNT_CO_TOL_INFEAS_ = 'MSK_DPAR_INTPNT_CO_TOL_INFEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1634
try:
    MSK_DPAR_INTPNT_CO_TOL_MU_RED_ = 'MSK_DPAR_INTPNT_CO_TOL_MU_RED'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1635
try:
    MSK_DPAR_INTPNT_CO_TOL_NEAR_REL_ = 'MSK_DPAR_INTPNT_CO_TOL_NEAR_REL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1636
try:
    MSK_DPAR_INTPNT_CO_TOL_PFEAS_ = 'MSK_DPAR_INTPNT_CO_TOL_PFEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1637
try:
    MSK_DPAR_INTPNT_CO_TOL_REL_GAP_ = 'MSK_DPAR_INTPNT_CO_TOL_REL_GAP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1638
try:
    MSK_DPAR_INTPNT_NL_MERIT_BAL_ = 'MSK_DPAR_INTPNT_NL_MERIT_BAL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1639
try:
    MSK_DPAR_INTPNT_NL_TOL_DFEAS_ = 'MSK_DPAR_INTPNT_NL_TOL_DFEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1640
try:
    MSK_DPAR_INTPNT_NL_TOL_MU_RED_ = 'MSK_DPAR_INTPNT_NL_TOL_MU_RED'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1641
try:
    MSK_DPAR_INTPNT_NL_TOL_NEAR_REL_ = 'MSK_DPAR_INTPNT_NL_TOL_NEAR_REL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1642
try:
    MSK_DPAR_INTPNT_NL_TOL_PFEAS_ = 'MSK_DPAR_INTPNT_NL_TOL_PFEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1643
try:
    MSK_DPAR_INTPNT_NL_TOL_REL_GAP_ = 'MSK_DPAR_INTPNT_NL_TOL_REL_GAP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1644
try:
    MSK_DPAR_INTPNT_NL_TOL_REL_STEP_ = 'MSK_DPAR_INTPNT_NL_TOL_REL_STEP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1645
try:
    MSK_DPAR_INTPNT_TOL_DFEAS_ = 'MSK_DPAR_INTPNT_TOL_DFEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1646
try:
    MSK_DPAR_INTPNT_TOL_DSAFE_ = 'MSK_DPAR_INTPNT_TOL_DSAFE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1647
try:
    MSK_DPAR_INTPNT_TOL_INFEAS_ = 'MSK_DPAR_INTPNT_TOL_INFEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1648
try:
    MSK_DPAR_INTPNT_TOL_MU_RED_ = 'MSK_DPAR_INTPNT_TOL_MU_RED'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1649
try:
    MSK_DPAR_INTPNT_TOL_PATH_ = 'MSK_DPAR_INTPNT_TOL_PATH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1650
try:
    MSK_DPAR_INTPNT_TOL_PFEAS_ = 'MSK_DPAR_INTPNT_TOL_PFEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1651
try:
    MSK_DPAR_INTPNT_TOL_PSAFE_ = 'MSK_DPAR_INTPNT_TOL_PSAFE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1652
try:
    MSK_DPAR_INTPNT_TOL_REL_GAP_ = 'MSK_DPAR_INTPNT_TOL_REL_GAP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1653
try:
    MSK_DPAR_INTPNT_TOL_REL_STEP_ = 'MSK_DPAR_INTPNT_TOL_REL_STEP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1654
try:
    MSK_DPAR_INTPNT_TOL_STEP_SIZE_ = 'MSK_DPAR_INTPNT_TOL_STEP_SIZE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1655
try:
    MSK_DPAR_LOWER_OBJ_CUT_ = 'MSK_DPAR_LOWER_OBJ_CUT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1656
try:
    MSK_DPAR_LOWER_OBJ_CUT_FINITE_TRH_ = 'MSK_DPAR_LOWER_OBJ_CUT_FINITE_TRH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1657
try:
    MSK_DPAR_MIO_DISABLE_TERM_TIME_ = 'MSK_DPAR_MIO_DISABLE_TERM_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1658
try:
    MSK_DPAR_MIO_HEURISTIC_TIME_ = 'MSK_DPAR_MIO_HEURISTIC_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1659
try:
    MSK_DPAR_MIO_MAX_TIME_ = 'MSK_DPAR_MIO_MAX_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1660
try:
    MSK_DPAR_MIO_MAX_TIME_APRX_OPT_ = 'MSK_DPAR_MIO_MAX_TIME_APRX_OPT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1661
try:
    MSK_DPAR_MIO_NEAR_TOL_ABS_GAP_ = 'MSK_DPAR_MIO_NEAR_TOL_ABS_GAP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1662
try:
    MSK_DPAR_MIO_NEAR_TOL_REL_GAP_ = 'MSK_DPAR_MIO_NEAR_TOL_REL_GAP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1663
try:
    MSK_DPAR_MIO_REL_ADD_CUT_LIMITED_ = 'MSK_DPAR_MIO_REL_ADD_CUT_LIMITED'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1664
try:
    MSK_DPAR_MIO_REL_GAP_CONST_ = 'MSK_DPAR_MIO_REL_GAP_CONST'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1665
try:
    MSK_DPAR_MIO_TOL_ABS_GAP_ = 'MSK_DPAR_MIO_TOL_ABS_GAP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1666
try:
    MSK_DPAR_MIO_TOL_ABS_RELAX_INT_ = 'MSK_DPAR_MIO_TOL_ABS_RELAX_INT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1667
try:
    MSK_DPAR_MIO_TOL_FEAS_ = 'MSK_DPAR_MIO_TOL_FEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1668
try:
    MSK_DPAR_MIO_TOL_REL_GAP_ = 'MSK_DPAR_MIO_TOL_REL_GAP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1669
try:
    MSK_DPAR_MIO_TOL_REL_RELAX_INT_ = 'MSK_DPAR_MIO_TOL_REL_RELAX_INT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1670
try:
    MSK_DPAR_MIO_TOL_X_ = 'MSK_DPAR_MIO_TOL_X'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1671
try:
    MSK_DPAR_NONCONVEX_TOL_FEAS_ = 'MSK_DPAR_NONCONVEX_TOL_FEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1672
try:
    MSK_DPAR_NONCONVEX_TOL_OPT_ = 'MSK_DPAR_NONCONVEX_TOL_OPT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1673
try:
    MSK_DPAR_OPTIMIZER_MAX_TIME_ = 'MSK_DPAR_OPTIMIZER_MAX_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1674
try:
    MSK_DPAR_PRESOLVE_TOL_ABS_LINDEP_ = 'MSK_DPAR_PRESOLVE_TOL_ABS_LINDEP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1675
try:
    MSK_DPAR_PRESOLVE_TOL_AIJ_ = 'MSK_DPAR_PRESOLVE_TOL_AIJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1676
try:
    MSK_DPAR_PRESOLVE_TOL_REL_LINDEP_ = 'MSK_DPAR_PRESOLVE_TOL_REL_LINDEP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1677
try:
    MSK_DPAR_PRESOLVE_TOL_S_ = 'MSK_DPAR_PRESOLVE_TOL_S'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1678
try:
    MSK_DPAR_PRESOLVE_TOL_X_ = 'MSK_DPAR_PRESOLVE_TOL_X'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1679
try:
    MSK_DPAR_QCQO_REFORMULATE_REL_DROP_TOL_ = 'MSK_DPAR_QCQO_REFORMULATE_REL_DROP_TOL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1680
try:
    MSK_DPAR_SIM_LU_TOL_REL_PIV_ = 'MSK_DPAR_SIM_LU_TOL_REL_PIV'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1681
try:
    MSK_DPAR_SIMPLEX_ABS_TOL_PIV_ = 'MSK_DPAR_SIMPLEX_ABS_TOL_PIV'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1682
try:
    MSK_DPAR_UPPER_OBJ_CUT_ = 'MSK_DPAR_UPPER_OBJ_CUT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1683
try:
    MSK_DPAR_UPPER_OBJ_CUT_FINITE_TRH_ = 'MSK_DPAR_UPPER_OBJ_CUT_FINITE_TRH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1685
try:
    MSK_IPAR_ALLOC_ADD_QNZ_ = 'MSK_IPAR_ALLOC_ADD_QNZ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1686
try:
    MSK_IPAR_ANA_SOL_BASIS_ = 'MSK_IPAR_ANA_SOL_BASIS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1687
try:
    MSK_IPAR_ANA_SOL_PRINT_VIOLATED_ = 'MSK_IPAR_ANA_SOL_PRINT_VIOLATED'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1688
try:
    MSK_IPAR_AUTO_SORT_A_BEFORE_OPT_ = 'MSK_IPAR_AUTO_SORT_A_BEFORE_OPT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1689
try:
    MSK_IPAR_AUTO_UPDATE_SOL_INFO_ = 'MSK_IPAR_AUTO_UPDATE_SOL_INFO'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1690
try:
    MSK_IPAR_BASIS_SOLVE_USE_PLUS_ONE_ = 'MSK_IPAR_BASIS_SOLVE_USE_PLUS_ONE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1691
try:
    MSK_IPAR_BI_CLEAN_OPTIMIZER_ = 'MSK_IPAR_BI_CLEAN_OPTIMIZER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1692
try:
    MSK_IPAR_BI_IGNORE_MAX_ITER_ = 'MSK_IPAR_BI_IGNORE_MAX_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1693
try:
    MSK_IPAR_BI_IGNORE_NUM_ERROR_ = 'MSK_IPAR_BI_IGNORE_NUM_ERROR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1694
try:
    MSK_IPAR_BI_MAX_ITERATIONS_ = 'MSK_IPAR_BI_MAX_ITERATIONS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1695
try:
    MSK_IPAR_CACHE_LICENSE_ = 'MSK_IPAR_CACHE_LICENSE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1696
try:
    MSK_IPAR_CHECK_CONVEXITY_ = 'MSK_IPAR_CHECK_CONVEXITY'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1697
try:
    MSK_IPAR_COMPRESS_STATFILE_ = 'MSK_IPAR_COMPRESS_STATFILE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1698
try:
    MSK_IPAR_CONCURRENT_NUM_OPTIMIZERS_ = 'MSK_IPAR_CONCURRENT_NUM_OPTIMIZERS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1699
try:
    MSK_IPAR_CONCURRENT_PRIORITY_DUAL_SIMPLEX_ = 'MSK_IPAR_CONCURRENT_PRIORITY_DUAL_SIMPLEX'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1700
try:
    MSK_IPAR_CONCURRENT_PRIORITY_FREE_SIMPLEX_ = 'MSK_IPAR_CONCURRENT_PRIORITY_FREE_SIMPLEX'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1701
try:
    MSK_IPAR_CONCURRENT_PRIORITY_INTPNT_ = 'MSK_IPAR_CONCURRENT_PRIORITY_INTPNT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1702
try:
    MSK_IPAR_CONCURRENT_PRIORITY_PRIMAL_SIMPLEX_ = 'MSK_IPAR_CONCURRENT_PRIORITY_PRIMAL_SIMPLEX'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1703
try:
    MSK_IPAR_FEASREPAIR_OPTIMIZE_ = 'MSK_IPAR_FEASREPAIR_OPTIMIZE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1704
try:
    MSK_IPAR_INFEAS_GENERIC_NAMES_ = 'MSK_IPAR_INFEAS_GENERIC_NAMES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1705
try:
    MSK_IPAR_INFEAS_PREFER_PRIMAL_ = 'MSK_IPAR_INFEAS_PREFER_PRIMAL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1706
try:
    MSK_IPAR_INFEAS_REPORT_AUTO_ = 'MSK_IPAR_INFEAS_REPORT_AUTO'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1707
try:
    MSK_IPAR_INFEAS_REPORT_LEVEL_ = 'MSK_IPAR_INFEAS_REPORT_LEVEL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1708
try:
    MSK_IPAR_INTPNT_BASIS_ = 'MSK_IPAR_INTPNT_BASIS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1709
try:
    MSK_IPAR_INTPNT_DIFF_STEP_ = 'MSK_IPAR_INTPNT_DIFF_STEP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1710
try:
    MSK_IPAR_INTPNT_FACTOR_DEBUG_LVL_ = 'MSK_IPAR_INTPNT_FACTOR_DEBUG_LVL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1711
try:
    MSK_IPAR_INTPNT_FACTOR_METHOD_ = 'MSK_IPAR_INTPNT_FACTOR_METHOD'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1712
try:
    MSK_IPAR_INTPNT_HOTSTART_ = 'MSK_IPAR_INTPNT_HOTSTART'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1713
try:
    MSK_IPAR_INTPNT_MAX_ITERATIONS_ = 'MSK_IPAR_INTPNT_MAX_ITERATIONS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1714
try:
    MSK_IPAR_INTPNT_MAX_NUM_COR_ = 'MSK_IPAR_INTPNT_MAX_NUM_COR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1715
try:
    MSK_IPAR_INTPNT_MAX_NUM_REFINEMENT_STEPS_ = 'MSK_IPAR_INTPNT_MAX_NUM_REFINEMENT_STEPS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1716
try:
    MSK_IPAR_INTPNT_OFF_COL_TRH_ = 'MSK_IPAR_INTPNT_OFF_COL_TRH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1717
try:
    MSK_IPAR_INTPNT_ORDER_METHOD_ = 'MSK_IPAR_INTPNT_ORDER_METHOD'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1718
try:
    MSK_IPAR_INTPNT_REGULARIZATION_USE_ = 'MSK_IPAR_INTPNT_REGULARIZATION_USE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1719
try:
    MSK_IPAR_INTPNT_SCALING_ = 'MSK_IPAR_INTPNT_SCALING'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1720
try:
    MSK_IPAR_INTPNT_SOLVE_FORM_ = 'MSK_IPAR_INTPNT_SOLVE_FORM'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1721
try:
    MSK_IPAR_INTPNT_STARTING_POINT_ = 'MSK_IPAR_INTPNT_STARTING_POINT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1722
try:
    MSK_IPAR_LIC_TRH_EXPIRY_WRN_ = 'MSK_IPAR_LIC_TRH_EXPIRY_WRN'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1723
try:
    MSK_IPAR_LICENSE_ALLOW_OVERUSE_ = 'MSK_IPAR_LICENSE_ALLOW_OVERUSE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1724
try:
    MSK_IPAR_LICENSE_DEBUG_ = 'MSK_IPAR_LICENSE_DEBUG'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1725
try:
    MSK_IPAR_LICENSE_PAUSE_TIME_ = 'MSK_IPAR_LICENSE_PAUSE_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1726
try:
    MSK_IPAR_LICENSE_SUPPRESS_EXPIRE_WRNS_ = 'MSK_IPAR_LICENSE_SUPPRESS_EXPIRE_WRNS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1727
try:
    MSK_IPAR_LICENSE_WAIT_ = 'MSK_IPAR_LICENSE_WAIT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1728
try:
    MSK_IPAR_LOG_ = 'MSK_IPAR_LOG'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1729
try:
    MSK_IPAR_LOG_BI_ = 'MSK_IPAR_LOG_BI'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1730
try:
    MSK_IPAR_LOG_BI_FREQ_ = 'MSK_IPAR_LOG_BI_FREQ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1731
try:
    MSK_IPAR_LOG_CHECK_CONVEXITY_ = 'MSK_IPAR_LOG_CHECK_CONVEXITY'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1732
try:
    MSK_IPAR_LOG_CONCURRENT_ = 'MSK_IPAR_LOG_CONCURRENT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1733
try:
    MSK_IPAR_LOG_CUT_SECOND_OPT_ = 'MSK_IPAR_LOG_CUT_SECOND_OPT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1734
try:
    MSK_IPAR_LOG_EXPAND_ = 'MSK_IPAR_LOG_EXPAND'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1735
try:
    MSK_IPAR_LOG_FACTOR_ = 'MSK_IPAR_LOG_FACTOR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1736
try:
    MSK_IPAR_LOG_FEAS_REPAIR_ = 'MSK_IPAR_LOG_FEAS_REPAIR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1737
try:
    MSK_IPAR_LOG_FILE_ = 'MSK_IPAR_LOG_FILE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1738
try:
    MSK_IPAR_LOG_HEAD_ = 'MSK_IPAR_LOG_HEAD'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1739
try:
    MSK_IPAR_LOG_INFEAS_ANA_ = 'MSK_IPAR_LOG_INFEAS_ANA'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1740
try:
    MSK_IPAR_LOG_INTPNT_ = 'MSK_IPAR_LOG_INTPNT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1741
try:
    MSK_IPAR_LOG_MIO_ = 'MSK_IPAR_LOG_MIO'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1742
try:
    MSK_IPAR_LOG_MIO_FREQ_ = 'MSK_IPAR_LOG_MIO_FREQ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1743
try:
    MSK_IPAR_LOG_NONCONVEX_ = 'MSK_IPAR_LOG_NONCONVEX'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1744
try:
    MSK_IPAR_LOG_OPTIMIZER_ = 'MSK_IPAR_LOG_OPTIMIZER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1745
try:
    MSK_IPAR_LOG_ORDER_ = 'MSK_IPAR_LOG_ORDER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1746
try:
    MSK_IPAR_LOG_PARAM_ = 'MSK_IPAR_LOG_PARAM'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1747
try:
    MSK_IPAR_LOG_PRESOLVE_ = 'MSK_IPAR_LOG_PRESOLVE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1748
try:
    MSK_IPAR_LOG_RESPONSE_ = 'MSK_IPAR_LOG_RESPONSE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1749
try:
    MSK_IPAR_LOG_SENSITIVITY_ = 'MSK_IPAR_LOG_SENSITIVITY'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1750
try:
    MSK_IPAR_LOG_SENSITIVITY_OPT_ = 'MSK_IPAR_LOG_SENSITIVITY_OPT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1751
try:
    MSK_IPAR_LOG_SIM_ = 'MSK_IPAR_LOG_SIM'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1752
try:
    MSK_IPAR_LOG_SIM_FREQ_ = 'MSK_IPAR_LOG_SIM_FREQ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1753
try:
    MSK_IPAR_LOG_SIM_MINOR_ = 'MSK_IPAR_LOG_SIM_MINOR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1754
try:
    MSK_IPAR_LOG_SIM_NETWORK_FREQ_ = 'MSK_IPAR_LOG_SIM_NETWORK_FREQ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1755
try:
    MSK_IPAR_LOG_STORAGE_ = 'MSK_IPAR_LOG_STORAGE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1756
try:
    MSK_IPAR_MAX_NUM_WARNINGS_ = 'MSK_IPAR_MAX_NUM_WARNINGS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1757
try:
    MSK_IPAR_MIO_BRANCH_DIR_ = 'MSK_IPAR_MIO_BRANCH_DIR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1758
try:
    MSK_IPAR_MIO_BRANCH_PRIORITIES_USE_ = 'MSK_IPAR_MIO_BRANCH_PRIORITIES_USE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1759
try:
    MSK_IPAR_MIO_CONSTRUCT_SOL_ = 'MSK_IPAR_MIO_CONSTRUCT_SOL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1760
try:
    MSK_IPAR_MIO_CONT_SOL_ = 'MSK_IPAR_MIO_CONT_SOL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1761
try:
    MSK_IPAR_MIO_CUT_LEVEL_ROOT_ = 'MSK_IPAR_MIO_CUT_LEVEL_ROOT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1762
try:
    MSK_IPAR_MIO_CUT_LEVEL_TREE_ = 'MSK_IPAR_MIO_CUT_LEVEL_TREE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1763
try:
    MSK_IPAR_MIO_FEASPUMP_LEVEL_ = 'MSK_IPAR_MIO_FEASPUMP_LEVEL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1764
try:
    MSK_IPAR_MIO_HEURISTIC_LEVEL_ = 'MSK_IPAR_MIO_HEURISTIC_LEVEL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1765
try:
    MSK_IPAR_MIO_HOTSTART_ = 'MSK_IPAR_MIO_HOTSTART'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1766
try:
    MSK_IPAR_MIO_KEEP_BASIS_ = 'MSK_IPAR_MIO_KEEP_BASIS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1767
try:
    MSK_IPAR_MIO_LOCAL_BRANCH_NUMBER_ = 'MSK_IPAR_MIO_LOCAL_BRANCH_NUMBER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1768
try:
    MSK_IPAR_MIO_MAX_NUM_BRANCHES_ = 'MSK_IPAR_MIO_MAX_NUM_BRANCHES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1769
try:
    MSK_IPAR_MIO_MAX_NUM_RELAXS_ = 'MSK_IPAR_MIO_MAX_NUM_RELAXS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1770
try:
    MSK_IPAR_MIO_MAX_NUM_SOLUTIONS_ = 'MSK_IPAR_MIO_MAX_NUM_SOLUTIONS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1771
try:
    MSK_IPAR_MIO_MODE_ = 'MSK_IPAR_MIO_MODE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1772
try:
    MSK_IPAR_MIO_MT_USER_CB_ = 'MSK_IPAR_MIO_MT_USER_CB'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1773
try:
    MSK_IPAR_MIO_NODE_OPTIMIZER_ = 'MSK_IPAR_MIO_NODE_OPTIMIZER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1774
try:
    MSK_IPAR_MIO_NODE_SELECTION_ = 'MSK_IPAR_MIO_NODE_SELECTION'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1775
try:
    MSK_IPAR_MIO_OPTIMIZER_MODE_ = 'MSK_IPAR_MIO_OPTIMIZER_MODE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1776
try:
    MSK_IPAR_MIO_PRESOLVE_AGGREGATE_ = 'MSK_IPAR_MIO_PRESOLVE_AGGREGATE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1777
try:
    MSK_IPAR_MIO_PRESOLVE_PROBING_ = 'MSK_IPAR_MIO_PRESOLVE_PROBING'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1778
try:
    MSK_IPAR_MIO_PRESOLVE_USE_ = 'MSK_IPAR_MIO_PRESOLVE_USE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1779
try:
    MSK_IPAR_MIO_ROOT_OPTIMIZER_ = 'MSK_IPAR_MIO_ROOT_OPTIMIZER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1780
try:
    MSK_IPAR_MIO_STRONG_BRANCH_ = 'MSK_IPAR_MIO_STRONG_BRANCH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1781
try:
    MSK_IPAR_MIO_USE_MULTITHREADED_OPTIMIZER_ = 'MSK_IPAR_MIO_USE_MULTITHREADED_OPTIMIZER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1782
try:
    MSK_IPAR_MT_SPINCOUNT_ = 'MSK_IPAR_MT_SPINCOUNT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1783
try:
    MSK_IPAR_NONCONVEX_MAX_ITERATIONS_ = 'MSK_IPAR_NONCONVEX_MAX_ITERATIONS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1784
try:
    MSK_IPAR_NUM_THREADS_ = 'MSK_IPAR_NUM_THREADS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1785
try:
    MSK_IPAR_OPF_MAX_TERMS_PER_LINE_ = 'MSK_IPAR_OPF_MAX_TERMS_PER_LINE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1786
try:
    MSK_IPAR_OPF_WRITE_HEADER_ = 'MSK_IPAR_OPF_WRITE_HEADER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1787
try:
    MSK_IPAR_OPF_WRITE_HINTS_ = 'MSK_IPAR_OPF_WRITE_HINTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1788
try:
    MSK_IPAR_OPF_WRITE_PARAMETERS_ = 'MSK_IPAR_OPF_WRITE_PARAMETERS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1789
try:
    MSK_IPAR_OPF_WRITE_PROBLEM_ = 'MSK_IPAR_OPF_WRITE_PROBLEM'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1790
try:
    MSK_IPAR_OPF_WRITE_SOL_BAS_ = 'MSK_IPAR_OPF_WRITE_SOL_BAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1791
try:
    MSK_IPAR_OPF_WRITE_SOL_ITG_ = 'MSK_IPAR_OPF_WRITE_SOL_ITG'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1792
try:
    MSK_IPAR_OPF_WRITE_SOL_ITR_ = 'MSK_IPAR_OPF_WRITE_SOL_ITR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1793
try:
    MSK_IPAR_OPF_WRITE_SOLUTIONS_ = 'MSK_IPAR_OPF_WRITE_SOLUTIONS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1794
try:
    MSK_IPAR_OPTIMIZER_ = 'MSK_IPAR_OPTIMIZER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1795
try:
    MSK_IPAR_PARAM_READ_CASE_NAME_ = 'MSK_IPAR_PARAM_READ_CASE_NAME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1796
try:
    MSK_IPAR_PARAM_READ_IGN_ERROR_ = 'MSK_IPAR_PARAM_READ_IGN_ERROR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1797
try:
    MSK_IPAR_PRESOLVE_ELIM_FILL_ = 'MSK_IPAR_PRESOLVE_ELIM_FILL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1798
try:
    MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES_ = 'MSK_IPAR_PRESOLVE_ELIMINATOR_MAX_NUM_TRIES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1799
try:
    MSK_IPAR_PRESOLVE_ELIMINATOR_USE_ = 'MSK_IPAR_PRESOLVE_ELIMINATOR_USE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1800
try:
    MSK_IPAR_PRESOLVE_LEVEL_ = 'MSK_IPAR_PRESOLVE_LEVEL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1801
try:
    MSK_IPAR_PRESOLVE_LINDEP_ABS_WORK_TRH_ = 'MSK_IPAR_PRESOLVE_LINDEP_ABS_WORK_TRH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1802
try:
    MSK_IPAR_PRESOLVE_LINDEP_REL_WORK_TRH_ = 'MSK_IPAR_PRESOLVE_LINDEP_REL_WORK_TRH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1803
try:
    MSK_IPAR_PRESOLVE_LINDEP_USE_ = 'MSK_IPAR_PRESOLVE_LINDEP_USE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1804
try:
    MSK_IPAR_PRESOLVE_MAX_NUM_REDUCTIONS_ = 'MSK_IPAR_PRESOLVE_MAX_NUM_REDUCTIONS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1805
try:
    MSK_IPAR_PRESOLVE_USE_ = 'MSK_IPAR_PRESOLVE_USE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1806
try:
    MSK_IPAR_PRIMAL_REPAIR_OPTIMIZER_ = 'MSK_IPAR_PRIMAL_REPAIR_OPTIMIZER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1807
try:
    MSK_IPAR_QO_SEPARABLE_REFORMULATION_ = 'MSK_IPAR_QO_SEPARABLE_REFORMULATION'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1808
try:
    MSK_IPAR_READ_ANZ_ = 'MSK_IPAR_READ_ANZ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1809
try:
    MSK_IPAR_READ_CON_ = 'MSK_IPAR_READ_CON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1810
try:
    MSK_IPAR_READ_CONE_ = 'MSK_IPAR_READ_CONE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1811
try:
    MSK_IPAR_READ_DATA_COMPRESSED_ = 'MSK_IPAR_READ_DATA_COMPRESSED'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1812
try:
    MSK_IPAR_READ_DATA_FORMAT_ = 'MSK_IPAR_READ_DATA_FORMAT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1813
try:
    MSK_IPAR_READ_KEEP_FREE_CON_ = 'MSK_IPAR_READ_KEEP_FREE_CON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1814
try:
    MSK_IPAR_READ_LP_DROP_NEW_VARS_IN_BOU_ = 'MSK_IPAR_READ_LP_DROP_NEW_VARS_IN_BOU'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1815
try:
    MSK_IPAR_READ_LP_QUOTED_NAMES_ = 'MSK_IPAR_READ_LP_QUOTED_NAMES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1816
try:
    MSK_IPAR_READ_MPS_FORMAT_ = 'MSK_IPAR_READ_MPS_FORMAT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1817
try:
    MSK_IPAR_READ_MPS_KEEP_INT_ = 'MSK_IPAR_READ_MPS_KEEP_INT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1818
try:
    MSK_IPAR_READ_MPS_OBJ_SENSE_ = 'MSK_IPAR_READ_MPS_OBJ_SENSE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1819
try:
    MSK_IPAR_READ_MPS_RELAX_ = 'MSK_IPAR_READ_MPS_RELAX'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1820
try:
    MSK_IPAR_READ_MPS_WIDTH_ = 'MSK_IPAR_READ_MPS_WIDTH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1821
try:
    MSK_IPAR_READ_QNZ_ = 'MSK_IPAR_READ_QNZ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1822
try:
    MSK_IPAR_READ_TASK_IGNORE_PARAM_ = 'MSK_IPAR_READ_TASK_IGNORE_PARAM'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1823
try:
    MSK_IPAR_READ_VAR_ = 'MSK_IPAR_READ_VAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1824
try:
    MSK_IPAR_SENSITIVITY_ALL_ = 'MSK_IPAR_SENSITIVITY_ALL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1825
try:
    MSK_IPAR_SENSITIVITY_OPTIMIZER_ = 'MSK_IPAR_SENSITIVITY_OPTIMIZER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1826
try:
    MSK_IPAR_SENSITIVITY_TYPE_ = 'MSK_IPAR_SENSITIVITY_TYPE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1827
try:
    MSK_IPAR_SIM_BASIS_FACTOR_USE_ = 'MSK_IPAR_SIM_BASIS_FACTOR_USE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1828
try:
    MSK_IPAR_SIM_DEGEN_ = 'MSK_IPAR_SIM_DEGEN'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1829
try:
    MSK_IPAR_SIM_DUAL_CRASH_ = 'MSK_IPAR_SIM_DUAL_CRASH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1830
try:
    MSK_IPAR_SIM_DUAL_PHASEONE_METHOD_ = 'MSK_IPAR_SIM_DUAL_PHASEONE_METHOD'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1831
try:
    MSK_IPAR_SIM_DUAL_RESTRICT_SELECTION_ = 'MSK_IPAR_SIM_DUAL_RESTRICT_SELECTION'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1832
try:
    MSK_IPAR_SIM_DUAL_SELECTION_ = 'MSK_IPAR_SIM_DUAL_SELECTION'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1833
try:
    MSK_IPAR_SIM_EXPLOIT_DUPVEC_ = 'MSK_IPAR_SIM_EXPLOIT_DUPVEC'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1834
try:
    MSK_IPAR_SIM_HOTSTART_ = 'MSK_IPAR_SIM_HOTSTART'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1835
try:
    MSK_IPAR_SIM_HOTSTART_LU_ = 'MSK_IPAR_SIM_HOTSTART_LU'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1836
try:
    MSK_IPAR_SIM_INTEGER_ = 'MSK_IPAR_SIM_INTEGER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1837
try:
    MSK_IPAR_SIM_MAX_ITERATIONS_ = 'MSK_IPAR_SIM_MAX_ITERATIONS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1838
try:
    MSK_IPAR_SIM_MAX_NUM_SETBACKS_ = 'MSK_IPAR_SIM_MAX_NUM_SETBACKS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1839
try:
    MSK_IPAR_SIM_NON_SINGULAR_ = 'MSK_IPAR_SIM_NON_SINGULAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1840
try:
    MSK_IPAR_SIM_PRIMAL_CRASH_ = 'MSK_IPAR_SIM_PRIMAL_CRASH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1841
try:
    MSK_IPAR_SIM_PRIMAL_PHASEONE_METHOD_ = 'MSK_IPAR_SIM_PRIMAL_PHASEONE_METHOD'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1842
try:
    MSK_IPAR_SIM_PRIMAL_RESTRICT_SELECTION_ = 'MSK_IPAR_SIM_PRIMAL_RESTRICT_SELECTION'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1843
try:
    MSK_IPAR_SIM_PRIMAL_SELECTION_ = 'MSK_IPAR_SIM_PRIMAL_SELECTION'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1844
try:
    MSK_IPAR_SIM_REFACTOR_FREQ_ = 'MSK_IPAR_SIM_REFACTOR_FREQ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1845
try:
    MSK_IPAR_SIM_REFORMULATION_ = 'MSK_IPAR_SIM_REFORMULATION'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1846
try:
    MSK_IPAR_SIM_SAVE_LU_ = 'MSK_IPAR_SIM_SAVE_LU'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1847
try:
    MSK_IPAR_SIM_SCALING_ = 'MSK_IPAR_SIM_SCALING'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1848
try:
    MSK_IPAR_SIM_SCALING_METHOD_ = 'MSK_IPAR_SIM_SCALING_METHOD'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1849
try:
    MSK_IPAR_SIM_SOLVE_FORM_ = 'MSK_IPAR_SIM_SOLVE_FORM'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1850
try:
    MSK_IPAR_SIM_STABILITY_PRIORITY_ = 'MSK_IPAR_SIM_STABILITY_PRIORITY'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1851
try:
    MSK_IPAR_SIM_SWITCH_OPTIMIZER_ = 'MSK_IPAR_SIM_SWITCH_OPTIMIZER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1852
try:
    MSK_IPAR_SOL_FILTER_KEEP_BASIC_ = 'MSK_IPAR_SOL_FILTER_KEEP_BASIC'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1853
try:
    MSK_IPAR_SOL_FILTER_KEEP_RANGED_ = 'MSK_IPAR_SOL_FILTER_KEEP_RANGED'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1854
try:
    MSK_IPAR_SOL_READ_NAME_WIDTH_ = 'MSK_IPAR_SOL_READ_NAME_WIDTH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1855
try:
    MSK_IPAR_SOL_READ_WIDTH_ = 'MSK_IPAR_SOL_READ_WIDTH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1856
try:
    MSK_IPAR_SOLUTION_CALLBACK_ = 'MSK_IPAR_SOLUTION_CALLBACK'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1857
try:
    MSK_IPAR_TIMING_LEVEL_ = 'MSK_IPAR_TIMING_LEVEL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1858
try:
    MSK_IPAR_WARNING_LEVEL_ = 'MSK_IPAR_WARNING_LEVEL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1859
try:
    MSK_IPAR_WRITE_BAS_CONSTRAINTS_ = 'MSK_IPAR_WRITE_BAS_CONSTRAINTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1860
try:
    MSK_IPAR_WRITE_BAS_HEAD_ = 'MSK_IPAR_WRITE_BAS_HEAD'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1861
try:
    MSK_IPAR_WRITE_BAS_VARIABLES_ = 'MSK_IPAR_WRITE_BAS_VARIABLES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1862
try:
    MSK_IPAR_WRITE_DATA_COMPRESSED_ = 'MSK_IPAR_WRITE_DATA_COMPRESSED'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1863
try:
    MSK_IPAR_WRITE_DATA_FORMAT_ = 'MSK_IPAR_WRITE_DATA_FORMAT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1864
try:
    MSK_IPAR_WRITE_DATA_PARAM_ = 'MSK_IPAR_WRITE_DATA_PARAM'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1865
try:
    MSK_IPAR_WRITE_FREE_CON_ = 'MSK_IPAR_WRITE_FREE_CON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1866
try:
    MSK_IPAR_WRITE_GENERIC_NAMES_ = 'MSK_IPAR_WRITE_GENERIC_NAMES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1867
try:
    MSK_IPAR_WRITE_GENERIC_NAMES_IO_ = 'MSK_IPAR_WRITE_GENERIC_NAMES_IO'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1868
try:
    MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_CONIC_ITEMS_ = 'MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_CONIC_ITEMS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1869
try:
    MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_ITEMS_ = 'MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_ITEMS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1870
try:
    MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_NL_ITEMS_ = 'MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_NL_ITEMS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1871
try:
    MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_PSD_ITEMS_ = 'MSK_IPAR_WRITE_IGNORE_INCOMPATIBLE_PSD_ITEMS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1872
try:
    MSK_IPAR_WRITE_INT_CONSTRAINTS_ = 'MSK_IPAR_WRITE_INT_CONSTRAINTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1873
try:
    MSK_IPAR_WRITE_INT_HEAD_ = 'MSK_IPAR_WRITE_INT_HEAD'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1874
try:
    MSK_IPAR_WRITE_INT_VARIABLES_ = 'MSK_IPAR_WRITE_INT_VARIABLES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1875
try:
    MSK_IPAR_WRITE_LP_LINE_WIDTH_ = 'MSK_IPAR_WRITE_LP_LINE_WIDTH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1876
try:
    MSK_IPAR_WRITE_LP_QUOTED_NAMES_ = 'MSK_IPAR_WRITE_LP_QUOTED_NAMES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1877
try:
    MSK_IPAR_WRITE_LP_STRICT_FORMAT_ = 'MSK_IPAR_WRITE_LP_STRICT_FORMAT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1878
try:
    MSK_IPAR_WRITE_LP_TERMS_PER_LINE_ = 'MSK_IPAR_WRITE_LP_TERMS_PER_LINE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1879
try:
    MSK_IPAR_WRITE_MPS_INT_ = 'MSK_IPAR_WRITE_MPS_INT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1880
try:
    MSK_IPAR_WRITE_PRECISION_ = 'MSK_IPAR_WRITE_PRECISION'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1881
try:
    MSK_IPAR_WRITE_SOL_BARVARIABLES_ = 'MSK_IPAR_WRITE_SOL_BARVARIABLES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1882
try:
    MSK_IPAR_WRITE_SOL_CONSTRAINTS_ = 'MSK_IPAR_WRITE_SOL_CONSTRAINTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1883
try:
    MSK_IPAR_WRITE_SOL_HEAD_ = 'MSK_IPAR_WRITE_SOL_HEAD'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1884
try:
    MSK_IPAR_WRITE_SOL_IGNORE_INVALID_NAMES_ = 'MSK_IPAR_WRITE_SOL_IGNORE_INVALID_NAMES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1885
try:
    MSK_IPAR_WRITE_SOL_VARIABLES_ = 'MSK_IPAR_WRITE_SOL_VARIABLES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1886
try:
    MSK_IPAR_WRITE_TASK_INC_SOL_ = 'MSK_IPAR_WRITE_TASK_INC_SOL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1887
try:
    MSK_IPAR_WRITE_XML_MODE_ = 'MSK_IPAR_WRITE_XML_MODE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1889
try:
    MSK_IINF_ANA_PRO_NUM_CON_ = 'MSK_IINF_ANA_PRO_NUM_CON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1890
try:
    MSK_IINF_ANA_PRO_NUM_CON_EQ_ = 'MSK_IINF_ANA_PRO_NUM_CON_EQ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1891
try:
    MSK_IINF_ANA_PRO_NUM_CON_FR_ = 'MSK_IINF_ANA_PRO_NUM_CON_FR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1892
try:
    MSK_IINF_ANA_PRO_NUM_CON_LO_ = 'MSK_IINF_ANA_PRO_NUM_CON_LO'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1893
try:
    MSK_IINF_ANA_PRO_NUM_CON_RA_ = 'MSK_IINF_ANA_PRO_NUM_CON_RA'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1894
try:
    MSK_IINF_ANA_PRO_NUM_CON_UP_ = 'MSK_IINF_ANA_PRO_NUM_CON_UP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1895
try:
    MSK_IINF_ANA_PRO_NUM_VAR_ = 'MSK_IINF_ANA_PRO_NUM_VAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1896
try:
    MSK_IINF_ANA_PRO_NUM_VAR_BIN_ = 'MSK_IINF_ANA_PRO_NUM_VAR_BIN'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1897
try:
    MSK_IINF_ANA_PRO_NUM_VAR_CONT_ = 'MSK_IINF_ANA_PRO_NUM_VAR_CONT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1898
try:
    MSK_IINF_ANA_PRO_NUM_VAR_EQ_ = 'MSK_IINF_ANA_PRO_NUM_VAR_EQ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1899
try:
    MSK_IINF_ANA_PRO_NUM_VAR_FR_ = 'MSK_IINF_ANA_PRO_NUM_VAR_FR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1900
try:
    MSK_IINF_ANA_PRO_NUM_VAR_INT_ = 'MSK_IINF_ANA_PRO_NUM_VAR_INT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1901
try:
    MSK_IINF_ANA_PRO_NUM_VAR_LO_ = 'MSK_IINF_ANA_PRO_NUM_VAR_LO'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1902
try:
    MSK_IINF_ANA_PRO_NUM_VAR_RA_ = 'MSK_IINF_ANA_PRO_NUM_VAR_RA'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1903
try:
    MSK_IINF_ANA_PRO_NUM_VAR_UP_ = 'MSK_IINF_ANA_PRO_NUM_VAR_UP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1904
try:
    MSK_IINF_CONCURRENT_FASTEST_OPTIMIZER_ = 'MSK_IINF_CONCURRENT_FASTEST_OPTIMIZER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1905
try:
    MSK_IINF_INTPNT_FACTOR_DIM_DENSE_ = 'MSK_IINF_INTPNT_FACTOR_DIM_DENSE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1906
try:
    MSK_IINF_INTPNT_ITER_ = 'MSK_IINF_INTPNT_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1907
try:
    MSK_IINF_INTPNT_NUM_THREADS_ = 'MSK_IINF_INTPNT_NUM_THREADS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1908
try:
    MSK_IINF_INTPNT_SOLVE_DUAL_ = 'MSK_IINF_INTPNT_SOLVE_DUAL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1909
try:
    MSK_IINF_MIO_CONSTRUCT_NUM_ROUNDINGS_ = 'MSK_IINF_MIO_CONSTRUCT_NUM_ROUNDINGS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1910
try:
    MSK_IINF_MIO_CONSTRUCT_SOLUTION_ = 'MSK_IINF_MIO_CONSTRUCT_SOLUTION'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1911
try:
    MSK_IINF_MIO_INITIAL_SOLUTION_ = 'MSK_IINF_MIO_INITIAL_SOLUTION'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1912
try:
    MSK_IINF_MIO_NUM_ACTIVE_NODES_ = 'MSK_IINF_MIO_NUM_ACTIVE_NODES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1913
try:
    MSK_IINF_MIO_NUM_BASIS_CUTS_ = 'MSK_IINF_MIO_NUM_BASIS_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1914
try:
    MSK_IINF_MIO_NUM_BRANCH_ = 'MSK_IINF_MIO_NUM_BRANCH'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1915
try:
    MSK_IINF_MIO_NUM_CARDGUB_CUTS_ = 'MSK_IINF_MIO_NUM_CARDGUB_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1916
try:
    MSK_IINF_MIO_NUM_CLIQUE_CUTS_ = 'MSK_IINF_MIO_NUM_CLIQUE_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1917
try:
    MSK_IINF_MIO_NUM_COEF_REDC_CUTS_ = 'MSK_IINF_MIO_NUM_COEF_REDC_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1918
try:
    MSK_IINF_MIO_NUM_CONTRA_CUTS_ = 'MSK_IINF_MIO_NUM_CONTRA_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1919
try:
    MSK_IINF_MIO_NUM_DISAGG_CUTS_ = 'MSK_IINF_MIO_NUM_DISAGG_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1920
try:
    MSK_IINF_MIO_NUM_FLOW_COVER_CUTS_ = 'MSK_IINF_MIO_NUM_FLOW_COVER_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1921
try:
    MSK_IINF_MIO_NUM_GCD_CUTS_ = 'MSK_IINF_MIO_NUM_GCD_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1922
try:
    MSK_IINF_MIO_NUM_GOMORY_CUTS_ = 'MSK_IINF_MIO_NUM_GOMORY_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1923
try:
    MSK_IINF_MIO_NUM_GUB_COVER_CUTS_ = 'MSK_IINF_MIO_NUM_GUB_COVER_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1924
try:
    MSK_IINF_MIO_NUM_INT_SOLUTIONS_ = 'MSK_IINF_MIO_NUM_INT_SOLUTIONS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1925
try:
    MSK_IINF_MIO_NUM_KNAPSUR_COVER_CUTS_ = 'MSK_IINF_MIO_NUM_KNAPSUR_COVER_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1926
try:
    MSK_IINF_MIO_NUM_LATTICE_CUTS_ = 'MSK_IINF_MIO_NUM_LATTICE_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1927
try:
    MSK_IINF_MIO_NUM_LIFT_CUTS_ = 'MSK_IINF_MIO_NUM_LIFT_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1928
try:
    MSK_IINF_MIO_NUM_OBJ_CUTS_ = 'MSK_IINF_MIO_NUM_OBJ_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1929
try:
    MSK_IINF_MIO_NUM_PLAN_LOC_CUTS_ = 'MSK_IINF_MIO_NUM_PLAN_LOC_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1930
try:
    MSK_IINF_MIO_NUM_RELAX_ = 'MSK_IINF_MIO_NUM_RELAX'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1931
try:
    MSK_IINF_MIO_NUMCON_ = 'MSK_IINF_MIO_NUMCON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1932
try:
    MSK_IINF_MIO_NUMINT_ = 'MSK_IINF_MIO_NUMINT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1933
try:
    MSK_IINF_MIO_NUMVAR_ = 'MSK_IINF_MIO_NUMVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1934
try:
    MSK_IINF_MIO_OBJ_BOUND_DEFINED_ = 'MSK_IINF_MIO_OBJ_BOUND_DEFINED'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1935
try:
    MSK_IINF_MIO_TOTAL_NUM_CUTS_ = 'MSK_IINF_MIO_TOTAL_NUM_CUTS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1936
try:
    MSK_IINF_MIO_USER_OBJ_CUT_ = 'MSK_IINF_MIO_USER_OBJ_CUT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1937
try:
    MSK_IINF_OPT_NUMCON_ = 'MSK_IINF_OPT_NUMCON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1938
try:
    MSK_IINF_OPT_NUMVAR_ = 'MSK_IINF_OPT_NUMVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1939
try:
    MSK_IINF_OPTIMIZE_RESPONSE_ = 'MSK_IINF_OPTIMIZE_RESPONSE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1940
try:
    MSK_IINF_RD_NUMBARVAR_ = 'MSK_IINF_RD_NUMBARVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1941
try:
    MSK_IINF_RD_NUMCON_ = 'MSK_IINF_RD_NUMCON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1942
try:
    MSK_IINF_RD_NUMCONE_ = 'MSK_IINF_RD_NUMCONE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1943
try:
    MSK_IINF_RD_NUMINTVAR_ = 'MSK_IINF_RD_NUMINTVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1944
try:
    MSK_IINF_RD_NUMQ_ = 'MSK_IINF_RD_NUMQ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1945
try:
    MSK_IINF_RD_NUMVAR_ = 'MSK_IINF_RD_NUMVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1946
try:
    MSK_IINF_RD_PROTYPE_ = 'MSK_IINF_RD_PROTYPE'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1947
try:
    MSK_IINF_SIM_DUAL_DEG_ITER_ = 'MSK_IINF_SIM_DUAL_DEG_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1948
try:
    MSK_IINF_SIM_DUAL_HOTSTART_ = 'MSK_IINF_SIM_DUAL_HOTSTART'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1949
try:
    MSK_IINF_SIM_DUAL_HOTSTART_LU_ = 'MSK_IINF_SIM_DUAL_HOTSTART_LU'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1950
try:
    MSK_IINF_SIM_DUAL_INF_ITER_ = 'MSK_IINF_SIM_DUAL_INF_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1951
try:
    MSK_IINF_SIM_DUAL_ITER_ = 'MSK_IINF_SIM_DUAL_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1952
try:
    MSK_IINF_SIM_NETWORK_DUAL_DEG_ITER_ = 'MSK_IINF_SIM_NETWORK_DUAL_DEG_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1953
try:
    MSK_IINF_SIM_NETWORK_DUAL_HOTSTART_ = 'MSK_IINF_SIM_NETWORK_DUAL_HOTSTART'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1954
try:
    MSK_IINF_SIM_NETWORK_DUAL_HOTSTART_LU_ = 'MSK_IINF_SIM_NETWORK_DUAL_HOTSTART_LU'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1955
try:
    MSK_IINF_SIM_NETWORK_DUAL_INF_ITER_ = 'MSK_IINF_SIM_NETWORK_DUAL_INF_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1956
try:
    MSK_IINF_SIM_NETWORK_DUAL_ITER_ = 'MSK_IINF_SIM_NETWORK_DUAL_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1957
try:
    MSK_IINF_SIM_NETWORK_PRIMAL_DEG_ITER_ = 'MSK_IINF_SIM_NETWORK_PRIMAL_DEG_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1958
try:
    MSK_IINF_SIM_NETWORK_PRIMAL_HOTSTART_ = 'MSK_IINF_SIM_NETWORK_PRIMAL_HOTSTART'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1959
try:
    MSK_IINF_SIM_NETWORK_PRIMAL_HOTSTART_LU_ = 'MSK_IINF_SIM_NETWORK_PRIMAL_HOTSTART_LU'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1960
try:
    MSK_IINF_SIM_NETWORK_PRIMAL_INF_ITER_ = 'MSK_IINF_SIM_NETWORK_PRIMAL_INF_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1961
try:
    MSK_IINF_SIM_NETWORK_PRIMAL_ITER_ = 'MSK_IINF_SIM_NETWORK_PRIMAL_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1962
try:
    MSK_IINF_SIM_NUMCON_ = 'MSK_IINF_SIM_NUMCON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1963
try:
    MSK_IINF_SIM_NUMVAR_ = 'MSK_IINF_SIM_NUMVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1964
try:
    MSK_IINF_SIM_PRIMAL_DEG_ITER_ = 'MSK_IINF_SIM_PRIMAL_DEG_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1965
try:
    MSK_IINF_SIM_PRIMAL_DUAL_DEG_ITER_ = 'MSK_IINF_SIM_PRIMAL_DUAL_DEG_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1966
try:
    MSK_IINF_SIM_PRIMAL_DUAL_HOTSTART_ = 'MSK_IINF_SIM_PRIMAL_DUAL_HOTSTART'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1967
try:
    MSK_IINF_SIM_PRIMAL_DUAL_HOTSTART_LU_ = 'MSK_IINF_SIM_PRIMAL_DUAL_HOTSTART_LU'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1968
try:
    MSK_IINF_SIM_PRIMAL_DUAL_INF_ITER_ = 'MSK_IINF_SIM_PRIMAL_DUAL_INF_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1969
try:
    MSK_IINF_SIM_PRIMAL_DUAL_ITER_ = 'MSK_IINF_SIM_PRIMAL_DUAL_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1970
try:
    MSK_IINF_SIM_PRIMAL_HOTSTART_ = 'MSK_IINF_SIM_PRIMAL_HOTSTART'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1971
try:
    MSK_IINF_SIM_PRIMAL_HOTSTART_LU_ = 'MSK_IINF_SIM_PRIMAL_HOTSTART_LU'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1972
try:
    MSK_IINF_SIM_PRIMAL_INF_ITER_ = 'MSK_IINF_SIM_PRIMAL_INF_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1973
try:
    MSK_IINF_SIM_PRIMAL_ITER_ = 'MSK_IINF_SIM_PRIMAL_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1974
try:
    MSK_IINF_SIM_SOLVE_DUAL_ = 'MSK_IINF_SIM_SOLVE_DUAL'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1975
try:
    MSK_IINF_SOL_BAS_PROSTA_ = 'MSK_IINF_SOL_BAS_PROSTA'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1976
try:
    MSK_IINF_SOL_BAS_SOLSTA_ = 'MSK_IINF_SOL_BAS_SOLSTA'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1977
try:
    MSK_IINF_SOL_INT_PROSTA_ = 'MSK_IINF_SOL_INT_PROSTA'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1978
try:
    MSK_IINF_SOL_INT_SOLSTA_ = 'MSK_IINF_SOL_INT_SOLSTA'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1979
try:
    MSK_IINF_SOL_ITG_PROSTA_ = 'MSK_IINF_SOL_ITG_PROSTA'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1980
try:
    MSK_IINF_SOL_ITG_SOLSTA_ = 'MSK_IINF_SOL_ITG_SOLSTA'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1981
try:
    MSK_IINF_SOL_ITR_PROSTA_ = 'MSK_IINF_SOL_ITR_PROSTA'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1982
try:
    MSK_IINF_SOL_ITR_SOLSTA_ = 'MSK_IINF_SOL_ITR_SOLSTA'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1983
try:
    MSK_IINF_STO_NUM_A_CACHE_FLUSHES_ = 'MSK_IINF_STO_NUM_A_CACHE_FLUSHES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1984
try:
    MSK_IINF_STO_NUM_A_REALLOC_ = 'MSK_IINF_STO_NUM_A_REALLOC'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1985
try:
    MSK_IINF_STO_NUM_A_TRANSPOSES_ = 'MSK_IINF_STO_NUM_A_TRANSPOSES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1987
try:
    MSK_DINF_BI_CLEAN_DUAL_TIME_ = 'MSK_DINF_BI_CLEAN_DUAL_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1988
try:
    MSK_DINF_BI_CLEAN_PRIMAL_DUAL_TIME_ = 'MSK_DINF_BI_CLEAN_PRIMAL_DUAL_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1989
try:
    MSK_DINF_BI_CLEAN_PRIMAL_TIME_ = 'MSK_DINF_BI_CLEAN_PRIMAL_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1990
try:
    MSK_DINF_BI_CLEAN_TIME_ = 'MSK_DINF_BI_CLEAN_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1991
try:
    MSK_DINF_BI_DUAL_TIME_ = 'MSK_DINF_BI_DUAL_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1992
try:
    MSK_DINF_BI_PRIMAL_TIME_ = 'MSK_DINF_BI_PRIMAL_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1993
try:
    MSK_DINF_BI_TIME_ = 'MSK_DINF_BI_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1994
try:
    MSK_DINF_CONCURRENT_TIME_ = 'MSK_DINF_CONCURRENT_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1995
try:
    MSK_DINF_INTPNT_DUAL_FEAS_ = 'MSK_DINF_INTPNT_DUAL_FEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1996
try:
    MSK_DINF_INTPNT_DUAL_OBJ_ = 'MSK_DINF_INTPNT_DUAL_OBJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1997
try:
    MSK_DINF_INTPNT_FACTOR_NUM_FLOPS_ = 'MSK_DINF_INTPNT_FACTOR_NUM_FLOPS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1998
try:
    MSK_DINF_INTPNT_OPT_STATUS_ = 'MSK_DINF_INTPNT_OPT_STATUS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 1999
try:
    MSK_DINF_INTPNT_ORDER_TIME_ = 'MSK_DINF_INTPNT_ORDER_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2000
try:
    MSK_DINF_INTPNT_PRIMAL_FEAS_ = 'MSK_DINF_INTPNT_PRIMAL_FEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2001
try:
    MSK_DINF_INTPNT_PRIMAL_OBJ_ = 'MSK_DINF_INTPNT_PRIMAL_OBJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2002
try:
    MSK_DINF_INTPNT_TIME_ = 'MSK_DINF_INTPNT_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2003
try:
    MSK_DINF_MIO_CONSTRUCT_SOLUTION_OBJ_ = 'MSK_DINF_MIO_CONSTRUCT_SOLUTION_OBJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2004
try:
    MSK_DINF_MIO_HEURISTIC_TIME_ = 'MSK_DINF_MIO_HEURISTIC_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2005
try:
    MSK_DINF_MIO_OBJ_ABS_GAP_ = 'MSK_DINF_MIO_OBJ_ABS_GAP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2006
try:
    MSK_DINF_MIO_OBJ_BOUND_ = 'MSK_DINF_MIO_OBJ_BOUND'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2007
try:
    MSK_DINF_MIO_OBJ_INT_ = 'MSK_DINF_MIO_OBJ_INT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2008
try:
    MSK_DINF_MIO_OBJ_REL_GAP_ = 'MSK_DINF_MIO_OBJ_REL_GAP'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2009
try:
    MSK_DINF_MIO_OPTIMIZER_TIME_ = 'MSK_DINF_MIO_OPTIMIZER_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2010
try:
    MSK_DINF_MIO_ROOT_OPTIMIZER_TIME_ = 'MSK_DINF_MIO_ROOT_OPTIMIZER_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2011
try:
    MSK_DINF_MIO_ROOT_PRESOLVE_TIME_ = 'MSK_DINF_MIO_ROOT_PRESOLVE_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2012
try:
    MSK_DINF_MIO_TIME_ = 'MSK_DINF_MIO_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2013
try:
    MSK_DINF_MIO_USER_OBJ_CUT_ = 'MSK_DINF_MIO_USER_OBJ_CUT'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2014
try:
    MSK_DINF_OPTIMIZER_TIME_ = 'MSK_DINF_OPTIMIZER_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2015
try:
    MSK_DINF_PRESOLVE_ELI_TIME_ = 'MSK_DINF_PRESOLVE_ELI_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2016
try:
    MSK_DINF_PRESOLVE_LINDEP_TIME_ = 'MSK_DINF_PRESOLVE_LINDEP_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2017
try:
    MSK_DINF_PRESOLVE_TIME_ = 'MSK_DINF_PRESOLVE_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2018
try:
    MSK_DINF_PRIMAL_REPAIR_PENALTY_OBJ_ = 'MSK_DINF_PRIMAL_REPAIR_PENALTY_OBJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2019
try:
    MSK_DINF_QCQO_REFORMULATE_TIME_ = 'MSK_DINF_QCQO_REFORMULATE_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2020
try:
    MSK_DINF_RD_TIME_ = 'MSK_DINF_RD_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2021
try:
    MSK_DINF_SIM_DUAL_TIME_ = 'MSK_DINF_SIM_DUAL_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2022
try:
    MSK_DINF_SIM_FEAS_ = 'MSK_DINF_SIM_FEAS'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2023
try:
    MSK_DINF_SIM_NETWORK_DUAL_TIME_ = 'MSK_DINF_SIM_NETWORK_DUAL_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2024
try:
    MSK_DINF_SIM_NETWORK_PRIMAL_TIME_ = 'MSK_DINF_SIM_NETWORK_PRIMAL_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2025
try:
    MSK_DINF_SIM_NETWORK_TIME_ = 'MSK_DINF_SIM_NETWORK_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2026
try:
    MSK_DINF_SIM_OBJ_ = 'MSK_DINF_SIM_OBJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2027
try:
    MSK_DINF_SIM_PRIMAL_DUAL_TIME_ = 'MSK_DINF_SIM_PRIMAL_DUAL_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2028
try:
    MSK_DINF_SIM_PRIMAL_TIME_ = 'MSK_DINF_SIM_PRIMAL_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2029
try:
    MSK_DINF_SIM_TIME_ = 'MSK_DINF_SIM_TIME'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2030
try:
    MSK_DINF_SOL_BAS_DUAL_OBJ_ = 'MSK_DINF_SOL_BAS_DUAL_OBJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2031
try:
    MSK_DINF_SOL_BAS_DVIOLCON_ = 'MSK_DINF_SOL_BAS_DVIOLCON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2032
try:
    MSK_DINF_SOL_BAS_DVIOLVAR_ = 'MSK_DINF_SOL_BAS_DVIOLVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2033
try:
    MSK_DINF_SOL_BAS_PRIMAL_OBJ_ = 'MSK_DINF_SOL_BAS_PRIMAL_OBJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2034
try:
    MSK_DINF_SOL_BAS_PVIOLCON_ = 'MSK_DINF_SOL_BAS_PVIOLCON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2035
try:
    MSK_DINF_SOL_BAS_PVIOLVAR_ = 'MSK_DINF_SOL_BAS_PVIOLVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2036
try:
    MSK_DINF_SOL_ITG_PRIMAL_OBJ_ = 'MSK_DINF_SOL_ITG_PRIMAL_OBJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2037
try:
    MSK_DINF_SOL_ITG_PVIOLBARVAR_ = 'MSK_DINF_SOL_ITG_PVIOLBARVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2038
try:
    MSK_DINF_SOL_ITG_PVIOLCON_ = 'MSK_DINF_SOL_ITG_PVIOLCON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2039
try:
    MSK_DINF_SOL_ITG_PVIOLCONES_ = 'MSK_DINF_SOL_ITG_PVIOLCONES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2040
try:
    MSK_DINF_SOL_ITG_PVIOLITG_ = 'MSK_DINF_SOL_ITG_PVIOLITG'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2041
try:
    MSK_DINF_SOL_ITG_PVIOLVAR_ = 'MSK_DINF_SOL_ITG_PVIOLVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2042
try:
    MSK_DINF_SOL_ITR_DUAL_OBJ_ = 'MSK_DINF_SOL_ITR_DUAL_OBJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2043
try:
    MSK_DINF_SOL_ITR_DVIOLBARVAR_ = 'MSK_DINF_SOL_ITR_DVIOLBARVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2044
try:
    MSK_DINF_SOL_ITR_DVIOLCON_ = 'MSK_DINF_SOL_ITR_DVIOLCON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2045
try:
    MSK_DINF_SOL_ITR_DVIOLCONES_ = 'MSK_DINF_SOL_ITR_DVIOLCONES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2046
try:
    MSK_DINF_SOL_ITR_DVIOLVAR_ = 'MSK_DINF_SOL_ITR_DVIOLVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2047
try:
    MSK_DINF_SOL_ITR_PRIMAL_OBJ_ = 'MSK_DINF_SOL_ITR_PRIMAL_OBJ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2048
try:
    MSK_DINF_SOL_ITR_PVIOLBARVAR_ = 'MSK_DINF_SOL_ITR_PVIOLBARVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2049
try:
    MSK_DINF_SOL_ITR_PVIOLCON_ = 'MSK_DINF_SOL_ITR_PVIOLCON'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2050
try:
    MSK_DINF_SOL_ITR_PVIOLCONES_ = 'MSK_DINF_SOL_ITR_PVIOLCONES'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2051
try:
    MSK_DINF_SOL_ITR_PVIOLVAR_ = 'MSK_DINF_SOL_ITR_PVIOLVAR'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2053
try:
    MSK_LIINF_BI_CLEAN_DUAL_DEG_ITER_ = 'MSK_LIINF_BI_CLEAN_DUAL_DEG_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2054
try:
    MSK_LIINF_BI_CLEAN_DUAL_ITER_ = 'MSK_LIINF_BI_CLEAN_DUAL_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2055
try:
    MSK_LIINF_BI_CLEAN_PRIMAL_DEG_ITER_ = 'MSK_LIINF_BI_CLEAN_PRIMAL_DEG_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2056
try:
    MSK_LIINF_BI_CLEAN_PRIMAL_DUAL_DEG_ITER_ = 'MSK_LIINF_BI_CLEAN_PRIMAL_DUAL_DEG_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2057
try:
    MSK_LIINF_BI_CLEAN_PRIMAL_DUAL_ITER_ = 'MSK_LIINF_BI_CLEAN_PRIMAL_DUAL_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2058
try:
    MSK_LIINF_BI_CLEAN_PRIMAL_DUAL_SUB_ITER_ = 'MSK_LIINF_BI_CLEAN_PRIMAL_DUAL_SUB_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2059
try:
    MSK_LIINF_BI_CLEAN_PRIMAL_ITER_ = 'MSK_LIINF_BI_CLEAN_PRIMAL_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2060
try:
    MSK_LIINF_BI_DUAL_ITER_ = 'MSK_LIINF_BI_DUAL_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2061
try:
    MSK_LIINF_BI_PRIMAL_ITER_ = 'MSK_LIINF_BI_PRIMAL_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2062
try:
    MSK_LIINF_INTPNT_FACTOR_NUM_NZ_ = 'MSK_LIINF_INTPNT_FACTOR_NUM_NZ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2063
try:
    MSK_LIINF_MIO_INTPNT_ITER_ = 'MSK_LIINF_MIO_INTPNT_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2064
try:
    MSK_LIINF_MIO_SIMPLEX_ITER_ = 'MSK_LIINF_MIO_SIMPLEX_ITER'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2065
try:
    MSK_LIINF_RD_NUMANZ_ = 'MSK_LIINF_RD_NUMANZ'
except:
    pass

# C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2066
try:
    MSK_LIINF_RD_NUMQNZ_ = 'MSK_LIINF_RD_NUMQNZ'
except:
    pass

MSKintt = MSKint32t # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2074

MSKidxt = MSKint32t # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2075

MSKlidxt = MSKint32t # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2076

MSKlintt = MSKint32t # C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h: 2077

threadlocaleinfostruct = struct_threadlocaleinfostruct # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 674

threadmbcinfostruct = struct_threadmbcinfostruct # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 653

__lc_time_data = struct___lc_time_data # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 656

localeinfo_struct = struct_localeinfo_struct # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 661

tagLC_ID = struct_tagLC_ID # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 669

lconv = struct_lconv # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/_mingw.h: 691

_div_t = struct__div_t # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 49

_ldiv_t = struct__ldiv_t # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdlib.h: 54

_heapinfo = struct__heapinfo # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/malloc.h: 50

_iobuf = struct__iobuf # c:\\users\\roots\\anaconda\\mingw\\bin\\../lib/gcc/x86_64-w64-mingw32/4.7.0/../../../../x86_64-w64-mingw32/include/stdio.h: 26

# No inserted files

