require_relative 'mkmf.rb'

extension_name = 'nmatrix'

$INSTALLFILES = [
  ['ruby_nmatrix.h'       , '$(archdir)'],
  # ['ruby_nmatrix.hpp'     , '$(archdir)'],
  ['nmatrix_config.h', '$(archdir)'],
]

$DEBUG = true
$CFLAGS = ["-Wall -Werror=return-type",$CFLAGS].join(" ")
$CXXFLAGS = ["-Wall -Werror=return-type",$CXXFLAGS].join(" ")
$CPPFLAGS = ["-Wall -Werror=return-type",$CPPFLAGS].join(" ")


LIBDIR      = RbConfig::CONFIG['libdir']
INCLUDEDIR  = RbConfig::CONFIG['includedir']

HEADER_DIRS = [
  '/opt/local/include',
  '/usr/local/include',
  INCLUDEDIR,
  '/usr/include',
]

LIB_DIRS = [
  '/opt/local/lib',
  '/usr/local/lib',
  LIBDIR,
  '/usr/lib',
]

dir_config(extension_name, HEADER_DIRS, LIB_DIRS)

have_library('blas')
have_library('lapacke')

basenames = %w{ruby_nmatrix}
$objs = basenames.map { |b| "#{b}.o"   }
$srcs = basenames.map { |b| "#{b}.cpp" }

create_conf_h("nmatrix_config.h")
create_makefile(extension_name)
