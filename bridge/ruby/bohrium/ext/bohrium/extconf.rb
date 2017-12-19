require "mkmf"
require "erb"

_, bhxxlib = dir_config("bhxx")
dir_config("bohrium")

abort "Couldn't find C++ std"        unless have_library("stdc++")
abort "Couldn't find 'boost_system'" unless have_library("boost_system")
abort "Couldn't find 'bhxx'"         unless have_library("bhxx")

# If we have OpenMP, use it.
if have_library("omp")
  $CPPFLAGS << " -fopenmp=libomp"
  $LDFLAGS << " -lomp"
end

$CPPFLAGS << " -std=c++11"
$LDFLAGS << " -lboost_system -lbhxx -Wl,-rpath,#{bhxxlib}"

Dir[File.expand_path("#{__dir__}/templates/*.erb")].each do |fname|
  erb = ERB.new(File.read(fname))
  erb.filename = fname
  new_fname = File.basename(fname, File.extname(fname))
  File.open(File.expand_path("#{__dir__}/#{new_fname}"), "w") do |f|
    f.write(erb.result)
  end
end

create_makefile("bohrium/bohrium")
