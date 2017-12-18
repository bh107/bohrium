require "mkmf"
require "erb"

_, bhxxlib = dir_config("bhxx")
dir_config("bohrium")

abort "Couldn't find C++ std"        unless have_library("stdc++")
abort "Couldn't find 'boost_system'" unless have_library("boost_system")
abort "Couldn't find 'bhxx'"         unless have_library("bhxx")

$CPPFLAGS << " -std=c++11"
$LDFLAGS << " -lboost_system -lbhxx -Wl,-rpath,#{bhxxlib}"

%w(bohrium.cpp arithmetic.hpp trigonometry.hpp).each do |fname|
  filename = File.expand_path("#{__dir__}/templates/#{fname}.erb")
  erb = ERB.new(File.read(filename))
  erb.filename = filename
  File.open(File.expand_path("#{__dir__}/#{fname}"), "w") do |f|
    f.write(erb.result)
  end
end

create_makefile("bohrium/bohrium")
