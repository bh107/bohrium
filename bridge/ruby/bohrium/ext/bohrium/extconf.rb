require "mkmf"
require "erb"
require "json"

require_relative "bohrium_helper.rb"

# Need 'bhxx' and 'bohrium' paths
_, bhxxlib = dir_config("bhxx")
dir_config("bohrium")

# Fail if we can't find the stuff we need
abort "Couldn't find C++ std"        unless have_library("stdc++")
abort "Couldn't find 'boost_system'" unless have_library("boost_system")
abort "Couldn't find 'bhxx'"         unless have_library("bhxx")

# If we have OpenMP, use it.
if have_library("omp")
  $CPPFLAGS << " -fopenmp=libomp"
  $LDFLAGS << " -lomp"
end

# Append CPP and LD flags
$CPPFLAGS << " -std=c++11"
$LDFLAGS << " -lboost_system -lbhxx -Wl,-rpath,#{bhxxlib}"

# All opcodes
@opcodes = JSON.parse(File.read(File.expand_path("#{__dir__}/../../../../../core/codegen/opcodes.json")))

# Opcodes with ["A", "A"] layout
@opcodes_one_arg = convert_opcodes(@opcodes.select do |opcode|
  # Only look at layouts with array equal array (one argument array)
  opcode["layout"].include?(["A", "A"])
end)

# Opcodes with three operands and BH_BOOL as the result
@opcodes_two_args_boolean_result = convert_opcodes(@opcodes.select do |opcode|
  # Only look at layouts with array equal array (one argument array)
  opcode["nop"] == 3 && opcode["types"].map(&:first).uniq == ["BH_BOOL"]
end.reject do |opcode|
  # Remove SCATTER and GATHER as they are special
  ["BH_LOGICAL_AND_REDUCE", "BH_LOGICAL_OR_REDUCE", "BH_LOGICAL_XOR_REDUCE"].include?(opcode["opcode"])
end, false)

# Opcodes with ["A", "A", "A"] layout
@opcodes_two_args = convert_opcodes(@opcodes.select do |opcode|
  # Only look at layouts with array `op` array (two argument arrays)
  opcode["layout"].include?(["A", "A", "A"])
end.reject do |opcode|
  # Remove SCATTER and GATHER as they are special
  ["BH_SCATTER", "BH_GATHER"].include?(opcode["opcode"])
end.reject do |opcode|
  # Reject the ones already part of the above
  name = opcode["opcode"].sub(/^BH_/, "").downcase
  @opcodes_two_args_boolean_result.keys.include?(name)
end)

# Opcodes with ["A", "A", "K"] layout
@opcodes_two_args_constant = convert_opcodes(@opcodes.select do |opcode|
  # Only look at layouts with array `op` array (two argument arrays)
  opcode["layout"].include?(["A", "A", "K"])
end.reject do |opcode|
  ["BH_ARG_MAXIMUM_REDUCE", "BH_ARG_MINIMUM_REDUCE"].include?(opcode["opcode"])
end.reject do |opcode|
  # Reject the ones already part of the above
  name = opcode["opcode"].sub(/^BH_/, "").downcase
  @opcodes_two_args.keys.include?(name) || @opcodes_two_args_boolean_result.keys.include?(name)
end, false)

# Create 'hpp' and 'cpp' from templates
Dir[File.expand_path("#{__dir__}/templates/*.erb")].each do |fname|
  erb = ERB.new(File.read(fname))
  erb.filename = fname
  new_fname = File.basename(fname, File.extname(fname))
  File.open(File.expand_path("#{__dir__}/#{new_fname}"), "w") do |f|
    f.write(erb.result)
  end
end

# Create the actual makefile
create_makefile("bohrium/bohrium")
