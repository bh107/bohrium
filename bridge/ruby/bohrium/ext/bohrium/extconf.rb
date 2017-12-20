require "mkmf"
require "erb"
require "json"

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

def convert_opcodes(opcodes)
  opcodes.each_with_object(Hash.new) do |opcode, hash|
    types = opcode["types"].select do |types|
      # For now, only look at methods that has the same input and output types
      # ... and no complex numbers
      types.uniq.size == 1 && !(types.include?("BH_COMPLEX64") || types.include?("BH_COMPLEX128"))
    end.map do |types|
      # Remove bits, convert uint to int and remove BH_
      types.map { |t| t.gsub(/\d+$/, "").gsub(/U?INT/, "int64_t").gsub("BH_", "").downcase }.first
    end.each_with_object(Hash.new) do |type, thash|
      thash[type] = case type
                    when "int64_t" then ["T_FIXNUM", "T_BIGNUM"]
                    when "float"   then ["T_FLOAT"]
                    when "bool"    then ["T_TRUE", "T_FALSE"]
                    end
    end
    next if types.empty?

    name = opcode["opcode"].sub(/^BH_/, "").downcase
    hash[name] = { types: types, layouts: opcode["layout"] }
  end
end

# All opcodes
@opcodes = JSON.parse(File.read(File.expand_path("#{__dir__}/../../../../../core/codegen/opcodes.json")))

# Opcodes with ["A", "A"] layout
@opcodes_one_arg = convert_opcodes(@opcodes.select do |opcode|
  # Only look at layouts with array equal array (one argument array)
  opcode["layout"].include?(["A", "A"])
end)

# Opcodes with ["A", "A", "A"] layout
@opcodes_two_args = convert_opcodes(@opcodes.select do |opcode|
  # Only look at layouts with array `op` array (two argument arrays)
  opcode["layout"].include?(["A", "A", "A"])
end.reject do |opcode|
  # Remove SCATTER and GATHER as they are special
  ["BH_SCATTER", "BH_GATHER"].include?(opcode["opcode"])
end)

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
