def type_converter(type)
    case type
    when "T_BIGNUM"
    when "T_FIXNUM"
        return "NUM2INT"
    when "T_FLOAT"
        return "NUM2DBL"
    when "T_TRUE"
    when "T_FALSE"
        return ""
    else
        fail "Wrong type! Got #{type}."
    end
end

def convert_opcodes(opcodes, same_input=true)
  opcodes.each_with_object(Hash.new) do |opcode, hash|
    found_types = opcode["types"].select do |types|
      # For now, only look at methods that has the same input and output types
      !same_input || types.uniq.size == 1
    end.reject do |types|
      # ... and no complex numbers
      types.include?("BH_COMPLEX64") || types.include?("BH_COMPLEX128")
    end.map do |types|
      # Remove bits, convert uint to int and remove BH_
      types.map { |t| t.gsub(/\d+$/, "").gsub(/U?INT/, "int64_t").gsub("BH_", "").downcase }[1]
    end.each_with_object(Hash.new) do |type, thash|
      thash[type] = case type
                    when "int64_t" then ["T_FIXNUM", "T_BIGNUM"]
                    when "float"   then ["T_FLOAT"]
                    when "bool"    then ["T_TRUE", "T_FALSE"]
                    end
    end
    next if found_types.empty?

    name = opcode["opcode"].sub(/^BH_/, "").downcase
    hash[name] = { types: found_types, layouts: opcode["layout"] }
  end
end
