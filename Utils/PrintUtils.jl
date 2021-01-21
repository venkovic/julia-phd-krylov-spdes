function printlnln(str::String, width=2)
  println()
  println(str)
  print(" " ^ width)
end
  
function space_println(str::String, width=4)
  println(" " ^ width * str)
end